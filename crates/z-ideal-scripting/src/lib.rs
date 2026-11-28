//! Restricted Lua VM, validated scene bindings and lifecycle dispatch.
//!
//! Script instances own their VM and expose only generation-safe entity handles,
//! typed serde-backed components, prefab tickets and deferred scene commands.

#![forbid(unsafe_code)]

mod host;
pub use host::{ScriptDiagnostic, ScriptHost, ScriptHostError, ScriptId};

use mlua::{
    AnyUserData, Function, HookTriggers, Lua, LuaSerdeExt, StdLib, Table, UserData, Value, VmState,
};
use std::{
    cell::{Cell, RefCell},
    collections::{HashMap, HashSet},
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use z_ideal_math::{Quat, Transform, Vec3};
use z_ideal_scene::{
    CommandResults, Commands, Component, Entity, EntityTarget, PendingPrefab, Prefab,
    PrefabInstance, PrefabOverrides, SceneEntityId, World, WorldIdentity,
};

/// Stable package identifier for diagnostics and tooling.
pub const CRATE_NAME: &str = "z-ideal-scripting";
/// Version of the public Lua scripting contract.
pub const SCRIPT_API_VERSION: u32 = 1;
/// Lua VM instructions between execution-budget checks.
pub const INSTRUCTION_CHECK_INTERVAL: u32 = 1_000;
/// Largest accepted per-callback instruction budget.
pub const MAX_SCRIPT_INSTRUCTION_BUDGET: u64 = 100_000_000;
/// Maximum accepted UTF-8 source size, in bytes.
pub const MAX_SCRIPT_SOURCE_BYTES: usize = 1024 * 1024;
/// Maximum serialized JSON event payload size, in bytes.
pub const MAX_SCRIPT_EVENT_BYTES: usize = 64 * 1024;
/// Maximum nesting depth accepted for JSON event payloads.
pub const MAX_SCRIPT_EVENT_DEPTH: usize = 64;

/// Configuration for one isolated Lua script instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScriptConfig {
    /// Maximum Lua VM instructions allowed during each lifecycle callback.
    ///
    /// The VM checks this budget every INSTRUCTION_CHECK_INTERVAL instructions,
    /// so a callback may execute up to one check interval beyond the configured
    /// budget before it is interrupted.
    pub instruction_budget: u64,
}

impl Default for ScriptConfig {
    fn default() -> Self {
        Self {
            instruction_budget: 100_000,
        }
    }
}

/// Lifecycle phase used in script execution diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptPhase {
    /// Top-level chunk evaluation during loading.
    Load,
    /// One-time initialization callback.
    Start,
    /// Fixed simulation update callback.
    FixedUpdate,
    /// Per-frame update callback.
    Update,
    /// Event callback.
    Event,
    /// One-time shutdown callback.
    Shutdown,
}

/// Script lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptState {
    /// Source has loaded successfully but start has not completed.
    Loaded,
    /// Start completed; updates are permitted.
    Started,
    /// Shutdown completed or the loaded instance was explicitly shut down.
    Shutdown,
}

/// Errors returned by script loading and lifecycle operations.
#[derive(Debug, thiserror::Error)]
pub enum ScriptError {
    /// The execution budget is too small to perform one budget check.
    #[error(
        "instruction budget must be between {INSTRUCTION_CHECK_INTERVAL} and {MAX_SCRIPT_INSTRUCTION_BUDGET}"
    )]
    InvalidInstructionBudget,
    /// The source exceeds the configured hard size limit.
    #[error("script source is {actual} bytes; maximum is {maximum} bytes")]
    SourceTooLarge { actual: usize, maximum: usize },
    /// Lua VM initialization failed.
    #[error("failed to initialize Lua VM: {0}")]
    VmInitialization(#[source] mlua::Error),
    /// Source evaluation or lifecycle callback failed.
    #[error("{phase:?} failed for script '{source_name}': {message}")]
    Execution {
        source_name: String,
        phase: ScriptPhase,
        message: String,
    },
    /// The chunk did not return a script table.
    #[error("script '{source_name}' must return a table")]
    MissingScriptTable { source_name: String },
    /// A lifecycle field exists but is not a function.
    #[error("script '{source_name}' field '{callback}' must be a function or nil")]
    InvalidCallback {
        source_name: String,
        callback: &'static str,
    },
    /// The requested lifecycle operation is not valid in the current state.
    #[error("cannot perform {operation} while script '{source_name}' is {state:?}")]
    InvalidLifecycle {
        source_name: String,
        operation: &'static str,
        state: ScriptState,
    },
    /// Delta time must be finite and non-negative.
    #[error("delta time must be finite and non-negative")]
    InvalidDeltaTime,
    /// Event names must be non-empty and no longer than 128 bytes.
    #[error("event name must be non-empty and at most 128 bytes")]
    InvalidEventName,
    /// Event payload exceeds the bounded JSON size.
    #[error("event payload is {actual} bytes; maximum is {maximum} bytes")]
    EventPayloadTooLarge { actual: usize, maximum: usize },
    /// Event payload could not be serialized as JSON.
    #[error("event payload serialization failed: {0}")]
    EventSerialization(String),
    /// Event payload exceeds the maximum nested JSON depth.
    #[error("event payload nesting exceeds {MAX_SCRIPT_EVENT_DEPTH} levels")]
    EventPayloadTooDeep,
    /// A script instance cannot be moved between distinct ECS worlds.
    #[error("script '{source_name}' is already attached to a different world")]
    WorldMismatch { source_name: String },
    /// A started script must be shut down before it can be reloaded.
    #[error("script '{source_name}' must be shut down before reload")]
    ReloadWhileStarted { source_name: String },
    /// A registered component could not be serialized for the callback snapshot.
    #[error("component '{component_id}' snapshot failed: {message}")]
    ComponentSnapshot {
        component_id: String,
        message: String,
    },
    /// Applying script-issued scene commands failed.
    #[error("scene commands failed during {phase:?} for script '{source_name}': {message}")]
    CommandApplication {
        source_name: String,
        phase: ScriptPhase,
        message: String,
    },
}

/// Component registry errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScriptRegistryError {
    /// Component IDs must be non-empty.
    #[error("component ID must not be empty")]
    EmptyComponentId,
    /// Component IDs must be unique within one script runtime.
    #[error("component ID is already registered: {0}")]
    DuplicateComponentId(&'static str),
    /// Prefab names must be non-empty.
    #[error("prefab name must not be empty")]
    EmptyPrefabName,
    /// Prefab names must be unique within one script runtime.
    #[error("prefab name is already registered: {0}")]
    DuplicatePrefabName(String),
}

type EncodeComponent = fn(&World, Entity) -> Result<Option<serde_json::Value>, String>;
type QueueComponentInsert =
    fn(&mut Commands, EntityTarget, serde_json::Value) -> Result<(), String>;
type QueueComponentRemove = fn(&mut Commands, EntityTarget);

#[derive(Clone, Copy)]
struct ComponentBinding {
    encode: EncodeComponent,
    queue_insert: QueueComponentInsert,
    queue_remove: QueueComponentRemove,
}

fn encode_component<T: Component>(
    world: &World,
    entity: Entity,
) -> Result<Option<serde_json::Value>, String> {
    world
        .get::<T>(entity)
        .map(serde_json::to_value)
        .transpose()
        .map_err(|error| error.to_string())
}

fn queue_component_insert<T: Component>(
    commands: &mut Commands,
    target: EntityTarget,
    value: serde_json::Value,
) -> Result<(), String> {
    let component = serde_json::from_value::<T>(value).map_err(|error| error.to_string())?;
    commands.insert(target, component);
    Ok(())
}

fn queue_component_remove<T: Component>(commands: &mut Commands, target: EntityTarget) {
    commands.remove::<T>(target);
}

/// One isolated Lua VM and one script table.
///
/// The VM is deliberately not Send or Sync; callers should execute callbacks
/// on the engine's owning script thread. Each callback receives the script
/// table as its first argument, allowing scripts to keep instance state there.
pub struct ScriptInstance {
    lua: Lua,
    source_name: String,
    script: Table,
    start: Option<Function>,
    fixed_update: Option<Function>,
    update: Option<Function>,
    on_event: Option<Function>,
    shutdown: Option<Function>,
    state: ScriptState,
    paused: bool,
    instruction_budget: u64,
    executed_instructions: Arc<AtomicU64>,
    world_identity: Option<WorldIdentity>,
    attached_entity: Option<Entity>,
    component_registry: HashMap<&'static str, ComponentBinding>,
    prefab_registry: Rc<HashMap<String, Prefab>>,
}

impl ScriptInstance {
    /// Compiles and evaluates source in a new restricted Lua VM.
    ///
    /// The source chunk must return a table. Optional start(self, engine),
    /// fixed_update(self, delta_seconds, engine),
    /// update(self, delta_seconds, engine),
    /// on_event(self, event_name, json_payload, engine), and
    /// shutdown(self, engine) fields define lifecycle callbacks. Top-level
    /// code and each callback are independently subject to the configured
    /// instruction budget.
    pub fn load(
        source_name: impl Into<String>,
        source: &str,
        config: ScriptConfig,
    ) -> Result<Self, ScriptError> {
        let source_name = source_name.into();
        if config.instruction_budget < u64::from(INSTRUCTION_CHECK_INTERVAL)
            || config.instruction_budget > MAX_SCRIPT_INSTRUCTION_BUDGET
        {
            return Err(ScriptError::InvalidInstructionBudget);
        }
        if source.len() > MAX_SCRIPT_SOURCE_BYTES {
            return Err(ScriptError::SourceTooLarge {
                actual: source.len(),
                maximum: MAX_SCRIPT_SOURCE_BYTES,
            });
        }

        let lua = Lua::new_with(
            StdLib::TABLE | StdLib::STRING | StdLib::MATH | StdLib::UTF8,
            mlua::LuaOptions::default(),
        )
        .map_err(ScriptError::VmInitialization)?;

        // Base-library file loading and GC controls are removed. OS, IO,
        // package, debug and coroutine libraries are never loaded.
        let globals = lua.globals();
        for name in [
            "collectgarbage",
            "dofile",
            "load",
            "loadfile",
            "pcall",
            "print",
            "require",
            "warn",
            "xpcall",
        ] {
            globals
                .set(name, Value::Nil)
                .map_err(|error| ScriptError::Execution {
                    source_name: source_name.clone(),
                    phase: ScriptPhase::Load,
                    message: error.to_string(),
                })?;
        }
        globals
            .set("Z_IDEAL_SCRIPT_API_VERSION", SCRIPT_API_VERSION)
            .map_err(|error| ScriptError::Execution {
                source_name: source_name.clone(),
                phase: ScriptPhase::Load,
                message: error.to_string(),
            })?;

        let executed_instructions = Arc::new(AtomicU64::new(0));
        let instruction_counter = Arc::clone(&executed_instructions);
        let instruction_budget = config.instruction_budget;
        lua.set_hook(
            HookTriggers::new().every_nth_instruction(INSTRUCTION_CHECK_INTERVAL),
            move |_, _| {
                let executed = instruction_counter
                    .fetch_add(u64::from(INSTRUCTION_CHECK_INTERVAL), Ordering::Relaxed)
                    .saturating_add(u64::from(INSTRUCTION_CHECK_INTERVAL));
                if executed > instruction_budget {
                    return Err(mlua::Error::RuntimeError(format!(
                        "instruction budget exceeded ({instruction_budget})"
                    )));
                }
                Ok(VmState::Continue)
            },
        )
        .map_err(|error| ScriptError::Execution {
            source_name: source_name.clone(),
            phase: ScriptPhase::Load,
            message: error.to_string(),
        })?;

        let value = lua
            .load(source)
            .set_name(source_name.clone())
            .eval::<Value>()
            .map_err(|error| ScriptError::Execution {
                source_name: source_name.clone(),
                phase: ScriptPhase::Load,
                message: error.to_string(),
            })?;
        let script = match value {
            Value::Table(table) => table,
            _ => return Err(ScriptError::MissingScriptTable { source_name }),
        };
        let start = read_callback(&script, "start", &source_name)?;
        let fixed_update = read_callback(&script, "fixed_update", &source_name)?;
        let update = read_callback(&script, "update", &source_name)?;
        let on_event = read_callback(&script, "on_event", &source_name)?;
        let shutdown = read_callback(&script, "shutdown", &source_name)?;

        Ok(Self {
            lua,
            source_name,
            script,
            start,
            fixed_update,
            update,
            on_event,
            shutdown,
            state: ScriptState::Loaded,
            paused: false,
            instruction_budget,
            executed_instructions,
            world_identity: None,
            attached_entity: None,
            component_registry: HashMap::new(),
            prefab_registry: Rc::new(HashMap::new()),
        })
    }

    /// Registers a typed component for JSON-compatible Lua reads and writes.
    ///
    /// The component must also be registered in each World used by this
    /// instance. Values are deserialized and validated by World on commit.
    pub fn register_component<T: Component>(&mut self) -> Result<(), ScriptRegistryError> {
        if T::ID.trim().is_empty() {
            return Err(ScriptRegistryError::EmptyComponentId);
        }
        if self.component_registry.contains_key(T::ID) {
            return Err(ScriptRegistryError::DuplicateComponentId(T::ID));
        }
        self.component_registry.insert(
            T::ID,
            ComponentBinding {
                encode: encode_component::<T>,
                queue_insert: queue_component_insert::<T>,
                queue_remove: queue_component_remove::<T>,
            },
        );
        Ok(())
    }

    /// Registers a validated prefab under a script-visible name.
    pub fn register_prefab(
        &mut self,
        name: impl Into<String>,
        prefab: Prefab,
    ) -> Result<(), ScriptRegistryError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(ScriptRegistryError::EmptyPrefabName);
        }
        if self.prefab_registry.contains_key(&name) {
            return Err(ScriptRegistryError::DuplicatePrefabName(name));
        }
        Rc::make_mut(&mut self.prefab_registry).insert(name, prefab);
        Ok(())
    }

    /// Returns the source name associated with diagnostics.
    #[must_use]
    pub fn source_name(&self) -> &str {
        &self.source_name
    }

    /// Returns the current lifecycle state.
    #[must_use]
    pub const fn state(&self) -> ScriptState {
        self.state
    }

    /// Starts the script once.
    pub fn start(&mut self) -> Result<(), ScriptError> {
        self.require_state("start", ScriptState::Loaded)?;
        if let Some(callback) = &self.start {
            self.call(
                ScriptPhase::Start,
                callback,
                (self.script.clone(), Value::Nil),
            )?;
        }
        self.state = ScriptState::Started;
        Ok(())
    }

    /// Replaces this script with a newly loaded source.
    ///
    /// Reload is permitted only while Loaded or Shutdown. A Started instance
    /// must first complete shutdown (with its bound World when scene APIs were
    /// used). The replacement receives a fresh VM and empty script state;
    /// failed replacement loading leaves the original instance unchanged.
    pub fn reload(&mut self, source: &str, config: ScriptConfig) -> Result<(), ScriptError> {
        if self.state == ScriptState::Started {
            return Err(ScriptError::ReloadWhileStarted {
                source_name: self.source_name.clone(),
            });
        }
        let replacement = Self::load(self.source_name.clone(), source, config)?;
        *self = replacement;
        Ok(())
    }

    /// Sets whether fixed and variable updates are currently suppressed.
    ///
    /// Pausing does not change lifecycle state and does not invoke shutdown.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Returns whether update callbacks are currently suppressed.
    #[must_use]
    pub const fn is_paused(&self) -> bool {
        self.paused
    }

    /// Runs one fixed simulation update with a finite, non-negative delta in seconds.
    pub fn fixed_update(&mut self, delta_seconds: f64) -> Result<(), ScriptError> {
        self.require_state("fixed_update", ScriptState::Started)?;
        validate_delta_time(delta_seconds)?;
        if self.paused {
            return Ok(());
        }
        if let Some(callback) = self.fixed_update.clone() {
            self.call(
                ScriptPhase::FixedUpdate,
                &callback,
                (self.script.clone(), delta_seconds, Value::Nil),
            )?;
        }
        Ok(())
    }

    /// Runs one fixed simulation update with the scoped engine scene API.
    pub fn fixed_update_with_world(
        &mut self,
        world: &mut World,
        delta_seconds: f64,
    ) -> Result<(), ScriptError> {
        self.require_state("fixed_update", ScriptState::Started)?;
        validate_delta_time(delta_seconds)?;
        if self.paused {
            return Ok(());
        }
        self.bind_world(world)?;
        let context = self.create_command_context(ScriptPhase::FixedUpdate, world)?;
        let callback = self.fixed_update.clone();
        if let Some(callback) = callback {
            let result = self.call(
                ScriptPhase::FixedUpdate,
                &callback,
                (self.script.clone(), delta_seconds, context.api.clone()),
            );
            context.active.set(false);
            if let Err(error) = result {
                invalidate_pending_handles(&context.handles);
                invalidate_prefab_tickets(&context.prefab_tickets);
                return Err(error);
            }
        } else {
            context.active.set(false);
        }
        self.apply_command_context(world, ScriptPhase::FixedUpdate, context)
    }

    /// Runs one frame update with a finite, non-negative delta in seconds.
    pub fn update(&mut self, delta_seconds: f64) -> Result<(), ScriptError> {
        self.require_state("update", ScriptState::Started)?;
        validate_delta_time(delta_seconds)?;
        if self.paused {
            return Ok(());
        }
        if let Some(callback) = &self.update {
            self.call(
                ScriptPhase::Update,
                callback,
                (self.script.clone(), delta_seconds, Value::Nil),
            )?;
        }
        Ok(())
    }

    /// Shuts down the script. A loaded but not-started script transitions
    /// directly to Shutdown without invoking a callback.
    pub fn shutdown(&mut self) -> Result<(), ScriptError> {
        match self.state {
            ScriptState::Loaded => {
                self.state = ScriptState::Shutdown;
                Ok(())
            }
            ScriptState::Started => {
                if let Some(callback) = &self.shutdown {
                    self.call(
                        ScriptPhase::Shutdown,
                        callback,
                        (self.script.clone(), Value::Nil),
                    )?;
                }
                self.state = ScriptState::Shutdown;
                Ok(())
            }
            ScriptState::Shutdown => Err(ScriptError::InvalidLifecycle {
                source_name: self.source_name.clone(),
                operation: "shutdown",
                state: self.state,
            }),
        }
    }

    /// Starts the script with a scoped engine API for deferred entity commands.
    ///
    /// The API table is passed as the second callback argument. It is valid only
    /// for the duration of this callback. Commands are discarded if Lua fails;
    /// successful callbacks apply commands FIFO before returning.
    pub fn start_with_world(&mut self, world: &mut World) -> Result<(), ScriptError> {
        self.require_state("start", ScriptState::Loaded)?;
        self.bind_world(world)?;
        let context = self.create_command_context(ScriptPhase::Start, world)?;
        let callback = self.start.clone();
        if let Some(callback) = callback {
            let result = self.call(
                ScriptPhase::Start,
                &callback,
                (self.script.clone(), context.api.clone()),
            );
            context.active.set(false);
            if let Err(error) = result {
                invalidate_pending_handles(&context.handles);
                invalidate_prefab_tickets(&context.prefab_tickets);
                return Err(error);
            }
        } else {
            context.active.set(false);
        }
        self.state = ScriptState::Started;
        self.apply_command_context(world, ScriptPhase::Start, context)
    }

    /// Runs a frame update with a scoped engine API for deferred entity commands.
    ///
    /// Delta time is in seconds. The API table is passed as the third callback
    /// argument after self and delta_seconds.
    pub fn update_with_world(
        &mut self,
        world: &mut World,
        delta_seconds: f64,
    ) -> Result<(), ScriptError> {
        self.require_state("update", ScriptState::Started)?;
        validate_delta_time(delta_seconds)?;
        if self.paused {
            return Ok(());
        }
        self.bind_world(world)?;
        let context = self.create_command_context(ScriptPhase::Update, world)?;
        let callback = self.update.clone();
        if let Some(callback) = callback {
            let result = self.call(
                ScriptPhase::Update,
                &callback,
                (self.script.clone(), delta_seconds, context.api.clone()),
            );
            context.active.set(false);
            if let Err(error) = result {
                invalidate_pending_handles(&context.handles);
                invalidate_prefab_tickets(&context.prefab_tickets);
                return Err(error);
            }
        } else {
            context.active.set(false);
        }
        self.apply_command_context(world, ScriptPhase::Update, context)
    }

    /// Dispatches a JSON event to the optional on_event(self, name, payload, engine) callback.
    pub fn dispatch_event(
        &mut self,
        name: &str,
        payload: &serde_json::Value,
    ) -> Result<(), ScriptError> {
        self.require_state("dispatch_event", ScriptState::Started)?;
        validate_event(name, payload)?;
        let Some(callback) = self.on_event.clone() else {
            return Ok(());
        };
        let payload = self
            .lua
            .to_value(payload)
            .map_err(|error| self.execution_error(ScriptPhase::Event, error))?;
        self.call(
            ScriptPhase::Event,
            &callback,
            (self.script.clone(), name, payload, Value::Nil),
        )
    }

    /// Dispatches a JSON event with a scoped engine API for deferred scene commands.
    pub fn dispatch_event_with_world(
        &mut self,
        world: &mut World,
        name: &str,
        payload: &serde_json::Value,
    ) -> Result<(), ScriptError> {
        self.require_state("dispatch_event", ScriptState::Started)?;
        validate_event(name, payload)?;
        self.bind_world(world)?;
        let context = self.create_command_context(ScriptPhase::Event, world)?;
        let callback = self.on_event.clone();
        if let Some(callback) = callback {
            let payload = self
                .lua
                .to_value(payload)
                .map_err(|error| self.execution_error(ScriptPhase::Event, error))?;
            let result = self.call(
                ScriptPhase::Event,
                &callback,
                (self.script.clone(), name, payload, context.api.clone()),
            );
            context.active.set(false);
            if let Err(error) = result {
                invalidate_pending_handles(&context.handles);
                invalidate_prefab_tickets(&context.prefab_tickets);
                return Err(error);
            }
        } else {
            context.active.set(false);
        }
        self.apply_command_context(world, ScriptPhase::Event, context)
    }

    /// Shuts down the script with a scoped engine API for deferred entity commands.
    pub fn shutdown_with_world(&mut self, world: &mut World) -> Result<(), ScriptError> {
        match self.state {
            ScriptState::Loaded => {
                self.bind_world(world)?;
                self.state = ScriptState::Shutdown;
                Ok(())
            }
            ScriptState::Started => {
                self.bind_world(world)?;
                let context = self.create_command_context(ScriptPhase::Shutdown, world)?;
                let callback = self.shutdown.clone();
                if let Some(callback) = callback {
                    let result = self.call(
                        ScriptPhase::Shutdown,
                        &callback,
                        (self.script.clone(), context.api.clone()),
                    );
                    context.active.set(false);
                    if let Err(error) = result {
                        invalidate_pending_handles(&context.handles);
                        invalidate_prefab_tickets(&context.prefab_tickets);
                        return Err(error);
                    }
                } else {
                    context.active.set(false);
                }
                self.state = ScriptState::Shutdown;
                self.apply_command_context(world, ScriptPhase::Shutdown, context)
            }
            ScriptState::Shutdown => Err(ScriptError::InvalidLifecycle {
                source_name: self.source_name.clone(),
                operation: "shutdown",
                state: self.state,
            }),
        }
    }

    pub(crate) fn bind_entity(&mut self, world: &World, entity: Entity) -> Result<(), ScriptError> {
        self.bind_world(world)?;
        if !world.is_alive(entity) {
            return Err(ScriptError::Execution {
                source_name: self.source_name.clone(),
                phase: ScriptPhase::Load,
                message: "cannot bind script to a stale entity".to_owned(),
            });
        }
        self.attached_entity = Some(entity);
        Ok(())
    }

    pub(crate) fn bind_world(&mut self, world: &World) -> Result<(), ScriptError> {
        let identity = world.identity();
        match &self.world_identity {
            Some(bound) if bound != &identity => Err(ScriptError::WorldMismatch {
                source_name: self.source_name.clone(),
            }),
            Some(_) => Ok(()),
            None => {
                self.world_identity = Some(identity);
                Ok(())
            }
        }
    }

    fn create_command_context(
        &self,
        phase: ScriptPhase,
        world: &World,
    ) -> Result<CallbackCommands, ScriptError> {
        let commands = Rc::new(RefCell::new(Commands::new()));
        let handles = Rc::new(RefCell::new(Vec::new()));
        let prefab_tickets = Rc::new(RefCell::new(Vec::new()));
        let destroyed_prefabs = Rc::new(RefCell::new(Vec::new()));
        let active = Rc::new(Cell::new(true));
        let api = self
            .lua
            .create_table()
            .map_err(|error| self.execution_error(phase, error))?;
        if let Some(entity) = self.attached_entity {
            let handle = Rc::new(RefCell::new(Some(EntityTarget::Existing(entity))));
            let userdata = self
                .lua
                .create_userdata(ScriptEntity { target: handle })
                .map_err(|error| self.execution_error(phase, error))?;
            api.set("entity", userdata)
                .map_err(|error| self.execution_error(phase, error))?;
        }
        let entities = self
            .lua
            .create_table()
            .map_err(|error| self.execution_error(phase, error))?;

        let spawn_commands = Rc::clone(&commands);
        let spawn_handles = Rc::clone(&handles);
        let spawn_active = Rc::clone(&active);
        entities
            .set(
                "spawn",
                self.lua
                    .create_function(move |lua, ()| {
                        ensure_context_active(&spawn_active)?;
                        let pending = spawn_commands
                            .try_borrow_mut()
                            .map_err(|_| {
                                mlua::Error::RuntimeError(
                                    "entity command buffer is already borrowed".into(),
                                )
                            })?
                            .spawn()
                            .map_err(mlua::Error::external)?;
                        let handle = Rc::new(RefCell::new(Some(EntityTarget::Pending(pending))));
                        spawn_handles
                            .try_borrow_mut()
                            .map_err(|_| {
                                mlua::Error::RuntimeError(
                                    "entity handle registry is already borrowed".into(),
                                )
                            })?
                            .push(Rc::clone(&handle));
                        lua.create_userdata(ScriptEntity { target: handle })
                    })
                    .map_err(|error| self.execution_error(phase, error))?,
            )
            .map_err(|error| self.execution_error(phase, error))?;

        let destroy_commands = Rc::clone(&commands);
        let destroy_active = Rc::clone(&active);
        entities
            .set(
                "destroy",
                self.lua
                    .create_function(move |_, entity: AnyUserData| {
                        ensure_context_active(&destroy_active)?;
                        let target = entity_target(&entity)?;
                        destroy_commands
                            .try_borrow_mut()
                            .map_err(|_| {
                                mlua::Error::RuntimeError(
                                    "entity command buffer is already borrowed".into(),
                                )
                            })?
                            .despawn(target);
                        Ok(())
                    })
                    .map_err(|error| self.execution_error(phase, error))?,
            )
            .map_err(|error| self.execution_error(phase, error))?;

        let parent_commands = Rc::clone(&commands);
        let parent_active = Rc::clone(&active);
        entities
            .set(
                "set_parent",
                self.lua
                    .create_function(
                        move |_, (child, parent): (AnyUserData, Option<AnyUserData>)| {
                            ensure_context_active(&parent_active)?;
                            let child = entity_target(&child)?;
                            let parent = parent.as_ref().map(entity_target).transpose()?;
                            parent_commands
                                .try_borrow_mut()
                                .map_err(|_| {
                                    mlua::Error::RuntimeError(
                                        "entity command buffer is already borrowed".into(),
                                    )
                                })?
                                .set_parent(child, parent);
                            Ok(())
                        },
                    )
                    .map_err(|error| self.execution_error(phase, error))?,
            )
            .map_err(|error| self.execution_error(phase, error))?;

        let world_entities = world.entities();
        let alive_entities: Rc<HashSet<Entity>> = Rc::new(world_entities.iter().copied().collect());
        let mut component_snapshot = HashMap::new();
        for entity in &world_entities {
            for (component_id, binding) in &self.component_registry {
                if let Some(value) = (binding.encode)(world, *entity).map_err(|message| {
                    ScriptError::ComponentSnapshot {
                        component_id: (*component_id).to_owned(),
                        message,
                    }
                })? {
                    component_snapshot.insert((*entity, (*component_id).to_owned()), value);
                }
            }
        }
        let component_snapshot = Rc::new(component_snapshot);
        let registered_component_ids: Rc<HashSet<String>> = Rc::new(
            self.component_registry
                .keys()
                .map(|id| (*id).to_owned())
                .collect(),
        );

        let transform_snapshot: Rc<HashMap<Entity, Transform>> = Rc::new(
            world_entities
                .iter()
                .copied()
                .filter_map(|entity| {
                    world
                        .get::<Transform>(entity)
                        .copied()
                        .map(|transform| (entity, transform))
                })
                .collect(),
        );
        let transforms = self
            .lua
            .create_table()
            .map_err(|error| self.execution_error(phase, error))?;

        let get_alive = Rc::clone(&alive_entities);
        let get_snapshot = Rc::clone(&transform_snapshot);
        let get_active = Rc::clone(&active);
        transforms
            .set(
                "get",
                self.lua
                    .create_function(move |lua, entity: AnyUserData| {
                        ensure_context_active(&get_active)?;
                        match entity_target(&entity)? {
                            EntityTarget::Existing(entity) => {
                                if !get_alive.contains(&entity) {
                                    return Err(mlua::Error::RuntimeError(
                                        "stale entity handle".into(),
                                    ));
                                }
                                match get_snapshot.get(&entity) {
                                    Some(transform) => {
                                        Ok(Value::Table(transform_to_lua(lua, *transform)?))
                                    }
                                    None => Ok(Value::Nil),
                                }
                            }
                            EntityTarget::Pending(_) => Ok(Value::Nil),
                        }
                    })
                    .map_err(|error| self.execution_error(phase, error))?,
            )
            .map_err(|error| self.execution_error(phase, error))?;

        let set_commands = Rc::clone(&commands);
        let set_active = Rc::clone(&active);
        transforms
            .set(
                "set",
                self.lua
                    .create_function(move |_, (entity, value): (AnyUserData, Table)| {
                        ensure_context_active(&set_active)?;
                        let target = entity_target(&entity)?;
                        let translation = read_vec3(&value, "translation")?;
                        let rotation = read_quat(&value, "rotation")?;
                        let scale = read_vec3(&value, "scale")?;
                        let transform = Transform::new(translation, rotation, scale)
                            .map_err(mlua::Error::external)?;
                        set_commands
                            .try_borrow_mut()
                            .map_err(|_| {
                                mlua::Error::RuntimeError(
                                    "entity command buffer is already borrowed".into(),
                                )
                            })?
                            .insert(target, transform);
                        Ok(())
                    })
                    .map_err(|error| self.execution_error(phase, error))?,
            )
            .map_err(|error| self.execution_error(phase, error))?;

        let prefabs = self
            .lua
            .create_table()
            .map_err(|error| self.execution_error(phase, error))?;
        let prefab_registry = Rc::clone(&self.prefab_registry);
        let prefab_commands = Rc::clone(&commands);
        let prefab_ticket_handles = Rc::clone(&prefab_tickets);
        let prefab_active = Rc::clone(&active);
        prefabs
            .set(
                "instantiate",
                self.lua
                    .create_function(move |lua, name: String| {
                        ensure_context_active(&prefab_active)?;
                        let prefab = prefab_registry.get(&name).cloned().ok_or_else(|| {
                            mlua::Error::RuntimeError(format!("prefab is not registered: {name}"))
                        })?;
                        let pending = prefab_commands
                            .try_borrow_mut()
                            .map_err(|_| {
                                mlua::Error::RuntimeError(
                                    "entity command buffer is already borrowed".into(),
                                )
                            })?
                            .instantiate_prefab(prefab, PrefabOverrides::default())
                            .map_err(mlua::Error::external)?;
                        let state =
                            Rc::new(RefCell::new(Some(PrefabTicketState::Pending(pending))));
                        prefab_ticket_handles
                            .try_borrow_mut()
                            .map_err(|_| {
                                mlua::Error::RuntimeError(
                                    "prefab ticket registry is already borrowed".into(),
                                )
                            })?
                            .push(Rc::clone(&state));
                        lua.create_userdata(ScriptPrefabTicket { state })
                    })
                    .map_err(|error| self.execution_error(phase, error))?,
            )
            .map_err(|error| self.execution_error(phase, error))?;

        let prefab_entity_active = Rc::clone(&active);
        prefabs
            .set(
                "entity",
                self.lua
                    .create_function(move |lua, (ticket, source_id): (AnyUserData, u64)| {
                        ensure_context_active(&prefab_entity_active)?;
                        let ticket = ticket.borrow::<ScriptPrefabTicket>()?;
                        let state = ticket.state.try_borrow().map_err(|_| {
                            mlua::Error::RuntimeError("prefab ticket is already borrowed".into())
                        })?;
                        match state.as_ref() {
                            Some(PrefabTicketState::Ready(instance)) => {
                                let source_id =
                                    SceneEntityId::new(source_id).map_err(mlua::Error::external)?;
                                let entity = instance.entity(source_id).ok_or_else(|| {
                                    mlua::Error::RuntimeError(
                                        "source entity is not part of this prefab".into(),
                                    )
                                })?;
                                lua.create_userdata(ScriptEntity {
                                    target: Rc::new(RefCell::new(Some(EntityTarget::Existing(
                                        entity,
                                    )))),
                                })
                            }
                            Some(PrefabTicketState::Pending(_)) => Err(mlua::Error::RuntimeError(
                                "prefab ticket resolves after this callback commits".into(),
                            )),
                            None => Err(mlua::Error::RuntimeError(
                                "prefab ticket is no longer valid".into(),
                            )),
                        }
                    })
                    .map_err(|error| self.execution_error(phase, error))?,
            )
            .map_err(|error| self.execution_error(phase, error))?;

        let prefab_destroy_commands = Rc::clone(&commands);
        let prefab_destroyed = Rc::clone(&destroyed_prefabs);
        let prefab_destroy_active = Rc::clone(&active);
        prefabs
            .set(
                "destroy",
                self.lua
                    .create_function(move |_, ticket: AnyUserData| {
                        ensure_context_active(&prefab_destroy_active)?;
                        let ticket = ticket.borrow::<ScriptPrefabTicket>()?;
                        let state = Rc::clone(&ticket.state);
                        let instance = {
                            let borrowed = state.try_borrow().map_err(|_| {
                                mlua::Error::RuntimeError(
                                    "prefab ticket is already borrowed".into(),
                                )
                            })?;
                            match borrowed.as_ref() {
                                Some(PrefabTicketState::Ready(instance)) => instance.clone(),
                                Some(PrefabTicketState::Pending(_)) => {
                                    return Err(mlua::Error::RuntimeError(
                                        "cannot destroy a prefab before its instantiation commits"
                                            .into(),
                                    ));
                                }
                                None => {
                                    return Err(mlua::Error::RuntimeError(
                                        "prefab ticket is no longer valid".into(),
                                    ));
                                }
                            }
                        };
                        let mut commands =
                            prefab_destroy_commands.try_borrow_mut().map_err(|_| {
                                mlua::Error::RuntimeError(
                                    "entity command buffer is already borrowed".into(),
                                )
                            })?;
                        for (_, entity) in instance.entities() {
                            commands.despawn(EntityTarget::Existing(entity));
                        }
                        drop(commands);
                        prefab_destroyed
                            .try_borrow_mut()
                            .map_err(|_| {
                                mlua::Error::RuntimeError(
                                    "prefab ticket registry is already borrowed".into(),
                                )
                            })?
                            .push(state);
                        Ok(())
                    })
                    .map_err(|error| self.execution_error(phase, error))?,
            )
            .map_err(|error| self.execution_error(phase, error))?;

        let components = self
            .lua
            .create_table()
            .map_err(|error| self.execution_error(phase, error))?;

        let component_get_snapshot = Rc::clone(&component_snapshot);
        let component_get_alive = Rc::clone(&alive_entities);
        let component_get_ids = Rc::clone(&registered_component_ids);
        let component_get_active = Rc::clone(&active);
        components
            .set(
                "get",
                self.lua
                    .create_function(move |lua, (entity, component_id): (AnyUserData, String)| {
                        ensure_context_active(&component_get_active)?;
                        if !component_get_ids.contains(&component_id) {
                            return Err(mlua::Error::RuntimeError(format!(
                                "component is not registered with this script: {component_id}"
                            )));
                        }
                        match entity_target(&entity)? {
                            EntityTarget::Existing(entity) => {
                                if !component_get_alive.contains(&entity) {
                                    return Err(mlua::Error::RuntimeError(
                                        "stale entity handle".into(),
                                    ));
                                }
                                match component_get_snapshot.get(&(entity, component_id)) {
                                    Some(value) => Ok(lua.to_value(value)?),
                                    None => Ok(Value::Nil),
                                }
                            }
                            EntityTarget::Pending(_) => Ok(Value::Nil),
                        }
                    })
                    .map_err(|error| self.execution_error(phase, error))?,
            )
            .map_err(|error| self.execution_error(phase, error))?;

        let component_set_commands = Rc::clone(&commands);
        let component_set_registry = self.component_registry.clone();
        let component_set_ids = Rc::clone(&registered_component_ids);
        let component_set_active = Rc::clone(&active);
        components.set("set", self.lua.create_function(move |lua, (entity, component_id, value): (AnyUserData, String, Value)| {
            ensure_context_active(&component_set_active)?;
            if !component_set_ids.contains(&component_id) {
                return Err(mlua::Error::RuntimeError(format!("component is not registered with this script: {component_id}")));
            }
            let target = entity_target(&entity)?;
            let value = lua.from_value::<serde_json::Value>(value)?;
            let binding = component_set_registry.get(component_id.as_str())
                .ok_or_else(|| mlua::Error::RuntimeError(format!("component is not registered with this script: {component_id}")))?;
            component_set_commands.try_borrow_mut()
                .map_err(|_| mlua::Error::RuntimeError("entity command buffer is already borrowed".into()))
                .and_then(|mut commands| (binding.queue_insert)(&mut commands, target, value)
                    .map_err(mlua::Error::RuntimeError))?;
            Ok(())
        }).map_err(|error| self.execution_error(phase, error))?)
            .map_err(|error| self.execution_error(phase, error))?;

        let component_remove_commands = Rc::clone(&commands);
        let component_remove_registry = self.component_registry.clone();
        let component_remove_ids = Rc::clone(&registered_component_ids);
        let component_remove_active = Rc::clone(&active);
        components
            .set(
                "remove",
                self.lua
                    .create_function(move |_, (entity, component_id): (AnyUserData, String)| {
                        ensure_context_active(&component_remove_active)?;
                        if !component_remove_ids.contains(&component_id) {
                            return Err(mlua::Error::RuntimeError(format!(
                                "component is not registered with this script: {component_id}"
                            )));
                        }
                        let target = entity_target(&entity)?;
                        let binding = component_remove_registry
                            .get(component_id.as_str())
                            .ok_or_else(|| {
                                mlua::Error::RuntimeError(format!(
                                    "component is not registered with this script: {component_id}"
                                ))
                            })?;
                        let mut commands =
                            component_remove_commands.try_borrow_mut().map_err(|_| {
                                mlua::Error::RuntimeError(
                                    "entity command buffer is already borrowed".into(),
                                )
                            })?;
                        (binding.queue_remove)(&mut commands, target);
                        Ok(())
                    })
                    .map_err(|error| self.execution_error(phase, error))?,
            )
            .map_err(|error| self.execution_error(phase, error))?;

        api.set("entities", entities)
            .map_err(|error| self.execution_error(phase, error))?;
        api.set("transforms", transforms)
            .map_err(|error| self.execution_error(phase, error))?;
        api.set("components", components)
            .map_err(|error| self.execution_error(phase, error))?;
        api.set("prefabs", prefabs)
            .map_err(|error| self.execution_error(phase, error))?;
        Ok(CallbackCommands {
            api,
            commands,
            handles,
            prefab_tickets,
            destroyed_prefabs,
            active,
        })
    }

    fn apply_command_context(
        &self,
        world: &mut World,
        phase: ScriptPhase,
        context: CallbackCommands,
    ) -> Result<(), ScriptError> {
        let commands = std::mem::take(&mut *context.commands.borrow_mut());
        match commands.apply(world) {
            Ok(results) => {
                resolve_pending_handles(&context.handles, &results);
                resolve_prefab_tickets(&context.prefab_tickets, &results);
                invalidate_destroyed_prefabs(&context.destroyed_prefabs);
                Ok(())
            }
            Err(error) => {
                if let Some(results) = error.partial_results() {
                    resolve_pending_handles(&context.handles, results);
                    resolve_prefab_tickets(&context.prefab_tickets, results);
                } else {
                    invalidate_pending_handles(&context.handles);
                    invalidate_prefab_tickets(&context.prefab_tickets);
                }
                invalidate_destroyed_prefabs(&context.destroyed_prefabs);
                Err(ScriptError::CommandApplication {
                    source_name: self.source_name.clone(),
                    phase,
                    message: error.to_string(),
                })
            }
        }
    }

    fn execution_error(&self, phase: ScriptPhase, error: mlua::Error) -> ScriptError {
        ScriptError::Execution {
            source_name: self.source_name.clone(),
            phase,
            message: error.to_string(),
        }
    }

    fn require_state(
        &self,
        operation: &'static str,
        required: ScriptState,
    ) -> Result<(), ScriptError> {
        if self.state == required {
            return Ok(());
        }
        Err(ScriptError::InvalidLifecycle {
            source_name: self.source_name.clone(),
            operation,
            state: self.state,
        })
    }

    fn call<A>(&self, phase: ScriptPhase, callback: &Function, args: A) -> Result<(), ScriptError>
    where
        A: mlua::IntoLuaMulti,
    {
        self.executed_instructions.store(0, Ordering::Relaxed);
        callback
            .call::<()>(args)
            .map_err(|error| ScriptError::Execution {
                source_name: self.source_name.clone(),
                phase,
                message: error.to_string(),
            })?;
        let _keep_vm_alive = &self.lua;
        Ok(())
    }

    /// Returns the configured per-callback instruction budget.
    #[must_use]
    pub const fn instruction_budget(&self) -> u64 {
        self.instruction_budget
    }
}

type EntityHandleCell = Rc<RefCell<Option<EntityTarget>>>;
type EntityHandleRegistry = Rc<RefCell<Vec<EntityHandleCell>>>;
type PrefabTicketCell = Rc<RefCell<Option<PrefabTicketState>>>;
type PrefabTicketRegistry = Rc<RefCell<Vec<PrefabTicketCell>>>;

struct ScriptEntity {
    target: EntityHandleCell,
}

impl UserData for ScriptEntity {}

enum PrefabTicketState {
    Pending(PendingPrefab),
    Ready(PrefabInstance),
}

struct ScriptPrefabTicket {
    state: PrefabTicketCell,
}

impl UserData for ScriptPrefabTicket {}

struct CallbackCommands {
    api: Table,
    commands: Rc<RefCell<Commands>>,
    handles: EntityHandleRegistry,
    prefab_tickets: PrefabTicketRegistry,
    destroyed_prefabs: PrefabTicketRegistry,
    active: Rc<Cell<bool>>,
}

fn ensure_context_active(active: &Cell<bool>) -> mlua::Result<()> {
    if active.get() {
        Ok(())
    } else {
        Err(mlua::Error::RuntimeError(
            "engine callback API has expired".into(),
        ))
    }
}

fn entity_target(entity: &AnyUserData) -> mlua::Result<EntityTarget> {
    let entity = entity.borrow::<ScriptEntity>()?;
    entity
        .target
        .try_borrow()
        .map_err(|_| mlua::Error::RuntimeError("entity handle is already borrowed".into()))?
        .clone()
        .ok_or_else(|| mlua::Error::RuntimeError("entity handle is no longer valid".into()))
}

fn invalidate_pending_handles(handles: &RefCell<Vec<EntityHandleCell>>) {
    if let Ok(handles) = handles.try_borrow() {
        for handle in handles.iter() {
            if let Ok(mut target) = handle.try_borrow_mut()
                && matches!(*target, Some(EntityTarget::Pending(_)))
            {
                *target = None;
            }
        }
    }
}

fn invalidate_prefab_tickets(tickets: &RefCell<Vec<PrefabTicketCell>>) {
    if let Ok(tickets) = tickets.try_borrow() {
        for ticket in tickets.iter() {
            if let Ok(mut state) = ticket.try_borrow_mut()
                && matches!(*state, Some(PrefabTicketState::Pending(_)))
            {
                *state = None;
            }
        }
    }
}

fn resolve_prefab_tickets(tickets: &RefCell<Vec<PrefabTicketCell>>, results: &CommandResults) {
    if let Ok(tickets) = tickets.try_borrow() {
        for ticket in tickets.iter() {
            if let Ok(mut state) = ticket.try_borrow_mut()
                && let Some(PrefabTicketState::Pending(pending)) = state.as_ref()
            {
                *state = results
                    .prefab(pending)
                    .cloned()
                    .map(PrefabTicketState::Ready);
            }
        }
    }
}

fn invalidate_destroyed_prefabs(tickets: &RefCell<Vec<PrefabTicketCell>>) {
    if let Ok(tickets) = tickets.try_borrow() {
        for ticket in tickets.iter() {
            if let Ok(mut state) = ticket.try_borrow_mut() {
                *state = None;
            }
        }
    }
}

fn resolve_pending_handles(handles: &RefCell<Vec<EntityHandleCell>>, results: &CommandResults) {
    if let Ok(handles) = handles.try_borrow() {
        for handle in handles.iter() {
            if let Ok(mut target) = handle.try_borrow_mut()
                && let Some(EntityTarget::Pending(pending)) = target.as_ref()
            {
                *target = results.entity(pending).map(EntityTarget::Existing);
            }
        }
    }
}

fn validate_event(name: &str, payload: &serde_json::Value) -> Result<(), ScriptError> {
    if name.trim().is_empty() || name.len() > 128 {
        return Err(ScriptError::InvalidEventName);
    }
    let actual = serde_json::to_vec(payload)
        .map_err(|error| ScriptError::EventSerialization(error.to_string()))?
        .len();
    if actual > MAX_SCRIPT_EVENT_BYTES {
        return Err(ScriptError::EventPayloadTooLarge {
            actual,
            maximum: MAX_SCRIPT_EVENT_BYTES,
        });
    }
    let mut pending = vec![(payload, 0_usize)];
    while let Some((value, depth)) = pending.pop() {
        if depth > MAX_SCRIPT_EVENT_DEPTH {
            return Err(ScriptError::EventPayloadTooDeep);
        }
        match value {
            serde_json::Value::Array(values) => {
                pending.extend(values.iter().map(|value| (value, depth + 1)));
            }
            serde_json::Value::Object(values) => {
                pending.extend(values.values().map(|value| (value, depth + 1)));
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_delta_time(delta_seconds: f64) -> Result<(), ScriptError> {
    if delta_seconds.is_finite() && delta_seconds >= 0.0 {
        Ok(())
    } else {
        Err(ScriptError::InvalidDeltaTime)
    }
}

fn read_array<const N: usize>(table: &Table) -> mlua::Result<[f32; N]> {
    if table.raw_len() != N {
        return Err(mlua::Error::RuntimeError(format!(
            "expected an array with {N} numbers"
        )));
    }
    let mut values = [0.0; N];
    for (index, value) in values.iter_mut().enumerate() {
        *value = table.raw_get(index + 1)?;
    }
    Ok(values)
}

fn read_vec3(table: &Table, field: &str) -> mlua::Result<Vec3> {
    let values = read_array::<3>(&table.get::<Table>(field)?)?;
    Ok(Vec3::new(values[0], values[1], values[2]))
}

fn read_quat(table: &Table, field: &str) -> mlua::Result<Quat> {
    let values = read_array::<4>(&table.get::<Table>(field)?)?;
    Ok(Quat::from_xyzw(values[0], values[1], values[2], values[3]))
}

fn transform_to_lua(lua: &Lua, transform: Transform) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    let translation = transform.translation().to_array();
    let rotation = transform.rotation().to_array();
    let scale = transform.scale().to_array();
    table.set("translation", lua.create_sequence_from(translation)?)?;
    table.set("rotation", lua.create_sequence_from(rotation)?)?;
    table.set("scale", lua.create_sequence_from(scale)?)?;
    Ok(table)
}

fn read_callback(
    script: &Table,
    callback: &'static str,
    source_name: &str,
) -> Result<Option<Function>, ScriptError> {
    match script
        .get::<Value>(callback)
        .map_err(|error| ScriptError::Execution {
            source_name: source_name.to_owned(),
            phase: ScriptPhase::Load,
            message: error.to_string(),
        })? {
        Value::Nil => Ok(None),
        Value::Function(function) => Ok(Some(function)),
        _ => Err(ScriptError::InvalidCallback {
            source_name: source_name.to_owned(),
            callback,
        }),
    }
}

#[cfg(test)]
mod tests;
