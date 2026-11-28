//! Deterministic ownership and dispatch for scripts attached to scene entities.

use std::collections::BTreeMap;

use serde_json::Value as JsonValue;
use z_ideal_scene::{Entity, World};

use crate::{ScriptError, ScriptInstance, ScriptPhase, ScriptState};

/// Stable insertion-order identifier for one script attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScriptId(u64);

impl ScriptId {
    /// Returns the stable numeric attachment identifier.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Diagnostic produced when one attached script fails a lifecycle operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptDiagnostic {
    /// Attachment that produced the diagnostic.
    pub script_id: ScriptId,
    /// Entity to which the script is attached.
    pub entity: Entity,
    /// Lifecycle operation that failed.
    pub phase: ScriptPhase,
    /// Human-readable source-aware error.
    pub message: String,
}

/// Errors returned by script attachment management.
#[derive(Debug, thiserror::Error)]
pub enum ScriptHostError {
    /// An attachment target is not alive in the supplied world.
    #[error("cannot attach script to stale entity {0:?}")]
    StaleEntity(Entity),
    /// The script is not in the Loaded state when attached.
    #[error("only a Loaded script can be attached; current state is {0:?}")]
    InvalidScriptState(ScriptState),
    /// Attachment identifiers are exhausted.
    #[error("script attachment identifier space is exhausted")]
    IdExhausted,
    /// The requested attachment does not exist.
    #[error("script attachment {0:?} does not exist")]
    UnknownScript(ScriptId),
    /// Script binding or shutdown failed.
    #[error(transparent)]
    Script(#[from] ScriptError),
}

struct AttachedScript {
    entity: Entity,
    instance: ScriptInstance,
}

/// Owns script instances attached to entities and dispatches their lifecycle.
///
/// Scripts are visited in attachment insertion order. Multiple scripts may be
/// attached to one entity. A callback failure is recorded and does not prevent
/// other scripts from running. A failed script remains attached; the host/editor
/// decides whether to retry, inspect or detach it.
#[derive(Default)]
pub struct ScriptHost {
    scripts: BTreeMap<ScriptId, AttachedScript>,
    next_id: u64,
    paused: bool,
}

impl ScriptHost {
    /// Creates an empty script host.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Attaches a Loaded script instance to a live entity in this world.
    ///
    /// The script is bound to this World's opaque identity at attachment time.
    pub fn attach(
        &mut self,
        world: &World,
        entity: Entity,
        mut instance: ScriptInstance,
    ) -> Result<ScriptId, ScriptHostError> {
        if !world.is_alive(entity) {
            return Err(ScriptHostError::StaleEntity(entity));
        }
        if instance.state() != ScriptState::Loaded {
            return Err(ScriptHostError::InvalidScriptState(instance.state()));
        }
        instance.bind_entity(world, entity)?;
        instance.set_paused(self.paused);

        let next_id = self
            .next_id
            .checked_add(1)
            .ok_or(ScriptHostError::IdExhausted)?;
        let id = ScriptId(next_id);
        self.next_id = next_id;
        self.scripts.insert(id, AttachedScript { entity, instance });
        Ok(id)
    }

    /// Detaches one script after successful shutdown.
    ///
    /// If shutdown fails, the attachment is retained so the caller can inspect
    /// the error and retry orderly teardown.
    pub fn detach(&mut self, world: &mut World, id: ScriptId) -> Result<(), ScriptHostError> {
        let attached = self
            .scripts
            .get_mut(&id)
            .ok_or(ScriptHostError::UnknownScript(id))?;
        if attached.instance.state() != ScriptState::Shutdown {
            attached.instance.shutdown_with_world(world)?;
        }
        self.scripts.remove(&id);
        Ok(())
    }

    /// Returns the entity associated with an attachment.
    #[must_use]
    pub fn attached_entity(&self, id: ScriptId) -> Option<Entity> {
        self.scripts.get(&id).map(|attached| attached.entity)
    }

    /// Returns the lifecycle state for an attachment.
    #[must_use]
    pub fn script_state(&self, id: ScriptId) -> Option<ScriptState> {
        self.scripts
            .get(&id)
            .map(|attached| attached.instance.state())
    }

    /// Returns the number of attached scripts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scripts.len()
    }

    /// Returns whether the host has no attached scripts.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scripts.is_empty()
    }

    /// Pauses or resumes fixed and variable updates for every attached script.
    ///
    /// Event and shutdown callbacks remain available while paused.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
        for attached in self.scripts.values_mut() {
            attached.instance.set_paused(paused);
        }
    }

    /// Returns whether fixed and variable updates are suppressed.
    #[must_use]
    pub const fn is_paused(&self) -> bool {
        self.paused
    }

    /// Starts all Loaded scripts in insertion order.
    pub fn start_all(&mut self, world: &mut World) -> Vec<ScriptDiagnostic> {
        let mut diagnostics = self.prune_stale(world);
        for (id, attached) in &mut self.scripts {
            if !world.is_alive(attached.entity) {
                continue;
            }
            if attached.instance.state() == ScriptState::Loaded
                && let Err(error) = attached.instance.start_with_world(world)
            {
                diagnostics.push(diagnostic(*id, attached.entity, ScriptPhase::Start, error));
            }
        }
        diagnostics.extend(self.prune_stale(world));
        diagnostics
    }

    /// Runs one fixed simulation step for every Started script.
    pub fn fixed_update_all(
        &mut self,
        world: &mut World,
        fixed_delta_seconds: f64,
    ) -> Vec<ScriptDiagnostic> {
        let mut diagnostics = self.prune_stale(world);
        for (id, attached) in &mut self.scripts {
            if !world.is_alive(attached.entity) {
                continue;
            }
            if attached.instance.state() == ScriptState::Started
                && let Err(error) = attached
                    .instance
                    .fixed_update_with_world(world, fixed_delta_seconds)
            {
                diagnostics.push(diagnostic(
                    *id,
                    attached.entity,
                    ScriptPhase::FixedUpdate,
                    error,
                ));
            }
        }
        diagnostics.extend(self.prune_stale(world));
        diagnostics
    }

    /// Runs one frame update for every Started script.
    pub fn update_all(&mut self, world: &mut World, delta_seconds: f64) -> Vec<ScriptDiagnostic> {
        let mut diagnostics = self.prune_stale(world);
        for (id, attached) in &mut self.scripts {
            if !world.is_alive(attached.entity) {
                continue;
            }
            if attached.instance.state() == ScriptState::Started
                && let Err(error) = attached.instance.update_with_world(world, delta_seconds)
            {
                diagnostics.push(diagnostic(*id, attached.entity, ScriptPhase::Update, error));
            }
        }
        diagnostics.extend(self.prune_stale(world));
        diagnostics
    }

    /// Dispatches one JSON event to every Started script in insertion order.
    pub fn dispatch_event_all(
        &mut self,
        world: &mut World,
        name: &str,
        payload: &JsonValue,
    ) -> Vec<ScriptDiagnostic> {
        let mut diagnostics = self.prune_stale(world);
        for (id, attached) in &mut self.scripts {
            if !world.is_alive(attached.entity) {
                continue;
            }
            if attached.instance.state() == ScriptState::Started
                && let Err(error) = attached
                    .instance
                    .dispatch_event_with_world(world, name, payload)
            {
                diagnostics.push(diagnostic(*id, attached.entity, ScriptPhase::Event, error));
            }
        }
        diagnostics.extend(self.prune_stale(world));
        diagnostics
    }

    /// Shuts down every non-shutdown script without detaching it.
    pub fn shutdown_all(&mut self, world: &mut World) -> Vec<ScriptDiagnostic> {
        let mut diagnostics = self.prune_stale(world);
        for (id, attached) in &mut self.scripts {
            if !world.is_alive(attached.entity) {
                continue;
            }
            if attached.instance.state() != ScriptState::Shutdown
                && let Err(error) = attached.instance.shutdown_with_world(world)
            {
                diagnostics.push(diagnostic(
                    *id,
                    attached.entity,
                    ScriptPhase::Shutdown,
                    error,
                ));
            }
        }
        diagnostics.extend(self.prune_stale(world));
        diagnostics
    }

    fn prune_stale(&mut self, world: &mut World) -> Vec<ScriptDiagnostic> {
        let stale: Vec<_> = self
            .scripts
            .iter()
            .filter_map(|(id, attached)| (!world.is_alive(attached.entity)).then_some(*id))
            .collect();
        let mut diagnostics = Vec::new();
        for id in stale {
            if let Some(mut attached) = self.scripts.remove(&id)
                && attached.instance.state() != ScriptState::Shutdown
                && let Err(error) = attached.instance.shutdown_with_world(world)
            {
                diagnostics.push(diagnostic(
                    id,
                    attached.entity,
                    ScriptPhase::Shutdown,
                    error,
                ));
            }
        }
        diagnostics
    }
}

fn diagnostic(
    script_id: ScriptId,
    entity: Entity,
    phase: ScriptPhase,
    error: ScriptError,
) -> ScriptDiagnostic {
    ScriptDiagnostic {
        script_id,
        entity,
        phase,
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{ScriptHost, ScriptHostError, ScriptId};
    use crate::{ScriptConfig, ScriptInstance};
    use serde::{Deserialize, Serialize};
    use z_ideal_scene::{Component, World};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Counter {
        value: u32,
    }

    impl Component for Counter {
        const ID: &'static str = "test.script_host_counter";
    }

    fn world_with_entities(count: usize) -> (World, Vec<z_ideal_scene::Entity>) {
        let mut world = World::new();
        world
            .register_component::<Counter>()
            .expect("register counter");
        let mut entities = Vec::new();
        for _ in 0..count {
            let entity = world.spawn().expect("spawn entity");
            world
                .insert(entity, Counter { value: 0 })
                .expect("insert counter");
            entities.push(entity);
        }
        (world, entities)
    }

    fn script(source: &str) -> ScriptInstance {
        let mut script =
            ScriptInstance::load("host.lua", source, ScriptConfig::default()).expect("load script");
        script
            .register_component::<Counter>()
            .expect("register counter binding");
        script
    }

    #[test]
    fn multiple_scripts_on_one_entity_run_in_attachment_order() {
        let (mut world, entities) = world_with_entities(1);
        let entity = entities[0];
        let mut host = ScriptHost::new();
        let first = script(
            r#"
            return {
                update = function(self, dt, engine)
                    engine.components.set(engine.entity, "test.script_host_counter", { value = 2 })
                end
            }
        "#,
        );
        let second = script(
            r#"
            return {
                update = function(self, dt, engine)
                    local value = engine.components.get(engine.entity, "test.script_host_counter")
                    assert(value.value == 2)
                    value.value = 3
                    engine.components.set(engine.entity, "test.script_host_counter", value)
                end
            }
        "#,
        );
        let first_id = host.attach(&world, entity, first).expect("attach first");
        let second_id = host.attach(&world, entity, second).expect("attach second");
        assert_eq!(host.attached_entity(first_id), Some(entity));
        assert_eq!(host.attached_entity(second_id), Some(entity));
        assert!(host.start_all(&mut world).is_empty());
        assert!(host.update_all(&mut world, 0.016).is_empty());
        assert_eq!(world.get::<Counter>(entity), Some(&Counter { value: 3 }));
    }

    #[test]
    fn one_script_error_does_not_prevent_other_scripts_from_updating() {
        let (mut world, entities) = world_with_entities(2);
        let mut host = ScriptHost::new();
        host.attach(
            &world,
            entities[0],
            script("return { update = function() error('expected failure') end }"),
        )
        .expect("attach failing script");
        host.attach(
            &world,
            entities[1],
            script(r#"
                return {
                    update = function(self, dt, engine)
                        engine.components.set(engine.entity, "test.script_host_counter", { value = 9 })
                    end
                }
            "#),
        ).expect("attach working script");
        assert!(host.start_all(&mut world).is_empty());
        let diagnostics = host.update_all(&mut world, 0.016);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].entity, entities[0]);
        assert_eq!(
            world.get::<Counter>(entities[1]),
            Some(&Counter { value: 9 })
        );
    }

    #[test]
    fn paused_host_suppresses_updates_but_still_dispatches_events() {
        let (mut world, entities) = world_with_entities(1);
        let entity = entities[0];
        let mut host = ScriptHost::new();
        host.attach(
            &world,
            entity,
            script(r#"
                return {
                    update = function(self, dt, engine)
                        engine.components.set(engine.entity, "test.script_host_counter", { value = 5 })
                    end,
                    on_event = function(self, name, payload, engine)
                        engine.components.set(engine.entity, "test.script_host_counter", { value = 8 })
                    end
                }
            "#),
        ).expect("attach script");
        assert!(host.start_all(&mut world).is_empty());
        host.set_paused(true);
        assert!(host.is_paused());
        assert!(host.update_all(&mut world, 0.016).is_empty());
        assert_eq!(world.get::<Counter>(entity), Some(&Counter { value: 0 }));
        assert!(
            host.dispatch_event_all(&mut world, "changed", &serde_json::json!({}))
                .is_empty()
        );
        assert_eq!(world.get::<Counter>(entity), Some(&Counter { value: 8 }));
    }

    #[test]
    fn despawned_entity_automatically_detaches_its_scripts() {
        let (mut world, entities) = world_with_entities(1);
        let mut host = ScriptHost::new();
        host.attach(&world, entities[0], script("return {}"))
            .expect("attach");
        assert!(host.start_all(&mut world).is_empty());
        world
            .despawn(entities[0])
            .expect("despawn attachment target");
        assert!(host.update_all(&mut world, 0.016).is_empty());
        assert!(host.is_empty());
    }

    #[test]
    fn attachment_rejects_stale_entities_and_non_loaded_instances() {
        let (mut world, entities) = world_with_entities(1);
        let mut host = ScriptHost::new();
        let stale = entities[0];
        world.despawn(stale).expect("despawn");
        assert!(matches!(
            host.attach(&world, stale, script("return {}")),
            Err(ScriptHostError::StaleEntity(_))
        ));

        let mut started = script("return {}");
        started.start().expect("start without host");
        let other = world.spawn().expect("spawn another");
        assert!(matches!(
            host.attach(&world, other, started),
            Err(ScriptHostError::InvalidScriptState(_))
        ));
    }

    #[test]
    fn despawning_attachment_stops_later_scripts_on_that_entity() {
        let (mut world, entities) = world_with_entities(1);
        let mut host = ScriptHost::new();
        host.attach(
                &world,
                entities[0],
                script("return { update = function(self, dt, engine) engine.entities.destroy(engine.entity) end }"),
            ).expect("attach destroyer");
        host.attach(
            &world,
            entities[0],
            script("return { update = function() error('must not run after despawn') end }"),
        )
        .expect("attach observer");
        assert!(host.start_all(&mut world).is_empty());
        assert!(host.update_all(&mut world, 0.016).is_empty());
        assert!(host.is_empty());
        assert!(!world.is_alive(entities[0]));
    }

    #[test]
    fn detach_runs_shutdown_and_removes_attachment_only_after_success() {
        let (mut world, entities) = world_with_entities(1);
        let entity = entities[0];
        let mut host = ScriptHost::new();
        let instance = script(
            r#"
                return {
                    shutdown = function(self, engine)
                        engine.components.set(engine.entity, "test.script_host_counter", { value = 14 })
                    end
                }
            "#,
        );
        let id = host.attach(&world, entity, instance).expect("attach");
        assert!(host.start_all(&mut world).is_empty());
        host.detach(&mut world, id).expect("detach");
        assert!(host.is_empty());
        assert_eq!(world.get::<Counter>(entity), Some(&Counter { value: 14 }));
    }

    #[test]
    fn host_fixed_update_shutdown_and_state_queries_cover_full_lifecycle() {
        let (mut world, entities) = world_with_entities(1);
        let entity = entities[0];
        let mut host = ScriptHost::new();
        let id = host.attach(
            &world,
            entity,
            script(r#"
                return {
                    fixed_update = function(self, dt, engine)
                        engine.components.set(engine.entity, "test.script_host_counter", { value = 6 })
                    end,
                    shutdown = function(self, engine)
                        engine.components.set(engine.entity, "test.script_host_counter", { value = 11 })
                    end
                }
            "#),
        ).expect("attach");
        assert_eq!(host.len(), 1);
        assert!(id.get() > 0);
        assert_eq!(host.script_state(id), Some(crate::ScriptState::Loaded));
        assert!(host.start_all(&mut world).is_empty());
        assert_eq!(host.script_state(id), Some(crate::ScriptState::Started));
        assert!(host.fixed_update_all(&mut world, 0.02).is_empty());
        assert_eq!(world.get::<Counter>(entity), Some(&Counter { value: 6 }));
        assert!(host.shutdown_all(&mut world).is_empty());
        assert_eq!(host.script_state(id), Some(crate::ScriptState::Shutdown));
        assert_eq!(world.get::<Counter>(entity), Some(&Counter { value: 11 }));
        assert!(host.shutdown_all(&mut world).is_empty());
    }

    #[test]
    fn failed_shutdown_keeps_attachment_available_for_retry() {
        let (mut world, entities) = world_with_entities(1);
        let mut host = ScriptHost::new();
        let id = host
            .attach(
                &world,
                entities[0],
                script("return { shutdown = function() error('shutdown failed') end }"),
            )
            .expect("attach");
        assert!(host.start_all(&mut world).is_empty());
        let diagnostics = host.shutdown_all(&mut world);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(host.script_state(id), Some(crate::ScriptState::Started));
        assert_eq!(host.len(), 1);
        assert!(matches!(
            host.detach(&mut world, id),
            Err(ScriptHostError::Script(_))
        ));
        assert_eq!(host.len(), 1);
    }

    #[test]
    fn host_reports_unknown_attachments_and_identifier_exhaustion() {
        let (mut world, entities) = world_with_entities(1);
        let mut host = ScriptHost::new();
        assert!(matches!(
            host.detach(&mut world, ScriptId(99)),
            Err(ScriptHostError::UnknownScript(_))
        ));
        host.next_id = u64::MAX;
        assert!(matches!(
            host.attach(&world, entities[0], script("return {}")),
            Err(ScriptHostError::IdExhausted)
        ));
        assert!(host.is_empty());
    }
}
