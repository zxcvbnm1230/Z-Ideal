//! Fixed-step orchestration for scene scripts and physics.
use std::{
    collections::{HashMap, HashSet},
    time::Duration,
};

use super::ScriptSet;
use serde_json::{Value as JsonValue, json};
use thiserror::Error;
use z_ideal_assets::AssetUri;
use z_ideal_math::Transform;
use z_ideal_physics::{
    FixedStepAccumulator, FixedStepConfig, FixedStepError, PhysicsScene, PhysicsSceneError,
    PhysicsSceneEvent, PhysicsWorldConfig,
};
use z_ideal_scene::{Collider, Entity, RigidBody, World, WorldError};
use z_ideal_scripting::{
    ScriptConfig, ScriptDiagnostic, ScriptError, ScriptHost, ScriptHostError, ScriptId,
    ScriptInstance, ScriptRegistryError,
};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RuntimeSimulationConfig {
    /// Box3D world configuration.
    pub physics: PhysicsWorldConfig,
    /// Engine fixed-step accumulation and catch-up policy.
    pub fixed_step: FixedStepConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimePhysicsOperation {
    /// Synchronize scene components before a fixed step.
    SynchronizeBeforeStep,
    /// Advance the physics backend by one fixed step.
    Step,
    /// Write simulated transforms back to the scene.
    Writeback,
    /// Translate physics events to scene entities.
    ReadEvents,
    /// Synchronize scene changes made during the variable update.
    SynchronizeAfterFrame,
}

#[derive(Debug, Error)]
pub enum RuntimeSimulationError {
    #[error("physics fixed step {physics:?} does not match runtime fixed step {runtime:?}")]
    FixedStepDurationMismatch {
        /// Physics backend fixed-step duration.
        physics: Duration,
        /// Runtime accumulator fixed-step duration.
        runtime: Duration,
    },
    #[error(transparent)]
    FixedStep(#[from] FixedStepError),
    #[error(transparent)]
    Scene(#[from] WorldError),
    #[error("failed to initialize runtime physics scene: {0}")]
    PhysicsInitialization(#[source] z_ideal_physics::PhysicsError),
    #[error(
        "runtime physics operation {operation:?} failed after {completed_steps} of {planned_steps} planned steps: {source}"
    )]
    Physics {
        /// Operation that failed.
        operation: RuntimePhysicsOperation,
        /// Number of fixed steps planned for this frame.
        planned_steps: u32,
        /// Number of fully completed fixed steps in the frame.
        completed_steps: u32,
        #[source]
        source: PhysicsSceneError,
    },
    #[error(transparent)]
    Script(#[from] ScriptHostError),
    #[error(transparent)]
    ScriptRegistry(#[from] ScriptRegistryError),
    #[error(transparent)]
    ScriptLoad(#[from] ScriptError),
    #[error(transparent)]
    ScriptPackage(#[from] super::RuntimeScriptPackageError),
    #[error("entity {entity:?} references script {uri}, but no script package is mounted")]
    MissingScriptPackage {
        /// Entity with the script reference.
        entity: Entity,
        /// Missing package script URI.
        uri: AssetUri,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeFrameReport {
    /// Number of completed fixed simulation steps.
    pub fixed_steps: u32,
    /// Fractional progress toward the next fixed step.
    pub interpolation_alpha: f64,
    /// Whole fixed-step time discarded by the catch-up policy.
    pub dropped_time: Duration,
    /// Physics events generated during this frame, in deterministic order.
    pub physics_events: Vec<PhysicsSceneEvent>,
    /// Script errors and warnings produced during callback dispatch.
    pub script_diagnostics: Vec<ScriptDiagnostic>,
}

pub struct RuntimeSimulation {
    world: World,
    physics: PhysicsScene,
    scripts: ScriptHost,
    fixed_steps: FixedStepAccumulator,
    script_package: Option<super::RuntimeScriptPackage>,
    package_attachments: HashMap<(Entity, AssetUri), ScriptId>,
}

impl RuntimeSimulation {
    pub fn new(config: RuntimeSimulationConfig) -> Result<Self, RuntimeSimulationError> {
        if config.physics.fixed_step != config.fixed_step.step_duration() {
            return Err(RuntimeSimulationError::FixedStepDurationMismatch {
                physics: config.physics.fixed_step,
                runtime: config.fixed_step.step_duration(),
            });
        }
        let mut world = World::new();
        world.register_component::<Transform>()?;
        world.register_component::<RigidBody>()?;
        world.register_component::<Collider>()?;
        world.register_component::<ScriptSet>()?;
        let physics = PhysicsScene::new(config.physics)
            .map_err(RuntimeSimulationError::PhysicsInitialization)?;
        Ok(Self {
            world,
            physics,
            scripts: ScriptHost::new(),
            fixed_steps: FixedStepAccumulator::new(config.fixed_step),
            script_package: None,
            package_attachments: HashMap::new(),
        })
    }

    #[must_use]
    pub const fn world(&self) -> &World {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    #[must_use]
    pub const fn scripts(&self) -> &ScriptHost {
        &self.scripts
    }

    pub fn scripts_mut(&mut self) -> &mut ScriptHost {
        &mut self.scripts
    }

    #[must_use]
    pub const fn physics_scene(&self) -> &PhysicsScene {
        &self.physics
    }

    pub fn physics_scene_mut(&mut self) -> &mut PhysicsScene {
        &mut self.physics
    }

    pub fn attach_script_asset(
        &mut self,
        entity: Entity,
        asset: &z_ideal_assets::ScriptAsset,
        config: ScriptConfig,
    ) -> Result<ScriptId, RuntimeSimulationError> {
        let mut instance = ScriptInstance::load_asset(asset, config)?;
        instance.register_component::<Transform>()?;
        instance.register_component::<RigidBody>()?;
        instance.register_component::<Collider>()?;
        Ok(self.scripts.attach(&self.world, entity, instance)?)
    }

    pub fn attach_script_instance(
        &mut self,
        entity: Entity,
        instance: ScriptInstance,
    ) -> Result<ScriptId, RuntimeSimulationError> {
        Ok(self.scripts.attach(&self.world, entity, instance)?)
    }

    /// Loads a verified source asset from a cooked package and attaches it.
    ///
    /// The package manifest and every source file are verified when the package
    /// is opened; the selected source is re-verified immediately before loading.
    pub fn attach_script_from_package(
        &mut self,
        entity: Entity,
        package: &super::RuntimeScriptPackage,
        uri: &AssetUri,
        config: ScriptConfig,
    ) -> Result<ScriptId, RuntimeSimulationError> {
        let asset = package.load_asset(uri)?;
        self.attach_script_asset(entity, &asset, config)
    }

    /// Mounts a verified script package for scene-driven ScriptSet components.
    ///
    /// Existing package-managed scripts are shut down and detached before the
    /// new package becomes active. Direct ScriptHost attachments are unaffected.
    pub fn mount_script_package(
        &mut self,
        package: super::RuntimeScriptPackage,
    ) -> Result<(), RuntimeSimulationError> {
        self.detach_package_attachments()?;
        self.script_package = Some(package);
        Ok(())
    }

    /// Unmounts the package after orderly shutdown of its managed attachments.
    pub fn unmount_script_package(&mut self) -> Result<(), RuntimeSimulationError> {
        self.detach_package_attachments()?;
        self.script_package = None;
        Ok(())
    }

    fn detach_package_attachments(&mut self) -> Result<(), RuntimeSimulationError> {
        let mut attachments: Vec<_> = self
            .package_attachments
            .iter()
            .map(|(key, id)| (key.clone(), *id))
            .collect();
        attachments.sort_by(
            |((left_entity, left_uri), _), ((right_entity, right_uri), _)| {
                left_entity
                    .index()
                    .cmp(&right_entity.index())
                    .then_with(|| left_entity.generation().cmp(&right_entity.generation()))
                    .then_with(|| left_uri.cmp(right_uri))
            },
        );
        for (key, id) in attachments {
            let (entity, _) = &key;
            if self.world.is_alive(*entity) && self.scripts.attached_entity(id).is_some() {
                self.scripts.detach(&mut self.world, id)?;
            }
            self.package_attachments.remove(&key);
        }
        Ok(())
    }

    fn synchronize_script_attachments(&mut self) -> Result<(), RuntimeSimulationError> {
        let mut desired = Vec::new();
        let mut desired_set = HashSet::new();
        for entity in self.world.entities_with::<ScriptSet>() {
            if let Some(set) = self.world.get::<ScriptSet>(entity) {
                for reference in &set.scripts {
                    if reference.enabled {
                        let key = (entity, reference.uri.clone());
                        if desired_set.insert(key.clone()) {
                            desired.push(key);
                        }
                    }
                }
            }
        }

        let stale: Vec<_> = self
            .package_attachments
            .keys()
            .filter(|key| !desired_set.contains(*key))
            .cloned()
            .collect();
        let mut stale = stale;
        stale.sort_by(|(left_entity, left_uri), (right_entity, right_uri)| {
            left_entity
                .index()
                .cmp(&right_entity.index())
                .then_with(|| left_entity.generation().cmp(&right_entity.generation()))
                .then_with(|| left_uri.cmp(right_uri))
        });
        for key in stale {
            if let Some(id) = self.package_attachments.get(&key).copied() {
                let (entity, _) = &key;
                if self.world.is_alive(*entity) && self.scripts.attached_entity(id).is_some() {
                    self.scripts.detach(&mut self.world, id)?;
                }
                self.package_attachments.remove(&key);
            }
        }

        for (entity, uri) in desired {
            if let Some(id) = self
                .package_attachments
                .get(&(entity, uri.clone()))
                .copied()
                && self.scripts.attached_entity(id) == Some(entity)
            {
                continue;
            }
            self.package_attachments.remove(&(entity, uri.clone()));
            let package = self.script_package.as_ref().ok_or_else(|| {
                RuntimeSimulationError::MissingScriptPackage {
                    entity,
                    uri: uri.clone(),
                }
            })?;
            let asset = package.load_asset(&uri)?;
            let id = self.attach_script_asset(entity, &asset, ScriptConfig::default())?;
            self.package_attachments.insert((entity, uri), id);
        }
        Ok(())
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.fixed_steps.set_paused(paused);
        self.scripts.set_paused(paused);
    }

    #[must_use]
    pub const fn is_paused(&self) -> bool {
        self.fixed_steps.is_paused()
    }

    pub fn dispatch_event(&mut self, name: &str, payload: &JsonValue) -> Vec<ScriptDiagnostic> {
        let diagnostics = self
            .scripts
            .dispatch_event_all(&mut self.world, name, payload);
        log_script_diagnostics(&diagnostics);
        diagnostics
    }

    pub fn advance(
        &mut self,
        frame_delta: Duration,
    ) -> Result<RuntimeFrameReport, RuntimeSimulationError> {
        self.synchronize_script_attachments()?;
        let mut diagnostics = Vec::new();
        append_script_diagnostics(&mut diagnostics, self.scripts.start_all(&mut self.world));
        let plan = self.fixed_steps.advance(frame_delta)?;
        let fixed_delta_seconds = self.fixed_steps.config().step_duration().as_secs_f64();
        let mut completed_steps = 0;
        let mut physics_events = Vec::new();

        for _ in 0..plan.steps {
            append_script_diagnostics(
                &mut diagnostics,
                self.scripts
                    .fixed_update_all(&mut self.world, fixed_delta_seconds),
            );
            self.physics
                .synchronize(&mut self.world)
                .map_err(|source| RuntimeSimulationError::Physics {
                    operation: RuntimePhysicsOperation::SynchronizeBeforeStep,
                    planned_steps: plan.steps,
                    completed_steps,
                    source,
                })?;
            self.physics
                .step()
                .map_err(|source| RuntimeSimulationError::Physics {
                    operation: RuntimePhysicsOperation::Step,
                    planned_steps: plan.steps,
                    completed_steps,
                    source: source.into(),
                })?;
            self.physics
                .writeback_dynamic_transforms(&mut self.world)
                .map_err(|source| RuntimeSimulationError::Physics {
                    operation: RuntimePhysicsOperation::Writeback,
                    planned_steps: plan.steps,
                    completed_steps,
                    source,
                })?;
            let step_events =
                self.physics
                    .events()
                    .map_err(|source| RuntimeSimulationError::Physics {
                        operation: RuntimePhysicsOperation::ReadEvents,
                        planned_steps: plan.steps,
                        completed_steps,
                        source,
                    })?;
            for event in &step_events {
                let (name, payload) = physics_event_payload(*event);
                append_script_diagnostics(
                    &mut diagnostics,
                    self.scripts
                        .dispatch_event_all(&mut self.world, name, &payload),
                );
            }
            physics_events.extend(step_events);
            completed_steps += 1;
        }

        append_script_diagnostics(
            &mut diagnostics,
            self.scripts
                .update_all(&mut self.world, frame_delta.as_secs_f64()),
        );
        self.physics
            .synchronize(&mut self.world)
            .map_err(|source| RuntimeSimulationError::Physics {
                operation: RuntimePhysicsOperation::SynchronizeAfterFrame,
                planned_steps: plan.steps,
                completed_steps,
                source,
            })?;

        Ok(RuntimeFrameReport {
            fixed_steps: completed_steps,
            interpolation_alpha: plan.interpolation_alpha,
            dropped_time: plan.dropped_time,
            physics_events,
            script_diagnostics: diagnostics,
        })
    }

    pub fn shutdown(&mut self) -> Vec<ScriptDiagnostic> {
        let diagnostics = self.scripts.shutdown_all(&mut self.world);
        log_script_diagnostics(&diagnostics);
        diagnostics
    }
}

fn physics_event_payload(event: PhysicsSceneEvent) -> (&'static str, JsonValue) {
    let entity =
        |entity: Entity| json!({ "index": entity.index(), "generation": entity.generation() });
    match event {
        PhysicsSceneEvent::ContactBegin { entity_a, entity_b } => (
            "physics.contact_begin",
            json!({ "entity_a": entity(entity_a), "entity_b": entity(entity_b) }),
        ),
        PhysicsSceneEvent::ContactEnd { entity_a, entity_b } => (
            "physics.contact_end",
            json!({ "entity_a": entity(entity_a), "entity_b": entity(entity_b) }),
        ),
        PhysicsSceneEvent::SensorBegin { sensor, visitor } => (
            "physics.sensor_begin",
            json!({ "sensor": entity(sensor), "visitor": entity(visitor) }),
        ),
        PhysicsSceneEvent::SensorEnd { sensor, visitor } => (
            "physics.sensor_end",
            json!({ "sensor": entity(sensor), "visitor": entity(visitor) }),
        ),
    }
}

fn append_script_diagnostics(accumulated: &mut Vec<ScriptDiagnostic>, next: Vec<ScriptDiagnostic>) {
    log_script_diagnostics(&next);
    accumulated.extend(next);
}

fn log_script_diagnostics(diagnostics: &[ScriptDiagnostic]) {
    for diagnostic in diagnostics {
        tracing::error!(
            target: "z_ideal::script",
            script_id = diagnostic.script_id.get(),
            entity_index = diagnostic.entity.index(),
            entity_generation = diagnostic.entity.generation(),
            phase = ?diagnostic.phase,
            message = %diagnostic.message,
            "script callback failed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::time::Duration;
    use z_ideal_math::Vec3;
    use z_ideal_scene::{Component, RigidBodyType};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Counter {
        value: u32,
    }

    impl Component for Counter {
        const ID: &'static str = "test.runtime_simulation_counter";
    }

    fn config(step: Duration) -> RuntimeSimulationConfig {
        RuntimeSimulationConfig {
            physics: PhysicsWorldConfig {
                fixed_step: step,
                ..PhysicsWorldConfig::default()
            },
            fixed_step: FixedStepConfig::new(step, 8).expect("fixed-step policy"),
        }
    }

    fn counter_script() -> ScriptInstance {
        let mut script = ScriptInstance::load(
            "counter.lua",
            r#"
                return {
                    start = function(self, engine)
                        self.fixed_count = 0
                    end,
                    fixed_update = function(self, dt, engine)
                        self.fixed_count = self.fixed_count + 1
                    end,
                    update = function(self, dt, engine)
                        engine.components.set(
                            engine.entity,
                            "test.runtime_simulation_counter",
                            { value = self.fixed_count }
                        )
                    end
                }
            "#,
            ScriptConfig::default(),
        )
        .expect("load counter script");
        script
            .register_component::<Counter>()
            .expect("register script component");
        script
    }

    #[test]
    fn rejects_mismatched_physics_and_script_step_durations() {
        assert!(matches!(
            RuntimeSimulation::new(RuntimeSimulationConfig {
                physics: PhysicsWorldConfig::default(),
                fixed_step: FixedStepConfig::new(Duration::from_millis(10), 8)
                    .expect("valid accumulator"),
            }),
            Err(RuntimeSimulationError::FixedStepDurationMismatch { .. })
        ));
    }

    #[test]
    fn advances_script_lifecycle_using_bounded_fixed_steps() {
        let mut simulation =
            RuntimeSimulation::new(config(Duration::from_millis(10))).expect("simulation");
        let entity = simulation.world_mut().spawn().expect("spawn");
        simulation
            .world_mut()
            .register_component::<Counter>()
            .expect("register Counter");
        simulation
            .world_mut()
            .insert(entity, Counter { value: 0 })
            .expect("insert Counter");
        simulation
            .attach_script_instance(entity, counter_script())
            .expect("attach script");

        let report = simulation
            .advance(Duration::from_millis(55))
            .expect("advance frame");
        assert_eq!(report.fixed_steps, 5);
        assert_eq!(report.dropped_time, Duration::ZERO);
        assert!((report.interpolation_alpha - 0.5).abs() < 1.0e-12);
        assert!(report.script_diagnostics.is_empty());
        assert_eq!(
            simulation.world().get::<Counter>(entity),
            Some(&Counter { value: 5 })
        );
    }

    #[test]
    fn synchronizes_and_steps_scene_physics_with_the_same_fixed_clock() {
        let mut simulation =
            RuntimeSimulation::new(config(Duration::from_millis(10))).expect("simulation");
        let entity = simulation.world_mut().spawn().expect("spawn");
        simulation
            .world_mut()
            .insert(
                entity,
                Transform::new(
                    Vec3::new(0.0, 10.0, 0.0),
                    z_ideal_math::Quat::IDENTITY,
                    Vec3::ONE,
                )
                .expect("transform"),
            )
            .expect("insert transform");
        simulation
            .world_mut()
            .insert(entity, RigidBody::default())
            .expect("insert rigid body");
        simulation
            .world_mut()
            .insert(entity, Collider::sphere(0.5))
            .expect("insert collider");

        let report = simulation
            .advance(Duration::from_millis(100))
            .expect("advance physics");
        assert_eq!(report.fixed_steps, 8);
        assert_eq!(report.dropped_time, Duration::from_millis(20));
        assert_eq!(simulation.physics_scene().body_count(), 1);
        let transform = simulation
            .world()
            .get::<Transform>(entity)
            .expect("dynamic transform");
        assert!(transform.translation().y < 10.0);
    }

    #[test]
    fn pause_suppresses_fixed_and_variable_script_updates() {
        let mut simulation =
            RuntimeSimulation::new(config(Duration::from_millis(10))).expect("simulation");
        let entity = simulation.world_mut().spawn().expect("spawn");
        simulation
            .world_mut()
            .register_component::<Counter>()
            .expect("register Counter");
        simulation
            .world_mut()
            .insert(entity, Counter { value: 0 })
            .expect("insert Counter");
        simulation
            .attach_script_instance(entity, counter_script())
            .expect("attach script");

        simulation.set_paused(true);
        let paused = simulation
            .advance(Duration::from_millis(50))
            .expect("paused frame");
        assert_eq!(paused.fixed_steps, 0);
        assert_eq!(paused.dropped_time, Duration::from_millis(50));
        assert_eq!(
            simulation.world().get::<Counter>(entity),
            Some(&Counter { value: 0 })
        );

        simulation.set_paused(false);
        let resumed = simulation
            .advance(Duration::from_millis(10))
            .expect("resumed frame");
        assert_eq!(resumed.fixed_steps, 1);
        assert_eq!(
            simulation.world().get::<Counter>(entity),
            Some(&Counter { value: 1 })
        );
    }

    #[test]
    fn dispatches_external_events_and_runs_shutdown() {
        let mut simulation =
            RuntimeSimulation::new(config(Duration::from_millis(10))).expect("simulation");
        let entity = simulation.world_mut().spawn().expect("spawn");
        simulation
            .world_mut()
            .register_component::<Counter>()
            .expect("register Counter");
        simulation
            .world_mut()
            .insert(entity, Counter { value: 0 })
            .expect("insert Counter");
        let mut script = ScriptInstance::load(
            "events.lua",
            r#"
                return {
                    on_event = function(self, name, payload, engine)
                        engine.components.set(
                            engine.entity,
                            "test.runtime_simulation_counter",
                            { value = payload.value }
                        )
                    end,
                    shutdown = function(self, engine)
                        engine.components.set(
                            engine.entity,
                            "test.runtime_simulation_counter",
                            { value = 99 }
                        )
                    end
                }
            "#,
            ScriptConfig::default(),
        )
        .expect("load event script");
        script
            .register_component::<Counter>()
            .expect("register component");
        simulation
            .attach_script_instance(entity, script)
            .expect("attach script");
        simulation.advance(Duration::ZERO).expect("start script");

        let diagnostics = simulation.dispatch_event("test.changed", &json!({ "value": 7 }));
        assert!(diagnostics.is_empty());
        assert_eq!(
            simulation.world().get::<Counter>(entity),
            Some(&Counter { value: 7 })
        );
        assert!(simulation.shutdown().is_empty());
        assert_eq!(
            simulation.world().get::<Counter>(entity),
            Some(&Counter { value: 99 })
        );
    }

    #[test]
    fn reports_script_errors_without_aborting_the_frame() {
        let mut simulation =
            RuntimeSimulation::new(config(Duration::from_millis(10))).expect("simulation");
        let failing_entity = simulation.world_mut().spawn().expect("spawn failing");
        let healthy_entity = simulation.world_mut().spawn().expect("spawn healthy");
        let failing = ScriptInstance::load(
            "failing.lua",
            "return { update = function() error('expected') end }",
            ScriptConfig::default(),
        )
        .expect("load failing script");
        let healthy = counter_script();
        simulation
            .attach_script_instance(failing_entity, failing)
            .expect("attach failing");
        simulation
            .world_mut()
            .register_component::<Counter>()
            .expect("register Counter");
        simulation
            .world_mut()
            .insert(healthy_entity, Counter { value: 0 })
            .expect("insert Counter");
        simulation
            .attach_script_instance(healthy_entity, healthy)
            .expect("attach healthy");

        let report = simulation
            .advance(Duration::ZERO)
            .expect("script errors are diagnostic");
        assert_eq!(report.script_diagnostics.len(), 1);
        assert_eq!(report.script_diagnostics[0].entity, failing_entity);
        assert_eq!(
            simulation.world().get::<Counter>(healthy_entity),
            Some(&Counter { value: 0 })
        );
    }

    #[test]
    fn dispatches_sensor_events_to_scripts_with_generation_safe_entity_data() {
        let mut simulation =
            RuntimeSimulation::new(config(Duration::from_millis(10))).expect("simulation");
        let sensor = simulation.world_mut().spawn().expect("sensor entity");
        let visitor = simulation.world_mut().spawn().expect("visitor entity");
        simulation
            .world_mut()
            .register_component::<Counter>()
            .expect("register Counter");
        simulation
            .world_mut()
            .insert(sensor, Counter { value: 0 })
            .expect("insert Counter");
        let mut sensor_body = RigidBody {
            body_type: RigidBodyType::Static,
            ..RigidBody::default()
        };
        sensor_body.linear_velocity = Vec3::ZERO;
        let mut sensor_collider = Collider::sphere(1.0);
        sensor_collider.sensor = true;
        sensor_collider.sensor_events = true;
        let mut visitor_collider = Collider::sphere(0.5);
        visitor_collider.sensor_events = true;
        simulation
            .world_mut()
            .insert(sensor, sensor_body)
            .expect("insert sensor body");
        simulation
            .world_mut()
            .insert(sensor, sensor_collider)
            .expect("insert sensor collider");
        simulation
            .world_mut()
            .insert(visitor, RigidBody::default())
            .expect("insert visitor body");
        simulation
            .world_mut()
            .insert(visitor, visitor_collider)
            .expect("insert visitor collider");

        let mut script = ScriptInstance::load(
            "sensor.lua",
            r#"
                return {
                    on_event = function(self, name, payload, engine)
                        if name == "physics.sensor_begin" then
                            engine.components.set(
                                engine.entity,
                                "test.runtime_simulation_counter",
                                { value = payload.visitor.index }
                            )
                        end
                    end
                }
            "#,
            ScriptConfig::default(),
        )
        .expect("load sensor script");
        script
            .register_component::<Counter>()
            .expect("register component");
        simulation
            .attach_script_instance(sensor, script)
            .expect("attach sensor script");

        let report = simulation
            .advance(Duration::from_millis(10))
            .expect("advance one physics step");
        assert!(report.physics_events.iter().any(|event| matches!(
            event,
            PhysicsSceneEvent::SensorBegin {
                sensor: event_sensor,
                visitor: event_visitor,
            } if *event_sensor == sensor && *event_visitor == visitor
        )));
        assert_eq!(
            simulation.world().get::<Counter>(sensor),
            Some(&Counter {
                value: visitor.index(),
            })
        );
    }

    #[test]
    fn runtime_built_in_script_bindings_accept_physics_components() {
        let mut simulation =
            RuntimeSimulation::new(config(Duration::from_millis(10))).expect("simulation");
        let entity = simulation.world_mut().spawn().expect("spawn");
        let asset = z_ideal_assets::LuaScriptImporter::default()
            .import(
                z_ideal_assets::AssetUri::new("scripts/physics.lua").expect("URI"),
                br#"return { start = function(self, engine) engine.components.set(engine.entity, "z_ideal.physics.rigid_body", { body_type = "static", gravity_scale = 1.0, linear_damping = 0.0, angular_damping = 0.0, allow_sleep = true, bullet = false, linear_velocity = {0,0,0} }) end }"#,
            )
            .expect("import script");
        simulation
            .attach_script_asset(entity, &asset, ScriptConfig::default())
            .expect("attach imported script");
        let report = simulation
            .advance(Duration::ZERO)
            .expect("run script start");
        assert!(report.script_diagnostics.is_empty());
        assert_eq!(
            simulation
                .world()
                .get::<RigidBody>(entity)
                .map(|body| body.body_type),
            Some(RigidBodyType::Static)
        );
        assert_eq!(
            simulation
                .world()
                .get::<Collider>(entity)
                .map(|collider| collider.shape),
            None
        );
    }
}

#[cfg(test)]
mod package_attachment_tests {
    use super::*;
    use crate::{RuntimeScriptPackage, RuntimeScriptPackageError};
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };
    use z_ideal_assets::{AssetUri, ScriptBuildPipeline};
    use z_ideal_scripting::ScriptState;

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestTree(PathBuf);

    impl TestTree {
        fn new() -> Self {
            let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "z-ideal-runtime-package-attach-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create test tree");
            Self(path)
        }
    }

    impl Drop for TestTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn attaches_and_runs_a_verified_cooked_script() {
        let tree = TestTree::new();
        let source = tree.0.join("source");
        fs::create_dir_all(&source).expect("source directory");
        fs::write(
            source.join("player.lua"),
            "return { update = function(self, dt, engine) assert(engine.entity ~= nil) end }",
        )
        .expect("write script");
        let package_dir = tree.0.join("package");
        ScriptBuildPipeline::default()
            .cook(&source, &package_dir)
            .expect("cook package");
        let package = RuntimeScriptPackage::open(&package_dir).expect("open package");

        let mut simulation = RuntimeSimulation::new(RuntimeSimulationConfig {
            physics: PhysicsWorldConfig {
                fixed_step: Duration::from_millis(10),
                ..PhysicsWorldConfig::default()
            },
            fixed_step: FixedStepConfig::new(Duration::from_millis(10), 8).expect("fixed step"),
        })
        .expect("simulation");
        let entity = simulation.world_mut().spawn().expect("spawn entity");
        let script_id = simulation
            .attach_script_from_package(
                entity,
                &package,
                &AssetUri::new("player.lua").expect("script URI"),
                ScriptConfig::default(),
            )
            .expect("attach package script");
        assert_eq!(simulation.scripts().len(), 1);
        assert_eq!(
            simulation.scripts().attached_entity(script_id),
            Some(entity)
        );

        let report = simulation.advance(Duration::ZERO).expect("run script");
        assert!(report.script_diagnostics.is_empty());
        assert_eq!(
            simulation.scripts().script_state(script_id),
            Some(ScriptState::Started)
        );
        assert!(simulation.shutdown().is_empty());
    }

    #[test]
    fn rejects_tampered_package_source_before_script_attachment() {
        let tree = TestTree::new();
        let source = tree.0.join("source");
        fs::create_dir_all(&source).expect("source directory");
        fs::write(source.join("player.lua"), "return {}").expect("write script");
        let package_dir = tree.0.join("package");
        ScriptBuildPipeline::default()
            .cook(&source, &package_dir)
            .expect("cook package");
        let package = RuntimeScriptPackage::open(&package_dir).expect("open package");
        fs::write(
            package_dir.join("scripts/player.lua"),
            "return { update = function() end }",
        )
        .expect("tamper source");

        let mut simulation = RuntimeSimulation::new(RuntimeSimulationConfig {
            physics: PhysicsWorldConfig {
                fixed_step: Duration::from_millis(10),
                ..PhysicsWorldConfig::default()
            },
            fixed_step: FixedStepConfig::new(Duration::from_millis(10), 8).expect("fixed step"),
        })
        .expect("simulation");
        let entity = simulation.world_mut().spawn().expect("spawn entity");
        assert!(matches!(
            simulation.attach_script_from_package(
                entity,
                &package,
                &AssetUri::new("player.lua").expect("script URI"),
                ScriptConfig::default(),
            ),
            Err(RuntimeSimulationError::ScriptPackage(
                RuntimeScriptPackageError::Build(
                    z_ideal_assets::ScriptBuildError::SourceMetadataMismatch { .. }
                )
            ))
        ));
        assert!(simulation.scripts().is_empty());
    }
}

#[cfg(test)]
mod scene_script_attachment_tests {
    use super::*;
    use crate::{RuntimeScriptPackage, ScriptReference, ScriptSet};
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };
    use z_ideal_assets::{AssetUri, ScriptBuildPipeline};

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestTree(PathBuf);

    impl TestTree {
        fn new() -> Self {
            let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "z-ideal-runtime-scene-script-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create test tree");
            Self(path)
        }
    }

    impl Drop for TestTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn simulation() -> RuntimeSimulation {
        RuntimeSimulation::new(RuntimeSimulationConfig {
            physics: PhysicsWorldConfig {
                fixed_step: Duration::from_millis(10),
                ..PhysicsWorldConfig::default()
            },
            fixed_step: FixedStepConfig::new(Duration::from_millis(10), 8).expect("fixed step"),
        })
        .expect("simulation")
    }

    fn package(tree: &TestTree, source_text: &str) -> RuntimeScriptPackage {
        let source = tree.0.join("source");
        fs::create_dir_all(&source).expect("source directory");
        fs::write(source.join("actor.lua"), source_text).expect("write source");
        let package = tree.0.join("package");
        ScriptBuildPipeline::default()
            .cook(&source, &package)
            .expect("cook package");
        RuntimeScriptPackage::open(package).expect("open package")
    }

    #[test]
    fn scene_script_set_attaches_in_order_and_shutdown_detaches_resources() {
        let tree = TestTree::new();
        let package = package(
            &tree,
            r#"
                return {
                    start = function(self, engine)
                        self.spawned = engine.entities.spawn()
                    end,
                    shutdown = function(self, engine)
                        engine.entities.destroy(self.spawned)
                    end
                }
            "#,
        );
        let mut simulation = simulation();
        simulation
            .mount_script_package(package)
            .expect("mount package");
        let entity = simulation.world_mut().spawn().expect("spawn owner");
        simulation
            .world_mut()
            .insert(
                entity,
                ScriptSet::new(vec![ScriptReference::new(
                    AssetUri::new("actor.lua").expect("URI"),
                )])
                .expect("script set"),
            )
            .expect("insert ScriptSet");

        let started = simulation.advance(Duration::ZERO).expect("start script");
        assert!(started.script_diagnostics.is_empty());
        assert_eq!(simulation.scripts().len(), 1);
        assert_eq!(simulation.world().entities().len(), 2);

        simulation
            .world_mut()
            .remove::<ScriptSet>(entity)
            .expect("remove ScriptSet");
        let detached = simulation
            .advance(Duration::ZERO)
            .expect("detach removed script");
        assert!(detached.script_diagnostics.is_empty());
        assert!(simulation.scripts().is_empty());
        assert_eq!(simulation.world().entities(), vec![entity]);
    }

    #[test]
    fn enabled_scene_script_requires_a_mounted_package() {
        let mut simulation = simulation();
        let entity = simulation.world_mut().spawn().expect("spawn");
        simulation
            .world_mut()
            .insert(
                entity,
                ScriptSet::new(vec![ScriptReference::new(
                    AssetUri::new("actor.lua").expect("URI"),
                )])
                .expect("script set"),
            )
            .expect("insert ScriptSet");
        assert!(matches!(
            simulation.advance(Duration::ZERO),
            Err(RuntimeSimulationError::MissingScriptPackage {
                entity: missing_entity,
                ..
            }) if missing_entity == entity
        ));
        assert!(simulation.scripts().is_empty());
    }

    #[test]
    fn disabled_scene_script_does_not_require_a_package_or_attach() {
        let mut simulation = simulation();
        let entity = simulation.world_mut().spawn().expect("spawn");
        simulation
            .world_mut()
            .insert(
                entity,
                ScriptSet::new(vec![ScriptReference::with_enabled(
                    AssetUri::new("actor.lua").expect("URI"),
                    false,
                )])
                .expect("script set"),
            )
            .expect("insert ScriptSet");
        let report = simulation.advance(Duration::ZERO).expect("advance");
        assert!(report.script_diagnostics.is_empty());
        assert!(simulation.scripts().is_empty());
    }

    #[test]
    fn changing_scene_script_reference_detaches_previous_instance() {
        let tree = TestTree::new();
        let package = package(
            &tree,
            "return { start = function(self, engine) self.started = true end }",
        );
        let mut simulation = simulation();
        simulation.mount_script_package(package).expect("mount");
        let entity = simulation.world_mut().spawn().expect("spawn");
        let reference = |uri: &str| {
            ScriptSet::new(vec![ScriptReference::new(AssetUri::new(uri).expect("URI"))])
                .expect("script set")
        };
        simulation
            .world_mut()
            .insert(entity, reference("actor.lua"))
            .expect("insert script set");
        simulation.advance(Duration::ZERO).expect("attach");
        assert_eq!(simulation.scripts().len(), 1);

        simulation
            .world_mut()
            .insert(entity, ScriptSet::default())
            .expect("clear script references");
        simulation.advance(Duration::ZERO).expect("detach");
        assert!(simulation.scripts().is_empty());
    }
}

#[cfg(test)]
mod script_set_order_tests {
    use super::*;
    use crate::{RuntimeScriptPackage, ScriptReference, ScriptSet};
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };
    use z_ideal_assets::ScriptBuildPipeline;
    use z_ideal_math::Vec3;

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn scripts_execute_in_scene_reference_order() {
        let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "z-ideal-script-order-{}-{sequence}",
            std::process::id()
        ));
        let source = root.join("source");
        let package_dir = root.join("package");
        fs::create_dir_all(&source).expect("source directory");
        fs::write(
            source.join("first.lua"),
            r#"
                return {
                    update = function(self, dt, engine)
                        engine.transforms.set(engine.entity, {
                            translation = {1, 0, 0},
                            rotation = {0, 0, 0, 1},
                            scale = {1, 1, 1}
                        })
                    end
                }
            "#,
        )
        .expect("write first script");
        fs::write(
            source.join("second.lua"),
            r#"
                return {
                    update = function(self, dt, engine)
                        local transform = engine.transforms.get(engine.entity)
                        assert(transform.translation[1] == 1)
                        transform.translation[1] = 2
                        engine.transforms.set(engine.entity, transform)
                    end
                }
            "#,
        )
        .expect("write second script");
        ScriptBuildPipeline::default()
            .cook(&source, &package_dir)
            .expect("cook scripts");
        let package = RuntimeScriptPackage::open(&package_dir).expect("open package");
        let mut simulation = RuntimeSimulation::new(RuntimeSimulationConfig {
            physics: PhysicsWorldConfig {
                fixed_step: Duration::from_millis(10),
                ..PhysicsWorldConfig::default()
            },
            fixed_step: FixedStepConfig::new(Duration::from_millis(10), 8).expect("fixed step"),
        })
        .expect("simulation");
        simulation
            .mount_script_package(package)
            .expect("mount package");
        let entity = simulation.world_mut().spawn().expect("spawn entity");
        simulation
            .world_mut()
            .insert(entity, Transform::default())
            .expect("insert transform");
        simulation
            .world_mut()
            .insert(
                entity,
                ScriptSet::new(vec![
                    ScriptReference::new(
                        z_ideal_assets::AssetUri::new("first.lua").expect("first URI"),
                    ),
                    ScriptReference::new(
                        z_ideal_assets::AssetUri::new("second.lua").expect("second URI"),
                    ),
                ])
                .expect("script set"),
            )
            .expect("insert script set");

        let report = simulation.advance(Duration::ZERO).expect("advance scripts");
        assert!(report.script_diagnostics.is_empty());
        assert_eq!(
            simulation
                .world()
                .get::<Transform>(entity)
                .expect("updated transform")
                .translation(),
            Vec3::new(2.0, 0.0, 0.0)
        );
        let _ = fs::remove_dir_all(&root);
    }
}
