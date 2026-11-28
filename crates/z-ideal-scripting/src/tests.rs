pub(crate) use crate::*;

#[cfg(test)]
mod lifecycle_tests {
    use super::{
        INSTRUCTION_CHECK_INTERVAL, MAX_SCRIPT_SOURCE_BYTES, ScriptConfig, ScriptError,
        ScriptInstance, ScriptPhase, ScriptState,
    };

    fn load(source: &str) -> Result<ScriptInstance, ScriptError> {
        ScriptInstance::load("test.lua", source, ScriptConfig::default())
    }

    #[test]
    fn crate_name_and_api_version_are_stable() {
        assert_eq!(super::CRATE_NAME, "z-ideal-scripting");
        assert_eq!(super::SCRIPT_API_VERSION, 1);
    }

    #[test]
    fn loads_script_table_and_reports_source_name() {
        let script = load("return { value = 42 }").expect("load script");
        assert_eq!(script.source_name(), "test.lua");
        assert_eq!(script.state(), ScriptState::Loaded);
        assert_eq!(script.instruction_budget(), 100_000);
    }

    #[test]
    fn rejects_chunk_that_does_not_return_a_table() {
        assert!(matches!(
            load("return 42"),
            Err(ScriptError::MissingScriptTable { .. })
        ));
    }

    #[test]
    fn rejects_non_function_lifecycle_fields() {
        assert!(matches!(
            load("return { update = 12 }"),
            Err(ScriptError::InvalidCallback {
                callback: "update",
                ..
            })
        ));
    }

    #[test]
    fn rejects_source_larger_than_hard_limit() {
        let source = " ".repeat(MAX_SCRIPT_SOURCE_BYTES + 1);
        assert!(matches!(
            load(&source),
            Err(ScriptError::SourceTooLarge { .. })
        ));
    }

    #[test]
    fn rejects_too_small_instruction_budget() {
        assert!(matches!(
            ScriptInstance::load(
                "budget.lua",
                "return {}",
                ScriptConfig {
                    instruction_budget: u64::from(INSTRUCTION_CHECK_INTERVAL) - 1
                }
            ),
            Err(ScriptError::InvalidInstructionBudget)
        ));
    }

    #[test]
    fn start_update_and_shutdown_follow_canonical_order() {
        let source = r#"
            return {
                start = function(self) self.started = true end,
                update = function(self, dt)
                    assert(self.started)
                    self.delta = dt
                end,
                shutdown = function(self)
                    assert(self.delta == 0.25)
                    self.stopped = true
                end
            }
        "#;
        let mut script = load(source).expect("load script");
        script.start().expect("start");
        assert_eq!(script.state(), ScriptState::Started);
        script.update(0.25).expect("update");
        script.shutdown().expect("shutdown");
        assert_eq!(script.state(), ScriptState::Shutdown);
    }

    #[test]
    fn lifecycle_callbacks_are_optional() {
        let mut script = load("return {}").expect("load script");
        script.start().expect("start");
        script.update(0.0).expect("update");
        script.shutdown().expect("shutdown");
    }

    #[test]
    fn update_rejects_non_finite_and_negative_delta() {
        let mut script = load("return {}").expect("load script");
        script.start().expect("start");
        assert!(matches!(
            script.update(f64::NAN),
            Err(ScriptError::InvalidDeltaTime)
        ));
        assert!(matches!(
            script.update(-0.01),
            Err(ScriptError::InvalidDeltaTime)
        ));
    }

    #[test]
    fn lifecycle_calls_must_follow_state_machine() {
        let mut script = load("return {}").expect("load script");
        assert!(matches!(
            script.update(0.1),
            Err(ScriptError::InvalidLifecycle { .. })
        ));
        script.start().expect("start");
        script.shutdown().expect("shutdown");
        assert!(matches!(
            script.update(0.1),
            Err(ScriptError::InvalidLifecycle { .. })
        ));
    }

    #[test]
    fn script_runtime_error_is_reported_and_does_not_advance_lifecycle() {
        let mut script =
            load("return { start = function() error('start failed') end }").expect("load script");
        assert!(matches!(
            script.start(),
            Err(ScriptError::Execution {
                phase: ScriptPhase::Start,
                ..
            })
        ));
        assert_eq!(script.state(), ScriptState::Loaded);
    }

    #[test]
    fn instruction_budget_interrupts_infinite_loop() {
        let mut script = ScriptInstance::load(
            "loop.lua",
            "return { start = function() while true do end end }",
            ScriptConfig {
                instruction_budget: u64::from(INSTRUCTION_CHECK_INTERVAL) * 4,
            },
        )
        .expect("load loop script");
        assert!(matches!(
            script.start(),
            Err(ScriptError::Execution {
                phase: ScriptPhase::Start,
                ..
            })
        ));
        assert_eq!(script.state(), ScriptState::Loaded);
    }

    #[test]
    fn restricted_standard_library_excludes_system_and_dynamic_load_apis() {
        let mut script = load(
            r#"
            return {
                start = function(self)
                    self.blocked = {
                        os = os, io = io, package = package, debug = debug,
                        coroutine = coroutine, collectgarbage = collectgarbage,
                        dofile = dofile, load = load, loadfile = loadfile, require = require,
                        pcall = pcall, xpcall = xpcall, print = print, warn = warn
                    }
                end
            }
        "#,
        )
        .expect("load script");
        script.start().expect("start");
    }

    #[test]
    fn protected_calls_are_unavailable_so_budget_errors_cannot_be_swallowed() {
        let mut script =
            load("return { start = function() assert(pcall == nil and xpcall == nil) end }")
                .expect("load script");
        script.start().expect("start without protected calls");
    }

    #[test]
    fn top_level_chunk_is_also_instruction_limited() {
        let result = ScriptInstance::load(
            "top_level_loop.lua",
            "while true do end",
            ScriptConfig {
                instruction_budget: u64::from(INSTRUCTION_CHECK_INTERVAL) * 2,
            },
        );
        assert!(matches!(
            result,
            Err(ScriptError::Execution {
                phase: ScriptPhase::Load,
                ..
            })
        ));
    }

    #[test]
    fn shutdown_callback_failure_leaves_instance_started_for_diagnostics() {
        let mut script =
            load("return { shutdown = function() error('stop failed') end }").expect("load script");
        script.start().expect("start");
        assert!(matches!(
            script.shutdown(),
            Err(ScriptError::Execution {
                phase: ScriptPhase::Shutdown,
                ..
            })
        ));
        assert_eq!(script.state(), ScriptState::Started);
    }
}

#[cfg(test)]
mod scene_binding_tests {
    use super::{ScriptConfig, ScriptError, ScriptInstance, ScriptState};
    use z_ideal_scene::World;

    fn load(source: &str) -> ScriptInstance {
        ScriptInstance::load("scene_bindings.lua", source, ScriptConfig::default())
            .expect("load script")
    }

    #[test]
    fn script_can_spawn_parent_and_child_then_destroy_child_next_callback() {
        let mut script = load(
            r#"
            return {
                start = function(self, engine)
                    self.root = engine.entities.spawn()
                    self.child = engine.entities.spawn()
                    engine.entities.set_parent(self.child, self.root)
                end,
                update = function(self, dt, engine)
                    engine.entities.destroy(self.child)
                end
            }
        "#,
        );
        let mut world = World::new();
        script
            .start_with_world(&mut world)
            .expect("start with scene");
        let entities = world.entities();
        assert_eq!(entities.len(), 2);
        assert_eq!(world.parent_of(entities[1]), Some(entities[0]));

        script
            .update_with_world(&mut world, 1.0 / 60.0)
            .expect("update with scene");
        assert_eq!(world.entities(), vec![entities[0]]);
        assert_eq!(world.parent_of(entities[0]), None);
    }

    #[test]
    fn failed_lua_callback_discards_queued_scene_commands() {
        let mut script = load(
            r#"
            return {
                start = function(self, engine)
                    engine.entities.spawn()
                    error("abort callback")
                end
            }
        "#,
        );
        let mut world = World::new();
        assert!(matches!(
            script.start_with_world(&mut world),
            Err(ScriptError::Execution { .. })
        ));
        assert!(world.entities().is_empty());
        assert_eq!(script.state(), ScriptState::Loaded);
    }

    #[test]
    fn engine_api_tables_expire_after_the_callback_that_received_them() {
        let mut script = load(
            r#"
            return {
                start = function(self, engine)
                    self.saved_engine = engine
                end,
                update = function(self, dt, engine)
                    self.saved_engine.entities.spawn()
                end
            }
        "#,
        );
        let mut world = World::new();
        script.start_with_world(&mut world).expect("start");
        assert!(matches!(
            script.update_with_world(&mut world, 0.1),
            Err(ScriptError::Execution { .. })
        ));
        assert!(world.entities().is_empty());
    }

    #[test]
    fn script_instance_rejects_being_reused_with_a_different_world() {
        let mut script = load("return {}");
        let mut first = World::new();
        let mut second = World::new();
        script.start_with_world(&mut first).expect("start");
        assert!(matches!(
            script.update_with_world(&mut second, 0.1),
            Err(ScriptError::WorldMismatch { .. })
        ));
    }

    #[test]
    fn stale_entity_command_is_reported_after_fifo_partial_application() {
        let mut script = load(
            r#"
            return {
                start = function(self, engine)
                    self.entity = engine.entities.spawn()
                end,
                update = function(self, dt, engine)
                    engine.entities.destroy(self.entity)
                    engine.entities.destroy(self.entity)
                end
            }
        "#,
        );
        let mut world = World::new();
        script.start_with_world(&mut world).expect("start");
        assert_eq!(world.entities().len(), 1);
        assert!(matches!(
            script.update_with_world(&mut world, 0.1),
            Err(ScriptError::CommandApplication { .. })
        ));
        assert!(world.entities().is_empty());
        assert_eq!(script.state(), ScriptState::Started);
    }
}

#[cfg(test)]
mod transform_binding_tests {
    use super::{ScriptConfig, ScriptError, ScriptInstance};
    use z_ideal_math::{Quat, Transform, Vec3};
    use z_ideal_scene::World;

    fn load(source: &str) -> ScriptInstance {
        ScriptInstance::load("transform_bindings.lua", source, ScriptConfig::default())
            .expect("load script")
    }

    fn world_with_transforms() -> World {
        let mut world = World::new();
        world
            .register_component::<Transform>()
            .expect("register Transform");
        world
    }

    #[test]
    fn scripts_can_set_and_read_validated_local_transforms() {
        let mut script = load(
            r#"
            return {
                start = function(self, engine)
                    self.entity = engine.entities.spawn()
                    engine.transforms.set(self.entity, {
                        translation = {1, 2, 3},
                        rotation = {0, 0, 0, 1},
                        scale = {1, 1, 1}
                    })
                end,
                update = function(self, dt, engine)
                    local value = engine.transforms.get(self.entity)
                    assert(value.translation[1] == 1)
                    value.translation[1] = 4
                    engine.transforms.set(self.entity, value)
                end
            }
        "#,
        );
        let mut world = world_with_transforms();
        script.start_with_world(&mut world).expect("start");
        let entity = world.entities()[0];
        assert_eq!(
            world
                .get::<Transform>(entity)
                .expect("transform")
                .translation(),
            Vec3::new(1.0, 2.0, 3.0)
        );

        script
            .update_with_world(&mut world, 1.0 / 60.0)
            .expect("update");
        assert_eq!(
            world
                .get::<Transform>(entity)
                .expect("transform")
                .translation(),
            Vec3::new(4.0, 2.0, 3.0)
        );
        assert_eq!(
            world
                .get::<Transform>(entity)
                .expect("transform")
                .rotation(),
            Quat::IDENTITY
        );
    }

    #[test]
    fn invalid_transform_values_are_rejected_without_mutating_world() {
        let mut script = load(
            r#"
            return {
                start = function(self, engine)
                    self.entity = engine.entities.spawn()
                    engine.transforms.set(self.entity, {
                        translation = {0, 0, 0},
                        rotation = {0, 0, 0, 0},
                        scale = {1, 1, 1}
                    })
                end
            }
        "#,
        );
        let mut world = world_with_transforms();
        assert!(matches!(
            script.start_with_world(&mut world),
            Err(ScriptError::Execution { .. })
        ));
        assert!(world.entities().is_empty());
    }

    #[test]
    fn transform_read_rejects_stale_entity_handles() {
        let mut script = load(
            r#"
            return {
                start = function(self, engine)
                    self.entity = engine.entities.spawn()
                    engine.transforms.set(self.entity, {
                        translation = {0, 0, 0},
                        rotation = {0, 0, 0, 1},
                        scale = {1, 1, 1}
                    })
                end,
                update = function(self, dt, engine)
                    engine.transforms.get(self.entity)
                end
            }
        "#,
        );
        let mut world = world_with_transforms();
        script.start_with_world(&mut world).expect("start");
        let entity = world.entities()[0];
        world.despawn(entity).expect("external despawn");
        assert!(matches!(
            script.update_with_world(&mut world, 0.1),
            Err(ScriptError::Execution { .. })
        ));
    }
}

#[cfg(test)]
mod fixed_update_tests {
    use super::{ScriptConfig, ScriptError, ScriptInstance};

    #[test]
    fn fixed_update_runs_only_when_started_and_not_paused() {
        let mut script = ScriptInstance::load(
            "fixed.lua",
            r#"
                return {
                    fixed_update = function(self, dt)
                        self.steps = (self.steps or 0) + 1
                        self.fixed_dt = dt
                    end,
                    update = function(self, dt)
                        assert(self.steps == 1)
                        assert(self.fixed_dt == 0.02)
                    end
                }
            "#,
            ScriptConfig::default(),
        )
        .expect("load script");
        script.start().expect("start");
        script.set_paused(true);
        assert!(script.is_paused());
        script.fixed_update(0.02).expect("paused fixed update");
        script.update(0.1).expect("paused update");
        script.set_paused(false);
        assert!(!script.is_paused());
        script.fixed_update(0.02).expect("fixed update");
        script.update(0.1).expect("frame update");
    }

    #[test]
    fn fixed_update_rejects_invalid_delta_time() {
        let mut script = ScriptInstance::load(
            "fixed.lua",
            "return { fixed_update = function() end }",
            ScriptConfig::default(),
        )
        .expect("load script");
        script.start().expect("start");
        assert!(matches!(
            script.fixed_update(f64::INFINITY),
            Err(ScriptError::InvalidDeltaTime)
        ));
    }
}

#[cfg(test)]
mod event_tests {
    use super::{MAX_SCRIPT_EVENT_BYTES, ScriptConfig, ScriptError, ScriptInstance};
    use serde_json::json;
    use z_ideal_scene::World;

    fn load(source: &str) -> ScriptInstance {
        ScriptInstance::load("events.lua", source, ScriptConfig::default()).expect("load script")
    }

    #[test]
    fn event_callback_receives_json_payload_and_name() {
        let mut script = load(
            r#"
            return {
                on_event = function(self, name, payload)
                    assert(name == "score.changed")
                    assert(payload.value == 17)
                    self.received = true
                end,
                update = function(self, dt)
                    assert(self.received == true)
                end
            }
        "#,
        );
        script.start().expect("start");
        script
            .dispatch_event("score.changed", &json!({"value": 17}))
            .expect("dispatch");
        script.update(0.016).expect("verify event");
    }

    #[test]
    fn event_callback_can_queue_scene_commands() {
        let mut script = load(
            r#"
            return {
                on_event = function(self, name, payload, engine)
                    if name == "spawn.requested" then
                        self.spawned = engine.entities.spawn()
                    end
                end
            }
        "#,
        );
        let mut world = World::new();
        script.start_with_world(&mut world).expect("start");
        script
            .dispatch_event_with_world(&mut world, "spawn.requested", &json!({}))
            .expect("dispatch");
        assert_eq!(world.entities().len(), 1);
    }

    #[test]
    fn invalid_event_name_and_oversized_payload_are_rejected() {
        let mut script = load("return { on_event = function() end }");
        script.start().expect("start");
        assert!(matches!(
            script.dispatch_event("", &json!({})),
            Err(ScriptError::InvalidEventName)
        ));
        assert!(matches!(
            script.dispatch_event(&"n".repeat(129), &json!({})),
            Err(ScriptError::InvalidEventName)
        ));
        let payload = json!({"data": "x".repeat(MAX_SCRIPT_EVENT_BYTES)});
        assert!(matches!(
            script.dispatch_event("large", &payload),
            Err(ScriptError::EventPayloadTooLarge { .. })
        ));
    }

    #[test]
    fn event_callback_failure_discards_queued_commands() {
        let mut script = load(
            r#"
            return {
                on_event = function(self, name, payload, engine)
                    engine.entities.spawn()
                    error("event rejected")
                end
            }
        "#,
        );
        let mut world = World::new();
        script.start_with_world(&mut world).expect("start");
        assert!(matches!(
            script.dispatch_event_with_world(&mut world, "bad.event", &json!({})),
            Err(ScriptError::Execution { .. })
        ));
        assert!(world.entities().is_empty());
    }
}

#[cfg(test)]
mod generic_component_tests {
    use super::{ScriptConfig, ScriptError, ScriptInstance, ScriptRegistryError};
    use serde::{Deserialize, Serialize};
    use z_ideal_scene::{Component, World};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Health {
        current: u32,
    }

    impl Component for Health {
        const ID: &'static str = "test.script_health";
    }

    fn registered_world() -> World {
        let mut world = World::new();
        world
            .register_component::<Health>()
            .expect("register Health");
        world
    }

    fn load(source: &str) -> ScriptInstance {
        ScriptInstance::load("components.lua", source, ScriptConfig::default())
            .expect("load script")
    }

    #[test]
    fn scripts_can_read_write_and_remove_registered_components() {
        let mut script = load(
            r#"
            return {
                start = function(self, engine)
                    self.entity = engine.entities.spawn()
                    assert(engine.components.get(self.entity, "test.script_health") == nil)
                    engine.components.set(self.entity, "test.script_health", { current = 12 })
                end,
                update = function(self, dt, engine)
                    local health = engine.components.get(self.entity, "test.script_health")
                    assert(health.current == 12)
                    health.current = 19
                    engine.components.set(self.entity, "test.script_health", health)
                end,
                on_event = function(self, name, payload, engine)
                    if name == "remove.health" then
                        engine.components.remove(self.entity, "test.script_health")
                    end
                end
            }
        "#,
        );
        script
            .register_component::<Health>()
            .expect("register script component");
        let mut world = registered_world();
        script.start_with_world(&mut world).expect("start");
        let entity = world.entities()[0];
        assert_eq!(world.get::<Health>(entity), Some(&Health { current: 12 }));

        script.update_with_world(&mut world, 0.016).expect("update");
        assert_eq!(world.get::<Health>(entity), Some(&Health { current: 19 }));

        script
            .dispatch_event_with_world(&mut world, "remove.health", &serde_json::json!({}))
            .expect("remove component");
        assert!(world.get::<Health>(entity).is_none());
    }

    #[test]
    fn component_registry_rejects_duplicate_stable_ids() {
        let mut script = load("return {}");
        script
            .register_component::<Health>()
            .expect("first registration");
        assert_eq!(
            script.register_component::<Health>(),
            Err(ScriptRegistryError::DuplicateComponentId(Health::ID))
        );
    }

    #[test]
    fn unknown_component_ids_and_invalid_payloads_are_reported() {
        let mut script = load(
            r#"
            return {
                start = function(self, engine)
                    self.entity = engine.entities.spawn()
                    engine.components.set(self.entity, "missing.component", {})
                end
            }
        "#,
        );
        script
            .register_component::<Health>()
            .expect("register script component");
        let mut world = registered_world();
        assert!(matches!(
            script.start_with_world(&mut world),
            Err(ScriptError::Execution { .. })
        ));
        assert!(world.entities().is_empty());

        let mut invalid = load(
            r#"
            return {
                start = function(self, engine)
                    self.entity = engine.entities.spawn()
                    engine.components.set(self.entity, "test.script_health", {
                        current = 1, unexpected = true
                    })
                end
            }
        "#,
        );
        invalid
            .register_component::<Health>()
            .expect("register component");
        assert!(matches!(
            invalid.start_with_world(&mut world),
            Err(ScriptError::Execution { .. })
        ));
        assert!(world.entities().is_empty());
    }
}

#[cfg(test)]
mod prefab_binding_tests {
    use super::{ScriptConfig, ScriptError, ScriptInstance, ScriptRegistryError};
    use z_ideal_scene::{Prefab, SceneDocument, SceneEntityId, SceneEntityRecord, World};

    fn prefab() -> Prefab {
        let source_id = SceneEntityId::new(10).expect("source ID");
        let mut document = SceneDocument::new();
        document.next_scene_entity_id = 11;
        document
            .entities
            .push(SceneEntityRecord::new(source_id, None));
        Prefab::new(document).expect("validated prefab")
    }

    fn load(source: &str) -> ScriptInstance {
        ScriptInstance::load("prefabs.lua", source, ScriptConfig::default()).expect("load script")
    }

    #[test]
    fn scripts_can_instantiate_prefab_and_resolve_entity_on_next_callback() {
        let mut script = load(
            r#"
            return {
                start = function(self, engine)
                    self.instance = engine.prefabs.instantiate("crate")
                end,
                update = function(self, dt, engine)
                    local entity = engine.prefabs.entity(self.instance, 10)
                    engine.entities.destroy(entity)
                end
            }
        "#,
        );
        script
            .register_prefab("crate", prefab())
            .expect("register prefab");
        let mut world = World::new();
        script.start_with_world(&mut world).expect("instantiate");
        assert_eq!(world.entities().len(), 1);
        script
            .update_with_world(&mut world, 0.016)
            .expect("resolve and destroy");
        assert!(world.entities().is_empty());
    }

    #[test]
    fn scripts_can_destroy_an_entire_prefab_instance() {
        let mut script = load(
            r#"
            return {
                start = function(self, engine)
                    self.instance = engine.prefabs.instantiate("crate")
                end,
                on_event = function(self, name, payload, engine)
                    engine.prefabs.destroy(self.instance)
                end
            }
        "#,
        );
        script
            .register_prefab("crate", prefab())
            .expect("register prefab");
        let mut world = World::new();
        script.start_with_world(&mut world).expect("instantiate");
        assert_eq!(world.entities().len(), 1);
        script
            .dispatch_event_with_world(&mut world, "remove.prefab", &serde_json::json!({}))
            .expect("destroy prefab");
        assert!(world.entities().is_empty());
    }

    #[test]
    fn prefab_ticket_cannot_be_resolved_before_commit() {
        let mut script = load(
            r#"
            return {
                start = function(self, engine)
                    local ticket = engine.prefabs.instantiate("crate")
                    engine.prefabs.entity(ticket, 10)
                end
            }
        "#,
        );
        script
            .register_prefab("crate", prefab())
            .expect("register prefab");
        let mut world = World::new();
        assert!(matches!(
            script.start_with_world(&mut world),
            Err(ScriptError::Execution { .. })
        ));
        assert!(world.entities().is_empty());
    }

    #[test]
    fn prefab_registry_rejects_duplicate_names() {
        let mut script = load("return {}");
        script.register_prefab("crate", prefab()).expect("register");
        assert_eq!(
            script.register_prefab("crate", prefab()),
            Err(ScriptRegistryError::DuplicatePrefabName("crate".to_owned()))
        );
    }
}

#[cfg(test)]
mod reload_tests {
    use super::{ScriptConfig, ScriptError, ScriptInstance, ScriptState};

    #[test]
    fn reload_replaces_source_after_shutdown_with_fresh_state() {
        let mut script = ScriptInstance::load(
            "reload.lua",
            "return { start = function(self) self.value = 1 end }",
            ScriptConfig::default(),
        )
        .expect("load initial script");
        script.start().expect("start initial script");
        assert!(matches!(
            script.reload("return {}", ScriptConfig::default()),
            Err(ScriptError::ReloadWhileStarted { .. })
        ));
        script.shutdown().expect("shutdown initial script");
        script.reload(
            "return { start = function(self) self.value = 2 end, update = function(self) assert(self.value == 2) end }",
            ScriptConfig::default(),
        ).expect("reload");
        assert_eq!(script.state(), ScriptState::Loaded);
        script.start().expect("start replacement");
        script.update(0.016).expect("update replacement");
    }

    #[test]
    fn failed_reload_preserves_the_previous_loaded_instance() {
        let mut script = ScriptInstance::load(
            "reload.lua",
            "return { start = function(self) self.value = 7 end, update = function(self) assert(self.value == 7) end }",
            ScriptConfig::default(),
        ).expect("load initial script");
        assert!(matches!(
            script.reload("return 17", ScriptConfig::default()),
            Err(ScriptError::MissingScriptTable { .. })
        ));
        script.start().expect("start original script");
        script.update(0.016).expect("verify original script");
    }
}

#[cfg(test)]
mod partial_prefab_binding_tests {
    use super::{ScriptConfig, ScriptError, ScriptInstance};
    use z_ideal_scene::{Prefab, SceneDocument, SceneEntityId, SceneEntityRecord, World};

    fn prefab() -> Prefab {
        let source_id = SceneEntityId::new(10).expect("source ID");
        let mut document = SceneDocument::new();
        document.next_scene_entity_id = 11;
        document
            .entities
            .push(SceneEntityRecord::new(source_id, None));
        Prefab::new(document).expect("prefab")
    }

    #[test]
    fn prefab_ticket_is_resolved_even_when_a_later_command_fails() {
        let mut script = ScriptInstance::load(
            "partial_prefab.lua",
            r#"
            return {
                start = function(self, engine)
                    self.old = engine.entities.spawn()
                end,
                update = function(self, dt, engine)
                    self.ticket = engine.prefabs.instantiate("crate")
                    engine.entities.destroy(self.old)
                end,
                on_event = function(self, name, payload, engine)
                    engine.prefabs.destroy(self.ticket)
                end
            }
        "#,
            ScriptConfig::default(),
        )
        .expect("load script");
        script
            .register_prefab("crate", prefab())
            .expect("register prefab");
        let mut world = World::new();
        script.start_with_world(&mut world).expect("start");
        let old = world.entities()[0];
        world.despawn(old).expect("make saved script handle stale");

        assert!(matches!(
            script.update_with_world(&mut world, 0.016),
            Err(ScriptError::CommandApplication { .. })
        ));
        assert_eq!(world.entities().len(), 1);
        script
            .dispatch_event_with_world(&mut world, "cleanup", &serde_json::json!({}))
            .expect("resolve partial prefab ticket and destroy");
        assert!(world.entities().is_empty());
    }
}

#[cfg(test)]
mod event_depth_tests {
    use super::{MAX_SCRIPT_EVENT_DEPTH, ScriptConfig, ScriptError, ScriptInstance};

    #[test]
    fn deeply_nested_event_payload_is_rejected_before_lua_conversion() {
        let mut script = ScriptInstance::load(
            "deep_event.lua",
            "return { on_event = function() end }",
            ScriptConfig::default(),
        )
        .expect("load script");
        script.start().expect("start");
        let mut payload = serde_json::json!(null);
        for _ in 0..=MAX_SCRIPT_EVENT_DEPTH {
            payload = serde_json::Value::Array(vec![payload]);
        }
        assert!(matches!(
            script.dispatch_event("deep", &payload),
            Err(ScriptError::EventPayloadTooDeep)
        ));
    }
}

#[cfg(test)]
mod budget_cap_tests {
    use super::{MAX_SCRIPT_INSTRUCTION_BUDGET, ScriptConfig, ScriptError, ScriptInstance};

    #[test]
    fn instruction_budget_cannot_overflow_counter() {
        assert!(matches!(
            ScriptInstance::load(
                "huge_budget.lua",
                "return {}",
                ScriptConfig {
                    instruction_budget: MAX_SCRIPT_INSTRUCTION_BUDGET + 1,
                },
            ),
            Err(ScriptError::InvalidInstructionBudget)
        ));
    }
}

#[cfg(test)]
mod world_update_pause_tests {
    use super::{ScriptConfig, ScriptInstance};
    use serde_json::json;
    use z_ideal_scene::World;

    #[test]
    fn fixed_update_with_world_queues_commands_and_respects_pause() {
        let mut script = ScriptInstance::load(
            "fixed_world.lua",
            "return { fixed_update = function(self, dt, engine) engine.entities.spawn() end }",
            ScriptConfig::default(),
        )
        .expect("load script");
        let mut world = World::new();
        script.start_with_world(&mut world).expect("start");
        script.set_paused(true);
        script
            .fixed_update_with_world(&mut world, 0.02)
            .expect("paused fixed update");
        assert!(world.entities().is_empty());
        script.set_paused(false);
        script
            .fixed_update_with_world(&mut world, 0.02)
            .expect("fixed update");
        assert_eq!(world.entities().len(), 1);
    }

    #[test]
    fn events_are_still_dispatched_while_updates_are_paused() {
        let mut script = ScriptInstance::load(
            "paused_events.lua",
            "return { on_event = function(self, name, payload, engine) engine.entities.spawn() end }",
            ScriptConfig::default(),
        ).expect("load script");
        let mut world = World::new();
        script.start_with_world(&mut world).expect("start");
        script.set_paused(true);
        script
            .dispatch_event_with_world(&mut world, "spawn", &json!({}))
            .expect("event while paused");
        assert_eq!(world.entities().len(), 1);
    }
}

#[cfg(test)]
mod registry_edge_tests {
    use crate::{ScriptConfig, ScriptError, ScriptInstance, ScriptRegistryError};
    use serde::{Deserialize, Serialize, Serializer};
    use z_ideal_scene::{
        Component, Prefab, SceneDocument, SceneEntityId, SceneEntityRecord, World,
    };

    #[derive(Debug, Serialize, Deserialize)]
    struct EmptyIdComponent;

    impl Component for EmptyIdComponent {
        const ID: &'static str = "";
    }

    #[derive(Debug, Deserialize)]
    struct FailingSerializeComponent;

    impl Serialize for FailingSerializeComponent {
        fn serialize<S: Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom(
                "intentional serialization failure",
            ))
        }
    }

    impl Component for FailingSerializeComponent {
        const ID: &'static str = "test.failing_serialize";
    }

    fn prefab() -> Prefab {
        let id = SceneEntityId::new(4).expect("source ID");
        let mut document = SceneDocument::new();
        document.next_scene_entity_id = 5;
        document.entities.push(SceneEntityRecord::new(id, None));
        Prefab::new(document).expect("prefab")
    }

    #[test]
    fn registry_rejects_empty_component_ids_and_prefab_names() {
        let mut script = ScriptInstance::load("registry.lua", "return {}", ScriptConfig::default())
            .expect("load script");
        assert_eq!(
            script.register_component::<EmptyIdComponent>(),
            Err(ScriptRegistryError::EmptyComponentId)
        );
        assert_eq!(
            script.register_prefab("  ", prefab()),
            Err(ScriptRegistryError::EmptyPrefabName)
        );
    }

    #[test]
    fn component_snapshot_serialization_errors_are_returned_before_callback() {
        let mut world = World::new();
        world
            .register_component::<FailingSerializeComponent>()
            .expect("register component");
        let entity = world.spawn().expect("spawn");
        world
            .insert(entity, FailingSerializeComponent)
            .expect("insert");

        let mut script = ScriptInstance::load(
            "snapshot_error.lua",
            "return { start = function() error('callback must not run') end }",
            ScriptConfig::default(),
        )
        .expect("load script");
        script
            .register_component::<FailingSerializeComponent>()
            .expect("register script component");
        assert!(matches!(
            script.start_with_world(&mut world),
            Err(ScriptError::ComponentSnapshot { .. })
        ));
        assert_eq!(script.state(), crate::ScriptState::Loaded);
        assert!(world.is_alive(entity));
    }
}

#[cfg(test)]
mod shutdown_prefab_ticket_tests {
    use crate::{ScriptConfig, ScriptError, ScriptInstance};
    use z_ideal_scene::{Prefab, SceneDocument, SceneEntityId, SceneEntityRecord, World};

    fn prefab() -> Prefab {
        let source_id = SceneEntityId::new(5).expect("source ID");
        let mut document = SceneDocument::new();
        document.next_scene_entity_id = 6;
        document
            .entities
            .push(SceneEntityRecord::new(source_id, None));
        Prefab::new(document).expect("prefab")
    }

    #[test]
    fn failed_shutdown_invalidates_new_uncommitted_prefab_tickets() {
        let mut script = ScriptInstance::load(
            "shutdown_ticket.lua",
            r#"
            return {
                shutdown = function(self, engine)
                    self.ticket = engine.prefabs.instantiate("crate")
                    error("shutdown callback failed")
                end,
                on_event = function(self, name, payload, engine)
                    engine.prefabs.entity(self.ticket, 5)
                end
            }
        "#,
            ScriptConfig::default(),
        )
        .expect("load");
        script.register_prefab("crate", prefab()).expect("register");
        let mut world = World::new();
        script.start_with_world(&mut world).expect("start");
        assert!(matches!(
            script.shutdown_with_world(&mut world),
            Err(ScriptError::Execution { .. })
        ));
        assert!(matches!(
            script.dispatch_event_with_world(&mut world, "check", &serde_json::json!({})),
            Err(ScriptError::Execution { .. })
        ));
        assert!(world.entities().is_empty());
    }
}
