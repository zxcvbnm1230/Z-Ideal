//! Entity-component world, generation-safe entities, and transform hierarchy.

#![forbid(unsafe_code)]

mod commands;
mod component;
mod entity;
mod hierarchy;
mod schedule;
mod world;

pub use commands::{CommandError, Commands, EntityTarget, PendingEntity};
pub use component::{Component, ComponentInfo};
pub use entity::Entity;
pub use hierarchy::HierarchyError;
pub use schedule::{Schedule, ScheduleError};
pub use world::{World, WorldError};

use z_ideal_math::Transform;

/// Built-in local transform component.
impl Component for Transform {
    const ID: &'static str = "z_ideal.transform";
    const VERSION: u32 = 1;

    fn validate(&self) -> Result<(), String> {
        Transform::validate(self).map_err(|error| error.to_string())
    }
}

/// Stable package identifier for diagnostics and tooling.
pub const CRATE_NAME: &str = "z-ideal-scene";

#[cfg(test)]
mod tests {
    use super::{CRATE_NAME, Component, Entity, HierarchyError, Transform, World, WorldError};
    use serde::{Deserialize, Serialize};
    use z_ideal_math::{Quat, Vec3};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Health(f32);

    impl Component for Health {
        const ID: &'static str = "sample.health";
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct DuplicateId;

    impl Component for DuplicateId {
        const ID: &'static str = "sample.health";
    }

    fn world_with_health() -> World {
        let mut world = World::new();
        world
            .register_component::<Health>()
            .expect("register Health");
        world
    }

    #[test]
    fn crate_name_is_not_empty() {
        assert!(!CRATE_NAME.is_empty());
    }

    #[test]
    fn entity_generation_changes_when_slot_is_reused() {
        let mut world = World::new();
        let first = world.spawn().expect("spawn first");
        world.despawn(first).expect("despawn first");
        let second = world.spawn().expect("spawn second");
        assert_eq!(first.index(), second.index());
        assert_ne!(first.generation(), second.generation());
        assert!(!world.is_alive(first));
        assert!(world.is_alive(second));
    }

    #[test]
    fn stale_entity_cannot_access_reused_slot() {
        let mut world = world_with_health();
        let first = world.spawn().expect("spawn first");
        world.insert(first, Health(10.0)).expect("insert component");
        world.despawn(first).expect("despawn first");
        let second = world.spawn().expect("spawn second");
        assert!(world.get::<Health>(first).is_none());
        assert_eq!(
            world.insert(first, Health(20.0)),
            Err(WorldError::StaleEntity(first))
        );
        assert!(world.get::<Health>(second).is_none());
    }

    #[test]
    fn component_insert_get_mutate_and_remove() {
        let mut world = world_with_health();
        let entity = world.spawn().expect("spawn");
        assert_eq!(world.insert(entity, Health(12.0)), Ok(None));
        assert_eq!(world.get::<Health>(entity), Some(&Health(12.0)));
        world.get_mut::<Health>(entity).expect("get mutable").0 = 24.0;
        assert_eq!(world.get::<Health>(entity), Some(&Health(24.0)));
        assert_eq!(world.remove::<Health>(entity), Ok(Some(Health(24.0))));
        assert!(world.get::<Health>(entity).is_none());
    }

    #[test]
    fn component_must_be_registered_before_use() {
        let mut world = World::new();
        let entity = world.spawn().expect("spawn");
        assert_eq!(
            world.insert(entity, Health(1.0)),
            Err(WorldError::ComponentNotRegistered("sample.health"))
        );
    }

    #[test]
    fn stable_component_ids_cannot_be_reused() {
        let mut world = World::new();
        world
            .register_component::<Health>()
            .expect("register Health");
        assert_eq!(
            world.register_component::<DuplicateId>(),
            Err(WorldError::DuplicateComponentId("sample.health"))
        );
    }

    #[test]
    fn invalid_transform_component_is_rejected() {
        let mut world = World::new();
        world
            .register_component::<Transform>()
            .expect("register Transform");
        let entity = world.spawn().expect("spawn");
        let invalid = serde_json::from_str::<Transform>(
            r#"{"translation":[0.0,0.0,0.0],"rotation":[0.0,0.0,0.0,0.0],"scale":[1.0,1.0,1.0]}"#,
        )
        .expect("deserialize shape");
        assert!(matches!(
            world.insert(entity, invalid),
            Err(WorldError::InvalidComponent { .. })
        ));
    }

    #[test]
    fn hierarchy_rejects_cycles_without_mutating_existing_parent() {
        let mut world = World::new();
        let root = world.spawn().expect("root");
        let child = world.spawn().expect("child");
        let grandchild = world.spawn().expect("grandchild");
        world.set_parent(child, Some(root)).expect("parent child");
        world
            .set_parent(grandchild, Some(child))
            .expect("parent grandchild");
        assert!(matches!(
            world.set_parent(root, Some(grandchild)),
            Err(WorldError::Hierarchy(HierarchyError::Cycle))
        ));
        assert_eq!(world.parent_of(child), Some(root));
        assert_eq!(world.parent_of(grandchild), Some(child));
        assert_eq!(world.parent_of(root), None);
    }

    #[test]
    fn self_parenting_is_rejected() {
        let mut world = World::new();
        let entity = world.spawn().expect("spawn");
        assert!(matches!(
            world.set_parent(entity, Some(entity)),
            Err(WorldError::Hierarchy(HierarchyError::SelfParent))
        ));
    }

    #[test]
    fn despawning_parent_detaches_children() {
        let mut world = World::new();
        let parent = world.spawn().expect("parent");
        let child = world.spawn().expect("child");
        world.set_parent(child, Some(parent)).expect("set parent");
        world.despawn(parent).expect("despawn parent");
        assert_eq!(world.parent_of(child), None);
        assert!(world.children_of(parent).is_empty());
    }

    #[test]
    fn world_transform_composes_parent_and_local_translation() {
        let mut world = World::new();
        world
            .register_component::<Transform>()
            .expect("register Transform");
        let parent = world.spawn().expect("parent");
        let child = world.spawn().expect("child");
        world
            .insert(
                parent,
                Transform::new(Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE)
                    .expect("parent transform"),
            )
            .expect("insert parent transform");
        world
            .insert(
                child,
                Transform::new(Vec3::new(0.0, 2.0, 0.0), Quat::IDENTITY, Vec3::ONE)
                    .expect("child transform"),
            )
            .expect("insert child transform");
        world.set_parent(child, Some(parent)).expect("set parent");
        let position = world
            .world_transform(child)
            .expect("world transform")
            .transform_point3(Vec3::ZERO);
        assert_eq!(position, Vec3::new(3.0, 2.0, 0.0));
    }

    #[test]
    fn entities_with_component_are_sorted_and_current() {
        let mut world = world_with_health();
        let first = world.spawn().expect("first");
        let second = world.spawn().expect("second");
        world.insert(second, Health(2.0)).expect("insert second");
        world.insert(first, Health(1.0)).expect("insert first");
        assert_eq!(world.entities_with::<Health>(), vec![first, second]);
    }

    #[test]
    fn invalid_entities_return_stale_entity_error() {
        let mut world = World::new();
        let fabricated = Entity::from_parts_for_test(500, 1);
        assert_eq!(
            world.despawn(fabricated),
            Err(WorldError::StaleEntity(fabricated))
        );
    }
}
