//! Synchronization between serializable scene physics components and Box3D.

use std::collections::HashMap;

use thiserror::Error;
use z_ideal_math::{Transform, Vec3};
use z_ideal_scene::{
    Collider, ColliderShape as SceneColliderShape, Entity, RigidBody, RigidBodyType,
    World as SceneWorld, WorldError as SceneWorldError,
};

use crate::{
    ColliderDesc, ColliderShape, PhysicsBodyHandle, PhysicsBodyType, PhysicsError, PhysicsWorld,
    PhysicsWorldConfig, RigidBodyDesc,
};

/// Errors produced while synchronizing a scene with its physics world.
#[derive(Debug, Error)]
pub enum PhysicsSceneError {
    /// Physics backend operation failed.
    #[error(transparent)]
    Physics(#[from] PhysicsError),
    /// Scene operation failed.
    #[error(transparent)]
    Scene(#[from] SceneWorldError),
    /// A physics entity must be a root entity; parented physics transforms are not yet supported.
    #[error("physics entity {0:?} must be a root entity")]
    ParentTransformUnsupported(Entity),
    /// Physics entities currently require unit local scale.
    #[error("physics entity {0:?} must have unit scale")]
    NonUnitScale(Entity),
    /// Scene transform could not be represented by a physics body.
    #[error("physics entity {0:?} has a transform that cannot be represented")]
    InvalidTransform(Entity),
}

#[derive(Debug, Clone, Copy)]
struct Binding {
    handle: PhysicsBodyHandle,
    rigid_body: RigidBody,
    collider: Collider,
}

/// Owns a physics world and synchronizes it with one ECS scene.
///
/// Physics entities must currently be root entities with unit scale. This
/// avoids silently dropping parent-induced shear or collider scale. Static
/// and kinematic transforms are pushed from the scene during synchronization;
/// dynamic transforms are written back after simulation steps.
pub struct PhysicsScene {
    physics: PhysicsWorld,
    bindings: HashMap<Entity, Binding>,
}

impl PhysicsScene {
    /// Creates a scene synchronization layer around a new physics world.
    pub fn new(config: PhysicsWorldConfig) -> Result<Self, PhysicsError> {
        Ok(Self {
            physics: PhysicsWorld::new(config)?,
            bindings: HashMap::new(),
        })
    }

    /// Returns the underlying physics world.
    #[must_use]
    pub const fn physics_world(&self) -> &PhysicsWorld {
        &self.physics
    }

    /// Returns mutable access to the underlying physics world.
    pub fn physics_world_mut(&mut self) -> &mut PhysicsWorld {
        &mut self.physics
    }

    /// Advances exactly one configured fixed simulation step.
    pub fn step(&mut self) -> Result<(), PhysicsError> {
        self.physics.step()
    }

    /// Creates, updates, and removes physics bodies to match scene components.
    ///
    /// An entity participates only when it has both RigidBody and Collider.
    /// Removing either component or despawning the entity destroys its body.
    pub fn synchronize(&mut self, scene: &mut SceneWorld) -> Result<(), PhysicsSceneError> {
        scene.register_component::<Transform>()?;
        scene.register_component::<RigidBody>()?;
        scene.register_component::<Collider>()?;

        let entities = scene.entities_with::<RigidBody>();
        let active: Vec<_> = entities
            .into_iter()
            .filter(|entity| scene.get::<Collider>(*entity).is_some())
            .collect();

        // Validate the complete scene input before changing any native body.
        // This prevents a malformed later entity from leaving earlier entities
        // partially synchronized during the same call.
        let mut desired = Vec::with_capacity(active.len());
        for entity in active {
            if scene.parent_of(entity).is_some() {
                return Err(PhysicsSceneError::ParentTransformUnsupported(entity));
            }
            let transform = scene.get::<Transform>(entity).copied().unwrap_or_default();
            if !transform.scale().abs_diff_eq(Vec3::ONE, 1.0e-6) {
                return Err(PhysicsSceneError::NonUnitScale(entity));
            }
            let rigid_body = *scene
                .get::<RigidBody>(entity)
                .ok_or(SceneWorldError::StaleEntity(entity))?;
            let collider = *scene
                .get::<Collider>(entity)
                .ok_or(SceneWorldError::StaleEntity(entity))?;
            desired.push((entity, transform, rigid_body, collider));
        }
        let active_entities: Vec<_> = desired.iter().map(|(entity, _, _, _)| *entity).collect();

        let mut stale: Vec<_> = self
            .bindings
            .keys()
            .copied()
            .filter(|entity| !active_entities.contains(entity))
            .collect();
        stale.sort_unstable();
        for entity in stale {
            if let Some(binding) = self.bindings.get(&entity).copied() {
                self.physics.destroy_body(binding.handle)?;
                self.bindings.remove(&entity);
            }
        }

        for (entity, transform, rigid_body, collider) in desired {
            let changed = self.bindings.get(&entity).is_some_and(|binding| {
                binding.rigid_body != rigid_body || binding.collider != collider
            });
            if changed {
                let binding = self.bindings[&entity];
                self.physics.destroy_body(binding.handle)?;
                self.bindings.remove(&entity);
            }

            if !self.bindings.contains_key(&entity) {
                let body_desc = RigidBodyDesc {
                    body_type: match rigid_body.body_type {
                        RigidBodyType::Static => PhysicsBodyType::Static,
                        RigidBodyType::Kinematic => PhysicsBodyType::Kinematic,
                        RigidBodyType::Dynamic => PhysicsBodyType::Dynamic,
                    },
                    position: transform.translation(),
                    rotation: transform.rotation(),
                    linear_velocity: rigid_body.linear_velocity,
                    gravity_scale: rigid_body.gravity_scale,
                    linear_damping: rigid_body.linear_damping,
                    angular_damping: rigid_body.angular_damping,
                    allow_sleep: rigid_body.allow_sleep,
                    bullet: rigid_body.bullet,
                };
                let collider_desc = ColliderDesc {
                    shape: match collider.shape {
                        SceneColliderShape::Sphere { radius } => ColliderShape::Sphere { radius },
                        SceneColliderShape::Box { half_extents } => {
                            ColliderShape::Box { half_extents }
                        }
                    },
                    density: collider.density,
                    friction: collider.friction,
                    restitution: collider.restitution,
                    sensor: collider.sensor,
                    contact_events: collider.contact_events,
                    sensor_events: collider.sensor_events,
                };
                let handle = self.physics.create_body(body_desc, collider_desc)?;
                self.bindings.insert(
                    entity,
                    Binding {
                        handle,
                        rigid_body,
                        collider,
                    },
                );
            } else if rigid_body.body_type != RigidBodyType::Dynamic {
                let handle = self.bindings[&entity].handle;
                self.physics.set_body_transform(
                    handle,
                    transform.translation(),
                    transform.rotation(),
                )?;
            }
        }
        Ok(())
    }

    /// Copies dynamic body poses back into local scene transforms.
    ///
    /// This method does not perform interpolation. Callers may use the
    /// FixedStepPlan interpolation alpha for presentation separately.
    pub fn writeback_dynamic_transforms(
        &self,
        scene: &mut SceneWorld,
    ) -> Result<(), PhysicsSceneError> {
        for (entity, binding) in &self.bindings {
            if binding.rigid_body.body_type != RigidBodyType::Dynamic || !scene.is_alive(*entity) {
                continue;
            }
            let state = self.physics.body_state(binding.handle)?;
            let existing = scene.get::<Transform>(*entity).copied().unwrap_or_default();
            let transform = Transform::new(state.position, state.rotation, existing.scale())
                .map_err(|_| PhysicsSceneError::InvalidTransform(*entity))?;
            if let Some(current) = scene.get_mut::<Transform>(*entity) {
                *current = transform;
            } else {
                scene.insert(*entity, transform)?;
            }
        }
        Ok(())
    }

    /// Returns the number of scene entities currently mapped to physics bodies.
    #[must_use]
    pub fn body_count(&self) -> usize {
        self.bindings.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use z_ideal_math::{Quat, Transform, Vec3};

    fn scene_with_physics_components() -> SceneWorld {
        let mut scene = SceneWorld::new();
        scene.register_component::<Transform>().expect("transform");
        scene.register_component::<RigidBody>().expect("rigid body");
        scene.register_component::<Collider>().expect("collider");
        scene
    }

    #[test]
    fn synchronizes_scene_components_and_removes_bodies_when_components_are_removed() {
        let mut scene = scene_with_physics_components();
        let entity = scene.spawn().expect("entity");
        scene
            .insert(
                entity,
                Transform::new(Vec3::new(0.0, 3.0, 0.0), Quat::IDENTITY, Vec3::ONE)
                    .expect("transform"),
            )
            .expect("insert transform");
        scene
            .insert(entity, RigidBody::default())
            .expect("rigid body");
        scene
            .insert(entity, Collider::sphere(0.5))
            .expect("collider");
        let mut physics = PhysicsScene::new(PhysicsWorldConfig::default()).expect("physics");
        physics.synchronize(&mut scene).expect("sync");
        assert_eq!(physics.body_count(), 1);
        scene.remove::<Collider>(entity).expect("remove collider");
        physics.synchronize(&mut scene).expect("remove body");
        assert_eq!(physics.body_count(), 0);
    }

    #[test]
    fn dynamic_simulation_writes_back_scene_transform() {
        let mut scene = scene_with_physics_components();
        let entity = scene.spawn().expect("entity");
        scene
            .insert(
                entity,
                Transform::new(Vec3::new(0.0, 3.0, 0.0), Quat::IDENTITY, Vec3::ONE)
                    .expect("transform"),
            )
            .expect("insert transform");
        scene
            .insert(entity, RigidBody::default())
            .expect("rigid body");
        scene
            .insert(entity, Collider::sphere(0.5))
            .expect("collider");
        let mut physics = PhysicsScene::new(PhysicsWorldConfig::default()).expect("physics");
        physics.synchronize(&mut scene).expect("sync");
        for _ in 0..30 {
            physics.step().expect("step");
        }
        physics
            .writeback_dynamic_transforms(&mut scene)
            .expect("writeback");
        assert!(
            scene
                .get::<Transform>(entity)
                .expect("transform")
                .translation()
                .y
                < 3.0
        );
    }

    #[test]
    fn static_and_kinematic_transforms_are_pushed_from_scene() {
        let mut scene = scene_with_physics_components();
        let entity = scene.spawn().expect("entity");
        scene
            .insert(
                entity,
                RigidBody {
                    body_type: RigidBodyType::Kinematic,
                    ..RigidBody::default()
                },
            )
            .expect("rigid body");
        scene
            .insert(entity, Collider::sphere(0.5))
            .expect("collider");
        let mut physics = PhysicsScene::new(PhysicsWorldConfig::default()).expect("physics");
        physics.synchronize(&mut scene).expect("initial sync");
        let moved = Transform::new(
            Vec3::new(4.0, 2.0, 1.0),
            Quat::from_rotation_y(0.5),
            Vec3::ONE,
        )
        .expect("moved transform");
        scene.insert(entity, moved).expect("move entity");
        physics.synchronize(&mut scene).expect("second sync");
        let binding = physics.bindings[&entity];
        let state = physics
            .physics
            .body_state(binding.handle)
            .expect("body state");
        assert!(state.position.abs_diff_eq(Vec3::new(4.0, 2.0, 1.0), 1.0e-5));
    }

    #[test]
    fn rejects_parented_physics_entities_and_non_unit_scale() {
        let mut scene = scene_with_physics_components();
        let valid_entity = scene.spawn().expect("valid entity");
        scene
            .insert(valid_entity, RigidBody::default())
            .expect("valid rigid body");
        scene
            .insert(valid_entity, Collider::sphere(0.5))
            .expect("valid collider");
        let parent = scene.spawn().expect("parent");
        let entity = scene.spawn().expect("entity");
        scene
            .set_parent(entity, Some(parent))
            .expect("parent entity");
        scene
            .insert(entity, RigidBody::default())
            .expect("rigid body");
        scene
            .insert(entity, Collider::sphere(0.5))
            .expect("collider");
        let mut physics = PhysicsScene::new(PhysicsWorldConfig::default()).expect("physics");
        assert!(matches!(
            physics.synchronize(&mut scene),
            Err(PhysicsSceneError::ParentTransformUnsupported(found)) if found == entity
        ));
        scene.set_parent(entity, None).expect("unparent");
        scene
            .insert(
                entity,
                Transform::new(Vec3::ZERO, Quat::IDENTITY, Vec3::splat(2.0)).expect("scaled"),
            )
            .expect("insert scaled transform");
        assert!(matches!(
            physics.synchronize(&mut scene),
            Err(PhysicsSceneError::NonUnitScale(found)) if found == entity
        ));
        assert_eq!(physics.body_count(), 0);
        assert_eq!(physics.physics_world().body_count(), 0);
    }

    #[test]
    fn despawning_a_scene_entity_releases_its_body() {
        let mut scene = scene_with_physics_components();
        let entity = scene.spawn().expect("entity");
        scene
            .insert(entity, RigidBody::default())
            .expect("rigid body");
        scene
            .insert(entity, Collider::sphere(0.5))
            .expect("collider");
        let mut physics = PhysicsScene::new(PhysicsWorldConfig::default()).expect("physics");
        physics.synchronize(&mut scene).expect("sync");
        scene.despawn(entity).expect("despawn");
        physics.synchronize(&mut scene).expect("cleanup");
        assert_eq!(physics.body_count(), 0);
        assert_eq!(physics.physics_world().body_count(), 0);
    }

    #[test]
    fn descriptor_changes_recreate_the_native_body() {
        let mut scene = scene_with_physics_components();
        let entity = scene.spawn().expect("entity");
        scene
            .insert(entity, RigidBody::default())
            .expect("rigid body");
        scene
            .insert(entity, Collider::sphere(0.5))
            .expect("collider");
        let mut physics = PhysicsScene::new(PhysicsWorldConfig::default()).expect("physics");
        physics.synchronize(&mut scene).expect("initial sync");
        let old = physics.bindings[&entity].handle;
        scene
            .insert(entity, Collider::sphere(1.0))
            .expect("replace collider");
        physics.synchronize(&mut scene).expect("recreate body");
        let new = physics.bindings[&entity].handle;
        assert_ne!(old, new);
        assert_eq!(physics.body_count(), 1);
    }
}
