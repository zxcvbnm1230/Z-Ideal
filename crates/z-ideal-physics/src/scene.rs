//! Synchronization between serializable scene physics components and Box3D.

use std::collections::HashMap;

use thiserror::Error;
use z_ideal_math::{Mat4, Quat, Transform, Vec3};
use z_ideal_scene::{
    Collider, ColliderShape as SceneColliderShape, Entity, RigidBody, RigidBodyType,
    World as SceneWorld, WorldError as SceneWorldError,
};

use crate::{
    ColliderDesc, ColliderShape, PhysicsBodyHandle, PhysicsBodyType, PhysicsError, PhysicsEvent,
    PhysicsOverlapHit, PhysicsRayCast, PhysicsRayHit, PhysicsShapeCast, PhysicsShapeCastHit,
    PhysicsShapeOverlap, PhysicsWorld, PhysicsWorldConfig, RigidBodyDesc,
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
    /// A sphere collider cannot represent non-uniform world scale.
    #[error("sphere collider on physics entity {0:?} requires uniform world scale")]
    NonUniformSphereScale(Entity),
    /// Scene transform could not be represented by a physics body.
    #[error("physics entity {0:?} has a transform that cannot be represented")]
    InvalidTransform(Entity),
    /// A physics result referenced a body not currently mapped to this scene.
    #[error("physics body {0:?} has no live scene entity mapping")]
    UnmappedBody(PhysicsBodyHandle),
}

/// A physics event translated into stable scene entity references.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsSceneEvent {
    /// Two solid scene colliders began touching.
    ContactBegin { entity_a: Entity, entity_b: Entity },
    /// Two solid scene colliders stopped touching.
    ContactEnd { entity_a: Entity, entity_b: Entity },
    /// A scene entity began overlapping a sensor entity.
    SensorBegin { sensor: Entity, visitor: Entity },
    /// A scene entity stopped overlapping a sensor entity.
    SensorEnd { sensor: Entity, visitor: Entity },
}

/// A ray-cast hit translated into a scene entity reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsSceneRayHit {
    /// Entity whose collider was hit.
    pub entity: Entity,
    /// World-space intersection point.
    pub point: Vec3,
    /// Surface normal at the intersection point.
    pub normal: Vec3,
    /// Fraction along the ray displacement.
    pub fraction: f32,
}

/// A shape-cast hit translated into a scene entity reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsSceneShapeCastHit {
    /// Entity whose collider was hit.
    pub entity: Entity,
    /// World-space intersection point.
    pub point: Vec3,
    /// Surface normal at the intersection point.
    pub normal: Vec3,
    /// Fraction along the shape-cast translation.
    pub fraction: f32,
}

/// A shape-overlap result translated into a scene entity reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicsSceneOverlapHit {
    /// Entity whose collider overlaps the query shape.
    pub entity: Entity,
}

#[derive(Debug, Clone, Copy)]
struct Binding {
    handle: PhysicsBodyHandle,
    rigid_body: RigidBody,
    collider: Collider,
    collider_scale: Vec3,
}

/// Owns a physics world and synchronizes it with one ECS scene.
///
/// Parent transforms are composed into world-space body poses. Collider scale
/// is baked into primitive dimensions; sphere colliders require uniform world
/// scale, while boxes support non-uniform scale. Transform hierarchies that
/// produce shear, reflection or singular scale are rejected. Static and
/// kinematic transforms are pushed from the scene; dynamic world poses are
/// converted back to local transforms after simulation steps.
pub struct PhysicsScene {
    physics: PhysicsWorld,
    bindings: HashMap<Entity, Binding>,
    retired_bindings: HashMap<PhysicsBodyHandle, Entity>,
    event_bindings: HashMap<PhysicsBodyHandle, Entity>,
}

fn decompose_world_matrix(
    matrix: Mat4,
    entity: Entity,
) -> Result<(Vec3, Quat, Vec3), PhysicsSceneError> {
    if !matrix.is_finite() || matrix.determinant().abs() <= f32::MIN_POSITIVE {
        return Err(PhysicsSceneError::InvalidTransform(entity));
    }
    let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
    if !scale.is_finite()
        || scale.min_element() <= f32::MIN_POSITIVE
        || !rotation.is_finite()
        || !translation.is_finite()
    {
        return Err(PhysicsSceneError::InvalidTransform(entity));
    }
    let recomposed = Mat4::from_scale_rotation_translation(scale, rotation, translation);
    if !recomposed.abs_diff_eq(matrix, 1.0e-4) {
        return Err(PhysicsSceneError::InvalidTransform(entity));
    }
    Ok((translation, rotation.normalize(), scale))
}

fn hierarchy_depth(scene: &SceneWorld, entity: Entity) -> usize {
    let mut depth = 0;
    let mut parent = scene.parent_of(entity);
    while let Some(current) = parent {
        depth += 1;
        parent = scene.parent_of(current);
    }
    depth
}

fn scale_collider(
    collider: Collider,
    scale: Vec3,
    entity: Entity,
) -> Result<SceneColliderShape, PhysicsSceneError> {
    Ok(match collider.shape {
        SceneColliderShape::Sphere { radius } => {
            if !scale.abs_diff_eq(Vec3::splat(scale.x), 1.0e-6) {
                return Err(PhysicsSceneError::NonUniformSphereScale(entity));
            }
            SceneColliderShape::Sphere {
                radius: radius * scale.x,
            }
        }
        SceneColliderShape::Box { half_extents } => SceneColliderShape::Box {
            half_extents: half_extents * scale,
        },
    })
}

impl PhysicsScene {
    /// Creates a scene synchronization layer around a new physics world.
    pub fn new(config: PhysicsWorldConfig) -> Result<Self, PhysicsError> {
        Ok(Self {
            physics: PhysicsWorld::new(config)?,
            bindings: HashMap::new(),
            retired_bindings: HashMap::new(),
            event_bindings: HashMap::new(),
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
        self.physics.step()?;
        self.event_bindings = self
            .bindings
            .iter()
            .map(|(entity, binding)| (binding.handle, *entity))
            .chain(self.retired_bindings.drain())
            .collect();
        Ok(())
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

        // Validate and decompose the complete scene input before changing any
        // native body. World matrices that contain shear/reflection/singular
        // scale cannot be represented by the rigid-body plus primitive-shape API.
        let mut desired = Vec::with_capacity(active.len());
        for entity in active {
            let world_matrix = scene.world_transform(entity)?;
            let (world_position, world_rotation, collider_scale) =
                decompose_world_matrix(world_matrix, entity)?;
            let rigid_body = *scene
                .get::<RigidBody>(entity)
                .ok_or(SceneWorldError::StaleEntity(entity))?;
            let collider = *scene
                .get::<Collider>(entity)
                .ok_or(SceneWorldError::StaleEntity(entity))?;
            let scaled_collider = scale_collider(collider, collider_scale, entity)?;
            desired.push((
                entity,
                world_position,
                world_rotation,
                collider_scale,
                rigid_body,
                collider,
                scaled_collider,
            ));
        }
        let active_entities: Vec<_> = desired.iter().map(|(entity, ..)| *entity).collect();

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
                self.retired_bindings.insert(binding.handle, entity);
                self.bindings.remove(&entity);
            }
        }

        for (
            entity,
            world_position,
            world_rotation,
            collider_scale,
            rigid_body,
            collider,
            scaled_collider,
        ) in desired
        {
            let existing_binding = self.bindings.get(&entity).copied();
            let changed = existing_binding.is_some_and(|binding| {
                binding.rigid_body != rigid_body
                    || binding.collider != collider
                    || !binding.collider_scale.abs_diff_eq(collider_scale, 1.0e-6)
            });
            let preserved_state = if changed {
                let binding = existing_binding.expect("changed binding exists");
                let state = (binding.rigid_body == rigid_body
                    && rigid_body.body_type == RigidBodyType::Dynamic)
                    .then(|| self.physics.body_state(binding.handle))
                    .transpose()?;
                self.physics.destroy_body(binding.handle)?;
                self.retired_bindings.insert(binding.handle, entity);
                self.bindings.remove(&entity);
                state
            } else {
                None
            };

            if !self.bindings.contains_key(&entity) {
                let (position, rotation, linear_velocity) = preserved_state.map_or(
                    (world_position, world_rotation, rigid_body.linear_velocity),
                    |state| (state.position, state.rotation, state.linear_velocity),
                );
                let body_desc = RigidBodyDesc {
                    body_type: match rigid_body.body_type {
                        RigidBodyType::Static => PhysicsBodyType::Static,
                        RigidBodyType::Kinematic => PhysicsBodyType::Kinematic,
                        RigidBodyType::Dynamic => PhysicsBodyType::Dynamic,
                    },
                    position,
                    rotation,
                    linear_velocity,
                    gravity_scale: rigid_body.gravity_scale,
                    linear_damping: rigid_body.linear_damping,
                    angular_damping: rigid_body.angular_damping,
                    allow_sleep: rigid_body.allow_sleep,
                    bullet: rigid_body.bullet,
                };
                let collider_desc = ColliderDesc {
                    shape: match scaled_collider {
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
                    category_bits: collider.category_bits,
                    mask_bits: collider.mask_bits,
                };
                let handle = self.physics.create_body(body_desc, collider_desc)?;
                self.bindings.insert(
                    entity,
                    Binding {
                        handle,
                        rigid_body,
                        collider,
                        collider_scale,
                    },
                );
            } else if rigid_body.body_type != RigidBodyType::Dynamic {
                let handle = self.bindings[&entity].handle;
                self.physics
                    .set_body_transform(handle, world_position, world_rotation)?;
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
        let mut dynamic_entities: Vec<_> = self
            .bindings
            .iter()
            .filter_map(|(entity, binding)| {
                (binding.rigid_body.body_type == RigidBodyType::Dynamic && scene.is_alive(*entity))
                    .then_some(*entity)
            })
            .collect();
        // Parent bodies must be written before children so each child's local
        // pose is solved against the parent's newly simulated world transform.
        dynamic_entities.sort_by_key(|entity| (hierarchy_depth(scene, *entity), *entity));

        for entity in dynamic_entities {
            let binding = self.bindings[&entity];
            let state = self.physics.body_state(binding.handle)?;
            let existing = scene.get::<Transform>(entity).copied().unwrap_or_default();
            let (local_position, local_rotation) = if let Some(parent) = scene.parent_of(entity) {
                let parent_matrix = scene.world_transform(parent)?;
                let (_, parent_rotation, _) = decompose_world_matrix(parent_matrix, parent)?;
                let inverse = parent_matrix.inverse();
                if !inverse.is_finite() {
                    return Err(PhysicsSceneError::InvalidTransform(entity));
                }
                (
                    inverse.transform_point3(state.position),
                    parent_rotation.inverse() * state.rotation,
                )
            } else {
                (state.position, state.rotation)
            };
            let transform = Transform::new(local_position, local_rotation, existing.scale())
                .map_err(|_| PhysicsSceneError::InvalidTransform(entity))?;
            if let Some(current) = scene.get_mut::<Transform>(entity) {
                *current = transform;
            } else {
                scene.insert(entity, transform)?;
            }
        }
        Ok(())
    }

    /// Returns latest-step physics events translated to scene entities.
    ///
    /// An unmapped body is reported as an error rather than silently dropping
    /// an event. Removed entities retain their identity in end-event snapshots.
    pub fn events(&self) -> Result<Vec<PhysicsSceneEvent>, PhysicsSceneError> {
        self.physics
            .events()
            .iter()
            .map(|event| {
                let entity_for = |handle| {
                    self.event_bindings
                        .get(&handle)
                        .copied()
                        .or_else(|| {
                            self.bindings.iter().find_map(|(entity, binding)| {
                                (binding.handle == handle).then_some(*entity)
                            })
                        })
                        .or_else(|| self.retired_bindings.get(&handle).copied())
                        .ok_or(PhysicsSceneError::UnmappedBody(handle))
                };
                Ok(match *event {
                    PhysicsEvent::ContactBegin { body_a, body_b } => {
                        let mut entities = [entity_for(body_a)?, entity_for(body_b)?];
                        entities.sort_unstable();
                        PhysicsSceneEvent::ContactBegin {
                            entity_a: entities[0],
                            entity_b: entities[1],
                        }
                    }
                    PhysicsEvent::ContactEnd { body_a, body_b } => {
                        let mut entities = [entity_for(body_a)?, entity_for(body_b)?];
                        entities.sort_unstable();
                        PhysicsSceneEvent::ContactEnd {
                            entity_a: entities[0],
                            entity_b: entities[1],
                        }
                    }
                    PhysicsEvent::SensorBegin { sensor, visitor } => {
                        PhysicsSceneEvent::SensorBegin {
                            sensor: entity_for(sensor)?,
                            visitor: entity_for(visitor)?,
                        }
                    }
                    PhysicsEvent::SensorEnd { sensor, visitor } => PhysicsSceneEvent::SensorEnd {
                        sensor: entity_for(sensor)?,
                        visitor: entity_for(visitor)?,
                    },
                })
            })
            .collect()
    }

    /// Casts a world-space ray and translates hits to scene entities.
    pub fn cast_ray(
        &self,
        ray: PhysicsRayCast,
    ) -> Result<Vec<PhysicsSceneRayHit>, PhysicsSceneError> {
        self.physics
            .cast_ray(ray)?
            .into_iter()
            .map(|hit: PhysicsRayHit| {
                Ok(PhysicsSceneRayHit {
                    entity: self.entity_for_body(hit.body)?,
                    point: hit.point,
                    normal: hit.normal,
                    fraction: hit.fraction,
                })
            })
            .collect()
    }

    /// Sweeps a convex primitive through the physics scene and resolves hits to entities.
    pub fn cast_shape(
        &self,
        query: PhysicsShapeCast,
    ) -> Result<Vec<PhysicsSceneShapeCastHit>, PhysicsSceneError> {
        self.physics
            .cast_shape(query)?
            .into_iter()
            .map(|hit: PhysicsShapeCastHit| {
                Ok(PhysicsSceneShapeCastHit {
                    entity: self.entity_for_body(hit.body)?,
                    point: hit.point,
                    normal: hit.normal,
                    fraction: hit.fraction,
                })
            })
            .collect()
    }

    /// Finds scene colliders overlapping a convex query primitive.
    pub fn overlap_shape(
        &self,
        query: PhysicsShapeOverlap,
    ) -> Result<Vec<PhysicsSceneOverlapHit>, PhysicsSceneError> {
        self.physics
            .overlap_shape(query)?
            .into_iter()
            .map(|hit: PhysicsOverlapHit| {
                Ok(PhysicsSceneOverlapHit {
                    entity: self.entity_for_body(hit.body)?,
                })
            })
            .collect()
    }

    fn entity_for_body(&self, handle: PhysicsBodyHandle) -> Result<Entity, PhysicsSceneError> {
        self.bindings
            .iter()
            .find_map(|(entity, binding)| (binding.handle == handle).then_some(*entity))
            .ok_or(PhysicsSceneError::UnmappedBody(handle))
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
    fn parented_dynamic_body_uses_world_pose_and_writes_back_local_pose() {
        let mut scene = scene_with_physics_components();
        let parent = scene.spawn().expect("parent");
        let parent_transform = Transform::new(
            Vec3::new(4.0, 2.0, 1.0),
            Quat::from_rotation_y(0.5),
            Vec3::ONE,
        )
        .expect("parent transform");
        scene
            .insert(parent, parent_transform)
            .expect("insert parent transform");
        let entity = scene.spawn().expect("child");
        scene.set_parent(entity, Some(parent)).expect("set parent");
        scene
            .insert(
                entity,
                Transform::new(
                    Vec3::new(1.0, 0.0, 0.0),
                    Quat::from_rotation_x(0.2),
                    Vec3::ONE,
                )
                .expect("child transform"),
            )
            .expect("insert child transform");
        scene
            .insert(entity, RigidBody::default())
            .expect("rigid body");
        scene
            .insert(entity, Collider::sphere(0.5))
            .expect("collider");

        let mut physics = PhysicsScene::new(PhysicsWorldConfig::default()).expect("physics");
        physics.synchronize(&mut scene).expect("sync parented body");
        let binding = physics.bindings[&entity];
        let initial_state = physics
            .physics
            .body_state(binding.handle)
            .expect("initial body state");
        let initial_world_position = scene
            .world_transform(entity)
            .expect("world transform")
            .transform_point3(Vec3::ZERO);
        assert!(
            initial_state
                .position
                .abs_diff_eq(initial_world_position, 1.0e-5)
        );

        for _ in 0..10 {
            physics.step().expect("step");
        }
        physics
            .writeback_dynamic_transforms(&mut scene)
            .expect("writeback child local transform");
        let state = physics
            .physics
            .body_state(binding.handle)
            .expect("body state after simulation");
        let world_position = scene
            .world_transform(entity)
            .expect("updated world transform")
            .transform_point3(Vec3::ZERO);
        assert!(world_position.abs_diff_eq(state.position, 1.0e-4));
        assert!(scene.parent_of(entity).is_some());
    }

    #[test]
    fn uniform_parent_scale_is_baked_into_sphere_radius_and_recreated_on_change() {
        let mut scene = scene_with_physics_components();
        let parent = scene.spawn().expect("parent");
        scene
            .insert(
                parent,
                Transform::new(Vec3::ZERO, Quat::IDENTITY, Vec3::splat(2.0)).expect("parent scale"),
            )
            .expect("insert parent scale");
        let entity = scene.spawn().expect("child");
        scene.set_parent(entity, Some(parent)).expect("set parent");
        scene
            .insert(
                entity,
                RigidBody {
                    body_type: RigidBodyType::Static,
                    ..RigidBody::default()
                },
            )
            .expect("rigid body");
        scene
            .insert(entity, Collider::sphere(0.5))
            .expect("collider");
        let mut physics = PhysicsScene::new(PhysicsWorldConfig::default()).expect("physics");
        physics.synchronize(&mut scene).expect("initial sync");
        let ray = PhysicsRayCast::new(Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -3.0));
        let first_hit = physics.cast_ray(ray).expect("first ray")[0];
        assert!((first_hit.fraction - 2.0 / 3.0).abs() < 1.0e-4);

        scene
            .insert(
                parent,
                Transform::new(Vec3::new(5.0, 0.0, 0.0), Quat::IDENTITY, Vec3::splat(3.0))
                    .expect("updated parent scale"),
            )
            .expect("update parent scale");
        physics.synchronize(&mut scene).expect("rescale collider");
        let moved_ray = PhysicsRayCast::new(Vec3::new(5.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -3.0));
        let second_hit = physics.cast_ray(moved_ray).expect("second ray")[0];
        assert!((second_hit.fraction - 0.5).abs() < 1.0e-4);
        assert_eq!(physics.body_count(), 1);
    }

    #[test]
    fn box_colliders_support_non_uniform_parent_scale() {
        let mut scene = scene_with_physics_components();
        let entity = scene.spawn().expect("entity");
        scene
            .insert(
                entity,
                Transform::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 1.0, 1.0))
                    .expect("non-uniform scale"),
            )
            .expect("insert transform");
        scene
            .insert(
                entity,
                RigidBody {
                    body_type: RigidBodyType::Static,
                    ..RigidBody::default()
                },
            )
            .expect("rigid body");
        scene
            .insert(entity, Collider::cuboid(Vec3::splat(0.5)))
            .expect("box collider");
        let mut physics = PhysicsScene::new(PhysicsWorldConfig::default()).expect("physics");
        physics.synchronize(&mut scene).expect("sync scaled box");
        let ray = PhysicsRayCast::new(Vec3::new(3.0, 0.0, 0.0), Vec3::new(-6.0, 0.0, 0.0));
        let hit = physics.cast_ray(ray).expect("box ray")[0];
        assert!((hit.fraction - 1.0 / 3.0).abs() < 1.0e-4);
    }

    #[test]
    fn dynamic_parent_and_child_writeback_is_hierarchy_ordered() {
        let mut scene = scene_with_physics_components();
        let parent = scene.spawn().expect("parent");
        scene
            .insert(
                parent,
                Transform::new(Vec3::new(0.0, 5.0, 0.0), Quat::IDENTITY, Vec3::ONE)
                    .expect("parent transform"),
            )
            .expect("insert parent transform");
        scene
            .insert(parent, RigidBody::default())
            .expect("parent body");
        scene
            .insert(parent, Collider::sphere(0.4))
            .expect("parent collider");
        let child = scene.spawn().expect("child");
        scene.set_parent(child, Some(parent)).expect("parent child");
        scene
            .insert(
                child,
                Transform::new(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE)
                    .expect("child transform"),
            )
            .expect("insert child transform");
        scene
            .insert(child, RigidBody::default())
            .expect("child body");
        scene
            .insert(child, Collider::sphere(0.4))
            .expect("child collider");

        let mut physics = PhysicsScene::new(PhysicsWorldConfig::default()).expect("physics");
        physics.synchronize(&mut scene).expect("sync");
        for _ in 0..10 {
            physics.step().expect("step");
        }
        physics
            .writeback_dynamic_transforms(&mut scene)
            .expect("hierarchical writeback");

        for entity in [parent, child] {
            let binding = physics.bindings[&entity];
            let state = physics
                .physics
                .body_state(binding.handle)
                .expect("body state");
            let world_position = scene
                .world_transform(entity)
                .expect("world transform")
                .transform_point3(Vec3::ZERO);
            assert!(world_position.abs_diff_eq(state.position, 1.0e-4));
        }
    }

    #[test]
    fn dynamic_collider_rescaling_preserves_simulated_body_state() {
        let mut scene = scene_with_physics_components();
        let parent = scene.spawn().expect("parent");
        scene
            .insert(parent, Transform::default())
            .expect("parent transform");
        let entity = scene.spawn().expect("child");
        scene
            .set_parent(entity, Some(parent))
            .expect("parent child");
        scene
            .insert(entity, RigidBody::default())
            .expect("rigid body");
        scene
            .insert(entity, Collider::sphere(0.5))
            .expect("collider");
        let mut physics = PhysicsScene::new(PhysicsWorldConfig::default()).expect("physics");
        physics.synchronize(&mut scene).expect("initial sync");
        for _ in 0..5 {
            physics.step().expect("step");
        }
        let binding = physics.bindings[&entity];
        let before = physics
            .physics
            .body_state(binding.handle)
            .expect("body state before scale");
        scene
            .insert(
                parent,
                Transform::new(Vec3::ZERO, Quat::IDENTITY, Vec3::splat(2.0))
                    .expect("scaled parent"),
            )
            .expect("rescale parent");
        physics.synchronize(&mut scene).expect("rescale collider");
        let after_binding = physics.bindings[&entity];
        let after = physics
            .physics
            .body_state(after_binding.handle)
            .expect("body state after scale");
        assert!(after.position.abs_diff_eq(before.position, 1.0e-5));
        assert!(
            after
                .linear_velocity
                .abs_diff_eq(before.linear_velocity, 1.0e-5)
        );
        assert_eq!(physics.body_count(), 1);
    }

    #[test]
    fn rejects_non_uniform_sphere_scale_and_sheared_world_transforms() {
        let mut scene = scene_with_physics_components();
        let entity = scene.spawn().expect("entity");
        scene
            .insert(
                entity,
                Transform::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 1.0, 1.0))
                    .expect("non-uniform scale"),
            )
            .expect("insert transform");
        scene
            .insert(entity, RigidBody::default())
            .expect("rigid body");
        scene
            .insert(entity, Collider::sphere(0.5))
            .expect("sphere collider");
        let mut physics = PhysicsScene::new(PhysicsWorldConfig::default()).expect("physics");
        assert!(matches!(
            physics.synchronize(&mut scene),
            Err(PhysicsSceneError::NonUniformSphereScale(found)) if found == entity
        ));
        assert_eq!(physics.body_count(), 0);

        scene
            .remove::<RigidBody>(entity)
            .expect("remove invalid sphere body");
        let parent = scene.spawn().expect("parent");
        scene
            .insert(
                parent,
                Transform::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 1.0, 1.0))
                    .expect("parent scale"),
            )
            .expect("insert parent transform");
        let child = scene.spawn().expect("child");
        scene.set_parent(child, Some(parent)).expect("parent child");
        scene
            .insert(
                child,
                Transform::new(Vec3::ZERO, Quat::from_rotation_z(0.5), Vec3::ONE)
                    .expect("rotated child"),
            )
            .expect("insert child transform");
        scene
            .insert(child, RigidBody::default())
            .expect("child rigid body");
        scene
            .insert(child, Collider::cuboid(Vec3::splat(0.5)))
            .expect("child box collider");
        assert!(matches!(
            physics.synchronize(&mut scene),
            Err(PhysicsSceneError::InvalidTransform(found)) if found == child
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

#[cfg(test)]
mod query_event_tests {
    use super::*;
    use z_ideal_math::Quat;

    fn scene_with_physics_components() -> SceneWorld {
        let mut scene = SceneWorld::new();
        scene.register_component::<Transform>().expect("transform");
        scene.register_component::<RigidBody>().expect("rigid body");
        scene.register_component::<Collider>().expect("collider");
        scene
    }

    #[test]
    fn scene_ray_cast_hits_are_sorted_and_resolve_to_entities() {
        let mut scene = scene_with_physics_components();
        let far = scene.spawn().expect("far entity");
        let near = scene.spawn().expect("near entity");
        for (entity, y) in [(far, 0.0), (near, 2.0)] {
            scene
                .insert(
                    entity,
                    Transform::new(Vec3::new(0.0, y, 0.0), Quat::IDENTITY, Vec3::ONE)
                        .expect("transform"),
                )
                .expect("insert transform");
            scene
                .insert(
                    entity,
                    RigidBody {
                        body_type: RigidBodyType::Static,
                        ..RigidBody::default()
                    },
                )
                .expect("rigid body");
            scene
                .insert(entity, Collider::cuboid(Vec3::splat(0.5)))
                .expect("collider");
        }
        let mut physics = PhysicsScene::new(PhysicsWorldConfig::default()).expect("physics");
        physics.synchronize(&mut scene).expect("sync");
        let hits = physics
            .cast_ray(PhysicsRayCast::new(
                Vec3::new(0.0, 5.0, 0.0),
                Vec3::new(0.0, -10.0, 0.0),
            ))
            .expect("ray cast");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].entity, near);
        assert_eq!(hits[1].entity, far);
    }

    #[test]
    fn scene_sensor_end_events_keep_the_destroyed_entity_reference_for_the_step() {
        let mut scene = scene_with_physics_components();
        let sensor = scene.spawn().expect("sensor entity");
        let visitor = scene.spawn().expect("visitor entity");
        scene
            .insert(
                sensor,
                RigidBody {
                    body_type: RigidBodyType::Static,
                    ..RigidBody::default()
                },
            )
            .expect("sensor body");
        scene
            .insert(
                sensor,
                Collider {
                    sensor: true,
                    sensor_events: true,
                    ..Collider::cuboid(Vec3::splat(1.0))
                },
            )
            .expect("sensor collider");
        scene
            .insert(visitor, RigidBody::default())
            .expect("visitor body");
        scene
            .insert(
                visitor,
                Collider {
                    sensor_events: true,
                    ..Collider::sphere(0.25)
                },
            )
            .expect("visitor collider");
        let mut physics = PhysicsScene::new(PhysicsWorldConfig {
            gravity: Vec3::ZERO,
            ..PhysicsWorldConfig::default()
        })
        .expect("physics");
        physics.synchronize(&mut scene).expect("initial sync");
        physics.step().expect("begin step");
        assert!(
            physics
                .events()
                .expect("begin events")
                .contains(&PhysicsSceneEvent::SensorBegin { sensor, visitor })
        );

        scene.despawn(visitor).expect("despawn visitor");
        physics
            .synchronize(&mut scene)
            .expect("remove visitor body");
        physics.step().expect("end step");
        assert!(
            physics
                .events()
                .expect("end events")
                .contains(&PhysicsSceneEvent::SensorEnd { sensor, visitor })
        );
    }

    #[test]
    fn scene_sensor_events_resolve_to_scene_entities() {
        let mut scene = scene_with_physics_components();
        let sensor = scene.spawn().expect("sensor entity");
        let visitor = scene.spawn().expect("visitor entity");
        scene
            .insert(
                sensor,
                RigidBody {
                    body_type: RigidBodyType::Static,
                    ..RigidBody::default()
                },
            )
            .expect("sensor body");
        scene
            .insert(
                sensor,
                Collider {
                    sensor: true,
                    sensor_events: true,
                    ..Collider::cuboid(Vec3::splat(1.0))
                },
            )
            .expect("sensor collider");
        scene
            .insert(visitor, RigidBody::default())
            .expect("visitor body");
        scene
            .insert(
                visitor,
                Collider {
                    sensor_events: true,
                    ..Collider::sphere(0.25)
                },
            )
            .expect("visitor collider");
        let mut physics = PhysicsScene::new(PhysicsWorldConfig {
            gravity: Vec3::ZERO,
            ..PhysicsWorldConfig::default()
        })
        .expect("physics");
        physics.synchronize(&mut scene).expect("sync");
        physics.step().expect("step");
        assert!(
            physics
                .events()
                .expect("scene events")
                .contains(&PhysicsSceneEvent::SensorBegin { sensor, visitor })
        );
    }
}

#[cfg(test)]
mod scene_shape_query_tests {
    use super::*;
    use crate::PhysicsQueryShape;
    use z_ideal_math::{Quat, Transform, Vec3};

    fn scene_with_physics_components() -> SceneWorld {
        let mut scene = SceneWorld::new();
        scene.register_component::<Transform>().expect("transform");
        scene.register_component::<RigidBody>().expect("rigid body");
        scene.register_component::<Collider>().expect("collider");
        scene
    }

    #[test]
    fn scene_shape_cast_and_overlap_queries_resolve_entity_handles() {
        let mut scene = scene_with_physics_components();
        let entity = scene.spawn().expect("entity");
        scene
            .insert(
                entity,
                RigidBody {
                    body_type: RigidBodyType::Static,
                    ..RigidBody::default()
                },
            )
            .expect("rigid body");
        scene
            .insert(entity, Collider::cuboid(Vec3::splat(0.5)))
            .expect("collider");

        let mut physics = PhysicsScene::new(PhysicsWorldConfig {
            gravity: Vec3::ZERO,
            ..PhysicsWorldConfig::default()
        })
        .expect("physics scene");
        physics.synchronize(&mut scene).expect("sync");

        let overlap = physics
            .overlap_shape(PhysicsShapeOverlap::new(
                Vec3::ZERO,
                Quat::IDENTITY,
                PhysicsQueryShape::Sphere { radius: 1.0 },
            ))
            .expect("overlap");
        assert_eq!(overlap, vec![PhysicsSceneOverlapHit { entity }]);

        let cast = physics
            .cast_shape(PhysicsShapeCast::new(
                Vec3::new(-3.0, 0.0, 0.0),
                Quat::IDENTITY,
                Vec3::new(6.0, 0.0, 0.0),
                PhysicsQueryShape::Sphere { radius: 0.25 },
            ))
            .expect("cast");
        assert_eq!(cast.len(), 1);
        assert_eq!(cast[0].entity, entity);
        assert!(cast[0].fraction.is_finite());
    }
}

#[cfg(test)]
mod scene_world_access_tests {
    use super::*;
    use z_ideal_math::Vec3;

    #[test]
    fn scene_exposes_its_physics_world_with_mutable_configuration() {
        let mut scene = PhysicsScene::new(PhysicsWorldConfig::default()).expect("physics scene");
        scene
            .physics_world_mut()
            .set_gravity(Vec3::ZERO)
            .expect("set gravity");
        assert_eq!(scene.physics_world().config().gravity, Vec3::ZERO);
        assert_eq!(scene.body_count(), 0);
    }
}
