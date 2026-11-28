//! Backend-neutral world, rigid-body and collider API backed by Box3D.

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use boxddd::{
    BodyDefBuilder, BodyType as Box3dBodyType, BoxHull, Foundation, FoundationConfig, Pos,
    Quat as Box3dQuat, ShapeDef, Sphere, StallThreshold, Vec3 as Box3dVec3, World as Box3dWorld,
};
use z_ideal_math::{Quat, Vec3};

static NEXT_WORLD_ID: AtomicU64 = AtomicU64::new(1);

/// Configuration errors for PhysicsWorldConfig.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PhysicsConfigError {
    /// Gravity contains a non-finite component.
    #[error("physics gravity must contain only finite values")]
    InvalidGravity,
    /// The world scale must be finite and greater than zero.
    #[error("physics length units per meter must be finite and greater than zero")]
    InvalidLengthScale,
    /// The fixed step must be positive and representable as f32 seconds.
    #[error("physics fixed step must be positive and representable as f32 seconds")]
    InvalidFixedStep,
    /// At least one solver substep is required.
    #[error("physics solver substep count must be greater than zero")]
    InvalidSolverSubsteps,
}

/// Configuration for a Box3D-backed physics world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsWorldConfig {
    /// World gravity in engine units per second squared.
    pub gravity: Vec3,
    /// Number of engine length units represented by one meter.
    pub length_units_per_meter: f32,
    /// Fixed simulation step duration.
    pub fixed_step: Duration,
    /// Box3D solver substeps for each fixed simulation step.
    pub solver_substeps: u32,
}

impl Default for PhysicsWorldConfig {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            length_units_per_meter: 1.0,
            fixed_step: Duration::from_nanos(16_666_667),
            solver_substeps: 4,
        }
    }
}

impl PhysicsWorldConfig {
    /// Validates the configuration before initializing Box3D.
    pub fn validate(self) -> Result<(), PhysicsConfigError> {
        if !self.gravity.is_finite() {
            return Err(PhysicsConfigError::InvalidGravity);
        }
        if !self.length_units_per_meter.is_finite() || self.length_units_per_meter <= 0.0 {
            return Err(PhysicsConfigError::InvalidLengthScale);
        }
        let step_seconds = self.fixed_step.as_secs_f64();
        if self.fixed_step.is_zero()
            || !step_seconds.is_finite()
            || step_seconds > f64::from(f32::MAX)
            || (step_seconds as f32) == 0.0
        {
            return Err(PhysicsConfigError::InvalidFixedStep);
        }
        if self.solver_substeps == 0 || self.solver_substeps > i32::MAX as u32 {
            return Err(PhysicsConfigError::InvalidSolverSubsteps);
        }
        Ok(())
    }
}

/// Physics world construction or operation failure.
#[derive(Debug, thiserror::Error)]
pub enum PhysicsError {
    /// The configuration is invalid.
    #[error(transparent)]
    InvalidConfig(#[from] PhysicsConfigError),
    /// The Box3D backend rejected an operation.
    #[error("Box3D operation failed: {0}")]
    Backend(#[from] boxddd::Error),
    /// A body descriptor contains invalid values.
    #[error("rigid-body descriptor is invalid")]
    InvalidRigidBody,
    /// A collider descriptor contains invalid values.
    #[error("collider descriptor is invalid")]
    InvalidCollider,
    /// A requested runtime body state contains invalid values.
    #[error("rigid-body state must contain finite values and a non-zero rotation")]
    InvalidBodyState,
    /// The body handle is stale or belongs to a different physics world.
    #[error("physics body handle is stale or belongs to another world")]
    StaleBodyHandle,
    /// The world has exhausted the representable body slot index range.
    #[error("physics body slot capacity is exhausted")]
    BodyCapacityExceeded,
    /// A unique physics-world identifier could not be allocated.
    #[error("physics world identifier space is exhausted")]
    WorldIdExhausted,
    /// Native body creation failed and rollback also failed.
    #[error("body creation failed ({creation}); rollback also failed ({rollback})")]
    CreationRollbackFailed {
        creation: boxddd::Error,
        rollback: boxddd::Error,
    },
}

/// Stable body type independent of Box3D.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsBodyType {
    /// Immovable body.
    Static,
    /// Body moved by application-controlled target transforms.
    Kinematic,
    /// Body affected by simulation forces and contacts.
    Dynamic,
}

/// Rigid-body creation parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidBodyDesc {
    /// Simulation behavior.
    pub body_type: PhysicsBodyType,
    /// Initial world-space position.
    pub position: Vec3,
    /// Initial world-space orientation.
    pub rotation: Quat,
    /// Initial linear velocity.
    pub linear_velocity: Vec3,
    /// Scale applied to world gravity.
    pub gravity_scale: f32,
    /// Linear velocity damping.
    pub linear_damping: f32,
    /// Angular velocity damping.
    pub angular_damping: f32,
    /// Whether this body may sleep.
    pub allow_sleep: bool,
    /// Whether continuous collision handling is enabled for this body.
    pub bullet: bool,
}

impl Default for RigidBodyDesc {
    fn default() -> Self {
        Self {
            body_type: PhysicsBodyType::Dynamic,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            gravity_scale: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            allow_sleep: true,
            bullet: false,
        }
    }
}

impl RigidBodyDesc {
    fn validate(self) -> Result<(), PhysicsError> {
        let rotation_length = self.rotation.length_squared();
        if !self.position.is_finite()
            || !self.linear_velocity.is_finite()
            || !self.rotation.is_finite()
            || !rotation_length.is_finite()
            || rotation_length <= f32::EPSILON
            || !self.gravity_scale.is_finite()
            || !self.linear_damping.is_finite()
            || self.linear_damping < 0.0
            || !self.angular_damping.is_finite()
            || self.angular_damping < 0.0
        {
            return Err(PhysicsError::InvalidRigidBody);
        }
        Ok(())
    }
}

/// Supported primitive collider geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColliderShape {
    /// Sphere centered on the body's local origin.
    Sphere { radius: f32 },
    /// Axis-aligned box in body-local coordinates, specified by half extents.
    Box { half_extents: Vec3 },
}

/// Collider creation parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColliderDesc {
    /// Primitive collision geometry.
    pub shape: ColliderShape,
    /// Density used to calculate dynamic body mass.
    pub density: f32,
    /// Surface friction coefficient.
    pub friction: f32,
    /// Surface restitution coefficient.
    pub restitution: f32,
    /// Whether the shape is a non-solid sensor.
    pub sensor: bool,
    /// Whether contact begin/end events should be generated.
    pub contact_events: bool,
    /// Whether sensor overlap events should be generated.
    pub sensor_events: bool,
}

impl ColliderDesc {
    /// Creates a sphere collider with default material values.
    #[must_use]
    pub const fn sphere(radius: f32) -> Self {
        Self {
            shape: ColliderShape::Sphere { radius },
            density: 1.0,
            friction: 0.6,
            restitution: 0.0,
            sensor: false,
            contact_events: false,
            sensor_events: false,
        }
    }

    /// Creates a box collider with default material values.
    #[must_use]
    pub const fn cuboid(half_extents: Vec3) -> Self {
        Self {
            shape: ColliderShape::Box { half_extents },
            density: 1.0,
            friction: 0.6,
            restitution: 0.0,
            sensor: false,
            contact_events: false,
            sensor_events: false,
        }
    }

    fn validate(self) -> Result<(), PhysicsError> {
        let valid_shape = match self.shape {
            ColliderShape::Sphere { radius } => radius.is_finite() && radius > 0.0,
            ColliderShape::Box { half_extents } => {
                half_extents.is_finite() && half_extents.min_element() > 0.0
            }
        };
        if !valid_shape
            || !self.density.is_finite()
            || self.density < 0.0
            || !self.friction.is_finite()
            || self.friction < 0.0
            || !self.restitution.is_finite()
            || self.restitution < 0.0
        {
            return Err(PhysicsError::InvalidCollider);
        }
        Ok(())
    }
}

/// World-unique, generation-checked rigid-body handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicsBodyHandle {
    world_id: u64,
    index: u32,
    generation: u32,
}

/// Current simulated body state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsBodyState {
    /// World-space position.
    pub position: Vec3,
    /// World-space orientation.
    pub rotation: Quat,
    /// World-space linear velocity.
    pub linear_velocity: Vec3,
}

struct BodySlot {
    generation: u32,
    native: Option<boxddd::BodyId>,
}

/// One independently owned Box3D simulation world.
///
/// Box3D process configuration is frozen after its Foundation is initialized.
/// Every PhysicsWorld in one process must therefore use the same
/// length_units_per_meter; conflicting values are returned as backend errors.
pub struct PhysicsWorld {
    world_id: u64,
    config: PhysicsWorldConfig,
    foundation: &'static Foundation,
    native: Box3dWorld,
    bodies: Vec<BodySlot>,
    free_slots: Vec<u32>,
}

impl PhysicsWorld {
    /// Creates a physics world and initializes Box3D's process-wide foundation.
    pub fn new(config: PhysicsWorldConfig) -> Result<Self, PhysicsError> {
        config.validate()?;
        let foundation = Foundation::initialize(FoundationConfig {
            length_units_per_meter: config.length_units_per_meter,
            stall_threshold: StallThreshold::Disabled,
        })?;
        let world_def = foundation
            .world_def_builder()
            .gravity(to_box3d_vec3(config.gravity))
            .build()?;
        let native = foundation.create_world(world_def)?;
        let world_id = allocate_world_id()?;
        Ok(Self {
            world_id,
            config,
            foundation,
            native,
            bodies: Vec::new(),
            free_slots: Vec::new(),
        })
    }

    /// Returns the configuration used by this world.
    #[must_use]
    pub const fn config(&self) -> PhysicsWorldConfig {
        self.config
    }

    /// Returns the number of live rigid bodies.
    #[must_use]
    pub fn body_count(&self) -> usize {
        self.bodies
            .iter()
            .filter(|slot| slot.native.is_some())
            .count()
    }

    /// Replaces world gravity.
    pub fn set_gravity(&mut self, gravity: Vec3) -> Result<(), PhysicsError> {
        if !gravity.is_finite() {
            return Err(PhysicsConfigError::InvalidGravity.into());
        }
        self.native.set_gravity(to_box3d_vec3(gravity))?;
        self.config.gravity = gravity;
        Ok(())
    }

    /// Creates a body and its collider transactionally.
    ///
    /// If shape creation fails after native body creation, the body is destroyed
    /// before the error is returned. If rollback also fails, both errors are
    /// retained in CreationRollbackFailed.
    pub fn create_body(
        &mut self,
        body: RigidBodyDesc,
        collider: ColliderDesc,
    ) -> Result<PhysicsBodyHandle, PhysicsError> {
        body.validate()?;
        collider.validate()?;
        if self.free_slots.is_empty() && self.bodies.len() >= u32::MAX as usize {
            return Err(PhysicsError::BodyCapacityExceeded);
        }

        let body_def = self.foundation_body_def(body).build()?;
        let native_body = self.native.create_body(body_def)?;
        if let Err(creation) = self.create_native_collider(native_body, collider) {
            return match self.native.destroy_body(native_body) {
                Ok(()) => Err(PhysicsError::Backend(creation)),
                Err(rollback) => Err(PhysicsError::CreationRollbackFailed { creation, rollback }),
            };
        }

        let (index, generation) = if let Some(index) = self.free_slots.pop() {
            let slot = &mut self.bodies[index as usize];
            debug_assert!(slot.native.is_none());
            slot.native = Some(native_body);
            (index, slot.generation)
        } else {
            let index =
                u32::try_from(self.bodies.len()).map_err(|_| PhysicsError::BodyCapacityExceeded)?;
            self.bodies.push(BodySlot {
                generation: 1,
                native: Some(native_body),
            });
            (index, 1)
        };

        Ok(PhysicsBodyHandle {
            world_id: self.world_id,
            index,
            generation,
        })
    }

    /// Destroys a body and invalidates its handle.
    pub fn destroy_body(&mut self, handle: PhysicsBodyHandle) -> Result<(), PhysicsError> {
        let native = self.native_body(handle)?;
        self.native.destroy_body(native)?;
        let slot = &mut self.bodies[handle.index as usize];
        slot.native = None;
        if let Some(next_generation) = slot.generation.checked_add(1) {
            slot.generation = next_generation;
            self.free_slots.push(handle.index);
        }
        Ok(())
    }

    /// Advances exactly one configured fixed simulation step.
    pub fn step(&mut self) -> Result<(), PhysicsError> {
        self.native.step(
            self.config.fixed_step.as_secs_f32(),
            self.config.solver_substeps as i32,
        )?;
        Ok(())
    }

    /// Replaces a body's world-space position and orientation.
    pub fn set_body_transform(
        &mut self,
        handle: PhysicsBodyHandle,
        position: Vec3,
        rotation: Quat,
    ) -> Result<(), PhysicsError> {
        let rotation_length = rotation.length_squared();
        if !position.is_finite()
            || !rotation.is_finite()
            || !rotation_length.is_finite()
            || rotation_length <= f32::EPSILON
        {
            return Err(PhysicsError::InvalidBodyState);
        }
        let native = self.native_body(handle)?;
        self.native
            .set_body_transform(native, to_box3d_pos(position), to_box3d_quat(rotation))?;
        Ok(())
    }

    /// Replaces a body's world-space linear velocity.
    pub fn set_body_linear_velocity(
        &mut self,
        handle: PhysicsBodyHandle,
        velocity: Vec3,
    ) -> Result<(), PhysicsError> {
        if !velocity.is_finite() {
            return Err(PhysicsError::InvalidBodyState);
        }
        let native = self.native_body(handle)?;
        self.native
            .set_body_linear_velocity(native, to_box3d_vec3(velocity))?;
        Ok(())
    }

    /// Reads the current simulated state of a body.
    pub fn body_state(&self, handle: PhysicsBodyHandle) -> Result<PhysicsBodyState, PhysicsError> {
        let native = self.native_body(handle)?;
        let position = self.native.body_position(native)?;
        let rotation = self.native.body_rotation(native)?;
        let velocity = self.native.body_linear_velocity(native)?;
        Ok(PhysicsBodyState {
            position: Vec3::new(position.x, position.y, position.z),
            rotation: Quat::from_xyzw(rotation.v.x, rotation.v.y, rotation.v.z, rotation.s),
            linear_velocity: Vec3::new(velocity.x, velocity.y, velocity.z),
        })
    }

    fn native_body(&self, handle: PhysicsBodyHandle) -> Result<boxddd::BodyId, PhysicsError> {
        if handle.world_id != self.world_id {
            return Err(PhysicsError::StaleBodyHandle);
        }
        self.bodies
            .get(handle.index as usize)
            .filter(|slot| slot.generation == handle.generation)
            .and_then(|slot| slot.native)
            .ok_or(PhysicsError::StaleBodyHandle)
    }

    fn foundation_body_def(&self, body: RigidBodyDesc) -> BodyDefBuilder {
        let body_type = match body.body_type {
            PhysicsBodyType::Static => Box3dBodyType::Static,
            PhysicsBodyType::Kinematic => Box3dBodyType::Kinematic,
            PhysicsBodyType::Dynamic => Box3dBodyType::Dynamic,
        };
        self.foundation
            .body_def_builder()
            .body_type(body_type)
            .position(to_box3d_pos(body.position))
            .rotation(to_box3d_quat(body.rotation))
            .linear_velocity(to_box3d_vec3(body.linear_velocity))
            .gravity_scale(body.gravity_scale)
            .linear_damping(body.linear_damping)
            .angular_damping(body.angular_damping)
            .enable_sleep(body.allow_sleep)
            .bullet(body.bullet)
    }

    fn create_native_collider(
        &mut self,
        body: boxddd::BodyId,
        collider: ColliderDesc,
    ) -> Result<(), boxddd::Error> {
        let shape_def: ShapeDef = self
            .foundation
            .shape_def_builder()
            .density(collider.density)
            .friction(collider.friction)
            .restitution(collider.restitution)
            .sensor(collider.sensor)
            .enable_contact_events(collider.contact_events)
            .enable_sensor_events(collider.sensor_events)
            .build()?;
        match collider.shape {
            ColliderShape::Sphere { radius } => {
                self.native.create_sphere_shape(
                    body,
                    &shape_def,
                    &Sphere::new(Box3dVec3::ZERO, radius),
                )?;
            }
            ColliderShape::Box { half_extents } => {
                let hull = BoxHull::new(half_extents.x, half_extents.y, half_extents.z)?;
                self.native.create_hull_shape(body, &shape_def, &hull)?;
            }
        }
        Ok(())
    }
}

fn allocate_world_id() -> Result<u64, PhysicsError> {
    let mut current = NEXT_WORLD_ID.load(Ordering::Relaxed);
    loop {
        let next = current
            .checked_add(1)
            .ok_or(PhysicsError::WorldIdExhausted)?;
        match NEXT_WORLD_ID.compare_exchange_weak(
            current,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return Ok(current),
            Err(observed) => current = observed,
        }
    }
}

fn to_box3d_vec3(value: Vec3) -> Box3dVec3 {
    Box3dVec3::new(value.x, value.y, value.z)
}

fn to_box3d_pos(value: Vec3) -> Pos {
    Pos::new(value.x, value.y, value.z)
}

fn to_box3d_quat(value: Quat) -> Box3dQuat {
    let value = value.normalize();
    Box3dQuat::new(Box3dVec3::new(value.x, value.y, value.z), value.w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_world_configuration_before_initialization() {
        let config = PhysicsWorldConfig {
            gravity: Vec3::new(f32::NAN, 0.0, 0.0),
            ..PhysicsWorldConfig::default()
        };
        assert!(matches!(
            PhysicsWorld::new(config),
            Err(PhysicsError::InvalidConfig(
                PhysicsConfigError::InvalidGravity
            ))
        ));
    }

    #[test]
    fn rejects_invalid_body_and_collider_descriptors() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default()).expect("world");
        assert!(matches!(
            world.create_body(
                RigidBodyDesc {
                    position: Vec3::splat(f32::INFINITY),
                    ..RigidBodyDesc::default()
                },
                ColliderDesc::sphere(0.5),
            ),
            Err(PhysicsError::InvalidRigidBody)
        ));
        assert!(matches!(
            world.create_body(RigidBodyDesc::default(), ColliderDesc::sphere(0.0)),
            Err(PhysicsError::InvalidCollider)
        ));
        assert_eq!(world.body_count(), 0);
    }

    #[test]
    fn dynamic_sphere_moves_under_gravity_after_fixed_steps() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default()).expect("world");
        let body = world
            .create_body(
                RigidBodyDesc {
                    position: Vec3::new(0.0, 5.0, 0.0),
                    ..RigidBodyDesc::default()
                },
                ColliderDesc::sphere(0.5),
            )
            .expect("create sphere body");
        for _ in 0..30 {
            world.step().expect("fixed step");
        }
        let state = world.body_state(body).expect("body state");
        assert!(state.position.y < 5.0);
        assert!(state.linear_velocity.y < 0.0);
    }

    #[test]
    fn body_handle_is_invalidated_on_destroy_and_generation_changes_on_reuse() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default()).expect("world");
        let first = world
            .create_body(RigidBodyDesc::default(), ColliderDesc::sphere(0.5))
            .expect("first body");
        world.destroy_body(first).expect("destroy first");
        assert!(matches!(
            world.body_state(first),
            Err(PhysicsError::StaleBodyHandle)
        ));
        let second = world
            .create_body(RigidBodyDesc::default(), ColliderDesc::sphere(0.5))
            .expect("second body");
        assert_eq!(first.index, second.index);
        assert_ne!(first.generation, second.generation);
    }

    #[test]
    fn handles_cannot_be_used_across_worlds() {
        let mut first_world =
            PhysicsWorld::new(PhysicsWorldConfig::default()).expect("first world");
        let second_world = PhysicsWorld::new(PhysicsWorldConfig::default()).expect("second world");
        let body = first_world
            .create_body(RigidBodyDesc::default(), ColliderDesc::sphere(0.5))
            .expect("body");
        assert!(matches!(
            second_world.body_state(body),
            Err(PhysicsError::StaleBodyHandle)
        ));
    }

    #[test]
    fn runtime_body_state_can_be_updated_and_read_back() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default()).expect("world");
        let body = world
            .create_body(RigidBodyDesc::default(), ColliderDesc::sphere(0.5))
            .expect("body");
        let position = Vec3::new(3.0, 4.0, 5.0);
        let rotation = Quat::from_rotation_y(0.5);
        let velocity = Vec3::new(1.0, 2.0, 3.0);
        world
            .set_body_transform(body, position, rotation)
            .expect("set body transform");
        world
            .set_body_linear_velocity(body, velocity)
            .expect("set body velocity");
        let state = world.body_state(body).expect("body state");
        assert!((state.position - position).length() < 1.0e-5);
        assert!(state.rotation.abs_diff_eq(rotation, 1.0e-5));
        assert!((state.linear_velocity - velocity).length() < 1.0e-5);
    }

    #[test]
    fn runtime_body_state_rejects_invalid_values_without_mutation() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default()).expect("world");
        let body = world
            .create_body(RigidBodyDesc::default(), ColliderDesc::sphere(0.5))
            .expect("body");
        let before = world.body_state(body).expect("before state");
        assert!(matches!(
            world.set_body_transform(body, Vec3::splat(f32::NAN), Quat::IDENTITY),
            Err(PhysicsError::InvalidBodyState)
        ));
        assert!(matches!(
            world.set_body_transform(body, Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)),
            Err(PhysicsError::InvalidBodyState)
        ));
        assert!(matches!(
            world.set_body_linear_velocity(body, Vec3::splat(f32::INFINITY)),
            Err(PhysicsError::InvalidBodyState)
        ));
        assert_eq!(world.body_state(body).expect("after state"), before);
    }

    #[test]
    fn static_box_collider_and_dynamic_sphere_can_share_a_world() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default()).expect("world");
        let ground = world
            .create_body(
                RigidBodyDesc {
                    body_type: PhysicsBodyType::Static,
                    position: Vec3::new(0.0, -1.0, 0.0),
                    ..RigidBodyDesc::default()
                },
                ColliderDesc::cuboid(Vec3::new(5.0, 0.5, 5.0)),
            )
            .expect("ground");
        let ball = world
            .create_body(
                RigidBodyDesc {
                    position: Vec3::new(0.0, 2.0, 0.0),
                    ..RigidBodyDesc::default()
                },
                ColliderDesc::sphere(0.5),
            )
            .expect("ball");
        for _ in 0..120 {
            world.step().expect("fixed step");
        }
        let ground_state = world.body_state(ground).expect("ground state");
        let ball_state = world.body_state(ball).expect("ball state");
        assert_eq!(ground_state.position.y, -1.0);
        assert!(ball_state.position.y > -0.6);
        assert!(ball_state.position.y < 1.0);
    }
}
