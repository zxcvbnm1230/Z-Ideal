//! Serializable scene components describing physics participation.

use serde::{Deserialize, Serialize};
use z_ideal_math::Vec3;

use crate::Component;

/// Simulation behavior for a scene rigid body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RigidBodyType {
    /// Immovable body.
    Static,
    /// Body moved by application-controlled transforms.
    Kinematic,
    /// Body moved by the physics simulation.
    Dynamic,
}

/// Physics body settings stored with a scene entity.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RigidBody {
    /// Simulation behavior.
    pub body_type: RigidBodyType,
    /// Scale applied to world gravity.
    pub gravity_scale: f32,
    /// Linear velocity damping.
    pub linear_damping: f32,
    /// Angular velocity damping.
    pub angular_damping: f32,
    /// Whether the body may sleep.
    pub allow_sleep: bool,
    /// Whether continuous collision handling is enabled.
    pub bullet: bool,
    /// Initial linear velocity when the physics body is created.
    pub linear_velocity: Vec3,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            body_type: RigidBodyType::Dynamic,
            gravity_scale: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            allow_sleep: true,
            bullet: false,
            linear_velocity: Vec3::ZERO,
        }
    }
}

impl Component for RigidBody {
    const ID: &'static str = "z_ideal.physics.rigid_body";

    fn validate(&self) -> Result<(), String> {
        if !self.gravity_scale.is_finite()
            || !self.linear_damping.is_finite()
            || self.linear_damping < 0.0
            || !self.angular_damping.is_finite()
            || self.angular_damping < 0.0
            || !self.linear_velocity.is_finite()
        {
            return Err("rigid-body values must be finite and damping must be non-negative".into());
        }
        Ok(())
    }
}

/// Primitive collider geometry represented in scene data.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ColliderShape {
    /// Sphere centered at the entity origin.
    Sphere { radius: f32 },
    /// Box specified by positive half extents.
    Box { half_extents: Vec3 },
}

/// Collider material and event settings stored with a scene entity.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Collider {
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

impl Collider {
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
}

impl Component for Collider {
    const ID: &'static str = "z_ideal.physics.collider";

    fn validate(&self) -> Result<(), String> {
        let shape_valid = match self.shape {
            ColliderShape::Sphere { radius } => radius.is_finite() && radius > 0.0,
            ColliderShape::Box { half_extents } => {
                half_extents.is_finite() && half_extents.min_element() > 0.0
            }
        };
        if !shape_valid
            || !self.density.is_finite()
            || self.density < 0.0
            || !self.friction.is_finite()
            || self.friction < 0.0
            || !self.restitution.is_finite()
            || self.restitution < 0.0
        {
            return Err("collider geometry and material values are invalid".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::World;

    #[test]
    fn physics_components_are_strictly_serialized_and_validated() {
        let mut world = World::new();
        world
            .register_component::<RigidBody>()
            .expect("register rigid body");
        world
            .register_component::<Collider>()
            .expect("register collider");
        let entity = world.spawn().expect("spawn entity");
        world
            .insert(entity, RigidBody::default())
            .expect("insert rigid body");
        world
            .insert(entity, Collider::sphere(0.5))
            .expect("insert collider");

        let unknown_field = r#"{"body_type":"dynamic","gravity_scale":1.0,"linear_damping":0.0,"angular_damping":0.0,"allow_sleep":true,"bullet":false,"linear_velocity":[0.0,0.0,0.0],"unexpected":true}"#;
        assert!(serde_json::from_str::<RigidBody>(unknown_field).is_err());

        let invalid = Collider {
            shape: ColliderShape::Sphere { radius: 0.0 },
            ..Collider::sphere(1.0)
        };
        assert!(matches!(
            world.insert(entity, invalid),
            Err(crate::WorldError::InvalidComponent {
                id: "z_ideal.physics.collider",
                ..
            })
        ));
    }
}
