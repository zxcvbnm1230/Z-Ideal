//! Engine math types and explicit world-space conventions.
//!
//! Z Ideal uses a right-handed coordinate system, +Y as up, and -Z as forward.
//! Angles are radians. Matrices use glam's column-major representation and are
//! composed for column vectors: a child world matrix is parent * child local.

#![forbid(unsafe_code)]

pub use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

/// World-space right direction.
pub const RIGHT: Vec3 = Vec3::X;
/// World-space up direction.
pub const UP: Vec3 = Vec3::Y;
/// World-space forward direction.
pub const FORWARD: Vec3 = Vec3::NEG_Z;

/// A validated local-space translation, rotation, and scale.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transform {
    translation: Vec3,
    rotation: Quat,
    scale: Vec3,
}

/// Invalid transform input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TransformError {
    /// Translation contains a NaN or infinity.
    #[error("translation must contain only finite values")]
    NonFiniteTranslation,
    /// Rotation contains invalid values or has zero length.
    #[error("rotation must be a finite, non-zero quaternion")]
    InvalidRotation,
    /// Scale contains a NaN or infinity.
    #[error("scale must contain only finite values")]
    NonFiniteScale,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    /// Creates a transform after validating and normalizing its rotation.
    pub fn new(translation: Vec3, rotation: Quat, scale: Vec3) -> Result<Self, TransformError> {
        if !translation.is_finite() {
            return Err(TransformError::NonFiniteTranslation);
        }
        if !scale.is_finite() {
            return Err(TransformError::NonFiniteScale);
        }
        let length_squared = rotation.length_squared();
        if !rotation.is_finite() || !length_squared.is_finite() || length_squared <= f32::EPSILON {
            return Err(TransformError::InvalidRotation);
        }
        Ok(Self {
            translation,
            rotation: rotation.normalize(),
            scale,
        })
    }

    /// Returns the local translation.
    #[must_use]
    pub const fn translation(&self) -> Vec3 {
        self.translation
    }

    /// Returns the local rotation.
    #[must_use]
    pub const fn rotation(&self) -> Quat {
        self.rotation
    }

    /// Returns the local scale.
    #[must_use]
    pub const fn scale(&self) -> Vec3 {
        self.scale
    }

    /// Replaces the translation after checking it is finite.
    pub fn set_translation(&mut self, translation: Vec3) -> Result<(), TransformError> {
        if !translation.is_finite() {
            return Err(TransformError::NonFiniteTranslation);
        }
        self.translation = translation;
        Ok(())
    }

    /// Replaces the rotation after checking and normalizing it.
    pub fn set_rotation(&mut self, rotation: Quat) -> Result<(), TransformError> {
        let length_squared = rotation.length_squared();
        if !rotation.is_finite() || !length_squared.is_finite() || length_squared <= f32::EPSILON {
            return Err(TransformError::InvalidRotation);
        }
        self.rotation = rotation.normalize();
        Ok(())
    }

    /// Replaces the scale after checking it is finite.
    pub fn set_scale(&mut self, scale: Vec3) -> Result<(), TransformError> {
        if !scale.is_finite() {
            return Err(TransformError::NonFiniteScale);
        }
        self.scale = scale;
        Ok(())
    }

    /// Validates values loaded from serialized data.
    pub fn validate(&self) -> Result<(), TransformError> {
        if !self.translation.is_finite() {
            return Err(TransformError::NonFiniteTranslation);
        }
        if !self.scale.is_finite() {
            return Err(TransformError::NonFiniteScale);
        }
        let length_squared = self.rotation.length_squared();
        if !self.rotation.is_finite()
            || !length_squared.is_finite()
            || (length_squared - 1.0).abs() > 1.0e-3
        {
            return Err(TransformError::InvalidRotation);
        }
        Ok(())
    }

    /// Builds the local-to-parent matrix.
    #[must_use]
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

#[cfg(test)]
mod tests {
    use super::{FORWARD, Quat, RIGHT, Transform, TransformError, UP, Vec3};

    #[test]
    fn coordinate_convention_is_explicit() {
        assert_eq!(RIGHT, Vec3::X);
        assert_eq!(UP, Vec3::Y);
        assert_eq!(FORWARD, Vec3::NEG_Z);
    }

    #[test]
    fn transform_defaults_to_identity() {
        assert_eq!(Transform::default().matrix(), glam::Mat4::IDENTITY);
    }

    #[test]
    fn constructor_normalizes_rotation() {
        let transform =
            Transform::new(Vec3::ZERO, Quat::from_xyzw(0.0, 2.0, 0.0, 0.0), Vec3::ONE)
                .expect("valid rotation");
        assert!((transform.rotation().length_squared() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn rejects_non_finite_translation_and_scale() {
        assert_eq!(
            Transform::new(Vec3::splat(f32::NAN), Quat::IDENTITY, Vec3::ONE),
            Err(TransformError::NonFiniteTranslation)
        );
        assert_eq!(
            Transform::new(Vec3::ZERO, Quat::IDENTITY, Vec3::splat(f32::INFINITY)),
            Err(TransformError::NonFiniteScale)
        );
    }

    #[test]
    fn rejects_zero_quaternion() {
        assert_eq!(
            Transform::new(
                Vec3::ZERO,
                Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
                Vec3::ONE
            ),
            Err(TransformError::InvalidRotation)
        );
    }

    #[test]
    fn matrix_applies_translation() {
        let transform =
            Transform::new(Vec3::new(2.0, 3.0, 4.0), Quat::IDENTITY, Vec3::ONE)
                .expect("valid transform");
        assert_eq!(
            transform.matrix().transform_point3(Vec3::ZERO),
            Vec3::new(2.0, 3.0, 4.0)
        );
    }
}
