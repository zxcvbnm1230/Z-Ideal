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

/// A validated perspective camera projection using right-handed, zero-to-one depth.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerspectiveCamera {
    vertical_fov_radians: f32,
    aspect_ratio: f32,
    near_plane: f32,
    far_plane: f32,
}

/// Invalid perspective camera parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CameraError {
    /// Vertical field of view must be finite and strictly between zero and pi.
    #[error("vertical field of view must be finite and between zero and pi radians")]
    InvalidFieldOfView,
    /// Aspect ratio must be finite and greater than zero.
    #[error("camera aspect ratio must be finite and greater than zero")]
    InvalidAspectRatio,
    /// Near and far planes must be finite, positive, and ordered.
    #[error("camera clipping planes must be finite, positive, and near must be less than far")]
    InvalidClipPlanes,
    /// Valid parameters would overflow the finite projection matrix representation.
    #[error("camera parameters produce a non-finite projection matrix")]
    ProjectionOverflow,
}

impl PerspectiveCamera {
    /// Creates a validated perspective projection.
    pub fn new(
        vertical_fov_radians: f32,
        aspect_ratio: f32,
        near_plane: f32,
        far_plane: f32,
    ) -> Result<Self, CameraError> {
        let camera = Self {
            vertical_fov_radians,
            aspect_ratio,
            near_plane,
            far_plane,
        };
        camera.validate()?;
        Ok(camera)
    }

    /// Validates values loaded from serialized scene data.
    pub fn validate(&self) -> Result<(), CameraError> {
        if !self.vertical_fov_radians.is_finite()
            || self.vertical_fov_radians <= 0.0
            || self.vertical_fov_radians >= std::f32::consts::PI
        {
            return Err(CameraError::InvalidFieldOfView);
        }
        if !self.aspect_ratio.is_finite() || self.aspect_ratio <= 0.0 {
            return Err(CameraError::InvalidAspectRatio);
        }
        if !self.near_plane.is_finite()
            || !self.far_plane.is_finite()
            || self.near_plane <= 0.0
            || self.far_plane <= self.near_plane
        {
            return Err(CameraError::InvalidClipPlanes);
        }
        let focal = 1.0 / (self.vertical_fov_radians * 0.5).tan();
        let depth_scale = self.far_plane / (self.near_plane - self.far_plane);
        let projection_values = [
            focal / self.aspect_ratio,
            focal,
            depth_scale,
            self.near_plane * depth_scale,
        ];
        if projection_values.iter().any(|value| !value.is_finite()) {
            return Err(CameraError::ProjectionOverflow);
        }
        Ok(())
    }

    /// Returns the vertical field of view in radians.
    #[must_use]
    pub const fn vertical_fov_radians(&self) -> f32 {
        self.vertical_fov_radians
    }
    /// Returns the camera aspect ratio.
    #[must_use]
    pub const fn aspect_ratio(&self) -> f32 {
        self.aspect_ratio
    }
    /// Returns the near clipping distance.
    #[must_use]
    pub const fn near_plane(&self) -> f32 {
        self.near_plane
    }
    /// Returns the far clipping distance.
    #[must_use]
    pub const fn far_plane(&self) -> f32 {
        self.far_plane
    }

    /// Recomputes the aspect ratio after a drawable-size change.
    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) -> Result<(), CameraError> {
        let previous = self.aspect_ratio;
        self.aspect_ratio = aspect_ratio;
        if let Err(error) = self.validate() {
            self.aspect_ratio = previous;
            return Err(error);
        }
        Ok(())
    }

    /// Builds a right-handed projection matrix with Vulkan-compatible zero-to-one depth.
    pub fn projection_matrix(&self) -> Result<Mat4, CameraError> {
        self.validate()?;
        let focal = 1.0 / (self.vertical_fov_radians * 0.5).tan();
        let depth_scale = self.far_plane / (self.near_plane - self.far_plane);
        Ok(Mat4::from_cols(
            Vec4::new(focal / self.aspect_ratio, 0.0, 0.0, 0.0),
            Vec4::new(0.0, focal, 0.0, 0.0),
            Vec4::new(0.0, 0.0, depth_scale, -1.0),
            Vec4::new(0.0, 0.0, self.near_plane * depth_scale, 0.0),
        ))
    }
}

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
    use super::{
        CameraError, FORWARD, PerspectiveCamera, Quat, RIGHT, Transform, TransformError, UP, Vec3,
    };

    #[test]
    fn perspective_camera_validates_projection_and_resize_aspect() {
        let mut camera =
            PerspectiveCamera::new(1.0, 16.0 / 9.0, 0.1, 1000.0).expect("valid camera");
        let projection = camera.projection_matrix().expect("finite projection");
        assert!(projection.is_finite());
        let near_clip = projection * glam::Vec4::new(0.0, 0.0, -0.1, 1.0);
        let far_clip = projection * glam::Vec4::new(0.0, 0.0, -1000.0, 1.0);
        assert!(near_clip.z.abs() < 1.0e-5 * near_clip.w.abs());
        assert!((far_clip.z / far_clip.w - 1.0).abs() < 1.0e-5);
        camera
            .set_aspect_ratio(4.0 / 3.0)
            .expect("valid resized aspect");
        assert!((camera.aspect_ratio() - 4.0 / 3.0).abs() < 1.0e-6);
        assert_eq!(
            camera.set_aspect_ratio(0.0),
            Err(CameraError::InvalidAspectRatio)
        );
        assert_eq!(
            PerspectiveCamera::new(1.0, 1.0, 10.0, 1.0),
            Err(CameraError::InvalidClipPlanes)
        );
        assert_eq!(
            PerspectiveCamera::new(1.0, f32::from_bits(1), 0.1, 100.0),
            Err(CameraError::ProjectionOverflow)
        );
    }

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
        let transform = Transform::new(Vec3::ZERO, Quat::from_xyzw(0.0, 2.0, 0.0, 0.0), Vec3::ONE)
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
            Transform::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 0.0), Vec3::ONE),
            Err(TransformError::InvalidRotation)
        );
    }

    #[test]
    fn matrix_applies_translation() {
        let transform = Transform::new(Vec3::new(2.0, 3.0, 4.0), Quat::IDENTITY, Vec3::ONE)
            .expect("valid transform");
        assert_eq!(
            transform.matrix().transform_point3(Vec3::ZERO),
            Vec3::new(2.0, 3.0, 4.0)
        );
    }
}
