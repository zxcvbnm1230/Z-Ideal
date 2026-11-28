//! Renderer-neutral extraction of camera and mesh state from an ECS world.

use z_ideal_math::{Mat4, PerspectiveCamera};
use z_ideal_scene::{Entity, World, WorldError};

use crate::GltfMeshInstance;

/// Camera matrices and mesh instances captured from one consistent world read.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderSceneSnapshot {
    /// Entity selected as the active camera.
    pub camera_entity: Entity,
    /// World-to-view matrix derived from the camera entity's world transform.
    pub view: Mat4,
    /// Vulkan-compatible right-handed zero-to-one-depth projection matrix.
    pub projection: Mat4,
    /// Mesh instances in deterministic entity order.
    pub meshes: Vec<RenderMeshInstance>,
}

/// One mesh instance with a stable source asset reference and world transform.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderMeshInstance {
    /// Entity that owns this mesh instance.
    pub entity: Entity,
    /// Project-relative source asset URI and imported mesh/node indices.
    pub source: GltfMeshInstance,
    /// Object-to-world transform.
    pub model: Mat4,
}

/// Errors while extracting render data from a world.
#[derive(Debug, thiserror::Error)]
pub enum RenderSceneExtractionError {
    /// The requested camera entity is stale or lacks a perspective camera component.
    #[error("entity {0:?} is not a live perspective camera")]
    MissingCamera(Entity),
    /// The camera's world transform is singular or non-finite.
    #[error("camera entity {0:?} has a non-invertible or non-finite world transform")]
    InvalidCameraTransform(Entity),
    /// A mesh entity's world transform is non-finite.
    #[error("mesh entity {0:?} has a non-finite world transform")]
    InvalidMeshTransform(Entity),
    /// ECS hierarchy evaluation failed.
    #[error(transparent)]
    World(#[from] WorldError),
    /// Camera projection parameters are invalid.
    #[error("camera projection is invalid: {0}")]
    Camera(#[from] z_ideal_math::CameraError),
}

/// Extracts camera and mesh state without exposing renderer or Vulkan types.
///
/// The caller explicitly selects the active camera; this avoids implicit,
/// nondeterministic camera selection when a world contains multiple cameras.
/// Meshes are returned in entity order, and their asset references remain
/// logical URI/index pairs for a later asset-resolution stage.
pub fn extract_render_scene(
    world: &World,
    camera_entity: Entity,
) -> Result<RenderSceneSnapshot, RenderSceneExtractionError> {
    let camera = world
        .get::<PerspectiveCamera>(camera_entity)
        .ok_or(RenderSceneExtractionError::MissingCamera(camera_entity))?;
    let camera_world = world.world_transform(camera_entity)?;
    let determinant = camera_world.determinant();
    if !camera_world.is_finite() || !determinant.is_finite() || determinant.abs() <= f32::EPSILON {
        return Err(RenderSceneExtractionError::InvalidCameraTransform(
            camera_entity,
        ));
    }
    let view = camera_world.inverse();
    if !view.is_finite() {
        return Err(RenderSceneExtractionError::InvalidCameraTransform(
            camera_entity,
        ));
    }
    let projection = camera.projection_matrix()?;

    let mut meshes = Vec::new();
    for entity in world.entities_with::<GltfMeshInstance>() {
        let model = world.world_transform(entity)?;
        if !model.is_finite() {
            return Err(RenderSceneExtractionError::InvalidMeshTransform(entity));
        }
        let source = world
            .get::<GltfMeshInstance>(entity)
            .ok_or(WorldError::StaleEntity(entity))?
            .clone();
        meshes.push(RenderMeshInstance {
            entity,
            source,
            model,
        });
    }

    Ok(RenderSceneSnapshot {
        camera_entity,
        view,
        projection,
        meshes,
    })
}

#[cfg(test)]
mod tests {
    use super::{RenderSceneExtractionError, extract_render_scene};
    use crate::GltfMeshInstance;
    use z_ideal_assets::AssetUri;
    use z_ideal_math::{PerspectiveCamera, Quat, Transform, Vec3};
    use z_ideal_scene::World;

    fn world_with_render_components() -> World {
        let mut world = World::new();
        world
            .register_component::<Transform>()
            .expect("register transform");
        world
            .register_component::<PerspectiveCamera>()
            .expect("register camera");
        world
            .register_component::<GltfMeshInstance>()
            .expect("register mesh instance");
        world
    }

    #[test]
    fn extracts_camera_and_mesh_world_matrices_deterministically() {
        let mut world = world_with_render_components();
        let camera_parent = world.spawn().expect("camera parent");
        let camera = world.spawn().expect("camera");
        let mesh_parent = world.spawn().expect("mesh parent");
        let mesh = world.spawn().expect("mesh");
        world
            .insert(
                camera_parent,
                Transform::new(Vec3::new(0.0, 0.0, 5.0), Quat::IDENTITY, Vec3::ONE)
                    .expect("camera parent transform"),
            )
            .expect("insert camera parent");
        world
            .insert(
                camera,
                PerspectiveCamera::new(1.0, 16.0 / 9.0, 0.1, 100.0).expect("camera"),
            )
            .expect("insert camera");
        world
            .set_parent(camera, Some(camera_parent))
            .expect("parent camera");
        world
            .insert(
                mesh_parent,
                Transform::new(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE)
                    .expect("mesh parent transform"),
            )
            .expect("insert mesh parent");
        world
            .insert(
                mesh,
                Transform::new(Vec3::new(0.0, 3.0, 0.0), Quat::IDENTITY, Vec3::ONE)
                    .expect("mesh transform"),
            )
            .expect("insert mesh transform");
        world
            .set_parent(mesh, Some(mesh_parent))
            .expect("parent mesh");
        world
            .insert(
                mesh,
                GltfMeshInstance {
                    source_uri: AssetUri::new("models/scene.gltf").expect("asset URI"),
                    mesh_index: 4,
                    source_node_index: 7,
                },
            )
            .expect("insert mesh instance");

        let snapshot = extract_render_scene(&world, camera).expect("extract render scene");
        assert_eq!(snapshot.camera_entity, camera);
        assert_eq!(
            snapshot.view.transform_point3(Vec3::new(0.0, 0.0, 5.0)),
            Vec3::ZERO
        );
        assert!(snapshot.projection.is_finite());
        assert_eq!(snapshot.meshes.len(), 1);
        assert_eq!(snapshot.meshes[0].entity, mesh);
        assert_eq!(
            snapshot.meshes[0].model.transform_point3(Vec3::ZERO),
            Vec3::new(2.0, 3.0, 0.0)
        );
        assert_eq!(snapshot.meshes[0].source.mesh_index, 4);
    }

    #[test]
    fn rejects_missing_camera_and_singular_camera_transform() {
        let mut world = world_with_render_components();
        let entity = world.spawn().expect("entity");
        assert!(matches!(
            extract_render_scene(&world, entity),
            Err(RenderSceneExtractionError::MissingCamera(_))
        ));
        world
            .insert(
                entity,
                PerspectiveCamera::new(1.0, 1.0, 0.1, 10.0).expect("camera"),
            )
            .expect("insert camera");
        world
            .insert(
                entity,
                Transform::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(0.0, 1.0, 1.0))
                    .expect("singular transform"),
            )
            .expect("insert transform");
        assert!(matches!(
            extract_render_scene(&world, entity),
            Err(RenderSceneExtractionError::InvalidCameraTransform(_))
        ));
    }

    #[test]
    fn empty_mesh_world_still_extracts_camera() {
        let mut world = world_with_render_components();
        let camera = world.spawn().expect("camera");
        world
            .insert(
                camera,
                PerspectiveCamera::new(1.0, 1.0, 0.1, 10.0).expect("camera"),
            )
            .expect("insert camera");
        let snapshot = extract_render_scene(&world, camera).expect("extract");
        assert!(snapshot.meshes.is_empty());
    }
}
