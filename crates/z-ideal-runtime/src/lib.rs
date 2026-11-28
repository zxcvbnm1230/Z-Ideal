//! Packaged game startup, scene loading, and runtime loop; no editor dependency.
#![forbid(unsafe_code)]

mod gltf_material;
mod gltf_pbr;
mod gltf_scene;
mod render_scene;
mod scene_renderer;
mod shadow;
mod vulkan_scene;

pub use gltf_material::{
    GltfMaterialError, GltfMaterialTextureResource, GltfMaterialTextureResources,
    GltfMaterialTextureSet, GltfMaterialTextureViews, GltfMaterialUniform,
    GltfPbrEnvironmentResources, GltfPbrLocalLight, GltfPbrLocalLightKind, GltfPbrSceneUniform,
    MAX_GLTF_PBR_LOCAL_LIGHTS, apply_gltf_material_pipeline_state, write_gltf_material_descriptors,
    write_gltf_pbr_scene_descriptor,
};
pub use gltf_pbr::{GLTF_PBR_FRAGMENT_SHADER, GLTF_PBR_VERTEX_SHADER, compile_gltf_pbr_shaders};
pub use gltf_scene::{
    GltfMeshInstance, GltfNodeName, GltfSceneInstance, GltfSceneInstantiationError,
    ImportedNodeEntity, instantiate_default_gltf_scene, instantiate_gltf_scene,
};
pub use render_scene::{
    RenderMeshInstance, RenderSceneExtractionError, RenderSceneSnapshot, TransparentSortError,
    extract_render_scene, sort_transparent_mesh_instances_back_to_front,
};
pub use scene_renderer::{VulkanSceneRenderer, VulkanSceneRendererError, VulkanSceneViewport};
pub use shadow::{
    SHADOW_FRAGMENT_SHADER, SHADOW_VERTEX_SHADER, VulkanShadowRenderer, VulkanShadowRendererError,
    compile_shadow_shaders,
};
pub use vulkan_scene::{
    GltfMaterialTextureBinding, GltfMaterialTextureSlot, VulkanSceneMeshes, VulkanSceneMeshesError,
};

/// Returns the crate's stable package name for diagnostics and tooling.
pub const CRATE_NAME: &str = "z-ideal-runtime";

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use z_ideal_assets::{AssetUri, GltfImport, ImportedMesh, ImportedNode, ImportedScene};
    use z_ideal_math::{Transform, Vec3};
    use z_ideal_scene::World;

    fn fixture() -> GltfImport {
        GltfImport {
            source_path: PathBuf::from("models/scene.gltf"),
            meshes: vec![ImportedMesh {
                source_index: 0,
                name: Some("triangle".to_owned()),
                primitives: Vec::new(),
            }],
            nodes: vec![
                ImportedNode {
                    source_index: 0,
                    name: Some("root".to_owned()),
                    translation: [1.0, 0.0, 0.0],
                    rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                    mesh_index: None,
                    children: vec![1],
                },
                ImportedNode {
                    source_index: 1,
                    name: Some("mesh".to_owned()),
                    translation: [0.0, 2.0, 0.0],
                    rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                    scale: [2.0, 2.0, 2.0],
                    mesh_index: Some(0),
                    children: Vec::new(),
                },
                ImportedNode {
                    source_index: 2,
                    name: Some("unused".to_owned()),
                    translation: [50.0, 0.0, 0.0],
                    rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                    mesh_index: None,
                    children: Vec::new(),
                },
            ],
            scenes: vec![ImportedScene {
                source_index: 0,
                name: Some("main".to_owned()),
                root_nodes: vec![0],
            }],
            default_scene: Some(0),
            materials: Vec::new(),
            textures: Vec::new(),
            images: Vec::new(),
            dependency_uris: Vec::new(),
        }
    }

    #[test]
    fn instantiates_scene_nodes_transforms_hierarchy_and_mesh_references() {
        let import = fixture();
        let uri = AssetUri::new("models/scene.gltf").expect("valid URI");
        let mut world = World::new();

        let instance = instantiate_default_gltf_scene(&mut world, &import, uri.clone())
            .expect("instantiate default scene");

        assert_eq!(instance.scene_index, 0);
        assert_eq!(instance.nodes.len(), 2);
        let root = instance.entity_for_node(0).expect("root entity");
        let mesh = instance.entity_for_node(1).expect("mesh entity");
        assert_eq!(instance.roots, vec![root]);
        assert_eq!(instance.entity_for_node(2), None);
        assert_eq!(world.parent_of(mesh), Some(root));
        assert_eq!(
            world
                .get::<GltfNodeName>(root)
                .map(|name| name.value.as_str()),
            Some("root")
        );
        assert_eq!(
            world
                .get::<GltfNodeName>(mesh)
                .map(|name| name.value.as_str()),
            Some("mesh")
        );
        assert_eq!(
            world.get::<GltfMeshInstance>(mesh),
            Some(&GltfMeshInstance {
                source_uri: uri,
                mesh_index: 0,
                source_node_index: 1,
            })
        );
        assert_eq!(
            world
                .get::<Transform>(mesh)
                .expect("local transform")
                .scale(),
            Vec3::splat(2.0)
        );
        assert_eq!(
            world
                .world_transform(mesh)
                .expect("world transform")
                .transform_point3(Vec3::ZERO),
            Vec3::new(1.0, 2.0, 0.0)
        );
    }

    #[test]
    fn runtime_components_round_trip_through_scene_serialization() {
        let uri = AssetUri::new("models/scene.gltf").expect("valid URI");
        let mut world = World::new();
        world
            .register_component::<Transform>()
            .expect("register Transform");
        world
            .register_component::<GltfMeshInstance>()
            .expect("register mesh instance");
        world
            .register_component::<GltfNodeName>()
            .expect("register node name");
        let entity = world.spawn().expect("spawn");
        world
            .insert(
                entity,
                GltfMeshInstance {
                    source_uri: uri,
                    mesh_index: 7,
                    source_node_index: 3,
                },
            )
            .expect("insert mesh instance");
        world
            .insert(
                entity,
                GltfNodeName {
                    value: "mesh".to_owned(),
                },
            )
            .expect("insert node name");

        let serialized = world.to_scene_json().expect("serialize scene");
        let mut restored = World::new();
        restored
            .register_component::<Transform>()
            .expect("register Transform");
        restored
            .register_component::<GltfMeshInstance>()
            .expect("register mesh instance");
        restored
            .register_component::<GltfNodeName>()
            .expect("register node name");
        restored
            .load_scene_json(&serialized)
            .expect("restore scene");
        let restored_entity = restored
            .entities_with::<GltfMeshInstance>()
            .into_iter()
            .next()
            .expect("restored mesh instance");
        assert_eq!(
            restored.get::<GltfMeshInstance>(restored_entity),
            world.get::<GltfMeshInstance>(entity)
        );
        assert_eq!(
            restored.get::<GltfNodeName>(restored_entity),
            world.get::<GltfNodeName>(entity)
        );
    }

    #[test]
    fn invalid_scene_reference_fails_before_mutating_world() {
        let mut import = fixture();
        import.scenes[0].root_nodes = vec![99];
        let mut world = World::new();
        let existing = world.spawn().expect("existing entity");
        let uri = AssetUri::new("models/scene.gltf").expect("valid URI");

        assert!(matches!(
            instantiate_gltf_scene(&mut world, &import, uri, 0),
            Err(GltfSceneInstantiationError::MissingReference {
                kind: "root node",
                index: 99
            })
        ));
        assert_eq!(world.entities(), vec![existing]);
    }

    #[test]
    fn malformed_reachable_hierarchy_is_rejected_without_partial_entities() {
        let mut import = fixture();
        import.nodes[1].children = vec![0];
        let mut world = World::new();
        let existing = world.spawn().expect("existing entity");
        let uri = AssetUri::new("models/scene.gltf").expect("valid URI");

        assert!(matches!(
            instantiate_gltf_scene(&mut world, &import, uri, 0),
            Err(GltfSceneInstantiationError::InvalidNode { .. })
        ));
        assert_eq!(world.entities(), vec![existing]);
    }

    #[test]
    fn missing_default_scene_is_reported() {
        let mut import = fixture();
        import.default_scene = None;
        let mut world = World::new();
        let uri = AssetUri::new("models/scene.gltf").expect("valid URI");
        assert!(matches!(
            instantiate_default_gltf_scene(&mut world, &import, uri),
            Err(GltfSceneInstantiationError::MissingDefaultScene)
        ));
        assert!(world.entities().is_empty());
    }

    #[test]
    fn crate_name_is_not_empty() {
        assert!(!CRATE_NAME.is_empty());
    }
}
