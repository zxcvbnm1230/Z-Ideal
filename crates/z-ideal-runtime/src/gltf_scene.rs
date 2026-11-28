//! Transactional instantiation of imported glTF scenes into an ECS world.

use serde::{Deserialize, Serialize};
use z_ideal_assets::{AssetUri, GltfImport};
use z_ideal_math::{Quat, Transform, TransformError, Vec3};
use z_ideal_scene::{Component, Entity, World, WorldError};

/// Mesh reference attached to a node entity instantiated from a glTF document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GltfMeshInstance {
    /// Project-relative URI of the imported glTF source asset.
    pub source_uri: AssetUri,
    /// Index into the source import's mesh table.
    pub mesh_index: usize,
    /// Index of the source glTF node that owns this instance.
    pub source_node_index: usize,
}

impl Component for GltfMeshInstance {
    const ID: &'static str = "z_ideal.gltf.mesh_instance";
}

/// Optional source name retained on an instantiated node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GltfNodeName {
    /// Original glTF node name.
    pub value: String,
}

impl Component for GltfNodeName {
    const ID: &'static str = "z_ideal.gltf.node_name";
}

/// Mapping from a source glTF node index to its generation-safe ECS entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImportedNodeEntity {
    /// Source node index in the imported document.
    pub source_node_index: usize,
    /// Entity created for this node.
    pub entity: Entity,
}

/// The ECS entities created for one imported glTF scene.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GltfSceneInstance {
    /// Source scene index.
    pub scene_index: usize,
    /// Instantiated root entities in source scene order.
    pub roots: Vec<Entity>,
    /// Instantiated nodes in deterministic depth-first source order.
    pub nodes: Vec<ImportedNodeEntity>,
}

impl GltfSceneInstance {
    /// Resolves a source node index to its entity in this instance.
    #[must_use]
    pub fn entity_for_node(&self, source_node_index: usize) -> Option<Entity> {
        self.nodes
            .iter()
            .find(|entry| entry.source_node_index == source_node_index)
            .map(|entry| entry.entity)
    }
}

/// Errors while validating or instantiating an imported glTF scene.
#[derive(Debug, thiserror::Error)]
pub enum GltfSceneInstantiationError {
    /// Requested scene index is absent.
    #[error("glTF scene index {0} does not exist")]
    MissingScene(usize),
    /// The document does not declare a default scene.
    #[error("glTF document does not declare a default scene")]
    MissingDefaultScene,
    /// Publicly constructible import data failed structural validation.
    #[error("invalid glTF node {node}: {reason}")]
    InvalidNode { node: usize, reason: String },
    /// A referenced node or mesh index is out of range.
    #[error("glTF scene references missing {kind} index {index}")]
    MissingReference { kind: &'static str, index: usize },
    /// ECS component registration, entity, or hierarchy operation failed.
    #[error(transparent)]
    World(#[from] WorldError),
    /// A source transform could not be represented as a validated engine transform.
    #[error("glTF node {node} has an invalid transform: {source}")]
    InvalidTransform {
        node: usize,
        #[source]
        source: TransformError,
    },
    /// Cleanup failed after an earlier instantiation error.
    #[error("instantiation failed ({original}) and rollback also failed ({rollback})")]
    RollbackFailed {
        original: Box<GltfSceneInstantiationError>,
        rollback: WorldError,
    },
}

/// Instantiates the document's declared default scene.
pub fn instantiate_default_gltf_scene(
    world: &mut World,
    import: &GltfImport,
    source_uri: AssetUri,
) -> Result<GltfSceneInstance, GltfSceneInstantiationError> {
    let scene_index = import
        .default_scene
        .ok_or(GltfSceneInstantiationError::MissingDefaultScene)?;
    instantiate_gltf_scene(world, import, source_uri, scene_index)
}

/// Instantiates one glTF scene as ECS entities.
///
/// Every selected source node becomes one entity with a local transform. Parent
/// relationships are reproduced in the ECS hierarchy. Nodes with a mesh receive a
/// mesh-instance component that refers to the source URI and imported mesh index.
/// Node names are retained when present.
///
/// The operation validates the selected hierarchy before creating entities. If an ECS
/// operation fails after creation begins, all entities created by this call are despawned.
/// Existing world entities are not modified.
pub fn instantiate_gltf_scene(
    world: &mut World,
    import: &GltfImport,
    source_uri: AssetUri,
    scene_index: usize,
) -> Result<GltfSceneInstance, GltfSceneInstantiationError> {
    let scene = import
        .scenes
        .get(scene_index)
        .ok_or(GltfSceneInstantiationError::MissingScene(scene_index))?;
    let ordered_nodes = validate_and_order_nodes(import, &scene.root_nodes)?;

    world.register_component::<Transform>()?;
    world.register_component::<GltfMeshInstance>()?;
    world.register_component::<GltfNodeName>()?;

    let mut created = Vec::with_capacity(ordered_nodes.len());
    let mut entities_by_node = vec![None; import.nodes.len()];
    let mut result_nodes = Vec::with_capacity(ordered_nodes.len());
    let mut roots = Vec::with_capacity(scene.root_nodes.len());

    let creation = (|| {
        for (node_index, parent_index) in &ordered_nodes {
            let node = &import.nodes[*node_index];
            let entity = world.spawn()?;
            created.push(entity);

            let transform = Transform::new(
                Vec3::from_array(node.translation),
                Quat::from_xyzw(
                    node.rotation_xyzw[0],
                    node.rotation_xyzw[1],
                    node.rotation_xyzw[2],
                    node.rotation_xyzw[3],
                ),
                Vec3::from_array(node.scale),
            )
            .map_err(|source| GltfSceneInstantiationError::InvalidTransform {
                node: *node_index,
                source,
            })?;
            world.insert(entity, transform)?;

            if let Some(mesh_index) = node.mesh_index {
                world.insert(
                    entity,
                    GltfMeshInstance {
                        source_uri: source_uri.clone(),
                        mesh_index,
                        source_node_index: *node_index,
                    },
                )?;
            }
            if let Some(name) = &node.name {
                world.insert(
                    entity,
                    GltfNodeName {
                        value: name.clone(),
                    },
                )?;
            }
            if let Some(parent_index) = parent_index {
                let parent = entities_by_node[*parent_index].ok_or_else(|| {
                    GltfSceneInstantiationError::InvalidNode {
                        node: *node_index,
                        reason: "parent was not instantiated before its child".to_owned(),
                    }
                })?;
                world.set_parent(entity, Some(parent))?;
            } else {
                roots.push(entity);
            }

            entities_by_node[*node_index] = Some(entity);
            result_nodes.push(ImportedNodeEntity {
                source_node_index: node.source_index,
                entity,
            });
        }
        Ok::<(), GltfSceneInstantiationError>(())
    })();

    if let Err(original) = creation {
        for entity in created.into_iter().rev() {
            if let Err(rollback) = world.despawn(entity) {
                return Err(GltfSceneInstantiationError::RollbackFailed {
                    original: Box::new(original),
                    rollback,
                });
            }
        }
        return Err(original);
    }

    Ok(GltfSceneInstance {
        scene_index,
        roots,
        nodes: result_nodes,
    })
}

fn validate_and_order_nodes(
    import: &GltfImport,
    roots: &[usize],
) -> Result<Vec<(usize, Option<usize>)>, GltfSceneInstantiationError> {
    let mut visited = vec![false; import.nodes.len()];
    let mut ordered = Vec::new();
    let mut stack = Vec::new();

    for root in roots.iter().rev() {
        if *root >= import.nodes.len() {
            return Err(GltfSceneInstantiationError::MissingReference {
                kind: "root node",
                index: *root,
            });
        }
        stack.push((*root, None));
    }

    while let Some((index, parent)) = stack.pop() {
        if visited[index] {
            return Err(GltfSceneInstantiationError::InvalidNode {
                node: index,
                reason: "node is reachable more than once (cycle or multiple parents)".to_owned(),
            });
        }
        visited[index] = true;
        let node = &import.nodes[index];
        if node.source_index != index {
            return Err(GltfSceneInstantiationError::InvalidNode {
                node: index,
                reason: format!(
                    "source index {} does not match its position in the node table",
                    node.source_index
                ),
            });
        }
        if node
            .mesh_index
            .is_some_and(|mesh| mesh >= import.meshes.len())
        {
            return Err(GltfSceneInstantiationError::MissingReference {
                kind: "mesh",
                index: node.mesh_index.unwrap_or(usize::MAX),
            });
        }
        Transform::new(
            Vec3::from_array(node.translation),
            Quat::from_xyzw(
                node.rotation_xyzw[0],
                node.rotation_xyzw[1],
                node.rotation_xyzw[2],
                node.rotation_xyzw[3],
            ),
            Vec3::from_array(node.scale),
        )
        .map_err(|source| GltfSceneInstantiationError::InvalidTransform {
            node: index,
            source,
        })?;

        ordered.push((index, parent));
        for child in node.children.iter().rev() {
            if *child >= import.nodes.len() {
                return Err(GltfSceneInstantiationError::MissingReference {
                    kind: "child node",
                    index: *child,
                });
            }
            stack.push((*child, Some(index)));
        }
    }

    Ok(ordered)
}
