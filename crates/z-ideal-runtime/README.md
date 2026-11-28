# z-ideal-runtime

Editor-independent runtime integration for loading engine assets into a game world.

## glTF scene instantiation

instantiate_gltf_scene and instantiate_default_gltf_scene convert a CPU-side GltfImport into ECS entities. Each selected source node receives a validated local Transform; node parentage is reproduced in the world hierarchy. Optional names are stored in GltfNodeName. Nodes with a mesh receive GltfMeshInstance, which records the project-relative source URI, source mesh-table index, and source node index.

The caller owns the imported asset and the World. The ECS component stores a logical URI and indices, not a pointer or borrowed reference, so asset lifetime remains explicit and the renderer can resolve the mesh data through the asset system. A GltfSceneInstance provides deterministic source-node-to-entity mappings.

The operation validates the selected reachable hierarchy before creating entities. If an ECS operation fails during creation, entities created by that operation are despawned in reverse creation order. Component registration remains in the world, as component registries are world-level schema state.

This is not yet a renderer scene-extraction path: mesh instances are references only. GPU mesh upload, materials, textures, skins, morph targets, and animations are separate remaining work.

## Renderer-neutral scene extraction

`extract_render_scene(&World, camera_entity)` creates an owned `RenderSceneSnapshot` for a caller-selected perspective camera. It contains the camera view and Vulkan-compatible projection matrices plus mesh-instance records in deterministic entity order. Each mesh record contains its entity, source asset URI/index reference, and composed object-to-world matrix. Camera and mesh transforms include ancestor transforms.

Camera selection is explicit when multiple cameras exist. Extraction rejects missing cameras, non-invertible/non-finite camera transforms, invalid projections, and non-finite mesh world transforms. The snapshot owns its mesh references and matrices and does not borrow the world. It deliberately contains no Vulkan handles or renderer backend types; asset resolution and GPU upload remain separate steps.
