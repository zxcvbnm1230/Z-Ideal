# z-ideal-runtime

Editor-independent runtime integration for loading engine assets into a game world.

## glTF scene instantiation

`instantiate_gltf_scene` and `instantiate_default_gltf_scene` convert a CPU-side `GltfImport` into ECS entities. Each selected source node receives a validated local `Transform`; node parentage is reproduced in the world hierarchy. Optional names are stored in `GltfNodeName`. Nodes with a mesh receive `GltfMeshInstance`, which records the project-relative source URI, source mesh-table index, and source node index.

The caller owns the imported asset and the `World`. The ECS component stores a logical URI and indices, not a pointer or borrowed reference, so asset lifetime remains explicit and the renderer can resolve the mesh data through the asset system. A `GltfSceneInstance` provides deterministic source-node-to-entity mappings.

The operation validates the selected reachable hierarchy before creating entities. If an ECS operation fails during creation, entities created by that operation are despawned in reverse creation order. Component registration remains in the world, as component registries are world-level schema state.

## Renderer-neutral scene extraction

`extract_render_scene(&World, camera_entity)` creates an owned `RenderSceneSnapshot` for a caller-selected perspective camera. It contains the camera view and Vulkan-compatible projection matrices plus mesh-instance records in deterministic entity order. Each mesh record contains its entity, source asset URI/index reference, and composed object-to-world matrix. Camera and mesh transforms include ancestor transforms.

Camera selection is explicit when multiple cameras exist. Extraction rejects missing cameras, non-invertible/non-finite camera transforms, invalid projections, and non-finite mesh world transforms. The snapshot owns its mesh references and matrices and does not borrow the world. It deliberately contains no Vulkan handles or renderer backend types.

## Vulkan scene mesh adapter

`VulkanSceneMeshes::upload` resolves the distinct `(AssetUri, mesh_index)` references in a snapshot against an `AssetCache<GltfImport>`, using the stable `AssetKind::Mesh` identity. It uploads every primitive in each referenced imported mesh through `MeshGpu`. Missing assets, missing or ambiguous mesh indices, empty meshes, and unsupported/malformed primitives are returned as explicit errors; partial GPU allocations are dropped if construction fails.

The adapter retains strong `AssetHandle<GltfImport>` values for the imported assets and owns the resulting `MeshGpu` resources. It borrows the Vulkan device, so it must be dropped before that device. Call `record_upload_barriers` while recording commands outside the consuming render pass; Vulkan does not permit this ordinary barrier inside a render pass. Then bind a compatible graphics pipeline inside the render pass and call `record_draws` to draw every primitive for each snapshot instance using the snapshot view-projection matrix and each instance's model matrix. `record_draws_with_materials` adds a per-primitive callback with the pinned optional `ImportedMaterial`, allowing callers to bind the appropriate material descriptors immediately before each draw.

The caller must bind a graphics pipeline compatible with `MeshGpu::vertex_binding`, `MeshGpu::vertex_attributes`, and its 64-byte vertex push-constant range before recording. `imported_asset` returns the pinned CPU asset, and `material_for_primitive` resolves the optional material attached to a source primitive. The material_texture_bindings_for_primitive method resolves all populated standard metallic-roughness slots into semantic bindings. It marks base-color/emissive images as sRGB and metallic-roughness/normal/occlusion images as linear, and returns each slot's texture index and UV set. A source texture used in both sRGB and linear roles requires separate GPU image resources. The current mesh format carries UV0 only; nonzero UV sets are explicitly rejected by this resolver. The hardware regression exercises base-color binding and factor sampling and checks the semantic mapping for all five standard slots. Reusable GPU material descriptor ownership/binding, full PBR shading, alpha-mask shader discard, alpha-blend sorting/blending, and double-sided pipeline variants remain outstanding. The caller still owns render-pass/framebuffer setup, viewport/scissor, pipeline selection, queue submission, and frame synchronization. VulkanHdrColorTarget supplies a validated floating-point attachment/sampled image; the deterministic GPU regression exercises a caller-owned fullscreen Reinhard tone-mapping pass.

## Vulkan regression coverage

The ignored vulkan_scene integration tests exercise mesh-reference resolution, upload lifetime and error paths, and an end-to-end imported sample scene. The sample test imports tests/fixtures/sample_scene.gltf, instantiates its default scene into an ECS world, extracts a camera snapshot, resolves its imported base-color material and texture, uploads the referenced mesh and sRGB texture with generated mip levels, writes the combined image sampler and base-color factor descriptors, renders to a floating-point HDR target, tone maps through a fullscreen Reinhard pass into an sRGB RGBA8 target, reads the pixels back, and checks known center and clear pixels. A separate 7x3 texture upload exercises non-power-of-two mip generation.

Run these tests on a Vulkan host with the Khronos validation layer:

cargo test -p z-ideal-runtime --test vulkan_scene --all-features -- --ignored --nocapture
