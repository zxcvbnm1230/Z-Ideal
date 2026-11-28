# Phase 3 — Asset pipeline and first renderer

**Status:** implementation substantially complete; final cross-platform CI and exit-gate review are pending. This document records tested behavior and explicit limitations; it is not a phase-completion claim.

## Roadmap scope

- glTF 2.0 / GLB import, textures, and material conversion.
- Asset handles, cache, dependency graph, and metadata.
- Mesh rendering, camera, depth, HDR target, and tone mapping.

**Exit gate:** deterministic sample-scene rendering and robust malformed-asset handling.

## Implemented foundation

### Asset contracts

- Stable 128-bit IDs derive from explicit stable asset-kind strings and validated project-relative URIs. The catalog detects collisions rather than assuming the hash is collision-free.
- Versioned asset metadata records source URI, kind, content fingerprint, importer identity, and direct dependencies. Unknown serialized fields and unsupported metadata schema versions are rejected.
- The dependency catalog supports forward references during discovery, deterministic enumeration, transactional cycle rejection, and an explicit complete-graph validation operation.
- Typed caches own strong references. Evicting an entry does not invalidate outstanding handles; replacing an entry requires explicit removal.

### glTF import

- Pinned gltf 1.4.1 parses glTF 2.0 JSON and GLB, external and data-URI buffers/images, and decodes source images.
- Mesh primitives preserve topology, positions, optional normals/tangents and UV0/UV1, indices, and explicit material references. TEXCOORD_2+ vertex attributes are rejected rather than silently discarded. Nodes preserve source indices, names, validated local TRS transforms, mesh references, and ordered child indices; scenes preserve named root-node lists and the declared default scene. The importer rejects out-of-range child/mesh/root references, multiple parents, cyclic hierarchies, and scene roots that are also children. Missing indices become sequential indices. Position/normal/tangent/UV finiteness, attribute counts, index bounds, and triangle index counts are checked.
- Metallic-roughness materials preserve base color, metallic and roughness factors, alpha mode/cutoff, double-sided state, all five standard texture slots, per-slot UV-set indices, normal scale, occlusion strength, and emissive factor.
- Texture source-image references, filters, and wrapping modes are converted to engine-facing types. Embedded data URIs are not reported as external catalog dependencies. External paths are percent-decoded, constrained to the source directory, and canonicalized before resource loading; absolute paths, URI schemes, traversal, missing files, and symlink escapes are rejected.
- Parser-specific glTF types do not leak through the public asset import structures.

### First-renderer building blocks

- PerspectiveCamera validates field of view, aspect ratio, and near/far planes and builds a right-handed zero-to-one-depth projection. Its aspect ratio can be updated after drawable-size changes.
- PerspectiveCamera is registered as a validated scene component with stable ID z_ideal.camera.perspective.
- The runtime crate extracts an owned, renderer-neutral render snapshot from a caller-selected perspective camera. It includes view/projection matrices and deterministic mesh-instance records with composed world transforms and logical asset references. Camera selection is explicit; singular/non-finite camera transforms and non-finite mesh transforms are rejected.
- The runtime crate instantiates selected glTF scenes transactionally into ECS entities. It preserves local transforms, parent/child relationships, optional node names, source node indices, and mesh-instance references to a project-relative source URI and imported mesh index. It supports explicit scene selection and the declared default scene. Malformed reachable hierarchy data is rejected before entities are created; ECS failures trigger rollback of entities created by the operation. Runtime components participate in versioned scene serialization.
- MeshGpu validates triangle-list geometry and uploads interleaved position, UV0, normal, tangent, and UV1 vertices plus 32-bit indices into explicitly owned Vulkan upload buffers. Missing normals and tangents are generated for triangle lists; mirrored-UV tangent handedness is preserved. Attribute lengths, finite values, tangent handedness, and index ranges are checked before vertex-basis generation. The vertex layout is exposed as matching Vulkan binding/attribute descriptions. MeshGpu::record_draw uses a validated 64-byte MVP push constant; record_draw_pbr uses a 128-byte MVP-plus-model block for world-space normal transforms.
- VulkanSceneMeshes is a runtime-side Vulkan adapter. It resolves each distinct logical mesh reference against AssetCache<GltfImport>, uploads all primitives through MeshGpu, pins strong CPU asset handles, and reports missing/ambiguous/empty mesh references or upload errors explicitly. Construction is transactional. `record_upload_barriers` records HOST_WRITE -> VERTEX_INPUT synchronization before the consuming render pass; `record_draws` records one indexed draw per primitive instance inside the pass; `record_draws_with_materials` invokes a caller-provided binder with each primitive's pinned optional `ImportedMaterial` immediately before its draw. The caller binds a compatible pipeline and owns render-pass, framebuffer, viewport/scissor, queue submission, and frame synchronization.
- A checked-in glTF sample fixture now has a hardware-validated end-to-end regression: import the file, instantiate its default scene into ECS, extract the camera/mesh snapshot, resolve and upload the asset through VulkanSceneMeshes, render offscreen, copy RGBA8 pixels to host memory, and assert exact center/clear pixel values.
- VulkanSceneRenderer builds four fixed-function pipeline variants for opaque/masked versus blended and single- versus double-sided materials. It chooses a variant per primitive, binds the scene descriptor set, restores viewport/scissor after pipeline changes, and delegates per-material descriptor binding. The caller explicitly owns pipeline layouts, shader modules, render pass/framebuffer, target/depth resources, descriptor allocation, and frame synchronization.
- VulkanTexture2D converts imported image channel formats to explicitly selected linear or sRGB Vulkan formats, validates physical-device sampling/transfer/blit features and dimensions, uploads decoded pixels through a host-visible staging buffer, generates a complete mip chain with linear blits, creates a full-chain image view and sampler from glTF filter/wrap state, and records all image transitions/copies to SHADER_READ_ONLY_OPTIMAL. Unsupported color-space/format or mip-generation combinations return structured errors. The staging buffer remains owned until the upload object is dropped; upload recording is one-shot and must complete before sampling.
- VulkanHdrColorTarget owns a validated R16G16B16A16_SFLOAT color image suitable for color-attachment rendering and subsequent sampling. The deterministic imported-scene GPU regression renders the material into this HDR target, samples it in a second fullscreen pass, applies a Reinhard tone-map operator, and reads the RGBA8 output back for exact pixel assertions.

## Known limitations and next work

- GltfImporter::import_with_options enforces bounded source-document reads (64 MiB default), per-external-file and aggregate external-resource budgets (256 MiB and 512 MiB defaults), strict decoded image dimensions (16,384 by 16,384), decoded output bytes per image (256 MiB), and aggregate decoded image bytes (512 MiB). Decoded output size is checked before allocating the pixel buffer and the image decoder receives matching limits; some decoder-internal temporary allocations remain best-effort. External buffer and image reads are bounded to the configured per-file limit, and actual bytes across distinct canonical paths are checked against the aggregate limit. Canonical-path validation still retains a symlink/file-replacement time-of-check/time-of-use race; treat imports as trusted project content until that is resolved.
- Embedded image data URIs currently require base64 encoding. Image channel formats and decoded bytes are preserved. VulkanTexture2D requires the caller to select linear or sRGB interpretation; sRGB is supported for 8-bit channel formats, while 16-bit and floating-point formats remain linear-only. It generates a complete mip chain using linear blits and rejects devices/formats lacking the required features. The PBR regression exercises all five semantic material slots and neutral fallbacks. VulkanHdrColorTarget supplies the floating-point target, while render-pass construction and the tone-map pipeline remain caller-owned.
- ECS instantiation preserves node transforms, hierarchy, names, and mesh references, but does not yet instantiate skinning, morph targets, or animation channels. KHR_texture_transform, TEXCOORD_2+, and extension-defined material behavior are unsupported.
- Asset catalog registration is currently an explicit API; the importer does not yet compute a complete transitive content fingerprint or automatically populate metadata and dependency records.
- VulkanSceneMeshes remains the geometry/asset adapter and pins CPU asset handles. GltfMaterialTextureSet owns five semantic GPU textures and explicit neutral fallbacks; write_gltf_material_descriptors and write_gltf_pbr_scene_descriptor populate validated descriptor sets and return fence-retained resource leases. The checked-in PBR shader pair implements a Cook-Torrance metallic-roughness direct-light model, normal/occlusion/emissive sampling, alpha-mask discard, and double-sided normal orientation. VulkanSceneRenderer owns opaque/masked/blended and single-/double-sided pipeline variants and orchestrates per-primitive material binding/draws. It borrows caller-owned shader modules, pipeline layout, and render pass; target/framebuffer/depth allocation, descriptor allocation, and frame synchronization remain caller-owned. BLEND sorting uses a stable camera-space object-origin approximation; intersecting or large transparent meshes may need bounds-aware application sorting.
- Malformed-resource regression coverage now includes truncated external buffers, accessors extending beyond buffer views, malformed embedded images, traversal and symlink escapes, invalid indices, cyclic/multi-parent nodes, and source/per-file/aggregate/decoded-image budgets. Unsupported image encodings/extensions and race-resistant opening of external resources remain outstanding.
- The PBR model currently supports one directional light and a constant ambient approximation. Image-based lighting/environment maps, point/spot or clustered lighting, shadows, SSAO, and non-core material extensions remain Phase 4 or later work. Transparent ordering is object-origin based, not bounds aware.
- The Vulkan CI workflow runs lavapipe offscreen readback, runtime scene-mesh upload, and X11 swapchain/depth validation under Xvfb. Hardware-specific driver coverage remains separate.

## Current validation

Local validation for the current Phase 3 implementation:

- `cargo fmt --all -- --check` — passed.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — passed.
- `cargo test --workspace --all-features` — 243 passed, 5 GPU integration tests ignored by default, 0 failed.
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps` — passed.
- `git diff --check` — passed.

All five ignored GPU integration tests were run manually on the connected NVIDIA GeForce 940MX host with Khronos validation enabled: the three runtime scene tests (including five-slot PBR material upload, four-variant VulkanSceneRenderer pipeline selection, actual PBR draw and lit/clear pixel readback), offscreen renderer readback, and X11 swapchain/depth presentation. All passed with no Vulkan validation errors. The imported glTF -> ECS -> snapshot -> mesh upload -> sRGB texture/mip upload -> HDR render -> Reinhard tone-map -> sRGB RGBA8 pixel-readback regression also passed.

GitHub CI for the current changes has not yet been run; Windows, macOS, Ubuntu, and the lavapipe/X11 Vulkan workflow remain the final cross-platform gate.
