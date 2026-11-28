# Phase 3 — Asset pipeline and first renderer

**Status:** implementation underway. This document records implemented behavior and the remaining exit work; it is not a phase-completion claim.

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
- Mesh primitives preserve topology, positions, optional normals and UV0, indices, and explicit material references. Nodes preserve source indices, names, validated local TRS transforms, mesh references, and ordered child indices; scenes preserve named root-node lists and the declared default scene. The importer rejects out-of-range child/mesh/root references, multiple parents, cyclic hierarchies, and scene roots that are also children. Missing indices become sequential indices. Position/normal/UV finiteness, attribute counts, index bounds, and triangle index counts are checked.
- Metallic-roughness materials preserve base color, metallic and roughness factors, alpha mode/cutoff, double-sided state, and base-color, metallic-roughness, normal, occlusion, and emissive texture indices.
- Texture source-image references, filters, and wrapping modes are converted to engine-facing types. Embedded data URIs are not reported as external catalog dependencies. External paths are percent-decoded, constrained to the source directory, and canonicalized before resource loading; absolute paths, URI schemes, traversal, missing files, and symlink escapes are rejected.
- Parser-specific glTF types do not leak through the public asset import structures.

### First-renderer building blocks

- PerspectiveCamera validates field of view, aspect ratio, and near/far planes and builds a right-handed zero-to-one-depth projection. Its aspect ratio can be updated after drawable-size changes.
- PerspectiveCamera is registered as a validated scene component with stable ID z_ideal.camera.perspective.
- The runtime crate extracts an owned, renderer-neutral render snapshot from a caller-selected perspective camera. It includes view/projection matrices and deterministic mesh-instance records with composed world transforms and logical asset references. Camera selection is explicit; singular/non-finite camera transforms and non-finite mesh transforms are rejected.
- The runtime crate instantiates selected glTF scenes transactionally into ECS entities. It preserves local transforms, parent/child relationships, optional node names, source node indices, and mesh-instance references to a project-relative source URI and imported mesh index. It supports explicit scene selection and the declared default scene. Malformed reachable hierarchy data is rejected before entities are created; ECS failures trigger rollback of entities created by the operation. Runtime components participate in versioned scene serialization.
- MeshGpu validates triangle-list geometry and uploads interleaved position/UV vertices plus 32-bit indices into explicitly owned Vulkan upload buffers. Its vertex layout is exposed as matching Vulkan binding/attribute descriptions. MeshGpu::record_draw composes view-projection and model matrices, writes a validated 64-byte column-major vertex push constant, binds the mesh buffers, and records an indexed draw.
- VulkanSceneMeshes is a runtime-side Vulkan adapter. It resolves each distinct logical mesh reference against AssetCache<GltfImport>, uploads all primitives through MeshGpu, pins strong CPU asset handles, and reports missing/ambiguous/empty mesh references or upload errors explicitly. Construction is transactional. `record_upload_barriers` records HOST_WRITE -> VERTEX_INPUT synchronization before the consuming render pass; `record_draws` records one indexed draw per primitive instance inside the pass; `record_draws_with_materials` invokes a caller-provided binder with each primitive's pinned optional `ImportedMaterial` immediately before its draw. The caller binds a compatible pipeline and owns render-pass, framebuffer, viewport/scissor, queue submission, and frame synchronization.
- A checked-in glTF sample fixture now has a hardware-validated end-to-end regression: import the file, instantiate its default scene into ECS, extract the camera/mesh snapshot, resolve and upload the asset through VulkanSceneMeshes, render offscreen, copy RGBA8 pixels to host memory, and assert exact center/clear pixel values.
- VulkanTexture2D converts imported image channel formats to explicitly selected linear or sRGB Vulkan formats, validates physical-device sampling/transfer/blit features and dimensions, uploads decoded pixels through a host-visible staging buffer, generates a complete mip chain with linear blits, creates a full-chain image view and sampler from glTF filter/wrap state, and records all image transitions/copies to SHADER_READ_ONLY_OPTIMAL. Unsupported color-space/format or mip-generation combinations return structured errors. The staging buffer remains owned until the upload object is dropped; upload recording is one-shot and must complete before sampling.
- VulkanHdrColorTarget owns a validated R16G16B16A16_SFLOAT color image suitable for color-attachment rendering and subsequent sampling. The deterministic imported-scene GPU regression renders the material into this HDR target, samples it in a second fullscreen pass, applies a Reinhard tone-map operator, and reads the RGBA8 output back for exact pixel assertions.

## Known limitations and next work

- GltfImporter::import_with_options enforces bounded source-document reads (64 MiB default), per-external-file and aggregate external-resource budgets (256 MiB and 512 MiB defaults), strict decoded image dimensions (16,384 by 16,384), decoded output bytes per image (256 MiB), and aggregate decoded image bytes (512 MiB). Decoded output size is checked before allocating the pixel buffer and the image decoder receives matching limits; some decoder-internal temporary allocations remain best-effort. External buffer and image reads are bounded to the configured per-file limit, and actual bytes across distinct canonical paths are checked against the aggregate limit. Canonical-path validation still retains a symlink/file-replacement time-of-check/time-of-use race; treat imports as trusted project content until that is resolved.
- Embedded image data URIs currently require base64 encoding. Image channel formats and decoded bytes are preserved. VulkanTexture2D requires the caller to select linear or sRGB interpretation; sRGB is supported for 8-bit channel formats, while 16-bit and floating-point formats remain linear-only. It generates a complete mip chain using linear blits and rejects devices/formats lacking the required features. The base-color descriptor/factor contract is exercised by the sample regression; other glTF material slots are not yet integrated. VulkanHdrColorTarget supplies the floating-point target, while render-pass construction and the tone-map pipeline remain caller-owned.
- ECS instantiation preserves node transforms, hierarchy, names, and mesh references, but does not yet instantiate skinning, morph targets, animation channels, tangents, or extension-defined material behavior.
- Asset catalog registration is currently an explicit API; the importer does not yet compute a complete transitive content fingerprint or automatically populate metadata and dependency records.
- VulkanSceneMeshes remains a geometry adapter, but it now exposes pinned imported assets and resolves each primitive's optional ImportedMaterial. The hardware sample regression uses that material to upload its base-color texture, write a combined-image-sampler descriptor and base-color-factor uniform buffer, and sample both in the fragment shader. This is a tested base-color material contract, not yet a reusable per-material scene adapter: metallic-roughness, normal, occlusion, and emissive slots, alpha modes, and double-sided pipeline selection remain outstanding. HDR target allocation and a deterministic Reinhard tone-map pass are implemented and hardware-tested, but a complete runtime-owned scene renderer remains outstanding.
- Malformed-resource regression coverage now includes truncated external buffers, accessors extending beyond buffer views, malformed embedded images, traversal and symlink escapes, invalid indices, cyclic/multi-parent nodes, and source/per-file/aggregate/decoded-image budgets. Unsupported image encodings/extensions and race-resistant opening of external resources remain outstanding.
- The Vulkan CI workflow runs lavapipe offscreen readback, runtime scene-mesh upload, and X11 swapchain/depth validation under Xvfb. Hardware-specific driver coverage remains separate.

## Current validation

Local validation after external/decoded-image budgets, malformed-asset hardening, base-color descriptor integration, mip generation, and HDR/tone-map changes:

- `cargo fmt --all -- --check` — passed.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — passed.
- `cargo test --workspace --all-features` — 228 passed, 4 GPU integration tests ignored by default, 0 failed.
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps` — passed.
- `git diff --check` — passed.

Both runtime Vulkan integration tests were run manually on the connected NVIDIA GeForce 940MX host with Khronos validation enabled. The imported glTF -> ECS -> snapshot -> mesh upload -> sRGB texture/mip upload -> base-color descriptor/factor binding -> HDR render -> Reinhard tone-map -> sRGB RGBA8 pixel-readback regression passed, as did the adapter/lifetime/error test. No Vulkan validation errors were reported.

GitHub CI on commit `545442a` passed the Windows, macOS, and Ubuntu quality jobs. The Vulkan workflow also passed lavapipe offscreen readback, runtime scene mesh/material binding and HDR tone-mapping validation, and X11 swapchain/depth validation.
