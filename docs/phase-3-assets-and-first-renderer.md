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
- VulkanSceneMeshes is a runtime-side Vulkan adapter. It resolves each distinct logical mesh reference against AssetCache<GltfImport>, uploads all primitives through MeshGpu, pins strong CPU asset handles, and reports missing/ambiguous/empty mesh references or upload errors explicitly. Construction is transactional. `record_upload_barriers` records HOST_WRITE -> VERTEX_INPUT synchronization before the consuming render pass; `record_draws` records one indexed draw per primitive instance inside the pass. The caller binds a compatible pipeline and owns render-pass, framebuffer, viewport/scissor, queue submission, and frame synchronization.

## Known limitations and next work

- GltfImporter::import_with_options now enforces pre-allocation source-document, per-external-file, and aggregate external-file byte budgets (defaults: 64 MiB, 256 MiB, and 512 MiB). Embedded data URIs and GLB chunks are bounded by the source-document limit. Decoded-image expansion is not yet bounded, and canonical-path validation still has a symlink time-of-check/time-of-use race; treat imports as trusted project content until those are resolved.
- Image channel formats and decoded bytes are preserved; color-space interpretation, GPU texture upload, mip generation, and sampler creation from imported texture state are not yet integrated.
- ECS instantiation preserves node transforms, hierarchy, names, and mesh references, but does not yet instantiate skinning, morph targets, animation channels, tangents, or extension-defined material behavior.
- Asset catalog registration is currently an explicit API; the importer does not yet compute a complete transitive content fingerprint or automatically populate metadata and dependency records.
- VulkanSceneMeshes uploads geometry only. Imported material/texture binding, shader material contracts, HDR color targets, tone mapping, and a complete runtime-owned scene renderer remain outstanding.
- Malformed-resource testing must be expanded to truncated buffers, invalid accessors, malformed images, unsupported encodings/extensions, decoded-image allocation limits, and symlink race conditions. Source, per-file external, and aggregate external byte budgets now have regression coverage.
- The Vulkan CI workflow runs lavapipe offscreen readback, runtime scene-mesh upload, and X11 swapchain/depth validation under Xvfb. Hardware-specific driver coverage remains separate.

## Current validation

Local validation after the runtime Vulkan adapter and glTF byte-budget changes:

- `cargo fmt --all -- --check` — passed.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — passed.
- `cargo test --workspace --all-features` — 210 passed, 3 GPU integration tests ignored by default, 0 failed.
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps` — passed.
- `git diff --check` — passed.

The runtime Vulkan integration test was run manually on the connected NVIDIA Vulkan host with Khronos validation enabled. It verifies explicit missing-asset, missing-mesh, and unsupported-topology errors; shared-mesh upload deduplication; retained asset-handle lifetime after cache eviction; upload barriers recorded before the render pass; and indexed draw recording for two instances. It completed with no Vulkan validation errors.

GitHub CI on commit `3f47c6c` passed the Windows, macOS, and Ubuntu quality jobs, and the Vulkan workflow passed lavapipe offscreen readback, runtime scene-mesh upload, and X11 swapchain/depth validation. The current glTF import-budget changes still require a new CI run after they are committed and pushed.
