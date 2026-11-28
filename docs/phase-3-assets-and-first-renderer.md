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
- MeshGpu validates triangle-list geometry and uploads interleaved position/UV vertices plus 32-bit indices into explicitly owned Vulkan upload buffers. Its vertex layout is exposed as matching Vulkan binding/attribute descriptions. `MeshGpu::record_draw` now composes view-projection and model matrices, writes a validated 64-byte column-major vertex push constant, binds the mesh buffers, and records an indexed draw.

## Known limitations and next work

- Import currently delegates file loading and image decoding to gltf/image without configurable pre-allocation byte or pixel budgets. Treat project imports as trusted input until bounded external-resource loading is implemented.
- Image channel formats and decoded bytes are preserved; color-space interpretation, GPU texture upload, mip generation, and sampler creation from imported texture state are not yet integrated.
- ECS instantiation preserves node transforms, hierarchy, names, and mesh references, but does not yet instantiate skinning, morph targets, animation channels, tangents, or extension-defined material behavior.
- Asset catalog registration is currently an explicit API; the importer does not yet compute a complete transitive content fingerprint or automatically populate metadata and dependency records.
- MeshGpu supports indexed triangle-list upload and camera/model-matrix draw recording. It uses host-visible upload memory and leaves host/GPU barriers, render-pass setup, pipeline binding, and viewport/scissor state to the caller. There is not yet an end-to-end runtime snapshot-to-renderer adapter, automatic asset resolution, material/texture binding, or production scene renderer.
- The renderer now has a camera/model matrix draw helper and an offscreen GPU test that draws overlapping meshes with depth testing; the nearer mesh wins regardless of draw order. A runtime snapshot-to-renderer adapter, asset resolution, imported material/texture binding, HDR color targets, tone mapping, and deterministic end-to-end sample-scene image regression remain outstanding.
- Malformed-resource testing must be expanded to truncated buffers, invalid accessors, malformed images, unsupported encodings/extensions, symlink race conditions, and configured resource budgets.

## Current validation

Full workspace checks passed on the connected Linux host:

- cargo fmt --all -- --check — passed.
- cargo clippy --workspace --all-targets --all-features -- -D warnings — passed.
- cargo test --workspace --all-features — 207 passed, 2 GPU integration tests ignored by default, 0 failed.
- RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps — passed.

The asset tests cover identity, metadata, cache lifetime, dependency graph behavior, glTF/GLB mesh and node import, embedded image decoding, material texture references, external dependency reporting, path-traversal rejection, malformed primitives, invalid indices, and topology-specific index counts. Runtime tests cover default-scene selection, entity/node mapping, transforms, hierarchy, mesh-instance references, transactional validation failures, scene serialization round trips, camera extraction, composed view/model matrices, deterministic mesh ordering, and singular-camera rejection. The renderer has CPU-side mesh-layout and matrix-encoding tests; its manually executed offscreen Vulkan test exercises the GPU draw/depth path.

The ignored offscreen Vulkan integration test was run manually with VK_LAYER_KHRONOS_validation on an NVIDIA GeForce 940MX. It uploads two indexed MeshGpu triangles, pushes the camera/model matrix, uses a D32 depth attachment and depth-tested pipeline, verifies that the farther triangle cannot overwrite the nearer one, exercises the graphics-family-0 / transfer-family-1 ownership path, performs image readback and deterministic pixel checks, and passes without validation errors.

No complete glTF-to-renderer sample-scene path, HDR target, or tone mapping is claimed yet.
