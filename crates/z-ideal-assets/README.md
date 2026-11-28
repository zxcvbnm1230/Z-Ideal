# z-ideal-assets

Renderer-independent asset identity, metadata, dependency tracking, cache ownership, and source import.

## Current contracts

- AssetUri accepts normalized project-relative logical paths and rejects absolute paths, empty components, dot segments, backslashes, NUL, and URI-reserved path characters.
- AssetId is a deterministic 128-bit identifier derived from a stable asset-kind tag and URI. AssetCatalog detects collisions; the hash is not treated as collision-proof.
- AssetMetadata is versioned, rejects unknown serialized fields, records content fingerprints and direct dependencies, and validates identity/importer metadata.
- AssetCatalog supports forward dependency discovery, transactional cycle rejection, deterministic iteration, and a complete-graph validation step.
- AssetCache<T> owns strong references. Handles remain valid after cache eviction; duplicate insertion is an explicit error, not an implicit reload.
- GltfImporter reads glTF 2.0 JSON and GLB documents, external/data-URI buffers and images, indexed/non-indexed mesh primitives, optional normals and UV0, validated node hierarchies with local TRS transforms and mesh references, scene roots/default scene, texture samplers, and metallic-roughness material factors and texture slots. Parser-specific types are converted into engine-facing enums and structures.
- GltfImporter::import_with_options enforces source-document, per-external-file, and aggregate external-resource byte limits (64 MiB, 256 MiB, and 512 MiB by default), plus strict image dimensions (16,384 by 16,384), decoded bytes per image (256 MiB), and aggregate decoded image bytes (512 MiB).

## Import boundaries

The importer percent-decodes external URIs, rejects absolute paths, schemes, traversal, queries/fragments, missing resources, and canonical paths outside the source directory before loading buffers/images. Source document reads are bounded while reading. External-file budgets are preflighted from canonical file metadata and enforced again against bytes actually read for both buffers and images. Reads are bounded to the configured per-file limit and aggregate actual bytes are tracked across distinct canonical paths. Canonical path validation still has a symlink/file-replacement time-of-check/time-of-use race. Embedded data URIs and GLB binary chunks are bounded by the source-document limit.

Image dimensions and the exact decoded output byte count are checked before allocating the output pixel buffer; the image decoder also receives matching width, height, and allocation limits. Some decoder-internal temporary allocations use the image library's best-effort max_alloc limit rather than a strict guarantee. Only base64 data URIs are supported for embedded images. Treat imports as trusted project content until external-resource loading is race-resistant. Decoded channel formats and bytes are preserved; color-space interpretation and GPU mip generation are not performed here.

The runtime crate instantiates imported scene nodes into ECS entities. GPU mesh upload lives in z-ideal-renderer and currently accepts indexed triangle lists with positions and optional UV0. VulkanTexture2D uploads imported images and sampler state with explicit linear/sRGB selection and GPU-generated mip chains. The runtime Vulkan sample currently exercises base-color texture and factor descriptor binding; other metallic-roughness/normal/occlusion/emissive material slots and full PBR shading remain outstanding, along with normal/tangent generation, skins, morph targets, and animation playback.

## Tests

cargo test -p z-ideal-assets --all-features covers URI validation, stable identities, cache eviction/handle lifetime, metadata round trips, missing dependencies, dependency cycles, glTF and GLB import, decoded image/material texture references, external dependency reporting, path-traversal rejection, source/per-file/aggregate external byte budgets, strict image dimensions, per-image and aggregate decoded-image byte budgets, truncated external buffers, accessors exceeding buffer views, traversal/symlink escapes, malformed images, invalid indices, malformed primitives, and topology-specific index counts.
