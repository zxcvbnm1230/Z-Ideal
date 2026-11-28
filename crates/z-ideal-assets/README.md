# z-ideal-assets

Renderer-independent asset identity, metadata, dependency tracking, cache ownership, and source import.

## Current contracts

- AssetUri accepts normalized project-relative logical paths and rejects absolute paths, empty components, dot segments, backslashes, NUL, and URI-reserved path characters.
- AssetId is a deterministic 128-bit identifier derived from a stable asset-kind tag and URI. AssetCatalog detects collisions; the hash is not treated as collision-proof.
- AssetMetadata is versioned, rejects unknown serialized fields, records content fingerprints and direct dependencies, and validates identity/importer metadata.
- AssetCatalog supports forward dependency discovery, transactional cycle rejection, deterministic iteration, and a complete-graph validation step.
- AssetCache<T> owns strong references. Handles remain valid after cache eviction; duplicate insertion is an explicit error, not an implicit reload.
- GltfImporter reads glTF 2.0 JSON and GLB documents, external/data-URI buffers and images, indexed/non-indexed mesh primitives, optional normals/tangents and UV0/UV1, validated node hierarchies with local TRS transforms and mesh references, scene roots/default scene, texture samplers, and metallic-roughness material factors, all five standard texture slots, per-slot UV-set indices, normal scale, occlusion strength, alpha mode/cutoff, and double-sided state. Core factors are checked for finite values and glTF ranges before import succeeds. Parser-specific types are converted into engine-facing enums and structures.
- GltfImporter::import_with_options enforces source-document, per-external-file, and aggregate external-resource byte limits (64 MiB, 256 MiB, and 512 MiB by default), plus strict image dimensions (16,384 by 16,384), decoded bytes per image (256 MiB), and aggregate decoded image bytes (512 MiB).
- HdrEnvironmentImporter decodes Radiance RGBE `.hdr` equirectangular maps to scene-linear RGB32F, preserves radiance above one, requires a 2:1 projection, rejects invalid samples, and enforces encoded-file, dimension, and decoded-storage budgets. Defaults are 256 MiB encoded, 16,384x8,192 pixels, and 512 MiB decoded.

## Import boundaries

The importer percent-decodes external URIs, rejects absolute paths, schemes, traversal, queries/fragments, missing resources, and canonical paths outside the source directory before loading buffers/images. Source document reads are bounded while reading. External-file budgets are preflighted from canonical file metadata and enforced again against bytes actually read for both buffers and images. Reads are bounded to the configured per-file limit and aggregate actual bytes are tracked across distinct canonical paths. Canonical path validation still has a symlink/file-replacement time-of-check/time-of-use race. Embedded data URIs and GLB binary chunks are bounded by the source-document limit.

Image dimensions and the exact decoded output byte count are checked before allocating the output pixel buffer; the image decoder also receives matching width, height, and allocation limits. Some decoder-internal temporary allocations use the image library's best-effort max_alloc limit rather than a strict guarantee. Only base64 data URIs are supported for embedded images. Treat imports as trusted project content until external-resource loading is race-resistant. Decoded channel formats and bytes are preserved; color-space interpretation and GPU mip generation are not performed here.

The HDRI importer currently supports Radiance RGBE `.hdr` equirectangular files only; EXR, cubemap source layouts, orientation metadata conversion, and GPU environment preprocessing are not implemented yet. Texture-transform extensions are not interpreted. Per-slot UV-set indices are preserved for UV0 and UV1; references to higher TEXCOORD sets are rejected rather than silently ignored. GPU mesh upload lives in z-ideal-renderer and currently accepts indexed triangle lists with positions, optional normals/tangents, and UV0/UV1. Missing normals and tangents are generated for triangle lists, including tangent handedness for mirrored UV orientation. The runtime crate instantiates imported scene nodes into ECS entities and provides the five-slot metallic-roughness PBR material path. Skins, morph targets, animation playback, and texture-transform extensions remain unsupported.

## Tests

cargo test -p z-ideal-assets --all-features covers bounded HDRI decoding, linear-radiance preservation, projection validation, and encoded/decoded resource budgets, URI validation, stable identities, cache eviction/handle lifetime, metadata round trips, missing dependencies, dependency cycles, glTF and GLB import, decoded image/material texture references, external dependency reporting, path-traversal rejection, source/per-file/aggregate external byte budgets, strict image dimensions, per-image and aggregate decoded-image byte budgets, truncated external buffers, accessors exceeding buffer views, traversal/symlink escapes, malformed images, invalid indices, malformed primitives, and topology-specific index counts.
