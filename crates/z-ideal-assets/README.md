# z-ideal-assets

Renderer-independent asset identity, metadata, dependency tracking, cache ownership, and source import.

## Current contracts

- AssetUri accepts normalized project-relative logical paths and rejects absolute paths, empty components, dot segments, backslashes, NUL, and URI-reserved path characters.
- AssetId is a deterministic 128-bit identifier derived from a stable asset-kind tag and URI. AssetCatalog detects collisions; the hash is not treated as collision-proof.
- AssetMetadata is versioned, rejects unknown serialized fields, records content fingerprints and direct dependencies, and validates identity/importer metadata.
- AssetCatalog supports forward dependency discovery, transactional cycle rejection, deterministic iteration, and a complete-graph validation step.
- AssetCache<T> owns strong references. Handles remain valid after cache eviction; duplicate insertion is an explicit error, not an implicit reload.
- GltfImporter reads glTF 2.0 JSON and GLB documents, external/data-URI buffers and images, indexed/non-indexed mesh primitives, optional normals and UV0, validated node hierarchies with local TRS transforms and mesh references, scene roots/default scene, texture samplers, and metallic-roughness material factors and texture slots. Parser-specific types are converted into engine-facing enums and structures.
- GltfImporter::import_with_options enforces a maximum source-document size, per-external-file size, and aggregate external-resource size. Defaults are 64 MiB, 256 MiB, and 512 MiB respectively.

## Import boundaries

The importer percent-decodes external URIs, rejects absolute paths, schemes, traversal, queries/fragments, missing resources, and canonical paths outside the source directory before loading buffers/images. Source document reads are bounded while reading. External-file budgets are preflighted from canonical file metadata before gltf loads the resources. The remaining metadata-to-open interval has a symlink/file-replacement time-of-check/time-of-use race. Embedded data URIs and GLB binary chunks are bounded by the source-document limit.

Decoded-image expansion is not yet bounded, so a small compressed image may still expand to a large decoded allocation. Treat imports as trusted project content until decoded-image limits and race-resistant external-resource loading are implemented. Decoded channel formats and bytes are preserved; color-space interpretation and GPU mip generation are not performed here.

The runtime crate instantiates imported scene nodes into ECS entities. GPU mesh upload lives in z-ideal-renderer and currently accepts indexed triangle lists with positions and optional UV0. GPU texture upload, imported material binding, normal/tangent generation, skins, morph targets, and animation playback remain outstanding.

## Tests

cargo test -p z-ideal-assets --all-features covers URI validation, stable identities, cache eviction/handle lifetime, metadata round trips, missing dependencies, dependency cycles, glTF and GLB import, decoded image/material texture references, external dependency reporting, path-traversal rejection, source/per-file/aggregate byte-budget enforcement, invalid indices, malformed primitives, and topology-specific index counts.
