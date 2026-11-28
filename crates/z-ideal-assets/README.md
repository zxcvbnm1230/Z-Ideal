# z-ideal-assets

Renderer-independent asset identity, metadata, dependency tracking, cache ownership, and source import.

## Current contracts

- AssetUri accepts normalized project-relative logical paths and rejects absolute paths, empty components, dot segments, backslashes, NUL, and URI-reserved path characters.
- AssetId is a deterministic 128-bit identifier derived from a stable asset-kind tag and URI. AssetCatalog detects collisions; the hash is not treated as collision-proof.
- AssetMetadata is versioned, rejects unknown serialized fields, records content fingerprints and direct dependencies, and validates identity/importer metadata.
- AssetCatalog supports forward dependency discovery, transactional cycle rejection, deterministic iteration, and a complete-graph validation step.
- AssetCache<T> owns strong references. Handles remain valid after cache eviction; duplicate insertion is an explicit error, not an implicit reload.
- GltfImporter reads glTF 2.0 JSON and GLB documents, external/data-URI buffers and images, indexed/non-indexed mesh primitives, optional normals and UV0, validated node hierarchies with local TRS transforms and mesh references, scene roots/default scene, texture samplers, and metallic-roughness material factors and texture slots. Parser-specific types are converted into engine-facing enums and structures.

## Import boundaries

The importer percent-decodes external URIs, rejects absolute paths, schemes, traversal, queries/fragments, missing resources, and canonical paths outside the source directory before loading buffers/images. It delegates the actual reads and image decoding to gltf and image. It does not yet expose configurable pre-allocation byte/pixel budgets; import only trusted project assets. It preserves decoded image channel format and bytes rather than converting color spaces or building GPU mip chains. Imported nodes/scenes remain CPU-side descriptions and are not yet instantiated into ECS worlds. It records normalized external buffer/image URIs but does not yet build catalog entries automatically.

The first GPU mesh upload path lives in z-ideal-renderer; it currently accepts indexed triangle lists with positions and optional UV0. It does not yet upload textures, generate normals/tangents, import node transforms/skins/morph targets/animations, or create a complete scene render graph.

## Tests

cargo test -p z-ideal-assets --all-features covers URI validation, stable identities, cache eviction/handle lifetime, metadata round trips, missing dependencies, dependency cycles, glTF and GLB import, decoded image/material texture references, external dependency reporting, path-traversal rejection, invalid indices, malformed primitives, and topology-specific index counts.
