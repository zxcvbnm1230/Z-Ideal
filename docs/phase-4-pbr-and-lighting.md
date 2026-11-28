# Phase 4 — PBR and lighting

**Status:** in progress. The first vertical slice adds bounded, scene-linear Radiance RGBE equirectangular environment-map import. This is an importer foundation only; it does not yet provide GPU cubemaps, irradiance convolution, specular prefiltering, or image-based lighting in the PBR shader.

## Roadmap scope

- Metallic-roughness materials and texture slots (the baseline PBR path was introduced in Phase 3).
- HDRI import, irradiance maps, and prefiltered environment maps.
- Directional, point, and spot lighting as scoped.
- Filtered shadow maps and screen-space ambient occlusion (SSAO).

**Exit gate:** reference scenes and deterministic image-regression tests with documented tolerances. Phase 4 remains open until all selected lighting paths and regression scenes meet this gate.

## Implemented foundation

### Radiance HDRI import

- `HdrEnvironmentImporter` reads Radiance RGBE `.hdr` files with a bounded encoded-file read and decoder allocation/dimension limits.
- Decoded output is scene-linear RGB32F and retains values above one; no display-referred tone mapping or clamping occurs during import.
- The initial contract is explicitly limited to 2:1 equirectangular projections. Other projections and EXR are rejected/not supported rather than reinterpreted.
- Every decoded radiance sample is checked for finiteness and non-negativity. Invalid data returns a structured error with pixel/channel location.
- Import options independently bound encoded source bytes, width, height, and decoded storage. Defaults are 256 MiB, 16,384x8,192, and 512 MiB respectively.
- Unit tests cover a valid two-pixel Radiance fixture, scene-linear values above one, malformed headers, projection rejection, invalid options, and encoded/decoded/dimension budgets.

## Remaining work

- Add a renderer-owned float HDR texture upload path with explicit linear format/feature validation and staging-resource retention.
- Convert equirectangular HDRI into a six-face cube image with validated image views, transitions, and sampling.
- Generate diffuse irradiance and roughness-prefiltered specular environment levels; validate shader interfaces and numerical behavior.
- Extend the PBR scene/material contract to sample environment lighting and combine it with direct lighting in linear space.
- Add point/spot light data and bounded light evaluation, with explicit limits and deterministic selection/order.
- Implement filtered shadow maps and SSAO as independently testable render stages.
- Add reference scenes and image regression comparisons with documented tolerances on supported Vulkan CI and hardware runners.

## Current validation

The HDRI importer is covered by `cargo test -p z-ideal-assets --all-features`. Full workspace validation and cross-platform CI are required before this phase increment is considered merged. GPU environment-map and lighting paths are not implemented or claimed.
