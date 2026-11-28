# Phase 4 — PBR and lighting

**Status:** in progress. The first increments add bounded, scene-linear Radiance RGBE equirectangular import, a linear RGBA32F Vulkan upload path, and six-face cube conversion/upload. Diffuse irradiance convolution, roughness-prefiltered specular maps, and image-based lighting in the PBR shader are not implemented yet.

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

### Renderer-side HDR texture and cube environment

- `VulkanTexture2D::from_hdr_environment` converts imported RGB radiance into linear RGBA32F storage, preserving values above one and adding an opaque alpha channel. It explicitly checks the format features required for sampling and linear mip blits. The generated ordinary mip chain is not roughness prefiltering.
- `EnvironmentCubeMap::from_equirectangular` bilinearly resamples the source into Vulkan cube layer order (+X, -X, +Y, -Y, +Z, -Z), wrapping longitude and clamping at the poles. Output is bounded to 512 MiB and face dimensions up to 4,096.
- `VulkanEnvironmentCube` uploads the six RGBA32F faces into a cube-compatible image, creates a cube view and clamp-to-edge sampler, and records staging/image barriers with retained resource lifetimes. It validates sampled-image and linear-filter format support rather than silently substituting a format.
- CPU tests cover all-face constant-radiance preservation, directional face orientation, invalid inputs, and storage bounds. A validation-layer GPU test uploads both the equirectangular float texture and cube image, copies all six cube faces back, checks exact float values, and verifies no Vulkan validation errors on the NVIDIA development device.

## Remaining work

- Generate diffuse irradiance convolution and roughness-prefiltered specular environment levels; validate shader interfaces and numerical behavior.
- Extend the PBR scene/material contract to sample environment lighting and combine it with direct lighting in linear space.
- Add point/spot light data and bounded light evaluation, with explicit limits and deterministic selection/order.
- Implement filtered shadow maps and SSAO as independently testable render stages.
- Add reference scenes and image regression comparisons with documented tolerances on supported Vulkan CI and hardware runners.

## Current validation

HDRI importer and cube conversion CPU tests are covered by workspace tests. The ignored `z-ideal-renderer` test `hdr_environment_upload_preserves_linear_radiance_and_is_validation_clean` was manually run on the NVIDIA GeForce 940MX with Khronos validation enabled; 2D texture and six-face cube readback passed with no validation errors. Cross-platform CI is required before this increment is considered merged. Irradiance/specular prefiltering and lighting integration remain outstanding.
