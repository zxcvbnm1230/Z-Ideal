# Phase 4 — PBR and lighting

**Status:** in progress. Implemented foundations include bounded, scene-linear Radiance RGBE equirectangular import, six-face cube conversion/upload, deterministic diffuse irradiance convolution, GGX roughness-prefiltered specular mip generation/upload, and runtime PBR image-based lighting using both environment maps.

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
- `EnvironmentLightingMaps::from_equirectangular` produces a cosine-weighted diffuse irradiance cube and GGX importance-sampled specular mip chain using deterministic Hammersley points. Diffuse values store the hemispherical irradiance integral (the PBR Lambert term divides by PI); specular mip roughness increases linearly from zero to one.
- Convolution validates the source and options, bounds sample count to 4,096, generated storage to 512 MiB, and total work to 100 million source samples. Invalid numeric output is rejected. The implementation is a deterministic CPU reference path, not a GPU compute convolution.
- `VulkanEnvironmentCube::from_cube_mips` uploads explicit mip contents into a cube-compatible RGBA32F image, creates a full-chain cube view and trilinear clamp-to-edge sampler, and records staging/image barriers with retained resource lifetimes. It validates sampled-image, transfer-source/destination, and linear-filter format support. It does not generate GGX levels by ordinary box-filtered mip blits.
- CPU tests cover all-face constant-radiance preservation, directional face orientation, diffuse integral normalization, deterministic convolution, and the reduction/spreading of a directional HDR lobe as roughness increases. The validation-layer GPU regression uploads irradiance and a two-level GGX chain, copies both specular levels back, checks exact float values, and verifies no Vulkan validation errors on the NVIDIA development device.

## Remaining work

- Add point/spot light data and bounded light evaluation, with explicit limits and deterministic selection/order.
- Implement filtered shadow maps and SSAO as independently testable render stages.
- Add reference scenes and image regression comparisons with documented tolerances on supported Vulkan CI and hardware runners.

## Current validation

HDRI import, cube conversion, and convolution reference tests are covered by workspace tests. The ignored `z-ideal-renderer` test `hdr_environment_upload_preserves_linear_radiance_and_is_validation_clean` uploads the 2D HDR texture, base cube, diffuse irradiance cube, and a two-level GGX specular cube; it reads the base cube and both specular levels back and compares exact float values. It passed manually on the NVIDIA GeForce 940MX with Khronos validation enabled. IBL shader/descriptor integration, expanded lights, shadows, SSAO, and reference-scene image regression remain outstanding.
