# Phase 4 — PBR and lighting

**Status:** complete and validated. The selected Phase 4 scope includes bounded scene-linear HDRI import, diffuse/specular environment lighting, metallic-roughness PBR, directional/point/spot direct lights, directional PCF shadows, depth-based SSAO, and deterministic reference-image regression.

## Roadmap scope

- Metallic-roughness materials and texture slots (the baseline PBR path was introduced in Phase 3).
- HDRI import, irradiance maps, and prefiltered environment maps.
- Directional, point, and spot lighting as scoped.
- Directional filtered shadow maps and screen-space ambient occlusion (SSAO); point/spot shadow maps and cascades are not currently selected.

**Exit gate:** met. The two-material PBR reference scene exercises directional, point, and spot lighting, directional PCF, environment lighting, and AO composition. The 32x32 RGBA8 golden uses exact alpha, a 16/255 per-channel tolerance, and at most 16 edge-pixel outliers; the separate SSAO regression checks a synthetic depth discontinuity.

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

## Implemented lighting stages and scope limits

- Directional-light PBR shadow sampling uses a caller-owned depth comparison image, light view-projection matrix, depth bias, and 3x3 PCF. The depth-only render stage records scene meshes with a dedicated 64-byte MVP path.
- `VulkanSsaoRenderer` dispatches a bounded 8x8 compute pass using depth reconstruction, eight deterministic screen-space directions, caller-provided inverse projection, pixel/world radii, bias, and intensity. It writes `R32_SFLOAT` AO, transitions it for fragment sampling, and the PBR shader consumes it through set 1 binding 4 with an explicit strength control.
- The reference scene contains two distinct metallic-roughness material descriptor sets and evaluates one directional, one point, and one spot light. It renders with directional PCF and screen-space AO enabled, then compares the complete 32x32 RGBA8 image against the checked-in golden.
- Explicit limitations: environment convolution is a deterministic CPU reference path, not GPU compute; specular IBL uses an analytic split-sum approximation rather than a BRDF LUT; the current shadow map is directional-only (no point/spot maps or cascades); SSAO is a single eight-direction depth pass without temporal accumulation or a denoising blur. These are documented scope boundaries, not silently substituted implementations.

## Current validation

Workspace validation on the final Phase 4 commit passed formatting, Clippy with warnings denied, all workspace tests, Rust documentation with warnings denied, and `git diff --check`. The workspace reported 262 passed tests and 8 intentionally ignored Vulkan tests. On the NVIDIA GeForce 940MX with Khronos validation enabled, seven ignored Vulkan tests passed: HDR environment upload/readback, offscreen rendering, four runtime scene tests (including the two-material PBR golden), and the SSAO compute/readback regression. The PBR golden compares every RGBA8 pixel to `tests/fixtures/pbr_reference.rgba8`, requires exact alpha, permits at most 16 pixels outside a 16/255 RGB channel tolerance, and separately checks representative pixels for both material variants and the clear target. The SSAO test renders a synthetic near/far depth discontinuity, dispatches the compute pass, reads back AO, verifies finite values in [0,1] and measurable occlusion, and checks Khronos validation. GitHub CI passed the quality matrix on Windows, macOS, and Ubuntu 24.04; the Vulkan workflow passed on lavapipe. The HDR environment upload regression was also validated on the NVIDIA device with exact float readback.
