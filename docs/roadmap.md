# Z Ideal Implementation Roadmap

This is a dependency-ordered roadmap, not a claim that these systems are already implemented.

## Phase 0 — Project foundation
- Cargo workspace, stable toolchain, formatting/lint/test CI.
- `AGENTS.md`, subsystem skills, architecture decision records.
- Structured logging, error conventions, configuration, build metadata.
- Cross-platform build smoke tests.

**Exit gate:** clean workspace builds on Windows, macOS, and Ubuntu 24.04+; CI gates are active.

## Phase 1 — Core, math, ECS, and scenes
- Math types and coordinate conventions.
- Generation-safe entities and component storage.
- Transform hierarchy and deterministic scheduling.
- Scene serialization, validation, and versioning.
- Prefab creation and instantiation.

**Exit gate:** exhaustive ECS/scene unit and integration tests; round-trip and migration fixtures.

## Phase 2 — Platform and Vulkan foundation
- Window/input/platform layer.
- Vulkan instance/device/queue selection.
- Validation layers, swapchain lifecycle, resource wrappers.
- Shader compilation and reflection pipeline.
Currently in progress: see [Phase 2 status](phase-2-platform-vulkan.md).

**Exit gate:** stable startup/shutdown, resize/recreation, validation-clean smoke test on each supported platform.

## Phase 3 — Asset pipeline and first renderer
- glTF 2.0/GLB import, textures, material conversion.
- Asset handles, cache, dependency graph and metadata.
- Mesh rendering, camera, depth, HDR target and tone mapping.

**Exit gate:** deterministic sample scene rendering and robust malformed-asset handling.

## Phase 4 — PBR and lighting
- Metallic-roughness materials and texture slots.
- HDRI import, irradiance and prefiltered environment maps.
- Directional/point/spot lighting as scoped.
- Filtered shadow maps and SSAO.

**Exit gate:** reference scenes and image regression tests with documented tolerances.

## Phase 5 — Physics and Lua
- Box3D adapter, collider/body components, queries and events.
- Fixed-step simulation.
- Lua VM, script assets, lifecycle, entity/component API.
- Prefab spawn and entity create/destroy commands.

**Exit gate:** every public physics and Lua API path covered by automated tests.

## Phase 6 — Audio and editor
- miniaudio device and resource lifecycle.
- Spatial audio components.
- Editor hierarchy, inspector, viewport, gizmos, asset browser and play mode.

**Exit gate:** editor authoring workflow integration tests and stable play/stop state restoration.

## Phase 7 — Export and release hardening
- Asset cooking and runtime manifest.
- Editor-free runtime packaging.
- Clean-room launch tests.
- Performance profiling, crash diagnostics, dependency/license review and release documentation.

**Exit gate:** reproducible release artifacts and clean-environment tests on all supported platforms.

A phase exits only when its tests, documentation, and platform requirements are met. See `docs/adr` for architectural decisions.
