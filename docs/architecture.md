# Z Ideal Architecture

## Workspace boundaries

| Crate | Responsibility | Dependency direction |
|---|---|---|
| z-ideal-core | lifecycle, time, logging, events, config, common errors | foundational; no engine-crate dependencies |
| z-ideal-math | math types and conversions | standalone |
| z-ideal-scene | ECS, hierarchy, transforms, serialization, prefabs | core, math |
| z-ideal-assets | IDs, import, metadata, dependency graph, cache | core |
| z-ideal-renderer | Vulkan device/resources/rendering | core, math, assets |
| z-ideal-physics | Box3D adapter and engine physics API | core, math, scene |
| z-ideal-scripting | Lua VM and safe bindings | core, scene |
| z-ideal-audio | miniaudio integration and spatial audio | core, math |
| z-ideal-platform | window/input/platform services | core |
| z-ideal-test-support | fixtures and test utilities | test-only consumers |
| z-ideal-editor | authoring UI and tools | engine-facing crates |
| z-ideal-runtime | game startup and packaged-game support | engine-facing crates; never editor |
| apps/editor | editor executable | z-ideal-editor |
| apps/runtime | runtime executable | z-ideal-runtime |

No cyclic crate dependencies. The test-support crate must never be a production dependency. Editor dependencies must never flow into runtime.

Dependency rows above describe the intended architecture. A crate adds a direct Cargo dependency only when its implementation actually uses that crate; future subsystem dependencies are added with the corresponding implementation rather than kept as unused manifest entries.

## Cross-cutting contracts

- Stable Rust; workspace-managed dependencies.
- Structured errors at subsystem boundaries.
- Explicit ownership and destruction order for native/GPU/audio resources.
- Unsafe and FFI isolated with documented invariants.
- Versioned, validated serialization with explicit migrations.
- Fixed-step simulation and documented coordinate/unit conventions.
- Deterministic behavior where feasible; nondeterminism documented.
