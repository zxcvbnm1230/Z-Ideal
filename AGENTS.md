# Z Ideal — Agent Development Guidelines

## Mission
Build a dependable, cross-platform 3D game engine for Windows, macOS, and Ubuntu 24.04+. Prioritize correctness, maintainability, observability, reproducibility, and tested behavior over speed of implementation.

## Non-negotiable rules
1. Do not introduce hacks, placeholder implementations presented as complete, silent fallbacks, or ignored errors.
2. Do not claim a feature works until it is implemented, tested, and its limitations are documented.
3. Keep engine subsystems behind narrow public interfaces. Do not let editor code leak into runtime crates.
4. Keep unsafe Rust and FFI isolated, documented, and minimized. Every unsafe block must explain its safety invariants.
5. Treat GPU resources, audio resources, physics handles, and asset handles as explicitly owned resources with defined lifetimes.
6. Make serialization versioned and migration-aware. Never silently discard unknown or malformed data.
7. Keep public APIs documented. Changes to public APIs require tests and documentation updates.
8. Add regression tests for every fixed bug.
9. Never disable, skip, weaken, or delete a failing test merely to make CI green. Explain and resolve the underlying failure.
10. Pin and audit native dependencies and build requirements across all supported platforms.

## Before changing code
- Read the relevant crate's `README.md`, module docs, tests, and nearby implementation.
- Check workspace conventions and existing abstractions before introducing new ones.
- Identify affected platforms, public APIs, serialized formats, and runtime packaging.
- Prefer the smallest cohesive change that preserves architectural boundaries.

## Rust conventions
- Use stable Rust unless a documented requirement justifies otherwise.
- Use workspace-managed dependency versions; avoid duplicate versions without a written reason.
- Run `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, and relevant `cargo test` commands.
- Return structured errors at subsystem boundaries. Do not use `unwrap()`/`expect()` in production paths unless an invariant is locally proven and documented.
- Avoid panics for invalid user-authored scenes, assets, scripts, or project files.
- Do not expose backend-specific types from public engine APIs unless explicitly intended.
- Keep deterministic behavior where feasible; document nondeterministic systems and sources.

## Architecture boundaries
- `z-ideal-core`: lifecycle, time, logging, events, configuration, common errors.
- `z-ideal-math`: engine math types and conversions; no renderer or physics dependency.
- `z-ideal-scene`: entities, components, hierarchy, transforms, serialization, prefabs.
- `z-ideal-assets`: asset IDs, import pipeline, metadata, dependency graph, cache and lifetime.
- `z-ideal-renderer`: Vulkan device/resource management, render graph, shaders and rendering.
- `z-ideal-physics`: Box3D adapter and engine-facing physics API.
- `z-ideal-scripting`: Lua VM, bindings, lifecycle and script diagnostics.
- `z-ideal-audio`: miniaudio integration and spatial audio API.
- `z-ideal-platform`: windows, input, timing, filesystem and platform services.
- `z-ideal-editor`: editor UI and authoring tools; must not be a dependency of runtime.
- `z-ideal-runtime`: game startup, scene loading, runtime loop and packaged-game support.
- `z-ideal-test-support`: fixtures and test utilities; production crates must not depend on it.

Avoid cyclic crate dependencies. If two systems need to communicate, prefer events, explicit interfaces, or a small shared contract crate over mutual dependencies.

## Required validation
```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```
Report exact commands and outcomes. If a platform or GPU test cannot run, say so explicitly; never imply it passed.
