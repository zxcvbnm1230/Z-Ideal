# Z Ideal

Z Ideal is a cross-platform 3D game engine targeting Windows, macOS, and Ubuntu 24.04+.

## Project status

**Phase 1 — core, math, ECS and scenes (ready for review).** The current Phase 1 branch implements the math conventions and Transform contract, generation-safe entities, registered sparse-set components, transform hierarchy, deterministic scheduling, deferred structural commands, versioned scene serialization with migrations, and prefab instantiation with entity-reference remapping and overrides. Phase 1 is not merged yet. Rendering, physics, Lua scripting, audio, editor workflows, and game export remain future phases.

## Requirements

- Stable Rust (toolchain pinned to the stable channel; minimum version currently 1.85)
- Git
- Platform-native Vulkan SDK/loader and portability support will be required for renderer development in Phase 2

## Workspace validation

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

## Architecture

See [docs/architecture.md](docs/architecture.md), [docs/roadmap.md](docs/roadmap.md), and the architecture decision records in [docs/adr](docs/adr).

## Development

Read [AGENTS.md](AGENTS.md) before changing code. CI validates formatting, Clippy, tests, and documentation on Ubuntu, Windows, and macOS.
