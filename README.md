# Z Ideal

Z Ideal is a cross-platform 3D game engine targeting Windows, macOS, and Ubuntu 24.04+.

## Project status

**Phase 0 — foundation.** The workspace and CI baseline are being established. Engine subsystems are intentionally skeletal until their contracts and tests are implemented in their scheduled phases. No renderer, physics backend, editor workflow, or game-export capability is claimed complete.

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
