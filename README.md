# Z Ideal

Z Ideal is a cross-platform 3D game engine targeting Windows, macOS, and Ubuntu 24.04+.

## Project status

**Phase 1 — core, math, ECS and scenes: merged to main.** The engine now has documented math conventions and a validated Transform contract, generation-safe entities, registered sparse-set components, transform hierarchy, deterministic scheduling, deferred structural commands, versioned scene serialization with migrations, and prefab instantiation with entity-reference remapping and overrides.

**Phase 2 — platform and Vulkan: in progress.** The current implementation adds a cross-platform native window/event-loop boundary, normalized per-window keyboard and mouse state, centralized input routing, a native-window manager with focus/resize/minimize/restore tracking, Vulkan loader and instance initialization, validation diagnostics, native window surfaces, deterministic graphics/presentation queue selection, logical-device ownership, and swapchain creation/recreation with image views and acquire/present status handling. Owned Vulkan fence and binary-semaphore wrappers are implemented with explicit device lifetimes and conservative idle-before-destroy behavior. Owned Vulkan buffers support device-local, upload, and readback memory policies with bounds-checked host access and atom-aligned non-coherent cache maintenance. Dedicated-allocation Vulkan images, lifetime-bound image views, and validated samplers are implemented. Descriptor-set layouts/pools and complete descriptor updates for uniform/storage buffers, sampled images, separate samplers, combined image samplers, and storage images are implemented with resource leases, device/format/usage/layout validation, and lifetime retention. Lifetime-bound pipeline layouts validate push-constant ranges. Graphics/compute pipeline wrappers, shader compilation/reflection, frame scheduling, and GPU-backed tests remain incomplete. No rendered frame is claimed.

## Requirements

- Stable Rust (workspace MSRV: 1.85)
- Git
- A Vulkan loader and a Vulkan-capable driver at runtime
- Khronos Vulkan SDK and validation layers recommended for development and validation

On macOS, the Vulkan portability path is required. The available portability implementation depends on the target Mac and installed Vulkan SDK/driver; consult the current Vulkan SDK guidance before setting up a development machine.

## Workspace validation

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --workspace --all-features --no-deps
```

## Architecture

See [docs/architecture.md](docs/architecture.md), [docs/roadmap.md](docs/roadmap.md), [docs/phase-2-platform-vulkan.md](docs/phase-2-platform-vulkan.md), and the architecture decision records in [docs/adr](docs/adr).

## Development

Read [AGENTS.md](AGENTS.md) before changing code. CI validates formatting, Clippy, tests, and documentation on Ubuntu, Windows, and macOS.
