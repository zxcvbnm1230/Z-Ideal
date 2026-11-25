# Phase 2 — Platform and Vulkan

## Exit scope

Phase 2 is complete only when the engine has cross-platform window/input integration, Vulkan initialization and device/queue selection, validation-layer support, presentation-capable queues and a complete swapchain lifecycle, explicitly owned GPU resource wrappers, reproducible shader compilation/reflection, and relevant automated tests. GPU-backed validation must run on suitable runners where available; unrun hardware tests must be identified rather than reported as passing.

## Implemented in the current Phase 2 branch

- Pinned workspace dependencies for ash 0.38.0, ash-window 0.13.0, raw-window-handle 0.6.2, and winit 0.30.13.
- Platform-native event-loop ownership through winit's ApplicationHandler lifecycle.
- Validated initial window configuration and a window-creation boundary that runs on the native event-loop thread.
- Normalized per-window keyboard and mouse state with held/pressed/released transitions, focus-loss release handling, cursor position, and accumulated wheel deltas.
- Dynamic Vulkan loader discovery with actionable loader and API errors.
- Instance-extension and validation-layer discovery before instance creation.
- Optional Khronos validation layer and debug-utils callback; requested validation fails clearly if unavailable.
- Vulkan portability enumeration is enabled when advertised; macOS initialization fails clearly if that extension is absent.
- Deterministic physical-device preference and graphics-queue-family selection.
- Portability-subset device extension enabled when advertised.
- Platform-required Vulkan instance extensions enumerated from the native display handle and validated before instance creation.
- Owned window surfaces with surface capability, format, present-mode, and queue-support queries.
- Physical-device selection requires both graphics and presentation support for a window surface and the swapchain device extension.
- Separate graphics and presentation queues are created when a single queue family cannot serve both roles.
- Explicit instance, debug-messenger, surface, and logical-device destruction ownership.

## Not implemented yet

- Central input event routing between multiple windows and the engine event bus.
- Swapchain creation, image views, acquire/present synchronization, resize/minimize/restore, out-of-date/suboptimal handling, and recreation.
- Swapchain creation, resize/minimize/restore, out-of-date/suboptimal handling, and recreation.
- GPU buffer/image/view/sampler/descriptor/pipeline/synchronization wrappers.
- Shader compiler integration, reflection, and pipeline-layout validation.
- GPU-backed smoke tests, offscreen rendering tests, and validation-layer clean-run evidence.

## Platform notes

The engine targets Vulkan portability, not a hard-coded macOS translation layer. The available Vulkan-on-Metal implementation is provided by the system's installed SDK/driver. Khronos currently recommends KosmicKrisp on Apple Silicon and notes MoltenVK for Intel Macs and compatibility use. Platform support is not considered validated until the corresponding CI and hardware checks pass.

## Validation

The normal CI matrix must run formatting, Clippy with warnings denied, workspace tests, and documentation on Ubuntu 24.04, Windows, and macOS. Runtime Vulkan initialization and GPU-backed tests require runners with a working Vulkan loader, suitable driver, and validation layers; compile-only CI does not prove those runtime capabilities.
