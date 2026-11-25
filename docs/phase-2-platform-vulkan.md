# Phase 2 — Platform and Vulkan

## Exit scope

Phase 2 is complete only when the engine has cross-platform window/input integration, Vulkan initialization and device/queue selection, validation-layer support, presentation-capable queues and a complete swapchain lifecycle, explicitly owned GPU resource wrappers, reproducible shader compilation/reflection, and relevant automated tests. GPU-backed validation must run on suitable runners where available; unrun hardware tests must be identified rather than reported as passing.

## Implemented in the current Phase 2 branch

- Pinned workspace dependencies for ash 0.38.0 and winit 0.30.13. Additional surface/window-handle dependencies will be introduced when surface ownership is implemented.
- Platform-native event-loop ownership through winit's ApplicationHandler lifecycle.
- Validated initial window configuration and a window-creation boundary that runs on the native event-loop thread.
- Dynamic Vulkan loader discovery with actionable loader and API errors.
- Instance-extension and validation-layer discovery before instance creation.
- Optional Khronos validation layer and debug-utils callback; requested validation fails clearly if unavailable.
- Vulkan portability enumeration is enabled when advertised; macOS initialization fails clearly if that extension is absent.
- Deterministic physical-device preference and graphics-queue-family selection.
- Portability-subset device extension enabled when advertised.
- Explicit instance, debug-messenger, and logical-device destruction ownership.

## Not implemented yet

- Normalized input state and event routing contract beyond the native window-event callback.
- Vulkan surface creation and present-queue selection.
- Swapchain creation, resize/minimize/restore, out-of-date/suboptimal handling, and recreation.
- GPU buffer/image/view/sampler/descriptor/pipeline/synchronization wrappers.
- Shader compiler integration, reflection, and pipeline-layout validation.
- GPU-backed smoke tests, offscreen rendering tests, and validation-layer clean-run evidence.

## Platform notes

The engine targets Vulkan portability, not a hard-coded macOS translation layer. The available Vulkan-on-Metal implementation is provided by the system's installed SDK/driver. Khronos currently recommends KosmicKrisp on Apple Silicon and notes MoltenVK for Intel Macs and compatibility use. Platform support is not considered validated until the corresponding CI and hardware checks pass.

## Validation

The normal CI matrix must run formatting, Clippy with warnings denied, workspace tests, and documentation on Ubuntu 24.04, Windows, and macOS. Runtime Vulkan initialization and GPU-backed tests require runners with a working Vulkan loader, suitable driver, and validation layers; compile-only CI does not prove those runtime capabilities.
