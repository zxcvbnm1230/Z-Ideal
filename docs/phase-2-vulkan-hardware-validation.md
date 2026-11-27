# Phase 2 — Vulkan hardware validation procedure

This procedure supplements the automated Ubuntu 24.04 Mesa lavapipe job. Lavapipe validates Vulkan command execution, resource lifetime, synchronization, offscreen rendering, and X11 swapchain presentation, but it is a software implementation and does not establish physical-GPU driver correctness.

## Linux physical-GPU smoke test

1. Install the Vulkan loader, the vendor's Vulkan ICD/driver, and `VK_LAYER_KHRONOS_validation`.
2. Confirm the loader enumerates the intended physical GPU:

   ```sh
   vulkaninfo --summary
   ```

   The test is unavailable if `vkEnumeratePhysicalDevices` fails or no physical device is listed. Do not treat a successful build as runtime validation.
3. Run the deterministic offscreen graphics test:

   ```sh
   cargo test -p z-ideal-renderer --all-features --test vulkan_offscreen -- --ignored --nocapture
   ```

   It compiles/reflects shaders, creates a graphics pipeline, draws a triangle, copies the image to host-visible memory, checks pixels, and fails on Vulkan validation errors. Its output reports the selected graphics and transfer queue-family indices. When they differ, the test performs a release/semaphore/acquire ownership transfer; when they match, it tests transfer-queue submission without a family ownership change. A cross-family ownership result requires a device where those family indices differ; the current lavapipe runner reports both as family 0.
4. Run the native X11 presentation test on an X11 desktop, Xwayland, or Xvfb:

   ```sh
   cargo test -p z-ideal-renderer --all-features --test vulkan_swapchain -- --ignored --nocapture
   ```

   It creates a native surface and swapchain, acquires and presents a depth-attachment frame, verifies that zero-sized recreation leaves the existing swapchain ready, recreates the swapchain, submits and presents a second frame, and fails on validation errors. It currently recreates at the same extent; it does not synthesize a native resize event. For a virtual display, install `xvfb`, `xauth`, and `libxkbcommon-x11-0`, then prefix the command with `xvfb-run -a`.

Record the GPU model, driver version, Vulkan API version, selected queue-family indices, test commands, and exact outcomes with any hardware validation report.

## Completed physical-GPU validation record

**Date:** 2026-11-27

- **Host:** CachyOS Linux, kernel `7.3.6-3-cachyos`; X11 test ran on the Xwayland display (`DISPLAY=:0`).
- **GPU:** NVIDIA GeForce 940MX, NVIDIA driver `580.196.07`, Vulkan API `1.4.312`.
- **Validation layer:** `VK_LAYER_KHRONOS_validation` version `1.4.363`.
- **Offscreen queue families:** graphics `0`, transfer `1`; the test exercised the cross-family release, semaphore handoff, acquire, transfer copy, and readback path.
- **Presentation queue family:** `0`.

Commands and outcomes (run with the Khronos validation layer available through `VK_LAYER_PATH` and `LD_LIBRARY_PATH`):

```sh
cargo test -p z-ideal-renderer --all-features --test vulkan_offscreen -- --ignored --nocapture
```

**Passed.** The selected device was `NVIDIA GeForce 940MX`; triangle rendering, deterministic image readback/pixel checks, pooled readback comparison, and cross-family transfer ownership completed with no Vulkan validation errors.

```sh
cargo test -p z-ideal-renderer --all-features --test vulkan_swapchain -- --ignored --nocapture
```

**Passed.** The selected device was `NVIDIA GeForce 940MX`; X11 surface/swapchain creation, a depth-attachment frame, zero-extent recreation deferral, same-extent swapchain recreation, a second frame submission/presentation, and teardown completed with no Vulkan validation errors.

This record establishes one Linux NVIDIA driver path only. It does not validate Ubuntu 24.04, Windows, macOS, Apple Vulkan portability, other vendor drivers, or a size-changing swapchain recreation.

## Completed native-window validation record

**Date:** 2026-11-27

Command:

```sh
WINIT_UNIX_BACKEND=x11 cargo run -p z-ideal-platform --example window_smoke
```

**Passed** on the same CachyOS Xwayland desktop. The example ran `PlatformEventLoop` on the process main thread, created a 64x64 native window, requested a 96x64 client area, received the native resize event, verified the updated physical extent and incremented surface generation through `WindowManager`, removed the window, and exited the event loop cleanly. This validates the Linux X11 path only.

## Platform coverage boundary

The automated quality matrix compiles and runs ordinary tests on Ubuntu 24.04, Windows, and macOS. The dedicated runtime Vulkan job currently runs on Ubuntu 24.04 with Mesa lavapipe and Xvfb. Windows/macOS physical-GPU presentation and vendor-driver validation remain unverified until suitable runners or developer machines execute equivalent tests.
