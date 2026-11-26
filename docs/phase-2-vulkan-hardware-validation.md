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

   It creates a native surface and swapchain, acquires a frame, records a render pass with a depth attachment, presents it, and fails on validation errors. For a virtual display, install `xvfb`, `xauth`, and `libxkbcommon-x11-0`, then prefix the command with `xvfb-run -a`.

Record the GPU model, driver version, Vulkan API version, selected queue-family indices, test commands, and exact outcomes with any hardware validation report.

## Platform coverage boundary

The automated quality matrix compiles and runs ordinary tests on Ubuntu 24.04, Windows, and macOS. The dedicated runtime Vulkan job currently runs on Ubuntu 24.04 with Mesa lavapipe and Xvfb. Windows/macOS physical-GPU presentation and vendor-driver validation remain unverified until suitable runners or developer machines execute equivalent tests.
