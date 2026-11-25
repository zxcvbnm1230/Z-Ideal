# ADR-0002: Vulkan as the initial graphics backend

- Status: Accepted
- Decision: Vulkan is the sole initial graphics backend on Windows, Linux and macOS. macOS uses Vulkan portability support; the engine detects and enables the portability-enumeration instance extension and the portability-subset device extension when advertised.
- Rationale: One explicit backend reduces initial scope while retaining cross-platform capability.
- Platform note: The Vulkan-on-Metal implementation is supplied by the installed Vulkan SDK/driver, not bundled or hard-coded by the engine. Khronos has deprecated MoltenVK in favor of KosmicKrisp on Apple Silicon; MoltenVK remains relevant for Intel Macs and compatibility environments. The engine therefore targets the Vulkan portability contract rather than depending on a specific implementation name.
- Consequences: Detect and validate loader, portability extensions, features, queues and validation layers. Fail with actionable diagnostics when requirements are unavailable. Do not claim macOS support until tested on macOS hardware/runners.
