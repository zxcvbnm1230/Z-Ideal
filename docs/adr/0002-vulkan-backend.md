# ADR-0002: Vulkan as the initial graphics backend

- Status: Accepted
- Decision: Vulkan is the sole initial graphics backend on Windows, Linux and macOS. macOS uses the Vulkan portability path, normally MoltenVK.
- Rationale: One explicit backend reduces initial scope while retaining cross-platform capability.
- Consequences: Detect and validate loader, portability extensions, features, queues and validation layers. Fail with actionable diagnostics when requirements are unavailable. Do not claim macOS support until tested on macOS hardware/runners.
