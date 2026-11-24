# Development Roadmap

1. **Phase 0 — Foundation:** workspace, stable toolchain, quality gates, logging/errors/config/build metadata, platform build smoke tests.
2. **Phase 1 — Core, math, ECS and scenes:** math conventions, generation-safe entities, component storage, transforms, deterministic scheduling, versioned serialization and prefabs.
3. **Phase 2 — Platform and Vulkan:** window/input, Vulkan initialization/device/queues, validation, swapchain, resource wrappers and shader compile/reflection.
4. **Phase 3 — Assets and first renderer:** glTF/GLB import, textures/materials, asset cache, mesh/camera/depth/HDR/tone mapping.
5. **Phase 4 — PBR and lighting:** metallic-roughness, HDRI/IBL, lights, shadows and SSAO.
6. **Phase 5 — Physics and Lua:** Box3D adapter, colliders/bodies/queries/events, fixed step, Lua VM/API/lifecycle and entity commands.
7. **Phase 6 — Audio and editor:** miniaudio, spatial audio, hierarchy/inspector/viewport/gizmos/assets/play mode.
8. **Phase 7 — Export and release:** asset cooking, runtime packaging, clean-room launch, profiling, diagnostics, licensing and docs.

A phase exits only when its tests, documentation, and platform requirements are met. See `docs/adr` for architectural decisions.
