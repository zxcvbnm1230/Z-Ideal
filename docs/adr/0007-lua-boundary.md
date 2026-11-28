# ADR-0007: Lua scripting boundary

- Status: Accepted
- Decision: Lua VM and bindings live only in `z-ideal-scripting`. Scripts use validated generation-safe handles and deferred commands, never internal pointers or direct ECS storage access. The implementation pins mlua 0.12.3 with vendored Lua 5.4.
- Lifecycle: load, start, host-driven fixed_update, update, on_event, shutdown. ScriptHost attaches multiple scripts to live entities in insertion order, exposes the attached entity as callback-scoped engine.entity, collects diagnostics and prunes stale attachments. Callback-scoped engine APIs expire on return. Reload requires shutdown and creates a fresh VM/script state.
- Commands: a failed Lua callback discards its command buffer. Successful callbacks apply commands FIFO; application is sequential, not transactional, and reports partial entity/prefab creation results.
- Security: restricted standard libraries, 1 MiB source cap, 100,000-instruction default per-callback budget checked every 1,000 instructions, 64 KiB/64-level event payload cap. Memory and wall-clock quotas are not implemented; scripts are trusted input until those controls exist.
- Consequences: entity handles are opaque and World-bound; components are registered by stable ID and converted through serde; prefabs are host-registered and resolved through post-commit tickets. Script asset cooking, editor diagnostics integration, packaged game-loop scheduling, memory quotas and running hot reload remain deferred.
