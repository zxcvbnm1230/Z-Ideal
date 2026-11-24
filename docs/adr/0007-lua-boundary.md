# ADR-0007: Lua scripting boundary

- Status: Accepted
- Decision: Lua VM and bindings live only in `z-ideal-scripting`. Scripts use validated handles and commands, never internal pointers or direct ECS storage access.
- Consequences: Define API versioning, lifecycle, execution limits, reload behavior and command-buffer semantics. Test invalid arguments and stale handles.
