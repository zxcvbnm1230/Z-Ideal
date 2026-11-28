# z-ideal-scripting

The scripting crate owns the Lua VM boundary. Lua values and functions remain
inside this crate. Scripts communicate with the ECS through generation-safe
entity userdata, a typed component registry, and scoped command APIs.

## Script source assets

The assets crate provides LuaScriptImporter for caller-supplied source bytes.
It accepts project-relative .lua URIs, valid UTF-8, and source up to 1 MiB.
An optional first-line header of the form -- z-ideal-script-api: 1 declares
the API version; omitting it selects the current version. Unsupported or
malformed versions are rejected during import.

The importer returns ScriptAsset with the original source, logical URI,
version and deterministic 64-bit FNV-1a content fingerprint. Its metadata is
registered as AssetKind::Script with importer ID z-ideal-lua/1 and no implicit
dependencies. The fingerprint is non-cryptographic and must not be used for
authenticity decisions. Import does not compile or execute Lua. The runtime
compiles an imported asset through ScriptInstance::load_asset before attaching
it to ScriptHost. Filesystem discovery, build cooking and bytecode generation
are not implemented; the caller supplies source bytes to the importer.

## Runtime and lifecycle

The runtime pins **mlua 0.12.3** with **Lua 5.4** built from vendored sources.
No host Lua installation is required.

Each `ScriptInstance` owns one Lua VM and one script table. A source chunk must
return a table with optional lifecycle functions:

```lua
return {
    start = function(self, engine)
        self.elapsed = 0.0
        self.root = engine.entities.spawn()
    end,

    fixed_update = function(self, fixed_delta_seconds, engine)
        self.elapsed = self.elapsed + fixed_delta_seconds
    end,

    update = function(self, delta_seconds, engine)
        -- Frame/presentation-oriented script work.
    end,

    on_event = function(self, event_name, payload, engine)
        -- Handle a bounded JSON event.
    end,

    shutdown = function(self, engine)
        engine.entities.destroy(self.root)
    end,
}
```

The supported lifecycle is `load -> start -> fixed_update* -> update* ->
event* -> shutdown`. All callbacks are optional. Update/event callbacks are
only legal after start. Delta values are seconds and must be finite and
non-negative. Fixed updates are driven by the host; ScriptInstance does not
own a timestep accumulator. Pausing suppresses fixed and variable update
callbacks, but events and shutdown remain available.

The script table is passed as the first callback argument. The scoped engine
API table is the second argument to start/shutdown, and the third argument to
fixed_update/update. Event callbacks receive name and payload before the API
table. The API table expires when that callback returns; retaining it and
calling it later returns an error.

Use `start_with_world`, `fixed_update_with_world`, `update_with_world`,
`dispatch_event_with_world` and `shutdown_with_world` to enable scene access.
The instance binds to the identity of the first World it uses and rejects
later calls with another World. Scene replacement preserves World identity,
while ordinary scene loading invalidates prior live entity generations.

The VM and script table are owned by ScriptInstance. The VM is not Send or Sync
and must be driven on its owning engine thread.

## Lua scene API

- `engine.entities.spawn()` queues an entity creation and returns opaque
  userdata. The handle is remapped to a generation-safe runtime entity after
  commit and can be retained on the script table.
- `engine.entities.destroy(entity)` queues entity destruction.
- `engine.entities.set_parent(child, parent_or_nil)` queues a hierarchy
  change; nil makes the child a root.
- `engine.transforms.get(entity)` reads the local Transform snapshot captured
  at callback entry. It returns nil when the live entity has no Transform and
  reports an error for stale handles.
- `engine.transforms.set(entity, transform)` queues a validated Transform
  replacement. The table requires `translation={x,y,z}`,
  `rotation={x,y,z,w}` and `scale={x,y,z}` arrays.
- `engine.components.get(entity, component_id)` reads a registered component
  as a JSON-compatible Lua value. Missing components return nil.
- `engine.components.set(entity, component_id, value)` deserializes and
  queues a component replacement. World insertion performs the component's
  normal validation before commit.
- `engine.components.remove(entity, component_id)` queues component removal.

Register user component types on the ScriptInstance with
`register_component::<T>()`, and also register them in every World with
`World::register_component::<T>()`. Component IDs are stable serialized IDs,
not Rust type names. Lua component values use serde-compatible JSON types;
unknown IDs, invalid fields and invalid component values are errors. Component
reads are callback-entry snapshots; writes/removals are deferred.

Transform values are local-space. Z Ideal uses a right-handed coordinate
system, +Y up and -Z forward. Quaternion arrays use x,y,z,w order. The math
layer validates finite translation/scale and finite non-zero rotation, which
is normalized before insertion. The host must register Transform in the World
before scripts can set it.

## Prefabs

The host registers validated prefabs by name with
`ScriptInstance::register_prefab`.

- `engine.prefabs.instantiate(name)` queues an instance and returns an opaque
  ticket.
- `engine.prefabs.entity(ticket, source_entity_id)` resolves one entity from
  a committed instance. A ticket cannot be resolved during the same callback
  that queued instantiation; resolve it in a later callback.
- `engine.prefabs.destroy(ticket)` queues destruction of all entities in the
  instance. The ticket becomes invalid after successful command application.

Prefab instantiation itself is transactional and preserves runtime handles
for existing scene entities. A failed later command in the same command buffer
can still leave earlier successful commands committed; partial results retain
the prefab ticket mapping so the host/script can recover.

## Command semantics

Each callback receives a fresh command buffer. Commands are applied FIFO after
the Lua callback succeeds. A Lua error discards that callback's buffer. The
scene command buffer itself is sequential, not transactional: if a queued
command fails during application, earlier commands remain applied and later
commands are discarded. Errors include partial entity/prefab creation results
where available. A start callback is considered started after Lua succeeds
even if subsequent command application reports an error, avoiding accidental
replay of the callback's side effects.

Transform and generic component reads are snapshots from callback entry;
writes are deferred until after the callback. A set followed by a get in the
same callback therefore reads the pre-callback value.

## Events, reload and API version

Events are dispatched through the optional
`on_event(self, event_name, json_payload, engine)` callback. Names are limited
to 128 bytes, serialized JSON payloads to 64 KiB and nesting depth to 64.
Events remain dispatchable
while updates are paused.

`Z_IDEAL_SCRIPT_API_VERSION` reports API version 1. Reload replaces the VM and
script state only while the instance is Loaded or Shutdown. A Started script
must first be shut down; use the world-aware shutdown method when it owns scene
resources. The replacement starts in Loaded state with fresh script state.
Failed replacement loading leaves the original instance unchanged. Automatic
hot reload of a running script and state migration are not implemented.

## Execution boundary

The VM loads Lua's table, string, math and UTF-8 libraries in addition to the
language base environment. OS, IO, package, debug and coroutine libraries are
not loaded. File/dynamic loading helpers, protected-call helpers (pcall and
xpcall), explicit garbage-collector controls and console output globals are
removed from the base environment. Source is limited to 1 MiB. Each top-level
chunk and lifecycle callback has an independent instruction budget (100,000
by default, capped at 100,000,000), checked every 1,000 Lua VM instructions.
Each VM also has a per-instance allocator limit (16 MiB by default, configurable
from 256 KiB through 64 MiB). Allocation beyond the limit returns an execution
error and discards that callback's uncommitted scene commands. The
memory_used_bytes and memory_limit_bytes accessors expose VM memory diagnostics.
This allocator limit does not include Rust-side component snapshots or other
host allocations. Protected calls are unavailable so a script cannot catch
and suppress instruction-budget or allocation errors.

**This is not yet a complete hostile-code sandbox.** There is no wall-clock
deadline or quota for expensive native host APIs. Instruction accounting does
not bound time spent inside a native call. Only run trusted project scripts
until those additional controls and a formal trust policy are in place.

## Remaining Phase 5 work

- Filesystem discovery/build cooking for imported script source and
  runtime/editor diagnostics integration.
- Wall-clock and host API quotas, plus a documented trusted/untrusted script policy.
- Automatic hot reload of running scripts with explicit state migration.
- Broader integration into the packaged runtime loop and editor play mode.
