# Phase 5 — Physics and Lua

**Status:** in progress. Fixed-step timing, the Box3D world/body adapter,
serializable scene physics components, ECS synchronization, collision filters,
ray/shape queries, overlap queries, contact/sensor event delivery, Lua source
import, and a Lua VM with scene bindings, lifecycle, events, prefab commands
and bounded instruction, elapsed-time and Lua-VM memory execution are
implemented. Script filesystem discovery and transactional package cooking are
implemented. Runtime/editor integration, hard deadlines for native host calls
and host API quotas remain outstanding. The VM allocator
quota does not include Rust-side component snapshots or other host allocations.

## Roadmap scope

- Box3D adapter, collider/body components, queries and events.
- Fixed-step simulation.
- Lua VM, lifecycle, typed component API, event dispatch and reload-after-shutdown.
- Deferred entity and prefab commands with generation-safe script handles.

**Exit gate:** every public physics and Lua API path is covered by automated
tests. This gate is not yet met.

## Implemented foundation

### Fixed-step accumulator

- FixedStepAccumulator converts frame durations into a bounded count of
  fixed simulation steps.
- The default policy is 16,666,667 ns per step (nearest-nanosecond 60 Hz) and
  at most eight catch-up steps per frame.
- When the catch-up limit is reached, excess whole steps are dropped while the
  fractional remainder is retained for presentation interpolation.
- Pausing clears pending time. Frame time supplied while paused is ignored and
  reported in the plan; resuming starts with an empty accumulator.
- Duration overflow returns a structured error and leaves accumulator state
  unchanged.
- Interpolation is presentation-only and must not be written back into the
  simulation state.

### Box3D backend and world adapter

- ADR-0004 selects exact dependency boxddd 0.4.0, with all backend-specific
  types contained in z-ideal-physics.
- Workspace MSRV is raised from Rust 1.85 to 1.92 to satisfy boxddd's
  documented MSRV. The physics crate's full test suite passes on Rust 1.92;
  the development machine uses Rust 1.100.
- PhysicsWorld owns a native Box3D world and exposes engine math types,
  validated world/body/collider descriptors and world-unique, generation-safe
  body handles.
- Initial primitive colliders are spheres and boxes; body types are static,
  kinematic and dynamic.
- Body plus collider creation is transactional. A failed collider creation
  triggers body rollback, and a rollback failure preserves both errors.
- Stale handles and handles from another PhysicsWorld are rejected.
- PhysicsWorld::step advances one configured fixed step with an explicit
  Box3D solver substep count.
- Body world transforms and linear velocities can be updated through
  validated engine-facing methods.
- The Box3D Foundation freezes process-wide length_units_per_meter at first
  initialization. Conflicting later configurations are returned as errors.

### Scene components and ECS synchronization

- z-ideal-scene provides serializable, validated RigidBody and Collider
  components with stable IDs and strict unknown-field rejection.
- RigidBody captures body type, gravity/damping, sleep and bullet settings,
  and initial linear velocity.
- Collider captures sphere/box geometry, material properties, sensor state,
  event-generation flags and collision category/mask bitsets.
- Newly added category/mask fields default to all bits when loading older
  serialized Collider payloads that predate filtering.
- PhysicsScene owns a PhysicsWorld and a generation-safe mapping from scene
  Entity handles to physics body handles.
- synchronize creates bodies for entities with both components, updates
  static/kinematic transforms, recreates bodies after descriptor changes,
  and destroys bodies after component removal or entity despawn.
- writeback_dynamic_transforms copies simulated dynamic poses into scene
  Transform components after fixed simulation steps. Presentation
  interpolation remains separate.
- Physics entities are currently required to be root entities with unit local
  scale. Parent transforms and scaled colliders return explicit errors; they
  are not silently approximated.
- Entities with only one of RigidBody or Collider do not participate.

### Collision filtering, queries and events

- Collider category and mask bitsets are passed to Box3D shape filters.
  The backend's collision filter rules are used without a separate engine-side
  approximation.
- PhysicsRayCast supports world-space origin/displacement and query category
  and mask bitsets. PhysicsWorld::cast_ray returns world-space points/normals
  and hit fractions, sorted by fraction and then stable engine body handle.
- Convex shape casts and overlap queries support sphere, box and Y-aligned
  capsule proxies with world-space pose, displacement and query filters.
  Casts sort by fraction/handle; overlaps sort by handle. PhysicsScene maps both
  result types to scene entities.
- Contact and sensor begin/end events are copied into owned engine-facing
  snapshots after each successful fixed step.
- Contact pairs are canonicalized by engine body handle. Event ordering is
  deterministic by event kind (contact begin, contact end, sensor begin,
  sensor end), then by handle pair. Sensor/visitor roles remain explicit.
- Shape-to-body mappings preserve retired handles for one step, allowing
  end events after body destruction to retain their original identity.
- PhysicsScene translates physics body handles to scene entities for ray and
  shape-query hits and event snapshots, including the entity identity for a
  body removed during the immediately preceding synchronization.

Parent transforms, collider scaling, joints, mesh/compound/height-field
colliders are not implemented.

### Lua VM, lifecycle and scene API

- ADR-0007 keeps the VM and all bindings in z-ideal-scripting. The runtime pins
  mlua 0.12.3 with vendored Lua 5.4; no host Lua library is required.
- Each ScriptInstance owns one VM and one script table. The lifecycle is
  load, start, fixed_update, update, on_event and shutdown. Callbacks are
  optional; fixed updates are host-driven. Pausing suppresses fixed and
  variable updates while allowing events and shutdown.
- The VM loads table, string, math and UTF-8 libraries. OS, IO, package,
  debug and coroutine libraries are not loaded; dynamic/file loading,
  protected-call helpers, explicit GC controls and console output globals are
  removed. Source is limited to 1 MiB; each chunk/callback has a default
  100,000-instruction budget checked every 1,000 instructions.
- Each VM has a 16 MiB default allocator limit, configurable from 256 KiB to
  64 MiB. This limit excludes Rust-side host allocations. Each chunk/callback
  also has a 50 ms default elapsed-time budget, configurable from 1 ms to 5 s
  and checked every 1,000 VM instructions. The hook cannot interrupt a long
  native host call, and native host API quotas are not implemented; run trusted
  project scripts until those controls and a formal trust policy exist.
- Engine API tables are callback-scoped and become invalid after callback
  return. Script instances bind to one opaque World identity; use of another
  World is rejected.
- Scripts can queue entity spawn/destroy/parent changes, read/write validated
  Transform values, and read/write/remove serde-backed registered components.
  Component reads are callback-entry snapshots; writes/removals are deferred.
- Host-registered validated prefabs can be instantiated and destroyed through
  deferred commands. A prefab ticket resolves to its source-entity mapping
  after the callback commits; it cannot be resolved in the same callback.
- A failed Lua callback discards its command buffer. Command application is
  FIFO and sequential, not transactional; a later command failure preserves
  earlier changes and returns partial entity/prefab creation results.
- ScriptHost owns instances attached to live entities, assigns engine.entity,
  supports multiple scripts per entity in insertion order, isolates failures
  into diagnostics and prunes despawned attachments. The runtime crate
  re-exports the host API.
- Events use JSON payloads bounded to 64 KiB and 64 nesting levels, with names
  bounded to 128 bytes. Reload replaces the VM only while Loaded or Shutdown;
  running hot reload and state migration are not implemented.
- Current focused coverage: 62 scripting tests, 46 scene tests, 40 physics
  tests and 53 asset tests pass on the development host. These are focused
  results, not the full workspace gate.

### Script asset discovery and cooking

- ScriptBuildPipeline recursively discovers Lua sources and sorts them by
  normalized project-relative URI. Symbolic links, invalid UTF-8 paths,
  malformed/unsupported API headers and invalid source are errors.
- Per-file source reads are bounded to the importer limit (1 MiB by default),
  and packages are capped at 16,384 scripts and 256 MiB combined source.
- Cooking writes original source files plus a versioned manifest.json into a
  unique sibling staging directory, then publishes the completed directory
  with a rename. The destination must not already exist; incremental updates
  and overwrite are intentionally unsupported.
- Manifest loading validates schema/API versions, script-kind/importer metadata,
  sorted unique URIs, dependency closure and exact package-relative paths.
  Every source is re-imported and checked against its recorded content hash.
  Symlinked package roots, manifests and sources are rejected.
- ScriptBuildManifest::load_script_asset loads and re-verifies an individual
  source for ScriptInstance::load_asset.
- Portable standard filesystem APIs cannot provide a race-free directory walk
  against a concurrently hostile writer. Source and package trees must remain
  stable while discovery/loading runs; symlinks are rejected.

## Backend evaluation

boxddd is an experimental 0.x safe wrapper over Box3D. Its documented native
desktop support covers Windows, Linux and macOS; its safe API uses opaque
owner-scoped IDs and fallible operations. The competing box3d-rust crate is a
pure-Rust port with a direct C-mirror API. The selected integration prioritizes
the safe, ergonomic wrapper boundary. Exact dependency versions are pinned.
Native build and runtime behavior must continue to be verified by CI on all
supported desktop targets.

## Remaining work

- Add joints, mesh/compound/height-field colliders, parent-transform support and
  collider scaling only when the public scene/physics contracts are specified.
- Connect script diagnostics to editor tooling and the engine logging pipeline;
  invoke the script package pipeline from the project build workflow.
- Add hard deadlines for native host calls, native API quotas and a documented
  trusted/untrusted script policy.
- Drive the runtime-exposed ScriptHost from packaged startup, fixed-step
  scheduling and editor play mode; the packaged app has no complete game-loop
  scheduler yet.
- Add running hot reload with explicit state migration, if required by product
  behavior.
