# Phase 5 — Physics and Lua

**Status:** in progress. Fixed-step timing, the Box3D world/body adapter,
serializable scene physics components, ECS synchronization, collision filters,
ray queries, and contact/sensor event delivery are implemented. Lua has not
yet been integrated.

## Roadmap scope

- Box3D adapter, collider/body components, queries and events.
- Fixed-step simulation.
- Lua VM, script assets, lifecycle, entity/component API.
- Prefab spawn and entity create/destroy commands.

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
- Contact and sensor begin/end events are copied into owned engine-facing
  snapshots after each successful fixed step.
- Contact pairs are canonicalized by engine body handle. Event ordering is
  deterministic by event kind (contact begin, contact end, sensor begin,
  sensor end), then by handle pair. Sensor/visitor roles remain explicit.
- Shape-to-body mappings preserve retired handles for one step, allowing
  end events after body destruction to retain their original identity.
- PhysicsScene translates physics body handles to scene entities for ray hits
  and event snapshots, including the entity identity for a body removed during
  the immediately preceding synchronization.

Shape casts and overlap queries are not implemented. Parent transforms,
collider scaling, joints, mesh/compound/height-field colliders are also not
implemented.

## Backend evaluation

boxddd is an experimental 0.x safe wrapper over Box3D. Its documented native
desktop support covers Windows, Linux and macOS; its safe API uses opaque
owner-scoped IDs and fallible operations. The competing box3d-rust crate is a
pure-Rust port with a direct C-mirror API. The selected integration prioritizes
the safe, ergonomic wrapper boundary. Exact dependency versions are pinned.
Native build and runtime behavior must continue to be verified by CI on all
supported desktop targets.

## Remaining work

- Add shape casts and overlap queries, and continue validating event ordering
  and lifecycle behavior across mutation edge cases.
- Add Lua VM, bounded execution and diagnostics, script asset contract,
  lifecycle and pause/reload behavior.
- Bind entity/component access through validated handles and deferred scene
  commands, including prefab spawn and entity create/destroy.
