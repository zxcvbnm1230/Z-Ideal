# Phase 5 — Physics and Lua

**Status:** in progress. Fixed-step timing, the initial Box3D world/body
adapter, serializable scene physics components, and ECS synchronization are
implemented. Events/queries and Lua have not yet been integrated.

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
- Collider captures sphere/box geometry, material properties, sensor state
  and event-generation flags.
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

The scene synchronizer does not yet deliver collision/sensor events, expose
ray/shape queries, support parented transforms or collider scaling, or provide
joints, mesh/compound/height-field colliders.

## Backend evaluation

boxddd is an experimental 0.x safe wrapper over Box3D. Its documented native
desktop support covers Windows, Linux and macOS; its safe API uses opaque
owner-scoped IDs and fallible operations. The competing box3d-rust crate is a
pure-Rust port with a direct C-mirror API. The selected integration prioritizes
the safe, ergonomic wrapper boundary. Exact dependency versions are pinned.
Native build and runtime behavior must continue to be verified by CI on all
supported desktop targets.

## Remaining work

- Add collision filtering, ray/shape queries and deterministic ordered
  collision/sensor event delivery; specify and test event ordering.
- Add Lua VM, bounded execution and diagnostics, script asset contract,
  lifecycle and pause/reload behavior.
- Bind entity/component access through validated handles and deferred scene
  commands, including prefab spawn and entity create/destroy.
