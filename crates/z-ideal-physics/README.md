# z-ideal-physics

Engine-facing physics contracts and the isolated Box3D adapter. Backend types
are not part of the engine-facing public API.

## Fixed-step timing

FixedStepAccumulator converts variable frame durations into a bounded number
of fixed simulation steps. The default policy uses a 16,666,667 ns step and a
maximum of eight catch-up steps per frame. When that limit is reached, excess
whole steps are dropped and the fractional remainder is retained. The returned
interpolation alpha is for presentation only; interpolated transforms must not
be written back into simulation state.

Pausing clears pending accumulated time, and frame durations supplied while
paused are ignored and reported as dropped_time. Resuming starts with an empty
accumulator. Callers execute each returned simulation step sequentially.

## Box3D world adapter

PhysicsWorld owns one Box3D world and exposes engine math types, validated
body/collider descriptors, and world-unique generation-checked body handles.
The Box3D Foundation is process-wide: the first PhysicsWorld freezes
length_units_per_meter for the process, and later worlds must request the same
scale. A conflicting scale is reported by the backend and is never silently
converted.

The initial adapter supports static, kinematic and dynamic rigid bodies with
sphere or box colliders. Body creation is transactional: if collider creation
fails after body creation, the adapter destroys the partially created body.
If rollback also fails, both errors are returned. Destroying a body invalidates
its handle; reusing a slot increments its generation. Handles from another
PhysicsWorld are rejected.

PhysicsWorld::step advances exactly one configured fixed step and uses the
configured Box3D solver substep count. The separate FixedStepAccumulator
decides how many such calls the runtime should make for a rendered frame.
Bodies can also have their world transform and linear velocity updated through
validated engine-facing methods.

## Current scope boundaries

The adapter does not yet synchronize ECS Transform components, process
collision/sensor events, expose ray/shape queries, or support joints, mesh
colliders, compound colliders, or height fields. These are later Phase 5
increments, not implemented capabilities. Box3D-specific IDs and types remain
private to this crate.

The pinned backend is boxddd 0.4.0, an experimental 0.x safe wrapper around
Box3D. It builds the vendored native C engine for supported desktop targets.
The workspace Rust MSRV is 1.92 to satisfy the backend's documented MSRV.
