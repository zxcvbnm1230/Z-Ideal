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
sphere or box colliders. Collider filters use category and mask bitsets;
collisions require the Box3D filter rules to accept the pair. Body creation is
transactional: if shape creation fails after native body creation, the adapter
destroys the partially created body. If rollback also fails, both errors are
returned. Destroying a body invalidates its handle; reusing a slot increments
its generation. Handles from another PhysicsWorld are rejected.

PhysicsWorld::step advances exactly one configured fixed step and uses the
configured Box3D solver substep count. The separate FixedStepAccumulator
decides how many such calls the runtime should make for a rendered frame.
Bodies can also have their world transform and linear velocity updated through
validated engine-facing methods.

## Queries and events

PhysicsWorld::cast_ray accepts a world-space origin and a displacement to the
ray endpoint. Query category/mask bits filter candidate shapes. Results contain
engine body handles, world-space hit point and normal, and a normalized hit
fraction. Box3D traversal order is unspecified; the adapter sorts results by
fraction, then by body slot and generation for ties.

ColliderDesc enables contact and sensor begin/end event generation. After each
successful PhysicsWorld::step, events() exposes an owned snapshot for that
step. Contact pairs are canonicalized by body handle; event kinds are ordered
ContactBegin, ContactEnd, SensorBegin, SensorEnd, then by body-handle pair.
Sensor events preserve sensor/visitor roles. Destroyed-body shape identities
are retained through the next step so end events can still map to the original
generation-safe body handle. A subsequent step replaces the event snapshot.

PhysicsScene translates event handles and ray hits to scene Entity handles.
Read scene events after stepping and before the next step replaces the
snapshot. If a body was destroyed during synchronization, its entity handle
is still available in the resulting end-event snapshot, even if the entity
itself has been despawned. Scene ray queries resolve only currently live
physics bindings.

## ECS scene synchronization

The scene crate provides serializable, validated RigidBody and Collider
components. PhysicsScene owns a PhysicsWorld and maps live scene Entity
handles to generation-safe physics handles.

Call synchronize before stepping to create bodies for entities carrying both
components, update static/kinematic poses, recreate bodies when descriptors
change, and destroy bodies after component removal or entity despawn. After
one or more fixed steps, call writeback_dynamic_transforms to copy dynamic
poses into scene Transform components. It does not perform presentation
interpolation.

Current transform contract: physics entities must be root entities and have
unit local scale. Parent transforms, shear, and scaled colliders are rejected
rather than approximated. Dynamic scene transforms are written in world space
because parented physics entities are not yet supported. Scene entities with
only one of RigidBody or Collider do not participate.

## Current scope boundaries

Shape casts, overlap queries, joints, mesh colliders, compound colliders,
height fields, parented physics transforms and collider scaling are not
implemented. These are later Phase 5 increments, not implemented capabilities.
Box3D-specific IDs and types remain private to this crate.

The pinned backend is boxddd 0.4.0, an experimental 0.x safe wrapper around
Box3D. It builds the vendored native C engine for supported desktop targets.
The workspace Rust MSRV is 1.92 to satisfy the backend's documented MSRV.
