# Phase 1 — Core, Math, ECS and Scenes

## Scope
Phase 1 establishes engine math conventions, generation-safe entities, authoritative component registration, component storage, transform hierarchy, deterministic scheduling, versioned scene serialization and prefab instantiation.

## Math conventions
- Right-handed world coordinates.
- +Y is up and -Z is forward.
- Angles are radians.
- Matrices are column-major and compose for column vectors; child world matrix = parent world matrix * child local matrix.
- The engine-facing API is exposed through z-ideal-math; backend crates must convert at subsystem boundaries.

## Entity and component contracts
- An Entity is a world-local (slot index, generation) handle. It is not a persistent serialized identifier.
- A destroyed slot increments its generation before reuse. If its generation counter is exhausted, the slot is retired rather than wrapping.
- Component types register one stable ID and non-zero schema version with each world. Duplicate stable IDs are rejected.
- Component storage is sparse-set based. Structural mutation is owned by World; system-driven structural changes are queued in Commands and applied at system synchronization points.
- Component insertion validates component invariants.

## Transform hierarchy
- Parent relationships are stored by the owning world.
- Reparenting validates both handles and rejects self-parenting and cycles before changing existing edges.
- Destroying a parent detaches its children; it does not recursively destroy them.
- Missing local Transform components are treated as identity for world-matrix composition.

## Deterministic scheduling
- Systems declare dependencies by stable system name.
- The scheduler topologically sorts dependencies; registration order breaks ties between simultaneously ready systems.
- Systems run serially and read the World immutably.
- Structural changes are queued in a per-system command buffer and applied FIFO after that system succeeds, before the next system begins.
- A failed system discards its own command buffer. Command application is sequential, not transactional; if a command fails, earlier commands in that buffer remain applied.

## Phase completion requirements
- Versioned scene documents, stable serialized component IDs, explicit migrations, malformed-input rejection and round-trip fixtures.
- Prefab instantiation with entity-reference remapping and explicit override semantics.
- Unit, integration, fixture, and cross-platform CI coverage for these contracts.

## Implementation status
Implemented in this Phase 1 slice: math conventions and validated Transform, generation-safe entity lifecycle, registered sparse-set component storage, cycle-safe hierarchy operations, deterministic dependency-aware scheduling, and deferred command buffers. Versioned scene serialization/migrations and prefab workflows remain incomplete and are not represented as implemented.
