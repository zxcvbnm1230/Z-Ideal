# Phase 5 — Physics and Lua

**Status:** in progress. The first increment establishes the fixed-step timing
contract. The Box3D backend and Lua VM/bindings have not yet been integrated.

## Roadmap scope

- Box3D adapter, collider/body components, queries and events.
- Fixed-step simulation.
- Lua VM, script assets, lifecycle, entity/component API.
- Prefab spawn and entity create/destroy commands.

**Exit gate:** every public physics and Lua API path is covered by automated
tests. This gate is not yet met.

## Implemented foundation

### Fixed-step accumulator

- `FixedStepAccumulator` converts frame durations into a bounded count of
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

The accumulator is independent of the physics backend. It does not yet own a
physics world, collider/body handles, scene synchronization, or event queues.

## Backend evaluation

The accepted physics boundary keeps Box3D-specific types inside
`z-ideal-physics`. Candidate evaluation is ongoing:

- `boxddd` provides a safe, ergonomic wrapper over Box3D's C API, including
  owner-scoped IDs and fallible operations. Its current documented native
  support covers Windows, Linux, and macOS; its documented Rust MSRV is 1.92,
  above this workspace's current 1.85 declaration.
- `box3d-rust` is a pure-Rust port and documents a direct C-mirror API rather
  than an ergonomic wrapper. Its determinism claims and API surface require
  integration-level verification before adoption.

No backend dependency has been added yet. Selection requires a reproducible
build and behavior check on all supported desktop targets, a license and
dependency audit, and a deliberate decision about the workspace MSRV.

## Remaining work

- Select and pin a Box3D implementation; isolate it behind backend-neutral
  engine types and validate world/body/shape ownership.
- Add validated rigid-body and collider components, synchronization with
  scene transforms, collision filtering, queries, and ordered event delivery.
- Specify and test physics event ordering at the world/runtime boundary.
- Add the Lua VM, bounded execution and diagnostics, script asset contract,
  lifecycle and pause/reload behavior.
- Bind entity/component access through validated handles and deferred scene
  commands, including prefab spawn and entity create/destroy.
- Cover all public API paths and malformed/stale inputs with automated tests.
