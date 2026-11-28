# ADR-0005: Fixed simulation timestep

- Status: Accepted
- Decision: Simulation uses a fixed-timestep accumulator. The default policy
  uses a 16,666,667 ns step and allows at most eight catch-up steps per frame.
  When the limit is reached, excess whole steps are dropped and the fractional
  remainder is retained for presentation interpolation. Pausing clears pending
  accumulated time; frame time while paused is ignored. Interpolated state is
  presentation-only and must not be fed back into simulation.
- Event ordering: the physics-world integration layer must preserve backend
  event order within each completed fixed step. Events from a step are drained
  and delivered after that step completes and before the next fixed step begins.
  Commands produced by those callbacks are deferred to the next fixed-step
  synchronization boundary; callbacks do not mutate an active backend step.
- Consequences: The accumulator reports bounded work, interpolation alpha and
  discarded time. The physics-world integration owns scene synchronization,
  event draining and deferred command application. Tests cover catch-up limits,
  excess-time policy, pause/resume, interpolation remainder and overflow.
