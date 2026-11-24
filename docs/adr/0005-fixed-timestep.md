# ADR-0005: Fixed simulation timestep

- Status: Accepted
- Decision: Simulation uses a fixed-timestep accumulator.
- Consequences: Before implementation, specify maximum catch-up steps, excess-time policy, pause behavior, interpolation and event ordering; cover them with deterministic tests.
