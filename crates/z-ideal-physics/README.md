# z-ideal-physics

Engine-facing physics contracts and the isolated Box3D adapter. Backend types
are not part of the engine-facing public API.

## Fixed-step timing

`FixedStepAccumulator` converts variable frame durations into a bounded number
of fixed simulation steps. The default policy uses a 16,666,667 ns step and a
maximum of eight catch-up steps per frame. When that limit is reached, excess
whole steps are dropped and the fractional remainder is retained. The returned
interpolation alpha is for presentation only; interpolated transforms must not
be written back into simulation state.

Pausing clears pending accumulated time, and frame durations supplied while
paused are ignored and reported as `dropped_time`. Resuming starts with an empty
accumulator. Callers execute each returned simulation step sequentially. Physics
event delivery order is defined by the world integration layer, not by this
time accumulator.

The default 60 Hz interval is represented to the nearest nanosecond, so it is
not an exact rational 1/60 second. Applications requiring a different rate or
catch-up budget must construct an explicit `FixedStepConfig`.
