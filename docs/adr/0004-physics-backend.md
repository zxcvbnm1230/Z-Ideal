# ADR-0004: Box3D integration boundary

- Status: Accepted
- Decision: Keep all Box3D-specific types behind `z-ideal-physics`. Evaluate safe binding/pure Rust candidates `boxddd` and `box3d-rust` for supported targets, API quality, licensing, maintenance, determinism and tests.
- Consequences: Pin exact versions/commits while dependencies remain 0.x. Do not expose backend types through public engine APIs.
