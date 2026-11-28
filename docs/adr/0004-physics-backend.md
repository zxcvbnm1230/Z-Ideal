# ADR-0004: Box3D integration boundary

- Status: Accepted
- Decision: Keep all Box3D-specific types behind z-ideal-physics. Pin
  boxddd 0.4.0 exactly and use its safe Rust API for the native Box3D backend.
  Do not expose Box3D IDs, definitions, or native handles through engine APIs.
- Rationale: boxddd provides engine-agnostic worlds, bodies, shapes, queries,
  events and owned value types, with opaque owner-scoped IDs and fallible safe
  operations. Its native desktop support matrix covers Windows, Linux and
  macOS. The competing box3d-rust crate provides a pure-Rust port with a
  direct C-mirror API; the current integration favors the safer ergonomic
  wrapper boundary rather than adopting that API surface directly.
- Compatibility: boxddd 0.4.0 documents Rust 1.92 as its MSRV. The workspace
  rust-version is therefore raised from 1.85 to 1.92. The backend and its
  vendored Box3D source are MIT OR Apache-2.0 / MIT licensed, respectively.
- Consequences: Pin exact versions while dependencies remain 0.x. Audit native
  build behavior in Windows, macOS and Ubuntu CI. Box3D Foundation configuration
  is process-wide and immutable after initialization; all physics worlds in a
  process must use the same length_units_per_meter. The backend is experimental
  and remains subject to regression testing and dependency review.
