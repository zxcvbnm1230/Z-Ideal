# ADR-0003: Math library selection

- Status: Accepted
- Decision: Use the Rust-native glam crate behind the z-ideal-math crate boundary. Pin the evaluated release exactly in the workspace manifest until the dependency reaches 1.0.
- Evaluation: The requested Rust glm crate (glm-rs, version 0.3.0) provides a GLSL-like API, but its own release notes list quaternion support and SIMD support as future plans, and its docs.rs page reports 52.73% item documentation. Those gaps matter for a 3D engine whose transform system requires quaternions. glam 0.34.3 provides vectors, matrices, quaternions, affine transforms, and SIMD implementations for x86/x86_64 and AArch64, and documents Rust 1.85 as its MSRV, matching this workspace's declared minimum.
- Consequences: Engine code imports math types through z-ideal-math, not directly from glam. Coordinate, handedness, up/forward axis, angle, and matrix composition conventions are documented in that crate. Renderer and physics integrations must convert at their boundaries.
- References: https://docs.rs/glm/latest/glm/, https://docs.rs/crate/glm/latest, https://docs.rs/glam/latest/glam/, https://docs.rs/crate/glam/latest.
