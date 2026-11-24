# ADR-0001: Cargo workspace and crate boundaries

- Status: Accepted
- Decision: Use a Cargo workspace with reusable subsystem crates, separate editor/runtime libraries, and separate editor/runtime executables.
- Rationale: Enforces dependency boundaries and allows runtime builds to exclude editor code.
- Consequences: Dependencies must remain acyclic; runtime must never depend on editor; test-support is test-only.
