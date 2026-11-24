# ADR-0006: Versioned scene and prefab schemas

- Status: Accepted
- Decision: Scene and prefab formats are versioned and use stable component identifiers.
- Consequences: Deserialization validates schema and references; migrations are explicit and fixture-tested. Unknown or malformed data is never silently discarded.
