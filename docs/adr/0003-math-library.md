# ADR-0003: Math library evaluation

- Status: Accepted for evaluation; final dependency selection deferred.
- Decision: Evaluate the Rust `glm` crate first for a Rust-native GLM-style API. Avoid C++ GLM FFI unless a concrete requirement justifies it.
- Consequences: Keep `z-ideal-math` as the authoritative boundary and document conventions and conversions.
