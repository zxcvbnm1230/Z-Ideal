# Testing and Quality Strategy

## Test pyramid
- Unit tests for algorithms, validation, state machines and error paths.
- Integration tests across crate boundaries and lifecycle transitions.
- Contract tests for serialization, scripting APIs and subsystem adapters.
- Versioned fixtures, including the planned **Z Ideal Test World**, covering components, assets, materials, lights, physics, audio, cameras, prefabs and Lua API.
- Offscreen rendering and physics integration tests where supported.
- Platform smoke tests on Windows, macOS and Ubuntu 24.04+.
- Clean-room exported-game launch tests before release.

## Required CI gates
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-features`
- Documentation build with warnings denied.
- Dependency/license/security audit as dependency inventory becomes established.
- OS matrix; GPU validation where capable runners exist.
- Clean-room export validation in release phase.

## Reporting
Every task reports exact commands, pass/fail outcomes, skipped tests and reasons, platform coverage, and known limitations. A test is never weakened or removed to conceal a defect.
