//! Foundational engine services: configuration, diagnostics and build metadata.
#![forbid(unsafe_code)]

pub mod build_info;
pub mod config;
pub mod logging;

pub use config::EngineConfig;

/// Stable package identifier for diagnostics and tooling.
pub const CRATE_NAME: &str = "z-ideal-core";
