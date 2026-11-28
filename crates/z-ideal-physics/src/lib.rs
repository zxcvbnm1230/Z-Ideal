//! Engine-facing physics API and isolated Box3D adapter.
//!
//! The fixed-step accumulator is backend-independent and defines the timing
//! contract used by physics-world integrations. Backend-specific types remain
//! private to this crate.
#![forbid(unsafe_code)]

mod fixed_step;

pub use fixed_step::{
    FixedStepAccumulator, FixedStepConfig, FixedStepConfigError, FixedStepError, FixedStepPlan,
};

/// Returns the crate's stable package name for diagnostics and tooling.
pub const CRATE_NAME: &str = "z-ideal-physics";

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name_is_not_empty() {
        assert!(!super::CRATE_NAME.is_empty());
    }
}
