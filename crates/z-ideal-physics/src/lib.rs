//! Engine-facing physics API and isolated Box3D adapter.
//!
//! Backend-specific types remain private to this crate. Applications interact
//! through validated descriptors, world-unique handles and engine math types.
#![forbid(unsafe_code)]

mod fixed_step;
mod world;

pub use fixed_step::{
    FixedStepAccumulator, FixedStepConfig, FixedStepConfigError, FixedStepError, FixedStepPlan,
};
pub use world::{
    ColliderDesc, ColliderShape, PhysicsBodyHandle, PhysicsBodyState, PhysicsBodyType,
    PhysicsConfigError, PhysicsError, PhysicsWorld, PhysicsWorldConfig, RigidBodyDesc,
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
