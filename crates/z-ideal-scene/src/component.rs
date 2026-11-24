//! Authoritative component identity and validation contract.

use std::any::TypeId;

/// A type stored in a World.
///
/// Every component must provide a globally stable serialized identifier.
/// The identifier is independent of its Rust type name and module path.
pub trait Component: Send + Sync + 'static {
    /// Stable schema identifier used by scene serialization and tooling.
    const ID: &'static str;

    /// Component schema version. Increment when its serialized representation changes.
    const VERSION: u32 = 1;

    /// Validates component invariants before insertion into a world.
    fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

/// Public, inspectable registration metadata for a component type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentInfo {
    /// Stable serialized identifier.
    pub id: &'static str,
    /// Component schema version.
    pub version: u32,
    pub(crate) type_id: TypeId,
}

impl ComponentInfo {
    pub(crate) fn new(id: &'static str, version: u32, type_id: TypeId) -> Self {
        Self {
            id,
            version,
            type_id,
        }
    }

    /// Returns the Rust type identity for registry checks.
    #[must_use]
    pub const fn type_id(&self) -> TypeId {
        self.type_id
    }
}
