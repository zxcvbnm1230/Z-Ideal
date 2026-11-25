//! Authoritative component identity, validation and migration contract.

use std::any::TypeId;

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

/// A serializable type stored in a World.
///
/// Every persistent component provides a globally stable identifier and a
/// schema version independent of its Rust type name and module path. Use
/// serde's deny_unknown_fields on component structs so unexpected fields are
/// rejected rather than silently ignored.
pub trait Component: Serialize + DeserializeOwned + Send + Sync + 'static {
    /// Stable schema identifier used by scene serialization and tooling.
    const ID: &'static str;

    /// Component schema version. Increment when its serialized representation changes.
    const VERSION: u32 = 1;

    /// Validates component invariants before insertion into a world.
    fn validate(&self) -> Result<(), String> {
        Ok(())
    }

    /// Migrates serialized data from an older schema version.
    ///
    /// The default accepts only the current version. Components that change
    /// their serialized representation must implement explicit migrations.
    fn migrate(from_version: u32, data: Value) -> Result<Value, String> {
        if from_version == Self::VERSION {
            Ok(data)
        } else {
            Err(format!(
                "no migration is registered from schema version {from_version}"
            ))
        }
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
