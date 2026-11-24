//! Versioned scene document schema and validation.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Current serialized scene document schema version.
pub const SCENE_SCHEMA_VERSION: u32 = 1;

/// Stable serialized entity identifier, distinct from a runtime Entity handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SceneEntityId(u64);

impl SceneEntityId {
    /// Creates a non-zero persistent entity identifier.
    pub fn new(value: u64) -> Result<Self, SceneError> {
        if value == 0 || value == u64::MAX {
            return Err(SceneError::InvalidSceneEntityId(value));
        }
        Ok(Self(value))
    }

    pub(crate) const fn new_unchecked(value: u64) -> Self {
        Self(value)
    }

    /// Returns the numeric serialized identifier.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Versioned component payload stored in a scene document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SerializedComponent {
    /// Component schema version at the time this data was written.
    pub schema_version: u32,
    /// Component-specific serialized payload.
    pub data: Value,
}

/// One entity and its serialized components.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneEntityRecord {
    /// Stable identifier within this scene document.
    pub id: SceneEntityId,
    /// Stable identifier of the parent, if any.
    pub parent: Option<SceneEntityId>,
    /// Components keyed by their stable registered component IDs.
    pub components: BTreeMap<String, SerializedComponent>,
}

impl SceneEntityRecord {
    /// Creates an entity record with no components.
    #[must_use]
    pub fn new(id: SceneEntityId, parent: Option<SceneEntityId>) -> Self {
        Self {
            id,
            parent,
            components: BTreeMap::new(),
        }
    }
}

/// Versioned, deterministic scene document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneDocument {
    /// Scene schema version.
    pub schema_version: u32,
    /// Entities in stable serialized-ID order.
    pub entities: Vec<SceneEntityRecord>,
}

impl SceneDocument {
    /// Creates an empty document using the current schema version.
    #[must_use]
    pub fn new() -> Self {
        Self {
            schema_version: SCENE_SCHEMA_VERSION,
            entities: Vec::new(),
        }
    }

    /// Validates document-level schema, IDs, parent references and component metadata.
    pub fn validate(&self) -> Result<(), SceneError> {
        if self.schema_version != SCENE_SCHEMA_VERSION {
            return Err(SceneError::UnsupportedSceneVersion(self.schema_version));
        }

        let mut ids = HashSet::with_capacity(self.entities.len());
        for entity in &self.entities {
            let id = entity.id.get();
            if id == 0 || id == u64::MAX {
                return Err(SceneError::InvalidSceneEntityId(id));
            }
            if !ids.insert(entity.id) {
                return Err(SceneError::DuplicateSceneEntityId(entity.id));
            }
            if entity.parent == Some(entity.id) {
                return Err(SceneError::SelfParent(entity.id));
            }
            for (component_id, component) in &entity.components {
                if component_id.trim().is_empty() || component.schema_version == 0 {
                    return Err(SceneError::InvalidComponentMetadata {
                        component_id: component_id.clone(),
                    });
                }
            }
        }

        for entity in &self.entities {
            if let Some(parent) = entity.parent {
                if !ids.contains(&parent) {
                    return Err(SceneError::MissingParent {
                        entity: entity.id,
                        parent,
                    });
                }
            }
        }
        Ok(())
    }
}

impl Default for SceneDocument {
    fn default() -> Self {
        Self::new()
    }
}

/// Scene serialization, migration and validation failures.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SceneError {
    /// The document schema version is not supported.
    #[error("unsupported scene schema version {0}; supported version is 1")]
    UnsupportedSceneVersion(u32),
    /// Persistent entity IDs must be non-zero and leave room for future allocation.
    #[error("invalid persistent scene entity ID: {0}")]
    InvalidSceneEntityId(u64),
    /// A persistent entity ID occurs more than once.
    #[error("duplicate persistent scene entity ID: {0:?}")]
    DuplicateSceneEntityId(SceneEntityId),
    /// A parent ID does not identify an entity in this document.
    #[error("entity {entity:?} refers to missing parent {parent:?}")]
    MissingParent {
        /// Entity declaring the parent.
        entity: SceneEntityId,
        /// Parent ID that was not found.
        parent: SceneEntityId,
    },
    /// An entity cannot parent itself.
    #[error("entity cannot parent itself: {0:?}")]
    SelfParent(SceneEntityId),
    /// A component ID or schema version is invalid.
    #[error("invalid component metadata for ID {component_id:?}")]
    InvalidComponentMetadata {
        /// Invalid component identifier.
        component_id: String,
    },
    /// The document references a component not registered by the target world.
    #[error("scene references an unregistered component ID: {0}")]
    UnknownComponent(String),
    /// A component migration failed.
    #[error("migration for component {component_id} from schema {from_version} to {to_version} failed: {message}")]
    ComponentMigration {
        /// Stable component identifier.
        component_id: &'static str,
        /// Serialized schema version.
        from_version: u32,
        /// Current component schema version.
        to_version: u32,
        /// Migration failure detail.
        message: String,
    },
    /// Component data could not be deserialized or validated.
    #[error("invalid data for component {component_id}: {message}")]
    InvalidComponentData {
        /// Stable component identifier.
        component_id: &'static str,
        /// Deserialization failure detail.
        message: String,
    },
    /// Component data could not be serialized.
    #[error("failed to serialize component {component_id}: {message}")]
    ComponentSerialization {
        /// Stable component identifier.
        component_id: String,
        /// Serialization failure detail.
        message: String,
    },
    /// Persistent scene ID allocation is exhausted.
    #[error("persistent scene entity ID space is exhausted")]
    SceneEntityIdExhausted,
    /// JSON encoding or decoding failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// The world rejected a scene operation.
    #[error(transparent)]
    World(#[from] crate::WorldError),
}
