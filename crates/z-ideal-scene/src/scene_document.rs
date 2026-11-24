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



/// Serializable reference to another entity in a scene.
///
/// The reserved JSON marker is remapped when a prefab is instantiated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SceneEntityRef {
    #[serde(rename = "$entity_ref")]
    id: SceneEntityId,
}

impl SceneEntityRef {
    /// Creates a reference to a persistent scene entity ID.
    #[must_use]
    pub const fn new(id: SceneEntityId) -> Self {
        Self { id }
    }

    /// Returns the referenced persistent scene entity ID.
    #[must_use]
    pub const fn id(self) -> SceneEntityId {
        self.id
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
    /// Next persistent entity ID to allocate, preserving deleted-ID gaps.
    pub next_scene_entity_id: u64,
    /// Entities in stable serialized-ID order.
    pub entities: Vec<SceneEntityRecord>,
}

impl SceneDocument {
    /// Creates an empty document using the current schema version.
    #[must_use]
    pub fn new() -> Self {
        Self {
            schema_version: SCENE_SCHEMA_VERSION,
            next_scene_entity_id: 1,
            entities: Vec::new(),
        }
    }

    /// Validates document-level schema, IDs, parent references and component metadata.
    pub fn validate(&self) -> Result<(), SceneError> {
        if self.schema_version != SCENE_SCHEMA_VERSION {
            return Err(SceneError::UnsupportedSceneVersion(self.schema_version));
        }

        let maximum_id = self
            .entities
            .iter()
            .map(|entity| entity.id.get())
            .max()
            .unwrap_or(0);
        if self.next_scene_entity_id == 0 || self.next_scene_entity_id <= maximum_id {
            return Err(SceneError::NextSceneEntityIdNotAhead {
                next: self.next_scene_entity_id,
                maximum: maximum_id,
            });
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

    /// Validates that every serialized entity reference resolves within this scene.
    pub fn validate_entity_references(&self) -> Result<(), SceneError> {
        self.validate()?;
        let ids: HashSet<_> = self.entities.iter().map(|entity| entity.id).collect();
        for entity in &self.entities {
            for (component_id, component) in &entity.components {
                let mut references = Vec::new();
                collect_entity_references(&component.data, &mut references)?;
                for target in references {
                    if !ids.contains(&target) {
                        return Err(SceneError::MissingEntityReference {
                            entity: entity.id,
                            component_id: component_id.clone(),
                            target,
                        });
                    }
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



pub(crate) fn remap_entity_references(
    value: &mut Value,
    remap: &std::collections::HashMap<SceneEntityId, SceneEntityId>,
) -> Result<(), SceneError> {
    match value {
        Value::Object(object) => {
            if let Some(reference) = object.get("$entity_ref") {
                if object.len() != 1 {
                    return Err(SceneError::InvalidEntityReference(
                        "the reserved $entity_ref marker must be the only object field".to_owned(),
                    ));
                }
                let raw = reference.as_u64().ok_or_else(|| {
                    SceneError::InvalidEntityReference(
                        "$entity_ref must contain an unsigned integer".to_owned(),
                    )
                })?;
                let old_id = SceneEntityId::new(raw)?;
                if let Some(new_id) = remap.get(&old_id) {
                    object.insert("$entity_ref".to_owned(), Value::from(new_id.get()));
                }
                Ok(())
            } else {
                for child in object.values_mut() {
                    remap_entity_references(child, remap)?;
                }
                Ok(())
            }
        }
        Value::Array(values) => {
            for child in values {
                remap_entity_references(child, remap)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn collect_entity_references(
    value: &Value,
    references: &mut Vec<SceneEntityId>,
) -> Result<(), SceneError> {
    match value {
        Value::Object(object) => {
            if let Some(reference) = object.get("$entity_ref") {
                if object.len() != 1 {
                    return Err(SceneError::InvalidEntityReference(
                        "the reserved $entity_ref marker must be the only object field".to_owned(),
                    ));
                }
                let raw = reference.as_u64().ok_or_else(|| {
                    SceneError::InvalidEntityReference(
                        "$entity_ref must contain an unsigned integer".to_owned(),
                    )
                })?;
                references.push(SceneEntityId::new(raw)?);
            } else {
                for child in object.values() {
                    collect_entity_references(child, references)?;
                }
            }
            Ok(())
        }
        Value::Array(values) => {
            for child in values {
                collect_entity_references(child, references)?;
            }
            Ok(())
        }
        _ => Ok(()),
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
    #[error(
        "migration for component {component_id} from schema {from_version} to {to_version} failed: {message}"
    )]
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
    /// The next persistent entity ID is not greater than all IDs in the document.
    #[error("next scene entity ID {next} must be greater than maximum existing ID {maximum}")]
    NextSceneEntityIdNotAhead {
        /// Next ID to allocate.
        next: u64,
        /// Largest ID present in the document.
        maximum: u64,
    },
    /// A reserved serialized entity-reference marker is malformed.
    #[error("invalid serialized entity reference: {0}")]
    InvalidEntityReference(String),
    /// A component references an entity that is absent from the scene.
    #[error("entity {entity:?} component {component_id} references missing entity {target:?}")]
    MissingEntityReference {
        /// Entity containing the component.
        entity: SceneEntityId,
        /// Stable component identifier.
        component_id: String,
        /// Referenced entity ID that was not found.
        target: SceneEntityId,
    },
    /// Persistent scene ID allocation is exhausted.
    #[error("persistent scene entity ID space is exhausted")]
    SceneEntityIdExhausted,
    /// JSON encoding or decoding failed.
    #[error("invalid scene JSON: {0}")]
    Json(String),
    /// The world rejected a scene operation.
    #[error(transparent)]
    World(#[from] crate::WorldError),
}

impl From<serde_json::Error> for SceneError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};
    use serde_json::{Value, json};

    use super::{
        SCENE_SCHEMA_VERSION, SceneDocument, SceneEntityId, SceneEntityRecord, SceneError,
        SerializedComponent,
    };
    use crate::{Component, World};
    use z_ideal_math::{Quat, Transform, Vec3};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Health {
        current: u32,
    }

    impl Component for Health {
        const ID: &'static str = "test.scene_health";
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct MigratingValue {
        value: u32,
    }

    impl Component for MigratingValue {
        const ID: &'static str = "test.migrating_value";
        const VERSION: u32 = 2;

        fn migrate(from_version: u32, data: Value) -> Result<Value, String> {
            match from_version {
                1 => {
                    let old_value = data
                        .get("legacy_value")
                        .and_then(Value::as_u64)
                        .ok_or_else(|| "legacy_value must be an unsigned integer".to_owned())?;
                    Ok(json!({ "value": old_value }))
                }
                2 => Ok(data),
                _ => Err(format!(
                    "unsupported test migration source version {from_version}"
                )),
            }
        }
    }

    fn test_id(value: u64) -> SceneEntityId {
        SceneEntityId::new(value).expect("valid scene entity ID")
    }

    fn world_with_components() -> World {
        let mut world = World::new();
        world
            .register_component::<Health>()
            .expect("register Health");
        world
            .register_component::<Transform>()
            .expect("register Transform");
        world
    }

    #[test]
    fn scene_json_round_trip_preserves_ids_components_and_hierarchy() {
        let mut source = world_with_components();
        let parent = source.spawn().expect("spawn parent");
        let child = source.spawn().expect("spawn child");
        source
            .insert(parent, Health { current: 100 })
            .expect("insert parent health");
        source
            .insert(child, Health { current: 40 })
            .expect("insert child health");
        source
            .insert(
                parent,
                Transform::new(Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE)
                    .expect("parent transform"),
            )
            .expect("insert parent transform");
        source
            .set_parent(child, Some(parent))
            .expect("set child parent");

        let parent_id = source.scene_entity_id(parent).expect("parent scene ID");
        let child_id = source.scene_entity_id(child).expect("child scene ID");
        let first_json = source.to_scene_json().expect("serialize source");
        assert_eq!(first_json, source.to_scene_json().expect("serialize again"));

        let mut loaded = world_with_components();
        loaded.load_scene_json(&first_json).expect("load scene");
        let loaded_parent = loaded
            .entity_for_scene_id(parent_id)
            .expect("resolve parent ID");
        let loaded_child = loaded
            .entity_for_scene_id(child_id)
            .expect("resolve child ID");

        assert_eq!(
            loaded.get::<Health>(loaded_parent),
            Some(&Health { current: 100 })
        );
        assert_eq!(
            loaded.get::<Health>(loaded_child),
            Some(&Health { current: 40 })
        );
        assert_eq!(loaded.parent_of(loaded_child), Some(loaded_parent));
        assert_eq!(
            loaded
                .world_transform(loaded_child)
                .expect("child world transform")
                .transform_point3(Vec3::ZERO),
            Vec3::new(3.0, 0.0, 0.0)
        );
    }

    #[test]
    fn scene_load_is_transactional_when_component_is_unknown() {
        let mut world = World::new();
        world
            .register_component::<Health>()
            .expect("register Health");
        let existing = world.spawn().expect("spawn existing");
        world
            .insert(existing, Health { current: 88 })
            .expect("insert existing health");

        let mut record = SceneEntityRecord::new(test_id(10), None);
        record.components.insert(
            "unknown.component".to_owned(),
            SerializedComponent {
                schema_version: 1,
                data: json!({ "value": 1 }),
            },
        );
        let document = SceneDocument {
            schema_version: SCENE_SCHEMA_VERSION,
            next_scene_entity_id: 11,
            entities: vec![record],
        };

        assert_eq!(
            world.load_scene_document(&document),
            Err(SceneError::UnknownComponent("unknown.component".to_owned()))
        );
        assert!(world.is_alive(existing));
        assert_eq!(world.get::<Health>(existing), Some(&Health { current: 88 }));
    }

    #[test]
    fn component_schema_migration_is_explicit_and_applied() {
        let mut world = World::new();
        world
            .register_component::<MigratingValue>()
            .expect("register migrating component");
        let mut record = SceneEntityRecord::new(test_id(7), None);
        record.components.insert(
            MigratingValue::ID.to_owned(),
            SerializedComponent {
                schema_version: 1,
                data: json!({ "legacy_value": 73 }),
            },
        );
        let document = SceneDocument {
            schema_version: SCENE_SCHEMA_VERSION,
            next_scene_entity_id: 8,
            entities: vec![record],
        };

        world
            .load_scene_document(&document)
            .expect("migrate old component");
        let entity = world
            .entity_for_scene_id(test_id(7))
            .expect("resolve migrated entity");
        assert_eq!(
            world.get::<MigratingValue>(entity),
            Some(&MigratingValue { value: 73 })
        );
    }

    #[test]
    fn default_component_migration_rejects_unknown_versions() {
        let mut world = World::new();
        world
            .register_component::<Health>()
            .expect("register Health");
        let mut record = SceneEntityRecord::new(test_id(1), None);
        record.components.insert(
            Health::ID.to_owned(),
            SerializedComponent {
                schema_version: 9,
                data: json!({ "current": 5 }),
            },
        );
        let document = SceneDocument {
            schema_version: SCENE_SCHEMA_VERSION,
            next_scene_entity_id: 2,
            entities: vec![record],
        };

        assert!(matches!(
            world.load_scene_document(&document),
            Err(SceneError::ComponentMigration {
                component_id: "test.scene_health",
                from_version: 9,
                to_version: 1,
                ..
            })
        ));
    }

    #[test]
    fn component_payload_rejects_unknown_fields() {
        let mut world = World::new();
        world
            .register_component::<Health>()
            .expect("register Health");
        let mut record = SceneEntityRecord::new(test_id(1), None);
        record.components.insert(
            Health::ID.to_owned(),
            SerializedComponent {
                schema_version: 1,
                data: json!({ "current": 5, "unexpected": true }),
            },
        );
        let document = SceneDocument {
            schema_version: SCENE_SCHEMA_VERSION,
            next_scene_entity_id: 2,
            entities: vec![record],
        };

        assert!(matches!(
            world.load_scene_document(&document),
            Err(SceneError::InvalidComponentData {
                component_id: "test.scene_health",
                ..
            })
        ));
    }

    #[test]
    fn document_validation_rejects_duplicate_and_missing_parent_ids() {
        let duplicate = SceneDocument {
            schema_version: SCENE_SCHEMA_VERSION,
            next_scene_entity_id: 4,
            entities: vec![
                SceneEntityRecord::new(test_id(3), None),
                SceneEntityRecord::new(test_id(3), None),
            ],
        };
        assert_eq!(
            duplicate.validate(),
            Err(SceneError::DuplicateSceneEntityId(test_id(3)))
        );

        let missing_parent = SceneDocument {
            schema_version: SCENE_SCHEMA_VERSION,
            next_scene_entity_id: 4,
            entities: vec![SceneEntityRecord::new(test_id(3), Some(test_id(99)))],
        };
        assert_eq!(
            missing_parent.validate(),
            Err(SceneError::MissingParent {
                entity: test_id(3),
                parent: test_id(99),
            })
        );
    }

    #[test]
    fn scene_schema_version_is_checked() {
        let document = SceneDocument {
            schema_version: SCENE_SCHEMA_VERSION + 1,
            next_scene_entity_id: 1,
            entities: Vec::new(),
        };
        assert_eq!(
            document.validate(),
            Err(SceneError::UnsupportedSceneVersion(
                SCENE_SCHEMA_VERSION + 1
            ))
        );
    }

    #[test]
    fn scene_document_preserves_deleted_entity_id_gaps() {
        let mut source = World::new();
        let retained = source.spawn().expect("spawn retained entity");
        let deleted = source.spawn().expect("spawn entity to delete");
        source.despawn(deleted).expect("despawn entity");

        let document = source.to_scene_document().expect("serialize source");
        assert_eq!(document.next_scene_entity_id, 3);

        let mut restored = World::new();
        restored
            .load_scene_document(&document)
            .expect("restore source");
        let new_entity = restored.spawn().expect("spawn after restore");
        assert_eq!(
            restored.scene_entity_id(new_entity),
            SceneEntityId::new(3)
        );
        let retained_id = source.scene_entity_id(retained).expect("retained scene ID");
        assert!(restored.entity_for_scene_id(retained_id).is_some());
    }

    #[test]
    fn malformed_json_and_unknown_document_fields_are_rejected() {
        let mut world = World::new();
        assert!(matches!(
            world.load_scene_json(r#"{"schema_version":1,"entities":[],"unexpected":true}"#),
            Err(SceneError::Json(_))
        ));
        assert!(matches!(
            world.load_scene_json("not-json"),
            Err(SceneError::Json(_))
        ));
    }
}
