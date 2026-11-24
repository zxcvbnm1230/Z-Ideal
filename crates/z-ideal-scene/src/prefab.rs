//! Prefab documents, entity remapping and explicit component overrides.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{
    SceneDocument, SceneEntityId, SceneEntityRecord, SceneError, SerializedComponent, World,
    scene_document::remap_entity_references,
};

#[derive(Debug, Clone)]
enum ComponentOverride {
    Set(SerializedComponent),
    Remove,
}

/// Explicit per-entity component overrides applied when instantiating a prefab.
#[derive(Debug, Clone, Default)]
pub struct PrefabOverrides {
    components: BTreeMap<SceneEntityId, BTreeMap<String, ComponentOverride>>,
}

impl PrefabOverrides {
    /// Adds or replaces a component on a prefab entity.
    pub fn set_component(
        &mut self,
        entity: SceneEntityId,
        component_id: impl Into<String>,
        component: SerializedComponent,
    ) -> Result<(), SceneError> {
        let component_id = component_id.into();
        if component_id.trim().is_empty() || component.schema_version == 0 {
            return Err(SceneError::InvalidComponentMetadata { component_id });
        }
        self.components
            .entry(entity)
            .or_default()
            .insert(component_id, ComponentOverride::Set(component));
        Ok(())
    }

    /// Removes a component from a prefab entity in the instantiated copy.
    pub fn remove_component(&mut self, entity: SceneEntityId, component_id: impl Into<String>) {
        self.components
            .entry(entity)
            .or_default()
            .insert(component_id.into(), ComponentOverride::Remove);
    }
}

/// Reusable, validated scene fragment.
#[derive(Debug, Clone, PartialEq)]
pub struct Prefab {
    document: SceneDocument,
}

impl Prefab {
    /// Creates a prefab from a scene document.
    ///
    /// Entity references may target entities outside the prefab; those
    /// references are validated against the destination world at instantiation.
    pub fn new(document: SceneDocument) -> Result<Self, SceneError> {
        document.validate()?;
        Ok(Self { document })
    }

    /// Returns the source prefab document.
    #[must_use]
    pub const fn document(&self) -> &SceneDocument {
        &self.document
    }

    /// Instantiates the prefab without component overrides.
    pub fn instantiate(&self, world: &mut World) -> Result<PrefabInstance, SceneError> {
        self.instantiate_with_overrides(world, &PrefabOverrides::default())
    }

    /// Instantiates the prefab into an existing world.
    ///
    /// The operation is transactional: the destination world is replaced only
    /// after the merged scene, remapped references, component payloads and
    /// hierarchy have all passed validation and deserialization.
    pub fn instantiate_with_overrides(
        &self,
        world: &mut World,
        overrides: &PrefabOverrides,
    ) -> Result<PrefabInstance, SceneError> {
        self.document.validate()?;
        let source_ids: HashSet<_> = self
            .document
            .entities
            .iter()
            .map(|entity| entity.id)
            .collect();
        for entity in overrides.components.keys() {
            if !source_ids.contains(entity) {
                return Err(SceneError::UnknownPrefabEntity(*entity));
            }
        }

        let mut merged = world.to_scene_document()?;
        let mut next_id = world.next_scene_entity_id();
        let mut remap = HashMap::with_capacity(self.document.entities.len());

        for source in &self.document.entities {
            let instance_id = SceneEntityId::new(next_id)?;
            next_id = next_id
                .checked_add(1)
                .ok_or(SceneError::SceneEntityIdExhausted)?;
            remap.insert(source.id, instance_id);
        }

        for source in &self.document.entities {
            let mut record = SceneEntityRecord::new(
                remap[&source.id],
                source.parent.map(|parent| remap[&parent]),
            );
            record.components = source.components.clone();

            for component in record.components.values_mut() {
                remap_entity_references(&mut component.data, &remap)?;
            }

            if let Some(entity_overrides) = overrides.components.get(&source.id) {
                for (component_id, component_override) in entity_overrides {
                    match component_override {
                        ComponentOverride::Set(component) => {
                            let mut component = component.clone();
                            remap_entity_references(&mut component.data, &remap)?;
                            record.components.insert(component_id.clone(), component);
                        }
                        ComponentOverride::Remove => {
                            record.components.remove(component_id);
                        }
                    }
                }
            }
            merged.entities.push(record);
        }

        merged.next_scene_entity_id = next_id;
        merged.entities.sort_by_key(|entity| entity.id);
        merged.validate()?;
        world.load_scene_document(&merged)?;

        let mut instance_entities = BTreeMap::new();
        for (source_id, instance_id) in remap {
            let entity = world
                .entity_for_scene_id(instance_id)
                .ok_or(SceneError::PrefabInstanceEntityMissing(source_id))?;
            instance_entities.insert(source_id, entity);
        }
        Ok(PrefabInstance {
            entities: instance_entities,
        })
    }
}

/// Mapping from source prefab IDs to runtime entities in one instantiated copy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefabInstance {
    entities: BTreeMap<SceneEntityId, crate::Entity>,
}

impl PrefabInstance {
    /// Resolves a source prefab ID to its instantiated runtime entity.
    #[must_use]
    pub fn entity(&self, source_id: SceneEntityId) -> Option<crate::Entity> {
        self.entities.get(&source_id).copied()
    }

    /// Iterates source prefab IDs and their instantiated runtime entities in ID order.
    pub fn entities(&self) -> impl Iterator<Item = (SceneEntityId, crate::Entity)> + '_ {
        self.entities
            .iter()
            .map(|(source, entity)| (*source, *entity))
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    use super::{Prefab, PrefabOverrides};
    use crate::{
        Component, SceneDocument, SceneEntityId, SceneEntityRecord, SceneEntityRef, SceneError,
        SerializedComponent, World,
    };

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Health {
        current: u32,
    }

    impl Component for Health {
        const ID: &'static str = "test.prefab_health";
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Link {
        target: SceneEntityRef,
    }

    impl Component for Link {
        const ID: &'static str = "test.prefab_link";
    }

    fn id(value: u64) -> SceneEntityId {
        SceneEntityId::new(value).expect("valid scene ID")
    }

    fn registered_world() -> World {
        let mut world = World::new();
        world
            .register_component::<Health>()
            .expect("register Health");
        world.register_component::<Link>().expect("register Link");
        world
    }

    fn linked_prefab(target: SceneEntityId) -> Prefab {
        let source = id(10);
        let mut record = SceneEntityRecord::new(source, None);
        record.components.insert(
            Link::ID.to_owned(),
            SerializedComponent {
                schema_version: 1,
                data: serde_json::to_value(Link {
                    target: SceneEntityRef::new(target),
                })
                .expect("serialize link"),
            },
        );
        Prefab::new(SceneDocument {
            schema_version: 1,
            next_scene_entity_id: 11,
            entities: vec![record],
        })
        .expect("create prefab")
    }

    #[test]
    fn instantiation_remaps_internal_references_and_applies_overrides() {
        let mut world = registered_world();
        let existing = world.spawn().expect("spawn existing entity");
        let existing_id = world.scene_entity_id(existing).expect("existing scene ID");

        let source = id(10);
        let target = id(20);
        let mut source_record = SceneEntityRecord::new(source, None);
        source_record.components.insert(
            Link::ID.to_owned(),
            SerializedComponent {
                schema_version: 1,
                data: serde_json::to_value(Link {
                    target: SceneEntityRef::new(target),
                })
                .expect("serialize link"),
            },
        );
        let mut target_record = SceneEntityRecord::new(target, Some(source));
        target_record.components.insert(
            Health::ID.to_owned(),
            SerializedComponent {
                schema_version: 1,
                data: json!({ "current": 5 }),
            },
        );
        let prefab = Prefab::new(SceneDocument {
            schema_version: 1,
            next_scene_entity_id: 21,
            entities: vec![source_record, target_record],
        })
        .expect("create prefab");

        let mut overrides = PrefabOverrides::default();
        overrides
            .set_component(
                target,
                Health::ID,
                SerializedComponent {
                    schema_version: 1,
                    data: json!({ "current": 99 }),
                },
            )
            .expect("set health override");

        let instance = prefab
            .instantiate_with_overrides(&mut world, &overrides)
            .expect("instantiate prefab");
        let instantiated_source = instance.entity(source).expect("resolve source");
        let instantiated_target = instance.entity(target).expect("resolve target");
        let instantiated_target_id = world
            .scene_entity_id(instantiated_target)
            .expect("instantiated target ID");

        assert_eq!(
            world
                .get::<Link>(instantiated_source)
                .expect("source link")
                .target
                .id(),
            instantiated_target_id
        );
        assert_ne!(instantiated_target_id, target);
        assert_eq!(
            world.parent_of(instantiated_target),
            Some(instantiated_source)
        );
        assert_eq!(
            world.get::<Health>(instantiated_target),
            Some(&Health { current: 99 })
        );
        assert_eq!(world.entity_for_scene_id(existing_id), Some(existing));
    }

    #[test]
    fn external_entity_references_are_preserved() {
        let mut world = registered_world();
        let existing = world.spawn().expect("spawn external entity");
        let existing_id = world.scene_entity_id(existing).expect("external scene ID");
        let prefab = linked_prefab(existing_id);

        let instance = prefab.instantiate(&mut world).expect("instantiate prefab");
        let entity = instance.entity(id(10)).expect("resolve prefab entity");
        assert_eq!(
            world
                .get::<Link>(entity)
                .expect("link component")
                .target
                .id(),
            existing_id
        );
    }

    #[test]
    fn remove_override_removes_component_from_instance() {
        let mut world = registered_world();
        let source = id(10);
        let mut record = SceneEntityRecord::new(source, None);
        record.components.insert(
            Health::ID.to_owned(),
            SerializedComponent {
                schema_version: 1,
                data: json!({ "current": 12 }),
            },
        );
        let prefab = Prefab::new(SceneDocument {
            schema_version: 1,
            next_scene_entity_id: 11,
            entities: vec![record],
        })
        .expect("create prefab");
        let mut overrides = PrefabOverrides::default();
        overrides.remove_component(source, Health::ID);

        let instance = prefab
            .instantiate_with_overrides(&mut world, &overrides)
            .expect("instantiate with removal");
        let entity = instance.entity(source).expect("resolve entity");
        assert!(world.get::<Health>(entity).is_none());
    }

    #[test]
    fn invalid_external_reference_fails_without_mutating_world() {
        let mut world = registered_world();
        let existing = world.spawn().expect("spawn existing");
        let existing_id = world.scene_entity_id(existing).expect("existing ID");
        let prefab = linked_prefab(id(999));

        assert!(matches!(
            prefab.instantiate(&mut world),
            Err(SceneError::MissingEntityReference { target, .. }) if target == id(999)
        ));
        assert_eq!(world.entity_for_scene_id(existing_id), Some(existing));
        assert_eq!(world.entities().len(), 1);
    }

    #[test]
    fn overrides_must_target_entities_in_the_prefab() {
        let mut world = registered_world();
        let prefab = linked_prefab(id(999));
        let mut overrides = PrefabOverrides::default();
        overrides
            .set_component(
                id(999),
                Health::ID,
                SerializedComponent {
                    schema_version: 1,
                    data: json!({ "current": 1 }),
                },
            )
            .expect("add override");

        assert_eq!(
            prefab.instantiate_with_overrides(&mut world, &overrides),
            Err(SceneError::UnknownPrefabEntity(id(999)))
        );
        assert!(world.entities().is_empty());
    }
}
