//! Explicitly owned ECS world and sparse-set component storage.

use std::{
    any::{Any, TypeId},
    collections::HashMap,
};

use serde_json::Value;

use crate::{
    Component, ComponentInfo, Entity, SceneDocument, SceneEntityId, SceneEntityRecord, SceneError,
    SerializedComponent, SCENE_SCHEMA_VERSION,
    hierarchy::{Hierarchy, HierarchyError},
};
use z_ideal_math::{Mat4, Transform};

/// ECS world operation failures.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorldError {
    /// The entity handle is stale or does not belong to this live world.
    #[error("entity handle is stale: {0:?}")]
    StaleEntity(Entity),
    /// The world has exhausted its representable entity slots.
    #[error("entity capacity exceeds the u32 slot index range")]
    EntityCapacityExceeded,
    /// Persistent scene entity ID allocation is exhausted.
    #[error("persistent scene entity ID space is exhausted")]
    SceneEntityIdExhausted,
    /// Component storage must be registered before use.
    #[error("component is not registered: {0}")]
    ComponentNotRegistered(&'static str),
    /// A stable component ID was already assigned to another Rust type.
    #[error("component ID is already registered: {0}")]
    DuplicateComponentId(&'static str),
    /// Component validation failed.
    #[error("component {id} failed validation: {reason}")]
    InvalidComponent { id: &'static str, reason: String },
    /// Internal storage did not match its registered Rust type.
    #[error("component storage type mismatch for {0}")]
    StorageTypeMismatch(&'static str),
    /// Hierarchy operation failed.
    #[error("invalid hierarchy operation: {0}")]
    Hierarchy(#[from] HierarchyError),
}

#[derive(Debug, Clone, Copy)]
struct EntitySlot {
    generation: u32,
    alive: bool,
    scene_id: SceneEntityId,
}

trait ErasedStorage: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn remove_entity(&mut self, entity: Entity);
    fn serialize_entity(&self, entity: Entity) -> Result<Option<Value>, serde_json::Error>;
}

#[derive(Debug)]
struct SparseSet<T> {
    dense_entities: Vec<Entity>,
    dense_values: Vec<T>,
    sparse: Vec<Option<usize>>,
}

impl<T> Default for SparseSet<T> {
    fn default() -> Self {
        Self {
            dense_entities: Vec::new(),
            dense_values: Vec::new(),
            sparse: Vec::new(),
        }
    }
}

impl<T> SparseSet<T> {
    fn get(&self, entity: Entity) -> Option<&T> {
        let dense_index = self
            .sparse
            .get(entity.index() as usize)
            .copied()
            .flatten()?;
        (self.dense_entities.get(dense_index).copied() == Some(entity))
            .then(|| &self.dense_values[dense_index])
    }

    fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        let dense_index = self
            .sparse
            .get(entity.index() as usize)
            .copied()
            .flatten()?;
        if self.dense_entities.get(dense_index).copied() != Some(entity) {
            return None;
        }
        self.dense_values.get_mut(dense_index)
    }

    fn insert(&mut self, entity: Entity, value: T) -> Option<T> {
        let slot_index = entity.index() as usize;
        if let Some(dense_index) = self.sparse.get(slot_index).copied().flatten() {
            if self.dense_entities.get(dense_index).copied() == Some(entity) {
                return Some(std::mem::replace(
                    &mut self.dense_values[dense_index],
                    value,
                ));
            }
        }

        if self.sparse.len() <= slot_index {
            self.sparse.resize(slot_index + 1, None);
        }
        let dense_index = self.dense_values.len();
        self.dense_entities.push(entity);
        self.dense_values.push(value);
        self.sparse[slot_index] = Some(dense_index);
        None
    }

    fn remove(&mut self, entity: Entity) -> Option<T> {
        let slot_index = entity.index() as usize;
        let dense_index = self.sparse.get(slot_index).copied().flatten()?;
        if self.dense_entities.get(dense_index).copied() != Some(entity) {
            return None;
        }

        self.sparse[slot_index] = None;
        self.dense_entities.swap_remove(dense_index);
        let removed = self.dense_values.swap_remove(dense_index);
        if let Some(moved_entity) = self.dense_entities.get(dense_index).copied() {
            self.sparse[moved_entity.index() as usize] = Some(dense_index);
        }
        Some(removed)
    }

    fn entities(&self) -> &[Entity] {
        &self.dense_entities
    }
}

impl<T: Component> ErasedStorage for SparseSet<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn remove_entity(&mut self, entity: Entity) {
        let _ = self.remove(entity);
    }

    fn serialize_entity(&self, entity: Entity) -> Result<Option<Value>, serde_json::Error> {
        self.get(entity).map(serde_json::to_value).transpose()
    }
}

type ComponentDecoder = fn(&mut World, Entity, u32, Value) -> Result<(), SceneError>;
type StorageFactory = fn() -> Box<dyn ErasedStorage>;

#[derive(Clone)]
struct ComponentDescriptor {
    info: ComponentInfo,
    decoder: ComponentDecoder,
    storage_factory: StorageFactory,
}

fn create_storage<T: Component>() -> Box<dyn ErasedStorage> {
    Box::<SparseSet<T>>::default()
}

fn decode_component<T: Component>(
    world: &mut World,
    entity: Entity,
    version: u32,
    data: Value,
) -> Result<(), SceneError> {
    let migrated = T::migrate(version, data).map_err(|message| SceneError::ComponentMigration {
        component_id: T::ID,
        from_version: version,
        to_version: T::VERSION,
        message,
    })?;
    let component = serde_json::from_value::<T>(migrated).map_err(|error| {
        SceneError::InvalidComponentData {
            component_id: T::ID,
            message: error.to_string(),
        }
    })?;
    world.insert(entity, component)?;
    Ok(())
}

/// A single owned ECS world.
///
/// Structural operations require mutable world access. Systems use the
/// scheduler command buffer to defer structural changes to synchronization points.
pub struct World {
    slots: Vec<EntitySlot>,
    free_slots: Vec<u32>,
    next_scene_id: u64,
    storages: HashMap<TypeId, Box<dyn ErasedStorage>>,
    component_by_type: HashMap<TypeId, ComponentDescriptor>,
    component_by_id: HashMap<&'static str, TypeId>,
    hierarchy: Hierarchy,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// Creates an empty world with no registered component types.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_slots: Vec::new(),
            next_scene_id: 1,
            storages: HashMap::new(),
            component_by_type: HashMap::new(),
            component_by_id: HashMap::new(),
            hierarchy: Hierarchy::default(),
        }
    }

    /// Registers a component type and its stable ID.
    ///
    /// Registration is idempotent for the same Rust type. A stable ID may not
    /// be reused by a different Rust type.
    pub fn register_component<T: Component>(&mut self) -> Result<(), WorldError> {
        let type_id = TypeId::of::<T>();
        if self.component_by_type.contains_key(&type_id) {
            return Ok(());
        }
        if T::ID.is_empty() || T::VERSION == 0 {
            return Err(WorldError::InvalidComponent {
                id: T::ID,
                reason: "component ID and schema version must be non-empty/non-zero".to_owned(),
            });
        }
        if self.component_by_id.contains_key(T::ID) {
            return Err(WorldError::DuplicateComponentId(T::ID));
        }

        self.storages.insert(type_id, create_storage::<T>());
        self.component_by_id.insert(T::ID, type_id);
        self.component_by_type.insert(
            type_id,
            ComponentDescriptor {
                info: ComponentInfo::new(T::ID, T::VERSION, type_id),
                decoder: decode_component::<T>,
                storage_factory: create_storage::<T>,
            },
        );
        Ok(())
    }

    /// Returns registered component metadata sorted by stable ID.
    #[must_use]
    pub fn registered_components(&self) -> Vec<ComponentInfo> {
        let mut components: Vec<_> = self
            .component_by_type
            .values()
            .map(|descriptor| descriptor.info.clone())
            .collect();
        components.sort_by_key(|component| component.id);
        components
    }

    /// Creates a new entity with a fresh persistent scene ID.
    pub fn spawn(&mut self) -> Result<Entity, WorldError> {
        let next_scene_id = self
            .next_scene_id
            .checked_add(1)
            .ok_or(WorldError::SceneEntityIdExhausted)?;
        let scene_id = SceneEntityId::new_unchecked(self.next_scene_id);
        let entity = self.spawn_with_scene_id(scene_id)?;
        self.next_scene_id = next_scene_id;
        Ok(entity)
    }

    fn spawn_with_scene_id(&mut self, scene_id: SceneEntityId) -> Result<Entity, WorldError> {
        if let Some(index) = self.free_slots.pop() {
            let slot = &mut self.slots[index as usize];
            slot.alive = true;
            slot.scene_id = scene_id;
            return Ok(Entity::new(index, slot.generation));
        }

        let index =
            u32::try_from(self.slots.len()).map_err(|_| WorldError::EntityCapacityExceeded)?;
        self.slots.push(EntitySlot {
            generation: 1,
            alive: true,
            scene_id,
        });
        Ok(Entity::new(index, 1))
    }

    /// Returns the persistent scene ID for a live entity.
    pub fn scene_entity_id(&self, entity: Entity) -> Result<SceneEntityId, WorldError> {
        if !self.is_alive(entity) {
            return Err(WorldError::StaleEntity(entity));
        }
        Ok(self.slots[entity.index() as usize].scene_id)
    }

    /// Resolves a persistent scene ID to its current runtime handle.
    #[must_use]
    pub fn entity_for_scene_id(&self, scene_id: SceneEntityId) -> Option<Entity> {
        self.slots
            .iter()
            .enumerate()
            .find(|(_, slot)| slot.alive && slot.scene_id == scene_id)
            .map(|(index, slot)| Entity::new(index as u32, slot.generation))
    }

    /// Returns whether the exact entity generation is alive in this world.
    #[must_use]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.slots
            .get(entity.index() as usize)
            .is_some_and(|slot| slot.alive && slot.generation == entity.generation())
    }

    /// Returns all live entities sorted by slot index.
    #[must_use]
    pub fn entities(&self) -> Vec<Entity> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.alive)
            .map(|(index, slot)| Entity::new(index as u32, slot.generation))
            .collect()
    }

    /// Destroys an entity, its components, and its hierarchy edges.
    pub fn despawn(&mut self, entity: Entity) -> Result<(), WorldError> {
        if !self.is_alive(entity) {
            return Err(WorldError::StaleEntity(entity));
        }

        self.hierarchy.remove_entity(entity);
        for storage in self.storages.values_mut() {
            storage.remove_entity(entity);
        }

        let slot = &mut self.slots[entity.index() as usize];
        slot.alive = false;
        if let Some(next_generation) = slot.generation.checked_add(1) {
            slot.generation = next_generation;
            self.free_slots.push(entity.index());
        }
        Ok(())
    }

    /// Inserts or replaces a component on a live entity.
    pub fn insert<T: Component>(
        &mut self,
        entity: Entity,
        component: T,
    ) -> Result<Option<T>, WorldError> {
        if !self.is_alive(entity) {
            return Err(WorldError::StaleEntity(entity));
        }
        if !self.component_by_type.contains_key(&TypeId::of::<T>()) {
            return Err(WorldError::ComponentNotRegistered(T::ID));
        }
        component
            .validate()
            .map_err(|reason| WorldError::InvalidComponent { id: T::ID, reason })?;

        let storage = self
            .storages
            .get_mut(&TypeId::of::<T>())
            .and_then(|storage| storage.as_any_mut().downcast_mut::<SparseSet<T>>())
            .ok_or(WorldError::StorageTypeMismatch(T::ID))?;
        Ok(storage.insert(entity, component))
    }

    /// Returns an immutable component reference.
    #[must_use]
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        if !self.is_alive(entity) {
            return None;
        }
        self.storages
            .get(&TypeId::of::<T>())?
            .as_any()
            .downcast_ref::<SparseSet<T>>()?
            .get(entity)
    }

    /// Returns a mutable component reference.
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        if !self.is_alive(entity) {
            return None;
        }
        self.storages
            .get_mut(&TypeId::of::<T>())?
            .as_any_mut()
            .downcast_mut::<SparseSet<T>>()?
            .get_mut(entity)
    }

    /// Removes a component from a live entity.
    pub fn remove<T: Component>(&mut self, entity: Entity) -> Result<Option<T>, WorldError> {
        if !self.is_alive(entity) {
            return Err(WorldError::StaleEntity(entity));
        }
        if !self.component_by_type.contains_key(&TypeId::of::<T>()) {
            return Err(WorldError::ComponentNotRegistered(T::ID));
        }
        let storage = self
            .storages
            .get_mut(&TypeId::of::<T>())
            .and_then(|storage| storage.as_any_mut().downcast_mut::<SparseSet<T>>())
            .ok_or(WorldError::StorageTypeMismatch(T::ID))?;
        Ok(storage.remove(entity))
    }

    /// Returns entities that currently have T, sorted by entity slot index.
    #[must_use]
    pub fn entities_with<T: Component>(&self) -> Vec<Entity> {
        let Some(storage) = self.storages.get(&TypeId::of::<T>()) else {
            return Vec::new();
        };
        let Some(storage) = storage.as_any().downcast_ref::<SparseSet<T>>() else {
            return Vec::new();
        };
        let mut entities = storage.entities().to_vec();
        entities.sort_unstable();
        entities.retain(|entity| self.is_alive(*entity));
        entities
    }

    /// Changes an entity's parent. Passing None makes it a root entity.
    pub fn set_parent(&mut self, child: Entity, parent: Option<Entity>) -> Result<(), WorldError> {
        if !self.is_alive(child) {
            return Err(WorldError::StaleEntity(child));
        }
        if let Some(parent) = parent {
            if !self.is_alive(parent) {
                return Err(WorldError::StaleEntity(parent));
            }
        }
        self.hierarchy.set_parent(child, parent)?;
        Ok(())
    }

    /// Returns an entity's parent, if any.
    #[must_use]
    pub fn parent_of(&self, entity: Entity) -> Option<Entity> {
        self.is_alive(entity)
            .then(|| self.hierarchy.parent_of(entity))
            .flatten()
    }

    /// Returns direct children in deterministic entity order.
    #[must_use]
    pub fn children_of(&self, entity: Entity) -> Vec<Entity> {
        if self.is_alive(entity) {
            self.hierarchy.children_of(entity)
        } else {
            Vec::new()
        }
    }

    /// Computes the world matrix by composing ancestor transforms root-to-leaf.
    ///
    /// Entities without a registered Transform component use identity local
    /// transforms. Stale entities are rejected.
    pub fn world_transform(&self, entity: Entity) -> Result<Mat4, WorldError> {
        if !self.is_alive(entity) {
            return Err(WorldError::StaleEntity(entity));
        }

        let mut chain = vec![entity];
        let mut cursor = self.hierarchy.parent_of(entity);
        while let Some(parent) = cursor {
            if !self.is_alive(parent) {
                return Err(WorldError::StaleEntity(parent));
            }
            chain.push(parent);
            cursor = self.hierarchy.parent_of(parent);
        }

        let mut world_matrix = Mat4::IDENTITY;
        for current in chain.into_iter().rev() {
            let local = self.get::<Transform>(current).copied().unwrap_or_default();
            world_matrix *= local.matrix();
        }
        Ok(world_matrix)
    }

    /// Serializes the world to a validated, deterministic scene document.
    pub fn to_scene_document(&self) -> Result<SceneDocument, SceneError> {
        let mut entities = self.entities();
        entities.sort_by_key(|entity| {
            self.scene_entity_id(*entity)
                .map(SceneEntityId::get)
                .unwrap_or_default()
        });

        let mut records = Vec::with_capacity(entities.len());
        let mut record_indices = HashMap::with_capacity(entities.len());
        for entity in &entities {
            let id = self.scene_entity_id(*entity)?;
            let parent = self
                .parent_of(*entity)
                .map(|parent| self.scene_entity_id(parent))
                .transpose()?;
            record_indices.insert(*entity, records.len());
            records.push(SceneEntityRecord::new(id, parent));
        }

        for (type_id, storage) in &self.storages {
            let descriptor = self
                .component_by_type
                .get(type_id)
                .ok_or(WorldError::StorageTypeMismatch("component registry"))?;
            for entity in &entities {
                if let Some(data) = storage.serialize_entity(*entity).map_err(|error| {
                    SceneError::ComponentSerialization {
                        component_id: descriptor.info.id.to_owned(),
                        message: error.to_string(),
                    }
                })? {
                    let index = record_indices
                        .get(entity)
                        .copied()
                        .ok_or(WorldError::StaleEntity(*entity))?;
                    records[index].components.insert(
                        descriptor.info.id.to_owned(),
                        SerializedComponent {
                            schema_version: descriptor.info.version,
                            data,
                        },
                    );
                }
            }
        }

        let document = SceneDocument {
            schema_version: SCENE_SCHEMA_VERSION,
            entities: records,
        };
        document.validate()?;
        Ok(document)
    }

    /// Serializes the world as pretty-printed JSON.
    pub fn to_scene_json(&self) -> Result<String, SceneError> {
        Ok(serde_json::to_string_pretty(&self.to_scene_document()?)?)
    }

    /// Loads a scene document transactionally.
    ///
    /// The existing world is replaced only after all entities, components,
    /// migrations, and hierarchy relationships have been validated.
    pub fn load_scene_document(&mut self, document: &SceneDocument) -> Result<(), SceneError> {
        document.validate()?;
        let max_id = document
            .entities
            .iter()
            .map(|entity| entity.id.get())
            .max()
            .unwrap_or(0);
        let next_scene_id = max_id
            .checked_add(1)
            .ok_or(SceneError::SceneEntityIdExhausted)?;

        let mut staging = self.empty_with_registry();
        staging.next_scene_id = next_scene_id.max(1);
        let mut entities_by_id = HashMap::with_capacity(document.entities.len());

        for record in &document.entities {
            let entity = staging.spawn_with_scene_id(record.id)?;
            entities_by_id.insert(record.id, entity);
        }

        for record in &document.entities {
            let entity = entities_by_id[&record.id];
            for (component_id, component) in &record.components {
                let type_id = staging
                    .component_by_id
                    .get(component_id.as_str())
                    .copied()
                    .ok_or_else(|| SceneError::UnknownComponent(component_id.clone()))?;
                let decoder = staging
                    .component_by_type
                    .get(&type_id)
                    .map(|descriptor| descriptor.decoder)
                    .ok_or(WorldError::StorageTypeMismatch("component registry"))?;
                decoder(
                    &mut staging,
                    entity,
                    component.schema_version,
                    component.data.clone(),
                )?;
            }

            if let Some(parent_id) = record.parent {
                let parent = entities_by_id.get(&parent_id).copied().ok_or(
                    SceneError::MissingParent {
                        entity: record.id,
                        parent: parent_id,
                    },
                )?;
                staging.set_parent(entity, Some(parent))?;
            }
        }

        *self = staging;
        Ok(())
    }

    /// Parses and loads a JSON scene document transactionally.
    pub fn load_scene_json(&mut self, source: &str) -> Result<(), SceneError> {
        let document: SceneDocument = serde_json::from_str(source)?;
        self.load_scene_document(&document)
    }

    fn empty_with_registry(&self) -> Self {
        let mut world = Self::new();
        world.component_by_type = self.component_by_type.clone();
        world.component_by_id = self.component_by_id.clone();
        for (type_id, descriptor) in &world.component_by_type {
            world
                .storages
                .insert(*type_id, (descriptor.storage_factory)());
        }
        world
    }
}
