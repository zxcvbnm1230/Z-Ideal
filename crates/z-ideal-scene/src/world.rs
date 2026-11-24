//! Explicitly owned ECS world and sparse-set component storage.

use std::{
    any::{Any, TypeId},
    collections::HashMap,
};

use crate::{
    Component, ComponentInfo, Entity,
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
}

trait ErasedStorage: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn remove_entity(&mut self, entity: Entity);
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

impl<T: Send + Sync + 'static> ErasedStorage for SparseSet<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn remove_entity(&mut self, entity: Entity) {
        let _ = self.remove(entity);
    }
}

/// A single owned ECS world.
///
/// Structural operations require mutable world access. Systems will use the
/// scheduler command buffer to defer structural changes to synchronization points.
pub struct World {
    slots: Vec<EntitySlot>,
    free_slots: Vec<u32>,
    storages: HashMap<TypeId, Box<dyn ErasedStorage>>,
    component_by_type: HashMap<TypeId, ComponentInfo>,
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

        self.storages
            .insert(type_id, Box::<SparseSet<T>>::default());
        self.component_by_id.insert(T::ID, type_id);
        self.component_by_type
            .insert(type_id, ComponentInfo::new(T::ID, T::VERSION, type_id));
        Ok(())
    }

    /// Returns registered component metadata sorted by stable ID.
    #[must_use]
    pub fn registered_components(&self) -> Vec<ComponentInfo> {
        let mut components: Vec<_> = self.component_by_type.values().cloned().collect();
        components.sort_by_key(|component| component.id);
        components
    }

    /// Creates a new entity, reusing a free slot with an incremented generation.
    pub fn spawn(&mut self) -> Result<Entity, WorldError> {
        if let Some(index) = self.free_slots.pop() {
            let slot = &mut self.slots[index as usize];
            slot.alive = true;
            return Ok(Entity::new(index, slot.generation));
        }

        let index =
            u32::try_from(self.slots.len()).map_err(|_| WorldError::EntityCapacityExceeded)?;
        self.slots.push(EntitySlot {
            generation: 1,
            alive: true,
        });
        Ok(Entity::new(index, 1))
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
}
