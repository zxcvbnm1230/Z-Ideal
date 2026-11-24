//! Parent/child relationships with cycle-safe updates.

use std::collections::{BTreeSet, HashMap};

use crate::Entity;

/// Invalid hierarchy operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum HierarchyError {
    /// An entity cannot parent itself.
    #[error("an entity cannot be its own parent")]
    SelfParent,
    /// The requested parent would create a cycle.
    #[error("the requested parent would create a hierarchy cycle")]
    Cycle,
}

/// World-owned parent/child index.
#[derive(Debug, Default)]
pub(crate) struct Hierarchy {
    parents: HashMap<Entity, Entity>,
    children: HashMap<Entity, BTreeSet<Entity>>,
}

impl Hierarchy {
    pub(crate) fn parent_of(&self, entity: Entity) -> Option<Entity> {
        self.parents.get(&entity).copied()
    }

    pub(crate) fn children_of(&self, entity: Entity) -> Vec<Entity> {
        self.children
            .get(&entity)
            .map(|children| children.iter().copied().collect())
            .unwrap_or_default()
    }

    pub(crate) fn set_parent(
        &mut self,
        child: Entity,
        parent: Option<Entity>,
    ) -> Result<(), HierarchyError> {
        if let Some(parent) = parent {
            if child == parent {
                return Err(HierarchyError::SelfParent);
            }
            let mut cursor = Some(parent);
            while let Some(ancestor) = cursor {
                if ancestor == child {
                    return Err(HierarchyError::Cycle);
                }
                cursor = self.parents.get(&ancestor).copied();
            }
        }

        if let Some(old_parent) = self.parents.remove(&child) {
            let remove_old_parent = if let Some(children) = self.children.get_mut(&old_parent) {
                children.remove(&child);
                children.is_empty()
            } else {
                false
            };
            if remove_old_parent {
                self.children.remove(&old_parent);
            }
        }

        if let Some(parent) = parent {
            self.parents.insert(child, parent);
            self.children.entry(parent).or_default().insert(child);
        }
        Ok(())
    }

    pub(crate) fn remove_entity(&mut self, entity: Entity) {
        if let Some(parent) = self.parents.remove(&entity) {
            let remove_parent = if let Some(children) = self.children.get_mut(&parent) {
                children.remove(&entity);
                children.is_empty()
            } else {
                false
            };
            if remove_parent {
                self.children.remove(&parent);
            }
        }

        if let Some(children) = self.children.remove(&entity) {
            for child in children {
                self.parents.remove(&child);
            }
        }
    }
}
