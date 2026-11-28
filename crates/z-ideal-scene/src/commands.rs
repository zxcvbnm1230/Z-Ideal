//! Deferred, ordered world mutations produced by systems.

use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::Arc,
};

use crate::{Component, Entity, Prefab, PrefabInstance, PrefabOverrides, World, WorldError};

/// A handle to an entity that will be created when a command buffer is applied.
#[derive(Debug, Clone)]
pub struct PendingEntity {
    buffer_identity: Arc<()>,
    id: u64,
}

impl PartialEq for PendingEntity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.buffer_identity, &other.buffer_identity) && self.id == other.id
    }
}

impl Eq for PendingEntity {}

impl Hash for PendingEntity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.buffer_identity) as usize).hash(state);
        self.id.hash(state);
    }
}

/// A handle to a prefab instance that will be created when a command buffer is applied.
#[derive(Debug, Clone)]
pub struct PendingPrefab {
    buffer_identity: Arc<()>,
    id: u64,
}

impl PartialEq for PendingPrefab {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.buffer_identity, &other.buffer_identity) && self.id == other.id
    }
}

impl Eq for PendingPrefab {}

impl Hash for PendingPrefab {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.buffer_identity) as usize).hash(state);
        self.id.hash(state);
    }
}

/// Identifies either an existing world entity or an entity reserved in this command buffer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntityTarget {
    /// An entity that already exists in the world.
    Existing(Entity),
    /// An entity created earlier in the same command buffer.
    Pending(PendingEntity),
}

impl From<Entity> for EntityTarget {
    fn from(entity: Entity) -> Self {
        Self::Existing(entity)
    }
}

impl From<PendingEntity> for EntityTarget {
    fn from(entity: PendingEntity) -> Self {
        Self::Pending(entity)
    }
}

/// Command-buffer application failures.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CommandError {
    /// Pending entity token space was exhausted.
    #[error("pending entity token space was exhausted")]
    PendingIdExhausted,
    /// Pending prefab token space was exhausted.
    #[error("pending prefab token space was exhausted")]
    PendingPrefabIdExhausted,
    /// Prefab instantiation failed during command application.
    #[error("prefab instantiation failed: {0}")]
    PrefabInstantiation(String),
    /// A command referenced a pending entity whose spawn command was not applied first.
    #[error("pending entity has not been spawned in this command buffer: {0:?}")]
    UnresolvedPendingEntity(PendingEntity),
    /// The world rejected a queued command.
    #[error(transparent)]
    World(#[from] WorldError),
    /// Command application failed after earlier entity/prefab creations committed.
    #[error("command application stopped after partial changes: {cause}")]
    PartialApplication {
        /// The command that failed.
        cause: Box<CommandError>,
        /// Handles for entities and prefab instances created before the failure.
        results: CommandResults,
    },
}

impl CommandError {
    /// Returns handles produced before a sequential command-application failure.
    #[must_use]
    pub fn partial_results(&self) -> Option<&CommandResults> {
        match self {
            Self::PartialApplication { results, .. } => Some(results),
            _ => None,
        }
    }
}

type EntityOperation =
    Box<dyn FnOnce(&mut World, Entity) -> Result<(), WorldError> + Send + 'static>;

enum Command {
    Spawn(PendingEntity),
    Despawn(EntityTarget),
    SetParent {
        child: EntityTarget,
        parent: Option<EntityTarget>,
    },
    InstantiatePrefab {
        pending: PendingPrefab,
        prefab: Prefab,
        overrides: PrefabOverrides,
    },
    Insert {
        target: EntityTarget,
        operation: EntityOperation,
    },
    Remove {
        target: EntityTarget,
        operation: EntityOperation,
    },
}

/// Ordered command buffer for structural changes.
///
/// Commands are applied FIFO after the system that produced them succeeds.
/// A failed system discards its buffer. Application is sequential rather than
/// transactional: if a command fails, earlier commands in that buffer remain
/// applied and later commands are discarded. If earlier entity or prefab
/// creation commands committed, the returned PartialApplication error retains
/// their CommandResults so callers can still resolve those handles.
pub struct Commands {
    buffer_identity: Arc<()>,
    commands: Vec<Command>,
    next_pending_id: u64,
    next_pending_prefab_id: u64,
}

impl Default for Commands {
    fn default() -> Self {
        Self {
            buffer_identity: Arc::new(()),
            commands: Vec::new(),
            next_pending_id: 0,
            next_pending_prefab_id: 0,
        }
    }
}

impl Commands {
    /// Creates an empty command buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Queues creation of an entity and returns a token usable by later commands
    /// in this same buffer.
    pub fn spawn(&mut self) -> Result<PendingEntity, CommandError> {
        let next = self
            .next_pending_id
            .checked_add(1)
            .ok_or(CommandError::PendingIdExhausted)?;
        let pending = PendingEntity {
            buffer_identity: Arc::clone(&self.buffer_identity),
            id: self.next_pending_id,
        };
        self.next_pending_id = next;
        self.commands.push(Command::Spawn(pending.clone()));
        Ok(pending)
    }

    /// Queues instantiation of a validated prefab. The returned ticket can be
    /// resolved through CommandResults after the command buffer is applied.
    pub fn instantiate_prefab(
        &mut self,
        prefab: Prefab,
        overrides: PrefabOverrides,
    ) -> Result<PendingPrefab, CommandError> {
        let next = self
            .next_pending_prefab_id
            .checked_add(1)
            .ok_or(CommandError::PendingPrefabIdExhausted)?;
        let pending = PendingPrefab {
            buffer_identity: Arc::clone(&self.buffer_identity),
            id: self.next_pending_prefab_id,
        };
        self.next_pending_prefab_id = next;
        self.commands.push(Command::InstantiatePrefab {
            pending: pending.clone(),
            prefab,
            overrides,
        });
        Ok(pending)
    }

    /// Queues insertion or replacement of a component on an existing or pending entity.
    pub fn insert<T: Component>(&mut self, target: impl Into<EntityTarget>, component: T) {
        self.commands.push(Command::Insert {
            target: target.into(),
            operation: Box::new(move |world, entity| world.insert(entity, component).map(|_| ())),
        });
    }

    /// Queues removal of a component from an existing or pending entity.
    pub fn remove<T: Component>(&mut self, target: impl Into<EntityTarget>) {
        self.commands.push(Command::Remove {
            target: target.into(),
            operation: Box::new(|world, entity| world.remove::<T>(entity).map(|_| ())),
        });
    }

    /// Queues destruction of an existing or earlier pending entity.
    ///
    /// A pending target must have had its spawn command queued earlier in this
    /// buffer. The operation is resolved when the buffer is applied.
    pub fn despawn(&mut self, entity: impl Into<EntityTarget>) {
        self.commands.push(Command::Despawn(entity.into()));
    }

    /// Queues a parent change. Pending targets must have had their spawn command
    /// queued earlier in this buffer.
    pub fn set_parent(&mut self, child: impl Into<EntityTarget>, parent: Option<EntityTarget>) {
        self.commands.push(Command::SetParent {
            child: child.into(),
            parent,
        });
    }

    /// Applies commands FIFO and returns the runtime handles assigned to
    /// pending entities.
    ///
    /// Application is sequential, not transactional: if a command fails,
    /// earlier commands remain applied and later commands are discarded.
    pub fn apply(self, world: &mut World) -> Result<CommandResults, CommandError> {
        let mut spawned = HashMap::new();
        let mut spawned_prefabs = HashMap::new();
        for command in self.commands {
            let result = match command {
                Command::Spawn(pending) => world
                    .spawn()
                    .map(|entity| {
                        spawned.insert(pending, entity);
                    })
                    .map_err(CommandError::World),
                Command::Despawn(target) => resolve(target, &spawned)
                    .and_then(|entity| world.despawn(entity).map_err(CommandError::World)),
                Command::SetParent { child, parent } => {
                    resolve(child, &spawned).and_then(|child| {
                        let parent = parent.map(|target| resolve(target, &spawned)).transpose()?;
                        world.set_parent(child, parent).map_err(CommandError::World)
                    })
                }
                Command::InstantiatePrefab {
                    pending,
                    prefab,
                    overrides,
                } => match prefab.instantiate_with_overrides(world, &overrides) {
                    Ok(instance) => {
                        spawned_prefabs.insert(pending, instance);
                        Ok(())
                    }
                    Err(error) => Err(CommandError::PrefabInstantiation(error.to_string())),
                },
                Command::Insert { target, operation } | Command::Remove { target, operation } => {
                    resolve(target, &spawned)
                        .and_then(|entity| operation(world, entity).map_err(CommandError::World))
                }
            };
            if let Err(cause) = result {
                let results = CommandResults {
                    spawned,
                    spawned_prefabs,
                };
                if results.spawned.is_empty() && results.spawned_prefabs.is_empty() {
                    return Err(cause);
                }
                return Err(CommandError::PartialApplication {
                    cause: Box::new(cause),
                    results,
                });
            }
        }
        Ok(CommandResults {
            spawned,
            spawned_prefabs,
        })
    }
}

/// Runtime entities created while applying one command buffer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommandResults {
    spawned: HashMap<PendingEntity, Entity>,
    spawned_prefabs: HashMap<PendingPrefab, PrefabInstance>,
}

impl CommandResults {
    /// Resolves a pending token created by the applied command buffer.
    #[must_use]
    pub fn entity(&self, pending: &PendingEntity) -> Option<Entity> {
        self.spawned.get(pending).copied()
    }

    /// Resolves a prefab ticket created by the applied command buffer.
    #[must_use]
    pub fn prefab(&self, pending: &PendingPrefab) -> Option<&PrefabInstance> {
        self.spawned_prefabs.get(pending)
    }

    /// Returns the number of pending entities successfully spawned.
    #[must_use]
    pub fn spawned_count(&self) -> usize {
        self.spawned.len()
    }
}

fn resolve(
    target: EntityTarget,
    spawned: &HashMap<PendingEntity, Entity>,
) -> Result<Entity, CommandError> {
    match target {
        EntityTarget::Existing(entity) => Ok(entity),
        EntityTarget::Pending(pending) => spawned
            .get(&pending)
            .copied()
            .ok_or(CommandError::UnresolvedPendingEntity(pending)),
    }
}

#[cfg(test)]
mod tests {
    use super::{CommandError, Commands};
    use crate::World;

    #[test]
    fn apply_returns_handles_for_spawned_pending_entities() {
        let mut world = World::new();
        let mut commands = Commands::new();
        let pending = commands.spawn().expect("queue spawn");
        let results = commands.apply(&mut world).expect("apply commands");
        let entity = results.entity(&pending).expect("resolve pending entity");
        assert!(world.is_alive(entity));
        assert_eq!(results.spawned_count(), 1);
    }

    #[test]
    fn pending_entity_can_be_destroyed_in_same_buffer() {
        let mut world = World::new();
        let mut commands = Commands::new();
        let pending = commands.spawn().expect("queue spawn");
        commands.despawn(pending.clone());
        let results = commands.apply(&mut world).expect("apply commands");
        let entity = results.entity(&pending).expect("resolve pending entity");
        assert!(!world.is_alive(entity));
    }

    #[test]
    fn pending_tokens_cannot_cross_command_buffers() {
        let mut world = World::new();
        let mut first = Commands::new();
        let foreign = first.spawn().expect("queue foreign spawn");
        let mut second = Commands::new();
        second.set_parent(foreign.clone(), None);
        assert!(matches!(
            second.apply(&mut world),
            Err(CommandError::UnresolvedPendingEntity(pending)) if pending == foreign
        ));
        assert!(world.entities().is_empty());
    }
}

#[cfg(test)]
mod prefab_command_tests {
    use super::Commands;
    use crate::{Prefab, SceneDocument, SceneEntityId, SceneEntityRecord, World};

    fn prefab() -> (Prefab, SceneEntityId) {
        let source_id = SceneEntityId::new(12).expect("source ID");
        let mut document = SceneDocument::new();
        document.next_scene_entity_id = 13;
        document
            .entities
            .push(SceneEntityRecord::new(source_id, None));
        (Prefab::new(document).expect("prefab"), source_id)
    }

    #[test]
    fn prefab_command_returns_instance_mapping_after_commit() {
        let mut world = World::new();
        let (prefab, source_id) = prefab();
        let mut commands = Commands::new();
        let ticket = commands
            .instantiate_prefab(prefab, Default::default())
            .expect("queue prefab");
        let results = commands.apply(&mut world).expect("apply prefab");
        let instance = results.prefab(&ticket).expect("resolve prefab ticket");
        let entity = instance.entity(source_id).expect("resolve prefab entity");
        assert!(world.is_alive(entity));
    }

    #[test]
    fn prefab_ticket_does_not_resolve_in_another_command_buffer() {
        let mut world = World::new();
        let (prefab, _) = prefab();
        let mut first = Commands::new();
        let ticket = first
            .instantiate_prefab(prefab, Default::default())
            .expect("queue prefab");
        let second = Commands::new();
        let results = second.apply(&mut world).expect("apply empty buffer");
        assert!(results.prefab(&ticket).is_none());
        assert!(world.entities().is_empty());
    }
}

#[cfg(test)]
mod partial_command_tests {
    use super::Commands;
    use crate::{Prefab, SceneDocument, SceneEntityId, SceneEntityRecord, World};

    #[test]
    fn partial_application_error_preserves_prefab_instance_results() {
        let mut world = World::new();
        let stale = world.spawn().expect("allocate stale handle");
        world.despawn(stale).expect("make stale handle");
        let source_id = SceneEntityId::new(12).expect("source ID");
        let mut document = SceneDocument::new();
        document.next_scene_entity_id = 13;
        document
            .entities
            .push(SceneEntityRecord::new(source_id, None));
        let prefab = Prefab::new(document).expect("prefab");

        let mut commands = Commands::new();
        let ticket = commands
            .instantiate_prefab(prefab, Default::default())
            .expect("queue prefab");
        commands.despawn(stale);
        let error = commands.apply(&mut world).expect_err("stale despawn fails");
        let results = error.partial_results().expect("partial command results");
        let instance = results
            .prefab(&ticket)
            .expect("prefab was committed before failure");
        let entity = instance.entity(source_id).expect("prefab entity mapping");
        assert!(world.is_alive(entity));
    }
}
