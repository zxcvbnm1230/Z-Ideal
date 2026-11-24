//! Deferred, ordered world mutations produced by systems.

use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::Arc,
};

use crate::{Component, Entity, World, WorldError};

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
    /// A command referenced a pending entity whose spawn command was not applied first.
    #[error("pending entity has not been spawned in this command buffer: {0:?}")]
    UnresolvedPendingEntity(PendingEntity),
    /// The world rejected a queued command.
    #[error(transparent)]
    World(#[from] WorldError),
}

type EntityOperation =
    Box<dyn FnOnce(&mut World, Entity) -> Result<(), WorldError> + Send + 'static>;

enum Command {
    Spawn(PendingEntity),
    Despawn(Entity),
    SetParent {
        child: EntityTarget,
        parent: Option<EntityTarget>,
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
/// applied and later commands are discarded.
pub struct Commands {
    buffer_identity: Arc<()>,
    commands: Vec<Command>,
    next_pending_id: u64,
}

impl Default for Commands {
    fn default() -> Self {
        Self {
            buffer_identity: Arc::new(()),
            commands: Vec::new(),
            next_pending_id: 0,
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

    /// Queues destruction of an existing entity.
    pub fn despawn(&mut self, entity: Entity) {
        self.commands.push(Command::Despawn(entity));
    }

    /// Queues a parent change. Pending targets must have had their spawn command
    /// queued earlier in this buffer.
    pub fn set_parent(&mut self, child: impl Into<EntityTarget>, parent: Option<EntityTarget>) {
        self.commands.push(Command::SetParent {
            child: child.into(),
            parent,
        });
    }

    pub(crate) fn apply(self, world: &mut World) -> Result<(), CommandError> {
        let mut spawned = HashMap::new();
        for command in self.commands {
            match command {
                Command::Spawn(pending) => {
                    let entity = world.spawn()?;
                    spawned.insert(pending, entity);
                }
                Command::Despawn(entity) => world.despawn(entity)?,
                Command::SetParent { child, parent } => {
                    let child = resolve(child, &spawned)?;
                    let parent = parent.map(|target| resolve(target, &spawned)).transpose()?;
                    world.set_parent(child, parent)?;
                }
                Command::Insert { target, operation } | Command::Remove { target, operation } => {
                    let entity = resolve(target, &spawned)?;
                    operation(world, entity)?;
                }
            }
        }
        Ok(())
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
