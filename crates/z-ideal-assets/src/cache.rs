//! Strongly typed, reference-counted asset cache.
use crate::AssetId;
use std::{collections::HashMap, sync::Arc};

/// A live asset reference. Existing handles remain valid after cache eviction.
pub struct AssetHandle<T> {
    id: AssetId,
    value: Arc<T>,
}
impl<T> Clone for AssetHandle<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            value: Arc::clone(&self.value),
        }
    }
}
impl<T> AssetHandle<T> {
    /// Returns the stable asset identifier.
    #[must_use]
    pub const fn id(&self) -> AssetId {
        self.id
    }
    /// Borrows the cached asset value.
    #[must_use]
    pub fn get(&self) -> &T {
        &self.value
    }
    /// Returns the number of strong references to this value.
    #[must_use]
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.value)
    }
}

/// In-memory cache for one asset value type.
pub struct AssetCache<T> {
    entries: HashMap<AssetId, Arc<T>>,
}
impl<T> Default for AssetCache<T> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }
}
impl<T> AssetCache<T> {
    /// Creates an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Inserts a value. Replacing an existing entry is explicit and returns an error.
    pub fn insert(&mut self, id: AssetId, value: T) -> Result<AssetHandle<T>, AssetCacheError> {
        if self.entries.contains_key(&id) {
            return Err(AssetCacheError::AlreadyCached(id));
        }
        let value = Arc::new(value);
        self.entries.insert(id, Arc::clone(&value));
        Ok(AssetHandle { id, value })
    }
    /// Returns a cloned strong handle when the asset is cached.
    #[must_use]
    pub fn get(&self, id: AssetId) -> Option<AssetHandle<T>> {
        self.entries.get(&id).map(|value| AssetHandle {
            id,
            value: Arc::clone(value),
        })
    }
    /// Removes the cache's reference. Outstanding handles continue owning the asset.
    pub fn remove(&mut self, id: AssetId) -> Option<AssetHandle<T>> {
        self.entries
            .remove(&id)
            .map(|value| AssetHandle { id, value })
    }
    /// Returns the number of cached values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// Returns whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Cache insertion failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AssetCacheError {
    /// An entry with this ID is already present; use explicit eviction first.
    #[error("asset {0:?} is already cached")]
    AlreadyCached(AssetId),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssetKind, AssetUri};
    #[test]
    fn eviction_does_not_invalidate_existing_handles() {
        let id = AssetId::from_uri(AssetKind::Mesh, &AssetUri::new("mesh.glb").expect("URI"));
        let mut cache = AssetCache::new();
        let first = cache.insert(id, vec![1_u8, 2, 3]).expect("insert");
        let second = cache.get(id).expect("cached");
        assert_eq!(first.strong_count(), 3);
        assert!(cache.remove(id).is_some());
        assert_eq!(cache.len(), 0);
        assert_eq!(second.get(), &[1, 2, 3]);
    }
    #[test]
    fn duplicate_insert_is_not_an_implicit_reload() {
        let id = AssetId::from_uri(AssetKind::Mesh, &AssetUri::new("mesh.glb").expect("URI"));
        let mut cache = AssetCache::new();
        cache.insert(id, 1).expect("insert");
        assert!(
            matches!(cache.insert(id, 2), Err(AssetCacheError::AlreadyCached(found)) if found == id)
        );
    }
}
