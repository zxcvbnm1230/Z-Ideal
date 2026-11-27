//! Versioned asset metadata and deterministic dependency graph validation.
use crate::{AssetId, AssetKind, AssetUri};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Current serialized asset-metadata schema.
pub const ASSET_METADATA_SCHEMA_VERSION: u32 = 1;

/// Versioned metadata for one imported or source asset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetMetadata {
    /// Metadata schema version.
    pub schema_version: u32,
    /// Stable asset identity.
    pub id: AssetId,
    /// Project-relative source URI.
    pub uri: AssetUri,
    /// Asset category.
    pub kind: AssetKind,
    /// Stable content fingerprint, computed by the importer/build pipeline.
    pub content_hash: u64,
    /// Direct dependencies, in deterministic order.
    pub dependencies: BTreeSet<AssetId>,
    /// Importer identifier and version.
    pub importer: String,
}
impl AssetMetadata {
    /// Creates metadata with an identity derived from kind and URI.
    #[must_use]
    pub fn new(
        uri: AssetUri,
        kind: AssetKind,
        content_hash: u64,
        dependencies: BTreeSet<AssetId>,
        importer: impl Into<String>,
    ) -> Self {
        let id = AssetId::from_uri(kind, &uri);
        Self {
            schema_version: ASSET_METADATA_SCHEMA_VERSION,
            id,
            uri,
            kind,
            content_hash,
            dependencies,
            importer: importer.into(),
        }
    }
    /// Validates schema, identity, importer metadata, and direct self-dependencies.
    pub fn validate(&self) -> Result<(), AssetCatalogError> {
        if self.schema_version != ASSET_METADATA_SCHEMA_VERSION {
            return Err(AssetCatalogError::UnsupportedMetadataVersion(
                self.schema_version,
            ));
        }
        if self.id != AssetId::from_uri(self.kind, &self.uri) {
            return Err(AssetCatalogError::IdentityMismatch(self.id));
        }
        if self.importer.trim().is_empty() {
            return Err(AssetCatalogError::MissingImporter(self.id));
        }
        if self.dependencies.contains(&self.id) {
            return Err(AssetCatalogError::DependencyCycle(vec![self.id]));
        }
        Ok(())
    }
}

/// Registry of known assets and their direct dependencies.
#[derive(Debug, Default)]
pub struct AssetCatalog {
    entries: BTreeMap<AssetId, AssetMetadata>,
}
impl AssetCatalog {
    /// Creates an empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Registers metadata. A hash collision or conflicting duplicate is rejected.
    pub fn register(&mut self, metadata: AssetMetadata) -> Result<(), AssetCatalogError> {
        metadata.validate()?;
        if let Some(existing) = self.entries.get(&metadata.id) {
            if existing.uri != metadata.uri || existing.kind != metadata.kind {
                return Err(AssetCatalogError::IdentityCollision(metadata.id));
            }
            if existing != &metadata {
                return Err(AssetCatalogError::ConflictingMetadata(metadata.id));
            }
            return Ok(());
        }
        let id = metadata.id;
        self.entries.insert(id, metadata);
        if let Err(error) = self.validate_acyclic() {
            self.entries.remove(&id);
            return Err(error);
        }
        Ok(())
    }
    /// Looks up metadata by stable ID.
    #[must_use]
    pub fn get(&self, id: AssetId) -> Option<&AssetMetadata> {
        self.entries.get(&id)
    }
    /// Returns catalog entries in stable ID order.
    pub fn entries(&self) -> impl Iterator<Item = &AssetMetadata> {
        self.entries.values()
    }
    /// Rejects dependencies that have not been registered yet.
    pub fn validate_complete(&self) -> Result<(), AssetCatalogError> {
        for metadata in self.entries.values() {
            for dependency in &metadata.dependencies {
                if !self.entries.contains_key(dependency) {
                    return Err(AssetCatalogError::MissingDependency {
                        asset: metadata.id,
                        dependency: *dependency,
                    });
                }
            }
        }
        self.validate_acyclic()
    }
    fn validate_acyclic(&self) -> Result<(), AssetCatalogError> {
        let mut states = HashMap::<AssetId, u8>::new();
        let mut stack = Vec::new();
        for id in self.entries.keys().copied() {
            self.visit(id, &mut states, &mut stack)?;
        }
        Ok(())
    }
    fn visit(
        &self,
        id: AssetId,
        states: &mut HashMap<AssetId, u8>,
        stack: &mut Vec<AssetId>,
    ) -> Result<(), AssetCatalogError> {
        match states.get(&id).copied() {
            Some(2) => return Ok(()),
            Some(1) => {
                let start = stack.iter().position(|entry| *entry == id).unwrap_or(0);
                let mut cycle = stack[start..].to_vec();
                cycle.push(id);
                return Err(AssetCatalogError::DependencyCycle(cycle));
            }
            _ => {}
        }
        states.insert(id, 1);
        stack.push(id);
        if let Some(metadata) = self.entries.get(&id) {
            for dependency in &metadata.dependencies {
                if self.entries.contains_key(dependency) {
                    self.visit(*dependency, states, stack)?;
                }
            }
        }
        stack.pop();
        states.insert(id, 2);
        Ok(())
    }
}

/// Asset catalog validation failures.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AssetCatalogError {
    /// Serialized metadata schema is unsupported.
    #[error("unsupported asset metadata schema version {0}")]
    UnsupportedMetadataVersion(u32),
    /// Metadata ID does not match its URI and kind.
    #[error("asset metadata identity does not match its URI and kind: {0:?}")]
    IdentityMismatch(AssetId),
    /// Importer identifier must not be empty.
    #[error("asset metadata has no importer identifier: {0:?}")]
    MissingImporter(AssetId),
    /// Distinct logical assets generated the same ID.
    #[error("asset identity hash collision: {0:?}")]
    IdentityCollision(AssetId),
    /// Same identity was registered with different metadata.
    #[error("asset metadata conflicts with existing entry: {0:?}")]
    ConflictingMetadata(AssetId),
    /// A referenced dependency has not been registered.
    #[error("asset {asset:?} references missing dependency {dependency:?}")]
    MissingDependency { asset: AssetId, dependency: AssetId },
    /// The dependency graph contains a cycle.
    #[error("asset dependency cycle: {0:?}")]
    DependencyCycle(Vec<AssetId>),
}

#[cfg(test)]
mod tests {
    use super::*;
    fn metadata(uri: &str, dependencies: BTreeSet<AssetId>) -> AssetMetadata {
        AssetMetadata::new(
            AssetUri::new(uri).expect("URI"),
            AssetKind::Mesh,
            42,
            dependencies,
            "test/1",
        )
    }
    #[test]
    fn catalog_requires_complete_dependencies_at_build_validation() {
        let dependency = metadata("textures/albedo.png", BTreeSet::new()).id;
        let root = metadata("models/ship.glb", BTreeSet::from([dependency]));
        let mut catalog = AssetCatalog::new();
        catalog
            .register(root.clone())
            .expect("forward dependency is allowed during discovery");
        assert!(matches!(
            catalog.validate_complete(),
            Err(AssetCatalogError::MissingDependency { .. })
        ));
        catalog
            .register(metadata("textures/albedo.png", BTreeSet::new()))
            .expect("dependency");
        assert!(catalog.validate_complete().is_ok());
    }
    #[test]
    fn dependency_cycles_are_rejected_transactionally() {
        let a = metadata("a", BTreeSet::new());
        let b = metadata("b", BTreeSet::from([a.id]));
        let a = metadata("a", BTreeSet::from([b.id]));
        let mut catalog = AssetCatalog::new();
        catalog.register(b).expect("register first node");
        assert!(matches!(
            catalog.register(a),
            Err(AssetCatalogError::DependencyCycle(_))
        ));
        assert_eq!(catalog.entries().count(), 1);
    }
    #[test]
    fn metadata_round_trips_and_rejects_unknown_fields() {
        let metadata = metadata("models/a.glb", BTreeSet::new());
        let json = serde_json::to_string(&metadata).expect("serialize");
        assert_eq!(
            serde_json::from_str::<AssetMetadata>(&json).expect("deserialize"),
            metadata
        );
        assert!(
            serde_json::from_str::<AssetMetadata>(
                &json.replace("\"importer\":", "\"unexpected\":")
            )
            .is_err()
        );
    }
}
