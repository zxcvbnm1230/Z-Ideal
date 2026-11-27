//! Asset identity, metadata, dependency tracking, caching, and glTF 2.0 import.
#![forbid(unsafe_code)]
mod cache;
mod catalog;
mod gltf_import;
mod identity;
pub use cache::{AssetCache, AssetCacheError, AssetHandle};
pub use catalog::{ASSET_METADATA_SCHEMA_VERSION, AssetCatalog, AssetCatalogError, AssetMetadata};
pub use gltf_import::{
    GltfImport, GltfImportError, GltfImporter, ImagePixelFormat, ImportedImage, ImportedMaterial,
    ImportedMesh, ImportedPrimitive, ImportedTexture, MaterialAlphaMode, PrimitiveTopology,
    TextureMagFilter, TextureMinFilter, TextureWrapMode,
};
pub use identity::{AssetId, AssetIdentityError, AssetKind, AssetUri};
/// Stable package identifier for diagnostics and tooling.
pub const CRATE_NAME: &str = "z-ideal-assets";
