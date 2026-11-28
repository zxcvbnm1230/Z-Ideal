//! Asset identity, metadata, dependency tracking, caching, and typed source importers.
#![forbid(unsafe_code)]
mod cache;
mod catalog;
mod gltf_import;
mod hdri;
mod identity;
mod script;
mod script_build;
pub use cache::{AssetCache, AssetCacheError, AssetHandle};
pub use catalog::{ASSET_METADATA_SCHEMA_VERSION, AssetCatalog, AssetCatalogError, AssetMetadata};
pub use gltf_import::{
    GltfImport, GltfImportError, GltfImportOptions, GltfImporter, ImagePixelFormat, ImportedImage,
    ImportedMaterial, ImportedMesh, ImportedNode, ImportedPrimitive, ImportedScene,
    ImportedTexture, MaterialAlphaMode, PrimitiveTopology, TextureMagFilter, TextureMinFilter,
    TextureWrapMode,
};
pub use hdri::{
    HdrEnvironmentImportError, HdrEnvironmentImportOptions, HdrEnvironmentImporter,
    HdrEnvironmentMap,
};
pub use identity::{AssetId, AssetIdentityError, AssetKind, AssetUri};
pub use script::{
    LUA_SCRIPT_IMPORTER_ID, LuaScriptImporter, MAX_SCRIPT_ASSET_BYTES, SCRIPT_ASSET_API_VERSION,
    ScriptAsset, ScriptAssetImportError,
};
pub use script_build::{
    CookedScriptEntry, MAX_SCRIPT_BUILD_MANIFEST_BYTES, MAX_SCRIPT_PACKAGE_SCRIPTS,
    MAX_SCRIPT_PACKAGE_SOURCE_BYTES, SCRIPT_BUILD_MANIFEST_FILE, SCRIPT_BUILD_MANIFEST_VERSION,
    ScriptBuildError, ScriptBuildManifest, ScriptBuildPipeline,
};
/// Stable package identifier for diagnostics and tooling.
pub const CRATE_NAME: &str = "z-ideal-assets";
