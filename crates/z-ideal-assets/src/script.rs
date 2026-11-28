//! Bounded Lua source import and versioned script asset metadata.

use crate::{AssetKind, AssetMetadata, AssetUri};
use std::{collections::BTreeSet, sync::Arc};

/// Current version of the Z Ideal Lua script API.
pub const SCRIPT_ASSET_API_VERSION: u32 = 1;
/// Maximum source size accepted by the Lua importer.
pub const MAX_SCRIPT_ASSET_BYTES: usize = 1024 * 1024;
/// Stable importer identifier recorded in asset metadata.
pub const LUA_SCRIPT_IMPORTER_ID: &str = "z-ideal-lua/1";
const API_HEADER_PREFIX: &str = "-- z-ideal-script-api:";

/// Imported Lua source with a validated URI and API version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptAsset {
    uri: AssetUri,
    source: Arc<str>,
    api_version: u32,
    content_hash: u64,
}

impl ScriptAsset {
    /// Returns the project-relative source URI.
    #[must_use]
    pub fn uri(&self) -> &AssetUri {
        &self.uri
    }

    /// Returns the original UTF-8 Lua source.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Returns the declared or default script API version.
    #[must_use]
    pub const fn api_version(&self) -> u32 {
        self.api_version
    }

    /// Returns the deterministic 64-bit source fingerprint.
    ///
    /// This non-cryptographic hash is intended for asset invalidation, not
    /// authenticity or security decisions.
    #[must_use]
    pub const fn content_hash(&self) -> u64 {
        self.content_hash
    }

    /// Creates catalog metadata for this source asset.
    #[must_use]
    pub fn metadata(&self) -> AssetMetadata {
        AssetMetadata::new(
            self.uri.clone(),
            AssetKind::Script,
            self.content_hash,
            BTreeSet::new(),
            LUA_SCRIPT_IMPORTER_ID,
        )
    }
}

/// Imports bounded UTF-8 Lua source from caller-provided asset bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LuaScriptImporter {
    max_source_bytes: usize,
}

impl Default for LuaScriptImporter {
    fn default() -> Self {
        Self {
            max_source_bytes: MAX_SCRIPT_ASSET_BYTES,
        }
    }
}

impl LuaScriptImporter {
    /// Creates an importer with a stricter per-source size limit.
    pub fn new(max_source_bytes: usize) -> Result<Self, ScriptAssetImportError> {
        if max_source_bytes == 0 || max_source_bytes > MAX_SCRIPT_ASSET_BYTES {
            return Err(ScriptAssetImportError::InvalidSizeLimit {
                requested: max_source_bytes,
                maximum: MAX_SCRIPT_ASSET_BYTES,
            });
        }
        Ok(Self { max_source_bytes })
    }

    /// Returns the configured maximum source size in bytes.
    #[must_use]
    pub const fn max_source_bytes(&self) -> usize {
        self.max_source_bytes
    }

    /// Imports a .lua asset without executing or compiling its source.
    ///
    /// An optional first-line header of the form
    /// -- z-ideal-script-api: 1 declares the script API version. If omitted,
    /// the current API version is assumed.
    pub fn import(
        &self,
        uri: AssetUri,
        bytes: &[u8],
    ) -> Result<ScriptAsset, ScriptAssetImportError> {
        if !uri
            .as_str()
            .rsplit('.')
            .next()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("lua"))
        {
            return Err(ScriptAssetImportError::UnsupportedExtension(uri));
        }
        if bytes.is_empty() {
            return Err(ScriptAssetImportError::EmptySource(uri));
        }
        if bytes.len() > self.max_source_bytes {
            return Err(ScriptAssetImportError::SourceTooLarge {
                uri,
                actual: bytes.len(),
                maximum: self.max_source_bytes,
            });
        }
        let source = std::str::from_utf8(bytes)
            .map_err(|_| ScriptAssetImportError::InvalidUtf8(uri.clone()))?;
        let api_version = parse_api_version(source, &uri)?;
        if api_version != SCRIPT_ASSET_API_VERSION {
            return Err(ScriptAssetImportError::UnsupportedApiVersion {
                uri,
                found: api_version,
                supported: SCRIPT_ASSET_API_VERSION,
            });
        }

        Ok(ScriptAsset {
            uri,
            source: Arc::from(source),
            api_version,
            content_hash: fnv1a64(bytes),
        })
    }
}

fn parse_api_version(source: &str, uri: &AssetUri) -> Result<u32, ScriptAssetImportError> {
    let first_line = source.lines().next().unwrap_or_default();
    if !first_line.starts_with("-- z-ideal-script-api") {
        return Ok(SCRIPT_ASSET_API_VERSION);
    }
    let Some(version) = first_line.strip_prefix(API_HEADER_PREFIX) else {
        return Err(ScriptAssetImportError::InvalidApiHeader(uri.clone()));
    };
    version
        .trim()
        .parse::<u32>()
        .map_err(|_| ScriptAssetImportError::InvalidApiHeader(uri.clone()))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Lua script import failures.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScriptAssetImportError {
    /// Configured size limit is zero or exceeds the hard maximum.
    #[error("invalid Lua source size limit {requested}; maximum is {maximum}")]
    InvalidSizeLimit {
        /// Requested importer limit.
        requested: usize,
        /// Hard source limit.
        maximum: usize,
    },
    /// Source URI does not end in .lua.
    #[error("Lua script URI has an unsupported extension: {0}")]
    UnsupportedExtension(AssetUri),
    /// Source file is empty.
    #[error("Lua script source is empty: {0}")]
    EmptySource(AssetUri),
    /// Source exceeds the configured size limit.
    #[error("Lua script {uri} is {actual} bytes; maximum is {maximum}")]
    SourceTooLarge {
        /// Source URI.
        uri: AssetUri,
        /// Actual source size.
        actual: usize,
        /// Configured maximum.
        maximum: usize,
    },
    /// Source is not valid UTF-8.
    #[error("Lua script source is not valid UTF-8: {0}")]
    InvalidUtf8(AssetUri),
    /// First-line API header is malformed.
    #[error("Lua script API header is malformed: {0}")]
    InvalidApiHeader(AssetUri),
    /// Source declares an API version unsupported by this importer.
    #[error("Lua script {uri} requires API version {found}; supported version is {supported}")]
    UnsupportedApiVersion {
        /// Source URI.
        uri: AssetUri,
        /// Version declared by the source.
        found: u32,
        /// Version supported by this importer.
        supported: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uri(path: &str) -> AssetUri {
        AssetUri::new(path).expect("asset URI")
    }

    #[test]
    fn imports_versioned_lua_source_and_emits_catalog_metadata() {
        let source = b"-- z-ideal-script-api: 1\nreturn { start = function() end }";
        let asset = LuaScriptImporter::default()
            .import(uri("scripts/player.lua"), source)
            .expect("import");
        assert_eq!(asset.uri().as_str(), "scripts/player.lua");
        assert_eq!(asset.source(), std::str::from_utf8(source).expect("UTF-8"));
        assert_eq!(asset.api_version(), SCRIPT_ASSET_API_VERSION);
        assert_eq!(asset.content_hash(), fnv1a64(source));

        let metadata = asset.metadata();
        assert_eq!(metadata.kind, AssetKind::Script);
        assert_eq!(metadata.uri, asset.uri().clone());
        assert_eq!(metadata.content_hash, asset.content_hash());
        assert_eq!(metadata.importer, LUA_SCRIPT_IMPORTER_ID);
        metadata.validate().expect("valid metadata");
    }

    #[test]
    fn defaults_api_version_and_hash_changes_with_source() {
        let importer = LuaScriptImporter::default();
        let first = importer
            .import(uri("scripts/a.lua"), b"return {}")
            .expect("import first");
        let second = importer
            .import(uri("scripts/b.lua"), b"return { update = function() end }")
            .expect("import second");
        assert_eq!(first.api_version(), SCRIPT_ASSET_API_VERSION);
        assert_ne!(first.content_hash(), second.content_hash());
    }

    #[test]
    fn rejects_invalid_uri_extension_empty_source_and_utf8() {
        let importer = LuaScriptImporter::default();
        assert!(matches!(
            importer.import(uri("scripts/player.txt"), b"return {}"),
            Err(ScriptAssetImportError::UnsupportedExtension(_))
        ));
        assert!(matches!(
            importer.import(uri("scripts/empty.lua"), b""),
            Err(ScriptAssetImportError::EmptySource(_))
        ));
        assert!(matches!(
            importer.import(uri("scripts/invalid.lua"), &[0xff, 0xfe]),
            Err(ScriptAssetImportError::InvalidUtf8(_))
        ));
    }

    #[test]
    fn enforces_hard_and_configured_size_limits() {
        assert!(matches!(
            LuaScriptImporter::new(0),
            Err(ScriptAssetImportError::InvalidSizeLimit { .. })
        ));
        assert!(matches!(
            LuaScriptImporter::new(MAX_SCRIPT_ASSET_BYTES + 1),
            Err(ScriptAssetImportError::InvalidSizeLimit { .. })
        ));
        let importer = LuaScriptImporter::new(4).expect("small importer");
        assert!(matches!(
            importer.import(uri("scripts/large.lua"), b"12345"),
            Err(ScriptAssetImportError::SourceTooLarge {
                actual: 5,
                maximum: 4,
                ..
            })
        ));
    }

    #[test]
    fn rejects_malformed_and_unsupported_api_versions() {
        let importer = LuaScriptImporter::default();
        assert!(matches!(
            importer.import(
                uri("scripts/bad.lua"),
                b"-- z-ideal-script-api: nope\nreturn {}"
            ),
            Err(ScriptAssetImportError::InvalidApiHeader(_))
        ));
        assert!(matches!(
            importer.import(
                uri("scripts/missing_colon.lua"),
                b"-- z-ideal-script-api 1\nreturn {}"
            ),
            Err(ScriptAssetImportError::InvalidApiHeader(_))
        ));
        assert!(matches!(
            importer.import(
                uri("scripts/new.lua"),
                b"-- z-ideal-script-api: 2\nreturn {}"
            ),
            Err(ScriptAssetImportError::UnsupportedApiVersion {
                found: 2,
                supported: 1,
                ..
            })
        ));
    }
}
