//! Deterministic filesystem discovery and transactional cooking for Lua scripts.
use crate::{
    AssetCatalog, AssetCatalogError, AssetMetadata, AssetUri, LuaScriptImporter,
    SCRIPT_ASSET_API_VERSION, ScriptAssetImportError,
};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

/// Serialized schema version for a cooked script package manifest.
pub const SCRIPT_BUILD_MANIFEST_VERSION: u32 = 1;
/// Name of the manifest written to a cooked script package.
pub const SCRIPT_BUILD_MANIFEST_FILE: &str = "manifest.json";
/// Maximum serialized script package manifest size.
pub const MAX_SCRIPT_BUILD_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
/// Maximum number of Lua source files in one package.
pub const MAX_SCRIPT_PACKAGE_SCRIPTS: usize = 16_384;
/// Maximum combined Lua source bytes in one package.
pub const MAX_SCRIPT_PACKAGE_SOURCE_BYTES: usize = 256 * 1024 * 1024;
static TEMP_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// One script entry in a cooked package manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CookedScriptEntry {
    /// Validated imported asset metadata.
    pub metadata: AssetMetadata,
    /// Script API version declared by source.
    pub api_version: u32,
    /// Package-relative path of the copied Lua source.
    pub source_file: String,
}

/// Deterministic index of all Lua scripts in one cooked package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScriptBuildManifest {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Script API version supported by this package.
    pub script_api_version: u32,
    /// Scripts sorted by logical URI.
    pub scripts: Vec<CookedScriptEntry>,
}

impl ScriptBuildManifest {
    /// Validates manifest versions, source paths, metadata and dependency closure.
    pub fn validate(&self) -> Result<(), ScriptBuildError> {
        if self.schema_version != SCRIPT_BUILD_MANIFEST_VERSION {
            return Err(ScriptBuildError::UnsupportedManifestVersion {
                found: self.schema_version,
                supported: SCRIPT_BUILD_MANIFEST_VERSION,
            });
        }
        if self.script_api_version != SCRIPT_ASSET_API_VERSION {
            return Err(ScriptBuildError::UnsupportedScriptApiVersion {
                found: self.script_api_version,
                supported: SCRIPT_ASSET_API_VERSION,
            });
        }
        if self.scripts.len() > MAX_SCRIPT_PACKAGE_SCRIPTS {
            return Err(ScriptBuildError::TooManyScripts {
                actual: self.scripts.len(),
                maximum: MAX_SCRIPT_PACKAGE_SCRIPTS,
            });
        }
        let mut previous_uri: Option<&str> = None;
        let mut catalog = AssetCatalog::new();
        for entry in &self.scripts {
            if entry.metadata.kind != crate::AssetKind::Script
                || entry.metadata.importer != crate::LUA_SCRIPT_IMPORTER_ID
            {
                return Err(ScriptBuildError::InvalidScriptMetadata(
                    entry.metadata.uri.clone(),
                ));
            }
            if entry.api_version != self.script_api_version {
                return Err(ScriptBuildError::EntryApiVersionMismatch {
                    uri: entry.metadata.uri.clone(),
                    found: entry.api_version,
                    expected: self.script_api_version,
                });
            }
            let expected_path = format!("scripts/{}", entry.metadata.uri.as_str());
            validate_package_relative_path(&entry.source_file)?;
            if entry.source_file != expected_path {
                return Err(ScriptBuildError::InvalidSourcePath {
                    uri: entry.metadata.uri.clone(),
                    path: entry.source_file.clone(),
                });
            }
            if let Some(previous) = previous_uri
                && previous >= entry.metadata.uri.as_str()
            {
                return Err(ScriptBuildError::UnsortedOrDuplicateUri(
                    entry.metadata.uri.clone(),
                ));
            }
            previous_uri = Some(entry.metadata.uri.as_str());
            catalog.register(entry.metadata.clone())?;
        }
        catalog.validate_complete()?;
        Ok(())
    }

    /// Loads and validates a manifest from a cooked package directory.
    pub fn load(package_directory: impl AsRef<Path>) -> Result<Self, ScriptBuildError> {
        let package = package_directory.as_ref();
        let package_metadata =
            fs::symlink_metadata(package).map_err(|source| ScriptBuildError::Io {
                path: package.to_path_buf(),
                source,
            })?;
        if package_metadata.file_type().is_symlink() {
            return Err(ScriptBuildError::SymlinkNotAllowed(package.to_path_buf()));
        }
        if !package_metadata.is_dir() {
            return Err(ScriptBuildError::NotDirectory(package.to_path_buf()));
        }
        let canonical_root = fs::canonicalize(package).map_err(|source| ScriptBuildError::Io {
            path: package.to_path_buf(),
            source,
        })?;
        let path = package.join(SCRIPT_BUILD_MANIFEST_FILE);
        let file_metadata = fs::symlink_metadata(&path).map_err(|source| ScriptBuildError::Io {
            path: path.clone(),
            source,
        })?;
        if file_metadata.file_type().is_symlink() {
            return Err(ScriptBuildError::SymlinkNotAllowed(path));
        }
        if !file_metadata.is_file() {
            return Err(ScriptBuildError::NotRegularFile(path));
        }
        let canonical_path = fs::canonicalize(&path).map_err(|source| ScriptBuildError::Io {
            path: path.clone(),
            source,
        })?;
        if !canonical_path.starts_with(&canonical_root) {
            return Err(ScriptBuildError::PathEscapesRoot(path));
        }
        let bytes = read_bounded_file(&canonical_path, MAX_SCRIPT_BUILD_MANIFEST_BYTES)?;
        let manifest: Self =
            serde_json::from_slice(&bytes).map_err(|source| ScriptBuildError::InvalidManifest {
                path: canonical_path.clone(),
                source,
            })?;
        manifest.validate()?;
        manifest.verify_sources(package_directory)?;
        Ok(manifest)
    }

    /// Loads one source asset from the package after rechecking its recorded metadata.
    pub fn load_script_asset(
        &self,
        package_directory: impl AsRef<Path>,
        uri: &AssetUri,
    ) -> Result<crate::ScriptAsset, ScriptBuildError> {
        self.validate()?;
        let entry = self
            .scripts
            .iter()
            .find(|entry| &entry.metadata.uri == uri)
            .ok_or_else(|| ScriptBuildError::UnknownScript(uri.clone()))?;
        let package = package_directory.as_ref();
        let canonical_root = canonical_package_root(package)?;
        let path = package.join(&entry.source_file);
        let metadata = fs::symlink_metadata(&path).map_err(|source| ScriptBuildError::Io {
            path: path.clone(),
            source,
        })?;
        if metadata.file_type().is_symlink() {
            return Err(ScriptBuildError::SymlinkNotAllowed(path));
        }
        if !metadata.is_file() {
            return Err(ScriptBuildError::NotRegularFile(path));
        }
        let canonical_path = fs::canonicalize(&path).map_err(|source| ScriptBuildError::Io {
            path: path.clone(),
            source,
        })?;
        if !canonical_path.starts_with(&canonical_root) {
            return Err(ScriptBuildError::PathEscapesRoot(path));
        }
        let bytes = read_bounded_file(&canonical_path, crate::MAX_SCRIPT_ASSET_BYTES)?;
        let asset = LuaScriptImporter::default()
            .import(uri.clone(), &bytes)
            .map_err(|source| ScriptBuildError::Import {
                path: canonical_path.clone(),
                source,
            })?;
        if asset.metadata() != entry.metadata || asset.api_version() != entry.api_version {
            return Err(ScriptBuildError::SourceMetadataMismatch {
                path: canonical_path,
                uri: uri.clone(),
            });
        }
        Ok(asset)
    }

    /// Verifies packaged source files against manifest metadata and rejects symlinks.
    pub fn verify_sources(
        &self,
        package_directory: impl AsRef<Path>,
    ) -> Result<(), ScriptBuildError> {
        self.validate()?;
        canonical_package_root(package_directory.as_ref())?;
        let mut total_source_bytes = 0_usize;
        for entry in &self.scripts {
            let asset = self.load_script_asset(package_directory.as_ref(), &entry.metadata.uri)?;
            total_source_bytes = total_source_bytes.checked_add(asset.source().len()).ok_or(
                ScriptBuildError::PackageSourceTooLarge {
                    maximum: MAX_SCRIPT_PACKAGE_SOURCE_BYTES,
                },
            )?;
            if total_source_bytes > MAX_SCRIPT_PACKAGE_SOURCE_BYTES {
                return Err(ScriptBuildError::PackageSourceTooLarge {
                    maximum: MAX_SCRIPT_PACKAGE_SOURCE_BYTES,
                });
            }
        }
        Ok(())
    }
}

/// Discovers and cooks Lua scripts rooted at a project script directory.
#[derive(Debug, Clone, Default)]
pub struct ScriptBuildPipeline {
    importer: LuaScriptImporter,
}

impl ScriptBuildPipeline {
    /// Creates a pipeline with a stricter source size limit.
    pub fn new(importer: LuaScriptImporter) -> Self {
        Self { importer }
    }

    /// Discovers all .lua files recursively, rejecting symlinks and invalid paths.
    ///
    /// The returned assets are sorted by their normalized project-relative URI.
    pub fn discover(
        &self,
        scripts_directory: impl AsRef<Path>,
    ) -> Result<Vec<crate::ScriptAsset>, ScriptBuildError> {
        let root = scripts_directory.as_ref();
        let metadata = fs::symlink_metadata(root).map_err(|source| ScriptBuildError::Io {
            path: root.to_path_buf(),
            source,
        })?;
        if metadata.file_type().is_symlink() {
            return Err(ScriptBuildError::SymlinkNotAllowed(root.to_path_buf()));
        }
        if !metadata.is_dir() {
            return Err(ScriptBuildError::NotDirectory(root.to_path_buf()));
        }
        let canonical_root = fs::canonicalize(root).map_err(|source| ScriptBuildError::Io {
            path: root.to_path_buf(),
            source,
        })?;
        let mut files = Vec::new();
        collect_lua_files(&canonical_root, &canonical_root, &mut files)?;
        files.sort_by(|left, right| left.0.cmp(&right.0));

        if files.len() > MAX_SCRIPT_PACKAGE_SCRIPTS {
            return Err(ScriptBuildError::TooManyScripts {
                actual: files.len(),
                maximum: MAX_SCRIPT_PACKAGE_SCRIPTS,
            });
        }
        let mut assets = Vec::with_capacity(files.len());
        let mut total_source_bytes = 0_usize;
        for (uri, path) in files {
            let bytes = read_bounded_file(&path, self.importer.max_source_bytes())?;
            total_source_bytes = total_source_bytes.checked_add(bytes.len()).ok_or(
                ScriptBuildError::PackageSourceTooLarge {
                    maximum: MAX_SCRIPT_PACKAGE_SOURCE_BYTES,
                },
            )?;
            if total_source_bytes > MAX_SCRIPT_PACKAGE_SOURCE_BYTES {
                return Err(ScriptBuildError::PackageSourceTooLarge {
                    maximum: MAX_SCRIPT_PACKAGE_SOURCE_BYTES,
                });
            }
            let uri = AssetUri::new(uri).map_err(|source| ScriptBuildError::InvalidAssetUri {
                path: path.clone(),
                source,
            })?;
            let asset =
                self.importer
                    .import(uri, &bytes)
                    .map_err(|source| ScriptBuildError::Import {
                        path: path.clone(),
                        source,
                    })?;
            assets.push(asset);
        }
        Ok(assets)
    }

    /// Builds a complete cooked package at a new destination directory.
    ///
    /// The destination must not already exist. Work is performed in a unique
    /// sibling staging directory; the completed directory is published with a
    /// single rename. A failed build removes its staging directory and leaves
    /// the requested destination absent.
    pub fn cook(
        &self,
        scripts_directory: impl AsRef<Path>,
        package_directory: impl AsRef<Path>,
    ) -> Result<ScriptBuildManifest, ScriptBuildError> {
        let assets = self.discover(&scripts_directory)?;
        let destination = package_directory.as_ref();
        if destination.exists() || fs::symlink_metadata(destination).is_ok() {
            return Err(ScriptBuildError::DestinationExists(
                destination.to_path_buf(),
            ));
        }
        let parent = destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|source| ScriptBuildError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
        let canonical_source_root =
            fs::canonicalize(scripts_directory.as_ref()).map_err(|source| {
                ScriptBuildError::Io {
                    path: scripts_directory.as_ref().to_path_buf(),
                    source,
                }
            })?;
        let canonical_parent = fs::canonicalize(parent).map_err(|source| ScriptBuildError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
        let candidate = canonical_parent.join(destination.file_name().ok_or_else(|| {
            ScriptBuildError::InvalidPackagePath(destination.display().to_string())
        })?);
        if candidate.starts_with(&canonical_source_root) {
            return Err(ScriptBuildError::OutputInsideSourceTree {
                source_root: canonical_source_root,
                destination: candidate,
            });
        }

        let staging =
            create_staging_directory(parent, destination.file_name().unwrap_or_default())?;
        let result = write_package(&staging, assets);
        let manifest = match result {
            Ok(manifest) => manifest,
            Err(error) => {
                let _ = fs::remove_dir_all(&staging);
                return Err(error);
            }
        };
        if destination.exists() || fs::symlink_metadata(destination).is_ok() {
            let _ = fs::remove_dir_all(&staging);
            return Err(ScriptBuildError::DestinationExists(
                destination.to_path_buf(),
            ));
        }
        if let Err(source) = fs::rename(&staging, destination) {
            let _ = fs::remove_dir_all(&staging);
            return Err(ScriptBuildError::Io {
                path: destination.to_path_buf(),
                source,
            });
        }
        Ok(manifest)
    }
}

fn canonical_package_root(package: &Path) -> Result<PathBuf, ScriptBuildError> {
    let metadata = fs::symlink_metadata(package).map_err(|source| ScriptBuildError::Io {
        path: package.to_path_buf(),
        source,
    })?;
    if metadata.file_type().is_symlink() {
        return Err(ScriptBuildError::SymlinkNotAllowed(package.to_path_buf()));
    }
    if !metadata.is_dir() {
        return Err(ScriptBuildError::NotDirectory(package.to_path_buf()));
    }
    fs::canonicalize(package).map_err(|source| ScriptBuildError::Io {
        path: package.to_path_buf(),
        source,
    })
}

fn collect_lua_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<(String, PathBuf)>,
) -> Result<(), ScriptBuildError> {
    let directory_metadata =
        fs::symlink_metadata(directory).map_err(|source| ScriptBuildError::Io {
            path: directory.to_path_buf(),
            source,
        })?;
    if directory_metadata.file_type().is_symlink() {
        return Err(ScriptBuildError::SymlinkNotAllowed(directory.to_path_buf()));
    }
    let canonical_directory =
        fs::canonicalize(directory).map_err(|source| ScriptBuildError::Io {
            path: directory.to_path_buf(),
            source,
        })?;
    if !canonical_directory.starts_with(root) {
        return Err(ScriptBuildError::PathEscapesRoot(directory.to_path_buf()));
    }
    let mut entries = fs::read_dir(&canonical_directory)
        .map_err(|source| ScriptBuildError::Io {
            path: canonical_directory.clone(),
            source,
        })?
        .map(|entry| {
            entry.map_err(|source| ScriptBuildError::Io {
                path: canonical_directory.clone(),
                source,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(|source| ScriptBuildError::Io {
            path: path.clone(),
            source,
        })?;
        if metadata.file_type().is_symlink() {
            return Err(ScriptBuildError::SymlinkNotAllowed(path));
        }
        if metadata.is_dir() {
            let canonical_path =
                fs::canonicalize(&path).map_err(|source| ScriptBuildError::Io {
                    path: path.clone(),
                    source,
                })?;
            if !canonical_path.starts_with(root) {
                return Err(ScriptBuildError::PathEscapesRoot(path));
            }
            collect_lua_files(root, &canonical_path, files)?;
        } else if metadata.is_file()
            && path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("lua"))
        {
            let canonical_path =
                fs::canonicalize(&path).map_err(|source| ScriptBuildError::Io {
                    path: path.clone(),
                    source,
                })?;
            if !canonical_path.starts_with(root) {
                return Err(ScriptBuildError::PathEscapesRoot(path));
            }
            let relative = canonical_path
                .strip_prefix(root)
                .map_err(|_| ScriptBuildError::PathEscapesRoot(canonical_path.clone()))?;
            let uri = path_to_uri(relative)
                .ok_or_else(|| ScriptBuildError::NonUtf8Path(canonical_path.clone()))?;
            files.push((uri, canonical_path));
            if files.len() > MAX_SCRIPT_PACKAGE_SCRIPTS {
                return Err(ScriptBuildError::TooManyScripts {
                    actual: files.len(),
                    maximum: MAX_SCRIPT_PACKAGE_SCRIPTS,
                });
            }
        }
    }
    Ok(())
}

fn path_to_uri(path: &Path) -> Option<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(value) => parts.push(value.to_str()?),
            _ => return None,
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("/"))
    }
}

fn validate_package_relative_path(path: &str) -> Result<(), ScriptBuildError> {
    let parsed = Path::new(path);
    if parsed.is_absolute()
        || parsed
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || path.contains('\\')
        || path.contains('\0')
    {
        return Err(ScriptBuildError::InvalidPackagePath(path.to_owned()));
    }
    Ok(())
}

fn create_staging_directory(
    parent: &Path,
    name: &std::ffi::OsStr,
) -> Result<PathBuf, ScriptBuildError> {
    for _ in 0..128 {
        let sequence = TEMP_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let staging = parent.join(format!(
            ".{}.zideal-staging-{}-{sequence}",
            name.to_string_lossy(),
            std::process::id()
        ));
        match fs::create_dir(&staging) {
            Ok(()) => return Ok(staging),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(ScriptBuildError::Io {
                    path: staging,
                    source,
                });
            }
        }
    }
    Err(ScriptBuildError::StagingNameExhausted(parent.to_path_buf()))
}

fn write_package(
    staging: &Path,
    assets: Vec<crate::ScriptAsset>,
) -> Result<ScriptBuildManifest, ScriptBuildError> {
    let mut entries = Vec::with_capacity(assets.len());
    for asset in assets {
        let relative = format!("scripts/{}", asset.uri().as_str());
        validate_package_relative_path(&relative)?;
        let output = staging.join(&relative);
        let parent = output
            .parent()
            .ok_or_else(|| ScriptBuildError::InvalidPackagePath(relative.clone()))?;
        fs::create_dir_all(parent).map_err(|source| ScriptBuildError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
        write_new_file(&output, asset.source().as_bytes())?;
        entries.push(CookedScriptEntry {
            metadata: asset.metadata(),
            api_version: asset.api_version(),
            source_file: relative,
        });
    }
    entries.sort_by(|left, right| left.metadata.uri.cmp(&right.metadata.uri));
    let manifest = ScriptBuildManifest {
        schema_version: SCRIPT_BUILD_MANIFEST_VERSION,
        script_api_version: SCRIPT_ASSET_API_VERSION,
        scripts: entries,
    };
    manifest.validate()?;
    let manifest_path = staging.join(SCRIPT_BUILD_MANIFEST_FILE);
    let bytes =
        serde_json::to_vec_pretty(&manifest).map_err(ScriptBuildError::SerializeManifest)?;
    write_new_file(&manifest_path, &bytes)?;
    Ok(manifest)
}

fn read_bounded_file(path: &Path, maximum: usize) -> Result<Vec<u8>, ScriptBuildError> {
    let file = fs::File::open(path).map_err(|source| ScriptBuildError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut bytes = Vec::new();
    let read_limit = u64::try_from(maximum).unwrap_or(u64::MAX).saturating_add(1);
    file.take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|source| ScriptBuildError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() > maximum {
        return Err(ScriptBuildError::FileTooLarge {
            path: path.to_path_buf(),
            actual_at_least: bytes.len(),
            maximum,
        });
    }
    Ok(bytes)
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), ScriptBuildError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| ScriptBuildError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    file.write_all(bytes)
        .map_err(|source| ScriptBuildError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    file.sync_all().map_err(|source| ScriptBuildError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

/// Filesystem discovery and script cooking errors.
#[derive(Debug, thiserror::Error)]
pub enum ScriptBuildError {
    /// Filesystem operation failed.
    #[error("script build filesystem error at {path}: {source}")]
    Io {
        /// Path associated with the failed operation.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// Input root is not a directory.
    #[error("script source root is not a directory: {0}")]
    NotDirectory(PathBuf),
    /// Symbolic links are rejected to keep discovery within the selected root.
    #[error("symbolic links are not allowed in script source trees: {0}")]
    SymlinkNotAllowed(PathBuf),
    /// Discovered path escaped the canonical source root.
    #[error("script path escaped source root: {0}")]
    PathEscapesRoot(PathBuf),
    /// A discovered path cannot be represented as a UTF-8 asset URI.
    #[error("script path is not valid UTF-8: {0}")]
    NonUtf8Path(PathBuf),
    /// The discovered project contains too many Lua files.
    #[error("script package contains {actual} scripts; maximum is {maximum}")]
    TooManyScripts {
        /// Discovered script count.
        actual: usize,
        /// Maximum accepted count.
        maximum: usize,
    },
    /// Combined source size exceeds the package limit.
    #[error("combined script source exceeds package limit of {maximum} bytes")]
    PackageSourceTooLarge {
        /// Maximum aggregate source size.
        maximum: usize,
    },
    /// File exceeds its configured bound before parsing or import.
    #[error("file at {path} is at least {actual_at_least} bytes; maximum is {maximum}")]
    FileTooLarge {
        /// Discovered source path.
        path: PathBuf,
        /// Bytes read up to the bounded over-limit sentinel.
        actual_at_least: usize,
        /// Configured maximum.
        maximum: usize,
    },
    /// A discovered path cannot form a valid logical asset URI.
    #[error("invalid asset URI for script at {path}: {source}")]
    InvalidAssetUri {
        /// Discovered source path.
        path: PathBuf,
        /// URI validation failure.
        #[source]
        source: crate::AssetIdentityError,
    },
    /// A Lua source file failed bounded import.
    #[error("failed to import script at {path}: {source}")]
    Import {
        /// Discovered source path.
        path: PathBuf,
        /// Import failure.
        #[source]
        source: ScriptAssetImportError,
    },
    /// A cooked manifest schema version is unsupported.
    #[error("unsupported script manifest version {found}; supported version is {supported}")]
    UnsupportedManifestVersion {
        /// Found manifest schema.
        found: u32,
        /// Supported manifest schema.
        supported: u32,
    },
    /// A cooked manifest requires an unsupported script API.
    #[error("unsupported script API version {found}; supported version is {supported}")]
    UnsupportedScriptApiVersion {
        /// Found API version.
        found: u32,
        /// Supported API version.
        supported: u32,
    },
    /// Manifest entry does not describe a Lua script imported by this importer.
    #[error("manifest entry is not valid Lua script metadata: {0}")]
    InvalidScriptMetadata(AssetUri),
    /// Manifest entry API version does not match package API version.
    #[error("script {uri} declares API version {found}; package expects {expected}")]
    EntryApiVersionMismatch {
        /// Script URI.
        uri: AssetUri,
        /// Found API version.
        found: u32,
        /// Expected API version.
        expected: u32,
    },
    /// Manifest script paths must be exactly scripts followed by the validated logical URI.
    #[error("invalid cooked source path {path:?} for script {uri}")]
    InvalidSourcePath {
        /// Logical source URI.
        uri: AssetUri,
        /// Cooked source path.
        path: String,
    },
    /// A package source path does not identify a regular file.
    #[error("cooked script source is not a regular file: {0}")]
    NotRegularFile(PathBuf),
    /// Source bytes do not match the manifest's hash or API metadata.
    #[error("cooked script source does not match manifest metadata at {path} ({uri})")]
    SourceMetadataMismatch { path: PathBuf, uri: AssetUri },
    /// Package paths must be relative and normalized.
    #[error("invalid package-relative path: {0:?}")]
    InvalidPackagePath(String),
    /// Entries must be sorted and unique by URI.
    #[error("script manifest entries are unsorted or duplicated: {0}")]
    UnsortedOrDuplicateUri(AssetUri),
    /// Requested logical URI is not present in the package manifest.
    #[error("script asset is not present in cooked package: {0}")]
    UnknownScript(AssetUri),
    /// Output path is inside the source tree and would be rediscovered as input.
    #[error("script package output {destination} is inside source tree {source_root}")]
    OutputInsideSourceTree {
        /// Canonical source directory.
        source_root: PathBuf,
        /// Canonical destination candidate.
        destination: PathBuf,
    },
    /// Destination already exists; cooking never intentionally overwrites a package.
    #[error("script package destination already exists: {0}")]
    DestinationExists(PathBuf),
    /// A unique staging directory could not be allocated.
    #[error("could not allocate a unique staging directory under {0}")]
    StagingNameExhausted(PathBuf),
    /// Manifest serialization failed.
    #[error("failed to serialize script manifest: {0}")]
    SerializeManifest(#[source] serde_json::Error),
    /// Manifest parsing failed.
    #[error("failed to parse script manifest at {path}: {source}")]
    InvalidManifest {
        /// Manifest path.
        path: PathBuf,
        /// Parsing error.
        #[source]
        source: serde_json::Error,
    },
    /// Asset catalog rejected the cooked manifest.
    #[error(transparent)]
    Catalog(#[from] AssetCatalogError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestTree(PathBuf);

    impl TestTree {
        fn new() -> Self {
            let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "z-ideal-script-build-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create test tree");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn discovers_lua_recursively_in_deterministic_uri_order() {
        let tree = TestTree::new();
        let root = tree.path().join("source");
        fs::create_dir_all(root.join("nested")).expect("nested directory");
        fs::write(root.join("z.lua"), "return {}").expect("write z");
        fs::write(root.join("nested/a.LUA"), "return {}").expect("write nested");
        fs::write(root.join("ignored.txt"), "not lua").expect("write ignored");

        let assets = ScriptBuildPipeline::default()
            .discover(&root)
            .expect("discover");
        assert_eq!(assets.len(), 2);
        assert_eq!(assets[0].uri().as_str(), "nested/a.LUA");
        assert_eq!(assets[1].uri().as_str(), "z.lua");
    }

    #[test]
    fn cooks_manifest_and_source_files_transactionally() {
        let tree = TestTree::new();
        let root = tree.path().join("source");
        fs::create_dir_all(root.join("nested")).expect("nested directory");
        fs::write(
            root.join("nested/player.lua"),
            "-- z-ideal-script-api: 1\nreturn {}",
        )
        .expect("write script");
        fs::write(root.join("helper.lua"), "return {}").expect("write helper");
        let package = tree.path().join("build").join("scripts");

        let manifest = ScriptBuildPipeline::default()
            .cook(&root, &package)
            .expect("cook");
        assert_eq!(manifest.scripts.len(), 2);
        assert_eq!(manifest.scripts[0].metadata.uri.as_str(), "helper.lua");
        assert_eq!(
            fs::read_to_string(package.join("scripts/nested/player.lua")).expect("cooked source"),
            "-- z-ideal-script-api: 1\nreturn {}"
        );
        let loaded = ScriptBuildManifest::load(&package).expect("load manifest");
        assert_eq!(loaded, manifest);
        let uri = AssetUri::new("nested/player.lua").expect("script URI");
        let asset = loaded
            .load_script_asset(&package, &uri)
            .expect("load cooked script source");
        assert_eq!(asset.uri(), &uri);
        assert_eq!(asset.source(), "-- z-ideal-script-api: 1\nreturn {}");
        assert!(matches!(
            loaded.load_script_asset(
                &package,
                &AssetUri::new("missing.lua").expect("missing URI")
            ),
            Err(ScriptBuildError::UnknownScript(_))
        ));
        assert!(matches!(
            ScriptBuildPipeline::default().cook(&root, &package),
            Err(ScriptBuildError::DestinationExists(_))
        ));
    }

    #[test]
    fn enforces_custom_per_file_source_limit_before_import() {
        let tree = TestTree::new();
        let root = tree.path().join("source");
        fs::create_dir_all(&root).expect("source root");
        fs::write(root.join("large.lua"), "12345").expect("large script");
        let importer = LuaScriptImporter::new(4).expect("small importer");
        let pipeline = ScriptBuildPipeline::new(importer);
        assert!(matches!(
            pipeline.discover(&root),
            Err(ScriptBuildError::FileTooLarge {
                maximum: 4,
                actual_at_least: 5,
                ..
            })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_entries() {
        let tree = TestTree::new();
        let root = tree.path().join("source");
        let outside = tree.path().join("outside.lua");
        fs::create_dir_all(&root).expect("source root");
        fs::write(&outside, "return {}").expect("outside script");
        std::os::unix::fs::symlink(&outside, root.join("escape.lua")).expect("create symlink");
        assert!(matches!(
            ScriptBuildPipeline::default().discover(&root),
            Err(ScriptBuildError::SymlinkNotAllowed(_))
        ));
    }

    #[test]
    fn rejects_non_directory_roots() {
        let tree = TestTree::new();
        let file = tree.path().join("source.lua");
        fs::write(&file, "return {}").expect("source file");
        assert!(matches!(
            ScriptBuildPipeline::default().discover(&file),
            Err(ScriptBuildError::NotDirectory(_))
        ));
    }

    #[test]
    fn rejects_cooking_output_inside_source_tree() {
        let tree = TestTree::new();
        let root = tree.path().join("source");
        fs::create_dir_all(&root).expect("source root");
        fs::write(root.join("a.lua"), "return {}").expect("source");
        let package = root.join("cooked");
        assert!(matches!(
            ScriptBuildPipeline::default().cook(&root, &package),
            Err(ScriptBuildError::OutputInsideSourceTree { .. })
        ));
        assert!(!package.exists());
    }

    #[test]
    fn rejects_invalid_script_and_preserves_absent_destination() {
        let tree = TestTree::new();
        let root = tree.path().join("source");
        fs::create_dir_all(&root).expect("source root");
        fs::write(root.join("invalid.lua"), b"\xff").expect("invalid bytes");
        let package = tree.path().join("package");
        assert!(matches!(
            ScriptBuildPipeline::default().cook(&root, &package),
            Err(ScriptBuildError::Import { .. })
        ));
        assert!(!package.exists());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_cooked_source_and_manifest() {
        let tree = TestTree::new();
        let root = tree.path().join("source");
        fs::create_dir_all(&root).expect("source root");
        fs::write(root.join("a.lua"), "return {}").expect("source");
        let package = tree.path().join("package");
        ScriptBuildPipeline::default()
            .cook(&root, &package)
            .expect("cook");
        let outside = tree.path().join("outside.lua");
        fs::write(&outside, "return {}").expect("outside");
        fs::remove_file(package.join("scripts/a.lua")).expect("remove source");
        std::os::unix::fs::symlink(&outside, package.join("scripts/a.lua"))
            .expect("source symlink");
        assert!(matches!(
            ScriptBuildManifest::load(&package),
            Err(ScriptBuildError::SymlinkNotAllowed(_))
        ));

        let package2 = tree.path().join("package2");
        ScriptBuildPipeline::default()
            .cook(&root, &package2)
            .expect("cook second");
        let outside_manifest = tree.path().join("manifest.json");
        fs::copy(package2.join(SCRIPT_BUILD_MANIFEST_FILE), &outside_manifest)
            .expect("copy manifest");
        fs::remove_file(package2.join(SCRIPT_BUILD_MANIFEST_FILE)).expect("remove manifest");
        std::os::unix::fs::symlink(&outside_manifest, package2.join(SCRIPT_BUILD_MANIFEST_FILE))
            .expect("manifest symlink");
        assert!(matches!(
            ScriptBuildManifest::load(&package2),
            Err(ScriptBuildError::SymlinkNotAllowed(_))
        ));
    }

    #[test]
    fn rejects_cooked_source_tampering() {
        let tree = TestTree::new();
        let root = tree.path().join("source");
        fs::create_dir_all(&root).expect("source root");
        fs::write(root.join("a.lua"), "return {}").expect("source");
        let package = tree.path().join("package");
        ScriptBuildPipeline::default()
            .cook(&root, &package)
            .expect("cook");
        fs::write(package.join("scripts/a.lua"), "return { changed = true }").expect("tamper");
        assert!(matches!(
            ScriptBuildManifest::load(&package),
            Err(ScriptBuildError::SourceMetadataMismatch { .. })
        ));
    }

    #[test]
    fn validates_manifest_paths_and_order() {
        let tree = TestTree::new();
        let root = tree.path().join("source");
        fs::create_dir_all(&root).expect("source root");
        fs::write(root.join("a.lua"), "return {}").expect("script");
        let package = tree.path().join("package");
        let mut manifest = ScriptBuildPipeline::default()
            .cook(&root, &package)
            .expect("cook");
        manifest.scripts[0].source_file = "../escape.lua".to_owned();
        assert!(matches!(
            manifest.validate(),
            Err(ScriptBuildError::InvalidPackagePath(_))
        ));
        assert!(matches!(
            manifest.load_script_asset(&package, &AssetUri::new("a.lua").expect("script URI")),
            Err(ScriptBuildError::InvalidPackagePath(_))
        ));
    }
}
