//! Verified cooked Lua script package access for packaged runtimes.
use std::{
    fs,
    path::{Path, PathBuf},
};

use thiserror::Error;
use z_ideal_assets::{AssetUri, ScriptAsset, ScriptBuildError, ScriptBuildManifest};

/// A loaded script package whose manifest and all source fingerprints were verified.
#[derive(Debug, Clone)]
pub struct RuntimeScriptPackage {
    directory: PathBuf,
    manifest: ScriptBuildManifest,
}

impl RuntimeScriptPackage {
    /// Opens a cooked package and verifies its manifest and every source file.
    ///
    /// The package directory must remain stable while assets are loaded. Each
    /// subsequent source load is independently re-verified against the manifest.
    pub fn open(directory: impl AsRef<Path>) -> Result<Self, RuntimeScriptPackageError> {
        let directory = directory.as_ref().to_path_buf();
        // Load through the caller path first so a symlinked package root is
        // rejected before canonicalization follows it.
        let manifest = ScriptBuildManifest::load(&directory)?;
        let canonical_directory =
            fs::canonicalize(&directory).map_err(|source| ScriptBuildError::Io {
                path: directory.clone(),
                source,
            })?;
        Ok(Self {
            directory: canonical_directory,
            manifest,
        })
    }

    /// Returns the package directory used to load this package.
    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Returns the validated manifest.
    #[must_use]
    pub const fn manifest(&self) -> &ScriptBuildManifest {
        &self.manifest
    }

    /// Returns the number of script assets in the package.
    #[must_use]
    pub fn len(&self) -> usize {
        self.manifest.scripts.len()
    }

    /// Returns whether this package contains no script assets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.manifest.scripts.is_empty()
    }

    /// Returns the sorted logical URIs in the package.
    pub fn script_uris(&self) -> impl ExactSizeIterator<Item = &AssetUri> {
        self.manifest
            .scripts
            .iter()
            .map(|entry| &entry.metadata.uri)
    }

    /// Loads one script source and re-verifies its recorded content fingerprint.
    pub fn load_asset(&self, uri: &AssetUri) -> Result<ScriptAsset, RuntimeScriptPackageError> {
        Ok(self.manifest.load_script_asset(&self.directory, uri)?)
    }
}

/// Errors opening a cooked script package or loading one of its assets.
#[derive(Debug, Error)]
pub enum RuntimeScriptPackageError {
    /// The package manifest or source files failed validation.
    #[error(transparent)]
    Build(#[from] ScriptBuildError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };
    use z_ideal_assets::ScriptBuildPipeline;

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestTree(PathBuf);

    impl TestTree {
        fn new() -> Self {
            let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "z-ideal-runtime-script-package-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create test tree");
            Self(path)
        }
    }

    impl Drop for TestTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn opens_package_and_loads_verified_script_assets_in_uri_order() {
        let tree = TestTree::new();
        let source = tree.0.join("source");
        fs::create_dir_all(source.join("nested")).expect("source directories");
        fs::write(source.join("z.lua"), "return { update = function() end }")
            .expect("write z script");
        fs::write(source.join("nested/a.lua"), "return {}").expect("write a script");
        let package_dir = tree.0.join("package");
        ScriptBuildPipeline::default()
            .cook(&source, &package_dir)
            .expect("cook package");

        let package = RuntimeScriptPackage::open(&package_dir).expect("open package");
        assert_eq!(
            package.directory(),
            fs::canonicalize(&package_dir).expect("canonical package path")
        );
        assert!(package.directory().is_absolute());
        assert_eq!(package.len(), 2);
        assert!(!package.is_empty());
        assert_eq!(
            package
                .script_uris()
                .map(AssetUri::as_str)
                .collect::<Vec<_>>(),
            ["nested/a.lua", "z.lua"]
        );
        let uri = AssetUri::new("nested/a.lua").expect("asset URI");
        let asset = package.load_asset(&uri).expect("load verified source");
        assert_eq!(asset.uri(), &uri);
        assert_eq!(asset.source(), "return {}");
    }

    #[test]
    fn rejects_source_tampering_after_package_open() {
        let tree = TestTree::new();
        let source = tree.0.join("source");
        fs::create_dir_all(&source).expect("source directory");
        fs::write(source.join("player.lua"), "return {}").expect("write source");
        let package_dir = tree.0.join("package");
        ScriptBuildPipeline::default()
            .cook(&source, &package_dir)
            .expect("cook package");
        let package = RuntimeScriptPackage::open(&package_dir).expect("open package");
        fs::write(
            package_dir.join("scripts/player.lua"),
            "return { changed = true }",
        )
        .expect("tamper source");

        assert!(matches!(
            package.load_asset(&AssetUri::new("player.lua").expect("URI")),
            Err(RuntimeScriptPackageError::Build(
                ScriptBuildError::SourceMetadataMismatch { .. }
            ))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_package_root_before_canonicalization() {
        let tree = TestTree::new();
        let source = tree.0.join("source");
        fs::create_dir_all(&source).expect("source directory");
        let package_dir = tree.0.join("package");
        ScriptBuildPipeline::default()
            .cook(&source, &package_dir)
            .expect("cook empty package");
        let alias = tree.0.join("package-alias");
        std::os::unix::fs::symlink(&package_dir, &alias).expect("package symlink");
        assert!(matches!(
            RuntimeScriptPackage::open(alias),
            Err(RuntimeScriptPackageError::Build(
                ScriptBuildError::SymlinkNotAllowed(_)
            ))
        ));
    }

    #[test]
    fn rejects_unknown_script_uris() {
        let tree = TestTree::new();
        let source = tree.0.join("source");
        fs::create_dir_all(&source).expect("source directory");
        let package_dir = tree.0.join("package");
        ScriptBuildPipeline::default()
            .cook(&source, &package_dir)
            .expect("cook empty package");
        let package = RuntimeScriptPackage::open(&package_dir).expect("open package");
        assert!(package.is_empty());
        assert!(matches!(
            package.load_asset(&AssetUri::new("missing.lua").expect("URI")),
            Err(RuntimeScriptPackageError::Build(
                ScriptBuildError::UnknownScript(_)
            ))
        ));
    }
}
