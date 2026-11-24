//! Compile-time build metadata exposed to diagnostics and support reports.

/// Metadata embedded by Cargo at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuildInfo {
    /// Package version.
    pub version: &'static str,
    /// Target operating system.
    pub target_os: &'static str,
    /// Target CPU architecture.
    pub target_arch: &'static str,
}

impl BuildInfo {
    /// Returns metadata for this compiled engine build.
    #[must_use]
    pub const fn current() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION"),
            target_os: std::env::consts::OS,
            target_arch: std::env::consts::ARCH,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BuildInfo;

    #[test]
    fn current_metadata_contains_version_and_target() {
        let info = BuildInfo::current();
        assert!(!info.version.is_empty());
        assert!(!info.target_os.is_empty());
        assert!(!info.target_arch.is_empty());
    }
}
