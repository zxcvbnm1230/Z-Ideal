//! Stable, deterministic asset identities and validated logical URIs.
use serde::{Deserialize, Serialize};
use std::fmt;

/// Asset category used to keep logical identifiers distinct across asset types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetKind {
    Mesh,
    Material,
    Texture,
    Scene,
    Shader,
    Audio,
    Script,
    Other,
}

impl AssetKind {
    const fn stable_name(self) -> &'static str {
        match self {
            Self::Mesh => "mesh",
            Self::Material => "material",
            Self::Texture => "texture",
            Self::Scene => "scene",
            Self::Shader => "shader",
            Self::Audio => "audio",
            Self::Script => "script",
            Self::Other => "other",
        }
    }
}

/// Validated project-relative logical asset URI.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct AssetUri(String);

impl AssetUri {
    /// Creates a URI using forward slashes and no absolute or parent traversal components.
    pub fn new(value: impl Into<String>) -> Result<Self, AssetIdentityError> {
        let value = value.into();
        if value.is_empty()
            || value.starts_with('/')
            || value.contains('\\')
            || value.contains('\0')
            || value.contains(':')
            || value.contains('?')
            || value.contains('#')
            || value
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == "..")
        {
            return Err(AssetIdentityError::InvalidUri(value));
        }
        Ok(Self(value))
    }
    /// Returns the normalized logical URI.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl TryFrom<String> for AssetUri {
    type Error = AssetIdentityError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}
impl From<AssetUri> for String {
    fn from(value: AssetUri) -> Self {
        value.0
    }
}
impl fmt::Display for AssetUri {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Stable 128-bit identifier derived from an asset kind and logical URI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AssetId(u128);
impl AssetId {
    /// Derives a stable identifier. Catalog insertion detects collisions rather than assuming
    /// the hash is globally collision-free.
    #[must_use]
    pub fn from_uri(kind: AssetKind, uri: &AssetUri) -> Self {
        let mut bytes = Vec::with_capacity(uri.0.len() + 24);
        bytes.extend_from_slice(kind.stable_name().as_bytes());
        bytes.push(b':');
        bytes.extend_from_slice(uri.0.as_bytes());
        let low = fnv1a64(0xcbf2_9ce4_8422_2325, &bytes);
        let high = fnv1a64(0x8422_2325_cbf2_9ce4, &bytes);
        Self((u128::from(high) << 64) | u128::from(low))
    }
    /// Returns the identifier as an integer.
    #[must_use]
    pub const fn as_u128(self) -> u128 {
        self.0
    }
}
fn fnv1a64(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Invalid logical asset identity input.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AssetIdentityError {
    /// URI is empty, absolute, contains parent traversal, or is not normalized.
    #[error("invalid project-relative asset URI: {0:?}")]
    InvalidUri(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ids_are_stable_and_kind_specific() {
        let uri = AssetUri::new("models/ship.glb").expect("valid URI");
        assert_eq!(
            AssetId::from_uri(AssetKind::Mesh, &uri),
            AssetId::from_uri(AssetKind::Mesh, &uri)
        );
        assert_ne!(
            AssetId::from_uri(AssetKind::Mesh, &uri),
            AssetId::from_uri(AssetKind::Texture, &uri)
        );
    }
    #[test]
    fn rejects_absolute_and_traversal_uris() {
        for value in [
            "",
            "/models/ship.glb",
            "../ship.glb",
            "models/../ship.glb",
            "models\\ship.glb",
            "models//ship.glb",
            "C:/models/ship.glb",
            "models/a?b.glb",
        ] {
            assert!(AssetUri::new(value).is_err(), "accepted {value:?}");
        }
    }
}
