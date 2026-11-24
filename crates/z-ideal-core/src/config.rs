//! Validated, versioned engine configuration.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Current configuration schema version.
pub const CONFIG_SCHEMA_VERSION: u32 = 1;

/// Engine-wide configuration loaded from a TOML document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineConfig {
    /// Schema version; unsupported versions are rejected.
    pub schema_version: u32,
    /// Logging configuration.
    #[serde(default)]
    pub logging: LoggingConfig,
}

/// Logging configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoggingConfig {
    /// Default tracing filter, e.g. "info" or "z_ideal=debug".
    pub filter: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            filter: "info".to_owned(),
        }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            schema_version: CONFIG_SCHEMA_VERSION,
            logging: LoggingConfig::default(),
        }
    }
}

/// Configuration loading and validation failures.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// TOML syntax or data type error.
    #[error("invalid configuration document: {0}")]
    Parse(#[from] toml::de::Error),
    /// Schema version is not supported.
    #[error("unsupported configuration schema version {found}; supported version is {supported}")]
    UnsupportedVersion { found: u32, supported: u32 },
}

impl EngineConfig {
    /// Parses and validates a TOML configuration document.
    ///
    /// Unknown fields are rejected rather than silently discarded.
    pub fn from_toml(source: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(source)?;
        if config.schema_version != CONFIG_SCHEMA_VERSION {
            return Err(ConfigError::UnsupportedVersion {
                found: config.schema_version,
                supported: CONFIG_SCHEMA_VERSION,
            });
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::{CONFIG_SCHEMA_VERSION, ConfigError, EngineConfig};

    #[test]
    fn default_configuration_has_current_schema() {
        assert_eq!(
            EngineConfig::default().schema_version,
            CONFIG_SCHEMA_VERSION
        );
    }

    #[test]
    fn parses_valid_configuration() {
        let config = EngineConfig::from_toml("schema_version = 1\n").expect("valid fixture");
        assert_eq!(config, EngineConfig::default());
    }

    #[test]
    fn rejects_unknown_fields() {
        let result = EngineConfig::from_toml("schema_version = 1\nunknown = true\n");
        assert!(matches!(result, Err(ConfigError::Parse(_))));
    }

    #[test]
    fn rejects_unsupported_schema_version() {
        let result = EngineConfig::from_toml("schema_version = 99\n");
        assert!(matches!(
            result,
            Err(ConfigError::UnsupportedVersion { found: 99, .. })
        ));
    }
}
