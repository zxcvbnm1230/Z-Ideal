//! Process-wide tracing subscriber initialization.

use thiserror::Error;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Logging initialization errors.
#[derive(Debug, Error)]
pub enum LoggingError {
    /// Filter syntax is invalid.
    #[error("invalid log filter: {0}")]
    InvalidFilter(#[from] tracing_subscriber::filter::ParseError),
    /// A global subscriber was already installed.
    #[error("a global tracing subscriber is already installed")]
    AlreadyInitialized,
}

/// Initializes structured process logging exactly once.
///
/// An explicit filter takes precedence over the `RUST_LOG` environment variable.
pub fn initialize_logging(filter: Option<&str>) -> Result<(), LoggingError> {
    let env_filter = match filter {
        Some(value) => EnvFilter::try_new(value)?,
        None => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
    };
    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .try_init()
        .map_err(|_| LoggingError::AlreadyInitialized)
}
