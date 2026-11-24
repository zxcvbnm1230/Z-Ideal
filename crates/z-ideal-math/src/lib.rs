//! Engine math types and coordinate conventions.
#![forbid(unsafe_code)]

/// Returns the crate's stable package name for diagnostics and tooling.
pub const CRATE_NAME: &str = "z-ideal-math";

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name_is_not_empty() {
        assert!(!super::CRATE_NAME.is_empty());
    }
}
