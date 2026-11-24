//! Z Ideal editor process entry point.
#![forbid(unsafe_code)]

fn main() {
    if let Err(error) = z_ideal_core::logging::initialize_logging(None) {
        eprintln!("failed to initialize logging: {error}");
        std::process::exit(1);
    }
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "Z Ideal editor foundation initialized");
    println!("Z Ideal Editor — foundation build; editor functionality is not implemented yet.");
}
