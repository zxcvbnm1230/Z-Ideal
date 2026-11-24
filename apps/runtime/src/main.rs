//! Z Ideal packaged-game process entry point.
#![forbid(unsafe_code)]

fn main() {
    if let Err(error) = z_ideal_core::logging::initialize_logging(None) {
        eprintln!("failed to initialize logging: {error}");
        std::process::exit(1);
    }
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "Z Ideal runtime foundation initialized");
    println!("Z Ideal Runtime — foundation build; game runtime functionality is not implemented yet.");
}
