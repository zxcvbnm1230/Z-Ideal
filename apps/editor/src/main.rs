//! Z Ideal editor process entry point and asset build commands.
#![forbid(unsafe_code)]

use std::{env, ffi::OsString, path::PathBuf, process::ExitCode};

use z_ideal_assets::ScriptBuildPipeline;

enum Command {
    Launch,
    Help,
    CookScripts { source: PathBuf, output: PathBuf },
}

fn main() -> ExitCode {
    if let Err(error) = z_ideal_core::logging::initialize_logging(None) {
        eprintln!("failed to initialize logging: {error}");
        return ExitCode::FAILURE;
    }

    match parse_command(env::args_os().skip(1)) {
        Ok(Command::Help) => {
            print_help();
            ExitCode::SUCCESS
        }
        Ok(Command::CookScripts { source, output }) => {
            match ScriptBuildPipeline::default().cook(&source, &output) {
                Ok(manifest) => {
                    tracing::info!(
                        source = %source.display(),
                        output = %output.display(),
                        scripts = manifest.scripts.len(),
                        api_version = manifest.script_api_version,
                        "cooked Lua script package"
                    );
                    println!(
                        "Cooked {} Lua scripts into {}",
                        manifest.scripts.len(),
                        output.display()
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    tracing::error!(%error, "Lua script package cooking failed");
                    eprintln!("script package build failed: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Ok(Command::Launch) => {
            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                subsystem = z_ideal_editor::CRATE_NAME,
                "Z Ideal editor foundation initialized"
            );
            println!(
                "Z Ideal Editor — foundation build; editor functionality is not implemented yet."
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            eprintln!("Run z-ideal-editor-app --help for usage.");
            ExitCode::from(2)
        }
    }
}

fn parse_command(args: impl IntoIterator<Item = OsString>) -> Result<Command, String> {
    let mut args = args.into_iter();
    let Some(command) = args.next() else {
        return Ok(Command::Launch);
    };
    match command.to_str() {
        Some("--help" | "-h" | "help") => Ok(Command::Help),
        Some("cook-scripts") => {
            let mut source = None;
            let mut output = None;
            while let Some(option) = args.next() {
                let option_name = option
                    .to_str()
                    .ok_or_else(|| "command options must be valid UTF-8".to_owned())?;
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {option_name}"))?;
                match option_name {
                    "--source" if source.is_none() => source = Some(PathBuf::from(value)),
                    "--output" if output.is_none() => output = Some(PathBuf::from(value)),
                    "--source" | "--output" => {
                        return Err(format!("duplicate option: {option_name}"));
                    }
                    _ => return Err(format!("unknown option: {option_name}")),
                }
            }
            let source = source.ok_or_else(|| "missing required --source".to_owned())?;
            let output = output.ok_or_else(|| "missing required --output".to_owned())?;
            Ok(Command::CookScripts { source, output })
        }
        Some(other) => Err(format!("unknown command: {other}")),
        None => Err("command name must be valid UTF-8".to_owned()),
    }
}

fn print_help() {
    println!(
        "Z Ideal Editor\n\
         Usage:\n\
           z-ideal-editor-app\n\
           z-ideal-editor-app --help\n\
           z-ideal-editor-app cook-scripts --source <directory> --output <new-directory>\n\
         \n\
         Commands:\n\
           cook-scripts  Discover, validate and cook Lua sources into a versioned package.\n\
         \n\
         The output directory must not already exist."
    );
}

#[cfg(test)]
mod tests {
    use super::{Command, parse_command};
    use std::{ffi::OsString, path::Path};

    fn args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    #[test]
    fn no_arguments_launches_editor_foundation() {
        assert!(matches!(parse_command([]), Ok(Command::Launch)));
    }

    #[test]
    fn parses_help_aliases() {
        assert!(matches!(
            parse_command(args(&["--help"])),
            Ok(Command::Help)
        ));
        assert!(matches!(parse_command(args(&["-h"])), Ok(Command::Help)));
        assert!(matches!(parse_command(args(&["help"])), Ok(Command::Help)));
    }

    #[test]
    fn parses_script_cooking_paths() {
        assert!(matches!(
            parse_command(args(&[
                "cook-scripts",
                "--source",
                "assets/scripts",
                "--output",
                "build/scripts"
            ])),
            Ok(Command::CookScripts { source, output })
                if source == Path::new("assets/scripts")
                    && output == Path::new("build/scripts")
        ));
    }

    #[test]
    fn rejects_missing_duplicate_and_unknown_options() {
        assert!(parse_command(args(&["cook-scripts", "--source", "scripts"])).is_err());
        assert!(
            parse_command(args(&[
                "cook-scripts",
                "--source",
                "a",
                "--source",
                "b",
                "--output",
                "out"
            ]))
            .is_err()
        );
        assert!(
            parse_command(args(&[
                "cook-scripts",
                "--source",
                "a",
                "--output",
                "out",
                "--force"
            ]))
            .is_err()
        );
        assert!(parse_command(args(&["not-a-command"])).is_err());
    }
}
