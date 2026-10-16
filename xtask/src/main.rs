//! `cargo xtask <command>` entry point.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use xtask::docs::{self, Doc};
use xtask::layering;

const USAGE: &str = "usage: cargo xtask <layering | docs-config [PATH] | docs-errors [PATH]>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["layering"] => run_layering(),
        [cmd] | [cmd, _] if Doc::from_command(cmd).is_some() => run_docs(&args),
        _ => Err(USAGE.to_owned()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::FAILURE
        }
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
}

fn run_docs(args: &[String]) -> Result<(), String> {
    let doc = args
        .first()
        .and_then(|cmd| Doc::from_command(cmd))
        .ok_or_else(|| USAGE.to_owned())?;
    let path = args.get(1).map(Path::new);
    docs::emit(&doc.markdown(), path, &mut std::io::stdout().lock())
}

fn run_layering() -> Result<(), String> {
    let root = workspace_root();
    let map_text = std::fs::read_to_string(root.join("xtask/layers.toml"))
        .map_err(|e| format!("cannot read xtask/layers.toml: {e}"))?;
    let map = layering::parse_map(&map_text)?;
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let out = Command::new(cargo)
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(&root)
        .output()
        .map_err(|e| format!("failed to run cargo metadata: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let json = String::from_utf8_lossy(&out.stdout);
    let errors = layering::check(&json, &map)?;
    if errors.is_empty() {
        println!("layering: OK");
        Ok(())
    } else {
        Err(errors
            .iter()
            .map(|e| format!("error: {e}"))
            .collect::<Vec<_>>()
            .join("\n"))
    }
}
