//! `cargo xtask <command>` entry point.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use xtask::layering;

fn main() -> ExitCode {
    let cmd = std::env::args().nth(1);
    let result = match cmd.as_deref() {
        Some("layering") => run_layering(),
        _ => Err("usage: cargo xtask layering".to_owned()),
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
