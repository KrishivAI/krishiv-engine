// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask metrics-gate`: the default embedded build stays free of
//! telemetry SDKs and encoder crates.
//!
//! The Prometheus exposition is handwritten over the metrics snapshot and
//! the OTLP exporter is not built (P2-T125 unit 2 ships Prometheus only),
//! so `cargo tree -e normal -p krishiv` (default features) must contain no
//! `opentelemetry*` crate and no Prometheus encoder dependency. The check
//! is pure ([`check_tree`]) over captured `cargo tree` output, so it is
//! unit-testable; [`run`] shells out to the real command.

use std::path::Path;
use std::process::Command;

/// Crate-name fragments forbidden in the default build's dependency tree.
pub const FORBIDDEN_FRAGMENTS: &[&str] = &["opentelemetry", "prometheus"];

/// Checks captured `cargo tree` output: `Ok` when no line names a
/// forbidden crate, otherwise the offending lines.
///
/// # Errors
/// Names every line that mentions a crate from [`FORBIDDEN_FRAGMENTS`].
pub fn check_tree(output: &str) -> Result<(), String> {
    let hits: Vec<&str> = output
        .lines()
        .filter(|line| FORBIDDEN_FRAGMENTS.iter().any(|f| line.contains(f)))
        .collect();
    if hits.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "default build depends on forbidden telemetry crates:\n{}",
            hits.join("\n")
        ))
    }
}

/// Runs `cargo tree -e normal -p krishiv --locked` in `root` and checks
/// its output with [`check_tree`].
///
/// # Errors
/// Returns the cargo failure or the [`check_tree`] violations.
pub fn run(root: &Path) -> Result<(), String> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let out = Command::new(cargo)
        .args([
            "tree", "-e", "normal", "-p", "krishiv", "--locked", "--color", "never",
        ])
        .current_dir(root)
        .output()
        .map_err(|e| format!("failed to run cargo tree: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo tree failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    check_tree(&String::from_utf8_lossy(&out.stdout))
}
