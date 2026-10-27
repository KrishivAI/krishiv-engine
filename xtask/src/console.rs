// SPDX-License-Identifier: Apache-2.0
//! The default-build purity check for the tokio-console diagnostics (P2-T129):
//! the default and `dist` builds must contain no `console-subscriber` and no
//! tokio `tracing` feature. `RUSTFLAGS="--cfg tokio_unstable"` is a rustc
//! cfg, not a cargo feature, so `cargo tree -e features` cannot see it: it
//! is covered instead by the cfg-gated naming (names are absent without it)
//! and the separate target directory the diagnostics guide mandates.
#![expect(
    clippy::print_stdout,
    reason = "CLI command: the check result is its interface"
)]

use std::path::Path;
use std::process::Command;

/// Lines of `cargo tree -e features -p krishiv` that must never appear in a
/// default build: the console subscriber itself, and tokio's `tracing`
/// feature (only the `tokio-console` feature may enable it).
#[must_use]
pub fn console_leaks(tree: &str) -> Vec<String> {
    tree.lines()
        .filter(|line| {
            let line = line.trim();
            line.starts_with("console-subscriber v")
                || line == "console-subscriber"
                || line.starts_with("tokio feature \"tracing\"")
        })
        .map(str::to_owned)
        .collect()
}

/// Runs `cargo tree -e features -p krishiv` and fails on any console leak.
///
/// # Errors
/// Returns the cargo failure or the offending lines.
pub fn run_console_check(root: &Path) -> Result<(), String> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let out = Command::new(cargo)
        .args([
            "tree", "--locked", "-e", "features", "-p", "krishiv", "--prefix", "none", "--color",
            "never",
        ])
        .current_dir(root)
        .output()
        .map_err(|e| format!("failed to run cargo tree: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo tree failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let leaks = console_leaks(&String::from_utf8_lossy(&out.stdout));
    if leaks.is_empty() {
        println!("tokio-console-check: OK (no console-subscriber, no tokio tracing feature)");
        Ok(())
    } else {
        Err(format!(
            "the default build leaks tokio-console diagnostics:\n{}",
            leaks.join("\n")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_trees_pass() {
        assert!(console_leaks("").is_empty());
        assert!(
            console_leaks("krishiv v0.1.0-alpha\ntokio v1.53.3\ntokio feature \"default\"\n")
                .is_empty()
        );
    }

    #[test]
    fn the_subscriber_and_the_tracing_feature_are_leaks() {
        let tree = "krishiv v0.1.0-alpha\nconsole-subscriber v0.5.0\ntokio v1.53.3\ntokio feature \"tracing\"\n";
        assert_eq!(console_leaks(tree).len(), 2);
        // Starred (deduplicated) lines count too.
        let starred =
            console_leaks("tokio feature \"tracing\" (*)\nconsole-subscriber v0.5.0 (*)\n");
        assert_eq!(starred.len(), 2);
    }

    #[test]
    fn other_features_are_not_leaks() {
        let tree = "tokio feature \"rt\"\ntracing feature \"std\"\ntracing-subscriber v0.3.23\nconsole-api v0.9.0\n";
        assert!(console_leaks(tree).is_empty());
    }
}
