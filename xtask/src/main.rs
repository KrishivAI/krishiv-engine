// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask <command>` entry point.

#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "CLI binary: stdout/stderr are its interface"
)]

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use xtask::budgets::{self, Budgets};
use xtask::dev;
use xtask::docs::{self, Doc};
use xtask::layering;
use xtask::policy::{self, DiffArgs};
use xtask::registry_coverage;

const USAGE: &str = "usage: cargo xtask <layering | docs-config [PATH] | docs-errors [PATH] | budgets [--build-time] [--write-baseline] | setup [--install] | doctor | stack up|down|status [--kafka] | pr-size|changes|dco --base REF [--head REF] [--repo DIR] | workflow-lint [DIR] | spdx [ROOT] | source-guard [ROOT] | registry-coverage>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["layering"] => run_layering(),
        ["registry-coverage"] => registry_coverage::run(&workspace_root()),
        ["budgets", flags @ ..] => run_budgets(flags),
        ["setup"] => dev::run_setup_full(&workspace_root(), false),
        ["setup", "--install"] => dev::run_setup_full(&workspace_root(), true),
        ["doctor"] => dev::run_doctor(&workspace_root()),
        ["stack", action] => dev::run_stack(&workspace_root(), action, false),
        ["stack", action, "--kafka"] => dev::run_stack(&workspace_root(), action, true),
        ["pr-size", flags @ ..] => {
            DiffArgs::parse(flags, &workspace_root()).and_then(|a| policy::run_pr_size(&a))
        }
        ["changes", flags @ ..] => {
            DiffArgs::parse(flags, &workspace_root()).and_then(|a| policy::run_changes(&a))
        }
        ["dco", flags @ ..] => {
            DiffArgs::parse(flags, &workspace_root()).and_then(|a| policy::run_dco(&a))
        }
        ["workflow-lint"] => policy::run_workflow_lint(&workspace_root().join(".github/workflows")),
        ["workflow-lint", dir] => policy::run_workflow_lint(Path::new(dir)),
        ["spdx"] => policy::run_spdx(&workspace_root()),
        ["spdx", root] => policy::run_spdx(Path::new(root)),
        ["source-guard"] => policy::run_source_guard(&workspace_root()),
        ["source-guard", root] => policy::run_source_guard(Path::new(root)),
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

fn cargo_cmd() -> Command {
    Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
}

fn run_budgets(flags: &[&str]) -> Result<(), String> {
    let mut build_time = false;
    let mut write_baseline = false;
    for f in flags {
        match *f {
            "--build-time" => build_time = true,
            "--write-baseline" => write_baseline = true,
            _ => return Err(USAGE.to_owned()),
        }
    }
    let root = workspace_root();

    let out = cargo_cmd()
        .args(["metadata", "--format-version", "1", "--color", "never"])
        .current_dir(&root)
        .output()
        .map_err(|e| format!("failed to run cargo metadata: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let meta_json = String::from_utf8_lossy(&out.stdout);
    let transitive_crates_all_features = budgets::count_transitive(&meta_json)?;
    let tree = cargo_cmd()
        .args([
            "tree",
            "-p",
            "krishiv-cli",
            "-e",
            "normal",
            "--prefix",
            "none",
            "--locked",
            "--color",
            "never",
        ])
        .current_dir(&root)
        .output()
        .map_err(|e| format!("failed to run cargo tree: {e}"))?;
    if !tree.status.success() {
        return Err(format!(
            "cargo tree failed: {}",
            String::from_utf8_lossy(&tree.stderr)
        ));
    }
    let transitive_crates = budgets::count_tree_crates(&String::from_utf8_lossy(&tree.stdout));
    let meta: serde_json::Value =
        serde_json::from_str(&meta_json).map_err(|e| format!("bad cargo metadata: {e}"))?;
    let target_dir = meta
        .get("target_directory")
        .and_then(serde_json::Value::as_str)
        .map_or_else(|| root.join("target"), PathBuf::from);

    // A clean build uses a throwaway target dir so the cache cannot hide cost.
    let build_dir = if build_time {
        target_dir.join("budget-clean")
    } else {
        target_dir.clone()
    };
    if build_time && build_dir.exists() {
        std::fs::remove_dir_all(&build_dir)
            .map_err(|e| format!("cannot clean {}: {e}", build_dir.display()))?;
    }
    let started = std::time::Instant::now();
    let status = cargo_cmd()
        .args([
            "build",
            "--release",
            "--locked",
            "-p",
            "krishiv-cli",
            "--target-dir",
        ])
        .arg(&build_dir)
        .current_dir(&root)
        .status()
        .map_err(|e| format!("failed to run cargo build: {e}"))?;
    if !status.success() {
        return Err("release build of krishiv-cli failed".to_owned());
    }
    let clean_build_secs = build_time.then(|| started.elapsed().as_secs());
    let bin = build_dir
        .join("release")
        .join(format!("krishiv{}", std::env::consts::EXE_SUFFIX));
    let binary_bytes = std::fs::metadata(&bin)
        .map_err(|e| format!("cannot stat {}: {e}", bin.display()))?
        .len();

    let now = Budgets {
        transitive_crates,
        transitive_crates_all_features: Some(transitive_crates_all_features),
        binary_bytes: Some(binary_bytes),
        clean_build_secs,
    };
    let json = serde_json::to_string_pretty(&now).map_err(|e| e.to_string())? + "\n";
    let baseline_path = root.join("budgets.json");
    if write_baseline {
        std::fs::write(&baseline_path, &json)
            .map_err(|e| format!("cannot write budgets.json: {e}"))?;
        println!("wrote {}\n{json}", baseline_path.display());
        return Ok(());
    }
    // Machine-readable measurement for CI artifacts.
    std::fs::write(target_dir.join("budgets-current.json"), &json)
        .map_err(|e| format!("cannot write budgets-current.json: {e}"))?;

    let base: Budgets = serde_json::from_str(
        &std::fs::read_to_string(&baseline_path)
            .map_err(|e| format!("cannot read budgets.json: {e}"))?,
    )
    .map_err(|e| format!("bad budgets.json: {e}"))?;
    let cmp = budgets::compare(&base, &now);
    let summary = format!("## Budgets\n\n{}", cmp.markdown);
    println!("{summary}");
    if let Some(path) = std::env::var_os("GITHUB_STEP_SUMMARY") {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
            .map_err(|e| format!("cannot open step summary: {e}"))?;
        writeln!(f, "{summary}").map_err(|e| e.to_string())?;
    }
    if cmp.violations.is_empty() {
        return Ok(());
    }
    let approved = std::env::var("BUDGET_APPROVED").is_ok_and(|v| v == "1");
    for v in &cmp.violations {
        eprintln!("{}: {v}", if approved { "approved" } else { "error" });
    }
    if approved {
        let note = "Budget exceeded but approved. Refresh the baseline: `cargo xtask budgets --write-baseline` and commit budgets.json.";
        println!("{note}");
        if let Some(path) = std::env::var_os("GITHUB_STEP_SUMMARY") {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(path)
            {
                let _ = writeln!(f, "{note}");
            }
        }
    }
    if approved {
        Ok(())
    } else {
        Err("budget exceeded; add the `budget-approved` label or set BUDGET_APPROVED=1".to_owned())
    }
}
