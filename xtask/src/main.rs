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
use xtask::codeql_proxy;
use xtask::coverage;
use xtask::dev;
use xtask::docs::{self, Doc};
use xtask::format_neutral;
use xtask::layering;
use xtask::policy::{self, DiffArgs};
use xtask::precheck;
use xtask::profile;
use xtask::registry_coverage;

const USAGE: &str = "usage: cargo xtask <layering | docs-config [PATH] | docs-errors [PATH] | budgets [--build-time] [--measure-only | --write-baseline] | budgets --compare FILE [--baseline FILE] | budgets --precheck [--base REF] [--repo DIR] | budgets --cache-main [--repo DIR] | wheel-sizes DIR [--budgets FILE] [--write | --measure-only FILE] | wheel-sizes --compare FILE [--budgets FILE | --baseline FILE] | setup [--install] | doctor | stack up|down|status [--kafka] | coverage [--no-branch] [--all-features] [--html] [-p PKG]... | coverage-summary [--markdown] [--json] [PATH] | coverage-scope --base REF [--head REF] [--repo DIR] | diff-coverage --base REF [--head REF] [--lcov PATH[:PREFIX]]... [--fail-under PCT] | pr-size|changes|dco --base REF [--head REF] [--repo DIR] | workflow-lint [DIR] | spdx [ROOT] | source-guard [ROOT] | registry-coverage | tokio-console-check | check-format-neutral [ROOT] | profile <QUERY | -f FILE> [--table NAME=PATH]... [--conf KEY=VALUE]... [--tool auto|perf|samply] [--out DIR] [--no-flamegraph] | precheck [--base REF] [--fast] [--full] [--allow-stale] [--coverage-dependents] [--memory-cap SIZE] | codeql-proxy [--base REF] | golden [--tag T] [--jdk 17|21] [--manifest F] [--out F] [--engine reject-all|sql-parse|session] [--cases F] [--ratchet check|update] [--known-failures F]>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["layering"] => run_layering(),
        ["tokio-console-check"] => run_console_check(),
        ["golden", flags @ ..] => run_golden(flags),
        ["precheck", flags @ ..] => precheck::run_precheck(&workspace_root(), flags),
        ["codeql-proxy", flags @ ..] => codeql_proxy::run(&workspace_root(), flags),
        ["registry-coverage"] => registry_coverage::run(&workspace_root()),
        ["budgets", flags @ ..] => run_budgets(flags),
        ["wheel-sizes", "--compare", file, flags @ ..] => {
            run_wheel_sizes_compare(Path::new(file), flags)
        }
        ["wheel-sizes", dir, flags @ ..] => run_wheel_sizes(Path::new(dir), flags),
        ["setup"] => dev::run_setup_full(&workspace_root(), false),
        ["setup", "--install"] => dev::run_setup_full(&workspace_root(), true),
        ["doctor"] => dev::run_doctor(&workspace_root()),
        ["stack", action] => dev::run_stack(&workspace_root(), action, false),
        ["stack", action, "--kafka"] => dev::run_stack(&workspace_root(), action, true),
        ["coverage", flags @ ..] => coverage::run_coverage(&workspace_root(), flags),
        ["profile", flags @ ..] => profile::run_profile(&workspace_root(), flags),
        ["coverage-summary", flags @ ..] => coverage::run_summary(&workspace_root(), flags),
        ["coverage-scope", flags @ ..] => coverage::run_coverage_scope(&workspace_root(), flags),
        ["diff-coverage", flags @ ..] => coverage::run_diff_coverage(&workspace_root(), flags),
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
        ["check-format-neutral"] => format_neutral::run_check(&workspace_root()),
        ["check-format-neutral", root] => format_neutral::run_check(Path::new(root)),
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

/// Run the golden-file runner (`krishiv-testkit` `golden-report`); the report goes to
/// `target/golden-report.json` unless `--out` is given.
fn run_golden(flags: &[&str]) -> Result<(), String> {
    let root = workspace_root();
    let default_out = root.join("target/golden-report.json");
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&root)
        .args([
            "run",
            "--quiet",
            "-p",
            "krishiv-testkit",
            "--bin",
            "golden-report",
            "--",
        ])
        .args(flags);
    if !flags.contains(&"--out") {
        cmd.arg("--out").arg(default_out);
    }
    match cmd.status() {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!("golden-report failed: {status}")),
        Err(e) => Err(format!("cannot run cargo: {e}")),
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

/// `cargo xtask tokio-console-check`: the default build contains no
/// `console-subscriber` and no tokio `tracing` feature (P2-T129).
fn run_console_check() -> Result<(), String> {
    xtask::console::run_console_check(&workspace_root())
}

fn run_layering() -> Result<(), String> {
    let root = workspace_root();
    let map_text = std::fs::read_to_string(root.join("xtask/layers.toml"))
        .map_err(|e| format!("cannot read xtask/layers.toml: {e}"))?;
    let map = layering::parse_map(&map_text)?;
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let out = Command::new(cargo)
        .args([
            "metadata",
            "--format-version",
            "1",
            "--all-features",
            "--locked",
        ])
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
    let mode = budgets::parse_flags(flags).ok_or_else(|| USAGE.to_owned())?;
    let root = workspace_root();
    if let budgets::Mode::Precheck { base, repo } = &mode {
        return run_budgets_precheck(repo.as_deref().unwrap_or(&root), base);
    }
    if let budgets::Mode::CacheMain { repo } = &mode {
        return run_budgets_cache_main(repo.as_deref().unwrap_or(&root));
    }
    let baseline_path = root.join("budgets.json");
    let mut baseline_file = baseline_path.clone();
    let now = match &mode {
        budgets::Mode::Compare { measured, baseline } => {
            if let Some(b) = baseline {
                baseline_file.clone_from(b);
            }
            read_budgets(measured)?
        }
        budgets::Mode::Check { build_time }
        | budgets::Mode::Measure { build_time }
        | budgets::Mode::WriteBaseline { build_time } => {
            let (mut now, target_dir) = measure_budgets(&root, *build_time)?;
            // Wheel sizes are measured by `wheel-sizes`, not here: keep the
            // recorded ones.
            if baseline_path.exists() {
                now.wheel_bytes = read_budgets(&baseline_path)?.wheel_bytes;
            }
            let json = serde_json::to_string_pretty(&now).map_err(|e| e.to_string())? + "\n";
            if matches!(mode, budgets::Mode::WriteBaseline { .. }) {
                std::fs::write(&baseline_path, &json)
                    .map_err(|e| format!("cannot write budgets.json: {e}"))?;
                println!("wrote {}\n{json}", baseline_path.display());
                return Ok(());
            }
            // Machine-readable measurement for CI artifacts.
            let current = target_dir.join("budgets-current.json");
            std::fs::write(&current, &json)
                .map_err(|e| format!("cannot write budgets-current.json: {e}"))?;
            if matches!(mode, budgets::Mode::Measure { .. }) {
                let summary = format!("## Budget measurement\n\n```json\n{json}```\n");
                println!("{summary}");
                append_step_summary(&summary);
                println!("wrote {}", current.display());
                return Ok(());
            }
            now
        }
        // Handled by the early returns above; the arm keeps the match
        // exhaustive so a future mode is noticed here.
        budgets::Mode::Precheck { .. } | budgets::Mode::CacheMain { .. } => {
            return Err("internal error: unreachable budgets mode".to_owned());
        }
    };
    let base = read_budgets(&baseline_file)?;
    report_budget_comparison(&budgets::compare(&base, &now))
}

/// Prints the comparison table and applies the label approvals. CI's
/// `--compare` and the local `--precheck` share this, so both gates apply
/// the same rule to the same [`budgets::compare`] output.
fn report_budget_comparison(cmp: &budgets::Comparison) -> Result<(), String> {
    let summary = format!("## Budgets\n\n{}", cmp.markdown);
    println!("{summary}");
    append_step_summary(&summary);
    let env_flag = |k: &str| std::env::var(k).is_ok_and(|v| v == "1");
    for v in cmp.violations.iter().chain(&cmp.unsafe_violations) {
        eprintln!("{v}");
    }
    if cmp.verdict(env_flag("BUDGET_APPROVED"), env_flag("UNSAFE_APPROVED"))? {
        let note = "Growth approved by the label. Nothing to record: once this merges, the main measurement is the next baseline.";
        println!("{note}");
        append_step_summary(note);
    }
    Ok(())
}

/// `budgets --precheck --base REF [--repo DIR]`: the precheck budgets step.
/// Measures `DIR` and compares it with the cached measurement of the
/// merge-base with `REF`, through [`report_budget_comparison`] (the same rule
/// CI applies). With no cached measurement yet it skips instead of failing:
/// CI checks the delta. `budgets.json` is never read here.
fn run_budgets_precheck(root: &Path, base: &str) -> Result<(), String> {
    let sha = budgets::merge_base(root, base)?;
    let baseline = budgets::main_baseline_path(&cargo_target_dir(root)?, &sha)
        .ok_or_else(|| format!("unsafe main commit id {sha:?}"))?;
    if !baseline.exists() {
        println!("{}", budgets::NO_MAIN_BASELINE);
        eprintln!(
            "hint: on a main checkout, run `cargo xtask budgets --cache-main` to measure it once"
        );
        return Ok(());
    }
    let main = read_budgets(&baseline)?;
    let (now, _) = measure_budgets(root, false)?;
    report_budget_comparison(&budgets::compare(&main, &now))
}

/// `budgets --cache-main [--repo DIR]`: measures `DIR` and caches it as the
/// main measurement of `HEAD`, for later `--precheck` runs. Run on a main
/// checkout; the cache is keyed per commit under the target directory, so
/// each main commit is measured once.
fn run_budgets_cache_main(root: &Path) -> Result<(), String> {
    let sha = budgets::head_sha(root)?;
    if !budgets::working_tree_clean(root)? {
        return Err(format!(
            "cannot cache the main measurement of {sha}: the working tree has uncommitted changes; commit or stash them before `--cache-main`"
        ));
    }
    let path = budgets::main_baseline_path(&cargo_target_dir(root)?, &sha)
        .ok_or_else(|| format!("unsafe commit id {sha:?}"))?;
    let (now, _) = measure_budgets(root, false)?;
    let json = serde_json::to_string_pretty(&now).map_err(|e| e.to_string())? + "\n";
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    std::fs::write(&path, &json).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    println!(
        "cached the main measurement for {sha} in {}",
        path.display()
    );
    Ok(())
}

/// The cargo target directory for `root` (honours `CARGO_TARGET_DIR`).
fn cargo_target_dir(root: &Path) -> Result<PathBuf, String> {
    let out = cargo_cmd()
        .args([
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--color",
            "never",
        ])
        .current_dir(root)
        .output()
        .map_err(|e| format!("failed to run cargo metadata: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let meta: serde_json::Value = serde_json::from_str(&String::from_utf8_lossy(&out.stdout))
        .map_err(|e| format!("bad cargo metadata: {e}"))?;
    Ok(meta
        .get("target_directory")
        .and_then(serde_json::Value::as_str)
        .map_or_else(|| root.join("target"), PathBuf::from))
}

/// `wheel-sizes DIR [--budgets FILE] [--write | --measure-only OUT]`: compares
/// (or records) the size of every wheel in `DIR` with the `wheel_bytes` of
/// `budgets.json` (or `FILE`) (P2-T103). `--measure-only OUT` writes the
/// measured sizes as JSON to `OUT` and gates nothing: the wheels workflow
/// uploads it, and the label-aware gate in pr-policy.yml compares it with
/// `wheel-sizes --compare OUT`, so a label added later can approve growth.
fn run_wheel_sizes(dir: &Path, flags: &[&str]) -> Result<(), String> {
    let (mut write, mut path) = (false, workspace_root().join("budgets.json"));
    let mut measure_only: Option<PathBuf> = None;
    let mut it = flags.iter();
    while let Some(f) = it.next() {
        match *f {
            "--write" => write = true,
            "--budgets" => path = PathBuf::from(it.next().ok_or_else(|| USAGE.to_owned())?),
            "--measure-only" => {
                measure_only = Some(PathBuf::from(it.next().ok_or_else(|| USAGE.to_owned())?));
            }
            _ => return Err(USAGE.to_owned()),
        }
    }
    if write && measure_only.is_some() {
        return Err(USAGE.to_owned());
    }
    let mut measured = std::collections::BTreeMap::new();
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    for e in entries {
        let e = e.map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
        let name = e.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".whl") {
            continue;
        }
        let key = xtask::wheels::wheel_key(&name)
            .ok_or_else(|| format!("not a wheel file name: {name}"))?;
        let bytes = e
            .metadata()
            .map_err(|e| format!("cannot stat {name}: {e}"))?
            .len();
        if measured.insert(key.clone(), bytes).is_some() {
            return Err(format!(
                "two wheels with the tags {key} in {}",
                dir.display()
            ));
        }
    }
    if measured.is_empty() {
        return Err(format!("no wheels in {}", dir.display()));
    }
    if let Some(out) = measure_only {
        let json = serde_json::to_string_pretty(&measured).map_err(|e| e.to_string())? + "\n";
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        std::fs::write(&out, json).map_err(|e| format!("cannot write {}: {e}", out.display()))?;
        println!("measured {} wheels into {}", measured.len(), out.display());
        return Ok(());
    }
    let mut base = read_budgets(&path)?;
    if write {
        // Merge: recording a subset of the targets keeps the others.
        base.wheel_bytes
            .get_or_insert_with(Default::default)
            .extend(measured);
        let json = serde_json::to_string_pretty(&base).map_err(|e| e.to_string())? + "\n";
        std::fs::write(&path, &json).map_err(|e| format!("cannot write budgets.json: {e}"))?;
        println!("wrote {}", path.display());
        return Ok(());
    }
    gate_wheel_sizes(&base.wheel_bytes.unwrap_or_default(), &measured)
}

/// Reads a `--measure-only` file: wheel key to bytes.
fn read_wheel_sizes(file: &Path) -> Result<std::collections::BTreeMap<String, u64>, String> {
    let text = std::fs::read_to_string(file)
        .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
    let sizes: std::collections::BTreeMap<String, u64> =
        serde_json::from_str(&text).map_err(|e| format!("bad {}: {e}", file.display()))?;
    if sizes.is_empty() {
        return Err(format!("no wheel sizes in {}", file.display()));
    }
    Ok(sizes)
}

/// `wheel-sizes --compare FILE [--budgets FILE | --baseline FILE]`: gates the
/// sizes measured by `--measure-only` (`FILE`). The baseline is the
/// `wheel_bytes` of `budgets.json` (or of `--budgets FILE`), or, with
/// `--baseline FILE`, another `--measure-only` file (CI: the measurement of
/// main).
fn run_wheel_sizes_compare(file: &Path, flags: &[&str]) -> Result<(), String> {
    let mut budgets_path = workspace_root().join("budgets.json");
    let mut baseline: Option<PathBuf> = None;
    let mut budgets_given = false;
    let mut it = flags.iter();
    while let Some(f) = it.next() {
        match *f {
            "--budgets" => {
                budgets_path = PathBuf::from(it.next().ok_or_else(|| USAGE.to_owned())?);
                budgets_given = true;
            }
            "--baseline" => {
                baseline = Some(PathBuf::from(it.next().ok_or_else(|| USAGE.to_owned())?));
            }
            _ => return Err(USAGE.to_owned()),
        }
    }
    if budgets_given && baseline.is_some() {
        return Err(USAGE.to_owned());
    }
    let measured = read_wheel_sizes(file)?;
    let recorded = match baseline {
        Some(b) => read_wheel_sizes(&b)?,
        None => read_budgets(&budgets_path)?.wheel_bytes.unwrap_or_default(),
    };
    gate_wheel_sizes(&recorded, &measured)
}

/// Prints the comparison; a violation fails unless `BUDGET_APPROVED=1`.
fn gate_wheel_sizes(
    recorded: &std::collections::BTreeMap<String, u64>,
    measured: &std::collections::BTreeMap<String, u64>,
) -> Result<(), String> {
    let cmp = xtask::wheels::compare(recorded, measured);
    let summary = format!("## Wheel sizes\n\n{}", cmp.markdown);
    println!("{summary}");
    append_step_summary(&summary);
    if cmp.violations.is_empty() {
        return Ok(());
    }
    for v in &cmp.violations {
        eprintln!("{v}");
    }
    if std::env::var("BUDGET_APPROVED").is_ok_and(|v| v == "1") {
        let note = "Growth approved by the label. Nothing to record: once this merges, the main measurement is the next baseline.";
        println!("{note}");
        append_step_summary(note);
        Ok(())
    } else {
        Err(
            "wheel size budget exceeded; add the `budget-approved` label or set BUDGET_APPROVED=1"
                .to_owned(),
        )
    }
}

/// Appends `text` to the GitHub job summary when running in Actions.
fn append_step_summary(text: &str) {
    use std::io::Write;
    if let Some(path) = std::env::var_os("GITHUB_STEP_SUMMARY")
        && let Ok(mut f) = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
    {
        let _ = writeln!(f, "{text}");
    }
}

fn read_budgets(path: &Path) -> Result<Budgets, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("bad {}: {e}", path.display()))
}

/// Measures the budgets of the current checkout; returns them with the
/// cargo target directory.
fn measure_budgets(root: &Path, build_time: bool) -> Result<(Budgets, PathBuf), String> {
    let out = cargo_cmd()
        .args(["metadata", "--format-version", "1", "--color", "never"])
        .current_dir(root)
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
        .current_dir(root)
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
            "--profile",
            "dist",
            "--locked",
            "-p",
            "krishiv-cli",
            "--target-dir",
        ])
        .arg(&build_dir)
        .current_dir(root)
        .status()
        .map_err(|e| format!("failed to run cargo build: {e}"))?;
    if !status.success() {
        return Err("dist-profile build of krishiv-cli failed".to_owned());
    }
    let clean_build_secs = build_time.then(|| started.elapsed().as_secs());
    let bin = build_dir
        .join("dist")
        .join(format!("krishiv{}", std::env::consts::EXE_SUFFIX));
    let binary_bytes = std::fs::metadata(&bin)
        .map_err(|e| format!("cannot stat {}: {e}", bin.display()))?
        .len();
    let now = Budgets {
        transitive_crates,
        transitive_crates_all_features: Some(transitive_crates_all_features),
        binary_bytes: Some(binary_bytes),
        clean_build_secs,
        unsafe_sites: Some(xtask::lint::unsafe_sites_per_crate(root)?),
        wheel_bytes: None,
    };
    Ok((now, target_dir))
}
