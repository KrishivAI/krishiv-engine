// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask precheck`: one local command that runs what the required PR CI
//! checks run, scoped to what changed against the base.
//!
//! Everything that decides *what* to run is pure (paths in, [`Step`]s out) so it
//! is testable without a checkout; [`run`] does the I/O. The mapping from every
//! required CI check to a precheck step lives in [`mapping`] and is policy-tested
//! (`xtask/tests/precheck.rs`), so the two cannot drift apart.
#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "CLI command: the summary table is its interface"
)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

/// The required status checks of branch protection on `main`
/// (`gh api repos/KrishivAI/krishiv-engine/branches/main/protection/required_status_checks`).
/// Update this list together with the branch protection; the policy test then
/// demands a mapping for any new entry.
pub const REQUIRED_CI_CHECKS: &[&str] = &[
    "changes",
    "policy",
    "text-lint",
    "fmt",
    "clippy",
    "test",
    "deny",
    "machete",
    "hack",
    "layering",
    "msrv",
    "docs",
    "python-quality",
    "wheels-ok",
    "diff-coverage",
    "CodSpeed Performance Analysis",
    "dependency-review",
    "zizmor",
    "scan-pr / osv-scan",
    "budgets",
    "wheel-sizes",
];

/// Required checks that are not jobs of this repository's workflows (a GitHub
/// App, or a reusable workflow under another name).
pub const EXTERNAL_CI_CHECKS: &[&str] = &["CodSpeed Performance Analysis", "scan-pr / osv-scan"];

/// Every step id [`plan`] reports, in run order.
pub const STEP_IDS: &[&str] = &[
    "changes",
    "fmt",
    "policy",
    "pr-size",
    "workflow-lint",
    "zizmor",
    "text-lint",
    "layering",
    "deny",
    "machete",
    "clippy",
    "doc",
    "nextest",
    "python-quality",
    "python-tests",
    "diff-coverage",
    "diff-coverage-python",
    "budgets",
];

/// How a required CI check is covered by precheck.
#[derive(Debug, PartialEq, Eq)]
pub enum Mapping {
    /// Covered by these precheck step ids.
    Local(&'static [&'static str]),
    /// Not reproducible locally; the reason is printed by the docs and tests.
    CiOnly(&'static str),
}

/// The precheck coverage of a required CI check, `None` when it has none (the
/// policy test fails then).
#[must_use]
pub fn mapping(check: &str) -> Option<Mapping> {
    Some(match check {
        "changes" => Mapping::Local(&["changes"]),
        "policy" => Mapping::Local(&["policy", "pr-size", "workflow-lint"]),
        "text-lint" => Mapping::Local(&["text-lint"]),
        "fmt" => Mapping::Local(&["fmt"]),
        "clippy" => Mapping::Local(&["clippy"]),
        "test" => Mapping::Local(&["nextest"]),
        "deny" => Mapping::Local(&["deny"]),
        "machete" => Mapping::Local(&["machete"]),
        "layering" => Mapping::Local(&["layering"]),
        "docs" => Mapping::Local(&["doc"]),
        "python-quality" => Mapping::Local(&["python-quality", "python-tests"]),
        "diff-coverage" => Mapping::Local(&["diff-coverage", "diff-coverage-python"]),
        "zizmor" => Mapping::Local(&["zizmor"]),
        "budgets" => Mapping::Local(&["budgets"]),
        "hack" => {
            Mapping::CiOnly("cargo hack --each-feature is a per-feature rebuild of every crate")
        }
        "msrv" => Mapping::CiOnly("needs a second (MSRV) toolchain"),
        "wheels-ok" => Mapping::CiOnly(
            "needs the maturin wheel matrix; python-quality and bench tests run locally",
        ),
        "wheel-sizes" => Mapping::CiOnly("needs the release wheels built by the wheel matrix"),
        "dependency-review" => Mapping::CiOnly("GitHub dependency graph service"),
        "CodSpeed Performance Analysis" => {
            Mapping::CiOnly("CodSpeed GitHub App on benchmark runners")
        }
        "scan-pr / osv-scan" => {
            Mapping::CiOnly("reusable OSV workflow against the vulnerability database")
        }
        _ => return None,
    })
}

/// A workspace package, as far as scoping needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pkg {
    /// Package name.
    pub name: String,
    /// Directory relative to the workspace root, `/`-separated, no trailing `/`.
    pub dir: String,
    /// Names of the packages it depends on (any kind).
    pub deps: Vec<String>,
}

/// Parses `cargo metadata --no-deps --format-version 1` output.
///
/// # Errors
/// Returns a message when the JSON is malformed.
pub fn parse_packages(json: &str, root: &Path) -> Result<Vec<Pkg>, String> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("bad cargo metadata: {e}"))?;
    let pkgs = v
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .ok_or("cargo metadata has no packages")?;
    let root = root.to_string_lossy().replace('\\', "/");
    let mut out = Vec::new();
    for p in pkgs {
        let name = p.get("name").and_then(serde_json::Value::as_str);
        let manifest = p.get("manifest_path").and_then(serde_json::Value::as_str);
        let (Some(name), Some(manifest)) = (name, manifest) else {
            return Err("cargo metadata package without name or manifest_path".to_owned());
        };
        let manifest = manifest.replace('\\', "/");
        let dir = manifest
            .strip_suffix("/Cargo.toml")
            .and_then(|d| d.strip_prefix(root.trim_end_matches('/')))
            .map_or("", |d| d.trim_start_matches('/'))
            .to_owned();
        let deps = p
            .get("dependencies")
            .and_then(serde_json::Value::as_array)
            .map(|ds| {
                ds.iter()
                    .filter_map(|d| d.get("name").and_then(serde_json::Value::as_str))
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        out.push(Pkg {
            name: name.to_owned(),
            dir,
            deps,
        });
    }
    Ok(out)
}

/// Which packages a Rust command applies to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    /// No Rust-relevant change.
    None,
    /// A workspace-wide input changed (manifests, toolchain, lint config).
    Workspace,
    /// Only these packages changed.
    Packages {
        /// Packages with changed files (clippy and doc run on these).
        touched: Vec<String>,
        /// `touched` plus everything that depends on them, transitively (tests run on these).
        with_dependents: Vec<String>,
    },
}

/// What a set of changed paths means for the checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Changes {
    /// The changed paths (repository-relative).
    pub paths: Vec<String>,
    /// Rust package scope.
    pub scope: Scope,
    /// `Cargo.toml` or `Cargo.lock` (root or crate) changed.
    pub manifests: bool,
    /// Something under `.github/` changed.
    pub github: bool,
    /// Something under `python/` changed.
    pub python: bool,
    /// Something under `bench/` changed.
    pub bench: bool,
    /// Bench sources, tests or lock changed (the coverage gate's `bench` rule).
    pub bench_cov: bool,
    /// Something under `tests/scripts/` changed.
    pub scripts: bool,
}

/// Root files whose change can alter every package.
const WORKSPACE_WIDE: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "clippy.toml",
    "rustfmt.toml",
    ".rustfmt.toml",
];

fn is_workspace_wide(path: &str) -> bool {
    WORKSPACE_WIDE.contains(&path) || path.starts_with(".cargo/") || path.starts_with("proto/")
}

impl Changes {
    /// Classifies `paths` against the workspace `pkgs`.
    #[must_use]
    pub fn detect(paths: &[String], pkgs: &[Pkg]) -> Self {
        let paths: Vec<String> = paths
            .iter()
            .map(|p| p.trim().to_owned())
            .filter(|p| !p.is_empty())
            .collect();
        let any = |f: &dyn Fn(&str) -> bool| paths.iter().any(|p| f(p));
        let manifests = any(&|p| {
            p.rsplit('/')
                .next()
                .is_some_and(|n| n == "Cargo.toml" || n == "Cargo.lock")
        });
        let scope = if any(&is_workspace_wide) {
            Scope::Workspace
        } else {
            let touched: BTreeSet<&str> = pkgs
                .iter()
                .filter(|pkg| {
                    paths.iter().any(|p| {
                        p.strip_prefix(pkg.dir.as_str())
                            .is_some_and(|rest| rest.starts_with('/'))
                    })
                })
                .map(|pkg| pkg.name.as_str())
                .collect();
            if touched.is_empty() {
                Scope::None
            } else {
                Scope::Packages {
                    with_dependents: dependents_closure(&touched, pkgs),
                    touched: touched.iter().map(|s| (*s).to_owned()).collect(),
                }
            }
        };
        Self {
            scope,
            manifests,
            github: any(&|p| p.starts_with(".github/")),
            python: any(&|p| p.starts_with("python/")),
            bench: any(&|p| p.starts_with("bench/")),
            bench_cov: any(&|p| {
                p.starts_with("bench/src/")
                    || p.starts_with("bench/tests/")
                    || p == "bench/pyproject.toml"
                    || p == "bench/uv.lock"
            }),
            scripts: any(&|p| p.starts_with("tests/scripts/")),
            paths,
        }
    }
}

/// `touched` plus every package that depends on one of them, transitively.
#[must_use]
pub fn dependents_closure(touched: &BTreeSet<&str>, pkgs: &[Pkg]) -> Vec<String> {
    let mut set: BTreeSet<String> = touched.iter().map(|s| (*s).to_owned()).collect();
    loop {
        let before = set.len();
        for pkg in pkgs {
            if pkg.deps.iter().any(|d| set.contains(d)) {
                set.insert(pkg.name.clone());
            }
        }
        if set.len() == before {
            return set.into_iter().collect();
        }
    }
}

/// Parsed command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Opts {
    /// Base ref the diff is taken against.
    pub base: String,
    /// Skip coverage and budgets.
    pub fast: bool,
    /// Run budgets even without a manifest change.
    pub full: bool,
    /// Run heavy steps under a `systemd-run` memory cap (e.g. `12G`).
    pub memory_cap: Option<String>,
}

impl Opts {
    /// Parses `[--base REF] [--fast] [--full] [--memory-cap SIZE]`.
    ///
    /// # Errors
    /// Returns a usage error for unknown flags, missing values or a bad size.
    pub fn parse(flags: &[&str]) -> Result<Self, String> {
        let mut o = Self {
            base: "origin/main".to_owned(),
            fast: false,
            full: false,
            memory_cap: None,
        };
        let mut it = flags.iter();
        while let Some(f) = it.next() {
            match *f {
                "--fast" => o.fast = true,
                "--full" => o.full = true,
                "--base" => {
                    let v = it.next().ok_or("--base needs a ref")?;
                    if v.is_empty() || v.starts_with('-') {
                        return Err(format!("--base needs a ref, got {v:?}"));
                    }
                    (*v).clone_into(&mut o.base);
                }
                "--memory-cap" => {
                    let v = it.next().ok_or("--memory-cap needs a size such as 12G")?;
                    let digits = v.trim_end_matches(['K', 'M', 'G', 'T']);
                    let ok = v.len() > digits.len()
                        && v.len() - digits.len() == 1
                        && digits.parse::<u64>().is_ok_and(|n| n > 0);
                    if !ok {
                        return Err(format!("--memory-cap expects e.g. 12G, got {v:?}"));
                    }
                    o.memory_cap = Some((*v).to_owned());
                }
                other => return Err(format!("unknown precheck flag {other}")),
            }
        }
        Ok(o)
    }
}

/// One process to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cmd {
    /// Program (resolved on `PATH`).
    pub program: String,
    /// Arguments.
    pub args: Vec<String>,
    /// Extra environment.
    pub env: Vec<(String, String)>,
    /// Working directory relative to the workspace root (`""` is the root).
    pub dir: String,
    /// Compiles or runs the workspace: put it under the memory cap.
    pub heavy: bool,
}

impl Cmd {
    fn new(program: &str, args: &[&str]) -> Self {
        Self {
            program: program.to_owned(),
            args: args.iter().map(|s| (*s).to_owned()).collect(),
            env: Vec::new(),
            dir: String::new(),
            heavy: false,
        }
    }
    fn heavy(mut self) -> Self {
        self.heavy = true;
        self
    }
    fn dir(mut self, dir: &str) -> Self {
        dir.clone_into(&mut self.dir);
        self
    }
    fn env(mut self, k: &str, v: &str) -> Self {
        self.env.push((k.to_owned(), v.to_owned()));
        self
    }
    fn arg(mut self, a: impl Into<String>) -> Self {
        self.args.push(a.into());
        self
    }
    fn args_owned(mut self, a: impl IntoIterator<Item = String>) -> Self {
        self.args.extend(a);
        self
    }
}

/// What a step does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Run these commands in order; the first failure fails the step.
    Run(Vec<Cmd>),
    /// Nothing to run; an informational pass.
    Note(String),
    /// Not run, with the reason (a change-scoping skip or a missing tool).
    Skip(String),
}

/// One row of the precheck.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    /// Step id (one of [`STEP_IDS`]).
    pub id: &'static str,
    /// What to do.
    pub action: Action,
    /// Parts that were left out of a `Run` (tools not installed), reported as `partial`.
    pub notes: Vec<String>,
}

fn skip(id: &'static str, why: impl Into<String>) -> Step {
    Step {
        id,
        action: Action::Skip(why.into()),
        notes: Vec::new(),
    }
}

fn run(id: &'static str, cmds: Vec<Cmd>) -> Step {
    Step {
        id,
        action: Action::Run(cmds),
        notes: Vec::new(),
    }
}

/// A `run` whose tool may be missing: a missing tool turns it into a `Skip`.
fn needs(id: &'static str, tool: &str, has: &dyn Fn(&str) -> bool, cmds: Vec<Cmd>) -> Step {
    if has(tool) {
        run(id, cmds)
    } else {
        skip(id, format!("not installed: {tool}"))
    }
}

fn pkg_args(scope: &Scope, dependents: bool) -> Vec<String> {
    match scope {
        Scope::Packages {
            touched,
            with_dependents,
        } => {
            let names = if dependents { with_dependents } else { touched };
            names
                .iter()
                .flat_map(|n| ["-p".to_owned(), n.clone()])
                .collect()
        }
        Scope::Workspace | Scope::None => vec!["--workspace".to_owned()],
    }
}

fn scope_label(scope: &Scope) -> String {
    match scope {
        Scope::None => "no Rust change".to_owned(),
        Scope::Workspace => "whole workspace".to_owned(),
        Scope::Packages {
            touched,
            with_dependents,
        } => format!(
            "{} touched crate(s), {} with dependents",
            touched.len(),
            with_dependents.len()
        ),
    }
}

const NO_RUST: &str = "no Rust crate changed";

/// Builds the ordered steps (one per [`STEP_IDS`] entry) for `ch`.
///
/// `has` reports whether a binary is installed; `xtask` is the path of this
/// executable, used to run the other xtask commands without a rebuild.
#[must_use]
#[expect(clippy::too_many_lines, reason = "one flat table of checks reads best")]
pub fn plan(ch: &Changes, o: &Opts, has: &dyn Fn(&str) -> bool, xtask: &str) -> Vec<Step> {
    let rust = ch.scope != Scope::None;
    let base = o.base.as_str();
    let x = |args: &[&str]| Cmd::new(xtask, args);
    let mut steps = Vec::new();

    steps.push(Step {
        id: "changes",
        action: Action::Note(format!(
            "{} path(s) vs {base}; Rust: {}; python: {}; bench: {}; .github: {}; manifests: {}",
            ch.paths.len(),
            scope_label(&ch.scope),
            ch.python,
            ch.bench,
            ch.github,
            ch.manifests
        )),
        notes: Vec::new(),
    });

    steps.push(if rust {
        run(
            "fmt",
            vec![Cmd::new("cargo", &["fmt", "--all", "--", "--check"])],
        )
    } else {
        skip("fmt", NO_RUST)
    });

    steps.push(run(
        "policy",
        vec![
            x(&["spdx"]),
            x(&["source-guard"]),
            x(&["registry-coverage"]),
            x(&["dco", "--base", base]),
            Cmd::new(
                "bash",
                &[".github/scripts/check-large-files.sh", base, "HEAD"],
            ),
        ],
    ));
    steps.push(run("pr-size", vec![x(&["pr-size", "--base", base])]));

    steps.push(if ch.github {
        run("workflow-lint", vec![x(&["workflow-lint"])])
    } else {
        skip("workflow-lint", "no .github change")
    });
    steps.push(if ch.github {
        needs(
            "zizmor",
            "zizmor",
            has,
            vec![Cmd::new("zizmor", &[".github"])],
        )
    } else {
        skip("zizmor", "no .github change")
    });

    steps.push(text_lint(has));

    steps.push(if rust {
        run("layering", vec![x(&["layering"])])
    } else {
        skip("layering", NO_RUST)
    });
    steps.push(if ch.manifests {
        needs(
            "deny",
            "cargo-deny",
            has,
            vec![Cmd::new("cargo", &["deny", "--locked", "check"])],
        )
    } else {
        skip("deny", "no Cargo.toml/Cargo.lock change")
    });
    steps.push(if ch.manifests {
        needs(
            "machete",
            "cargo-machete",
            has,
            vec![Cmd::new("cargo", &["machete"])],
        )
    } else {
        skip("machete", "no Cargo.toml/Cargo.lock change")
    });

    if rust {
        steps.push(run(
            "clippy",
            vec![
                Cmd::new("cargo", &["clippy", "--locked", "--all-targets"])
                    .args_owned(pkg_args(&ch.scope, false))
                    .arg("--")
                    .arg("-D")
                    .arg("warnings")
                    .heavy(),
            ],
        ));
        steps.push(run(
            "doc",
            vec![
                Cmd::new("cargo", &["doc", "--no-deps", "--locked"])
                    .args_owned(pkg_args(&ch.scope, false))
                    .env("RUSTDOCFLAGS", "-D warnings")
                    .heavy(),
            ],
        ));
        steps.push(needs(
            "nextest",
            "cargo-nextest",
            has,
            vec![
                Cmd::new("cargo", &["nextest", "run", "--locked", "--no-tests=pass"])
                    .args_owned(pkg_args(&ch.scope, true))
                    .env("KRISHIV_REQUIRE_TARGETS", "baseline,x86_64v3")
                    .heavy(),
                Cmd::new("cargo", &["test", "--doc", "--locked"])
                    .args_owned(pkg_args(&ch.scope, true))
                    .heavy(),
            ],
        ));
    } else {
        for id in ["clippy", "doc", "nextest"] {
            steps.push(skip(id, NO_RUST));
        }
    }

    steps.push(python_quality(ch, has));
    steps.push(python_tests(ch, has));

    steps.push(if o.fast {
        skip("diff-coverage", "--fast")
    } else if !rust {
        skip("diff-coverage", NO_RUST)
    } else if !has("cargo-llvm-cov") {
        skip("diff-coverage", "not installed: cargo-llvm-cov")
    } else if !has("cargo-nextest") {
        skip("diff-coverage", "not installed: cargo-nextest")
    } else {
        run(
            "diff-coverage",
            vec![
                x(&["coverage", "--no-branch"]).heavy(),
                x(&["diff-coverage", "--base", base, "--fail-under", "90"]),
            ],
        )
    });
    steps.push(if o.fast {
        skip("diff-coverage-python", "--fast")
    } else if ch.bench_cov {
        if has("uv") {
            run(
                "diff-coverage-python",
                vec![x(&[
                    "diff-coverage",
                    "--base",
                    base,
                    "--lcov",
                    "target/precheck/bench.lcov:bench",
                    "--fail-under",
                    "90",
                ])],
            )
        } else {
            skip("diff-coverage-python", "not installed: uv")
        }
    } else if ch.python {
        skip(
            "diff-coverage-python",
            "python/ coverage needs the maturin wheel build (CI wheels job)",
        )
    } else {
        skip("diff-coverage-python", "no python/bench change")
    });

    steps.push(if o.fast {
        skip("budgets", "--fast")
    } else if !(ch.manifests || o.full) {
        skip("budgets", "no Cargo.toml/Cargo.lock change (use --full)")
    } else {
        run("budgets", vec![x(&["budgets"]).heavy()])
    });
    steps
}

fn text_lint(has: &dyn Fn(&str) -> bool) -> Step {
    let mut cmds = Vec::new();
    let mut notes = Vec::new();
    if has("typos") {
        cmds.push(Cmd::new("typos", &[]));
    } else {
        notes.push("typos not installed".to_owned());
    }
    if has("taplo") {
        cmds.push(Cmd::new("taplo", &["fmt", "--check"]));
    } else {
        notes.push("taplo not installed".to_owned());
    }
    if has("markdownlint-cli2") {
        cmds.push(Cmd::new("markdownlint-cli2", &["**/*.md"]));
    } else if has("docker") {
        cmds.push(Cmd::new(
            "docker",
            &[
                "run",
                "--rm",
                "-v",
                "{ROOT}:/workdir",
                "davidanson/markdownlint-cli2:v0.23.3",
                "**/*.md",
            ],
        ));
    } else {
        notes.push("markdownlint-cli2 (or docker) not installed".to_owned());
    }
    if cmds.is_empty() {
        skip("text-lint", format!("not installed: {}", notes.join(", ")))
    } else {
        Step {
            id: "text-lint",
            action: Action::Run(cmds),
            notes,
        }
    }
}

/// Directories to run the ruff/ty quality gate in, with the `uv` invocation CI uses.
fn python_dirs(ch: &Changes) -> Vec<(&'static str, &'static [&'static str])> {
    let mut v: Vec<(&'static str, &'static [&'static str])> = Vec::new();
    if ch.python {
        v.push(("python", &[]));
    }
    if ch.bench {
        v.push(("bench", &[]));
    }
    if ch.scripts {
        v.push(("tests/scripts", &["--project", "../../python"]));
    }
    v
}

fn python_quality(ch: &Changes, has: &dyn Fn(&str) -> bool) -> Step {
    let dirs = python_dirs(ch);
    if dirs.is_empty() {
        return skip(
            "python-quality",
            "no python/, bench/ or tests/scripts/ change",
        );
    }
    let mut cmds = Vec::new();
    for (dir, project) in dirs {
        let proj = |c: Cmd| c.args_owned(project.iter().map(|s| (*s).to_owned()));
        cmds.push(proj(
            Cmd::new("uv", &["sync", "--frozen", "--no-install-project"]).dir(dir),
        ));
        for tool in [
            &["ruff", "check"][..],
            &["ruff", "format", "--check"],
            &["ty", "check"],
        ] {
            let mut c = Cmd::new("uv", &["run", "--frozen", "--no-sync"]).dir(dir);
            c = proj(c).args_owned(tool.iter().map(|s| (*s).to_owned()));
            if dir == "tests/scripts" {
                c = c.arg(".");
            }
            cmds.push(c);
        }
    }
    needs("python-quality", "uv", has, cmds)
}

fn python_tests(ch: &Changes, has: &dyn Fn(&str) -> bool) -> Step {
    if !ch.bench && !ch.python {
        return skip("python-tests", "no python/ or bench/ change");
    }
    if !ch.bench {
        return skip(
            "python-tests",
            "python/ tests need the maturin wheel build (CI wheels job)",
        );
    }
    let mut step = needs(
        "python-tests",
        "uv",
        has,
        vec![
            Cmd::new("uv", &["sync", "--frozen"]).dir("bench"),
            Cmd::new(
                "uv",
                &[
                    "run",
                    "--frozen",
                    "--no-sync",
                    "pytest",
                    "-q",
                    "--cov=src/krishiv_bench",
                    "--cov-report=lcov:../target/precheck/bench.lcov",
                ],
            )
            .dir("bench"),
        ],
    );
    if ch.python {
        step.notes.push(
            "python/ tests need the maturin wheel build (CI wheels job); bench/ only".to_owned(),
        );
    }
    step
}

/// Outcome of one step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Ran and passed.
    Pass,
    /// Ran but part of it was left out (a tool is not installed).
    Partial,
    /// Failed.
    Fail,
    /// Not run.
    Skipped,
}

/// A finished step for the summary table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    /// Step id.
    pub id: &'static str,
    /// Result.
    pub status: Status,
    /// Reason or detail.
    pub detail: String,
    /// Wall-clock seconds.
    pub secs: u64,
}

/// Whether the run fails: any step failed.
#[must_use]
pub fn failed(outcomes: &[Outcome]) -> bool {
    outcomes.iter().any(|o| o.status == Status::Fail)
}

/// The summary table.
#[must_use]
pub fn render_table(outcomes: &[Outcome]) -> String {
    let w = outcomes
        .iter()
        .map(|o| o.id.len())
        .max()
        .unwrap_or(4)
        .max(5);
    let mut s = format!("{:<w$}  {:<7}  {:>5}  detail\n", "check", "status", "secs");
    for o in outcomes {
        let status = match o.status {
            Status::Pass => "pass",
            Status::Partial => "partial",
            Status::Fail => "FAIL",
            Status::Skipped => "skipped",
        };
        s.push_str(&format!(
            "{:<w$}  {:<7}  {:>5}  {}\n",
            o.id, status, o.secs, o.detail
        ));
    }
    s
}

/// Prefix that runs a command under a `systemd-run` memory cap.
#[must_use]
pub fn cap_prefix(cap: &str) -> Vec<String> {
    ["systemd-run", "--user", "--scope", "-q", "-p"]
        .iter()
        .map(|s| (*s).to_owned())
        .chain([
            format!("MemoryMax={cap}"),
            "-p".to_owned(),
            "MemorySwapMax=0".to_owned(),
            "--".to_owned(),
        ])
        .collect()
}

fn on_path(bin: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|p| {
        std::env::split_paths(&p).any(|d| {
            let f = d.join(bin);
            f.is_file()
        })
    })
}

fn git_lines(root: &Path, args: &[&str]) -> Result<Vec<String>, String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|e| format!("failed to run git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_owned)
        .collect())
}

/// Committed changes against the merge base, plus uncommitted and untracked files.
fn changed_paths(root: &Path, base: &str) -> Result<Vec<String>, String> {
    let mut all: BTreeSet<String> = BTreeSet::new();
    all.extend(git_lines(
        root,
        &["diff", "--name-only", &format!("{base}...HEAD")],
    )?);
    all.extend(git_lines(root, &["diff", "--name-only", "HEAD"])?);
    all.extend(git_lines(
        root,
        &["ls-files", "--others", "--exclude-standard"],
    )?);
    Ok(all.into_iter().collect())
}

fn exec(root: &Path, cmd: &Cmd, cap: Option<&str>, log: &mut String) -> Result<(), String> {
    let mut argv: Vec<String> = Vec::new();
    if cmd.heavy
        && let Some(c) = cap
    {
        argv.extend(cap_prefix(c));
    }
    argv.push(cmd.program.clone());
    let root_str = root.to_string_lossy();
    argv.extend(cmd.args.iter().map(|a| a.replace("{ROOT}", &root_str)));
    let (prog, rest) = argv.split_first().ok_or("empty command")?;
    let mut c = Command::new(prog);
    c.args(rest).current_dir(root.join(&cmd.dir));
    for (k, v) in &cmd.env {
        c.env(k, v);
    }
    log.push_str(&format!("$ {}\n", argv.join(" ")));
    let out = c
        .output()
        .map_err(|e| format!("cannot run {}: {e}", cmd.program))?;
    log.push_str(&String::from_utf8_lossy(&out.stdout));
    log.push_str(&String::from_utf8_lossy(&out.stderr));
    if out.status.success() {
        Ok(())
    } else {
        Err(format!("`{}` failed ({})", argv.join(" "), out.status))
    }
}

fn run_step(root: &Path, step: &Step, cap: Option<&str>, log_dir: &Path) -> Outcome {
    let started = Instant::now();
    let (status, detail) = match &step.action {
        Action::Note(n) => (Status::Pass, n.clone()),
        Action::Skip(why) => (Status::Skipped, why.clone()),
        Action::Run(cmds) => {
            let mut log = String::new();
            let mut result = Ok(());
            for c in cmds {
                if let Err(e) = exec(root, c, cap, &mut log) {
                    result = Err(e);
                    break;
                }
            }
            let log_path = log_dir.join(format!("{}.log", step.id));
            let _ = std::fs::write(&log_path, &log);
            match result {
                Ok(()) if step.notes.is_empty() => (Status::Pass, String::new()),
                Ok(()) => (Status::Partial, step.notes.join("; ")),
                Err(e) => {
                    let tail: Vec<&str> = log.lines().rev().take(30).collect();
                    eprintln!(
                        "--- {} failed; last lines of {} ---",
                        step.id,
                        log_path.display()
                    );
                    for l in tail.iter().rev() {
                        eprintln!("{l}");
                    }
                    (Status::Fail, format!("{e}; log: {}", log_path.display()))
                }
            }
        }
    };
    Outcome {
        id: step.id,
        status,
        detail,
        secs: started.elapsed().as_secs(),
    }
}

/// `cargo xtask precheck [--base REF] [--fast] [--full] [--memory-cap SIZE]`.
///
/// # Errors
/// Returns the usage or environment error, or the failed-check summary.
pub fn run_precheck(root: &Path, flags: &[&str]) -> Result<(), String> {
    let o = Opts::parse(flags)?;
    if o.memory_cap.is_some() && !on_path("systemd-run") {
        return Err("--memory-cap needs systemd-run, which is not installed".to_owned());
    }
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let meta = Command::new(cargo)
        .args([
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--color",
            "never",
        ])
        .current_dir(root)
        .output()
        .map_err(|e| format!("failed to run cargo metadata: {e}"))?;
    if !meta.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&meta.stderr)
        ));
    }
    let pkgs = parse_packages(&String::from_utf8_lossy(&meta.stdout), root)?;
    let paths = changed_paths(root, &o.base)?;
    let ch = Changes::detect(&paths, &pkgs);
    let exe = std::env::current_exe().map_err(|e| format!("cannot find the xtask binary: {e}"))?;
    let exe = exe.to_string_lossy().into_owned();
    let steps = plan(&ch, &o, &on_path, &exe);
    let log_dir: PathBuf = root.join("target/precheck");
    std::fs::create_dir_all(&log_dir)
        .map_err(|e| format!("cannot create {}: {e}", log_dir.display()))?;
    let mut outcomes = Vec::new();
    for (i, step) in steps.iter().enumerate() {
        eprintln!("[{}/{}] {}", i + 1, steps.len(), step.id);
        outcomes.push(run_step(root, step, o.memory_cap.as_deref(), &log_dir));
    }
    println!("\n{}", render_table(&outcomes));
    if failed(&outcomes) {
        let names: Vec<&str> = outcomes
            .iter()
            .filter(|o| o.status == Status::Fail)
            .map(|o| o.id)
            .collect();
        Err(format!("precheck failed: {}", names.join(", ")))
    } else {
        println!("precheck: OK");
        Ok(())
    }
}
