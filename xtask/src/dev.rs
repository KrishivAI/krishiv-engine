// SPDX-License-Identifier: Apache-2.0
//! Developer environment (P2-T123): `doctor`, `setup` and `stack up|down|status`.
//!
//! The decision logic is pure and unit-tested; the probes that run external
//! tools are thin wrappers. Nothing here ever runs `sudo`: fixes are printed
//! for the developer to run, and `setup --install` only executes the
//! user-level ones (`rustup`, `cargo install`, `uv`).
#![expect(clippy::print_stdout, reason = "command output is the interface")]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Docker compose project owned by the benchmark baselines. `stack` only ever
/// addresses this project, never another one on the machine.
pub const STACK_PROJECT: &str = "krishiv-bench";

/// Compose file of the baselines stack, relative to the workspace root.
pub const STACK_FILE: &str = "bench/baselines/docker-compose.yml";

/// Free space below which `doctor` fails, in GiB.
const DISK_FAIL_GIB: u64 = 5;
/// Free space below which `doctor` warns, in GiB.
const DISK_WARN_GIB: u64 = 30;

/// Outcome of one doctor check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Requirement met.
    Ok,
    /// Optional or advisory item is missing.
    Warn,
    /// A prerequisite is missing; `doctor` exits non-zero.
    Fail,
}

/// A user-level command that fixes a check (run by `setup --install`).
pub type AutoFix = Vec<String>;

/// One doctor finding.
#[derive(Debug, Clone)]
pub struct Check {
    /// What was checked.
    pub name: String,
    /// Result.
    pub status: Status,
    /// What was found.
    pub detail: String,
    /// The named fix (a command or instruction) when not [`Status::Ok`].
    pub fix: Option<String>,
    /// A command `setup --install` may run for this fix.
    pub auto: Option<AutoFix>,
}

impl Check {
    fn ok(name: &str, detail: impl Into<String>) -> Self {
        Self::new(name, Status::Ok, detail, None, None)
    }

    fn new(
        name: &str,
        status: Status,
        detail: impl Into<String>,
        fix: Option<String>,
        auto: Option<AutoFix>,
    ) -> Self {
        Self {
            name: name.to_owned(),
            status,
            detail: detail.into(),
            fix,
            auto,
        }
    }

    fn missing(
        name: &str,
        status: Status,
        detail: impl Into<String>,
        fix: &str,
        auto: Option<AutoFix>,
    ) -> Self {
        Self::new(name, status, detail, Some(fix.to_owned()), auto)
    }
}

fn argv(parts: &[&str]) -> Option<AutoFix> {
    Some(parts.iter().map(|p| (*p).to_owned()).collect())
}

/// Whether any check failed.
#[must_use]
pub fn has_failures(checks: &[Check]) -> bool {
    checks.iter().any(|c| c.status == Status::Fail)
}

/// Renders the checks as a report; every non-OK line is followed by its fix.
#[must_use]
pub fn render(checks: &[Check]) -> String {
    let mut out = String::new();
    for c in checks {
        let tag = match c.status {
            Status::Ok => "ok  ",
            Status::Warn => "warn",
            Status::Fail => "FAIL",
        };
        out.push_str(&format!("[{tag}] {}: {}\n", c.name, c.detail));
        if let Some(fix) = &c.fix {
            out.push_str(&format!("       fix: {fix}\n"));
        }
    }
    out
}

/// `channel` of a `rust-toolchain.toml`.
///
/// # Errors
/// Returns an error when the text is not TOML or has no `toolchain.channel`.
pub fn parse_channel(text: &str) -> Result<String, String> {
    let v: toml::Value =
        toml::from_str(text).map_err(|e| format!("bad rust-toolchain.toml: {e}"))?;
    v.get("toolchain")
        .and_then(|t| t.get("channel"))
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| "rust-toolchain.toml has no toolchain.channel".to_owned())
}

/// `[workspace.package] rust-version` (the MSRV) of a workspace manifest.
///
/// # Errors
/// Returns an error when the manifest has no `workspace.package.rust-version`.
pub fn parse_msrv(text: &str) -> Result<String, String> {
    let v: toml::Value = toml::from_str(text).map_err(|e| format!("bad Cargo.toml: {e}"))?;
    v.get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("rust-version"))
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| "Cargo.toml has no workspace.package.rust-version".to_owned())
}

/// The version number from `rustc --version` output (`rustc 1.99.1 (...)`).
#[must_use]
pub fn rustc_version(out: &str) -> Option<String> {
    out.split_whitespace().nth(1).map(str::to_owned)
}

/// Whether `rustup toolchain list` output contains toolchain `want` (`1.97`
/// matches `1.97-x86_64-...` and `1.97.0-x86_64-...`).
#[must_use]
pub fn has_toolchain(list: &str, want: &str) -> bool {
    list.lines().any(|l| {
        let name = l.split_whitespace().next().unwrap_or("");
        name == want
            || name.strip_prefix(want).is_some_and(|rest| {
                rest.starts_with('-') || (rest.starts_with('.') && want.matches('.').count() < 2)
            })
    })
}

/// Parses `/proc/sys/kernel/perf_event_paranoid`.
#[must_use]
pub fn parse_paranoid(text: &str) -> Option<i32> {
    text.trim().parse().ok()
}

/// Available KiB from `df -Pk <dir>` output (fourth column of the data row).
#[must_use]
pub fn parse_df_avail_kib(out: &str) -> Option<u64> {
    out.lines().nth(1)?.split_whitespace().nth(3)?.parse().ok()
}

/// Classifies free disk space in GiB.
#[must_use]
pub fn disk_status(avail_gib: u64) -> Status {
    if avail_gib < DISK_FAIL_GIB {
        Status::Fail
    } else if avail_gib < DISK_WARN_GIB {
        Status::Warn
    } else {
        Status::Ok
    }
}

/// Whether a Python version string (`3.11.17`, `3.14.8t`) is at least 3.11.
#[must_use]
pub fn python_ok(version: &str) -> bool {
    let mut it = version.split('.');
    let major = it.next().and_then(|s| s.parse::<u32>().ok());
    let minor = it
        .next()
        .and_then(|s| s.trim_end_matches('t').parse::<u32>().ok());
    matches!((major, minor), (Some(3), Some(m)) if m >= 11)
}

/// Arguments after `docker` for `stack <action>`; always scoped to
/// [`STACK_PROJECT`] so no other compose project can be touched.
///
/// # Errors
/// Returns a usage error for an unknown action.
pub fn stack_args(action: &str, kafka: bool, root: &Path) -> Result<Vec<String>, String> {
    let mut a: Vec<String> = ["compose", "-p", STACK_PROJECT, "-f"]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    a.push(root.join(STACK_FILE).display().to_string());
    if kafka {
        a.extend(["--profile".to_owned(), "kafka".to_owned()]);
    }
    let tail: &[&str] = match action {
        "up" => &["up", "-d", "--wait"],
        // No `-v`: named volumes (MinIO data) survive `down`.
        "down" => &["down", "--remove-orphans"],
        "status" => &["ps"],
        other => return Err(format!("unknown stack action {other}; use up|down|status")),
    };
    a.extend(tail.iter().map(|s| (*s).to_owned()));
    Ok(a)
}

fn probe(cmd: &str, args: &[&str], dir: &Path) -> Option<String> {
    let out = Command::new(cmd)
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

fn data_dir() -> PathBuf {
    std::env::var_os("KRISHIV_DATA").map_or_else(
        || {
            std::env::var_os("HOME")
                .map_or_else(|| PathBuf::from("."), PathBuf::from)
                .join("krishiv-data")
        },
        PathBuf::from,
    )
}

fn cargo_tool(name: &str, required: bool, root: &Path) -> Check {
    let bin = format!("cargo-{name}");
    probe("cargo", &[name, "--version"], root).map_or_else(
        || {
            let status = if required { Status::Fail } else { Status::Warn };
            let fix = format!("cargo install --locked {bin}");
            Check::missing(
                &bin,
                status,
                "not found on PATH",
                &fix,
                argv(&["cargo", "install", "--locked", &bin]),
            )
        },
        |v| Check::ok(&bin, v.lines().next().unwrap_or("").to_owned()),
    )
}

fn toolchain_checks(root: &Path, checks: &mut Vec<Check>) {
    let rustup = probe("rustup", &["--version"], root);
    if rustup.is_none() {
        checks.push(Check::missing(
            "rustup",
            Status::Fail,
            "not found on PATH",
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh",
            None,
        ));
    }
    let channel = std::fs::read_to_string(root.join("rust-toolchain.toml"))
        .map_err(|e| e.to_string())
        .and_then(|t| parse_channel(&t));
    let installed = probe("rustc", &["--version"], root).and_then(|o| rustc_version(&o));
    match (installed, channel) {
        (None, ch) => {
            let want = ch.unwrap_or_else(|_| "stable".to_owned());
            checks.push(Check::missing(
                "rustc",
                Status::Fail,
                "not found on PATH",
                &format!("rustup toolchain install {want}  (install rustup first if missing)"),
                argv(&["rustup", "toolchain", "install", &want]),
            ));
        }
        (Some(have), Ok(want)) if have == want => {
            checks.push(Check::ok(
                "rustc",
                format!("{have} (matches rust-toolchain.toml)"),
            ));
        }
        (Some(have), Ok(want)) => checks.push(Check::missing(
            "rustc",
            Status::Fail,
            format!("{have} active, rust-toolchain.toml pins {want}"),
            &format!("rustup toolchain install {want}"),
            argv(&["rustup", "toolchain", "install", &want]),
        )),
        (Some(have), Err(e)) => checks.push(Check::ok("rustc", format!("{have} ({e})"))),
    }
    let msrv = std::fs::read_to_string(root.join("Cargo.toml"))
        .map_err(|e| e.to_string())
        .and_then(|t| parse_msrv(&t));
    if let Ok(msrv) = msrv {
        let list = probe("rustup", &["toolchain", "list"], root).unwrap_or_default();
        let name = format!("MSRV toolchain {msrv}");
        if has_toolchain(&list, &msrv) {
            checks.push(Check::ok(&name, "installed (cargo +MSRV check)"));
        } else {
            checks.push(Check::missing(
                &name,
                Status::Warn,
                "not installed; CI runs `cargo check` on it",
                &format!("rustup toolchain install {msrv} --profile minimal"),
                argv(&[
                    "rustup",
                    "toolchain",
                    "install",
                    &msrv,
                    "--profile",
                    "minimal",
                ]),
            ));
        }
    }
    checks.push(cargo_tool("nextest", true, root));
    for t in ["llvm-cov", "deny", "hack", "machete"] {
        checks.push(cargo_tool(t, false, root));
    }
}

fn python_checks(root: &Path, checks: &mut Vec<Check>) {
    if probe("uv", &["--version"], root).is_none() {
        checks.push(Check::missing(
            "uv",
            Status::Fail,
            "not found on PATH",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            None,
        ));
        return;
    }
    checks.push(Check::ok(
        "uv",
        probe("uv", &["--version"], root).unwrap_or_default(),
    ));
    let (status, name) = (Status::Fail, "python >= 3.11");
    match probe("uv", &["python", "find", ">=3.11"], root) {
        Some(p) => checks.push(Check::ok(name, p)),
        None => checks.push(Check::missing(
            name,
            status,
            "no CPython 3.11+ found",
            "uv python install 3.11",
            argv(&["uv", "python", "install", "3.11"]),
        )),
    }
    match probe("uv", &["python", "find", "3.14t"], root) {
        Some(p) => checks.push(Check::ok("python 3.14t (free-threaded)", p)),
        None => checks.push(Check::missing(
            "python 3.14t (free-threaded)",
            Status::Warn,
            "not found; needed for the cp314t wheel",
            "uv python install 3.14t",
            argv(&["uv", "python", "install", "3.14t"]),
        )),
    }
    if probe("maturin", &["--version"], root).is_some() {
        checks.push(Check::ok("maturin", "on PATH"));
    } else {
        checks.push(Check::missing(
            "maturin",
            Status::Warn,
            "not on PATH (`uvx maturin` also works)",
            "uv tool install maturin",
            argv(&["uv", "tool", "install", "maturin"]),
        ));
    }
}

fn host_checks(root: &Path, checks: &mut Vec<Check>) {
    if probe("git", &["--version"], root).is_none() {
        checks.push(Check::missing(
            "git",
            Status::Fail,
            "not found on PATH",
            "install git with your package manager (for example: sudo apt install git)",
            None,
        ));
    } else {
        let hooks = probe("git", &["config", "core.hooksPath"], root).unwrap_or_default();
        if hooks == ".githooks" {
            checks.push(Check::ok("git hooks", "core.hooksPath = .githooks"));
        } else {
            checks.push(Check::missing(
                "git hooks",
                Status::Warn,
                "not installed",
                "cargo xtask setup",
                None,
            ));
        }
    }
    let docker = probe("docker", &["compose", "version"], root);
    checks.push(docker.map_or_else(
        || {
            Check::missing(
                "docker compose",
                Status::Warn,
                "not found; needed for `stack`, actionlint and markdownlint",
                "install Docker Engine and the compose plugin: https://docs.docker.com/engine/install/",
                None,
            )
        },
        |v| Check::ok("docker compose", v),
    ));
    if let Some(p) = std::fs::read_to_string("/proc/sys/kernel/perf_event_paranoid")
        .ok()
        .and_then(|t| parse_paranoid(&t))
    {
        if p <= 1 {
            checks.push(Check::ok("perf_event_paranoid", p.to_string()));
        } else {
            checks.push(Check::missing(
                "perf_event_paranoid",
                Status::Warn,
                format!("{p}; hardware counters for benchmarks need <= 1"),
                "sudo sysctl kernel.perf_event_paranoid=1  (run it yourself; xtask never uses sudo)",
                None,
            ));
        }
    }
    let target = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from);
    let probe_dir = target
        .as_deref()
        .filter(|p| p.exists())
        .unwrap_or(root)
        .to_path_buf();
    let avail = probe("df", &["-Pk", &probe_dir.display().to_string()], root)
        .and_then(|o| parse_df_avail_kib(&o))
        .map(|k| k / (1024 * 1024));
    if let Some(gib) = avail {
        let st = disk_status(gib);
        let detail = format!("{gib} GiB free on {}", probe_dir.display());
        if st == Status::Ok {
            checks.push(Check::ok("disk space", detail));
        } else {
            checks.push(Check::missing(
                "disk space",
                st,
                detail,
                "free space, or point CARGO_TARGET_DIR at a larger disk (one directory per worktree)",
                None,
            ));
        }
    }
    let linked_worktree = root.join(".git").is_file();
    match (&target, linked_worktree) {
        (Some(t), _) => checks.push(Check::ok("CARGO_TARGET_DIR", t.display().to_string())),
        (None, true) => checks.push(Check::missing(
            "CARGO_TARGET_DIR",
            Status::Warn,
            "unset in a linked git worktree; builds share nothing with other worktrees",
            &format!(
                "export CARGO_TARGET_DIR=<big disk>/krishiv-target/{}",
                root.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("worktree")
            ),
            None,
        )),
        (None, false) => checks.push(Check::ok("CARGO_TARGET_DIR", "default (target/)")),
    }
    let data = data_dir();
    if data.is_dir() {
        checks.push(Check::ok("datasets dir", data.display().to_string()));
    } else {
        checks.push(Check::missing(
            "datasets dir",
            Status::Warn,
            format!("{} does not exist", data.display()),
            &format!(
                "mkdir -p {}  (then see bench/README.md to generate data)",
                data.display()
            ),
            None,
        ));
    }
}

/// Runs every doctor check against the workspace at `root`.
#[must_use]
pub fn run_checks(root: &Path) -> Vec<Check> {
    let mut checks = Vec::new();
    host_checks(root, &mut checks);
    toolchain_checks(root, &mut checks);
    python_checks(root, &mut checks);
    checks
}

/// `xtask doctor`: prints the report; fails when a prerequisite is missing.
///
/// # Errors
/// Returns the list of failed prerequisites with their fixes.
pub fn run_doctor(root: &Path) -> Result<(), String> {
    let checks = run_checks(root);
    print!("{}", render(&checks));
    if has_failures(&checks) {
        let failed: Vec<&str> = checks
            .iter()
            .filter(|c| c.status == Status::Fail)
            .map(|c| c.name.as_str())
            .collect();
        Err(format!(
            "doctor: missing prerequisites: {}. Apply the fixes above.",
            failed.join(", ")
        ))
    } else {
        println!("doctor: all required prerequisites present");
        Ok(())
    }
}

/// `xtask setup [--install]`: installs the git hooks, validates the
/// environment and prints (or, with `--install`, runs) the user-level fixes.
///
/// # Errors
/// Returns an error when the hooks cannot be installed or an install fails.
pub fn run_setup_full(root: &Path, install: bool) -> Result<(), String> {
    crate::policy::run_setup(root)?;
    let checks = run_checks(root);
    print!("{}", render(&checks));
    for c in checks.iter().filter(|c| c.status != Status::Ok) {
        let Some(fix) = &c.fix else { continue };
        if install && let Some(cmd) = &c.auto {
            println!("setup: running `{}`", cmd.join(" "));
            let status = Command::new(&cmd[0])
                .args(&cmd[1..])
                .current_dir(root)
                .status()
                .map_err(|e| format!("failed to run {}: {e}", cmd[0]))?;
            if !status.success() {
                return Err(format!("`{}` failed", cmd.join(" ")));
            }
        } else if !install {
            println!("setup: to fix {}: {fix}", c.name);
        }
    }
    println!(
        "setup: done; `cargo xtask doctor` re-checks (add `--install` to run user-level fixes)"
    );
    Ok(())
}

/// `xtask stack up|down|status [--kafka]` against the `krishiv-bench` project.
///
/// # Errors
/// Returns an error when docker is missing, the data directory is missing for
/// `up`, or the compose command fails.
pub fn run_stack(root: &Path, action: &str, kafka: bool) -> Result<(), String> {
    let args = stack_args(action, kafka, root)?;
    if probe("docker", &["compose", "version"], root).is_none() {
        return Err(
            "docker compose not found; install Docker Engine + the compose plugin".to_owned(),
        );
    }
    let data = data_dir();
    if action == "up" && !data.is_dir() {
        return Err(format!(
            "datasets dir {} does not exist; run `mkdir -p {}` (compose would create it root-owned)",
            data.display(),
            data.display()
        ));
    }
    println!("stack: docker {}", args.join(" "));
    let status = Command::new("docker")
        .args(&args)
        .env("KRISHIV_DATA", &data)
        .current_dir(root)
        .status()
        .map_err(|e| format!("failed to run docker: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("docker compose {action} failed"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_and_msrv() {
        assert_eq!(
            parse_channel("[toolchain]\nchannel = \"1.99.1\"\n").unwrap_or_default(),
            "1.99.1"
        );
        assert!(parse_channel("[toolchain]\n").is_err());
        let m = "[workspace.package]\nrust-version = \"1.97\"\n";
        assert_eq!(parse_msrv(m).unwrap_or_default(), "1.97");
        assert!(parse_msrv("[workspace]\n").is_err());
    }

    #[test]
    fn versions() {
        assert_eq!(
            rustc_version("rustc 1.99.1 (abc 2026-10-01)").as_deref(),
            Some("1.99.1")
        );
        assert_eq!(rustc_version(""), None);
        assert!(python_ok("3.11.17") && python_ok("3.14.8t") && python_ok("3.12"));
        assert!(!python_ok("3.10.22") && !python_ok("2.7") && !python_ok("x"));
    }

    #[test]
    fn toolchain_list_matching() {
        let l = "stable-x86_64-unknown-linux-gnu\n1.97-x86_64-unknown-linux-gnu\n1.99.1-x86_64-unknown-linux-gnu (active)\n";
        assert!(has_toolchain(l, "1.97"));
        assert!(has_toolchain(l, "1.99.1"));
        assert!(!has_toolchain(l, "1.9"));
        assert!(!has_toolchain(l, "1.99.2"));
        assert!(has_toolchain("1.97.0-x86_64-unknown-linux-gnu\n", "1.97"));
    }

    #[test]
    fn system_parsers() {
        assert_eq!(parse_paranoid("2\n"), Some(2));
        assert_eq!(parse_paranoid("x"), None);
        let df = "Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/x 100 50 52428800 50% /\n";
        assert_eq!(parse_df_avail_kib(df), Some(52_428_800));
        assert_eq!(parse_df_avail_kib("Filesystem\n"), None);
        assert_eq!(disk_status(4), Status::Fail);
        assert_eq!(disk_status(10), Status::Warn);
        assert_eq!(disk_status(100), Status::Ok);
    }

    #[test]
    fn report_names_the_fix() {
        let c = vec![
            Check::ok("a", "fine"),
            Check::missing("b", Status::Fail, "gone", "install b", None),
        ];
        let text = render(&c);
        assert!(text.contains("[ok  ] a: fine"));
        assert!(text.contains("[FAIL] b: gone\n       fix: install b"));
        assert!(has_failures(&c));
        assert!(!has_failures(&c[..1]));
    }

    #[test]
    fn stack_is_scoped_to_the_bench_project() {
        let root = Path::new("/r");
        let up = stack_args("up", false, root).unwrap_or_default();
        assert_eq!(&up[..4], ["compose", "-p", "krishiv-bench", "-f"]);
        assert!(up.ends_with(&["up".to_owned(), "-d".to_owned(), "--wait".to_owned()]));
        let down = stack_args("down", true, root).unwrap_or_default();
        assert!(down.contains(&"kafka".to_owned()));
        assert!(!down.contains(&"-v".to_owned()));
        assert!(stack_args("rm", false, root).is_err());
    }
}
