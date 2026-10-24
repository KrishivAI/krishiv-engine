// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask profile <QUERY | -f FILE>`: the query profile as JSON plus a
//! sampling-profiler flamegraph.
//!
//! The decision logic (argument parsing, tool discovery over a `PATH` string,
//! command-line construction, output-directory handling and the profile
//! summary) is pure and unit-tested; [`run_profile`] does the I/O. Nothing
//! here ever runs `sudo` or changes the system's `perf_event_paranoid`.
#![expect(clippy::print_stdout, reason = "command output is the interface")]

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Which sampler to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolChoice {
    /// `perf` with the inferno binaries when all three exist, else `samply`.
    Auto,
    /// `perf record` plus `inferno-collapse-perf` / `inferno-flamegraph`.
    Perf,
    /// `samply record`.
    Samply,
}

/// The sampler [`select_tool`] picked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedTool {
    /// `perf` plus the inferno binaries.
    Perf,
    /// `samply`.
    Samply,
}

/// Which sampler binaries exist on a `PATH`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Discovery {
    /// `perf` was found.
    pub perf: bool,
    /// `inferno-collapse-perf` was found.
    pub collapse: bool,
    /// `inferno-flamegraph` was found.
    pub flamegraph: bool,
    /// `samply` was found.
    pub samply: bool,
}

/// The parsed `cargo xtask profile` command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileArgs {
    /// SQL text, when the query was passed inline.
    pub query: Option<String>,
    /// File holding the SQL, when `-f` was passed.
    pub file: Option<PathBuf>,
    /// `--table NAME=PATH` entries.
    pub tables: Vec<(String, String)>,
    /// `--conf KEY=VALUE` entries.
    pub confs: Vec<(String, String)>,
    /// `--tool` choice.
    pub tool: ToolChoice,
    /// `--out DIR`, when given.
    pub out: Option<PathBuf>,
    /// `--no-flamegraph` was passed.
    pub no_flamegraph: bool,
}

/// `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`, so the release CLI binary
/// the sampler sees resolves symbols.
pub const LINE_TABLES_ENV: (&str, &str) = ("CARGO_PROFILE_RELEASE_DEBUG", "line-tables-only");

/// Name of the query-profile document the CLI writes.
pub const PROFILE_JSON: &str = "profile.json";
/// `perf record` output in the out directory.
pub const PERF_DATA: &str = "perf.data";
/// The rendered flamegraph in the out directory.
pub const FLAMEGRAPH_SVG: &str = "flamegraph.svg";
/// `samply record --save-only` output in the out directory (samply's own
/// format, not the query profile).
pub const SAMPLY_OUT: &str = "profile.json.gz";

/// Splits `KEY=VALUE`, rejecting a missing `=` or an empty key.
fn key_value(s: &str, flag: &str) -> Result<(String, String), String> {
    match s.split_once('=') {
        Some((k, v)) if !k.trim().is_empty() => Ok((k.trim().to_owned(), v.to_owned())),
        _ => Err(format!("{flag} expects NAME=VALUE, got {s:?}")),
    }
}

/// Parses the `cargo xtask profile` command line.
///
/// # Errors
/// Returns a usage error for unknown flags, missing values, a missing query,
/// or both inline SQL and `-f`.
pub fn parse_args(flags: &[&str]) -> Result<ProfileArgs, String> {
    let mut args = ProfileArgs {
        query: None,
        file: None,
        tables: Vec::new(),
        confs: Vec::new(),
        tool: ToolChoice::Auto,
        out: None,
        no_flamegraph: false,
    };
    let mut it = flags.iter();
    while let Some(f) = it.next() {
        match *f {
            "--table" => {
                let v = it.next().ok_or("--table needs NAME=PATH")?;
                args.tables.push(key_value(v, "--table")?);
            }
            "--conf" => {
                let v = it.next().ok_or("--conf needs KEY=VALUE")?;
                args.confs.push(key_value(v, "--conf")?);
            }
            "--tool" => {
                let v = it.next().ok_or("--tool needs auto|perf|samply")?;
                args.tool = match *v {
                    "auto" => ToolChoice::Auto,
                    "perf" => ToolChoice::Perf,
                    "samply" => ToolChoice::Samply,
                    _ => return Err(format!("--tool expects auto|perf|samply, got {v:?}")),
                };
            }
            "--out" => {
                let v = it.next().ok_or("--out needs a directory")?;
                args.out = Some(PathBuf::from(v));
            }
            "--no-flamegraph" => args.no_flamegraph = true,
            "-f" => {
                let v = it.next().ok_or("-f needs a file")?;
                if args.file.is_some() {
                    return Err("pass -f only once".to_owned());
                }
                args.file = Some(PathBuf::from(v));
            }
            other if other.starts_with('-') => return Err(format!("unknown flag {other}")),
            q => {
                if args.query.is_some() {
                    return Err("pass a single query; quote SQL with spaces".to_owned());
                }
                args.query = Some((*q).to_owned());
            }
        }
    }
    if args.query.is_some() && args.file.is_some() {
        return Err("pass either SQL text or -f FILE, not both".to_owned());
    }
    if args.query.is_none() && args.file.is_none() {
        return Err("no query given; pass SQL text or -f FILE".to_owned());
    }
    Ok(args)
}

/// Whether `bin` exists in `dir`. The suffix is a parameter so tests can
/// cover the platform-suffix lookup on any host.
fn has_bin_with_suffix(dir: &Path, bin: &str, suffix: &str) -> bool {
    if dir.join(bin).is_file() {
        return true;
    }
    !suffix.is_empty() && dir.join(format!("{bin}{suffix}")).is_file()
}

/// Whether `bin` exists in `dir` (with or without the platform suffix, so
/// fake executables in tests need no extension).
fn has_bin(dir: &Path, bin: &str) -> bool {
    has_bin_with_suffix(dir, bin, std::env::consts::EXE_SUFFIX)
}

/// Which sampler binaries the colon-separated `PATH` string holds. Pure over
/// its argument so tests can point it at a temp directory of fakes.
#[must_use]
pub fn discover_on_path(path: &str) -> Discovery {
    let mut d = Discovery {
        perf: false,
        collapse: false,
        flamegraph: false,
        samply: false,
    };
    for dir in std::env::split_paths(path) {
        if dir.as_os_str().is_empty() {
            continue;
        }
        d.perf = d.perf || has_bin(&dir, "perf");
        d.collapse = d.collapse || has_bin(&dir, "inferno-collapse-perf");
        d.flamegraph = d.flamegraph || has_bin(&dir, "inferno-flamegraph");
        d.samply = d.samply || has_bin(&dir, "samply");
    }
    d
}

/// What to print when no sampler is available: which tools were looked for
/// and how to install each. The `perf_event_paranoid` note is separate (see
/// [`perf_permission_note`]): it applies when `perf record` runs but the
/// kernel refuses, not when the tools are missing.
#[must_use]
pub fn no_sampler_message(d: &Discovery) -> String {
    let mut missing: Vec<&str> = Vec::new();
    if !d.perf {
        missing.push("perf");
    }
    if !d.collapse {
        missing.push("inferno-collapse-perf");
    }
    if !d.flamegraph {
        missing.push("inferno-flamegraph");
    }
    if !d.samply {
        missing.push("samply");
    }
    format!(
        "no sampling profiler found on PATH (missing: {}); install one of:\n\
         - perf: the distribution's `linux-tools` package (for example `apt install linux-tools-generic`)\n\
         - inferno: `cargo install inferno` (provides `inferno-collapse-perf` and `inferno-flamegraph`)\n\
         - samply: `cargo install samply`",
        missing.join(", ")
    )
}

/// What to print when `perf record` fails because the kernel blocks
/// unprivileged profiling. xtask never changes the setting itself; a
/// privileged user lowers it.
#[must_use]
pub fn perf_permission_note() -> String {
    "the kernel's `perf_event_paranoid` setting blocks unprivileged `perf record`; \
     a privileged user can lower it (for example `sysctl kernel.perf_event_paranoid=1`). \
     xtask never changes this setting and never runs `sudo`"
        .to_owned()
}

/// Whether `perf record`'s stderr reports a permission failure (the kernel
/// refusing the run, not a bad flag or a missing binary).
#[must_use]
pub fn is_perf_permission_error(stderr: &str) -> bool {
    let lower = stderr.to_lowercase();
    lower.contains("perf_event_paranoid")
        || lower.contains("permission")
        || lower.contains("operation not permitted")
        || lower.contains("access denied")
}

/// Picks the sampler: explicit `--tool` values need their binaries;
/// `auto` prefers `perf` with both inferno binaries and falls back to
/// `samply` (so `perf` without inferno still profiles via `samply`).
///
/// # Errors
/// Returns the install hints when the requested binaries are missing.
pub fn select_tool(requested: &ToolChoice, d: &Discovery) -> Result<SelectedTool, String> {
    match requested {
        ToolChoice::Perf => {
            if !d.perf {
                return Err(format!(
                    "`--tool perf` needs `perf` on PATH (the distribution's `linux-tools` package).\n{}",
                    no_sampler_message(d)
                ));
            }
            if !d.collapse || !d.flamegraph {
                return Err(
                    "`--tool perf` needs `inferno-collapse-perf` and `inferno-flamegraph` on PATH (`cargo install inferno`)".to_owned(),
                );
            }
            Ok(SelectedTool::Perf)
        }
        ToolChoice::Samply => {
            if !d.samply {
                return Err(format!(
                    "`--tool samply` needs `samply` on PATH (`cargo install samply`).\n{}",
                    no_sampler_message(d)
                ));
            }
            Ok(SelectedTool::Samply)
        }
        ToolChoice::Auto => {
            if d.perf && d.collapse && d.flamegraph {
                Ok(SelectedTool::Perf)
            } else if d.samply {
                Ok(SelectedTool::Samply)
            } else {
                Err(no_sampler_message(d))
            }
        }
    }
}

/// `cargo build -p krishiv-cli --release` (with [`LINE_TABLES_ENV`] set by
/// the caller).
#[must_use]
pub fn cargo_build_argv() -> Vec<OsString> {
    ["build", "-p", "krishiv-cli", "--release"]
        .iter()
        .map(OsString::from)
        .collect()
}

/// `krishiv sql --explain-analyze --json ...`: the query plus its tables and
/// conf. `query` is `(inline SQL, None)` or `((), Some(file))`; the SQL is
/// one argument (`-e`), never interpolated into a shell.
#[must_use]
pub fn cli_sql_argv(
    query: &Option<String>,
    file: &Option<PathBuf>,
    tables: &[(String, String)],
    confs: &[(String, String)],
) -> Vec<OsString> {
    let mut argv = vec![
        OsString::from("sql"),
        OsString::from("--explain-analyze"),
        OsString::from("--json"),
    ];
    for (name, path) in tables {
        argv.push(OsString::from("--table"));
        argv.push(OsString::from(format!("{name}={path}")));
    }
    for (key, value) in confs {
        argv.push(OsString::from("--conf"));
        argv.push(OsString::from(format!("{key}={value}")));
    }
    if let Some(q) = query {
        argv.push(OsString::from("-e"));
        argv.push(OsString::from(q));
    }
    if let Some(f) = file {
        argv.push(OsString::from("-f"));
        argv.push(f.as_os_str().to_owned());
    }
    argv
}

/// `perf record -F 999 -g --call-graph dwarf -o PERF_DATA -- CLI...`.
#[must_use]
pub fn perf_record_argv(perf_data: &Path, cli_bin: &Path, cli_args: &[OsString]) -> Vec<OsString> {
    let mut argv = vec![
        OsString::from("perf"),
        OsString::from("record"),
        OsString::from("-F"),
        OsString::from("999"),
        OsString::from("-g"),
        OsString::from("--call-graph"),
        OsString::from("dwarf"),
        OsString::from("-o"),
        perf_data.as_os_str().to_owned(),
        OsString::from("--"),
        cli_bin.as_os_str().to_owned(),
    ];
    argv.extend(cli_args.iter().cloned());
    argv
}

/// `perf script -i PERF_DATA`.
#[must_use]
pub fn perf_script_argv(perf_data: &Path) -> Vec<OsString> {
    vec![
        OsString::from("perf"),
        OsString::from("script"),
        OsString::from("-i"),
        perf_data.as_os_str().to_owned(),
    ]
}

/// `inferno-collapse-perf` (stdin folded stacks, stdout collapsed).
#[must_use]
pub fn collapse_argv() -> Vec<OsString> {
    vec![OsString::from("inferno-collapse-perf")]
}

/// `inferno-flamegraph` (stdin collapsed stacks, stdout SVG).
#[must_use]
pub fn flamegraph_argv() -> Vec<OsString> {
    vec![OsString::from("inferno-flamegraph")]
}

/// `samply record --save-only -o OUT -- CLI...` (samply's own format).
#[must_use]
pub fn samply_record_argv(out: &Path, cli_bin: &Path, cli_args: &[OsString]) -> Vec<OsString> {
    let mut argv = vec![
        OsString::from("samply"),
        OsString::from("record"),
        OsString::from("--save-only"),
        OsString::from("-o"),
        out.as_os_str().to_owned(),
        OsString::from("--"),
        cli_bin.as_os_str().to_owned(),
    ];
    argv.extend(cli_args.iter().cloned());
    argv
}

/// The default out directory: `target/xtask-profile/<UTC timestamp>` under
/// the workspace root. `now_secs` is seconds since the Unix epoch (UTC); it
/// is a parameter so the path is deterministic in tests.
#[must_use]
pub fn default_out_dir(root: &Path, now_secs: u64) -> PathBuf {
    root.join("target")
        .join("xtask-profile")
        .join(now_secs.to_string())
}

/// Current time as seconds since the Unix epoch (UTC), or zero when the clock
/// is before it.
#[must_use]
pub fn utc_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Creates the out directory, refusing to overwrite an existing one.
///
/// # Errors
/// Returns a message when the directory already exists or cannot be created.
pub fn create_out_dir(dir: &Path) -> Result<(), String> {
    if dir.exists() {
        return Err(format!(
            "refusing to overwrite existing {}: pass --out DIR",
            dir.display()
        ));
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))
}

/// Reads the query: inline SQL, or the file `-f` named (relative to `root`).
/// `parse_args` guarantees exactly one source; the last arm is defensive.
///
/// # Errors
/// Returns a message when the file cannot be read or both sources are set.
fn resolve_query_text(
    query: &Option<String>,
    file: &Option<PathBuf>,
    root: &Path,
) -> Result<String, String> {
    match (query, file) {
        (Some(q), None) => Ok(q.clone()),
        (None, Some(f)) => std::fs::read_to_string(root.join(f))
            .map_err(|e| format!("cannot read {}: {e}", f.display())),
        _ => Err("pass either SQL text or -f FILE, not both".to_owned()),
    }
}

fn u64_of(v: &serde_json::Value, key: &str) -> Result<u64, String> {
    v.get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("profile: {key} is not an unsigned integer"))
}

/// One operator's compute time, for the summary.
struct SlowOp {
    name: String,
    compute_ns: u64,
}

/// Summarizes the profile JSON the CLI wrote: the query wall time and the
/// three slowest operators by compute time.
///
/// # Errors
/// Returns a message when the document is not a query profile.
pub fn summarize_profile(text: &str) -> Result<String, String> {
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("profile is not JSON: {e}"))?;
    let query = v
        .get("query")
        .ok_or_else(|| "profile has no query member".to_owned())?;
    let wall_ns = u64_of(query, "wall_time_ns")?;
    let ops = v
        .get("operators")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "profile has no operators array".to_owned())?;
    let mut slowest: Vec<SlowOp> = Vec::new();
    for op in ops {
        let name = op
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "profile has an operator without a name".to_owned())?;
        let metrics = op
            .get("metrics")
            .ok_or_else(|| format!("profile operator {name:?} has no metrics"))?;
        slowest.push(SlowOp {
            name: name.to_owned(),
            compute_ns: u64_of(metrics, "compute_time_ns")?,
        });
    }
    slowest.sort_by_key(|op| std::cmp::Reverse(op.compute_ns));
    slowest.truncate(3);
    let mut out = format!(
        "query wall time: {} ns ({:.3} ms)\nslowest operators:",
        wall_ns,
        wall_ns as f64 / 1_000_000.0
    );
    for op in &slowest {
        out.push_str(&format!(
            "\n- {}: {} ns ({:.3} ms) compute",
            op.name,
            op.compute_ns,
            op.compute_ns as f64 / 1_000_000.0
        ));
    }
    Ok(out)
}

/// The error for a failed `perf record` run: the command with stderr, plus
/// the [`perf_permission_note`] when the kernel refused the run.
#[must_use]
pub fn perf_record_error(stderr: &str) -> String {
    let mut msg = format!("`perf record` failed: {}", stderr.trim());
    if is_perf_permission_error(stderr) {
        msg.push_str(&format!("\n{}", perf_permission_note()));
    }
    msg
}

/// Runs `prog args...` in `root`, returning stdout on success and the
/// command with stderr on failure.
fn run_argv(prog: &OsStr, args: &[OsString], root: &Path) -> Result<Vec<u8>, String> {
    let out = Command::new(prog)
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|e| format!("cannot run {}: {e}", prog.to_string_lossy()))?;
    if out.status.success() {
        Ok(out.stdout)
    } else {
        let mut shown = vec![prog.to_string_lossy().into_owned()];
        shown.extend(args.iter().map(|a| a.to_string_lossy().into_owned()));
        Err(format!(
            "`{}` failed: {}",
            shown.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
}

/// Runs the query under `perf record`, then pipes `perf script` through the
/// inferno binaries into `flamegraph.svg`. No shell is involved: the pipe is
/// three processes with wired stdio. The binary paths are parameters so tests
/// can substitute stubs; production passes the `PATH`-resolved tool names.
fn run_perf_with(
    root: &Path,
    cli_bin: &Path,
    cli_args: &[OsString],
    out_dir: &Path,
    perf_bin: &OsStr,
    collapse_bin: &OsStr,
    flamegraph_bin: &OsStr,
) -> Result<(), String> {
    let perf_data = out_dir.join(PERF_DATA);
    let svg = out_dir.join(FLAMEGRAPH_SVG);
    let record = perf_record_argv(&perf_data, cli_bin, cli_args);
    let out = Command::new(perf_bin)
        .args(&record[1..])
        .current_dir(root)
        .output()
        .map_err(|e| format!("cannot run perf: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        return Err(perf_record_error(&stderr));
    }
    // `perf script -i perf.data | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg`.
    let script = perf_script_argv(&perf_data);
    let mut script_child = Command::new(perf_bin)
        .args(&script[1..])
        .current_dir(root)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("cannot run perf script: {e}"))?;
    let script_out = script_child
        .stdout
        .take()
        .ok_or_else(|| "cannot pipe perf script output".to_owned())?;
    let mut collapse_child = Command::new(collapse_bin)
        .args(&collapse_argv()[1..])
        .stdin(script_out)
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| {
            let _ = script_child.kill();
            format!("cannot run inferno-collapse-perf: {e}")
        })?;
    let collapse_out = collapse_child
        .stdout
        .take()
        .ok_or_else(|| "cannot pipe inferno-collapse-perf output".to_owned())?;
    let svg_file = std::fs::File::create(&svg).map_err(|e| {
        let _ = script_child.kill();
        let _ = collapse_child.kill();
        format!("cannot write {}: {e}", svg.display())
    })?;
    let flame_status = Command::new(flamegraph_bin)
        .args(&flamegraph_argv()[1..])
        .stdin(collapse_out)
        .stdout(svg_file)
        .status()
        .map_err(|e| {
            let _ = script_child.kill();
            let _ = collapse_child.kill();
            format!("cannot run inferno-flamegraph: {e}")
        })?;
    // The pipe closes once every stage is reaped; a later stage failing
    // hides nothing because each stage's error names it.
    let collapse_status = collapse_child
        .wait()
        .map_err(|e| format!("inferno-collapse-perf failed: {e}"))?;
    let script_status = script_child
        .wait()
        .map_err(|e| format!("perf script failed: {e}"))?;
    if !script_status.success() {
        return Err("`perf script` failed".to_owned());
    }
    if !collapse_status.success() {
        return Err("`inferno-collapse-perf` failed".to_owned());
    }
    if !flame_status.success() {
        return Err("`inferno-flamegraph` failed".to_owned());
    }
    println!("wrote {}", svg.display());
    Ok(())
}

/// Runs the query under `perf record` plus the inferno binaries from `PATH`.
fn run_perf(
    root: &Path,
    cli_bin: &Path,
    cli_args: &[OsString],
    out_dir: &Path,
) -> Result<(), String> {
    run_perf_with(
        root,
        cli_bin,
        cli_args,
        out_dir,
        OsStr::new("perf"),
        OsStr::new("inferno-collapse-perf"),
        OsStr::new("inferno-flamegraph"),
    )
}

/// Runs the query under `samply record --save-only`. The binary is a
/// parameter so tests can substitute a stub; production passes `samply`.
fn run_samply_with(
    root: &Path,
    cli_bin: &Path,
    cli_args: &[OsString],
    out_dir: &Path,
    samply_bin: &OsStr,
) -> Result<(), String> {
    let out = out_dir.join(SAMPLY_OUT);
    let argv = samply_record_argv(&out, cli_bin, cli_args);
    let status = Command::new(samply_bin)
        .args(&argv[1..])
        .current_dir(root)
        .status()
        .map_err(|e| format!("cannot run samply: {e}"))?;
    if !status.success() {
        return Err(format!("`samply record` failed ({status})"));
    }
    println!("wrote {}", out.display());
    Ok(())
}

/// Runs the query under `samply record --save-only` from `PATH`.
fn run_samply(
    root: &Path,
    cli_bin: &Path,
    cli_args: &[OsString],
    out_dir: &Path,
) -> Result<(), String> {
    run_samply_with(root, cli_bin, cli_args, out_dir, OsStr::new("samply"))
}

/// `cargo xtask profile <QUERY | -f FILE>` with an explicit environment:
/// `cargo` is the build binary, `path` the `PATH` string the sampler search
/// uses. [`run_profile`] reads both from the process environment; tests pass
/// fakes here instead of mutating it.
///
/// # Errors
/// Returns a message when the arguments are bad, the build or the query
/// fails, the out directory exists, or no sampler is installed (the profile
/// JSON is still written and reported first).
pub fn run_profile_with_env(
    root: &Path,
    flags: &[&str],
    cargo: &OsStr,
    path: &str,
) -> Result<(), String> {
    let args = parse_args(flags)?;
    let query_text = resolve_query_text(&args.query, &args.file, root)?;
    let out_dir = args
        .out
        .clone()
        .unwrap_or_else(|| default_out_dir(root, utc_now_secs()));
    create_out_dir(&out_dir)?;

    // (1) The release CLI with line tables, so the sampler resolves symbols.
    let build_status = Command::new(cargo)
        .args(cargo_build_argv())
        .env(LINE_TABLES_ENV.0, LINE_TABLES_ENV.1)
        .current_dir(root)
        .status()
        .map_err(|e| format!("cannot run cargo build: {e}"))?;
    if !build_status.success() {
        return Err("cargo build -p krishiv-cli --release failed".to_owned());
    }
    let bin = root
        .join("target")
        .join("release")
        .join(format!("krishiv{}", std::env::consts::EXE_SUFFIX));
    if !bin.is_file() {
        return Err(format!("built no binary at {}", bin.display()));
    }

    // (2) The query profile as JSON.
    let query_src = if args.query.is_some() {
        (Some(query_text), None)
    } else {
        (None, args.file.clone())
    };
    let cli_args = cli_sql_argv(&query_src.0, &query_src.1, &args.tables, &args.confs);
    let stdout = run_argv(bin.as_os_str(), &cli_args, root)
        .map_err(|e| format!("krishiv sql --explain-analyze --json failed: {e}"))?;
    let profile_path = out_dir.join(PROFILE_JSON);
    std::fs::write(&profile_path, &stdout)
        .map_err(|e| format!("cannot write {}: {e}", profile_path.display()))?;
    println!("wrote {}", profile_path.display());

    // (3) The flamegraph, unless asked off.
    let mut sampler_err: Option<String> = None;
    if !args.no_flamegraph {
        match select_tool(&args.tool, &discover_on_path(path)) {
            Ok(SelectedTool::Perf) => {
                if let Err(e) = run_perf(root, &bin, &cli_args, &out_dir) {
                    sampler_err = Some(e);
                }
            }
            Ok(SelectedTool::Samply) => {
                if let Err(e) = run_samply(root, &bin, &cli_args, &out_dir) {
                    sampler_err = Some(e);
                }
            }
            Err(e) => sampler_err = Some(e),
        }
    }

    // The summary reads the profile the CLI just wrote.
    let text = std::fs::read_to_string(&profile_path)
        .map_err(|e| format!("cannot read {}: {e}", profile_path.display()))?;
    let summary = summarize_profile(&text)?;
    println!("output: {}", out_dir.display());
    println!("{summary}");
    if let Some(e) = sampler_err {
        return Err(e);
    }
    Ok(())
}

/// `cargo xtask profile <QUERY | -f FILE> [--table NAME=PATH]... [--conf KEY=VALUE]...
/// [--tool auto|perf|samply] [--out DIR] [--no-flamegraph]`: builds the CLI in
/// release with line tables, writes `<out>/profile.json` via
/// `krishiv sql --explain-analyze --json`, then samples the query again for a
/// flamegraph unless `--no-flamegraph` was passed.
///
/// # Errors
/// As [`run_profile_with_env`]; `CARGO` and `PATH` come from the process
/// environment.
pub fn run_profile(root: &Path, flags: &[&str]) -> Result<(), String> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let path = std::env::var("PATH").unwrap_or_default();
    run_profile_with_env(root, flags, &cargo, &path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_text_comes_from_sql_or_file() {
        let root = tempfile::Builder::new()
            .prefix("xtask-profile-query-")
            .tempdir()
            .expect("tempdir");
        assert_eq!(
            resolve_query_text(&Some("SELECT 1".to_owned()), &None, root.path()),
            Ok("SELECT 1".to_owned())
        );
        std::fs::write(root.path().join("q.sql"), "SELECT 2").expect("write");
        assert_eq!(
            resolve_query_text(&None, &Some(PathBuf::from("q.sql")), root.path()),
            Ok("SELECT 2".to_owned())
        );
        assert!(
            resolve_query_text(&None, &Some(PathBuf::from("missing.sql")), root.path())
                .unwrap_err()
                .contains("cannot read")
        );
        // Defensive: `parse_args` never produces these, but the error names
        // the fix if it ever happens.
        assert!(
            resolve_query_text(&None, &None, root.path())
                .unwrap_err()
                .contains("not both")
        );
        assert!(
            resolve_query_text(
                &Some("SELECT 1".to_owned()),
                &Some(PathBuf::from("q.sql")),
                root.path(),
            )
            .unwrap_err()
            .contains("not both")
        );
    }

    #[test]
    fn suffix_lookup_covers_plain_and_suffixed_names() {
        let dir = tempfile::Builder::new()
            .prefix("xtask-profile-suffix-")
            .tempdir()
            .expect("tempdir");
        std::fs::write(dir.path().join("plain"), "x").expect("write");
        std::fs::write(dir.path().join("suffixed.exe"), "x").expect("write");
        assert!(has_bin_with_suffix(dir.path(), "plain", ".exe"));
        assert!(has_bin_with_suffix(dir.path(), "plain", ""));
        assert!(has_bin_with_suffix(dir.path(), "suffixed", ".exe"));
        assert!(!has_bin_with_suffix(dir.path(), "suffixed", ""));
        assert!(!has_bin_with_suffix(dir.path(), "missing", ".exe"));
        assert!(!has_bin_with_suffix(dir.path(), "missing", ""));
        assert!(has_bin(dir.path(), "plain"));
        assert!(!has_bin(dir.path(), "missing"));
    }

    #[test]
    fn argv_runner_reports_stdout_and_failures() {
        let root = tempfile::Builder::new()
            .prefix("xtask-profile-argv-")
            .tempdir()
            .expect("tempdir");
        let out =
            run_argv(OsStr::new("echo"), &[OsString::from("hello")], root.path()).expect("echo");
        assert_eq!(out, b"hello\n");
        let e = run_argv(OsStr::new("false"), &[], root.path()).unwrap_err();
        assert!(e.contains("failed"), "{e}");
        let e = run_argv(
            OsStr::new("/nonexistent/xtask-profile-xyz"),
            &[],
            root.path(),
        )
        .unwrap_err();
        assert!(e.contains("cannot run"), "{e}");
    }

    /// A stub that exits 1 when its first argument equals `fail_arg` (and
    /// prints `stderr` when non-empty). Stages have distinct argument
    /// shapes (`record ...`, `script ...`, none for the inferno tools), so
    /// each failure is deterministic without shared state. Unix only: the
    /// stubs are shell scripts.
    #[cfg(unix)]
    fn arg_stub(dir: &Path, name: &str, fail_arg: &str, stderr: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let body = format!(
            "#!/bin/sh\nif [ -n '{stderr}' ]; then echo '{stderr}' >&2; fi\nif [ \"$1\" = '{fail_arg}' ]; then exit 1; fi\nexit 0\n",
        );
        let p = dir.join(name);
        std::fs::write(&p, body).expect("write stub");
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        p
    }

    #[cfg(unix)]
    fn unit_out_dir(name: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::Builder::new()
            .prefix(&format!("xtask-profile-{name}-"))
            .tempdir()
            .expect("tempdir");
        let out = dir.path().join("out");
        std::fs::create_dir_all(&out).expect("mkdir");
        (dir, out)
    }

    #[test]
    #[cfg(unix)]
    fn perf_pipe_succeeds_with_true_stubs() {
        let (_dir, out) = unit_out_dir("perf-ok");
        let bin = PathBuf::from("/bin/true");
        run_perf_with(
            Path::new("/"),
            Path::new("/b/krishiv"),
            &[],
            &out,
            bin.as_os_str(),
            bin.as_os_str(),
            bin.as_os_str(),
        )
        .expect("perf with true stubs");
        assert!(out.join(FLAMEGRAPH_SVG).is_file());
    }

    #[test]
    #[cfg(unix)]
    fn perf_record_failure_names_the_stage() {
        let (_dir, out) = unit_out_dir("perf-record");
        let e = run_perf_with(
            Path::new("/"),
            Path::new("/b/krishiv"),
            &[],
            &out,
            OsStr::new("/bin/false"),
            OsStr::new("/bin/true"),
            OsStr::new("/bin/true"),
        )
        .unwrap_err();
        assert!(e.contains("`perf record` failed"), "{e}");
        assert!(!e.contains("perf_event_paranoid"), "{e}");
    }

    #[test]
    #[cfg(unix)]
    fn perf_record_permission_failure_adds_the_kernel_note() {
        let dir = tempfile::Builder::new()
            .prefix("xtask-profile-perf-perm-")
            .tempdir()
            .expect("tempdir");
        let failing = arg_stub(
            dir.path(),
            "perf",
            "record",
            "Error: No permission to enable cycles event (perf_event_paranoid)",
        );
        let out = dir.path().join("out");
        std::fs::create_dir_all(&out).expect("mkdir");
        let e = run_perf_with(
            Path::new("/"),
            Path::new("/b/krishiv"),
            &[],
            &out,
            failing.as_os_str(),
            OsStr::new("/bin/true"),
            OsStr::new("/bin/true"),
        )
        .unwrap_err();
        assert!(e.contains("`perf record` failed"), "{e}");
        assert!(e.contains("perf_event_paranoid"), "{e}");
        assert!(e.contains("never"), "{e}");
    }

    #[test]
    #[cfg(unix)]
    fn perf_reports_spawn_and_stage_failures() {
        let e = run_perf_with(
            Path::new("/"),
            Path::new("/b/krishiv"),
            &[],
            Path::new("/nonexistent-out-dir-xyz"),
            OsStr::new("/nonexistent/xtask-profile-xyz"),
            OsStr::new("/bin/true"),
            OsStr::new("/bin/true"),
        )
        .unwrap_err();
        assert!(e.contains("cannot run perf"), "{e}");

        // The record succeeds but the script stage fails.
        let dir = tempfile::Builder::new()
            .prefix("xtask-profile-perf-script-")
            .tempdir()
            .expect("tempdir");
        let stub = arg_stub(dir.path(), "tool", "script", "");
        let out = dir.path().join("out");
        std::fs::create_dir_all(&out).expect("mkdir");
        let e = run_perf_with(
            Path::new("/"),
            Path::new("/b/krishiv"),
            &[],
            &out,
            stub.as_os_str(),
            OsStr::new("/bin/true"),
            OsStr::new("/bin/true"),
        )
        .unwrap_err();
        assert!(e.contains("`perf script` failed"), "{e}");

        // The collapse stage fails (it takes no arguments).
        let dir = tempfile::Builder::new()
            .prefix("xtask-profile-perf-collapse-")
            .tempdir()
            .expect("tempdir");
        let stub = arg_stub(dir.path(), "tool", "", "");
        let out = dir.path().join("out");
        std::fs::create_dir_all(&out).expect("mkdir");
        let e = run_perf_with(
            Path::new("/"),
            Path::new("/b/krishiv"),
            &[],
            &out,
            OsStr::new("/bin/true"),
            stub.as_os_str(),
            OsStr::new("/bin/true"),
        )
        .unwrap_err();
        assert!(e.contains("`inferno-collapse-perf` failed"), "{e}");

        // The flamegraph stage fails (it takes no arguments).
        let dir = tempfile::Builder::new()
            .prefix("xtask-profile-perf-flame-")
            .tempdir()
            .expect("tempdir");
        let stub = arg_stub(dir.path(), "tool", "", "");
        let out = dir.path().join("out");
        std::fs::create_dir_all(&out).expect("mkdir");
        let e = run_perf_with(
            Path::new("/"),
            Path::new("/b/krishiv"),
            &[],
            &out,
            OsStr::new("/bin/true"),
            OsStr::new("/bin/true"),
            stub.as_os_str(),
        )
        .unwrap_err();
        assert!(e.contains("`inferno-flamegraph` failed"), "{e}");

        // A missing collapse binary fails to spawn.
        let dir = tempfile::Builder::new()
            .prefix("xtask-profile-perf-spawn-")
            .tempdir()
            .expect("tempdir");
        let out = dir.path().join("out");
        std::fs::create_dir_all(&out).expect("mkdir");
        let e = run_perf_with(
            Path::new("/"),
            Path::new("/b/krishiv"),
            &[],
            &out,
            OsStr::new("/bin/true"),
            OsStr::new("/nonexistent/xtask-profile-xyz"),
            OsStr::new("/bin/true"),
        )
        .unwrap_err();
        assert!(e.contains("cannot run inferno-collapse-perf"), "{e}");

        // An unwritable svg path fails with its path.
        let dir = tempfile::Builder::new()
            .prefix("xtask-profile-perf-svg-")
            .tempdir()
            .expect("tempdir");
        let blocker = dir.path().join("out");
        std::fs::write(&blocker, "file, not a directory").expect("write");
        let e = run_perf_with(
            Path::new("/"),
            Path::new("/b/krishiv"),
            &[],
            &blocker,
            OsStr::new("/bin/true"),
            OsStr::new("/bin/true"),
            OsStr::new("/bin/true"),
        )
        .unwrap_err();
        assert!(e.contains("cannot write"), "{e}");
    }

    #[test]
    #[cfg(unix)]
    fn samply_runner_reports_its_outcome() {
        let dir = tempfile::Builder::new()
            .prefix("xtask-profile-samply-")
            .tempdir()
            .expect("tempdir");
        let out = dir.path().join("out");
        std::fs::create_dir_all(&out).expect("mkdir");
        run_samply_with(
            Path::new("/"),
            Path::new("/b/krishiv"),
            &[],
            &out,
            OsStr::new("/bin/true"),
        )
        .expect("samply with true stub");
        let e = run_samply_with(
            Path::new("/"),
            Path::new("/b/krishiv"),
            &[],
            &out,
            OsStr::new("/bin/false"),
        )
        .unwrap_err();
        assert!(e.contains("`samply record` failed"), "{e}");
        let e = run_samply_with(
            Path::new("/"),
            Path::new("/b/krishiv"),
            &[],
            &out,
            OsStr::new("/nonexistent/xtask-profile-xyz"),
        )
        .unwrap_err();
        assert!(e.contains("cannot run samply"), "{e}");
    }
}
