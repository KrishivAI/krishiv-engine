// SPDX-License-Identifier: Apache-2.0
//! Tests for `cargo xtask profile`: argument parsing, tool discovery with
//! fake executables, command-line construction, out-directory handling, the
//! failure messages, and `run_profile` with stub build/query binaries.
//! Nothing here runs `perf`, `samply`, the inferno tools or the real build.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

use xtask::profile::{
    FLAMEGRAPH_SVG, LINE_TABLES_ENV, PERF_DATA, PROFILE_FORMAT, PROFILE_JSON, PROFILE_VERSION,
    ProfileArgs, SAMPLY_OUT, SelectedTool, ToolChoice, cargo_build_argv, cli_bin_path,
    cli_sql_argv, collapse_argv, create_out_dir, default_out_dir, discover_on_path,
    flamegraph_argv, is_perf_permission_error, no_sampler_message, perf_permission_note,
    perf_record_argv, perf_record_error, perf_script_argv, run_profile, run_profile_with_env,
    samply_record_argv, select_tool, summarize_profile, utc_now_millis,
};

fn tmp(name: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(&format!("xtask-profile-{name}-"))
        .tempdir()
        .unwrap()
}

fn touch(dir: &Path, name: &str) {
    std::fs::write(dir.join(name), "#!/bin/sh\nexit 0\n").unwrap();
    // Discovery requires the executable bit (Unix): a non-executable file is
    // not a usable sampler, so the fakes are runnable.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir.join(name), std::fs::Permissions::from_mode(0o755)).unwrap();
    }
}

fn join(parts: &[&Path]) -> String {
    std::env::join_paths(parts.iter())
        .unwrap()
        .into_string()
        .unwrap()
}

fn args(flags: &[&str]) -> ProfileArgs {
    xtask::profile::parse_args(flags).unwrap()
}

// Argument parsing.

#[test]
fn parses_every_flag() {
    let a = args(&[
        "SELECT 1",
        "--table",
        "t=/data/t",
        "--table",
        "u=/data/u",
        "--conf",
        "k=v",
        "--conf",
        "a=b=c",
        "--tool",
        "samply",
        "--out",
        "/tmp/o",
        "--no-flamegraph",
    ]);
    assert_eq!(a.query.as_deref(), Some("SELECT 1"));
    assert_eq!(a.file, None);
    assert_eq!(
        a.tables,
        vec![
            ("t".to_owned(), "/data/t".to_owned()),
            ("u".to_owned(), "/data/u".to_owned()),
        ]
    );
    assert_eq!(
        a.confs,
        vec![
            ("k".to_owned(), "v".to_owned()),
            ("a".to_owned(), "b=c".to_owned()),
        ]
    );
    assert_eq!(a.tool, ToolChoice::Samply);
    assert_eq!(a.out, Some(PathBuf::from("/tmp/o")));
    assert!(a.no_flamegraph);
}

#[test]
fn parses_defaults_and_file_query() {
    let a = args(&["-f", "q.sql"]);
    assert_eq!(a.query, None);
    assert_eq!(a.file, Some(PathBuf::from("q.sql")));
    assert!(a.tables.is_empty() && a.confs.is_empty());
    assert_eq!(a.tool, ToolChoice::Auto);
    assert_eq!(a.out, None);
    assert!(!a.no_flamegraph);
    assert_eq!(args(&["SELECT 1"]).tool, ToolChoice::Auto);
    assert_eq!(args(&["SELECT 1", "--tool", "perf"]).tool, ToolChoice::Perf);
}

#[test]
fn missing_query_is_an_error() {
    for flags in [
        vec![],
        vec!["--tool", "perf"],
        vec!["--table", "t=/p"],
        vec!["--no-flamegraph"],
    ] {
        let e = xtask::profile::parse_args(&flags).unwrap_err();
        assert!(e.contains("no query given"), "{flags:?}: {e}");
    }
}

#[test]
fn query_and_file_together_is_an_error() {
    let e = xtask::profile::parse_args(&["SELECT 1", "-f", "q.sql"]).unwrap_err();
    assert!(e.contains("not both"), "{e}");
    // The file may come first; the query is still rejected.
    let e = xtask::profile::parse_args(&["-f", "q.sql", "SELECT 1"]).unwrap_err();
    assert!(e.contains("not both"), "{e}");
}

#[test]
fn two_positional_queries_are_an_error() {
    let e = xtask::profile::parse_args(&["SELECT 1", "SELECT 2"]).unwrap_err();
    assert!(e.contains("single query"), "{e}");
}

#[test]
fn unknown_flags_are_an_error() {
    for f in ["--bogus", "--flamegraph", "-x"] {
        let e = xtask::profile::parse_args(&["SELECT 1", f]).unwrap_err();
        assert!(e.contains("unknown flag"), "{f}: {e}");
    }
}

#[test]
fn flags_missing_their_value_are_an_error() {
    for f in ["--table", "--conf", "--tool", "--out", "-f"] {
        let e = xtask::profile::parse_args(&["SELECT 1", f]).unwrap_err();
        assert!(e.contains("needs"), "{f}: {e}");
    }
}

#[test]
fn bad_key_value_pairs_are_an_error() {
    for (flag, bad) in [
        ("--table", "no-equals"),
        ("--table", "=v"),
        ("--conf", "no-equals"),
        ("--conf", "=v"),
    ] {
        let e = xtask::profile::parse_args(&["SELECT 1", flag, bad]).unwrap_err();
        assert!(e.contains("NAME=VALUE"), "{flag} {bad}: {e}");
    }
}

#[test]
fn bad_tool_is_an_error() {
    let e = xtask::profile::parse_args(&["SELECT 1", "--tool", "strace"]).unwrap_err();
    assert!(e.contains("--tool expects auto|perf|samply"), "{e}");
}

#[test]
fn duplicate_file_flag_is_an_error() {
    let e = xtask::profile::parse_args(&["-f", "a.sql", "-f", "b.sql"]).unwrap_err();
    assert!(e.contains("only once"), "{e}");
}

// Tool discovery.

#[test]
fn discovery_with_fake_executables() {
    let dir = tmp("discover");
    // Empty entries are skipped: nothing is found.
    assert_eq!(
        discover_on_path(""),
        xtask::profile::Discovery {
            perf: false,
            collapse: false,
            flamegraph: false,
            samply: false,
        }
    );
    touch(dir.path(), "perf");
    let p = join(&[dir.path()]);
    let d = discover_on_path(&p);
    assert!(d.perf && !d.collapse && !d.flamegraph && !d.samply);
    touch(dir.path(), "samply");
    let d = discover_on_path(&p);
    assert!(d.perf && d.samply && !d.collapse);
    touch(dir.path(), "inferno-collapse-perf");
    touch(dir.path(), "inferno-flamegraph");
    let d = discover_on_path(&p);
    assert!(d.perf && d.samply && d.collapse && d.flamegraph);
    // Empty `PATH` entries do not hide the fakes.
    let d = discover_on_path(&format!(":{p}:"));
    assert!(d.perf && d.samply && d.collapse && d.flamegraph);
}

#[test]
fn discovery_searches_every_path_entry() {
    let a = tmp("discover-a");
    let b = tmp("discover-b");
    touch(a.path(), "samply");
    touch(b.path(), "perf");
    let d = discover_on_path(&join(&[a.path(), b.path()]));
    assert!(d.perf && d.samply && !d.collapse && !d.flamegraph);
}

#[test]
fn discovery_ignores_non_files() {
    let dir = tmp("discover-dir");
    // A directory named like a tool is not an executable.
    std::fs::create_dir(dir.path().join("perf")).unwrap();
    assert!(!discover_on_path(&join(&[dir.path()])).perf);
}

#[test]
#[cfg(unix)]
fn discovery_ignores_non_executable_files() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tmp("discover-noexec");
    // A non-executable file named like a tool would only fail at spawn:
    // discovery skips it.
    let p = dir.path().join("perf");
    std::fs::write(&p, "#!/bin/sh\nexit 0\n").unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(!discover_on_path(&join(&[dir.path()])).perf);
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(discover_on_path(&join(&[dir.path()])).perf);
}

// Tool selection.

#[test]
fn auto_prefers_perf_then_samply() {
    let all = xtask::profile::Discovery {
        perf: true,
        collapse: true,
        flamegraph: true,
        samply: true,
    };
    assert_eq!(select_tool(&ToolChoice::Auto, &all), Ok(SelectedTool::Perf));
    // `perf` without inferno falls back to `samply`.
    let no_inferno = xtask::profile::Discovery {
        perf: true,
        collapse: false,
        flamegraph: false,
        samply: true,
    };
    assert_eq!(
        select_tool(&ToolChoice::Auto, &no_inferno),
        Ok(SelectedTool::Samply)
    );
    let samply_only = xtask::profile::Discovery {
        perf: false,
        collapse: false,
        flamegraph: false,
        samply: true,
    };
    assert_eq!(
        select_tool(&ToolChoice::Auto, &samply_only),
        Ok(SelectedTool::Samply)
    );
    // One inferno binary is not enough for `perf`.
    let half = xtask::profile::Discovery {
        perf: true,
        collapse: true,
        flamegraph: false,
        samply: true,
    };
    assert_eq!(
        select_tool(&ToolChoice::Auto, &half),
        Ok(SelectedTool::Samply)
    );
}

#[test]
fn auto_without_any_sampler_names_the_installs() {
    let none = xtask::profile::Discovery {
        perf: false,
        collapse: false,
        flamegraph: false,
        samply: false,
    };
    let e = select_tool(&ToolChoice::Auto, &none).unwrap_err();
    assert!(e.contains("no sampling profiler"), "{e}");
    assert!(e.contains("linux-tools"), "{e}");
    assert!(e.contains("cargo install samply"), "{e}");
    assert!(e.contains("cargo install inferno"), "{e}");
    for tool in [
        "perf",
        "inferno-collapse-perf",
        "inferno-flamegraph",
        "samply",
    ] {
        assert!(e.contains(tool), "{tool}: {e}");
    }
}

#[test]
fn auto_with_perf_but_no_inferno_and_no_samply_mentions_inferno() {
    let d = xtask::profile::Discovery {
        perf: true,
        collapse: false,
        flamegraph: false,
        samply: false,
    };
    let e = select_tool(&ToolChoice::Auto, &d).unwrap_err();
    assert!(e.contains("inferno-collapse-perf"), "{e}");
    assert!(e.contains("perf"), "{e}");
}

#[test]
fn explicit_tool_needs_its_binaries() {
    let all = xtask::profile::Discovery {
        perf: true,
        collapse: true,
        flamegraph: true,
        samply: true,
    };
    assert_eq!(select_tool(&ToolChoice::Perf, &all), Ok(SelectedTool::Perf));
    assert_eq!(
        select_tool(&ToolChoice::Samply, &all),
        Ok(SelectedTool::Samply)
    );
    let none = xtask::profile::Discovery {
        perf: false,
        collapse: false,
        flamegraph: false,
        samply: false,
    };
    let e = select_tool(&ToolChoice::Perf, &none).unwrap_err();
    assert!(e.contains("needs `perf`"), "{e}");
    assert!(e.contains("linux-tools"), "{e}");
    let e = select_tool(&ToolChoice::Samply, &none).unwrap_err();
    assert!(e.contains("needs `samply`"), "{e}");
    assert!(e.contains("cargo install samply"), "{e}");
    // Explicit `perf` without the inferno binaries names them.
    let no_inferno = xtask::profile::Discovery {
        perf: true,
        collapse: true,
        flamegraph: false,
        samply: false,
    };
    let e = select_tool(&ToolChoice::Perf, &no_inferno).unwrap_err();
    assert!(e.contains("inferno-flamegraph"), "{e}");
    assert!(e.contains("cargo install inferno"), "{e}");
}

// Failure message text.

#[test]
fn no_sampler_message_lists_missing_tools_and_installs() {
    let all = xtask::profile::Discovery {
        perf: true,
        collapse: true,
        flamegraph: true,
        samply: true,
    };
    let m = no_sampler_message(&all);
    assert!(m.contains("cargo install samply"), "{m}");
    let one = xtask::profile::Discovery {
        perf: true,
        collapse: true,
        flamegraph: true,
        samply: false,
    };
    let m = no_sampler_message(&one);
    assert!(m.contains("missing: samply"), "{m}");
    assert!(m.contains("linux-tools"), "{m}");
}

#[test]
fn permission_detection_and_note() {
    assert!(is_perf_permission_error(
        "Error: No permission to enable cycles event"
    ));
    assert!(is_perf_permission_error("perf_event_paranoid is set to 4"));
    assert!(is_perf_permission_error("Operation not permitted"));
    assert!(is_perf_permission_error("Access denied opening the events"));
    assert!(!is_perf_permission_error(""));
    assert!(!is_perf_permission_error(
        "failed to open perf.data: no such file"
    ));
    // A file permission failure is not the kernel refusing profiling: it
    // must not attract the `perf_event_paranoid` note.
    assert!(!is_perf_permission_error(
        "could not open output file: Permission denied"
    ));
    let note = perf_permission_note();
    assert!(note.contains("perf_event_paranoid"), "{note}");
    assert!(note.contains("sudo"), "{note}");
}

#[test]
fn perf_record_error_adds_the_note_only_for_kernel_refusals() {
    let e = perf_record_error("Error: No permission to enable cycles event\n");
    assert!(e.contains("`perf record` failed"), "{e}");
    assert!(e.contains("perf_event_paranoid"), "{e}");
    let e = perf_record_error("bad option: --frobnicate\n");
    assert!(e.contains("`perf record` failed: bad option"), "{e}");
    assert!(!e.contains("perf_event_paranoid"), "{e}");
    // An unrelated permission failure (writing the output file) is reported
    // without the kernel note.
    let e = perf_record_error("could not open output file: Permission denied\n");
    assert!(e.contains("`perf record` failed"), "{e}");
    assert!(!e.contains("perf_event_paranoid"), "{e}");
}

// Command-line construction.

fn os(strs: &[&str]) -> Vec<OsString> {
    strs.iter().map(OsString::from).collect()
}

#[test]
fn command_lines_are_deterministic() {
    assert_eq!(
        cargo_build_argv(),
        os(&["build", "-p", "krishiv-cli", "--release"])
    );
    assert_eq!(
        LINE_TABLES_ENV,
        ("CARGO_PROFILE_RELEASE_DEBUG", "line-tables-only")
    );
    assert_eq!(collapse_argv(), os(&["inferno-collapse-perf"]));
    assert_eq!(flamegraph_argv(), os(&["inferno-flamegraph"]));
}

#[test]
fn cli_sql_argv_for_inline_query() {
    let argv = cli_sql_argv(
        &Some("SELECT 1".to_owned()),
        &None,
        &[("t".to_owned(), "/d/t".to_owned())],
        &[("k".to_owned(), "v=v".to_owned())],
    );
    assert_eq!(
        argv,
        os(&[
            "sql",
            "--explain-analyze",
            "--json",
            "--table",
            "t=/d/t",
            "--conf",
            "k=v=v",
            "-e",
            "SELECT 1",
        ])
    );
    // The SQL is one argument: spaces and metacharacters are never split.
    let argv = cli_sql_argv(
        &Some("SELECT * FROM t; rm -rf /".to_owned()),
        &None,
        &[],
        &[],
    );
    let last = argv.last().unwrap();
    assert_eq!(last, OsStr::new("SELECT * FROM t; rm -rf /"));
}

#[test]
fn cli_sql_argv_for_file_query() {
    let argv = cli_sql_argv(
        &None,
        &Some(PathBuf::from("q.sql")),
        &[("t".to_owned(), "/d/t".to_owned())],
        &[],
    );
    assert_eq!(
        argv,
        os(&[
            "sql",
            "--explain-analyze",
            "--json",
            "--table",
            "t=/d/t",
            "-f",
            "q.sql",
        ])
    );
}

#[test]
fn sampler_argv_match_the_documented_commands() {
    let cli = os(&["sql", "--explain-analyze", "--json", "-e", "SELECT 1"]);
    let record = perf_record_argv(Path::new("/o/perf.data"), Path::new("/b/krishiv"), &cli);
    assert_eq!(
        record,
        os(&[
            "perf",
            "record",
            "-F",
            "999",
            "-g",
            "--call-graph",
            "dwarf",
            "-o",
            "/o/perf.data",
            "--",
            "/b/krishiv",
            "sql",
            "--explain-analyze",
            "--json",
            "-e",
            "SELECT 1",
        ])
    );
    assert_eq!(
        perf_script_argv(Path::new("/o/perf.data")),
        os(&["perf", "script", "-i", "/o/perf.data"])
    );
    let samply = samply_record_argv(
        Path::new("/o/profile.json.gz"),
        Path::new("/b/krishiv"),
        &cli,
    );
    assert_eq!(
        samply,
        os(&[
            "samply",
            "record",
            "--save-only",
            "-o",
            "/o/profile.json.gz",
            "--",
            "/b/krishiv",
            "sql",
            "--explain-analyze",
            "--json",
            "-e",
            "SELECT 1",
        ])
    );
    assert_eq!(PROFILE_JSON, "profile.json");
    assert_eq!(PERF_DATA, "perf.data");
    assert_eq!(FLAMEGRAPH_SVG, "flamegraph.svg");
    assert_eq!(SAMPLY_OUT, "profile.json.gz");
}

// Out directory.

#[test]
fn default_out_dir_and_clock() {
    let root = Path::new("/w");
    assert_eq!(
        default_out_dir(root, 1_728_000_000_000),
        PathBuf::from("/w/target/xtask-profile/1728000000000")
    );
    assert!(utc_now_millis() > 0);
}

#[test]
fn create_out_dir_makes_parents_and_refuses_to_overwrite() {
    let dir = tmp("outdir");
    let nested = dir.path().join("a").join("b");
    create_out_dir(&nested).unwrap();
    assert!(nested.is_dir());
    let e = create_out_dir(&nested).unwrap_err();
    assert!(e.contains("refusing to overwrite"), "{e}");
    assert!(e.contains("--out"), "{e}");
    let file = dir.path().join("f");
    std::fs::write(&file, "x").unwrap();
    let e = create_out_dir(&file).unwrap_err();
    assert!(e.contains("refusing to overwrite"), "{e}");
    // A parent that is a file cannot hold the new directory.
    let e = create_out_dir(&file.join("sub")).unwrap_err();
    assert!(e.contains("cannot create"), "{e}");
}

// Profile summary.

const PROFILE: &str = r#"{"format":"krishiv.query-profile","version":1,"query":{"wall_time_ns":3000000},"operators":[
{"name":"Scan","metrics":{"compute_time_ns":100}},
{"name":"Filter","metrics":{"compute_time_ns":300}},
{"name":"HashJoin","metrics":{"compute_time_ns":200}},
{"name":"Sort","metrics":{"compute_time_ns":50}}]}"#;

#[test]
fn summary_shows_wall_time_and_three_slowest() {
    let s = summarize_profile(PROFILE).unwrap();
    assert_eq!(
        s,
        "query wall time: 3000000 ns (3.000 ms)\nslowest operators:\n\
         - Filter: 300 ns (0.000 ms) compute\n\
         - HashJoin: 200 ns (0.000 ms) compute\n\
         - Scan: 100 ns (0.000 ms) compute"
    );
}

#[test]
fn summary_handles_few_operators() {
    let one = r#"{"format":"krishiv.query-profile","version":1,"query":{"wall_time_ns":7},"operators":[
{"name":"Only","metrics":{"compute_time_ns":7}}]}"#;
    let s = summarize_profile(one).unwrap();
    assert_eq!(
        s,
        "query wall time: 7 ns (0.000 ms)\nslowest operators:\n\
         - Only: 7 ns (0.000 ms) compute"
    );
    let empty = r#"{"format":"krishiv.query-profile","version":1,"query":{"wall_time_ns":7},"operators":[]}"#;
    let s = summarize_profile(empty).unwrap();
    assert_eq!(s, "query wall time: 7 ns (0.000 ms)\nslowest operators:");
}

#[test]
fn summary_checks_the_profile_format_and_version() {
    assert_eq!(PROFILE_FORMAT, "krishiv.query-profile");
    assert_eq!(PROFILE_VERSION, 1);
    let e = summarize_profile(
        r#"{"format":"other.format","version":1,"query":{"wall_time_ns":1},"operators":[]}"#,
    )
    .unwrap_err();
    assert!(e.contains("krishiv.query-profile"), "{e}");
    let e = summarize_profile(
        r#"{"format":"krishiv.query-profile","version":2,"query":{"wall_time_ns":1},"operators":[]}"#,
    )
    .unwrap_err();
    assert!(e.contains("version 2 is not supported"), "{e}");
    let e = summarize_profile(
        r#"{"format":"krishiv.query-profile","query":{"wall_time_ns":1},"operators":[]}"#,
    )
    .unwrap_err();
    assert!(e.contains("no version"), "{e}");
    // A document without the format member is not a query profile either.
    let e = summarize_profile(r#"{"query":{"wall_time_ns":1},"operators":[]}"#).unwrap_err();
    assert!(e.contains("krishiv.query-profile"), "{e}");
}

#[test]
fn summary_rejects_multi_statement_output() {
    // The CLI prints one JSON line per statement: two statements are JSON
    // Lines, not one JSON document.
    let line = r#"{"format":"krishiv.query-profile","version":1,"query":{"wall_time_ns":1},"operators":[]}"#;
    let two = format!("{line}\n{line}\n");
    let e = summarize_profile(&two).unwrap_err();
    assert!(e.contains("more than one JSON document"), "{e}");
    assert!(e.contains("single statement"), "{e}");
}

#[test]
fn summary_rejects_bad_documents() {
    assert!(summarize_profile("{").unwrap_err().contains("not JSON"));
    assert!(
        summarize_profile(r#"{"format":"krishiv.query-profile","version":1,"operators":[]}"#)
            .unwrap_err()
            .contains("no query")
    );
    assert!(
        summarize_profile(
            r#"{"format":"krishiv.query-profile","version":1,"query":{},"operators":[]}"#
        )
        .unwrap_err()
        .contains("wall_time_ns")
    );
    assert!(
        summarize_profile(
            r#"{"format":"krishiv.query-profile","version":1,"query":{"wall_time_ns":1}}"#
        )
        .unwrap_err()
        .contains("operators")
    );
    assert!(
        summarize_profile(
            r#"{"format":"krishiv.query-profile","version":1,"query":{"wall_time_ns":1},"operators":[{}]}"#
        )
        .unwrap_err()
        .contains("without a name")
    );
    assert!(
        summarize_profile(
            r#"{"format":"krishiv.query-profile","version":1,"query":{"wall_time_ns":1},"operators":[{"name":"X"}]}"#
        )
        .unwrap_err()
        .contains("no metrics")
    );
    assert!(
        summarize_profile(
            r#"{"format":"krishiv.query-profile","version":1,"query":{"wall_time_ns":1},"operators":[{"name":"X","metrics":{}}]}"#
        )
        .unwrap_err()
        .contains("compute_time_ns")
    );
}

// `run_profile_with_env` with stub binaries (unix only: the stubs are shell
// scripts). The environment is explicit, so no test mutates the process.

#[cfg(unix)]
fn write_exe(dir: &Path, name: &str, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

/// A fake workspace root whose `target/release/krishiv` prints `stdout_text`
/// and exits `code`. The stub records the arguments it received, one per
/// line, in `target/release/krishiv.args`.
#[cfg(unix)]
fn fake_root(dir: &Path, stdout_text: &str, code: i32) -> PathBuf {
    let bin_dir = dir.join("target").join("release");
    std::fs::create_dir_all(&bin_dir).unwrap();
    write_exe(
        &bin_dir,
        "krishiv",
        &format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$(dirname \"$0\")/krishiv.args\"\ncat <<'PROFILE_EOF'\n{stdout_text}\nPROFILE_EOF\nexit {code}\n",
        ),
    );
    dir.to_owned()
}

/// Arguments the `fake_root` stub saw, one entry per argv element.
#[cfg(unix)]
fn recorded_cli_args(dir: &Path) -> Vec<String> {
    let p = dir.join("target").join("release").join("krishiv.args");
    std::fs::read_to_string(&p)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

#[cfg(unix)]
fn fake_cargo(dir: &Path, code: i32) -> PathBuf {
    write_exe(dir, "cargo", &format!("#!/bin/sh\nexit {code}\n"))
}

#[cfg(unix)]
fn host_path() -> String {
    std::env::var("PATH").unwrap_or_default()
}

#[test]
#[cfg(unix)]
fn run_profile_reads_the_environment_for_cargo_and_path() {
    // `run_profile` only forwards the process environment (covered here
    // without running anything: parsing fails first).
    let dir = tmp("run-env");
    let e = run_profile(dir.path(), &[]).unwrap_err();
    assert!(e.contains("no query given"), "{e}");
}

#[test]
#[cfg(unix)]
fn run_profile_writes_profile_json_without_flamegraph() {
    let dir = tmp("run-ok");
    fake_root(dir.path(), PROFILE, 0);
    let cargo = fake_cargo(dir.path(), 0);
    let out = dir.path().join("out");
    run_profile_with_env(
        dir.path(),
        &[
            "SELECT 1",
            "--table",
            "t=/d/t",
            "--conf",
            "k=v",
            "--out",
            out.to_str().unwrap(),
            "--no-flamegraph",
        ],
        cargo.as_os_str(),
        &host_path(),
        None,
    )
    .unwrap();
    let written = std::fs::read_to_string(out.join("profile.json")).unwrap();
    assert_eq!(written, format!("{PROFILE}\n"));
    assert_eq!(
        recorded_cli_args(dir.path()),
        [
            "sql",
            "--explain-analyze",
            "--json",
            "--table",
            "t=/d/t",
            "--conf",
            "k=v",
            "-e",
            "SELECT 1",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect::<Vec<_>>()
    );
}

#[test]
#[cfg(unix)]
fn run_profile_reads_the_query_file_and_defaults_the_out_dir() {
    let dir = tmp("run-file");
    fake_root(dir.path(), PROFILE, 0);
    let cargo = fake_cargo(dir.path(), 0);
    std::fs::write(dir.path().join("q.sql"), "SELECT 2").unwrap();
    run_profile_with_env(
        dir.path(),
        &["-f", "q.sql", "--no-flamegraph"],
        cargo.as_os_str(),
        &host_path(),
        None,
    )
    .unwrap();
    // The default out directory holds the profile.
    let base = dir.path().join("target").join("xtask-profile");
    let mut entries: Vec<_> = std::fs::read_dir(&base).unwrap().collect();
    assert_eq!(entries.len(), 1, "{:?}", base);
    let entry = entries.pop().unwrap().unwrap();
    let written = std::fs::read_to_string(entry.path().join("profile.json")).unwrap();
    assert_eq!(written, format!("{PROFILE}\n"));
    assert_eq!(
        recorded_cli_args(dir.path()),
        ["sql", "--explain-analyze", "--json", "-f", "q.sql",]
            .iter()
            .map(|s| (*s).to_owned())
            .collect::<Vec<_>>()
    );
}

#[test]
#[cfg(unix)]
fn run_profile_reports_the_json_before_the_missing_sampler_error() {
    let dir = tmp("run-nosampler");
    fake_root(dir.path(), PROFILE, 0);
    let cargo = fake_cargo(dir.path(), 0);
    let empty = tmp("empty-path");
    let out = dir.path().join("out");
    let e = run_profile_with_env(
        dir.path(),
        &["SELECT 1", "--out", out.to_str().unwrap()],
        cargo.as_os_str(),
        empty.path().to_str().unwrap(),
        None,
    )
    .unwrap_err();
    assert!(e.contains("no sampling profiler"), "{e}");
    assert!(e.contains("cargo install samply"), "{e}");
    // The profile step still ran and is on disk.
    assert!(out.join("profile.json").is_file());
}

#[test]
#[cfg(unix)]
fn run_profile_fails_on_build_binary_and_cli_errors() {
    let dir = tmp("run-fail");
    let path = host_path();
    // The build fails: nothing else runs.
    let failing = fake_cargo(dir.path(), 1);
    let e = run_profile_with_env(
        dir.path(),
        &[
            "SELECT 1",
            "--out",
            dir.path().join("o1").to_str().unwrap(),
            "--no-flamegraph",
        ],
        failing.as_os_str(),
        &path,
        None,
    )
    .unwrap_err();
    assert!(e.contains("cargo build"), "{e}");

    // The build succeeds but the binary is missing.
    let passing = write_exe(dir.path(), "cargo-ok", "#!/bin/sh\nexit 0\n");
    let e = run_profile_with_env(
        dir.path(),
        &[
            "SELECT 1",
            "--out",
            dir.path().join("o2").to_str().unwrap(),
            "--no-flamegraph",
        ],
        passing.as_os_str(),
        &path,
        None,
    )
    .unwrap_err();
    assert!(e.contains("built no binary"), "{e}");

    // The CLI fails: its stderr is in the error.
    let bin_dir = dir.path().join("target").join("release");
    std::fs::create_dir_all(&bin_dir).unwrap();
    write_exe(&bin_dir, "krishiv", "#!/bin/sh\necho boom >&2\nexit 1\n");
    let e = run_profile_with_env(
        dir.path(),
        &[
            "SELECT 1",
            "--out",
            dir.path().join("o3").to_str().unwrap(),
            "--no-flamegraph",
        ],
        passing.as_os_str(),
        &path,
        None,
    )
    .unwrap_err();
    assert!(e.contains("boom"), "{e}");
}

#[test]
#[cfg(unix)]
fn run_profile_refuses_to_overwrite_and_needs_a_query_file() {
    let dir = tmp("run-out");
    fake_root(dir.path(), PROFILE, 0);
    let cargo = fake_cargo(dir.path(), 0);
    let out = dir.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let path = host_path();
    let e = run_profile_with_env(
        dir.path(),
        &[
            "SELECT 1",
            "--out",
            out.to_str().unwrap(),
            "--no-flamegraph",
        ],
        cargo.as_os_str(),
        &path,
        None,
    )
    .unwrap_err();
    assert!(e.contains("refusing to overwrite"), "{e}");
    let e = run_profile_with_env(
        dir.path(),
        &["-f", "missing.sql", "--no-flamegraph"],
        cargo.as_os_str(),
        &path,
        None,
    )
    .unwrap_err();
    assert!(e.contains("cannot read"), "{e}");
    let e = run_profile_with_env(
        dir.path(),
        &["--no-flamegraph"],
        cargo.as_os_str(),
        &path,
        None,
    )
    .unwrap_err();
    assert!(e.contains("no query given"), "{e}");
}

#[test]
#[cfg(unix)]
fn run_profile_names_an_unrunnable_cargo() {
    let dir = tmp("run-cargo");
    let e = run_profile_with_env(
        dir.path(),
        &[
            "SELECT 1",
            "--out",
            dir.path().join("o").to_str().unwrap(),
            "--no-flamegraph",
        ],
        OsStr::new("/nonexistent/xtask-profile-cargo"),
        &host_path(),
        None,
    )
    .unwrap_err();
    assert!(e.contains("cannot run cargo build"), "{e}");
}

#[test]
fn cli_bin_path_defaults_to_the_workspace_target_dir() {
    let root = Path::new("/w");
    assert_eq!(
        cli_bin_path(root, None),
        PathBuf::from(format!(
            "/w/target/release/krishiv{}",
            std::env::consts::EXE_SUFFIX
        ))
    );
    assert_eq!(
        cli_bin_path(root, Some(OsStr::new("/t"))),
        PathBuf::from(format!(
            "/t/release/krishiv{}",
            std::env::consts::EXE_SUFFIX
        ))
    );
}

#[test]
#[cfg(unix)]
fn run_profile_finds_the_binary_under_cargo_target_dir() {
    let dir = tmp("run-targetdir");
    // The build drops the binary in `CARGO_TARGET_DIR`, not `target/`: the
    // fake cargo pretends to, and the lookup must follow.
    let custom = dir.path().join("custom-target");
    let bin_dir = custom.join("release");
    std::fs::create_dir_all(&bin_dir).unwrap();
    write_exe(
        &bin_dir,
        "krishiv",
        &format!("#!/bin/sh\ncat <<'PROFILE_EOF'\n{PROFILE}\nPROFILE_EOF\nexit 0\n"),
    );
    let cargo = fake_cargo(dir.path(), 0);
    let out = dir.path().join("out");
    run_profile_with_env(
        dir.path(),
        &[
            "SELECT 1",
            "--out",
            out.to_str().unwrap(),
            "--no-flamegraph",
        ],
        cargo.as_os_str(),
        &host_path(),
        Some(custom.as_os_str()),
    )
    .unwrap();
    let written = std::fs::read_to_string(out.join("profile.json")).unwrap();
    assert!(written.contains("wall_time_ns"), "{written}");
}

#[test]
fn profile_subcommand_is_wired_into_the_binary() {
    // No query: the `profile` arm runs and reports the usage error.
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .arg("profile")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("no query given"),
        "{out:?}"
    );
    // An unknown flag is a usage error, not the top-level usage text.
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["profile", "SELECT 1", "--bogus"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("unknown flag"),
        "{out:?}"
    );
}
