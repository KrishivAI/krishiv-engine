// SPDX-License-Identifier: Apache-2.0
//! The line protocol over the checked-in TPC-H `lineitem` sample.
#![cfg(test)]

use std::path::PathBuf;

use krishiv_bench_runner::{Runner, RunnerConfig, decode_path, encode_path, serve};

fn data_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tpch/q1-lineitem.parquet");
    std::fs::copy(fixture, dir.path().join("lineitem.parquet")).unwrap();
    std::fs::write(dir.path().join("notes.txt"), "not a table").unwrap();
    dir
}

fn runner(dir: &tempfile::TempDir) -> Runner {
    Runner::new(&RunnerConfig {
        data: dir.path().to_str().unwrap().to_owned(),
        threads: 2,
        io_threads: None,
        memory_limit: "1g".to_owned(),
        tables: Vec::new(),
        conf: Vec::new(),
    })
    .unwrap()
}

#[test]
fn serves_run_dump_and_errors() {
    let dir = data_dir();
    let sql = dir.path().join("q.sql");
    std::fs::write(
        &sql,
        "SELECT l_returnflag, count(*) AS n FROM lineitem GROUP BY l_returnflag",
    )
    .unwrap();
    let bad = dir.path().join("bad.sql");
    std::fs::write(&bad, "SELECT nope FROM lineitem").unwrap();
    let out = dir.path().join("out");
    let script = format!(
        "RUN {sql}\n\nDUMP {sql} {out}\nRUN {bad}\nRUN /does/not/exist.sql\nBOGUS\nQUIT\nRUN {sql}\n",
        sql = encode_path(sql.to_str().unwrap()),
        bad = encode_path(bad.to_str().unwrap()),
        out = encode_path(out.to_str().unwrap()),
    );
    let mut reply = Vec::new();
    serve(&runner(&dir), script.as_bytes(), &mut reply).unwrap();
    let reply = String::from_utf8(reply).unwrap();
    let lines: Vec<&str> = reply.lines().collect();
    assert_eq!(
        lines[0],
        format!("READY {} lineitem", env!("CARGO_PKG_VERSION"))
    );
    assert!(lines[1].starts_with("OK 3 "), "{lines:?}");
    assert_eq!(lines[2], "");
    assert!(lines[3].starts_with("OK 3 "), "{lines:?}");
    assert!(lines[4].starts_with("ERR "), "{lines:?}");
    assert!(lines[5].starts_with("ERR cannot read"), "{lines:?}");
    assert_eq!(lines[6], "ERR bad command: BOGUS");
    assert_eq!(lines.len(), 7, "QUIT stops serving: {lines:?}");
    assert!(out.exists());
}

#[test]
fn run_sql_reports_rows() {
    let dir = data_dir();
    let (rows, secs) = runner(&dir)
        .run_sql("SELECT * FROM lineitem LIMIT 5")
        .unwrap();
    assert_eq!(rows, 5);
    assert!(secs >= 0.0);
}

#[test]
fn a_directory_of_parts_is_one_table_and_an_empty_directory_is_refused() {
    let dir = data_dir();
    let parts = dir.path().join("orders");
    std::fs::create_dir(&parts).unwrap();
    std::fs::copy(
        dir.path().join("lineitem.parquet"),
        parts.join("part-0.parquet"),
    )
    .unwrap();
    assert_eq!(runner(&dir).tables(), ["lineitem", "orders"]);
    let empty = tempfile::tempdir().unwrap();
    let err = Runner::new(&RunnerConfig {
        data: empty.path().to_str().unwrap().to_owned(),
        threads: 1,
        io_threads: None,
        memory_limit: "1g".to_owned(),
        tables: Vec::new(),
        conf: Vec::new(),
    })
    .unwrap_err();
    assert!(err.to_string().contains("no <table>.parquet"), "{err}");
    let missing = Runner::new(&RunnerConfig {
        data: "/does/not/exist".to_owned(),
        threads: 1,
        io_threads: None,
        memory_limit: "1g".to_owned(),
        tables: Vec::new(),
        conf: Vec::new(),
    });
    assert!(missing.unwrap_err().to_string().contains("cannot read"));
}

/// `--conf` settings reach the session.
#[test]
fn conf_settings_reach_the_session() {
    let dir = data_dir();
    let key = "krishiv.exec.join.rowLayout";
    let r = |conf: Vec<(String, String)>| {
        Runner::new(&RunnerConfig {
            data: dir.path().to_str().unwrap().to_owned(),
            threads: 1,
            io_threads: None,
            memory_limit: "1g".to_owned(),
            tables: Vec::new(),
            conf,
        })
        .unwrap()
    };
    assert_eq!(r(Vec::new()).conf(key).as_deref(), Some("true"));
    let off = r(vec![(key.to_owned(), "false".to_owned())]);
    assert_eq!(off.conf(key).as_deref(), Some("false"));
    // The built-in setting is kept next to `--conf` (normalized).
    assert_eq!(off.conf("krishiv.memory.limit").as_deref(), Some("1GB"));
}

#[test]
fn arguments() {
    let args = |s: &str| RunnerConfig::from_args(s.split_whitespace().map(str::to_owned));
    let c = args("--data /d --threads 4 --memory-limit 2g").unwrap();
    assert_eq!(
        (c.data.as_str(), c.threads, c.memory_limit.as_str()),
        ("/d", 4, "2g")
    );
    assert!(args("--data /d").unwrap().threads >= 1);
    assert_eq!(args("--data /d").unwrap().io_threads, None);
    let sizes = |s: &str| args(s).unwrap().executor_config().unwrap();
    // The I/O runtime's blocking threads come from `krishiv.io.blockingThreads`, validated.
    assert_eq!(sizes("--data /d").io_blocking_threads, 8);
    assert_eq!(
        sizes("--data /d --conf krishiv.io.blockingThreads=3").io_blocking_threads,
        3
    );
    for bad in ["0", "-1", "x"] {
        let c = args(&format!(
            "--data /d --conf krishiv.io.blockingThreads={bad}"
        ))
        .unwrap();
        let e = c.executor_config().unwrap_err().to_string();
        assert!(e.contains("krishiv.io.blockingThreads"), "{bad}: {e}");
    }
    assert_eq!(
        sizes("--data /d --threads 3").io_threads,
        krishiv_exec::ExecutorConfig::default().io_threads
    );
    let six = sizes("--data /d --threads 3 --io-threads 6");
    assert_eq!(
        (six.compute_threads, six.io_threads, six.blocking_threads),
        (3, 6, 3)
    );
    assert_eq!(
        args("--data /d --io-threads 6").unwrap().io_threads,
        Some(6)
    );
    assert_eq!(
        args("--data /d --io-threads 0").unwrap().io_threads,
        Some(1)
    );
    // The parser's own error (an unknown flag would say so instead).
    let e = args("--data /d --io-threads x").unwrap_err().to_string();
    assert!(
        e.contains("--io-threads: invalid digit found in string") && !e.contains("unknown"),
        "{e}"
    );
    let c = args("--data /d --conf a.b=1 --conf c=x=y").unwrap();
    assert_eq!(
        c.conf,
        vec![
            ("a.b".to_owned(), "1".to_owned()),
            ("c".to_owned(), "x=y".to_owned())
        ]
    );
    assert!(args("--data /d --conf novalue").is_err());
    assert!(args("--data /d --conf =v").is_err());
    // A blank key is no key.
    let blank = RunnerConfig::from_args(["--data", "/d", "--conf", " =v"].map(str::to_owned));
    assert!(blank.unwrap_err().to_string().contains("key=value"));
    assert!(
        args("--threads 4")
            .unwrap_err()
            .to_string()
            .contains("--data")
    );
    assert!(
        args("--data")
            .unwrap_err()
            .to_string()
            .contains("needs a value")
    );
    assert!(
        args("--data /d --threads x")
            .unwrap_err()
            .to_string()
            .contains("--threads")
    );
    assert!(
        args("--wat")
            .unwrap_err()
            .to_string()
            .contains("unknown argument")
    );
}

#[test]
fn path_codec_round_trips_and_rejects_garbage() {
    for p in [
        "/plain/a.sql",
        "/with space/a b.sql",
        "/pct%/\u{e9}t\u{e9}/x\ty\n.sql",
    ] {
        let e = encode_path(p);
        assert!(!e.contains(char::is_whitespace), "{e}");
        assert_eq!(decode_path(&e).unwrap(), p);
    }
    assert_eq!(encode_path("a b"), "a%20b");
    for bad in ["%", "%4", "%zz", "%FF"] {
        assert!(
            decode_path(bad)
                .unwrap_err()
                .to_string()
                .contains("bad path encoding"),
            "{bad}"
        );
    }
}

#[test]
fn dump_runs_the_query_once_and_leaves_no_view() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tpch/q1-lineitem.parquet");
    let spaced = dir.path().join("with space");
    std::fs::create_dir(&spaced).unwrap();
    std::fs::copy(fixture, spaced.join("lineitem.parquet")).unwrap();
    let r = Runner::new(&RunnerConfig {
        data: spaced.to_str().unwrap().to_owned(),
        threads: 1,
        io_threads: None,
        memory_limit: "1g".to_owned(),
        tables: Vec::new(),
        conf: Vec::new(),
    })
    .unwrap();
    let sql = spaced.join("q file.sql");
    std::fs::write(&sql, "SELECT * FROM lineitem WHERE false").unwrap();
    let out = spaced.join("out dir");
    let line = format!(
        "DUMP {} {}",
        encode_path(sql.to_str().unwrap()),
        encode_path(out.to_str().unwrap())
    );
    // An empty result still writes (the schema comes from the plan, not the batches).
    let reply = r.handle_line(&line).unwrap();
    assert!(reply.starts_with("OK 0 "), "{reply}");
    assert!(out.exists());
    // The dump view is gone: dumping again works and a table of that name does not resolve.
    assert!(r.handle_line(&line).unwrap().starts_with("OK 0 "));
    let probe = spaced.join("probe.sql");
    std::fs::write(&probe, "SELECT * FROM __krishiv_bench_dump").unwrap();
    let reply = r
        .handle_line(&format!("RUN {}", encode_path(probe.to_str().unwrap())))
        .unwrap();
    assert!(reply.starts_with("ERR "), "{reply}");
    let bad = r.handle_line("DUMP %zz /x").unwrap();
    assert!(bad.starts_with("ERR bad path encoding"), "{bad}");
}

#[test]
fn the_binary_serves_stdin_and_reports_startup_errors() {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let bin = env!("CARGO_BIN_EXE_krishiv-bench-runner");
    let dir = data_dir();
    let mut child = Command::new(bin)
        .args(["--data", dir.path().to_str().unwrap(), "--threads", "1"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"BOGUS\nQUIT\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.starts_with("READY "), "{text}");
    assert!(text.contains("ERR bad command: BOGUS"), "{text}");

    let failed = Command::new(bin).output().unwrap();
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("--data DIR is required"));
}

fn reply_for(r: &Runner, cmd: &str, sql: &std::path::Path) -> String {
    r.handle_line(&format!("{cmd} {}", encode_path(sql.to_str().unwrap())))
        .unwrap()
}

#[test]
fn profile_reports_the_phases_and_matches_run() {
    let dir = data_dir();
    let sql = dir.path().join("q.sql");
    std::fs::write(
        &sql,
        "SELECT l_returnflag, count(*) FROM lineitem GROUP BY 1",
    )
    .unwrap();
    let r = runner(&dir);
    let reply = reply_for(&r, "PROFILE", &sql);
    let mut parts = reply.splitn(4, ' ');
    assert_eq!(parts.next(), Some("OK"), "{reply}");
    assert_eq!(parts.next(), Some("3"), "{reply}");
    let total: f64 = parts.next().unwrap().parse().unwrap();
    let json = parts.next().unwrap();
    assert!(!json.contains('\n'));
    let phases = ["parse_analyze", "plan_execute"].map(|name| {
        let key = format!("\"name\":\"{name}\",\"seconds\":");
        let at = json
            .find(&key)
            .unwrap_or_else(|| panic!("{name} in {json}"))
            + key.len();
        let num: String = json[at..]
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        num.parse::<f64>().unwrap()
    });
    assert!(phases.iter().all(|p| *p >= 0.0), "{json}");
    assert!(
        (phases[0] + phases[1] - total).abs() < 1e-4,
        "phases {phases:?} must add up to {total}"
    );
    // Memory: the limit, what the pool granted, spills and the resident set.
    let field = |key: &str| -> Option<u64> {
        let mem = &json[json.find("\"memory\":{").unwrap()..];
        let at = mem.find(&format!("\"{key}\":")).unwrap() + key.len() + 3;
        let v: String = mem[at..].chars().take_while(char::is_ascii_digit).collect();
        v.parse().ok()
    };
    assert_eq!(field("limit_bytes"), Some(1 << 30), "{json}");
    assert!(
        field("peak_reserved_bytes").is_some_and(|b| b > 0),
        "{json}"
    );
    assert_eq!(field("spill_files"), Some(0), "{json}");
    assert_eq!(field("spill_bytes"), Some(0), "{json}");
    if cfg!(target_os = "linux") {
        let (before, peak) = (field("rss_before_bytes"), field("peak_rss_bytes"));
        assert!(before.is_some_and(|b| b > 0), "{json}");
        assert!(peak >= before, "{json}");
    }
    for key in [
        "\"peak_reserved_bytes\":",
        "\"spill_requests\":0",
        "\"spill_files\":0",
        "\"spill_bytes\":0",
        "\"blocked_memory_seconds\":",
        "\"drivers\":",
        "\"batches_in\":",
        "\"batches_out\":",
        "\"morsels\":",
        "\"max_driver_compute_seconds\":",
        "\"blocked_sink_seconds\":",
        "\"extra\":{",
        "\"pipelines\":[{\"id\":",
        "\"started_seconds\":",
        "\"detailed\":false",
        "\"scan_rows_decoded\":",
        "\"hash_resizes\":",
    ] {
        assert!(json.contains(key), "{key} in {json}");
    }
    assert!(!json.contains("\"probe_lookups\""), "{json}");
    if cfg!(feature = "heap-stats") {
        // A number (the test binary does not install the counting allocator,
        // so it may be 0 here).
        assert!(field("allocations").is_some(), "{json}");
    } else {
        assert!(json.contains("\"allocations\":null"), "{json}");
    }
    assert!(reply_for(&r, "RUN", &sql).starts_with("OK 3 "));
    assert!(
        r.handle_line("PROFILE")
            .unwrap()
            .starts_with("ERR bad command")
    );
}

/// `PROFILE_DETAILED` adds the probe-length counters of a detailed profile and
/// returns the same rows as `PROFILE`.
#[test]
fn profile_detailed_counts_probe_lengths() {
    let dir = data_dir();
    let sql = dir.path().join("q.sql");
    std::fs::write(
        &sql,
        "SELECT a.l_returnflag, count(*) FROM lineitem a \
         JOIN (SELECT DISTINCT l_returnflag FROM lineitem) b ON a.l_returnflag = b.l_returnflag GROUP BY 1",
    )
    .unwrap();
    let r = runner(&dir);
    let plain = reply_for(&r, "PROFILE", &sql);
    let detailed = reply_for(&r, "PROFILE_DETAILED", &sql);
    let rows = |reply: &str| reply.split(' ').nth(1).map(str::to_owned);
    assert!(detailed.starts_with("OK "), "{detailed}");
    assert_eq!(rows(&plain), rows(&detailed));
    assert!(detailed.contains("\"detailed\":true"), "{detailed}");
    for key in [
        "\"probe_lookups\":",
        "\"probe_key_compares\":",
        "\"max_probe_length\":",
    ] {
        assert!(detailed.contains(key), "{key} in {detailed}");
        assert!(!plain.contains(key), "{key} in {plain}");
    }
}

#[test]
fn plan_replies_the_profile_document() {
    let dir = data_dir();
    let sql = dir.path().join("q.sql");
    std::fs::write(
        &sql,
        "SELECT l_returnflag, count(*) FROM lineitem WHERE l_quantity > 0 GROUP BY 1",
    )
    .unwrap();
    let r = runner(&dir);
    let reply = reply_for(&r, "PLAN", &sql);
    let mut parts = reply.splitn(4, ' ');
    assert_eq!(parts.next(), Some("OK"), "{reply}");
    assert_eq!(parts.next(), Some("3"), "{reply}");
    assert!(parts.next().unwrap().parse::<f64>().unwrap() >= 0.0);
    let json = parts.next().unwrap();
    assert!(!json.contains('\n'));
    // The document loads back: the plan tree, its operators and their row counts.
    let profile = krishiv_session::QueryProfile::from_json(json).unwrap();
    // The plan holds the aggregate and the lineitem scan.
    let mut names = Vec::new();
    let mut stack = vec![&profile.plan];
    while let Some(n) = stack.pop() {
        names.push((n.name.clone(), n.detail.clone()));
        stack.extend(n.children.iter());
    }
    assert!(names.iter().any(|(n, _)| n == "HashAggregate"), "{names:?}");
    assert!(
        names
            .iter()
            .any(|(n, d)| n == "Scan" && d.contains("lineitem")),
        "{names:?}"
    );
    // The aggregate's output side reports the three groups (`l_returnflag` A, N, R).
    let agg_out: u64 = profile
        .metrics
        .operators
        .values()
        .filter(|o| o.name == "HashAggregate" && o.role == krishiv_exec::OperatorRole::Source)
        .map(|o| o.metrics.rows_out)
        .sum();
    assert_eq!(agg_out, 3, "{json}");
    assert!(
        r.handle_line("PLAN")
            .unwrap()
            .starts_with("ERR bad command"),
        "PLAN needs a file"
    );
}

#[test]
fn each_query_file_is_attributed_its_own_rows() {
    // Two different queries share neither a result nor a reply: the row count comes from the query that ran,
    // whatever the order or repetition.
    let dir = data_dir();
    let one = dir.path().join("one.sql");
    std::fs::write(&one, "SELECT 1 AS one").unwrap();
    let seven = dir.path().join("seven.sql");
    std::fs::write(&seven, "SELECT l_quantity FROM lineitem LIMIT 7").unwrap();
    let r = runner(&dir);
    let rows = |sql: &std::path::Path| -> u64 {
        let reply = reply_for(&r, "RUN", sql);
        let p: Vec<&str> = reply.split(' ').collect();
        assert_eq!(p[0], "OK", "{reply}");
        assert!(p[2].parse::<f64>().unwrap() >= 0.0, "{reply}");
        p[1].parse().unwrap()
    };
    let order = [&one, &seven, &seven, &one];
    assert_eq!(order.map(|q| rows(q)), [1, 7, 7, 1]);
}

#[test]
fn conf_is_applied_after_the_memory_limit_and_unknown_keys_are_rejected() {
    let dir = data_dir();
    let new = |conf: Vec<(&str, &str)>| {
        Runner::new(&RunnerConfig {
            data: dir.path().to_str().unwrap().to_owned(),
            threads: 1,
            io_threads: None,
            memory_limit: "1g".to_owned(),
            tables: Vec::new(),
            conf: conf
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v.to_owned()))
                .collect(),
        })
    };
    // The valid --memory-limit is overridden by a later, invalid conf value: proves conf reaches the session.
    let bad = new(vec![("krishiv.memory.limit", "not-a-size")]).unwrap_err();
    assert!(bad.to_string().contains("not-a-size"), "{bad}");
    // An unknown key would otherwise be dropped with only a warning.
    let unknown = new(vec![("no.such.setting", "1")]).unwrap_err();
    assert!(unknown.to_string().contains("no.such.setting"), "{unknown}");
    assert!(unknown.to_string().contains("not applied"), "{unknown}");
    assert!(new(vec![("krishiv.memory.limit", "2g")]).is_ok());
}

#[test]
fn table_arguments_register_views_over_urls() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tpch/q1-lineitem.parquet");
    let args = |extra: &[&str]| {
        RunnerConfig::from_args(
            ["--table", &format!("lineitem=file://{}", fixture.display())]
                .into_iter()
                .chain(extra.iter().copied())
                .map(str::to_owned),
        )
    };
    // No `--data` needed; the table reads through its URL.
    let config = args(&[
        "--threads",
        "2",
        "--conf",
        "fs.s3a.endpoint=http://127.0.0.1:9",
    ])
    .unwrap();
    assert!(config.data.is_empty());
    let r = Runner::new(&config).unwrap();
    assert_eq!(r.tables(), ["lineitem"]);
    let reply = r.handle_line(&format!(
        "RUN {}",
        encode_path(&{
            let sql = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(sql.path(), "SELECT count(*) FROM lineitem").unwrap();
            sql.into_temp_path()
                .keep()
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned()
        })
    ));
    assert!(reply.unwrap().starts_with("OK 1 "), "one count row");
    // A bad spec, and neither source, are refused.
    assert!(RunnerConfig::from_args(["--table", "x"].map(str::to_owned)).is_err());
    assert!(RunnerConfig::from_args(["--threads", "2"].map(str::to_owned)).is_err());
}

#[test]
fn remote_table_urls_get_their_own_errors_not_a_credentials_error() {
    for url in [
        "hf://datasets/cornell-movie-review-data/rotten_tomatoes/validation.parquet",
        "http://127.0.0.1:1/datasets/a/b/resolve/main/c.parquet",
        "s3://fw/p.parquet",
    ] {
        // Loopback endpoints and no retries keep every case off the network and fast.
        let config = RunnerConfig::from_args(
            [
                "--conf",
                "fs.hf.endpoint=http://127.0.0.1:1",
                "--conf",
                "fs.s3a.endpoint=http://127.0.0.1:1",
                "--conf",
                "fs.s3a.access.key=k",
                "--conf",
                "fs.s3a.secret.key=s",
                "--conf",
                "fs.s3a.path.style.access=true",
                "--conf",
                "krishiv.io.maxRetries=0",
                "--table",
                &format!("fw={url}"),
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let err = Runner::new(&config)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        // The path classifier's own message, not a store's "lacked valid authentication credentials".
        assert!(!err.contains("paths with credentials"), "{url}: {err}");
    }
}

/// `--table` repeated with one name lists several files of one table.
#[test]
fn a_repeated_table_flag_collects_its_urls() {
    let config = RunnerConfig::from_args(
        [
            "--table",
            "fw=https://example.com/a.parquet",
            "--table",
            "fw=https://example.com/b.parquet",
            "--table",
            "other=https://example.com/c.parquet",
        ]
        .map(str::to_owned),
    )
    .unwrap();
    assert_eq!(config.tables.len(), 2);
    assert_eq!(config.tables[0].0, "fw");
    assert_eq!(config.tables[0].1.len(), 2);
}

/// A `.parquet` file under `--data` that is not Parquet fails the runner's start with the read error,
/// not a missing-table message.
#[test]
fn a_corrupt_table_file_fails_the_start() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("broken.parquet"), "not parquet").unwrap();
    let err = Runner::new(&RunnerConfig {
        data: dir.path().to_str().unwrap().to_owned(),
        threads: 1,
        io_threads: None,
        memory_limit: "1g".to_owned(),
        tables: Vec::new(),
        conf: Vec::new(),
    })
    .err()
    .map(|e| e.to_string())
    .unwrap_or_default();
    assert!(!err.is_empty(), "a corrupt table must not start the runner");
    assert!(!err.contains("no <table>.parquet"), "{err}");
}

/// A writer that fails on the first write, as a closed pipe does.
struct BrokenPipe;

impl std::io::Write for BrokenPipe {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Err(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "closed",
        ))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The READY line is the first write: a harness that went away ends `serve` with the I/O error.
#[test]
fn serve_reports_a_closed_output() {
    let dir = data_dir();
    let r = runner(&dir);
    let err = serve(&r, std::io::Cursor::new(b"QUIT\n"), &mut BrokenPipe).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::BrokenPipe);
}

/// `--build-info` exits non-zero when stdout cannot be written (`/dev/full` fails every write).
#[cfg(target_os = "linux")]
#[test]
fn build_info_fails_when_stdout_is_full() {
    use std::process::{Command, Stdio};

    let full = std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/full")
        .unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_krishiv-bench-runner"))
        .arg("--build-info")
        .stdout(Stdio::from(full))
        .status()
        .unwrap();
    assert!(!status.success());
}

/// Every flag that takes a value refuses to end the command line, and a bad number is a config error.
#[test]
fn flags_without_values_and_bad_numbers_are_config_errors() {
    for flag in [
        "--data",
        "--table",
        "--threads",
        "--memory-limit",
        "--io-threads",
        "--conf",
    ] {
        let err = RunnerConfig::from_args([flag.to_owned()]).unwrap_err();
        assert!(err.to_string().contains(flag), "{flag}: {err}");
    }
    for args in [
        ["--data", "d", "--threads", "x"],
        ["--data", "d", "--io-threads", "x"],
        ["--data", "d", "--conf", "no-equals"],
        ["--data", "d", "--bogus", "1"],
    ] {
        assert!(
            RunnerConfig::from_args(args.map(str::to_owned)).is_err(),
            "{args:?}"
        );
    }
}

/// A memory limit that does not parse stops the start before any table is read.
#[test]
fn an_invalid_memory_limit_fails_the_start() {
    let dir = data_dir();
    let config = |limit: &str| RunnerConfig {
        data: dir.path().to_str().unwrap().to_owned(),
        threads: 1,
        io_threads: None,
        memory_limit: limit.to_owned(),
        tables: Vec::new(),
        conf: Vec::new(),
    };
    assert!(Runner::new(&config("lots")).is_err());
}

/// A writer that accepts bytes and fails to flush, or fails once `ok_writes` writes have gone through.
struct Flaky {
    ok_writes: usize,
    flush_fails: bool,
}

impl std::io::Write for Flaky {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.ok_writes == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "closed",
            ));
        }
        self.ok_writes -= 1;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.flush_fails {
            Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "closed",
            ))
        } else {
            Ok(())
        }
    }
}

/// A reader whose first read fails.
struct BadInput;

impl std::io::Read for BadInput {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
        Err(std::io::Error::other("read failed"))
    }
}

/// `serve` ends with the I/O error of the flush after READY, of an unreadable input, and of a reply write.
#[test]
fn serve_reports_flush_input_and_reply_failures() {
    let dir = data_dir();
    let r = runner(&dir);
    // READY is written (one write per `writeln!` piece is at most a few), but the flush fails.
    let mut flush = Flaky {
        ok_writes: usize::MAX,
        flush_fails: true,
    };
    assert!(serve(&r, std::io::Cursor::new(b"QUIT\n"), &mut flush).is_err());
    // READY and its flush succeed; reading the next line fails.
    let mut ok = Flaky {
        ok_writes: usize::MAX,
        flush_fails: false,
    };
    let err = serve(&r, std::io::BufReader::new(BadInput), &mut ok).unwrap_err();
    assert_eq!(err.to_string(), "read failed");
    // READY goes through; the reply to a command cannot be written.
    let mut few = Flaky {
        ok_writes: 4,
        flush_fails: false,
    };
    assert!(serve(&r, std::io::Cursor::new(b"BOGUS\nQUIT\n"), &mut few).is_err());
}

/// The binary reports a startup failure of the runner (a data directory that is not there) on stderr
/// and exits non-zero.
#[test]
fn the_binary_reports_a_runner_that_cannot_start() {
    use std::process::Command;

    let out = Command::new(env!("CARGO_BIN_EXE_krishiv-bench-runner"))
        .args(["--data", "/nonexistent/krishiv-bench-data"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot read"));
}

/// The binary reports a closed stdout (`/dev/full`) when it serves, and exits non-zero.
#[cfg(target_os = "linux")]
#[test]
fn the_binary_reports_a_stdout_that_cannot_be_written() {
    use std::process::{Command, Stdio};

    let dir = data_dir();
    let full = std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/full")
        .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_krishiv-bench-runner"))
        .args(["--data", dir.path().to_str().unwrap(), "--threads", "1"])
        .stdin(Stdio::null())
        .stdout(Stdio::from(full))
        .output()
        .unwrap();
    assert!(!out.status.success());
}
