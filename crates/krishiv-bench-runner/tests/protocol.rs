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
        conf: Vec::new(),
    })
    .unwrap_err();
    assert!(err.to_string().contains("no <table>.parquet"), "{err}");
    let missing = Runner::new(&RunnerConfig {
        data: "/does/not/exist".to_owned(),
        threads: 1,
        io_threads: None,
        memory_limit: "1g".to_owned(),
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
            conf,
        })
        .unwrap()
    };
    assert_eq!(r(Vec::new()).conf(key).as_deref(), Some("false"));
    let on = r(vec![(key.to_owned(), "true".to_owned())]);
    assert_eq!(on.conf(key).as_deref(), Some("true"));
    // The built-in setting is kept next to `--conf` (normalized).
    assert_eq!(on.conf("krishiv.memory.limit").as_deref(), Some("1GB"));
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
        "\"extra\":{",
    ] {
        assert!(json.contains(key), "{key} in {json}");
    }
    assert!(reply_for(&r, "RUN", &sql).starts_with("OK 3 "));
    assert!(
        r.handle_line("PROFILE")
            .unwrap()
            .starts_with("ERR bad command")
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
