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
        memory_limit: "1g".to_owned(),
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
        memory_limit: "1g".to_owned(),
    })
    .unwrap_err();
    assert!(err.to_string().contains("no <table>.parquet"), "{err}");
    let missing = Runner::new(&RunnerConfig {
        data: "/does/not/exist".to_owned(),
        threads: 1,
        memory_limit: "1g".to_owned(),
    });
    assert!(missing.unwrap_err().to_string().contains("cannot read"));
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
        memory_limit: "1g".to_owned(),
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
