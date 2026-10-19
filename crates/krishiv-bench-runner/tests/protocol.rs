// SPDX-License-Identifier: Apache-2.0
//! The line protocol over the checked-in TPC-H `lineitem` sample.
#![cfg(test)]

use std::path::PathBuf;

use krishiv_bench_runner::{Runner, RunnerConfig, serve};

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
        sql = sql.display(),
        bad = bad.display(),
        out = out.display(),
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
    assert!(err.contains("no <table>.parquet"), "{err}");
    let missing = Runner::new(&RunnerConfig {
        data: "/does/not/exist".to_owned(),
        threads: 1,
        memory_limit: "1g".to_owned(),
    });
    assert!(missing.unwrap_err().contains("cannot read"));
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
    assert!(args("--threads 4").unwrap_err().contains("--data"));
    assert!(args("--data").unwrap_err().contains("needs a value"));
    assert!(
        args("--data /d --threads x")
            .unwrap_err()
            .contains("--threads")
    );
    assert!(args("--wat").unwrap_err().contains("unknown argument"));
}
