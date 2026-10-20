// SPDX-License-Identifier: Apache-2.0
//! `krishiv sql`: -e, -f, stdin, formats, --table/--conf, errors, TPC-H Q1.
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::io::Write;
use std::process::{Command, Output, Stdio};

fn krishiv(args: &[&str], stdin: Option<&str>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_krishiv"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn krishiv");
    let mut pipe = child.stdin.take().expect("stdin");
    if let Some(s) = stdin {
        pipe.write_all(s.as_bytes()).expect("write stdin");
    }
    drop(pipe);
    child.wait_with_output().expect("wait")
}

fn out(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).expect("utf-8")
}
fn err(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).expect("utf-8")
}

const LINEITEM: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/tpch/q1-lineitem.parquet"
);

#[test]
fn execute_prints_a_table() {
    let o = krishiv(&["sql", "-e", "select 1 as a, 'x' as b"], None);
    assert!(o.status.success(), "{}", err(&o));
    let t = out(&o);
    assert!(t.contains("| a | b |"), "{t}");
    assert!(t.contains("| 1 | x |"), "{t}");
    assert!(t.trim_end().ends_with("1 row"), "{t}");
}

#[test]
fn file_runs_every_statement() {
    let dir = tempfile::tempdir().expect("tempdir");
    let f = dir.path().join("q.sql");
    std::fs::write(&f, "select 1 as a; -- c;\nselect 'p;q' as b;").expect("write");
    let o = krishiv(
        &["sql", "-o", "csv", "-f", f.to_str().expect("utf-8")],
        None,
    );
    assert!(o.status.success(), "{}", err(&o));
    assert_eq!(out(&o), "a\n1\nb\np;q\n");
}

#[test]
fn stdin_is_read_without_flags() {
    let o = krishiv(
        &["sql", "-o", "csv"],
        Some("select 'a,\"b' as s, cast(null as int) as n"),
    );
    assert!(o.status.success(), "{}", err(&o));
    assert_eq!(out(&o), "s,n\n\"a,\"\"b\",\n");
}

#[test]
fn json_lines() {
    let o = krishiv(
        &[
            "sql",
            "-o",
            "json",
            "-e",
            "select 1 as a, 'x\"' as b, 1.5 as c, cast(null as int) as d",
        ],
        None,
    );
    assert!(o.status.success(), "{}", err(&o));
    assert_eq!(out(&o), "{\"a\":1,\"b\":\"x\\\"\",\"c\":1.5,\"d\":null}\n");
}

#[test]
fn conf_is_applied() {
    let strict = krishiv(&["sql", "-e", "select 1 / 0"], None);
    assert_eq!(strict.status.code(), Some(1));
    assert!(err(&strict).contains("DIVIDE_BY_ZERO"), "{}", err(&strict));
    let lax = krishiv(
        &[
            "sql",
            "-o",
            "csv",
            "--conf",
            "spark.sql.ansi.enabled=false",
            "-e",
            "select 1 / 0 as x",
        ],
        None,
    );
    assert!(lax.status.success(), "{}", err(&lax));
    assert_eq!(out(&lax), "x\n\n");
    let bad = krishiv(&["sql", "--conf", "novalue", "-e", "select 1"], None);
    assert_eq!(bad.status.code(), Some(2));
}

#[test]
fn syntax_error_exits_1_with_class_sqlstate_and_caret() {
    let o = krishiv(&["sql", "-e", "xyz 1"], None);
    assert_eq!(o.status.code(), Some(1));
    assert!(out(&o).is_empty());
    let e = err(&o);
    assert!(e.contains("[PARSE_SYNTAX_ERROR]"), "{e}");
    assert!(e.contains("SQLSTATE: 42601"), "{e}");
    assert!(
        e.contains("(line 1, pos 0)\n\n== SQL ==\nxyz 1\n^^^"),
        "{e}"
    );
}

#[test]
fn caret_line_counts_from_the_file_start() {
    let o = krishiv(&["sql", "-e", "select 1;\n\nselect  from x"], None);
    assert_eq!(o.status.code(), Some(1));
    assert!(err(&o).contains("(line 3, pos 8)"), "{}", err(&o));
}

#[test]
fn unknown_table_and_missing_file_fail() {
    let o = krishiv(&["sql", "-e", "select * from nope"], None);
    assert_eq!(o.status.code(), Some(1));
    assert!(err(&o).contains("TABLE_OR_VIEW_NOT_FOUND"), "{}", err(&o));
    let o = krishiv(&["sql", "-f", "/nonexistent/x.sql"], None);
    assert_eq!(o.status.code(), Some(1));
    let o = krishiv(
        &["sql", "--table", "t=/nonexistent", "-e", "select 1"],
        None,
    );
    assert_eq!(o.status.code(), Some(1));
}

#[test]
fn timing_goes_to_stderr() {
    let o = krishiv(
        &["sql", "--timing", "-o", "csv", "-e", "select 1 as a"],
        None,
    );
    assert!(err(&o).starts_with("Time: "), "{}", err(&o));
    assert_eq!(out(&o), "a\n1\n");
}

/// TPC-H Q1 over the checked-in `lineitem` fixture; with `KRISHIV_TPCH_SF1`
/// set to a directory holding `lineitem.parquet`, the same query at SF1.
#[test]
fn tpch_q1_over_a_parquet_table() {
    let path = std::env::var("KRISHIV_TPCH_SF1")
        .map_or_else(|_| LINEITEM.to_owned(), |d| format!("{d}/lineitem.parquet"));
    let q1 = "select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, count(*) as count_order \
              from lineitem where l_shipdate <= date '1998-09-02' \
              group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus";
    let table = format!("lineitem={path}");
    let o = krishiv(&["sql", "-o", "csv", "--table", &table, "-e", q1], None);
    assert!(o.status.success(), "{}", err(&o));
    let want = "l_returnflag,l_linestatus,sum_qty,count_order\n";
    if std::env::var("KRISHIV_TPCH_SF1").is_ok() {
        // TPC-H SF1 Q1 reference rows.
        let t = out(&o);
        assert!(t.starts_with(want), "{t}");
        assert!(t.contains("A,F,37734107.00,1478493\n"), "{t}");
        assert!(t.contains("N,O,74476040.00,2920374\n"), "{t}");
    } else {
        let rows = "A,F,38726.00,1490\nN,F,780.00,33\nN,O,74790.00,2915\nR,F,36245.00,1430\n";
        assert_eq!(out(&o), format!("{want}{rows}"));
    }
}

#[test]
fn dash_file_reads_stdin_and_types_render() {
    let q = "select cast(1.50 as decimal(5,2)) as d, date '2020-01-02' as dt, true as b, 2.5 as f";
    let o = krishiv(&["sql", "-o", "json", "-f", "-"], Some(q));
    assert!(o.status.success(), "{}", err(&o));
    let t = out(&o);
    assert!(
        t.contains("\"d\":1.50") && t.contains("\"dt\":\"2020-01-02\"") && t.contains("\"b\":true"),
        "{t}"
    );
}

#[test]
fn execute_conflicts_with_file() {
    let o = krishiv(&["sql", "-e", "select 1", "-f", "x.sql"], None);
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn table_registers_a_view_and_empty_input_is_ok() {
    let o = krishiv(
        &[
            "sql",
            "--table",
            &format!("t={LINEITEM}"),
            "-o",
            "csv",
            "-e",
            "select count(*) as n from t",
        ],
        None,
    );
    assert!(
        o.status.success() && out(&o).starts_with("n\n"),
        "{}",
        err(&o)
    );
    let o = krishiv(&["sql"], Some("  ;; "));
    assert!(o.status.success() && out(&o).is_empty());
}

#[test]
fn csv_distinguishes_null_from_empty_and_zone_is_checked() {
    let o = krishiv(
        &[
            "sql",
            "-o",
            "csv",
            "-e",
            "select '' as e, cast(null as string) as n",
        ],
        None,
    );
    assert_eq!(out(&o), "e,n\n\"\",\n");
    let o = krishiv(
        &[
            "sql",
            "--conf",
            "spark.sql.session.timeZone=Mars/Base",
            "-e",
            "select 1",
        ],
        None,
    );
    assert_eq!(o.status.code(), Some(1));
    assert!(err(&o).contains("not supported"), "{}", err(&o));
    let o = krishiv(
        &[
            "sql",
            "-o",
            "csv",
            "-e",
            "select 1 as a; -- trailing comment",
        ],
        None,
    );
    assert!(o.status.success(), "{}", err(&o));
    assert_eq!(out(&o), "a\n1\n");
}

#[test]
fn explain_analyze_prints_the_profile() {
    let view = format!("lineitem={LINEITEM}");
    let q = "select l_returnflag, count(*) from lineitem group by l_returnflag";
    let o = krishiv(
        &["sql", "--table", &view, "--explain-analyze", "-e", q],
        None,
    );
    assert!(o.status.success(), "{}", err(&o));
    let t = out(&o);
    assert!(t.starts_with("== Physical Plan (analyzed) ==\n"), "{t}");
    assert!(t.contains("HashAggregate"), "{t}");
    assert!(t.contains("groups=3"), "{t}");
    let o = krishiv(
        &[
            "sql",
            "--table",
            &view,
            "--explain-analyze",
            "--json",
            "-e",
            q,
        ],
        None,
    );
    assert!(o.status.success(), "{}", err(&o));
    let p = krishiv::QueryProfile::from_json(out(&o).trim()).expect("profile JSON");
    let result = p
        .metrics
        .operators
        .values()
        .find(|e| e.node.is_none())
        .expect("result sink");
    assert_eq!(result.metrics.rows_in, 3);
}

#[test]
fn explain_verbose_prints_the_rules() {
    let q = "select a from (select 1 as a, 2 as b) where a > 0";
    let o = krishiv(&["sql", "--explain-verbose", "-e", q], None);
    assert!(o.status.success(), "{}", err(&o));
    let t = out(&o);
    assert!(t.starts_with("== Parsed Logical Plan ==\n"), "{t}");
    assert!(t.contains("== Optimizer: "), "{t}");
    let o = krishiv(&["sql", "--explain-verbose", "--json", "-e", q], None);
    assert!(o.status.success(), "{}", err(&o));
    let v = krishiv::ExplainVerbose::from_json(out(&o).trim()).expect("verbose JSON");
    assert!(!v.steps.is_empty());
}

#[test]
fn explain_flags_are_checked() {
    let o = krishiv(&["sql", "--json", "-e", "select 1"], None);
    assert_eq!(o.status.code(), Some(1));
    assert!(
        err(&o).contains("--json needs --explain-analyze or --explain-verbose"),
        "{}",
        err(&o)
    );
    let o = krishiv(
        &[
            "sql",
            "--explain-analyze",
            "--explain-verbose",
            "-e",
            "select 1",
        ],
        None,
    );
    assert_eq!(o.status.code(), Some(2));
    assert!(err(&o).contains("cannot be used with"), "{}", err(&o));
}
