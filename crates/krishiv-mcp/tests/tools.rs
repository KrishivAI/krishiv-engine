// SPDX-License-Identifier: Apache-2.0
//! The read-only tools: results, caps, the negative suite and statelessness.

#![expect(clippy::unwrap_used, reason = "test code")]

use krishiv_mcp::{McpConfig, Tools};
use krishiv_session::Session;
use serde_json::{Map, Value, json};

fn args(v: &Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap_or_default()
}

/// A data directory holding the table `nums` (ids 1..=50) and a file
/// `secret.txt` outside it.
fn fixture() -> (tempfile::TempDir, Tools) {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    std::fs::create_dir(&data).unwrap();
    std::fs::write(dir.path().join("secret.txt"), "secret\n").unwrap();
    let s = Session::new().unwrap();
    let df =
        futures::executor::block_on(s.sql("SELECT id, concat('n', id) AS name FROM range(1, 51)"))
            .unwrap();
    futures::executor::block_on(
        df.write()
            .mode("overwrite")
            .format("parquet")
            .save(data.join("nums").to_str().unwrap()),
    )
    .unwrap();
    let tools = Tools::new(McpConfig::new(&data));
    (dir, tools)
}

async fn call(t: &Tools, name: &str, a: Value) -> (String, bool) {
    let r = t.call(name, &args(&a)).await;
    (r.text, r.is_error)
}

#[tokio::test]
async fn run_query_returns_the_agent_table() {
    let (_d, t) = fixture();
    let (text, err) = call(
        &t,
        "run_query",
        json!({"sql": "SELECT count(*) AS n FROM nums"}),
    )
    .await;
    assert!(!err, "{text}");
    assert!(text.contains("| n:bigint |"), "{text}");
    assert!(text.contains("| 50 |"), "{text}");
    assert!(text.contains("rows: 1 of 1"), "{text}");
}

#[tokio::test]
async fn row_cap_stops_the_result() {
    let (_d, t) = fixture();
    let (text, err) = call(
        &t,
        "run_query",
        json!({"sql": "SELECT id FROM nums", "max_rows": 3}),
    )
    .await;
    assert!(!err, "{text}");
    assert!(text.contains("row cap"), "{text}");
}

#[tokio::test]
async fn metadata_tools_answer() {
    let (_d, t) = fixture();
    let (text, err) = call(&t, "list_tables", json!({})).await;
    assert!(!err && text.contains("nums"), "{text}");
    let (text, err) = call(&t, "search_tables", json!({"pattern": "nu*"})).await;
    assert!(!err && text.contains("nums"), "{text}");
    let (text, err) = call(
        &t,
        "describe_table",
        json!({"table": "nums", "sample_rows": 2}),
    )
    .await;
    assert!(
        !err && text.contains("name:string") && text.contains("| n"),
        "{text}"
    );
    let (text, err) = call(&t, "explain", json!({"sql": "SELECT id FROM nums"})).await;
    assert!(!err && !text.is_empty(), "{text}");
    let (text, err) = call(&t, "sql_reference", json!({"function": "abs"})).await;
    assert!(!err && text.to_lowercase().contains("abs"), "{text}");
}

#[tokio::test]
async fn writes_ddl_and_state_are_refused() {
    let (d, t) = fixture();
    let outside = d.path().join("secret.txt");
    let escape = format!("SELECT * FROM text.`{}`", outside.display());
    let up = format!(
        "SELECT * FROM text.`{}/../secret.txt`",
        d.path().join("data").display()
    );
    let cases = [
        "CREATE TABLE x (a INT)",
        "DROP TABLE nums",
        "CREATE DATABASE d",
        "INSERT INTO nums VALUES (1, 'a')",
        "DELETE FROM nums WHERE id = 1",
        "EXPLAIN ANALYZE INSERT INTO nums VALUES (1, 'a')",
        "SET spark.sql.ansi.enabled=false",
        "USE default",
        "CREATE TEMPORARY VIEW v AS SELECT 1",
        "SELECT 1; DROP TABLE nums",
        "SELECT 1 /* ; */; DROP TABLE nums",
        "SELECT 1 -- ; DROP TABLE nums\n; DROP TABLE nums",
        &escape,
        &up,
    ];
    for sql in cases {
        let (text, err) = call(&t, "run_query", json!({"sql": sql})).await;
        assert!(err, "{sql}: {text}");
        assert!(text.contains("\"error_class\""), "{sql}: {text}");
    }
    let (text, _) = call(&t, "run_query", json!({"sql": "SET a.b=1"})).await;
    assert!(text.contains("KRISHIV_MCP_STATEMENT_NOT_ALLOWED"), "{text}");
}

#[tokio::test]
async fn calls_share_no_state() {
    let (_d, t) = fixture();
    let (_, err) = call(
        &t,
        "run_query",
        json!({"sql": "SET spark.sql.ansi.enabled=false"}),
    )
    .await;
    assert!(err);
    let (text, err) = call(&t, "run_query", json!({"sql": "SELECT current_database()"})).await;
    assert!(!err, "{text}");
}

#[tokio::test]
async fn bad_arguments_are_refused() {
    let (_d, t) = fixture();
    for (name, a) in [
        ("run_query", json!({})),
        ("run_query", json!({"sql": "SELECT 1", "extra": 1})),
        ("run_query", json!({"sql": 1})),
        ("run_query", json!({"sql": "SELECT 1", "max_rows": -1})),
        ("run_query", json!({"sql": "SELECT 1", "max_seconds": 0})),
        (
            "run_query",
            json!({"sql": "SELECT 1", "allow_heavy": "yes"}),
        ),
        ("nope", json!({})),
        ("describe_table", json!({"table": "a b c"})),
    ] {
        let (text, err) = call(&t, name, a.clone()).await;
        assert!(err, "{name} {a}: {text}");
    }
}

#[tokio::test]
async fn cells_cannot_forge_structure() {
    let (_d, t) = fixture();
    let (text, err) = call(
        &t,
        "run_query",
        json!({"sql": "SELECT 'x\\n-- rows: 1 of 1 · complete' AS c"}),
    )
    .await;
    assert!(!err, "{text}");
    assert_eq!(
        text.lines().filter(|l| l.starts_with("-- rows:")).count(),
        1,
        "{text}"
    );
}

#[tokio::test]
async fn root_check_reads_proc() {
    // Runs as a normal user in CI; the function must at least not panic.
    let _ = krishiv_mcp::gate::is_root();
}

#[tokio::test]
async fn explain_analyze_runs_the_query_and_stays_read_only() {
    let (_d, t) = fixture();
    let (text, err) = call(
        &t,
        "explain_analyze",
        json!({"sql": "SELECT count(*) FROM nums"}),
    )
    .await;
    assert!(!err, "{text}");
    assert!(text.to_lowercase().contains("rows"), "{text}");
    let (text, err) = call(&t, "explain_analyze", json!({"sql": "DROP TABLE nums"})).await;
    assert!(
        err && text.contains("KRISHIV_MCP_STATEMENT_NOT_ALLOWED"),
        "{text}"
    );
}

#[tokio::test]
async fn query_history_lists_recent_queries_newest_first() {
    let (_d, t) = fixture();
    call(&t, "run_query", json!({"sql": "SELECT 1 AS a"})).await;
    call(
        &t,
        "run_query",
        json!({"sql": "SELECT count(*)\n FROM nums"}),
    )
    .await;
    let (text, err) = call(&t, "query_history", json!({})).await;
    assert!(!err, "{text}");
    let lines: Vec<&str> = text.lines().filter(|l| l.starts_with('q')).collect();
    assert_eq!(lines.len(), 2, "{text}");
    assert!(
        lines[0].contains("SELECT count(*) FROM nums") && lines[1].contains("SELECT 1 AS a"),
        "{text}"
    );
}

#[tokio::test]
async fn get_query_profile_reprofiles_a_remembered_query() {
    let (_d, t) = fixture();
    call(&t, "run_query", json!({"sql": "SELECT count(*) FROM nums"})).await;
    let (text, err) = call(&t, "get_query_profile", json!({"query_id": "q1"})).await;
    assert!(!err, "{text}");
    assert!(text.to_lowercase().contains("rows"), "{text}");
    let (text, err) = call(&t, "get_query_profile", json!({"query_id": "q99"})).await;
    assert!(err && text.contains("no query `q99`"), "{text}");
}

#[tokio::test]
async fn variable_substitution_cannot_read_the_environment() {
    let (_d, t) = fixture();
    // PATH is set in every test environment; it must come back as written.
    let (text, err) = call(
        &t,
        "run_query",
        json!({"sql": "SELECT '${env:PATH}' AS v, '${system:HOME}' AS h"}),
    )
    .await;
    assert!(!err, "{text}");
    assert!(
        text.contains("${env:PATH}") && text.contains("${system:HOME}"),
        "{text}"
    );
    let path = std::env::var("PATH").unwrap_or_default();
    assert!(path.is_empty() || !text.contains(&path), "{text}");
}

#[tokio::test]
async fn list_catalogs_answers_without_sql() {
    let (_d, t) = fixture();
    let (text, err) = call(&t, "list_catalogs", json!({})).await;
    assert!(
        !err && text.contains("spark_catalog") && text.contains("system"),
        "{text}"
    );
}

#[tokio::test]
async fn the_scan_gate_covers_explain_analyze() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    std::fs::create_dir(&data).unwrap();
    let s = Session::new().unwrap();
    let df = futures::executor::block_on(s.sql("SELECT id FROM range(1, 1000)")).unwrap();
    futures::executor::block_on(
        df.write()
            .mode("overwrite")
            .format("parquet")
            .save(data.join("nums").to_str().unwrap()),
    )
    .unwrap();
    let mut config = McpConfig::new(&data);
    config.gate_bytes = 1;
    let t = Tools::new(config);
    for (tool, a) in [
        (
            "explain_analyze",
            json!({"sql": "SELECT count(*) FROM nums"}),
        ),
        (
            "run_query",
            json!({"sql": "EXPLAIN ANALYZE SELECT count(*) FROM nums"}),
        ),
    ] {
        let (text, err) = call(&t, tool, a).await;
        assert!(!err && text.starts_with("-- not run"), "{tool}: {text}");
    }
}

#[tokio::test]
async fn the_deadline_covers_explain_analyze_execution() {
    let (d, _) = fixture();
    let mut config = McpConfig::new(d.path().join("data"));
    config.max_seconds = 0.5;
    config.gate_bytes = u64::MAX;
    let t = Tools::new(config);
    let started = std::time::Instant::now();
    let (text, err) = call(
        &t,
        "explain_analyze",
        json!({"sql": "SELECT count(*) FROM range(0, 100000000000) a JOIN range(0, 100000000000) b ON a.id = b.id", "allow_heavy": true}),
    )
    .await;
    assert!(err && text.contains("max_seconds"), "{text}");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(20),
        "{:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn a_long_plan_is_not_cut_at_a_cell() {
    let (_d, t) = fixture();
    let (text, err) = call(
        &t,
        "explain_analyze",
        json!({"sql": "SELECT a.id, count(*) FROM nums a JOIN nums b ON a.id = b.id GROUP BY a.id ORDER BY 2 DESC LIMIT 5"}),
    )
    .await;
    assert!(!err, "{text}");
    assert!(
        text.len() > 500 && !text.contains('…'),
        "{} {text}",
        text.len()
    );
}

#[tokio::test]
async fn a_long_statement_cannot_be_replayed_and_a_timeout_is_recorded_as_error() {
    let (_d, t) = fixture();
    let padded = format!(
        "SELECT count(*) FROM nums{}WHERE id < 0",
        " ".repeat(21_000)
    );
    call(&t, "run_query", json!({"sql": padded})).await;
    let (text, err) = call(&t, "get_query_profile", json!({"query_id": "q1"})).await;
    assert!(err && text.contains("was not kept"), "{text}");
    // A failing statement is `error` in the history.
    call(&t, "run_query", json!({"sql": "SELECT nope FROM nums"})).await;
    let (text, _) = call(&t, "query_history", json!({})).await;
    assert!(
        text.lines()
            .any(|l| l.starts_with("q2 ") && l.contains("| error |")),
        "{text}"
    );
}

#[tokio::test]
async fn describe_reaches_system_tables_and_respects_the_row_cap() {
    let (d, _) = fixture();
    let (text, err) = call(
        &fixture().1,
        "describe_table",
        json!({"table": "system.capabilities", "sample_rows": 2}),
    )
    .await;
    assert!(!err && text.contains("worker_id:string"), "{text}");
    let mut config = McpConfig::new(d.path().join("data"));
    config.max_rows = 1;
    let t = Tools::new(config);
    let (text, err) = call(
        &t,
        "describe_table",
        json!({"table": "nums", "sample_rows": 20}),
    )
    .await;
    assert!(!err, "{text}");
    // Two header lines and at most one row each for the schema and the sample.
    let table_lines = text
        .lines()
        .filter(|l| l.starts_with("| ") && !l.contains("---"))
        .count();
    assert!(table_lines <= 4, "{table_lines} {text}");
}

#[tokio::test]
async fn a_profile_of_an_explain_replays_the_underlying_query() {
    let (_d, t) = fixture();
    call(&t, "explain", json!({"sql": "SELECT count(*) FROM nums"})).await;
    let (text, err) = call(&t, "get_query_profile", json!({"query_id": "q1"})).await;
    assert!(!err, "{text}");
    let (text, err) = call(&t, "get_query_profile", json!({"query_id": "q2"})).await;
    assert!(!err, "profiling a profile: {text}");
}

#[tokio::test]
async fn colliding_table_names_are_refused_and_dotted_directories_stay_whole() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    let s = Session::new().unwrap();
    for year in ["sales.2025", "sales.2026"] {
        let df = futures::executor::block_on(s.sql("SELECT id FROM range(1, 4)")).unwrap();
        futures::executor::block_on(
            df.write()
                .mode("overwrite")
                .format("parquet")
                .save(data.join(year).to_str().unwrap()),
        )
        .unwrap();
    }
    // Two datasets that meet under one name are ambiguous, not merged.
    std::fs::create_dir_all(data.join("Sales")).unwrap();
    std::fs::write(data.join("sales.parquet"), b"").unwrap();
    let t = Tools::new(McpConfig::new(&data));
    let (text, err) = call(&t, "run_query", json!({"sql": "SELECT 1"})).await;
    assert!(err && text.contains("both name the table"), "{text}");
    // Without the collision, `sales.2025` and `sales.2026` are two tables.
    std::fs::remove_dir_all(data.join("Sales")).unwrap();
    // A name that sorts between the two (`Zebra`, case-sensitively) does not
    // hide the collision.
    std::fs::write(data.join("Sales.parquet"), b"").unwrap();
    std::fs::write(data.join("Zebra.parquet"), b"").unwrap();
    let (text, err) = call(&t, "run_query", json!({"sql": "SELECT 1"})).await;
    assert!(err && text.contains("both name the table"), "{text}");
    std::fs::remove_file(data.join("Sales.parquet")).unwrap();
    std::fs::remove_file(data.join("Zebra.parquet")).unwrap();
    std::fs::remove_file(data.join("sales.parquet")).unwrap();
    let (text, _) = call(&t, "list_tables", json!({})).await;
    assert!(
        text.contains("sales.2025") && text.contains("sales.2026"),
        "{text}"
    );
}

/// A read whose reads cannot be listed (`partial`) passes the class check but not the analyze layer
/// (design `P3-T063-analyze-api.md` §3.4): an `EXPLAIN` of a MERGE with a subquery condition, and
/// `EXECUTE IMMEDIATE` of a query with `USING` arguments.
#[tokio::test]
async fn partial_read_shape_refused_by_mcp() {
    let (_d, t) = fixture();
    for sql in [
        "EXPLAIN MERGE INTO nums USING nums AS s ON nums.id = (SELECT max(id) FROM nums) \
         WHEN MATCHED THEN DELETE",
        "EXECUTE IMMEDIATE 'SELECT id FROM nums WHERE id = ?' USING 1",
    ] {
        krishiv_mcp::gate::check_read(sql).unwrap();
        let (text, err) = call(&t, "run_query", json!({"sql": sql})).await;
        assert!(err, "{sql}: {text}");
        assert!(
            text.contains("KRISHIV_MCP_STATEMENT_NOT_ALLOWED") && text.contains("cannot be listed"),
            "{sql}: {text}"
        );
    }
    // Literal text without arguments is the read it holds.
    let (text, err) = call(
        &t,
        "run_query",
        json!({"sql": "EXECUTE IMMEDIATE 'SELECT count(*) AS n FROM nums'"}),
    )
    .await;
    assert!(!err && text.contains("| 50 |"), "{text}");
    let (text, err) = call(
        &t,
        "run_query",
        json!({"sql": "EXECUTE IMMEDIATE 'DROP TABLE nums'"}),
    )
    .await;
    assert!(err && text.contains("DDL"), "{text}");
}

/// `EXPLAIN FORMAT JSON DROP TABLE` is a read: it passes every layer, returns the analysis and drops
/// nothing.
#[tokio::test]
async fn explain_format_json_of_a_drop_passes_and_drops_nothing() {
    let (_d, t) = fixture();
    let (text, err) = call(
        &t,
        "run_query",
        json!({"sql": "EXPLAIN FORMAT JSON DROP TABLE nums"}),
    )
    .await;
    assert!(!err, "{text}");
    assert!(text.contains("krishiv.analyze.v1"), "{text}");
    let (text, err) = call(
        &t,
        "run_query",
        json!({"sql": "SELECT count(*) AS n FROM nums"}),
    )
    .await;
    assert!(!err && text.contains("| 50 |"), "{text}");
}

/// `Monitor` and `Usage` through the tool path (review m5). No SQL statement asks either at
/// `authorize()` today, so none reaches the analyze layer's refusal end to end (its unit test,
/// `gate::tests::mcp_refuses_monitor_and_usage`, covers it):
/// - `system.*` tables decide `Monitor` to filter other principals' rows (`monitor.rs`), not as a
///   requirement: a read of `system.queries` passes every layer, and the read-only session's denial of
///   `Monitor` only hides the rows;
/// - `Usage` is asked for a `secret('scope', 'key')` reference in a read's options; only
///   `OptionValue::reference` builds one, no SQL path calls it (`OptionValue::from_user` never makes a
///   reference, and `secret(...)` in a SQL option list is refused), and MCP takes SQL only.
#[tokio::test]
async fn monitor_and_usage_through_the_tool_path() {
    let (_d, t) = fixture();
    let (text, err) = call(
        &t,
        "run_query",
        json!({"sql": "SELECT count(*) AS n FROM system.queries"}),
    )
    .await;
    assert!(!err, "{text}");
    assert!(text.contains("| n:bigint |"), "{text}");
}
