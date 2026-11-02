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
