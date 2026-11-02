// SPDX-License-Identifier: Apache-2.0
//! The `system` catalog (P2-T126 unit 1).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

/// The first column of `sql` as strings (the query casts to STRING itself).
fn strings(sql: &str) -> Vec<String> {
    let s = Session::new().unwrap();
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let b = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for batch in &b {
        let c = batch.column(0);
        if let Some(v) = c.as_string_view_opt() {
            out.extend((0..v.len()).map(|i| v.value(i).to_owned()));
        } else {
            let v = c.as_string::<i32>();
            out.extend((0..v.len()).map(|i| v.value(i).to_owned()));
        }
    }
    out
}

fn error(sql: &str) -> String {
    let s = Session::new().unwrap();
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => format!("{sql}: it succeeded"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn capabilities_lists_the_build() {
    let r = strings(
        "SELECT concat_ws('|', kind, name, value) FROM system.capabilities \
         WHERE kind = 'build' ORDER BY name",
    );
    assert_eq!(r[0], "build|engine|krishiv");
    let r = strings("SELECT value FROM system.capabilities WHERE name = 'system_tables'");
    assert_eq!(r, ["1"]);
}

#[test]
fn the_catalog_name_is_matched_without_case() {
    let r = strings("SELECT worker_id FROM SYSTEM.Capabilities LIMIT 1");
    assert_eq!(r, ["local"]);
}

#[test]
fn an_unknown_system_table_is_not_found() {
    let e = error("SELECT * FROM system.nope");
    assert!(e.contains("TABLE_OR_VIEW_NOT_FOUND"), "{e}");
}

#[test]
fn nothing_can_be_created_or_dropped_in_system() {
    for sql in [
        "CREATE DATABASE system",
        "CREATE SCHEMA IF NOT EXISTS system",
        "DROP DATABASE system",
        "CREATE DATABASE spark_catalog.system",
        "DROP SCHEMA SPARK_CATALOG.System",
        "CREATE TABLE system.x (a INT) USING parquet",
        "CREATE TABLE system.capabilities (a INT) USING parquet",
        "DROP TABLE system.capabilities",
        "INSERT INTO system.capabilities VALUES ('a', 'b', 'c', 'd', 'e')",
        "CREATE VIEW system.v AS SELECT 1",
    ] {
        let e = error(sql);
        assert!(e.contains("KRISHIV_RESERVED_NAME"), "{sql}: {e}");
    }
}

#[test]
fn memory_lists_the_pool_tree() {
    let r = strings("SELECT path FROM system.memory WHERE kind = 'pool' AND parent IS NULL");
    assert_eq!(r, ["process"]);
    // Rows are the tree as the scan was planned: before the reading query's own pool exists.
    let r = strings(
        "SELECT CAST(sum(reserved_bytes) >= 0 AS STRING) FROM system.memory WHERE kind = 'consumer'",
    );
    assert_eq!(r, ["true"]);
    let r = strings("SELECT worker_id FROM system.memory LIMIT 1");
    assert_eq!(r, ["local"]);
}

#[test]
fn a_view_over_a_system_table_ignores_a_session_table_of_the_same_name() {
    let s = Session::new().unwrap();
    let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "x",
        arrow_schema::DataType::Int32,
        false,
    )]));
    s.register_batches("capabilities", schema, vec![]).unwrap();
    let df = block_on(s.sql("SELECT name FROM system.capabilities")).unwrap();
    block_on(df.create_or_replace_temp_view("v")).unwrap();
    let rows = block_on(async {
        s.sql("SELECT count(*) FROM v WHERE name = 'engine'")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert!(rows[0].num_rows() == 1);
}

fn run_all(s: &Session, sql: &str) -> Result<(), String> {
    block_on(async { s.sql(sql).await?.collect().await })
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn strings_in(s: &Session, sql: &str) -> Vec<String> {
    let b = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let mut out = Vec::new();
    for batch in &b {
        let c = batch.column(0);
        if let Some(v) = c.as_string_view_opt() {
            out.extend((0..v.len()).map(|i| v.value(i).to_owned()));
        } else {
            let v = c.as_string::<i32>();
            out.extend((0..v.len()).map(|i| v.value(i).to_owned()));
        }
    }
    out
}

#[test]
fn queries_lists_finished_and_failed_statements_with_their_text() {
    let s = Session::new().unwrap();
    run_all(&s, "SELECT 1 AS one, 'secret-literal' AS s").unwrap();
    run_all(&s, "SELECT * FROM range(3)").unwrap();
    let ids = strings_in(
        &s,
        "SELECT query_id FROM system.queries ORDER BY submitted_at",
    );
    assert!(ids.len() >= 2, "{ids:?}");
    // Every id is a UUIDv7 and they differ.
    let mut seen = std::collections::BTreeSet::new();
    for id in &ids {
        assert_eq!(id.len(), 36, "{id}");
        assert_eq!(&id[14..15], "7", "{id}");
        assert!(seen.insert(id.clone()), "{id}");
    }
    let states = strings_in(&s, "SELECT DISTINCT state FROM system.queries");
    assert!(
        states
            .iter()
            .all(|x| ["finished", "failed", "running", "cancelled"].contains(&x.as_str())),
        "{states:?}"
    );
    let texts = strings_in(
        &s,
        "SELECT query_text FROM system.queries WHERE query_text LIKE 'SELECT 1 AS one%'",
    );
    assert_eq!(texts.len(), 1, "{texts:?}");
    // Literals are kept (`queryText.literals = keep`, the design default).
    assert!(texts[0].contains("secret-literal"), "{}", texts[0]);
    let walls = strings_in(
        &s,
        "SELECT CAST(count(*) AS STRING) FROM system.queries WHERE state = 'finished' AND wall_ns > 0 AND ended_at >= submitted_at",
    );
    assert!(walls[0].parse::<u32>().unwrap() >= 1, "{walls:?}");
}
