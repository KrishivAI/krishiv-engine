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
