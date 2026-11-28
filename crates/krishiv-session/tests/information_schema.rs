// SPDX-License-Identifier: Apache-2.0
//! `information_schema.catalogs`, `.schemata`, `.tables`, `.columns` and `.views` over the session catalog.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn session(dir: &std::path::Path) -> Session {
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", dir.to_str().unwrap())
        .unwrap();
    s
}

fn run(s: &Session, sql: &str) {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

fn rows(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for batch in df {
        for r in 0..batch.num_rows() {
            let cells: Vec<String> = (0..batch.num_columns())
                .map(|c| arrow_cast::display::array_value_to_string(batch.column(c), r).unwrap())
                .collect();
            out.push(cells.join("|"));
        }
    }
    out
}

#[test]
fn schemata_tables_and_columns_list_the_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE DATABASE db1");
    run(
        &s,
        "CREATE TABLE db1.t (a INT NOT NULL, b STRING) USING parquet",
    );
    run(&s, "CREATE VIEW db1.v AS SELECT 1 AS x");
    assert_eq!(
        rows(
            &s,
            "SELECT * FROM information_schema.schemata ORDER BY schema_name"
        ),
        ["spark_catalog|db1", "spark_catalog|default"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT table_schema, table_name, table_type FROM information_schema.tables ORDER BY 2"
        ),
        ["db1|t|BASE TABLE", "db1|v|VIEW"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT table_name, column_name, ordinal_position, data_type, is_nullable \
             FROM information_schema.columns WHERE table_name = 't' ORDER BY ordinal_position"
        ),
        ["t|a|1|int|YES", "t|b|2|string|YES"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT table_schema, table_name, view_definition FROM information_schema.views"
        ),
        ["db1|v|SELECT 1 AS x"]
    );
}

#[test]
fn catalogs_lists_the_session_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    assert_eq!(
        rows(&s, "SELECT * FROM information_schema.catalogs"),
        ["spark_catalog"]
    );
}
