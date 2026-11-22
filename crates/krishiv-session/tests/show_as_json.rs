// SPDX-License-Identifier: Apache-2.0
//! `SHOW TABLES | SHOW TABLE EXTENDED | SHOW PARTITIONS ... AS JSON` (Spark 4.3): one `json_metadata` document.
//! Expected texts are from a Spark 4.3.0 run (`spark-sql`, 2026-11-22).
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::builder().build().expect("session");
    for stmt in [
        "CREATE TABLE j3(a INT, b STRING) USING parquet PARTITIONED BY (b)",
        "INSERT INTO j3 VALUES (1, 'x')",
        "CREATE TEMPORARY VIEW tv AS SELECT 1",
    ] {
        block_on(async { s.sql(stmt).await?.collect().await }).expect("setup");
    }
    s
}

fn json(s: &Session, sql: &str) -> String {
    let batches = block_on(async { s.sql(sql).await?.collect().await }).expect("runs");
    let schema = batches[0].schema();
    assert_eq!(schema.fields().len(), 1);
    assert_eq!(schema.field(0).name(), "json_metadata");
    assert_eq!(
        batches
            .iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        1
    );
    arrow_cast::display::array_value_to_string(batches[0].column(0), 0).expect("text")
}

#[test]
fn show_tables_lists_tables_and_temporary_views() {
    let s = session();
    assert_eq!(
        json(&s, "SHOW TABLES AS JSON"),
        r#"{"tables":[{"name":"j3","namespace":["default"],"isTemporary":false},{"name":"tv","namespace":[],"isTemporary":true}]}"#
    );
    assert_eq!(
        json(&s, "SHOW TABLES IN default LIKE 'j*' AS JSON"),
        r#"{"tables":[{"name":"j3","namespace":["default"],"isTemporary":false}]}"#
    );
    assert_eq!(
        json(&s, "SHOW TABLES LIKE 'zz' AS JSON"),
        r#"{"tables":[]}"#
    );
}

#[test]
fn show_table_extended_gives_the_type() {
    let s = session();
    block_on(async { s.sql("CREATE VIEW v3 AS SELECT 1").await?.collect().await }).expect("view");
    assert_eq!(
        json(&s, "SHOW TABLE EXTENDED LIKE 'j3' AS JSON"),
        r#"{"tables":[{"name":"j3","catalog":"spark_catalog","namespace":["default"],"type":"MANAGED","isTemporary":false}]}"#
    );
    assert_eq!(
        json(&s, "SHOW TABLE EXTENDED LIKE 'v3' AS JSON"),
        r#"{"tables":[{"name":"v3","catalog":"spark_catalog","namespace":["default"],"type":"VIEW","isTemporary":false}]}"#
    );
    assert_eq!(
        json(&s, "SHOW TABLE EXTENDED IN default LIKE 'tv' AS JSON"),
        r#"{"tables":[{"name":"tv","catalog":"spark_catalog","namespace":[],"type":"VIEW","isTemporary":true}]}"#
    );
}

#[test]
fn show_table_extended_with_a_partition_is_refused() {
    let s = session();
    let e = block_on(async {
        s.sql("SHOW TABLE EXTENDED LIKE 'j3' PARTITION (b='x') AS JSON")
            .await?
            .collect()
            .await
    })
    .expect_err("refused");
    assert_eq!(
        e.error_class().name(),
        "UNSUPPORTED_FEATURE.SHOW_TABLE_EXTENDED_JSON_WITH_PARTITION"
    );
}

#[test]
fn show_partitions_gives_one_object_per_partition() {
    let s = session();
    assert_eq!(
        json(&s, "SHOW PARTITIONS j3 AS JSON"),
        r#"{"partitions":[{"b":"x"}]}"#
    );
    assert_eq!(
        json(&s, "SHOW PARTITIONS j3 PARTITION (b='x') AS JSON"),
        r#"{"partitions":[{"b":"x"}]}"#
    );
}
