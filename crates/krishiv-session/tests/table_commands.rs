// SPDX-License-Identifier: Apache-2.0
//! `CREATE TABLE`, `DROP TABLE`, `INSERT` and `TRUNCATE TABLE` on the
//! permanent catalog through `Session::sql`: what the tables read back, the
//! files they leave, and Spark's error classes.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

/// A session whose warehouse is `dir`.
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

fn err(s: &Session, sql: &str) -> KrishivError {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_err()
}

fn class(s: &Session, sql: &str) -> String {
    err(s, sql).error_class().name().to_owned()
}

/// The first column of `sql` as strings (`~` for NULL).
fn col0(s: &Session, sql: &str) -> Vec<String> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        let c = b.column(0);
        for i in 0..b.num_rows() {
            out.push(if c.is_null(i) {
                "~".to_owned()
            } else {
                match c.data_type() {
                    ArrowType::Utf8View => c.as_string_view().value(i).to_owned(),
                    ArrowType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
                    other => panic!("{sql}: unexpected type {other}"),
                }
            });
        }
    }
    out
}

#[test]
fn create_insert_select_and_drop_a_parquet_table() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT, b STRING) USING parquet");
    assert!(dir.path().join("t").is_dir());
    run(&s, "INSERT INTO t VALUES (1, 'x'), (2, 'y')");
    run(&s, "INSERT INTO t VALUES (3, NULL)");
    assert_eq!(
        col0(
            &s,
            "SELECT concat_ws('|', CAST(a AS STRING), b) FROM t ORDER BY a"
        ),
        ["1|x", "2|y", "3"]
    );
    // Qualified spellings find the same table.
    assert_eq!(
        col0(&s, "SELECT CAST(count(*) AS STRING) FROM default.t"),
        ["3"]
    );
    assert_eq!(
        col0(
            &s,
            "SELECT CAST(count(*) AS STRING) FROM spark_catalog.default.t"
        ),
        ["3"]
    );
    run(&s, "INSERT OVERWRITE t VALUES (9, 'z')");
    assert_eq!(
        col0(&s, "SELECT concat_ws('|', CAST(a AS STRING), b) FROM t"),
        ["9|z"]
    );
    run(&s, "DROP TABLE t");
    assert!(!dir.path().join("t").exists());
    assert_eq!(class(&s, "SELECT * FROM t"), "TABLE_OR_VIEW_NOT_FOUND");
    run(&s, "DROP TABLE IF EXISTS t");
    assert_eq!(class(&s, "DROP TABLE t"), "TABLE_OR_VIEW_NOT_FOUND");
}

#[test]
fn csv_tables_read_back_what_they_were_given() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE c (id BIGINT, name STRING) USING csv");
    run(&s, "INSERT INTO c VALUES (1, 'a'), (2, 'b')");
    assert_eq!(
        col0(
            &s,
            "SELECT concat_ws('|', CAST(id AS STRING), name) FROM c ORDER BY id"
        ),
        ["1|a", "2|b"]
    );
}

#[test]
fn create_table_errors() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT)");
    assert_eq!(
        class(&s, "CREATE TABLE t (a INT)"),
        "TABLE_OR_VIEW_ALREADY_EXISTS"
    );
    run(&s, "CREATE TABLE IF NOT EXISTS t (a INT, b INT)");
    assert_eq!(col0(&s, "SELECT CAST(count(*) AS STRING) FROM t"), ["0"]);
    assert_eq!(
        class(&s, "CREATE TABLE d (a INT) USING delta"),
        "DATA_SOURCE_NOT_FOUND"
    );
    assert_eq!(
        class(&s, "CREATE TABLE h (a INT) STORED AS parquet"),
        "NOT_SUPPORTED_COMMAND_WITHOUT_HIVE_SUPPORT"
    );
    assert_eq!(class(&s, "CREATE TABLE nodb.x (a INT)"), "SCHEMA_NOT_FOUND");
    assert_eq!(
        class(&s, "CREATE TABLE dup (a INT, A INT)"),
        "COLUMN_ALREADY_EXISTS"
    );
    // OR REPLACE swaps the columns and empties the table.
    run(&s, "INSERT INTO t VALUES (1)");
    run(&s, "CREATE OR REPLACE TABLE t (x STRING)");
    assert_eq!(col0(&s, "SELECT x FROM t"), Vec::<String>::new());
}

#[test]
fn insert_matches_columns_and_applies_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (a INT, b STRING, c DECIMAL(5,2) DEFAULT 1.5) USING parquet",
    );
    run(&s, "INSERT INTO t VALUES (1, 'x', 2.25)");
    run(&s, "INSERT INTO t (b, a) VALUES ('y', 2)");
    run(
        &s,
        "INSERT INTO t BY NAME SELECT 'z' AS b, 3 AS a, 4.5 AS c",
    );
    assert_eq!(
        col0(
            &s,
            "SELECT concat_ws('|', CAST(a AS STRING), b, CAST(c AS STRING)) FROM t ORDER BY a"
        ),
        ["1|x|2.25", "2|y|1.50", "3|z|4.50"]
    );
    assert_eq!(
        class(&s, "INSERT INTO t VALUES (1, 'x')"),
        "INSERT_COLUMN_ARITY_MISMATCH.NOT_ENOUGH_DATA_COLUMNS"
    );
    assert_eq!(
        class(&s, "INSERT INTO t VALUES (1, 'x', 1, 4)"),
        "INSERT_COLUMN_ARITY_MISMATCH.TOO_MANY_DATA_COLUMNS"
    );
    assert_eq!(
        class(&s, "INSERT INTO t VALUES ('abc', 'o', 1)"),
        "INCOMPATIBLE_DATA_FOR_TABLE.CANNOT_SAFELY_CAST"
    );
    assert_eq!(
        class(&s, "INSERT INTO t (a, nope) VALUES (1, 2)"),
        "INCOMPATIBLE_DATA_FOR_TABLE.EXTRA_COLUMNS"
    );
    assert_eq!(
        class(&s, "INSERT INTO missing VALUES (1)"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

#[test]
fn insert_into_a_view_and_other_wrong_targets() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TEMPORARY VIEW v AS SELECT 1 AS a");
    assert_eq!(
        class(&s, "INSERT INTO v VALUES (1)"),
        "EXPECT_TABLE_NOT_VIEW.NO_ALTERNATIVE"
    );
    assert_eq!(class(&s, "DROP TABLE v"), "WRONG_COMMAND_FOR_OBJECT_TYPE");
    run(&s, "CREATE TABLE t (a INT)");
    assert_eq!(
        class(&s, "INSERT INTO t PARTITION (a = 1) VALUES (1)"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

#[test]
fn ctas_writes_then_adds_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE u USING parquet AS SELECT 1 AS id, 'q' AS s",
    );
    assert_eq!(
        col0(&s, "SELECT concat_ws('|', CAST(id AS STRING), s) FROM u"),
        ["1|q"]
    );
    // A failing query leaves no table and no files.
    assert_eq!(
        class(&s, "CREATE TABLE bad AS SELECT * FROM nope"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    assert!(!dir.path().join("bad").exists());
    assert_eq!(class(&s, "SELECT * FROM bad"), "TABLE_OR_VIEW_NOT_FOUND");
}

#[test]
fn create_table_like_copies_columns() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE a (x INT, y STRING)");
    run(&s, "CREATE TABLE b LIKE a");
    run(&s, "INSERT INTO b VALUES (1, 'k')");
    assert_eq!(col0(&s, "SELECT y FROM b"), ["k"]);
    run(&s, "CREATE TEMPORARY VIEW v AS SELECT 1 AS p, 'q' AS r");
    run(&s, "CREATE TABLE c LIKE v");
    run(&s, "INSERT INTO c VALUES (2, 'r')");
    assert_eq!(col0(&s, "SELECT r FROM c"), ["r"]);
    assert_eq!(
        class(&s, "CREATE TABLE d LIKE nope"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

#[test]
fn truncate_keeps_the_directory() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (1), (2)");
    run(&s, "TRUNCATE TABLE t");
    assert!(dir.path().join("t").is_dir());
    assert_eq!(col0(&s, "SELECT CAST(count(*) AS STRING) FROM t"), ["0"]);
    run(&s, "INSERT INTO t VALUES (5)");
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM t"), ["5"]);
    assert_eq!(class(&s, "TRUNCATE TABLE nope"), "TABLE_OR_VIEW_NOT_FOUND");
}

#[test]
fn external_tables_keep_their_files_when_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    let loc = dir.path().join("ext");
    run(
        &s,
        &format!(
            "CREATE EXTERNAL TABLE e (a INT) USING parquet LOCATION '{}'",
            loc.display()
        ),
    );
    run(&s, "INSERT INTO e VALUES (7)");
    run(&s, "DROP TABLE e");
    assert!(loc.is_dir());
    // Re-created over the same files, the rows are still there.
    run(
        &s,
        &format!(
            "CREATE EXTERNAL TABLE e (a INT) USING parquet LOCATION '{}'",
            loc.display()
        ),
    );
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM e"), ["7"]);
}

#[test]
fn temporary_views_shadow_tables_of_the_same_bare_name() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (1)");
    run(&s, "CREATE TEMPORARY VIEW t AS SELECT 99 AS a");
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM t"), ["99"]);
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM default.t"), ["1"]);
    assert_eq!(
        class(&s, "INSERT INTO t VALUES (2)"),
        "EXPECT_TABLE_NOT_VIEW.NO_ALTERNATIVE"
    );
}

#[test]
fn a_session_without_a_warehouse_uses_its_own_directory() {
    let s = Session::new().unwrap();
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (3)");
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM t"), ["3"]);
    let _ = ErrorClass::DivideByZero;
}
