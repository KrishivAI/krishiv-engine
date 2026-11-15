// SPDX-License-Identifier: Apache-2.0
//! `SHOW CREATE TABLE` of tables with nested column types prints Spark's `StructType.toDDL`: `ARRAY<..>`,
//! `MAP<.., ..>` and `STRUCT<name: type, ..>` with field names quoted where needed, field comments, every
//! string as `STRING COLLATE UTF8_BINARY` and a declared `CHAR(n)` / `VARCHAR(n)` kept at its place. The
//! expected texts are Spark 4.2.0's output for the same statements (`spark-shell`, `local[1]`).
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn run(s: &Session, sql: &str) {
    block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
}

fn show_create(s: &Session, name: &str) -> String {
    let df = block_on(s.sql(&format!("SHOW CREATE TABLE {name}"))).unwrap();
    let batches = block_on(df.collect()).unwrap();
    batches[0].column(0).as_string_view().value(0).to_owned()
}

#[test]
fn declared_nested_columns_print_as_spark_prints_them() {
    let s = Session::new().unwrap();
    run(
        &s,
        "CREATE TABLE t1 (i INT NOT NULL, st STRING, a ARRAY<INT>, m MAP<INT, STRING>, \
         s STRUCT<x: INT, `y z`: STRING COMMENT 'c1', `1`: DECIMAL(10,2)>, \
         n ARRAY<STRUCT<k: BIGINT, v: MAP<STRING, ARRAY<DATE>>>> COMMENT 'top', ts TIMESTAMP_NTZ, \
         b BINARY, c CHAR(3), vc VARCHAR(5)) USING parquet",
    );
    assert_eq!(
        show_create(&s, "t1"),
        "CREATE TABLE spark_catalog.default.t1 (\n  i INT,\n  st STRING COLLATE UTF8_BINARY,\n  \
         a ARRAY<INT>,\n  m MAP<INT, STRING COLLATE UTF8_BINARY>,\n  \
         s STRUCT<x: INT, `y z`: STRING COLLATE UTF8_BINARY COMMENT 'c1', `1`: DECIMAL(10,2)>,\n  \
         n ARRAY<STRUCT<k: BIGINT, v: MAP<STRING COLLATE UTF8_BINARY, ARRAY<DATE>>>> COMMENT 'top',\n  \
         ts TIMESTAMP_NTZ,\n  b BINARY,\n  c CHAR(3) COLLATE UTF8_BINARY,\n  \
         vc VARCHAR(5) COLLATE UTF8_BINARY)\nUSING parquet\n"
    );
}

#[test]
fn deep_nesting_and_intervals_in_a_json_table() {
    let s = Session::new().unwrap();
    run(
        &s,
        "CREATE TABLE t3 (s STRUCT<a: STRUCT<b: ARRAY<MAP<STRING, STRUCT<c: FLOAT>>>>>, \
         ym INTERVAL YEAR TO MONTH, dt INTERVAL DAY TO SECOND, ds INTERVAL HOUR) USING json",
    );
    assert_eq!(
        show_create(&s, "t3"),
        "CREATE TABLE spark_catalog.default.t3 (\n  \
         s STRUCT<a: STRUCT<b: ARRAY<MAP<STRING COLLATE UTF8_BINARY, STRUCT<c: FLOAT>>>>>,\n  \
         ym INTERVAL YEAR TO MONTH,\n  dt INTERVAL DAY TO SECOND,\n  ds INTERVAL HOUR)\nUSING json\n"
    );
}

#[test]
fn nested_char_and_varchar_keep_their_lengths() {
    let s = Session::new().unwrap();
    run(
        &s,
        "CREATE TABLE t4 (s STRUCT<a: VARCHAR(3), b: CHAR(2)>, v ARRAY<VARCHAR(4)>) USING parquet",
    );
    assert_eq!(
        show_create(&s, "t4"),
        "CREATE TABLE spark_catalog.default.t4 (\n  \
         s STRUCT<a: VARCHAR(3) COLLATE UTF8_BINARY, b: CHAR(2) COLLATE UTF8_BINARY>,\n  \
         v ARRAY<VARCHAR(4) COLLATE UTF8_BINARY>)\nUSING parquet\n"
    );
}

#[test]
fn nested_field_names_are_quoted_unless_identifiers() {
    let s = Session::new().unwrap();
    run(
        &s,
        "CREATE TABLE t6 (`Weird``n` STRUCT<`a``b`: INT, `select`: INT, `_x`: INT, `X`: INT, `9a`: INT, \
         `a-b`: INT, `é`: INT, `it's`: STRING COMMENT 'it\\'s \\\\ x'>) USING parquet",
    );
    assert_eq!(
        show_create(&s, "t6"),
        "CREATE TABLE spark_catalog.default.t6 (\n  \
         `Weird``n` STRUCT<`a``b`: INT, select: INT, _x: INT, X: INT, `9a`: INT, `a-b`: INT, `é`: INT, \
         `it's`: STRING COLLATE UTF8_BINARY COMMENT 'it\\'s \\ x'>)\nUSING parquet\n"
    );
}

/// A struct built in the query (`named_struct`) has non-nullable fields, which a stored table makes nullable
/// (Spark prints `s STRUCT<x: INT>`); that case comes with the stored-table nullability fix.
#[test]
fn ctas_columns_print_their_inferred_nested_types() {
    let s = Session::new().unwrap();
    run(
        &s,
        "CREATE TABLE t2 USING parquet AS SELECT array(1) a, map(1,'x') m, array(array(1.5d)) d",
    );
    assert_eq!(
        show_create(&s, "t2"),
        "CREATE TABLE spark_catalog.default.t2 (\n  a ARRAY<INT>,\n  \
         m MAP<INT, STRING COLLATE UTF8_BINARY>,\n  d ARRAY<ARRAY<DOUBLE>>)\nUSING parquet\n"
    );
    run(
        &s,
        "CREATE TABLE t8 USING parquet AS SELECT array('a') a, \
         map('k', timestamp'2020-01-01 00:00:00') m",
    );
    assert_eq!(
        show_create(&s, "t8"),
        "CREATE TABLE spark_catalog.default.t8 (\n  a ARRAY<STRING COLLATE UTF8_BINARY>,\n  \
         m MAP<STRING COLLATE UTF8_BINARY, TIMESTAMP>)\nUSING parquet\n"
    );
}

#[test]
fn a_column_name_starting_with_a_digit_is_quoted() {
    let s = Session::new().unwrap();
    run(&s, "CREATE TABLE t9 (`9a` INT, `select` INT) USING parquet");
    assert_eq!(
        show_create(&s, "t9"),
        "CREATE TABLE spark_catalog.default.t9 (\n  `9a` INT,\n  select INT)\nUSING parquet\n"
    );
}
