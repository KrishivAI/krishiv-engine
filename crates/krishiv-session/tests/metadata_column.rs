// SPDX-License-Identifier: Apache-2.0
//! The hidden `_metadata` column of file-source tables (Spark's
//! `FileFormat` metadata): bare and qualified, left out of `*`, expandable,
//! usable in every clause and across joins, and absent from tables that are
//! not file sources.
#![cfg(test)]

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use futures::executor::block_on;
use krishiv_session::Session;

fn rows(s: &Session, sql: &str) -> Vec<RecordBatch> {
    block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap()
}

fn count(s: &Session, sql: &str) -> usize {
    rows(s, sql).iter().map(RecordBatch::num_rows).sum()
}

fn names(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.schema())
        .unwrap()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

fn session() -> Session {
    let s = Session::new().unwrap();
    for sql in [
        "CREATE TABLE p(id INT, name STRING) USING PARQUET",
        "INSERT INTO p VALUES (1, 'a'), (2, 'b')",
        "CREATE TABLE c(id INT, v STRING) USING CSV",
        "INSERT INTO c VALUES (1, 'x')",
    ] {
        block_on(s.sql(sql)).unwrap();
    }
    s
}

#[test]
fn the_column_is_hidden_from_star_and_read_by_name() {
    let s = session();
    assert_eq!(names(&s, "SELECT * FROM p"), ["id", "name"]);
    assert_eq!(
        names(&s, "SELECT d.* FROM p AS d, c WHERE d.id = c.id"),
        ["id", "name"]
    );
    assert_eq!(
        names(
            &s,
            "SELECT _metadata.file_name, p._metadata.file_size FROM p"
        ),
        ["file_name", "file_size"]
    );
    assert_eq!(
        names(&s, "SELECT _metadata.* FROM p"),
        [
            "file_path",
            "file_name",
            "file_size",
            "file_block_start",
            "file_block_length",
            "file_modification_time",
            "row_index"
        ]
    );
    // Only Parquet has row positions.
    assert_eq!(names(&s, "SELECT _metadata.* FROM c").len(), 6);
    assert!(block_on(s.sql("SELECT _metadata.row_index FROM c")).is_err());
}

#[test]
fn the_values_are_the_files() {
    let s = session();
    let b = rows(&s, "SELECT _metadata.row_index, id FROM p ORDER BY id");
    let idx = b[0].column(0).as_primitive::<Int64Type>();
    assert_eq!((idx.value(0), idx.value(1)), (0, 1));
    let b = rows(&s, "SELECT _metadata.file_path FROM p");
    let path = b[0].column(0).as_string_view().value(0).to_owned();
    assert!(
        path.starts_with("file:/") && path.ends_with(".parquet"),
        "{path}"
    );
}

#[test]
fn clauses_aliases_and_subqueries_reach_it() {
    let s = session();
    assert_eq!(
        count(&s, "SELECT * FROM p WHERE _metadata.row_index > 0"),
        1
    );
    assert_eq!(
        count(&s, "SELECT id FROM p ORDER BY _metadata.row_index DESC"),
        2
    );
    assert_eq!(count(&s, "SELECT x._metadata.file_name FROM p x"), 2);
    assert_eq!(
        count(
            &s,
            "SELECT t.file_name FROM (SELECT _metadata.file_name FROM p) t"
        ),
        2
    );
    assert_eq!(
        count(
            &s,
            "SELECT _metadata.file_name, count(*) FROM p GROUP BY _metadata.file_name"
        ),
        1
    );
    assert_eq!(count(&s, "SELECT max(_metadata.row_index) FROM p"), 1);
}

#[test]
fn joins_carry_both_sides() {
    let s = session();
    assert_eq!(
        count(
            &s,
            "SELECT p.id, p._metadata.row_index, c._metadata.file_size FROM p LEFT JOIN c ON p.id = c.id"
        ),
        2
    );
    assert_eq!(
        count(
            &s,
            "SELECT a.id FROM p a JOIN p b ON a.id = b.id WHERE a._metadata.row_index <= b._metadata.row_index"
        ),
        2
    );
    assert_eq!(
        count(
            &s,
            "SELECT p.id FROM p JOIN c ON p.id = c.id ORDER BY c._metadata.file_size"
        ),
        count(&s, "SELECT p.id FROM p JOIN c ON p.id = c.id")
    );
}

#[test]
fn tables_that_are_not_file_sources_and_columns_of_the_name_have_none() {
    let s = session();
    block_on(s.sql("CREATE TEMPORARY VIEW v AS SELECT 1 AS a")).unwrap();
    assert!(block_on(s.sql("SELECT _metadata FROM v")).is_err());
    block_on(s.sql("CREATE TABLE m(_metadata INT) USING PARQUET")).unwrap();
    block_on(s.sql("INSERT INTO m VALUES (5)")).unwrap();
    assert_eq!(names(&s, "SELECT * FROM m"), ["_metadata"]);
}

#[test]
fn a_struct_field_groups_and_expands() {
    let s = Session::new().unwrap();
    let q = "SELECT s.a, count(*) FROM (SELECT named_struct('a', 1) s) t GROUP BY s.a";
    assert_eq!(count(&s, q), 1);
    assert_eq!(
        names(
            &s,
            "SELECT s.* FROM (SELECT named_struct('a', 1, 'b', 'x') s) t"
        ),
        ["a", "b"]
    );
}
