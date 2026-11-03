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

#[test]
fn order_by_and_having_match_a_grouped_struct_field() {
    let s = Session::new().unwrap();
    let from = "FROM (SELECT named_struct('a', 1) s) t";
    assert_eq!(
        count(
            &s,
            &format!("SELECT s.a, count(*) {from} GROUP BY s.a ORDER BY s.a")
        ),
        1
    );
    assert_eq!(
        count(
            &s,
            &format!("SELECT count(*) {from} GROUP BY s.a HAVING s.a > 0")
        ),
        1
    );
}

#[test]
fn order_by_a_metadata_field_after_group_by_all() {
    let s = session();
    let q = "SELECT _metadata.file_name, count(*) FROM p GROUP BY ALL ORDER BY _metadata.file_name";
    assert_eq!(count(&s, q), 1);
}

#[test]
fn a_struct_column_of_the_name_is_a_table_column_not_the_hidden_one() {
    let s = Session::new().unwrap();
    for sql in [
        "CREATE TABLE r(id INT, _metadata STRUCT<file_path: STRING, x: INT>) USING PARQUET",
        "INSERT INTO r VALUES (1, named_struct('file_path', 'mine', 'x', 7))",
    ] {
        block_on(s.sql(sql)).unwrap();
    }
    assert_eq!(names(&s, "SELECT * FROM r"), ["id", "_metadata"]);
    assert_eq!(
        count(&s, "SELECT _metadata.x FROM r WHERE _metadata.x = 7"),
        1
    );
}

#[test]
fn a_late_fetched_sort_limit_can_filter_on_the_hidden_column() {
    let s = session();
    let sql = "SELECT name FROM p WHERE _metadata.file_size > 0 ORDER BY id LIMIT 1";
    block_on(s.sql("SET krishiv.exec.lateFetch.maxRows = 100")).unwrap();
    assert_eq!(count(&s, sql), 1);
}

#[test]
fn a_table_column_shaped_like_the_hidden_one_is_still_a_table_column() {
    let s = Session::new().unwrap();
    for sql in [
        "CREATE TABLE q(id INT, _metadata STRUCT<file_path: STRING>) USING PARQUET",
        "INSERT INTO q VALUES (1, named_struct('file_path', 'mine'))",
    ] {
        block_on(s.sql(sql)).unwrap();
    }
    assert_eq!(
        count(&s, "SELECT id FROM q WHERE _metadata.file_path = 'mine'"),
        1
    );
}

#[test]
fn joins_read_the_hidden_column_in_on_and_using() {
    let s = session();
    assert_eq!(
        count(
            &s,
            "SELECT p.id FROM p JOIN c ON p._metadata.file_name IS NOT NULL AND c._metadata.file_name IS NOT NULL AND p.id = c.id"
        ),
        1
    );
    assert_eq!(
        count(
            &s,
            "SELECT p.id FROM p JOIN c ON p._metadata.file_size >= 0 AND c._metadata.file_size >= 0"
        ),
        2
    );
    assert_eq!(
        names(&s, "SELECT * FROM p JOIN c ON p._metadata.file_size >= 0"),
        ["id", "name", "id", "v"]
    );
    assert_eq!(
        count(&s, "SELECT p._metadata.file_name FROM p JOIN c USING (id)"),
        1
    );
    assert_eq!(
        count(&s, "SELECT c._metadata.file_name FROM p NATURAL JOIN c"),
        1
    );
}

#[test]
fn a_qualifier_naming_a_table_wins_over_a_struct_column_in_nested_except() {
    let s = Session::new().unwrap();
    assert_eq!(
        names(
            &s,
            "SELECT t.* EXCEPT (b) FROM (SELECT named_struct('a', 1, 'b', 2) AS t, 3 AS b) t"
        ),
        ["t"]
    );
}

#[test]
fn struct_expansion_of_the_hidden_column_takes_except() {
    let s = session();
    assert_eq!(
        names(
            &s,
            "SELECT _metadata.* EXCEPT (file_size, row_index) FROM p"
        ),
        [
            "file_path",
            "file_name",
            "file_block_start",
            "file_block_length",
            "file_modification_time"
        ]
    );
}

#[test]
fn a_table_column_named_metadata_merged_by_using_stays_in_a_qualified_star() {
    let s = session();
    assert_eq!(
        names(
            &s,
            "SELECT r.* FROM (SELECT 1 AS _metadata) l JOIN (SELECT 1 AS _metadata, 2 AS x) r USING (_metadata)"
        ),
        ["_metadata", "x"]
    );
}

#[test]
fn chained_using_joins_keep_the_metadata_of_every_input() {
    let s = session();
    block_on(s.sql("CREATE TABLE d(id INT, w STRING) USING PARQUET")).unwrap();
    block_on(s.sql("INSERT INTO d VALUES (1, 'z')")).unwrap();
    assert_eq!(
        count(
            &s,
            "SELECT p._metadata.file_name FROM p JOIN c USING (id) JOIN d USING (id)"
        ),
        1
    );
    assert_eq!(
        count(
            &s,
            "SELECT d._metadata.file_name FROM p JOIN c USING (id) JOIN d USING (id)"
        ),
        1
    );
}
