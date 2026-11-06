// SPDX-License-Identifier: Apache-2.0
//! The multi-table insert `FROM source INSERT ... SELECT ... [INSERT ... SELECT ...]` (Hive syntax): each clause
//! writes its own rows of the shared source. Rows and errors are from a Spark 4.2.0 run.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn run(s: &Session, sql: &str) {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

fn class(s: &Session, sql: &str) -> String {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => panic!("{sql}: no error"),
        Err(e) => e.error_class().name().to_owned(),
    }
}

/// The rows of `sql` (one or two columns of INT/STRING), each as text joined by `,`.
fn rows(s: &Session, sql: &str) -> Vec<String> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| {
                    if c.is_null(i) {
                        return "NULL".to_owned();
                    }
                    match c.data_type() {
                        arrow_schema::DataType::Int32 => c
                            .as_primitive::<arrow_array::types::Int32Type>()
                            .value(i)
                            .to_string(),
                        arrow_schema::DataType::Int64 => c
                            .as_primitive::<arrow_array::types::Int64Type>()
                            .value(i)
                            .to_string(),
                        arrow_schema::DataType::Utf8View => c.as_string_view().value(i).to_owned(),
                        arrow_schema::DataType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
                        other => format!("<{other}>"),
                    }
                })
                .collect();
            out.push(cells.join(","));
        }
    }
    out
}

fn session() -> (Session, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::builder()
        .config("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .build()
        .unwrap();
    for ddl in [
        "CREATE TABLE a(level INT) USING csv",
        "CREATE TABLE b(level INT) USING csv",
        "CREATE TABLE c(x INT, y STRING) USING parquet",
        "CREATE TEMPORARY VIEW src AS SELECT * FROM VALUES (1, 'p'), (2, 'q'), (3, 'r') AS t(k, v)",
    ] {
        run(&s, ddl);
    }
    (s, dir)
}

#[test]
fn each_clause_writes_its_rows_of_the_source() {
    let (s, _dir) = session();
    run(
        &s,
        "FROM src INSERT INTO a SELECT k WHERE k > 1 INSERT INTO b SELECT k + 10 WHERE k < 3",
    );
    assert_eq!(rows(&s, "SELECT * FROM a ORDER BY 1"), ["2", "3"]);
    assert_eq!(rows(&s, "SELECT * FROM b ORDER BY 1"), ["11", "12"]);
    run(
        &s,
        "FROM src INSERT OVERWRITE TABLE a SELECT k INSERT INTO b SELECT k",
    );
    assert_eq!(rows(&s, "SELECT * FROM a ORDER BY 1"), ["1", "2", "3"]);
    assert_eq!(
        rows(&s, "SELECT * FROM b ORDER BY 1"),
        ["1", "2", "3", "11", "12"]
    );
}

#[test]
fn clauses_may_share_a_target_group_and_order() {
    let (s, _dir) = session();
    run(
        &s,
        "FROM src INSERT INTO c SELECT k, v INSERT INTO c SELECT k * 2, upper(v) WHERE k = 1",
    );
    assert_eq!(
        rows(&s, "SELECT * FROM c ORDER BY 1, 2"),
        ["1,p", "2,P", "2,q", "3,r"]
    );
    run(
        &s,
        "FROM src INSERT INTO a SELECT count(*) GROUP BY v INSERT INTO b SELECT max(k)",
    );
    assert_eq!(rows(&s, "SELECT * FROM a ORDER BY 1"), ["1", "1", "1"]);
    assert_eq!(rows(&s, "SELECT * FROM b ORDER BY 1"), ["3"]);
    run(
        &s,
        "FROM src INSERT INTO a SELECT k ORDER BY k LIMIT 1 INSERT INTO b SELECT k ORDER BY k DESC LIMIT 1",
    );
    assert_eq!(rows(&s, "SELECT * FROM b ORDER BY 1"), ["3", "3"]);
    // Target columns and a source alias.
    run(
        &s,
        "FROM src s INSERT INTO a (level) SELECT s.k WHERE s.k = 2",
    );
    assert_eq!(rows(&s, "SELECT count(*) FROM a WHERE level = 2"), ["1"]);
}

/// `cte-recursion` 52/53: a common table expression in front of the shared source.
#[test]
fn a_recursive_cte_feeds_every_clause() {
    let (s, _dir) = session();
    run(&s, "CREATE TABLE rt2(level INT) USING csv");
    run(
        &s,
        "WITH RECURSIVE r(level) AS (VALUES (0) UNION ALL SELECT level + 1 FROM r WHERE level < 9) \
         FROM r INSERT INTO rt2 SELECT * INSERT INTO rt2 SELECT *",
    );
    let got = rows(&s, "SELECT * FROM rt2 ORDER BY 1");
    let want: Vec<String> = (0..10)
        .flat_map(|i| [i.to_string(), i.to_string()])
        .collect();
    assert_eq!(got, want);
}

/// A clause that does not resolve fails the statement before any clause writes.
#[test]
fn a_clause_that_does_not_resolve_writes_nothing() {
    let (s, _dir) = session();
    run(
        &s,
        "CREATE TABLE bk(level INT) USING parquet CLUSTERED BY (level) INTO 2 BUCKETS",
    );
    for (sql, want) in [
        (
            "FROM src INSERT INTO a SELECT k INSERT INTO nope SELECT k",
            "TABLE_OR_VIEW_NOT_FOUND",
        ),
        (
            "FROM src INSERT INTO a SELECT k INSERT INTO b SELECT zz",
            "UNRESOLVED_COLUMN.WITH_SUGGESTION",
        ),
        (
            "FROM src INSERT INTO a SELECT k INSERT INTO b SELECT k, v",
            "INSERT_COLUMN_ARITY_MISMATCH.TOO_MANY_DATA_COLUMNS",
        ),
        // Every target resolves before any clause's columns.
        (
            "FROM src INSERT INTO a SELECT zz INSERT INTO nope SELECT k",
            "TABLE_OR_VIEW_NOT_FOUND",
        ),
        (
            "FROM src INSERT INTO a SELECT k, k INSERT INTO b SELECT zz",
            "INSERT_COLUMN_ARITY_MISMATCH.TOO_MANY_DATA_COLUMNS",
        ),
        // A write the engine refuses before writing (Spark writes bucketed tables; Krishiv does not
        // yet) refuses the statement before an earlier clause writes.
        (
            "FROM src INSERT INTO a SELECT k INSERT INTO bk SELECT k",
            "KRISHIV_UNSUPPORTED_FEATURE",
        ),
    ] {
        assert_eq!(class(&s, sql), want, "{sql}");
        assert_eq!(rows(&s, "SELECT count(*) FROM a"), ["0"], "{sql}");
    }
}

/// Spark 4.2.0: the clauses run in order, each reading the source as it is when it runs, so a clause after one that
/// wrote the source table reads the new rows; of two overwrites of one table the last wins.
#[test]
fn clauses_run_in_order() {
    let (s, _dir) = session();
    run(&s, "INSERT INTO a VALUES (1), (2)");
    run(&s, "FROM a INSERT INTO a SELECT * INSERT INTO b SELECT *");
    assert_eq!(rows(&s, "SELECT * FROM a ORDER BY 1"), ["1", "1", "2", "2"]);
    assert_eq!(rows(&s, "SELECT * FROM b ORDER BY 1"), ["1", "1", "2", "2"]);
    run(
        &s,
        "FROM src INSERT OVERWRITE TABLE c SELECT k, v WHERE k = 1 \
         INSERT OVERWRITE TABLE c SELECT k, v WHERE k > 1",
    );
    assert_eq!(rows(&s, "SELECT x FROM c ORDER BY 1"), ["2", "3"]);
    run(
        &s,
        "FROM src INSERT OVERWRITE TABLE c SELECT k, v WHERE k = 1 INSERT INTO c SELECT k + 10, v",
    );
    assert_eq!(
        rows(&s, "SELECT x FROM c ORDER BY 1"),
        ["1", "11", "12", "13"]
    );
    run(
        &s,
        "FROM src INSERT INTO c SELECT k + 20, v INSERT OVERWRITE TABLE c SELECT k, v WHERE k = 3",
    );
    assert_eq!(rows(&s, "SELECT x FROM c ORDER BY 1"), ["3"]);
}
