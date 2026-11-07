// SPDX-License-Identifier: Apache-2.0
//! SQL-bodied functions (T184) as Spark 4.2.0 runs them in `sql-udf.sql`: a
//! table function's arguments read the left side of a `LATERAL` join and its
//! output is qualified by its name; `STRUCT(a, b)` over parameters names the
//! fields after them; a call whose body is its parameter keeps the call's
//! name over a grouping key.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, RecordBatch};
use futures::executor::block_on;
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::new().unwrap();
    for sql in [
        "CREATE TEMPORARY VIEW t1(c1, c2) AS VALUES (0, 1), (0, 2), (1, 2)",
        "CREATE TEMPORARY VIEW t2(c1, c2) AS VALUES (0, 2), (0, 3)",
        "CREATE TEMPORARY FUNCTION f1(x INT) RETURNS TABLE (a INT) RETURN SELECT x",
        "CREATE TEMPORARY FUNCTION f2(x INT) RETURNS TABLE (a INT) RETURN SELECT c2 FROM t2 WHERE c1 = x",
        "CREATE TEMPORARY FUNCTION s(a INT, b STRING) RETURN STRUCT(a, b)",
        "CREATE TEMPORARY FUNCTION id(x INT) RETURNS INT RETURN x",
    ] {
        block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    }
    s
}

fn run(s: &Session, sql: &str) -> Vec<RecordBatch> {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap()
}

fn ints(batches: &[RecordBatch]) -> Vec<Vec<i32>> {
    let mut rows = vec![];
    for b in batches {
        for i in 0..b.num_rows() {
            rows.push(
                (0..b.num_columns())
                    .map(|c| b.column(c).as_primitive::<Int32Type>().value(i))
                    .collect(),
            );
        }
    }
    rows.sort();
    rows
}

#[test]
fn lateral_table_functions_read_the_left_side() {
    let s = session();
    let out = run(&s, "SELECT * FROM t1, LATERAL f1(c1)");
    assert_eq!(
        ints(&out),
        vec![vec![0, 1, 0], vec![0, 2, 0], vec![1, 2, 1]]
    );
    let out = run(&s, "SELECT * FROM t1, LATERAL f2(c1)");
    assert_eq!(
        ints(&out),
        vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 2], vec![0, 2, 3]]
    );
    // The output is qualified by the function's name.
    let out = run(&s, "SELECT f1.a FROM t1, LATERAL f1(c1)");
    assert_eq!(ints(&out), vec![vec![0], vec![0], vec![1]]);
}

#[test]
fn calls_are_named_as_spark_names_them() {
    let s = session();
    let df = block_on(s.sql("SELECT s(1, 'test')")).unwrap();
    let schema = block_on(df.schema()).unwrap();
    assert_eq!(
        format!("{:?}", schema.field(0).data_type())
            .matches("name: \"a\"")
            .count()
            + format!("{:?}", schema.field(0).data_type())
                .matches("name: \"b\"")
                .count(),
        2,
        "{schema:?}"
    );
    let df = block_on(s.sql("SELECT c1, id(c1) FROM t1 GROUP BY c1")).unwrap();
    let schema = block_on(df.schema()).unwrap();
    assert_eq!(schema.field(1).name(), "id(c1)");
    // Spark 4.2.0 names `grouping` over the call as written.
    let df = block_on(
        s.sql("SELECT id(c1), c2, GROUPING(id(c1)), SUM(c1) FROM t1 GROUP BY ROLLUP(id(c1), c2)"),
    )
    .unwrap();
    let schema = block_on(df.schema()).unwrap();
    assert_eq!(schema.field(2).name(), "grouping(id(c1))");
}

/// `x IN (one row of values)` over a column stays refused: Spark answers it
/// two-valued (a NULL `x` or value gives false, and `NOT` over it is
/// null-aware), which an equality would not (review of #505, checked on Spark
/// 4.2.0). A constant NULL is folded first, to NULL, as Spark's
/// `NullPropagation` does.
#[test]
fn in_over_one_row_of_values_is_refused() {
    let s = session();
    for sql in [
        "SELECT c1 IN (SELECT 1) FROM t1",
        "SELECT coalesce(c1 IN (SELECT 1), true) FROM t1",
        "SELECT c1 FROM t1 WHERE (c1 IN (SELECT 1)) IS NULL",
        "SELECT NOT (c1 IN (SELECT 1)) FROM t1",
    ] {
        let e = block_on(async { s.sql(sql).await?.collect().await }).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{sql}: {e}"
        );
    }
    let out = run(&s, "SELECT CAST(NULL AS INT) IN (SELECT 1)");
    assert!(out[0].column(0).is_null(0));
}

/// Spark 4.2.0 (`sql-udf.sql` 772-787): a table function's body keeps the
/// ANSI mode it was created under, as a scalar function's does.
#[test]
fn table_functions_keep_their_creation_ansi_mode() {
    let s = Session::new().unwrap();
    for sql in [
        "SET spark.sql.ansi.enabled=false",
        "CREATE TEMPORARY FUNCTION ct() RETURNS TABLE (a INT) RETURN SELECT CAST('a' AS INT)",
        "CREATE TEMPORARY FUNCTION dt() RETURNS TABLE (a INT) RETURN SELECT 'a' + 1",
        "SET spark.sql.ansi.enabled=true",
    ] {
        run(&s, sql);
    }
    for f in ["ct", "dt"] {
        let out = run(&s, &format!("SELECT * FROM {f}()"));
        assert!(out[0].column(0).is_null(0), "{f}");
    }
}
