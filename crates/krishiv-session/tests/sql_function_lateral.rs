// SPDX-License-Identifier: Apache-2.0
//! SQL-bodied functions (T184) as Spark 4.2.0 runs them in `sql-udf.sql`: a
//! table function's arguments read the left side of a `LATERAL` join and its
//! output is qualified by its name; `STRUCT(a, b)` over parameters names the
//! fields after them; a call whose body is its parameter keeps the call's
//! name over a grouping key.
#![cfg(test)]

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
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
}
