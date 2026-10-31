// SPDX-License-Identifier: Apache-2.0
//! The boundary between pipe operators (Spark's `PipeOperator`): `WHERE` and `ORDER BY` after `|>` read only the
//! columns the operator before them output. Expected results and errors are from a Spark 4.2.0 run.
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::new().unwrap();
    for v in [
        "CREATE TEMPORARY VIEW t AS SELECT * FROM VALUES (10, 2) AS t(x, y)",
        "CREATE TEMPORARY VIEW o AS SELECT * FROM VALUES (1), (2) AS o(y)",
    ] {
        block_on(async { s.sql(v).await?.collect().await }).unwrap();
    }
    s
}

fn class(s: &Session, sql: &str) -> String {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => panic!("{sql}: no error"),
        Err(e) => e.error_class().name().to_owned(),
    }
}

/// A column dropped by `SELECT` cannot be read by the next `WHERE` or `ORDER BY`.
#[test]
fn a_dropped_column_is_not_visible_to_the_next_operator() {
    let s = session();
    for sql in [
        "TABLE t |> SELECT x, length(y) AS z |> WHERE x + length(y) < 4",
        "TABLE t |> SELECT x, length(y) AS z |> ORDER BY x, y",
        "(SELECT x, sum(length(y)) AS sum_len FROM t GROUP BY x) |> WHERE sum(length(y)) = 3",
    ] {
        assert_eq!(class(&s, sql), "UNRESOLVED_COLUMN.WITH_SUGGESTION", "{sql}");
    }
}

/// Inside `EXISTS`, `y` is the outer query's, not the dropped `t.y`.
#[test]
fn a_dropped_column_resolves_to_the_enclosing_query() {
    let s = session();
    let batches = block_on(async {
        s.sql("SELECT * FROM o WHERE EXISTS (TABLE t |> SELECT x |> WHERE y = 1)")
            .await?
            .collect()
            .await
    })
    .unwrap();
    let rows: Vec<i32> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_primitive::<arrow_array::types::Int32Type>()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(rows, [1]);
}

/// A column the previous operator output is still visible.
#[test]
fn an_output_column_stays_visible() {
    let s = session();
    block_on(async {
        s.sql("TABLE t |> SELECT x, length(y) AS z |> WHERE x + z < 40 |> ORDER BY z")
            .await?
            .collect()
            .await
    })
    .unwrap();
}
