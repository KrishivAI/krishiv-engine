// SPDX-License-Identifier: Apache-2.0
//! `max_by(x, y, k)` / `min_by(x, y, k)` over a window. Expected rows are Spark 4.2.0's, from the `max-min-by-k`
//! golden (statement 22).
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn run(s: &Session, sql: &str) -> Vec<String> {
    let batches = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let (top, low) = (b.column(0).as_string_view(), b.column(1).as_string_view());
        out.extend((0..b.num_rows()).map(|i| format!("{} | {}", top.value(i), low.value(i))));
    }
    out
}

#[test]
fn the_k_best_over_a_partition_window_match_spark() {
    let d = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", d.path().to_str().unwrap())
        .unwrap();
    for sql in [
        "create table dept_data (dept string, emp string, salary int) using parquet",
        "insert into dept_data values ('Eng','Alice',120000),('Eng','Bob',95000),('Eng','Carol',110000),\
         ('Sales','Dave',80000),('Sales','Eve',75000),('Sales','Frank',85000)",
    ] {
        block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    }
    let rows = run(
        &s,
        "SELECT cast(max_by(emp, salary, 2) OVER (PARTITION BY dept) as string), \
         cast(min_by(emp, salary, 2) OVER (PARTITION BY dept) as string) \
         FROM dept_data ORDER BY dept, emp",
    );
    assert_eq!(
        rows,
        [
            "[Alice, Carol] | [Bob, Carol]",
            "[Alice, Carol] | [Bob, Carol]",
            "[Alice, Carol] | [Bob, Carol]",
            "[Frank, Dave] | [Eve, Dave]",
            "[Frank, Dave] | [Eve, Dave]",
            "[Frank, Dave] | [Eve, Dave]",
        ]
    );
}

#[test]
fn the_unaliased_column_is_named_as_written() {
    let s = Session::new().unwrap();
    let df = block_on(async {
        s.sql(
            "select max_by(c, a, 2) over (partition by b) from values (1, 'x', 'p') as t(a, b, c)",
        )
        .await
    })
    .unwrap();
    let schema = block_on(df.schema()).unwrap();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    assert_eq!(
        names,
        [
            "max_by(c, a, 2) OVER (PARTITION BY b ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)"
        ]
    );
}

#[test]
fn an_upper_case_call_is_named_like_the_lower_case_one() {
    let s = Session::new().unwrap();
    let df = block_on(async {
        s.sql(
            "select MAX_BY(c, a, 2) over (partition by b) from values (1, 'x', 'p') as t(a, b, c)",
        )
        .await
    })
    .unwrap();
    let schema = block_on(df.schema()).unwrap();
    assert!(
        schema.fields()[0]
            .name()
            .starts_with("max_by(c, a, 2) OVER ("),
        "{:?}",
        schema.fields()[0].name()
    );
}

#[test]
fn a_map_payload_over_a_window_sorts_by_the_key_as_the_aggregate_does() {
    let s = Session::new().unwrap();
    let batches = block_on(async {
        s.sql(
            "select cast(max_by(m, y, 2) over (partition by 1) as string) \
             from values (map('a', 1), 3), (map('b', 2), 1), (map('c', 3), 2) as t(m, y)",
        )
        .await?
        .collect()
        .await
    })
    .unwrap();
    let col = batches[0].column(0).as_string_view();
    assert_eq!(col.value(0), "[{a -> 1}, {c -> 3}]");
}
