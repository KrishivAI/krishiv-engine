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
