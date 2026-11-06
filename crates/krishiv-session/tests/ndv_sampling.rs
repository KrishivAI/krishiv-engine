// SPDX-License-Identifier: Apache-2.0
//! Plan-time distinct-value samples (P2-T164 unit U6): an equality on a string column of plain Parquet files,
//! which store no distinct count, is estimated from a sample of the column instead of the default selectivity.
#![cfg(test)]

use arrow_array::Array as _;
use futures::executor::block_on;
use krishiv_session::Session;

/// The `rowCount` `EXPLAIN COST` gives the first `Filter` line of `sql`.
fn filter_rows(s: &Session, sql: &str) -> f64 {
    let text = block_on(async {
        let df = s.sql(&format!("EXPLAIN COST {sql}")).await.unwrap();
        let batches = df.collect().await.unwrap();
        let mut text = String::new();
        for b in &batches {
            use arrow_array::cast::AsArray as _;
            let c = b.column(0);
            for i in 0..c.len() {
                match c.as_string_opt::<i32>() {
                    Some(a) => text.push_str(a.value(i)),
                    None => text.push_str(c.as_string_view().value(i)),
                }
                text.push('\n');
            }
        }
        text
    });
    let line = text
        .lines()
        .find(|l| l.contains("Filter ("))
        .unwrap_or_else(|| panic!("no Filter in\n{text}"));
    let count = line
        .split("rowCount=")
        .nth(1)
        .and_then(|r| r.split(')').next())
        .unwrap_or_else(|| panic!("no rowCount in {line}"));
    count.parse().unwrap_or_else(|e| panic!("{count}: {e}"))
}

/// The file table `name` under `dir`, as a query reads it.
fn table(dir: &std::path::Path, name: &str) -> String {
    format!("parquet.`{}`", dir.join(name).display())
}

/// A session over two Parquet tables: `nation(n_name)` with 25 distinct names, and `orders(status)` of 20,000 rows
/// with three statuses.
fn session(dir: &std::path::Path, sampling: bool) -> Session {
    // The tables are read by path: `parquet.`<dir>/nation`` and `parquet.`<dir>/orders``.
    let s = Session::builder().build().unwrap();
    s.conf()
        .set(
            "krishiv.optimizer.ndvSampling.enabled",
            if sampling { "true" } else { "false" },
        )
        .unwrap();
    // Small budget: `orders` is sampled, not read whole; `nation` fits it.
    s.conf()
        .set("krishiv.optimizer.ndvSampling.rows", "4096")
        .unwrap();
    let nation = dir.join("nation");
    let orders = dir.join("orders");
    if !nation.exists() {
        let n =
            block_on(s.sql("SELECT concat('NATION', CAST(id AS STRING)) AS n_name FROM range(25)"))
                .unwrap();
        block_on(n.write().parquet(nation.to_str().unwrap())).unwrap();
        let o = block_on(s.sql(
            "SELECT id, CASE WHEN id % 50 = 0 THEN 'P' WHEN id % 2 = 0 THEN 'F' ELSE 'O' END AS status FROM range(20000)",
        ))
        .unwrap();
        block_on(o.write().parquet(orders.to_str().unwrap())).unwrap();
    }
    s
}

#[test]
fn an_equality_on_a_sampled_string_column_divides_by_its_distinct_count() {
    let dir = tempfile::tempdir().unwrap();
    let off = session(dir.path(), false);
    let on = session(dir.path(), true);
    let nation = format!(
        "SELECT * FROM {} WHERE n_name = 'NATION7'",
        table(dir.path(), "nation")
    );
    let orders = format!(
        "SELECT * FROM {} WHERE status = 'F'",
        table(dir.path(), "orders")
    );
    let (nation, orders) = (nation.as_str(), orders.as_str());
    // Without a distinct count, `col = literal` takes the default selectivity (1/5).
    assert_eq!(filter_rows(&off, nation), 5.0);
    assert_eq!(filter_rows(&off, orders), 4000.0);
    // Sampled: `nation` is read whole (exactly 25 names), `orders` from its first 4,096 rows (3 statuses, none
    // seen once, so the estimate stays 3).
    assert_eq!(filter_rows(&on, nation), 1.0);
    let rows = filter_rows(&on, orders);
    // `EXPLAIN COST` prints three significant digits.
    assert!((rows - 20_000.0 / 3.0).abs() < 70.0, "{rows}");
    // A second query reads the cached samples (same answer).
    assert_eq!(filter_rows(&on, nation), 1.0);
}

#[test]
fn only_compared_columns_without_a_distinct_count_are_sampled() {
    let dir = tempfile::tempdir().unwrap();
    let on = session(dir.path(), true);
    // `id` is integral: its footer range bounds its distinct count, so the sample does not change it; a range
    // predicate on `status` is not an equality and keeps its default.
    let orders = table(dir.path(), "orders");
    let by_id = filter_rows(&on, &format!("SELECT * FROM {orders} WHERE id = 7"));
    assert_eq!(by_id, 1.0);
    let range = filter_rows(&on, &format!("SELECT * FROM {orders} WHERE status > 'F'"));
    assert!(range > 1.0, "{range}");
}

#[test]
fn an_in_list_on_a_sampled_column_uses_its_distinct_count() {
    let dir = tempfile::tempdir().unwrap();
    let on = session(dir.path(), true);
    let orders = table(dir.path(), "orders");
    // Two of the three statuses: 2/3 of the 20,000 rows, not the default.
    let rows = filter_rows(
        &on,
        &format!("SELECT * FROM {orders} WHERE status IN ('F', 'O')"),
    );
    assert!((rows - 20_000.0 * 2.0 / 3.0).abs() < 140.0, "{rows}");
}

#[test]
fn a_sample_out_of_time_keeps_the_default_and_is_tried_again_with_more_time() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), true);
    let nation = format!(
        "SELECT * FROM {} WHERE n_name = 'NATION7'",
        table(dir.path(), "nation")
    );
    s.conf()
        .set("krishiv.optimizer.ndvSampling.timeout", "0s")
        .unwrap();
    assert_eq!(
        filter_rows(&s, &nation),
        5.0,
        "no time: the default selectivity"
    );
    assert_eq!(
        filter_rows(&s, &nation),
        5.0,
        "recorded: not tried again with no more time"
    );
    s.conf()
        .set("krishiv.optimizer.ndvSampling.timeout", "10s")
        .unwrap();
    assert_eq!(filter_rows(&s, &nation), 1.0, "more time: sampled");
}
