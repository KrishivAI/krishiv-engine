// SPDX-License-Identifier: Apache-2.0
//! Discovered key facts (design P2b A5 (d), unit 2): over plain Parquet files,
//! which declare no key, `krishiv.optimizer.discoverKeys` lets the session
//! count the dimension's key once, so `PayloadAboveAggregate` fires; a
//! repeated key never fires it; the answers equal the rule off.
#![cfg(test)]

use arrow_array::Array as _;
use futures::executor::block_on;
use krishiv_session::Session;

const QUERY: &str = "SELECT c_custkey, c_name, sum(o_price) AS revenue, count(*) AS n \
                     FROM orders, customer WHERE o_custkey = c_custkey \
                     GROUP BY c_custkey, c_name";

/// Writes `customer(c_custkey, c_name)` (200 keys, written sorted; with
/// `dup`, key 5 once more under another name) and `orders(o_custkey,
/// o_price)` (2,000 rows over every key) under `dir`.
fn write_tables(dir: &std::path::Path, dup: bool) {
    let s = Session::builder().build().unwrap();
    let customer = if dup {
        "SELECT id AS c_custkey, concat('Customer#', CAST(id AS STRING)) AS c_name FROM range(1, 201) \
         UNION ALL SELECT CAST(5 AS BIGINT), 'Customer#again'"
    } else {
        "SELECT id AS c_custkey, concat('Customer#', CAST(id AS STRING)) AS c_name FROM range(1, 201)"
    };
    let c = block_on(s.sql(customer)).unwrap();
    block_on(c.write().parquet(dir.join("customer").to_str().unwrap())).unwrap();
    let o = block_on(s.sql(
        "SELECT id % 200 + 1 AS o_custkey, CAST(id % 97 AS DECIMAL(12, 2)) AS o_price FROM range(2000)",
    ))
    .unwrap();
    block_on(o.write().parquet(dir.join("orders").to_str().unwrap())).unwrap();
}

/// A session reading the tables as views, with the rule and the key counts
/// on or off.
fn session(dir: &std::path::Path, rule: bool, discover: bool) -> Session {
    let s = Session::builder().build().unwrap();
    let on = |b: bool| if b { "true" } else { "false" };
    s.conf()
        .set("krishiv.optimizer.payloadAboveAggregate", on(rule))
        .unwrap();
    s.conf()
        .set("krishiv.optimizer.discoverKeys", on(discover))
        .unwrap();
    for t in ["customer", "orders"] {
        block_on(s.sql(&format!(
            "CREATE TEMPORARY VIEW {t} USING parquet OPTIONS (path '{}')",
            dir.join(t).display()
        )))
        .unwrap();
    }
    s
}

fn text(s: &Session, sql: &str) -> String {
    block_on(async {
        let batches = s.sql(sql).await.unwrap().collect().await.unwrap();
        let mut out = String::new();
        for b in &batches {
            use arrow_array::cast::AsArray as _;
            let c = b.column(0);
            for i in 0..c.len() {
                match c.as_string_opt::<i32>() {
                    Some(a) => out.push_str(a.value(i)),
                    None => out.push_str(c.as_string_view().value(i)),
                }
                out.push('\n');
            }
        }
        out
    })
}

/// The query's rows, as sorted text.
fn rows(s: &Session) -> Vec<String> {
    let batches = block_on(async { s.sql(QUERY).await.unwrap().collect().await.unwrap() });
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            let row: Vec<String> = (0..b.num_columns())
                .map(|c| arrow_cast::display::array_value_to_string(b.column(c), i).unwrap())
                .collect();
            out.push(row.join("|"));
        }
    }
    out.sort();
    out
}

/// The optimized plan's aggregate groups by the orders' key alone.
fn moved(s: &Session) -> bool {
    let plan = text(s, &format!("EXPLAIN EXTENDED {QUERY}"));
    let optimized: String = plan
        .split("== Optimized Logical Plan ==")
        .nth(1)
        .and_then(|p| p.split("== Physical Plan ==").next())
        .unwrap_or_else(|| panic!("no optimized plan in\n{plan}"))
        .to_owned();
    let agg = optimized
        .lines()
        .find(|l| l.contains("Aggregate ["))
        .unwrap_or_else(|| panic!("no aggregate in\n{optimized}"));
    agg.contains("Aggregate [o_custkey#") && !agg.contains("c_name")
}

#[test]
fn a_discovered_key_moves_the_payload_above_the_aggregate() {
    let dir = tempfile::tempdir().unwrap();
    write_tables(dir.path(), false);
    let off = session(dir.path(), false, false);
    let rule_only = session(dir.path(), true, false);
    let on = session(dir.path(), true, true);
    assert!(!moved(&off));
    // Without a count the files prove no key.
    assert!(!moved(&rule_only));
    assert!(moved(&on));
    assert_eq!(rows(&on), rows(&off));
    assert_eq!(rows(&on).len(), 200);
    // EXPLAIN VERBOSE says what the count found; a later query reads the
    // cached fact.
    let verbose = text(&on, &format!("EXPLAIN VERBOSE {QUERY}"));
    assert!(
        verbose.contains("discoverKeys: customer: key [c_custkey]: cached unique"),
        "{verbose}"
    );
    assert!(
        verbose.contains("moved the join on [o_custkey = c_custkey]"),
        "{verbose}"
    );
}

#[test]
fn a_repeated_key_never_moves_and_the_answers_stay() {
    let dir = tempfile::tempdir().unwrap();
    write_tables(dir.path(), true);
    let off = session(dir.path(), false, false);
    let on = session(dir.path(), true, true);
    assert!(!moved(&on));
    assert_eq!(rows(&on), rows(&off));
    // Key 5 under two names: two groups.
    assert_eq!(rows(&on).len(), 201);
    let verbose = text(&on, &format!("EXPLAIN VERBOSE {QUERY}"));
    assert!(
        verbose.contains("key [c_custkey]: ") && verbose.contains("not unique"),
        "{verbose}"
    );
}

#[test]
fn with_the_rule_off_nothing_is_counted() {
    let dir = tempfile::tempdir().unwrap();
    write_tables(dir.path(), false);
    let s = session(dir.path(), false, true);
    let verbose = text(&s, &format!("EXPLAIN VERBOSE {QUERY}"));
    assert!(!verbose.contains("discoverKeys"), "{verbose}");
}
