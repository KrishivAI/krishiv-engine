// SPDX-License-Identifier: Apache-2.0
//! Discovered key facts (design P2b A5 (d), unit 2): over plain Parquet files,
//! which declare no key, `krishiv.optimizer.discoverKeys` lets the session
//! count the dimension's key once, so `PayloadAboveAggregate` fires; a
//! repeated key never fires it; the answers equal the rule off.
#![cfg(test)]
mod literal_path;
use literal_path::sql_path;

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
            sql_path(&dir.join(t))
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

/// A file changed after a count: the next query counts again and, over a
/// repeated key now, the rule no longer fires (review M6).
#[test]
fn a_changed_file_is_counted_again() {
    let dir = tempfile::tempdir().unwrap();
    write_tables(dir.path(), false);
    let on = session(dir.path(), true, true);
    assert!(moved(&on));
    // `customer` rewritten with key 5 twice.
    std::fs::remove_dir_all(dir.path().join("customer")).unwrap();
    std::fs::remove_dir_all(dir.path().join("orders")).unwrap();
    write_tables(dir.path(), true);
    let again = session(dir.path(), true, true);
    assert!(!moved(&again));
    let verbose = text(&again, &format!("EXPLAIN VERBOSE {QUERY}"));
    assert!(
        verbose.contains("key [c_custkey]: cached not unique")
            || verbose.contains("key [c_custkey]: discovered not unique"),
        "{verbose}"
    );
    assert_eq!(rows(&again), rows(&session(dir.path(), false, false)));
}

/// `REFRESH TABLE` and `CLEAR CACHE` forget the facts: the next count reads
/// again (review M1).
#[test]
fn refresh_and_clear_cache_forget_the_facts() {
    let dir = tempfile::tempdir().unwrap();
    write_tables(dir.path(), false);
    let on = session(dir.path(), true, true);
    let note = |s: &Session| {
        let verbose = text(s, &format!("EXPLAIN VERBOSE {QUERY}"));
        verbose
            .lines()
            .find(|l| l.contains("discoverKeys: customer"))
            .unwrap_or_else(|| panic!("no count in\n{verbose}"))
            .to_owned()
    };
    assert!(note(&on).contains("discovered unique"));
    assert!(note(&on).contains("cached unique"));
    block_on(on.sql("REFRESH TABLE customer")).unwrap();
    assert!(note(&on).contains("discovered unique"));
    block_on(on.sql("CLEAR CACHE")).unwrap();
    assert!(note(&on).contains("discovered unique"));
}

/// The count reserves from the session's pool: under a small
/// `krishiv.session.memoryLimit` the hash path is refused, the fact stays
/// unknown and the rule does not fire (review I1).
#[test]
fn a_count_past_the_session_memory_limit_proves_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let w = Session::builder().build().unwrap();
    // Written in descending order: the hash path.
    let c = block_on(w.sql(
        "SELECT 400001 - id AS c_custkey, concat('C', CAST(id AS STRING)) AS c_name \
         FROM range(1, 400001) ORDER BY c_custkey DESC",
    ))
    .unwrap();
    block_on(
        c.write()
            .parquet(dir.path().join("customer").to_str().unwrap()),
    )
    .unwrap();
    let o = block_on(w.sql(
        "SELECT id % 400000 + 1 AS o_custkey, CAST(id % 97 AS DECIMAL(12, 2)) AS o_price \
         FROM range(800000)",
    ))
    .unwrap();
    block_on(
        o.write()
            .parquet(dir.path().join("orders").to_str().unwrap()),
    )
    .unwrap();
    let s = Session::builder()
        .config("krishiv.session.memoryLimit", "3m")
        .build()
        .unwrap();
    for (k, v) in [
        ("krishiv.optimizer.payloadAboveAggregate", "true"),
        ("krishiv.optimizer.discoverKeys", "true"),
    ] {
        s.conf().set(k, v).unwrap();
    }
    for t in ["customer", "orders"] {
        block_on(s.sql(&format!(
            "CREATE TEMPORARY VIEW {t} USING parquet OPTIONS (path '{}')",
            sql_path(&dir.path().join(t))
        )))
        .unwrap();
    }
    let verbose = text(&s, &format!("EXPLAIN VERBOSE {QUERY}"));
    assert!(
        verbose.contains("key [c_custkey]: unknown, the pool refused the count's memory"),
        "{verbose}"
    );
    assert!(!moved(&s));
}

/// `REFRESH <path>` forgets the facts as `REFRESH TABLE` does (review N2).
#[test]
fn refresh_of_a_path_forgets_the_facts() {
    let dir = tempfile::tempdir().unwrap();
    write_tables(dir.path(), false);
    let on = session(dir.path(), true, true);
    let note = || {
        let verbose = text(&on, &format!("EXPLAIN VERBOSE {QUERY}"));
        verbose
            .lines()
            .find(|l| l.contains("discoverKeys: customer"))
            .unwrap_or_else(|| panic!("no count in\n{verbose}"))
            .to_owned()
    };
    assert!(note().contains("discovered unique"));
    assert!(note().contains("cached unique"));
    block_on(on.sql(&format!(
        "REFRESH '{}'",
        sql_path(&dir.path().join("customer"))
    )))
    .unwrap();
    assert!(note().contains("discovered unique"));
}
