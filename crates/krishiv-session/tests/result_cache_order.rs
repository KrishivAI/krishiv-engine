// SPDX-License-Identifier: Apache-2.0
//! The result cache returns rows in the order an uncached run does (Comet 1.2.0 correctness
//! check 2, over P2-T094): with and without `ORDER BY`, and under `krishiv.deterministicOrder`.
//!
//! A hit replays the stored batches, so it is the order of the run that filled the entry. What
//! that order is promised to be depends on the query: a total `ORDER BY` is the same on every
//! run, so uncached, fill and hit all agree; an unordered query is the same on every run only
//! under `krishiv.deterministicOrder`, so there the three agree too; without it a hit must equal
//! the fill run and hold the uncached run's rows as a multiset.
#![cfg(test)]

use arrow_array::RecordBatch;
use futures::executor::block_on;
use krishiv_session::Session;

fn run(s: &Session, sql: &str) -> Vec<RecordBatch> {
    block_on(async { s.sql(sql).await.unwrap().collect().await.unwrap() })
}

/// The batches as text, in the order they came, batch boundaries included. Every row is written out:
/// the `Debug` of a whole array elides its middle elements, so a reorder there would not show.
fn shape(batches: &[RecordBatch]) -> String {
    batches
        .iter()
        .map(|b| flat(std::slice::from_ref(b)).join(";"))
        .collect::<Vec<_>>()
        .join("|")
}

/// The rows as lines, in the order they came (batch boundaries do not show).
fn flat(batches: &[RecordBatch]) -> Vec<String> {
    let mut lines = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            lines.push(format!("{:?}", b.slice(r, 1).columns()));
        }
    }
    lines
}

/// The rows as sorted lines, for comparing contents without the order.
fn multiset(batches: &[RecordBatch]) -> Vec<String> {
    let mut lines = flat(batches);
    lines.sort();
    lines
}

/// What the `== Result Cache ==` section of EXPLAIN says about `sql`.
fn verdict(s: &Session, sql: &str) -> String {
    let text = format!("{:?}", run(s, &format!("EXPLAIN {sql}")));
    let section = text.split("== Result Cache ==").nth(1).unwrap_or("");
    section.split("\\n").nth(1).unwrap_or("").trim().to_owned()
}

fn session(deterministic: bool) -> Session {
    let s = Session::new().unwrap();
    if deterministic {
        s.conf().set("krishiv.deterministicOrder", "true").unwrap();
    }
    run(
        &s,
        "CREATE TABLE d USING parquet AS \
         SELECT id, id % 7 AS g, CAST(id AS DECIMAL(20,2)) AS m FROM range(60000)",
    );
    s
}

/// Queries whose order is fixed by an `ORDER BY` on a unique key.
const ORDERED: [&str; 3] = [
    "SELECT id, m FROM d WHERE id % 3 = 0 ORDER BY id",
    "SELECT id, g FROM d ORDER BY id DESC LIMIT 5000",
    "SELECT g, sum(m) AS s FROM d GROUP BY g ORDER BY g",
];

/// Queries with no total order: no `ORDER BY`, a partial one (ties), a join and a union.
const UNORDERED: [&str; 5] = [
    "SELECT id, m FROM d WHERE id % 3 = 0",
    "SELECT id, g FROM d ORDER BY g",
    "SELECT g, sum(m), count(*) FROM d GROUP BY g",
    "SELECT a.id, b.g FROM d a JOIN d b ON a.id = b.id WHERE a.id < 5000",
    "SELECT id FROM d WHERE id % 5 = 0 UNION ALL SELECT id FROM d WHERE id % 7 = 0",
];

/// Runs `sql` uncached, then as the fill run, then as a hit; checks that the second run really
/// was a miss and the third a hit.
fn three_runs(s: &Session, sql: &str) -> (Vec<RecordBatch>, Vec<RecordBatch>, Vec<RecordBatch>) {
    s.conf()
        .set("krishiv.resultCache.enabled", "false")
        .unwrap();
    let uncached = run(s, sql);
    s.conf().set("krishiv.resultCache.enabled", "true").unwrap();
    let before = verdict(s, sql);
    assert!(
        before.starts_with("miss"),
        "`{sql}` before the fill run: {before}"
    );
    let fill = run(s, sql);
    let after = verdict(s, sql);
    assert!(
        after.starts_with("hit"),
        "`{sql}` after the fill run: {after}"
    );
    let hit = run(s, sql);
    (uncached, fill, hit)
}

#[test]
fn a_total_order_by_comes_back_in_the_same_order_from_the_cache() {
    for deterministic in [false, true] {
        let s = session(deterministic);
        for sql in ORDERED {
            let (uncached, fill, hit) = three_runs(&s, sql);
            assert_eq!(
                multiset(&uncached),
                multiset(&hit),
                "`{sql}` (deterministic: {deterministic}): the hit holds other rows"
            );
            assert_eq!(
                flat(&uncached),
                flat(&fill),
                "`{sql}`: fill differs from uncached"
            );
            assert_eq!(
                flat(&uncached),
                flat(&hit),
                "`{sql}`: hit differs from uncached"
            );
        }
    }
}

#[test]
fn under_deterministic_order_unordered_queries_hit_byte_identically() {
    let s = session(true);
    for sql in UNORDERED.into_iter().chain(ORDERED) {
        let (uncached, fill, hit) = three_runs(&s, sql);
        assert_eq!(
            shape(&uncached),
            shape(&fill),
            "`{sql}`: fill differs from uncached"
        );
        assert_eq!(
            shape(&fill),
            shape(&hit),
            "`{sql}`: hit differs from the fill run"
        );
    }
}

#[test]
fn without_deterministic_order_a_hit_replays_the_fill_run() {
    let s = session(false);
    for sql in UNORDERED {
        let (uncached, fill, hit) = three_runs(&s, sql);
        assert_eq!(
            shape(&fill),
            shape(&hit),
            "`{sql}`: hit differs from the fill run"
        );
        assert_eq!(
            multiset(&uncached),
            multiset(&hit),
            "`{sql}`: the hit holds other rows than an uncached run"
        );
        // Another hit is the same bytes again.
        assert_eq!(
            shape(&hit),
            shape(&run(&s, sql)),
            "`{sql}`: a second hit differs"
        );
    }
}
