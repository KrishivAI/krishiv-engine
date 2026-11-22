// SPDX-License-Identifier: Apache-2.0
//! Grouping sets aggregate their input before the expansion (`krishiv-optimizer` `eager` module
//! `expand`): where it fires, the executor returns what the aggregate over the expanded rows
//! returns, and the calls it cannot move stay above the expansion.

use proptest::prelude::*;

use super::partial_aggregate::{Row, physical, runtime, sorted, tables};
use super::run;
use super::tpch::{Order, Tpch};

/// Whether `q`'s statistics-ordered plan aggregates below its expansion.
fn fires(t: &Tpch, q: &str) -> bool {
    let plan = t.optimized(q, Order::Statistics).unwrap();
    let text = plan.root().to_string();
    let lines: Vec<&str> = text.lines().collect();
    lines.windows(2).any(|w| {
        w[0].trim_start_matches([' ', ':', '+', '-', '|'])
            .starts_with("Expand")
            && w[1].contains("PartialAggregate")
    })
}

/// `q`'s rows with the statistics-ordered plan and without statistics, where the rule never runs.
fn both(t: &Tpch, q: &str) -> (Vec<String>, Vec<String>) {
    let on = t.optimized(q, Order::Statistics).unwrap();
    let off = t.optimized(q, Order::Written).unwrap();
    (
        sorted(&run(&physical(t, &on).unwrap()).unwrap()),
        sorted(&run(&physical(t, &off).unwrap()).unwrap()),
    )
}

/// 240 rows over keys 0 to 11, with NULLs in every other column.
fn many() -> Vec<Row> {
    (0..240_i64)
        .map(|n| {
            (
                n % 12,
                (n % 7 != 0).then(|| i32::try_from(n).unwrap()),
                (n % 5 != 0).then(|| i128::from(n) * 100),
                (n % 3 != 0).then(|| u8::try_from(n % 9).unwrap()),
            )
        })
        .collect()
}

const WEIGHTS: [i64; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];

const FIRES: [&str; 6] = [
    // Every call eager aggregation moves, over a ROLLUP of the table's own columns.
    "SELECT k, s, count(i), count(*), sum(i), avg(i), sum(d), avg(d), min(i), max(d), min(s), \
     max(s) FROM f GROUP BY ROLLUP(k, s)",
    // A grouping expression, the grouping id and `grouping`, `HAVING`.
    "SELECT k % 3, s, grouping_id(), grouping(s), sum(d), count(*) FROM f \
     GROUP BY CUBE(k % 3, s) HAVING count(*) > 2",
    // Equal sets stay separate rows; a grand total and a set of another column.
    "SELECT k, sum(i), max(s) FROM f GROUP BY GROUPING SETS ((k), (k), (), (s))",
    // A filtered dimension, which eager aggregation leaves alone: the expansion still aggregates
    // its input.
    "SELECT w.wk, f.s, sum(f.i), avg(f.d), count(*) FROM f JOIN w ON f.k = w.wk \
     WHERE w.wk < 6 GROUP BY ROLLUP(w.wk, f.s)",
    // An argument that is also a grouping column.
    "SELECT i, sum(i), max(i), count(i) FROM f GROUP BY ROLLUP(i)",
    // The rewrite of several distinct calls is an expansion too.
    "SELECT count(DISTINCT s), count(DISTINCT i), max(d) FROM f",
];

/// Each stays above the expansion, for the reason in its comment.
const STAYS: [&str; 8] = [
    // Summation order: a `bigint` buffer that can overflow.
    "SELECT k, sum(k) FROM f GROUP BY ROLLUP(k)",
    // Summation order: floating point.
    "SELECT k, sum(CAST(i AS DOUBLE)) FROM f GROUP BY ROLLUP(k)",
    // An argument that can raise.
    "SELECT k, sum(i + 1) FROM f GROUP BY ROLLUP(k)",
    // Not one accumulator per group.
    "SELECT k, count(DISTINCT i) FROM f GROUP BY ROLLUP(k)",
    "SELECT k, sum(i) FILTER (WHERE k > 2) FROM f GROUP BY ROLLUP(k)",
    // No key: a keyless partial aggregate yields a row over empty input.
    "SELECT sum(i), count(*) FROM f GROUP BY GROUPING SETS ((), (), ())",
    // The multi-distinct rewrite with a key (the regular calls aggregate in an aggregate of their
    // own between the expansion and the merge), or a `FILTER` on its regular call.
    "SELECT k, count(DISTINCT s), count(DISTINCT i), count(*), sum(d) FROM f GROUP BY k",
    "SELECT count(DISTINCT s), count(DISTINCT i), max(d) FILTER (WHERE k > 2) FROM f",
];

#[test]
fn the_rule_fires_and_returns_the_same_rows() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = tables(dir.path(), &many(), &WEIGHTS, &rt);
    for q in FIRES {
        assert!(
            fires(&t, q),
            "{q}\n{}",
            t.optimized(q, Order::Statistics).unwrap().root()
        );
        let (on, off) = both(&t, q);
        assert_eq!(on, off, "{q}");
    }
}

#[test]
fn the_calls_it_cannot_move_stay_above_the_expansion() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = tables(dir.path(), &many(), &WEIGHTS, &rt);
    for q in STAYS {
        assert!(!fires(&t, q), "{q}");
        let (on, off) = both(&t, q);
        assert_eq!(on, off, "{q}");
    }
}

/// An empty table: what the rule leaves answers as the plan without it, for the sets that fire
/// and the ones that stay (Spark goldens `postgreSQL/groupingsets` #16 and #44: only empty sets).
#[test]
fn an_empty_input_returns_the_same_rows() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = tables(dir.path(), &[], &WEIGHTS, &rt);
    for q in FIRES.iter().chain(&STAYS) {
        let (on, off) = both(&t, q);
        assert_eq!(on, off, "{q}");
    }
}

/// Keys with as many values as rows: the partial aggregate does not reduce and passes the rows on
/// as their own states, with the same answer.
#[test]
fn keys_that_do_not_group_return_the_same_rows() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let rows: Vec<Row> = (0..3000_i64)
        .map(|n| {
            (
                n,
                Some(i32::try_from(n % 17).unwrap()),
                Some(i128::from(n)),
                None,
            )
        })
        .collect();
    let t = tables(dir.path(), &rows, &WEIGHTS, &rt);
    let q = "SELECT k, k % 5, sum(i), avg(d), count(*) FROM f GROUP BY ROLLUP(k % 5, k)";
    assert!(fires(&t, q));
    let (on, off) = both(&t, q);
    assert_eq!(on, off);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]
    /// Whatever the rows: the rule's plan answers as the aggregate over the expanded rows does,
    /// firing or not.
    #[test]
    fn the_rule_never_changes_an_answer(
        rows in prop::collection::vec(
            (
                0i64..12,
                prop::option::of(-1000i32..1000),
                prop::option::of(-100_000i128..100_000),
                prop::option::of(0u8..9),
            ),
            0..400,
        ),
        keys in prop::collection::vec(0i64..16, 0..60),
    ) {
        let rt = runtime();
        let dir = tempfile::tempdir().unwrap();
        let t = tables(dir.path(), &rows, &keys, &rt);
        for q in FIRES.iter().chain(&STAYS) {
            let (on, off) = both(&t, q);
            prop_assert_eq!(on, off, "{}", q);
        }
    }
}
