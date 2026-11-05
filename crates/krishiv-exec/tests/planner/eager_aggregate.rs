// SPDX-License-Identifier: Apache-2.0
//! The eager aggregation rule (design P2-T167 unit 2): where it fires the plan
//! returns what the aggregate over the join returns, and each of the design's
//! semantics rules keeps a call above the join.

use proptest::prelude::*;

use super::partial_aggregate::{Row, physical, runtime, sorted, tables};
use super::run;
use super::tpch::{Order, Tpch};

/// Whether `q`'s statistics-ordered plan runs a partial aggregate below a join (the rule's effect). The
/// two-phase aggregate the physical planner puts above a join, under its exchange, is not the rule.
fn fires(t: &Tpch, q: &str) -> bool {
    let plan = t.optimized(q, Order::Statistics).unwrap();
    partial_below_join(&physical(t, &plan).unwrap().to_string())
}

/// Whether a `PartialHashAggregate` line sits in the subtree of a `Join` line of the plan's text tree.
fn partial_below_join(text: &str) -> bool {
    let indent = |l: &str| l.len() - l.trim_start_matches([' ', ':', '+', '-']).len();
    let lines: Vec<&str> = text.lines().collect();
    lines.iter().enumerate().any(|(i, l)| {
        l.contains("Join")
            && lines[i + 1..]
                .iter()
                .take_while(|c| indent(c) > indent(l))
                .any(|c| c.contains("PartialHashAggregate"))
    })
}

/// `q`'s rows with the rule's plan and without statistics, where it never runs.
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

/// Every key once, and two keys `f` does not hold.
const WEIGHTS: [i64; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];

const FIRES: [&str; 8] = [
    // Inner, keyed by the other side's column.
    "SELECT w.wk, count(f.i), count(*), sum(f.i), avg(f.i), sum(f.d), avg(f.d), min(f.i), \
     max(f.d), min(f.s), max(f.s) FROM f JOIN w ON f.k = w.wk GROUP BY w.wk",
    // An expression of the calls above the merge.
    "SELECT w.wk % 3, sum(f.i) + count(f.i), avg(f.d) FROM f JOIN w ON f.k = w.wk GROUP BY w.wk % 3",
    // The null-supplying side: an unmatched row is the empty state.
    "SELECT w.wk, count(f.i), sum(f.i), avg(f.i), sum(f.d), avg(f.d), min(f.i), max(f.d) \
     FROM w LEFT JOIN f ON w.wk = f.k GROUP BY w.wk",
    "SELECT w.wk, count(f.i), sum(f.i) FROM f RIGHT JOIN w ON f.k = w.wk GROUP BY w.wk",
    // The preserved side counts its rows.
    "SELECT f.k, count(*), sum(f.i) FROM f LEFT JOIN w ON f.k = w.wk GROUP BY f.k",
    // A global aggregate, one row even when the join is empty.
    "SELECT count(f.i), sum(f.i), avg(f.d), min(f.s) FROM f JOIN w ON f.k = w.wk",
    // An expansion above the join passes the states on.
    "SELECT w.wk, w.wk % 2, avg(f.i), sum(f.d), count(f.s) FROM f JOIN w ON f.k = w.wk \
     GROUP BY ROLLUP(w.wk, w.wk % 2)",
    "SELECT w.wk, grouping(w.wk), count(*), sum(f.i) FROM f JOIN w ON f.k = w.wk \
     GROUP BY CUBE(w.wk)",
];

/// Each stays above the join, for the reason in its comment.
const STAYS: [&str; 13] = [
    // An argument that can raise on a row the join would drop.
    "SELECT w.wk, sum(f.i + 1) FROM f JOIN w ON f.k = w.wk GROUP BY w.wk",
    // Summation order: floating point.
    "SELECT w.wk, sum(CAST(f.i AS DOUBLE)) FROM f JOIN w ON f.k = w.wk GROUP BY w.wk",
    "SELECT w.wk, avg(CAST(f.i AS DOUBLE)) FROM f JOIN w ON f.k = w.wk GROUP BY w.wk",
    // Summation order: a buffer that can overflow.
    "SELECT w.wk, sum(f.k) FROM f JOIN w ON f.k = w.wk GROUP BY w.wk",
    "SELECT w.wk, sum(CAST(f.d AS DECIMAL(30, 2))) FROM f JOIN w ON f.k = w.wk GROUP BY w.wk",
    // An unmatched row is not the empty state.
    "SELECT w.wk, count(*) FROM w LEFT JOIN f ON w.wk = f.k GROUP BY w.wk",
    "SELECT w.wk, count(coalesce(f.i, 1)) FROM w LEFT JOIN f ON w.wk = f.k GROUP BY w.wk",
    // Not one accumulator per group.
    "SELECT w.wk, count(DISTINCT f.i) FROM f JOIN w ON f.k = w.wk GROUP BY w.wk",
    "SELECT w.wk, sum(f.i) FILTER (WHERE f.k > 2) FROM f JOIN w ON f.k = w.wk GROUP BY w.wk",
    // Arguments from both sides.
    "SELECT sum(f.i), max(w.wk) FROM f JOIN w ON f.k = w.wk",
    // A filtered dimension: its runtime filter prunes the scan.
    "SELECT w.wk, sum(f.i) FROM f JOIN w ON f.k = w.wk WHERE w.wk < 6 GROUP BY w.wk",
    // A key whose distinct count is a guess: strings have none.
    "SELECT f.s, sum(f.i) FROM f JOIN w ON f.k = w.wk GROUP BY f.s",
    // Two keys beyond the join's.
    "SELECT f.i, f.s, sum(f.i) FROM f JOIN w ON f.k = w.wk GROUP BY f.i, f.s",
];

#[test]
fn the_rule_fires_and_returns_the_same_rows() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = tables(dir.path(), &many(), &WEIGHTS, &rt);
    for q in FIRES {
        assert!(fires(&t, q), "{q}");
        let (on, off) = both(&t, q);
        assert_eq!(on, off, "{q}");
    }
}

#[test]
fn each_semantics_rule_keeps_the_call_above_the_join() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = tables(dir.path(), &many(), &WEIGHTS, &rt);
    for q in STAYS {
        assert!(!fires(&t, q), "{q}");
        let (on, off) = both(&t, q);
        assert_eq!(on, off, "{q}");
    }
}

/// Unmatched rows of an outer join: `count` of `f` 0, the rest NULL, and
/// `count(*)`, which stays above the join, 1.
#[test]
fn unmatched_outer_rows_read_as_spark_reads_them() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = tables(dir.path(), &many(), &WEIGHTS, &rt);
    let q = "SELECT w.wk, count(f.i), sum(f.i), avg(f.d) FROM w LEFT JOIN f ON w.wk = f.k \
             WHERE w.wk > 11 GROUP BY w.wk";
    let (on, off) = both(&t, q);
    assert_eq!(on, off);
    assert_eq!(on, ["12|0|NULL|NULL", "13|0|NULL|NULL"]);
    let q = "SELECT w.wk, count(*) FROM w LEFT JOIN f ON w.wk = f.k WHERE w.wk > 11 GROUP BY w.wk";
    assert_eq!(both(&t, q).0, ["12|1", "13|1"]);
}

/// A column only a residual reads stays readable above the partial aggregate:
/// it is one of the keys, and the condition sees each of its values.
#[test]
fn a_residual_condition_reads_a_pushed_key() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let rows: Vec<Row> = (0..2400_i64)
        .map(|n| (n % 12, Some(i32::try_from(n % 4).unwrap()), None, None))
        .collect();
    let t = tables(dir.path(), &rows, &WEIGHTS, &rt);
    let q = "SELECT w.wk, sum(f.i), count(f.i) FROM f LEFT JOIN w ON f.k = w.wk AND f.i + 9 > w.wk \
             GROUP BY w.wk";
    assert!(fires(&t, q));
    let plan = t.optimized(q, Order::Statistics).unwrap();
    assert!(
        physical(&t, &plan)
            .unwrap()
            .to_string()
            .contains("keys=[k#0, i#1]")
            || plan
                .root()
                .to_string()
                .contains("PartialAggregate [k#0, i#1]"),
        "{}",
        plan.root()
    );
    let (on, off) = both(&t, q);
    assert_eq!(on, off);
}

/// Whichever of two dimensions join ordering puts on top, the rule moves the
/// one the aggregate groups by there: one of these two needs the rotation.
#[test]
fn the_grouping_dimension_moves_to_the_top_join() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let rows: Vec<Row> = (0..2400_i64)
        .map(|n| (n % 12, Some(i32::try_from(n % 10).unwrap()), None, None))
        .collect();
    let t = tables(dir.path(), &rows, &WEIGHTS, &rt);
    for group in ["a", "b"] {
        let q = format!(
            "SELECT {group}.wk, count(f.s), min(f.k) FROM f JOIN w a ON f.k = a.wk \
             JOIN w b ON f.i = b.wk GROUP BY {group}.wk"
        );
        assert!(fires(&t, &q), "{q}");
        let (on, off) = both(&t, &q);
        assert_eq!(on, off, "{q}");
        let plan = t.optimized(&q, Order::Statistics).unwrap();
        let again = krishiv_optimizer::optimize_with(
            &plan,
            &krishiv_exec::planner::ProviderStatistics(std::sync::Arc::clone(&t.tables) as _),
        )
        .unwrap();
        assert!(
            again.structurally_eq(&plan),
            "{q}: not a fixed point:\n{}\n{}",
            plan.root(),
            again.root()
        );
    }
}

/// A join that drops most rows does not pay for the aggregate below it.
#[test]
fn a_join_that_drops_rows_is_left_alone() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = tables(dir.path(), &many(), &[3], &rt);
    let q = "SELECT w.wk, sum(f.i) FROM f JOIN w ON f.k = w.wk GROUP BY w.wk";
    assert!(!fires(&t, q));
    let (on, off) = both(&t, q);
    assert_eq!(on, off);
}

/// Keys with as many values as rows do not group anything.
#[test]
fn keys_that_do_not_group_are_left_alone() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let rows: Vec<Row> = (0..240_i64).map(|n| (n, Some(1), None, None)).collect();
    let keys: Vec<i64> = (0..240).collect();
    let t = tables(dir.path(), &rows, &keys, &rt);
    let q = "SELECT w.wk, sum(f.i) FROM f JOIN w ON f.k = w.wk GROUP BY w.wk";
    assert!(!fires(&t, q));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]
    /// Whatever the rows and join multiplicities: the rule's plan answers as
    /// the aggregate over the join does, firing or not.
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
