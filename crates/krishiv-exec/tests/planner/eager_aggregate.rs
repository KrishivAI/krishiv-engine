// SPDX-License-Identifier: Apache-2.0
//! The eager aggregation rule (design P2-T167 unit 2): where it fires the plan
//! returns what the aggregate over the join returns, and each of the design's
//! semantics rules keeps a call above the join.

use proptest::prelude::*;

use super::partial_aggregate::{
    Row, physical, runtime, sorted, tables, tables_with_attribute, tables_with_null_keys,
};
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

const FIRES: [&str; 10] = [
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
    // Exact decimal arithmetic never raises (TPC-DS q11's `ss_ext_list_price - ss_ext_discount_amt`).
    "SELECT w.wk, sum(f.d - CAST(f.i AS DECIMAL(10, 0))), avg(f.d + f.d) FROM f JOIN w ON f.k = w.wk \
     GROUP BY w.wk",
    // Nor does its division by a whole literal that keeps the integral digits (q04's `(...) / 2`).
    "SELECT w.wk, sum((f.d + f.d) / 2), avg(f.d / 3) FROM f JOIN w ON f.k = w.wk GROUP BY w.wk",
];

/// Each stays above the join, for the reason in its comment.
const STAYS: [&str; 14] = [
    // An argument that can raise on a row the join would drop.
    "SELECT w.wk, sum(f.i + 1) FROM f JOIN w ON f.k = w.wk GROUP BY w.wk",
    // A decimal divided by less than one: the quotient can outgrow the dividend.
    "SELECT w.wk, sum(f.d / 0.5) FROM f JOIN w ON f.k = w.wk GROUP BY w.wk",
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

/// A nondeterministic grouping key is evaluated once per joined row, not once per partial group, so the call stays
/// above the join. (Not in `STAYS`: the on and off plans order the join differently, and a seeded `rand` draws from
/// the rows in the order they arrive, so their answers are not comparable.)
const NONDETERMINISTIC_KEYS: [&str; 3] = [
    "SELECT count(*) FROM f JOIN w ON f.k = w.wk GROUP BY rand(0)",
    "SELECT w.wk, count(*) FROM f JOIN w ON f.k = w.wk GROUP BY w.wk, rand(0)",
    "SELECT x, wk, sum(i) FROM (SELECT w.wk, f.i, rand(0) AS x FROM f JOIN w ON f.k = w.wk) GROUP BY x, wk",
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

#[test]
fn a_nondeterministic_key_keeps_the_call_above_the_join() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = tables(dir.path(), &many(), &WEIGHTS, &rt);
    for q in NONDETERMINISTIC_KEYS {
        assert!(!fires(&t, q), "{q}");
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

/// A distinct runs below the join too (no calls: the partial aggregate only
/// drops repeated groups), and a distinct over a semi or anti join
/// (`INTERSECT`, `EXCEPT`) moves onto both inputs first (lever K5, TPC-DS
/// q14's `cross_items`).
const DISTINCTS: [(&str, bool); 7] = [
    ("SELECT DISTINCT w.wk FROM f JOIN w ON f.k = w.wk", true),
    (
        "SELECT w.wk FROM f JOIN w ON f.k = w.wk GROUP BY w.wk",
        true,
    ),
    (
        "SELECT w.wk FROM f JOIN w ON f.k = w.wk INTERSECT SELECT f.k FROM f WHERE f.i > 100",
        true,
    ),
    (
        "SELECT w.wk FROM f JOIN w ON f.k = w.wk EXCEPT SELECT f.k FROM f WHERE f.k < 6",
        true,
    ),
    // Chained: the left of the outer one is distinct already.
    (
        "SELECT f.k FROM f INTERSECT SELECT w.wk FROM w INTERSECT SELECT f.k FROM f WHERE f.i < 50",
        false,
    ),
    // Two columns, NULLs on one: `<=>` matches them.
    (
        "SELECT f.k, f.s FROM f INTERSECT SELECT f.k, f.s FROM f WHERE f.k > 3",
        false,
    ),
    // A floating-point column stays as written.
    (
        "SELECT f.k, CAST(f.i AS DOUBLE) FROM f INTERSECT SELECT f.k, CAST(f.i AS DOUBLE) FROM f",
        false,
    ),
];

#[test]
fn distincts_move_below_joins_and_return_the_same_rows() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = tables(dir.path(), &many(), &WEIGHTS, &rt);
    for (q, fires_here) in DISTINCTS {
        if fires_here {
            assert!(fires(&t, q), "{q}");
        }
        let (on, off) = both(&t, q);
        assert_eq!(on, off, "{q}");
        assert!(!on.is_empty(), "{q}");
    }
}

/// A nullable dimension key gets `InferJoinFilters`' `isnotnull` filter on
/// the dimension (TPC-DS q11's `customer`): the dimension is still read whole
/// for the rule, which fires, and the NULL keys the join drops stay dropped.
#[test]
fn a_null_only_filter_on_the_dimension_keeps_the_rule() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let keys: Vec<Option<i64>> = WEIGHTS
        .iter()
        .copied()
        .map(Some)
        .chain([None, None])
        .collect();
    let t = tables_with_null_keys(dir.path(), &many(), &keys, &rt);
    let q = "SELECT w.wk, count(f.i), sum(f.i), sum(f.d - CAST(f.i AS DECIMAL(10, 0))) \
             FROM f JOIN w ON f.k = w.wk GROUP BY w.wk";
    let plan = t.optimized(q, Order::Statistics).unwrap();
    assert!(
        plan.root().to_string().contains("isnotnull(wk"),
        "{}",
        plan.root()
    );
    assert!(fires(&t, q), "{}", plan.root());
    let (on, off) = both(&t, q);
    assert_eq!(on, off);
    assert_eq!(on.len(), 12);
    // Another filter on the dimension still keeps the call above the join.
    let q = "SELECT w.wk, sum(f.i) FROM f JOIN w ON f.k = w.wk WHERE w.wk < 6 GROUP BY w.wk";
    assert!(!fires(&t, q));
    assert_eq!(both(&t, q).0, both(&t, q).1);
}

/// Only a not-NULL filter on a plain `=` key of the join reads the dimension
/// whole: one on another column (a sparse attribute prunes the dimension, and
/// its runtime filter the fact side), or on a null-safe `<=>` key, where a
/// NULL key does match, keeps the call above the join (review of 37a66474c
/// finding 1).
#[test]
fn a_not_null_filter_off_the_plain_keys_keeps_the_call_above() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let keys: Vec<(i64, Option<i64>)> = WEIGHTS
        .iter()
        .map(|&k| (k, (k % 4 == 0).then_some(k)))
        .collect();
    let t = tables_with_attribute(dir.path(), &many(), &keys, &rt);
    let q = "SELECT w.wk, count(f.i), sum(f.i) FROM f JOIN w ON f.k = w.wk \
             WHERE w.wa IS NOT NULL GROUP BY w.wk";
    let plan = t.optimized(q, Order::Statistics).unwrap();
    assert!(
        plan.root().to_string().contains("isnotnull(wa"),
        "{}",
        plan.root()
    );
    assert!(!fires(&t, q), "{}", plan.root());
    let (on, off) = both(&t, q);
    assert_eq!(on, off);
    assert_eq!(on.len(), 3);
    // The same dimension unfiltered still takes the rule.
    let q = "SELECT w.wk, count(f.i), sum(f.i) FROM f JOIN w ON f.k = w.wk GROUP BY w.wk";
    assert!(fires(&t, q));

    let dir = tempfile::tempdir().unwrap();
    let keys: Vec<Option<i64>> = WEIGHTS
        .iter()
        .copied()
        .map(Some)
        .chain([None, None])
        .collect();
    let t = tables_with_null_keys(dir.path(), &many(), &keys, &rt);
    let q = "SELECT w.wk, count(f.i), sum(f.i) FROM f JOIN w ON f.k <=> w.wk \
             WHERE w.wk IS NOT NULL GROUP BY w.wk";
    let plan = t.optimized(q, Order::Statistics).unwrap();
    assert!(
        plan.root().to_string().contains("isnotnull(wk"),
        "{}",
        plan.root()
    );
    assert!(!fires(&t, q), "{}", plan.root());
    let (on, off) = both(&t, q);
    assert_eq!(on, off);
}

/// The notes of `q`'s optimization by statistics.
fn notes(t: &Tpch, q: &str) -> String {
    let (_, steps) = t.optimized_in(q, &[]).unwrap();
    steps
        .iter()
        .map(|s| s.note.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// A distinct over a semi or anti join whose inputs are two instances of one
/// CTE (`Shift`s of its body) splits onto both, NULL keys included, and
/// returns the rows it returned unsplit (review of 37a66474c finding 2a).
#[test]
fn a_distinct_splits_across_instances_of_one_cte() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    // `f.k` 10 and 11 have no `w` row: `x` is NULL for them.
    let t = tables(dir.path(), &many(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], &rt);
    let cte = "WITH c AS (SELECT w.wk AS x, f.i FROM f LEFT JOIN w ON f.k = w.wk) ";
    for (q, kind) in [
        (
            format!("{cte}SELECT x FROM c WHERE i > 10 INTERSECT SELECT x FROM c WHERE i < 200"),
            "LeftSemi",
        ),
        (
            format!("{cte}SELECT x FROM c WHERE i > 10 EXCEPT SELECT x FROM c WHERE x < 4"),
            "LeftAnti",
        ),
    ] {
        let plan = t.optimized(&q, Order::Statistics).unwrap();
        let text = plan.root().to_string();
        assert!(text.contains("Shift"), "{text}");
        let n = notes(&t, &q);
        assert!(
            n.contains(&format!("split across the {kind} join")),
            "{q}\n{n}\n{text}"
        );
        let (on, off) = both(&t, &q);
        assert_eq!(on, off, "{q}");
        assert!(on.iter().any(|r| r == "NULL"), "{q}: {on:?}");
    }
}

/// The split's refusals, by their notes: a floating-point key, and a
/// condition that reads a left column the distinct drops (review of
/// 37a66474c finding 2b).
#[test]
fn a_distinct_that_cannot_split_says_why() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = tables(dir.path(), &many(), &WEIGHTS, &rt);
    for (q, why) in [
        (
            "SELECT f.k, CAST(f.i AS DOUBLE) FROM f INTERSECT SELECT f.k, CAST(f.i AS DOUBLE) FROM f",
            "is a floating point, collated or nested column",
        ),
        (
            "SELECT f.k FROM f LEFT SEMI JOIN w ON f.k = w.wk AND f.i > w.wk GROUP BY f.k",
            "the condition reads a column of the left side the distinct drops",
        ),
    ] {
        let n = notes(&t, q);
        assert!(
            n.contains("distinct not split: ") && n.contains(why),
            "{q}\n{n}"
        );
        let (on, off) = both(&t, q);
        assert_eq!(on, off, "{q}");
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

/// 240 rows whose `k` (12 values) and `i` (9 values, from its range; a few NULLs) make 36 pairs:
/// keys that correlate, where the product of their counts (108) says an aggregate on both cannot
/// reduce 4 times.
fn correlated() -> Vec<Row> {
    (0..240_i64)
        .map(|n| {
            (
                n % 12,
                (n % 11 != 0).then(|| i32::try_from(n % 9).unwrap()),
                (n % 5 != 0).then(|| i128::from(n) * 100),
                (n % 3 != 0).then(|| u8::try_from(n % 9).unwrap()),
            )
        })
        .collect()
}

/// Correlated keys (perf-6 N3', TPC-DS q38/q87): the gate reads the largest distinct count of
/// one key, not the product, so the aggregate moves below the join and returns the rows it
/// returned above it.
#[test]
fn correlated_keys_move_below_the_join_and_return_the_same_rows() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = tables(dir.path(), &correlated(), &WEIGHTS, &rt);
    for q in [
        "SELECT DISTINCT w.wk, f.i FROM f JOIN w ON f.k = w.wk",
        "SELECT w.wk, f.i, count(*), min(f.d), max(f.d) FROM f JOIN w ON f.k = w.wk \
         GROUP BY w.wk, f.i",
    ] {
        assert!(fires(&t, q), "{q}\n{}", notes(&t, q));
        let (on, off) = both(&t, q);
        assert_eq!(on, off, "{q}");
        assert!(!on.is_empty(), "{q}");
    }
}

/// q38's shape: an `INTERSECT` of two distincts on correlated keys over the same dimension.
/// Both distincts move below their joins, NULL keys included, and the rows do not change (the
/// optimizer's `eager_correlated` test covers q38's own plan, where the dimension is read
/// through a `Shift`).
#[test]
fn an_intersect_of_correlated_distincts_moves_both_and_returns_the_same_rows() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = tables(dir.path(), &correlated(), &WEIGHTS, &rt);
    let q = "SELECT DISTINCT w.wk, f.i FROM f JOIN w ON f.k = w.wk WHERE f.d IS NOT NULL \
             INTERSECT SELECT DISTINCT w.wk, f.i FROM f JOIN w ON f.k = w.wk WHERE f.s IS NOT NULL";
    let n = notes(&t, q);
    let plan = t.optimized(q, Order::Statistics).unwrap();
    let text = plan.root().to_string();
    assert!(
        !n.contains("the other side is filtered or joined"),
        "{q}\n{n}\n{text}"
    );
    assert!(
        text.matches("PartialAggregate").count() >= 2,
        "{q}\n{n}\n{text}"
    );
    let (on, off) = both(&t, q);
    assert_eq!(on, off, "{q}");
    assert!(!on.is_empty(), "{q}");
}
