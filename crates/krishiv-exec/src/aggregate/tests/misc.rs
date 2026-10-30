// SPDX-License-Identifier: Apache-2.0
//! The `bit_and`/`bit_or`/`bit_xor`, `regr_*`, `mode`, `listagg` and
//! `try_sum`/`try_avg` aggregates: Spark 4.2's values on the golden
//! inputs (quoted next to each case), NULL handling, empty and all-NULL
//! groups, single-row groups, `DISTINCT` dedup through the planner's
//! distinct rewrite where Spark allows it, and merge-correctness (one
//! partition against eight; exact, or within `1e-9` relative for the
//! floating `regr_*`, as Spark's reassociation rounds).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int8Type, Int16Type, Int32Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, StringViewArray,
};
use arrow_schema::DataType;
use krishiv_types::batch::Batch;

use super::*;
use AggregateFunction::{
    BitAnd, BitOr, BitXor, Mode, ModeMax, ModeMin, RegrAvgx, RegrAvgy, RegrCount, RegrIntercept,
    RegrR2, RegrSlope, RegrSxx, RegrSxy, RegrSyy, TryAvg, TrySum,
};

fn i8s(v: &[Option<i8>]) -> ArrayRef {
    Arc::new(Int8Array::from(v.to_vec()))
}

fn i16s(v: &[Option<i16>]) -> ArrayRef {
    Arc::new(Int16Array::from(v.to_vec()))
}

fn i32s(v: &[Option<i32>]) -> ArrayRef {
    Arc::new(Int32Array::from(v.to_vec()))
}

fn i64s(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

fn f64s(v: &[Option<f64>]) -> ArrayRef {
    Arc::new(Float64Array::from(v.to_vec()))
}

fn strs(v: &[Option<&str>]) -> ArrayRef {
    Arc::new(StringViewArray::from(v.to_vec()))
}

/// Renders the cell types these tests use (the shared `cell` covers a
/// different set; see `rows`).
fn mcell(a: &ArrayRef, i: usize) -> String {
    if a.is_null(i) {
        return "NULL".to_owned();
    }
    match a.data_type() {
        DataType::Int8 => a.as_primitive::<Int8Type>().value(i).to_string(),
        DataType::Int16 => a.as_primitive::<Int16Type>().value(i).to_string(),
        DataType::Int32 => a.as_primitive::<Int32Type>().value(i).to_string(),
        DataType::Int64 => a.as_primitive::<Int64Type>().value(i).to_string(),
        DataType::Float64 => format!("{:?}", a.as_primitive::<Float64Type>().value(i)),
        DataType::Utf8 => a.as_string::<i32>().value(i).to_owned(),
        DataType::Utf8View => a.as_string_view().value(i).to_owned(),
        DataType::Binary => String::from_utf8_lossy(a.as_binary::<i32>().value(i)).into_owned(),
        DataType::BinaryView => String::from_utf8_lossy(a.as_binary_view().value(i)).into_owned(),
        DataType::Boolean => a.as_boolean().value(i).to_string(),
        _ => format!("<{}>", a.data_type()),
    }
}

fn mrows(batches: &[Batch]) -> Vec<String> {
    let mut out: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            let d = b.data();
            (0..d.num_rows()).map(move |r| {
                d.columns()
                    .iter()
                    .map(|c| mcell(c, r))
                    .collect::<Vec<_>>()
                    .join("|")
            })
        })
        .collect();
    out.sort();
    out
}

async fn mrun(plan: Arc<dyn crate::ExecNode>) -> Vec<String> {
    mrows(&exec(plan, Opts::default()).await.unwrap().0)
}

/// Runs `plan` with `slots` drivers (the default layout only).
async fn slots(plan: Arc<dyn crate::ExecNode>, slots: usize) -> Vec<String> {
    let o = Opts {
        slots,
        both_layouts: false,
        ..Opts::default()
    };
    super::rows(&exec(plan, o).await.unwrap().0)
}

/// `SELECT bit_xor(b3), ...` over the bitwise golden's rows.
fn bitwise_input() -> (arrow_schema::SchemaRef, Vec<Vec<ArrayRef>>) {
    let s = schema(vec![
        ("b1", DataType::Int32),
        ("b2", DataType::Int32),
        ("b3", DataType::Int32),
        ("b4", DataType::Int64),
    ]);
    let b = vec![
        i32s(&[Some(1), Some(2), Some(7)]),
        i32s(&[Some(1), Some(3), Some(7)]),
        i32s(&[Some(1), Some(4), Some(7)]),
        i64s(&[Some(1), None, Some(3)]),
    ];
    (s, vec![b])
}

/// The bitwise aggregates over `TINYINT` .. `BIGINT`: NULLs skipped, NULL
/// for no input. From `bitwise.sql` (`BIT_XOR(b3)` over `(1,1,1,1L),
/// (2,3,4,null), (7,7,7,3L)` gives 2).
#[tokio::test]
async fn bit_aggregates() {
    let (s, b) = bitwise_input();
    let aggs = vec![
        agg(&BitAnd, 0, &s),
        agg(&BitOr, 1, &s),
        agg(&BitXor, 2, &s),
        agg(&BitXor, 3, &s),
    ];
    assert_eq!(
        mrun(node(scan(&s, b), spec(&s, vec![], aggs))).await,
        ["0|7|2|2"]
    );
    // Grouped: `bit_xor(b3)` grouped by `b1` over the same rows gives
    // 1, 4 and 7.
    let (s, b) = bitwise_input();
    let aggs = vec![agg(&BitXor, 2, &s)];
    let mut rows = mrun(node(scan(&s, b), spec(&s, vec![0], aggs))).await;
    rows.sort();
    assert_eq!(rows, ["1|1", "2|4", "7|7"]);
}

/// `bit_and`/`bit_or` keep the input type; other types are refused at
/// analysis (pinned by the registry matrix), and the executor reads
/// every integral width.
#[tokio::test]
async fn bit_keeps_narrow_types_and_skips_nulls() {
    for (f, t, want) in [
        (BitAnd, DataType::Int8, "1"),
        (BitOr, DataType::Int16, "7"),
        (BitXor, DataType::Int32, "6"),
    ] {
        let s = schema(vec![("x", t.clone())]);
        let col: ArrayRef = match t {
            DataType::Int8 => i8s(&[Some(3), None, Some(5)]),
            DataType::Int16 => i16s(&[Some(3), None, Some(5)]),
            _ => i32s(&[Some(3), None, Some(5)]),
        };
        let rows = mrun(node(
            scan(&s, vec![vec![col]]),
            spec(&s, vec![], vec![agg(&f, 0, &s)]),
        ))
        .await;
        assert_eq!(rows, [want], "{f:?}");
    }
    // No non-NULL input, and no rows at all: NULL.
    let s = schema(vec![("x", DataType::Int64)]);
    let none = mrun(node(
        scan(&s, vec![vec![i64s(&[None, None])]]),
        spec(&s, vec![], vec![agg(&BitAnd, 0, &s)]),
    ))
    .await;
    assert_eq!(none, ["NULL"]);
    let empty = mrun(node(
        scan(
            &s,
            vec![vec![Arc::new(Int64Array::from(Vec::<Option<i64>>::new()))]],
        ),
        spec(&s, vec![], vec![agg(&BitOr, 0, &s)]),
    ))
    .await;
    assert_eq!(empty, ["NULL"]);
}

/// One partition and eight partitions agree exactly (bitwise operations
/// are associative and commutative).
#[tokio::test]
async fn bit_partitions_agree() {
    let (s, one) = bitwise_input();
    let multi: Vec<Vec<ArrayRef>> = (0..4).map(|_| one[0].clone()).collect();
    for f in [&BitAnd, &BitOr, &BitXor] {
        let plan = || {
            node(
                scan(&s, multi.clone()),
                spec(&s, vec![], vec![agg(f, 2, &s)]),
            )
        };
        assert_eq!(slots(plan(), 1).await, slots(plan(), 8).await, "{f:?}");
    }
}

/// The regression golden's rows: `(1, 10, null), (2, 10, 11), (2, 20,
/// 22), (2, 25, null), (2, 30, 35), (2, null, 40)` as `(k, y, x)`.
fn regr_input() -> (arrow_schema::SchemaRef, Vec<Vec<ArrayRef>>) {
    let s = schema(vec![
        ("k", DataType::Int32),
        ("y", DataType::Float64),
        ("x", DataType::Float64),
    ]);
    let b = vec![
        i32s(&[Some(1), Some(2), Some(2), Some(2), Some(2), Some(2)]),
        f64s(&[
            Some(10.0),
            Some(10.0),
            Some(20.0),
            Some(25.0),
            Some(30.0),
            None,
        ]),
        f64s(&[None, Some(11.0), Some(22.0), None, Some(35.0), Some(40.0)]),
    ];
    (s, vec![b])
}

/// `regr_count` counts the rows where both arguments are non-NULL, exactly.
#[tokio::test]
async fn regr_count_counts_pairs() {
    let (s, b) = regr_input();
    let aggs = vec![agg(&RegrCount, 1, &s)];
    // Two arguments: the planner passes both columns; the executor reads
    // the pair. `agg` takes one column, so build the pair by hand.
    let mut e = aggs.into_iter().next().unwrap();
    e.args = vec![1, 2];
    assert_eq!(
        mrun(node(scan(&s, b), spec(&s, vec![], vec![e]))).await,
        ["3"]
    );
}

/// The `regr_*` values on the golden rows, within `1e-9` relative: from
/// `linear-regression.sql` (`regr_r2` is `0.997690531177829`,
/// `regr_sxx` is `288.6666666666667`, `regr_slope` is
/// `0.8314087759815242`, `regr_intercept` is `1.1547344110854496`).
#[tokio::test]
async fn regr_values_match_spark() {
    let (s, b) = regr_input();
    let mut aggs = Vec::new();
    for f in [
        &RegrAvgx,
        &RegrAvgy,
        &RegrSxx,
        &RegrSyy,
        &RegrSxy,
        &RegrSlope,
        &RegrIntercept,
        &RegrR2,
    ] {
        let mut e = agg(f, 1, &s);
        e.args = vec![1, 2];
        aggs.push(e);
    }
    let rows = mrun(node(scan(&s, b), spec(&s, vec![], aggs))).await;
    assert_eq!(rows.len(), 1);
    let got: Vec<f64> = rows[0].split('|').map(|v| v.parse().unwrap()).collect();
    let want: [f64; 8] = [
        22.666666666666668,
        20.0,
        288.6666666666667,
        200.0,
        240.0,
        0.8314087759815242,
        1.1547344110854496,
        0.997690531177829,
    ];
    for (g, w) in got.iter().zip(want) {
        let tol = 1e-9 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "{g} vs {w}");
    }
}

/// One-sided NULL pairs, empty groups, single-row groups and all-NULL
/// input: `regr_count` is 0, the rest NULL.
#[tokio::test]
async fn regr_nulls_and_edges() {
    let s = schema(vec![("y", DataType::Float64), ("x", DataType::Float64)]);
    let vs = |y: &[Option<f64>], x: &[Option<f64>]| vec![vec![f64s(y), f64s(x)]];
    let one = |f: &AggregateFunction, y: &[Option<f64>], x: &[Option<f64>]| {
        let mut e = agg(f, 0, &s);
        e.args = vec![0, 1];
        mrun(node(scan(&s, vs(y, x)), spec(&s, vec![], vec![e])))
    };
    // One-sided NULLs do not count.
    assert_eq!(
        one(&RegrCount, &[Some(1.0), None], &[None, Some(2.0)]).await,
        ["0"]
    );
    assert_eq!(
        one(&RegrAvgx, &[Some(1.0), None], &[None, Some(2.0)]).await,
        ["NULL"]
    );
    // All NULL, and no rows: NULL (count 0).
    assert_eq!(
        one(&RegrSlope, &[None, None], &[None, None]).await,
        ["NULL"]
    );
    assert_eq!(one(&RegrCount, &[], &[]).await, ["0"]);
    assert_eq!(one(&RegrR2, &[], &[]).await, ["NULL"]);
    // A single pair has no spread: slope, intercept and r2 are NULL, the
    // means are the values, the sums of squares are 0.
    assert_eq!(one(&RegrSlope, &[Some(2.0)], &[Some(4.0)]).await, ["NULL"]);
    assert_eq!(one(&RegrR2, &[Some(2.0)], &[Some(4.0)]).await, ["NULL"]);
    assert_eq!(one(&RegrAvgx, &[Some(2.0)], &[Some(4.0)]).await, ["4.0"]);
    assert_eq!(one(&RegrSxx, &[Some(2.0)], &[Some(4.0)]).await, ["0.0"]);
    // A constant independent variable: slope, intercept and r2 are NULL.
    assert_eq!(
        one(&RegrSlope, &[Some(1.0), Some(2.0)], &[Some(5.0), Some(5.0)]).await,
        ["NULL"]
    );
    assert_eq!(
        one(&RegrR2, &[Some(1.0), Some(2.0)], &[Some(5.0), Some(5.0)]).await,
        ["NULL"]
    );
    // A constant dependent variable with spread: r2 is 1.
    assert_eq!(
        one(&RegrR2, &[Some(7.0), Some(7.0)], &[Some(1.0), Some(3.0)]).await,
        ["1.0"]
    );
}

/// One partition and eight partitions agree: exactly for `regr_count`,
/// within `1e-9` relative for the floating `regr_*` (Spark reassociates
/// the pairwise sums when it merges, as the engine does).
#[tokio::test]
async fn regr_partitions_agree() {
    let (s, one) = regr_input();
    let multi: Vec<Vec<ArrayRef>> = (0..4).map(|_| one[0].clone()).collect();
    for f in [
        &RegrAvgx,
        &RegrAvgy,
        &RegrSxx,
        &RegrSyy,
        &RegrSxy,
        &RegrSlope,
        &RegrIntercept,
        &RegrR2,
    ] {
        let plan = || {
            let mut e = agg(f, 1, &s);
            e.args = vec![1, 2];
            node(scan(&s, multi.clone()), spec(&s, vec![], vec![e]))
        };
        let (a, b) = (slots(plan(), 1).await, slots(plan(), 8).await);
        assert_eq!(a.len(), 1, "{f:?}");
        assert_eq!(b.len(), 1, "{f:?}");
        let parse = |r: &str| r.parse::<f64>().unwrap();
        let (x, y) = (parse(&a[0]), parse(&b[0]));
        if x.is_nan() || y.is_nan() {
            assert!(x.is_nan() && y.is_nan(), "{f:?}");
        } else if x == 0.0 && y == 0.0 {
            assert_eq!(x, y, "{f:?}");
        } else {
            let tol = 1e-9 * x.abs().max(y.abs()).max(1.0);
            assert!((x - y).abs() <= tol, "{f:?}: {x} vs {y}");
        }
    }
    let plan = || {
        let mut e = agg(&RegrCount, 1, &s);
        e.args = vec![1, 2];
        node(scan(&s, multi.clone()), spec(&s, vec![], vec![e]))
    };
    assert_eq!(slots(plan(), 1).await, ["12"]);
    assert_eq!(slots(plan(), 1).await, slots(plan(), 8).await);
}

/// `mode` over the golden's grouped salaries: all values are unique, so
/// every group ties, and Spark's slot order picks `6627`, `5186`,
/// `11303` and `10563` (from `mode.sql`).
#[tokio::test]
async fn mode_breaks_ties_like_spark() {
    let s = schema(vec![("d", DataType::Utf8View), ("sal", DataType::Int32)]);
    let b = vec![vec![
        strs(&[
            Some("Accounting"),
            Some("Accounting"),
            Some("Accounting"),
            Some("Accounting"),
            Some("Accounting"),
            Some("Accounting"),
            Some("IT"),
            Some("IT"),
            Some("Sales"),
            Some("Sales"),
            Some("Sales"),
            Some("Sales"),
            Some("SCM"),
            Some("SCM"),
            Some("SCM"),
            Some("SCM"),
            Some("SCM"),
        ]),
        i32s(&[
            Some(8435),
            Some(9998),
            Some(8992),
            Some(8870),
            Some(11472),
            Some(6627),
            Some(8113),
            Some(5186),
            Some(9181),
            Some(9441),
            Some(6660),
            Some(10563),
            Some(10449),
            Some(6949),
            Some(11303),
            Some(11798),
            Some(10586),
        ]),
    ]];
    let rows = mrun(node(
        scan(&s, b),
        spec(&s, vec![0], vec![agg(&Mode, 1, &s)]),
    ))
    .await;
    assert_eq!(
        rows,
        ["Accounting|6627", "IT|5186", "SCM|11303", "Sales|10563",]
    );
}

/// `mode` tie-breaks: `(-10, 0, 10)` gives `0`; the departments give
/// `Accounting` (six of sixteen); the ordered forms give the smallest
/// and the largest.
#[tokio::test]
async fn mode_ties_and_ordered_forms() {
    let s = schema(vec![("x", DataType::Int32)]);
    let one = |f: &AggregateFunction, v: &[Option<i32>]| {
        mrun(node(
            scan(&s, vec![vec![i32s(v)]]),
            spec(&s, vec![], vec![agg(f, 0, &s)]),
        ))
    };
    assert_eq!(one(&Mode, &[-10, 0, 10].map(Some)).await, ["0"]);
    assert_eq!(one(&ModeMin, &[-10, 0, 10].map(Some)).await, ["-10"]);
    assert_eq!(one(&ModeMax, &[-10, 0, 10].map(Some)).await, ["10"]);
    // A clear winner beats the tie-break.
    assert_eq!(
        one(&Mode, &[Some(1), Some(2), Some(2), Some(3)]).await,
        ["2"]
    );
    assert_eq!(one(&ModeMin, &[Some(3), Some(1), Some(1)]).await, ["1"]);
    // NULLs are ignored.
    assert_eq!(one(&Mode, &[None, Some(5), Some(5), None]).await, ["5"]);
    // No non-NULL value, and no rows: NULL.
    assert_eq!(one(&Mode, &[None, None]).await, ["NULL"]);
    assert_eq!(one(&ModeMax, &[] as &[Option<i32>]).await, ["NULL"]);
}

/// `mode` over strings breaks ties in Spark's slot order (from
/// `mode.sql`: `mode(department)` is `Accounting`).
#[tokio::test]
async fn mode_strings() {
    let s = schema(vec![("d", DataType::Utf8View)]);
    let b = vec![vec![strs(&[
        Some("Accounting"),
        Some("IT"),
        Some("Sales"),
        Some("SCM"),
    ])]];
    // All four tie: Spark's slot order picks the lowest slot (`SCM`).
    assert_eq!(
        mrun(node(scan(&s, b), spec(&s, vec![], vec![agg(&Mode, 0, &s)]))).await,
        ["SCM"]
    );
}

/// One partition and eight partitions agree: exactly when the winner is
/// unique (the common case), deterministically otherwise.
#[tokio::test]
async fn mode_partitions_agree() {
    let s = schema(vec![("k", DataType::Int32), ("x", DataType::Int32)]);
    let batch = vec![
        i32s(&[Some(0), Some(0), Some(0), Some(1), Some(1)]),
        i32s(&[Some(1), Some(2), Some(2), Some(4), Some(4)]),
    ];
    let multi: Vec<Vec<ArrayRef>> = (0..4).map(|_| batch.clone()).collect();
    for f in [&Mode, &ModeMin, &ModeMax] {
        let plan = || {
            node(
                scan(&s, multi.clone()),
                spec(&s, vec![0], vec![agg(f, 1, &s)]),
            )
        };
        assert_eq!(slots(plan(), 1).await, slots(plan(), 8).await, "{f:?}");
    }
    // Both winners are unique: group 0 gives 2, group 1 gives 4.
    assert_eq!(
        slots(
            {
                let e = agg(&Mode, 1, &s);
                node(scan(&s, multi.clone()), spec(&s, vec![0], vec![e]))
            },
            1
        )
        .await,
        ["0|2", "1|4"]
    );
}

/// `listagg` without ordering keeps input order within a partition and
/// merges partitions in partition-index order: `df`'s `col2` groups give
/// `bc` and `bd` (from `listagg.sql`).
#[tokio::test]
async fn listagg_keeps_input_order() {
    let s = schema(vec![("a", DataType::Utf8View), ("b", DataType::Utf8View)]);
    let b = vec![vec![
        strs(&[Some("a"), Some("a"), Some("b"), Some("b"), None]),
        strs(&[Some("b"), Some("c"), Some("c"), Some("d"), None]),
    ]];
    let listagg = || {
        AggregateExpr::listagg(
            AggregateFunction::ListAgg {
                delimiter: Vec::new(),
                orders: Vec::new(),
            },
            1,
            DataType::Utf8View,
            Vec::new(),
            Vec::new(),
            "x",
        )
    };
    // Single-partition fold order is the input order.
    let rows = mrun(node(
        scan(&s, b.clone()),
        spec(&s, vec![0], vec![listagg()]),
    ))
    .await;
    assert_eq!(rows, ["NULL|NULL", "a|bc", "b|cd"]);
}

/// `listagg` delimiters: `|` joins, NULL (or a missing delimiter) joins
/// with nothing, and binary values join with binary delimiters.
#[tokio::test]
async fn listagg_delimiters() {
    let s = schema(vec![("v", DataType::Utf8View)]);
    let run = |delim: Vec<u8>| {
        let e = AggregateExpr::listagg(
            AggregateFunction::ListAgg {
                delimiter: delim,
                orders: Vec::new(),
            },
            0,
            DataType::Utf8View,
            Vec::new(),
            Vec::new(),
            "x",
        );
        mrun(node(
            scan(&s, vec![vec![strs(&[Some("a"), None, Some("b")])]]),
            spec(&s, vec![], vec![e]),
        ))
    };
    assert_eq!(run(b"|".to_vec()).await, ["a|b"]);
    assert_eq!(run(Vec::new()).await, ["ab"]);
    // Empty input, and all-NULL input: NULL, never empty.
    let empty = mrun(node(
        scan(&s, vec![vec![strs(&[] as &[Option<&str>])]]),
        spec(
            &s,
            vec![],
            vec![AggregateExpr::listagg(
                AggregateFunction::ListAgg {
                    delimiter: b"|".to_vec(),
                    orders: Vec::new(),
                },
                0,
                DataType::Utf8View,
                Vec::new(),
                Vec::new(),
                "x",
            )],
        ),
    ))
    .await;
    assert_eq!(empty, ["NULL"]);
    let nulls = mrun(node(
        scan(&s, vec![vec![strs(&[None, None])]]),
        spec(
            &s,
            vec![],
            vec![AggregateExpr::listagg(
                AggregateFunction::ListAgg {
                    delimiter: b"|".to_vec(),
                    orders: Vec::new(),
                },
                0,
                DataType::Utf8View,
                Vec::new(),
                Vec::new(),
                "x",
            )],
        ),
    ))
    .await;
    assert_eq!(nulls, ["NULL"]);
}

/// `listagg` with `WITHIN GROUP` ordering sorts once at evaluation:
/// `ORDER BY col1 DESC` over `(a, b), (a, c), (b, c), (b, d)` gives
/// `bbaa` (from `listagg.sql`).
#[tokio::test]
async fn listagg_ordering() {
    let s = schema(vec![("a", DataType::Utf8View), ("b", DataType::Utf8View)]);
    let b = vec![vec![
        strs(&[Some("a"), Some("a"), Some("b"), Some("b"), None]),
        strs(&[Some("b"), Some("c"), Some("c"), Some("d"), None]),
    ]];
    let ordered = |orders: Vec<(bool, bool)>, keys: Vec<usize>| {
        let e = AggregateExpr::listagg(
            AggregateFunction::ListAgg {
                delimiter: Vec::new(),
                orders,
            },
            0,
            DataType::Utf8View,
            keys.clone(),
            keys.iter().map(|_| DataType::Utf8View).collect(),
            "x",
        );
        mrun(node(scan(&s, b.clone()), spec(&s, vec![], vec![e])))
    };
    // `ORDER BY col1 DESC`.
    assert_eq!(ordered(vec![(true, false)], vec![0]).await, ["bbaa"]);
    // `ORDER BY col1` (ascending).
    assert_eq!(ordered(vec![(false, true)], vec![0]).await, ["aabb"]);
}

/// One partition and eight partitions agree exactly, ordered or not.
#[tokio::test]
async fn listagg_partitions_agree() {
    let s = schema(vec![("a", DataType::Utf8View), ("b", DataType::Utf8View)]);
    let one = vec![
        strs(&[Some("a"), Some("a"), Some("b"), Some("b"), None]),
        strs(&[Some("b"), Some("c"), Some("c"), Some("d"), None]),
    ];
    let multi: Vec<Vec<ArrayRef>> = (0..4).map(|_| one.clone()).collect();
    let plan = |orders: Vec<(bool, bool)>| {
        let (keys, key_types) = if orders.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            (vec![0], vec![DataType::Utf8View])
        };
        let e = AggregateExpr::listagg(
            AggregateFunction::ListAgg {
                delimiter: b"|".to_vec(),
                orders,
            },
            1,
            DataType::Utf8View,
            keys,
            key_types,
            "x",
        );
        node(scan(&s, multi.clone()), spec(&s, vec![], vec![e]))
    };
    assert_eq!(slots(plan(vec![]), 1).await, slots(plan(vec![]), 8).await);
    assert_eq!(
        slots(plan(vec![(true, false)]), 1).await,
        slots(plan(vec![(true, false)]), 8).await
    );
}

/// `DISTINCT` dedups before the accumulator runs (the planner's
/// distinct rewrite, as Spark does it): repeated values join once, and
/// an order-free aggregate agrees whatever the dedup order is.
#[tokio::test]
async fn distinct_dedups_before_the_accumulator() {
    let s = schema(vec![("v", DataType::Utf8View)]);
    let input = || {
        scan(
            &s,
            vec![vec![strs(&[Some("x"), Some("x"), Some("x"), None])]],
        )
    };
    let e = AggregateExpr::listagg(
        AggregateFunction::ListAgg {
            delimiter: b"|".to_vec(),
            orders: Vec::new(),
        },
        0,
        DataType::Utf8View,
        Vec::new(),
        Vec::new(),
        "x",
    );
    let plan = distinct_plan(input(), spec(&s, vec![], vec![e]), &[true]).unwrap();
    assert_eq!(mrun(plan).await, ["x"]);
    // `bit_xor(DISTINCT b)` over `1, 1, 2` (NULL skipped) gives `3`.
    let s = schema(vec![("b", DataType::Int32)]);
    let input = || scan(&s, vec![vec![i32s(&[Some(1), Some(1), Some(2), None])]]);
    let plan = distinct_plan(
        input(),
        spec(&s, vec![], vec![agg(&BitXor, 0, &s)]),
        &[true],
    )
    .unwrap();
    assert_eq!(mrun(plan).await, ["3"]);
}

/// `try_sum` is `sum` with NULL instead of an overflow error: `bigint`
/// overflow gives NULL, ANSI on or off; decimals behave the same.
#[tokio::test]
async fn try_sum_overflows_to_null() {
    let s = schema(vec![("x", DataType::Int64)]);
    let one = |v: &[Option<i64>], ansi: bool| {
        let mut sp = spec(&s, vec![], vec![agg(&TrySum, 0, &s)]);
        sp.ansi = ansi;
        mrun(node(scan(&s, vec![vec![i64s(v)]]), sp))
    };
    assert_eq!(one(&[Some(5), Some(10), Some(15)], true).await, ["30"]);
    assert_eq!(one(&[None, Some(10), Some(15)], true).await, ["25"]);
    assert_eq!(one(&[None, None], true).await, ["NULL"]);
    assert_eq!(one(&[], true).await, ["NULL"]);
    assert_eq!(one(&[Some(i64::MAX), Some(1)], true).await, ["NULL"]);
    assert_eq!(one(&[Some(i64::MAX), Some(1)], false).await, ["NULL"]);
    // Decimal overflow is NULL too, ANSI on or off.
    let d = schema(vec![("x", DataType::Decimal128(38, 0))]);
    let big = "98765432109876543210987654321098765432"
        .parse::<i128>()
        .unwrap();
    let col: ArrayRef = Arc::new(
        arrow_array::Decimal128Array::from(vec![Some(big), Some(big)])
            .with_precision_and_scale(38, 0)
            .unwrap(),
    );
    for ansi in [true, false] {
        let mut sp = spec(&d, vec![], vec![agg(&TrySum, 0, &d)]);
        sp.ansi = ansi;
        let rows = mrun(node(scan(&d, vec![vec![Arc::clone(&col)]]), sp)).await;
        assert_eq!(rows, ["NULL"], "ansi {ansi}");
    }
}

/// `try_avg` is `avg` with NULL instead of an overflow error.
#[tokio::test]
async fn try_avg_overflows_to_null() {
    let s = schema(vec![("x", DataType::Int64)]);
    let one = |v: &[Option<i64>], ansi: bool| {
        let mut sp = spec(&s, vec![], vec![agg(&TryAvg, 0, &s)]);
        sp.ansi = ansi;
        mrun(node(scan(&s, vec![vec![i64s(v)]]), sp))
    };
    assert_eq!(one(&[Some(5), Some(10), Some(15)], true).await, ["10.0"]);
    assert_eq!(one(&[None, Some(10), Some(15)], true).await, ["12.5"]);
    assert_eq!(one(&[None, None], true).await, ["NULL"]);
    assert_eq!(one(&[], true).await, ["NULL"]);
    // `avg` of integers sums in `double`: no overflow here.
    assert_eq!(
        one(&[Some(i64::MAX), Some(1)], true).await,
        [format!("{:?}", (i64::MAX as f64 + 1.0) / 2.0)]
    );
    // Decimal overflow is NULL, ANSI on or off.
    let d = schema(vec![("x", DataType::Decimal128(38, 0))]);
    let big = "98765432109876543210987654321098765432"
        .parse::<i128>()
        .unwrap();
    let col: ArrayRef = Arc::new(
        arrow_array::Decimal128Array::from(vec![Some(big), Some(big)])
            .with_precision_and_scale(38, 0)
            .unwrap(),
    );
    for ansi in [true, false] {
        let mut sp = spec(&d, vec![], vec![agg(&TryAvg, 0, &d)]);
        sp.ansi = ansi;
        let rows = mrun(node(scan(&d, vec![vec![Arc::clone(&col)]]), sp)).await;
        assert_eq!(rows, ["NULL"], "ansi {ansi}");
    }
}

/// One partition and eight partitions agree exactly for `try_sum` and
/// `try_avg` (including sticky overflow).
#[tokio::test]
async fn try_partitions_agree() {
    let s = schema(vec![("x", DataType::Int64)]);
    let one = vec![i64s(&[Some(1), Some(2), None, Some(i64::MAX), Some(1)])];
    let multi: Vec<Vec<ArrayRef>> = (0..4).map(|_| one.clone()).collect();
    for f in [&TrySum, &TryAvg] {
        let plan = || {
            node(
                scan(&s, multi.clone()),
                spec(&s, vec![], vec![agg(f, 0, &s)]),
            )
        };
        assert_eq!(slots(plan(), 1).await, slots(plan(), 8).await, "{f:?}");
    }
    // The overflowed sum is NULL in every partitioning.
    assert_eq!(
        slots(
            node(
                scan(&s, multi.clone()),
                spec(&s, vec![], vec![agg(&TrySum, 0, &s)])
            ),
            8
        )
        .await,
        ["NULL"]
    );
}
