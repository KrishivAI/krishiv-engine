// SPDX-License-Identifier: Apache-2.0
//! `collect_list`, `collect_set` (`array_agg`, which the analyzer resolves
//! as `collect_list`) and the element-wise `vector_sum` / `vector_avg`:
//! Spark 4.2 semantics (NULLs skipped, empty groups, first-seen sets with
//! NaN equal to NaN, `VECTOR_DIMENSION_MISMATCH`), merges of partial
//! states (through the partial phase and `DISTINCT`'s two levels),
//! partition-index-ordered merges, spills and the memory accounting that
//! reserves them. Every test that runs a plan through `exec` also runs
//! with the row-layout tables.

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, BinaryViewArray, Float32Array, Int32Array, Int64Array, ListArray, RecordBatch,
    StringViewArray, StructArray,
};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field};

use super::super::accumulator::Accumulator;
use super::*;
use crate::operators::{Sort, SortKey};
use crate::planner::nodes::{BreakerNode, OperatorNode, SourceNode};
use crate::window::{
    FrameBound, FrameUnits, Window, WindowFrame, WindowFunction, WindowFunctionPlan, WindowPlan,
};
use AggregateFunction::{CollectList, CollectSet, VectorAvg, VectorSum};

/// One collect or vector aggregate over column `col` of `s`.
fn agg(f: &AggregateFunction, col: usize, s: &SchemaRef) -> AggregateExpr {
    let t = s.field(col).data_type().clone();
    AggregateExpr::new(f.clone(), col, t, format!("{f:?}_{col}"))
}

/// `(k, x)`: `k` groups (`1`: rows with a NULL and a repeat, `2`: all
/// NULL, `3`: one row).
fn collect_input() -> (SchemaRef, Vec<Vec<ArrayRef>>) {
    let s = schema(vec![("k", DataType::Int64), ("x", DataType::Int64)]);
    let k = i64s(&[
        Some(1),
        Some(1),
        Some(1),
        Some(1),
        Some(2),
        Some(2),
        Some(3),
    ]);
    let x = i64s(&[Some(1), None, Some(2), Some(1), None, None, Some(9)]);
    (s, vec![vec![k, x]])
}

/// A list column of `float` vectors (`None`: a NULL vector).
fn vectors(v: &[Option<Vec<Option<f32>>>]) -> ArrayRef {
    let mut offsets = Vec::with_capacity(v.len() + 1);
    offsets.push(0_i32);
    let mut values = Vec::new();
    let mut validity = Vec::with_capacity(v.len());
    for vec in v {
        match vec {
            None => validity.push(false),
            Some(items) => {
                validity.push(true);
                values.extend(items.iter().copied());
            }
        }
        offsets.push(i32::try_from(values.len()).unwrap());
    }
    Arc::new(ListArray::new(
        Arc::new(Field::new("element", DataType::Float32, true)),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(Float32Array::from(values)),
        Some(NullBuffer::from(validity)),
    ))
}

fn vector_schema() -> SchemaRef {
    schema(vec![
        ("k", DataType::Int64),
        (
            "v",
            DataType::List(Arc::new(Field::new("element", DataType::Float32, true))),
        ),
    ])
}

/// `(k, v)`: `k` groups, `v` lists of floats with a NULL, an empty and a
/// NULL vector in group `1`, all-NULL vectors in group `2`, one row in
/// group `3`.
fn vector_input() -> (SchemaRef, Vec<Vec<ArrayRef>>) {
    let s = vector_schema();
    let k = i64s(&[
        Some(1),
        Some(1),
        Some(1),
        Some(1),
        Some(2),
        Some(2),
        Some(3),
    ]);
    #[rustfmt::skip]
    let v = vectors(&[
        Some(vec![Some(1.0), Some(2.0)]),
        Some(vec![Some(3.0), Some(4.0)]),
        None,
        Some(vec![]),
        None,
        None,
        Some(vec![Some(9.0)]),
    ]);
    (s, vec![vec![k, v]])
}

/// `a[row]` as Spark renders it: lists as `[1, 2]`, NULL as `NULL`.
fn shown(a: &ArrayRef, row: usize) -> String {
    if a.is_null(row) {
        return "NULL".to_owned();
    }
    match a.data_type() {
        DataType::List(_) => {
            let l = a.as_any().downcast_ref::<ListArray>().unwrap();
            let offsets = l.value_offsets();
            let start = usize::try_from(offsets[row].max(0)).unwrap_or(0);
            let len = usize::try_from(l.value_length(row).max(0)).unwrap_or(0);
            let v = l.values().slice(start, len);
            format!(
                "[{}]",
                (0..v.len())
                    .map(|i| shown(&v, i))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        DataType::Int32 => a
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(row)
            .to_string(),
        DataType::Int64 => a.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::Decimal128(_, s) => super::super::accumulator::plain(
            a.as_primitive::<arrow_array::types::Decimal128Type>()
                .value(row),
            *s,
        ),
        DataType::Float32 => format!("{:?}", a.as_primitive::<Float32Type>().value(row)),
        DataType::Float64 => format!("{:?}", a.as_primitive::<Float64Type>().value(row)),
        DataType::Utf8View => a.as_string_view().value(row).to_owned(),
        DataType::LargeUtf8 => a.as_string::<i64>().value(row).to_owned(),
        DataType::BinaryView => String::from_utf8_lossy(a.as_binary_view().value(row)).into_owned(),
        _ => format!("{:?}", a.slice(row, 1)).replace('\n', " "),
    }
}

/// Rows of `batches` as `a|b|c`, in order.
fn in_order(batches: &[Batch]) -> Vec<String> {
    batches
        .iter()
        .flat_map(|b| {
            let d = b.data();
            (0..d.num_rows()).map(move |r| {
                d.columns()
                    .iter()
                    .map(|c| shown(c, r))
                    .collect::<Vec<_>>()
                    .join("|")
            })
        })
        .collect()
}

/// Rows of `plan`'s output as `a|b|c`, sorted (lists render as
/// `[1, 2]`, unlike the debug format of the shared `rows` helper).
async fn sorted_rows(plan: Arc<dyn ExecNode>) -> Vec<String> {
    let mut out = in_order(&exec(plan, Opts::default()).await.unwrap().0);
    out.sort();
    out
}

/// Runs `plan` with one driver, keeping the row order.
async fn run_ordered(plan: Arc<dyn ExecNode>) -> Vec<String> {
    run_ordered_layout(plan, true).await
}

/// Runs `plan` with one driver, keeping the row order, with the
/// row-layout tables on (`rows`) or off (both checked by default).
async fn run_ordered_layout(plan: Arc<dyn ExecNode>, rows: bool) -> Vec<String> {
    let o = Opts {
        slots: 1,
        both_layouts: rows,
        ..Opts::default()
    };
    in_order(&exec(plan, o).await.unwrap().0)
}

/// `collect_list` keeps input order, skips NULLs and gives `[]` (never
/// NULL) for empty and all-NULL groups.
#[tokio::test]
async fn collect_list_keeps_input_order_and_skips_nulls() {
    let (s, b) = collect_input();
    let single = || {
        node(
            scan(&s, b.clone()),
            spec(&s, vec![0], vec![agg(&CollectList, 1, &s)]),
        )
    };
    assert_eq!(
        run_ordered(single()).await,
        ["1|[1, 2, 1]", "2|[]", "3|[9]"]
    );
    // Global: one row; `[]` over no rows.
    let global = || {
        node(
            scan(&s, b.clone()),
            spec(&s, vec![], vec![agg(&CollectList, 1, &s)]),
        )
    };
    assert_eq!(run_ordered(global()).await, ["[1, 2, 1, 9]"]);
    let none = node(
        scan(&s, vec![vec![i64s(&[]), i64s(&[])]]),
        spec(&s, vec![], vec![agg(&CollectList, 1, &s)]),
    );
    assert_eq!(run_ordered(none).await, ["[]"]);
}

/// `collect_set` dedups in first-seen order, skips NULLs and gives `[]`
/// for empty and all-NULL groups.
#[tokio::test]
async fn collect_set_dedups_in_first_seen_order() {
    let (s, b) = collect_input();
    let single = || {
        node(
            scan(&s, b.clone()),
            spec(&s, vec![0], vec![agg(&CollectSet, 1, &s)]),
        )
    };
    assert_eq!(run_ordered(single()).await, ["1|[1, 2]", "2|[]", "3|[9]"]);
    let global = || {
        node(
            scan(&s, b.clone()),
            spec(&s, vec![], vec![agg(&CollectSet, 1, &s)]),
        )
    };
    assert_eq!(run_ordered(global()).await, ["[1, 2, 9]"]);
}

/// NaN equals NaN and `-0.0` equals `0.0` in the set, as for grouping
/// keys (the shared key encoding); the list keeps every row.
#[tokio::test]
async fn collect_set_normalizes_floats_like_grouping_keys() {
    let s = schema(vec![("k", DataType::Int64), ("x", DataType::Float64)]);
    let b = vec![vec![
        i64s(&[Some(1), Some(1), Some(1), Some(1), Some(1)]),
        f64s(vec![
            Some(1.0),
            Some(f64::NAN),
            Some(f64::NAN),
            Some(-0.0),
            Some(0.0),
        ]),
    ]];
    let single = || {
        node(
            scan(&s, b.clone()),
            spec(&s, vec![0], vec![agg(&CollectSet, 1, &s)]),
        )
    };
    assert_eq!(
        run_ordered(single()).await,
        ["1|[1.0, NaN, -0.0]"],
        "first-seen order keeps the -0.0 that arrived first"
    );
    let single = || {
        node(
            scan(&s, b.clone()),
            spec(&s, vec![0], vec![agg(&CollectList, 1, &s)]),
        )
    };
    assert_eq!(
        run_ordered(single()).await,
        ["1|[1.0, NaN, NaN, -0.0, 0.0]"]
    );
}

/// `vector_sum` adds element-wise: NULL vectors are skipped, all-NULL is
/// NULL, empty vectors are neutral but counted.
#[tokio::test]
async fn vector_sum_adds_element_wise() {
    let (s, b) = vector_input();
    let single = || {
        node(
            scan(&s, b.clone()),
            spec(&s, vec![0], vec![agg(&VectorSum, 1, &s)]),
        )
    };
    assert_eq!(
        run_ordered_layout(single(), false).await,
        ["1|[4.0, 6.0]", "2|NULL", "3|[9.0]"]
    );
    // Global over one dimension.
    let s = vector_schema();
    let b = vec![vec![
        i64s(&[Some(1), Some(1)]),
        vectors(&[
            Some(vec![Some(10.0), Some(20.0)]),
            Some(vec![Some(3.0), Some(4.0)]),
        ]),
    ]];
    let global = || {
        node(
            scan(&s, b.clone()),
            spec(&s, vec![], vec![agg(&VectorSum, 1, &s)]),
        )
    };
    assert_eq!(run_ordered_layout(global(), false).await, ["[13.0, 24.0]"]);
}

/// A global aggregate over vectors of different dimensions is
/// `VECTOR_DIMENSION_MISMATCH` (group 1 has dimension 2, group 3 has
/// dimension 1).
#[tokio::test]
async fn vector_global_mix_of_dimensions_is_an_error() {
    let (s, b) = vector_input();
    let e = exec(
        node(
            scan(&s, b.clone()),
            spec(&s, vec![], vec![agg(&VectorSum, 1, &s)]),
        ),
        Opts {
            slots: 1,
            both_layouts: false,
            ..Opts::default()
        },
    )
    .await
    .err()
    .map(|e| (e.error_class(), e.to_string()));
    let (class, text) = e.expect("a dimension mix fails");
    assert_eq!(class, ErrorClass::VectorDimensionMismatch, "{text}");
}

/// `vector_avg` divides by the count of non-NULL vectors (empty vectors
/// count): `[1, 2]`, `[3, 4]` and `[]` give `[4/3, 6/3]`.
#[tokio::test]
async fn vector_avg_divides_by_non_null_vectors() {
    let (s, b) = vector_input();
    let single = || {
        node(
            scan(&s, b.clone()),
            spec(&s, vec![0], vec![agg(&VectorAvg, 1, &s)]),
        )
    };
    assert_eq!(
        run_ordered(single()).await,
        ["1|[1.3333334, 2.0]", "2|NULL", "3|[9.0]"]
    );
}

/// An empty-only group is an empty array, not NULL (and not an error).
#[tokio::test]
async fn empty_vectors_give_an_empty_array() {
    let s = vector_schema();
    let b = vec![vec![
        i64s(&[Some(1), Some(1)]),
        vectors(&[Some(vec![]), Some(vec![])]),
    ]];
    for f in [&VectorSum, &VectorAvg] {
        let single = || node(scan(&s, b.clone()), spec(&s, vec![0], vec![agg(f, 1, &s)]));
        assert_eq!(run_ordered(single()).await, ["1|[]"], "{f:?}");
    }
}

/// Vectors of different non-zero dimensions are
/// `VECTOR_DIMENSION_MISMATCH`, on update and on merge of partial states.
#[tokio::test]
async fn vector_dimension_mismatch_is_an_error() {
    let s = vector_schema();
    let two = vectors(&[Some(vec![Some(1.0), Some(2.0)])]);
    let three = vectors(&[Some(vec![Some(1.0), Some(2.0), Some(3.0)])]);
    let both = arrow_select::concat::concat(&[&two, &three]).unwrap();
    let b = vec![vec![i64s(&[Some(1), Some(1)]), both]];
    for f in [&VectorSum, &VectorAvg] {
        let e = exec(
            node(scan(&s, b.clone()), spec(&s, vec![0], vec![agg(f, 1, &s)])),
            Opts {
                slots: 1,
                both_layouts: false,
                ..Opts::default()
            },
        )
        .await
        .err()
        .map(|e| (e.error_class(), e.to_string()));
        let (class, text) = e.expect("a dimension mismatch fails");
        assert_eq!(class, ErrorClass::VectorDimensionMismatch, "{text}");
        assert!(text.contains("`vector_"), "{text}");
    }
    // Through partial states: states of different dimensions do not merge.
    let mut first = spec(&s, vec![0], vec![agg(&VectorSum, 1, &s)]);
    first.output = AggregateOutput::Partial;
    let second = second_phase(&first, None);
    let partial = PartialAggregate::new(first, BypassPolicy::default(), Some(7)).unwrap();
    let plan = node(PartialNode::new(scan(&s, b), partial), second);
    let e = exec(
        plan,
        Opts {
            slots: 1,
            both_layouts: false,
            ..Opts::default()
        },
    )
    .await
    .err()
    .map(|e| e.error_class());
    assert_eq!(e, Some(ErrorClass::VectorDimensionMismatch));
}

/// One partition and eight partitions give the same values: exactly for
/// the order-insensitive vector aggregates, after sorting for the
/// order-dependent collect ones.
#[tokio::test]
async fn one_and_eight_partitions_agree() {
    // Several batches so eight drivers each fold rows (one batch is one
    // morsel: a single batch would run on one driver either way).
    let (s, one) = collect_input();
    let multi: Vec<Vec<ArrayRef>> = (0..4).map(|_| one[0].clone()).collect();
    for f in [&CollectList, &CollectSet] {
        let single = || {
            node(
                scan(&s, multi.clone()),
                spec(&s, vec![0], vec![agg(f, 1, &s)]),
            )
        };
        let one_rows = run_ordered(single()).await;
        let eight = || {
            node(
                scan(&s, multi.clone()),
                spec(&s, vec![0], vec![agg(f, 1, &s)]),
            )
        };
        let eight_rows = run_ordered_slots(eight(), 8).await;
        // Same values, whatever the thread timing; the order is the next
        // test's subject.
        let (mut a, mut b) = (one_rows.clone(), eight_rows.clone());
        a.sort();
        b.sort();
        assert_eq!(a, b, "{f:?}");
        // And the eight-partition run is deterministic: the same seed
        // gives the same row order twice.
        assert_eq!(eight_rows, run_ordered_slots(eight(), 8).await, "{f:?}");
    }
    let (s, one) = vector_input();
    let multi: Vec<Vec<ArrayRef>> = (0..4).map(|_| one[0].clone()).collect();
    for f in [&VectorSum, &VectorAvg] {
        let single = || {
            node(
                scan(&s, multi.clone()),
                spec(&s, vec![0], vec![agg(f, 1, &s)]),
            )
        };
        let one_rows = run_ordered(single()).await;
        let eight_rows = run_ordered_slots(single(), 8).await;
        assert_eq!(one_rows, eight_rows, "{f:?}");
    }
    // Global too: one row, exact.
    let (s, one) = collect_input();
    let multi: Vec<Vec<ArrayRef>> = (0..4).map(|_| one[0].clone()).collect();
    let global = || {
        node(
            scan(&s, multi.clone()),
            spec(&s, vec![], vec![agg(&CollectList, 1, &s)]),
        )
    };
    let one_rows = run_ordered(global()).await;
    let eight_rows = run_ordered_slots(global(), 8).await;
    let (mut a, mut b) = (one_rows, eight_rows);
    a.sort();
    b.sort();
    assert_eq!(a, b);
}

/// Runs `plan` with `slots` drivers, keeping the row order (the default
/// layout only: both-layouts equivalence is covered elsewhere).
async fn run_ordered_slots(plan: Arc<dyn ExecNode>, slots: usize) -> Vec<String> {
    let o = Opts {
        slots,
        both_layouts: false,
        ..Opts::default()
    };
    in_order(&exec(plan, o).await.unwrap().0)
}

/// Merging partial states in partition-index order concatenates them, so
/// multi-partition `collect_list` is deterministic: whatever thread folds
/// first, partitions merge by index (the sink sorts its tables and runs
/// by `partition`, see `Run::partition`).
#[tokio::test]
async fn collect_merges_follow_partition_index_order() {
    use krishiv_types::batch::Weights;
    let raw = AggregateExpr::new(CollectList, 0, DataType::Int64, "c");
    let fold = |vals: &[Option<i64>]| {
        let mut a = Accumulator::new(&raw, true).unwrap();
        a.resize(1);
        a.update(&vec![0; vals.len()], &[i64s(vals)], &Weights::Const(1))
            .unwrap();
        a.state(&[0]).unwrap()
    };
    let left = fold(&[Some(1), None, Some(2)]);
    let right = fold(&[Some(3), Some(1)]);
    let merging = raw.merging(0);
    let merge = |first: &[ArrayRef], second: &[ArrayRef]| {
        let mut f = Accumulator::new(&merging, true).unwrap();
        f.resize(1);
        f.merge(&[0], first).unwrap();
        f.merge(&[0], second).unwrap();
        f.evaluate(&[0]).unwrap()
    };
    // Partition order: left then right.
    let got = merge(&left, &right);
    assert_eq!(shown(&got, 0), "[1, 2, 3, 1]");
    // The reverse merge concatenates in reverse: the merge order is what
    // the sink's partition index fixes, not thread timing.
    let got = merge(&right, &left);
    assert_eq!(shown(&got, 0), "[3, 1, 1, 2]");
    // The set dedups across partitions in first-seen order.
    let raw = AggregateExpr::new(CollectSet, 0, DataType::Int64, "c");
    let fold = |vals: &[Option<i64>]| {
        let mut a = Accumulator::new(&raw, true).unwrap();
        a.resize(1);
        a.update(&vec![0; vals.len()], &[i64s(vals)], &Weights::Const(1))
            .unwrap();
        a.state(&[0]).unwrap()
    };
    let left = fold(&[Some(1), Some(2)]);
    let right = fold(&[Some(2), None, Some(3)]);
    let merging = raw.merging(0);
    let mut f = Accumulator::new(&merging, true).unwrap();
    f.resize(1);
    f.merge(&[0], &left).unwrap();
    f.merge(&[0], &right).unwrap();
    assert_eq!(shown(&f.evaluate(&[0]).unwrap(), 0), "[1, 2, 3]");
    // The vector sums merge element-wise with the counts.
    let floats = || DataType::List(Arc::new(Field::new("element", DataType::Float32, true)));
    let raw = AggregateExpr::new(VectorSum, 0, floats(), "v");
    let fold = |rows: &[Option<Vec<Option<f32>>>]| {
        let mut a = Accumulator::new(&raw, true).unwrap();
        a.resize(1);
        a.update(&vec![0; rows.len()], &[vectors(rows)], &Weights::Const(1))
            .unwrap();
        a.state(&[0]).unwrap()
    };
    let left = fold(&[Some(vec![Some(1.0), Some(2.0)]), None]);
    let right = fold(&[Some(vec![Some(3.0), Some(4.0)])]);
    let merging = raw.merging(0);
    let mut f = Accumulator::new(&merging, true).unwrap();
    f.resize(1);
    f.merge(&[0], &left).unwrap();
    f.merge(&[0], &right).unwrap();
    let got = f.evaluate(&[0]).unwrap();
    assert_eq!(shown(&got, 0), "[4.0, 6.0]");
}

/// `collect_list(DISTINCT x)` skips NULLs and repeats through the
/// distinct rewrite; `collect_set` needs none (it dedups itself). A
/// `DISTINCT` pair over two argument lists is refused, as for the other
/// aggregates.
#[tokio::test]
async fn distinct_collects_skip_nulls_and_repeats() {
    let (s, b) = collect_input();
    let sp = spec(&s, vec![0], vec![agg(&CollectList, 1, &s)]);
    let plan = distinct_plan(scan(&s, b.clone()), sp, &[true]).unwrap();
    assert_eq!(sorted_rows(plan).await, ["1|[1, 2]", "2|[]", "3|[9]"]);
    // Global distinct.
    let sp = spec(&s, vec![], vec![agg(&CollectSet, 1, &s)]);
    let plan = distinct_plan(scan(&s, b.clone()), sp, &[true]).unwrap();
    assert_eq!(sorted_rows(plan).await, ["[1, 2, 9]"]);
    // Two distinct argument lists: one rewrite handles one list only.
    let sp = spec(
        &s,
        vec![0],
        vec![agg(&CollectList, 0, &s), agg(&CollectList, 1, &s)],
    );
    let e = distinct_plan(scan(&s, b), sp, &[true, true]).err();
    assert_eq!(
        e.map(|e| e.error_class()),
        Some(ErrorClass::KrishivUnsupportedFeature)
    );
}

/// Rows with weights other than 1 are refused, as for the other
/// order-sensitive aggregates (the planner keeps `FILTER (WHERE ...)`
/// out: it never reaches the accumulator).
#[tokio::test]
async fn collect_and_vector_reject_non_unit_weights() {
    use arrow_array::Int64Array;
    let s = schema(vec![("k", DataType::Int64), ("x", DataType::Int64)]);
    let cols = vec![i64s(&[Some(1), Some(1)]), i64s(&[Some(10), Some(5)])];
    let data = RecordBatch::try_new(Arc::clone(&s), cols).unwrap();
    let weighted = |f: &AggregateFunction| {
        let b = Batch::with_weights(data.clone(), Weights::Column(Int64Array::from(vec![2, -1])))
            .unwrap();
        node(
            Arc::new(Scan(Batches::new(Arc::clone(&s), vec![b]))),
            spec(&s, vec![0], vec![agg(f, 1, &s)]),
        )
    };
    for f in [&CollectList, &CollectSet] {
        let e = exec(weighted(f), Opts::default()).await.err();
        assert_eq!(
            e.map(|e| e.error_class()),
            Some(ErrorClass::KrishivUnsupportedFeature),
            "{f:?}"
        );
    }
    let (s, b) = vector_input();
    let cols = b.into_iter().next().unwrap();
    let data = RecordBatch::try_new(Arc::clone(&s), cols).unwrap();
    let b = Batch::with_weights(data, Weights::Column(Int64Array::from(vec![1; 7]))).unwrap();
    let unit = node(
        Arc::new(Scan(Batches::new(Arc::clone(&s), vec![b]))),
        spec(&s, vec![0], vec![agg(&VectorSum, 1, &s)]),
    );
    assert!(exec(unit, Opts::default()).await.is_ok());
}

/// Nested element types: strings (both carriers), decimals, arrays and
/// structs collect in order; the set dedups them by the shared key
/// encoding.
#[tokio::test]
async fn nested_element_types_collect() {
    // Strings, `Utf8` carrier against a `Utf8View` argument (what the
    // planner types): the chunks convert, never silently.
    let views = schema(vec![("k", DataType::Int64), ("x", DataType::Utf8View)]);
    let raw = schema(vec![("k", DataType::Int64), ("x", DataType::Utf8)]);
    let k = || i64s(&[Some(1), Some(1), Some(1), None, Some(1), None]);
    let words = || {
        Arc::new(StringViewArray::from_iter(vec![
            Some("b"),
            Some("a"),
            Some("b"),
            Some("z"),
            None,
            Some("a"),
        ])) as ArrayRef
    };
    let b = vec![vec![k(), words()]];
    let single = || {
        node(
            scan(&views, b.clone()),
            spec(&views, vec![0], vec![agg(&CollectList, 1, &views)]),
        )
    };
    assert_eq!(run_ordered(single()).await, ["1|[b, a, b]", "NULL|[z, a]"]);
    let single = || {
        node(
            scan(&views, b.clone()),
            spec(&views, vec![0], vec![agg(&CollectSet, 1, &views)]),
        )
    };
    assert_eq!(run_ordered(single()).await, ["1|[b, a]", "NULL|[z, a]"]);
    // The `Utf8` carrier normalizes to views on append.
    let strings = || {
        Arc::new(arrow_array::StringArray::from(vec![
            Some("b"),
            Some("a"),
            Some("b"),
            Some("z"),
            None,
            Some("a"),
        ])) as ArrayRef
    };
    let b = vec![vec![k(), strings()]];
    let single = || {
        node(
            scan(&raw, b.clone()),
            spec(&views, vec![0], vec![agg(&CollectSet, 1, &views)]),
        )
    };
    assert_eq!(run_ordered(single()).await, ["1|[b, a]", "NULL|[z, a]"]);
    // Binary views dedup by bytes; large strings collect in order.
    let bin = schema(vec![("k", DataType::Int64), ("x", DataType::BinaryView)]);
    let bytes = || {
        Arc::new(BinaryViewArray::from_iter(vec![
            Some(b"b".as_slice()),
            Some(b"a".as_slice()),
            Some(b"b".as_slice()),
            Some(b"z".as_slice()),
            None,
            Some(b"a".as_slice()),
        ])) as ArrayRef
    };
    let b = vec![vec![k(), bytes()]];
    let single = || {
        node(
            scan(&bin, b.clone()),
            spec(&bin, vec![0], vec![agg(&CollectSet, 1, &bin)]),
        )
    };
    assert_eq!(run_ordered(single()).await, ["1|[b, a]", "NULL|[z, a]"]);
    let large = schema(vec![("k", DataType::Int64), ("x", DataType::LargeUtf8)]);
    let big = || {
        Arc::new(arrow_array::LargeStringArray::from(vec![
            Some("bb"),
            Some("a"),
            Some("bb"),
            Some("z"),
            None,
            Some("a"),
        ])) as ArrayRef
    };
    let b = vec![vec![k(), big()]];
    let single = || {
        node(
            scan(&large, b.clone()),
            spec(&large, vec![0], vec![agg(&CollectList, 1, &large)]),
        )
    };
    assert_eq!(
        run_ordered(single()).await,
        ["1|[bb, a, bb]", "NULL|[z, a]"]
    );
    // Decimals.
    let ds = schema(vec![
        ("k", DataType::Int64),
        ("x", DataType::Decimal128(10, 2)),
    ]);
    let b = vec![vec![
        i64s(&[Some(1), Some(1), Some(1), None]),
        dec(&["1.50", "2.00", "1.50", "9.99"], 10, 2),
    ]];
    let single = || {
        node(
            scan(&ds, b.clone()),
            spec(&ds, vec![], vec![agg(&CollectSet, 1, &ds)]),
        )
    };
    assert_eq!(run_ordered(single()).await, ["[1.50, 2.00, 9.99]"]);
    // Arrays of `int`.
    let arrays = schema(vec![
        ("k", DataType::Int64),
        (
            "x",
            DataType::List(Arc::new(Field::new("element", DataType::Int32, true))),
        ),
    ]);
    let b = vec![vec![
        i64s(&[Some(1), Some(1), Some(1), Some(1)]),
        lists(&[Some(vec![1, 2]), Some(vec![3]), Some(vec![1, 2]), None]),
    ]];
    let single = || {
        node(
            scan(&arrays, b.clone()),
            spec(&arrays, vec![], vec![agg(&CollectList, 1, &arrays)]),
        )
    };
    assert_eq!(sorted_rows(single()).await, ["[[1, 2], [3], [1, 2]]"]);
    let single = || {
        node(
            scan(&arrays, b.clone()),
            spec(&arrays, vec![], vec![agg(&CollectSet, 1, &arrays)]),
        )
    };
    assert_eq!(sorted_rows(single()).await, ["[[1, 2], [3]]"]);
    // Structs: the list keeps every row; compare the collected `a` field.
    let struct_type = DataType::Struct(
        vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Utf8View, true),
        ]
        .into(),
    );
    let ts = schema(vec![("k", DataType::Int64), ("x", struct_type.clone())]);
    let b = vec![vec![
        i64s(&[Some(1), Some(1), Some(1)]),
        structs(
            &[Some(1), Some(2), Some(1)],
            &[Some("x"), Some("y"), Some("x")],
        ),
    ]];
    let out = exec(
        node(
            scan(&ts, b.clone()),
            spec(&ts, vec![], vec![agg(&CollectList, 1, &ts)]),
        ),
        Opts {
            slots: 1,
            both_layouts: false,
            ..Opts::default()
        },
    )
    .await
    .unwrap()
    .0;
    let col = &out[0].data().columns()[0];
    let l = col.as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(l.value_length(0), 3);
    let inner = l.values().as_any().downcast_ref::<StructArray>().unwrap();
    let a = inner
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(a.values(), &[1, 2, 1]);
    // A set over floats nested in arrays has no shared encoding: refused,
    // never silently grouped incorrectly.
    let nested = DataType::List(Arc::new(Field::new("element", DataType::Float64, true)));
    let e = AggregateExpr::new(CollectSet, 0, nested, "c").result_type();
    assert_eq!(
        e.map_err(|e| e.error_class()),
        Err(ErrorClass::KrishivUnsupportedFeature)
    );
}

/// A `list<int>` column of `rows` (`None`: NULL).
fn lists(rows: &[Option<Vec<i32>>]) -> ArrayRef {
    let mut offsets = vec![0_i32];
    let mut values = Vec::new();
    let mut validity = Vec::with_capacity(rows.len());
    for r in rows {
        match r {
            None => validity.push(false),
            Some(v) => {
                validity.push(true);
                values.extend_from_slice(v);
            }
        }
        offsets.push(i32::try_from(values.len()).unwrap());
    }
    Arc::new(ListArray::new(
        Arc::new(Field::new("element", DataType::Int32, true)),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(Int32Array::from(values)),
        Some(NullBuffer::from(validity)),
    ))
}

/// A struct `{a, b}` column.
fn structs(a: &[Option<i64>], b: &[Option<&str>]) -> ArrayRef {
    Arc::new(StructArray::from(vec![
        (Arc::new(Field::new("a", DataType::Int64, true)), i64s(a)),
        (
            Arc::new(Field::new("b", DataType::Utf8View, true)),
            Arc::new(StringViewArray::from_iter(b.iter().copied())) as ArrayRef,
        ),
    ]))
}

/// `collect_list` and `vector_sum` as window aggregates: the whole
/// partition, a running frame and a bounded `ROWS` frame fold rows in
/// order and skip NULLs, through the same accumulator interface the
/// other aggregates use.
#[tokio::test]
async fn collect_and_vector_over_window_frames() {
    use futures::StreamExt as _;
    let s = schema(vec![
        ("p", DataType::Int64),
        ("o", DataType::Int64),
        ("x", DataType::Int64),
    ]);
    let batches = vec![Batch::new(
        RecordBatch::try_new(
            Arc::clone(&s),
            vec![
                i64s(&[Some(1), Some(1), Some(1), Some(2), Some(2)]),
                i64s(&[Some(0), Some(1), Some(2), Some(0), Some(1)]),
                i64s(&[Some(1), None, Some(3), Some(7), Some(7)]),
            ],
        )
        .unwrap(),
    )];
    let collect = |frame: WindowFrame, name: &str| WindowFunctionPlan {
        function: WindowFunction::Aggregate(AggregateExpr::new(
            CollectList,
            2,
            DataType::Int64,
            name.to_owned(),
        )),
        frame,
        filter: None,
        name: name.to_owned(),
    };
    let plan = WindowPlan {
        partition_by: vec![0],
        order_by: vec![SortKey::asc(1)],
        functions: vec![
            collect(WindowFrame::running(FrameUnits::Rows), "c_run"),
            collect(WindowFrame::whole(FrameUnits::Rows), "c_whole"),
            collect(
                WindowFrame {
                    units: FrameUnits::Rows,
                    lower: FrameBound::Offset(-1),
                    upper: FrameBound::Offset(1),
                },
                "c_bounded",
            ),
        ],
    };
    let keys: Vec<SortKey> = vec![SortKey::asc(0), SortKey::asc(1)];
    let sort = Arc::new(Sort::new(Arc::clone(&s), &keys, None).unwrap());
    let op = Arc::new(Window::new(Arc::clone(&s), plan, true, None).unwrap());
    let plan: Arc<dyn ExecNode> = Arc::new(OperatorNode {
        input: Arc::new(BreakerNode {
            input: Arc::new(SourceNode(crate::testing::Batches::new(
                Arc::clone(&s),
                batches,
            ))),
            sink: Arc::clone(&sort) as Arc<dyn crate::Sink>,
            source: sort,
            ordered: true,
        }),
        op,
    });
    let runner = QueryRunner::new(TokioExecutor::deterministic(2).unwrap());
    let root = pool(1 << 30);
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("q", Some(1 << 30)),
    );
    let mut stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
    let mut out = Vec::new();
    while let Some(b) = stream.next().await {
        out.push(b.unwrap());
    }
    drop((stream, q));
    assert_eq!(root.reserved(), 0);
    // Input columns, then the three frames; the partitions sort by `o`.
    assert_eq!(
        in_order(&out),
        [
            "1|0|1|[1]|[1, 3]|[1]",
            "1|1|NULL|[1]|[1, 3]|[1, 3]",
            "1|2|3|[1, 3]|[1, 3]|[3]",
            "2|0|7|[7]|[7, 7]|[7, 7]",
            "2|1|7|[7, 7]|[7, 7]|[7, 7]",
        ]
    );
}

/// `vector_sum` over a whole-partition window frame: NULL vectors are
/// skipped, the sums accumulate in order.
#[tokio::test]
async fn vector_sum_over_a_window_partition() {
    use futures::StreamExt as _;
    let s = schema(vec![
        ("p", DataType::Int64),
        ("o", DataType::Int64),
        (
            "v",
            DataType::List(Arc::new(Field::new("element", DataType::Float32, true))),
        ),
    ]);
    let batches = vec![Batch::new(
        RecordBatch::try_new(
            Arc::clone(&s),
            vec![
                i64s(&[Some(1), Some(1), Some(1)]),
                i64s(&[Some(0), Some(1), Some(2)]),
                vectors(&[
                    Some(vec![Some(1.0), Some(2.0)]),
                    None,
                    Some(vec![Some(3.0), Some(4.0)]),
                ]),
            ],
        )
        .unwrap(),
    )];
    let plan = WindowPlan {
        partition_by: vec![0],
        order_by: vec![SortKey::asc(1)],
        functions: vec![WindowFunctionPlan {
            function: WindowFunction::Aggregate(AggregateExpr::new(
                VectorSum,
                2,
                DataType::List(Arc::new(Field::new("element", DataType::Float32, true))),
                "v",
            )),
            frame: WindowFrame::whole(FrameUnits::Rows),
            filter: None,
            name: "v".to_owned(),
        }],
    };
    let keys: Vec<SortKey> = vec![SortKey::asc(0), SortKey::asc(1)];
    let sort = Arc::new(Sort::new(Arc::clone(&s), &keys, None).unwrap());
    let op = Arc::new(Window::new(Arc::clone(&s), plan, true, None).unwrap());
    let plan: Arc<dyn ExecNode> = Arc::new(OperatorNode {
        input: Arc::new(BreakerNode {
            input: Arc::new(SourceNode(crate::testing::Batches::new(
                Arc::clone(&s),
                batches,
            ))),
            sink: Arc::clone(&sort) as Arc<dyn crate::Sink>,
            source: sort,
            ordered: true,
        }),
        op,
    });
    let runner = QueryRunner::new(TokioExecutor::deterministic(2).unwrap());
    let root = pool(1 << 30);
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("q", Some(1 << 30)),
    );
    let mut stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
    let mut out = Vec::new();
    while let Some(b) = stream.next().await {
        out.push(b.unwrap());
    }
    drop((stream, q));
    assert_eq!(root.reserved(), 0);
    assert_eq!(
        in_order(&out),
        [
            "1|0|[1.0, 2.0]|[4.0, 6.0]",
            "1|1|NULL|[4.0, 6.0]",
            "1|2|[3.0, 4.0]|[4.0, 6.0]",
        ]
    );
}

/// A large `collect_list` under a tiny budget spills its variable-width
/// state and still returns every value; without a spill directory the
/// reservation failure is a clean memory error. Either way the pool is
/// empty afterwards (the reservation never leaks).
#[tokio::test]
async fn collect_list_spills_and_recovers_memory() {
    // 2k groups of 10 values of ~30-byte strings in batches of 1000:
    // each batch reserves a little, the table outgrows a small budget and
    // spills along the way (the pool keeps 512KB of spill scratch: a 2MB
    // pool leaves ~1.5MB usable for the ~1.8MB table).
    let s = schema(vec![("k", DataType::Int64), ("x", DataType::Utf8View)]);
    let n = 4_000;
    let per = 10;
    let b: Vec<Vec<ArrayRef>> = (0..n * per)
        .collect::<Vec<_>>()
        .chunks(1000)
        .map(|chunk| {
            vec![
                Arc::new(Int64Array::from_iter(
                    chunk.iter().map(|&i| Some(i64::try_from(i % n).unwrap())),
                )) as ArrayRef,
                Arc::new(StringViewArray::from_iter(chunk.iter().map(|&i| {
                    ((i % 7) != 0).then(|| format!("value-{i:06}-padding-to-grow-bytes"))
                }))) as ArrayRef,
            ]
        })
        .collect();
    // Within-group order depends on the thread timing (partitions merge in
    // partition-index order, batches within a partition in morsel order),
    // so rows compare with their values sorted.
    let canonical = |batches: &[Batch]| {
        let mut out = in_order(batches);
        for row in &mut out {
            let (key, list) = row.split_once('|').unwrap();
            let inner = list.strip_prefix('[').unwrap().strip_suffix(']').unwrap();
            let mut values: Vec<&str> = if inner.is_empty() {
                Vec::new()
            } else {
                inner.split(", ").collect()
            };
            values.sort_unstable();
            *row = format!("{key}|[{}]", values.join(", "));
        }
        out.sort();
        out
    };
    let expected = canonical(
        &exec(single_collect(&s, &b), Opts::default())
            .await
            .unwrap()
            .0,
    );
    assert_eq!(expected.len(), n);
    for limit in [2 << 20, 1536 << 10] {
        let (_m, scope) = spill_dir();
        let mut sp = spec(&s, vec![0], vec![agg(&CollectList, 1, &s)]);
        sp.spill = Some(scope);
        let o = Opts {
            limit,
            ..Opts::default()
        };
        // Both layouts, fresh plans (counters add up across runs of one).
        let (off, m_off) = exec_with(node(scan(&s, b.clone()), sp.clone()), o, false)
            .await
            .unwrap();
        let (on, m_on) = exec_with(node(scan(&s, b.clone()), sp.clone()), o, true)
            .await
            .unwrap();
        assert_eq!(canonical(&off), expected, "limit {limit}");
        assert_eq!(canonical(&on), expected, "limit {limit} (row layout)");
        // Both layouts spilled.
        for m in [&m_off, &m_on] {
            assert!(extra(m, "HashAggregate", "spill_runs") > 0, "limit {limit}");
            assert_eq!(extra(m, "HashAggregate", "groups"), n as u64);
        }
    }
    // Unspillable: a memory error, and the pool is empty (asserted by
    // `exec` on the error path too).
    let o = Opts {
        limit: 1 << 20,
        ..Opts::default()
    };
    assert!(exec(single_collect(&s, &b), o).await.is_err());
}

/// `node` over one `collect_list` aggregate (grouped by column 0).
fn single_collect(s: &SchemaRef, b: &[Vec<ArrayRef>]) -> Arc<dyn ExecNode> {
    node(
        scan(s, b.to_vec()),
        spec(s, vec![0], vec![agg(&CollectList, 1, s)]),
    )
}

/// A grouped `collect_list` over no rows returns no rows; a global one
/// returns one empty array.
#[tokio::test]
async fn empty_input_collects() {
    let (s, _) = collect_input();
    let empty: Vec<Vec<ArrayRef>> = vec![];
    let grouped = node(
        scan(&s, empty.clone()),
        spec(&s, vec![0], vec![agg(&CollectSet, 1, &s)]),
    );
    assert_eq!(exec(grouped, Opts::default()).await.unwrap().0.len(), 0);
    let global = node(
        scan(&s, empty),
        spec(&s, vec![], vec![agg(&CollectList, 1, &s)]),
    );
    assert_eq!(run_ordered(global).await, ["[]"]);
}

/// An untyped NULL column holds no values: the collect aggregates give
/// `[]`, the vector ones NULL (snapshots still copy the empty states, so
/// running frames work too).
#[tokio::test]
async fn null_typed_input_collects() {
    use arrow_array::NullArray;
    let s = schema(vec![("k", DataType::Int64), ("x", DataType::Null)]);
    let b = vec![vec![
        i64s(&[Some(1), Some(1)]),
        Arc::new(NullArray::new(2)) as ArrayRef,
    ]];
    for (f, want) in [(CollectList, "1|[]"), (CollectSet, "1|[]")] {
        let single = || node(scan(&s, b.clone()), spec(&s, vec![0], vec![agg(&f, 1, &s)]));
        assert_eq!(run_ordered(single()).await, [want], "{f:?}");
    }
    let v = schema(vec![
        ("k", DataType::Int64),
        (
            "v",
            DataType::List(Arc::new(Field::new("element", DataType::Float32, true))),
        ),
    ]);
    // A NULL vector column against a vector argument: untyped NULLs are
    // skipped before the list cast, so the group is NULL, not an error.
    let b = vec![vec![
        i64s(&[Some(1), Some(1)]),
        Arc::new(NullArray::new(2)) as ArrayRef,
    ]];
    let single = || {
        node(
            scan(&s, b.clone()),
            spec(&s, vec![0], vec![agg(&VectorSum, 1, &v)]),
        )
    };
    assert_eq!(run_ordered(single()).await, ["1|NULL"]);
}
