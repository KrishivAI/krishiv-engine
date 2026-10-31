// SPDX-License-Identifier: Apache-2.0
//! `any_value`, `max_by` and `min_by`: semantics (Spark 4.2: the first row
//! on a tie, NULL keys ignored, a NULL value a legal result, NaN largest,
//! `-0.0 = 0.0`), merges of partial states, the row-layout tables, spills,
//! memory accounting and the types they take. Every test that runs a plan
//! through `exec` also runs it with the row-layout tables.

use arrow_array::types::TimestampMicrosecondType;
use arrow_array::{
    BooleanArray, Decimal128Array, Float32Array, Float64Array, Int16Array, PrimitiveArray,
};

use super::*;
use AggregateFunction::{AnyValue, MaxBy, MinBy};

fn strs(v: &[Option<&str>]) -> ArrayRef {
    Arc::new(StringViewArray::from(v.to_vec()))
}

fn f64s(v: &[Option<f64>]) -> ArrayRef {
    Arc::new(Float64Array::from(v.to_vec()))
}

/// `by(x, y)` over columns 1 and 2 of `s`.
fn by(f: &AggregateFunction, s: &SchemaRef) -> AggregateExpr {
    let name = format!("{f:?}");
    AggregateExpr::by(
        f.clone(),
        (1, 2),
        (
            s.field(1).data_type().clone(),
            s.field(2).data_type().clone(),
        ),
        name,
    )
}

/// Groups of `(x, y)` rows: `1` has a tie at the largest key across the
/// two batches (`c` first) and a NULL key; `2` has only NULL keys; `3`
/// holds a NULL value with the largest key; `4` a NULL value and key; `5`
/// one row.
///
/// `tie: false` makes `d`'s key 6, so no group has a tie (the order a
/// `DISTINCT` plan merges its groups in is not the input's).
fn by_input(tie: bool) -> (SchemaRef, Vec<Vec<ArrayRef>>) {
    let s = schema(vec![
        ("k", DataType::Int64),
        ("x", DataType::Utf8View),
        ("y", DataType::Int64),
    ]);
    let k = i64s(&[1, 1, 1, 1, 2, 2, 3, 3, 4, 5].map(Some));
    let x = strs(&[
        Some("a"),
        Some("b"),
        Some("c"),
        Some("d"),
        Some("p"),
        Some("q"),
        None,
        Some("z"),
        None,
        Some("m"),
    ]);
    let y = i64s(&[
        Some(3),
        None,
        Some(7),
        Some(if tie { 7 } else { 6 }),
        None,
        None,
        Some(9),
        Some(1),
        None,
        Some(5),
    ]);
    // The tie `c`/`d` straddles the batches.
    let cut = |a: &ArrayRef| (a.slice(0, 3), a.slice(3, 7));
    let ((k1, k2), (x1, x2), (y1, y2)) = (cut(&k), cut(&x), cut(&y));
    (s, vec![vec![k1, x1, y1], vec![k2, x2, y2]])
}

const BY_MAX: [&str; 5] = ["1|c", "2|NULL", "3|NULL", "4|NULL", "5|m"];
const BY_MIN: [&str; 5] = ["1|a", "2|NULL", "3|z", "4|NULL", "5|m"];

fn by_aggs(s: &SchemaRef) -> Vec<AggregateExpr> {
    vec![by(&MaxBy, s), by(&MinBy, s)]
}

/// `1|max|min` rows from the two single-aggregate expectations.
fn both(max: &[&str], min: &[&str]) -> Vec<String> {
    max.iter()
        .zip(min)
        .map(|(a, b)| format!("{a}|{}", b.split_once('|').map_or("", |p| p.1)))
        .collect()
}

/// Spark 4.2: `SELECT k, max_by(x, y), min_by(x, y) FROM t GROUP BY k`.
/// The first row wins a tie (also across batches); NULL keys are ignored; a
/// NULL value is a result; no non-NULL key gives NULL.
#[tokio::test]
async fn max_by_and_min_by_pick_the_extreme_key() {
    let (s, b) = by_input(true);
    let single = node(scan(&s, b.clone()), spec(&s, vec![0], by_aggs(&s)));
    let expected = both(&BY_MAX, &BY_MIN);
    assert_eq!(run_one(single).await, expected);
    // Global: the largest key is 9, whose value is NULL (a NULL value is a
    // legal result); the smallest is 1, of `z`.
    let global = node(scan(&s, b.clone()), spec(&s, vec![], by_aggs(&s)));
    assert_eq!(run_one(global).await, ["NULL|z"]);
    // Over no rows, and over rows whose keys are all NULL: NULL.
    let none = node(
        scan(&s, vec![vec![i64s(&[]), strs(&[]), i64s(&[])]]),
        spec(&s, vec![], by_aggs(&s)),
    );
    assert_eq!(run_one(none).await, ["NULL|NULL"]);
    let null_keys = node(
        scan(
            &s,
            vec![vec![
                i64s(&[Some(1), Some(1)]),
                strs(&[Some("a"), Some("b")]),
                i64s(&[None, None]),
            ]],
        ),
        spec(&s, vec![], by_aggs(&s)),
    );
    assert_eq!(run_one(null_keys).await, ["NULL|NULL"]);
}

/// Partial states from any split of the input merge to the single-phase
/// result: through the partial phase (batches in order) and as `DISTINCT`'s
/// first phase.
#[tokio::test]
async fn max_by_and_min_by_merge_partial_states() {
    let (s, b) = by_input(true);
    let expected = both(&BY_MAX, &BY_MIN);
    let mut first = spec(&s, vec![0], by_aggs(&s));
    first.output = AggregateOutput::Partial;
    // The partial states: value then key, per aggregate.
    let names: Vec<String> = first
        .output_schema()
        .unwrap()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(
        names,
        ["k", "MaxBy#value", "MaxBy#key", "MinBy#value", "MinBy#key"]
    );
    let second = second_phase(&first, None);
    let partial = PartialAggregate::new(first, BypassPolicy::default(), Some(7)).unwrap();
    let plan = node(PartialNode::new(scan(&s, b.clone()), partial), second);
    assert_eq!(run_one(plan).await, expected);
    // Bypass: one state row per input row, merged by the final phase.
    let mut first = spec(&s, vec![0], by_aggs(&s));
    first.output = AggregateOutput::Partial;
    let second = second_phase(&first, None);
    let always = BypassPolicy {
        min_rows: 1,
        max_group_percent: 0,
        ..BypassPolicy::default()
    };
    let partial = PartialAggregate::new(first, always, Some(7)).unwrap();
    let plan = node(PartialNode::new(scan(&s, b.clone()), partial), second);
    assert_eq!(run_one(plan).await, expected);
    // `count(DISTINCT x)` next to them: the other aggregates' states are
    // merged by the second phase (without a tie, as that order is the
    // hash order of the first phase's groups).
    let (s, b) = by_input(false);
    let mut aggs = by_aggs(&s);
    aggs.push(agg(&Count, 1, &s));
    let sp = spec(&s, vec![0], aggs);
    let plan = distinct_plan(scan(&s, b), sp, &[false, false, true]).unwrap();
    let want = [
        "1|c|a|4",
        "2|NULL|NULL|2",
        "3|NULL|z|1",
        "4|NULL|NULL|0",
        "5|m|m|1",
    ];
    assert_eq!(run_one(plan).await, want);
}

/// Spark's ordering of `double` keys: NaN is the largest (and equals
/// itself), `-0.0 = 0.0`, infinities order as usual; the first row wins ties.
#[tokio::test]
async fn float_keys_follow_spark_ordering() {
    let s = schema(vec![
        ("k", DataType::Int64),
        ("x", DataType::Int64),
        ("y", DataType::Float64),
    ]);
    let nan = f64::NAN;
    let k = i64s(&[1, 1, 2, 2, 3, 3, 4, 4, 4].map(Some));
    let x = i64s(&[1, 2, 1, 2, 1, 2, 1, 2, 3].map(Some));
    let y = f64s(&[
        Some(0.0),
        Some(-0.0),
        Some(nan),
        Some(f64::INFINITY),
        Some(nan),
        Some(nan),
        Some(f64::NEG_INFINITY),
        None,
        Some(f64::NEG_INFINITY),
    ]);
    let plan = node(
        scan(&s, vec![vec![k, x, y]]),
        spec(&s, vec![0], by_aggs(&s)),
    );
    // k | max_by | min_by: a `0.0`/`-0.0` tie keeps the first row; NaN beats
    // infinity; all-NaN keys tie.
    assert_eq!(run_one(plan).await, ["1|1|1", "2|1|2", "3|1|1", "4|1|1"]);
}

/// `any_value` is the first row's value, or with `ignoreNulls` the first
/// non-NULL one; through the partial phase as well.
#[tokio::test]
async fn any_value_is_the_first_value() {
    let (s, b) = picks_input();
    let aggs = vec![
        agg(
            &AnyValue {
                ignore_nulls: false,
            },
            1,
            &s,
        ),
        agg(&AnyValue { ignore_nulls: true }, 1, &s),
        agg(
            &AnyValue {
                ignore_nulls: false,
            },
            2,
            &s,
        ),
        agg(&AnyValue { ignore_nulls: true }, 2, &s),
    ];
    let expected = ["1|NULL|5|a|a", "2|NULL|NULL|NULL|NULL", "3|9|9|z|z"];
    let single = node(scan(&s, b.clone()), spec(&s, vec![0], aggs.clone()));
    assert_eq!(run_one(single).await, expected);
    let mut first = spec(&s, vec![0], aggs.clone());
    first.output = AggregateOutput::Partial;
    let second = second_phase(&first, None);
    let partial = PartialAggregate::new(first, BypassPolicy::default(), Some(7)).unwrap();
    let plan = node(PartialNode::new(scan(&s, b), partial), second);
    assert_eq!(run_one(plan).await, expected);
    // No rows: NULL.
    let empty = |t: DataType| arrow_array::new_empty_array(&t);
    let none = node(
        scan(
            &s,
            vec![vec![
                empty(DataType::Int64),
                empty(DataType::Int64),
                empty(DataType::Utf8View),
            ]],
        ),
        spec(&s, vec![], aggs),
    );
    assert_eq!(run_one(none).await, ["NULL|NULL|NULL|NULL"]);
}

/// A column of Spark type `t` holding `v` (`i64` stands for the value in
/// every type's domain; strings are zero-padded so they order as numbers).
fn typed(t: &DataType, v: &[Option<i64>]) -> ArrayRef {
    let on = |f: fn(i64) -> i128| v.iter().map(|x| x.map(f)).collect::<Vec<_>>();
    match t {
        DataType::Boolean => Arc::new(BooleanArray::from(
            v.iter().map(|x| x.map(|x| x % 2 != 0)).collect::<Vec<_>>(),
        )),
        DataType::Int8 => Arc::new(Int8Array::from(
            v.iter()
                .map(|x| x.map(|x| i8::try_from(x).unwrap()))
                .collect::<Vec<_>>(),
        )),
        DataType::Int16 => Arc::new(Int16Array::from(
            v.iter()
                .map(|x| x.map(|x| i16::try_from(x).unwrap()))
                .collect::<Vec<_>>(),
        )),
        DataType::Int32 => Arc::new(Int32Array::from(
            v.iter()
                .map(|x| x.map(|x| i32::try_from(x).unwrap()))
                .collect::<Vec<_>>(),
        )),
        DataType::Int64 => i64s(v),
        DataType::Float32 => Arc::new(Float32Array::from(
            v.iter().map(|x| x.map(|x| x as f32)).collect::<Vec<_>>(),
        )),
        DataType::Float64 => f64s(&v.iter().map(|x| x.map(|x| x as f64)).collect::<Vec<_>>()),
        DataType::Date32 => Arc::new(Date32Array::from(
            v.iter()
                .map(|x| x.map(|x| i32::try_from(x).unwrap()))
                .collect::<Vec<_>>(),
        )),
        DataType::Timestamp(..) => {
            Arc::new(PrimitiveArray::<TimestampMicrosecondType>::from(v.to_vec()))
        }
        DataType::Decimal128(p, s) => Arc::new(
            Decimal128Array::from(on(|x| i128::from(x) * 100))
                .with_precision_and_scale(*p, *s)
                .unwrap(),
        ),
        DataType::Utf8View => Arc::new(StringViewArray::from(
            v.iter()
                .map(|x| x.map(|x| format!("{x:05}")))
                .collect::<Vec<_>>(),
        )),
        t => panic!("no test column of {t}"),
    }
}

/// The types `max_by`/`min_by` take as value and as key.
fn by_types() -> Vec<DataType> {
    vec![
        DataType::Boolean,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::Float32,
        DataType::Float64,
        DataType::Decimal128(10, 2),
        DataType::Date32,
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
        DataType::Utf8View,
    ]
}

/// What `max_by`/`min_by` of rows `(g, x, y)` give per group, by a plain
/// loop: the first row with a non-NULL key and the extreme key.
fn by_oracle(
    rows: &[(i64, Option<i64>, Option<i64>)],
    max: bool,
    key: impl Fn(i64) -> i64,
) -> BTreeMap<i64, Option<i64>> {
    let mut best: BTreeMap<i64, Option<(i64, Option<i64>)>> = BTreeMap::new();
    for &(g, x, y) in rows {
        let slot = best.entry(g).or_insert(None);
        let Some(y) = y else { continue };
        let better = match slot {
            None => true,
            Some((cur, _)) => (key(y) > key(*cur)) == max && key(y) != key(*cur),
        };
        if better {
            *slot = Some((y, x));
        }
    }
    best.into_iter()
        .map(|(g, v)| (g, v.and_then(|(_, x)| x)))
        .collect()
}

/// Every value type as `x` and as `y`, grouped, through both phases and
/// both table layouts, against a plain loop.
#[tokio::test]
async fn every_value_and_key_type() {
    // Distinct non-negative keys per group (except for booleans), NULLs in
    // both columns, a NULL key on the largest value.
    let rows: Vec<(i64, Option<i64>, Option<i64>)> = vec![
        (1, Some(10), Some(5)),
        (1, Some(11), Some(9)),
        (1, None, None),
        (1, Some(12), Some(2)),
        (2, None, Some(7)),
        (2, Some(21), Some(3)),
        (3, Some(30), None),
        (4, Some(40), Some(1)),
    ];
    let g: Vec<Option<i64>> = rows.iter().map(|r| Some(r.0)).collect();
    let xs: Vec<Option<i64>> = rows.iter().map(|r| r.1).collect();
    let ys: Vec<Option<i64>> = rows.iter().map(|r| r.2).collect();
    for tx in by_types() {
        for ty in by_types() {
            let s = schema(vec![
                ("k", DataType::Int64),
                ("x", tx.clone()),
                ("y", ty.clone()),
            ]);
            let batch = vec![i64s(&g), typed(&tx, &xs), typed(&ty, &ys)];
            let key = |y: i64| if ty == DataType::Boolean { y % 2 } else { y };
            for (f, max) in [(&MaxBy, true), (&MinBy, false)] {
                let want = by_oracle(&rows, max, key);
                let render = |v: Option<i64>| cell(&typed(&tx, &[v]), 0);
                let expected: Vec<String> = {
                    let mut r: Vec<String> = want
                        .iter()
                        .map(|(g, v)| format!("{g}|{}", render(*v)))
                        .collect();
                    r.sort();
                    r
                };
                let sp = spec(&s, vec![0], vec![by(&f.clone(), &s)]);
                let single = node(scan(&s, vec![batch.clone()]), sp.clone());
                assert_eq!(run_one(single).await, expected, "{f:?} of {tx} by {ty}");
                // Two batches through the partial phase.
                let halves: Vec<Vec<ArrayRef>> = vec![
                    batch.iter().map(|c| c.slice(0, 3)).collect(),
                    batch.iter().map(|c| c.slice(3, 5)).collect(),
                ];
                let mut first = sp.clone();
                first.output = AggregateOutput::Partial;
                let second = second_phase(&first, None);
                let partial =
                    PartialAggregate::new(first, BypassPolicy::default(), Some(7)).unwrap();
                let plan = node(PartialNode::new(scan(&s, halves), partial), second);
                assert_eq!(
                    run_one(plan).await,
                    expected,
                    "{f:?} of {tx} by {ty}, two phases"
                );
            }
        }
    }
}

/// Group `k` holds `(x, y)`; the keys' order is not the values' order, so a
/// wrong pick shows.
fn spill_input(n: i64, keys: i64) -> (SchemaRef, Vec<Vec<ArrayRef>>) {
    let s = schema(vec![
        ("k", DataType::Int64),
        ("x", DataType::Utf8View),
        ("y", DataType::Utf8View),
    ]);
    let batches = (0..n)
        .step_by(1000)
        .map(|start| {
            let r = start..(start + 1000).min(n);
            let k: Vec<_> = r.clone().map(|i| Some(i % keys)).collect();
            let x: Vec<_> = r.clone().map(|i| Some(format!("value-{i:08}"))).collect();
            let y: Vec<_> = r
                .map(|i| (i % 11 != 0).then(|| format!("key-{:08}", (i * 7919) % 100_003)))
                .collect();
            vec![
                i64s(&k),
                Arc::new(StringViewArray::from(x)) as ArrayRef,
                Arc::new(StringViewArray::from(y)) as ArrayRef,
            ]
        })
        .collect();
    (s, batches)
}

/// `max_by`/`min_by` of strings by strings under a tiny budget: the runs
/// spill, merge partition by partition and give the in-memory result, which
/// is the plain loop's.
#[tokio::test]
async fn by_aggregates_spill_and_merge() {
    let (s, b) = spill_input(60_000, 40_000);
    let aggs = vec![by(&MaxBy, &s), by(&MinBy, &s)];
    let expected = run_rows(node(scan(&s, b.clone()), spec(&s, vec![0], aggs.clone()))).await;
    // The same answer by a plain loop (ties keep the first row).
    let mut best: BTreeMap<i64, [Option<(String, String)>; 2]> = BTreeMap::new();
    for i in 0..60_000_i64 {
        if i % 11 == 0 {
            continue;
        }
        let (x, y) = (
            format!("value-{i:08}"),
            format!("key-{:08}", (i * 7919) % 100_003),
        );
        let slot = best.entry(i % 40_000).or_default();
        if slot[0].as_ref().is_none_or(|(_, cur)| y > *cur) {
            slot[0] = Some((x.clone(), y.clone()));
        }
        if slot[1].as_ref().is_none_or(|(_, cur)| y < *cur) {
            slot[1] = Some((x, y));
        }
    }
    let mut plain: Vec<String> = (0..40_000_i64)
        .map(|k| {
            let v = best.get(&k);
            let pick = |n: usize| {
                v.and_then(|v| v[n].as_ref())
                    .map_or("NULL".to_owned(), |(x, _)| x.clone())
            };
            format!("{k}|{}|{}", pick(0), pick(1))
        })
        .collect();
    plain.sort();
    assert_eq!(expected, plain);
    let (_m, scope) = spill_dir();
    let mut sp = spec(&s, vec![0], aggs);
    sp.spill = Some(scope.clone());
    let o = Opts {
        limit: 3 << 20,
        ..Opts::default()
    };
    let (out, metrics) = exec_each(|| node(scan(&s, b.clone()), sp.clone()), o).await;
    assert_eq!(out, expected);
    for m in &metrics {
        assert!(extra(m, "HashAggregate", "spill_runs") > 0, "{m:?}");
        assert_eq!(extra(m, "HashAggregate", "groups"), 40_000);
    }
    // Spilled partial states merged in a second phase.
    let mut first = spec(&s, vec![0], vec![by(&MaxBy, &s), by(&MinBy, &s)]);
    first.output = AggregateOutput::Partial;
    let second = second_phase(&first, Some(scope));
    let partial = PartialAggregate::new(first, BypassPolicy::default(), Some(7)).unwrap();
    let plan = node(PartialNode::new(scan(&s, b), partial), second);
    assert_eq!(rows(&exec(plan, o).await.unwrap().0), expected);
}

/// A table that outgrows the pool without a spill directory fails the query
/// with a memory error and releases everything.
#[tokio::test]
async fn by_aggregates_without_spill_fail_cleanly() {
    let (s, b) = spill_input(60_000, 60_000);
    let plan = node(scan(&s, b), spec(&s, vec![0], vec![by(&MaxBy, &s)]));
    let o = Opts {
        limit: 1 << 20,
        ..Opts::default()
    };
    let e = exec(plan, o).await.err().map(|e| e.error_class());
    assert_eq!(e, Some(ErrorClass::UnableToAcquireMemory));
}

#[test]
fn types_states_and_errors() {
    let s = schema(vec![
        ("k", DataType::Int64),
        ("x", DataType::Decimal128(12, 2)),
        ("y", DataType::Date32),
    ]);
    let a = by(&MaxBy, &s);
    assert_eq!(a.result_type().unwrap(), DataType::Decimal128(12, 2));
    let fields = a.state_fields().unwrap();
    let types: Vec<_> = fields.iter().map(|f| f.data_type().clone()).collect();
    assert_eq!(types, [DataType::Decimal128(12, 2), DataType::Date32]);
    assert!(fields.iter().all(Field::is_nullable));
    assert_eq!(a.merging(4).args, [4, 5]);
    assert_eq!(a.args, [1, 2]);
    let any = AggregateExpr::new(AnyValue { ignore_nulls: true }, 1, DataType::Utf8View, "a");
    assert_eq!(any.result_type().unwrap(), DataType::Utf8View);
    assert_eq!(any.state_fields().unwrap().len(), 2);
    let unsupported = ErrorClass::KrishivUnsupportedFeature;
    // A type `min`/`max` do not take, as value or as key; a plain argument
    // type (no key) for `max_by`.
    let bad = |x: DataType, y: DataType| {
        AggregateExpr::by(MinBy, (0, 1), (x, y), "b")
            .result_type()
            .map_err(|e| e.error_class())
    };
    assert_eq!(bad(DataType::Null, DataType::Int64), Err(unsupported));
    assert_eq!(bad(DataType::Int64, DataType::Null), Err(unsupported));
    // Spark reports these as `DATATYPE_MISMATCH`; the engine has no
    // ordering for them at all, so they are unsupported here.
    let interval = DataType::Interval(arrow_schema::IntervalUnit::YearMonth);
    let array = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
    assert_eq!(bad(interval.clone(), DataType::Int64), Err(unsupported));
    assert_eq!(bad(DataType::Int64, interval), Err(unsupported));
    assert_eq!(bad(array.clone(), DataType::Int64), Err(unsupported));
    assert_eq!(bad(DataType::Int64, array), Err(unsupported));
    assert_eq!(
        AggregateExpr::new(MaxBy, 0, DataType::Int64, "b")
            .result_type()
            .map_err(|e| e.error_class()),
        Err(unsupported)
    );
    assert_eq!(
        AggregateExpr::new(
            AnyValue {
                ignore_nulls: false
            },
            0,
            DataType::Null,
            "b"
        )
        .result_type()
        .map_err(|e| e.error_class()),
        Err(unsupported)
    );
}

/// Weights other than 1 need incremental view maintenance, as for `min`.
#[tokio::test]
async fn weights_are_rejected() {
    let (s, b) = by_input(true);
    let aggs = vec![
        by(&MaxBy, &s),
        agg(
            &AnyValue {
                ignore_nulls: false,
            },
            1,
            &s,
        ),
    ];
    let weighted: Vec<Batch> = b
        .into_iter()
        .map(|cols| {
            let data = RecordBatch::try_new(Arc::clone(&s), cols).unwrap();
            Batch::with_weights(data, Weights::Const(2)).unwrap()
        })
        .collect();
    let plan: Arc<dyn ExecNode> = Arc::new(Scan(Batches::new(Arc::clone(&s), weighted)));
    let e = exec(node(plan, spec(&s, vec![0], aggs)), Opts::default())
        .await
        .err()
        .map(|e| e.error_class());
    assert_eq!(e, Some(ErrorClass::KrishivUnsupportedFeature));
}

/// A running window frame snapshots state per row; byte values would be
/// copied per row without being accounted, so they are refused, as is a
/// partial state. Other types snapshot.
#[test]
fn running_frames_snapshot_fixed_width_states_only() {
    let int = |a: &AggregateExpr| Accumulator::new(a, true).unwrap();
    let mk = |tx: DataType, ty: DataType| {
        let expr = AggregateExpr::by(MaxBy, (0, 1), (tx, ty), "b");
        int(&expr)
    };
    // Rows 0..3 of one partition (group 0); each row's state goes to its
    // own group 1..=3.
    let parts = [0u32; 3];
    let snaps = [1u32, 2, 3];
    let run = Running {
        parts: &parts,
        snapshots: &snaps,
    };
    let mut a = mk(DataType::Int64, DataType::Int64);
    a.resize(4);
    let x = i64s(&[Some(10), None, Some(30)]);
    let y = i64s(&[Some(1), Some(5), Some(5)]);
    a.update_running(&[0, 0, 0], &[x, y], &run).unwrap();
    // After each row: 10; then NULL (key 5 beats 1); then NULL again (a tie
    // keeps the first).
    let out = a.evaluate(&[1, 2, 3]).unwrap();
    let out = out.as_primitive::<Int64Type>();
    assert_eq!(
        (0..3)
            .map(|i| out.is_valid(i).then(|| out.value(i)))
            .collect::<Vec<_>>(),
        [Some(10), None, None]
    );
    for (tx, ty) in [
        (DataType::Utf8View, DataType::Int64),
        (DataType::Int64, DataType::Utf8View),
    ] {
        let mut a = mk(tx.clone(), ty.clone());
        a.resize(4);
        let cols = [typed(&tx, &[Some(1); 3]), typed(&ty, &[Some(1); 3])];
        let e = a.update_running(&[0, 0, 0], &cols, &run).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
        // The whole-partition fold takes them.
        let mut a = mk(tx.clone(), ty.clone());
        a.resize(1);
        a.update(&[0, 0, 0], &cols, &Weights::Const(1)).unwrap();
    }
    // Columns of other lengths than the group ids are an internal error,
    // not a panic.
    let mut a = mk(DataType::Int64, DataType::Int64);
    a.resize(1);
    let e = a
        .update(
            &[0, 0],
            &[i64s(&[Some(1)]), i64s(&[Some(1), Some(2)])],
            &Weights::Const(1),
        )
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}

/// The projected size bounds the size after the update, for byte values
/// and keys (what the reservation is made from).
#[test]
fn projected_size_bounds_the_size_after_an_update() {
    for (tx, ty) in [
        (DataType::Utf8View, DataType::Int64),
        (DataType::Int64, DataType::Utf8View),
        (DataType::Utf8View, DataType::Utf8View),
        (DataType::Int64, DataType::Int64),
    ] {
        let expr = AggregateExpr::by(MaxBy, (0, 1), (tx.clone(), ty.clone()), "b");
        let mut a = Accumulator::new(&expr, true).unwrap();
        let v: Vec<Option<i64>> = (0..200).map(Some).collect();
        let (x, y) = (typed(&tx, &v), typed(&ty, &v));
        let projected = a.projected(200, Some(&x), Some(&y));
        a.reserve(200);
        a.resize(200);
        let groups: Vec<u32> = (0..200).collect();
        a.update(&groups, &[x, y], &Weights::Const(1)).unwrap();
        assert!(
            a.size() <= projected,
            "{tx} by {ty}: {} > {projected}",
            a.size()
        );
        // Replacing every value by a longer one stays within the bound too.
        let longer: Vec<Option<i64>> = (1000..1200).map(Some).collect();
        let (x, y) = (typed(&tx, &longer), typed(&ty, &longer));
        let projected = a.projected(0, Some(&x), Some(&y));
        a.update(&groups, &[x, y], &Weights::Const(1)).unwrap();
        assert!(
            a.size() <= projected,
            "{tx} by {ty}: {} > {projected}",
            a.size()
        );
    }
}

/// A column of another physical kind than the state's (a plan that typed
/// its argument wrongly) is an internal error, not a panic or a wrong
/// answer.
#[test]
fn columns_of_the_wrong_kind_are_internal_errors() {
    let expr = AggregateExpr::by(MaxBy, (0, 1), (DataType::Int64, DataType::Int64), "b");
    let mut a = Accumulator::new(&expr, true).unwrap();
    a.resize(1);
    let ints = i64s(&[Some(1)]);
    let doubles = f64s(&[Some(1.0)]);
    let internal = |r: crate::Result<()>| r.unwrap_err().error_class();
    // The first row is stored: a key of another kind cannot be.
    let r = a.update(
        &[0],
        &[Arc::clone(&ints), Arc::clone(&doubles)],
        &Weights::Const(1),
    );
    assert_eq!(internal(r), ErrorClass::InternalError);
    a.update(
        &[0],
        &[Arc::clone(&ints), Arc::clone(&ints)],
        &Weights::Const(1),
    )
    .unwrap();
    // A later row is compared with the key held first.
    let r = a.update(
        &[0],
        &[Arc::clone(&ints), Arc::clone(&doubles)],
        &Weights::Const(1),
    );
    assert_eq!(internal(r), ErrorClass::InternalError);
    // A value of another kind, on a row that wins (key 2 beats key 1).
    let larger = i64s(&[Some(2)]);
    let r = a.update(&[0], &[doubles, larger], &Weights::Const(1));
    assert_eq!(internal(r), ErrorClass::InternalError);
}
