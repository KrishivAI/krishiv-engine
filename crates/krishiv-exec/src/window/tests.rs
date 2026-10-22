// SPDX-License-Identifier: Apache-2.0
//! Window operator tests: a property test against a naive per-row
//! reference over random partitions, orderings and batch splits; Spark's
//! `NTile`, exact decimal frames, filters, memory, weights and plan checks.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, BooleanArray, Decimal128Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::error::ErrorClass;
use krishiv_memory::{MemoryConfig, MemoryPool, SpillOwner, TreePool};
use krishiv_types::batch::{Batch, Weights};
use proptest::prelude::*;

use super::*;
use crate::aggregate::{AggInput, AggregateExpr, AggregateFunction};
use crate::operator::Operator;
use crate::operators::Sort;
use crate::planner::nodes::{BreakerNode, OperatorNode, SourceNode};
use crate::testing::Batches;
use crate::{
    ExecConfig, ExecNode, Executor, OperatorId, QueryContext, QueryId, QueryRunner, TaskContext,
    TokioClock, TokioExecutor, WorkloadClass,
};

fn pool(limit: usize) -> Arc<TreePool> {
    TreePool::with_clock(
        &MemoryConfig {
            limit: Some(limit),
            quantum: 16 << 10,
            spill_wait_timeout: Duration::from_secs(5),
            abort_release_timeout: Duration::from_secs(30),
            spill_scratch_reserve: krishiv_memory::DEFAULT_SPILL_SCRATCH_RESERVE,
        },
        Arc::new(TokioClock),
    )
}

/// `(p, o, id, v)`: partition key, order key, unique tie-break, value.
type Row = (Option<i64>, Option<i64>, i64, Option<i64>);

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("p", DataType::Int64, true),
        Field::new("o", DataType::Int64, true),
        Field::new("id", DataType::Int64, false),
        Field::new("v", DataType::Int64, true),
    ]))
}

fn batch(rows: &[Row]) -> Batch {
    let col = |f: fn(&Row) -> Option<i64>| -> ArrayRef {
        Arc::new(Int64Array::from_iter(rows.iter().map(f)))
    };
    let columns = vec![col(|r| r.0), col(|r| r.1), col(|r| Some(r.2)), col(|r| r.3)];
    Batch::new(RecordBatch::try_new(schema(), columns).unwrap())
}

fn agg(function: AggregateFunction, arg: Option<usize>) -> WindowFunction {
    WindowFunction::Aggregate(AggregateExpr {
        function,
        args: arg.into_iter().collect(),
        input: AggInput::Raw,
        arg_type: if arg.is_some() {
            DataType::Int64
        } else {
            DataType::Null
        },
        name: "a".to_owned(),
    })
}

fn plan(functions: Vec<(WindowFunction, WindowFrame)>, partitioned: bool) -> WindowPlan {
    WindowPlan {
        partition_by: if partitioned { vec![0] } else { Vec::new() },
        order_by: vec![SortKey::asc(1)],
        functions: functions
            .into_iter()
            .enumerate()
            .map(|(i, (function, frame))| WindowFunctionPlan {
                function,
                frame,
                filter: None,
                name: format!("w{i}"),
            })
            .collect(),
    }
}

/// Runs `rows` (split into batches of `split` rows) through a sort by the
/// partitioning, ordering and `id`, then the window; returns the output
/// batches in order. Checks the query pool is empty afterwards.
async fn run(
    rows: &[Row],
    split: usize,
    window: &WindowPlan,
    limit: usize,
) -> crate::Result<Vec<Batch>> {
    let s = schema();
    let batches: Vec<Batch> = rows.chunks(split.max(1)).map(batch).collect();
    let mut keys: Vec<SortKey> = window
        .partition_by
        .iter()
        .map(|&c| SortKey::asc(c))
        .collect();
    keys.extend(window.order_by.iter().copied());
    keys.push(SortKey::asc(2));
    let sort = Arc::new(Sort::new(Arc::clone(&s), &keys, None)?);
    let op = Arc::new(Window::new(Arc::clone(&s), window.clone(), true)?);
    let plan: Arc<dyn ExecNode> = Arc::new(OperatorNode {
        input: Arc::new(BreakerNode {
            input: Arc::new(SourceNode(Batches::new(s, batches))),
            sink: Arc::clone(&sort) as Arc<dyn crate::Sink>,
            source: sort,
            ordered: true,
        }),
        op,
    });
    let runner = QueryRunner::new(TokioExecutor::deterministic(2)?);
    let root = pool(limit);
    let config = ExecConfig {
        batch_size: 4,
        ..ExecConfig::default()
    };
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        config,
        root.child("q", Some(limit)),
    );
    let result = async {
        let mut stream = runner.execute(plan, Arc::clone(&q)).await?;
        let mut out = Vec::new();
        while let Some(b) = stream.next().await {
            out.push(b?);
        }
        Ok(out)
    }
    .await;
    drop(q);
    assert_eq!(root.reserved(), 0, "the query pool is empty at the end");
    result
}

/// One output value as text, for comparing.
fn cell(a: &ArrayRef, i: usize) -> String {
    if a.is_null(i) {
        return "null".to_owned();
    }
    match a.data_type() {
        DataType::Int32 => a.as_primitive::<Int32Type>().value(i).to_string(),
        DataType::Int64 => a.as_primitive::<Int64Type>().value(i).to_string(),
        DataType::Float64 => format!("{:?}", a.as_primitive::<Float64Type>().value(i)),
        DataType::Decimal128(_, s) => {
            crate::aggregate::plain(a.as_primitive::<Decimal128Type>().value(i), *s)
        }
        t => format!("<{t}>"),
    }
}

/// Each output row as `id: w0, w1, ...`, in output order.
fn outputs(batches: &[Batch], functions: usize) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        let d = b.data();
        let n = d.num_columns();
        for i in 0..d.num_rows() {
            let id = d.column(2).as_primitive::<Int64Type>().value(i);
            let cells: Vec<String> = (n - functions..n).map(|c| cell(d.column(c), i)).collect();
            out.push(format!("{id}: {}", cells.join(", ")));
        }
    }
    out
}

/// Spark's `NTile` as its update expressions run, row by row.
fn ntile_spark(rows: usize, buckets: i64) -> Vec<i64> {
    let n = i64::try_from(rows).unwrap();
    let (size, padding) = (n / buckets, n % buckets);
    let (mut row_number, mut bucket, mut threshold) = (0_i64, 0_i64, 0_i64);
    let mut out = Vec::new();
    for _ in 0..rows {
        let overflow = row_number >= threshold;
        let next_bucket = bucket + i64::from(overflow);
        let next_threshold = if overflow {
            threshold + size + i64::from(bucket < padding)
        } else {
            threshold
        };
        row_number += 1;
        bucket = next_bucket;
        threshold = next_threshold;
        out.push(bucket);
    }
    out
}

/// Spark's order of the keys: NULL first, then by value.
fn key(v: Option<i64>) -> (bool, i64) {
    (v.is_some(), v.unwrap_or(0))
}

/// The naive value of `f` (over `frame`) for each row of `rows`, which are
/// sorted by partition, order key and id; one partition when not
/// `partitioned`.
fn reference(
    rows: &[Row],
    f: &WindowFunction,
    frame: WindowFrame,
    partitioned: bool,
) -> Vec<String> {
    let mut out = Vec::new();
    let same_part = |a: &Row, b: &Row| !partitioned || a.0 == b.0;
    let to_f = |x: usize| f64::from(u32::try_from(x).unwrap());
    for r in rows {
        let part: Vec<&Row> = rows.iter().filter(|x| same_part(x, r)).collect();
        let pos = part.iter().position(|x| x.2 == r.2).unwrap();
        let n = part.len();
        let rank = 1 + part.iter().filter(|x| key(x.1) < key(r.1)).count();
        let peers_end = part.iter().filter(|x| key(x.1) <= key(r.1)).count();
        out.push(match f {
            WindowFunction::RowNumber => (pos + 1).to_string(),
            WindowFunction::Rank => rank.to_string(),
            WindowFunction::DenseRank => {
                let mut keys: Vec<(bool, i64)> = part
                    .iter()
                    .map(|x| key(x.1))
                    .filter(|k| *k < key(r.1))
                    .collect();
                keys.dedup();
                (keys.len() + 1).to_string()
            }
            WindowFunction::PercentRank => {
                let v = if n == 1 {
                    0.0
                } else {
                    to_f(rank - 1) / to_f(n - 1)
                };
                format!("{v:?}")
            }
            WindowFunction::CumeDist => format!("{:?}", to_f(peers_end) / to_f(n)),
            WindowFunction::Ntile(k) => ntile_spark(n, i64::from(*k))[pos].to_string(),
            WindowFunction::Aggregate(a) => {
                let end = match (frame.units, frame.upper) {
                    (_, FrameBound::UnboundedFollowing) => n,
                    (FrameUnits::Rows, _) => pos + 1,
                    (FrameUnits::Range, _) => peers_end,
                };
                let vals: Vec<i64> = part[..end].iter().filter_map(|x| x.3).collect();
                match a.function {
                    AggregateFunction::Count if a.args.is_empty() => end.to_string(),
                    AggregateFunction::Count => vals.len().to_string(),
                    _ if vals.is_empty() => "null".to_owned(),
                    AggregateFunction::Sum => vals.iter().sum::<i64>().to_string(),
                    AggregateFunction::Min => vals.iter().min().unwrap().to_string(),
                    AggregateFunction::Max => vals.iter().max().unwrap().to_string(),
                    AggregateFunction::Avg => {
                        // Spark sums integers as doubles, in order.
                        #[expect(clippy::cast_precision_loss, reason = "Spark's double average")]
                        let s = vals.iter().fold(0.0, |s, v| s + *v as f64);
                        format!("{:?}", s / to_f(vals.len()))
                    }
                    other => format!("<{other:?}>"),
                }
            }
        });
    }
    out
}

fn functions() -> Vec<(WindowFunction, WindowFrame)> {
    use AggregateFunction::{Avg, Count, Max, Min, Sum};
    let rows = WindowFrame::running(FrameUnits::Rows);
    let range = WindowFrame::running(FrameUnits::Range);
    let whole = WindowFrame::whole(FrameUnits::Rows);
    vec![
        (WindowFunction::RowNumber, rows),
        (WindowFunction::Rank, rows),
        (WindowFunction::DenseRank, rows),
        (WindowFunction::PercentRank, rows),
        (WindowFunction::CumeDist, range),
        (WindowFunction::Ntile(3), rows),
        (agg(Sum, Some(3)), rows),
        (agg(Sum, Some(3)), range),
        (agg(Sum, Some(3)), whole),
        (agg(Count, None), range),
        (agg(Count, Some(3)), whole),
        (agg(Min, Some(3)), range),
        (agg(Max, Some(3)), rows),
        (agg(Avg, Some(3)), range),
        (agg(Avg, Some(3)), WindowFrame::whole(FrameUnits::Range)),
    ]
}

fn rows_strategy() -> impl Strategy<Value = Vec<Row>> {
    let opt = |hi: i64| prop_oneof![1 => Just(None), 4 => (0..hi).prop_map(Some)];
    prop::collection::vec((opt(4), opt(5), opt(100)), 0..60).prop_map(|v| {
        v.into_iter()
            .enumerate()
            .map(|(i, (p, o, x))| (p, o, i64::try_from(i).unwrap(), x))
            .collect()
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Every function over random partitions, orderings with ties and NULLs,
    /// and batch splits (partitions span batches) gives the naive value.
    #[test]
    fn window_matches_a_naive_reference(
        rows in rows_strategy(),
        split in 1_usize..9,
        partitioned in any::<bool>(),
    ) {
        let functions = functions();
        let window = plan(functions.clone(), partitioned);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let out = rt.block_on(run(&rows, split, &window, 1 << 30)).unwrap();
        let got = outputs(&out, functions.len());
        let mut sorted = rows;
        sorted.sort_by_key(|r| (if partitioned { key(r.0) } else { (false, 0) }, key(r.1), r.2));
        let columns: Vec<Vec<String>> = functions
            .iter()
            .map(|(f, frame)| reference(&sorted, f, *frame, partitioned))
            .collect();
        let want: Vec<String> = sorted
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let cells: Vec<String> = columns.iter().map(|c| c[i].clone()).collect();
                format!("{}: {}", r.2, cells.join(", "))
            })
            .collect();
        prop_assert_eq!(got, want);
    }
}

#[test]
fn ntile_closed_form_is_sparks_algorithm() {
    for rows in 0..40 {
        for k in 1..45 {
            let got: Vec<i64> = (0..rows)
                .map(|r| i64::from(super::eval::ntile(r, rows, k).unwrap()))
                .collect();
            assert_eq!(
                got,
                ntile_spark(rows, i64::from(k)),
                "{rows} rows, {k} buckets"
            );
        }
    }
    assert_eq!(super::eval::ntile(0, 1, 0), None);
}

fn decimal_rows(values: &[i128], p: u8) -> (SchemaRef, Batch) {
    let s = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, false),
        Field::new("x", DataType::Decimal128(p, 0), true),
    ]));
    let i: ArrayRef = Arc::new(Int64Array::from_iter_values(
        (0..values.len()).map(|i| i64::try_from(i).unwrap()),
    ));
    let x: ArrayRef = Arc::new(
        Decimal128Array::from_iter_values(values.iter().copied())
            .with_precision_and_scale(p, 0)
            .unwrap(),
    );
    let b = RecordBatch::try_new(Arc::clone(&s), vec![i, x]).unwrap();
    (s, Batch::new(b))
}

fn task(limit: usize) -> (Arc<TreePool>, TaskContext) {
    let root = pool(limit);
    let q = QueryContext::new(
        QueryId(7),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("q", Some(limit)),
    );
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
    let ctx = TaskContext::new(q, executor, SpillOwner::new(), OperatorId(0), "window");
    (root, ctx)
}

/// Runs `op` over `input` directly (one driver), collecting its output.
fn drive(op: &Window, ctx: &TaskContext, input: &[Batch]) -> crate::Result<Vec<Batch>> {
    let mut local = op.local_state(ctx)?;
    let mut out = Vec::new();
    for b in input {
        loop {
            match op.execute(b, &mut local)? {
                OperatorOutput::NeedMoreInput(o) => {
                    out.extend(o);
                    break;
                }
                OperatorOutput::HaveMoreOutput(o) => out.push(o),
                other => {
                    return Err(crate::error::internal(format!("unexpected {other:?}")));
                }
            }
        }
    }
    while let Some(b) = op.finish(&mut local)? {
        out.push(b);
    }
    Ok(out)
}

fn decimal_window(function: AggregateFunction, frame: WindowFrame, p: u8, ansi: bool) -> Window {
    let s = decimal_rows(&[0], p).0;
    let plan = WindowPlan {
        partition_by: Vec::new(),
        order_by: vec![SortKey::asc(0)],
        functions: vec![WindowFunctionPlan {
            function: WindowFunction::Aggregate(AggregateExpr {
                function,
                args: vec![1],
                input: AggInput::Raw,
                arg_type: DataType::Decimal128(p, 0),
                name: "s".to_owned(),
            }),
            frame,
            filter: None,
            name: "s".to_owned(),
        }],
    };
    Window::new(s, plan, ansi).unwrap()
}

fn column_text(out: &[Batch], column: usize) -> Vec<String> {
    out.iter()
        .flat_map(|b| {
            let c = b.data().column(column);
            (0..c.len()).map(|i| cell(c, i)).collect::<Vec<_>>()
        })
        .collect()
}

/// Spark keeps a window's decimal sum exact and checks it when evaluated:
/// `M, 1, -5` leaves `decimal(38, 0)` after the second row and comes back
/// (Spark 4.2.0: ANSI off `[M, NULL, M - 4]`; ANSI on the second row raises
/// `NUMERIC_VALUE_OUT_OF_RANGE`; over the whole partition `M - 4` either
/// way).
#[tokio::test]
async fn decimal_frames_stay_exact_until_evaluated() {
    let m = 10_i128.pow(38) - 1;
    let (_, input) = decimal_rows(&[m, 1, -5], 38);
    let (_root, ctx) = task(1 << 30);
    let values = |ansi: bool, frame: WindowFrame| {
        let op = decimal_window(AggregateFunction::Sum, frame, 38, ansi);
        drive(&op, &ctx, std::slice::from_ref(&input)).map(|out| column_text(&out, 2))
    };
    let running = WindowFrame::running(FrameUnits::Rows);
    let whole = WindowFrame::whole(FrameUnits::Rows);
    let ms = m.to_string();
    let m4 = (m - 4).to_string();
    assert_eq!(
        values(false, running).unwrap(),
        [ms.as_str(), "null", m4.as_str()]
    );
    let e = values(true, running).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::NumericValueOutOfRangeWithSuggestion
    );
    assert!(
        e.to_string()
            .contains("100000000000000000000000000000000000000"),
        "{e}"
    );
    assert_eq!(
        values(true, whole).unwrap(),
        [m4.as_str(), m4.as_str(), m4.as_str()]
    );
    // Beyond `i128` the exact sum still comes back: M + M + M - M - M.
    let (_, wide) = decimal_rows(&[m, m, m, -m, -m], 38);
    let op = decimal_window(AggregateFunction::Sum, whole, 38, true);
    let out = drive(&op, &ctx, &[wide]).unwrap();
    assert_eq!(column_text(&out, 2)[0], ms);
}

/// Decimal `avg` of an exact sum: ANSI off NULL past the result's
/// precision, ANSI on Spark's error with the scale-39 quotient.
#[tokio::test]
async fn decimal_averages_round_as_spark() {
    let m = 10_i128.pow(38) - 1;
    let (_, input) = decimal_rows(&[m, 1, -5], 38);
    let (_root, ctx) = task(1 << 30);
    let running = WindowFrame::running(FrameUnits::Rows);
    let op = decimal_window(AggregateFunction::Avg, running, 38, false);
    let out = drive(&op, &ctx, std::slice::from_ref(&input)).unwrap();
    assert_eq!(column_text(&out, 2), ["null"; 3]);
    let op = decimal_window(AggregateFunction::Avg, running, 38, true);
    let e = drive(&op, &ctx, &[input]).unwrap_err();
    assert!(
        e.to_string().contains(
            "99999999999999999999999999999999999999.000000000000000000000000000000000000000"
        ),
        "{e}"
    );
    // Within range: (1 + 2) / 2 = 1.5000 in decimal(14, 4).
    let (_, small) = decimal_rows(&[1, 2], 10);
    let op = decimal_window(
        AggregateFunction::Avg,
        WindowFrame::whole(FrameUnits::Rows),
        10,
        true,
    );
    let out = drive(&op, &ctx, &[small]).unwrap();
    assert_eq!(column_text(&out, 2), ["1.5000", "1.5000"]);
}

/// `FILTER (WHERE ...)`: the aggregate reads the rows where the filter is
/// true (not NULL); `first` with `IGNORE NULLS` skips NULL values.
#[tokio::test]
async fn filters_and_ignore_nulls() {
    let s = Arc::new(Schema::new(vec![
        Field::new("o", DataType::Int64, false),
        Field::new("v", DataType::Int64, true),
        Field::new("f", DataType::Boolean, true),
    ]));
    let b = RecordBatch::try_new(
        Arc::clone(&s),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int64Array::from(vec![None, Some(10), Some(20), None])),
            Arc::new(BooleanArray::from(vec![
                Some(true),
                None,
                Some(true),
                Some(false),
            ])),
        ],
    )
    .unwrap();
    let f = |function, filter, frame| WindowFunctionPlan {
        function: WindowFunction::Aggregate(AggregateExpr {
            function,
            args: vec![1],
            input: AggInput::Raw,
            arg_type: DataType::Int64,
            name: "x".to_owned(),
        }),
        frame,
        filter,
        name: "x".to_owned(),
    };
    let running = WindowFrame::running(FrameUnits::Rows);
    let plan = WindowPlan {
        partition_by: Vec::new(),
        order_by: vec![SortKey::asc(0)],
        functions: vec![
            f(AggregateFunction::Sum, Some(2), running),
            f(
                AggregateFunction::Count,
                Some(2),
                WindowFrame::whole(FrameUnits::Rows),
            ),
            f(
                AggregateFunction::First { ignore_nulls: true },
                None,
                running,
            ),
            f(
                AggregateFunction::Last {
                    ignore_nulls: false,
                },
                None,
                running,
            ),
        ],
    };
    let op = Window::new(Arc::clone(&s), plan, true).unwrap();
    let (_root, ctx) = task(1 << 30);
    let out = drive(&op, &ctx, &[Batch::new(b)]).unwrap();
    assert_eq!(column_text(&out, 3), ["null", "null", "20", "20"]);
    assert_eq!(column_text(&out, 4), ["1", "1", "1", "1"]);
    assert_eq!(column_text(&out, 5), ["null", "10", "10", "10"]);
    assert_eq!(column_text(&out, 6), ["null", "10", "20", "null"]);
    let m = op.extra_metrics();
    assert!(
        m.iter()
            .any(|(k, v)| *k == "partitions" && *v == crate::Metric::Count(1)),
        "{m:?}"
    );
}

#[tokio::test]
async fn memory_is_reserved_before_a_batch_is_taken() {
    let rows: Vec<Row> = (0..1000)
        .map(|i| (Some(i % 7), Some(i), i, Some(i)))
        .collect();
    let window = plan(functions(), true);
    let op = Window::new(schema(), window.clone(), true).unwrap();
    let (root, ctx) = task(64 << 10);
    let mut local = op.local_state(&ctx).unwrap();
    match op.execute(&batch(&rows), &mut local).unwrap() {
        OperatorOutput::NeedMemory { bytes } => assert!(bytes > 64 << 10, "{bytes}"),
        other => panic!("expected NeedMemory, got {other:?}"),
    }
    // Nothing was taken: the operator finishes empty.
    assert!(op.finish(&mut local).unwrap().is_none());
    drop((local, ctx));
    assert_eq!(root.reserved(), 0);
    // Through the runner, a pool that cannot hold the partition fails the
    // query with the memory error, and frees everything.
    let e = run(&rows, 1000, &window, 96 << 10).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory, "{e}");
}

#[tokio::test]
async fn weights_other_than_one_are_refused() {
    let op = Window::new(schema(), plan(functions(), true), true).unwrap();
    let (_root, ctx) = task(1 << 30);
    let mut local = op.local_state(&ctx).unwrap();
    let rows: Vec<Row> = vec![(Some(1), Some(1), 0, Some(1))];
    let weighted = Batch::with_weights(batch(&rows).data().clone(), Weights::Const(2)).unwrap();
    let e = op.execute(&weighted, &mut local).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("weights other than 1"), "{e}");
}

#[tokio::test]
async fn empty_input_gives_no_rows() {
    let op = Window::new(schema(), plan(functions(), false), true).unwrap();
    let (_root, ctx) = task(1 << 30);
    let out = drive(&op, &ctx, &[batch(&[])]).unwrap();
    assert!(out.is_empty());
    assert_eq!(op.schema().fields().len(), 4 + functions().len());
}

#[test]
fn invalid_plans_are_rejected() {
    let window = |f: WindowFunctionPlan, partition_by: Vec<usize>| WindowPlan {
        partition_by,
        order_by: vec![SortKey::asc(1)],
        functions: vec![f],
    };
    let f = |function, frame, filter| WindowFunctionPlan {
        function,
        frame,
        filter,
        name: "x".to_owned(),
    };
    let rows = WindowFrame::running(FrameUnits::Rows);
    let class = |p: WindowPlan| Window::new(schema(), p, true).unwrap_err().error_class();
    let internal = ErrorClass::InternalError;
    assert_eq!(
        class(window(f(WindowFunction::RowNumber, rows, None), vec![9])),
        internal
    );
    assert_eq!(
        class(window(f(WindowFunction::Ntile(0), rows, None), vec![])),
        internal
    );
    assert_eq!(
        class(window(f(WindowFunction::RowNumber, rows, Some(0)), vec![])),
        internal
    );
    assert_eq!(
        class(window(
            f(agg(AggregateFunction::Sum, Some(9)), rows, None),
            vec![]
        )),
        internal
    );
    let following = WindowFrame {
        units: FrameUnits::Rows,
        lower: FrameBound::CurrentRow,
        upper: FrameBound::UnboundedFollowing,
    };
    assert_eq!(
        class(window(
            f(agg(AggregateFunction::Sum, Some(3)), following, None),
            vec![]
        )),
        ErrorClass::KrishivUnsupportedFeature
    );
    let op = Window::new(schema(), plan(functions(), true), true).unwrap();
    assert!(format!("{op:?}").contains("Window"));
}
