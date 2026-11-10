// SPDX-License-Identifier: Apache-2.0
//! Window operator tests: a property test against a naive per-row
//! reference over random partitions, orderings and batch splits; Spark's
//! `NTile`, exact decimal frames, filters, memory, weights and plan checks.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]
#![expect(
    clippy::disallowed_methods,
    reason = "test-only runtime plumbing and temp-dir cleanup; never runs on a compute worker"
)]

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
            resident_guard: true,
        },
        Arc::new(TokioClock),
    )
}

/// `(p, o, id, v)`: partition key, order key (small codes, see
/// [`Keys`]), unique tie-break, value. Column 4 is a string derived from
/// `v` ([`text`]).
type Row = (Option<i64>, Option<i64>, i64, Option<i64>);

/// The type the partition and order key codes are written as.
#[derive(Clone, Copy, Debug)]
enum Keys {
    Int,
    /// `-0.0`, `0.0`, NaN, `1.5`, `-inf`: Spark's equal `-0.0 = 0.0` and
    /// NaN largest.
    Float,
    Decimal,
    /// Byte order differs from the codes' order.
    Text,
}

const FLOATS: [f64; 5] = [-0.0, 0.0, f64::NAN, 1.5, f64::NEG_INFINITY];
const TEXTS: [&str; 5] = ["b", "a", "\u{e9}", "A", "aa"];

impl Keys {
    fn data_type(self) -> DataType {
        match self {
            Self::Int => DataType::Int64,
            Self::Float => DataType::Float64,
            Self::Decimal => DataType::Decimal128(10, 2),
            Self::Text => DataType::Utf8,
        }
    }

    fn column(self, codes: impl Iterator<Item = Option<i64>>) -> ArrayRef {
        let at = |k: i64| usize::try_from(k).unwrap() % 5;
        match self {
            Self::Int => Arc::new(Int64Array::from_iter(codes)),
            Self::Float => Arc::new(arrow_array::Float64Array::from_iter(
                codes.map(|k| k.map(|k| FLOATS[at(k)])),
            )),
            Self::Decimal => Arc::new(
                Decimal128Array::from_iter(codes.map(|k| k.map(|k| i128::from(k) * 137 - 200)))
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
            Self::Text => Arc::new(arrow_array::StringArray::from_iter(
                codes.map(|k| k.map(|k| TEXTS[at(k)])),
            )),
        }
    }

    /// The code's position in Spark's order: NULL first, then by value
    /// (floats: `-0.0 = 0.0`, NaN largest; strings by bytes).
    fn key(self, v: Option<i64>) -> (bool, Vec<u8>) {
        let Some(k) = v else {
            return (false, Vec::new());
        };
        let at = usize::try_from(k).unwrap() % 5;
        let bytes = match self {
            Self::Int => (k ^ i64::MIN).to_be_bytes().to_vec(),
            Self::Decimal => ((i128::from(k) * 137 - 200) ^ i128::MIN)
                .to_be_bytes()
                .to_vec(),
            Self::Float => {
                let f = FLOATS[at];
                let f = if f.is_nan() { f64::NAN.abs() } else { f + 0.0 };
                let b = f.to_bits();
                let b = if b >> 63 == 1 { !b } else { b | (1 << 63) };
                b.to_be_bytes().to_vec()
            }
            Self::Text => TEXTS[at].as_bytes().to_vec(),
        };
        (true, bytes)
    }
}

/// The string column's value of a row's `v`.
fn text(v: Option<i64>) -> Option<String> {
    v.map(|x| format!("s{:02}", (x * 37) % 100))
}

fn schema() -> SchemaRef {
    schema_of(Keys::Int)
}

fn schema_of(keys: Keys) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("p", keys.data_type(), true),
        Field::new("o", keys.data_type(), true),
        Field::new("id", DataType::Int64, false),
        Field::new("v", DataType::Int64, true),
        Field::new("s", DataType::Utf8, true),
    ]))
}

fn batch(rows: &[Row]) -> Batch {
    batch_of(Keys::Int, rows)
}

fn batch_of(keys: Keys, rows: &[Row]) -> Batch {
    let columns: Vec<ArrayRef> = vec![
        keys.column(rows.iter().map(|r| r.0)),
        keys.column(rows.iter().map(|r| r.1)),
        Arc::new(Int64Array::from_iter(rows.iter().map(|r| Some(r.2)))),
        Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.3))),
        Arc::new(arrow_array::StringArray::from_iter(
            rows.iter().map(|r| text(r.3)),
        )),
    ];
    Batch::new(RecordBatch::try_new(schema_of(keys), columns).unwrap())
}

pub(super) fn agg(function: AggregateFunction, arg: Option<usize>) -> WindowFunction {
    WindowFunction::Aggregate(AggregateExpr {
        function,
        args: arg.into_iter().collect(),
        input: AggInput::Raw,
        mask: None,
        arg_type: match arg {
            Some(4) => DataType::Utf8View,
            Some(_) => DataType::Int64,
            None => DataType::Null,
        },
        name: "a".to_owned(),
        element: None,
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
    run_keys(Keys::Int, rows, split, window, limit).await
}

async fn run_keys(
    keys_as: Keys,
    rows: &[Row],
    split: usize,
    window: &WindowPlan,
    limit: usize,
) -> crate::Result<Vec<Batch>> {
    run_keys_ansi(keys_as, rows, split, window, limit, true).await
}

async fn run_keys_ansi(
    keys_as: Keys,
    rows: &[Row],
    split: usize,
    window: &WindowPlan,
    limit: usize,
    ansi: bool,
) -> crate::Result<Vec<Batch>> {
    let s = schema_of(keys_as);
    let batches: Vec<Batch> = rows
        .chunks(split.max(1))
        .map(|c| batch_of(keys_as, c))
        .collect();
    let mut keys: Vec<SortKey> = window
        .partition_by
        .iter()
        .map(|&c| SortKey::asc(c))
        .collect();
    keys.extend(window.order_by.iter().copied());
    keys.push(SortKey::asc(2));
    let sort = Arc::new(Sort::new(Arc::clone(&s), &keys, None)?);
    let op = Arc::new(Window::new(Arc::clone(&s), window.clone(), ansi, None)?);
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
        DataType::Utf8View => a.as_string_view().value(i).to_owned(),
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

/// The naive value of `f` (over `frame`) for each row of `rows`, which are
/// sorted by partition key, order key and id (keys written as `kt`); one
/// partition when not `partitioned`.
fn reference(
    kt: Keys,
    rows: &[Row],
    f: &WindowFunction,
    frame: WindowFrame,
    partitioned: bool,
) -> Vec<String> {
    let mut out = Vec::new();
    let key = |v: Option<i64>| kt.key(v);
    let same_part = |a: &Row, b: &Row| !partitioned || key(a.0) == key(b.0);
    let to_f = |x: usize| f64::from(u32::try_from(x).unwrap());
    for r in rows {
        let part: Vec<&Row> = rows.iter().filter(|x| same_part(x, r)).collect();
        let pos = part.iter().position(|x| x.2 == r.2).unwrap();
        let n = part.len();
        let rank = 1 + part.iter().filter(|x| key(x.1) < key(r.1)).count();
        let peers_end = part.iter().filter(|x| key(x.1) <= key(r.1)).count();
        let peers_start = rank - 1;
        let at = |k: i64| {
            let i = i64::try_from(pos).unwrap().saturating_add(k);
            usize::try_from(i.clamp(0, i64::try_from(n).unwrap())).unwrap()
        };
        let start = match (frame.units, frame.lower) {
            (_, FrameBound::UnboundedPreceding) => 0,
            (FrameUnits::Rows, FrameBound::CurrentRow) => pos,
            (FrameUnits::Range, FrameBound::CurrentRow) => peers_start,
            (_, FrameBound::Offset(k)) => at(k),
            (_, b) => {
                out.push(format!("<lower {b:?}>"));
                continue;
            }
        };
        let end = match (frame.units, frame.upper) {
            (_, FrameBound::UnboundedFollowing) => n,
            (FrameUnits::Rows, FrameBound::CurrentRow) => pos + 1,
            (FrameUnits::Range, FrameBound::CurrentRow) => peers_end,
            (_, FrameBound::Offset(k)) => at(k + 1),
            (_, b) => {
                out.push(format!("<upper {b:?}>"));
                continue;
            }
        };
        let show = |v: Option<i64>| v.map_or_else(|| "null".to_owned(), |v| v.to_string());
        out.push(match f {
            WindowFunction::Offset {
                offset,
                default,
                ignore_nulls,
                ..
            } => {
                let p = i64::try_from(pos).unwrap();
                let target = if !*ignore_nulls || *offset == 0 {
                    usize::try_from(p + offset).ok().filter(|&j| j < n)
                } else if *offset > 0 {
                    let after: Vec<usize> = (pos + 1..n).filter(|&j| part[j].3.is_some()).collect();
                    after.get(usize::try_from(offset - 1).unwrap()).copied()
                } else {
                    let before: Vec<usize> =
                        (0..pos).rev().filter(|&j| part[j].3.is_some()).collect();
                    before.get(usize::try_from(-offset - 1).unwrap()).copied()
                };
                match (target, default) {
                    (Some(j), _) => show(part[j].3),
                    // The default is the `id` column.
                    (None, Some(_)) => r.2.to_string(),
                    (None, None) => "null".to_owned(),
                }
            }
            WindowFunction::NthValue {
                n: nth,
                ignore_nulls,
                ..
            } => {
                let frame_rows = &part[start..end.max(start)];
                let k = usize::try_from(nth - 1).unwrap();
                let v = if *ignore_nulls {
                    frame_rows.iter().filter_map(|x| x.3).nth(k)
                } else {
                    frame_rows.get(k).and_then(|x| x.3)
                };
                show(v)
            }
            WindowFunction::RowNumber => (pos + 1).to_string(),
            WindowFunction::Rank => rank.to_string(),
            WindowFunction::DenseRank => {
                let mut keys: Vec<(bool, Vec<u8>)> = part
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
                let frame_rows = &part[start..end.max(start)];
                if a.args == [4] {
                    // The string column: by bytes; first/last by row.
                    let texts: Vec<Option<String>> = frame_rows.iter().map(|x| text(x.3)).collect();
                    let set: Vec<&String> = texts.iter().flatten().collect();
                    let v = match &a.function {
                        AggregateFunction::Min => set.iter().min().map(|s| (*s).clone()),
                        AggregateFunction::Max => set.iter().max().map(|s| (*s).clone()),
                        AggregateFunction::First { ignore_nulls: true } => {
                            set.first().map(|s| (*s).clone())
                        }
                        AggregateFunction::Last { ignore_nulls: true } => {
                            set.last().map(|s| (*s).clone())
                        }
                        AggregateFunction::First { .. } => texts.first().cloned().flatten(),
                        AggregateFunction::Last { .. } => texts.last().cloned().flatten(),
                        other => Some(format!("<{other:?}>")),
                    };
                    v.unwrap_or_else(|| "null".to_owned())
                } else {
                    let vals: Vec<i64> = frame_rows.iter().filter_map(|x| x.3).collect();
                    match &a.function {
                        AggregateFunction::Count if a.args.is_empty() => {
                            frame_rows.len().to_string()
                        }
                        AggregateFunction::Count => vals.len().to_string(),
                        AggregateFunction::First { ignore_nulls } => {
                            let mut picked = frame_rows.iter().map(|x| x.3);
                            let v = if *ignore_nulls {
                                vals.first().copied()
                            } else {
                                picked.next().flatten()
                            };
                            v.map_or_else(|| "null".to_owned(), |v| v.to_string())
                        }
                        AggregateFunction::Last { ignore_nulls } => {
                            let v = if *ignore_nulls {
                                vals.last().copied()
                            } else {
                                frame_rows.last().and_then(|x| x.3)
                            };
                            v.map_or_else(|| "null".to_owned(), |v| v.to_string())
                        }
                        _ if vals.is_empty() => "null".to_owned(),
                        AggregateFunction::Sum => vals.iter().sum::<i64>().to_string(),
                        AggregateFunction::Min => vals.iter().min().unwrap().to_string(),
                        AggregateFunction::Max => vals.iter().max().unwrap().to_string(),
                        AggregateFunction::Avg => {
                            // Spark sums integers as doubles, in order.
                            #[expect(
                                clippy::cast_precision_loss,
                                reason = "Spark's double average"
                            )]
                            let s = vals.iter().fold(0.0, |s, v| s + *v as f64);
                            format!("{:?}", s / to_f(vals.len()))
                        }
                        AggregateFunction::AnyValue { .. }
                        | AggregateFunction::MaxBy
                        | AggregateFunction::MinBy
                        | AggregateFunction::StddevSamp
                        | AggregateFunction::StddevPop
                        | AggregateFunction::VarSamp
                        | AggregateFunction::VarPop
                        | AggregateFunction::Skewness
                        | AggregateFunction::Kurtosis
                        | AggregateFunction::Corr
                        | AggregateFunction::CovarSamp
                        | AggregateFunction::CovarPop
                        | AggregateFunction::RegrCount
                        | AggregateFunction::RegrAvgx
                        | AggregateFunction::RegrAvgy
                        | AggregateFunction::RegrSxx
                        | AggregateFunction::RegrSyy
                        | AggregateFunction::RegrSxy
                        | AggregateFunction::RegrSlope
                        | AggregateFunction::RegrIntercept
                        | AggregateFunction::RegrR2
                        | AggregateFunction::BitAnd
                        | AggregateFunction::BitOr
                        | AggregateFunction::BitXor
                        | AggregateFunction::Mode
                        | AggregateFunction::ModeMin
                        | AggregateFunction::ModeMax
                        | AggregateFunction::ListAgg { .. }
                        | AggregateFunction::TrySum
                        | AggregateFunction::TryAvg
                        | AggregateFunction::CollectList
                        | AggregateFunction::CollectSet
                        | AggregateFunction::VectorSum
                        | AggregateFunction::VectorAvg
                        | AggregateFunction::Percentile { .. }
                        | AggregateFunction::PercentileApprox { .. }
                        | AggregateFunction::PercentileDisc { .. }
                        | AggregateFunction::ApproxCountDistinct { .. }
                        | AggregateFunction::Udf(_) => {
                            unreachable!("no window test aggregates with this function")
                        }
                    }
                }
            }
        });
    }
    out
}

fn functions() -> Vec<(WindowFunction, WindowFrame)> {
    use AggregateFunction::{Avg, Count, First, Last, Max, Min, Sum};
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
        (agg(Max, Some(4)), rows),
        (agg(Min, Some(4)), range),
        (agg(First { ignore_nulls: true }, Some(4)), rows),
        (
            agg(
                Last {
                    ignore_nulls: false,
                },
                Some(4),
            ),
            range,
        ),
        (agg(Last { ignore_nulls: true }, Some(4)), whole),
        (
            agg(
                First {
                    ignore_nulls: false,
                },
                Some(4),
            ),
            whole,
        ),
    ]
}

/// Frames that do not start at the partition's first row or end at the
/// current row (P2-T055 part 2), over every aggregate shape.
fn frame_functions() -> Vec<(WindowFunction, WindowFrame)> {
    use AggregateFunction::{Avg, Count, First, Last, Max, Min, Sum};
    let rows = |lower, upper| WindowFrame {
        units: FrameUnits::Rows,
        lower,
        upper,
    };
    let around = rows(FrameBound::Offset(-1), FrameBound::Offset(1));
    let ahead = rows(FrameBound::Offset(2), FrameBound::Offset(3));
    let before = rows(FrameBound::UnboundedPreceding, FrameBound::Offset(-1));
    let to_end = rows(FrameBound::Offset(-1), FrameBound::UnboundedFollowing);
    let rows_rest = rows(FrameBound::CurrentRow, FrameBound::UnboundedFollowing);
    let range_rest = WindowFrame {
        units: FrameUnits::Range,
        lower: FrameBound::CurrentRow,
        upper: FrameBound::UnboundedFollowing,
    };
    let empty = rows(FrameBound::Offset(1), FrameBound::CurrentRow);
    vec![
        (agg(Sum, Some(3)), around),
        (agg(Count, None), ahead),
        (agg(Count, Some(3)), around),
        (agg(Min, Some(3)), to_end),
        (agg(Max, Some(3)), before),
        (agg(Avg, Some(3)), around),
        (agg(Sum, Some(3)), range_rest),
        (agg(Sum, Some(3)), empty),
        (
            agg(
                First {
                    ignore_nulls: false,
                },
                Some(3),
            ),
            ahead,
        ),
        (agg(Last { ignore_nulls: true }, Some(3)), around),
        (agg(Max, Some(4)), around),
        (agg(Min, Some(4)), range_rest),
        (agg(First { ignore_nulls: true }, Some(4)), to_end),
        (
            agg(
                Last {
                    ignore_nulls: false,
                },
                Some(4),
            ),
            before,
        ),
        (agg(Last { ignore_nulls: true }, Some(4)), rows_rest),
        (
            agg(
                First {
                    ignore_nulls: false,
                },
                Some(4),
            ),
            empty,
        ),
        (nth(2, false), WindowFrame::running(FrameUnits::Range)),
        (nth(1, true), around),
        (nth(3, true), WindowFrame::whole(FrameUnits::Rows)),
        (nth(2, false), to_end),
        (
            lag_lead(-1, None, false),
            rows(FrameBound::Offset(-1), FrameBound::Offset(-1)),
        ),
        (
            lag_lead(2, Some(2), false),
            rows(FrameBound::Offset(2), FrameBound::Offset(2)),
        ),
        (
            lag_lead(-2, Some(2), true),
            rows(FrameBound::Offset(-2), FrameBound::Offset(-2)),
        ),
        (
            lag_lead(1, None, true),
            rows(FrameBound::Offset(1), FrameBound::Offset(1)),
        ),
        (
            lag_lead(0, None, true),
            rows(FrameBound::Offset(0), FrameBound::Offset(0)),
        ),
    ]
}

/// `nth_value(v, n)` (column 3).
fn nth(n: i64, ignore_nulls: bool) -> WindowFunction {
    WindowFunction::NthValue {
        input: 3,
        n,
        ignore_nulls,
        data_type: DataType::Int64,
    }
}

/// `lead(v, offset, id)` (`lag` for a negative offset; the default is the
/// `id` column when given).
fn lag_lead(offset: i64, default: Option<usize>, ignore_nulls: bool) -> WindowFunction {
    WindowFunction::Offset {
        input: 3,
        offset,
        default,
        ignore_nulls,
        data_type: DataType::Int64,
    }
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

pub(super) fn task(limit: usize) -> (Arc<TreePool>, TaskContext) {
    task_with(limit, ExecConfig::default())
}

fn task_with(limit: usize, config: ExecConfig) -> (Arc<TreePool>, TaskContext) {
    let root = pool(limit);
    let q = QueryContext::new(
        QueryId(7),
        WorkloadClass::Batch,
        config,
        root.child("q", Some(limit)),
    );
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
    let ctx = TaskContext::new(q, executor, SpillOwner::new(), OperatorId(0), "window");
    (root, ctx)
}

/// Runs `op` over `input` directly (one driver), collecting its output.
pub(super) fn drive(op: &Window, ctx: &TaskContext, input: &[Batch]) -> crate::Result<Vec<Batch>> {
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
                mask: None,
                arg_type: DataType::Decimal128(p, 0),
                name: "s".to_owned(),
                element: None,
            }),
            frame,
            filter: None,
            name: "s".to_owned(),
        }],
    };
    Window::new(s, plan, ansi, None).unwrap()
}

fn column_text(out: &[Batch], column: usize) -> Vec<String> {
    out.iter()
        .flat_map(|b| {
            let c = b.data().column(column);
            (0..c.len()).map(|i| cell(c, i)).collect::<Vec<_>>()
        })
        .collect()
}

/// Sorted rows in `parts` partitions with ties in the order key, and
/// their boundary flags.
fn sorted_unit(rows: i64, parts: i64) -> (Batch, Vec<bool>, Vec<bool>) {
    let data: Vec<Row> = (0..rows)
        .map(|i| (Some(i * parts / rows), Some(i / 3), i, Some(i % 11)))
        .collect();
    let b = batch(&data);
    let part: Vec<bool> = (0..data.len())
        .map(|i| i == 0 || data[i].0 != data[i - 1].0)
        .collect();
    let peer: Vec<bool> = (0..data.len())
        .map(|i| part[i] || data[i].1 != data[i - 1].1)
        .collect();
    (b, part, peer)
}

// Spilling the buffered rows (P2-T055 part 2).

/// A spill directory of its own: the manager deletes what it wrote, and
/// the guard the directory itself.
pub(super) struct SpillDir {
    manager: Option<krishiv_memory::SpillManager>,
    dir: std::path::PathBuf,
}

impl Drop for SpillDir {
    fn drop(&mut self) {
        drop(self.manager.take());
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

pub(super) fn spill_scope() -> (SpillDir, krishiv_memory::SpillScope) {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join(format!("window-unit-{}-{n}", std::process::id()));
    let manager = krishiv_memory::SpillManager::open(&krishiv_memory::SpillConfig {
        dir: dir.clone(),
        ..krishiv_memory::SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    (
        SpillDir {
            manager: Some(manager),
            dir,
        },
        scope,
    )
}

/// How [`drive_spilling`] spills.
#[derive(Clone, Copy, Debug)]
pub(super) struct Spills {
    /// The buffered rows spill after every this many batches (0: only when
    /// the operator needs memory).
    pub(super) every: usize,
    /// After the input, all but this many free bytes of the pool are taken,
    /// so partitions that do not fit are evaluated from disk.
    pub(super) slack: Option<usize>,
}

/// Runs `op` over `input` directly as a driver does, with spills (see
/// [`Spills`]); a request for memory in the flush phase is granted while
/// the pool has it. Collects the output.
pub(super) async fn drive_spilling(
    op: &Window,
    ctx: &TaskContext,
    limit: usize,
    root: &TreePool,
    input: &[Batch],
    spills: Spills,
) -> crate::Result<Vec<Batch>> {
    let mut local = op.local_state(ctx)?;
    let mut out = Vec::new();
    for (k, b) in input.iter().enumerate() {
        loop {
            match op.execute(b, &mut local)? {
                OperatorOutput::NeedMoreInput(o) => {
                    out.extend(o);
                    break;
                }
                OperatorOutput::HaveMoreOutput(o) => out.push(o),
                OperatorOutput::NeedMemory { bytes } => {
                    if op.spill(&mut local, bytes, ctx).await? == 0 {
                        return Err(crate::error::internal("nothing spilled".to_owned()));
                    }
                }
                OperatorOutput::Finished(_) => {
                    return Err(crate::error::internal("unexpected Finished".to_owned()));
                }
            }
        }
        if spills.every > 0 && (k + 1) % spills.every == 0 {
            op.spill(&mut local, 1, ctx).await?;
        }
    }
    while let Some(b) = op.finish(&mut local)? {
        out.push(b);
    }
    op.end_input(&mut local, ctx).await?;
    let pool = Arc::clone(ctx.query().pool());
    let mut hog = krishiv_memory::Reservation::new(Arc::clone(&pool), pool.register("hog", None));
    if let Some(slack) = spills.slack {
        take_all_but(&mut hog, limit.saturating_sub(root.reserved()), slack);
    }
    loop {
        match op.flush(&mut local, ctx).await? {
            crate::operator::FlushOutput::Batch(b) => out.push(b),
            crate::operator::FlushOutput::Continue => {}
            crate::operator::FlushOutput::Done => break,
            crate::operator::FlushOutput::NeedMemory { bytes } => {
                let mut r = ctx.consumer().reservation();
                if r.try_grow(bytes).is_ok() {
                    r.shrink_to_credit(bytes);
                }
            }
        }
    }
    drop((hog, local));
    ctx.consumer().consumer().release_credit();
    Ok(out)
}

/// Grows `hog` by all the pool grants (trying from `free` bytes down),
/// then gives back `slack`.
fn take_all_but(hog: &mut krishiv_memory::Reservation, free: usize, slack: usize) {
    let mut step = free;
    while step > 0 {
        if hog.try_grow(step).is_err() {
            step /= 2;
        }
    }
    hog.shrink(slack);
}

/// `drive_spilling` next to `drive` without a spill scope: the outputs
/// must be equal, row for row and in order. Returns the spilling window's
/// counters (spills, external partitions).
async fn compare_spilling(
    rows: &[Row],
    split: usize,
    window: &WindowPlan,
    spills: Spills,
) -> (u64, u64) {
    let counters = compared_spilling(rows, split, window, spills).await;
    (
        counters.spills.load(std::sync::atomic::Ordering::Relaxed),
        counters.external.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// [`compare_spilling`], returning the operator's counters.
async fn compared_spilling(
    rows: &[Row],
    split: usize,
    window: &WindowPlan,
    spills: Spills,
) -> Arc<super::spill::Counters> {
    let batches: Vec<Batch> = rows.chunks(split.max(1)).map(batch).collect();
    let mut sorted = batches.clone();
    // Sorted by partition, order and id, as the sort below a window gives.
    let all: Vec<Row> = {
        let mut v = rows.to_vec();
        let p = |r: &Row| {
            if window.partition_by.is_empty() {
                None
            } else {
                r.0
            }
        };
        let key = window.order_by.first().copied().unwrap_or(SortKey::asc(1));
        v.sort_by_key(|r| {
            (
                p(r).map(|x| x + 1).unwrap_or(0),
                match r.1 {
                    None if key.nulls_first => i64::MIN,
                    None => i64::MAX,
                    Some(x) if key.descending => -x,
                    Some(x) => x,
                },
                r.2,
            )
        });
        v
    };
    sorted.clear();
    sorted.extend(all.chunks(split.max(1)).map(batch));
    let plain = Window::new(schema(), window.clone(), true, None).unwrap();
    let (_r, ctx) = task(1 << 30);
    let want = outputs(
        &drive(&plain, &ctx, &sorted).unwrap(),
        window.functions.len(),
    );
    let (_manager, scope) = spill_scope();
    let op = Window::new(schema(), window.clone(), true, Some(scope)).unwrap();
    let limit = 1 << 30;
    // Batches of 4 rows: complete rows reach a batch while rows wait after
    // a spill, and in the flush phase.
    let config = ExecConfig {
        batch_size: 4,
        ..ExecConfig::default()
    };
    let (root, ctx) = task_with(limit, config);
    let got = drive_spilling(&op, &ctx, limit, &root, &sorted, spills)
        .await
        .unwrap();
    assert_eq!(outputs(&got, window.functions.len()), want, "{spills:?}");
    drop((got, ctx));
    assert_eq!(root.reserved(), 0, "the query pool is empty at the end");
    Arc::clone(&op.counters)
}

fn big_rows_strategy() -> impl Strategy<Value = Vec<Row>> {
    let opt = |hi: i64| prop_oneof![1 => Just(None), 4 => (0..hi).prop_map(Some)];
    prop::collection::vec((opt(4), opt(6), opt(100)), 0..240).prop_map(|v| {
        v.into_iter()
            .enumerate()
            .map(|(i, (p, o, x))| (p, o, i64::try_from(i).unwrap(), x))
            .collect()
    })
}

/// [`spilled_batches_of_alternating_sizes_read_back_from_a_full_pool`] with batches of 20 and `large` rows: the
/// external partition's whole-batch evaluation does not fit, so it is evaluated in fewer rows per step.
async fn alternating_sizes_from_a_full_pool(
    large: usize,
    functions: Vec<(WindowFunction, WindowFrame)>,
    narrows: bool,
) {
    let rows: Vec<Row> = (0..3_000)
        .map(|i| (Some(0), Some(i), i, Some(i % 1_000)))
        .collect();
    let window = plan(functions, false);
    // Batches of 20 and 300 rows, alternating (the rows are already in order).
    let mut sorted = Vec::new();
    let (mut at, mut k) = (0, 0);
    while at < rows.len() {
        let n = if k % 2 == 0 { 20 } else { large }.min(rows.len() - at);
        sorted.push(batch(&rows[at..at + n]));
        at += n;
        k += 1;
    }
    let plain = Window::new(schema(), window.clone(), true, None).unwrap();
    let (_r, ctx) = task(1 << 30);
    let want = outputs(
        &drive(&plain, &ctx, &sorted).unwrap(),
        window.functions.len(),
    );
    let (_manager, scope) = spill_scope();
    let op = Window::new(schema(), window.clone(), true, Some(scope)).unwrap();
    let limit = 1 << 30;
    let (root, ctx) = task(limit);
    let spills = Spills {
        every: 1,
        slack: Some(0),
    };
    let got = drive_spilling(&op, &ctx, limit, &root, &sorted, spills)
        .await
        .unwrap();
    assert!(
        op.counters
            .spills
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0,
        "the window spilled"
    );
    if narrows {
        assert!(
            op.counters
                .narrowed
                .load(std::sync::atomic::Ordering::Relaxed)
                > 0,
            "the external partition was evaluated in narrower steps ({large} rows)"
        );
    }
    assert_eq!(outputs(&got, window.functions.len()), want, "{large} rows");
    drop((got, ctx));
    assert_eq!(root.reserved(), 0, "the query pool is empty at the end");
}

/// A source of `count` batches made on demand by `make(k)` (an input
/// larger than the test should hold).
struct Generated<F> {
    schema: SchemaRef,
    count: i64,
    make: F,
}

impl<F: Fn(i64) -> Batch + Send + Sync> Generated<F> {
    fn new(schema: SchemaRef, count: i64, make: F) -> Arc<Self> {
        Arc::new(Self {
            schema,
            count,
            make,
        })
    }
}

impl<F: Fn(i64) -> Batch + Send + Sync> crate::Source for Generated<F> {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn global_state(&self, ctx: &TaskContext) -> crate::Result<crate::operator::GlobalState> {
        let n = usize::try_from(self.count).unwrap_or(0);
        Ok(Arc::new(crate::MorselCursor::new(
            n,
            ctx.query().config().seed,
        )))
    }

    fn max_parallelism(&self, _global: &crate::operator::GlobalState) -> usize {
        1
    }

    fn local_state(
        &self,
        _ctx: &TaskContext,
        _global: &crate::operator::GlobalState,
    ) -> crate::Result<crate::operator::LocalState> {
        Ok(Box::new(()))
    }

    fn get_data<'a>(
        &'a self,
        _local: &'a mut crate::operator::LocalState,
        global: &'a crate::operator::GlobalState,
    ) -> futures::future::BoxFuture<'a, crate::Result<Option<Batch>>> {
        let k = global
            .downcast_ref::<crate::MorselCursor>()
            .and_then(crate::MorselCursor::claim);
        let b = k.map(|k| (self.make)(i64::try_from(k).unwrap()));
        Box::pin(std::future::ready(Ok(b)))
    }
}

/// Rows per partition of [`one_large_partition_spills_through_the_runner`]
/// (`KRISHIV_WINDOW_SPILL_ROWS`; the plan's 2 GB case is about 40 million),
/// and the query's memory limit (`KRISHIV_WINDOW_SPILL_LIMIT`).
fn large_case() -> (i64, usize) {
    let var = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<u64>().ok());
    let rows = var("KRISHIV_WINDOW_SPILL_ROWS").map_or(200_000, |v| i64::try_from(v).unwrap());
    let limit = var("KRISHIV_WINDOW_SPILL_LIMIT").map_or(8 << 20, |v| usize::try_from(v).unwrap());
    (rows, limit)
}

async fn refused_on_disk(function: WindowFunction, frame: WindowFrame) {
    let rows: Vec<Row> = (0..3_000)
        .map(|i| (Some(0), Some(i), i, Some(i % 7)))
        .collect();
    let window = plan(vec![(function, frame)], true);
    let (_dir, scope) = spill_scope();
    let op = Window::new(schema(), window, true, Some(scope)).unwrap();
    let limit = 1 << 30;
    let (root, ctx) = task(limit);
    let batches: Vec<Batch> = rows.chunks(100).map(batch).collect();
    let spills = Spills {
        every: 1,
        slack: Some(96 << 10),
    };
    let e = drive_spilling(&op, &ctx, limit, &root, &batches, spills)
        .await
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    assert!(e.to_string().contains("larger than memory"), "{e}");
}

/// [`run_keys`] over a sort bucketed by the partitioning keys, read by four
/// drivers: each driver's window sees whole partitions, in order.
async fn run_bucketed(
    keys_as: Keys,
    rows: &[Row],
    split: usize,
    window: &WindowPlan,
) -> crate::Result<Vec<Batch>> {
    let s = schema_of(keys_as);
    let batches: Vec<Batch> = rows
        .chunks(split.max(1))
        .map(|c| batch_of(keys_as, c))
        .collect();
    let mut keys: Vec<SortKey> = window
        .partition_by
        .iter()
        .map(|&c| SortKey::asc(c))
        .collect();
    keys.extend(window.order_by.iter().copied());
    keys.push(SortKey::asc(2));
    let sort = Sort::new(Arc::clone(&s), &keys, None)?.with_hash_buckets(&window.partition_by)?;
    let sort = Arc::new(sort);
    let op = Arc::new(Window::new(Arc::clone(&s), window.clone(), true, None)?);
    let plan: Arc<dyn ExecNode> = Arc::new(OperatorNode {
        input: Arc::new(BreakerNode {
            input: Arc::new(SourceNode(Batches::new(s, batches))),
            sink: Arc::clone(&sort) as Arc<dyn crate::Sink>,
            source: Arc::clone(&sort) as Arc<dyn crate::Source>,
            ordered: false,
        }),
        op,
    });
    let runner = QueryRunner::new(TokioExecutor::deterministic(4)?);
    let root = pool(1 << 30);
    let config = ExecConfig {
        batch_size: 4,
        ..ExecConfig::default()
    };
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        config,
        root.child("q", Some(1 << 30)),
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
    assert_eq!(sort.metrics().buckets, 16);
    result
}

#[path = "group_limit_tests.rs"]
mod group_limit_tests;

/// A frame from `lower` to `upper`.
pub(super) fn between(units: FrameUnits, lower: FrameBound, upper: FrameBound) -> WindowFrame {
    WindowFrame {
        units,
        lower,
        upper,
    }
}

/// Every frame a partition on disk evaluates through `disk.rs`.
fn disk_frames() -> Vec<WindowFrame> {
    use FrameBound::{CurrentRow, Offset, RangeInt, UnboundedFollowing, UnboundedPreceding};
    use FrameUnits::{Range, Rows};
    vec![
        between(Rows, Offset(-2), Offset(3)),
        between(Rows, Offset(-1), CurrentRow),
        between(Rows, Offset(2), Offset(1)),
        between(Rows, Offset(40), Offset(60)),
        // Whole batches of empty frames come before the first row a frame reads.
        between(Rows, Offset(-1000), Offset(-900)),
        between(Rows, CurrentRow, UnboundedFollowing),
        between(Rows, Offset(-3), UnboundedFollowing),
        between(Rows, UnboundedPreceding, Offset(-1)),
        between(Rows, UnboundedPreceding, Offset(2)),
        between(Range, CurrentRow, UnboundedFollowing),
        between(Range, RangeInt(-3), RangeInt(2)),
        between(Range, RangeInt(-4), CurrentRow),
        between(Range, CurrentRow, RangeInt(4)),
        between(Range, RangeInt(1), RangeInt(3)),
        between(Range, UnboundedPreceding, RangeInt(1)),
        between(Range, UnboundedPreceding, RangeInt(-1)),
        between(Range, RangeInt(-2), UnboundedFollowing),
    ]
}

/// The aggregates, of integers and strings, over `frame`, and `nth_value`.
fn over(frame: WindowFrame) -> Vec<(WindowFunction, WindowFrame)> {
    use AggregateFunction::{Avg, Count, First, Last, Max, Min, Sum};
    let first = First {
        ignore_nulls: false,
    };
    let last = Last { ignore_nulls: true };
    vec![
        (agg(Sum, Some(3)), frame),
        (agg(Avg, Some(3)), frame),
        (agg(Count, None), frame),
        (agg(Min, Some(3)), frame),
        (agg(first.clone(), Some(3)), frame),
        (agg(last.clone(), Some(3)), frame),
        (agg(Max, Some(4)), frame),
        (agg(first, Some(4)), frame),
        (agg(last, Some(4)), frame),
        (nth(2, false), frame),
        (nth(3, true), frame),
    ]
}

fn skewed_rows(n: i64) -> Vec<Row> {
    (0..n)
        .map(|i| {
            // One large partition between two small ones.
            let p = if i < 50 {
                0
            } else if i < n - 50 {
                1
            } else {
                2
            };
            (
                Some(p),
                (i % 7 != 0).then_some(i / 5),
                i,
                (i % 13 != 0).then_some(i % 17),
            )
        })
        .collect()
}

mod part1;
mod part2;
