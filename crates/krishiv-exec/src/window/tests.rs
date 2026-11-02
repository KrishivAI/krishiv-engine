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

fn agg(function: AggregateFunction, arg: Option<usize>) -> WindowFunction {
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
                        | AggregateFunction::ApproxCountDistinct { .. } => {
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Every function over random partitions, orderings with ties and NULLs,
    /// and batch splits (partitions span batches) gives the naive value.
    #[test]
    fn window_matches_a_naive_reference(
        rows in rows_strategy(),
        split in 1_usize..9,
        partitioned in any::<bool>(),
        kt in prop_oneof![Just(Keys::Int), Just(Keys::Float), Just(Keys::Decimal), Just(Keys::Text)],
    ) {
        let functions = functions();
        let window = plan(functions.clone(), partitioned);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let out = rt.block_on(run_keys(kt, &rows, split, &window, 1 << 30)).unwrap();
        let got = outputs(&out, functions.len());
        let mut sorted = rows;
        sorted.sort_by_key(|r| {
            let p = if partitioned { kt.key(r.0) } else { (false, Vec::new()) };
            (p, kt.key(r.1), r.2)
        });
        let columns: Vec<Vec<String>> = functions
            .iter()
            .map(|(f, frame)| reference(kt, &sorted, f, *frame, partitioned))
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Frames with offsets and frames to the partition's end give the
    /// naive value over each row's frame, with ANSI on (integer sums in
    /// order) and off (integer sums through the segment tree).
    #[test]
    fn frames_match_a_naive_reference(
        rows in rows_strategy(),
        split in 1_usize..9,
        partitioned in any::<bool>(),
        ansi in any::<bool>(),
    ) {
        let functions = frame_functions();
        let window = plan(functions.clone(), partitioned);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let out = rt
            .block_on(run_keys_ansi(Keys::Int, &rows, split, &window, 1 << 30, ansi))
            .unwrap();
        let got = outputs(&out, functions.len());
        let mut sorted = rows;
        sorted.sort_by_key(|r| {
            let p = if partitioned { Keys::Int.key(r.0) } else { (false, Vec::new()) };
            (p, Keys::Int.key(r.1), r.2)
        });
        let columns: Vec<Vec<String>> = functions
            .iter()
            .map(|(f, frame)| reference(Keys::Int, &sorted, f, *frame, partitioned))
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
                mask: None,
                arg_type: DataType::Decimal128(p, 0),
                name: "s".to_owned(),
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
            mask: None,
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
    let op = Window::new(Arc::clone(&s), plan, true, None).unwrap();
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
    let op = Window::new(schema(), window.clone(), true, None).unwrap();
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
    let op = Window::new(schema(), plan(functions(), true), true, None).unwrap();
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
    let op = Window::new(schema(), plan(functions(), false), true, None).unwrap();
    let (_root, ctx) = task(1 << 30);
    let out = drive(&op, &ctx, &[batch(&[])]).unwrap();
    assert!(out.is_empty());
    assert_eq!(op.schema().fields().len(), 5 + functions().len());
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
    let class = |p: WindowPlan| {
        Window::new(schema(), p, true, None)
            .unwrap_err()
            .error_class()
    };
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
    // RANGE frames with value bounds are a later part of P2-T055.
    let following = WindowFrame {
        units: FrameUnits::Range,
        lower: FrameBound::Offset(-1),
        upper: FrameBound::CurrentRow,
    };
    assert_eq!(
        class(window(
            f(agg(AggregateFunction::Sum, Some(3)), following, None),
            vec![]
        )),
        ErrorClass::KrishivUnsupportedFeature
    );
    let op = Window::new(schema(), plan(functions(), true), true, None).unwrap();
    assert!(format!("{op:?}").contains("Window"));
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

/// What a unit's evaluation reserves matches what it holds: at least its
/// outputs and row maps, and at most twice that (the rest is what it
/// holds while it runs: states, group ids, gather indices).
#[test]
fn reserved_cost_matches_what_evaluation_holds() {
    let numeric: Vec<(WindowFunction, WindowFrame)> = functions()
        .into_iter()
        .filter(|(f, _)| !matches!(f, WindowFunction::Aggregate(a) if a.args == [4]))
        .collect();
    for (rows, parts) in [(10_000, 100), (10_000, 1), (4_096, 4_096)] {
        for f in &numeric {
            let window = plan(vec![f.clone()], true);
            let (b, part, peer) = sorted_unit(rows, parts);
            let n = b.num_rows();
            let slice = super::eval::Slice {
                batch: b.data(),
                offset: 0,
                len: n,
                part_start: &part,
                peer_start: &peer,
            };
            let out = super::eval::evaluate(&window, &[slice], true).unwrap();
            let count = |v: &[bool]| v.iter().filter(|&&x| x).count();
            // Outputs, row maps (`part_of`, `peer_of`) and the partition and
            // peer-group vectors.
            let held: usize = out
                .iter()
                .map(|c| Array::get_array_memory_size(c.as_ref()))
                .sum::<usize>()
                + 8 * n
                + 20 * count(&part)
                + 16 * count(&peer);
            let cost = super::eval::cost(&window, n, count(&part), count(&peer), 0).unwrap();
            assert!(cost >= held, "{f:?}: reserved {cost} < held {held}");
            assert!(cost <= 2 * held, "{f:?}: reserved {cost} for {held} held");
        }
    }
}

/// A partitioned window runs within a limit that holds a few batches and
/// their exact evaluation cost (the earlier flat per-row estimate needed
/// several times more and failed here).
#[tokio::test]
async fn partitioned_windows_run_at_a_tight_limit() {
    let window = plan(
        vec![
            (WindowFunction::Rank, WindowFrame::running(FrameUnits::Rows)),
            (
                agg(AggregateFunction::Sum, Some(3)),
                WindowFrame::running(FrameUnits::Range),
            ),
        ],
        true,
    );
    let rows: Vec<Row> = (0..40_000)
        .map(|i| (Some(i / 100), Some(i / 3), i, Some(i % 11)))
        .collect();
    let per_batch = 4_000;
    let batches: Vec<Batch> = rows.chunks(per_batch).map(batch).collect();
    // A batch: 40 partitions, 1334 peer groups.
    let exact = batches[0].data().get_array_memory_size()
        + 2 * per_batch
        + super::eval::cost(&window, per_batch, 40, 1_334, 0).unwrap();
    let limit = 4 * exact;
    // The flat estimate this replaces reserved 2 + 16 + 64 + 16 bytes per
    // row besides the input.
    assert!(batches.len() * per_batch * 98 > limit);
    let op = Window::new(schema(), window, true, None).unwrap();
    let (root, ctx) = task(limit);
    let out = drive(&op, &ctx, &batches).unwrap();
    assert_eq!(out.iter().map(Batch::num_rows).sum::<usize>(), 40_000);
    drop((out, ctx));
    assert_eq!(root.reserved(), 0);
}

/// String results are views of the input's buffers: a running `max` over
/// one partition of 1 KB strings reserves and holds about the input, not
/// a copy of the maximum per row.
#[tokio::test]
async fn string_results_share_the_input_buffers() {
    let rows = 10_000;
    let s = Arc::new(Schema::new(vec![
        Field::new("o", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    let values: Vec<String> = (0..rows).map(|i| format!("{i:0>1024}")).collect();
    let input = arrow_array::StringArray::from_iter_values(&values);
    let data_ptr = input.values().as_ptr();
    let b = RecordBatch::try_new(
        Arc::clone(&s),
        vec![
            Arc::new(Int64Array::from_iter_values(0..rows)),
            Arc::new(input),
        ],
    )
    .unwrap();
    let input_bytes = b.get_array_memory_size();
    let plan = WindowPlan {
        partition_by: Vec::new(),
        order_by: vec![SortKey::asc(0)],
        functions: vec![WindowFunctionPlan {
            function: WindowFunction::Aggregate(AggregateExpr {
                function: AggregateFunction::Max,
                args: vec![1],
                input: AggInput::Raw,
                mask: None,
                arg_type: DataType::Utf8View,
                name: "m".to_owned(),
            }),
            frame: WindowFrame::running(FrameUnits::Rows),
            filter: None,
            name: "m".to_owned(),
        }],
    };
    let op = Window::new(Arc::clone(&s), plan, true, None).unwrap();
    // A copy per row would need 10 MB more than the input.
    let (root, ctx) = task(input_bytes + input_bytes / 2);
    let out = drive(&op, &ctx, &[Batch::new(b)]).unwrap();
    let m = out[0].data().column(2).as_string_view();
    assert_eq!(m.value(9_999), values[9_999]);
    assert!(m.data_buffers().iter().any(|d| d.as_ptr() == data_ptr));
    drop((out, ctx));
    assert_eq!(root.reserved(), 0);
}

#[test]
fn windows_require_their_partitioning() {
    assert_eq!(
        plan(functions(), true).required_distribution(),
        RequiredDistribution::Clustered(vec![0])
    );
    assert_eq!(
        plan(functions(), false).required_distribution(),
        RequiredDistribution::Single
    );
}

// Spilling the buffered rows (P2-T055 part 2).

/// A spill directory of its own: the manager deletes what it wrote, and
/// the guard the directory itself.
struct SpillDir {
    manager: Option<krishiv_memory::SpillManager>,
    dir: std::path::PathBuf,
}

impl Drop for SpillDir {
    fn drop(&mut self) {
        drop(self.manager.take());
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn spill_scope() -> (SpillDir, krishiv_memory::SpillScope) {
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
struct Spills {
    /// The buffered rows spill after every this many batches (0: only when
    /// the operator needs memory).
    every: usize,
    /// After the input, all but this many free bytes of the pool are taken,
    /// so partitions that do not fit are evaluated from disk.
    slack: Option<usize>,
}

/// Runs `op` over `input` directly as a driver does, with spills (see
/// [`Spills`]); a request for memory in the flush phase is granted while
/// the pool has it. Collects the output.
async fn drive_spilling(
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// With spills at any batch, and partitions read back in memory or
    /// evaluated from disk, every function gives what it gives without a
    /// spill, in the same order.
    #[test]
    fn spilled_windows_match_in_memory_ones(
        rows in big_rows_strategy(),
        split in 1_usize..12,
        every in 0_usize..4,
        slack in prop_oneof![Just(None), (0_usize..8192).prop_map(Some)],
        partitioned in any::<bool>(),
    ) {
        let window = plan(functions(), partitioned);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let spills = Spills { every, slack };
        let (spilled, _) = rt.block_on(compare_spilling(&rows, split, &window, spills));
        prop_assert!(every == 0 || rows.len() < split.max(1) * every || spilled > 0);
    }
}

/// A partition that does not fit in memory is copied to a file of its own
/// and evaluated from there, with every function, partitioned or not.
#[tokio::test]
async fn a_partition_larger_than_memory_is_evaluated_from_disk() {
    for partitioned in [false, true] {
        let rows: Vec<Row> = (0..6_000)
            .map(|i| {
                // One large partition between two small ones.
                let p = if i < 100 {
                    0
                } else if i < 5_900 {
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
            .collect();
        let window = plan(functions(), partitioned);
        // Room for evaluating a batch from disk, not for the partition.
        let spills = Spills {
            every: 1,
            slack: Some(96 << 10),
        };
        let (spilled, external) = compare_spilling(&rows, 250, &window, spills).await;
        assert!(spilled > 0);
        assert!(external >= 1, "partitioned {partitioned}: {external}");
    }
}

/// The ROWS and RANGE aggregates of strings, decimals and filtered rows
/// from disk equal their in-memory values.
#[tokio::test]
async fn exact_and_string_frames_from_disk() {
    use AggregateFunction::{Avg, Max, Min, Sum};
    let (s, b) = decimal_rows(
        &(0..3_000)
            .map(|i| i128::from(i % 97) * 1_000_003)
            .collect::<Vec<_>>(),
        20,
    );
    let frames = [
        WindowFrame::running(FrameUnits::Rows),
        WindowFrame::running(FrameUnits::Range),
        WindowFrame::whole(FrameUnits::Rows),
    ];
    let mut functions = Vec::new();
    for (k, frame) in frames.iter().enumerate() {
        for function in [Sum, Avg, Min, Max] {
            let name = format!("d{k}{function:?}");
            functions.push(WindowFunctionPlan {
                function: WindowFunction::Aggregate(AggregateExpr {
                    function: function.clone(),
                    args: vec![1],
                    input: AggInput::Raw,
                    mask: None,
                    arg_type: DataType::Decimal128(20, 0),
                    name: "d".to_owned(),
                }),
                frame: *frame,
                filter: None,
                name,
            });
        }
    }
    let plan = WindowPlan {
        partition_by: Vec::new(),
        order_by: vec![SortKey::asc(0)],
        functions,
    };
    let batches: Vec<Batch> = (0..b.num_rows())
        .step_by(100)
        .map(|i| Batch::new(b.data().slice(i, 100.min(b.num_rows() - i))))
        .collect();
    let plain = Window::new(Arc::clone(&s), plan.clone(), true, None).unwrap();
    let (_r, ctx) = task(1 << 30);
    let want = drive(&plain, &ctx, &batches).unwrap();
    let (_manager, scope) = spill_scope();
    let op = Window::new(s, plan.clone(), true, Some(scope)).unwrap();
    let limit = 1 << 30;
    let (root, ctx) = task(limit);
    let spills = Spills {
        every: 1,
        slack: Some(0),
    };
    let got = drive_spilling(&op, &ctx, limit, &root, &batches, spills)
        .await
        .unwrap();
    assert!(
        op.counters
            .external
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0
    );
    for c in 2..2 + plan.functions.len() {
        assert_eq!(column_text(&got, c), column_text(&want, c), "column {c}");
    }
}

/// A spill whose write fails loses the rows it took: every later call
/// fails with that error instead of returning a short result. The write
/// fails with an I/O error (the scope's directory is gone), not a
/// cancellation, so the stored error's class differs from the placeholder
/// the spill holds while the blocking write runs.
#[tokio::test]
async fn a_failed_spill_fails_every_later_call() {
    let rows: Vec<Row> = (0..400)
        .map(|i| (Some(i / 300), Some(i), i, Some(i)))
        .collect();
    let (_manager, scope) = spill_scope();
    let op = Window::new(schema(), plan(functions(), true), true, Some(scope.clone())).unwrap();
    let (_root, ctx) = task(1 << 30);
    let mut local = op.local_state(&ctx).unwrap();
    assert!(matches!(
        op.execute(&batch(&rows), &mut local).unwrap(),
        OperatorOutput::NeedMoreInput(_)
    ));
    std::fs::remove_dir_all(scope.dir()).unwrap();
    let e = op.spill(&mut local, 1, &ctx).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivSpillIoError, "{e}");
    let class = e.error_class();
    let again = op.execute(&batch(&rows[..1]), &mut local).unwrap_err();
    assert_eq!(again.error_class(), class, "{again}");
    assert_eq!(op.finish(&mut local).unwrap_err().error_class(), class);
    assert_eq!(
        op.end_input(&mut local, &ctx)
            .await
            .unwrap_err()
            .error_class(),
        class
    );
    assert_eq!(
        op.flush(&mut local, &ctx).await.unwrap_err().error_class(),
        class
    );
}

/// A spill dropped mid-write (its query cancelled) leaves the window
/// failed with `OPERATION_CANCELED`; a refused scratch reservation spills
/// nothing and loses nothing.
#[tokio::test]
async fn a_dropped_spill_leaves_the_window_cancelled() {
    use futures::FutureExt;
    let rows: Vec<Row> = (0..400)
        .map(|i| (Some(i / 300), Some(i), i, Some(i)))
        .collect();
    let (_manager, scope) = spill_scope();
    let op = Window::new(schema(), plan(functions(), true), true, Some(scope)).unwrap();
    let (_root, ctx) = task(1 << 30);
    let mut local = op.local_state(&ctx).unwrap();
    op.execute(&batch(&rows), &mut local).unwrap();
    // Polled once: it waits for its blocking write, and is dropped.
    assert!(op.spill(&mut local, 1, &ctx).now_or_never().is_none());
    let e = op.finish(&mut local).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::OperationCanceled, "{e}");

    // A pool without room for the writer's scratch: the spill declines.
    let (_manager, scope) = spill_scope();
    let op = Window::new(schema(), plan(functions(), true), true, Some(scope)).unwrap();
    let limit = 256 << 10;
    let (root, ctx) = task(limit);
    let mut local = op.local_state(&ctx).unwrap();
    op.execute(&batch(&rows[..10]), &mut local).unwrap();
    let pool = Arc::clone(ctx.query().pool());
    let mut hog = krishiv_memory::Reservation::new(Arc::clone(&pool), pool.register("hog", None));
    take_all_but(&mut hog, limit - root.reserved(), 0);
    assert_eq!(op.spill(&mut local, 1, &ctx).await.unwrap(), 0);
    drop(hog);
    let out: usize = std::iter::from_fn(|| op.finish(&mut local).unwrap())
        .map(|b| b.num_rows())
        .sum();
    assert_eq!(out, 10);
}

/// A failed read in the flush phase fails it and every later step. The
/// read fails with an I/O error (the spill files are gone), not a
/// cancellation, so the stored error's class differs from the placeholder
/// the flush holds while the blocking read runs.
#[tokio::test]
async fn a_failed_read_back_fails_the_flush_phase() {
    let rows: Vec<Row> = (0..400)
        .map(|i| (Some(i / 300), Some(i), i, Some(i)))
        .collect();
    let (_manager, scope) = spill_scope();
    let op = Window::new(schema(), plan(functions(), true), true, Some(scope.clone())).unwrap();
    let (_root, ctx) = task(1 << 30);
    let mut local = op.local_state(&ctx).unwrap();
    op.execute(&batch(&rows), &mut local).unwrap();
    assert!(op.spill(&mut local, 1, &ctx).await.unwrap() > 0);
    assert!(op.finish(&mut local).unwrap().is_none());
    op.end_input(&mut local, &ctx).await.unwrap();
    std::fs::remove_dir_all(scope.dir()).unwrap();
    let e = op.flush(&mut local, &ctx).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivSpillIoError, "{e}");
    let again = op.flush(&mut local, &ctx).await.unwrap_err();
    assert_eq!(again.error_class(), e.error_class());
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

/// One partition many times the memory limit, through the sort and the
/// window with spilling: every row's values are checked against closed
/// forms as the output streams, and the pool is empty at the end.
#[tokio::test]
async fn one_large_partition_spills_through_the_runner() {
    use AggregateFunction::{Count, Sum};
    let (n, limit) = large_case();
    let rows = WindowFrame::running(FrameUnits::Rows);
    let range = WindowFrame::running(FrameUnits::Range);
    let window = plan(
        vec![
            (WindowFunction::RowNumber, rows),
            (WindowFunction::Rank, rows),
            (agg(Sum, Some(3)), rows),
            (agg(Count, None), range),
            (agg(Sum, Some(3)), WindowFrame::whole(FrameUnits::Rows)),
            // An exact percentile over the whole partition, folded from disk.
            (
                agg(
                    AggregateFunction::Percentile {
                        percentages: vec![0.5],
                        array: false,
                        reverse: false,
                    },
                    Some(3),
                ),
                WindowFrame::whole(FrameUnits::Rows),
            ),
            // Read through `disk.rs`: an offset function, a sliding frame, a
            // frame from the start, a frame over key values, `nth_value`.
            (lag_lead(-1, None, false), rows),
            (
                agg(Sum, Some(3)),
                between(
                    FrameUnits::Rows,
                    FrameBound::Offset(-1),
                    FrameBound::Offset(1),
                ),
            ),
            (
                agg(Count, None),
                between(
                    FrameUnits::Rows,
                    FrameBound::UnboundedPreceding,
                    FrameBound::Offset(-1),
                ),
            ),
            (
                agg(Count, None),
                between(
                    FrameUnits::Range,
                    FrameBound::RangeInt(-3),
                    FrameBound::RangeInt(3),
                ),
            ),
            (nth(2, false), range),
        ],
        true,
    );
    let s = schema();
    // Generated in batches, unsorted (descending ids), into the sort.
    let per = 8_192;
    let source = Generated::new(Arc::clone(&s), (n + per - 1) / per, {
        move |k: i64| {
            let hi = n - k * per;
            let lo = (hi - per).max(0);
            let rows: Vec<Row> = (lo..hi)
                .rev()
                .map(|i| (Some(0), Some(i / 3), i, Some(i % 11)))
                .collect();
            batch(&rows)
        }
    });
    let (_manager, scope) = spill_scope();
    let keys = [SortKey::asc(0), SortKey::asc(1), SortKey::asc(2)];
    let sort = Arc::new(Sort::new(Arc::clone(&s), &keys, Some(scope.clone())).unwrap());
    let op = Arc::new(Window::new(Arc::clone(&s), window, true, Some(scope)).unwrap());
    let node: Arc<dyn ExecNode> = Arc::new(OperatorNode {
        input: Arc::new(BreakerNode {
            input: Arc::new(SourceNode(source)),
            sink: Arc::clone(&sort) as Arc<dyn crate::Sink>,
            source: sort,
            ordered: true,
        }),
        op: Arc::clone(&op) as Arc<dyn crate::Operator>,
    });
    let runner = QueryRunner::new(TokioExecutor::deterministic(2).unwrap());
    let root = pool(limit);
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("q", Some(limit)),
    );
    let total: i64 = (0..n).map(|i| i % 11).sum();
    let median = {
        let mut v: Vec<i64> = (0..n).map(|i| i % 11).collect();
        v.sort_unstable();
        // percentile(x, 0.5): the mean of the two middle values.
        let (lo, hi) = ((v.len() - 1) / 2, v.len() / 2);
        (v[lo] + v[hi]) as f64 / 2.0
    };
    let mut stream = runner.execute(node, Arc::clone(&q)).await.unwrap();
    let mut next = 0_i64;
    let mut running = 0_i64;
    while let Some(b) = stream.next().await {
        let b = b.unwrap();
        let d = b.data();
        let col = |c: usize| d.column(c).as_primitive::<Int64Type>().clone();
        let int = |c: usize| d.column(c).as_primitive::<Int32Type>().clone();
        let (ids, rn, rank, sum, count, whole) = (col(2), int(5), int(6), col(7), col(8), col(9));
        let med = d
            .column(10)
            .as_primitive::<arrow_array::types::Float64Type>()
            .clone();
        for r in 0..d.num_rows() {
            let i = ids.value(r);
            assert_eq!(i, next, "rows in order");
            running += i % 11;
            assert_eq!(i64::from(rn.value(r)), i + 1);
            assert_eq!(i64::from(rank.value(r)), i / 3 * 3 + 1);
            assert_eq!(sum.value(r), running);
            assert_eq!(count.value(r), (i / 3 * 3 + 3).min(n));
            assert_eq!(whole.value(r), total);
            assert!((med.value(r) - median).abs() < 1e-9);
            let (lag, slide, before, near, second) = (col(11), col(12), col(13), col(14), col(15));
            if i == 0 {
                assert!(lag.is_null(r));
            } else {
                assert_eq!(lag.value(r), (i - 1) % 11);
            }
            let around: i64 = ((i - 1).max(0)..=(i + 1).min(n - 1)).map(|x| x % 11).sum();
            assert_eq!(slide.value(r), around);
            assert_eq!(before.value(r), i);
            let k = i / 3;
            assert_eq!(near.value(r), ((k + 3) * 3 + 3).min(n) - (k - 3).max(0) * 3);
            assert_eq!(second.value(r), 1);
            next += 1;
        }
    }
    assert_eq!(next, n);
    drop((stream, q));
    assert_eq!(root.reserved(), 0, "the query pool is empty at the end");
    assert!(
        op.counters
            .external
            .load(std::sync::atomic::Ordering::Relaxed)
            >= 1
    );
}

/// The flush phase's headroom refused: the window is failed, and a later
/// flush fails too instead of finding no rows.
#[tokio::test(start_paused = true)]
async fn a_refused_flush_headroom_fails_the_window() {
    let rows: Vec<Row> = (0..400)
        .map(|i| (Some(i / 300), Some(i), i, Some(i)))
        .collect();
    let (_manager, scope) = spill_scope();
    let op = Window::new(schema(), plan(functions(), true), true, Some(scope)).unwrap();
    let limit = 1 << 20;
    let (root, ctx) = task(limit);
    let mut local = op.local_state(&ctx).unwrap();
    op.execute(&batch(&rows), &mut local).unwrap();
    assert!(op.spill(&mut local, 1, &ctx).await.unwrap() > 0);
    assert!(op.finish(&mut local).unwrap().is_none());
    let pool = Arc::clone(ctx.query().pool());
    let mut hog = krishiv_memory::Reservation::new(Arc::clone(&pool), pool.register("hog", None));
    take_all_but(&mut hog, limit - root.reserved(), 0);
    let e = op.end_input(&mut local, &ctx).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory, "{e}");
    let again = op.flush(&mut local, &ctx).await.unwrap_err();
    assert_eq!(again.error_class(), ErrorClass::UnableToAcquireMemory);
    drop((hog, local));
}

/// The reservation for frames with offsets covers their outputs and row
/// maps (it also holds the frames, the tree and a chunk of scratch).
#[test]
fn frame_costs_cover_their_outputs() {
    for (rows, parts) in [(10_000, 100), (4_096, 4_096)] {
        for f in frame_functions() {
            let window = plan(vec![f.clone()], true);
            let (b, part, peer) = sorted_unit(rows, parts);
            let n = b.num_rows();
            let slice = super::eval::Slice {
                batch: b.data(),
                offset: 0,
                len: n,
                part_start: &part,
                peer_start: &peer,
            };
            let out = super::eval::evaluate(&window, &[slice], false).unwrap();
            let count = |v: &[bool]| v.iter().filter(|&&x| x).count();
            let held: usize = out
                .iter()
                .map(|c| Array::get_array_memory_size(c.as_ref()))
                .sum::<usize>()
                + 8 * n * 2;
            let cost = super::eval::cost(&window, n, count(&part), count(&peer), 0).unwrap();
            assert!(cost >= held, "{f:?}: reserved {cost} < held {held}");
        }
    }
}

/// `lag`/`lead`/`nth_value` reserve their peak: the input copied into
/// views (strings), the default column's views too, the counting rows, the
/// picks, the gather's locations and indices, the frames (`nth_value`) and
/// the result, all alive while the result is gathered.
#[test]
fn offset_function_costs_cover_their_scratch() {
    let rows = |lower, upper| WindowFrame {
        units: FrameUnits::Rows,
        lower,
        upper,
    };
    let lag_s = WindowFunction::Offset {
        input: 4,
        offset: -1,
        default: Some(4),
        ignore_nulls: false,
        data_type: DataType::Utf8View,
    };
    let nth_s = WindowFunction::NthValue {
        input: 4,
        n: 2,
        ignore_nulls: true,
        data_type: DataType::Utf8View,
    };
    let cases = [
        (
            lag_s,
            rows(FrameBound::Offset(-1), FrameBound::Offset(-1)),
            true,
            0,
        ),
        (nth_s, WindowFrame::whole(FrameUnits::Rows), false, 8),
        (
            nth(3, false),
            WindowFrame::whole(FrameUnits::Rows),
            false,
            8,
        ),
    ];
    for (function, frame, with_default, frames) in cases {
        let window = plan(vec![(function.clone(), frame)], true);
        let (b, part, peer) = sorted_unit(10_000, 100);
        let n = b.num_rows();
        let slice = super::eval::Slice {
            batch: b.data(),
            offset: 0,
            len: n,
            part_start: &part,
            peer_start: &peer,
        };
        let out = super::eval::evaluate(&window, &[slice], false).unwrap();
        // What an array holds of its own: views share the input's value
        // buffers.
        let own = |a: &ArrayRef| {
            let shared: usize = match a.data_type() {
                DataType::Utf8View => a
                    .as_string_view()
                    .data_buffers()
                    .iter()
                    .map(arrow_buffer::Buffer::capacity)
                    .sum(),
                _ => 0,
            };
            Array::get_array_memory_size(a.as_ref()) - shared
        };
        let output: usize = out.iter().map(own).sum();
        let views = match &function {
            WindowFunction::Offset { .. } | WindowFunction::NthValue { input: 4, .. } => {
                let v = super::eval::to_view(b.data().column(4)).unwrap();
                own(&v) * (1 + usize::from(with_default))
            }
            _ => 0,
        };
        // Counting rows, picks, gather locations and indices, frames.
        let scratch = n * (8 + 16 + 32 + frames);
        let held = output + views + scratch;
        let count = |v: &[bool]| v.iter().filter(|&&x| x).count();
        let cost = super::eval::cost(&window, n, count(&part), count(&peer), 0).unwrap();
        assert!(cost >= held, "{function:?}: reserved {cost} < held {held}");
    }
}

/// A segment tree over more rows than a chunk is built a chunk of parents
/// at a time and still gives each frame's value.
#[test]
fn trees_larger_than_a_chunk_match_the_reference() {
    use AggregateFunction::{Count, Max, Min, Sum};
    let frame = WindowFrame {
        units: FrameUnits::Rows,
        lower: FrameBound::Offset(-3),
        upper: FrameBound::Offset(2),
    };
    let functions: Vec<(WindowFunction, WindowFrame)> = [Sum, Count, Min, Max]
        .into_iter()
        .map(|f| (agg(f, Some(3)), frame))
        .collect();
    let n = i64::try_from(2 * super::frames::CHUNK + 777).unwrap();
    let data: Vec<Row> = (0..n)
        .map(|i| (Some(0), Some(i), i, (i % 13 != 0).then_some(i % 97 - 40)))
        .collect();
    let window = plan(functions, true);
    let b = batch(&data);
    let part: Vec<bool> = (0..data.len()).map(|i| i == 0).collect();
    let peer = vec![true; data.len()];
    let slice = super::eval::Slice {
        batch: b.data(),
        offset: 0,
        len: data.len(),
        part_start: &part,
        peer_start: &peer,
    };
    let out = super::eval::evaluate(&window, &[slice], false).unwrap();
    let got: Vec<String> = data
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let cells: Vec<String> = out.iter().map(|c| cell(c, i)).collect();
            format!("{}: {}", r.2, cells.join(", "))
        })
        .collect();
    // Each row's frame of non-NULL values, directly (one partition).
    let want: Vec<String> = (0..data.len())
        .map(|i| {
            let lo = i.saturating_sub(3);
            let hi = (i + 3).min(data.len());
            let vals: Vec<i64> = data[lo..hi].iter().filter_map(|r| r.3).collect();
            let show = |v: Option<i64>| v.map_or_else(|| "null".to_owned(), |v| v.to_string());
            let sum = (!vals.is_empty()).then(|| vals.iter().sum::<i64>());
            let count = i64::try_from(vals.len()).unwrap();
            format!(
                "{}: {}, {}, {}, {}",
                data[i].2,
                show(sum),
                count,
                show(vals.iter().min().copied()),
                show(vals.iter().max().copied())
            )
        })
        .collect();
    assert_eq!(got, want);
}

/// A running `collect_list` snapshots a growing copy per row: over a
/// partition larger than memory it is refused too.
#[tokio::test]
async fn a_growing_running_state_over_a_partition_on_disk_is_refused() {
    refused_on_disk(
        agg(AggregateFunction::CollectList, Some(3)),
        WindowFrame::running(FrameUnits::Rows),
    )
    .await;
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

/// The window's metrics count its input once: rows read back in the flush
/// phase (in memory or copied to a partition's own file) are not counted
/// again.
#[tokio::test]
async fn spilled_rows_are_counted_once() {
    let rows: Vec<Row> = (0..6_000)
        .map(|i| {
            let p = if i < 100 {
                0
            } else if i < 5_900 {
                1
            } else {
                2 + i % 5
            };
            (Some(p), Some(i), i, Some(i % 17))
        })
        .collect();
    let window = plan(functions(), true);
    let (_dir, scope) = spill_scope();
    let op = Window::new(schema(), window, true, Some(scope)).unwrap();
    let limit = 1 << 30;
    let (root, ctx) = task(limit);
    let mut sorted = rows.clone();
    sorted.sort_by_key(|r| (r.0, r.1));
    let batches: Vec<Batch> = sorted.chunks(250).map(batch).collect();
    let spills = Spills {
        every: 1,
        slack: Some(96 << 10),
    };
    let out = drive_spilling(&op, &ctx, limit, &root, &batches, spills)
        .await
        .unwrap();
    assert_eq!(out.iter().map(Batch::num_rows).sum::<usize>(), 6_000);
    assert!(
        op.counters
            .external
            .load(std::sync::atomic::Ordering::Relaxed)
            >= 1
    );
    let count = |name: &str| {
        op.extra_metrics()
            .into_iter()
            .find(|(n, _)| n == name)
            .map(|(_, m)| m)
    };
    assert!(
        matches!(count("rows"), Some(crate::metrics::Metric::Count(6_000))),
        "{:?}",
        count("rows")
    );
    // Partitions 0, 1, then 2..=6.
    assert!(
        matches!(count("partitions"), Some(crate::metrics::Metric::Count(7))),
        "{:?}",
        count("partitions")
    );
}

/// A batch whose first rows wait in the output keeps its bytes reserved
/// for them when the rest of it spills: they are not the spill's to free
/// (nor to fund its writer's scratch) until those rows are emitted.
#[tokio::test]
async fn bytes_of_rows_waiting_in_the_output_stay_reserved_through_a_spill() {
    // `a`: 8 rows of partition 0. `b`: 2 more rows of partition 0, then a
    // large partition 1, so `b`'s first rows are evaluated while the rest
    // of it stays buffered.
    let a: Vec<Row> = (0..8).map(|i| (Some(0), Some(i), i, Some(i))).collect();
    let b: Vec<Row> = (8..20_000)
        .map(|i| (Some(i64::from(i >= 10)), Some(i), i, Some(i)))
        .collect();
    let (_dir, scope) = spill_scope();
    let window = plan(
        vec![(
            WindowFunction::RowNumber,
            WindowFrame::running(FrameUnits::Rows),
        )],
        true,
    );
    let op = Window::new(schema(), window, true, Some(scope)).unwrap();
    let limit = 64 << 20;
    let config = ExecConfig {
        batch_size: 4,
        ..ExecConfig::default()
    };
    let (root, ctx) = task_with(limit, config);
    let mut local = op.local_state(&ctx).unwrap();
    let mut out = 0;
    let b = batch(&b);
    for input in [batch(&a), b.clone()] {
        match op.execute(&input, &mut local).unwrap() {
            OperatorOutput::NeedMoreInput(o) => out += o.map_or(0, |o| o.num_rows()),
            other => panic!("{other:?}"),
        }
    }
    // `a`'s rows went out; `b`'s first two wait in the output.
    assert_eq!(out, 8);
    // The buffered rows are all of `b`, whose bytes its waiting rows hold:
    // the spill writes them but frees nothing yet.
    assert_eq!(op.spill(&mut local, 1, &ctx).await.unwrap(), 0);
    let held = b.data().get_array_memory_size();
    assert!(root.reserved() >= held, "{} < {held}", root.reserved());
    let waiting = op.finish(&mut local).unwrap().unwrap();
    assert_eq!(waiting.num_rows(), 2);
    assert!(root.reserved() < held, "released with its last rows");
    out += waiting.num_rows();
    assert!(op.finish(&mut local).unwrap().is_none());
    op.end_input(&mut local, &ctx).await.unwrap();
    loop {
        match op.flush(&mut local, &ctx).await.unwrap() {
            crate::operator::FlushOutput::Batch(b) => out += b.num_rows(),
            crate::operator::FlushOutput::Done => break,
            _ => {}
        }
    }
    assert_eq!(out, 20_000);
    drop(local);
    ctx.consumer().consumer().release_credit();
    assert_eq!(root.reserved(), 0);
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Every function over a bucketed sort (partitions split among four
    /// drivers) gives the naive value of every row.
    #[test]
    fn bucketed_windows_match_the_naive_reference(
        rows in rows_strategy(),
        split in 1_usize..9,
        kt in prop_oneof![Just(Keys::Int), Just(Keys::Float), Just(Keys::Decimal), Just(Keys::Text)],
    ) {
        let functions = functions();
        let window = plan(functions.clone(), true);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let out = rt.block_on(run_bucketed(kt, &rows, split, &window)).unwrap();
        let mut got = outputs(&out, functions.len());
        got.sort();
        let mut sorted = rows;
        sorted.sort_by_key(|r| (kt.key(r.0), kt.key(r.1), r.2));
        let columns: Vec<Vec<String>> = functions
            .iter()
            .map(|(f, frame)| reference(kt, &sorted, f, *frame, true))
            .collect();
        let mut want: Vec<String> = sorted
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let cells: Vec<String> = columns.iter().map(|c| c[i].clone()).collect();
                format!("{}: {}", r.2, cells.join(", "))
            })
            .collect();
        want.sort();
        prop_assert_eq!(got, want);
    }
}

#[path = "group_limit_tests.rs"]
mod group_limit_tests;

/// A growing state's snapshots are priced by position in the partition, so
/// a skewed input costs its real quadratic share and the price is additive
/// across batches.
#[test]
fn snapshot_slots_follow_partition_sizes_across_batches() {
    use super::eval::snapshot_slots;
    // Partitions of 3 and 1 rows: 1+2+3 and 1.
    assert_eq!(snapshot_slots(&[true, false, false, true], 0), (7, 1));
    // One partition of 6 rows, whole or split into batches of 4 and 2.
    let whole = snapshot_slots(&[true, false, false, false, false, false], 0);
    let (a, open) = snapshot_slots(&[true, false, false, false], 0);
    let (b, open) = snapshot_slots(&[false, false], open);
    assert_eq!(whole, (21, 6));
    assert_eq!((a + b, open), (21, 6));
    // Skew is priced exactly: partitions of 100 and 2 rows.
    let mut flags = vec![false; 102];
    flags[0] = true;
    flags[100] = true;
    assert_eq!(snapshot_slots(&flags, 0).0, 5_050 + 3);
    assert_eq!(snapshot_slots(&[], 7), (0, 7));
}

/// The window cost of a running `collect_list` grows with the snapshot
/// elements; a whole-partition frame and a non-growing aggregate ignore them.
#[test]
fn window_cost_counts_snapshot_elements_of_growing_aggregates() {
    let running = WindowFrame::running(FrameUnits::Rows);
    let growing = plan(
        vec![(agg(AggregateFunction::CollectList, Some(3)), running)],
        true,
    );
    let summing = plan(vec![(agg(AggregateFunction::Sum, Some(3)), running)], true);
    let base = super::eval::cost(&growing, 10, 1, 10, 0).unwrap();
    assert_eq!(
        super::eval::cost(&growing, 10, 1, 10, 880).unwrap(),
        base + 880
    );
    let sum_base = super::eval::cost(&summing, 10, 1, 10, 0).unwrap();
    assert_eq!(
        super::eval::cost(&summing, 10, 1, 10, 880).unwrap(),
        sum_base
    );
}

/// A running `collect_list` charges its snapshots by the width of its
/// argument: long strings cost more per element than the 16-byte floor, a
/// narrow integer column does not.
#[test]
fn snapshot_width_follows_the_argument_column() {
    let rows: Vec<Row> = (0..50)
        .map(|i| (Some(0), Some(i), i, Some(i % 7)))
        .collect();
    let b = batch(&rows);
    let narrow = plan(
        vec![(
            agg(AggregateFunction::CollectList, Some(2)),
            WindowFrame::running(FrameUnits::Rows),
        )],
        true,
    );
    assert_eq!(super::eval::snapshot_width(&narrow, b.data()), 16);
    let mut columns = b.data().columns().to_vec();
    let long = "x".repeat(200);
    columns[4] = Arc::new(arrow_array::StringArray::from_iter_values(
        (0..50).map(|_| long.as_str()),
    ));
    let wide = RecordBatch::try_new(b.data().schema(), columns).unwrap();
    let list = plan(
        vec![(
            agg(AggregateFunction::CollectList, Some(4)),
            WindowFrame::running(FrameUnits::Rows),
        )],
        true,
    );
    assert!(super::eval::snapshot_width(&list, &wide) >= 200);
    // Distinct keys are kept besides the values: twice the width.
    let set = plan(
        vec![(
            agg(AggregateFunction::CollectSet, Some(4)),
            WindowFrame::running(FrameUnits::Rows),
        )],
        true,
    );
    assert!(super::eval::snapshot_width(&set, &wide) >= 400);
    // A whole-partition frame snapshots nothing.
    let whole = plan(
        vec![(
            agg(AggregateFunction::CollectList, Some(4)),
            WindowFrame::whole(FrameUnits::Rows),
        )],
        true,
    );
    assert_eq!(super::eval::snapshot_width(&whole, &wide), 16);
}

/// One large partition under a running `collect_list` needs a quadratic
/// reservation: under a tight limit the window fails with a memory error
/// before the snapshots are made, and releases what it reserved.
#[tokio::test]
async fn a_running_collect_list_over_a_big_partition_fails_under_a_tight_limit() {
    let window = plan(
        vec![(
            agg(AggregateFunction::CollectList, Some(2)),
            WindowFrame::running(FrameUnits::Rows),
        )],
        true,
    );
    let rows: Vec<Row> = (0..3_000)
        .map(|i| (Some(0), Some(i), i, Some(i % 7)))
        .collect();
    let batches: Vec<Batch> = rows.chunks(500).map(batch).collect();
    let op = Window::new(schema(), window, true, None).unwrap();
    let limit = 16 << 20;
    let (root, ctx) = task(limit);
    let e = drive(&op, &ctx, &batches).unwrap_err();
    // The window asks for memory (a spill) before it makes the snapshots;
    // the driver reports that request.
    assert!(e.to_string().contains("NeedMemory"), "{e}");
    drop(ctx);
    assert_eq!(root.reserved(), 0);
}

/// A frame from `lower` to `upper`.
fn between(units: FrameUnits, lower: FrameBound, upper: FrameBound) -> WindowFrame {
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

/// A partition larger than memory gives what it gives in memory, for every
/// frame of aggregates and `nth_value`.
#[tokio::test]
async fn frames_of_a_partition_on_disk_match_memory() {
    let rows = skewed_rows(3_000);
    let spills = Spills {
        every: 1,
        slack: Some(4 << 20),
    };
    for frame in disk_frames() {
        let window = plan(over(frame), true);
        let (spilled, external) = compare_spilling(&rows, 250, &window, spills).await;
        assert!(spilled > 0, "{frame:?}");
        assert!(external >= 1, "{frame:?}");
    }
}

/// `lag`/`lead` of integers and strings from disk: any offset (past the
/// partition too), defaults, `IGNORE NULLS`.
#[tokio::test]
async fn offsets_of_a_partition_on_disk_match_memory() {
    let frame = WindowFrame::whole(FrameUnits::Rows);
    let rows = skewed_rows(6_000);
    let spills = Spills {
        every: 1,
        slack: Some(1 << 20),
    };
    for offset in [-100_000, -300, -2, -1, 0, 1, 7, 300, 100_000] {
        let mut functions = Vec::new();
        for default in [None, Some(2)] {
            for ignore_nulls in [false, true] {
                functions.push((lag_lead(offset, default, ignore_nulls), frame));
            }
        }
        for ignore_nulls in [false, true] {
            functions.push((
                WindowFunction::Offset {
                    input: 4,
                    offset,
                    default: Some(4),
                    ignore_nulls,
                    data_type: DataType::Utf8View,
                },
                frame,
            ));
        }
        let window = plan(functions, true);
        let (_, external) = compare_spilling(&rows, 250, &window, spills).await;
        assert!(external >= 1, "lead by {offset}");
    }
}

/// Frames that do not fit fold their rows from the file, with the same
/// values: a frame to the partition's end and one wider than memory.
#[tokio::test]
async fn frames_wider_than_memory_fold_from_the_file() {
    use AggregateFunction::{Count, First, Max, Min, Sum};
    use FrameBound::{CurrentRow, Offset, UnboundedFollowing};
    use FrameUnits::Rows;
    // Few enough for the quadratic fold.
    let rows = skewed_rows(1_500);
    for frame in [
        between(Rows, CurrentRow, UnboundedFollowing),
        between(Rows, Offset(-1_000), Offset(1_000)),
    ] {
        let functions = vec![
            (agg(Sum, Some(3)), frame),
            (agg(Count, None), frame),
            (agg(Min, Some(3)), frame),
            (agg(Max, Some(4)), frame),
            (agg(First { ignore_nulls: true }, Some(4)), frame),
        ];
        let window = plan(functions, true);
        let spills = Spills {
            every: 1,
            slack: Some(96 << 10),
        };
        let counters = compared_spilling(&rows, 50, &window, spills).await;
        assert!(counters.external.load(std::sync::atomic::Ordering::Relaxed) >= 1);
        assert!(
            counters
                .file_folds
                .load(std::sync::atomic::Ordering::Relaxed)
                >= 1,
            "{frame:?}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Frames and offset functions from disk equal the in-memory ones over
    /// random rows, split anywhere.
    #[test]
    fn disk_frames_match_memory_on_random_rows(
        rows in big_rows_strategy(),
        split in 1_usize..12,
        which in 0_usize..16,
        offset in -9_i64..9,
        ignore_nulls in any::<bool>(),
        partitioned in any::<bool>(),
    ) {
        let frame = disk_frames()[which];
        let mut functions = over(frame);
        functions.push((lag_lead(offset, Some(2), ignore_nulls), frame));
        let window = plan(functions, partitioned);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let spills = Spills { every: 1, slack: Some(256 << 10) };
        rt.block_on(compare_spilling(&rows, split, &window, spills));
    }
}

/// An exact percentile's value map is charged as it grows: distinct values
/// far beyond the memory left are refused with the memory error (the pool
/// empty after), and few distinct values over the same rows complete.
#[tokio::test]
async fn an_exact_percentile_on_disk_is_charged() {
    let percentile = agg(
        AggregateFunction::Percentile {
            percentages: vec![0.5],
            array: false,
            reverse: false,
        },
        Some(3),
    );
    for frame in [
        WindowFrame::whole(FrameUnits::Rows),
        WindowFrame::running(FrameUnits::Rows),
        WindowFrame::running(FrameUnits::Range),
    ] {
        let window = plan(vec![(percentile.clone(), frame)], true);
        let spills = Spills {
            every: 1,
            slack: Some(256 << 10),
        };
        let few: Vec<Row> = (0..20_000)
            .map(|i| (Some(0), Some(i), i, Some(i % 11)))
            .collect();
        compare_spilling(&few, 500, &window, spills).await;

        let distinct: Vec<Row> = (0..200_000)
            .map(|i| (Some(0), Some(i), i, Some(i)))
            .collect();
        let (_dir, scope) = spill_scope();
        let op = Window::new(schema(), window, true, Some(scope)).unwrap();
        let limit = 1 << 30;
        let (root, ctx) = task(limit);
        let batches: Vec<Batch> = distinct.chunks(500).map(batch).collect();
        let e = drive_spilling(&op, &ctx, limit, &root, &batches, spills)
            .await
            .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory, "{e}");
        drop(ctx);
        assert_eq!(root.reserved(), 0, "the query pool is empty at the end");
    }
}

/// `RANGE` frames from disk over a descending key with NULLs last, as in
/// memory: both edges move the other way through the keys.
#[tokio::test]
async fn range_frames_on_disk_over_a_descending_key_match_memory() {
    use FrameBound::{CurrentRow, RangeInt, UnboundedFollowing, UnboundedPreceding};
    let rows = skewed_rows(3_000);
    let spills = Spills {
        every: 1,
        slack: Some(4 << 20),
    };
    for frame in [
        between(FrameUnits::Range, RangeInt(-3), RangeInt(2)),
        between(FrameUnits::Range, CurrentRow, RangeInt(4)),
        between(FrameUnits::Range, UnboundedPreceding, RangeInt(1)),
        between(FrameUnits::Range, RangeInt(-2), UnboundedFollowing),
    ] {
        for (descending, nulls_first) in [(true, false), (true, true), (false, false)] {
            let mut window = plan(over(frame), true);
            window.order_by = vec![SortKey {
                column: 1,
                descending,
                nulls_first,
            }];
            let (_, external) = compare_spilling(&rows, 250, &window, spills).await;
            assert!(external >= 1, "{frame:?} {descending} {nulls_first}");
        }
    }
}

/// Float and decimal `RANGE` keys and `FILTER` from disk, as in memory.
#[tokio::test]
async fn float_and_decimal_keys_and_filters_on_disk_match_memory() {
    use FrameBound::{Offset, RangeFloat, RangeInt, UnboundedFollowing};
    use arrow_array::{BooleanArray, Decimal128Array, Float64Array};
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("f", DataType::Float64, true),
        Field::new("d", DataType::Decimal128(12, 2), true),
        Field::new("v", DataType::Int64, true),
        Field::new("b", DataType::Boolean, true),
    ]));
    let n = 3_000_i64;
    let sorted_batches = || -> Vec<Batch> {
        // Ascending in both keys, NULLs first, NaN largest for the float.
        let ids: Vec<i64> = (0..n).collect();
        ids.chunks(250)
            .map(|c| {
                let f = Float64Array::from_iter(c.iter().map(|&i| match i {
                    i if i < 20 => None,
                    i if i >= n - 20 => Some(f64::NAN),
                    #[expect(clippy::cast_precision_loss, reason = "small test keys")]
                    i => Some((i / 4) as f64 / 2.0),
                }));
                let d = Decimal128Array::from_iter(
                    c.iter()
                        .map(|&i| (i >= 20).then_some(i128::from(i / 4) * 25)),
                )
                .with_data_type(DataType::Decimal128(12, 2));
                let v = Int64Array::from_iter(c.iter().map(|&i| (i % 13 != 0).then_some(i % 17)));
                let b =
                    BooleanArray::from_iter(c.iter().map(|&i| (i % 11 != 0).then_some(i % 3 != 0)));
                let columns: Vec<ArrayRef> =
                    vec![Arc::new(f), Arc::new(d), Arc::new(v), Arc::new(b)];
                Batch::new(RecordBatch::try_new(Arc::clone(&schema), columns).unwrap())
            })
            .collect()
    };
    let sum = || agg(AggregateFunction::Sum, Some(2));
    let filtered = |frame| WindowFunctionPlan {
        function: sum(),
        frame,
        filter: Some(3),
        name: "w".to_owned(),
    };
    let float = between(
        FrameUnits::Range,
        RangeFloat((-1.5_f64).to_bits()),
        RangeFloat(1.0_f64.to_bits()),
    );
    let cases = [
        (0, float),
        (1, between(FrameUnits::Range, RangeInt(-50), RangeInt(75))),
        (1, between(FrameUnits::Rows, Offset(-2), Offset(3))),
        (1, between(FrameUnits::Rows, Offset(-1), UnboundedFollowing)),
        (
            1,
            between(FrameUnits::Rows, FrameBound::UnboundedPreceding, Offset(-1)),
        ),
    ];
    for (column, frame) in cases {
        let window = WindowPlan {
            partition_by: Vec::new(),
            order_by: vec![SortKey::asc(column)],
            functions: vec![filtered(frame)],
        };
        let concat = |batches: &[Batch]| {
            let data: Vec<RecordBatch> = batches.iter().map(|b| b.data().clone()).collect();
            arrow_select::concat::concat_batches(&data[0].schema(), &data).unwrap()
        };
        let input = sorted_batches();
        let plain = Window::new(Arc::clone(&schema), window.clone(), true, None).unwrap();
        let (_r, ctx) = task(1 << 30);
        let want = concat(&drive(&plain, &ctx, &input).unwrap());
        let (_manager, scope) = spill_scope();
        let op = Window::new(Arc::clone(&schema), window, true, Some(scope)).unwrap();
        let limit = 1 << 30;
        let (root, ctx) = task(limit);
        let spills = Spills {
            every: 1,
            slack: Some(256 << 10),
        };
        let got = drive_spilling(&op, &ctx, limit, &root, &input, spills)
            .await
            .unwrap();
        assert_eq!(concat(&got), want, "{frame:?}");
        assert!(
            op.counters
                .external
                .load(std::sync::atomic::Ordering::Relaxed)
                >= 1,
            "{frame:?}"
        );
        drop((got, ctx));
        assert_eq!(root.reserved(), 0, "the query pool is empty at the end");
    }
}

/// A row `FILTER` drops adds nothing, so it can't overflow an ANSI `sum`,
/// from disk as in memory. Frames from the partition's start fold (`Prefix`)
/// and frames that don't fit fold from the file.
#[tokio::test]
async fn rows_a_filter_drops_do_not_overflow_on_disk() {
    use FrameBound::{Offset, UnboundedPreceding};
    use arrow_array::BooleanArray;
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, true),
        Field::new("b", DataType::Boolean, true),
    ]));
    let n = 3_000_i64;
    let ids: Vec<i64> = (0..n).collect();
    let input: Vec<Batch> = ids
        .chunks(250)
        .map(|c| {
            let k = Int64Array::from_iter_values(c.iter().copied());
            let v = Int64Array::from_iter_values(c.iter().map(|_| i64::MAX));
            let b = BooleanArray::from_iter(c.iter().map(|_| Some(false)));
            let columns: Vec<ArrayRef> = vec![Arc::new(k), Arc::new(v), Arc::new(b)];
            Batch::new(RecordBatch::try_new(Arc::clone(&schema), columns).unwrap())
        })
        .collect();
    for frame in [
        between(FrameUnits::Rows, UnboundedPreceding, Offset(-1)),
        between(FrameUnits::Rows, Offset(-2), Offset(3)),
    ] {
        let window = WindowPlan {
            partition_by: Vec::new(),
            order_by: vec![SortKey::asc(0)],
            functions: vec![WindowFunctionPlan {
                function: agg(AggregateFunction::Sum, Some(1)),
                frame,
                filter: Some(2),
                name: "w".to_owned(),
            }],
        };
        let (_manager, scope) = spill_scope();
        let op = Window::new(Arc::clone(&schema), window, true, Some(scope)).unwrap();
        let limit = 1 << 30;
        let (root, ctx) = task(limit);
        let spills = Spills {
            every: 1,
            slack: Some(256 << 10),
        };
        let got = drive_spilling(&op, &ctx, limit, &root, &input, spills)
            .await
            .unwrap();
        let rows: usize = got.iter().map(Batch::num_rows).sum();
        assert_eq!(rows, 3_000, "{frame:?}");
        for b in &got {
            let w = b.data().columns().last().unwrap();
            assert_eq!(w.null_count(), b.num_rows(), "{frame:?}");
        }
        assert!(
            op.counters
                .external
                .load(std::sync::atomic::Ordering::Relaxed)
                >= 1,
            "{frame:?}"
        );
        drop((got, ctx));
        assert_eq!(root.reserved(), 0, "the query pool is empty at the end");
    }
}
