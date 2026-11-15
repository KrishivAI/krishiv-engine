// SPDX-License-Identifier: Apache-2.0
//! Order-sensitive aggregates (`first`, `last`, the ties of `max_by` and
//! `min_by`, float `max`/`min` of `0.0` and `-0.0`) across drivers follow
//! the driver-partition order whatever order the drivers combine in, and
//! whether or not the aggregate spilled: the answer never depends on memory.
#![cfg(not(loom))]
#![expect(
    clippy::disallowed_methods,
    reason = "test-only runtime plumbing; never runs on a compute worker"
)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow_schema::{Field, Schema};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillOwner, TreePool};
use krishiv_types::batch::Batch;

use arrow_schema::DataType;

use super::slice_tests::batch;
use super::*;
use crate::operator::SinkStatus;
use crate::{
    ExecConfig, Executor, OperatorId, QueryContext, QueryId, Sink, Source, TaskContext, TokioClock,
    TokioExecutor, WorkloadClass,
};

const KEYS: i64 = 512;

/// What [`run`] aggregates.
#[derive(Clone, Copy)]
enum Mode {
    /// `(key, first o, last o)`.
    Picks,
    /// `(key, max_by(o, k), min_by(o, k))`: every row of a group ties on `k`.
    By,
    /// `(key, max(z), min(z))` as `f64` bits, where `z` is `0.0` in partition
    /// 0 and `-0.0` in the others: the zeros tie, so the sign says which
    /// partition's value was kept.
    Zeros,
}

/// Partition `p`'s rows for `mode`: keys `0..KEYS`, then the order column
/// counting from `p * 10_000` (or the zero of [`Mode::Zeros`]).
fn input(mode: Mode, p: u32) -> Batch {
    if !matches!(mode, Mode::Zeros) {
        return batch(false, i64::from(p) * 10_000, 0..KEYS);
    }
    let zero = if p == 0 { 0.0 } else { -0.0 };
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from_iter_values(0..KEYS)),
        Arc::new(Float64Array::from_iter_values((0..KEYS).map(|_| zero))),
    ];
    Batch::new(RecordBatch::try_new(zeros_schema(), cols).unwrap())
}

fn zeros_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("z", DataType::Float64, false),
    ]))
}

/// Drivers of partitions `order` (in that combine order) each fold
/// [`input`]; the drivers in `spilled` spill their table before they
/// combine. Returns the rows `mode` describes, sorted.
async fn run(mode: Mode, rows: bool, order: &[u32], spilled: &[u32]) -> Vec<[i64; 3]> {
    let dir = tempfile::Builder::new()
        .prefix("krishiv-merge-order-")
        .tempdir()
        .unwrap();
    let manager = krishiv_memory::SpillManager::open(&krishiv_memory::SpillConfig {
        dir: dir.path().join("spill"),
        ..krishiv_memory::SpillConfig::default()
    })
    .unwrap();
    let root = TreePool::with_clock(
        &MemoryConfig {
            resident_guard: false,
            ..MemoryConfig::default()
        },
        Arc::new(TokioClock),
    );
    let pool: Arc<dyn MemoryPool> = root.child("q", Some(256 << 20));
    let config = ExecConfig {
        agg_row_layout: rows,
        ..ExecConfig::default()
    };
    let q = QueryContext::new(QueryId(1), WorkloadClass::Batch, config, pool);
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
    let task = || {
        TaskContext::new(
            Arc::clone(&q),
            Arc::clone(&executor),
            SpillOwner::new(),
            OperatorId(0),
            "agg",
        )
    };
    let pick = |f| AggregateExpr::new(f, 2, DataType::Int64, "p");
    let by = |f| AggregateExpr::by(f, (2, 0), (DataType::Int64, DataType::Int64), "b");
    let zero = |f| AggregateExpr::new(f, 1, DataType::Float64, "z");
    let agg = HashAggregate::new(AggregateSpec {
        input: input(mode, 0).data().schema(),
        group_by: vec![0],
        aggregates: match mode {
            Mode::Picks => vec![
                pick(AggregateFunction::First {
                    ignore_nulls: false,
                }),
                pick(AggregateFunction::Last {
                    ignore_nulls: false,
                }),
            ],
            Mode::By => vec![by(AggregateFunction::MaxBy), by(AggregateFunction::MinBy)],
            Mode::Zeros => vec![zero(AggregateFunction::Max), zero(AggregateFunction::Min)],
        },
        output: AggregateOutput::Final,
        ansi: true,
        spill: Some(manager.scope().unwrap()),
        require_unit_weights: false,
        legacy_statistical: false,
        udafs: None,
    })
    .unwrap();
    let coordinator = task();
    let g = Sink::global_state(&*agg, &coordinator).unwrap();
    for &p in order {
        let d = task().with_partition(p);
        let mut l = Sink::local_state(&*agg, &d, &g).unwrap();
        let b = input(mode, p);
        assert!(matches!(
            agg.sink(b, &mut l, &g).unwrap(),
            SinkStatus::NeedMoreInput
        ));
        if spilled.contains(&p) {
            assert!(agg.spill(&mut l, 0, &d).await.unwrap() > 0, "partition {p}");
        }
        agg.combine(l, &g).unwrap();
    }
    agg.finalize(&g, &coordinator).await.unwrap();
    let src = coordinator.with_dependency(Arc::clone(&g));
    let sg = Source::global_state(&*agg, &src).unwrap();
    let mut sl = Source::local_state(&*agg, &src, &sg).unwrap();
    let mut out = Vec::new();
    while let Some(b) = agg.get_data(&mut sl, &sg).await.unwrap() {
        let d = b.data();
        let col = |i: usize| -> Vec<i64> {
            let c = d.column(i);
            if i > 0 && matches!(mode, Mode::Zeros) {
                let f = c.as_primitive::<Float64Type>();
                f.values()
                    .iter()
                    .map(|v| v.to_bits().cast_signed())
                    .collect()
            } else {
                c.as_primitive::<Int64Type>().values().to_vec()
            }
        };
        let cols: Vec<_> = (0..3).map(col).collect();
        out.extend((0..d.num_rows()).map(|r| std::array::from_fn(|i| cols[i][r])));
    }
    out.sort_unstable();
    drop((sl, sg, src, g));
    assert_eq!(root.reserved(), 0, "memory still held");
    out
}

/// Every combine order, spilled or not, in either table layout gives
/// `reference`'s rows (`mode` as [`run`]).
async fn same_in_every_run(mode: Mode, reference: &[[i64; 3]]) {
    for rows in [true, false] {
        for order in [[0, 1, 2], [2, 1, 0], [1, 2, 0]] {
            for spilled in [&[][..], &[1], &[0, 2]] {
                let got = run(mode, rows, &order, spilled).await;
                assert!(
                    got == reference,
                    "row layout {rows}, combine order {order:?}, spilled {spilled:?}: {:?}",
                    &got[..2]
                );
            }
        }
    }
}

/// `first` comes from partition 0 and `last` from partition 2 (the disk
/// phase's order: every table spilled).
#[tokio::test]
async fn first_and_last_follow_partition_order_spilled_or_not() {
    let reference = run(Mode::Picks, true, &[0, 1, 2], &[0, 1, 2]).await;
    let want: Vec<[i64; 3]> = (0..KEYS).map(|k| [k, k, 20_000 + k]).collect();
    assert_eq!(reference, want);
    same_in_every_run(Mode::Picks, &reference).await;
}

/// The ties of `max_by`/`min_by` keep the last row to arrive (Spark's
/// `MaxMinBy`), the last partition's, in every run.
#[tokio::test]
async fn max_by_and_min_by_ties_do_not_depend_on_memory() {
    let reference = run(Mode::By, true, &[0, 1, 2], &[0, 1, 2]).await;
    let want: Vec<[i64; 3]> = (0..KEYS).map(|k| [k, 20_000 + k, 20_000 + k]).collect();
    assert_eq!(reference, want);
    same_in_every_run(Mode::By, &reference).await;
}

/// Spark 4.2.0 keeps the first zero of a `0.0`/`-0.0` tie in input order,
/// across partitions too (`max` and `min` of `[0.0, -0.0]` in two partitions
/// are `0.0`, of `[-0.0, 0.0]` `-0.0`): partition 0's `0.0`, in every run.
#[tokio::test]
async fn float_max_and_min_zero_ties_follow_partition_order() {
    let reference = run(Mode::Zeros, true, &[0, 1, 2], &[0, 1, 2]).await;
    let plus = 0.0f64.to_bits().cast_signed();
    let want: Vec<[i64; 3]> = (0..KEYS).map(|k| [k, plus, plus]).collect();
    assert_eq!(reference, want);
    same_in_every_run(Mode::Zeros, &reference).await;
}

/// Only `max`/`min` over a float (or a struct or array holding one) observe
/// the merge order; other types have no ties between distinct values.
#[test]
fn only_float_extrema_observe_merge_order() {
    let f = |t: DataType| Arc::new(Field::new("x", t, true));
    let max = |t| AggregateExpr::new(AggregateFunction::Max, 0, t, "m");
    let min = |t| AggregateExpr::new(AggregateFunction::Min, 0, t, "m");
    assert!(observes_order(&max(DataType::Float64)));
    assert!(observes_order(&min(DataType::Float32)));
    assert!(observes_order(&max(DataType::Struct(
        vec![f(DataType::Int32), f(DataType::Float64)].into()
    ))));
    assert!(observes_order(&min(DataType::List(f(DataType::Struct(
        vec![f(DataType::Float32)].into()
    ))))));
    assert!(!observes_order(&max(DataType::Int64)));
    assert!(!observes_order(&min(DataType::Utf8)));
    assert!(!observes_order(&max(DataType::List(f(DataType::Int32)))));
    assert!(!observes_order(&AggregateExpr::new(
        AggregateFunction::Sum,
        0,
        DataType::Float64,
        "s"
    )));
}
