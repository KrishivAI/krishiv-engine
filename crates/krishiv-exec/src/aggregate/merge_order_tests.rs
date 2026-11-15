// SPDX-License-Identifier: Apache-2.0
//! Order-sensitive aggregates (`first`, `last`, the ties of `max_by` and
//! `min_by`) across drivers follow the driver-partition order whatever order
//! the drivers combine in, and whether or not the aggregate spilled: the
//! answer never depends on memory.
#![cfg(not(loom))]
#![expect(
    clippy::disallowed_methods,
    reason = "test-only runtime plumbing; never runs on a compute worker"
)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use krishiv_memory::{MemoryConfig, MemoryPool, SpillOwner, TreePool};

use arrow_schema::DataType;

use super::slice_tests::batch;
use super::*;
use crate::operator::SinkStatus;
use crate::{
    ExecConfig, Executor, OperatorId, QueryContext, QueryId, Sink, Source, TaskContext, TokioClock,
    TokioExecutor, WorkloadClass,
};

const KEYS: i64 = 512;

/// Drivers of partitions `order` (in that combine order) each fold keys
/// `0..KEYS` with the order column counting from `partition * 10_000`;
/// the drivers in `spilled` spill their table before they combine.
/// Returns `(key, first o, last o)` with `picks`, else `(key, max_by(o, k),
/// min_by(o, k))` (every row of a group ties on `k`), sorted.
async fn run(picks: bool, rows: bool, order: &[u32], spilled: &[u32]) -> Vec<[i64; 3]> {
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
    let agg = HashAggregate::new(AggregateSpec {
        input: batch(false, 0, 0..1).data().schema(),
        group_by: vec![0],
        aggregates: if picks {
            vec![
                pick(AggregateFunction::First {
                    ignore_nulls: false,
                }),
                pick(AggregateFunction::Last {
                    ignore_nulls: false,
                }),
            ]
        } else {
            vec![by(AggregateFunction::MaxBy), by(AggregateFunction::MinBy)]
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
        let b = batch(false, i64::from(p) * 10_000, 0..KEYS);
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
        let col = |i: usize| d.column(i).as_primitive::<Int64Type>().clone();
        let cols: Vec<_> = (0..3).map(col).collect();
        out.extend((0..d.num_rows()).map(|r| std::array::from_fn(|i| cols[i].value(r))));
    }
    out.sort_unstable();
    drop((sl, sg, src, g));
    assert_eq!(root.reserved(), 0, "memory still held");
    out
}

/// Every combine order, spilled or not, in either table layout gives
/// `reference`'s rows (`picks` as [`run`]).
async fn same_in_every_run(picks: bool, reference: &[[i64; 3]]) {
    for rows in [true, false] {
        for order in [[0, 1, 2], [2, 1, 0], [1, 2, 0]] {
            for spilled in [&[][..], &[1], &[0, 2]] {
                let got = run(picks, rows, &order, spilled).await;
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
    let reference = run(true, true, &[0, 1, 2], &[0, 1, 2]).await;
    let want: Vec<[i64; 3]> = (0..KEYS).map(|k| [k, k, 20_000 + k]).collect();
    assert_eq!(reference, want);
    same_in_every_run(true, &reference).await;
}

/// The ties of `max_by`/`min_by` keep the last row to arrive (Spark's
/// `MaxMinBy`), the last partition's, in every run.
#[tokio::test]
async fn max_by_and_min_by_ties_do_not_depend_on_memory() {
    let reference = run(false, true, &[0, 1, 2], &[0, 1, 2]).await;
    let want: Vec<[i64; 3]> = (0..KEYS).map(|k| [k, 20_000 + k, 20_000 + k]).collect();
    assert_eq!(reference, want);
    same_in_every_run(false, &reference).await;
}
