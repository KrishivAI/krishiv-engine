// SPDX-License-Identifier: Apache-2.0
//! Unit tests of [`Tail`], called directly.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_common::error::ErrorClass;
use krishiv_memory::{MemoryConfig, MemoryPool, SpillOwner, TreePool};
use krishiv_types::batch::{Batch, Weights};

use super::Tail;
use crate::operator::{Sink, SinkStatus, Source};
use crate::{
    ExecConfig, ExecutorConfig, OperatorId, QueryContext, QueryId, TaskContext, TokioExecutor,
    WorkloadClass,
};

/// A source's `get_data` is ready at once.
fn ready<T>(f: impl std::future::Future<Output = T>) -> T {
    futures::FutureExt::now_or_never(f).unwrap_or_else(|| unreachable!("pending"))
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]))
}

fn ints(a: &[i64]) -> Batch {
    let data = RecordBatch::try_new(schema(), vec![Arc::new(Int64Array::from(a.to_vec()))])
        .unwrap_or_else(|e| unreachable!("{e}"));
    Batch::new(data)
}

fn ctx(config: &MemoryConfig) -> (Arc<TreePool>, TaskContext) {
    let root = TreePool::new(config);
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("q", None),
    );
    let exec =
        TokioExecutor::new(&ExecutorConfig::default()).unwrap_or_else(|e| unreachable!("{e}"));
    let ctx = TaskContext::new(q, exec, SpillOwner::new(), OperatorId(0), "op");
    (root, ctx)
}

/// Several drivers' batches sunk into one `Tail(limit)`, combined in order.
/// Each driver is a list of batch sizes; rows are numbered across drivers.
/// Returns the emitted rows.
fn run_drivers(limit: u64, drivers: &[&[usize]]) -> Vec<i64> {
    let (_root, ctx) = ctx(&MemoryConfig::default());
    let t = Tail::new(schema(), limit);
    let global = Sink::global_state(&*t, &ctx).unwrap_or_else(|e| unreachable!("{e}"));
    let mut next = 0i64;
    for sizes in drivers {
        let mut local =
            Sink::local_state(&*t, &ctx, &global).unwrap_or_else(|e| unreachable!("{e}"));
        for &n in *sizes {
            let rows: Vec<i64> = (next..next + n as i64).collect();
            next += n as i64;
            match t
                .sink(ints(&rows), &mut local, &global)
                .unwrap_or_else(|e| unreachable!("{e}"))
            {
                SinkStatus::NeedMoreInput => {}
                SinkStatus::Finished => break,
                other => unreachable!("unexpected {other:?}"),
            }
        }
        t.combine(local, &global)
            .unwrap_or_else(|e| unreachable!("{e}"));
    }
    let mut local = Source::local_state(&*t, &ctx, &global).unwrap_or_else(|e| unreachable!("{e}"));
    let mut got = Vec::new();
    while let Some(b) =
        ready(t.get_data(&mut local, &global)).unwrap_or_else(|e| unreachable!("{e}"))
    {
        got.extend(
            b.data()
                .column(0)
                .as_primitive::<Int64Type>()
                .iter()
                .flatten(),
        );
    }
    got
}

fn run(limit: u64, sizes: &[usize]) -> Vec<i64> {
    run_drivers(limit, &[sizes])
}

#[test]
fn keeps_the_last_rows_in_order() {
    assert_eq!(run(4, &[3, 3, 3]), vec![5, 6, 7, 8]);
    assert_eq!(run(3, &[3, 3]), vec![3, 4, 5]); // exact batch
    assert_eq!(run(1, &[2, 2]), vec![3]);
    assert_eq!(run(5, &[10]), vec![5, 6, 7, 8, 9]); // longer than the limit
    assert_eq!(run(5, &[0, 2, 0, 1, 0]), vec![0, 1, 2]); // empty batches
}

#[test]
fn a_limit_past_the_input_returns_everything() {
    assert_eq!(run(100, &[3, 3]), (0..6).collect::<Vec<_>>());
    assert_eq!(run(u64::MAX, &[3, 3]), (0..6).collect::<Vec<_>>());
    assert_eq!(run(5, &[]), Vec::<i64>::new());
}

#[test]
fn the_last_rows_of_drivers_combined_in_order_are_the_last_rows_of_all() {
    // Two branches of a union, combined in branch order: the tail is of the
    // whole, not one tail per branch.
    assert_eq!(run_drivers(2, &[&[3, 3], &[2]]), vec![6, 7]);
    assert_eq!(run_drivers(4, &[&[3], &[1], &[3]]), vec![3, 4, 5, 6]);
    // A branch with fewer rows than the limit contributes all of them.
    assert_eq!(run_drivers(5, &[&[4], &[2]]), vec![1, 2, 3, 4, 5]);
    assert_eq!(run_drivers(3, &[&[4], &[0], &[0]]), vec![1, 2, 3]);
}

#[test]
fn tail_zero_is_empty_and_stops_the_input() {
    assert_eq!(run(0, &[3, 3]), Vec::<i64>::new());
    let (_root, ctx) = ctx(&MemoryConfig::default());
    let t = Tail::new(schema(), 0);
    let global = Sink::global_state(&*t, &ctx).unwrap_or_else(|e| unreachable!("{e}"));
    let mut local = Sink::local_state(&*t, &ctx, &global).unwrap_or_else(|e| unreachable!("{e}"));
    assert!(matches!(
        t.sink(ints(&[1]), &mut local, &global)
            .unwrap_or_else(|e| unreachable!("{e}")),
        SinkStatus::Finished
    ));
}

#[test]
fn memory_is_bounded_by_the_limit_and_released() {
    let (root, ctx) = ctx(&MemoryConfig {
        quantum: 1,
        ..MemoryConfig::default()
    });
    let t = Tail::new(schema(), 10);
    let global = Sink::global_state(&*t, &ctx).unwrap_or_else(|e| unreachable!("{e}"));
    let mut local = Sink::local_state(&*t, &ctx, &global).unwrap_or_else(|e| unreachable!("{e}"));
    for i in 0..200_i64 {
        let rows: Vec<i64> = (i * 1000..(i + 1) * 1000).collect();
        let out = t
            .sink(ints(&rows), &mut local, &global)
            .unwrap_or_else(|e| unreachable!("{e}"));
        assert!(matches!(out, SinkStatus::NeedMoreInput));
        // Ten i64 rows plus slack, never the 8 kB of a whole input batch.
        assert!(root.reserved() < 4096, "{}", root.reserved());
    }
    t.combine(local, &global)
        .unwrap_or_else(|e| unreachable!("{e}"));
    let mut src = Source::local_state(&*t, &ctx, &global).unwrap_or_else(|e| unreachable!("{e}"));
    while ready(t.get_data(&mut src, &global))
        .unwrap_or_else(|e| unreachable!("{e}"))
        .is_some()
    {}
    assert_eq!(root.reserved(), 0);
}

#[test]
fn a_refused_reservation_asks_for_memory_and_keeps_its_state() {
    let (_root, ctx) = ctx(&MemoryConfig {
        limit: Some(1),
        quantum: 1,
        ..MemoryConfig::default()
    });
    let t = Tail::new(schema(), 5);
    let global = Sink::global_state(&*t, &ctx).unwrap_or_else(|e| unreachable!("{e}"));
    let mut local = Sink::local_state(&*t, &ctx, &global).unwrap_or_else(|e| unreachable!("{e}"));
    assert!(matches!(
        t.sink(ints(&[1, 2]), &mut local, &global)
            .unwrap_or_else(|e| unreachable!("{e}")),
        SinkStatus::NeedMemory { .. }
    ));
    t.combine(local, &global)
        .unwrap_or_else(|e| unreachable!("{e}"));
    let mut src = Source::local_state(&*t, &ctx, &global).unwrap_or_else(|e| unreachable!("{e}"));
    assert!(
        ready(t.get_data(&mut src, &global))
            .unwrap_or_else(|e| unreachable!("{e}"))
            .is_none()
    );
}

#[test]
fn weighted_input_is_rejected() {
    let (_root, ctx) = ctx(&MemoryConfig::default());
    let t = Tail::new(schema(), 5);
    let global = Sink::global_state(&*t, &ctx).unwrap_or_else(|e| unreachable!("{e}"));
    let mut local = Sink::local_state(&*t, &ctx, &global).unwrap_or_else(|e| unreachable!("{e}"));
    let b = Batch::with_weights(ints(&[1]).into_parts().0, Weights::Const(2))
        .unwrap_or_else(|e| unreachable!("{e}"));
    let e = t.sink(b, &mut local, &global).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}
