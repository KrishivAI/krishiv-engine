// SPDX-License-Identifier: Apache-2.0
//! A batch whose worst case the pool refuses is applied in slices
//! (`sink::MIN_SLICE_ROWS`), and a refusal part-way through a batch
//! resumes where it stopped when the driver retries it.
#![cfg(not(loom))]
#![expect(
    clippy::disallowed_methods,
    reason = "test-only runtime plumbing; never runs on a compute worker"
)]

use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_memory::{ConsumerHandle, MemoryConfig, MemoryPool, SpillOwner, TreePool};
use krishiv_types::batch::Batch;

use super::sink::resumes_at;
use super::*;
use crate::operator::SinkStatus;
use crate::{
    ExecConfig, Executor, OperatorId, QueryContext, QueryId, Sink, Source, TaskContext, TokioClock,
    TokioExecutor, WorkloadClass,
};

/// Bytes of each `v` value: a 4,096-row batch carries ~800 KB of them,
/// which the worst case of a string `max` counts in full.
const VALUE: usize = 200;
const ROWS: usize = 4096;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Utf8, true),
    ]))
}

/// A batch of `keys` (the value is the key, padded to [`VALUE`] bytes).
fn batch(keys: impl Iterator<Item = i64>) -> Batch {
    let keys: Vec<i64> = keys.collect();
    let values: Vec<String> = keys.iter().map(|k| format!("{k:0>VALUE$}")).collect();
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(keys)),
        Arc::new(StringArray::from(values)),
    ];
    Batch::new(RecordBatch::try_new(schema(), cols).unwrap())
}

fn spec() -> AggregateSpec {
    let s = schema();
    AggregateSpec {
        input: Arc::clone(&s),
        group_by: vec![0],
        aggregates: vec![
            AggregateExpr::count_star("c"),
            AggregateExpr::new(AggregateFunction::Sum, 0, DataType::Int64, "s"),
            AggregateExpr::new(AggregateFunction::Max, 1, DataType::Utf8, "m"),
        ],
        output: AggregateOutput::Final,
        ansi: true,
        spill: None,
        require_unit_weights: false,
        legacy_statistical: false,
        udafs: None,
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Outcome {
    /// `(key, count, sum, max length)`, sorted.
    groups: Vec<(i64, i64, i64, usize)>,
    /// `NeedMemory` answers, and those after part of the batch applied.
    refused: usize,
    resumed: usize,
}

/// Aggregates `input` on one driver in a query pool of `limit + ballast`
/// bytes, `ballast` of them held by another consumer until the sink first
/// answers `NeedMemory` (as a peer's freed bytes would); then retries the
/// same batch, as the driver does.
async fn aggregate(input: &[Batch], rows: bool, limit: usize, ballast: usize) -> Outcome {
    let root = TreePool::with_clock(
        &MemoryConfig {
            limit: Some(1 << 30),
            quantum: 16 << 10,
            spill_wait_timeout: Duration::from_secs(5),
            abort_release_timeout: Duration::from_secs(30),
            spill_scratch_reserve: krishiv_memory::DEFAULT_SPILL_SCRATCH_RESERVE,
            resident_guard: true,
        },
        Arc::new(TokioClock),
    );
    let pool: Arc<dyn MemoryPool> = root.child("q", Some(limit + ballast));
    let mut held = ConsumerHandle::register(Arc::clone(&pool), "ballast", None).reservation();
    if ballast > 0 {
        // All the pool lets a consumer have but `limit` (it keeps some
        // back for spill scratch).
        let free = match held.try_grow(usize::MAX >> 1) {
            Err(krishiv_memory::MemoryError::Exhausted { available, .. }) => available,
            other => panic!("{other:?}"),
        };
        held.try_grow(free - limit).unwrap();
    }
    let config = ExecConfig {
        agg_row_layout: rows,
        ..ExecConfig::default()
    };
    let q = QueryContext::new(QueryId(1), WorkloadClass::Batch, config, pool);
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
    let ctx = TaskContext::new(q, executor, SpillOwner::new(), OperatorId(0), "agg");
    let agg = HashAggregate::new(spec()).unwrap();
    let g = Sink::global_state(&*agg, &ctx).unwrap();
    let mut l = Sink::local_state(&*agg, &ctx, &g).unwrap();
    let mut out = Outcome::default();
    for b in input {
        while let SinkStatus::NeedMemory { .. } = agg.sink(b.clone(), &mut l, &g).unwrap() {
            assert!(held.size() > 0, "refused with the pool's bytes free");
            out.refused += 1;
            out.resumed += usize::from(resumes_at(&l).is_some_and(|at| at > 0));
            held.shrink(held.size());
        }
    }
    agg.combine(l, &g).unwrap();
    agg.finalize(&g, &ctx).await.unwrap();
    let src = ctx.with_dependency(Arc::clone(&g));
    let sg = Source::global_state(&*agg, &src).unwrap();
    let mut sl = Source::local_state(&*agg, &src, &sg).unwrap();
    while let Some(b) = agg.get_data(&mut sl, &sg).await.unwrap() {
        let d = b.data();
        let col = |i: usize| d.column(i).as_primitive::<Int64Type>().clone();
        let (k, c, s) = (col(0), col(1), col(2));
        let m = d.column(3);
        for r in 0..d.num_rows() {
            let len = match m.data_type() {
                DataType::Utf8View => m.as_string_view().value(r).len(),
                _ => m.as_string::<i32>().value(r).len(),
            };
            out.groups.push((k.value(r), c.value(r), s.value(r), len));
        }
    }
    out.groups.sort_unstable();
    drop((sl, sg, src, g, held));
    assert_eq!(root.reserved(), 0, "memory still held");
    out
}

/// Eight batches over five keys in a 400 KB pool: a whole batch's worst
/// case (~1 MB, every row a new group with its value kept) never fits, its
/// slices do, and the groups are those of an unlimited run.
#[tokio::test]
async fn a_batch_the_pool_refuses_whole_is_applied_in_slices() {
    let input: Vec<Batch> = (0..8_i64)
        .map(|b| batch((0..ROWS as i64).map(move |i| (b * 7 + i) % 5)))
        .collect();
    for rows in [true, false] {
        let want = aggregate(&input, rows, 1 << 30, 0).await;
        assert_eq!(want.groups.len(), 5);
        let got = aggregate(&input, rows, 400 << 10, 0).await;
        assert_eq!(got.groups, want.groups, "row layout {rows}");
        assert_eq!(got.refused, 0, "row layout {rows}: a slice was refused");
    }
}

/// A batch of 1,024 new keys, then hits: under some pool sizes the table's
/// growth makes a later slice's memory refused after earlier slices of the
/// same batch were applied. The retry applies only the rest (a group's
/// count would double otherwise).
#[tokio::test]
async fn a_slice_refused_part_way_resumes_after_the_applied_rows() {
    let first = batch((0..ROWS as i64).map(|i| if i < 1024 { 1000 + i } else { i % 5 }));
    let rest: Vec<Batch> = (0..3_i64)
        .map(|b| batch((0..ROWS as i64).map(move |i| (b + i) % 5)))
        .collect();
    let input: Vec<Batch> = std::iter::once(first).chain(rest).collect();
    for rows in [true, false] {
        let want = aggregate(&input, rows, 1 << 30, 0).await;
        assert_eq!(want.groups.len(), 1029);
        let mut resumed = 0;
        for kib in (150..=700).step_by(25) {
            let got = aggregate(&input, rows, kib << 10, 4 << 20).await;
            assert_eq!(got.groups, want.groups, "row layout {rows}, {kib} KiB");
            resumed += got.resumed;
        }
        assert!(
            resumed > 0,
            "row layout {rows}: no pool size resumed a batch"
        );
    }
}
