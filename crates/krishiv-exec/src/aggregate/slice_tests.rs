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
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray, StringViewArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_common::error::ErrorClass;
use krishiv_memory::{ConsumerHandle, MemoryConfig, MemoryPool, SpillOwner, TreePool};
use krishiv_types::batch::Batch;

use super::sink::resumes_at;
use super::*;
use crate::operator::{GlobalState, SinkStatus};
use crate::{
    ExecConfig, Executor, OperatorId, QueryContext, QueryId, Sink, Source, TaskContext, TokioClock,
    TokioExecutor, WorkloadClass,
};

/// Bytes of each `v` value: a 4,096-row batch carries ~800 KB of them,
/// which the worst case of a string `max` counts in full.
const VALUE: usize = 200;
const ROWS: usize = 4096;

/// The value column's type: `Utf8`, or `Utf8View` as Parquet scans read.
fn value_type(view: bool) -> DataType {
    if view {
        DataType::Utf8View
    } else {
        DataType::Utf8
    }
}

fn schema(view: bool) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", value_type(view), true),
        Field::new("o", DataType::Int64, true),
    ]))
}

/// A batch of `keys` (the value is the key, padded to [`VALUE`] bytes;
/// `o` numbers the rows from `first`, for `first` and `last`).
pub(super) fn batch(view: bool, first: i64, keys: impl Iterator<Item = i64>) -> Batch {
    let keys: Vec<i64> = keys.collect();
    let values: Vec<String> = keys.iter().map(|k| format!("{k:0>VALUE$}")).collect();
    let values: ArrayRef = if view {
        Arc::new(StringViewArray::from_iter_values(values))
    } else {
        Arc::new(StringArray::from(values))
    };
    let order = Int64Array::from_iter_values(first..first + keys.len() as i64);
    let cols: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(keys)), values, Arc::new(order)];
    Batch::new(RecordBatch::try_new(schema(view), cols).unwrap())
}

pub(super) fn spec(view: bool) -> AggregateSpec {
    let s = schema(view);
    let pick = |f| AggregateExpr::new(f, 2, DataType::Int64, "p");
    AggregateSpec {
        input: Arc::clone(&s),
        group_by: vec![0],
        aggregates: vec![
            AggregateExpr::count_star("c"),
            AggregateExpr::new(AggregateFunction::Sum, 0, DataType::Int64, "s"),
            AggregateExpr::new(AggregateFunction::Max, 1, value_type(view), "m"),
            // Order-sensitive: slices fold each group's rows in input order.
            pick(AggregateFunction::First {
                ignore_nulls: false,
            }),
            pick(AggregateFunction::Last {
                ignore_nulls: false,
            }),
        ],
        output: AggregateOutput::Final,
        ansi: true,
        spill: None,
        require_unit_weights: false,
        moments: Default::default(),
        udafs: None,
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Outcome {
    /// `(key, count, sum, max length, first o, last o)`, sorted.
    groups: Vec<(i64, i64, i64, usize, i64, i64)>,
    /// `NeedMemory` answers, and those after part of the batch applied.
    refused: usize,
    resumed: usize,
}

/// Aggregates `input` on one driver in a query pool of `limit + ballast`
/// bytes, `ballast` of them held by another consumer until the sink first
/// answers `NeedMemory` (as a peer's freed bytes would); then retries the
/// same batch, as the driver does. With `misuse`, the first refusal after
/// part of a batch applied is answered with an equal batch in other
/// buffers instead, and then `combine`: both must fail.
async fn aggregate(
    input: &[Batch],
    view: bool,
    rows: bool,
    limit: usize,
    ballast: usize,
    misuse: bool,
) -> Outcome {
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
    let agg = HashAggregate::new(spec(view)).unwrap();
    let g = Sink::global_state(&*agg, &ctx).unwrap();
    let mut l = Sink::local_state(&*agg, &ctx, &g).unwrap();
    let mut out = Outcome::default();
    for b in input {
        while let SinkStatus::NeedMemory { .. } = agg.sink(b.clone(), &mut l, &g).unwrap() {
            assert!(held.size() > 0, "refused with the pool's bytes free");
            out.refused += 1;
            let resumed = resumes_at(&l).is_some_and(|at| at > 0);
            out.resumed += usize::from(resumed);
            held.shrink(held.size());
            if misuse && resumed {
                let copy = Batch::new(
                    arrow_select::concat::concat_batches(&b.data().schema(), [b.data()]).unwrap(),
                );
                let e = agg.sink(copy, &mut l, &g).unwrap_err();
                assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
                let e = agg.combine(l, &g).unwrap_err();
                assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
                drop((g, held));
                assert_eq!(root.reserved(), 0, "memory still held");
                return out;
            }
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
            let (f, l) = (col(4), col(5));
            out.groups.push((
                k.value(r),
                c.value(r),
                s.value(r),
                len,
                f.value(r),
                l.value(r),
            ));
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
    for (view, rows) in [(false, true), (false, false), (true, true), (true, false)] {
        let input: Vec<Batch> = (0..8_i64)
            .map(|b| {
                batch(
                    view,
                    b * 10_000,
                    (0..ROWS as i64).map(move |i| (b * 7 + i) % 5),
                )
            })
            .collect();
        let want = aggregate(&input, view, rows, 1 << 30, 0, false).await;
        assert_eq!(want.groups.len(), 5);
        let got = aggregate(&input, view, rows, 400 << 10, 0, false).await;
        let case = format!("view {view}, row layout {rows}");
        assert_eq!(got.groups, want.groups, "{case}");
        assert_eq!(got.refused, 0, "{case}: a slice was refused");
    }
}

/// A batch of 1,024 new keys, then hits: under some pool sizes the table's
/// growth makes a later slice's memory refused after earlier slices of the
/// same batch were applied. The retry applies only the rest (a group's
/// count would double otherwise). The sweep of pool sizes finds where that
/// happens under today's estimates: if a change to `projected` or
/// `transient` moves it, widen the sweep rather than drop the assertion.
#[tokio::test]
async fn a_slice_refused_part_way_resumes_after_the_applied_rows() {
    let view = false;
    let first = batch(
        view,
        0,
        (0..ROWS as i64).map(|i| if i < 1024 { 1000 + i } else { i % 5 }),
    );
    let rest: Vec<Batch> = (1..4_i64)
        .map(|b| batch(view, b * 10_000, (0..ROWS as i64).map(move |i| (b + i) % 5)))
        .collect();
    let input: Vec<Batch> = std::iter::once(first).chain(rest).collect();
    for rows in [true, false] {
        let want = aggregate(&input, view, rows, 1 << 30, 0, false).await;
        assert_eq!(want.groups.len(), 1029);
        let mut resumed = 0;
        for kib in (150..=700).step_by(25) {
            let got = aggregate(&input, view, rows, kib << 10, 4 << 20, false).await;
            assert_eq!(got.groups, want.groups, "row layout {rows}, {kib} KiB");
            resumed += got.resumed;
            if got.resumed > 0 {
                let misused = aggregate(&input, view, rows, kib << 10, 4 << 20, true).await;
                assert_eq!(misused.resumed, 1, "row layout {rows}, {kib} KiB");
            }
        }
        assert!(
            resumed > 0,
            "row layout {rows}: no pool size resumed a batch"
        );
    }
}

/// A batch refused whole while the driver's table holds more than the
/// shortfall: a spilling aggregate asks for the memory with nothing
/// applied, so the driver spills the table (as without slices), rather
/// than fill the pool slice by slice and leave the operators after it no
/// room; without a spill scope the batch is applied in slices.
#[tokio::test]
async fn a_spilling_table_that_frees_the_shortfall_spills_rather_than_slices() {
    for spilling in [true, false] {
        let dir = tempfile::Builder::new()
            .prefix("krishiv-slice-")
            .tempdir()
            .unwrap();
        let manager = krishiv_memory::SpillManager::open(&krishiv_memory::SpillConfig {
            dir: dir.path().join("spill"),
            ..krishiv_memory::SpillConfig::default()
        })
        .unwrap();
        let root = TreePool::new(&MemoryConfig {
            quantum: 16 << 10,
            ..MemoryConfig::default()
        });
        let pool: Arc<dyn MemoryPool> = root.child("q", Some(64 << 20));
        let q = QueryContext::new(
            QueryId(1),
            WorkloadClass::Batch,
            ExecConfig::default(),
            pool,
        );
        let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
        let ctx = TaskContext::new(q, executor, SpillOwner::new(), OperatorId(0), "agg");
        let spec = AggregateSpec {
            spill: spilling.then(|| manager.scope().unwrap()),
            ..spec(false)
        };
        let agg = HashAggregate::new(spec).unwrap();
        let g = Sink::global_state(&*agg, &ctx).unwrap();
        let mut l = Sink::local_state(&*agg, &ctx, &g).unwrap();
        // A first batch of new keys fits; the table then holds `held`.
        let a = batch(false, 0, 0..ROWS as i64);
        assert!(matches!(
            agg.sink(a, &mut l, &g).unwrap(),
            SinkStatus::NeedMoreInput
        ));
        let held = root.reserved();
        // Leave half of that free: the next batch of new keys (worst case
        // about `held`, ~1.3 MB with the 128 KiB spill headroom) is refused
        // whole, and the table (~1.2 MB) frees the rest.
        let pool = ctx.query().pool();
        let mut ballast = ConsumerHandle::register(Arc::clone(pool), "ballast", None).reservation();
        let free = match ballast.try_grow(usize::MAX >> 1) {
            Err(krishiv_memory::MemoryError::Exhausted { available, .. }) => available,
            other => panic!("{other:?}"),
        };
        ballast.try_grow(free - held / 2).unwrap();
        let b = batch(false, 10_000, ROWS as i64..2 * ROWS as i64);
        let status = agg.sink(b, &mut l, &g).unwrap();
        if spilling {
            assert!(
                matches!(status, SinkStatus::NeedMemory { .. }),
                "no spill asked"
            );
            assert_eq!(resumes_at(&l), None, "a slice was applied before the spill");
        } else {
            // Today's estimates refuse a later slice after 1,024 rows; a
            // batch that fit whole would be `NeedMoreInput` with no resume.
            assert!(
                resumes_at(&l).is_some(),
                "without a spill scope the batch is applied in slices"
            );
        }
        drop((l, g, ballast, ctx));
    }
}

/// A slice's copy is sized by its own rows: a view array's by its views
/// (a copy shares the data buffers), a `Utf8` array's by its offsets and
/// bytes; neither by its parent's buffers.
#[test]
fn own_bytes_counts_a_slices_rows() {
    let b = batch(true, 0, 0..ROWS as i64);
    let views = b.data().column(1);
    let slice = views.slice(0, 256);
    let got = super::table::own_bytes(slice.as_ref());
    assert!((256 * 16..256 * 16 + 512).contains(&got), "views: {got}");
    let b = batch(false, 0, 0..ROWS as i64);
    let strings = b.data().column(1);
    let whole = super::table::own_bytes(strings.as_ref());
    let slice = super::table::own_bytes(strings.slice(0, 256).as_ref());
    assert!(whole >= ROWS * VALUE, "whole: {whole}");
    assert!(
        slice >= 256 * VALUE && slice < whole / 8,
        "slice: {slice} of {whole}"
    );
}

/// A driver that spilled advertises 0 spillable bytes; another driver then
/// combines its table and ends, charging it to a consumer no one can ask to
/// spill. When the first driver's next batch is refused, its signal counts
/// the combined tables again, so its self-spill (which writes them) is
/// asked for them: otherwise arbitration finds nothing to spill and the
/// query fails with `UNABLE_TO_ACQUIRE_MEMORY` (the intermittent failure
/// of `krishiv-session` `metrics_registry::a_spilling_query_folds_spill_totals`).
#[tokio::test]
async fn a_refused_batch_counts_tables_combined_since_the_last_spill() {
    use krishiv_memory::Spillable as _;
    let dir = tempfile::Builder::new()
        .prefix("krishiv-slice-")
        .tempdir()
        .unwrap();
    let manager = krishiv_memory::SpillManager::open(&krishiv_memory::SpillConfig {
        dir: dir.path().join("spill"),
        ..krishiv_memory::SpillConfig::default()
    })
    .unwrap();
    let root = TreePool::new(&MemoryConfig {
        quantum: 16 << 10,
        ..MemoryConfig::default()
    });
    let pool: Arc<dyn MemoryPool> = root.child("q", Some(64 << 20));
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig::default(),
        pool,
    );
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
    let (a, b) = (task(), task());
    let spec = AggregateSpec {
        spill: Some(manager.scope().unwrap()),
        ..spec(false)
    };
    let agg = HashAggregate::new(spec).unwrap();
    let g = Sink::global_state(&*agg, &a).unwrap();
    // Driver A folds a batch of new keys, then spills its table.
    let mut la = Sink::local_state(&*agg, &a, &g).unwrap();
    let first = batch(false, 0, 0..ROWS as i64);
    assert!(matches!(
        agg.sink(first, &mut la, &g).unwrap(),
        SinkStatus::NeedMoreInput
    ));
    assert!(agg.spill(&mut la, usize::MAX, &a).await.unwrap() > 0);
    let signal = Arc::clone(a.consumer().signal());
    assert_eq!(signal.spillable_bytes(), 0);
    // Driver B folds its own keys, combines and ends.
    let before = root.reserved();
    let mut lb = Sink::local_state(&*agg, &b, &g).unwrap();
    let second = batch(false, 10_000, ROWS as i64..2 * ROWS as i64);
    assert!(matches!(
        agg.sink(second, &mut lb, &g).unwrap(),
        SinkStatus::NeedMoreInput
    ));
    agg.combine(lb, &g).unwrap();
    drop(b);
    let combined = root.reserved() - before;
    assert!(combined > 0);
    // Nearly no room left: A's next batch of new keys is refused whole.
    let mut ballast =
        ConsumerHandle::register(Arc::clone(a.query().pool()), "ballast", None).reservation();
    let free = match ballast.try_grow(usize::MAX >> 1) {
        Err(krishiv_memory::MemoryError::Exhausted { available, .. }) => available,
        other => panic!("{other:?}"),
    };
    ballast.try_grow(free - (64 << 10)).unwrap();
    let third = batch(false, 20_000, 2 * ROWS as i64..3 * ROWS as i64);
    assert!(matches!(
        agg.sink(third, &mut la, &g).unwrap(),
        SinkStatus::NeedMemory { .. }
    ));
    assert!(
        signal.spillable_bytes() >= combined,
        "A's signal advertises {} bytes; B combined {combined}",
        signal.spillable_bytes()
    );
    // A's spill writes the combined tables.
    let held = root.reserved();
    let freed = agg.spill(&mut la, usize::MAX, &a).await.unwrap();
    assert!(freed >= combined, "freed {freed} of {combined}");
    assert!(root.reserved() < held);
    drop((la, g, ballast, a));
}

/// A driver combines its table and ends while the other drivers still run:
/// the table is charged to the combined tables' own consumer, whose hook
/// spills it on request. A request in the same query that only that table
/// can fund is granted (before, the table sat on the ended driver's
/// consumer with no spill hook, and the request failed with
/// `UNABLE_TO_ACQUIRE_MEMORY`), and `finalize` reads the spilled run.
#[tokio::test]
async fn combined_tables_spill_when_a_round_asks_them() {
    let dir = tempfile::Builder::new()
        .prefix("krishiv-slice-")
        .tempdir()
        .unwrap();
    let manager = krishiv_memory::SpillManager::open(&krishiv_memory::SpillConfig {
        dir: dir.path().join("spill"),
        ..krishiv_memory::SpillConfig::default()
    })
    .unwrap();
    let root = TreePool::with_clock(
        &MemoryConfig {
            quantum: 16 << 10,
            ..MemoryConfig::default()
        },
        Arc::new(TokioClock),
    );
    let pool: Arc<dyn MemoryPool> = root.child("q", Some(64 << 20));
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig::default(),
        pool,
    );
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
    let spec = AggregateSpec {
        spill: Some(manager.scope().unwrap()),
        ..spec(false)
    };
    let agg = HashAggregate::new(spec).unwrap();
    let coordinator = task();
    let g = Sink::global_state(&*agg, &coordinator).unwrap();
    // Driver B folds new keys, combines and ends.
    let b = task();
    let mut lb = Sink::local_state(&*agg, &b, &g).unwrap();
    let keys = batch(false, 0, 0..ROWS as i64);
    assert!(matches!(
        agg.sink(keys, &mut lb, &g).unwrap(),
        SinkStatus::NeedMoreInput
    ));
    agg.combine(lb, &g).unwrap();
    drop(b);
    let combined = root.reserved();
    assert!(combined > 0);
    // Another consumer of the query fills the pool, then asks for half the
    // combined table's bytes more: only spilling that table makes room.
    let pool = q.pool();
    let mut ballast = ConsumerHandle::register(Arc::clone(pool), "ballast", None).reservation();
    let free = match ballast.try_grow(usize::MAX >> 1) {
        Err(krishiv_memory::MemoryError::Exhausted { available, .. }) => available,
        other => panic!("{other:?}"),
    };
    ballast.try_grow(free).unwrap();
    let mut wants = ConsumerHandle::register(Arc::clone(pool), "wants", None).reservation();
    tokio::time::timeout(Duration::from_secs(30), wants.grow(combined / 2))
        .await
        .expect("the round waited out its deadline")
        .expect("the combined table was not spilled");
    drop((wants, ballast));
    // Every group comes back from the spilled run.
    let fin = task();
    agg.finalize(&g, &fin).await.unwrap();
    let src = fin.with_dependency(Arc::clone(&g));
    let sg = Source::global_state(&*agg, &src).unwrap();
    let mut sl = Source::local_state(&*agg, &src, &sg).unwrap();
    let mut groups = 0;
    while let Some(out) = agg.get_data(&mut sl, &sg).await.unwrap() {
        groups += out.data().num_rows();
    }
    assert_eq!(groups, ROWS);
    drop((sl, sg, src, g, coordinator));
    assert_eq!(root.reserved(), 0, "memory still held");
}

/// While the combined tables' hook writes them, the last driver's
/// `spill_global` frees nothing: a free job would start the disk phase
/// before the hook's runs reach `runs`, and those groups would be lost.
/// Every group of both drivers comes back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spill_global_waits_out_a_combined_tables_write() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    let dir = tempfile::Builder::new()
        .prefix("krishiv-slice-")
        .tempdir()
        .unwrap();
    let manager = krishiv_memory::SpillManager::open(&krishiv_memory::SpillConfig {
        dir: dir.path().join("spill"),
        ..krishiv_memory::SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    // The first spill writer (the hook's) says so and waits for `release`.
    let (entered_tx, entered_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let gate = std::sync::Mutex::new((entered_tx, release_rx));
    let calls = AtomicUsize::new(0);
    scope.set_test_writer_hook(Arc::new(move || {
        if calls.fetch_add(1, Ordering::SeqCst) == 0 {
            let g = gate.lock().unwrap();
            g.0.send(()).unwrap();
            g.1.recv().unwrap();
        }
    }));
    let root = TreePool::with_clock(
        &MemoryConfig {
            quantum: 16 << 10,
            ..MemoryConfig::default()
        },
        Arc::new(TokioClock),
    );
    let pool: Arc<dyn MemoryPool> = root.child("q", Some(64 << 20));
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig::default(),
        pool,
    );
    let executor: Arc<dyn Executor> =
        TokioExecutor::new(&crate::ExecutorConfig::default()).unwrap();
    let task = || {
        TaskContext::new(
            Arc::clone(&q),
            Arc::clone(&executor),
            SpillOwner::new(),
            OperatorId(0),
            "agg",
        )
    };
    let spec = AggregateSpec {
        spill: Some(scope),
        ..spec(false)
    };
    let agg = HashAggregate::new(spec).unwrap();
    let coordinator = task();
    let g = Sink::global_state(&*agg, &coordinator).unwrap();
    // Driver D (the last) folds its keys; driver B folds others, combines
    // and ends.
    let (b, d) = (task(), task());
    let mut ld = Sink::local_state(&*agg, &d, &g).unwrap();
    let mut lb = Sink::local_state(&*agg, &b, &g).unwrap();
    let ok = |s: SinkStatus| assert!(matches!(s, SinkStatus::NeedMoreInput));
    ok(agg
        .sink(batch(false, 0, 0..ROWS as i64), &mut ld, &g)
        .unwrap());
    ok(agg
        .sink(
            batch(false, 10_000, ROWS as i64..2 * ROWS as i64),
            &mut lb,
            &g,
        )
        .unwrap());
    agg.combine(lb, &g).unwrap();
    drop(b);
    // A request in the query that only a spill funds: the hook takes B's
    // table and its write stops at the gate.
    let pool = Arc::clone(q.pool());
    let mut ballast = ConsumerHandle::register(Arc::clone(&pool), "ballast", None).reservation();
    let free = match ballast.try_grow(usize::MAX >> 1) {
        Err(krishiv_memory::MemoryError::Exhausted { available, .. }) => available,
        other => panic!("{other:?}"),
    };
    ballast.try_grow(free).unwrap();
    let mut wants = ConsumerHandle::register(pool, "wants", None).reservation();
    let asked = tokio::spawn(async move { wants.grow(64 << 10).await.map(|()| wants) });
    tokio::task::spawn_blocking(move || entered_rx.recv().unwrap())
        .await
        .unwrap();
    // D combines and, as the last driver, services a spill request.
    agg.combine(ld, &g).unwrap();
    assert_eq!(
        agg.spill_global(&g, 1, &d).await.unwrap(),
        0,
        "a free job ran while the hook wrote"
    );
    release_tx.send(()).unwrap();
    let wants = asked.await.unwrap().unwrap();
    drop((wants, ballast));
    agg.finalize(&g, &d).await.unwrap();
    let src = d.with_dependency(Arc::clone(&g));
    let sg = Source::global_state(&*agg, &src).unwrap();
    let mut sl = Source::local_state(&*agg, &src, &sg).unwrap();
    let mut groups = 0;
    while let Some(out) = agg.get_data(&mut sl, &sg).await.unwrap() {
        groups += out.data().num_rows();
    }
    assert_eq!(groups, 2 * ROWS);
    drop((sl, sg, src, g, coordinator));
    assert_eq!(root.reserved(), 0, "memory still held");
}

/// The pool refuses the scratch of the combined tables' hook write (here:
/// another consumer takes the memory as the writer starts). The write
/// frees nothing and is no error: the tables go back to the hook, which
/// offers them again, so a later round spills them, and `finalize` reads
/// every group. Before, the refusal failed the query.
#[tokio::test]
async fn a_combined_tables_write_refused_scratch_keeps_the_tables() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let dir = tempfile::Builder::new()
        .prefix("krishiv-slice-")
        .tempdir()
        .unwrap();
    let manager = krishiv_memory::SpillManager::open(&krishiv_memory::SpillConfig {
        dir: dir.path().join("spill"),
        ..krishiv_memory::SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    let root = TreePool::with_clock(
        &MemoryConfig {
            quantum: 16 << 10,
            ..MemoryConfig::default()
        },
        Arc::new(TokioClock),
    );
    let pool: Arc<dyn MemoryPool> = root.child("q", Some(64 << 20));
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig::default(),
        pool,
    );
    // The first writer's start takes every free byte of the query.
    let grab_reservation = || {
        ConsumerHandle::register(Arc::clone(q.pool()), "grab", None)
            .reservation()
            .for_spill_scratch()
    };
    let grab = Arc::new(std::sync::Mutex::new(grab_reservation()));
    let calls = Arc::new(AtomicUsize::new(0));
    {
        let (grab, calls) = (Arc::clone(&grab), Arc::clone(&calls));
        scope.set_test_writer_hook(Arc::new(move || {
            if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                // Scratch grants are exact and may take the pool's
                // spill-scratch floor: halve down to one byte.
                let mut g = grab.lock().unwrap();
                let mut n = 1usize << 40;
                while n > 0 {
                    if g.try_grow(n).is_err() {
                        n /= 2;
                    }
                }
            }
        }));
    }
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
    let spec = AggregateSpec {
        spill: Some(scope),
        ..spec(false)
    };
    let agg = HashAggregate::new(spec).unwrap();
    let coordinator = task();
    let g = Sink::global_state(&*agg, &coordinator).unwrap();
    let b = task();
    let mut lb = Sink::local_state(&*agg, &b, &g).unwrap();
    assert!(matches!(
        agg.sink(batch(false, 0, 0..ROWS as i64), &mut lb, &g)
            .unwrap(),
        SinkStatus::NeedMoreInput
    ));
    agg.combine(lb, &g).unwrap();
    drop(b);
    let combined = root.reserved();
    let pool = q.pool();
    let mut ballast = ConsumerHandle::register(Arc::clone(pool), "ballast", None).reservation();
    let free = match ballast.try_grow(usize::MAX >> 1) {
        Err(krishiv_memory::MemoryError::Exhausted { available, .. }) => available,
        other => panic!("{other:?}"),
    };
    ballast.try_grow(free).unwrap();
    // Round 1: the hook's write is refused its scratch; nothing is freed.
    // (Without abort: with nothing spillable left, the round would abort
    // the query instead.)
    let mut wants = ConsumerHandle::register(Arc::clone(pool), "wants", None).reservation();
    let first = tokio::time::timeout(
        Duration::from_secs(30),
        wants.grow_without_abort(combined / 2),
    )
    .await
    .expect("round 1 waited out its deadline");
    assert!(
        matches!(first, Err(krishiv_memory::MemoryError::Exhausted { .. })),
        "{first:?}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1, "the hook did not write");
    q.check_cancelled()
        .expect("the refused write failed the query");
    // The grab lets go; round 2 spills the tables the hook offers again.
    drop(std::mem::replace(
        &mut *grab.lock().unwrap(),
        grab_reservation(),
    ));
    tokio::time::timeout(Duration::from_secs(30), wants.grow(combined / 2))
        .await
        .expect("round 2 waited out its deadline")
        .expect("the kept tables were not offered again");
    assert!(calls.load(Ordering::SeqCst) > 1, "round 2 wrote no run");
    drop((wants, ballast));
    let fin = task();
    agg.finalize(&g, &fin).await.unwrap();
    let src = fin.with_dependency(Arc::clone(&g));
    let sg = Source::global_state(&*agg, &src).unwrap();
    let mut sl = Source::local_state(&*agg, &src, &sg).unwrap();
    let mut groups = 0;
    while let Some(out) = agg.get_data(&mut sl, &sg).await.unwrap() {
        groups += out.data().num_rows();
    }
    assert_eq!(groups, ROWS);
    drop((sl, sg, src, g, coordinator, grab));
    assert_eq!(root.reserved(), 0, "memory still held");
}

/// One query whose aggregate spills to a scope of its own, with driver B's
/// table combined (B ended), as the combined-tables hook tests need it.
struct Combined {
    _dir: tempfile::TempDir,
    root: Arc<TreePool>,
    q: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
    agg: Arc<HashAggregate>,
    g: Option<GlobalState>,
    coordinator: Option<TaskContext>,
}

impl Combined {
    /// `max_bytes` caps the spill files (`None`: the default cap); `hook`
    /// runs at each writer's start.
    fn new(max_bytes: Option<u64>, hook: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self::with_tables(max_bytes, hook, &[(0, 0)])
    }

    /// As [`new`](Self::new), with one combined table per `(partition,
    /// first)`: keys `0..ROWS`, the order column counting from `first`.
    fn with_tables(
        max_bytes: Option<u64>,
        hook: Arc<dyn Fn() + Send + Sync>,
        tables: &[(u32, i64)],
    ) -> Self {
        let dir = tempfile::Builder::new()
            .prefix("krishiv-slice-")
            .tempdir()
            .unwrap();
        let manager = krishiv_memory::SpillManager::open(&krishiv_memory::SpillConfig {
            dir: dir.path().join("spill"),
            max_bytes: max_bytes.unwrap_or(krishiv_memory::SpillConfig::default().max_bytes),
            ..krishiv_memory::SpillConfig::default()
        })
        .unwrap();
        let scope = manager.scope().unwrap();
        scope.set_test_writer_hook(hook);
        let root = TreePool::with_clock(
            &MemoryConfig {
                quantum: 16 << 10,
                // The process's resident set is not under test: an earlier
                // query of the same test would shift the pool's room.
                resident_guard: false,
                ..MemoryConfig::default()
            },
            Arc::new(TokioClock),
        );
        let pool: Arc<dyn MemoryPool> = root.child("q", Some(64 << 20));
        let q = QueryContext::new(
            QueryId(1),
            WorkloadClass::Batch,
            ExecConfig::default(),
            pool,
        );
        let executor: Arc<dyn Executor> =
            TokioExecutor::new(&crate::ExecutorConfig::default()).unwrap();
        let agg = HashAggregate::new(AggregateSpec {
            spill: Some(scope),
            ..spec(false)
        })
        .unwrap();
        let mut c = Self {
            _dir: dir,
            root,
            q,
            executor,
            agg,
            g: None,
            coordinator: None,
        };
        let coordinator = c.task();
        c.g = Some(Sink::global_state(&*c.agg, &coordinator).unwrap());
        c.coordinator = Some(coordinator);
        for &(partition, first) in tables {
            c.combine(partition, first);
        }
        c
    }

    /// A driver of `partition` folds keys `0..ROWS` (order from `first`),
    /// combines and ends.
    fn combine(&self, partition: u32, first: i64) {
        let d = self.task().with_partition(partition);
        let mut l = Sink::local_state(&*self.agg, &d, self.g()).unwrap();
        assert!(matches!(
            self.agg
                .sink(batch(false, first, 0..ROWS as i64), &mut l, self.g())
                .unwrap(),
            SinkStatus::NeedMoreInput
        ));
        self.agg.combine(l, self.g()).unwrap();
    }

    fn task(&self) -> TaskContext {
        TaskContext::new(
            Arc::clone(&self.q),
            Arc::clone(&self.executor),
            SpillOwner::new(),
            OperatorId(0),
            "agg",
        )
    }

    fn g(&self) -> &GlobalState {
        self.g.as_ref().unwrap()
    }

    /// Fills the query's pool, then asks in the background for `bytes`
    /// that only a spill of the combined tables funds.
    fn ask(
        &self,
        bytes: usize,
    ) -> (
        krishiv_memory::Reservation,
        tokio::task::JoinHandle<Result<krishiv_memory::Reservation, krishiv_memory::MemoryError>>,
    ) {
        let pool = self.q.pool();
        let mut ballast = ConsumerHandle::register(Arc::clone(pool), "ballast", None).reservation();
        let free = match ballast.try_grow(usize::MAX >> 1) {
            Err(krishiv_memory::MemoryError::Exhausted { available, .. }) => available,
            other => panic!("{other:?}"),
        };
        ballast.try_grow(free).unwrap();
        let mut wants = ConsumerHandle::register(Arc::clone(pool), "wants", None).reservation();
        let asked = tokio::spawn(async move { wants.grow(bytes).await.map(|()| wants) });
        (ballast, asked)
    }

    /// Finalizes on `fin` and counts the groups read back.
    async fn groups(&self, fin: TaskContext) -> crate::error::Result<usize> {
        Ok(self.rows(fin).await?.len())
    }

    /// Finalizes on `fin`: `(key, count, sum, first o, last o)`, sorted.
    async fn rows(&self, fin: TaskContext) -> crate::error::Result<Vec<[i64; 5]>> {
        self.agg.finalize(self.g(), &fin).await?;
        let src = fin.with_dependency(Arc::clone(self.g()));
        let sg = Source::global_state(&*self.agg, &src)?;
        let mut sl = Source::local_state(&*self.agg, &src, &sg)?;
        let mut rows = Vec::new();
        while let Some(out) = self.agg.get_data(&mut sl, &sg).await? {
            let d = out.data();
            let col = |i: usize| d.column(i).as_primitive::<Int64Type>().clone();
            let cols = [col(0), col(1), col(2), col(4), col(5)];
            rows.extend((0..d.num_rows()).map(|r| cols.each_ref().map(|c| c.value(r))));
        }
        rows.sort_unstable();
        Ok(rows)
    }

    /// Drops the query's states; returns what the pool still holds.
    fn end(mut self) -> usize {
        self.g = None;
        self.coordinator = None;
        let root = Arc::clone(&self.root);
        drop(self);
        root.reserved()
    }
}

/// A gate for the first spill writer: it says so and waits for `release`.
fn writer_gate() -> (
    Arc<dyn Fn() + Send + Sync>,
    std::sync::mpsc::Receiver<()>,
    std::sync::mpsc::Sender<()>,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let gate = std::sync::Mutex::new((entered_tx, release_rx));
    let calls = AtomicUsize::new(0);
    let hook = Arc::new(move || {
        if calls.fetch_add(1, Ordering::SeqCst) == 0 {
            let g = gate.lock().unwrap();
            g.0.send(()).unwrap();
            let _ = g.1.recv();
        }
    });
    (hook, entered_rx, release_tx)
}

/// Waits (30 s at most) for `f`.
async fn within<T>(f: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(30), f)
        .await
        .expect("waited out the test's deadline")
}

async fn entered(rx: std::sync::mpsc::Receiver<()>) {
    within(tokio::task::spawn_blocking(move || rx.recv().unwrap()))
        .await
        .unwrap();
}

/// A `finalize` that starts while the combined tables' hook writes them
/// waits for the write, then reads the runs it wrote: every group.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finalize_waits_out_a_combined_tables_write() {
    let (hook, entered_rx, release) = writer_gate();
    let c = Arc::new(Combined::new(None, hook));
    let (ballast, asked) = c.ask(64 << 10);
    entered(entered_rx).await;
    let fin = c.task();
    let finalizing = {
        let c = Arc::clone(&c);
        tokio::spawn(async move { c.groups(fin).await })
    };
    // An early signal only: the groups read below are what show that
    // `finalize` waited.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!finalizing.is_finished(), "finalize ran during the write");
    release.send(()).unwrap();
    let wants = within(asked)
        .await
        .unwrap()
        .expect("the write freed nothing");
    drop((wants, ballast));
    assert_eq!(within(finalizing).await.unwrap().unwrap(), ROWS);
    let c = Arc::into_inner(c).unwrap();
    assert_eq!(c.end(), 0, "memory still held");
}

/// A query cancelled while the combined tables' hook writes them ends
/// with an empty pool.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cancel_during_a_combined_tables_write_leaves_the_pool_empty() {
    let (hook, entered_rx, release) = writer_gate();
    let c = Combined::new(None, hook);
    let (ballast, asked) = c.ask(64 << 10);
    entered(entered_rx).await;
    c.q.cancel();
    release.send(()).unwrap();
    let wants = tokio::time::timeout(Duration::from_secs(30), asked)
        .await
        .expect("the request outlived the cancel")
        .unwrap();
    drop((wants, ballast));
    assert!(
        c.groups(c.task()).await.is_err(),
        "a cancelled query finished"
    );
    // As the runner does at query end: the hook's write runs out first.
    within(c.q.blocking_done()).await;
    assert_eq!(c.end(), 0, "memory still held");
}

/// A hook write that fails (the spill cap refuses the file) fails the
/// query, and its `finalize` with it: no result short of the lost groups.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failing_combined_tables_write_fails_the_query() {
    let c = Combined::new(Some(1), Arc::new(|| {}));
    let (ballast, asked) = c.ask(64 << 10);
    let wants = tokio::time::timeout(Duration::from_secs(30), asked)
        .await
        .expect("the request waited out its deadline")
        .unwrap();
    drop((wants, ballast));
    // The write drops the tables it lost before the query is failed.
    tokio::time::timeout(Duration::from_secs(30), async {
        while c.q.check_cancelled().is_ok() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("the query did not fail");
    assert!(
        c.groups(c.task()).await.is_err(),
        "finalize read a short result"
    );
    assert_eq!(c.end(), 0, "memory still held");
}

/// Four combined tables (driver partitions 0, 2 and 3, then 1 combining
/// while the hook writes): the write finishes partition 0's table, the
/// pool refuses partition 2's scratch. The hook keeps partitions 2 and 3
/// (3 not yet tried) and puts them back beside partition 1's; every group
/// (key, count, sum) is that of a run that never spilled, and `first`/`last`
/// follow driver-partition order. (Not compared with the unspilled run:
/// its in-memory row-table merge orders the drivers' tables the other way,
/// a defect of its own, `ops/queue-opus.md`.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_write_refused_part_way_keeps_the_tables_it_did_not_write() {
    let tables = [(0, 0), (2, 20_000), (3, 30_000)];
    let unspilled = Combined::with_tables(None, Arc::new(|| {}), &tables);
    unspilled.combine(1, 10_000);
    let unordered = |rows: Vec<[i64; 5]>| {
        rows.into_iter()
            .map(|r| [r[0], r[1], r[2]])
            .collect::<Vec<_>>()
    };
    let expected = unordered(unspilled.rows(unspilled.task()).await.unwrap());
    assert_eq!(unspilled.end(), 0);
    // Extra writers (`parts.rs` helpers) may still start partition 2's run
    // if the request takes partition 0's freed bytes only after the run
    // began: their credit covers the scratch, nothing is refused, and the
    // try proves nothing (its result is still checked). Repeat until one
    // is refused.
    for _ in 0..20 {
        if refused_part_way(&tables, &expected, &unordered).await {
            return;
        }
    }
    panic!("no try refused partition 2's scratch");
}

/// One try of [`a_write_refused_part_way_keeps_the_tables_it_did_not_write`]:
/// whether the hook's write was refused part-way (partition 0's run only).
async fn refused_part_way(
    tables: &[(u32, i64)],
    expected: &[[i64; 3]],
    unordered: &dyn Fn(Vec<[i64; 5]>) -> Vec<[i64; 3]>,
) -> bool {
    use std::sync::atomic::{AtomicBool, Ordering};
    // The first writer after partition 0's run (the first writer of
    // partition 2's) stops at the gate.
    let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let gate = std::sync::Mutex::new((entered_tx, release_rx));
    let agg = Arc::new(std::sync::OnceLock::<std::sync::Weak<HashAggregate>>::new());
    let fired = AtomicBool::new(false);
    let hook = {
        let agg = Arc::clone(&agg);
        Arc::new(move || {
            let ran = agg
                .get()
                .and_then(std::sync::Weak::upgrade)
                .is_some_and(|a| a.metrics().spill_runs >= 1);
            if ran && !fired.swap(true, Ordering::SeqCst) {
                let g = gate.lock().unwrap();
                g.0.send(()).unwrap();
                let _ = g.1.recv();
            }
        })
    };
    let c = Combined::with_tables(None, hook, tables);
    agg.set(Arc::downgrade(&c.agg)).unwrap();
    let held = c.root.reserved();
    // About one table's bytes: partition 0's run funds the request, and
    // partition 2's starts with no room for extra writers.
    let (ballast, asked) = c.ask(held / tables.len() - (64 << 10));
    entered(entered_rx).await;
    // Partition 1 combines mid-write (with the ballast's room); then every
    // free byte, the scratch floor included, is taken, so the gated writer
    // is refused.
    drop(ballast);
    // The request is funded (partition 0's run or the ballast) before the
    // grab, so its round never sees the grabbed pool.
    let wants = within(asked)
        .await
        .unwrap()
        .expect("the request was not funded");
    c.combine(1, 10_000);
    let mut grab = ConsumerHandle::register(Arc::clone(c.q.pool()), "grab", None)
        .reservation()
        .for_spill_scratch();
    let mut n = 1usize << 40;
    while n > 0 {
        if grab.try_grow(n).is_err() {
            n /= 2;
        }
    }
    release_tx.send(()).unwrap();
    // The grab holds until the hook's write ended.
    within(c.q.blocking_done()).await;
    drop((wants, grab));
    let refused = c.agg.metrics().spill_runs == 1;
    if refused {
        // Three tables still in memory (`held` was three): partitions 2
        // and 3 kept, partition 1's; partition 2's lent headroom may not
        // be back.
        let in_memory = c.root.reserved();
        assert!(in_memory > held * 5 / 6, "{in_memory} of {held}");
    }
    let got = c.rows(c.task()).await.unwrap();
    // Driver-partition order: `first` from partition 0, `last` from 3.
    for r in &got {
        assert_eq!([r[3], r[4]], [r[0], 30_000 + r[0]], "{r:?}");
    }
    assert_eq!(unordered(got), expected);
    assert_eq!(c.end(), 0, "memory still held");
    refused
}
