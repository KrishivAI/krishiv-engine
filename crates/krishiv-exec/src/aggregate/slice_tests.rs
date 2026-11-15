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
use crate::operator::SinkStatus;
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
fn batch(view: bool, first: i64, keys: impl Iterator<Item = i64>) -> Batch {
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

fn spec(view: bool) -> AggregateSpec {
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
        legacy_statistical: false,
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
