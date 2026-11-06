// SPDX-License-Identifier: Apache-2.0
//! The disk phase of a spilled aggregate in parallel tasks (design
//! `P2-agg-parallel-disk-merge` option 1): tasks merged several at once
//! give the rows of one at a time (an order-sensitive `collect_list`
//! included), each writes its own output file so the result is read by
//! several drivers, a task refused memory while others run waits instead
//! of splitting beside them, and a cancel mid-phase leaves nothing behind.

use arrow_array::StringViewArray;

use super::*;
use crate::testing::BlockingGate;
use AggregateFunction::CollectList;

/// `count(*)`, `sum`, `min`, `avg` and `collect_list` of `x` by `k`.
fn disk_aggs(s: &SchemaRef) -> Vec<AggregateExpr> {
    let mut aggs = grouped_aggs(s);
    aggs.push(agg(&CollectList, 1, s));
    aggs
}

/// `n` rows `k = i % keys`, `x` a ~30-byte string of `i` (NULL every 7th),
/// in batches of 1000: wide groups, so a partition outgrows a small pool.
fn wide_input(n: i64, keys: i64) -> (SchemaRef, Vec<Vec<ArrayRef>>) {
    let s = schema(vec![("k", DataType::Int64), ("x", DataType::Utf8View)]);
    let batches = (0..n)
        .step_by(1000)
        .map(|start| {
            let r = start..(start + 1000).min(n);
            let k: Vec<_> = r.clone().map(|i| Some(i % keys)).collect();
            let x = StringViewArray::from_iter(
                r.map(|i| (i % 7 != 0).then(|| format!("value-{i:08}-padding-to-grow"))),
            );
            vec![i64s(&k), Arc::new(x) as ArrayRef]
        })
        .collect();
    (s, batches)
}

/// `count(*)` and `collect_list(x)` by `k` (the string input has no sum).
fn wide_aggs(s: &SchemaRef) -> Vec<AggregateExpr> {
    vec![AggregateExpr::count_star("c"), agg(&CollectList, 1, s)]
}

/// What one run by hand returned.
struct DiskRun {
    /// Rows as `a|b|c`, sorted by row; each `collect_list` keeps its order.
    rows: Vec<String>,
    /// The output source's parallelism once finalized.
    parallel: usize,
}

/// Runs `agg` by hand under a pool of `limit` bytes: `drivers` drivers sink
/// `b` in turn (spilling when refused memory) and combine, everything left
/// in memory is spilled (so the whole result goes through the disk phase),
/// then the aggregate finalizes on `executor` and one reader reads it.
/// Checks that nothing leaks.
async fn disk_by_hand(
    agg: &Arc<HashAggregate>,
    s: &SchemaRef,
    b: &[Vec<ArrayRef>],
    executor: Arc<dyn Executor>,
    limit: usize,
    rows_layout: bool,
) -> DiskRun {
    let root = pool(limit);
    let config = ExecConfig {
        agg_row_layout: rows_layout,
        ..ExecConfig::default()
    };
    let q = QueryContext::new(
        QueryId(12),
        WorkloadClass::Batch,
        config,
        root.child("q", Some(limit)),
    );
    let ctx = |id, name| {
        let e = Arc::clone(&executor);
        TaskContext::new(Arc::clone(&q), e, SpillOwner::new(), OperatorId(id), name)
    };
    let sink_ctx = ctx(0, "agg");
    let g = Sink::global_state(&**agg, &sink_ctx).unwrap();
    for part in b.chunks(b.len().div_ceil(3)) {
        let mut l = Sink::local_state(&**agg, &sink_ctx, &g).unwrap();
        for cols in part {
            let batch = Batch::new(RecordBatch::try_new(Arc::clone(s), cols.clone()).unwrap());
            while let crate::SinkStatus::NeedMemory { .. } =
                agg.sink(batch.clone(), &mut l, &g).unwrap()
            {
                let freed = agg.spill(&mut l, 0, &sink_ctx).await.unwrap();
                assert!(freed > 0, "nothing to spill under {limit} B");
            }
        }
        agg.combine(l, &g).unwrap();
    }
    // The tables still in memory go to runs too.
    agg.spill_global(&g, 1, &sink_ctx).await.unwrap();
    let final_ctx = ctx(0, "agg");
    agg.finalize(&g, &final_ctx).await.unwrap();
    let src = ctx(1, "src").with_dependency(Arc::clone(&g));
    let sg = Source::global_state(&**agg, &src).unwrap();
    let parallel = agg.max_parallelism(&sg);
    let mut sl = Source::local_state(&**agg, &src, &sg).unwrap();
    let mut out = Vec::new();
    while let Some(b) = agg.get_data(&mut sl, &sg).await.unwrap() {
        out.push(b);
    }
    drop((sl, sg, src, g, sink_ctx, final_ctx));
    assert_eq!(root.reserved(), 0, "memory left after the query");
    DiskRun {
        rows: rows(&out),
        parallel,
    }
}

/// A row without its last column (the `collect_list`).
fn without_list(row: &str) -> &str {
    row.rsplit_once('|').map_or(row, |(head, _)| head)
}

/// Disk tasks merged several at once give exactly the rows of one at a
/// time: runs of one partition still merge in partition order inside a
/// task, so even `collect_list` keeps its order; only the groups' order
/// across partitions changes. Each task wrote its own output file, so the
/// result is read by several drivers.
#[tokio::test]
async fn disk_tasks_in_parallel_equal_one_at_a_time() {
    let (s, b) = grouped_input(120_000, 40_000);
    let expected = run_rows(single(&s, &b)).await;
    for layout in [false, true] {
        let run = |slots: usize, limit: usize| {
            let (s, b) = (Arc::clone(&s), b.clone());
            async move {
                let (_m, scope) = spill_dir();
                let mut sp = spec(&s, vec![0], disk_aggs(&s));
                sp.spill = Some(scope);
                let agg = HashAggregate::new(sp).unwrap();
                let executor: Arc<dyn Executor> = TokioExecutor::deterministic(slots).unwrap();
                let got = disk_by_hand(&agg, &s, &b, executor, limit, layout).await;
                (got, agg)
            }
        };
        for limit in [8 << 20, 3 << 20] {
            let (one, _) = run(1, limit).await;
            let (four, agg) = run(4, limit).await;
            assert_eq!(one.rows.len(), 40_000, "layout {layout}, {limit} B");
            assert_eq!(four.rows, one.rows, "layout {layout}, {limit} B");
            let plain: Vec<&str> = four.rows.iter().map(|r| without_list(r)).collect();
            assert_eq!(plain, expected, "layout {layout}, {limit} B");
            assert!(agg.metrics().spill_runs > 0);
            assert!(sink_metric(&agg, "disk_tasks") > 1, "layout {layout}");
            assert!(sink_metric(&agg, "disk_peak_tasks") > 1, "layout {layout}");
            assert!(four.parallel > 1, "{} readers", four.parallel);
            assert!(one.parallel > 1, "{} readers", one.parallel);
        }
    }
}

/// Partitions that do not fit beside each other under a tiny pool: a task
/// refused memory while others run waits for one of them to finish; only
/// a task alone splits, so no split ever runs beside another task or
/// split (the blocking pool has real threads here). The rows are those of
/// the in-memory aggregate.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refused_task_waits_for_the_others_before_it_splits() {
    let (s, b) = wide_input(80_000, 20_000);
    let canonical = |rows: &[String]| {
        let mut v: Vec<String> = rows.iter().map(|r| without_list(r).to_owned()).collect();
        v.sort();
        v
    };
    // Every key has four rows.
    let mut expected: Vec<String> = (0..20_000).map(|k| format!("{k}|4")).collect();
    expected.sort();
    let mut waits = 0;
    let mut splits = 0;
    for layout in [false, true] {
        for limit in [3 << 20, 2 << 20] {
            let (_m, scope) = spill_dir();
            let mut sp = spec(&s, vec![0], wide_aggs(&s));
            sp.spill = Some(scope);
            let agg = HashAggregate::new(sp).unwrap();
            let executor: Arc<dyn Executor> = TokioExecutor::new(&crate::ExecutorConfig {
                compute_threads: 4,
                io_threads: 1,
                blocking_threads: 4,
                io_blocking_threads: 2,
            })
            .unwrap();
            let got = disk_by_hand(&agg, &s, &b, executor, limit, layout).await;
            assert_eq!(canonical(&got.rows), expected, "layout {layout}, {limit} B");
            assert_eq!(agg.disk_split_overlaps(), 0, "layout {layout}, {limit} B");
            waits += sink_metric(&agg, "disk_waits");
            splits += agg.metrics().repartitions;
        }
    }
    // The case under test happened: tasks were refused beside others, and
    // partitions were split.
    assert!(waits > 0, "no task waited");
    assert!(splits > 0, "no partition split");
}

/// A query cancelled while its aggregate's disk tasks run ends only once
/// the tasks have let go of their memory and files: the pool is empty and
/// no spill file is left.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cancel_during_the_disk_phase_releases_everything() {
    let (spill, scope) = spill_dir();
    let (s, b) = grouped_input(100_000, 75_000);
    let mut sp = spec(&s, vec![0], disk_aggs(&s));
    sp.spill = Some(scope);
    let gate = BlockingGate::new();
    let agg = HashAggregate::with_disk_gate(sp, Arc::clone(&gate)).unwrap();
    let plan = AggregateNode::new(scan(&s, b), agg);
    let executor = TokioExecutor::new(&crate::ExecutorConfig {
        compute_threads: 2,
        io_threads: 1,
        blocking_threads: 2,
        io_blocking_threads: 2,
    })
    .unwrap();
    let runner = QueryRunner::new(executor);
    let limit = 3 << 20;
    let root = pool(limit);
    let q = QueryContext::new(
        QueryId(13),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("q", Some(limit)),
    );
    let mut stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
    let drained = tokio::spawn(async move {
        let mut rows = 0;
        while let Some(b) = stream.next().await {
            rows += b.map_or(0, |b| b.num_rows());
        }
        rows
    });
    while !gate.reached() {
        tokio::task::yield_now().await;
    }
    assert!(root.reserved() > 0);
    q.cancel();
    let (fq, froot) = (Arc::clone(&q), Arc::clone(&root));
    let at_end = tokio::spawn(async move {
        fq.finished().await;
        froot.reserved()
    });
    // The query must not end while a task still holds its table.
    for _ in 0..200 {
        if at_end.is_finished() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert!(
        !at_end.is_finished(),
        "the query ended beside a running task"
    );
    gate.open();
    assert_eq!(at_end.await.unwrap(), 0, "memory left when the query ended");
    assert_eq!(root.reserved(), 0, "memory left after the query");
    assert_eq!(drained.await.unwrap(), 0);
    assert_eq!(spill.manager.used_bytes(), 0, "spill files left");
    assert_eq!(
        q.error().map(|e| e.error_class()),
        Some(ErrorClass::OperationCanceled)
    );
}

/// A spill request that arrives while disk tasks run (review T1): the
/// finalizing driver takes it between task completions, the phase still
/// ends with the rows of the in-memory aggregate, and nothing leaks.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_spill_request_during_the_disk_phase_is_taken_between_tasks() {
    let (s, b) = grouped_input(100_000, 40_000);
    // The reference runs on a current-thread runtime of its own, as the
    // other tests' do.
    let expected = {
        let (s, b) = (Arc::clone(&s), b.clone());
        std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(run_rows(single(&s, &b)))
        })
        .join()
        .unwrap()
    };
    let (_m, scope) = spill_dir();
    let mut sp = spec(&s, vec![0], disk_aggs(&s));
    sp.spill = Some(scope);
    let gate = BlockingGate::new();
    let agg = HashAggregate::with_disk_gate(sp, Arc::clone(&gate)).unwrap();
    let executor: Arc<dyn Executor> = TokioExecutor::new(&crate::ExecutorConfig {
        compute_threads: 4,
        io_threads: 1,
        blocking_threads: 4,
        io_blocking_threads: 2,
    })
    .unwrap();
    let limit = 8 << 20;
    let root = pool(limit);
    let q = QueryContext::new(
        QueryId(14),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("q", Some(limit)),
    );
    let ctx = |id, name| {
        let e = Arc::clone(&executor);
        TaskContext::new(Arc::clone(&q), e, SpillOwner::new(), OperatorId(id), name)
    };
    let sink_ctx = ctx(0, "agg");
    let g = Sink::global_state(&*agg, &sink_ctx).unwrap();
    for part in b.chunks(b.len().div_ceil(3)) {
        let mut l = Sink::local_state(&*agg, &sink_ctx, &g).unwrap();
        for cols in part {
            let batch = Batch::new(RecordBatch::try_new(Arc::clone(&s), cols.clone()).unwrap());
            while let crate::SinkStatus::NeedMemory { .. } =
                agg.sink(batch.clone(), &mut l, &g).unwrap()
            {
                agg.spill(&mut l, 0, &sink_ctx).await.unwrap();
            }
        }
        agg.combine(l, &g).unwrap();
    }
    agg.spill_global(&g, 1, &sink_ctx).await.unwrap();
    let final_ctx = ctx(0, "agg");
    let asked_ctx = ctx(0, "agg");
    let (finalized, asked) = tokio::join!(agg.finalize(&g, &final_ctx), async {
        while !gate.reached() {
            tokio::task::yield_now().await;
        }
        // A task holds its table at the gate: ask the aggregate to free
        // memory, then let the tasks go on.
        let freed = agg.spill_global(&g, 1, &asked_ctx).await;
        gate.open();
        freed
    });
    finalized.unwrap();
    asked.unwrap();
    assert!(sink_metric(&agg, "disk_tasks") > 1);
    let src = ctx(1, "src").with_dependency(Arc::clone(&g));
    let sg = Source::global_state(&*agg, &src).unwrap();
    let mut sl = Source::local_state(&*agg, &src, &sg).unwrap();
    let mut out = Vec::new();
    while let Some(b) = agg.get_data(&mut sl, &sg).await.unwrap() {
        out.push(b);
    }
    let got: Vec<String> = rows(&out)
        .iter()
        .map(|r| without_list(r).to_owned())
        .collect();
    assert_eq!(got, expected);
    drop((sl, sg, src, g, sink_ctx, final_ctx, asked_ctx));
    assert_eq!(root.reserved(), 0, "memory left after the query");
}

/// Groups the final phase flushed before it spilled stay the first output
/// file beside the disk tasks' own files (review T2): after the first
/// task no task gets memory, so its groups are flushed and the rest go
/// through the disk phase. Every group comes out once, with its values.
#[tokio::test]
async fn flushed_groups_stay_beside_the_disk_tasks_files() {
    let (s, b) = grouped_input(200_000, 90_000);
    let expected = run_rows(single(&s, &b)).await;
    let (_m, scope) = spill_dir();
    let mut sp = spec(&s, vec![0], grouped_aggs(&s));
    sp.spill = Some(scope);
    let agg =
        HashAggregate::with_final_stall(sp, rowtable::Mode::Fixed { limit: 5_000 }, 1).unwrap();
    let got = by_hand(&agg, &s, &b, 3).await;
    assert_eq!(got.rows, expected);
    assert_eq!(sink_metric(&agg, "row_final_flushes"), 1);
    assert_eq!(sink_metric(&agg, "row_final_spills"), 1);
    assert!(sink_metric(&agg, "disk_tasks") > 0, "no disk task ran");
    assert_eq!(agg.metrics().groups, 90_000);
}
