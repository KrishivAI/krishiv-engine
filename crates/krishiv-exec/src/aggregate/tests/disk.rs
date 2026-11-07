// SPDX-License-Identifier: Apache-2.0
//! The disk phase of a spilled aggregate in parallel tasks (design
//! `P2-agg-parallel-disk-merge` option 1): tasks merged several at once
//! give the rows of one at a time (an order-sensitive `collect_list`
//! included), each writes its own output file so the result is read by
//! several drivers, a task refused memory while others run waits instead
//! of splitting beside them, and a cancel mid-phase leaves nothing behind.

use arrow_array::StringViewArray;
use krishiv_memory::{Reservation, Spillable};

use super::*;
use crate::testing::BlockingGate;
use AggregateFunction::CollectList;

/// How long any one wait of these tests may take: a regression fails the
/// test instead of leaving it spinning.
const PATIENCE: Duration = Duration::from_secs(60);

/// Spills a by-hand driver makes for one batch before the test fails.
const MAX_SPILLS: usize = 1000;

/// Awaits `fut`, failing the test after [`PATIENCE`].
async fn bounded<F: Future>(what: &str, fut: F) -> F::Output {
    match tokio::time::timeout(PATIENCE, fut).await {
        Ok(v) => v,
        Err(_) => panic!("timed out: {what}"),
    }
}

/// Polls `ready` every millisecond (sleeping, never spinning), failing the
/// test after [`PATIENCE`].
async fn wait_until(what: &str, ready: impl Fn() -> bool) {
    bounded(what, async {
        while !ready() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await;
}

/// Opens its gates when dropped, so a failing test never leaves a
/// blocking-pool thread parked at one.
struct OpenOnDrop(Vec<Arc<BlockingGate>>);

impl Drop for OpenOnDrop {
    fn drop(&mut self) {
        for g in &self.0 {
            g.open();
        }
    }
}

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
            let mut spills = 0;
            while let crate::SinkStatus::NeedMemory { .. } =
                agg.sink(batch.clone(), &mut l, &g).unwrap()
            {
                spills += 1;
                assert!(spills <= MAX_SPILLS, "a batch never fits under {limit} B");
                let freed = bounded("spill", agg.spill(&mut l, 0, &sink_ctx))
                    .await
                    .unwrap();
                assert!(freed > 0, "nothing to spill under {limit} B");
            }
        }
        agg.combine(l, &g).unwrap();
    }
    // The tables still in memory go to runs too.
    bounded("spill_global", agg.spill_global(&g, 1, &sink_ctx))
        .await
        .unwrap();
    let final_ctx = ctx(0, "agg");
    bounded("finalize", agg.finalize(&g, &final_ctx))
        .await
        .unwrap();
    let src = ctx(1, "src").with_dependency(Arc::clone(&g));
    let sg = Source::global_state(&**agg, &src).unwrap();
    let parallel = agg.max_parallelism(&sg);
    let mut sl = Source::local_state(&**agg, &src, &sg).unwrap();
    let mut out = Vec::new();
    while let Some(b) = bounded("get_data", agg.get_data(&mut sl, &sg))
        .await
        .unwrap()
    {
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
            waits += sink_metric(&agg, "disk_stalls");
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
    // After the executor: dropped first, so no blocking thread stays
    // parked at a gate while the executor shuts down.
    let _open = OpenOnDrop(vec![Arc::clone(&gate)]);
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
    wait_until("a disk task at the gate", || gate.reached()).await;
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
    assert_eq!(
        bounded("the query's end", at_end).await.unwrap(),
        0,
        "memory left when the query ended"
    );
    assert_eq!(root.reserved(), 0, "memory left after the query");
    assert_eq!(bounded("the stream", drained).await.unwrap(), 0);
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
    // After the executor: dropped first, so no blocking thread stays
    // parked at a gate while the executor shuts down.
    let _open = OpenOnDrop(vec![Arc::clone(&gate)]);
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
    sink_to_runs(&agg, &s, &b, &sink_ctx, &g).await;
    let final_ctx = ctx(0, "agg");
    let asked_ctx = ctx(0, "agg");
    let (finalized, asked) = bounded("the disk phase", async {
        tokio::join!(agg.finalize(&g, &final_ctx), async {
            wait_until("a disk task at the gate", || gate.reached()).await;
            // A task holds its table at the gate: ask the aggregate to free
            // memory, then let the tasks go on.
            let freed = agg.spill_global(&g, 1, &asked_ctx).await;
            gate.open();
            freed
        })
    })
    .await;
    finalized.unwrap();
    asked.unwrap();
    assert!(sink_metric(&agg, "disk_tasks") > 1);
    let src = ctx(1, "src").with_dependency(Arc::clone(&g));
    let out = read_all(&agg, &src).await;
    let got: Vec<String> = rows(&out)
        .iter()
        .map(|r| without_list(r).to_owned())
        .collect();
    assert_eq!(got, expected);
    drop((src, g, sink_ctx, final_ctx, asked_ctx));
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

/// Sinks `b` by hand in three drivers (each spilling when refused memory)
/// and combines them, then spills every table left to runs: the whole
/// result goes through the disk phase.
async fn sink_to_runs(
    agg: &HashAggregate,
    s: &SchemaRef,
    b: &[Vec<ArrayRef>],
    ctx: &TaskContext,
    g: &crate::GlobalState,
) {
    for part in b.chunks(b.len().div_ceil(3)) {
        let mut l = Sink::local_state(agg, ctx, g).unwrap();
        for cols in part {
            let batch = Batch::new(RecordBatch::try_new(Arc::clone(s), cols.clone()).unwrap());
            let mut spills = 0;
            while let crate::SinkStatus::NeedMemory { .. } =
                agg.sink(batch.clone(), &mut l, g).unwrap()
            {
                spills += 1;
                assert!(spills <= MAX_SPILLS, "a batch never fits");
                bounded("spill", agg.spill(&mut l, 0, ctx)).await.unwrap();
            }
        }
        agg.combine(l, g).unwrap();
    }
    bounded("spill_global", agg.spill_global(g, 1, ctx))
        .await
        .unwrap();
}

/// Reads a finalized aggregate's result through one source driver.
async fn read_all(agg: &Arc<HashAggregate>, src: &TaskContext) -> Vec<Batch> {
    let sg = Source::global_state(&**agg, src).unwrap();
    let mut sl = Source::local_state(&**agg, src, &sg).unwrap();
    let mut out = Vec::new();
    while let Some(b) = bounded("get_data", agg.get_data(&mut sl, &sg))
        .await
        .unwrap()
    {
        out.push(b);
    }
    out
}

/// Grows `r` by all the pool grants it, trying from `free` bytes down.
fn take_all(r: &mut Reservation, free: usize) {
    // Bounded: every grant consumes `step` bytes of a finite pool and
    // every refusal halves `step`.
    let mut step = free;
    while step > 0 {
        if r.try_grow(step).is_err() {
            step /= 2;
        }
    }
}

/// The rows of `run_rows(single(..))`, computed on a current-thread
/// runtime of its own (as the other tests' references are).
fn reference(s: &SchemaRef, b: &[Vec<ArrayRef>]) -> Vec<String> {
    let (s, b) = (Arc::clone(s), b.to_vec());
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(run_rows(single(&s, &b)))
    })
    .join()
    .unwrap()
}

/// A spill request taken while the only task left waits with all its input
/// merged (review M2): the task writes its groups once its write fits,
/// rather than being split, which would write its table out and read it
/// back. Here the task's write is refused while another consumer holds
/// the pool; that consumer lets go before the request is taken, so the
/// write fits then. Tasks run one at a time (`deterministicOrder`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_requested_merged_task_writes_instead_of_splitting() {
    let (s, b) = grouped_input(100_000, 40_000);
    let expected = reference(&s, &b);
    let (spill, scope) = spill_dir();
    let mut sp = spec(&s, vec![0], disk_aggs(&s));
    sp.spill = Some(scope);
    let (merged, done) = (BlockingGate::new(), BlockingGate::new());
    let agg = HashAggregate::with_disk_gates(sp, Arc::clone(&merged), Arc::clone(&done)).unwrap();
    let executor: Arc<dyn Executor> = TokioExecutor::new(&crate::ExecutorConfig {
        compute_threads: 4,
        io_threads: 1,
        blocking_threads: 4,
        io_blocking_threads: 2,
    })
    .unwrap();
    // After the executor: dropped first, so no blocking thread stays
    // parked at a gate while the executor shuts down.
    let _open = OpenOnDrop(vec![Arc::clone(&merged), Arc::clone(&done)]);
    let limit = 8 << 20;
    let root = pool(limit);
    let config = ExecConfig {
        deterministic_order: true,
        ..ExecConfig::default()
    };
    let q = QueryContext::new(
        QueryId(15),
        WorkloadClass::Batch,
        config,
        root.child("q", Some(limit)),
    );
    let ctx = |id, name| {
        let e = Arc::clone(&executor);
        TaskContext::new(Arc::clone(&q), e, SpillOwner::new(), OperatorId(id), name)
    };
    let sink_ctx = ctx(0, "agg");
    let g = Sink::global_state(&*agg, &sink_ctx).unwrap();
    sink_to_runs(&agg, &s, &b, &sink_ctx, &g).await;
    let final_ctx = ctx(0, "agg");
    let asked_ctx = ctx(0, "agg");
    let hog_ctx = ctx(1, "hog");
    let mut hog = hog_ctx.consumer().reservation();
    let (finalized, asked) = bounded("the disk phase", async {
        tokio::join!(agg.finalize(&g, &final_ctx), async {
            wait_until("a merged task", || merged.reached()).await;
            // The first task is merged and holds its table: another
            // consumer takes what is left of the pool and the aggregate is
            // asked to free memory; then the task goes on to its (refused)
            // write.
            take_all(&mut hog, limit - root.reserved());
            let freed = agg.spill_global(&g, 1, &asked_ctx).await;
            merged.open();
            wait_until("the task's job to end", || done.reached()).await;
            // Stalled on its write, alone: the memory is back before the
            // driver takes the request.
            hog.shrink(hog.size());
            done.open();
            freed
        })
    })
    .await;
    finalized.unwrap();
    // The merged task's branch writes its table; it frees nothing itself.
    assert_eq!(asked.unwrap(), 0);
    assert_eq!(agg.disk_merged_runs(), 1, "the merged task did not run");
    assert_eq!(agg.metrics().repartitions, 0, "a partition was split");
    assert_eq!(sink_metric(&agg, "disk_stalls"), 0, "stalled beside others");
    assert!(sink_metric(&agg, "disk_tasks") > 1);
    let src = ctx(2, "src").with_dependency(Arc::clone(&g));
    let out = read_all(&agg, &src).await;
    let got: Vec<String> = rows(&out)
        .iter()
        .map(|r| without_list(r).to_owned())
        .collect();
    assert_eq!(got, expected);
    drop((src, g, sink_ctx, final_ctx, asked_ctx, hog, hog_ctx));
    assert_eq!(root.reserved(), 0, "memory left after the query");
    assert_eq!(spill.manager.used_bytes(), 0, "spill files left");
}

/// A consumer standing in for another operator of the query, with a spill
/// hook the test services by hand.
fn spillable_hog(q: &Arc<QueryContext>) -> (Arc<krishiv_memory::SpillSignal>, Reservation) {
    let signal = Arc::new(krishiv_memory::SpillSignal::new());
    let hook: Arc<dyn Spillable> = Arc::clone(&signal) as _;
    let handle = krishiv_memory::ConsumerHandle::register(Arc::clone(q.pool()), "hog", Some(hook));
    (signal, handle.reservation())
}

/// The disk phase of a global aggregate (review T5): its one task holds a
/// single group, so it can never split; refused memory alone, it waits
/// through arbitration. When another consumer frees memory on request, the
/// task goes on and the one row comes out. When none can, the arbitrator
/// (`Escalation::Abort`, no victim to spill) aborts the requester's own
/// query and the task's `grow` fails with `UNABLE_TO_ACQUIRE_MEMORY`: the
/// failure is the arbitrator's policy; the phase's part is that it neither
/// splits nor writes a merged run. Either way nothing leaks and no spill
/// request is left pending.
#[tokio::test]
async fn a_lone_unsplittable_task_waits_for_the_arbitrator() {
    let (s, b) = wide_input(30_000, 1);
    let aggs = wide_aggs(&s);
    for spillable in [true, false] {
        let (spill, scope) = spill_dir();
        let mut sp = spec(&s, vec![], aggs.clone());
        sp.spill = Some(scope);
        let agg = HashAggregate::new(sp).unwrap();
        let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
        let limit = 8 << 20;
        let root = pool(limit);
        let q = QueryContext::new(
            QueryId(16),
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
        sink_to_runs(&agg, &s, &b, &sink_ctx, &g).await;
        assert!(agg.metrics().spill_runs > 0, "nothing went to disk");
        // Another consumer holds the whole pool from now on.
        let (signal, mut hog) = if spillable {
            spillable_hog(&q)
        } else {
            let h = krishiv_memory::ConsumerHandle::register(Arc::clone(q.pool()), "hog", None);
            (
                Arc::new(krishiv_memory::SpillSignal::new()),
                h.reservation(),
            )
        };
        take_all(&mut hog, limit - root.reserved());
        signal.set_spillable_bytes(hog.size());
        let final_ctx = ctx(0, "agg");
        let mut asked = false;
        let (finalized, ()) = bounded("the disk phase", async {
            tokio::join!(agg.finalize(&g, &final_ctx), async {
                if !spillable {
                    return;
                }
                // The task, refused alone, asks the arbitrator, which asks
                // the hog: it frees everything.
                wait_until("a spill request to the hog", || {
                    signal.spill_requested().is_some()
                })
                .await;
                asked = true;
                let freed = hog.size();
                hog.shrink(freed);
                signal.set_spillable_bytes(0);
                if let Some(service) = signal.begin() {
                    service.complete(freed);
                }
            })
        })
        .await;
        assert_eq!(agg.metrics().repartitions, 0, "spillable {spillable}");
        if spillable {
            finalized.unwrap();
            assert!(asked, "the arbitrator asked nobody");
            let src = ctx(2, "src").with_dependency(Arc::clone(&g));
            let out = read_all(&agg, &src).await;
            // Every input row in the one group.
            let got = rows(&out);
            assert_eq!(got.len(), 1);
            assert!(got[0].starts_with("30000|"), "{}", &got[0][..20]);
            drop(src);
        } else {
            let e = finalized.unwrap_err();
            assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory, "{e}");
            assert_eq!(agg.disk_merged_runs(), 0, "a merged run was written");
            // The request the phase took is not left behind: a later one
            // returns at once, freeing nothing.
            let later = bounded("a later request", agg.spill_global(&g, 1, &sink_ctx)).await;
            assert_eq!(later.unwrap(), 0);
        }
        drop((g, sink_ctx, final_ctx, hog));
        assert_eq!(root.reserved(), 0, "memory left, spillable {spillable}");
        assert_eq!(spill.manager.used_bytes(), 0, "spill files left");
    }
}

/// A disk task that fails (review T4; here an input file it reads is
/// corrupt) fails the query with its own error, not a cancellation, and the
/// query ends once every task let go: the pool is empty and no spill file
/// is left.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_task_error_fails_the_query_and_frees_everything() {
    let (spill, scope) = spill_dir();
    let dir = scope.dir().to_owned();
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
    // After the executor: dropped first, so no blocking thread stays
    // parked at a gate while the executor shuts down.
    let _open = OpenOnDrop(vec![Arc::clone(&gate)]);
    let limit = 3 << 20;
    let root = pool(limit);
    let q = QueryContext::new(
        QueryId(17),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("q", Some(limit)),
    );
    let mut stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
    let drained = tokio::spawn(async move {
        let mut errors = Vec::new();
        while let Some(b) = stream.next().await {
            if let Err(e) = b {
                errors.push(e.error_class());
            }
        }
        errors
    });
    wait_until("a disk task at the gate", || gate.reached()).await;
    // A task holds its table at the gate: every spill file is damaged, so
    // the next task to open one fails.
    assert!(corrupt_files(&dir) > 0, "no spill file to damage");
    gate.open();
    bounded("the query's end", q.finished()).await;
    assert_eq!(root.reserved(), 0, "memory left when the query ended");
    assert_eq!(
        q.error().map(|e| e.error_class()),
        Some(ErrorClass::KrishivSpillFileCorrupt)
    );
    assert_eq!(
        bounded("the stream", drained).await.unwrap(),
        vec![ErrorClass::KrishivSpillFileCorrupt]
    );
    assert_eq!(spill.manager.used_bytes(), 0, "spill files left");
}

/// Overwrites the first bytes (the magic) of every file under `dir`;
/// returns how many. That includes the `.spill.tmp` of a writer still open
/// on the other blocking thread: it is never finished or read, so the
/// asserted error is still the next *input* file's open.
fn corrupt_files(dir: &std::path::Path) -> usize {
    use std::io::Write;
    let mut n = 0;
    let mut dirs = vec![dir.to_owned()];
    while let Some(d) = dirs.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let path = e.path();
            if path.is_dir() {
                dirs.push(path);
            } else if let Ok(mut f) = std::fs::OpenOptions::new().write(true).open(&path) {
                f.write_all(b"XXXX").unwrap();
                n += 1;
            }
        }
    }
    n
}

/// Fills `agg` from three drivers under a pool large enough that nothing
/// spills while sinking, so the finalize state holds three tables.
async fn three_tables(
    agg: &Arc<HashAggregate>,
    q: &Arc<QueryContext>,
    executor: &Arc<dyn Executor>,
    s: &SchemaRef,
    b: &[Vec<ArrayRef>],
) -> (crate::operator::GlobalState, TaskContext) {
    let sink_ctx = TaskContext::new(
        Arc::clone(q),
        Arc::clone(executor),
        SpillOwner::new(),
        OperatorId(0),
        "agg",
    );
    let g = Sink::global_state(&**agg, &sink_ctx).unwrap();
    for part in b.chunks(b.len().div_ceil(3)) {
        let mut l = Sink::local_state(&**agg, &sink_ctx, &g).unwrap();
        for cols in part {
            let batch = Batch::new(RecordBatch::try_new(Arc::clone(s), cols.clone()).unwrap());
            assert!(matches!(
                agg.sink(batch, &mut l, &g).unwrap(),
                crate::SinkStatus::NeedMoreInput
            ));
        }
        agg.combine(l, &g).unwrap();
    }
    (g, sink_ctx)
}

fn real_threads() -> Arc<dyn Executor> {
    TokioExecutor::new(&crate::ExecutorConfig {
        compute_threads: 4,
        io_threads: 1,
        blocking_threads: 4,
        io_blocking_threads: 2,
    })
    .unwrap()
}

/// A spill request on the combined finalize state writes the drivers'
/// tables at once (design X2): two writers wait at the gate together,
/// which one writer at a time never does. The result is the in-memory one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_finalize_tables_spill_at_once() {
    let (_m, scope) = spill_dir();
    let (s, b) = grouped_input(60_000, 20_000);
    // The reference runs on the deterministic executor, which needs a
    // current-thread runtime of its own (this test has real threads).
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
    let mut sp = spec(&s, vec![0], disk_aggs(&s));
    sp.spill = Some(scope);
    let gate = BlockingGate::new();
    let agg = HashAggregate::with_free_gate(sp, Arc::clone(&gate)).unwrap();
    let executor = real_threads();
    let _open = OpenOnDrop(vec![Arc::clone(&gate)]);
    let limit = 256 << 20;
    let root = pool(limit);
    let q = QueryContext::new(
        QueryId(21),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("q", Some(limit)),
    );
    let (g, sink_ctx) = three_tables(&agg, &q, &executor, &s, &b).await;
    let (freed, ()) = tokio::join!(agg.spill_global(&g, 1, &sink_ctx), async {
        wait_until("two writers at the gate", || agg.spill_writers_peak() >= 2).await;
        gate.open();
    });
    assert!(freed.unwrap() > 0, "the tables were not written");
    assert!(sink_metric(&agg, "spill_writers_peak") >= 2);
    let ctx = |id, name| {
        TaskContext::new(
            Arc::clone(&q),
            Arc::clone(&executor),
            SpillOwner::new(),
            OperatorId(id),
            name,
        )
    };
    let final_ctx = ctx(0, "agg");
    bounded("finalize", agg.finalize(&g, &final_ctx))
        .await
        .unwrap();
    let src = ctx(1, "src").with_dependency(Arc::clone(&g));
    let out = read_all(&agg, &src).await;
    let mut got: Vec<String> = rows(&out)
        .iter()
        .map(|r| without_list(r).to_owned())
        .collect();
    got.sort();
    let mut expected = expected;
    expected.sort();
    assert_eq!(got, expected);
    drop((src, final_ctx, g, sink_ctx));
    assert_eq!(root.reserved(), 0, "memory left after the query");
}

/// A writer that fails (the spill directory turned read-only while the
/// writers wait at the gate) fails the spill with its error; every writer
/// is awaited, and once the state is dropped no spill file and no memory
/// is left.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failed_finalize_write_fails_the_spill_and_frees_everything() {
    use std::os::unix::fs::PermissionsExt as _;
    let (spill, scope) = spill_dir();
    let dir = scope.dir().to_owned();
    let (s, b) = grouped_input(60_000, 20_000);
    let mut sp = spec(&s, vec![0], disk_aggs(&s));
    sp.spill = Some(scope);
    let gate = BlockingGate::new();
    let agg = HashAggregate::with_free_gate(sp, Arc::clone(&gate)).unwrap();
    let executor = real_threads();
    let _open = OpenOnDrop(vec![Arc::clone(&gate)]);
    let limit = 256 << 20;
    let root = pool(limit);
    let q = QueryContext::new(
        QueryId(22),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("q", Some(limit)),
    );
    let (g, sink_ctx) = three_tables(&agg, &q, &executor, &s, &b).await;
    std::fs::create_dir_all(&dir).unwrap();
    let (freed, ()) = tokio::join!(agg.spill_global(&g, 1, &sink_ctx), async {
        wait_until("a writer at the gate", || gate.reached()).await;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o500)).unwrap();
        gate.open();
    });
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(
        freed.is_err(),
        "the spill succeeded without a writable directory"
    );
    assert_eq!(agg.metrics().spill_runs, 0, "a run was kept");
    drop((g, sink_ctx));
    assert_eq!(root.reserved(), 0, "memory left after the failed spill");
    assert_eq!(spill.manager.used_bytes(), 0, "spill files left");
}
