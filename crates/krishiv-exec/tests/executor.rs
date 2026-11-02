// SPDX-License-Identifier: Apache-2.0
//! The morsel executor end to end (P2-T050 acceptance): throughput and
//! exactness, cancellation, spill at batch boundaries, blocking-pool
//! isolation, limit, backpressure, error and panic isolation, the memory
//! contract and deterministic replay. Every test checks that the query's
//! memory is fully released when it ends.
#![cfg(not(loom))]
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]
#![expect(
    clippy::unwrap_in_result,
    reason = "test operators unwrap their own state; a failure is a test bug"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_schema::SchemaRef;
use futures::StreamExt;
use krishiv_common::config::Config;
use krishiv_common::error::{DivideByZero, ErrorClass, KrishivError};
use krishiv_exec::testing::{
    BlockingGate, Break, Buffer, Count, Filter, Limit, Pipe, RangeSource, Scan, schema,
};
use krishiv_exec::{
    DEFAULT_BATCH_SIZE, ExecConfig, ExecNode, Executor, ExecutorConfig, FlushOutput, GlobalState,
    LocalState, Operator, OperatorOutput, PipelineBuilder, QueryContext, QueryId, QueryRunner,
    Result, Scheduler, Sink, SinkStatus, Source, TaskContext, TokioClock, TokioExecutor,
    WorkloadClass,
};
use krishiv_io::FilterId;
use krishiv_memory::{BoxFuture, MemoryConfig, MemoryError, MemoryPool, Spillable, TreePool};
use krishiv_types::batch::Batch;

const MIB: usize = 1 << 20;

fn root(limit: usize) -> Arc<TreePool> {
    TreePool::with_clock(
        &MemoryConfig {
            limit: Some(limit),
            quantum: 64 << 10,
            spill_wait_timeout: Duration::from_secs(2),
            abort_release_timeout: Duration::from_secs(30),
            spill_scratch_reserve: krishiv_memory::DEFAULT_SPILL_SCRATCH_RESERVE,
            resident_guard: true,
        },
        Arc::new(TokioClock),
    )
}

/// Every test runs with the stall detector failing the query, with a
/// threshold generous enough for unoptimised and instrumented builds; a
/// test that sets its own threshold keeps it.
fn query(root: &TreePool, id: u64, mut config: ExecConfig) -> Arc<QueryContext> {
    if !config.fail_on_stall {
        config.fail_on_stall = true;
        config.stall_threshold = Duration::from_millis(500);
    }
    QueryContext::new(
        QueryId(id),
        WorkloadClass::Interactive,
        config,
        root.child(&format!("q{id}"), None),
    )
}

fn executor(threads: usize) -> Arc<TokioExecutor> {
    TokioExecutor::new(&ExecutorConfig {
        compute_threads: threads,
        io_threads: 1,
        blocking_threads: threads,
        io_blocking_threads: 2,
    })
    .unwrap()
}

/// Runs `plan` to the end: every batch, or the query's error. Returns once
/// every task has ended.
async fn run(
    runner: &QueryRunner,
    plan: Arc<dyn ExecNode>,
    q: &Arc<QueryContext>,
) -> Result<Vec<Batch>> {
    let mut stream = runner.execute(plan, Arc::clone(q)).await?;
    let mut out = Vec::new();
    let mut error = None;
    while let Some(b) = stream.next().await {
        match b {
            Ok(b) => out.push(b),
            Err(e) => error = Some(e),
        }
    }
    drop(stream);
    q.finished().await;
    error.map_or(Ok(out), Err)
}

fn values(batches: &[Batch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            b.data()
                .column(0)
                .as_primitive::<Int64Type>()
                .values()
                .to_vec()
        })
        .collect()
}

fn rows(batches: &[Batch]) -> usize {
    batches.iter().map(Batch::num_rows).sum()
}

fn scan(rows: u64) -> (Arc<RangeSource>, Arc<dyn ExecNode>) {
    let source = RangeSource::new(rows);
    (Arc::clone(&source), Arc::new(Scan(source)))
}

fn pipe(child: Arc<dyn ExecNode>, op: Arc<dyn Operator>) -> Arc<dyn ExecNode> {
    Arc::new(Pipe(child, op))
}

/// Calls `f` with each input's index, then passes the input on.
struct Hook<F>(F, AtomicU64);

fn hook<F: Fn(u64) -> Result<()> + Send + Sync + 'static>(f: F) -> Arc<dyn Operator> {
    Arc::new(Hook(f, AtomicU64::new(0)))
}

impl<F: Fn(u64) -> Result<()> + Send + Sync> Operator for Hook<F> {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn local_state(&self, _ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn execute(&self, input: &Batch, _local: &mut LocalState) -> Result<OperatorOutput> {
        (self.0)(self.1.fetch_add(1, Ordering::Relaxed))?;
        Ok(OperatorOutput::NeedMoreInput(Some(input.clone())))
    }
}

fn assert_released(root: &TreePool) {
    assert_eq!(
        root.reserved(),
        0,
        "the query's memory is released at its end"
    );
}

/// Acceptance: scan → filter → sink over 10 M rows, exact count and
/// checksum, at 1 and N threads.
#[tokio::test]
async fn scan_filter_sink_over_ten_million_rows() {
    const ROWS: u64 = 10_000_000;
    for threads in [1, 4] {
        let runner = QueryRunner::new(executor(threads));
        let root = root(1 << 30);
        let q = query(&root, 1, ExecConfig::default());
        let (_, scan) = scan(ROWS);
        let count = Count::new();
        let plan = Arc::new(Break(
            pipe(scan, Filter::new(|v| v % 2 == 0)),
            Arc::clone(&count),
            false,
        ));
        let out = run(&runner, plan, &q).await.unwrap();
        let half = (ROWS / 2) as i64;
        assert_eq!(
            values(&out),
            [half, half * (half - 1)],
            "count and sum of the even values"
        );
        assert_eq!(count.finalized.load(Ordering::Relaxed), 1);
        let m = q.metrics();
        let filter = m
            .operators
            .values()
            .find(|e| e.name.ends_with("Filter"))
            .unwrap();
        assert_eq!(
            (filter.metrics.rows_in, filter.metrics.rows_out),
            (ROWS, ROWS / 2)
        );
        assert_eq!(
            m.drivers.batches,
            ROWS.div_ceil(DEFAULT_BATCH_SIZE as u64) + 1
        );
        assert_eq!(m.drivers.drivers, threads as u64 + 1);
        assert!(m.drivers.yields >= m.drivers.batches);
        assert_released(&root);
    }
}

/// Acceptance: a cancelled query stops within 100 ms and releases all its
/// reservations.
#[tokio::test]
async fn a_cancelled_query_stops_within_100ms_and_releases_everything() {
    let runner = QueryRunner::new(executor(4));
    let root = root(1 << 30);
    let q = query(&root, 7, ExecConfig::default());
    let (_, scan) = scan(u64::MAX >> 8);
    let buffer = Buffer::new(true, None);
    let plan = Arc::new(Break(scan, Arc::clone(&buffer), false));
    let mut stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
    while buffer.sunk.load(Ordering::Relaxed) < 100 {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert!(root.reserved() > 0, "the buffer holds memory");
    let start = Instant::now();
    runner.cancel(QueryId(7));
    q.finished().await;
    let took = start.elapsed();
    assert!(took < Duration::from_millis(100), "stopped in {took:?}");
    assert_released(&root);
    let e = stream.next().await.unwrap().unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::OperationCanceled);
    assert!(stream.next().await.is_none());
    runner.cancel(QueryId(7)); // finished: ignored
}

/// Acceptance: the driver services a spill request at the next batch
/// boundary.
#[tokio::test(start_paused = true)]
async fn a_spill_request_is_serviced_at_the_next_batch() {
    let runner = QueryRunner::new(TokioExecutor::deterministic(1).unwrap());
    let root = root(1 << 30);
    let q = query(&root, 1, ExecConfig::default());
    let (_, scan) = scan(20 * DEFAULT_BATCH_SIZE as u64);
    let buffer = Buffer::new(true, None);
    let b = Arc::clone(&buffer);
    let trigger = hook(move |i| {
        if i == 5 {
            drop(b.signals()[0].request_spill(1));
        }
        Ok(())
    });
    let plan = Arc::new(Break(pipe(scan, trigger), Arc::clone(&buffer), false));
    let out = run(&runner, plan, &q).await.unwrap();
    assert_eq!(rows(&out), 20 * DEFAULT_BATCH_SIZE);
    // Batch 5 (the 6th) was sunk, then the boundary before batch 6 spilled.
    assert_eq!(buffer.spilled_at(), [6]);
    let sink = q
        .metrics()
        .operators
        .values()
        .find(|e| e.name.ends_with("Buffer"))
        .cloned()
        .unwrap();
    assert_eq!(sink.metrics.spills, 1);
    assert!(sink.metrics.spilled_bytes > 0);
    assert_released(&root);
}

/// Acceptance: a blocking call routed through the blocking pool does not
/// hold up a concurrent short query. Deterministic: with one worker thread,
/// the short query runs to completion while the long query's blocking call
/// is parked at a gate the test opens only afterwards. Were the call run on
/// the worker, the short query could not finish (the timeouts only turn
/// that hang into a failure; nothing is timed).
#[tokio::test]
async fn blocking_work_does_not_delay_a_short_query() {
    const HANG: Duration = Duration::from_secs(60);
    let exec = executor(1);
    let runner = QueryRunner::new(Arc::clone(&exec) as Arc<dyn Executor>);
    let root = root(1 << 30);
    let gate = BlockingGate::new();
    let (_, long_scan) = scan(DEFAULT_BATCH_SIZE as u64);
    let long = Buffer::with_finalize_gate(Arc::clone(&gate));
    let long_q = query(&root, 1, ExecConfig::default());
    let mut long_stream = runner
        .execute(Arc::new(Break(long_scan, long, false)), Arc::clone(&long_q))
        .await
        .unwrap();
    let reached = tokio::time::timeout(HANG, async {
        while !gate.reached() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await;
    let short = async {
        let (_, short_scan) = scan(10 * DEFAULT_BATCH_SIZE as u64);
        let plan: Arc<dyn ExecNode> = Arc::new(Break(short_scan, Count::new(), false));
        let q = query(&root, 2, ExecConfig::default());
        tokio::time::timeout(HANG, run(&runner, plan, &q)).await
    };
    let short = if reached.is_ok() {
        Some(short.await)
    } else {
        None
    };
    let long_blocked = !long_q.is_finished();
    gate.open(); // before asserting, so a failure never leaves it shut
    assert!(reached.is_ok(), "the long query's blocking call started");
    let short = short.and_then(Result::ok);
    assert!(
        short.is_some(),
        "the short query ran beside the blocking call"
    );
    short.into_iter().for_each(|r| drop(r.unwrap()));
    assert!(long_blocked, "the blocking call was still running");
    while long_stream.next().await.is_some() {}
    long_q.finished().await;
    assert_released(&root);
}

/// A panic in the blocking pool is an `INTERNAL_ERROR`; task handles
/// report aborts as cancellations.
#[tokio::test]
async fn blocking_pool_panics_and_task_aborts() {
    let exec = executor(2);
    exec.spawn_blocking(Box::new(|| {})).await.unwrap();
    let e = exec
        .spawn_blocking(Box::new(|| panic!("boom")))
        .await
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert!(e.to_string().contains("boom"));
    let io = exec.io_handle().spawn(async { 7 });
    assert_eq!(io.await.unwrap(), 7);
    assert!(format!("{exec:?}").contains("parallelism: 2"));
    // Aborting a task reports a cancellation.
    let t = exec.spawn(WorkloadClass::Batch, Box::pin(std::future::pending()));
    t.abort();
    assert_eq!(
        t.await.unwrap_err().error_class(),
        ErrorClass::OperationCanceled
    );
    // Dropping the executor inside async code does not block.
    drop(exec);
}

/// A streaming LIMIT stops its upstream: at most one batch is read past it.
#[tokio::test(start_paused = true)]
async fn limit_stops_upstream() {
    let runner = QueryRunner::new(TokioExecutor::deterministic(1).unwrap());
    let root = root(1 << 30);
    let q = query(&root, 1, ExecConfig::default());
    let (source, scan) = scan(1_000_000);
    let plan = pipe(scan, Limit::new(10_000));
    let out = run(&runner, plan, &q).await.unwrap();
    assert_eq!(rows(&out), 10_000);
    let read = source.produced.load(Ordering::Relaxed);
    assert!(
        read <= 10_000 + DEFAULT_BATCH_SIZE as u64,
        "read {read} rows"
    );
    assert_released(&root);
}

/// The result channel is bounded: drivers stop pulling while the caller
/// does not read; dropping the stream cancels the query.
#[tokio::test(start_paused = true)]
async fn backpressure_and_dropping_the_stream_cancels() {
    const SLOTS: u64 = 2;
    let runner = QueryRunner::new(TokioExecutor::deterministic(2).unwrap());
    let root = root(1 << 30);
    let q = query(
        &root,
        1,
        ExecConfig {
            morsel_rows: DEFAULT_BATCH_SIZE,
            ..ExecConfig::default()
        },
    );
    let (source, scan) = scan(u64::MAX >> 8);
    let mut stream = runner.execute(scan, Arc::clone(&q)).await.unwrap();
    assert!(stream.next().await.unwrap().is_ok());
    tokio::time::sleep(Duration::from_millis(50)).await;
    let read = source.produced.load(Ordering::Relaxed);
    // Capacity 2 per slot, one more accepted per blocked driver, one read.
    let bound = (2 * SLOTS + SLOTS + 1) * DEFAULT_BATCH_SIZE as u64;
    assert!(read <= bound, "read {read} rows while the caller was idle");
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        source.produced.load(Ordering::Relaxed),
        read,
        "drivers are parked"
    );
    drop(stream);
    q.finished().await;
    assert_eq!(
        q.error().unwrap().error_class(),
        ErrorClass::OperationCanceled
    );
    let sink = q
        .metrics()
        .operators
        .values()
        .find(|e| e.name == "Result")
        .cloned()
        .unwrap();
    assert!(sink.metrics.blocked_on_sink > Duration::ZERO);
    assert_released(&root);
}

/// The first error wins and cancels the other drivers; their secondary
/// "cancelled" errors are not reported.
#[tokio::test]
async fn the_first_error_wins() {
    let runner = QueryRunner::new(executor(4));
    let root = root(1 << 30);
    let q = query(
        &root,
        1,
        ExecConfig {
            morsel_rows: DEFAULT_BATCH_SIZE,
            ..ExecConfig::default()
        },
    );
    let (_, scan) = scan(1 << 24);
    let fail = hook(|i| {
        if i == 20 {
            return Err(DivideByZero {
                config: "\"spark.sql.ansi.enabled\"".into(),
            }
            .into());
        }
        Ok(())
    });
    let buffer = Buffer::new(true, None);
    let plan = Arc::new(Break(pipe(scan, fail), buffer, false));
    let e = run(&runner, plan, &q).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DivideByZero);
    assert_eq!(q.error(), Some(e));
    assert_released(&root);
}

/// A panic fails only its own query (`INTERNAL_ERROR`); a concurrent query
/// on the same executor completes.
#[tokio::test]
async fn a_panic_fails_only_its_query() {
    let runner = Arc::new(QueryRunner::new(executor(2)));
    let root = root(1 << 30);
    let bad = query(
        &root,
        1,
        ExecConfig {
            morsel_rows: DEFAULT_BATCH_SIZE,
            ..ExecConfig::default()
        },
    );
    let good = query(&root, 2, ExecConfig::default());
    let (_, scan_bad) = scan(1 << 22);
    let boom = hook(|i| {
        if i == 10 {
            panic!("operator bug")
        } else {
            Ok(())
        }
    });
    let bad_plan = Arc::new(Break(pipe(scan_bad, boom), Buffer::new(true, None), false));
    let (_, scan_good) = scan(1 << 20);
    let good_plan = Arc::new(Break(scan_good, Count::new(), false));
    let (r1, r2) = tokio::join!(run(&runner, bad_plan, &bad), run(&runner, good_plan, &good));
    let e = r1.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert!(e.to_string().contains("operator bug"), "{e}");
    assert_eq!(values(&r2.unwrap())[0], 1 << 20);
    assert_released(&root);
}

/// Two queries, one memory limit: the second needs memory while the first
/// holds spillable memory; the first spills at its next batch boundary.
/// `NeedMemory` retries take the grant credit, so nothing is reserved twice.
#[tokio::test(start_paused = true)]
async fn two_queries_one_limit_spill_while_allocating() {
    const HELD: i64 = 64 * DEFAULT_BATCH_SIZE as i64; // ~2 MiB
    let runner = QueryRunner::new(TokioExecutor::deterministic(1).unwrap());
    let root = root(4 * MIB);
    let q1 = query(&root, 1, ExecConfig::default());
    let q2 = query(&root, 2, ExecConfig::default());
    // q1 buffers its first ~2 MiB, then keeps scanning (filtered out), so
    // it passes batch boundaries while holding the memory.
    let gate = Arc::new(tokio::sync::Notify::new());
    let g = Arc::clone(&gate);
    let start_q2 = hook(move |i| {
        if i == 64 {
            g.notify_one();
        }
        Ok(())
    });
    let (_, scan1) = scan(400 * DEFAULT_BATCH_SIZE as u64);
    let holder = Buffer::new(true, None);
    let plan1 = Arc::new(Break(
        pipe(pipe(scan1, start_q2), Filter::new(|v| v < HELD)),
        Arc::clone(&holder),
        false,
    ));
    let (_, scan2) = scan(80 * DEFAULT_BATCH_SIZE as u64);
    let taker = Buffer::new(false, None);
    let plan2 = Arc::new(Break(scan2, Arc::clone(&taker), false));
    let first = async { rows(&run(&runner, plan1, &q1).await.unwrap()) };
    let second = async {
        gate.notified().await;
        rows(&run(&runner, plan2, &q2).await.unwrap())
    };
    let (n1, n2) = tokio::join!(first, second);
    assert_eq!((n1, n2), (64 * DEFAULT_BATCH_SIZE, 80 * DEFAULT_BATCH_SIZE));
    assert!(!holder.spilled_at().is_empty(), "q1 spilled for q2");
    assert!(taker.spilled_at().is_empty());
    let stats = root.stats();
    assert!(stats.spill_requests >= 1 && stats.aborts == 0, "{stats:?}");
    assert_released(&root);
}

/// Seeded deterministic mode replays the morsel order; unseeded claims
/// morsels in order.
#[tokio::test(start_paused = true)]
async fn deterministic_mode_replays_morsel_order() {
    async fn firsts(seed: Option<u64>) -> Vec<i64> {
        let runner = QueryRunner::new(TokioExecutor::deterministic(1).unwrap());
        let root = root(1 << 30);
        let config = ExecConfig {
            morsel_rows: DEFAULT_BATCH_SIZE,
            seed,
            ..ExecConfig::default()
        };
        let q = query(&root, 1, config);
        let (_, scan) = scan(16 * DEFAULT_BATCH_SIZE as u64);
        let out = run(&runner, scan, &q).await.unwrap();
        out.iter()
            .map(|b| b.data().column(0).as_primitive::<Int64Type>().value(0))
            .collect()
    }
    let a = firsts(Some(7)).await;
    assert_eq!(a, firsts(Some(7)).await);
    assert_ne!(a, firsts(Some(8)).await);
    let ordered = firsts(None).await;
    assert!(ordered.is_sorted());
    let mut sorted = a.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, ordered, "same morsels, different order");
}

/// Spins until this thread has run for `d` of CPU time. The stall detector
/// charges a poll its slice minus the thread's run-queue wait, so a spin
/// timed by the wall clock holds the CPU for less than `d` on a loaded
/// machine (the thread is preempted for part of it) and may not trip it.
#[cfg(unix)]
fn spin_on_cpu(d: Duration) {
    let cpu = || {
        let t = rustix::time::clock_gettime(rustix::time::ClockId::ThreadCPUTime);
        Duration::new(
            t.tv_sec.unsigned_abs(),
            u32::try_from(t.tv_nsec).unwrap_or(0),
        )
    };
    let start = cpu();
    while cpu().saturating_sub(start) < d {
        std::hint::spin_loop();
    }
}

/// Targets without `rustix::time` (Windows) time the spin by the wall clock:
/// the test compiles there, and is only as reliable as a wall-clock spin.
#[cfg(not(unix))]
fn spin_on_cpu(d: Duration) {
    let start = std::time::Instant::now();
    while start.elapsed() < d {
        std::hint::spin_loop();
    }
}

/// The stall detector fails a driver that does not yield for longer than
/// the threshold (in fail-on-stall mode).
#[tokio::test]
async fn the_stall_detector_fails_a_non_yielding_driver() {
    let runner = QueryRunner::new(executor(1));
    let root = root(1 << 30);
    let config = ExecConfig {
        fail_on_stall: true,
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let (_, scan) = scan(4 * DEFAULT_BATCH_SIZE as u64);
    let spin = hook(|_| {
        spin_on_cpu(Duration::from_millis(30));
        Ok(())
    });
    let e = run(&runner, pipe(scan, spin), &q).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert!(e.to_string().contains("without yielding"), "{e}");
    let stats = q.metrics().drivers;
    assert!(stats.stalls >= 1 && stats.max_slice >= Duration::from_millis(30));
    assert_released(&root);
}

/// A driver that blocks inside a poll (a sleep standing in for blocking
/// I/O) is a stall too: only run-queue waits are left out of a slice.
#[tokio::test]
async fn the_stall_detector_fails_a_blocking_driver() {
    let runner = QueryRunner::new(executor(1));
    let root = root(1 << 30);
    let config = ExecConfig {
        fail_on_stall: true,
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let (_, scan) = scan(4 * DEFAULT_BATCH_SIZE as u64);
    let block = hook(|_| {
        std::thread::sleep(Duration::from_millis(30));
        Ok(())
    });
    let e = run(&runner, pipe(scan, block), &q).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert!(e.to_string().contains("without yielding"), "{e}");
    assert!(q.metrics().drivers.stalls >= 1);
    assert_released(&root);
}

/// Emits 16 outputs per input with `HaveMoreOutput`.
struct Explode;

impl Operator for Explode {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn local_state(&self, _ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(0u32))
    }

    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput> {
        let n = local.downcast_mut::<u32>().unwrap();
        *n += 1;
        if *n <= 16 {
            return Ok(OperatorOutput::HaveMoreOutput(input.slice(0, 1)?));
        }
        *n = 0;
        Ok(OperatorOutput::NeedMoreInput(None))
    }
}

/// A driver yields once per source batch and once every 8 extra outputs.
#[tokio::test(start_paused = true)]
async fn yields_per_batch_and_every_eight_outputs() {
    let runner = QueryRunner::new(TokioExecutor::deterministic(1).unwrap());
    let root = root(1 << 30);
    let q = query(&root, 1, ExecConfig::default());
    let (_, scan) = scan(3 * DEFAULT_BATCH_SIZE as u64);
    let out = run(&runner, pipe(scan, Arc::new(Explode)), &q)
        .await
        .unwrap();
    assert_eq!(rows(&out), 3 * 16);
    let m = q.metrics();
    assert_eq!((m.drivers.batches, m.drivers.yields), (3, 3 + 3 * 2));
    let explode = m
        .operators
        .values()
        .find(|e| e.name.ends_with("Explode"))
        .unwrap();
    assert_eq!(
        (explode.metrics.batches_in, explode.metrics.batches_out),
        (3, 48)
    );
    assert_released(&root);
}

/// Planning errors are returned by `submit`; source state errors fail the
/// query; a deterministic executor needs a current-thread runtime.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn planning_and_setup_errors() {
    let e = TokioExecutor::deterministic(1).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    let runner = QueryRunner::new(executor(1));
    let root = root(1 << 30);

    struct NoSource;
    impl ExecNode for NoSource {
        fn build_pipelines(self: Arc<Self>, _b: &mut PipelineBuilder) -> Result<()> {
            Ok(())
        }
    }
    struct TwoSources;
    impl ExecNode for TwoSources {
        fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
            b.set_source(RangeSource::new(1))?;
            b.set_source(RangeSource::new(1)).map(drop)
        }
    }
    for plan in [
        Arc::new(NoSource) as Arc<dyn ExecNode>,
        Arc::new(TwoSources),
    ] {
        let e = runner
            .execute(plan, query(&root, 1, ExecConfig::default()))
            .await
            .err()
            .unwrap();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
    }
    // A breaker source that names no child pipeline.
    struct BadBreaker;
    impl ExecNode for BadBreaker {
        fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
            let count = Count::new();
            b.set_breaker_source(count, krishiv_exec::OperatorId(0))
                .map(drop)
        }
    }
    let e = runner
        .execute(Arc::new(BadBreaker), query(&root, 1, ExecConfig::default()))
        .await
        .err()
        .unwrap();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    // A query id that is already running is refused.
    let running = query(&root, 5, ExecConfig::default());
    let (_, scan5) = scan(u64::MAX >> 8);
    let stream = runner.execute(scan5, Arc::clone(&running)).await.unwrap();
    let (_, again) = scan(1);
    let e = runner
        .execute(again, query(&root, 5, ExecConfig::default()))
        .await
        .err()
        .unwrap();
    assert!(e.to_string().contains("already running"), "{e}");
    drop(stream);
    running.finished().await;
    // A breaker read before it was built (its source used alone).
    let buffer = Buffer::new(true, None);
    let q = query(&root, 2, ExecConfig::default());
    let e = run(&runner, Arc::new(Scan(buffer)), &q).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert_released(&root);
}

/// The row-layout keys reach `ExecConfig`: on and 3 radix bits by default,
/// and the radix bits are clamped to 3..=6.
#[test]
fn config_join_row_layout_keys() {
    let default = ExecConfig::from_config(&Config::new());
    assert!(default.agg_row_layout && default.join_row_layout);
    assert_eq!(default.join_radix_bits, 3);
    let read = |bits: &str| {
        let (config, _) = Config::from_pairs([
            ("krishiv.exec.agg.rowLayout", "false"),
            ("krishiv.exec.join.rowLayout", "false"),
            ("krishiv.exec.join.radixBits", bits),
        ])
        .unwrap();
        ExecConfig::from_config(&config)
    };
    let off = read("5");
    assert!(!off.agg_row_layout && !off.join_row_layout);
    assert_eq!(off.join_radix_bits, 5);
    assert_eq!(read("1").join_radix_bits, 3);
    assert_eq!(read("9").join_radix_bits, 6);
}

#[test]
fn config_merge_join_buffer_bytes() {
    let default = ExecConfig::from_config(&Config::new());
    assert_eq!(default.join_merge_buffer_bytes, 0);
    let (config, _) = Config::from_pairs([("krishiv.exec.join.smj.bufferBytes", "4m")]).unwrap();
    assert_eq!(
        ExecConfig::from_config(&config).join_merge_buffer_bytes,
        4 << 20
    );
}

/// Sort hash buckets are on by default; `krishiv.exec.sort.hashBuckets`
/// turns them off.
#[test]
fn config_sort_hash_buckets_key() {
    assert!(ExecConfig::from_config(&Config::new()).sort_hash_buckets);
    let (config, _) = Config::from_pairs([("krishiv.exec.sort.hashBuckets", "false")]).unwrap();
    assert!(!ExecConfig::from_config(&config).sort_hash_buckets);
}

/// `krishiv.exec.batchSize` sizes the batches; morsels feed several drivers;
/// an order-preserving breaker output keeps its order.
#[tokio::test]
async fn config_batch_size_and_ordered_breaker() {
    let (config, _) = Config::from_pairs([("krishiv.exec.batchSize", "1000")]).unwrap();
    let exec_config = ExecConfig::from_config(&config);
    assert_eq!(exec_config.batch_size, 1000);
    assert_eq!(
        ExecConfig::from_config(&Config::new()).batch_size,
        DEFAULT_BATCH_SIZE
    );
    let runner = QueryRunner::new(executor(2));
    let root = root(1 << 30);
    let q = query(
        &root,
        1,
        ExecConfig {
            morsel_rows: 5000,
            ..exec_config
        },
    );
    let (_, scan) = scan(20_000);
    let plan = Arc::new(Break(scan, Buffer::new(true, None), true));
    let out = run(&runner, plan, &q).await.unwrap();
    assert!(out.iter().all(|b| b.num_rows() == 1000));
    let mut v = values(&out);
    v.sort_unstable();
    assert_eq!(v, (0..20_000).collect::<Vec<i64>>());
    assert_released(&root);
    let e = KrishivError::from(MemoryError::Aborted { held: 1 });
    assert_eq!(e.error_class(), ErrorClass::KrishivQueryAbortedOutOfMemory);
}

/// Reserves `.1` bytes per driver up front and offers them for spilling if
/// `.2`; records its spill signals.
struct Hold(
    std::sync::Mutex<Vec<Arc<krishiv_memory::SpillSignal>>>,
    usize,
    bool,
);

fn hold(bytes: usize, spillable: bool) -> Arc<Hold> {
    Arc::new(Hold(std::sync::Mutex::default(), bytes, spillable))
}

impl Operator for Hold {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        let memory = ctx.consumer();
        let mut r = memory.reservation();
        r.try_grow(self.1)?;
        if self.2 {
            memory.signal().set_spillable_bytes(self.1);
        }
        self.0.lock().unwrap().push(Arc::clone(memory.signal()));
        Ok(Box::new(r))
    }

    fn execute(&self, input: &Batch, _local: &mut LocalState) -> Result<OperatorOutput> {
        Ok(OperatorOutput::NeedMoreInput(Some(input.clone())))
    }

    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        _target: usize,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        let r = local.downcast_mut::<krishiv_memory::Reservation>().unwrap();
        let freed = r.size();
        r.shrink(freed);
        Box::pin(async move { Ok(freed) })
    }
}

/// A driver parked on a full result channel still services spill requests
/// for its own consumers.
#[tokio::test(start_paused = true)]
async fn a_parked_driver_services_spill_requests() {
    let runner = QueryRunner::new(TokioExecutor::deterministic(1).unwrap());
    let root = root(8 * MIB);
    let q = query(&root, 1, ExecConfig::default());
    let hold = hold(MIB, true);
    let (_, scan) = scan(u64::MAX >> 8);
    let plan = pipe(scan, Arc::clone(&hold) as Arc<dyn Operator>);
    let stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await; // the driver parks on the full channel
    let signal = Arc::clone(&hold.0.lock().unwrap()[0]);
    assert_eq!(
        signal.request_spill(MIB).await,
        MIB,
        "serviced while parked"
    );
    assert!(root.reserved() < MIB);
    drop(stream);
    q.finished().await;
    assert_eq!(
        q.error().unwrap().error_class(),
        ErrorClass::OperationCanceled
    );
    let m = q.metrics();
    let op = m
        .operators
        .values()
        .find(|e| e.name.ends_with("Hold"))
        .unwrap();
    assert_eq!(
        (op.metrics.spills, op.metrics.peak_reserved),
        (1, MIB as u64)
    );
    assert_released(&root);
}

/// A driver parked on a full result channel fails at once with the abort
/// error when the arbitrator aborts its query, releasing its memory.
#[tokio::test(start_paused = true)]
async fn a_memory_abort_wakes_a_parked_driver() {
    let runner = QueryRunner::new(TokioExecutor::deterministic(1).unwrap());
    let root = root(4 * MIB);
    let q1 = query(&root, 1, ExecConfig::default());
    let (_, scan) = scan(u64::MAX >> 8);
    // Not spillable and above its fair share: the arbitrator aborts it.
    let plan = pipe(scan, hold(3 * MIB, false) as Arc<dyn Operator>);
    let mut stream = runner.execute(plan, Arc::clone(&q1)).await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    let q2 = root.child("q2", None);
    let id2 = q2.register("want", None);
    let start = tokio::time::Instant::now();
    q2.reserve(id2, 2 * MIB).await.unwrap();
    assert!(
        start.elapsed() < Duration::from_secs(30),
        "no release timeout"
    );
    assert_eq!(root.stats().aborts, 1);
    q1.finished().await;
    let mut last = None;
    while let Some(b) = stream.next().await {
        last = Some(b);
    }
    let e = last.unwrap().unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivQueryAbortedOutOfMemory);
    q2.unregister(id2);
    assert_released(&root);
}

/// Grows its reservation by every input's size; `NeedMemory` when the pool
/// refuses; spills everything when asked.
struct Greedy;

impl Operator for Greedy {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        assert_eq!(ctx.executor().parallelism(), 1);
        let memory = ctx.consumer();
        assert_eq!(memory.id(), ctx.consumer().id(), "registered once");
        Ok(Box::new((
            memory.reservation(),
            Arc::clone(memory.signal()),
        )))
    }

    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput> {
        let (r, signal) = local
            .downcast_mut::<(
                krishiv_memory::Reservation,
                Arc<krishiv_memory::SpillSignal>,
            )>()
            .unwrap();
        let bytes = input.data().get_array_memory_size();
        if r.try_grow(bytes).is_err() {
            return Ok(OperatorOutput::NeedMemory { bytes });
        }
        signal.set_spillable_bytes(r.size());
        Ok(OperatorOutput::NeedMoreInput(Some(input.clone())))
    }

    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        _target: usize,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        let (r, signal) = local
            .downcast_mut::<(
                krishiv_memory::Reservation,
                Arc<krishiv_memory::SpillSignal>,
            )>()
            .unwrap();
        let freed = r.size();
        r.shrink(freed);
        signal.set_spillable_bytes(0);
        Box::pin(async move { Ok(freed) })
    }
}

/// An operator and a sink that outgrow the limit get `NeedMemory`; the
/// driver spills their own state first, then retries the same input with
/// the bytes as grant credit.
#[tokio::test(start_paused = true)]
async fn need_memory_self_spills_and_retries() {
    let runner = QueryRunner::new(TokioExecutor::deterministic(1).unwrap());
    let root = root(MIB);
    let q = query(&root, 1, ExecConfig::default());
    let (_, scan) = scan(100 * DEFAULT_BATCH_SIZE as u64); // ~3 MiB through each
    let buffer = Buffer::new(true, None);
    let plan = Arc::new(Break(
        pipe(scan, Arc::new(Greedy)),
        Arc::clone(&buffer),
        false,
    ));
    let out = run(&runner, plan, &q).await.unwrap();
    assert_eq!(rows(&out), 100 * DEFAULT_BATCH_SIZE);
    let m = q.metrics();
    let greedy = m
        .operators
        .values()
        .find(|e| e.name.ends_with("Greedy"))
        .unwrap();
    // Deterministic mode times on the paused clock: a wait that needs no
    // timer costs no time, so check the spills, not the wait time.
    assert!(greedy.metrics.spills > 0);
    assert!(
        greedy.metrics.peak_reserved <= MIB as u64,
        "never double-reserved"
    );
    assert!(!buffer.spilled_at().is_empty());
    assert_released(&root);
}

/// Takes the first batch only: its sink returns `Finished`.
#[derive(Default)]
struct First(std::sync::Mutex<Option<Batch>>);

impl krishiv_exec::Sink for First {
    fn global_state(&self, _ctx: &TaskContext) -> Result<krishiv_exec::GlobalState> {
        Ok(Arc::new(()))
    }

    fn local_state(
        &self,
        _ctx: &TaskContext,
        _g: &krishiv_exec::GlobalState,
    ) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn sink(
        &self,
        input: Batch,
        _local: &mut LocalState,
        _g: &krishiv_exec::GlobalState,
    ) -> Result<krishiv_exec::SinkStatus> {
        self.0.lock().unwrap().get_or_insert(input);
        Ok(krishiv_exec::SinkStatus::Finished)
    }

    fn combine(&self, _local: LocalState, _g: &krishiv_exec::GlobalState) -> Result<()> {
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        _g: &'a krishiv_exec::GlobalState,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

impl krishiv_exec::Source for First {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn max_parallelism(&self, _g: &krishiv_exec::GlobalState) -> usize {
        1
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<krishiv_exec::GlobalState> {
        Ok(Arc::new(()))
    }

    fn local_state(
        &self,
        _ctx: &TaskContext,
        _g: &krishiv_exec::GlobalState,
    ) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn get_data<'a>(
        &'a self,
        _local: &'a mut LocalState,
        _g: &'a krishiv_exec::GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        let b = self.0.lock().unwrap().take();
        Box::pin(async move { Ok(b) })
    }
}

/// A sink that returns `Finished` stops its driver's source.
#[tokio::test(start_paused = true)]
async fn a_finished_sink_stops_its_source() {
    let runner = QueryRunner::new(TokioExecutor::deterministic(1).unwrap());
    let root = root(1 << 30);
    let q = query(&root, 1, ExecConfig::default());
    let (source, scan) = scan(1_000_000);
    let out = run(
        &runner,
        Arc::new(Break(scan, Arc::new(First::default()), false)),
        &q,
    )
    .await;
    assert_eq!(values(&out.unwrap())[..2], [0, 1]);
    assert_eq!(
        source.produced.load(Ordering::Relaxed),
        DEFAULT_BATCH_SIZE as u64
    );
    assert_released(&root);
}

/// Shutting the executor down ends its running queries (cancelled), and
/// dropping the last handle to a runtime from inside it does not block.
#[test]
fn shutting_down_the_executor_ends_running_queries() {
    let exec = executor(2);
    let runner = QueryRunner::new(Arc::clone(&exec) as Arc<dyn Executor>);
    let root = root(1 << 30);
    let q = query(&root, 1, ExecConfig::default());
    let caller = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (_, scan) = scan(u64::MAX >> 8);
    let mut stream = caller
        .block_on(runner.execute(scan, Arc::clone(&q)))
        .unwrap();
    drop(runner);
    drop(exec); // the running tasks still hold the executor
    caller.block_on(async {
        assert!(stream.next().await.unwrap().is_ok());
        drop(stream); // cancels; the last task drops the executor
        q.finished().await;
    });
    assert_released(&root);
}

/// A breaker's `finalize` runs while another query needs memory: the
/// arbitrator's request reaches the finalizing driver, which spills the
/// global state through `Sink::spill_global`.
#[tokio::test(start_paused = true)]
async fn finalize_spills_global_state_under_memory_pressure() {
    let runner = QueryRunner::new(TokioExecutor::deterministic(1).unwrap());
    let root = root(4 * MIB);
    let q1 = query(&root, 1, ExecConfig::default());
    let (_, scan) = scan(80 * DEFAULT_BATCH_SIZE as u64); // ~2.5 MiB
    let buffer = Buffer::with_finalize_yields(true, None, 10_000);
    let plan = Arc::new(Break(scan, Arc::clone(&buffer), false));
    let mut stream = runner.execute(plan, Arc::clone(&q1)).await.unwrap();
    while !buffer.finalizing.load(Ordering::Acquire) {
        tokio::task::yield_now().await;
    }
    assert!(buffer.spilled_at().is_empty());
    let q2 = root.child("q2", None);
    let id = q2.register("want", None);
    q2.reserve(id, 3 * MIB).await.unwrap();
    assert_eq!(buffer.spilled_at(), [80], "spilled during finalize");
    assert_eq!(root.stats().aborts, 0);
    let mut n = 0;
    while let Some(b) = stream.next().await {
        n += b.unwrap().num_rows();
    }
    assert_eq!(n, 80 * DEFAULT_BATCH_SIZE);
    q1.finished().await;
    q2.unregister(id);
    assert_released(&root);
}

/// Two child pipelines feed the root: a build side that finalizes and a
/// sibling that fails. The executor owns the finalized build state and drops
/// it at query end, so the pool is empty while the caller still holds the
/// plan.
#[tokio::test(start_paused = true)]
async fn breaker_state_is_released_when_a_sibling_fails() {
    struct Join(
        Arc<dyn ExecNode>,
        Arc<dyn ExecNode>,
        Arc<Buffer>,
        Arc<Count>,
    );
    impl ExecNode for Join {
        fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
            let build = b.child_pipeline(
                Arc::clone(&self.2) as Arc<dyn krishiv_exec::Sink>,
                Arc::clone(&self.0),
            )?;
            b.child_pipeline(
                Arc::clone(&self.3) as Arc<dyn krishiv_exec::Sink>,
                Arc::clone(&self.1),
            )?;
            b.set_breaker_source(Arc::clone(&self.2) as Arc<dyn krishiv_exec::Source>, build)
                .map(drop)
        }
    }
    let runner = QueryRunner::new(TokioExecutor::deterministic(1).unwrap());
    let root = root(1 << 30);
    let q = query(&root, 1, ExecConfig::default());
    let (_, build_scan) = scan(2 * DEFAULT_BATCH_SIZE as u64);
    let (_, probe_scan) = scan(64 * DEFAULT_BATCH_SIZE as u64);
    let fail = hook(|i| {
        if i == 32 {
            return Err(DivideByZero {
                config: String::new(),
            }
            .into());
        }
        Ok(())
    });
    let buffer = Buffer::new(true, None);
    let plan = Arc::new(Join(
        build_scan,
        pipe(probe_scan, fail),
        Arc::clone(&buffer),
        Count::new(),
    ));
    let held: Arc<dyn ExecNode> = Arc::clone(&plan) as Arc<dyn ExecNode>;
    let e = run(&runner, plan, &q).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DivideByZero);
    assert_eq!(
        buffer.sunk.load(Ordering::Relaxed),
        2,
        "the build side finished"
    );
    assert_released(&root);
    drop(held);
}

/// A range scan whose drivers each register a memory consumer (an I/O
/// buffer, say) and record its spill signal.
struct BufferedScan(
    Arc<RangeSource>,
    std::sync::Mutex<Vec<Arc<krishiv_memory::SpillSignal>>>,
);

impl krishiv_exec::Source for BufferedScan {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<krishiv_exec::GlobalState> {
        self.0.global_state(ctx)
    }

    fn max_parallelism(&self, g: &krishiv_exec::GlobalState) -> usize {
        self.0.max_parallelism(g)
    }

    fn local_state(&self, ctx: &TaskContext, g: &krishiv_exec::GlobalState) -> Result<LocalState> {
        let memory = ctx.consumer();
        memory.signal().set_spillable_bytes(MIB);
        self.1.lock().unwrap().push(Arc::clone(memory.signal()));
        self.0.local_state(ctx, g)
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        g: &'a krishiv_exec::GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        self.0.get_data(local, g)
    }
}

/// A spill request to a source's consumer (a source has no spill hook) is
/// completed with 0 bytes by the parked driver, which then still sees its
/// query's cancellation: it neither spins on the request nor hangs.
#[tokio::test]
async fn a_spill_request_to_a_source_is_answered_and_never_hangs() {
    let runner = QueryRunner::new(executor(1));
    let root = root(1 << 30);
    let q = query(&root, 1, ExecConfig::default());
    let source = Arc::new(BufferedScan(
        RangeSource::new(u64::MAX >> 8),
        std::sync::Mutex::default(),
    ));
    let plan = Arc::new(Scan(Arc::clone(&source) as Arc<dyn krishiv_exec::Source>));
    let stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
    while source.1.lock().unwrap().is_empty() {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    tokio::time::sleep(Duration::from_millis(20)).await; // parked on the full channel
    let signal = Arc::clone(&source.1.lock().unwrap()[0]);
    let freed = tokio::time::timeout(Duration::from_secs(5), signal.request_spill(MIB)).await;
    assert_eq!(freed, Ok(0), "answered while parked");
    drop(stream);
    tokio::time::timeout(Duration::from_secs(5), q.finished())
        .await
        .unwrap();
    assert_eq!(
        q.error().unwrap().error_class(),
        ErrorClass::OperationCanceled
    );
    assert_released(&root);
}

/// A flush-phase operator. With `spin` it swallows its input, and the first
/// driver to flush waits, like a grace join's exclusive driver, until no
/// other driver holds this operator's state; the others emit batches.
struct Gate {
    spin: bool,
    live: Arc<AtomicU64>,
    claimed: std::sync::atomic::AtomicBool,
}

fn gate(spin: bool) -> Arc<Gate> {
    Arc::new(Gate {
        spin,
        live: Arc::default(),
        claimed: std::sync::atomic::AtomicBool::new(false),
    })
}

struct GateState(Arc<AtomicU64>, Option<bool>);

impl Drop for GateState {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Operator for Gate {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn local_state(&self, _ctx: &TaskContext) -> Result<LocalState> {
        self.live.fetch_add(1, Ordering::AcqRel);
        Ok(Box::new(GateState(Arc::clone(&self.live), None)))
    }

    fn execute(&self, input: &Batch, _local: &mut LocalState) -> Result<OperatorOutput> {
        Ok(OperatorOutput::NeedMoreInput(
            (!self.spin).then(|| input.clone()),
        ))
    }

    fn needs_flush(&self) -> bool {
        true
    }

    fn flush<'a>(
        &'a self,
        local: &'a mut LocalState,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<FlushOutput>> {
        let state = local.downcast_mut::<GateState>().unwrap();
        let claimer = *state
            .1
            .get_or_insert_with(|| !self.claimed.swap(true, Ordering::AcqRel));
        let out = if !self.spin {
            FlushOutput::Done
        } else if claimer {
            if self.live.load(Ordering::Acquire) > 1 {
                FlushOutput::Continue
            } else {
                FlushOutput::Done
            }
        } else {
            let values = arrow_array::Int64Array::from_iter_values(0..100);
            let data = arrow_array::RecordBatch::try_new(schema(), vec![Arc::new(values)]);
            FlushOutput::Batch(Batch::new(data.unwrap()))
        };
        Box::pin(std::future::ready(Ok(out)))
    }
}

/// A driver whose output of a flushing operator is no longer wanted (a
/// `LIMIT` above it finished) drops that operator's state before it waits
/// at a later flush barrier, so a driver waiting for that state to go away
/// (a grace join's exclusive driver) can finish instead of deadlocking.
#[tokio::test]
async fn an_unwanted_operator_gives_up_its_shared_flush_work() {
    let runner = QueryRunner::new(executor(2));
    let root = root(64 * MIB);
    let config = ExecConfig {
        morsel_rows: 4096,
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let (_, scan) = scan(100_000);
    let plan = pipe(pipe(pipe(scan, gate(true)), Limit::new(1_000)), gate(false));
    let out = tokio::time::timeout(Duration::from_secs(30), run(&runner, plan, &q))
        .await
        .expect("the drivers deadlocked")
        .unwrap();
    assert_eq!(rows(&out), 1_000);
    assert_released(&root);
}

/// Its flush step asks the downstream `Hold` of its own driver to spill and
/// waits for the answer.
struct AskSpill(Arc<Hold>);

impl Operator for AskSpill {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn local_state(&self, _ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn execute(&self, input: &Batch, _local: &mut LocalState) -> Result<OperatorOutput> {
        Ok(OperatorOutput::NeedMoreInput(Some(input.clone())))
    }

    fn needs_flush(&self) -> bool {
        true
    }

    fn flush<'a>(
        &'a self,
        _local: &'a mut LocalState,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<FlushOutput>> {
        let signals = self.0.0.lock().unwrap().clone();
        Box::pin(async move {
            for s in signals {
                assert_eq!(s.request_spill(MIB).await, MIB);
            }
            Ok(FlushOutput::Done)
        })
    }
}

/// While an operator's flush step waits (a grace join loading or
/// repartitioning a partition), its driver still services spill requests
/// for its other consumers; they no longer wait for the step to end.
#[tokio::test(start_paused = true)]
async fn a_driver_in_a_flush_step_services_spill_requests() {
    let runner = QueryRunner::new(TokioExecutor::deterministic(1).unwrap());
    let root = root(8 * MIB);
    let q = query(&root, 1, ExecConfig::default());
    let hold = hold(MIB, true);
    let (_, scan) = scan(10_000);
    let ask: Arc<dyn Operator> = Arc::new(AskSpill(Arc::clone(&hold)));
    let plan = pipe(pipe(scan, ask), Arc::clone(&hold) as Arc<dyn Operator>);
    let out = tokio::time::timeout(Duration::from_secs(60), run(&runner, plan, &q))
        .await
        .expect("the spill request waited for the flush step")
        .unwrap();
    assert_eq!(rows(&out), 10_000);
    assert_released(&root);
}

/// A source that reserves `per_batch` working bytes through arbitration
/// for each batch (a scan's decoder buffers), released once the batch is
/// out.
struct GrowingScan(Arc<RangeSource>, usize);

struct GrowingLocal(LocalState, krishiv_memory::Reservation);

impl krishiv_exec::Source for GrowingScan {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<krishiv_exec::GlobalState> {
        self.0.global_state(ctx)
    }

    fn max_parallelism(&self, g: &krishiv_exec::GlobalState) -> usize {
        self.0.max_parallelism(g)
    }

    fn local_state(&self, ctx: &TaskContext, g: &krishiv_exec::GlobalState) -> Result<LocalState> {
        let inner = self.0.local_state(ctx, g)?;
        Ok(Box::new(GrowingLocal(inner, ctx.consumer().reservation())))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        g: &'a krishiv_exec::GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(async move {
            let l = local.downcast_mut::<GrowingLocal>().unwrap();
            l.1.grow(self.1).await?;
            let out = self.0.get_data(&mut l.0, g).await;
            l.1.shrink(self.1);
            out
        })
    }
}

/// A source growing through arbitration while its own driver's sink holds
/// the pool in spillable state: the arbitrator asks that sink (the source's
/// driver services it while the source waits) instead of failing the query.
#[tokio::test]
async fn a_source_waiting_for_memory_is_served_by_its_own_drivers_sink() {
    let runner = QueryRunner::new(executor(1));
    let root = root(4 * MIB);
    let q = query(&root, 1, ExecConfig::default());
    let sink = Buffer::new(true, None);
    let source = Arc::new(GrowingScan(
        RangeSource::new(1000 * DEFAULT_BATCH_SIZE as u64),
        MIB,
    ));
    let plan: Arc<dyn ExecNode> = Arc::new(Break(
        Arc::new(Scan(source as Arc<dyn krishiv_exec::Source>)),
        Arc::clone(&sink),
        false,
    ));
    let out = tokio::time::timeout(Duration::from_secs(60), run(&runner, plan, &q))
        .await
        .expect("no hang");
    out.unwrap();
    assert!(!sink.spilled_at().is_empty(), "the sink spilled");
    assert_eq!(root.stats().aborts, 0);
    assert_released(&root);
}

/// `UNION ALL` fan-in: several inputs into one pipeline.
struct FanIn(Vec<Arc<dyn ExecNode>>);

impl ExecNode for FanIn {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        b.fan_in(&self.0)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fan_in_runs_its_inputs_into_one_sink() {
    let runner = QueryRunner::new(executor(4));
    let root = root(1 << 30);
    // Inputs with operators of their own, a breaker under one, a nested
    // fan-in, then shared operators and a breaker above.
    let filtered = pipe(scan(10_000).1, Filter::new(|v| v % 2 == 0));
    let counted = Arc::new(Break(scan(7).1, Count::new(), false));
    let nested = Arc::new(FanIn(vec![scan(3).1, scan(4).1]));
    let union = Arc::new(FanIn(vec![filtered, counted, nested]));
    let plan = Arc::new(Break(
        pipe(union, Filter::new(|v| v < 5)),
        Count::new(),
        false,
    ));
    let out = run(&runner, plan, &query(&root, 1, ExecConfig::default()))
        .await
        .unwrap();
    // [count, sum]: 0, 2, 4 from the first input; the inner count's
    // [7, 21] is filtered out; 0..3 and 0..4 pass.
    assert_eq!(values(&out), [3 + 3 + 4, 6 + 3 + 6]);
    // A limit above a fan-in stops every input.
    let union = Arc::new(FanIn(vec![scan(1 << 20).1, scan(1 << 20).1]));
    let out = run(
        &runner,
        pipe(union, Limit::new(10)),
        &query(&root, 2, ExecConfig::default()),
    )
    .await
    .unwrap();
    assert_eq!(rows(&out), 10);
    assert_released(&root);
}

#[tokio::test]
async fn fan_in_errors() {
    let runner = QueryRunner::new(executor(1));
    let root = root(1 << 30);
    struct Started;
    impl ExecNode for Started {
        fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
            b.set_source(RangeSource::new(1))?;
            b.fan_in(&[scan(1).1])
        }
    }
    struct Failing;
    impl ExecNode for Failing {
        fn build_pipelines(self: Arc<Self>, _b: &mut PipelineBuilder) -> Result<()> {
            Err(DivideByZero {
                config: "\"spark.sql.ansi.enabled\"".into(),
            }
            .into())
        }
    }
    for plan in [
        Arc::new(FanIn(Vec::new())) as Arc<dyn ExecNode>,
        Arc::new(Started),
    ] {
        let e = runner
            .execute(plan, query(&root, 1, ExecConfig::default()))
            .await
            .err()
            .unwrap();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
    }
    // An input's planning error is the query's.
    let plan = Arc::new(FanIn(vec![scan(1).1, Arc::new(Failing)]));
    let e = runner
        .execute(plan, query(&root, 2, ExecConfig::default()))
        .await
        .err()
        .unwrap();
    assert_eq!(e.error_class(), ErrorClass::DivideByZero);
}

/// A breaker's finalized state goes once the pipeline reading it has ended,
/// not at query end: three breakers that cannot spill, each holding ~1 MiB
/// and read by the next, run under a 2.5 MiB limit, because at most two of
/// them hold memory at once.
#[tokio::test(start_paused = true)]
async fn a_finalized_state_is_released_after_its_last_reader() {
    let runner = QueryRunner::new(TokioExecutor::deterministic(1).unwrap());
    let root = root(5 * MIB / 2);
    let q = query(&root, 1, ExecConfig::default());
    let (_, scan) = scan(32 * DEFAULT_BATCH_SIZE as u64); // ~1 MiB
    let first = Arc::new(Break(scan, Buffer::new(false, None), false));
    let second = Arc::new(Break(first, Buffer::new(false, None), false));
    let third = Arc::new(Break(second, Buffer::new(false, None), false));
    let out = run(&runner, third, &q).await.unwrap();
    assert_eq!(rows(&out), 32 * DEFAULT_BATCH_SIZE);
    assert_eq!(root.stats().aborts, 0);
    assert_released(&root);
}

/// A [`Count`] whose sink publishes runtime filter 1.
struct Publisher(Arc<Count>);

impl Sink for Publisher {
    fn published_filters(&self) -> Vec<FilterId> {
        vec![FilterId(1)]
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        Sink::global_state(&*self.0, ctx)
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        Sink::local_state(&*self.0, ctx, global)
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        global: &GlobalState,
    ) -> Result<SinkStatus> {
        self.0.sink(input, local, global)
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        self.0.combine(local, global)
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        self.0.finalize(global, ctx)
    }
}

/// A range that applies runtime filter 1, noting how many times `publisher`
/// had finalized when its pipeline started.
struct Consumer {
    range: Arc<RangeSource>,
    publisher: Arc<Count>,
    finalized_at_start: AtomicU64,
}

impl Source for Consumer {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn runtime_filters(&self) -> Vec<FilterId> {
        vec![FilterId(1)]
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        let finalized = self.publisher.finalized.load(Ordering::Relaxed);
        self.finalized_at_start.store(finalized, Ordering::Relaxed);
        Source::global_state(&*self.range, ctx)
    }

    fn max_parallelism(&self, global: &GlobalState) -> usize {
        self.range.max_parallelism(global)
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        Source::local_state(&*self.range, ctx, global)
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        self.range.get_data(local, global)
    }
}

/// Two child pipelines no dependency orders: the publisher's and the
/// consumer's, below a pipeline reading the consumer's breaker.
struct Siblings {
    publisher: Arc<Publisher>,
    publisher_input: Arc<dyn ExecNode>,
    consumer: Arc<dyn ExecNode>,
    top: Arc<Count>,
}

impl ExecNode for Siblings {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        b.child_pipeline(
            Arc::clone(&self.publisher) as Arc<dyn Sink>,
            Arc::clone(&self.publisher_input),
        )?;
        let sink = b.child_pipeline(
            Arc::clone(&self.top) as Arc<dyn Sink>,
            Arc::clone(&self.consumer),
        )?;
        b.set_breaker_source(Arc::clone(&self.top) as Arc<dyn Source>, sink)?;
        Ok(())
    }
}

/// A scan's pipeline starts only once the pipeline publishing its runtime
/// filter has finalized, though no dependency orders them.
#[tokio::test]
async fn a_scan_starts_after_the_pipeline_publishing_its_filter() {
    let runner = QueryRunner::new(executor(2));
    let root = root(1 << 30);
    let q = query(&root, 1, ExecConfig::default());
    let count = Count::new();
    let (_, publisher_input) = scan(10_000);
    let consumer = Arc::new(Consumer {
        range: RangeSource::new(1_000),
        publisher: Arc::clone(&count),
        finalized_at_start: AtomicU64::new(u64::MAX),
    });
    let plan = Arc::new(Siblings {
        publisher: Arc::new(Publisher(count)),
        publisher_input,
        consumer: Arc::new(Scan(Arc::clone(&consumer) as Arc<dyn Source>)),
        top: Count::new(),
    });
    let out = run(&runner, plan, &q).await.unwrap();
    assert_eq!(values(&out)[0], 1_000);
    assert_eq!(consumer.finalized_at_start.load(Ordering::Relaxed), 1);
    assert_released(&root);
}
