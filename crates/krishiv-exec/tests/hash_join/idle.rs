// SPDX-License-Identifier: Apache-2.0
//! A finished build that waits for its probe is spilled before a running
//! consumer of the query spills its own state (study 30, M1 + M2): the
//! probe then reads the written partitions back, and the answer is the
//! in-memory one. A write of the idle build held at the executor tests
//! the two races of that write: the probe sealing the build while it is
//! in flight, and a cancel while it is in flight.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_exec::testing::{Batches, Scan};
use krishiv_exec::{
    ExecConfig, ExecNode, Executor, GlobalState, LocalState, QueryRunner, Result, Source,
    TaskContext, TaskHandle, TokioExecutor, WorkloadClass,
};
use krishiv_memory::{ConsumerHandle, SpillOwner, Spillable};
use krishiv_types::batch::Batch;

use super::util::*;
use super::{Case, JoinType, data, schema, skewed};

/// A scan that hands out nothing until the test opens it.
struct Gated {
    inner: Arc<Batches>,
    reached: Arc<AtomicBool>,
    /// Batches handed out after the gate opened.
    handed: Arc<AtomicUsize>,
    open: tokio::sync::watch::Receiver<bool>,
}

impl Source for Gated {
    fn name(&self) -> &'static str {
        "Gated"
    }

    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        self.inner.global_state(ctx)
    }

    fn max_parallelism(&self, global: &GlobalState) -> usize {
        self.inner.max_parallelism(global)
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        self.inner.local_state(ctx, global)
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(async move {
            self.reached.store(true, Ordering::Release);
            let mut open = self.open.clone();
            let _ = open.wait_for(|&o| o).await;
            let batch = self.inner.get_data(local, global).await?;
            if batch.is_some() {
                self.handed.fetch_add(1, Ordering::AcqRel);
            }
            Ok(batch)
        })
    }
}

/// Runs `case` with the probe side held back until the build has finished
/// and a consumer of the same query took `pressure` bytes more than the
/// pool had free; returns the rows and the query.
async fn run_held(
    case: &Case,
    probe: &[super::Row],
    build: &[super::Row],
    limit: usize,
    pressure: usize,
) -> (Vec<String>, Arc<krishiv_exec::QueryContext>) {
    let root = root(limit);
    let (_manager, scope) = spill();
    let config = ExecConfig {
        join_row_layout: case.row_layout,
        join_idle_spill_after: Duration::ZERO,
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let reached = Arc::new(AtomicBool::new(false));
    let (open, gate) = tokio::sync::watch::channel(false);
    let held: Arc<dyn ExecNode> = Arc::new(Scan(Arc::new(Gated {
        inner: Batches::new(schema(), batches(&data(probe), 256)),
        reached: Arc::clone(&reached),
        handed: Arc::default(),
        open: gate,
    })));
    let plan = join(
        case.spec(Some(scope.clone())),
        scan(&schema(), batches(&data(build), 256)),
        held,
    );
    let running = tokio::spawn({
        let q = Arc::clone(&q);
        async move { run(&runner_for(2, limit), plan, &q).await }
    });
    while !reached.load(Ordering::Acquire) {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    // The build has finished and its probe waits: another driver of the
    // query needs more than is free.
    let owner = SpillOwner::new();
    let filler = ConsumerHandle::register(
        Arc::clone(q.pool()),
        "filler",
        Some(owner.signal() as Arc<dyn Spillable>),
    );
    let free = q.pool().available().unwrap_or(0);
    owner
        .reserve_for(&filler, free + pressure, || async {})
        .await
        .unwrap();
    let mut taken = filler.reservation();
    taken.try_grow(free + pressure).unwrap();
    drop(taken);
    drop(filler);
    open.send(true).unwrap();
    let out = running.await.unwrap().unwrap();
    assert_released(&root);
    assert_eq!(spill_files(&scope), 0, "spill files are deleted");
    (render(&out), q)
}

#[tokio::test(start_paused = true)]
async fn an_idle_build_spills_before_the_requester_and_joins_from_disk() {
    let build = skewed(6_000, 1_500);
    let probe = skewed(3_000, 2_000);
    for join_type in [JoinType::Inner, JoinType::RightOuter, JoinType::FullOuter] {
        for row_layout in [false, true] {
            let case = Case {
                join_type,
                keys: 2,
                null_safe: vec![false, false],
                filter: false,
                row_layout,
            };
            let (rows, q) = run_held(&case, &probe, &build, 16 * MIB, 512 << 10).await;
            assert_eq!(rows, case.oracle(&probe, &build), "{case:?}");
            assert!(
                metric(&q, "HashJoinProbe", "idle_spills") > 0,
                "the idle build spilled ({case:?})"
            );
            assert!(metric(&q, "HashJoinProbe", "idle_spilled_bytes") > 0);
            assert!(metric(&q, "HashJoinProbe", "grace_partitions") > 0);
        }
    }
}

/// The deterministic executor, except that the first blocking job after
/// [`HeldWrite::arm`] waits for [`HeldWrite::release`]. The job waits on a
/// task of its own, so dropping the future that awaits it detaches the
/// job, as dropping the await of a running blocking job does.
#[derive(Debug)]
struct HeldWrite {
    inner: Arc<TokioExecutor>,
    armed: AtomicBool,
    entered: tokio::sync::watch::Sender<bool>,
    released: tokio::sync::watch::Sender<bool>,
    finished: tokio::sync::watch::Sender<bool>,
}

impl HeldWrite {
    fn new(slots: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: TokioExecutor::deterministic(slots).unwrap(),
            armed: AtomicBool::new(false),
            entered: tokio::sync::watch::channel(false).0,
            released: tokio::sync::watch::channel(false).0,
            finished: tokio::sync::watch::channel(false).0,
        })
    }

    fn arm(&self) {
        self.armed.store(true, Ordering::Release);
    }

    /// Waits until the held job has been handed to the executor.
    async fn entered(&self) {
        let mut entered = self.entered.subscribe();
        let _ = entered.wait_for(|&e| e).await;
    }

    fn release(&self) {
        self.released.send_replace(true);
    }

    /// Waits until the held job has run (released, it runs even when
    /// nothing awaits it any more).
    async fn finished(&self) {
        let mut finished = self.finished.subscribe();
        let _ = finished.wait_for(|&f| f).await;
    }
}

impl Executor for HeldWrite {
    fn spawn(&self, class: WorkloadClass, fut: BoxFuture<'static, ()>) -> TaskHandle {
        self.inner.spawn(class, fut)
    }

    fn spawn_blocking(
        &self,
        f: Box<dyn FnOnce() + Send + 'static>,
    ) -> BoxFuture<'static, Result<()>> {
        if !self.armed.swap(false, Ordering::AcqRel) {
            return self.inner.spawn_blocking(f);
        }
        self.entered.send_replace(true);
        let mut released = self.released.subscribe();
        let inner = Arc::clone(&self.inner);
        let finished = self.finished.clone();
        let job = self.inner.compute_handle().spawn(async move {
            let _ = released.wait_for(|&r| r).await;
            let ran = inner.spawn_blocking(f).await;
            finished.send_replace(true);
            ran
        });
        Box::pin(async move { job.await.unwrap() })
    }

    fn io_handle(&self) -> &tokio::runtime::Handle {
        self.inner.io_handle()
    }

    fn compute_handle(&self) -> &tokio::runtime::Handle {
        self.inner.compute_handle()
    }

    fn parallelism(&self) -> usize {
        self.inner.parallelism()
    }

    fn now(&self) -> std::time::Instant {
        self.inner.now()
    }
}

/// What happens while the idle build's write is held.
#[derive(Debug, Clone, Copy)]
enum During {
    /// The probe starts: its first read seals the build.
    Seal,
    /// The query is cancelled.
    Cancel,
}

/// The outcome of [`run_write_held`].
struct Held {
    rows: Result<Vec<String>>,
    q: Arc<krishiv_exec::QueryContext>,
    /// Probe batches handed out before the write was released.
    handed: usize,
}

/// As [`run_held`], but the idle build's write is held at the executor
/// while `during` happens, then released.
async fn run_write_held(
    case: &Case,
    probe: &[super::Row],
    build: &[super::Row],
    limit: usize,
    pressure: usize,
    during: During,
) -> Held {
    let root = root(limit);
    let (_manager, scope) = spill();
    let config = ExecConfig {
        join_row_layout: case.row_layout,
        join_idle_spill_after: Duration::ZERO,
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let reached = Arc::new(AtomicBool::new(false));
    let handed = Arc::new(AtomicUsize::new(0));
    let (open, gate) = tokio::sync::watch::channel(false);
    let held: Arc<dyn ExecNode> = Arc::new(Scan(Arc::new(Gated {
        inner: Batches::new(schema(), batches(&data(probe), 256)),
        reached: Arc::clone(&reached),
        handed: Arc::clone(&handed),
        open: gate,
    })));
    let plan = join(
        case.spec(Some(scope.clone())),
        scan(&schema(), batches(&data(build), 256)),
        held,
    );
    let executor = HeldWrite::new(2);
    let runner = QueryRunner::new(Arc::clone(&executor) as Arc<dyn Executor>);
    let running = tokio::spawn({
        let q = Arc::clone(&q);
        async move { run(&runner, plan, &q).await }
    });
    while !reached.load(Ordering::Acquire) {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    // The build has finished and its probe waits. The next blocking job is
    // the idle build's write, asked for by a consumer that needs more than
    // is free; it waits for that write.
    executor.arm();
    let pressing = tokio::spawn({
        let q = Arc::clone(&q);
        async move {
            let owner = SpillOwner::new();
            let filler = ConsumerHandle::register(
                Arc::clone(q.pool()),
                "filler",
                Some(owner.signal() as Arc<dyn Spillable>),
            );
            let want = q.pool().available().unwrap_or(0) + pressure;
            if owner.reserve_for(&filler, want, || async {}).await.is_ok() {
                let mut taken = filler.reservation();
                let _ = taken.try_grow(want);
            }
        }
    });
    // The idle build's write starts (a timeout fails the test).
    tokio::time::timeout(Duration::from_secs(30), executor.entered())
        .await
        .unwrap();
    match during {
        During::Seal => {
            open.send(true).unwrap();
        }
        During::Cancel => q.cancel(),
    }
    // Paused time: everything that can run without the write runs.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let handed_before = handed.load(Ordering::Acquire);
    executor.release();
    // A cancelled query may have dropped the probe scan already.
    open.send_replace(true);
    pressing.await.unwrap();
    // The query ends (a timeout fails the test).
    let out = tokio::time::timeout(Duration::from_secs(60), running)
        .await
        .unwrap()
        .unwrap();
    // The held write runs to its end before the files are counted.
    tokio::time::timeout(Duration::from_secs(30), executor.finished())
        .await
        .unwrap();
    assert_released(&root);
    assert_eq!(spill_files(&scope), 0, "spill files are deleted");
    Held {
        rows: out.map(|b| render(&b)),
        q,
        handed: handed_before,
    }
}

fn held_cases() -> impl Iterator<Item = Case> {
    [JoinType::Inner, JoinType::FullOuter]
        .into_iter()
        .flat_map(|join_type| {
            [false, true].map(|row_layout| Case {
                join_type,
                keys: 2,
                null_safe: vec![false, false],
                filter: false,
                row_layout,
            })
        })
}

#[tokio::test(start_paused = true)]
async fn a_build_sealed_while_its_idle_write_is_in_flight_joins_from_disk() {
    let build = skewed(6_000, 1_500);
    let probe = skewed(3_000, 2_000);
    for case in held_cases() {
        let held = run_write_held(&case, &probe, &build, 16 * MIB, 512 << 10, During::Seal).await;
        assert!(
            held.handed > 0,
            "the probe read before the write landed ({case:?})"
        );
        assert_eq!(held.rows.unwrap(), case.oracle(&probe, &build), "{case:?}");
        assert!(
            metric(&held.q, "HashJoinProbe", "idle_spills") > 0,
            "{case:?}"
        );
        assert!(
            metric(&held.q, "HashJoinProbe", "grace_partitions") > 0,
            "{case:?}"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn a_query_cancelled_while_its_idle_write_is_in_flight_releases_everything() {
    let build = skewed(6_000, 1_500);
    let probe = skewed(3_000, 2_000);
    for case in held_cases() {
        let held = run_write_held(&case, &probe, &build, 16 * MIB, 512 << 10, During::Cancel).await;
        let err = held.rows.expect_err("the query was cancelled");
        assert!(
            err.to_string().contains("OPERATION_CANCELED"),
            "{err} ({case:?})"
        );
        assert_eq!(
            metric(&held.q, "HashJoinProbe", "idle_spills"),
            0,
            "the cancelled write never landed ({case:?})"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn a_join_that_can_spill_reports_its_idle_spills_at_zero() {
    let build = skewed(600, 150);
    let probe = skewed(300, 200);
    let case = Case {
        join_type: JoinType::Inner,
        keys: 2,
        null_safe: vec![false, false],
        filter: false,
        row_layout: false,
    };
    let root = root(16 * MIB);
    let (_manager, scope) = spill();
    let q = query(&root, 1, ExecConfig::default());
    let plan = join(
        case.spec(Some(scope)),
        scan(&schema(), batches(&data(&build), 256)),
        scan(&schema(), batches(&data(&probe), 256)),
    );
    let out = run(&runner_for(2, 16 * MIB), plan, &q).await.unwrap();
    assert_eq!(render(&out), case.oracle(&probe, &build));
    let probe_metrics: Vec<_> = q
        .metrics()
        .operators
        .values()
        .filter(|e| e.name == "HashJoinProbe")
        .flat_map(|e| {
            e.extra
                .iter()
                .map(|(n, _)| n.to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    for name in ["idle_spills", "idle_spilled_bytes"] {
        assert!(
            probe_metrics.iter().any(|n| n == name),
            "{name} in {probe_metrics:?}"
        );
    }
    assert_eq!(metric(&q, "HashJoinProbe", "idle_spills"), 0);
}
