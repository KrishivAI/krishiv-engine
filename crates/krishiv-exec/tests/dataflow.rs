// SPDX-License-Identifier: Apache-2.0
//! The dataflow pipeline scheduler (`krishiv.exec.dataflowScheduling`): a
//! pipeline starts once the pipelines it reads have ended, not when its whole
//! wave has; a failure or cancellation while other groups still run drains
//! them, and the query's memory is released; results equal the wave runner's.
#![cfg(not(loom))]
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_schema::SchemaRef;
use futures::StreamExt;
use krishiv_common::error::{DivideByZero, ErrorClass};
use krishiv_exec::testing::{Break, Buffer, Count, Pipe, RangeSource, Scan, schema};
use krishiv_exec::{
    DEFAULT_BATCH_SIZE, ExecConfig, ExecNode, ExecutorConfig, GlobalState, LocalState, Operator,
    OperatorOutput, PipelineBuilder, QueryContext, QueryId, QueryRunner, Result, Sink, Source,
    TaskContext, TokioClock, TokioExecutor, WorkloadClass,
};
use krishiv_memory::{BoxFuture, MemoryConfig, MemoryPool, TreePool};
use krishiv_types::batch::Batch;
use tokio::sync::watch;

fn root() -> Arc<TreePool> {
    TreePool::with_clock(
        &MemoryConfig {
            limit: Some(1 << 30),
            quantum: 64 << 10,
            spill_wait_timeout: Duration::from_secs(2),
            abort_release_timeout: Duration::from_secs(30),
            spill_scratch_reserve: krishiv_memory::DEFAULT_SPILL_SCRATCH_RESERVE,
            resident_guard: true,
        },
        Arc::new(TokioClock),
    )
}

fn query(root: &TreePool, id: u64, dataflow: bool) -> Arc<QueryContext> {
    let config = ExecConfig {
        dataflow_scheduling: dataflow,
        fail_on_stall: true,
        stall_threshold: Duration::from_millis(500),
        ..ExecConfig::default()
    };
    QueryContext::new(
        QueryId(id),
        WorkloadClass::Interactive,
        config,
        root.child(&format!("q{id}"), None),
    )
}

fn runner() -> QueryRunner {
    QueryRunner::new(
        TokioExecutor::new(&ExecutorConfig {
            compute_threads: 4,
            io_threads: 1,
            blocking_threads: 4,
            io_blocking_threads: 2,
            release_when_idle: true,
        })
        .unwrap(),
    )
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

fn assert_released(root: &TreePool) {
    assert_eq!(
        root.reserved(),
        0,
        "the query's memory is released at its end"
    );
}

/// A range scan that passes `open_after` batches, then waits (async, so no
/// driver stalls) until the test opens its gate.
struct GatedScan {
    inner: Arc<RangeSource>,
    open_after: u64,
    served: AtomicU64,
    gate: watch::Receiver<bool>,
}

impl GatedScan {
    fn new(rows: u64, open_after: u64) -> (Arc<Self>, watch::Sender<bool>) {
        let (tx, gate) = watch::channel(false);
        let scan = Arc::new(Self {
            inner: RangeSource::new(rows),
            open_after,
            served: AtomicU64::new(0),
            gate,
        });
        (scan, tx)
    }
}

impl Source for GatedScan {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        self.inner.global_state(ctx)
    }

    fn max_parallelism(&self, g: &GlobalState) -> usize {
        self.inner.max_parallelism(g)
    }

    fn local_state(&self, ctx: &TaskContext, g: &GlobalState) -> Result<LocalState> {
        self.inner.local_state(ctx, g)
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        g: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(async move {
            if self.served.fetch_add(1, Ordering::AcqRel) >= self.open_after {
                let mut gate = self.gate.clone();
                // A dropped sender opens the gate too.
                let _ = gate.wait_for(|open| *open).await;
            }
            self.inner.get_data(local, g).await
        })
    }
}

/// Two independent child pipelines: `held` into `held_sink` (read by
/// nobody) and `read` into a buffer, which the root pipeline then reads.
/// Under waves the root waits for both; under dataflow only for `read`.
struct Two {
    held: Arc<dyn ExecNode>,
    held_sink: Arc<dyn Sink>,
    read: Arc<dyn ExecNode>,
    buffer: Arc<Buffer>,
}

impl ExecNode for Two {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        b.child_pipeline(Arc::clone(&self.held_sink), Arc::clone(&self.held))?;
        let read = b.child_pipeline(
            Arc::clone(&self.buffer) as Arc<dyn Sink>,
            Arc::clone(&self.read),
        )?;
        b.set_breaker_source(Arc::clone(&self.buffer) as Arc<dyn Source>, read)
            .map(drop)
    }
}

fn scan(rows: u64) -> Arc<dyn ExecNode> {
    Arc::new(Scan(RangeSource::new(rows)))
}

/// Fails its pipeline at the first batch.
struct Fail;

impl Operator for Fail {
    fn schema(&self) -> SchemaRef {
        schema()
    }

    fn local_state(&self, _ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn execute(&self, _input: &Batch, _local: &mut LocalState) -> Result<OperatorOutput> {
        Err(DivideByZero {
            config: String::new(),
        }
        .into())
    }
}

/// The plan of the tests: a gated `held` pipeline beside `read`.
fn two(
    held_rows: u64,
    open_after: u64,
    read: Arc<dyn ExecNode>,
) -> (Arc<Two>, watch::Sender<bool>, Arc<Count>) {
    let (gated, gate) = GatedScan::new(held_rows, open_after);
    let count = Count::new();
    let plan = Arc::new(Two {
        held: Arc::new(Scan(gated)),
        held_sink: Arc::clone(&count) as Arc<dyn Sink>,
        read,
        buffer: Buffer::new(false, None),
    });
    (plan, gate, count)
}

/// Two independent chains under one root: `held` (gated) into `held_count`,
/// and `scan -> c1 -> c2` (two breakers) into the root's buffer. The root
/// reads both, but `c2`'s pipeline reads only `c1`: under waves it waits for
/// the wave holding `held`, under dataflow only for `c1`.
fn chain(open_after: u64) -> (Arc<Two>, watch::Sender<bool>, Arc<Count>, Arc<Count>) {
    let c2 = Count::new();
    let read: Arc<dyn ExecNode> = Arc::new(Break(
        Arc::new(Break(scan(10_000), Count::new(), false)),
        Arc::clone(&c2),
        false,
    ));
    let (plan, gate, held_count) = two(8, open_after, read);
    (plan, gate, held_count, c2)
}

/// Whether `count` finalizes within `within`.
async fn finalizes(count: &Count, within: Duration) -> bool {
    tokio::time::timeout(within, async {
        while count.finalized.load(Ordering::Acquire) == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .is_ok()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pipeline_starts_when_its_own_inputs_end() {
    let runner = runner();
    let root = root();
    let (plan, gate, held, c2) = chain(0);
    let q = query(&root, 1, true);
    let mut stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
    // The held pipeline has not ended (its gate is shut), yet `c2`'s
    // pipeline, two breakers above an unrelated scan, already ran.
    assert!(
        finalizes(&c2, Duration::from_secs(30)).await,
        "c2's pipeline starts before the held one ends"
    );
    assert_eq!(held.finalized.load(Ordering::Acquire), 0);
    gate.send(true).unwrap();
    let mut out = Vec::new();
    while let Some(b) = stream.next().await {
        out.push(b.unwrap());
    }
    drop(stream);
    q.finished().await;
    assert_eq!(values(&out), [2, 10_000 + 49_995_000]);
    assert_eq!(
        held.finalized.load(Ordering::Acquire),
        1,
        "the held pipeline ran to its end"
    );
    assert_released(&root);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_wave_runner_holds_the_chain_until_its_wave_ends() {
    let runner = runner();
    let root = root();
    let (plan, gate, held, c2) = chain(0);
    let q = query(&root, 1, false);
    let mut stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
    assert!(
        !finalizes(&c2, Duration::from_millis(300)).await,
        "under waves c2's pipeline waits for the wave holding the held pipeline"
    );
    gate.send(true).unwrap();
    let mut out = Vec::new();
    while let Some(b) = stream.next().await {
        out.push(b.unwrap());
    }
    drop(stream);
    q.finished().await;
    assert_eq!(values(&out), [2, 10_000 + 49_995_000]);
    assert_eq!(held.finalized.load(Ordering::Acquire), 1);
    assert_released(&root);
}

/// A failing group while another is still running: the first error wins,
/// nothing else starts, the running group is drained and every byte is
/// released (the held pipeline holds batches in its buffer when it stops).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failure_drains_the_running_groups() {
    let runner = runner();
    let root = root();
    let failing: Arc<dyn ExecNode> =
        Arc::new(Pipe(scan(DEFAULT_BATCH_SIZE as u64), Arc::new(Fail)));
    let (gated, gate) = GatedScan::new(64 * DEFAULT_BATCH_SIZE as u64, 4);
    let held_buffer = Buffer::new(false, None);
    let plan = Arc::new(Two {
        held: Arc::new(Scan(gated)),
        held_sink: Arc::clone(&held_buffer) as Arc<dyn Sink>,
        read: failing,
        buffer: Buffer::new(false, None),
    });
    let q = query(&root, 1, true);
    let mut stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
    let mut error = None;
    while let Some(b) = stream.next().await {
        if let Err(e) = b {
            error = Some(e);
        }
    }
    drop(stream);
    q.finished().await;
    assert_eq!(error.unwrap().error_class(), ErrorClass::DivideByZero);
    assert_released(&root);
    drop(gate);
}

/// Cancelling while the held group waits and the other chain has ended
/// drains the held group and releases everything.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cancel_drains_the_running_groups() {
    let runner = runner();
    let root = root();
    let (plan, _gate, held, c2) = chain(0);
    let q = query(&root, 1, true);
    let mut stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
    assert!(finalizes(&c2, Duration::from_secs(30)).await);
    q.cancel();
    while stream.next().await.is_some() {}
    drop(stream);
    q.finished().await;
    assert_eq!(
        q.error().unwrap().error_class(),
        ErrorClass::OperationCanceled
    );
    assert_eq!(held.finalized.load(Ordering::Acquire), 0);
    assert_released(&root);
}

/// Nested breakers and independent pipelines give the same rows either way.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn results_equal_the_wave_runners() {
    let runner = runner();
    let root = root();
    let mut results = Vec::new();
    for (id, dataflow) in [(1, false), (2, true)] {
        let read: Arc<dyn ExecNode> = Arc::new(Break(
            Arc::new(Break(scan(10_000), Count::new(), false)),
            Count::new(),
            false,
        ));
        let (plan, gate, _) = two(5_000, u64::MAX, read);
        let q = query(&root, id, dataflow);
        let mut stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
        let mut out = Vec::new();
        while let Some(b) = stream.next().await {
            out.push(b.unwrap());
        }
        drop(stream);
        q.finished().await;
        drop(gate);
        results.push(values(&out));
    }
    assert_eq!(results[0], results[1]);
    // [count, sum] over the inner count's two rows, 10_000 and 49_995_000.
    assert_eq!(results[1], [2, 10_000 + 49_995_000]);
    assert_released(&root);
}
