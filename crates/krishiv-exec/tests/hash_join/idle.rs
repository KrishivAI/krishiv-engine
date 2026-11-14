// SPDX-License-Identifier: Apache-2.0
//! A finished build that waits for its probe is spilled before a running
//! consumer of the query spills its own state (study 30, M1 + M2): the
//! probe then reads the written partitions back, and the answer is the
//! in-memory one.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_exec::testing::{Batches, BlockingGate, Break, Buffer, Scan};
use krishiv_exec::{ExecConfig, ExecNode, GlobalState, LocalState, Result, Source, TaskContext};
use krishiv_memory::{ConsumerHandle, MemoryPool, SpillOwner, Spillable};
use krishiv_types::batch::Batch;

use super::util::*;
use super::{Case, JoinType, data, schema, skewed};

/// A scan that hands out nothing until the test opens it.
struct Gated {
    inner: Arc<Batches>,
    reached: Arc<AtomicBool>,
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
            self.inner.get_data(local, global).await
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
    let (rows, q, _) = run_held_with(case, probe, build, limit, pressure, None).await;
    (rows, q)
}

/// What the pool held while the sink above the probe finalized, and the
/// bytes that sink had reserved for its buffered rows.
struct AtFinalize {
    held: usize,
    sink_bytes: u64,
}

/// As [`run_held`]; with a `finalize_gate`, the join runs below a sink whose
/// `finalize` waits at it (on real threads: the deterministic executor
/// runs a blocking call on the test's own thread), and the result also
/// says what the pool held while it waited.
async fn run_held_with(
    case: &Case,
    probe: &[super::Row],
    build: &[super::Row],
    limit: usize,
    pressure: usize,
    finalize_gate: Option<Arc<BlockingGate>>,
) -> (
    Vec<String>,
    Arc<krishiv_exec::QueryContext>,
    Option<AtFinalize>,
) {
    const HANG: Duration = Duration::from_secs(60);
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
        open: gate,
    })));
    let plan = join(
        case.spec(Some(scope.clone())),
        scan(&schema(), batches(&data(build), 256)),
        held,
    );
    let (plan, runner) = match &finalize_gate {
        Some(gate) => {
            let sink = Buffer::with_finalize_gate(Arc::clone(gate));
            let plan: Arc<dyn ExecNode> = Arc::new(Break(plan, sink, false));
            (plan, runner(2))
        }
        None => (plan, runner_for(2, limit)),
    };
    let running = tokio::spawn({
        let q = Arc::clone(&q);
        async move { run(&runner, plan, &q).await }
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
    let held = match &finalize_gate {
        Some(gate) => {
            let reached = tokio::time::timeout(HANG, async {
                while !gate.reached() {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            })
            .await;
            let held = root.reserved();
            gate.open(); // before asserting, so a failure never leaves it shut
            assert!(reached.is_ok(), "the sink above the probe finalized");
            Some(held)
        }
        None => None,
    };
    let out = running.await.unwrap().unwrap();
    assert_released(&root);
    assert_eq!(spill_files(&scope), 0, "spill files are deleted");
    let at_finalize = held.map(|held| AtFinalize {
        held,
        sink_bytes: out
            .iter()
            .map(|b| b.data().get_array_memory_size() as u64)
            .sum(),
    });
    (render(&out), q, at_finalize)
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

/// An idle build that spilled some level-0 partitions after its finalize
/// still gives the in-memory ones back when its last probe driver is done:
/// while the sink above the probe finalizes, the pool holds that sink's
/// rows, not the build's tables (design `P2-q16-build-release`).
#[tokio::test]
async fn an_idle_spilled_build_goes_back_before_the_sink_above_the_probe_finalizes() {
    let build = skewed(6_000, 1_500);
    let probe = skewed(3_000, 2_000);
    for join_type in [JoinType::Inner, JoinType::RightOuter] {
        let case = Case {
            join_type,
            keys: 2,
            null_safe: vec![false, false],
            filter: false,
            row_layout: false,
        };
        let gate = BlockingGate::new();
        let (rows, q, at) =
            run_held_with(&case, &probe, &build, 16 * MIB, 512 << 10, Some(gate)).await;
        let at = at.unwrap();
        assert_eq!(rows, case.oracle(&probe, &build), "{case:?}");
        assert!(
            metric(&q, "HashJoinProbe", "idle_spills") > 0,
            "the idle build spilled ({case:?})"
        );
        let build_bytes = metric(&q, "HashJoinBuild", "steady_reserved_bytes");
        let spilled = metric(&q, "HashJoinProbe", "idle_spilled_bytes");
        // Most of the build stayed in memory: the release has bytes to return.
        assert!(
            spilled < build_bytes / 2,
            "{case:?}: spilled {spilled} of {build_bytes}"
        );
        // Measured: released, the pool holds the sink's rows and some tens of
        // KiB more; with the release skipped, about 60% of the build more.
        assert!(
            (at.held as u64) < build_bytes,
            "{case:?}: held {} at the sink's finalize, build {build_bytes}",
            at.held
        );
        let beside_sink = (at.held as u64).saturating_sub(at.sink_bytes);
        assert!(
            beside_sink < build_bytes / 4,
            "{case:?}: held {beside_sink} beside the sink's {} bytes, build {build_bytes}",
            at.sink_bytes
        );
    }
}
