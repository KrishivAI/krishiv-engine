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
use krishiv_exec::testing::{Batches, Scan};
use krishiv_exec::{ExecConfig, ExecNode, GlobalState, LocalState, Result, Source, TaskContext};
use krishiv_memory::{ConsumerHandle, SpillOwner, Spillable};
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
