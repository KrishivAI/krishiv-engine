// SPDX-License-Identifier: Apache-2.0
//! Unit tests of the pieces the end-to-end tests do not reach directly.

use std::sync::Arc;

use futures::FutureExt;
use krishiv_memory::{MemoryConfig, MemoryPool, SpillOwner, Spillable, TreePool};

use crate::channel::{Channel, ResultSink};
use crate::testing::{Buffer, Count, Filter, Limit, PassThrough, Pipe, RangeSource, Repeat, Scan};
use crate::{
    ExecConfig, ExecutorConfig, MorselCursor, Operator, OperatorId, QueryContext, QueryId, Sink,
    SinkStatus, Source, TaskContext, TokioExecutor, WorkloadClass,
};

fn ctx() -> (Arc<TreePool>, TaskContext) {
    let root = TreePool::new(&MemoryConfig::default());
    let q = QueryContext::new(
        QueryId(3),
        WorkloadClass::Agent,
        ExecConfig::default(),
        root.child("q", None),
    );
    let exec = TokioExecutor::new(&ExecutorConfig::default());
    let exec = exec.unwrap_or_else(|e| unreachable!("{e}"));
    (
        root,
        TaskContext::new(q, exec, SpillOwner::new(), OperatorId(9), "op"),
    )
}

#[test]
fn accessors_debug_and_defaults() {
    let (root, ctx) = ctx();
    assert_eq!(
        (ctx.op(), ctx.query().id(), ctx.query().class()),
        (OperatorId(9), QueryId(3), WorkloadClass::Agent)
    );
    assert!(ctx.executor().parallelism() >= 1);
    assert!(format!("{ctx:?}").contains("OperatorId(9)"));
    assert!(format!("{:?}", ctx.query()).contains("cancelled: false"));
    let m = ctx.consumer();
    assert_eq!(m.id(), ctx.consumer().id());
    assert!(format!("{m:?}").contains("MemoryHandle"));
    let cfg = ExecutorConfig::default();
    assert_eq!(cfg.compute_threads, cfg.blocking_threads);
    let cursor = MorselCursor::new(0, Some(1));
    assert!(cursor.is_empty() && cursor.claim().is_none());
    for status in [
        SinkStatus::NeedMoreInput,
        SinkStatus::Finished,
        SinkStatus::NeedMemory { bytes: 1 },
    ] {
        assert!(!format!("{status:?}").is_empty());
    }
    assert_eq!(
        format!("{:?}", SinkStatus::Blocked(Box::pin(async {}))),
        "Blocked"
    );
    for s in [
        RangeSource::new(1).schema(),
        Count::new().schema(),
        Buffer::new(true, None).schema(),
    ] {
        assert_eq!(s.fields().len(), 1);
    }
    for op in [
        Filter::new(|_| true) as Arc<dyn Operator>,
        Arc::new(PassThrough),
        Limit::new(1),
    ] {
        assert_eq!(op.schema().fields().len(), 1);
    }
    let repeat = Repeat::new(4, 1).unwrap_or_else(|e| unreachable!("{e}"));
    assert_eq!(repeat.schema().fields().len(), 1);
    let scan = format!(
        "{:?} {:?}",
        Scan(RangeSource::new(1)),
        Pipe(Arc::new(Scan(repeat)), Arc::new(PassThrough))
    );
    assert_eq!(scan, "Scan Pipe");
    let b = crate::PipelineBuilder::new(Arc::new(Count::default()));
    assert!(format!("{b:?}").contains("pipelines: 1"));
    drop(ctx);
    assert_eq!(root.reserved(), 0);
}

#[test]
fn default_spill_hooks_free_nothing() {
    let (_root, ctx) = ctx();
    let mut local: crate::LocalState = Box::new(());
    let op = PassThrough;
    assert_eq!(
        op.spill(&mut local, 10, &ctx)
            .now_or_never()
            .map(Result::ok),
        Some(Some(0))
    );
    let sink = Count::default();
    assert_eq!(
        Sink::spill(&sink, &mut local, 10, &ctx)
            .now_or_never()
            .map(Result::ok),
        Some(Some(0))
    );
    // A source's default parallelism is one driver.
    let g: crate::GlobalState = Arc::new(());
    assert_eq!(Source::max_parallelism(&sink, &g), 1);
}

#[test]
fn a_gone_receiver_finishes_the_result_sink() {
    let (_root, ctx) = ctx();
    let channel = Channel::new(4);
    let sink = ResultSink {
        channel: Arc::clone(&channel),
    };
    let g = sink
        .global_state(&ctx)
        .unwrap_or_else(|e| unreachable!("{e}"));
    let mut local = sink
        .local_state(&ctx, &g)
        .unwrap_or_else(|e| unreachable!("{e}"));
    let batch = Repeat::new(1, 1).unwrap_or_else(|e| unreachable!("{e}"));
    let mut l2: crate::LocalState = Box::new(());
    let one = batch
        .get_data(&mut l2, &g)
        .now_or_never()
        .and_then(Result::ok)
        .flatten();
    let one = one.unwrap_or_else(|| unreachable!());
    channel.drop_receiver();
    let status = sink.sink(one, &mut local, &g);
    assert!(matches!(status, Ok(SinkStatus::Finished)));
    let space = Arc::clone(&channel).space();
    assert_eq!(
        space.now_or_never(),
        Some(()),
        "a gone receiver never blocks"
    );
    // Spill requests to an operator without a spill hook complete with 0.
    let signal = Arc::clone(ctx.consumer().signal());
    let ticket = signal.request_spill(1);
    signal.complete(0);
    assert_eq!(ticket.now_or_never(), Some(0));
}

/// A query without a limit of its own reports the process limit it runs
/// under (the profile's `limit_bytes`), and a tighter query limit wins.
#[test]
fn metrics_report_the_limit_the_query_runs_under() {
    let root = TreePool::new(&MemoryConfig {
        limit: Some(64 << 20),
        ..MemoryConfig::default()
    });
    let query = |limit| {
        QueryContext::new(
            QueryId(4),
            WorkloadClass::Agent,
            ExecConfig::default(),
            root.child("q", limit),
        )
    };
    assert_eq!(query(None).metrics().memory.limit, Some(64 << 20));
    assert_eq!(query(Some(8 << 20)).metrics().memory.limit, Some(8 << 20));
    let unbounded = TreePool::new(&MemoryConfig {
        limit: None,
        ..MemoryConfig::default()
    });
    let q = QueryContext::new(
        QueryId(5),
        WorkloadClass::Agent,
        ExecConfig::default(),
        unbounded.child("q", None),
    );
    assert_eq!(q.metrics().memory.limit, None);
}

/// A cancellation with a reason ends the query and its drivers with the
/// reason's error; the first cancellation or failure wins, and a finished
/// query is never cancelled.
#[test]
fn a_cancellation_with_a_reason_is_the_error_of_the_query() {
    use krishiv_common::error::{InternalError, KrishivError};
    let root = TreePool::new(&MemoryConfig::default());
    let query = |n| {
        QueryContext::new(
            QueryId(n),
            WorkloadClass::Interactive,
            ExecConfig::default(),
            root.child("q", None),
        )
    };
    let reason = |m: &str| -> KrishivError {
        InternalError {
            message: m.to_owned(),
        }
        .into()
    };
    let q = query(1);
    assert!(q.check_cancelled().is_ok());
    assert!(q.cancel_with(reason("first")));
    assert!(!q.cancel_with(reason("second")), "the first reason wins");
    q.cancel();
    let seen = q.check_cancelled().unwrap_err();
    assert!(seen.to_string().contains("first"), "{seen}");
    assert_eq!(q.cancellation().to_string(), seen.to_string());
    assert!(q.error().unwrap().to_string().contains("first"));
    // A plain cancel first: drivers see OPERATION_CANCELED, and a later
    // reason changes nothing.
    let q = query(2);
    q.cancel();
    assert!(!q.cancel_with(reason("late")));
    assert_eq!(
        q.check_cancelled().unwrap_err().error_class().name(),
        "OPERATION_CANCELED"
    );
    // Finished: nothing to cancel.
    let q = query(3);
    q.mark_finished();
    assert!(!q.cancel_with(reason("after")));
    assert!(q.error().is_none());
}

/// Lanes partition the claim ordinals: each ordinal belongs to exactly one lane, in lane order, and
/// the seeded permutation maps ordinals to morsels as the shared claim does.
#[test]
fn strided_claims_deal_every_morsel_to_exactly_one_lane() {
    for seed in [None, Some(7)] {
        let cursor = MorselCursor::new(10, seed);
        let mut seen = Vec::new();
        for lane in 0..3 {
            let got: Vec<_> = (0..)
                .map_while(|j| cursor.claim_strided(lane, 3, j))
                .collect();
            assert!(got.iter().all(|(o, _)| *o % 3 == lane as u64));
            assert!(got.windows(2).all(|w| w[0].0 < w[1].0));
            seen.extend(got);
        }
        seen.sort_unstable();
        let shared: Vec<_> = {
            let c = MorselCursor::new(10, seed);
            std::iter::from_fn(|| c.claim_ordered()).collect()
        };
        assert_eq!(seen, shared);
    }
    let cursor = MorselCursor::new(4, None);
    assert!(cursor.claim_strided(0, 0, 0).is_none() && cursor.claim_strided(3, 3, 0).is_none());
    assert!(cursor.claim_strided(0, 2, usize::MAX).is_none());
}
