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
