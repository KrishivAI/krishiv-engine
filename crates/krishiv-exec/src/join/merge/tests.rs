// SPDX-License-Identifier: Apache-2.0
//! The sort-merge join source's metrics: its buckets' merges on the blocking
//! pool count on the source's own context.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillOwner, TreePool};
use krishiv_types::batch::Batch;

use super::SortMergeJoin;
use crate::context::{ExecConfig, OperatorId, QueryContext, QueryId, TaskContext};
use crate::executor::{Executor, TokioExecutor, WorkloadClass};
use crate::join::{JoinSpec, JoinType};
use crate::operator::{Sink, SinkStatus, Source};

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "the test runtime's `block_on`; never runs on a compute worker"
)]
async fn a_merge_joins_bucket_merges_count_on_its_context() {
    let executor: Arc<dyn Executor> = TokioExecutor::new(&crate::ExecutorConfig {
        compute_threads: 1,
        io_threads: 1,
        blocking_threads: 1,
        io_blocking_threads: 1,
        release_when_idle: false,
    })
    .unwrap_or_else(|e| unreachable!("{e}"));
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)]));
    let join = SortMergeJoin::try_new(JoinSpec {
        join_type: JoinType::Inner,
        probe_schema: Arc::clone(&schema),
        build_schema: Arc::clone(&schema),
        probe_keys: vec![0],
        build_keys: vec![0],
        null_safe: vec![false],
        filter: None,
        single_match: false,
        spill: None,
    })
    .unwrap();
    let root = TreePool::new(&MemoryConfig::default());
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig {
            batch_size: 1000,
            ..ExecConfig::default()
        },
        root.child("q", None),
    );
    let ctx = |op| {
        TaskContext::new(
            Arc::clone(&query),
            Arc::clone(&executor),
            SpillOwner::new(),
            OperatorId(op),
            "SortMergeJoin",
        )
    };
    let mut sorted = Vec::new();
    for (op, sort) in [(0, &join.left_sort), (1, &join.right_sort)] {
        let sink_ctx = ctx(op);
        let global = Sink::global_state(sort.as_ref(), &sink_ctx).unwrap();
        let mut local = Sink::local_state(sort.as_ref(), &sink_ctx, &global).unwrap();
        for b in 0..10_i64 {
            let k = (b * 1000..(b + 1) * 1000).map(|i| (i * 2_654_435_761) % 4_001);
            let k: ArrayRef = Arc::new(Int64Array::from_iter_values(k));
            let batch = Batch::new(RecordBatch::try_new(Arc::clone(&schema), vec![k]).unwrap());
            assert!(matches!(
                sort.sink(batch, &mut local, &global).unwrap(),
                SinkStatus::NeedMoreInput
            ));
        }
        sort.combine(local, &global).unwrap();
        sort.finalize(&global, &sink_ctx).await.unwrap();
        sorted.push((global, sink_ctx));
    }
    let source_ctx = ctx(2)
        .with_dependency(Arc::clone(&sorted[0].0))
        .with_second_dependency(Arc::clone(&sorted[1].0));
    let global = join.global_state(&source_ctx).unwrap();
    let mut local = join.local_state(&source_ctx, &global).unwrap();
    let mut rows = 0;
    while let Some(b) = join.get_data(&mut local, &global).await.unwrap() {
        rows += b.num_rows();
    }
    assert!(rows > 0);
    assert!(
        source_ctx.blocking_time() > std::time::Duration::ZERO,
        "the bucket merges ran off the join source's counter"
    );
    drop((local, global, source_ctx, sorted));
    assert_eq!(root.reserved(), 0);
}
