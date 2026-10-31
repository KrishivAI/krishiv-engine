// SPDX-License-Identifier: Apache-2.0
//! Unit tests of the sort internals: key canonicalization, per-batch order,
//! the merge's tie-break, weights through `interleave_batches`, and the
//! global spill driven directly.
#![expect(
    clippy::disallowed_methods,
    reason = "test-only temp-dir cleanup; never runs on a compute worker"
)]

use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_common::error::ErrorClass;
use krishiv_memory::{
    ConsumerHandle, MemoryConfig, MemoryPool, SpillConfig, SpillManager, SpillOwner, TreePool,
};
use krishiv_types::batch::{Batch, Weights};

use super::Sort;
use super::keys::{KeyEncoder, SortKey};
use super::merge::{
    Bucketing, Merger, Slice, SortedBatch, interleave_batches, merge_part, merge_splitters,
    parallel_merge, sort_slices,
};
use crate::context::{ExecConfig, OperatorId, QueryContext, QueryId, TaskContext};
use crate::executor::{Executor, TokioExecutor, WorkloadClass};
use crate::operator::{Sink, Source};

fn floats() -> (SchemaRef, Batch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("d", DataType::Float64, true),
        Field::new("f", DataType::Float32, true),
        Field::new("id", DataType::Int32, false),
    ]));
    let d = Float64Array::from(vec![
        Some(-f64::NAN),
        Some(0.0),
        Some(-0.0),
        Some(f64::INFINITY),
        None,
        Some(f64::NAN),
        Some(-1.0),
    ]);
    let f = Float32Array::from(vec![
        Some(-f32::NAN),
        Some(0.0),
        Some(-0.0),
        Some(f32::INFINITY),
        None,
        Some(f32::NAN),
        Some(-1.0),
    ]);
    let id = Int32Array::from_iter_values(0..7);
    let columns: Vec<ArrayRef> = vec![Arc::new(d), Arc::new(f), Arc::new(id)];
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).map(Batch::new);
    (schema, batch.unwrap_or_else(|e| unreachable!("{e}")))
}

fn ids(batch: &Batch) -> Vec<i32> {
    batch
        .data()
        .column(2)
        .as_primitive::<Int32Type>()
        .values()
        .to_vec()
}

#[test]
fn floats_order_as_in_spark() {
    let (schema, batch) = floats();
    for column in [0, 1] {
        let keys = KeyEncoder::new(&schema, &[SortKey::asc(column)]).unwrap();
        let rows = keys.encode(batch.data()).unwrap();
        // NaN = -NaN, both above +inf; -0.0 = 0.0.
        assert_eq!(rows.row(0), rows.row(5));
        assert!(rows.row(0) > rows.row(3));
        assert_eq!(rows.row(1), rows.row(2));
        let sorted = SortedBatch::new(batch.clone(), &keys, None).unwrap();
        // Ties stay in row order.
        assert_eq!(sorted.order, [4, 6, 1, 2, 3, 0, 5]);
        let desc = KeyEncoder::new(&schema, &[SortKey::desc(column)]).unwrap();
        let sorted = SortedBatch::new(batch.clone(), &desc, None).unwrap();
        assert_eq!(sorted.order, [0, 5, 3, 1, 2, 6, 4]);
    }
}

/// Decimals by value, binaries by unsigned bytes (orders from Spark 4.2.0).
#[test]
fn decimals_and_binaries_order_as_in_spark() {
    let d = arrow_array::Decimal128Array::from(vec![150, -100, 99_999, 0, -99_999]);
    let d: ArrayRef = Arc::new(d.with_precision_and_scale(5, 2).unwrap());
    let b = [vec![0x80], vec![0x7f, 0xff], vec![], vec![0], vec![0x7f]];
    let b: ArrayRef = Arc::new(arrow_array::BinaryArray::from_iter_values(b));
    let batch = Batch::new(RecordBatch::try_from_iter([("d", d), ("b", b)]).unwrap());
    for (column, order) in [(0, [4, 1, 3, 0, 2]), (1, [2, 3, 4, 1, 0])] {
        let keys = KeyEncoder::new(&batch.schema(), &[SortKey::asc(column)]).unwrap();
        assert_eq!(
            SortedBatch::new(batch.clone(), &keys, None).unwrap().order,
            order
        );
    }
}

#[test]
fn the_merge_breaks_ties_by_run_order() {
    let (schema, batch) = floats();
    let keys = KeyEncoder::new(&schema, &[SortKey::asc(0).nulls_first(false)]).unwrap();
    let runs: Vec<_> = [
        batch.slice(0, 4).unwrap(),
        batch.slice(4, 3).unwrap(),
        batch,
    ]
    .into_iter()
    .map(|b| Arc::new(SortedBatch::new(b, &keys, None).unwrap()))
    .collect();
    let root = TreePool::new(&MemoryConfig::default());
    let handle = ConsumerHandle::register(root.child("q", None), "sort", None);
    let mut merger = Merger::new(schema, keys, runs, &[], &handle).unwrap();
    let mut out = Vec::new();
    while let Some(b) = merger.next(3).unwrap() {
        assert!(b.num_rows() <= 3);
        out.extend(ids(&b));
    }
    // -1.0, ±0.0 ×4, inf ×2, NaN ×4 (run 0 before run 1 before run 2), nulls.
    assert_eq!(out, [6, 6, 1, 2, 1, 2, 3, 3, 0, 5, 0, 5, 4, 4]);
    assert!(merger.next(3).unwrap().is_none());
}

/// A merged batch is built from the batches it uses only (not every
/// buffered one): rows from constant-weight batches keep a constant weight.
#[test]
fn a_merged_batch_reads_only_the_batches_it_uses() {
    let (schema, batch) = floats();
    let keys = KeyEncoder::new(&schema, &[SortKey::asc(2)]).unwrap();
    let column = Weights::Column(Int64Array::from(vec![4, 5, 6]));
    let runs: Vec<_> = [(0, Weights::Const(2)), (4, column)]
        .into_iter()
        .map(|(start, w)| {
            let b = batch.slice(start, 3).unwrap();
            let b = Batch::with_weights(b.data().clone(), w).unwrap();
            Arc::new(SortedBatch::new(b, &keys, None).unwrap())
        })
        .collect();
    let root = TreePool::new(&MemoryConfig::default());
    let handle = ConsumerHandle::register(root.child("q", None), "sort", None);
    let mut merger = Merger::new(schema, keys, runs, &[], &handle).unwrap();
    let first = merger.next(3).unwrap().unwrap();
    assert_eq!(
        (ids(&first), first.weights()),
        (vec![0, 1, 2], &Weights::Const(2))
    );
    let rest = merger.next(3).unwrap().unwrap();
    assert_eq!(ids(&rest), [4, 5, 6]);
}

#[test]
fn interleaving_keeps_weights() {
    let (schema, batch) = floats();
    let column = Weights::Column(Int64Array::from_iter_values(0..7));
    let a = Batch::with_weights(batch.data().clone(), Weights::Const(2)).unwrap();
    let b = Batch::with_weights(batch.data().clone(), column).unwrap();
    let both = interleave_batches(&schema, &[a.clone(), b], &[(1, 6), (0, 0), (1, 3)]).unwrap();
    assert_eq!(ids(&both), [6, 0, 3]);
    assert_eq!(
        both.weights(),
        &Weights::Column(Int64Array::from(vec![6, 2, 3]))
    );
    let same = interleave_batches(&schema, &[a.clone(), a], &[(1, 1), (0, 2)]).unwrap();
    assert_eq!(same.weights(), &Weights::Const(2));
}

/// The global spill (during `finalize`) of two combined drivers' buffers,
/// then the source's merge of the spilled runs: weights and run order
/// survive; a sort without a spill scope frees nothing.
#[tokio::test(start_paused = true)]
async fn spill_global_spills_the_combined_buffers() {
    let (schema, batch) = floats();
    let weights = Weights::Column(Int64Array::from_iter_values(10..17));
    let batch = Batch::with_weights(batch.data().clone(), weights).unwrap();
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
    let root = TreePool::new(&MemoryConfig::default());
    let config = ExecConfig {
        batch_size: 3,
        ..ExecConfig::default()
    };
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        config,
        root.child("q", None),
    );
    let ctx = |op| {
        let owner = SpillOwner::new();
        TaskContext::new(
            Arc::clone(&query),
            Arc::clone(&executor),
            owner,
            OperatorId(op),
            "Sort",
        )
    };
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join(format!("sort-unit-global-{}", std::process::id()));
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.clone(),
        ..SpillConfig::default()
    })
    .unwrap();
    let keys = [SortKey::asc(2)];
    let unspillable = Sort::new(Arc::clone(&schema), &keys, None).unwrap();
    let sort = Sort::new(schema, &keys, Some(manager.scope().unwrap())).unwrap();
    let sink_ctx = ctx(0);
    let global = Sink::global_state(&sort, &sink_ctx).unwrap();
    for s in [&unspillable, &sort] {
        let mut local = Sink::local_state(s, &sink_ctx, &global).unwrap();
        assert_eq!(s.spill(&mut local, 1, &sink_ctx).await.unwrap(), 0);
        if std::ptr::eq(s, &sort) {
            for _ in 0..2 {
                let mut local = Sink::local_state(s, &sink_ctx, &global).unwrap();
                s.sink(batch.clone(), &mut local, &global).unwrap();
                s.combine(local, &global).unwrap();
            }
        }
    }
    assert_eq!(
        unspillable
            .spill_global(&global, 1, &sink_ctx)
            .await
            .unwrap(),
        0
    );
    assert!(sort.spill_global(&global, 1, &sink_ctx).await.unwrap() > 0);
    assert_eq!(sort.spill_global(&global, 1, &sink_ctx).await.unwrap(), 0);
    sort.finalize(&global, &sink_ctx).await.unwrap();
    assert_eq!(sort.metrics().runs_spilled, 2);

    let source_ctx = ctx(1).with_dependency(Arc::clone(&global));
    let source_global = Source::global_state(&sort, &source_ctx).unwrap();
    let mut local = Source::local_state(&sort, &source_ctx, &source_global).unwrap();
    let mut out = Vec::new();
    while let Some(b) = sort.get_data(&mut local, &source_global).await.unwrap() {
        assert!(b.num_rows() <= 3);
        out.push(b);
    }
    assert!(
        sort.get_data(&mut local, &source_global)
            .await
            .unwrap()
            .is_none()
    );
    // The final merge's run frames come out of its budget (three batch
    // sizes per run and an output batch; key rows may add a little), never
    // on top of it: topping the budget back up while the merger held its
    // frames reserved them twice (about 5 KB here instead of 2.7 KB).
    let batch_bytes = out.iter().map(|b| b.data().get_array_memory_size());
    let bound = 2 * 4 * batch_bytes.max().unwrap_or(0);
    assert!(source_ctx.consumer().consumer().peak() <= bound);
    let out = Batch::concat(&out[0].schema(), &out).unwrap();
    assert_eq!(ids(&out), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]);
    let w: Vec<i64> = (10..17).flat_map(|w| [w, w]).collect();
    assert_eq!(out.weights(), &Weights::Column(Int64Array::from(w)));
    assert_eq!(sort.metrics().final_fan_in, 2);
    drop((local, source_global, source_ctx, global, sink_ctx));
    assert_eq!(manager.used_bytes(), 0);
    assert_eq!(root.reserved(), 0);
    drop(manager);
    let _ = std::fs::remove_dir_all(dir);
}

/// Intermediate merges rewrite only as many runs as it takes to get down to
/// the fan-in: 16 runs at fan-in 8 merge 8, then 2 (10 runs rewritten, not
/// 16); 15 runs merge 8; a deep level merges full groups first.
#[test]
fn merge_groups_rewrite_no_more_than_needed() {
    use super::sort::group_size;
    let plan = |mut total: usize, fan_in: usize| {
        let mut groups = Vec::new();
        while total > fan_in {
            let g = group_size(total, fan_in);
            groups.push(g);
            total -= g - 1;
        }
        groups
    };
    assert_eq!(plan(16, 8), [8, 2]);
    assert_eq!(plan(15, 8), [8]);
    assert_eq!(plan(9, 8), [2]);
    assert_eq!(plan(3, 2), [2]);
    assert_eq!(plan(40, 4), [4; 12]);
}

/// A spill of a sorted result that fails (its spill directory is full)
/// fails the read: the reader gets the error, never a short result.
#[tokio::test(start_paused = true)]
async fn a_failed_spill_of_the_result_fails_the_read() {
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let batches: Vec<Batch> = (0..10_i64)
        .map(|b| {
            // Scattered values: the spilled rows do not compress away.
            let k = (b * 1000..(b + 1) * 1000).map(|i| (i * 2_654_435_761) % 1_000_003);
            let k: ArrayRef = Arc::new(Int64Array::from_iter_values(k));
            Batch::new(RecordBatch::try_new(Arc::clone(&schema), vec![k]).unwrap())
        })
        .collect();
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
    let root = TreePool::with_clock(
        &MemoryConfig {
            limit: Some(1 << 30),
            quantum: 1,
            ..MemoryConfig::default()
        },
        Arc::new(crate::executor::TokioClock),
    );
    let limit = 4 << 20;
    let config = ExecConfig {
        batch_size: 1000,
        ..ExecConfig::default()
    };
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        config,
        root.child("q", Some(limit)),
    );
    let ctx = |op| {
        TaskContext::new(
            Arc::clone(&query),
            Arc::clone(&executor),
            SpillOwner::new(),
            OperatorId(op),
            "Sort",
        )
    };
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join(format!("sort-unit-failed-{}", std::process::id()));
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.clone(),
        max_bytes: 4096,
        ..SpillConfig::default()
    })
    .unwrap();
    let sort = Sort::new(schema, &[SortKey::asc(0)], Some(manager.scope().unwrap())).unwrap();
    let sink_ctx = ctx(0);
    let global = Sink::global_state(&sort, &sink_ctx).unwrap();
    let mut local = Sink::local_state(&sort, &sink_ctx, &global).unwrap();
    for b in batches {
        assert!(matches!(
            sort.sink(b, &mut local, &global).unwrap(),
            crate::operator::SinkStatus::NeedMoreInput
        ));
    }
    sort.combine(local, &global).unwrap();
    sort.finalize(&global, &sink_ctx).await.unwrap();
    assert_eq!(sort.metrics().runs_spilled, 0, "sorted in memory");

    let source_ctx = ctx(1).with_dependency(Arc::clone(&global));
    let source_global = Source::global_state(&sort, &source_ctx).unwrap();
    let mut local = Source::local_state(&sort, &source_ctx, &source_global).unwrap();
    let first = sort.get_data(&mut local, &source_global).await.unwrap();
    assert_eq!(first.map(|b| b.num_rows()), Some(1000));
    // Another consumer of the query needs the memory the result holds: the
    // result's spill is asked for and fails on the full directory.
    let pool = Arc::clone(query.pool());
    let want = pool.register("want", None);
    let reserved = pool.reserve(want, 3_400_000).await;
    let failed = query
        .error()
        .unwrap_or_else(|| unreachable!("the spill failed"));
    assert_ne!(failed.error_class(), ErrorClass::UnableToAcquireMemory);
    let read = sort.get_data(&mut local, &source_global).await;
    let e = read.err().unwrap_or_else(|| unreachable!("a short result"));
    assert_eq!(e.error_class(), failed.error_class(), "{e}");
    drop(reserved);
    pool.unregister(want);
    drop((local, source_global, source_ctx, global, sink_ctx));
    assert_eq!(manager.used_bytes(), 0);
    assert_eq!(root.reserved(), 0);
    drop(manager);
    let _ = std::fs::remove_dir_all(dir);
}

/// A second spill request that reaches a sorted result after its rest was
/// spilled, and before the reader took the run (a request made on stale
/// spillable bytes), frees nothing and leaves the run to the reader: every
/// row is read.
#[tokio::test(start_paused = true)]
async fn a_late_spill_request_leaves_the_spilled_run_to_the_reader() {
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let batches: Vec<Batch> = (0..10_i64)
        .map(|b| {
            // Scattered values: the spilled rows do not compress away.
            let k = (b * 1000..(b + 1) * 1000).map(|i| (i * 2_654_435_761) % 1_000_003);
            let k: ArrayRef = Arc::new(Int64Array::from_iter_values(k));
            Batch::new(RecordBatch::try_new(Arc::clone(&schema), vec![k]).unwrap())
        })
        .collect();
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
    let root = TreePool::with_clock(
        &MemoryConfig {
            limit: Some(1 << 30),
            quantum: 1,
            ..MemoryConfig::default()
        },
        Arc::new(crate::executor::TokioClock),
    );
    let limit = 4 << 20;
    let config = ExecConfig {
        batch_size: 1000,
        ..ExecConfig::default()
    };
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        config,
        root.child("q", Some(limit)),
    );
    let ctx = |op| {
        TaskContext::new(
            Arc::clone(&query),
            Arc::clone(&executor),
            SpillOwner::new(),
            OperatorId(op),
            "Sort",
        )
    };
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join(format!("sort-unit-late-{}", std::process::id()));
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.clone(),
        ..SpillConfig::default()
    })
    .unwrap();
    let sort = Sort::new(schema, &[SortKey::asc(0)], Some(manager.scope().unwrap())).unwrap();
    let sink_ctx = ctx(0);
    let global = Sink::global_state(&sort, &sink_ctx).unwrap();
    let mut local = Sink::local_state(&sort, &sink_ctx, &global).unwrap();
    for b in batches {
        assert!(matches!(
            sort.sink(b, &mut local, &global).unwrap(),
            crate::operator::SinkStatus::NeedMoreInput
        ));
    }
    sort.combine(local, &global).unwrap();
    sort.finalize(&global, &sink_ctx).await.unwrap();
    assert_eq!(sort.metrics().runs_spilled, 0, "sorted in memory");

    let source_ctx = ctx(1).with_dependency(Arc::clone(&global));
    let source_global = Source::global_state(&sort, &source_ctx).unwrap();
    let mut local = Source::local_state(&sort, &source_ctx, &source_global).unwrap();
    let first = sort.get_data(&mut local, &source_global).await.unwrap();
    assert_eq!(first.map(|b| b.num_rows()), Some(1000));
    // Another consumer of the query needs the memory the result holds: the
    // rest of the merge is spilled.
    let pool = Arc::clone(query.pool());
    let want = pool.register("want", None);
    let reserved = pool.reserve(want, 3_400_000).await;
    assert!(reserved.is_ok(), "{reserved:?}");
    assert_eq!(sort.metrics().runs_spilled, 1);
    pool.release(want, 3_400_000);
    assert_eq!(
        super::sort::spill_result_again(&mut local).await.unwrap(),
        0
    );
    let mut keys = Vec::new();
    while let Some(b) = sort.get_data(&mut local, &source_global).await.unwrap() {
        let k = b
            .data()
            .column(0)
            .as_primitive::<arrow_array::types::Int64Type>();
        keys.extend(k.values().iter().copied());
    }
    assert_eq!(keys.len(), 9000, "every row after the first batch");
    assert!(keys.is_sorted());
    assert!(query.error().is_none());
    drop(reserved);
    pool.unregister(want);
    drop((local, source_global, source_ctx, global, sink_ctx));
    assert_eq!(manager.used_bytes(), 0);
    assert_eq!(root.reserved(), 0);
    drop(manager);
    let _ = std::fs::remove_dir_all(dir);
}

/// The merge budget covers the merger's per-source `slot`/`stamp` metadata,
/// not just the run frames and the output batch.
#[tokio::test(start_paused = true)]
async fn merge_budget_covers_per_source_metadata() {
    use super::merge::{SpillRun, metadata_bytes};
    use super::sort::merge_budget;

    assert_eq!(metadata_bytes(3), 3 * (8 + 4));

    let root = TreePool::new(&MemoryConfig::default());
    let handle = ConsumerHandle::register(root.child("q", None), "sort", None);
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join(format!("sort-unit-merge-budget-{}", std::process::id()));
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.clone(),
        ..SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let k: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
    let batch = Batch::new(RecordBatch::try_new(Arc::clone(&schema), vec![k]).unwrap());
    let mut runs = Vec::new();
    for (max_batch_bytes, row_bytes) in [(1000usize, 10usize), (2000, 20), (1500, 15)] {
        let scratch = handle.reservation().for_spill_scratch();
        let mut writer = scope.writer(Arc::clone(&schema), scratch).unwrap();
        writer.write(&batch).unwrap();
        let file = writer.finish().unwrap();
        runs.push(SpillRun {
            file,
            max_batch_bytes,
            row_bytes,
        });
    }
    let batch_rows = 100;
    let (budget, fan_in) = merge_budget(&runs, runs.len(), batch_rows, &handle, false)
        .await
        .unwrap();
    assert_eq!(fan_in, runs.len());
    let per_run = runs.iter().map(SpillRun::merge_bytes).max().unwrap();
    let row_bytes = runs.iter().map(|r| r.row_bytes).max().unwrap();
    let expected = per_run * fan_in + row_bytes * batch_rows + metadata_bytes(fan_in);
    assert_eq!(budget.size(), expected);
    drop((budget, runs, handle));
    assert_eq!(root.reserved(), 0);
    drop(manager);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn bucket_counts_follow_the_drivers() {
    use super::sort::bucket_bits;
    assert_eq!(bucket_bits(0), 0);
    assert_eq!(bucket_bits(1), 0);
    // At least eight buckets, four per driver, at most 4096.
    assert_eq!(bucket_bits(2), 3);
    assert_eq!(bucket_bits(4), 4);
    assert_eq!(bucket_bits(12), 6);
    assert_eq!(bucket_bits(5000), 12);
    assert_eq!(bucket_bits(usize::MAX), 12);
}

/// A merge of many slices of a bucketed sort sorts their rows instead of
/// running the loser tree, in the same stable order (ties by run, then
/// within a run); without memory for the sort it runs the tree. Without
/// hash buckets (one bucket) it always runs the tree: sorting all rows
/// again costs more than merging whole sorted batches.
#[test]
fn a_wide_bucket_merge_sorts_in_the_merge_order() {
    let (schema, batch) = floats();
    let keys = KeyEncoder::new(&schema, &[SortKey::asc(0).nulls_first(false)]).unwrap();
    let n = batch.num_rows();
    // 100 runs: rotations of the rows, so ties come from many runs.
    let rotations: Vec<Batch> = (0..100)
        .map(|i| {
            let rows: Vec<(usize, usize)> = (0..n).map(|r| (0, (r + i) % n)).collect();
            interleave_batches(&schema, std::slice::from_ref(&batch), &rows).unwrap()
        })
        .collect();
    let bucketing = Bucketing::new(Arc::clone(&keys), 1);
    let runs = |buckets: Option<&Bucketing>| -> Vec<Arc<SortedBatch>> {
        rotations
            .iter()
            .map(|b| Arc::new(SortedBatch::new(b.clone(), &keys, buckets).unwrap()))
            .collect()
    };
    let root = TreePool::new(&MemoryConfig::default());
    let handle = ConsumerHandle::register(root.child("q", None), "sort", None);
    let tight = TreePool::new(&MemoryConfig {
        limit: Some(1024),
        ..MemoryConfig::default()
    });
    let small = ConsumerHandle::register(tight.child("q", None), "sort", None);
    let drain = |mut m: Merger| {
        let mut out = Vec::new();
        while let Some(b) = m.next(37).unwrap() {
            out.extend(ids(&b));
        }
        out
    };
    let bucketed = runs(Some(&bucketing));
    let mut total = 0;
    for b in 0..bucketing.buckets() {
        let slices = bucketed
            .iter()
            .map(|s| (Arc::clone(s), s.bucket(b)))
            .collect();
        let tree = Merger::slices(Arc::clone(&schema), Arc::clone(&keys), slices, &[], &handle);
        let want = drain(tree.unwrap());
        total += want.len();
        let sorted = Merger::bucket(
            Arc::clone(&schema),
            Arc::clone(&keys),
            &bucketed,
            b,
            &handle,
        )
        .unwrap();
        assert_eq!(sorted.sorts_rows(), !want.is_empty(), "bucket {b}");
        assert_eq!(drain(sorted), want, "bucket {b}");
        // A pool too small for the keyed rows: the tree, same order.
        let fallback =
            Merger::bucket(Arc::clone(&schema), Arc::clone(&keys), &bucketed, b, &small).unwrap();
        assert!(!fallback.sorts_rows());
        assert_eq!(drain(fallback), want, "bucket {b}");
    }
    assert_eq!(total, 100 * n);
    // One bucket: the tree of `Merger::new`, never a sort of the rows.
    let plain = runs(None);
    let want = drain(
        Merger::new(
            Arc::clone(&schema),
            Arc::clone(&keys),
            plain.clone(),
            &[],
            &handle,
        )
        .unwrap(),
    );
    assert_eq!(want.len(), 100 * n);
    let merger = Merger::bucket(schema, keys, &plain, 0, &handle).unwrap();
    assert!(!merger.sorts_rows(), "an unbucketed merge runs the tree");
    assert_eq!(drain(merger), want);
}

/// `count` batches of `rows` rows of `key(batch, row)`.
fn int_batches(
    schema: &SchemaRef,
    count: i64,
    rows: i64,
    key: impl Fn(i64, i64) -> i64,
) -> Vec<Batch> {
    (0..count)
        .map(|b| {
            let k = (0..rows).map(|r| key(b, r));
            let k: ArrayRef = Arc::new(Int64Array::from_iter_values(k));
            Batch::new(RecordBatch::try_new(Arc::clone(schema), vec![k]).unwrap())
        })
        .collect()
}

fn int_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]))
}

fn scattered(b: i64, r: i64) -> i64 {
    (b * 1000 + r).wrapping_mul(2_654_435_761) % 1_000_003
}

/// `memory`'s parts between its splitters.
fn cut(memory: &[Slice], parts: usize) -> Vec<Vec<Slice>> {
    let splitters = merge_splitters(memory, parts).unwrap();
    let lows = std::iter::once(None).chain(splitters.iter().copied().map(Some));
    let highs = splitters.iter().copied().map(Some).chain([None]);
    lows.zip(highs)
        .map(|(lo, hi)| merge_part(memory, lo, hi).unwrap())
        .collect()
}

/// The merge's cuts keep every row once and in the tree's order, whatever the
/// number of parts: ties across many runs, NaN and NULL keys, runs of very
/// different lengths, and every key equal.
#[test]
fn parts_sorted_one_by_one_are_the_merge() {
    let (schema, batch) = floats();
    let keys = KeyEncoder::new(&schema, &[SortKey::asc(0).nulls_first(false)]).unwrap();
    let n = batch.num_rows();
    let rotations: Vec<Batch> = (0..100)
        .map(|i| {
            let rows: Vec<(usize, usize)> = (0..1 + i % n).map(|r| (0, (r + i) % n)).collect();
            interleave_batches(&schema, std::slice::from_ref(&batch), &rows).unwrap()
        })
        .collect();
    let root = TreePool::new(&MemoryConfig::default());
    let handle = ConsumerHandle::register(root.child("q", None), "sort", None);
    let runs: Vec<Arc<SortedBatch>> = rotations
        .iter()
        .map(|b| Arc::new(SortedBatch::new(b.clone(), &keys, None).unwrap()))
        .collect();
    let drain = |mut m: Merger| {
        let mut out = Vec::new();
        while let Some(b) = m.next(37).unwrap() {
            out.extend(ids(&b));
        }
        out
    };
    let want = drain(
        Merger::new(
            Arc::clone(&schema),
            Arc::clone(&keys),
            runs.clone(),
            &[],
            &handle,
        )
        .unwrap(),
    );
    let memory: Vec<Slice> = runs
        .iter()
        .map(|s| (Arc::clone(s), 0..s.order.len()))
        .collect();
    let rows: usize = memory.iter().map(|(_, r)| r.len()).sum();
    for parts in [1, 2, 3, 7, 16, rows, rows + 5] {
        let cut = cut(&memory, parts);
        assert!(cut.len() <= parts);
        let mut order = Vec::new();
        for part in &cut {
            assert_eq!(part.len(), memory.len());
            order.extend(sort_slices(part).unwrap());
        }
        let merger = Merger::from_order(
            Arc::clone(&schema),
            Arc::clone(&keys),
            &memory,
            order.clone(),
            (handle.reservation(), 0),
            1,
        );
        assert_eq!(drain(merger), want, "{parts} parts");
        // Gathered ahead, a few batches at a time: the same rows.
        let mut merger = Merger::from_order(
            Arc::clone(&schema),
            Arc::clone(&keys),
            &memory,
            order,
            (handle.reservation(), 0),
            parts.min(4),
        );
        let mut gathered = Vec::new();
        loop {
            let batches: Vec<Batch> = merger
                .gather_jobs(37)
                .iter()
                .map(|j| j.run().unwrap())
                .collect();
            merger.gathered(batches);
            let Some(batch) = merger.next(37).unwrap() else {
                break;
            };
            assert!(batch.num_rows() <= 37);
            gathered.extend(ids(&batch));
        }
        assert_eq!(gathered, want, "{parts} parts, gathered ahead");
    }
}

/// Equal keys do not pile up in one part, since the merge's order is total.
#[test]
fn equal_keys_spread_over_the_parts() {
    let schema = int_schema();
    let keys = KeyEncoder::new(&schema, &[SortKey::asc(0)]).unwrap();
    let memory: Vec<Slice> = int_batches(&schema, 100, 100, |_, _| 7)
        .into_iter()
        .map(|b| {
            let s = Arc::new(SortedBatch::new(b, &keys, None).unwrap());
            let all = 0..s.order.len();
            (s, all)
        })
        .collect();
    let cut = cut(&memory, 4);
    assert_eq!(cut.len(), 4);
    for part in &cut {
        let rows: usize = part.iter().map(|(_, r)| r.len()).sum();
        assert!((2000..=3000).contains(&rows), "{rows} rows in a part");
    }
}

#[test]
fn a_parallel_merge_needs_slices_rows_threads_and_memory() {
    let schema = int_schema();
    let keys = KeyEncoder::new(&schema, &[SortKey::asc(0)]).unwrap();
    let runs = |count, rows, buckets: Option<&Bucketing>| -> Vec<Arc<SortedBatch>> {
        int_batches(&schema, count, rows, scattered)
            .into_iter()
            .map(|b| Arc::new(SortedBatch::new(b, &keys, buckets).unwrap()))
            .collect()
    };
    let root = TreePool::new(&MemoryConfig::default());
    let handle = ConsumerHandle::register(root.child("q", None), "sort", None);
    let wide = runs(65, 1000, None);
    let m = parallel_merge(&wide, 8, 1000, &handle).unwrap();
    assert_eq!((m.memory.len(), m.parts), (65, 3), "rows bound the parts");
    assert_eq!(m.ready_bytes, 8 * 1000 * 3);
    assert!(m.scratch.size() > m.ready_bytes);
    drop(m);
    assert_eq!(parallel_merge(&wide, 2, 1000, &handle).unwrap().parts, 2);
    assert!(parallel_merge(&wide, 1, 1000, &handle).is_none(), "one thread");
    let few_slices = runs(64, 1000, None);
    assert!(parallel_merge(&few_slices, 8, 1000, &handle).is_none());
    let few_rows = runs(65, 100, None);
    assert!(parallel_merge(&few_rows, 8, 1000, &handle).is_none());
    let bucketing = Bucketing::new(Arc::clone(&keys), 1);
    let bucketed = runs(65, 1000, Some(&bucketing));
    assert!(parallel_merge(&bucketed, 8, 1000, &handle).is_none());
    let tight = TreePool::new(&MemoryConfig {
        limit: Some(1024),
        ..MemoryConfig::default()
    });
    let small = ConsumerHandle::register(tight.child("q", None), "sort", None);
    assert!(parallel_merge(&wide, 8, 1000, &small).is_none(), "the tree instead");
}

/// A sort's source over a wide in-memory result on two threads, with what it
/// needs to stay alive: the keys it reads, with the rest of the merge spilled
/// after the first batch if `spill_after_first`.
async fn read_wide_result(spill_after_first: bool) -> Vec<i64> {
    let schema = int_schema();
    let batches = int_batches(&schema, 100, 400, scattered);
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(2).unwrap();
    let root = TreePool::with_clock(
        &MemoryConfig {
            limit: Some(1 << 30),
            quantum: 1,
            ..MemoryConfig::default()
        },
        Arc::new(crate::executor::TokioClock),
    );
    let config = ExecConfig {
        batch_size: 1000,
        ..ExecConfig::default()
    };
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        config,
        root.child("q", None),
    );
    let ctx = |op| {
        TaskContext::new(
            Arc::clone(&query),
            Arc::clone(&executor),
            SpillOwner::new(),
            OperatorId(op),
            "Sort",
        )
    };
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join(format!(
            "sort-unit-parallel-{}-{spill_after_first}",
            std::process::id()
        ));
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.clone(),
        ..SpillConfig::default()
    })
    .unwrap();
    let sort = Sort::new(schema, &[SortKey::asc(0)], Some(manager.scope().unwrap())).unwrap();
    let sink_ctx = ctx(0);
    let global = Sink::global_state(&sort, &sink_ctx).unwrap();
    let mut local = Sink::local_state(&sort, &sink_ctx, &global).unwrap();
    for b in batches {
        sort.sink(b, &mut local, &global).unwrap();
    }
    sort.combine(local, &global).unwrap();
    sort.finalize(&global, &sink_ctx).await.unwrap();
    let source_ctx = ctx(1).with_dependency(Arc::clone(&global));
    let source_global = Source::global_state(&sort, &source_ctx).unwrap();
    let mut local = Source::local_state(&sort, &source_ctx, &source_global).unwrap();
    let mut keys = Vec::new();
    while let Some(b) = sort.get_data(&mut local, &source_global).await.unwrap() {
        let k = b
            .data()
            .column(0)
            .as_primitive::<arrow_array::types::Int64Type>();
        assert!(b.num_rows() <= 1000);
        let first = keys.is_empty();
        keys.extend(k.values().iter().copied());
        if first && spill_after_first {
            assert!(super::sort::spill_result_again(&mut local).await.unwrap() > 0);
            assert_eq!(sort.metrics().runs_spilled, 1);
        }
    }
    assert!(query.error().is_none());
    drop((local, source_global, source_ctx, global, sink_ctx));
    assert_eq!(manager.used_bytes(), 0);
    assert_eq!(root.reserved(), 0, "the merge's memory went back");
    drop(manager);
    let _ = std::fs::remove_dir_all(dir);
    keys
}

#[tokio::test(start_paused = true)]
async fn a_wide_result_reads_in_order_through_the_parallel_merge() {
    let mut want: Vec<i64> = (0..100)
        .flat_map(|b| (0..400).map(move |r| scattered(b, r)))
        .collect();
    want.sort_unstable();
    assert_eq!(read_wide_result(false).await, want);
    // The rest of the merge spilled after the first batch reads the same.
    assert_eq!(read_wide_result(true).await, want);
}

/// A cancelled parallel merge fails the read, and every later read: never a
/// short result.
#[tokio::test(start_paused = true)]
async fn a_cancelled_parallel_merge_fails_the_read() {
    let schema = int_schema();
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(2).unwrap();
    let root = TreePool::with_clock(
        &MemoryConfig::default(),
        Arc::new(crate::executor::TokioClock),
    );
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("q", None),
    );
    let ctx = |op| {
        TaskContext::new(
            Arc::clone(&query),
            Arc::clone(&executor),
            SpillOwner::new(),
            OperatorId(op),
            "Sort",
        )
    };
    let sort = Sort::new(Arc::clone(&schema), &[SortKey::asc(0)], None).unwrap();
    let sink_ctx = ctx(0);
    let global = Sink::global_state(&sort, &sink_ctx).unwrap();
    let mut local = Sink::local_state(&sort, &sink_ctx, &global).unwrap();
    for b in int_batches(&schema, 100, 400, scattered) {
        sort.sink(b, &mut local, &global).unwrap();
    }
    sort.combine(local, &global).unwrap();
    sort.finalize(&global, &sink_ctx).await.unwrap();
    let source_ctx = ctx(1).with_dependency(Arc::clone(&global));
    let source_global = Source::global_state(&sort, &source_ctx).unwrap();
    let mut local = Source::local_state(&sort, &source_ctx, &source_global).unwrap();
    query.cancel();
    assert!(sort.get_data(&mut local, &source_global).await.is_err());
    let mut again = Source::local_state(&sort, &source_ctx, &source_global).unwrap();
    assert!(sort.get_data(&mut again, &source_global).await.is_err());
    drop((local, again, source_global, source_ctx, global, sink_ctx));
    assert_eq!(root.reserved(), 0);
}

/// Rows sharing leading bytes order the same with those bytes skipped, also
/// when their keys are longer than the prefix and differ only past it.
#[test]
fn a_part_with_a_shared_prefix_sorts_past_it() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
        Field::new("c", DataType::Int64, false),
    ]));
    let keys = KeyEncoder::new(
        &schema,
        &[SortKey::asc(0), SortKey::asc(1), SortKey::asc(2)],
    )
    .unwrap();
    let memory: Vec<Slice> = (0..5_i64)
        .map(|b| {
            let col = |f: &dyn Fn(i64) -> i64| -> ArrayRef {
                Arc::new(Int64Array::from_iter_values((0..200).map(|r| f(b * 200 + r))))
            };
            let columns = vec![col(&|_| 42), col(&|i| i % 3), col(&|i| 1000 - i)];
            let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
            let s = Arc::new(SortedBatch::new(Batch::new(batch), &keys, None).unwrap());
            let all = 0..s.order.len();
            (s, all)
        })
        .collect();
    let order = sort_slices(&memory).unwrap();
    assert_eq!(order.len(), 1000);
    let row = |&(s, r): &(u32, u32)| memory[s as usize].0.rows.row(r as usize);
    assert!(order.windows(2).all(|w| row(&w[0]) <= row(&w[1])));
}

#[test]
fn the_shared_prefix_of_nothing_sorts_nothing() {
    let schema = int_schema();
    let keys = KeyEncoder::new(&schema, &[SortKey::asc(0)]).unwrap();
    let batch = int_batches(&schema, 1, 10, scattered).remove(0);
    let s = Arc::new(SortedBatch::new(batch, &keys, None).unwrap());
    assert!(sort_slices(&[(s, 3..3)]).unwrap().is_empty());
    assert!(sort_slices(&[]).unwrap().is_empty());
}
