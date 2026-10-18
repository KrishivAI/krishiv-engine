// SPDX-License-Identifier: Apache-2.0
//! Unit tests of the sort internals: key canonicalization, per-batch order,
//! the merge's tie-break, weights through `interleave_batches`, and the
//! global spill driven directly.

use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_memory::{
    ConsumerHandle, MemoryConfig, MemoryPool, SpillConfig, SpillManager, SpillOwner, TreePool,
};
use krishiv_types::batch::{Batch, Weights};

use super::Sort;
use super::keys::{KeyEncoder, SortKey};
use super::merge::{Merger, SortedBatch, interleave_batches};
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
        let sorted = SortedBatch::new(batch.clone(), &keys).unwrap();
        // Ties stay in row order.
        assert_eq!(sorted.order, [4, 6, 1, 2, 3, 0, 5]);
        let desc = KeyEncoder::new(&schema, &[SortKey::desc(column)]).unwrap();
        let sorted = SortedBatch::new(batch.clone(), &desc).unwrap();
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
        assert_eq!(SortedBatch::new(batch.clone(), &keys).unwrap().order, order);
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
    .map(|b| Arc::new(SortedBatch::new(b, &keys).unwrap()))
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
            Arc::new(SortedBatch::new(b, &keys).unwrap())
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
