// SPDX-License-Identifier: Apache-2.0
//! Unit tests of the stateless operators, called directly (the pipeline
//! tests are in `tests/operators.rs`).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::FutureExt;
use krishiv_common::error::ErrorClass;
use krishiv_memory::{MemoryConfig, MemoryPool, SpillOwner, TreePool};
use krishiv_types::batch::{Batch, Weights};

use super::*;
use crate::operator::{LocalState, Source};
use crate::{
    ExecConfig, ExecutorConfig, OperatorId, QueryContext, QueryId, TaskContext, TokioExecutor,
    WorkloadClass,
};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("s", DataType::Utf8, true),
    ]))
}

fn batch(a: &[Option<i64>]) -> Batch {
    let s: StringArray = a.iter().map(|v| v.map(|v| format!("s{v}"))).collect();
    let data = RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int64Array::from(a.to_vec())), Arc::new(s)],
    )
    .unwrap_or_else(|e| unreachable!("{e}"));
    Batch::new(data)
}

fn ints(a: &[i64]) -> Batch {
    batch(&a.iter().map(|v| Some(*v)).collect::<Vec<_>>())
}

fn weighted(a: &[i64], w: &[i64]) -> Batch {
    let (data, _) = ints(a).into_parts();
    Batch::with_weights(data, Weights::Column(Int64Array::from(w.to_vec())))
        .unwrap_or_else(|e| unreachable!("{e}"))
}

/// `a > 1`; a null `a` gives a null mask value.
fn gt1() -> Arc<dyn Predicate> {
    Arc::new(FnPredicate(|b: &RecordBatch| {
        Ok(b.column(0)
            .as_primitive::<Int64Type>()
            .iter()
            .map(|v| v.map(|v| v > 1))
            .collect::<BooleanArray>())
    }))
}

fn ctx() -> (Arc<TreePool>, TaskContext) {
    ctx_with(ExecConfig::default())
}

fn ctx_with(config: ExecConfig) -> (Arc<TreePool>, TaskContext) {
    let root = TreePool::new(&MemoryConfig::default());
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        config,
        root.child("q", None),
    );
    let exec =
        TokioExecutor::new(&ExecutorConfig::default()).unwrap_or_else(|e| unreachable!("{e}"));
    let ctx = TaskContext::new(q, exec, SpillOwner::new(), OperatorId(0), "op");
    (root, ctx)
}

fn schema_of(fields: &[(&str, DataType, bool)]) -> SchemaRef {
    Arc::new(Schema::new(
        fields
            .iter()
            .map(|(n, t, nullable)| Field::new(*n, t.clone(), *nullable))
            .collect::<Vec<_>>(),
    ))
}

fn scan_of(sizes: &[usize]) -> MemoryScan {
    let mut next = 0i64;
    let batches = sizes
        .iter()
        .map(|&n| {
            let rows: Vec<i64> = (next..next + n as i64).collect();
            next += n as i64;
            ints(&rows)
        })
        .collect();
    MemoryScan::try_new(schema(), batches).unwrap_or_else(|e| unreachable!("{e}"))
}

/// Drains a scan with one driver; returns the batches.
fn drain(scan: &MemoryScan, ctx: &TaskContext) -> (Vec<Batch>, u64) {
    let g = scan
        .global_state(ctx)
        .unwrap_or_else(|e| unreachable!("{e}"));
    let mut local = scan
        .local_state(ctx, &g)
        .unwrap_or_else(|e| unreachable!("{e}"));
    let mut out = Vec::new();
    while let Some(b) = scan
        .get_data(&mut local, &g)
        .now_or_never()
        .unwrap_or_else(|| unreachable!("memory scans do not wait"))
        .unwrap_or_else(|e| unreachable!("{e}"))
    {
        out.push(b);
    }
    (out, scan.morsels_claimed(&local))
}

#[test]
fn scan_slices_to_batch_size_and_groups_morsels() {
    let (_root, ctx) = ctx_with(ExecConfig {
        batch_size: 4,
        morsel_rows: 10,
        ..ExecConfig::default()
    });
    // 10 + 3 + 0 + 9 rows: morsels {10}, {3, 0, 9}; the 10 splits into 4+4+2.
    let scan = scan_of(&[10, 3, 0, 9]);
    let (out, morsels) = drain(&scan, &ctx);
    let sizes: Vec<_> = out.iter().map(Batch::num_rows).collect();
    assert_eq!(sizes, [4, 4, 2, 3, 4, 4, 1]);
    assert_eq!(morsels, 2);
    assert_eq!(scan.rows_read(), 22);
    let g = scan
        .global_state(&ctx)
        .unwrap_or_else(|e| unreachable!("{e}"));
    assert_eq!(scan.max_parallelism(&g), 2);
    assert_eq!(scan.name(), "MemoryScan");
    // No data: one (idle) driver, no morsel.
    let empty = scan_of(&[]);
    let g = empty
        .global_state(&ctx)
        .unwrap_or_else(|e| unreachable!("{e}"));
    assert_eq!(empty.max_parallelism(&g), 1);
    assert!(drain(&empty, &ctx).0.is_empty());
    assert_eq!(empty.morsels_claimed(&(Box::new(0u8) as LocalState)), 0);
}

#[test]
fn scan_pushdown_filters_before_projecting_and_projections_compose() {
    let (_root, ctx) = ctx();
    let mut scan = scan_of(&[2, 3, 3]);
    // The first batch (0, 1) is emptied by `a > 1`, which skips the second filter.
    assert_eq!(scan.push_filter(gt1()), FilterPushdown::Exact);
    // `a > 3` evaluated on the full schema although only `s` is read.
    let gt3: Arc<dyn Predicate> = Arc::new(FnPredicate(|b: &RecordBatch| {
        Ok(b.column(0)
            .as_primitive::<Int64Type>()
            .iter()
            .map(|v| v.map(|v| v > 3))
            .collect::<BooleanArray>())
    }));
    assert_eq!(scan.push_filter(gt3), FilterPushdown::Exact);
    scan.push_projection(&[1, 0])
        .unwrap_or_else(|e| unreachable!("{e}"));
    scan.push_projection(&[0])
        .unwrap_or_else(|e| unreachable!("{e}")); // composes to `s`
    assert_eq!(scan.schema().field(0).name(), "s");
    let (out, _) = drain(&scan, &ctx);
    assert_eq!(
        out.len(),
        2,
        "the first batch is filtered to nothing and skipped"
    );
    assert_eq!(
        out.iter()
            .flat_map(|b| b.data().column(0).as_string::<i32>().iter())
            .collect::<Vec<_>>(),
        [Some("s4"), Some("s5"), Some("s6"), Some("s7")]
    );
    assert_eq!(scan.rows_read(), 8);
    assert!(scan.push_projection(&[1]).is_err());
    assert!(scan.push_projection(&[5]).is_err());
}

#[test]
fn scan_rejects_foreign_batches_and_keeps_weights() {
    let other = schema_of(&[("z", DataType::Int64, true)]);
    assert!(MemoryScan::try_new(other, vec![ints(&[1])]).is_err());
    let (_root, ctx) = ctx();
    let w = weighted(&[1, 2], &[7, 8]);
    let scan =
        MemoryScan::try_new(schema(), vec![w.clone()]).unwrap_or_else(|e| unreachable!("{e}"));
    assert_eq!(drain(&scan, &ctx).0, [w]);
}

#[test]
fn scan_reports_inconsistent_state_as_internal_errors() {
    let (_root, ctx) = ctx();
    let scan = scan_of(&[1]);
    // A morsel that points past the scan's batches.
    let mut local = ScanLocal {
        batches: 5..6,
        offset: 0,
        claimed: 0,
    };
    let e = scan.next_piece(&mut local, 4).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    // A cursor that hands out a morsel the global state does not have.
    let global: crate::GlobalState = Arc::new(ScanGlobal {
        cursor: crate::MorselCursor::new(1, None),
        morsels: vec![],
        batch_rows: 4,
    });
    let mut local = scan
        .local_state(&ctx, &global)
        .unwrap_or_else(|e| unreachable!("{e}"));
    let e = scan.next_batch(&mut local, &global).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    // State of another operator.
    let other: crate::GlobalState = Arc::new(());
    assert!(scan.next_batch(&mut local, &other).is_err());
    assert_eq!(scan.max_parallelism(&other), 1);
    let mut wrong: LocalState = Box::new(());
    assert!(scan.next_batch(&mut wrong, &global).is_err());
}
