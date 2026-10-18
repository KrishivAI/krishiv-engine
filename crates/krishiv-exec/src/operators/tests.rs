// SPDX-License-Identifier: Apache-2.0
//! Unit tests of the stateless operators, called directly (the pipeline
//! tests are in `tests/operators.rs`).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Int64Array, RecordBatch, RecordBatchOptions, StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::FutureExt;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillOwner, TreePool};
use krishiv_types::batch::{Batch, Weights};

use super::*;
use crate::operator::{LocalState, Operator, OperatorOutput, Source};
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

fn col_a(b: &Batch) -> Vec<Option<i64>> {
    b.data()
        .column(0)
        .as_primitive::<Int64Type>()
        .iter()
        .collect()
}

fn unlocal() -> LocalState {
    Box::new(())
}

fn out_batch(o: OperatorOutput) -> Option<Batch> {
    match o {
        OperatorOutput::NeedMoreInput(b) => b,
        other => unreachable!("unexpected {other:?}"),
    }
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

#[test]
fn filter_keeps_selected_rows_with_their_weights() {
    let (_root, ctx) = ctx();
    let f = Filter::new(schema(), gt1());
    let mut local = f.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
    let out = out_batch(
        f.execute(&weighted(&[1, 2, 3], &[5, -1, 2]), &mut local)
            .unwrap_or_else(|e| unreachable!("{e}")),
    );
    let out = out.unwrap_or_else(|| unreachable!("rows survive"));
    assert_eq!(col_a(&out), [Some(2), Some(3)]);
    assert_eq!(
        out.weights(),
        &Weights::Column(Int64Array::from(vec![-1, 2]))
    );
    // A constant weight stays a constant.
    let (data, _) = ints(&[1, 2]).into_parts();
    let b = Batch::with_weights(data, Weights::Const(3)).unwrap_or_else(|e| unreachable!("{e}"));
    let out = out_batch(
        f.execute(&b, &mut local)
            .unwrap_or_else(|e| unreachable!("{e}")),
    );
    assert_eq!(out.map(|o| o.weights().clone()), Some(Weights::Const(3)));
    assert_eq!(f.name(), "Filter");
    assert_eq!(f.schema(), schema());
}

#[test]
fn filter_edge_cases_produce_no_output_and_null_does_not_select() {
    let f = Filter::new(schema(), gt1());
    let mut local = unlocal();
    // Empty batch.
    let empty = ints(&[]);
    assert!(
        out_batch(
            f.execute(&empty, &mut local)
                .unwrap_or_else(|e| unreachable!("{e}"))
        )
        .is_none()
    );
    // All-null column: every mask value is null.
    let nulls = batch(&[None, None, None]);
    assert!(
        out_batch(
            f.execute(&nulls, &mut local)
                .unwrap_or_else(|e| unreachable!("{e}"))
        )
        .is_none()
    );
    // Nothing passes.
    assert!(
        out_batch(
            f.execute(&ints(&[0, 1]), &mut local)
                .unwrap_or_else(|e| unreachable!("{e}"))
        )
        .is_none()
    );
    // Mixed null and passing rows.
    let mixed = batch(&[None, Some(5), None, Some(0)]);
    let out = out_batch(
        f.execute(&mixed, &mut local)
            .unwrap_or_else(|e| unreachable!("{e}")),
    );
    assert_eq!(out.map(|o| col_a(&o)), Some(vec![Some(5)]));
    // Everything passes: the output shares the input's buffers.
    let all = ints(&[2, 3]);
    let out = out_batch(
        f.execute(&all, &mut local)
            .unwrap_or_else(|e| unreachable!("{e}")),
    );
    let out = out.unwrap_or_else(|| unreachable!("all rows pass"));
    assert_eq!(
        out.data().column(0).to_data().buffers()[0].as_ptr(),
        all.data().column(0).to_data().buffers()[0].as_ptr()
    );
}

#[test]
fn filter_errors_propagate() {
    let bad_len = Filter::new(
        schema(),
        Arc::new(FnPredicate(|_: &RecordBatch| {
            Ok(BooleanArray::from(vec![true]))
        })),
    );
    let e = bad_len.execute(&ints(&[1, 2]), &mut unlocal()).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    let failing = Filter::new(
        schema(),
        Arc::new(FnPredicate(|_: &RecordBatch| {
            Err::<BooleanArray, _>(KrishivError::from(krishiv_common::error::InternalError {
                message: "boom".to_owned(),
            }))
        })),
    );
    assert!(failing.execute(&ints(&[1]), &mut unlocal()).is_err());
    assert!(
        format!(
            "{:?}",
            FnPredicate(|_: &RecordBatch| Ok::<_, KrishivError>(BooleanArray::from(vec![true])))
        )
        .contains("FnPredicate")
    );
}

/// `a + 1`, or an expression that misbehaves.
#[derive(Debug)]
struct PlusOne {
    ty: DataType,
    short: bool,
}

impl ScalarExpr for PlusOne {
    fn data_type(&self) -> DataType {
        self.ty.clone()
    }

    fn nullable(&self) -> bool {
        true
    }

    fn evaluate(&self, batch: &RecordBatch) -> crate::Result<ArrayRef> {
        let a = batch.column(0).as_primitive::<Int64Type>();
        let mut values: Vec<Option<i64>> = a.iter().map(|v| v.map(|v| v + 1)).collect();
        if self.short {
            values.pop();
        }
        Ok(Arc::new(Int64Array::from(values)))
    }
}

fn plus_one() -> Arc<dyn ScalarExpr> {
    Arc::new(PlusOne {
        ty: DataType::Int64,
        short: false,
    })
}

#[test]
fn project_reorders_renames_and_computes() {
    let p = Project::try_new(
        &schema(),
        vec![
            ProjectItem::column(1),
            ProjectItem::aliased(0, "x"),
            ProjectItem::expr("y", plus_one()),
            ProjectItem::column(1),
        ],
    )
    .unwrap_or_else(|e| unreachable!("{e}"));
    let names: Vec<_> = p
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, ["s", "x", "y", "s"]);
    let input = weighted(&[1, 2], &[3, -4]);
    let out = out_batch(
        p.execute(&input, &mut unlocal())
            .unwrap_or_else(|e| unreachable!("{e}")),
    )
    .unwrap_or_else(|| unreachable!("rows"));
    assert_eq!(
        out.data()
            .column(1)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec(),
        [1, 2]
    );
    assert_eq!(
        out.data()
            .column(2)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec(),
        [2, 3]
    );
    assert_eq!(out.weights(), input.weights());
    // Plain columns share buffers.
    assert_eq!(
        out.data().column(0).to_data().buffers()[1].as_ptr(),
        input.data().column(1).to_data().buffers()[1].as_ptr()
    );
    assert_eq!(p.name(), "Project");
}

#[test]
fn project_edge_cases() {
    let all_null = batch(&[None, None]);
    let p = Project::try_new(
        &schema(),
        vec![ProjectItem::column(0), ProjectItem::expr("y", plus_one())],
    )
    .unwrap_or_else(|e| unreachable!("{e}"));
    let out = out_batch(
        p.execute(&all_null, &mut unlocal())
            .unwrap_or_else(|e| unreachable!("{e}")),
    )
    .unwrap_or_else(|| unreachable!("rows"));
    assert_eq!(out.data().column(0).null_count(), 2);
    assert_eq!(out.data().column(1).null_count(), 2);
    // Empty input: no output.
    assert!(
        out_batch(
            p.execute(&ints(&[]), &mut unlocal())
                .unwrap_or_else(|e| unreachable!("{e}"))
        )
        .is_none()
    );
    // No columns: the row count survives.
    let none = Project::try_new(&schema(), vec![]).unwrap_or_else(|e| unreachable!("{e}"));
    let out = out_batch(
        none.execute(&ints(&[1, 2, 3]), &mut unlocal())
            .unwrap_or_else(|e| unreachable!("{e}")),
    );
    assert_eq!(
        out.map(|o| (o.num_rows(), o.data().num_columns())),
        Some((3, 0))
    );
}

#[test]
fn project_rejects_bad_plans_and_bad_evaluators() {
    let e = Project::try_new(&schema(), vec![ProjectItem::column(2)]).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    // An evaluator whose column is shorter than the batch, or of another type.
    for (ty, short) in [(DataType::Int64, true), (DataType::Utf8, false)] {
        let p = Project::try_new(
            &schema(),
            vec![ProjectItem::expr("y", Arc::new(PlusOne { ty, short }))],
        )
        .unwrap_or_else(|e| unreachable!("{e}"));
        let e = p.execute(&ints(&[1, 2]), &mut unlocal()).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
    }
    // An index that is valid for the plan but not for the batch.
    let p = Project::try_new(&schema(), vec![ProjectItem::column(1)])
        .unwrap_or_else(|e| unreachable!("{e}"));
    let narrow = Batch::new(
        RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(1)),
        )
        .unwrap_or_else(|e| unreachable!("{e}")),
    );
    assert!(p.execute(&narrow, &mut unlocal()).is_err());
}

fn run_limit(offset: u64, limit: u64, sizes: &[usize]) -> (Vec<i64>, usize) {
    let l = Limit::with_offset(schema(), offset, limit);
    let mut got = Vec::new();
    let mut next = 0i64;
    let mut calls = 0;
    for &n in sizes {
        let rows: Vec<i64> = (next..next + n as i64).collect();
        next += n as i64;
        calls += 1;
        let (out, done) = match l
            .execute(&ints(&rows), &mut unlocal())
            .unwrap_or_else(|e| unreachable!("{e}"))
        {
            OperatorOutput::NeedMoreInput(b) => (b, false),
            OperatorOutput::Finished(b) => (b, true),
            other => unreachable!("unexpected {other:?}"),
        };
        got.extend(out.iter().flat_map(col_a).flatten());
        if done {
            break;
        }
    }
    (got, calls)
}

#[test]
fn limit_counts_rows_across_batches() {
    assert_eq!(run_limit(0, 5, &[3, 3, 3]), ((0..5).collect(), 2));
    assert_eq!(run_limit(0, 3, &[3, 3]), ((0..3).collect(), 1)); // exact fit stops at once
    assert_eq!(run_limit(0, 100, &[3, 3]), ((0..6).collect(), 2)); // never reached
    assert_eq!(run_limit(0, 0, &[3, 3]), (vec![], 1)); // LIMIT 0
    assert_eq!(run_limit(2, 3, &[2, 2, 2]), (vec![2, 3, 4], 3)); // OFFSET skips a batch
    assert_eq!(run_limit(1, 1, &[3]), (vec![1], 1)); // slice in the middle
    assert_eq!(run_limit(5, 5, &[0, 2]), (vec![], 2)); // empty batches
    assert_eq!(run_limit(u64::MAX, u64::MAX, &[2, 2]), (vec![], 2)); // saturates
}

#[test]
fn limit_after_finished_and_weights() {
    let l = Limit::new(schema(), 1);
    assert!(matches!(
        l.execute(&ints(&[1, 2]), &mut unlocal())
            .unwrap_or_else(|e| unreachable!("{e}")),
        OperatorOutput::Finished(Some(_))
    ));
    // A batch already in flight in another driver gets nothing.
    assert!(matches!(
        l.execute(&ints(&[3]), &mut unlocal())
            .unwrap_or_else(|e| unreachable!("{e}")),
        OperatorOutput::Finished(None)
    ));
    // Weights other than insert-only are rejected, not miscounted.
    for b in [
        weighted(&[1], &[1]),
        Batch::with_weights(ints(&[1]).into_parts().0, Weights::Const(2))
            .unwrap_or_else(|e| unreachable!("{e}")),
    ] {
        let e = Limit::new(schema(), 5)
            .execute(&b, &mut unlocal())
            .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    }
    assert_eq!((l.name(), l.schema()), ("Limit", schema()));
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
