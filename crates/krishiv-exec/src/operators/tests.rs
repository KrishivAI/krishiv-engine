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
    let (_root, ctx) = ctx();
    let f = Filter::new(schema(), gt1());
    let mut local = f.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
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
    let (_root, ctx) = ctx();
    let mut local = bad_len
        .local_state(&ctx)
        .unwrap_or_else(|e| unreachable!("{e}"));
    let e = bad_len.execute(&ints(&[1, 2]), &mut local).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    let failing = Filter::new(
        schema(),
        Arc::new(FnPredicate(|_: &RecordBatch| {
            Err::<BooleanArray, _>(KrishivError::from(krishiv_common::error::InternalError {
                message: "boom".to_owned(),
            }))
        })),
    );
    assert!(failing.execute(&ints(&[1]), &mut local).is_err());
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
    let (_root, ctx) = ctx();
    let mut local = p.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
    let input = weighted(&[1, 2], &[3, -4]);
    let out = out_batch(
        p.execute(&input, &mut local)
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
    let (_root, ctx) = ctx();
    let all_null = batch(&[None, None]);
    let p = Project::try_new(
        &schema(),
        vec![ProjectItem::column(0), ProjectItem::expr("y", plus_one())],
    )
    .unwrap_or_else(|e| unreachable!("{e}"));
    let mut local = p.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
    let out = out_batch(
        p.execute(&all_null, &mut local)
            .unwrap_or_else(|e| unreachable!("{e}")),
    )
    .unwrap_or_else(|| unreachable!("rows"));
    assert_eq!(out.data().column(0).null_count(), 2);
    assert_eq!(out.data().column(1).null_count(), 2);
    // Empty input: no output.
    assert!(
        out_batch(
            p.execute(&ints(&[]), &mut local)
                .unwrap_or_else(|e| unreachable!("{e}"))
        )
        .is_none()
    );
    // No columns: the row count survives.
    let none = Project::try_new(&schema(), vec![]).unwrap_or_else(|e| unreachable!("{e}"));
    let mut none_local = none
        .local_state(&ctx)
        .unwrap_or_else(|e| unreachable!("{e}"));
    let out = out_batch(
        none.execute(&ints(&[1, 2, 3]), &mut none_local)
            .unwrap_or_else(|e| unreachable!("{e}")),
    );
    assert_eq!(
        out.map(|o| (o.num_rows(), o.data().num_columns())),
        Some((3, 0))
    );
}

#[test]
fn project_rejects_bad_plans_and_bad_evaluators() {
    let (_root, ctx) = ctx();
    let e = Project::try_new(&schema(), vec![ProjectItem::column(2)]).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    // An evaluator whose column is shorter than the batch, or of another type.
    for (ty, short) in [(DataType::Int64, true), (DataType::Utf8, false)] {
        let p = Project::try_new(
            &schema(),
            vec![ProjectItem::expr("y", Arc::new(PlusOne { ty, short }))],
        )
        .unwrap_or_else(|e| unreachable!("{e}"));
        let mut local = p.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
        let e = p.execute(&ints(&[1, 2]), &mut local).unwrap_err();
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
    let mut local = p.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
    assert!(p.execute(&narrow, &mut local).is_err());
}

#[test]
fn filter_and_project_reject_foreign_local_state() {
    // A stateful operator evaluated with another operator's state fails
    // closed instead of silently reading partition 0's stream.
    let e = Filter::new(schema(), gt1())
        .execute(&ints(&[2]), &mut unlocal())
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    let p = Project::try_new(&schema(), vec![ProjectItem::column(0)])
        .unwrap_or_else(|e| unreachable!("{e}"));
    let e = p.execute(&ints(&[2]), &mut unlocal()).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
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

#[test]
fn union_schema_unification() {
    let a = schema_of(&[
        ("c1", DataType::Int64, false),
        ("c2", DataType::Utf8, false),
    ]);
    let b = schema_of(&[("x", DataType::Int64, true), ("y", DataType::Utf8, false)]);
    let u = UnionAlign::unify(&[Arc::clone(&a), Arc::clone(&b)])
        .unwrap_or_else(|e| unreachable!("{e}"));
    let fields: Vec<_> = u
        .fields()
        .iter()
        .map(|f| (f.name().as_str(), f.is_nullable()))
        .collect();
    assert_eq!(fields, [("c1", true), ("c2", false)]);
    // Errors: no branch, arity, type.
    assert!(UnionAlign::unify(&[]).is_err());
    let one = schema_of(&[("c1", DataType::Int64, false)]);
    assert!(UnionAlign::unify(&[Arc::clone(&a), one]).is_err());
    let typed = schema_of(&[
        ("c1", DataType::Int32, false),
        ("c2", DataType::Utf8, false),
    ]);
    assert!(UnionAlign::unify(&[Arc::clone(&a), typed]).is_err());
    // A branch cannot be nullable where the output is not.
    assert!(UnionAlign::try_new(&b, a).is_err());
}

#[test]
fn union_align_relabels_without_touching_rows_or_weights() {
    let first = schema_of(&[("c1", DataType::Int64, false), ("c2", DataType::Utf8, true)]);
    let second = schema();
    let out =
        UnionAlign::unify(&[first, Arc::clone(&second)]).unwrap_or_else(|e| unreachable!("{e}"));
    let op = UnionAlign::try_new(&second, Arc::clone(&out)).unwrap_or_else(|e| unreachable!("{e}"));
    assert_eq!((op.name(), op.schema()), ("UnionAlign", Arc::clone(&out)));
    let input = weighted(&[1, 2], &[2, -1]);
    let aligned = out_batch(
        op.execute(&input, &mut unlocal())
            .unwrap_or_else(|e| unreachable!("{e}")),
    )
    .unwrap_or_else(|| unreachable!("rows"));
    assert_eq!(aligned.schema(), out);
    assert_eq!(aligned.weights(), input.weights());
    assert_eq!(col_a(&aligned), [Some(1), Some(2)]);
    assert!(
        out_batch(
            op.execute(&ints(&[]), &mut unlocal())
                .unwrap_or_else(|e| unreachable!("{e}"))
        )
        .is_none()
    );
    // An all-null branch column fits a nullable output.
    assert!(
        out_batch(
            op.execute(&batch(&[None]), &mut unlocal())
                .unwrap_or_else(|e| unreachable!("{e}"))
        )
        .is_some()
    );
    // A batch that does not match the declared branch schema is an engine bug.
    let wrong = Batch::new(
        RecordBatch::try_new(
            schema_of(&[("a", DataType::Int32, true), ("s", DataType::Utf8, true)]),
            vec![
                Arc::new(arrow_array::Int32Array::from(vec![1])),
                Arc::new(StringArray::from(vec!["x"])),
            ],
        )
        .unwrap_or_else(|e| unreachable!("{e}")),
    );
    assert!(op.execute(&wrong, &mut unlocal()).is_err());
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
    // 10 + 3 + 0 + 9 rows: morsels {10}, {3, 0, 9}; the 10 splits into 4+4+2
    // and the empty batch is still returned (one piece per call).
    let scan = scan_of(&[10, 3, 0, 9]);
    let (out, morsels) = drain(&scan, &ctx);
    let sizes: Vec<_> = out.iter().map(Batch::num_rows).collect();
    assert_eq!(sizes, [4, 4, 2, 3, 0, 4, 4, 1]);
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
fn scan_pushdown_filters_before_projecting_once_built() {
    let (_root, ctx) = ctx();
    let base = scan_of(&[2, 3, 3]);
    // `a > 3` evaluated on the full schema although only `s` is read.
    let gt3: Arc<dyn Predicate> = Arc::new(FnPredicate(|b: &RecordBatch| {
        Ok(b.column(0)
            .as_primitive::<Int64Type>()
            .iter()
            .map(|v| v.map(|v| v > 3))
            .collect::<BooleanArray>())
    }));
    let f1 = PushedFilter::opaque(gt1());
    assert_eq!(base.filter_pushdown(&f1), FilterPushdown::Exact);
    assert!(!base.supports_selection());
    assert_eq!(base.table_schema(), schema());
    let pushed = ScanPushdowns {
        projection: Some(vec![krishiv_io::ColumnPath::new("s")]),
        // The first batch (0, 1) is emptied by `a > 1`, which skips the second filter.
        filters: vec![f1, PushedFilter::opaque(gt3)],
        // Ignored: filters may drop rows.
        limit: Some(1),
        ..ScanPushdowns::default()
    };
    let scan = base
        .with_pushdowns(pushed.clone())
        .unwrap_or_else(|e| unreachable!("{e}"));
    assert_eq!(scan.schema().fields().len(), 1);
    assert_eq!(scan.schema().field(0).name(), "s");
    // The base scan is unchanged (pushdowns are not cumulative).
    assert_eq!(base.schema(), schema());
    let (out, _) = drain(&scan, &ctx);
    assert_eq!(
        out.len(),
        3,
        "the first batch is filtered to nothing but still returned"
    );
    assert_eq!(out[0].num_rows(), 0);
    assert_eq!(
        out.iter()
            .flat_map(|b| b.data().column(0).as_string::<i32>().iter())
            .collect::<Vec<_>>(),
        [Some("s4"), Some("s5"), Some("s6"), Some("s7")]
    );
    assert_eq!(scan.rows_read(), 8);
    assert!(base.build(pushed).is_ok());
    // Unknown columns and selections are refused.
    let bad = ScanPushdowns {
        projection: Some(vec![krishiv_io::ColumnPath::new("zz")]),
        ..ScanPushdowns::default()
    };
    assert!(base.with_pushdowns(bad).is_err());
    #[derive(Debug)]
    struct All;
    impl FileSelections for All {
        fn selection(&self, _file: &krishiv_io::FileRef) -> Option<krishiv_io::RowSelection> {
            None
        }
    }
    let sel = ScanPushdowns {
        selection: Some(Arc::new(All)),
        ..ScanPushdowns::default()
    };
    // `output_schema` refuses what `build` refuses, with the same error.
    let refused = |p: &ScanPushdowns| {
        let a = base.with_pushdowns(p.clone()).unwrap_err();
        let b = ScanPushdown::output_schema(&base, p).unwrap_err();
        assert_eq!(a.to_string(), b.to_string());
        assert_eq!(a.error_class(), ErrorClass::InternalError);
    };
    refused(&sel);
    let meta = ScanPushdowns {
        metadata: krishiv_io::MetadataCols {
            file_name: true,
            ..krishiv_io::MetadataCols::default()
        },
        ..ScanPushdowns::default()
    };
    refused(&meta);
    // Without a projection, every column.
    let all = ScanPushdown::output_schema(&base, &ScanPushdowns::default()).unwrap();
    assert_eq!(all, ScanPushdown::table_schema(&base));
}

#[test]
fn scan_limit_and_nested_projection() {
    let (_root, ctx) = ctx();
    // A limit without filters stops claiming once enough rows were returned.
    let base = scan_of(&[3, 3, 3]);
    let limited = base
        .with_pushdowns(ScanPushdowns {
            limit: Some(4),
            ..ScanPushdowns::default()
        })
        .unwrap_or_else(|e| unreachable!("{e}"));
    let (out, _) = drain(&limited, &ctx);
    assert_eq!(out.iter().map(Batch::num_rows).sum::<usize>(), 6);
    // A nested path prunes struct fields, zero-copy.
    let inner = arrow_schema::Fields::from(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("y", DataType::Int64, true),
    ]);
    let table = Arc::new(Schema::new(vec![Field::new(
        "st",
        DataType::Struct(inner.clone()),
        true,
    )]));
    let st = arrow_array::StructArray::new(
        inner,
        vec![
            Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![3, 4])) as ArrayRef,
        ],
        None,
    );
    let data = RecordBatch::try_new(Arc::clone(&table), vec![Arc::new(st)])
        .unwrap_or_else(|e| unreachable!("{e}"));
    let scan = MemoryScan::try_new(table, vec![Batch::new(data)])
        .and_then(|s| {
            s.with_pushdowns(ScanPushdowns {
                projection: Some(vec![krishiv_io::ColumnPath::nested(["st", "y"])]),
                ..ScanPushdowns::default()
            })
        })
        .unwrap_or_else(|e| unreachable!("{e}"));
    let (out, _) = drain(&scan, &ctx);
    let st = out[0].data().column(0).as_struct();
    assert_eq!(st.num_columns(), 1);
    assert_eq!(st.column(0).as_primitive::<Int64Type>().values(), &[3, 4]);
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
        position: None,
        rows: super::RowStage::new(&[], &schema(), super::RowStageSettings::default()),
        partition: 0,
    };
    let e = scan.next_piece(&mut local, 4, 0).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    // A cursor that hands out a morsel the global state does not have.
    let global: crate::GlobalState = Arc::new(ScanGlobal {
        cursor: crate::MorselCursor::new(1, None),
        morsels: vec![],
        batch_rows: 4,
        returned: std::sync::atomic::AtomicU64::new(0),
        runtime: Vec::new(),
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

/// Column `i` of the batch as a replication count.
#[derive(Debug)]
struct Column(usize);

impl ScalarExpr for Column {
    fn data_type(&self) -> DataType {
        DataType::Int64
    }

    fn nullable(&self) -> bool {
        true
    }

    fn evaluate(&self, batch: &RecordBatch) -> crate::error::Result<ArrayRef> {
        Ok(Arc::clone(batch.column(self.0)))
    }
}

/// Every output batch of `op` over `input`, feeding it until it asks for
/// more input.
fn replicate_all(op: &ReplicateRows, input: &Batch, local: &mut LocalState) -> Vec<Batch> {
    let mut out = Vec::new();
    loop {
        match op
            .execute(input, local)
            .unwrap_or_else(|e| unreachable!("{e}"))
        {
            OperatorOutput::HaveMoreOutput(b) => out.push(b),
            OperatorOutput::NeedMoreInput(b) => {
                out.extend(b);
                return out;
            }
            other => unreachable!("unexpected {other:?}"),
        }
    }
}

#[test]
fn replicate_rows_repeats_each_row_its_count_times() {
    let (_root, ctx) = ctx_with(ExecConfig {
        batch_size: 3,
        ..ExecConfig::default()
    });
    let op = ReplicateRows::new(schema(), Arc::new(Column(0)));
    assert_eq!(op.name(), "ReplicateRows");
    assert_eq!(op.schema(), schema());
    let mut local = op.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
    // NULL, 0 and negative counts drop the row; 4 copies span batches of 3.
    let input = batch(&[Some(2), None, Some(0), Some(4), Some(-1), Some(1)]);
    let out = replicate_all(&op, &input, &mut local);
    assert_eq!(
        out.iter().map(Batch::num_rows).collect::<Vec<_>>(),
        [3, 3, 1]
    );
    let rows: Vec<Option<i64>> = out.iter().flat_map(col_a).collect();
    assert_eq!(
        rows,
        [2, 2, 4, 4, 4, 4, 1].map(Some),
        "each row its count times, in order"
    );
    let s: Vec<String> = out
        .iter()
        .flat_map(|b| {
            b.data()
                .column(1)
                .as_string::<i32>()
                .iter()
                .map(|v| v.unwrap_or_default().to_owned())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(s, ["s2", "s2", "s4", "s4", "s4", "s4", "s1"]);
    // The next input starts afresh; an empty one and one of only zero
    // counts produce nothing.
    assert!(replicate_all(&op, &ints(&[]), &mut local).is_empty());
    assert!(replicate_all(&op, &ints(&[0, -5]), &mut local).is_empty());
    let out = replicate_all(&op, &ints(&[1]), &mut local);
    assert_eq!(out.iter().flat_map(col_a).collect::<Vec<_>>(), [Some(1)]);
}

#[test]
fn replicate_rows_keeps_weights_and_rejects_non_bigint_counts() {
    let (_root, ctx) = ctx();
    let op = ReplicateRows::new(schema(), Arc::new(Column(0)));
    let mut local = op.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
    let out = replicate_all(&op, &weighted(&[2, 1], &[5, -1]), &mut local);
    let [b] = out.as_slice() else {
        unreachable!("one batch: {out:?}")
    };
    assert_eq!(col_a(b), [2, 2, 1].map(Some));
    assert_eq!(
        b.weights(),
        &Weights::Column(Int64Array::from(vec![5, 5, -1]))
    );
    let out = replicate_all(&op, &ints(&[2]), &mut local);
    assert_eq!(out[0].weights(), &Weights::Const(1));
    let strings = ReplicateRows::new(schema(), Arc::new(Column(1)));
    let mut local = strings
        .local_state(&ctx)
        .unwrap_or_else(|e| unreachable!("{e}"));
    let err = strings
        .execute(&ints(&[1]), &mut local)
        .err()
        .unwrap_or_else(|| unreachable!("an error"));
    assert_eq!(err.error_class(), ErrorClass::InternalError);
}

/// A projection of `schema()`: `a` (or nulls of it), then `s` (or nulls).
fn expand_projection(null_a: bool, null_s: bool) -> Arc<Project> {
    let item = |index: usize, name: &str, null: bool| {
        if null {
            let nulls = NullOf::new(&schema(), index).unwrap_or_else(|e| unreachable!("{e}"));
            ProjectItem::expr(name, Arc::new(nulls))
        } else {
            ProjectItem::column(index)
        }
    };
    Project::try_new(&schema(), vec![item(0, "a", null_a), item(1, "s", null_s)])
        .unwrap_or_else(|e| unreachable!("{e}"))
}

#[test]
fn expand_streams_one_copy_per_projection_and_keeps_weights() {
    let (_root, ctx) = ctx();
    let op = Expand::try_new(vec![
        expand_projection(false, false),
        expand_projection(false, true),
        expand_projection(true, true),
    ])
    .unwrap_or_else(|e| unreachable!("{e}"));
    assert_eq!(op.name(), "Expand");
    assert_eq!(op.schema(), schema());
    let mut local = op.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
    let input = weighted(&[1, 2], &[3, -4]);
    // One copy per call: `HaveMoreOutput` until the last projection.
    let mut copies = Vec::new();
    for _ in 0..2 {
        let OperatorOutput::HaveMoreOutput(b) = op
            .execute(&input, &mut local)
            .unwrap_or_else(|e| unreachable!("{e}"))
        else {
            unreachable!("a copy and more to come")
        };
        copies.push(b);
    }
    copies.extend(out_batch(
        op.execute(&input, &mut local)
            .unwrap_or_else(|e| unreachable!("{e}")),
    ));
    assert_eq!(copies.len(), 3);
    assert_eq!(col_a(&copies[0]), [1, 2].map(Some));
    assert_eq!(col_a(&copies[1]), [1, 2].map(Some));
    assert_eq!(col_a(&copies[2]), [None, None]);
    let strings = |b: &Batch| -> Vec<Option<String>> {
        b.data()
            .column(1)
            .as_string::<i32>()
            .iter()
            .map(|v| v.map(str::to_owned))
            .collect()
    };
    assert_eq!(
        strings(&copies[0]),
        [Some("s1".to_owned()), Some("s2".to_owned())]
    );
    assert_eq!(strings(&copies[1]), [None, None]);
    assert!(copies.iter().all(|b| b.weights() == input.weights()));
    // The cursor starts again for the next input; an empty batch is skipped.
    assert!(
        out_batch(
            op.execute(&ints(&[]), &mut local)
                .unwrap_or_else(|e| unreachable!("{e}"))
        )
        .is_none()
    );
    assert!(matches!(
        op.execute(&ints(&[7]), &mut local),
        Ok(OperatorOutput::HaveMoreOutput(_))
    ));
}

#[test]
fn expand_rejects_bad_plans() {
    let class = |r: Result<Arc<Expand>, KrishivError>| {
        r.err()
            .unwrap_or_else(|| unreachable!("an error"))
            .error_class()
    };
    assert_eq!(class(Expand::try_new(vec![])), ErrorClass::InternalError);
    // A projection that outputs other columns.
    let narrow = Project::try_new(&schema(), vec![ProjectItem::column(0)])
        .unwrap_or_else(|e| unreachable!("{e}"));
    assert_eq!(
        class(Expand::try_new(vec![
            expand_projection(false, false),
            narrow
        ])),
        ErrorClass::InternalError
    );
    let renamed = Project::try_new(
        &schema(),
        vec![ProjectItem::aliased(0, "x"), ProjectItem::column(1)],
    )
    .unwrap_or_else(|e| unreachable!("{e}"));
    assert_eq!(
        class(Expand::try_new(vec![
            expand_projection(false, false),
            renamed
        ])),
        ErrorClass::InternalError
    );
    // Nulls of a column the input has not; and of a batch that lost it.
    let err = NullOf::new(&schema(), 2)
        .err()
        .unwrap_or_else(|| unreachable!("an error"));
    assert_eq!(err.error_class(), ErrorClass::InternalError);
    let nulls = NullOf::new(&schema(), 1).unwrap_or_else(|e| unreachable!("{e}"));
    assert_eq!(nulls.data_type(), DataType::Utf8);
    assert!(nulls.nullable());
    let one_column = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)])),
        vec![Arc::new(Int64Array::from(vec![1_i64]))],
    )
    .unwrap_or_else(|e| unreachable!("{e}"));
    let err = nulls
        .evaluate(&one_column)
        .err()
        .unwrap_or_else(|| unreachable!("an error"));
    assert_eq!(err.error_class(), ErrorClass::InternalError);
}

/// A stateful function sees the driver's partition: `rand(0)` draws
/// partition 0's stream on one driver and partition 1's on another, even
/// when partition 1 binds first, and the same partition draws the same
/// stream on a fresh evaluation.
#[test]
fn project_threads_the_drivers_partition_to_stateful_functions() {
    use arrow_array::NullArray;
    use arrow_array::types::Float64Type;
    use krishiv_common::config::Config;
    use krishiv_plan::{
        Expr, ExprNode, FunctionId, FunctionKind, Literal, ResolvedExpr, ResolvedFunction,
    };

    use crate::planner::eval::{Compiled, EvalEnv, Input, compile};

    // `rand(0)` analyzed: a non-nullable double of a folded seed.
    let call = |seed: i32| {
        let f = ResolvedFunction {
            id: FunctionId::Registry {
                name: "rand".to_owned(),
                id: 0,
            },
            kind: FunctionKind::Scalar,
            args: vec![Arc::new(Expr::new(ExprNode::Literal(Literal::Integer(
                seed,
            ))))],
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type: krishiv_types::DataType::Double,
            nullable: false,
            deterministic: false,
        };
        Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f)))
    };
    let one_row = || {
        let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Null, true)]));
        let data = RecordBatch::try_new(schema, vec![Arc::new(NullArray::new(1))])
            .unwrap_or_else(|e| unreachable!("{e}"));
        Batch::new(data)
    };
    let project = |seed: i32| {
        let config = Config::new();
        let env = EvalEnv::new(&config, Arc::from("test"));
        let schema = Arc::new(Schema::empty());
        let compiled = compile(
            &call(seed),
            Input {
                layout: &[],
                schema: &schema,
                env: &env,
                subqueries: &[],
            },
        )
        .unwrap_or_else(|e| unreachable!("{e}"));
        let input = one_row();
        Project::try_new(
            &input.schema(),
            vec![ProjectItem::expr(
                "r",
                Arc::new(Compiled(compiled)) as Arc<dyn crate::operators::expr::ScalarExpr>,
            )],
        )
        .unwrap_or_else(|e| unreachable!("{e}"))
    };
    let run = |p: &Arc<Project>, partition: u32| {
        let (_root, ctx) = ctx();
        let ctx = ctx.with_partition(partition);
        let mut local = p.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
        let out = out_batch(
            p.execute(&one_row(), &mut local)
                .unwrap_or_else(|e| unreachable!("{e}")),
        )
        .unwrap_or_else(|| unreachable!("a row"));
        out.data().column(0).as_primitive::<Float64Type>().value(0)
    };
    // Partition 1 binds first on the shared binding: partition 0 still
    // draws its own stream (each slot draws `base + slot`), and both are
    // Spark's first draws for their seeds.
    let p = project(0);
    assert_eq!(run(&p, 1), 0.6363787615254752);
    assert_eq!(run(&p, 0), 0.7604953758285915);
    // A fresh evaluation draws the same stream again: the stream is a pure
    // function of (seed, partition, row).
    assert_eq!(run(&project(0), 0), 0.7604953758285915);
}

use krishiv_plan::GeneratorKind;

/// A column of `data_type` read as the generator's argument.
#[derive(Debug)]
struct Typed(usize, DataType);

impl ScalarExpr for Typed {
    fn data_type(&self) -> DataType {
        self.1.clone()
    }

    fn nullable(&self) -> bool {
        true
    }

    fn evaluate(&self, batch: &RecordBatch) -> crate::error::Result<ArrayRef> {
        Ok(Arc::clone(batch.column(self.0)))
    }
}

/// An argument that is the same list column whatever the input is.
#[derive(Debug)]
struct Fixed(ArrayRef);

impl ScalarExpr for Fixed {
    fn data_type(&self) -> DataType {
        self.0.data_type().clone()
    }

    fn nullable(&self) -> bool {
        true
    }

    fn evaluate(&self, _batch: &RecordBatch) -> crate::error::Result<ArrayRef> {
        Ok(Arc::clone(&self.0))
    }
}

fn lists(rows: Vec<Option<Vec<Option<i64>>>>) -> ArrayRef {
    Arc::new(arrow_array::ListArray::from_iter_primitive::<Int64Type, _, _>(rows))
}

#[test]
fn generate_repeats_rows_with_positions_weights_and_outer_nulls() {
    let (_root, ctx) = ctx();
    let xs = lists(vec![
        Some(vec![Some(10), None]),
        Some(vec![]),
        None,
        Some(vec![Some(7)]),
    ]);
    let id: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4]));
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("xs", xs.data_type().clone(), true),
    ]));
    let data = RecordBatch::try_new(Arc::clone(&input_schema), vec![id, xs]).unwrap();
    let batch =
        Batch::with_weights(data, Weights::Column(Int64Array::from(vec![5, 6, 7, -1]))).unwrap();
    let arg = Arc::new(Typed(1, input_schema.field(1).data_type().clone()));
    for (outer, want_ids, want_pos) in [
        (false, vec![1, 1, 4], vec![Some(0), Some(1), Some(0)]),
        (
            true,
            vec![1, 1, 2, 3, 4],
            vec![Some(0), Some(1), None, None, Some(0)],
        ),
    ] {
        let op = Generate::try_new(
            &input_schema,
            GeneratorKind::PosExplode,
            outer,
            Arc::clone(&arg) as Arc<dyn ScalarExpr>,
            None,
        )
        .unwrap();
        assert_eq!(op.name(), "Generate");
        let mut local = op.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
        let OperatorOutput::NeedMoreInput(Some(out)) = op.execute(&batch, &mut local).unwrap()
        else {
            unreachable!("one batch");
        };
        let ids: Vec<i64> = out
            .data()
            .column(0)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec();
        assert_eq!(ids, want_ids, "outer {outer}");
        let pos: Vec<Option<i32>> = out
            .data()
            .column(2)
            .as_primitive::<arrow_array::types::Int32Type>()
            .iter()
            .collect();
        assert_eq!(pos, want_pos, "outer {outer}");
        let col: Vec<Option<i64>> = out
            .data()
            .column(3)
            .as_primitive::<Int64Type>()
            .iter()
            .collect();
        let want_col: Vec<Option<i64>> = if outer {
            vec![Some(10), None, None, None, Some(7)]
        } else {
            vec![Some(10), None, Some(7)]
        };
        assert_eq!(col, want_col, "outer {outer}");
        // Each copy keeps its row's weight.
        let weights = match out.weights() {
            Weights::Column(w) => w.values().to_vec(),
            Weights::Const(_) => unreachable!("column weights"),
        };
        let want_w: Vec<i64> = if outer {
            vec![5, 5, 6, 7, -1]
        } else {
            vec![5, 5, -1]
        };
        assert_eq!(weights, want_w, "outer {outer}");
        // An empty input produces nothing.
        let empty = Batch::new(RecordBatch::new_empty(Arc::clone(&input_schema)));
        assert!(matches!(
            op.execute(&empty, &mut local).unwrap(),
            OperatorOutput::NeedMoreInput(None)
        ));
    }
}

#[test]
fn generate_without_input_columns_and_with_an_all_empty_input() {
    let (_root, ctx) = ctx();
    let input_schema = Arc::new(Schema::empty());
    let one_row = || {
        Batch::new(
            RecordBatch::try_new_with_options(
                Arc::clone(&input_schema),
                vec![],
                &arrow_array::RecordBatchOptions::new().with_row_count(Some(1)),
            )
            .unwrap(),
        )
    };
    let arg = Arc::new(Fixed(lists(vec![Some(vec![Some(1), Some(2)])])));
    let op = Generate::try_new(&input_schema, GeneratorKind::Explode, false, arg, None).unwrap();
    let mut local = op.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
    let OperatorOutput::NeedMoreInput(Some(out)) = op.execute(&one_row(), &mut local).unwrap()
    else {
        unreachable!("one batch");
    };
    assert_eq!(
        out.data()
            .column(0)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec(),
        [1, 2]
    );
    // Rows that all have no element produce no batch.
    let none = Arc::new(Fixed(lists(vec![Some(vec![])])));
    let op = Generate::try_new(&input_schema, GeneratorKind::Explode, false, none, None).unwrap();
    let mut local = op.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
    assert!(matches!(
        op.execute(&one_row(), &mut local).unwrap(),
        OperatorOutput::NeedMoreInput(None)
    ));
}

#[test]
fn generate_refuses_arguments_it_cannot_take() {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
    for (kind, ty) in [
        (GeneratorKind::Explode, DataType::Int64),
        (GeneratorKind::PosExplode, DataType::Utf8),
        (
            GeneratorKind::Inline,
            DataType::new_list(DataType::Int64, true),
        ),
    ] {
        let err =
            Generate::try_new(&schema, kind, false, Arc::new(Typed(0, ty)), None).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::InternalError, "{kind:?}");
    }
    // A column that is not a list fails when it is evaluated.
    let op = Generate::try_new(
        &schema,
        GeneratorKind::Explode,
        false,
        Arc::new(Typed(0, DataType::new_list(DataType::Int64, true))),
        None,
    )
    .unwrap();
    let (_root, ctx) = ctx();
    let mut local = op.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
    let err = op.execute(&ints(&[1]), &mut local).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError);
}

#[test]
fn generate_does_not_carry_its_argument_column() {
    let (_root, ctx) = ctx();
    let xs = lists(vec![Some(vec![Some(1), Some(2)]), Some(vec![Some(3)])]);
    let id: ArrayRef = Arc::new(Int64Array::from(vec![10, 20]));
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("xs", xs.data_type().clone(), true),
        Field::new("id", DataType::Int64, false),
    ]));
    let batch = Batch::new(RecordBatch::try_new(Arc::clone(&input_schema), vec![xs, id]).unwrap());
    let arg = Arc::new(Typed(0, input_schema.field(0).data_type().clone()));
    let op = Generate::try_new(&input_schema, GeneratorKind::Explode, false, arg, Some(0)).unwrap();
    // `id`, then the element: the list column is read and dropped.
    assert_eq!(op.schema().fields().len(), 2);
    assert_eq!(op.schema().field(0).name(), "id");
    let mut local = op.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
    let OperatorOutput::NeedMoreInput(Some(out)) = op.execute(&batch, &mut local).unwrap() else {
        unreachable!("one batch");
    };
    assert_eq!(out.data().num_columns(), 2);
    assert_eq!(
        out.data()
            .column(0)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec(),
        [10, 10, 20]
    );
    assert_eq!(
        out.data()
            .column(1)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec(),
        [1, 2, 3]
    );
}

#[test]
fn generate_splits_its_output_at_the_batch_size() {
    let (_root, ctx) = ctx_with(ExecConfig {
        batch_size: 3,
        ..ExecConfig::default()
    });
    let input_schema = Arc::new(Schema::empty());
    let one_row = || {
        Batch::new(
            RecordBatch::try_new_with_options(
                Arc::clone(&input_schema),
                vec![],
                &arrow_array::RecordBatchOptions::new().with_row_count(Some(2)),
            )
            .unwrap(),
        )
    };
    // Two rows: 5 elements, then an empty array an outer generator keeps.
    let arg = Arc::new(Fixed(lists(vec![
        Some((1..=5).map(Some).collect()),
        Some(vec![]),
    ])));
    let op = Generate::try_new(&input_schema, GeneratorKind::PosExplode, true, arg, None).unwrap();
    let mut local = op.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
    let input = one_row();
    let mut sizes = Vec::new();
    let mut values = Vec::new();
    let mut positions = Vec::new();
    loop {
        let (out, last) = match op.execute(&input, &mut local).unwrap() {
            OperatorOutput::HaveMoreOutput(b) => (Some(b), false),
            OperatorOutput::NeedMoreInput(b) => (b, true),
            other => unreachable!("unexpected {other:?}"),
        };
        if let Some(b) = out {
            sizes.push(b.num_rows());
            let p = b
                .data()
                .column(0)
                .as_primitive::<arrow_array::types::Int32Type>();
            positions.extend(p.iter());
            let v = b.data().column(1).as_primitive::<Int64Type>();
            values.extend(v.iter());
        }
        if last {
            break;
        }
    }
    assert_eq!(sizes, [3, 3]);
    assert_eq!(values, [Some(1), Some(2), Some(3), Some(4), Some(5), None]);
    assert_eq!(
        positions,
        [Some(0), Some(1), Some(2), Some(3), Some(4), None]
    );
}

/// Every output batch of `op` over `input`, fed until it asks for more input.
fn generate_all(op: &Generate, input: &Batch, local: &mut LocalState) -> Vec<Batch> {
    let mut out = Vec::new();
    loop {
        match op
            .execute(input, local)
            .unwrap_or_else(|e| unreachable!("{e}"))
        {
            OperatorOutput::HaveMoreOutput(b) => out.push(b),
            OperatorOutput::NeedMoreInput(b) => {
                out.extend(b);
                return out;
            }
            other => unreachable!("unexpected {other:?}"),
        }
    }
}

#[test]
fn generate_with_batch_size_one_keeps_each_rows_weight_across_a_split() {
    let (_root, ctx) = ctx_with(ExecConfig {
        batch_size: 1,
        ..ExecConfig::default()
    });
    let xs = lists(vec![
        Some(vec![Some(10), Some(11), Some(12)]),
        Some(vec![Some(7)]),
    ]);
    let id: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("xs", xs.data_type().clone(), true),
    ]));
    let data = RecordBatch::try_new(Arc::clone(&input_schema), vec![id, xs]).unwrap();
    let input = Batch::with_weights(data, Weights::Column(Int64Array::from(vec![5, -2]))).unwrap();
    let arg = Arc::new(Typed(1, input_schema.field(1).data_type().clone()));
    let op = Generate::try_new(&input_schema, GeneratorKind::Explode, false, arg, Some(1)).unwrap();
    let mut local = op.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
    let out = generate_all(&op, &input, &mut local);
    assert_eq!(out.iter().map(Batch::num_rows).collect::<Vec<_>>(), [1; 4]);
    let ids: Vec<i64> = out
        .iter()
        .map(|b| b.data().column(0).as_primitive::<Int64Type>().value(0))
        .collect();
    let vals: Vec<i64> = out
        .iter()
        .map(|b| b.data().column(1).as_primitive::<Int64Type>().value(0))
        .collect();
    assert_eq!(ids, [1, 1, 1, 2]);
    assert_eq!(vals, [10, 11, 12, 7]);
    let weights: Vec<i64> = out
        .iter()
        .map(|b| match b.weights() {
            Weights::Column(w) => w.value(0),
            Weights::Const(w) => *w,
        })
        .collect();
    assert_eq!(weights, [5, 5, 5, -2], "the split keeps the row's weight");
}

#[test]
fn inline_skips_a_null_struct_and_splits_mid_row() {
    let (_root, ctx) = ctx_with(ExecConfig {
        batch_size: 2,
        ..ExecConfig::default()
    });
    let field = Arc::new(Field::new("a", DataType::Int64, true));
    let item = Arc::new(Field::new(
        "item",
        DataType::Struct(vec![Arc::clone(&field)].into()),
        true,
    ));
    let structs = arrow_array::StructArray::new(
        vec![field].into(),
        vec![Arc::new(Int64Array::from(vec![1, 0, 3])) as ArrayRef],
        Some(arrow_buffer::NullBuffer::from(vec![true, false, true])),
    );
    let list: ArrayRef = Arc::new(arrow_array::ListArray::new(
        item,
        arrow_buffer::OffsetBuffer::from_lengths([3]),
        Arc::new(structs),
        None,
    ));
    let input_schema = Arc::new(Schema::new(vec![Field::new(
        "xs",
        list.data_type().clone(),
        true,
    )]));
    let input = Batch::new(RecordBatch::try_new(Arc::clone(&input_schema), vec![list]).unwrap());
    let arg = Arc::new(Typed(0, input_schema.field(0).data_type().clone()));
    let op = Generate::try_new(&input_schema, GeneratorKind::Inline, false, arg, Some(0)).unwrap();
    let mut local = op.local_state(&ctx).unwrap_or_else(|e| unreachable!("{e}"));
    let out = generate_all(&op, &input, &mut local);
    assert_eq!(out.iter().map(Batch::num_rows).collect::<Vec<_>>(), [2, 1]);
    let vals: Vec<Option<i64>> = out
        .iter()
        .flat_map(|b| {
            b.data()
                .column(0)
                .as_primitive::<Int64Type>()
                .iter()
                .collect::<Vec<_>>()
        })
        .collect();
    // Spark's inline turns a NULL struct into a row of NULL fields.
    assert_eq!(vals, [Some(1), None, Some(3)]);
}
