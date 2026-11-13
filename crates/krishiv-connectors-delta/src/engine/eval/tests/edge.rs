// SPDX-License-Identifier: Apache-2.0
//! Edge cases of the evaluator beyond the parity table: nested nulls, `create_many` type
//! checks, error paths and type-check refusals.

use super::*;

/// `s: struct<x: int, t: struct<y: int>>` over two rows; `s` is NULL in row 1 while its
/// physical children hold 7 there.
fn nested_batch() -> RecordBatch {
    let t = StructArray::try_new(
        vec![Field::new("y", ArrowType::Int32, true)].into(),
        vec![ints(&[Some(7), Some(7)])],
        None,
    )
    .unwrap();
    let s = StructArray::try_new(
        vec![
            Field::new("x", ArrowType::Int32, true),
            Field::new("t", t.data_type().clone(), true),
        ]
        .into(),
        vec![ints(&[Some(7), Some(7)]), Arc::new(t)],
        Some(vec![true, false].into()),
    )
    .unwrap();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "s",
            s.data_type().clone(),
            true,
        )])),
        vec![Arc::new(s)],
    )
    .unwrap()
}

#[test]
fn a_null_struct_hides_its_children() {
    let b = nested_batch();
    let x = Expr::column(["s", "x"]);
    let got = evaluate_expression(&ctx(), &x, &b, Some(&DataType::INTEGER)).unwrap();
    assert_eq!(show(&got), ["7", "NULL"]);
    let y = Expr::column(["s", "t", "y"]);
    let got = evaluate_expression(&ctx(), &y, &b, Some(&DataType::INTEGER)).unwrap();
    assert_eq!(show(&got), ["7", "NULL"]);
    // An extracted struct carries its ancestors' nulls down to its own children.
    let t = evaluate_expression(&ctx(), &Expr::column(["s", "t"]), &b, None).unwrap();
    assert_eq!(show(&t), ["{y: 7}", "NULL"]);
    assert!(t.as_struct().column(0).is_null(1));
    // Through a predicate: `s.x = 7` is NULL, not true, on the NULL struct.
    let eq = Pred::eq(Expr::column(["s", "x"]), lit(7));
    let got = evaluate_predicate(&ctx(), &eq, &b, false).unwrap();
    assert_eq!(show(&(Arc::new(got) as ArrayRef)), ["true", "NULL"]);
    let null = Pred::is_null(Expr::column(["s", "x"]));
    let got = evaluate_predicate(&ctx(), &null, &b, false).unwrap();
    assert_eq!(show(&(Arc::new(got) as ArrayRef)), ["false", "true"]);
}

#[test]
fn data_skipping_keeps_a_file_whose_stats_struct_is_null() {
    // Stats `minValues.s.x` and `maxValues.s.x`: file 0 has real stats (10..20), file 1 has a
    // NULL `s` (no stats) whose physical `x` slots also say 10..20.
    let stats = |name: &str, v: i32| {
        let s = StructArray::try_new(
            vec![Field::new("x", ArrowType::Int32, true)].into(),
            vec![ints(&[Some(v), Some(v)])],
            Some(vec![true, false].into()),
        )
        .unwrap();
        let outer = StructArray::try_new(
            vec![Field::new("s", s.data_type().clone(), true)].into(),
            vec![Arc::new(s) as ArrayRef],
            None,
        )
        .unwrap();
        (
            Field::new(name, outer.data_type().clone(), true),
            Arc::new(outer) as ArrayRef,
        )
    };
    let (min_f, min) = stats("minValues", 10);
    let (max_f, max) = stats("maxValues", 20);
    let b =
        RecordBatch::try_new(Arc::new(Schema::new(vec![min_f, max_f])), vec![min, max]).unwrap();
    // `s.x = 7` as the kernel rewrites it: NOT(min > 7) AND NOT(max < 7), kept unless false.
    let skipping = Pred::and(
        Pred::not(Pred::gt(Expr::column(["minValues", "s", "x"]), lit(7))),
        Pred::not(Pred::lt(Expr::column(["maxValues", "s", "x"]), lit(7))),
    );
    let out = evaluate_predicate(&ctx(), &skipping, &b, false).unwrap();
    assert_eq!(
        show(&(Arc::new(out.clone()) as ArrayRef)),
        ["false", "NULL"]
    );
    let kept = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "output",
            ArrowType::Boolean,
            true,
        )])),
        vec![Arc::new(out)],
    )
    .unwrap();
    let filter = Pred::distinct(col("output"), lit(false));
    let keep = evaluate_predicate(&ctx(), &filter, &kept, false).unwrap();
    assert_eq!(show(&(Arc::new(keep) as ArrayRef)), ["false", "true"]);
}
