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

fn decimal(bits: i128, precision: u8, scale: u8) -> Scalar {
    Scalar::decimal(bits, precision, scale).unwrap()
}

fn one_field(name: &str, t: DataType) -> Arc<StructType> {
    Arc::new(StructType::try_new([StructField::nullable(name, t)]).unwrap())
}

#[test]
fn create_many_refuses_a_scalar_of_another_decimal_scale() {
    // The kernel's own `create_many` copies the raw bits (12.34 at scale 2 would read 1.234 at
    // scale 3); Krishiv refuses the row instead of rescaling.
    let field = DataType::decimal(5, 3).unwrap();
    let e = err(handler().create_many(
        one_field("d", field.clone()),
        vec![vec![decimal(1234, 4, 2)]],
    ));
    let text = e.to_string();
    assert!(text.contains("field 'd'"), "{text}");
    assert!(text.contains("does not match the declared type"), "{text}");
    // Same scale, other precision: still refused (the declared type is exact).
    assert!(
        handler()
            .create_many(
                one_field("d", field.clone()),
                vec![vec![decimal(1234, 4, 3)]]
            )
            .is_err()
    );
    // Nested in a struct: refused too.
    let inner = StructType::try_new([StructField::nullable("d", field.clone())]).unwrap();
    let wrong = Scalar::Struct(
        delta_kernel::expressions::StructData::try_new(
            vec![StructField::nullable("d", DataType::decimal(4, 2).unwrap())],
            vec![decimal(1234, 4, 2)],
        )
        .unwrap(),
    );
    assert!(
        handler()
            .create_many(
                one_field("s", DataType::Struct(Box::new(inner))),
                vec![vec![wrong]]
            )
            .is_err()
    );
    // A zoned timestamp into a `timestamp_ntz` field would only be relabelled: refused.
    assert!(
        handler()
            .create_many(
                one_field("t", DataType::TIMESTAMP_NTZ),
                vec![vec![Scalar::Timestamp(1)]]
            )
            .is_err()
    );
    // The exact type is accepted, and a NULL of any decimal type is a NULL.
    let data = handler()
        .create_many(
            one_field("d", field),
            vec![
                vec![decimal(12_340, 5, 3)],
                vec![Scalar::Null(DataType::decimal(4, 2).unwrap())],
            ],
        )
        .unwrap();
    let b = KrishivEngineData::batch_of(data.as_ref()).unwrap();
    assert_eq!(show(b.column(0)), ["12.340", "NULL"]);
}

fn eval_pred(p: &Pred, b: &RecordBatch) -> KernelResult<Vec<String>> {
    evaluate_predicate(&ctx(), p, b, false).map(|a| show(&(Arc::new(a) as ArrayRef)))
}

/// One-row columns of the string and binary encodings and of lists of each type.
fn encodings_batch() -> RecordBatch {
    use arrow_array::types::{
        Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int64Type,
        TimestampMicrosecondType,
    };
    use arrow_array::{
        BinaryViewArray, LargeListArray, LargeStringArray, PrimitiveArray, StringViewArray,
    };
    fn list_of<T: arrow_array::ArrowPrimitiveType>(
        values: PrimitiveArray<T>,
    ) -> arrow_array::ListArray {
        let field = Arc::new(Field::new("element", values.data_type().clone(), true));
        let len = i32::try_from(values.len()).unwrap();
        arrow_array::ListArray::new(
            field,
            arrow_buffer::OffsetBuffer::new(vec![0, len].into()),
            Arc::new(values),
            None,
        )
    }
    let strings = |values: ArrayRef| {
        let field = Arc::new(Field::new("element", values.data_type().clone(), true));
        let len = i32::try_from(values.len()).unwrap();
        Arc::new(arrow_array::ListArray::new(
            field,
            arrow_buffer::OffsetBuffer::new(vec![0, len].into()),
            values,
            None,
        )) as ArrayRef
    };
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("v", Arc::new(StringViewArray::from(vec!["x"]))),
        ("ls", Arc::new(LargeStringArray::from(vec!["y"]))),
        ("bv", Arc::new(BinaryViewArray::from(vec![&b"x"[..]]))),
        ("sl", strings(Arc::new(StringArray::from(vec!["a", "x"])))),
        (
            "vl",
            strings(Arc::new(StringViewArray::from(vec!["a", "x"]))),
        ),
        (
            "lsl",
            strings(Arc::new(LargeStringArray::from(vec!["a", "x"]))),
        ),
        (
            "bvl",
            strings(Arc::new(BinaryViewArray::from(vec![&b"x"[..]]))),
        ),
        (
            "ll",
            Arc::new(LargeListArray::from_iter_primitive::<Int32Type, _, _>(
                vec![Some(vec![Some(1), Some(2)])],
            )),
        ),
        (
            "i8",
            Arc::new(list_of(PrimitiveArray::<Int8Type>::from(vec![1_i8]))),
        ),
        (
            "i16",
            Arc::new(list_of(PrimitiveArray::<Int16Type>::from(vec![1_i16]))),
        ),
        (
            "i32",
            Arc::new(list_of(PrimitiveArray::<Int32Type>::from(vec![1]))),
        ),
        (
            "i64",
            Arc::new(list_of(PrimitiveArray::<Int64Type>::from(vec![1_i64]))),
        ),
        (
            "f32",
            Arc::new(list_of(PrimitiveArray::<Float32Type>::from(vec![1.5_f32]))),
        ),
        (
            "f64",
            Arc::new(list_of(PrimitiveArray::<Float64Type>::from(vec![1.5_f64]))),
        ),
        (
            "d",
            Arc::new(list_of(PrimitiveArray::<Date32Type>::from(vec![19_000]))),
        ),
        (
            "t",
            Arc::new(list_of(
                PrimitiveArray::<TimestampMicrosecondType>::from(vec![5_i64]).with_timezone("UTC"),
            )),
        ),
        (
            "dec",
            Arc::new(list_of(
                PrimitiveArray::<Decimal128Type>::from(vec![1234_i128])
                    .with_precision_and_scale(4, 2)
                    .unwrap(),
            )),
        ),
        ("bools", strings(Arc::new(BooleanArray::from(vec![true])))),
    ];
    let fields: Vec<Field> = columns
        .iter()
        .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
        .collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns.into_iter().map(|(_, a)| a).collect(),
    )
    .unwrap()
}

fn is_in(needle: impl Into<Scalar>, column: &str) -> Pred {
    Pred::binary(
        delta_kernel::expressions::BinaryPredicateOp::In,
        lit(needle),
        col(column),
    )
}

#[test]
fn comparisons_across_string_and_binary_encodings() {
    let b = encodings_batch();
    assert_eq!(
        eval_pred(&Pred::eq(col("v"), lit("x")), &b).unwrap(),
        ["true"]
    );
    assert_eq!(
        eval_pred(&Pred::eq(col("ls"), lit("x")), &b).unwrap(),
        ["false"]
    );
    assert_eq!(
        eval_pred(&Pred::eq(col("bv"), lit(Scalar::Binary(b"x".to_vec()))), &b).unwrap(),
        ["true"]
    );
    // NOT(a > 2) is a <= 2.
    let not_gt = Pred::not(Pred::gt(col("a"), lit(2)));
    assert_eq!(
        eval_pred(&not_gt, &batch()).unwrap(),
        ["true", "true", "false", "false", "false"]
    );
}

#[test]
fn in_over_each_list_encoding_and_element_type() {
    let b = encodings_batch();
    for (needle, column, want) in [
        (Scalar::from("x"), "sl", "true"),
        (Scalar::from("x"), "vl", "true"),
        (Scalar::from("z"), "lsl", "false"),
        (Scalar::from("x"), "lsl", "true"),
        (Scalar::Integer(2), "ll", "true"),
        (Scalar::Byte(1), "i8", "true"),
        (Scalar::Short(1), "i16", "true"),
        (Scalar::Long(1), "i64", "true"),
        (Scalar::Long(2), "i64", "false"),
        (Scalar::Float(1.5), "f32", "true"),
        (Scalar::Double(1.5), "f64", "true"),
        (Scalar::Date(19_000), "d", "true"),
        (Scalar::Timestamp(5), "t", "true"),
        (Scalar::decimal(1234, 4, 2).unwrap(), "dec", "true"),
    ] {
        let got = eval_pred(&is_in(needle.clone(), column), &b);
        assert_eq!(got.unwrap(), [want], "{needle} IN {column}");
    }
    // A binary needle over a list of binary views has no IN kernel.
    let e = err(eval_pred(&is_in(Scalar::Binary(b"x".to_vec()), "bvl"), &b));
    assert!(e.to_string().contains("Bad Comparison"), "{e}");
    let e = err(eval_pred(&is_in(true, "bools"), &b));
    assert!(e.to_string().contains("Bad Comparison"), "{e}");
    // A needle of another type than the list's elements is an error, not a panic.
    for (needle, column) in [
        (Scalar::Long(1), "i32"),
        (Scalar::Integer(1), "i64"),
        (Scalar::decimal(12_340, 5, 3).unwrap(), "dec"),
    ] {
        let e = err(eval_pred(&is_in(needle.clone(), column), &b));
        assert!(
            e.to_string().contains("Bad Comparison"),
            "{needle} IN {column}: {e}"
        );
    }
    // The right side must be a list column.
    let e = err(eval_pred(&is_in("x", "ls"), &b));
    assert!(e.to_string().contains("Cannot cast to list array"), "{e}");
}
