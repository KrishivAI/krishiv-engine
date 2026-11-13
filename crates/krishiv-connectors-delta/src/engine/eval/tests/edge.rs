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

use arrow_array::{LargeListArray, ListArray};
use delta_kernel::schema::MapType;

use crate::engine::eval::types::{check, conform};
use crate::engine::schema::to_arrow_type;

#[test]
fn check_widens_integers_into_wide_enough_decimals() {
    let dec = |p, s| DataType::decimal(p, s).unwrap();
    // An integer fits a decimal with as many integer digits: Int8 3, Int16 5, Int32 10, Int64 20.
    for (from, fits, narrow) in [
        (ArrowType::Int8, dec(3, 0), dec(2, 0)),
        (ArrowType::Int16, dec(7, 2), dec(6, 2)),
        (ArrowType::Int32, dec(12, 2), dec(11, 2)),
        (ArrowType::Int64, dec(20, 0), dec(19, 0)),
    ] {
        assert!(check(&fits, &from).is_ok(), "{from} into {fits}");
        let e = err(check(&narrow, &from));
        assert!(
            e.to_string().contains("Incorrect datatype"),
            "{from} into {narrow}: {e}"
        );
    }
    // A string or a float never widens into a decimal.
    assert!(check(&dec(38, 0), &ArrowType::Utf8).is_err());
    assert!(check(&dec(38, 0), &ArrowType::Float64).is_err());
}

#[test]
fn check_matches_lists_maps_and_structs() {
    let list = DataType::Array(Box::new(ArrayType::new(DataType::LONG, true)));
    let int_list = ArrowType::List(Arc::new(Field::new("item", ArrowType::Int32, true)));
    let large = ArrowType::LargeList(Arc::new(Field::new("item", ArrowType::Int64, true)));
    assert!(check(&list, &int_list).is_ok());
    assert!(check(&list, &large).is_ok());
    let strings = ArrowType::List(Arc::new(Field::new("item", ArrowType::Utf8, true)));
    assert!(check(&list, &strings).is_err());
    // A list type is not a primitive and a primitive is not a list.
    assert!(
        err(check(&list, &ArrowType::Int64))
            .to_string()
            .contains("Incorrect datatype")
    );
    let map = DataType::Map(Box::new(MapType::new(
        DataType::STRING,
        DataType::LONG,
        true,
    )));
    let arrow_map = to_arrow_type(&map).unwrap();
    assert!(check(&map, &arrow_map).is_ok());
    let ArrowType::Map(entries, sorted) = &arrow_map else {
        panic!("{arrow_map}")
    };
    // Entries that are not a key/value struct are refused, not indexed.
    let not_struct = ArrowType::Map(
        Arc::new(entries.as_ref().clone().with_data_type(ArrowType::Int32)),
        *sorted,
    );
    assert!(
        err(check(&map, &not_struct))
            .to_string()
            .contains("Incorrect datatype")
    );
    let ArrowType::Struct(kv) = entries.data_type() else {
        panic!("{entries}")
    };
    let mut three: Vec<_> = kv.iter().cloned().collect();
    three.push(Arc::new(Field::new("extra", ArrowType::Int32, true)));
    let three = ArrowType::Map(
        Arc::new(
            entries
                .as_ref()
                .clone()
                .with_data_type(ArrowType::Struct(three.into())),
        ),
        *sorted,
    );
    assert!(check(&map, &three).is_err());
    // A struct must have every kernel field; extra Arrow fields are ignored.
    let st = DataType::Struct(Box::new(
        StructType::try_new([
            StructField::nullable("a", DataType::INTEGER),
            StructField::nullable("b", DataType::STRING),
        ])
        .unwrap(),
    ));
    let only_a = ArrowType::Struct(
        vec![
            Field::new("a", ArrowType::Int32, true),
            Field::new("z", ArrowType::Utf8, true),
        ]
        .into(),
    );
    let e = err(check(&st, &only_a));
    assert!(e.to_string().contains("Missing Struct fields b"), "{e}");
    let both = ArrowType::Struct(
        vec![
            Field::new("b", ArrowType::Utf8, true),
            Field::new("a", ArrowType::Int16, true),
            Field::new("z", ArrowType::Utf8, true),
        ]
        .into(),
    );
    assert!(check(&st, &both).is_ok());
    let a_wrong = ArrowType::Struct(
        vec![
            Field::new("a", ArrowType::Utf8, true),
            Field::new("b", ArrowType::Utf8, true),
        ]
        .into(),
    );
    assert!(check(&st, &a_wrong).is_err());
}

#[test]
fn conform_rebuilds_lists_maps_and_large_lists_as_declared() {
    let ints = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        Some(vec![Some(1), None]),
        None,
        Some(vec![]),
    ]);
    let longs = DataType::Array(Box::new(ArrayType::new(DataType::LONG, true)));
    let target = to_arrow_type(&longs).unwrap();
    let out = conform(&(Arc::new(ints.clone()) as ArrayRef), &target).unwrap();
    assert_eq!(out.data_type(), &target);
    assert_eq!(show(&out), ["[1, NULL]", "NULL", "[]"]);
    // A null element where the declared type has none is an error, not a panic.
    let strict = to_arrow_type(&DataType::Array(Box::new(ArrayType::new(
        DataType::LONG,
        false,
    ))))
    .unwrap();
    assert!(conform(&(Arc::new(ints) as ArrayRef), &strict).is_err());
    // A large list (or a list view) becomes a list.
    let large =
        LargeListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![Some(3)]), None]);
    let out = conform(&(Arc::new(large) as ArrayRef), &target).unwrap();
    assert_eq!(out.data_type(), &target);
    assert_eq!(show(&out), ["[3]", "NULL"]);
    // A map's entries are rebuilt with the declared names and value type.
    let mut map = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    map.keys().append_value("k");
    map.values().append_value(5);
    map.append(true).unwrap();
    map.append(false).unwrap();
    let map: ArrayRef = Arc::new(map.finish());
    let declared = to_arrow_type(&DataType::Map(Box::new(MapType::new(
        DataType::STRING,
        DataType::LONG,
        true,
    ))))
    .unwrap();
    let out = conform(&map, &declared).unwrap();
    assert_eq!(out.data_type(), &declared);
    assert_eq!(show(&out), ["{k: 5}", "NULL"]);
}

#[test]
fn conform_refuses_shape_and_type_mismatches() {
    let one = StructArray::try_new(
        vec![Field::new("a", ArrowType::Int32, true)].into(),
        vec![ints(&[Some(1)])],
        None,
    )
    .unwrap();
    let two = ArrowType::Struct(
        vec![
            Field::new("a", ArrowType::Int32, true),
            Field::new("b", ArrowType::Int32, true),
        ]
        .into(),
    );
    let e = err(conform(&(Arc::new(one) as ArrayRef), &two));
    assert!(
        e.to_string()
            .contains("Passed struct had 1 columns, but the target type has 2"),
        "{e}"
    );
    // A string is not narrowed to an integer, nor a long to an int.
    let strings: ArrayRef = Arc::new(StringArray::from(vec!["1"]));
    let e = err(conform(&strings, &ArrowType::Int32));
    assert!(e.to_string().contains("Incorrect datatype"), "{e}");
    let longs: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![1]));
    assert!(conform(&longs, &ArrowType::Int32).is_err());
    // A list is not a struct.
    assert!(conform(&strings, &two).is_err());
}

fn struct_of(fields: &[(&str, DataType)]) -> DataType {
    DataType::Struct(Box::new(
        StructType::try_new(
            fields
                .iter()
                .map(|(n, t)| StructField::nullable(*n, t.clone())),
        )
        .unwrap(),
    ))
}

fn eval(e: &Expr, t: Option<&DataType>) -> KernelResult<ArrayRef> {
    evaluate_expression(&ctx(), e, &batch(), t)
}

#[test]
fn expressions_refuse_a_result_type_they_cannot_give() {
    let int = DataType::INTEGER;
    let e = err(eval(&Expr::StructPatch(patch()), None));
    assert!(e.to_string().contains("Data type is required"), "{e}");
    let e = err(eval(&Expr::StructPatch(patch()), Some(&int)));
    assert!(e.to_string().contains("Data type is required"), "{e}");
    let pred = Expr::from_pred(Pred::is_null(col("a")));
    let e = err(eval(&pred, Some(&int)));
    assert!(e.to_string().contains("produces boolean output"), "{e}");
    let to_json = Expr::unary(UnaryExpressionOp::ToJson, col("st"));
    let e = err(eval(&to_json, Some(&int)));
    assert!(e.to_string().contains("requires STRING output"), "{e}");
    let array = Expr::array([col("a")]);
    let e = err(eval(&array, Some(&int)));
    assert!(e.to_string().contains("requires a DataType::Array"), "{e}");
    // The struct's field count must match its declared type.
    let two = struct_of(&[("a", int.clone()), ("b", int.clone())]);
    let e = err(eval(&Expr::struct_from([col("a")]), Some(&two)));
    assert!(e.to_string().contains("field count mismatch"), "{e}");
    // COALESCE of no inputs, and of inputs that are not the requested type.
    let e = err(eval(&Expr::coalesce(Vec::<Expr>::new()), Some(&int)));
    assert!(e.to_string().contains("empty COALESCE"), "{e}");
    let e = err(eval(
        &Expr::coalesce([col("b"), col("a")]),
        Some(&DataType::LONG),
    ));
    assert!(e.to_string().contains("Requested result type"), "{e}");
    // ARRAY inputs of different types.
    let e = err(eval(&Expr::array([col("a"), col("s")]), None));
    assert!(e.to_string().contains("share the same element type"), "{e}");
}

#[test]
fn arithmetic_structs_and_arrays_with_declared_types() {
    let int = DataType::INTEGER;
    let minus = Expr::binary(BinaryExpressionOp::Minus, col("a"), lit(1));
    assert_eq!(
        show(&eval(&minus, Some(&int)).unwrap()),
        ["0", "1", "2", "3", "4"]
    );
    let times = Expr::binary(BinaryExpressionOp::Multiply, col("a"), col("b"));
    assert_eq!(
        show(&eval(&times, Some(&int)).unwrap()),
        ["NULL", "20", "NULL", "160", "NULL"]
    );
    // A nullability predicate with no NULLs of its own.
    let st =
        Expr::struct_with_nullability_from([col("a")], Expr::from_pred(Pred::gt(col("a"), lit(3))));
    let a = struct_of(&[("a", int.clone())]);
    assert_eq!(
        show(&eval(&st, Some(&a)).unwrap()),
        ["NULL", "NULL", "NULL", "{a: 4}", "{a: 5}"]
    );
    // A struct of no fields keeps the row count.
    let empty = DataType::Struct(Box::new(StructType::try_new([]).unwrap()));
    let out = eval(&Expr::struct_from(Vec::<Expr>::new()), Some(&empty)).unwrap();
    assert_eq!(out.len(), 5);
    assert_eq!(out.null_count(), 0);
    // ARRAY checked against its declared type.
    let ints = DataType::Array(Box::new(ArrayType::new(int, true)));
    let out = eval(&Expr::array([col("a"), col("b")]), Some(&ints)).unwrap();
    assert_eq!(show(&out)[1], "[2, 10]");
}

#[test]
fn struct_patch_prepends_appends_and_checks_its_schema() {
    let int = DataType::INTEGER;
    let mut p = patch();
    p.prepended_fields = vec![Arc::new(lit(1))];
    p.appended_fields = vec![Arc::new(lit(2))];
    let full = struct_of(&[
        ("first", int.clone()),
        ("x", int.clone()),
        ("k", int.clone()),
        ("last", int.clone()),
    ]);
    let out = eval(&Expr::StructPatch(p.clone()), Some(&full)).unwrap();
    assert_eq!(
        show(&out),
        [
            "{first: 1, x: 1, k: 7, last: 2}",
            "{first: 1, x: 2, k: 7, last: 2}",
            "NULL",
            "{first: 1, x: 4, k: 7, last: 2}",
            "{first: 1, x: 5, k: 7, last: 2}",
        ]
    );
    // One output field short, or one too many.
    let short = struct_of(&[
        ("first", int.clone()),
        ("x", int.clone()),
        ("k", int.clone()),
    ]);
    let e = err(eval(&Expr::StructPatch(p.clone()), Some(&short)));
    assert!(e.to_string().contains("Too few fields"), "{e}");
    let long = struct_of(&[
        ("first", int.clone()),
        ("x", int.clone()),
        ("k", int.clone()),
        ("last", int.clone()),
        ("extra", int.clone()),
    ]);
    let e = err(eval(&Expr::StructPatch(p), Some(&long)));
    assert!(e.to_string().contains("Too many fields"), "{e}");
    // A required patch of a field the input does not have.
    let mut missing = patch();
    missing.field_patches = HashMap::from([(
        "nope".to_owned(),
        ExpressionFieldPatch {
            keep_input: true,
            insertions: vec![],
            optional: false,
        },
    )]);
    let xy = struct_of(&[("x", int), ("y", DataType::STRING)]);
    let e = err(eval(&Expr::StructPatch(missing.clone()), Some(&xy)));
    assert!(e.to_string().contains("non-optional field patches"), "{e}");
    // The same patch marked optional is skipped.
    missing.field_patches.get_mut("nope").unwrap().optional = true;
    assert_eq!(
        show(&eval(&Expr::StructPatch(missing), Some(&xy)).unwrap())[0],
        "{x: 1, y: p}"
    );
}
#[test]
fn parse_json_reads_each_string_encoding() {
    use arrow_array::{LargeStringArray, StringViewArray};
    let row = r#"{"numRecords":3}"#;
    for strings in [
        Arc::new(StringArray::from(vec![row])) as ArrayRef,
        Arc::new(LargeStringArray::from(vec![row])),
        Arc::new(StringViewArray::from(vec![row])),
    ] {
        let out = super::json::parse_json(&ctx(), &strings, &stats_schema()).unwrap();
        assert_eq!(
            show(&out),
            ["{numRecords: 3, minValues: NULL}"],
            "{}",
            strings.data_type()
        );
    }
    let e = err(super::json::parse_json(
        &ctx(),
        &ints(&[Some(1)]),
        &stats_schema(),
    ));
    assert!(e.to_string().contains("Expected string array"), "{e}");
    // No rows: no result rows.
    let none: ArrayRef = Arc::new(StringArray::from(Vec::<&str>::new()));
    let out = super::json::parse_json(&ctx(), &none, &stats_schema()).unwrap();
    assert_eq!(out.len(), 0);
    // Two objects in one string make the batch unparseable: NULL stats for every row.
    let two: ArrayRef = Arc::new(StringArray::from(vec![
        row,
        r#"{"numRecords":1}{"numRecords":2}"#,
    ]));
    let out = super::json::parse_json(&ctx(), &two, &stats_schema()).unwrap();
    assert_eq!(show(&out), ["NULL", "NULL"]);
}

#[test]
fn map_to_struct_binary_ntz_and_refusals() {
    let schema = StructType::try_new([
        StructField::nullable("bin", DataType::BINARY),
        StructField::nullable("ntz", DataType::TIMESTAMP_NTZ),
    ])
    .unwrap();
    let to_struct = |b: &RecordBatch, schema: &StructType, options| {
        evaluate_expression(
            &ctx(),
            &Expr::map_to_struct(col("pv"), options),
            b,
            Some(&DataType::Struct(Box::new(schema.clone()))),
        )
    };
    let b = one_map(&[("bin", "ab"), ("ntz", "2024-01-02 03:04:05")]);
    let out = to_struct(&b, &schema, MapToStructOptions::default()).unwrap();
    assert_eq!(show(&out), ["{bin: 6162, ntz: 2024-01-02T03:04:05}"]);
    let bad = one_map(&[("ntz", "not a time")]);
    let e = err(to_struct(&bad, &schema, MapToStructOptions::default()));
    assert!(matches!(e, KernelError::ParseError(..)), "{e:?}");
    // UTC under any of its names is the default zone.
    for zone in ["UTC", "Etc/UTC", "Z"] {
        assert_eq!(
            partition_ts("2024-01-01 00:00:00", Some(zone)).unwrap(),
            Some(rfc3339("2024-01-01T00:00:00Z")),
            "{zone}"
        );
    }
    // Only primitive targets; a struct of no fields keeps the rows.
    let nested = StructType::try_new([StructField::nullable(
        "s",
        StructType::try_new([StructField::nullable("x", DataType::INTEGER)]).unwrap(),
    )])
    .unwrap();
    let e = err(to_struct(&b, &nested, MapToStructOptions::default()));
    assert!(
        e.to_string()
            .contains("only supports primitive target types"),
        "{e}"
    );
    let empty = StructType::try_new([]).unwrap();
    let out = to_struct(&b, &empty, MapToStructOptions::default()).unwrap();
    assert_eq!(out.len(), 1);
}

#[test]
fn evaluators_over_maps_and_an_empty_patch() {
    // A map field in the input schema matches a map column.
    let with_map = Arc::new(
        StructType::try_new([
            StructField::nullable("a", DataType::INTEGER),
            StructField::nullable(
                "pv",
                DataType::Map(Box::new(delta_kernel::schema::MapType::new(
                    DataType::STRING,
                    DataType::STRING,
                    true,
                ))),
            ),
        ])
        .unwrap(),
    );
    let eval = handler()
        .new_predicate_evaluator(with_map, Arc::new(Pred::is_null(col("pv"))))
        .unwrap();
    let out = eval.evaluate(&KrishivEngineData::new(batch())).unwrap();
    let out = KrishivEngineData::batch_of(out.as_ref()).unwrap();
    assert_eq!(
        show(out.column(0)),
        ["false", "false", "true", "false", "false"]
    );
    // An empty patch over an input path only renames: the struct's columns as the output.
    let empty = ExpressionStructPatch {
        input_path: Some(ColumnName::new(["st"])),
        field_patches: HashMap::new(),
        prepended_fields: vec![],
        appended_fields: vec![],
    };
    let renamed = struct_of(&[("xx", DataType::INTEGER), ("yy", DataType::STRING)]);
    let eval = handler()
        .new_expression_evaluator(input_schema(), Arc::new(Expr::StructPatch(empty)), renamed)
        .unwrap();
    let out = eval.evaluate(&KrishivEngineData::new(batch())).unwrap();
    let out = KrishivEngineData::batch_of(out.as_ref()).unwrap();
    assert_eq!(out.schema().field(0).name(), "xx");
    assert_eq!(show(out.column(1)), ["p", "q", "r", "s", "t"]);
}

#[test]
fn create_many_builds_map_columns() {
    use delta_kernel::expressions::MapData;
    use delta_kernel::schema::MapType;
    let map_type = MapType::new(DataType::STRING, DataType::LONG, true);
    let schema = one_field("m", DataType::Map(Box::new(map_type.clone())));
    let value = MapData::try_new(
        map_type.clone(),
        [
            (Scalar::from("a"), Scalar::Long(1)),
            (Scalar::from("b"), Scalar::Null(DataType::LONG)),
        ],
    )
    .unwrap();
    let data = handler()
        .create_many(
            schema,
            vec![
                vec![Scalar::Map(value)],
                vec![Scalar::Null(DataType::Map(Box::new(map_type)))],
            ],
        )
        .unwrap();
    let b = KrishivEngineData::batch_of(data.as_ref()).unwrap();
    assert_eq!(show(b.column(0)), ["{a: 1, b: NULL}", "NULL"]);
}
