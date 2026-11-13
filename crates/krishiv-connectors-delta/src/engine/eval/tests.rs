// SPDX-License-Identifier: Apache-2.0
//! `delta_eval_parity`: the kernel's evaluator tests (`delta_kernel` 0.30.1
//! `engine/arrow_expression/evaluate_expression.rs` and `tests.rs`) as a table, one row per
//! case with the kernel's expected value, plus the NULL and cast rows of the design note.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{Int32Builder, ListBuilder, MapBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, RecordBatch, StringArray, StructArray,
    TimestampMicrosecondArray,
};
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use delta_kernel::expressions::{
    BinaryExpressionOp, ColumnName, Expression as Expr, ExpressionFieldPatch,
    ExpressionStructPatch, MapToStructOptions, Predicate as Pred, Scalar, UnaryExpressionOp,
};
use delta_kernel::schema::{ArrayType, DataType, StructField, StructType};
use delta_kernel::{EvaluationHandler, KernelError};

use super::expr::evaluate_expression;
use super::predicate::evaluate_predicate;
use super::*;
use crate::engine::data::KrishivEngineData;
use crate::engine::relabel::{ARROW_UTC, relabel, with_zone};

fn ctx() -> Ctx {
    Ctx {
        limits: DeltaLimits::default(),
    }
}

fn col(name: &str) -> Expr {
    Expr::column([name])
}

fn lit(v: impl Into<Scalar>) -> Expr {
    Expr::literal(v)
}

fn null_int() -> Expr {
    Expr::literal(Scalar::Null(DataType::INTEGER))
}

fn ints(v: &[Option<i32>]) -> ArrayRef {
    Arc::new(Int32Array::from(v.to_vec()))
}

fn bools(v: &[Option<bool>]) -> ArrayRef {
    Arc::new(BooleanArray::from(v.to_vec()))
}

/// The five-row batch every case runs on.
fn batch() -> RecordBatch {
    let mut list = ListBuilder::new(Int32Builder::new());
    for row in [
        &[Some(1), Some(2)][..],
        &[Some(1), Some(2)],
        &[Some(1), None],
        &[None],
        &[Some(1), None],
    ] {
        for v in row {
            list.values().append_option(*v);
        }
        list.append(true);
    }
    let list = list.finish();
    let mut map = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    type Entries<'a> = Option<&'a [(&'a str, Option<&'a str>)]>;
    let maps: [Entries<'_>; 5] = [
        Some(&[
            ("p", Some("a")),
            ("d", Some("2024-01-02")),
            ("n", Some("7")),
            ("ts", Some("2024-01-01 00:00:00")),
            ("other", Some("x")),
        ]),
        Some(&[("p", Some("")), ("d", Some("")), ("n", Some(""))]),
        None,
        Some(&[("p", None), ("ts", Some("2024-01-01T01:00:00+01:00"))]),
        Some(&[("p", Some("first")), ("p", Some("last")), ("n", Some("-3"))]),
    ];
    for row in maps {
        match row {
            Some(entries) => {
                for (k, v) in entries {
                    map.keys().append_value(k);
                    map.values().append_option(*v);
                }
                map.append(true).unwrap();
            }
            None => map.append(false).unwrap(),
        }
    }
    let map = map.finish();
    let st = StructArray::try_new(
        vec![
            Field::new("x", ArrowType::Int32, true),
            Field::new("y", ArrowType::Utf8, true),
        ]
        .into(),
        vec![
            ints(&[Some(1), Some(2), Some(3), Some(4), Some(5)]),
            Arc::new(StringArray::from(vec!["p", "q", "r", "s", "t"])),
        ],
        Some(vec![true, true, false, true, true].into()),
    )
    .unwrap();
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("a", ints(&[Some(1), Some(2), Some(3), Some(4), Some(5)])),
        ("b", ints(&[None, Some(10), None, Some(40), None])),
        ("l", ints(&[Some(1), Some(1), None, None, Some(1)])),
        ("r", ints(&[Some(1), Some(2), None, Some(1), None])),
        ("n", ints(&[Some(-7), Some(7), Some(-1), Some(1), Some(0)])),
        ("zero", ints(&[Some(0); 5])),
        (
            "big",
            ints(&[Some(1000), Some(1), Some(-129), Some(127), None]),
        ),
        (
            "f",
            Arc::new(Float64Array::from(vec![1.0, -1.0, 0.0, 2.0, 7.0])),
        ),
        ("fz", Arc::new(Float64Array::from(vec![0.0; 5]))),
        (
            "bl",
            bools(&[Some(false), None, Some(true), None, Some(false)]),
        ),
        ("br", bools(&[None, Some(false), None, None, None])),
        (
            "bt",
            bools(&[Some(true), Some(true), Some(false), None, Some(false)]),
        ),
        ("bn", bools(&[None, None, None, Some(true), Some(true)])),
        (
            "s",
            Arc::new(StringArray::from(vec![
                Some("x"),
                None,
                Some("z"),
                Some("12"),
                Some("y"),
            ])),
        ),
        (
            "j",
            Arc::new(StringArray::from(vec![
                Some(r#"{"numRecords":3,"minValues":{"t":"2024-01-01T00:00:00.000Z","d":"1.50"}}"#),
                Some(
                    r#"{"numRecords":1,"minValues":{"t":"+275760-09-13T00:00:01Z","d":"123.45"}}"#,
                ),
                None,
                Some(r#"{"numRecords":2}"#),
                Some(r#"{"numRecords":"5","minValues":{"t":"not a time","d":"x"}}"#),
            ])),
        ),
        ("list", Arc::new(list)),
        ("pv", Arc::new(map)),
        ("st", Arc::new(st)),
        (
            "ts",
            Arc::new(
                TimestampMicrosecondArray::from(vec![
                    1_783_007_755_298_677,
                    1_783_007_755_999_900,
                    -1_500,
                    1_783_007_755_000_000,
                    0,
                ])
                .with_timezone("UTC"),
            ),
        ),
        (
            "ntz",
            Arc::new(TimestampMicrosecondArray::from(vec![
                1_783_007_755_298_677,
                1_783_007_755_999_900,
                -1_500,
                1_783_007_755_000_000,
                0,
            ])),
        ),
    ];
    let tj = StructArray::try_new(
        vec![
            Field::new("b", ArrowType::Int32, true),
            Field::new("ts", columns[18].1.data_type().clone(), true),
            Field::new("ntz", columns[19].1.data_type().clone(), true),
        ]
        .into(),
        vec![
            Arc::clone(&columns[1].1),
            Arc::clone(&columns[18].1),
            Arc::clone(&columns[19].1),
        ],
        Some(vec![true, true, false, true, true].into()),
    )
    .unwrap();
    let mut columns = columns;
    columns.push(("tj", Arc::new(tj)));
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

/// Each value of `a` as text (`NULL` for null); zoned timestamps shown at `+00:00`.
fn show(a: &ArrayRef) -> Vec<String> {
    let a = relabel(a, &with_zone(a.data_type(), ARROW_UTC)).unwrap();
    let options = FormatOptions::default().with_null("NULL");
    let f = ArrayFormatter::try_new(a.as_ref(), &options).unwrap();
    (0..a.len()).map(|i| f.value(i).to_string()).collect()
}

enum What {
    E(Expr, Option<DataType>),
    P(Pred),
}

/// The expected outcome: the values, or an error containing the text.
enum Want {
    V(&'static [&'static str]),
    Err(&'static str),
}

fn stats_schema() -> Arc<StructType> {
    Arc::new(
        StructType::try_new([
            StructField::nullable("numRecords", DataType::LONG),
            StructField::nullable(
                "minValues",
                StructType::try_new([
                    StructField::nullable("t", DataType::TIMESTAMP),
                    StructField::nullable("d", DataType::decimal(4, 2).unwrap()),
                ])
                .unwrap(),
            ),
        ])
        .unwrap(),
    )
}

fn partition_schema() -> StructType {
    StructType::try_new([
        StructField::nullable("p", DataType::STRING),
        StructField::nullable("d", DataType::DATE),
        StructField::nullable("n", DataType::INTEGER),
        StructField::nullable("ts", DataType::TIMESTAMP),
    ])
    .unwrap()
}

fn patch() -> ExpressionStructPatch {
    ExpressionStructPatch {
        input_path: Some(ColumnName::new(["st"])),
        field_patches: HashMap::from([(
            "y".to_owned(),
            ExpressionFieldPatch {
                keep_input: false,
                insertions: vec![Arc::new(lit(7))],
                optional: false,
            },
        )]),
        prepended_fields: vec![],
        appended_fields: vec![],
    }
}

fn patched_schema() -> StructType {
    StructType::try_new([
        StructField::nullable("x", DataType::INTEGER),
        StructField::nullable("k", DataType::INTEGER),
    ])
    .unwrap()
}

#[expect(clippy::too_many_lines, reason = "the parity table")]
fn cases() -> Vec<(&'static str, What, Want)> {
    use Want::{Err, V};
    use What::{E, P};
    let int = Some(DataType::INTEGER);
    let div = |l, r| Expr::binary(BinaryExpressionOp::Divide, l, r);
    vec![
        // Literals and columns.
        ("literal", E(lit(7), int.clone()), V(&["7"; 5])),
        (
            "column",
            E(col("b"), int.clone()),
            V(&["NULL", "10", "NULL", "40", "NULL"]),
        ),
        (
            "column_type_validation",
            E(col("s"), int.clone()),
            Err("Incorrect datatype"),
        ),
        (
            "literal_type_validation",
            E(lit("x"), int.clone()),
            Err("Incorrect datatype"),
        ),
        // Arithmetic: integer division truncates, by zero is an error; floats give inf/NaN.
        (
            "plus",
            E(
                Expr::binary(BinaryExpressionOp::Plus, col("a"), lit(1)),
                int.clone(),
            ),
            V(&["2", "3", "4", "5", "6"]),
        ),
        (
            "divide_truncates",
            E(div(col("n"), lit(2)), int.clone()),
            V(&["-3", "3", "0", "0", "0"]),
        ),
        (
            "divide_by_zero",
            E(div(col("a"), col("zero")), int.clone()),
            Err("Divide by zero"),
        ),
        (
            "float_divide_by_zero",
            E(div(col("f"), col("fz")), Some(DataType::DOUBLE)),
            V(&["inf", "-inf", "NaN", "inf", "inf"]),
        ),
        (
            "binary_type_validation",
            E(
                Expr::binary(BinaryExpressionOp::Plus, col("a"), lit(1)),
                Some(DataType::STRING),
            ),
            Err("Incorrect datatype"),
        ),
        // Casts: an unrepresentable value or an uncastable pair is NULL.
        (
            "cast_int_to_string",
            E(Expr::cast(col("a"), DataType::STRING), None),
            V(&["1", "2", "3", "4", "5"]),
        ),
        (
            "cast_bad_string_is_null",
            E(Expr::cast(col("s"), DataType::INTEGER), None),
            V(&["NULL", "NULL", "NULL", "12", "NULL"]),
        ),
        (
            "cast_overflow_is_null",
            E(Expr::cast(col("big"), DataType::BYTE), None),
            V(&["NULL", "1", "NULL", "127", "NULL"]),
        ),
        (
            "cast_unsupported_pair_is_all_null",
            E(Expr::cast(col("list"), DataType::INTEGER), None),
            V(&["NULL"; 5]),
        ),
        // Coalesce.
        (
            "coalesce",
            E(Expr::coalesce([col("b"), col("a")]), int.clone()),
            V(&["1", "10", "3", "40", "5"]),
        ),
        (
            "coalesce_type_mismatch",
            E(Expr::coalesce([col("b"), col("s")]), None),
            Err("has type"),
        ),
        (
            "coalesce_short_circuits",
            E(Expr::coalesce([col("a"), col("s")]), int.clone()),
            V(&["1", "2", "3", "4", "5"]),
        ),
        // Comparisons: a NULL side is NULL; NOT is pushed down.
        (
            "lt_null",
            P(Pred::lt(col("b"), lit(20))),
            V(&["NULL", "true", "NULL", "false", "NULL"]),
        ),
        (
            "not_lt_is_ge",
            P(Pred::not(Pred::lt(col("b"), lit(20)))),
            V(&["NULL", "false", "NULL", "true", "NULL"]),
        ),
        (
            "eq",
            P(Pred::eq(col("l"), col("r"))),
            V(&["true", "false", "NULL", "NULL", "NULL"]),
        ),
        // DISTINCT is null-safe.
        (
            "distinct_is_null_safe",
            P(Pred::distinct(col("l"), col("r"))),
            V(&["false", "true", "false", "true", "true"]),
        ),
        (
            "not_distinct",
            P(Pred::not(Pred::distinct(col("l"), col("r")))),
            V(&["true", "false", "true", "false", "false"]),
        ),
        (
            "is_null",
            P(Pred::is_null(col("b"))),
            V(&["true", "false", "true", "false", "true"]),
        ),
        (
            "is_not_null",
            P(Pred::is_not_null(col("b"))),
            V(&["false", "true", "false", "true", "false"]),
        ),
        // Kleene AND/OR (the kernel's eight cases across the rows).
        (
            "and_kleene",
            P(Pred::and(
                Pred::from_expr(col("bl")),
                Pred::from_expr(col("br")),
            )),
            V(&["false", "false", "NULL", "NULL", "false"]),
        ),
        (
            "or_kleene",
            P(Pred::or(
                Pred::from_expr(col("bt")),
                Pred::from_expr(col("bn")),
            )),
            V(&["true", "true", "NULL", "true", "true"]),
        ),
        (
            "not_and_is_or_of_nots",
            P(Pred::not(Pred::and(
                Pred::from_expr(col("bt")),
                Pred::from_expr(col("bn")),
            ))),
            V(&["NULL", "NULL", "true", "NULL", "true"]),
        ),
        ("empty_and_is_true", P(Pred::and_from([])), V(&["true"; 5])),
        ("empty_or_is_false", P(Pred::or_from([])), V(&["false"; 5])),
        // IN never gives NULL.
        (
            "in_list_column",
            P(Pred::binary(
                delta_kernel::expressions::BinaryPredicateOp::In,
                lit(1),
                col("list"),
            )),
            V(&["true", "true", "true", "false", "true"]),
        ),
        (
            "in_list_column_absent",
            P(Pred::binary(
                delta_kernel::expressions::BinaryPredicateOp::In,
                lit(9),
                col("list"),
            )),
            V(&["false"; 5]),
        ),
        (
            "in_null_needle",
            P(Pred::binary(
                delta_kernel::expressions::BinaryPredicateOp::In,
                null_int(),
                col("list"),
            )),
            V(&["false"; 5]),
        ),
        (
            "not_in",
            P(Pred::not(Pred::binary(
                delta_kernel::expressions::BinaryPredicateOp::In,
                lit(2),
                col("list"),
            ))),
            V(&["false", "false", "true", "true", "true"]),
        ),
        (
            "in_literal_array",
            P(Pred::binary(
                delta_kernel::expressions::BinaryPredicateOp::In,
                lit(2),
                lit(Scalar::Array(
                    delta_kernel::expressions::ArrayData::try_new(
                        ArrayType::new(DataType::INTEGER, true),
                        [Scalar::Integer(1), Scalar::Integer(2)],
                    )
                    .unwrap(),
                )),
            )),
            V(&["true"; 5]),
        ),
        (
            "in_literal_array_null_never_matches",
            P(Pred::binary(
                delta_kernel::expressions::BinaryPredicateOp::In,
                null_int(),
                lit(Scalar::Array(
                    delta_kernel::expressions::ArrayData::try_new(
                        ArrayType::new(DataType::INTEGER, true),
                        [Scalar::Integer(1), Scalar::Null(DataType::INTEGER)],
                    )
                    .unwrap(),
                )),
            )),
            V(&["false"; 5]),
        ),
        (
            "in_rejects_non_array",
            P(Pred::binary(
                delta_kernel::expressions::BinaryPredicateOp::In,
                lit(1),
                lit(1),
            )),
            Err("Invalid right value"),
        ),
        // Unknown is never evaluated (never read as NULL in a filter).
        (
            "unknown_predicate",
            P(Pred::unknown("udf")),
            Err("Unknown predicate"),
        ),
        (
            "unknown_expression",
            E(Expr::unknown("udf"), int),
            Err("Unknown expression"),
        ),
        // Struct, with the nullability predicate keeping only true rows.
        (
            "struct_nullability",
            E(
                Expr::struct_with_nullability_from(
                    [col("a")],
                    Expr::from_pred(Pred::from_expr(col("bl"))),
                ),
                Some(DataType::Struct(Box::new(
                    StructType::try_new([StructField::nullable("a", DataType::INTEGER)]).unwrap(),
                ))),
            ),
            V(&["NULL", "NULL", "{a: 3}", "NULL", "NULL"]),
        ),
        (
            "struct_needs_struct_type",
            E(Expr::struct_from([col("a")]), None),
            Err("expects a DataType::Struct"),
        ),
        // StructPatch: drop y, insert k after it, keep the source's null rows.
        (
            "struct_patch",
            E(
                Expr::StructPatch(patch()),
                Some(DataType::Struct(Box::new(patched_schema()))),
            ),
            V(&[
                "{x: 1, k: 7}",
                "{x: 2, k: 7}",
                "NULL",
                "{x: 4, k: 7}",
                "{x: 5, k: 7}",
            ]),
        ),
        // Array.
        (
            "array",
            E(Expr::array([col("a"), lit(0)]), None),
            V(&["[1, 0]", "[2, 0]", "[3, 0]", "[4, 0]", "[5, 0]"]),
        ),
        (
            "array_null_in_non_nullable",
            E(
                Expr::array([col("b")]),
                Some(DataType::Array(Box::new(ArrayType::new(
                    DataType::INTEGER,
                    false,
                )))),
            ),
            Err("declares non-nullable elements"),
        ),
        // ToJson: nulls omitted, a NULL row NULL, timestamps truncated to ms, `Z` on timestamp.
        (
            "to_json",
            E(Expr::unary(UnaryExpressionOp::ToJson, col("tj")), None),
            V(&[
                r#"{"ts":"2026-07-02T15:55:55.298Z","ntz":"2026-07-02T15:55:55.298"}"#,
                r#"{"b":10,"ts":"2026-07-02T15:55:55.999Z","ntz":"2026-07-02T15:55:55.999"}"#,
                "NULL",
                r#"{"b":40,"ts":"2026-07-02T15:55:55.000Z","ntz":"2026-07-02T15:55:55.000"}"#,
                r#"{"ts":"1970-01-01T00:00:00.000Z","ntz":"1970-01-01T00:00:00.000"}"#,
            ]),
        ),
        (
            "to_json_non_struct",
            E(Expr::unary(UnaryExpressionOp::ToJson, col("a")), None),
            Err("can only be applied to struct"),
        ),
        // ParseJson: bad timestamp/decimal cells are NULL; a NULL string a row of NULLs; a
        // number given as a string coerces.
        (
            "parse_json_stats",
            E(Expr::parse_json(col("j"), stats_schema()), None),
            V(&[
                "{numRecords: 3, minValues: {t: 2024-01-01T00:00:00Z, d: 1.50}}",
                "{numRecords: 1, minValues: {t: NULL, d: NULL}}",
                "{numRecords: NULL, minValues: NULL}",
                "{numRecords: 2, minValues: NULL}",
                "{numRecords: 5, minValues: {t: NULL, d: NULL}}",
            ]),
        ),
        (
            "parse_json_unparseable_batch_is_null",
            E(Expr::parse_json(col("s"), stats_schema()), None),
            V(&["NULL"; 5]),
        ),
        // MapToStruct: Delta's partition-value encoding.
        (
            "map_to_struct",
            E(
                Expr::map_to_struct(col("pv"), MapToStructOptions::default()),
                Some(DataType::Struct(Box::new(partition_schema()))),
            ),
            V(&[
                "{p: a, d: 2024-01-02, n: 7, ts: 2024-01-01T00:00:00Z}",
                "{p: , d: NULL, n: NULL, ts: NULL}",
                "NULL",
                "{p: NULL, d: NULL, n: NULL, ts: 2024-01-01T00:00:00Z}",
                "{p: last, d: NULL, n: -3, ts: NULL}",
            ]),
        ),
        (
            "map_to_struct_fixed_reader_zone",
            E(
                Expr::map_to_struct(
                    col("pv"),
                    MapToStructOptions::default().with_timestamp_timezone("+05:30"),
                ),
                Some(DataType::Struct(Box::new(partition_schema()))),
            ),
            V(&[
                "{p: a, d: 2024-01-02, n: 7, ts: 2023-12-31T18:30:00Z}",
                "{p: , d: NULL, n: NULL, ts: NULL}",
                "NULL",
                "{p: NULL, d: NULL, n: NULL, ts: 2024-01-01T00:00:00Z}",
                "{p: last, d: NULL, n: -3, ts: NULL}",
            ]),
        ),
        (
            "map_to_struct_needs_struct_type",
            E(
                Expr::map_to_struct(col("pv"), MapToStructOptions::default()),
                None,
            ),
            Err("requires a DataType::Struct"),
        ),
        (
            "map_to_struct_non_map",
            E(
                Expr::map_to_struct(col("a"), MapToStructOptions::default()),
                Some(DataType::Struct(Box::new(partition_schema()))),
            ),
            Err("requires a MapArray"),
        ),
    ]
}

#[test]
fn delta_eval_parity() {
    let b = batch();
    let mut failures = Vec::new();
    for (name, what, want) in cases() {
        let got = match what {
            What::E(e, t) => evaluate_expression(&ctx(), &e, &b, t.as_ref()),
            What::P(p) => {
                evaluate_predicate(&ctx(), &p, &b, false).map(|a| Arc::new(a) as ArrayRef)
            }
        };
        let ok = match (&got, &want) {
            (Ok(a), Want::V(v)) => show(a) == *v,
            (Err(e), Want::Err(text)) => e.to_string().contains(text),
            _ => false,
        };
        if !ok {
            let got = got.map(|a| show(&a)).map_err(|e| e.to_string());
            failures.push(format!("{name}: got {got:?}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn a_bad_partition_value_is_a_parse_error() {
    let mut map = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    map.keys().append_value("n");
    map.values().append_value("seven");
    map.append(true).unwrap();
    let map: ArrayRef = Arc::new(map.finish());
    let b = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "pv",
            map.data_type().clone(),
            true,
        )])),
        vec![map],
    )
    .unwrap();
    let e = evaluate_expression(
        &ctx(),
        &Expr::map_to_struct(col("pv"), MapToStructOptions::default()),
        &b,
        Some(&DataType::Struct(Box::new(partition_schema()))),
    )
    .unwrap_err();
    assert!(matches!(e, KernelError::ParseError(..)), "{e:?}");
}

#[test]
fn a_named_reader_zone_is_unsupported() {
    let e = evaluate_expression(
        &ctx(),
        &Expr::map_to_struct(
            col("pv"),
            MapToStructOptions::default().with_timestamp_timezone("America/Los_Angeles"),
        ),
        &batch(),
        Some(&DataType::Struct(Box::new(partition_schema()))),
    )
    .unwrap_err();
    assert!(matches!(e, KernelError::Unsupported(_)), "{e:?}");
}

#[test]
fn parse_json_applies_the_parse_limits() {
    let limits = DeltaLimits {
        max_depth: 2,
        ..DeltaLimits::default()
    };
    let deep = Arc::new(StringArray::from(vec![r#"{"a":{"b":{"c":1}}}"#])) as ArrayRef;
    let e = super::json::parse_json(&Ctx { limits }, &deep, &stats_schema()).unwrap_err();
    let class = crate::error::from_kernel(&e, "t").error_class();
    assert_eq!(
        class,
        krishiv_common::error::ErrorClass::KrishivCatalogLimitExceeded
    );
}

fn err<T>(r: KernelResult<T>) -> KernelError {
    match r {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    }
}

fn handler() -> KrishivEvaluationHandler {
    KrishivEvaluationHandler::new(DeltaLimits::default())
}

fn input_schema() -> Arc<StructType> {
    Arc::new(
        StructType::try_new([
            StructField::nullable("a", DataType::INTEGER),
            StructField::nullable("b", DataType::INTEGER),
        ])
        .unwrap(),
    )
}

#[test]
fn an_expression_evaluator_gives_the_declared_output() {
    let b = batch();
    // A non-struct output is one nullable column named `output`; an Int32 widens to `long`.
    let eval = handler()
        .new_expression_evaluator(input_schema(), Arc::new(col("a")), DataType::LONG)
        .unwrap();
    let out = eval.evaluate(&KrishivEngineData::new(b.clone())).unwrap();
    let out = KrishivEngineData::batch_of(out.as_ref()).unwrap();
    assert_eq!(out.schema().field(0).name(), "output");
    assert_eq!(out.column(0).data_type(), &ArrowType::Int64);
    // A struct output becomes the batch's columns, renamed to the declared names.
    let schema = StructType::try_new([
        StructField::not_null("first", DataType::INTEGER),
        StructField::nullable("second", DataType::INTEGER),
    ])
    .unwrap();
    let eval = handler()
        .new_expression_evaluator(
            input_schema(),
            Arc::new(Expr::struct_from([col("a"), col("b")])),
            DataType::Struct(Box::new(schema.clone())),
        )
        .unwrap();
    let out = eval.evaluate(&KrishivEngineData::new(b.clone())).unwrap();
    let out = KrishivEngineData::batch_of(out.as_ref()).unwrap();
    assert_eq!(
        out.schema()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    assert!(!out.schema().field(0).is_nullable());
    // A null in a non-nullable output column is an error, not a panic.
    let eval = handler()
        .new_expression_evaluator(
            input_schema(),
            Arc::new(Expr::struct_from([col("b"), col("a")])),
            DataType::Struct(Box::new(schema)),
        )
        .unwrap();
    assert!(eval.evaluate(&KrishivEngineData::new(b.clone())).is_err());
    // The empty patch only renames (column mapping).
    let eval = handler()
        .new_expression_evaluator(
            input_schema(),
            Arc::new(Expr::StructPatch(ExpressionStructPatch {
                input_path: Some(ColumnName::new(["st"])),
                field_patches: HashMap::new(),
                prepended_fields: vec![],
                appended_fields: vec![],
            })),
            DataType::Struct(Box::new(
                StructType::try_new([
                    StructField::nullable("phys_x", DataType::INTEGER),
                    StructField::nullable("phys_y", DataType::STRING),
                ])
                .unwrap(),
            )),
        )
        .unwrap();
    let out = eval.evaluate(&KrishivEngineData::new(b)).unwrap();
    let out = KrishivEngineData::batch_of(out.as_ref()).unwrap();
    assert_eq!(out.schema().field(1).name(), "phys_y");
    assert_eq!(out.num_rows(), 5);
}

#[test]
fn an_evaluator_checks_its_input_schema() {
    let b = batch();
    let missing =
        Arc::new(StructType::try_new([StructField::nullable("nope", DataType::INTEGER)]).unwrap());
    let eval = handler()
        .new_predicate_evaluator(missing, Arc::new(Pred::is_null(col("a"))))
        .unwrap();
    let e = err(eval.evaluate(&KrishivEngineData::new(b.clone())));
    assert!(e.to_string().contains("missing"), "{e}");
    let reversed = Arc::new(
        StructType::try_new([
            StructField::nullable("b", DataType::INTEGER),
            StructField::nullable("a", DataType::INTEGER),
        ])
        .unwrap(),
    );
    let eval = handler()
        .new_predicate_evaluator(reversed, Arc::new(Pred::is_null(col("a"))))
        .unwrap();
    let e = err(eval.evaluate(&KrishivEngineData::new(b.clone())));
    assert!(e.to_string().contains("out of order"), "{e}");
    let wrong =
        Arc::new(StructType::try_new([StructField::nullable("a", DataType::STRING)]).unwrap());
    let eval = handler()
        .new_predicate_evaluator(wrong, Arc::new(Pred::is_null(col("a"))))
        .unwrap();
    assert!(eval.evaluate(&KrishivEngineData::new(b)).is_err());
    // The data-skipping filter: DISTINCT(output, false) keeps true and NULL rows.
    let out_schema = Arc::new(
        StructType::try_new([StructField::nullable("output", DataType::BOOLEAN)]).unwrap(),
    );
    let filter = handler()
        .new_predicate_evaluator(out_schema, Arc::new(col("output").distinct(lit(false))))
        .unwrap();
    let skipping = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "output",
            ArrowType::Boolean,
            true,
        )])),
        vec![bools(&[Some(true), Some(false), None])],
    )
    .unwrap();
    let out = filter.evaluate(&KrishivEngineData::new(skipping)).unwrap();
    let out = KrishivEngineData::batch_of(out.as_ref()).unwrap();
    assert_eq!(show(out.column(0)), ["true", "false", "true"]);
}

#[test]
fn create_many_builds_rows() {
    let schema = Arc::new(
        StructType::try_new([
            StructField::nullable("i", DataType::INTEGER),
            StructField::nullable(
                "s",
                StructType::try_new([
                    StructField::not_null("x", DataType::STRING),
                    StructField::nullable("y", DataType::LONG),
                ])
                .unwrap(),
            ),
        ])
        .unwrap(),
    );
    let st = |x: &str, y: i64| {
        Scalar::Struct(
            delta_kernel::expressions::StructData::try_new(
                vec![
                    StructField::not_null("x", DataType::STRING),
                    StructField::nullable("y", DataType::LONG),
                ],
                vec![Scalar::String(x.to_owned()), Scalar::Long(y)],
            )
            .unwrap(),
        )
    };
    let data = handler()
        .create_many(
            Arc::clone(&schema),
            vec![
                vec![Scalar::Integer(1), st("a", 2)],
                vec![
                    Scalar::Null(DataType::INTEGER),
                    Scalar::Null(schema.field("s").unwrap().data_type().clone()),
                ],
            ],
        )
        .unwrap();
    let b = KrishivEngineData::batch_of(data.as_ref()).unwrap();
    assert_eq!(show(b.column(0)), ["1", "NULL"]);
    assert_eq!(show(b.column(1)), ["{x: a, y: 2}", "NULL"]);
    // No rows: an empty batch of the schema.
    let empty = handler().create_many(Arc::clone(&schema), vec![]).unwrap();
    assert_eq!(
        KrishivEngineData::batch_of(empty.as_ref())
            .unwrap()
            .num_rows(),
        0
    );
    // A row of the wrong width, or a wrong type, is an error.
    assert!(
        handler()
            .create_many(Arc::clone(&schema), vec![vec![Scalar::Integer(1)]])
            .err()
            .unwrap()
            .to_string()
            .contains("has 1 scalars but schema has 2 fields")
    );
    assert!(
        handler()
            .create_many(
                Arc::clone(&schema),
                vec![vec![
                    Scalar::String("x".into()),
                    Scalar::Null(DataType::LONG)
                ]]
            )
            .is_err()
    );
    // A null in a non-nullable top-level field is an error, not a panic.
    let strict =
        Arc::new(StructType::try_new([StructField::not_null("i", DataType::INTEGER)]).unwrap());
    assert!(
        handler()
            .create_many(strict, vec![vec![Scalar::Null(DataType::INTEGER)]])
            .is_err()
    );
}

#[test]
fn coalesce_picks_runs_from_each_input() {
    let b = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("x", ArrowType::Int32, true),
            Field::new("y", ArrowType::Int32, true),
        ])),
        vec![
            ints(&[Some(1), Some(2), None, None, Some(5), None]),
            ints(&[Some(9), None, Some(3), None, None, Some(6)]),
        ],
    )
    .unwrap();
    let a = evaluate_expression(&ctx(), &Expr::coalesce([col("x"), col("y")]), &b, None).unwrap();
    assert_eq!(
        a.as_primitive::<Int32Type>().iter().collect::<Vec<_>>(),
        [Some(1), Some(2), Some(3), None, Some(5), Some(6)]
    );
}

#[test]
fn relabelling_shares_the_buffers() {
    // A list of structs of `UTC` timestamps, as arrow-json gives them (`+00:00`): relabelled
    // without copying values or offsets (part-1 review nit 8).
    let b = batch();
    let ts = Arc::clone(b.column_by_name("ts").unwrap());
    let zoned = relabel(&ts, &with_zone(ts.data_type(), ARROW_UTC)).unwrap();
    let st = StructArray::try_new(
        vec![Field::new("t", zoned.data_type().clone(), true)].into(),
        vec![Arc::clone(&zoned)],
        None,
    )
    .unwrap();
    let list: ArrayRef = Arc::new(
        arrow_array::ListArray::try_new(
            Arc::new(Field::new("element", st.data_type().clone(), true)),
            arrow_buffer::OffsetBuffer::from_lengths([2, 3]),
            Arc::new(st),
            None,
        )
        .unwrap(),
    );
    let target = with_zone(list.data_type(), "UTC");
    let back = relabel(&list, &target).unwrap();
    assert_eq!(back.data_type(), &target);
    let leaf = back.as_list::<i32>().values().as_struct().column(0);
    assert_eq!(
        leaf.as_primitive::<arrow_array::types::TimestampMicrosecondType>()
            .values()
            .as_ptr(),
        ts.as_primitive::<arrow_array::types::TimestampMicrosecondType>()
            .values()
            .as_ptr()
    );
    assert_eq!(
        back.as_list::<i32>().offsets().as_ptr(),
        list.as_list::<i32>().offsets().as_ptr()
    );
}

fn one_map(entries: &[(&str, &str)]) -> RecordBatch {
    let mut map = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    for (k, v) in entries {
        map.keys().append_value(k);
        map.values().append_value(v);
    }
    map.append(true).unwrap();
    let map: ArrayRef = Arc::new(map.finish());
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "pv",
            map.data_type().clone(),
            true,
        )])),
        vec![map],
    )
    .unwrap()
}

/// A `timestamp` partition value under `MapToStruct` with reader zone `zone`, as micros.
fn partition_ts(raw: &str, zone: Option<&str>) -> KernelResult<Option<i64>> {
    let options = zone.map_or_else(MapToStructOptions::default, |z| {
        MapToStructOptions::default().with_timestamp_timezone(z)
    });
    let schema = StructType::try_new([StructField::nullable("t", DataType::TIMESTAMP)])?;
    let a = evaluate_expression(
        &ctx(),
        &Expr::map_to_struct(col("pv"), options),
        &one_map(&[("t", raw)]),
        Some(&DataType::Struct(Box::new(schema))),
    )?;
    let t = a.as_struct().column(0);
    Ok(t.is_valid(0).then(|| {
        t.as_primitive::<arrow_array::types::TimestampMicrosecondType>()
            .value(0)
    }))
}

fn rfc3339(s: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(s)
        .unwrap()
        .timestamp_micros()
}

/// The kernel's partition-timestamp grammar (`timestamp_timezone.rs`
/// `parses_compatible_partition_timestamps`, `rejects_unsupported_partition_timestamps`, and
/// `evaluate_expression.rs` `test_map_to_struct_timestamp_timezone`), with its expected values.
#[test]
fn partition_timestamp_grammar() {
    let accepted: &[(&str, Option<&str>, &str)] = &[
        (
            "2024-01-15 12:30:45.123456",
            None,
            "2024-01-15T12:30:45.123456Z",
        ),
        ("2024-01-15T17:30:45+0530", None, "2024-01-15T12:00:45Z"),
        ("2024-01-15T12:30:45+19:00", None, "2024-01-14T17:30:45Z"),
        ("2024-01-15T07:00:45-05:30", None, "2024-01-15T12:30:45Z"),
        ("2024-01-15 17:30:45 +05:30", None, "2024-01-15T12:00:45Z"),
        ("2024-01-15t12:30:45", None, "2024-01-15T12:30:45Z"),
        ("2024-01-15 123045", None, "2024-01-15T12:30:45Z"),
        ("2024-01-15", None, "2024-01-15T00:00:00Z"),
        (
            "2024-01-15 12:30:45.123456789123",
            None,
            "2024-01-15T12:30:45.123456Z",
        ),
        (
            "2024-01-15 12:30:45.123456",
            Some("+05:30"),
            "2024-01-15T07:00:45.123456Z",
        ),
        (
            "2024-01-15 12:30:45.123456",
            Some("+18:00"),
            "2024-01-14T18:30:45.123456Z",
        ),
        (
            "2024-01-15 12:30:45.123456",
            Some("-18:00"),
            "2024-01-16T06:30:45.123456Z",
        ),
        (
            "2024-01-15 12:30:45+02:00",
            Some("+05:30"),
            "2024-01-15T10:30:45Z",
        ),
        (
            "2024-01-15T12:30:45.123456Z",
            Some("+05:30"),
            "2024-01-15T12:30:45.123456Z",
        ),
        // A name of UTC needs no zone database, spaced or glued to the time, also over a
        // fixed reader zone (the value's zone wins), as Arrow's grammar reads it in the kernel.
        ("2024-01-15 12:30:45 UTC", None, "2024-01-15T12:30:45Z"),
        ("2024-01-15 12:30:45UTC", None, "2024-01-15T12:30:45Z"),
        ("2024-01-15T12:30:45.5GMT", None, "2024-01-15T12:30:45.5Z"),
        (
            "2024-01-15 12:30:45 Etc/UTC",
            Some("+05:30"),
            "2024-01-15T12:30:45Z",
        ),
        ("2024-01-15 12:30:45 Zulu", None, "2024-01-15T12:30:45Z"),
    ];
    for (raw, zone, want) in accepted {
        assert_eq!(
            partition_ts(raw, *zone).unwrap(),
            Some(rfc3339(want)),
            "{raw} in {zone:?}"
        );
    }
    let rejected = [
        // An offset and a name: the kernel's Arrow grammar parses `+02:00 UTC` as one zone.
        "2024-01-15 12:30:45+02:00 UTC",
        "2024-01-15 12:30:45Z UTC",
        "2024-01-15 12:30:45+02:00 America/New_York",
        "日時 America/New_York",
        "2024-01-15T12:30:45+05:30:15:00",
        "2024-01-15T12:30:45+053",
        "2024-01-15 01:30:45 PM",
        "15/01/2024 12:30:45",
        "12:30:45",
    ];
    for raw in rejected {
        let e = partition_ts(raw, None).unwrap_err();
        assert!(matches!(e, KernelError::ParseError(..)), "{raw}: {e:?}");
    }
    // The kernel resolves an embedded zone name with chrono-tz
    // (`embedded_iana_timezone_with_default_options`: 2024-01-15T17:30:45Z); Krishiv refuses it
    // by name, never with a wrong value. So is an unknown name the kernel rejects.
    for raw in [
        "2024-01-15 12:30:45 America/New_York",
        "2024-01-15 12:30:45 Foo/Bar",
        // A name without `/`, and one glued to the time (08a part 2 re-review: these were
        // INTERNAL_ERROR).
        "2024-01-15 12:30:45 CET",
        "2024-01-15 12:30:45CET",
    ] {
        let e = partition_ts(raw, Some("+05:30")).unwrap_err();
        assert!(
            matches!(&e, KernelError::Unsupported(m) if m.contains(raw.rsplit(' ').next().unwrap())),
            "{raw}: {e:?}"
        );
    }
}

/// The kernel's `rejects_noncanonical_reader_timezones`.
#[test]
fn invalid_reader_zones_are_refused() {
    for zone in [
        "",
        "+",
        "+05",
        "+0530",
        "+05:60",
        "+05:00:01",
        "+18:00:01",
        "+18:30",
        "+19:00",
        "+05:00:00:00",
        "Not/AZone",
    ] {
        assert!(
            partition_ts("2024-01-15 12:30:45", Some(zone)).is_err(),
            "{zone:?}"
        );
    }
}

/// The kernel's `test_map_to_struct_null_propagation_with_non_nullable_fields` and
/// `test_coalesce_map_to_struct_with_null_map_non_nullable_fields`: a NULL map is a NULL
/// struct, so non-nullable partition columns of non-add rows do not fail (checkpoint writes).
#[test]
fn a_null_map_is_a_null_struct_of_non_nullable_fields() {
    let mut map = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    for row in [
        Some([("region", "us"), ("id", "42")]),
        None,
        Some([("region", "eu"), ("id", "7")]),
    ] {
        match row {
            Some(entries) => {
                for (k, v) in entries {
                    map.keys().append_value(k);
                    map.values().append_value(v);
                }
                map.append(true).unwrap();
            }
            None => map.append(false).unwrap(),
        }
    }
    let map: ArrayRef = Arc::new(map.finish());
    let b = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "pv",
            map.data_type().clone(),
            true,
        )])),
        vec![Arc::clone(&map)],
    )
    .unwrap();
    let strict = StructType::try_new([
        StructField::not_null("region", DataType::STRING),
        StructField::not_null("id", DataType::INTEGER),
    ])
    .unwrap();
    let a = evaluate_expression(
        &ctx(),
        &Expr::map_to_struct(col("pv"), MapToStructOptions::default()),
        &b,
        Some(&DataType::Struct(Box::new(strict))),
    )
    .unwrap();
    assert_eq!(
        show(&a),
        ["{region: us, id: 42}", "NULL", "{region: eu, id: 7}"]
    );
    // COALESCE(pv_parsed, MAP_TO_STRUCT(pv)) with an all-NULL pv_parsed.
    let date = StructType::try_new([StructField::not_null("date", DataType::DATE)]).unwrap();
    let date_type = DataType::Struct(Box::new(date));
    let parsed = arrow_array::new_null_array(
        &crate::engine::schema::to_arrow_type(&date_type).unwrap(),
        2,
    );
    let mut map = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    map.keys().append_value("date");
    map.values().append_value("2024-01-15");
    map.append(true).unwrap();
    map.append(false).unwrap();
    let map: ArrayRef = Arc::new(map.finish());
    let b = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("pv_parsed", parsed.data_type().clone(), true),
            Field::new("pv", map.data_type().clone(), true),
        ])),
        vec![parsed, map],
    )
    .unwrap();
    let a = evaluate_expression(
        &ctx(),
        &Expr::coalesce([
            col("pv_parsed"),
            Expr::map_to_struct(col("pv"), MapToStructOptions::default()),
        ]),
        &b,
        Some(&date_type),
    )
    .unwrap();
    assert_eq!(show(&a), ["{date: 2024-01-15}", "NULL"]);
}

/// The kernel's `test_map_to_struct_duplicate_keys` (the rightmost entry wins) and
/// `test_parse_json_strict_leaf_errors_fall_back_to_all_null_struct`.
#[test]
fn duplicate_keys_and_strict_leaf_fallback() {
    let schema = StructType::try_new([StructField::nullable("x", DataType::STRING)]).unwrap();
    let a = evaluate_expression(
        &ctx(),
        &Expr::map_to_struct(col("pv"), MapToStructOptions::default()),
        &one_map(&[("x", "first"), ("x", "last")]),
        Some(&DataType::Struct(Box::new(schema))),
    )
    .unwrap();
    assert_eq!(show(&a), ["{x: last}"]);
    let strings = Arc::new(StringArray::from(vec![r#"{"a": "not_a_number"}"#])) as ArrayRef;
    let target =
        Arc::new(StructType::try_new([StructField::nullable("a", DataType::LONG)]).unwrap());
    let a = super::json::parse_json(&ctx(), &strings, &target).unwrap();
    assert_eq!((a.len(), a.null_count()), (1, 1));
}
