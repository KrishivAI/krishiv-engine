// SPDX-License-Identifier: Apache-2.0
//! The values of a pivot the client gave none for (Spark's `RelationalGroupedDataset.collectPivotValues`,
//! which the Connect planner calls while planning): `select(col).distinct().limit(max + 1).sort(col)`
//! over the pivot's input, collected. More than `spark.sql.pivotMaxValues` (10000) values is
//! `_LEGACY_ERROR_TEMP_1324`. The query runs through the whole pre-pass, as the client's own plan
//! does, because the pivot's input may hold local relations, catalog calls or Python UDFs.

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type,
    Int16Type, Int32Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType;
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature, LegacyErrorTemp1324};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::expression::{self, ExprType};
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect::translate::{UdfHost as TranslateHost, translate_expression};
use krishiv_plan::Literal;
use krishiv_session::QueryOptions;
use krishiv_types::DataType as SparkType;

use super::{Input, request_translated};
use crate::manager::SessionEntry;

/// Spark's default `spark.sql.pivotMaxValues`.
const DEFAULT_MAX_VALUES: i32 = 10_000;
/// The name the pivot column takes in the values query.
const VALUE: &str = "__krishiv_pivot_value";

fn relation(rel_type: RelType) -> pb::Relation {
    pb::Relation {
        common: None,
        rel_type: Some(rel_type),
    }
}

fn attribute(name: &str) -> pb::Expression {
    pb::Expression {
        common: None,
        expr_type: Some(ExprType::UnresolvedAttribute(
            expression::UnresolvedAttribute {
                unparsed_identifier: name.to_owned(),
                plan_id: None,
                is_metadata_column: None,
            },
        )),
    }
}

/// `input.select(column).distinct().limit(max + 1).sort(column)`, as Spark builds it.
fn values_query(input: &pb::Relation, column: &pb::Expression, max: i32) -> pb::Relation {
    let alias = pb::Expression {
        common: None,
        expr_type: Some(ExprType::Alias(Box::new(expression::Alias {
            expr: Some(Box::new(column.clone())),
            name: vec![VALUE.to_owned()],
            metadata: None,
        }))),
    };
    let projected = relation(RelType::Project(Box::new(pb::Project {
        input: Some(Box::new(input.clone())),
        expressions: vec![alias],
    })));
    let distinct = relation(RelType::Deduplicate(Box::new(pb::Deduplicate {
        input: Some(Box::new(projected)),
        column_names: Vec::new(),
        all_columns_as_keys: Some(true),
        within_watermark: None,
    })));
    let limited = relation(RelType::Limit(Box::new(pb::Limit {
        input: Some(Box::new(distinct)),
        limit: max.saturating_add(1),
    })));
    relation(RelType::Sort(Box::new(pb::Sort {
        input: Some(Box::new(limited)),
        order: vec![expression::SortOrder {
            child: Some(Box::new(attribute(VALUE))),
            direction: expression::sort_order::SortDirection::Ascending as i32,
            null_ordering: expression::sort_order::NullOrdering::SortNullsFirst as i32,
        }],
        is_global: Some(true),
    })))
}

/// The pivot column as Spark's `Column.toString` prints it (`expression.sql`), for the
/// too-many-values error: the translated expression's Spark SQL text.
fn column_text(entry: &SessionEntry, column: &pb::Expression) -> String {
    translate_expression(column, &super::translate_conf(entry.session()))
        .map_or_else(|_| "<expression>".to_owned(), |e| e.to_string())
}

/// The sorted distinct values of `column` over `input`.
///
/// # Errors
/// The errors of translating and running the query; `_LEGACY_ERROR_TEMP_1324` past the limit;
/// `KRISHIV_UNSUPPORTED_FEATURE` for a value type with no literal.
pub(super) async fn collect(
    entry: &SessionEntry,
    input: &pb::Relation,
    column: &pb::Expression,
    udfs: &mut dyn TranslateHost,
) -> Result<Vec<Literal>, KrishivError> {
    let max = match entry.session().conf().get("spark.sql.pivotMaxValues") {
        None => DEFAULT_MAX_VALUES,
        Some(v) => v
            .trim()
            .parse::<i32>()
            .map_err(|_| KrishivUnsupportedFeature {
                feature: format!("spark.sql.pivotMaxValues = {v:?}, which is not an integer"),
            })?,
    };
    let query = values_query(input, column, max);
    let requested = request_translated(entry, Input::Relation(&query), udfs).await?;
    let frame = requested
        .frame(entry.session(), QueryOptions::default())
        .await?;
    let batches = frame.collect().await?;
    let mut values = Vec::new();
    for batch in &batches {
        literals(batch, &mut values)?;
    }
    if i64::try_from(values.len()).unwrap_or(i64::MAX) > i64::from(max) {
        return Err(LegacyErrorTemp1324 {
            pivot_column: column_text(entry, column),
            max_values: max.to_string(),
            config: "spark.sql.pivotMaxValues".to_owned(),
        }
        .into());
    }
    Ok(values)
}

fn unsupported(data_type: &DataType) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: format!("a pivot over values of type {data_type} without explicit values"),
    }
    .into()
}

/// The bytes of one row of a binary-carried array (`BinaryView`, `Binary` or `LargeBinary`).
fn bytes_at(array: &dyn Array, row: usize) -> Option<&[u8]> {
    match array.data_type() {
        DataType::BinaryView => Some(array.as_binary_view().value(row)),
        DataType::Binary => Some(array.as_binary::<i32>().value(row)),
        DataType::LargeBinary => Some(array.as_binary::<i64>().value(row)),
        _ => None,
    }
}

/// The text of one row of a string-carried array (`Utf8View`, `Utf8` or `LargeUtf8`).
fn text_at(array: &dyn Array, row: usize) -> Option<&str> {
    match array.data_type() {
        DataType::Utf8View => Some(array.as_string_view().value(row)),
        DataType::Utf8 => Some(array.as_string::<i32>().value(row)),
        DataType::LargeUtf8 => Some(array.as_string::<i64>().value(row)),
        _ => None,
    }
}

/// One non-null value of a `string` column in any carrier it may arrive in: text, or bytes (a
/// string that is not valid UTF-8 is stored as binary with the type tag).
fn string_at(array: &dyn Array, row: usize) -> Result<Literal, KrishivError> {
    if let Some(t) = text_at(array, row) {
        return Ok(Literal::String(t.to_owned()));
    }
    match bytes_at(array, row).map(std::str::from_utf8) {
        Some(Ok(t)) => Ok(Literal::String(t.to_owned())),
        _ => Err(KrishivUnsupportedFeature {
            feature: "a pivot over string values that are not valid UTF-8 without explicit values"
                .to_owned(),
        }
        .into()),
    }
}

/// The first column of `batch` as literals, appended to `out`. The type is the column's Spark
/// type (the field's tag, else its Arrow type), and every carrier that type may arrive in is read.
fn literals(batch: &RecordBatch, out: &mut Vec<Literal>) -> Result<(), KrishivError> {
    let array = batch.column(0).as_ref();
    let spark = SparkType::from_arrow_field(batch.schema().field(0))?;
    for row in 0..array.len() {
        if array.is_null(row) {
            out.push(Literal::Null(None));
            continue;
        }
        out.push(match (&spark, array.data_type()) {
            (SparkType::Boolean, _) => Literal::Boolean(array.as_boolean().value(row)),
            (SparkType::Byte, _) => Literal::Byte(array.as_primitive::<Int8Type>().value(row)),
            (SparkType::Short, _) => Literal::Short(array.as_primitive::<Int16Type>().value(row)),
            (SparkType::Integer, _) => {
                Literal::Integer(array.as_primitive::<Int32Type>().value(row))
            }
            (SparkType::Long, _) => Literal::Long(array.as_primitive::<Int64Type>().value(row)),
            (SparkType::Float, _) => Literal::Float(array.as_primitive::<Float32Type>().value(row)),
            (SparkType::Double, _) => {
                Literal::Double(array.as_primitive::<Float64Type>().value(row))
            }
            (SparkType::String, _) => string_at(array, row)?,
            (SparkType::Binary, _) => match bytes_at(array, row) {
                Some(b) => Literal::Binary(b.to_vec()),
                None => return Err(unsupported(array.data_type())),
            },
            (SparkType::Date, _) => Literal::Date(array.as_primitive::<Date32Type>().value(row)),
            (SparkType::Timestamp, _) => {
                Literal::Timestamp(array.as_primitive::<TimestampMicrosecondType>().value(row))
            }
            (SparkType::TimestampNtz, _) => {
                Literal::TimestampNtz(array.as_primitive::<TimestampMicrosecondType>().value(row))
            }
            (SparkType::Decimal(d), DataType::Decimal128(..)) => Literal::Decimal {
                unscaled: array.as_primitive::<Decimal128Type>().value(row),
                data_type: *d,
            },
            (SparkType::Decimal(d), DataType::Decimal64(..)) => Literal::Decimal {
                unscaled: i128::from(array.as_primitive::<Decimal64Type>().value(row)),
                data_type: *d,
            },
            (SparkType::Decimal(d), DataType::Decimal32(..)) => Literal::Decimal {
                unscaled: i128::from(array.as_primitive::<Decimal32Type>().value(row)),
                data_type: *d,
            },
            _ => return Err(unsupported(array.data_type())),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
        Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray, StringArray,
        TimestampMicrosecondArray,
    };

    use super::*;

    fn run(array: ArrayRef) -> Result<Vec<Literal>, KrishivError> {
        let Ok(batch) = RecordBatch::try_from_iter([("v", array)]) else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        literals(&batch, &mut out).map(|()| out)
    }

    #[test]
    fn every_supported_value_type_becomes_its_literal() {
        let cases: Vec<(ArrayRef, Literal)> = vec![
            (
                Arc::new(BooleanArray::from(vec![true])),
                Literal::Boolean(true),
            ),
            (Arc::new(Int8Array::from(vec![1])), Literal::Byte(1)),
            (Arc::new(Int16Array::from(vec![2])), Literal::Short(2)),
            (Arc::new(Int32Array::from(vec![3])), Literal::Integer(3)),
            (Arc::new(Int64Array::from(vec![4])), Literal::Long(4)),
            (Arc::new(Float32Array::from(vec![1.5])), Literal::Float(1.5)),
            (
                Arc::new(Float64Array::from(vec![2.5])),
                Literal::Double(2.5),
            ),
            (
                Arc::new(StringArray::from(vec!["a"])),
                Literal::String("a".to_owned()),
            ),
            (
                Arc::new(LargeStringArray::from(vec!["b"])),
                Literal::String("b".to_owned()),
            ),
            (Arc::new(Date32Array::from(vec![7])), Literal::Date(7)),
            (
                Arc::new(TimestampMicrosecondArray::from(vec![9])),
                Literal::TimestampNtz(9),
            ),
            (
                Arc::new(TimestampMicrosecondArray::from(vec![9]).with_timezone("UTC")),
                Literal::Timestamp(9),
            ),
        ];
        for (array, want) in cases {
            assert_eq!(run(array).expect("literal"), vec![want]);
        }
    }

    #[test]
    fn a_decimal_keeps_its_precision_and_scale() {
        let a = Decimal128Array::from(vec![12345])
            .with_precision_and_scale(10, 2)
            .expect("decimal");
        let got = run(Arc::new(a)).expect("literal");
        assert!(matches!(
            got.as_slice(),
            [Literal::Decimal {
                unscaled: 12345,
                ..
            }]
        ));
    }

    #[test]
    fn a_null_value_is_a_null_literal() {
        let got = run(Arc::new(Int32Array::from(vec![None, Some(1)]))).expect("literal");
        assert_eq!(got, vec![Literal::Null(None), Literal::Integer(1)]);
    }

    #[test]
    fn a_binary_column_in_its_canonical_carrier_becomes_binary_literals() {
        let a = arrow_array::BinaryViewArray::from(vec![b"ab".as_slice()]);
        assert_eq!(
            run(Arc::new(a)).expect("literal"),
            vec![Literal::Binary(b"ab".to_vec())]
        );
        let a = arrow_array::LargeBinaryArray::from(vec![b"c".as_slice()]);
        assert_eq!(
            run(Arc::new(a)).expect("literal"),
            vec![Literal::Binary(b"c".to_vec())]
        );
    }

    #[test]
    fn a_string_column_carried_as_bytes_is_a_string_when_the_bytes_are_utf8() {
        let tagged = |a: arrow_array::BinaryViewArray, bad: bool| {
            let field = arrow_schema::Field::new("v", DataType::BinaryView, true).with_metadata(
                std::collections::HashMap::from([(
                    krishiv_types::arrow::SPARK_TYPE_METADATA_KEY.to_owned(),
                    "string".to_owned(),
                )]),
            );
            let batch = RecordBatch::try_new(
                Arc::new(arrow_schema::Schema::new(vec![field])),
                vec![Arc::new(a)],
            )
            .expect("batch");
            let mut out = Vec::new();
            let r = literals(&batch, &mut out).map(|()| out);
            assert_eq!(r.is_err(), bad, "{r:?}");
            r
        };
        let ok = tagged(
            arrow_array::BinaryViewArray::from(vec![b"hi".as_slice()]),
            false,
        );
        assert_eq!(ok.expect("ok"), vec![Literal::String("hi".to_owned())]);
        let _ = tagged(
            arrow_array::BinaryViewArray::from(vec![[0xff_u8, 0xfe].as_slice()]),
            true,
        );
    }

    #[test]
    fn a_decimal64_carrier_keeps_its_value() {
        let a = arrow_array::Decimal64Array::from(vec![123])
            .with_precision_and_scale(10, 2)
            .expect("decimal");
        let got = run(Arc::new(a)).expect("literal");
        assert!(matches!(
            got.as_slice(),
            [Literal::Decimal { unscaled: 123, .. }]
        ));
    }

    #[test]
    fn a_type_with_no_literal_is_refused() {
        let a = arrow_array::Time64NanosecondArray::from(vec![1]);
        let err = run(Arc::new(a)).expect_err("refused").to_string();
        assert!(err.contains("without explicit values"), "{err}");
    }
}
