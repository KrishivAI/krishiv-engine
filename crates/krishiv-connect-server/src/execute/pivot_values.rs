// SPDX-License-Identifier: Apache-2.0
//! The values of a pivot the client gave none for (Spark's `RelationalGroupedDataset.collectPivotValues`,
//! which the Connect planner calls while planning): `select(col).distinct().limit(max + 1).sort(col)`
//! over the pivot's input, collected. More than `spark.sql.pivotMaxValues` (10000) values is
//! `_LEGACY_ERROR_TEMP_1324`. The query runs through the whole pre-pass, as the client's own plan
//! does, because the pivot's input may hold local relations, catalog calls or Python UDFs.

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature, LegacyErrorTemp1324};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::expression::{self, ExprType};
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect::translate::UdfHost as TranslateHost;
use krishiv_plan::Literal;
use krishiv_session::QueryOptions;
use krishiv_types::DecimalType;

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

/// The pivot column as Spark's `Column.toString` prints it, for the too-many-values error (an
/// attribute's name; other expressions are not rendered).
fn column_text(column: &pb::Expression) -> String {
    match &column.expr_type {
        Some(ExprType::UnresolvedAttribute(a)) => a.unparsed_identifier.clone(),
        _ => "<expression>".to_owned(),
    }
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
    let max = entry
        .session()
        .conf()
        .get("spark.sql.pivotMaxValues")
        .and_then(|v| v.trim().parse::<i32>().ok())
        .unwrap_or(DEFAULT_MAX_VALUES);
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
            pivot_column: column_text(column),
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

/// The first column of `batch` as literals, appended to `out`.
fn literals(batch: &RecordBatch, out: &mut Vec<Literal>) -> Result<(), KrishivError> {
    let array = batch.column(0);
    let data_type = array.data_type();
    for row in 0..array.len() {
        if array.is_null(row) {
            out.push(Literal::Null(None));
            continue;
        }
        out.push(match data_type {
            DataType::Boolean => Literal::Boolean(array.as_boolean().value(row)),
            DataType::Int8 => Literal::Byte(array.as_primitive::<Int8Type>().value(row)),
            DataType::Int16 => Literal::Short(array.as_primitive::<Int16Type>().value(row)),
            DataType::Int32 => Literal::Integer(array.as_primitive::<Int32Type>().value(row)),
            DataType::Int64 => Literal::Long(array.as_primitive::<Int64Type>().value(row)),
            DataType::Float32 => Literal::Float(array.as_primitive::<Float32Type>().value(row)),
            DataType::Float64 => Literal::Double(array.as_primitive::<Float64Type>().value(row)),
            DataType::Utf8 => Literal::String(array.as_string::<i32>().value(row).to_owned()),
            DataType::LargeUtf8 => Literal::String(array.as_string::<i64>().value(row).to_owned()),
            DataType::Utf8View => Literal::String(array.as_string_view().value(row).to_owned()),
            DataType::Binary => Literal::Binary(array.as_binary::<i32>().value(row).to_vec()),
            DataType::Date32 => Literal::Date(array.as_primitive::<Date32Type>().value(row)),
            DataType::Timestamp(TimeUnit::Microsecond, tz) => {
                let v = array.as_primitive::<TimestampMicrosecondType>().value(row);
                if tz.is_some() {
                    Literal::Timestamp(v)
                } else {
                    Literal::TimestampNtz(v)
                }
            }
            DataType::Decimal128(p, s) => Literal::Decimal {
                unscaled: array.as_primitive::<Decimal128Type>().value(row),
                data_type: DecimalType::try_new(u32::from(*p), u32::try_from(*s).unwrap_or(0))?,
            },
            other => return Err(unsupported(other)),
        });
    }
    Ok(())
}
