// SPDX-License-Identifier: Apache-2.0
//! `ExecutePlanResponse.observed_metrics` (P3-T084): the metrics of the
//! `CollectMetrics` nodes of a plan, one `ObservedMetrics` per node name
//! (Spark's `SparkConnectPlanExecution.createObservedMetricsResponse`,
//! v4.3.0): the values as literals, the keys the metric column names, the
//! `plan_id` of the relation that carries the node.

use std::collections::BTreeMap;

use arrow_array::cast::AsArray as _;
use arrow_array::types::{
    Date32Type, DurationMicrosecondType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, IntervalMonthDayNanoType, IntervalYearMonthType, TimestampMicrosecondType,
};
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType as Arrow, Field, IntervalUnit, TimeUnit};
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_connect::proto::v4_3::execute_plan_response::ObservedMetrics;
use krishiv_connect::proto::v4_3::expression::Literal;
use krishiv_connect::proto::v4_3::expression::literal::{
    Array as ArrayLiteral, CalendarInterval, Decimal, LiteralType, Map as MapLiteral,
    Struct as StructLiteral,
};
use krishiv_plan::{LogicalPlan, PlanNode};
use krishiv_types::DataType;

use super::schema::proto_type;

/// The plan id of the relation of each observed name in `plan` (the first
/// carrying one; `0` when the relation has none).
pub(crate) fn plan_ids(plan: &LogicalPlan) -> BTreeMap<String, i64> {
    let mut out = BTreeMap::new();
    let mut stack = vec![plan];
    while let Some(p) = stack.pop() {
        if let PlanNode::CollectMetrics { name, .. } = &p.node {
            out.entry(name.clone())
                .or_insert_with(|| p.origin.plan_id.map_or(0, |id| id.0));
        }
        p.for_each_child(&mut |c| stack.push(c));
    }
    out
}

/// One `ObservedMetrics` for the one-row `row` of the node `name`; a node that
/// did not run (`LIMIT 0` above it) has no columns and so no values.
///
/// # Errors
/// A metric of a type that has no literal yet.
pub(crate) fn observed_metrics(
    name: &str,
    plan_id: i64,
    row: &RecordBatch,
) -> Result<ObservedMetrics, KrishivError> {
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for (field, column) in row.schema().fields().iter().zip(row.columns()) {
        keys.push(field.name().clone());
        values.push(literal(field, column.as_ref(), 0)?);
    }
    Ok(ObservedMetrics {
        name: name.to_owned(),
        values,
        keys,
        plan_id,
        root_error_idx: None,
        errors: Vec::new(),
    })
}

fn lit(t: LiteralType) -> Literal {
    Literal {
        literal_type: Some(t),
        data_type: None,
    }
}

fn unsupported(t: &Arrow) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: format!("an observed metric of type {t}"),
    }
    .into()
}

fn null(field: &Field) -> Result<Literal, KrishivError> {
    Ok(lit(LiteralType::Null(proto_type(
        &DataType::from_arrow_field(field)?,
    )?)))
}

#[expect(
    deprecated,
    reason = "Spark's clients still read the struct, map and array element types"
)]
fn literal(field: &Field, a: &dyn Array, i: usize) -> Result<Literal, KrishivError> {
    if a.is_null(i) {
        return null(field);
    }
    Ok(lit(match a.data_type() {
        Arrow::Boolean => LiteralType::Boolean(a.as_boolean().value(i)),
        Arrow::Int8 => LiteralType::Byte(i32::from(a.as_primitive::<Int8Type>().value(i))),
        Arrow::Int16 => LiteralType::Short(i32::from(a.as_primitive::<Int16Type>().value(i))),
        Arrow::Int32 => LiteralType::Integer(a.as_primitive::<Int32Type>().value(i)),
        Arrow::Int64 => LiteralType::Long(a.as_primitive::<Int64Type>().value(i)),
        Arrow::Float32 => LiteralType::Float(a.as_primitive::<Float32Type>().value(i)),
        Arrow::Float64 => LiteralType::Double(a.as_primitive::<Float64Type>().value(i)),
        Arrow::Utf8 => LiteralType::String(a.as_string::<i32>().value(i).to_owned()),
        Arrow::LargeUtf8 => LiteralType::String(a.as_string::<i64>().value(i).to_owned()),
        Arrow::Utf8View => LiteralType::String(a.as_string_view().value(i).to_owned()),
        Arrow::BinaryView => LiteralType::Binary(a.as_binary_view().value(i).to_vec()),
        Arrow::Binary => LiteralType::Binary(a.as_binary::<i32>().value(i).to_vec()),
        Arrow::LargeBinary => LiteralType::Binary(a.as_binary::<i64>().value(i).to_vec()),
        Arrow::Date32 => LiteralType::Date(a.as_primitive::<Date32Type>().value(i)),
        Arrow::Timestamp(TimeUnit::Microsecond, tz) => {
            let v = a.as_primitive::<TimestampMicrosecondType>().value(i);
            if tz.is_some() {
                LiteralType::Timestamp(v)
            } else {
                LiteralType::TimestampNtz(v)
            }
        }
        Arrow::Decimal128(precision, scale) => LiteralType::Decimal(Decimal {
            value: a
                .as_primitive::<arrow_array::types::Decimal128Type>()
                .value_as_string(i),
            precision: Some(i32::from(*precision)),
            scale: Some(i32::from(*scale)),
        }),
        Arrow::List(item) => {
            let list = a.as_list::<i32>().value(i);
            let elements = (0..list.len())
                .map(|j| literal(item, list.as_ref(), j))
                .collect::<Result<Vec<_>, _>>()?;
            let array = ArrayLiteral {
                element_type: Some(proto_type(&DataType::from_arrow_field(item)?)?),
                elements,
            };
            LiteralType::Array(array)
        }
        // Spark's client raises `UNSUPPORTED_LITERAL` for the year-month and calendar
        // forms (the server sends them), so they are sent as Spark's server sends them.
        Arrow::Duration(TimeUnit::Microsecond) => {
            LiteralType::DayTimeInterval(a.as_primitive::<DurationMicrosecondType>().value(i))
        }
        Arrow::Interval(IntervalUnit::YearMonth) => {
            LiteralType::YearMonthInterval(a.as_primitive::<IntervalYearMonthType>().value(i))
        }
        Arrow::Interval(IntervalUnit::MonthDayNano) => {
            let v = a.as_primitive::<IntervalMonthDayNanoType>().value(i);
            LiteralType::CalendarInterval(CalendarInterval {
                months: v.months,
                days: v.days,
                microseconds: v.nanoseconds / 1_000,
            })
        }
        Arrow::Struct(fields) => {
            let st = a.as_struct();
            if is_calendar(fields) {
                return calendar_struct(st, i);
            }
            let elements = fields
                .iter()
                .zip(st.columns())
                .map(|(f, c)| literal(f, c.as_ref(), i))
                .collect::<Result<Vec<_>, _>>()?;
            LiteralType::Struct(StructLiteral {
                struct_type: Some(proto_type(&DataType::from_arrow_field(field)?)?),
                elements,
            })
        }
        Arrow::Map(entries, _) => {
            let Arrow::Struct(kv) = entries.data_type() else {
                return Err(unsupported(a.data_type()));
            };
            let (key, value) = (&kv[0], &kv[1]);
            let map = a.as_map();
            let slice = map.value(i);
            let keys = (0..slice.len())
                .map(|j| literal(key, slice.column(0).as_ref(), j))
                .collect::<Result<Vec<_>, _>>()?;
            let values = (0..slice.len())
                .map(|j| literal(value, slice.column(1).as_ref(), j))
                .collect::<Result<Vec<_>, _>>()?;
            LiteralType::Map(MapLiteral {
                key_type: Some(proto_type(&DataType::from_arrow_field(key)?)?),
                value_type: Some(proto_type(&DataType::from_arrow_field(value)?)?),
                keys,
                values,
            })
        }
        other => return Err(unsupported(other)),
    }))
}

/// Whether `fields` are the session's calendar-interval carrier (`months`, `days`, `microseconds`).
fn is_calendar(fields: &arrow_schema::Fields) -> bool {
    fields.len() == 3
        && fields.iter().any(|f| {
            f.metadata()
                .contains_key(krishiv_types::arrow::CALENDAR_CHILD_METADATA_KEY)
        })
}

fn calendar_struct(st: &arrow_array::StructArray, i: usize) -> Result<Literal, KrishivError> {
    let col = |n: &str| {
        st.column_by_name(n)
            .ok_or_else(|| unsupported(st.data_type()))
    };
    Ok(lit(LiteralType::CalendarInterval(CalendarInterval {
        months: col("months")?.as_primitive::<Int32Type>().value(i),
        days: col("days")?.as_primitive::<Int32Type>().value(i),
        microseconds: col("microseconds")?.as_primitive::<Int64Type>().value(i),
    })))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int64Array, StringArray};
    use arrow_schema::Schema;

    use super::*;

    #[test]
    fn a_row_becomes_keys_and_literals() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("n", Arrow::Int64, false),
            Field::new("s", Arrow::Utf8, true),
        ]));
        let row = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![3])),
                Arc::new(StringArray::from(vec![None::<&str>])),
            ],
        )
        .unwrap();
        let m = observed_metrics("obs", 7, &row).unwrap();
        assert_eq!((m.name.as_str(), m.plan_id), ("obs", 7));
        assert_eq!(m.keys, ["n", "s"]);
        assert_eq!(m.values[0].literal_type, Some(LiteralType::Long(3)));
        assert!(matches!(
            m.values[1].literal_type,
            Some(LiteralType::Null(_))
        ));
    }

    #[test]
    fn interval_forms_are_spark_s() {
        use arrow_array::{
            DurationMicrosecondArray, IntervalMonthDayNanoArray, IntervalYearMonthArray,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", Arrow::Duration(TimeUnit::Microsecond), true),
            Field::new("y", Arrow::Interval(IntervalUnit::YearMonth), true),
            Field::new("c", Arrow::Interval(IntervalUnit::MonthDayNano), true),
        ]));
        let row = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(DurationMicrosecondArray::from(vec![172_800_000_000])),
                Arc::new(IntervalYearMonthArray::from(vec![2])),
                Arc::new(IntervalMonthDayNanoArray::from(vec![
                    arrow_array::types::IntervalMonthDayNano::new(1, 2, 3_723_500_000_000),
                ])),
            ],
        )
        .unwrap();
        let m = observed_metrics("obs", 1, &row).unwrap();
        assert_eq!(
            m.values[0].literal_type,
            Some(LiteralType::DayTimeInterval(172_800_000_000))
        );
        assert_eq!(
            m.values[1].literal_type,
            Some(LiteralType::YearMonthInterval(2))
        );
        assert_eq!(
            m.values[2].literal_type,
            Some(LiteralType::CalendarInterval(CalendarInterval {
                months: 1,
                days: 2,
                microseconds: 3_723_500_000
            }))
        );
    }

    #[test]
    fn a_node_that_did_not_run_has_no_values() {
        let row = RecordBatch::new_empty(Arc::new(Schema::empty()));
        let m = observed_metrics("obs", 0, &row).unwrap();
        assert!(m.keys.is_empty() && m.values.is_empty());
    }
}
