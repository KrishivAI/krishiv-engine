// SPDX-License-Identifier: Apache-2.0
//! ANSI intervals in Parquet, as Spark stores them (design P2-T001b §3): a
//! year-month interval is a plain `INT32`, a day-time interval a plain
//! `INT64`, and the Spark type survives only in the footer's row metadata
//! ([`ROW_METADATA_KEY`], which Spark writes for every file and reads first).
//!
//! The decoders still produce those integers, since arrow-rs takes no other
//! type for them. [`typed`] puts the footer's types on the file schema, and
//! [`as_intervals`] puts them on each decoded column.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{DurationMicrosecondType, Int32Type, Int64Type, IntervalYearMonthType};
use arrow_array::{Array, ArrayRef, LargeListArray, ListArray, MapArray, StructArray};
use arrow_schema::{DataType, FieldRef, Fields, IntervalUnit, Schema, TimeUnit};
use krishiv_types::DataType as SparkType;
use parquet::file::metadata::KeyValue;

use crate::error::{FormatError, FormatResult};

/// The footer key of Spark's schema JSON.
pub(super) const ROW_METADATA_KEY: &str = "org.apache.spark.sql.parquet.row.metadata";

const YEAR_MONTH: DataType = DataType::Interval(IntervalUnit::YearMonth);
const DAY_TIME: DataType = DataType::Duration(TimeUnit::Microsecond);

/// The footer value for `schema` (`None`: a type with no Spark form, which
/// the engine's own schemas don't hold).
pub(super) fn row_metadata(schema: &Schema) -> Option<String> {
    SparkType::from_arrow(&DataType::Struct(schema.fields().clone()))
        .ok()
        .map(|t| t.json())
}

/// `fields` with the ANSI interval types the footer gives their integers.
/// Footer text that doesn't parse is ignored, as Spark ignores it, and a
/// footer type that doesn't suit the stored type leaves the column alone.
pub(super) fn typed(fields: Vec<FieldRef>, footer: Option<&Vec<KeyValue>>) -> Vec<FieldRef> {
    let spark = footer
        .into_iter()
        .flatten()
        .find(|kv| kv.key == ROW_METADATA_KEY)
        .and_then(|kv| kv.value.as_deref())
        .and_then(|json| SparkType::from_json(json).ok());
    let Some(SparkType::Struct(spark)) = spark else {
        return fields;
    };
    fields
        .into_iter()
        .map(
            |f| match spark.fields().iter().find(|s| s.name() == f.name()) {
                Some(s) => typed_field(&f, s.data_type()),
                None => f,
            },
        )
        .collect()
}

fn typed_field(f: &FieldRef, spark: &SparkType) -> FieldRef {
    let retyped = |t: DataType| Arc::new(f.as_ref().clone().with_data_type(t));
    match (f.data_type(), spark) {
        (DataType::Int32 | &YEAR_MONTH, SparkType::YearMonthInterval { .. })
        | (DataType::Int64 | &DAY_TIME, SparkType::DayTimeInterval { .. }) => {
            let tagged = spark.to_arrow_field(f.name(), f.is_nullable());
            let mut metadata = f.metadata().clone();
            metadata.extend(tagged.metadata().clone());
            Arc::new(tagged.with_metadata(metadata))
        }
        (DataType::Struct(children), SparkType::Struct(s)) => {
            let children: Fields = children
                .iter()
                .map(|c| match s.fields().iter().find(|x| x.name() == c.name()) {
                    Some(x) => typed_field(c, x.data_type()),
                    None => Arc::clone(c),
                })
                .collect();
            retyped(DataType::Struct(children))
        }
        (DataType::List(e), SparkType::Array(a)) => {
            retyped(DataType::List(typed_field(e, a.element_type())))
        }
        (DataType::LargeList(e), SparkType::Array(a)) => {
            retyped(DataType::LargeList(typed_field(e, a.element_type())))
        }
        (DataType::Map(entries, sorted), SparkType::Map(m)) => {
            let DataType::Struct(kv) = entries.data_type() else {
                return Arc::clone(f);
            };
            let (Some(k), Some(v)) = (kv.first(), kv.get(1)) else {
                return Arc::clone(f);
            };
            let kv: Fields = [typed_field(k, m.key_type()), typed_field(v, m.value_type())]
                .into_iter()
                .collect();
            let entries = Arc::new(
                entries
                    .as_ref()
                    .clone()
                    .with_data_type(DataType::Struct(kv)),
            );
            retyped(DataType::Map(entries, *sorted))
        }
        _ => Arc::clone(f),
    }
}

/// Whether `t` is or holds an ANSI interval.
pub(super) fn holds_interval(t: &DataType) -> bool {
    match t {
        DataType::Struct(f) => f.iter().any(|c| holds_interval(c.data_type())),
        DataType::List(e) | DataType::LargeList(e) | DataType::Map(e, _) => {
            holds_interval(e.data_type())
        }
        t => *t == YEAR_MONTH || *t == DAY_TIME,
    }
}

/// `a`, decoded as integers, as `target` types its intervals, at any depth.
pub(super) fn as_intervals(a: &ArrayRef, target: &DataType) -> FormatResult<ArrayRef> {
    let bad = |e: arrow_schema::ArrowError| FormatError::Corrupt(e.to_string());
    if a.data_type() == target || !holds_interval(target) {
        return Ok(Arc::clone(a));
    }
    Ok(match (a.data_type(), target) {
        (DataType::Int32, &YEAR_MONTH) => Arc::new(
            a.as_primitive::<Int32Type>()
                .reinterpret_cast::<IntervalYearMonthType>(),
        ),
        (DataType::Int64, &DAY_TIME) => Arc::new(
            a.as_primitive::<Int64Type>()
                .reinterpret_cast::<DurationMicrosecondType>(),
        ),
        (DataType::Struct(_), DataType::Struct(to)) => {
            let s = a.as_struct();
            let columns = s
                .columns()
                .iter()
                .zip(to.iter())
                .map(|(c, t)| as_intervals(c, t.data_type()))
                .collect::<FormatResult<Vec<_>>>()?;
            Arc::new(
                StructArray::try_new_with_length(to.clone(), columns, s.nulls().cloned(), s.len())
                    .map_err(bad)?,
            )
        }
        (DataType::List(_), DataType::List(to)) => {
            let l = a.as_list::<i32>();
            let values = as_intervals(l.values(), to.data_type())?;
            Arc::new(
                ListArray::try_new(
                    Arc::clone(to),
                    l.offsets().clone(),
                    values,
                    l.nulls().cloned(),
                )
                .map_err(bad)?,
            )
        }
        (DataType::LargeList(_), DataType::LargeList(to)) => {
            let l = a.as_list::<i64>();
            let values = as_intervals(l.values(), to.data_type())?;
            Arc::new(
                LargeListArray::try_new(
                    Arc::clone(to),
                    l.offsets().clone(),
                    values,
                    l.nulls().cloned(),
                )
                .map_err(bad)?,
            )
        }
        (DataType::Map(_, sorted), DataType::Map(to, _)) => {
            let m = a.as_map();
            let entries: ArrayRef = Arc::new(m.entries().clone());
            let entries = as_intervals(&entries, to.data_type())?;
            Arc::new(
                MapArray::try_new(
                    Arc::clone(to),
                    m.offsets().clone(),
                    entries.as_struct().clone(),
                    m.nulls().cloned(),
                    *sorted,
                )
                .map_err(bad)?,
            )
        }
        _ => Arc::clone(a),
    })
}
