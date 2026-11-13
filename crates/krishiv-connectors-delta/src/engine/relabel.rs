// SPDX-License-Identifier: Apache-2.0
//! Arrays relabelled to a type that differs only in labels: field names, nullability, metadata
//! and timestamp zone labels.
//!
//! Without `chrono-tz` (ADR 0028) Arrow cannot parse the zone label `UTC`, so a `timestamp`
//! crosses arrow-json and arrow-cast labelled `+00:00` (the same instants) and is relabelled
//! here. The rebuild is shallow: zone labels are set on the leaves with `with_timezone_opt`,
//! and structs, lists and maps are rebuilt over their existing children and buffers. Nothing is
//! copied, and no offsets or UTF-8 are re-validated (the checks left are O(1) per array, plus
//! the null count of a child declared non-nullable).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType,
};
use arrow_array::{
    Array, ArrayRef, LargeListArray, ListArray, MapArray, RecordBatch, RecordBatchOptions,
    StructArray,
};
use arrow_schema::{
    ArrowError, DataType as ArrowType, Field, FieldRef, Fields, Schema, SchemaRef, TimeUnit,
};

/// The zone label timestamps carry through arrow-json and arrow-cast.
pub(crate) const ARROW_UTC: &str = "+00:00";

/// `t` with every zoned timestamp labelled `zone`.
#[must_use]
pub(crate) fn with_zone(t: &ArrowType, zone: &str) -> ArrowType {
    let field = |f: &FieldRef| {
        Arc::new(
            Field::new(f.name(), with_zone(f.data_type(), zone), f.is_nullable())
                .with_metadata(f.metadata().clone()),
        )
    };
    match t {
        ArrowType::Timestamp(unit, Some(_)) => ArrowType::Timestamp(*unit, Some(zone.into())),
        ArrowType::Struct(fields) => {
            ArrowType::Struct(fields.iter().map(field).collect::<Fields>())
        }
        ArrowType::List(f) => ArrowType::List(field(f)),
        ArrowType::LargeList(f) => ArrowType::LargeList(field(f)),
        ArrowType::Map(f, sorted) => ArrowType::Map(field(f), *sorted),
        other => other.clone(),
    }
}

/// `schema` with every zoned timestamp labelled `zone`.
#[must_use]
pub(crate) fn schema_with_zone(schema: &Schema, zone: &str) -> Schema {
    Schema::new_with_metadata(
        schema
            .fields()
            .iter()
            .map(|f| {
                Field::new(f.name(), with_zone(f.data_type(), zone), f.is_nullable())
                    .with_metadata(f.metadata().clone())
            })
            .collect::<Vec<_>>(),
        schema.metadata().clone(),
    )
}

fn mismatch(from: &ArrowType, to: &ArrowType) -> ArrowError {
    ArrowError::InvalidArgumentError(format!("cannot relabel {from} as {to}"))
}

fn timestamp(array: &ArrayRef, unit: TimeUnit, zone: Option<&Arc<str>>) -> ArrayRef {
    let zone = zone.cloned();
    match unit {
        TimeUnit::Second => Arc::new(
            array
                .as_primitive::<TimestampSecondType>()
                .clone()
                .with_timezone_opt(zone),
        ),
        TimeUnit::Millisecond => Arc::new(
            array
                .as_primitive::<TimestampMillisecondType>()
                .clone()
                .with_timezone_opt(zone),
        ),
        TimeUnit::Microsecond => Arc::new(
            array
                .as_primitive::<TimestampMicrosecondType>()
                .clone()
                .with_timezone_opt(zone),
        ),
        TimeUnit::Nanosecond => Arc::new(
            array
                .as_primitive::<TimestampNanosecondType>()
                .clone()
                .with_timezone_opt(zone),
        ),
    }
}

/// `array` relabelled as `target`, which must differ from its type only in field names,
/// nullability, metadata and timestamp zone labels.
///
/// # Errors
/// A shape mismatch, or nulls in a child `target` declares non-nullable.
pub(crate) fn relabel(array: &ArrayRef, target: &ArrowType) -> Result<ArrayRef, ArrowError> {
    if array.data_type() == target {
        return Ok(Arc::clone(array));
    }
    match (array.data_type(), target) {
        (ArrowType::Timestamp(from, _), ArrowType::Timestamp(to, zone)) if from == to => {
            Ok(timestamp(array, *to, zone.as_ref()))
        }
        (ArrowType::Struct(from), ArrowType::Struct(to)) if from.len() == to.len() => {
            let s = array.as_struct();
            let columns = s
                .columns()
                .iter()
                .zip(to.iter())
                .map(|(c, f)| relabel(c, f.data_type()))
                .collect::<Result<Vec<_>, _>>()?;
            let out = if columns.is_empty() {
                StructArray::new_empty_fields(s.len(), s.nulls().cloned())
            } else {
                StructArray::try_new(to.clone(), columns, s.nulls().cloned())?
            };
            Ok(Arc::new(out))
        }
        (ArrowType::List(_), ArrowType::List(to)) => {
            let l = array.as_list::<i32>();
            let values = relabel(l.values(), to.data_type())?;
            Ok(Arc::new(ListArray::try_new(
                Arc::clone(to),
                l.offsets().clone(),
                values,
                l.nulls().cloned(),
            )?))
        }
        (ArrowType::LargeList(_), ArrowType::LargeList(to)) => {
            let l = array.as_list::<i64>();
            let values = relabel(l.values(), to.data_type())?;
            Ok(Arc::new(LargeListArray::try_new(
                Arc::clone(to),
                l.offsets().clone(),
                values,
                l.nulls().cloned(),
            )?))
        }
        (ArrowType::Map(..), ArrowType::Map(to, sorted)) => {
            let m = array.as_map();
            let entries: ArrayRef = Arc::new(m.entries().clone());
            let entries = relabel(&entries, to.data_type())?;
            Ok(Arc::new(MapArray::try_new(
                Arc::clone(to),
                m.offsets().clone(),
                entries.as_struct().clone(),
                m.nulls().cloned(),
                *sorted,
            )?))
        }
        (from, to) => Err(mismatch(from, to)),
    }
}

/// `batch` relabelled as `schema` (see [`relabel`]).
///
/// # Errors
/// As [`relabel`], or a column count mismatch.
pub(crate) fn relabel_batch(
    batch: &RecordBatch,
    schema: SchemaRef,
) -> Result<RecordBatch, ArrowError> {
    if batch.schema().as_ref() == schema.as_ref() {
        return Ok(batch.clone());
    }
    if batch.num_columns() != schema.fields().len() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "cannot relabel {} columns as {}",
            batch.num_columns(),
            schema.fields().len()
        )));
    }
    let columns = batch
        .columns()
        .iter()
        .zip(schema.fields())
        .map(|(c, f)| relabel(c, f.data_type()))
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new_with_options(
        schema,
        columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
}
