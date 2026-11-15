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
            // Never empty: two structs of no fields have equal types and returned above.
            Ok(Arc::new(StructArray::try_new(
                to.clone(),
                columns,
                s.nulls().cloned(),
            )?))
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

#[cfg(test)]
mod tests {
    use arrow_array::builder::{Int32Builder, MapBuilder, StringBuilder};
    use arrow_array::{
        Int32Array, Int64Array, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray, TimestampSecondArray,
    };
    use arrow_schema::Field;

    use super::*;

    fn zoned(unit: TimeUnit) -> ArrayRef {
        match unit {
            TimeUnit::Second => {
                Arc::new(TimestampSecondArray::from(vec![1]).with_timezone(ARROW_UTC))
            }
            TimeUnit::Millisecond => {
                Arc::new(TimestampMillisecondArray::from(vec![1]).with_timezone(ARROW_UTC))
            }
            TimeUnit::Microsecond => {
                Arc::new(TimestampMicrosecondArray::from(vec![1]).with_timezone(ARROW_UTC))
            }
            TimeUnit::Nanosecond => {
                Arc::new(TimestampNanosecondArray::from(vec![1]).with_timezone(ARROW_UTC))
            }
        }
    }

    #[test]
    fn timestamps_of_every_unit_take_the_zone_label() {
        for unit in [
            TimeUnit::Second,
            TimeUnit::Millisecond,
            TimeUnit::Microsecond,
            TimeUnit::Nanosecond,
        ] {
            let a = zoned(unit);
            let target = ArrowType::Timestamp(unit, Some("UTC".into()));
            let out = relabel(&a, &target).unwrap();
            assert_eq!(out.data_type(), &target);
            assert_eq!(out.to_data().buffers(), a.to_data().buffers(), "{unit:?}");
            // No zone at all is a relabel too.
            let ntz = relabel(&a, &ArrowType::Timestamp(unit, None)).unwrap();
            assert_eq!(ntz.data_type(), &ArrowType::Timestamp(unit, None));
        }
        // Another unit is a conversion, not a relabel.
        let e = relabel(
            &zoned(TimeUnit::Second),
            &ArrowType::Timestamp(TimeUnit::Millisecond, None),
        )
        .unwrap_err();
        assert!(e.to_string().contains("cannot relabel"), "{e}");
    }

    #[test]
    fn nested_types_are_relabelled_over_their_buffers() {
        let ts = ArrowType::Timestamp(TimeUnit::Microsecond, Some(ARROW_UTC.into()));
        let utc = ArrowType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
        // with_zone reaches large lists and maps.
        let large = ArrowType::LargeList(Arc::new(Field::new("item", ts.clone(), true)));
        assert_eq!(
            with_zone(&large, "UTC"),
            ArrowType::LargeList(Arc::new(Field::new("item", utc, true)))
        );
        // A large list of zoned timestamps.
        let values = zoned(TimeUnit::Microsecond);
        let l: ArrayRef = Arc::new(LargeListArray::new(
            Arc::new(Field::new("item", ts, true)),
            arrow_buffer::OffsetBuffer::new(vec![0_i64, 1].into()),
            Arc::clone(&values),
            None,
        ));
        let out = relabel(&l, &with_zone(&large, "UTC")).unwrap();
        assert_eq!(out.data_type(), &with_zone(&large, "UTC"));
        // A list renamed.
        let list: ArrayRef = Arc::new(ListArray::from_iter_primitive::<
            arrow_array::types::Int32Type,
            _,
            _,
        >(vec![Some(vec![Some(1)]), None]));
        let renamed = ArrowType::List(Arc::new(Field::new("element", ArrowType::Int32, true)));
        let out = relabel(&list, &renamed).unwrap();
        assert_eq!(out.data_type(), &renamed);
        assert!(out.is_null(1));
        // A map with other entry names.
        let mut m = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
        m.keys().append_value("k");
        m.values().append_value(1);
        m.append(true).unwrap();
        let map: ArrayRef = Arc::new(m.finish());
        let ArrowType::Map(entries, sorted) = map.data_type() else {
            panic!("{}", map.data_type())
        };
        let ArrowType::Struct(kv) = entries.data_type() else {
            panic!("{entries}")
        };
        let kv: Fields = vec![
            kv[0].as_ref().clone().with_name("key"),
            kv[1].as_ref().clone().with_name("value"),
        ]
        .into();
        let target = ArrowType::Map(
            Arc::new(Field::new("key_value", ArrowType::Struct(kv), false)),
            *sorted,
        );
        let out = relabel(&map, &target).unwrap();
        assert_eq!(out.data_type(), &target);
        // A null child under a field declared non-nullable is refused.
        let nulls: ArrayRef = Arc::new(Int32Array::from(vec![None, Some(1)]));
        let st: ArrayRef = Arc::new(
            StructArray::try_new(
                vec![Field::new("a", ArrowType::Int32, true)].into(),
                vec![nulls],
                None,
            )
            .unwrap(),
        );
        let strict = ArrowType::Struct(vec![Field::new("a", ArrowType::Int32, false)].into());
        assert!(relabel(&st, &strict).is_err());
        // A shape or primitive change is refused.
        let ints: ArrayRef = Arc::new(Int32Array::from(vec![1]));
        let e = relabel(&ints, &ArrowType::Int64).unwrap_err();
        assert!(
            e.to_string().contains("cannot relabel Int32 as Int64"),
            "{e}"
        );
        let wide = ArrowType::Struct(
            vec![
                Field::new("a", ArrowType::Int32, true),
                Field::new("b", ArrowType::Int32, true),
            ]
            .into(),
        );
        assert!(relabel(&st, &wide).is_err());
    }

    #[test]
    fn batches_relabel_by_position() {
        let longs: ArrayRef = Arc::new(Int64Array::from(vec![1_i64]));
        let ts: ArrayRef = zoned(TimeUnit::Microsecond);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("a", ArrowType::Int64, true),
                Field::new("t", ts.data_type().clone(), true),
            ])),
            vec![longs, ts],
        )
        .unwrap();
        let same = relabel_batch(&batch, batch.schema()).unwrap();
        assert_eq!(same.schema(), batch.schema());
        let utc = Arc::new(schema_with_zone(&batch.schema(), "UTC"));
        let out = relabel_batch(&batch, Arc::clone(&utc)).unwrap();
        assert_eq!(out.schema(), utc);
        let one = Arc::new(Schema::new(vec![Field::new("a", ArrowType::Int64, true)]));
        let e = relabel_batch(&batch, one).unwrap_err();
        assert!(
            e.to_string().contains("cannot relabel 2 columns as 1"),
            "{e}"
        );
    }
}
