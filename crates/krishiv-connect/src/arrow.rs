// SPDX-License-Identifier: Apache-2.0
//! The Arrow shapes Spark Connect sends for result columns whose in-memory
//! form differs from Spark's (Spark 4.2's `ArrowUtils`, as a client
//! receives it).
//!
//! - `time(p)`: `Time64(Nanosecond)` with no field metadata, which is the
//!   in-memory form; nothing to convert.
//! - `geometry` / `geography`: `Struct<srid: Int32, wkb: Binary>`, both
//!   children non-null, the `wkb` child tagged `geometry: "true"` (or
//!   `geography: "true"`) and `srid: "<SRID of the type>"`, -1 for `ANY`.
//!   In memory the WKB child is a `BinaryView`, without the tags.
//! - `variant`: `Struct<value: Binary, metadata: Binary>` (that order), both
//!   children non-null, the `metadata` child tagged `variant: "true"`,
//!   without the Arrow extension name (Spark's `ArrowUtils.toArrowField`,
//!   v4.0.0 to v4.3.0). In memory the children are `metadata`
//!   then `value`, `BinaryView`, marked by `krishiv.variant`.
//!
//! The other result conversions (time zones, strings) belong to the result
//! path of the Connect server, which uses these.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BinaryArray, BinaryViewArray, StructArray};
use arrow_schema::{DataType as ArrowType, Field, Fields};
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_types::arrow::{
    SPATIAL_SRID_FIELD, SPATIAL_WKB_FIELD, VARIANT_METADATA_FIELD, VARIANT_VALUE_FIELD,
};
use krishiv_types::{DataType, Srid};

/// The children of a spatial column of type `data_type` as Spark Connect
/// sends them, or `None` for a type that is not spatial.
fn spatial_children(data_type: &DataType) -> Option<Fields> {
    let (tag, srid) = match data_type {
        DataType::Geometry { srid } => ("geometry", srid),
        DataType::Geography { srid } => ("geography", srid),
        _ => return None,
    };
    let metadata = HashMap::from([
        (tag.to_owned(), "true".to_owned()),
        ("srid".to_owned(), Srid::to_i32(*srid).to_string()),
    ]);
    Some(Fields::from(vec![
        Field::new(SPATIAL_SRID_FIELD, ArrowType::Int32, false),
        Field::new(SPATIAL_WKB_FIELD, ArrowType::Binary, false).with_metadata(metadata),
    ]))
}

/// The result field Spark Connect sends for a column `name` of a spatial
/// type, or `None` for a type that is not spatial.
pub fn spatial_field(name: &str, data_type: &DataType, nullable: bool) -> Option<Field> {
    let children = spatial_children(data_type)?;
    Some(Field::new(name, ArrowType::Struct(children), nullable))
}

/// Converts a spatial column from its in-memory form to the form Spark
/// Connect sends (see the module docs). The rows, SRIDs and nulls are
/// unchanged; the WKB bytes are copied into a `Binary` child.
///
/// # Errors
/// `INTERNAL_ERROR` when `data_type` is not spatial or `column` is not
/// the in-memory form of a spatial column (an engine bug).
pub fn spatial_column(column: &ArrayRef, data_type: &DataType) -> Result<ArrayRef, KrishivError> {
    let internal = |what: &str| -> KrishivError {
        InternalError {
            message: format!("a spatial result column {what}"),
        }
        .into()
    };
    let children = spatial_children(data_type).ok_or_else(|| internal("of another type"))?;
    let input = column
        .as_any()
        .downcast_ref::<StructArray>()
        .filter(|s| s.num_columns() == 2)
        .ok_or_else(|| internal("that is not a two-field struct"))?;
    // Where the struct itself is null (an outer join pads with null
    // structs, whose children Arrow leaves as arbitrary placeholders), the
    // children are not read: only the rows the struct holds are checked.
    let valid = |i: usize| input.nulls().is_none_or(|n| n.is_valid(i));
    let srid = input.column(0);
    let srid_ok = srid.data_type() == &ArrowType::Int32
        && (srid.null_count() == 0 || (0..input.len()).all(|i| !valid(i) || srid.is_valid(i)));
    if !srid_ok {
        return Err(internal("without a non-null Int32 SRID"));
    }
    let wkb = input
        .column(1)
        .as_any()
        .downcast_ref::<BinaryViewArray>()
        .filter(|w| w.null_count() == 0 || (0..input.len()).all(|i| !valid(i) || w.is_valid(i)))
        .ok_or_else(|| internal("without non-null WKB views"))?;
    // `Binary` has 32-bit offsets: a batch is split before it gets here.
    let bytes: usize = wkb.iter().map(|v| v.map_or(0, <[u8]>::len)).sum();
    if i32::try_from(bytes).is_err() {
        return Err(internal("with more than 2 GiB of WKB in one batch"));
    }
    // A null child under a null struct becomes an empty placeholder.
    let wkb: BinaryArray = wkb.iter().map(|v| Some(v.unwrap_or_default())).collect();
    let out = StructArray::try_new(
        children,
        vec![Arc::clone(srid), Arc::new(wkb)],
        input.nulls().cloned(),
    )
    .map_err(|e| internal(&format!("that Arrow refuses: {e}")))?;
    Ok(Arc::new(out))
}

/// The children of a variant column as Spark Connect sends them: the
/// `metadata` child carries the `variant: "true"` tag (Spark's
/// `ArrowUtils.toArrowField`; `isVariantField` and PySpark's `is_variant`
/// look for it there).
fn variant_children() -> Fields {
    Fields::from(vec![
        Field::new(VARIANT_VALUE_FIELD, ArrowType::Binary, false),
        Field::new(VARIANT_METADATA_FIELD, ArrowType::Binary, false)
            .with_metadata(HashMap::from([("variant".to_owned(), "true".to_owned())])),
    ])
}

/// The result field Spark Connect sends for a `variant` column `name`.
#[must_use]
pub fn variant_field(name: &str, nullable: bool) -> Field {
    Field::new(name, ArrowType::Struct(variant_children()), nullable)
}

/// Converts a variant column from its in-memory form to the form Spark
/// Connect sends (see the module docs): the same rows and nulls, the
/// children swapped into `value, metadata` order and copied to `Binary`.
///
/// # Errors
/// `INTERNAL_ERROR` when `column` is not an in-memory variant column (an
/// engine bug), or holds more than 2 GiB of bytes in one batch.
pub fn variant_column(column: &ArrayRef) -> Result<ArrayRef, KrishivError> {
    let internal = |what: &str| -> KrishivError {
        InternalError {
            message: format!("a variant result column {what}"),
        }
        .into()
    };
    let input = column
        .as_any()
        .downcast_ref::<StructArray>()
        .filter(|s| krishiv_types::arrow::is_variant_struct(s.data_type()))
        .ok_or_else(|| internal("that is not a variant struct"))?;
    let child = |name: &str| -> Result<BinaryArray, KrishivError> {
        let view = input
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<BinaryViewArray>())
            .ok_or_else(|| internal("without binary view children"))?;
        let bytes: usize = view.iter().map(|v| v.map_or(0, <[u8]>::len)).sum();
        if i32::try_from(bytes).is_err() {
            return Err(internal("with more than 2 GiB in one batch"));
        }
        Ok(view.iter().map(|v| Some(v.unwrap_or_default())).collect())
    };
    let value = child(VARIANT_VALUE_FIELD)?;
    let metadata = child(VARIANT_METADATA_FIELD)?;
    let out = StructArray::try_new(
        variant_children(),
        vec![Arc::new(value), Arc::new(metadata)],
        input.nulls().cloned(),
    )
    .map_err(|e| internal(&format!("that Arrow refuses: {e}")))?;
    Ok(Arc::new(out))
}

#[cfg(test)]
mod tests;
