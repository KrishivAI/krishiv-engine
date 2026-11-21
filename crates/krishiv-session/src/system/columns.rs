// SPDX-License-Identifier: Apache-2.0
//! Nested column builders of the `system` tables: histogram buckets (`system.metrics`) and advisor hints
//! (`system.queries.hints`).

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, Int64Array};
use arrow_schema::DataType as ArrowType;
use krishiv_types::DataType;

/// The element field and the struct fields of an Arrow list of structs; `None` for any other type (which an
/// array of structs never maps to).
fn list_of_struct(t: ArrowType) -> Option<(arrow_schema::FieldRef, arrow_schema::Fields)> {
    let ArrowType::List(element) = t else {
        return None;
    };
    let ArrowType::Struct(fields) = element.data_type().clone() else {
        return None;
    };
    Some((element, fields))
}

/// `ARRAY<STRUCT<le: DOUBLE, count: BIGINT>>`.
pub(super) fn buckets_type() -> DataType {
    use krishiv_types::{ArrayType, StructField, StructType};
    DataType::Array(
        ArrayType::new(
            DataType::Struct(StructType::new([
                StructField::new("le", DataType::Double, false),
                StructField::new("count", DataType::Long, false),
            ])),
            false,
        )
        .into(),
    )
}

/// An `ARRAY<STRUCT<le, count>>` column; a row with `None` is NULL.
pub(super) fn buckets_column(rows: &[Option<&[(f64, i64)]>]) -> ArrayRef {
    use arrow_array::{Float64Array, ListArray, StructArray};
    use arrow_buffer::{NullBuffer, OffsetBuffer};
    let Some((element, fields)) = list_of_struct(buckets_type().to_arrow()) else {
        return arrow_array::new_null_array(&buckets_type().to_arrow(), rows.len());
    };
    let les =
        Float64Array::from_iter_values(rows.iter().flatten().flat_map(|r| r.iter().map(|b| b.0)));
    let counts =
        Int64Array::from_iter_values(rows.iter().flatten().flat_map(|r| r.iter().map(|b| b.1)));
    let offsets = OffsetBuffer::from_lengths(rows.iter().map(|r| r.map_or(0, <[_]>::len)));
    let values = StructArray::new(fields, vec![Arc::new(les), Arc::new(counts)], None);
    let valid = NullBuffer::from_iter(rows.iter().map(Option::is_some));
    Arc::new(ListArray::new(
        element,
        offsets,
        Arc::new(values),
        Some(valid),
    ))
}

/// `ARRAY<STRUCT<code: STRING, node_id: INT, operator_id: INT, message: STRING>>`.
pub(super) fn hints_type() -> DataType {
    use krishiv_types::{ArrayType, StructField, StructType};
    DataType::Array(
        ArrayType::new(
            DataType::Struct(StructType::new([
                StructField::new("code", DataType::String, false),
                StructField::new("node_id", DataType::Integer, true),
                StructField::new("operator_id", DataType::Integer, true),
                StructField::new("message", DataType::String, false),
            ])),
            false,
        )
        .into(),
    )
}

/// One advisor hint of a finished query, as `system.queries.hints` lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HintRow {
    pub(crate) code: &'static str,
    pub(crate) node_id: Option<i32>,
    pub(crate) operator_id: Option<i32>,
    pub(crate) message: String,
}

impl HintRow {
    /// The row of `h`.
    pub(crate) fn of(h: &krishiv_exec::profile::Hint) -> Self {
        Self {
            code: h.code.as_str(),
            node_id: h.node.and_then(|n| i32::try_from(n).ok()),
            operator_id: h.operator.and_then(|o| i32::try_from(o.0).ok()),
            message: h.message.clone(),
        }
    }
}

/// An `ARRAY<STRUCT<code, node_id, operator_id, message>>` column; a row with `None` is NULL.
pub(super) fn hints_column(rows: &[Option<&[HintRow]>]) -> ArrayRef {
    use arrow_array::{ListArray, StringViewArray, StructArray};
    use arrow_buffer::{NullBuffer, OffsetBuffer};
    let Some((element, fields)) = list_of_struct(hints_type().to_arrow()) else {
        return arrow_array::new_null_array(&hints_type().to_arrow(), rows.len());
    };
    let all = || rows.iter().flatten().flat_map(|r| r.iter());
    let values = StructArray::new(
        fields,
        vec![
            Arc::new(StringViewArray::from_iter_values(all().map(|h| h.code))),
            Arc::new(Int32Array::from_iter(all().map(|h| h.node_id))),
            Arc::new(Int32Array::from_iter(all().map(|h| h.operator_id))),
            Arc::new(StringViewArray::from_iter_values(
                all().map(|h| h.message.as_str()),
            )),
        ],
        None,
    );
    let offsets = OffsetBuffer::from_lengths(rows.iter().map(|r| r.map_or(0, <[_]>::len)));
    let valid = NullBuffer::from_iter(rows.iter().map(Option::is_some));
    Arc::new(ListArray::new(
        element,
        offsets,
        Arc::new(values),
        Some(valid),
    ))
}
