// SPDX-License-Identifier: Apache-2.0
//! Struct kernels: construction (`named_struct`, `struct`) and field access
//! (`GetStructField`), over Arrow `StructArray`s directly.
//!
//! Construction packs one `StructArray` per batch: the struct itself is never
//! null (Spark's `CreateStruct` is non-nullable), so the struct validity is
//! all-valid and each child keeps its own validity. Field access extracts the
//! child and folds the struct validity into it: a null struct gives a null
//! field.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType as ArrowType, Fields};

use crate::error::KernelError;

/// Builds the struct of `children` (each of `rows` values) with `fields`.
/// The struct itself is never null.
///
/// # Errors
/// `INTERNAL_ERROR` when a child has the wrong length or type.
pub fn build(
    fields: Fields,
    children: Vec<ArrayRef>,
    rows: usize,
) -> Result<StructArray, KernelError> {
    if children.len() != fields.len() {
        return Err(KernelError::internal(format!(
            "a struct of {} fields with {} values",
            fields.len(),
            children.len()
        )));
    }
    for (i, c) in children.iter().enumerate() {
        if c.len() != rows {
            return Err(KernelError::internal(format!(
                "struct field {i} of {} rows for {rows} rows",
                c.len()
            )));
        }
        if c.data_type() != fields[i].data_type() {
            return Err(KernelError::internal(format!(
                "struct field {i} of {} where {} was bound",
                c.data_type(),
                fields[i].data_type()
            )));
        }
    }
    if fields.is_empty() {
        return Ok(StructArray::new_empty_fields(rows, None));
    }
    Ok(StructArray::new(fields, children, None))
}

/// The field `ordinal` of every row of `struct_array`: the child with the
/// struct validity folded in (a null struct gives a null field).
///
/// # Errors
/// `INTERNAL_ERROR` for a non-struct array or an out-of-range ordinal.
pub fn get_field(struct_array: &dyn Array, ordinal: usize) -> Result<ArrayRef, KernelError> {
    let s = struct_array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            KernelError::internal(format!("a struct field of {}", struct_array.data_type()))
        })?;
    let child = s.columns().get(ordinal).ok_or_else(|| {
        KernelError::internal(format!(
            "struct field {ordinal} of {} fields",
            s.columns().len()
        ))
    })?;
    Ok(match s.nulls() {
        None => Arc::clone(child),
        Some(validity) => fold_validity(child.as_ref(), validity)?,
    })
}

/// `child` with the nulls of `struct_validity` folded in.
fn fold_validity(child: &dyn Array, struct_validity: &NullBuffer) -> Result<ArrayRef, KernelError> {
    let merged = NullBuffer::union(child.nulls(), Some(struct_validity));
    let data = child
        .to_data()
        .into_builder()
        .nulls(merged)
        .build()
        .map_err(|e| KernelError::internal(e.to_string()))?;
    Ok(arrow_array::make_array(data))
}

/// Whether `t` is directly orderable by [`crate::compare`] (the leaves of a
/// struct comparison).
fn leaf_orderable(t: &ArrowType) -> bool {
    use arrow_schema::DataType as A;
    matches!(
        t,
        A::Boolean
            | A::Int8
            | A::Int16
            | A::Int32
            | A::Int64
            | A::Float32
            | A::Float64
            | A::Decimal128(..)
            | A::Date32
            | A::Timestamp(..)
            | A::Utf8
            | A::Utf8View
            | A::LargeUtf8
            | A::Binary
            | A::BinaryView
            | A::LargeBinary
            | A::Null
    )
}

/// Whether structs of Arrow type `t` can be compared field-wise.
pub fn comparable(t: &ArrowType) -> bool {
    match t {
        ArrowType::Struct(fields) => fields.iter().all(|f| {
            let dt = f.data_type();
            drilled_orderable(dt)
        }),
        _ => false,
    }
}

/// Whether `t` is orderable, recursing into structs.
fn drilled_orderable(t: &ArrowType) -> bool {
    match t {
        ArrowType::Struct(fields) => fields.iter().all(|f| drilled_orderable(f.data_type())),
        other => leaf_orderable(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::Field;

    fn int(values: Vec<Option<i32>>) -> ArrayRef {
        Arc::new(Int32Array::from(values))
    }

    fn str(values: Vec<Option<&str>>) -> ArrayRef {
        Arc::new(StringArray::from(values))
    }

    fn fields() -> Fields {
        vec![
            Field::new("a", ArrowType::Int32, true),
            Field::new("b", ArrowType::Utf8, true),
        ]
        .into()
    }

    #[test]
    fn build_packs_children_with_all_valid_struct() {
        let s = build(
            fields(),
            vec![int(vec![Some(1), None]), str(vec![Some("x"), None])],
            2,
        )
        .unwrap();
        assert_eq!(s.len(), 2);
        assert!(s.is_valid(0) && s.is_valid(1));
        assert!(s.column(0).is_null(1));
        assert!(s.column(1).is_null(1));
    }

    #[test]
    fn build_rejects_ragged_children() {
        let err = build(fields(), vec![int(vec![Some(1)]), str(vec![None, None])], 2)
            .unwrap_err()
            .to_string();
        assert!(err.contains("1 rows for 2 rows"), "{err}");
    }

    #[test]
    fn get_field_reads_child_and_folds_struct_nulls() {
        let s = build(
            fields(),
            vec![int(vec![Some(1), Some(2)]), str(vec![Some("x"), Some("y")])],
            2,
        )
        .unwrap();
        let col = get_field(&s, 1).unwrap();
        assert_eq!(col.len(), 2);

        // A null struct gives a null field.
        let nulls = NullBuffer::from_iter([false, true]);
        let nulled = StructArray::new(
            fields(),
            vec![int(vec![Some(1), Some(2)]), str(vec![Some("x"), Some("y")])],
            Some(nulls),
        );
        let col = get_field(&nulled, 0).unwrap();
        assert!(col.is_null(0));
        assert!(!col.is_null(1));
    }

    #[test]
    fn get_field_rejects_non_structs_and_bad_ordinals() {
        let err = get_field(int(vec![Some(1)]).as_ref(), 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("a struct field of"), "{err}");
        let s = build(fields(), vec![int(vec![Some(1)]), str(vec![Some("x")])], 1).unwrap();
        let err = get_field(&s, 7).unwrap_err().to_string();
        assert!(err.contains("struct field 7 of 2 fields"), "{err}");
    }

    #[test]
    fn build_and_read_empty_and_nested_structs() {
        let empty = build(Fields::empty(), Vec::new(), 0).unwrap();
        assert_eq!(empty.len(), 0);
        let inner = build(
            Fields::from(vec![Field::new("x", ArrowType::Int32, true)]),
            vec![int(vec![Some(1), None])],
            2,
        )
        .unwrap();
        let outer = build(
            Fields::from(vec![
                Field::new("in", ArrowType::Struct(inner.fields().clone()), true),
                Field::new("b", ArrowType::Int32, true),
            ]),
            vec![Arc::new(inner) as ArrayRef, int(vec![Some(2), Some(3)])],
            2,
        )
        .unwrap();
        assert_eq!(outer.len(), 2);
        let flat = get_field(&outer, 0).unwrap();
        let back = get_field(flat.as_ref(), 0).unwrap();
        assert_eq!(back.len(), 2);
        assert!(back.is_null(1));
    }

    #[test]
    fn build_and_read_across_chunk_boundaries() {
        let rows = 200_usize;
        let ints: Vec<Option<i32>> = (0..200_i32).map(Some).collect();
        let strs: Vec<Option<&str>> = (0..rows).map(|_| Some("v")).collect();
        let s = build(fields(), vec![int(ints), str(strs)], rows).unwrap();
        assert_eq!(s.len(), rows);
        let col = get_field(&s, 0).unwrap();
        assert_eq!(col.len(), rows);
        for i in [0, 63, 64, 65, 127, 128, 199] {
            assert!(!col.is_null(i));
        }
    }
}
