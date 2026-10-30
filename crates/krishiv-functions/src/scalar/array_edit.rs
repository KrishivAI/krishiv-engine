// SPDX-License-Identifier: Apache-2.0
//! Array editing: `array_append`, `array_prepend`, `array_insert` and
//! `array_compact`.
//!
//! `array_insert` counts from 1; a position past the end pads with NULLs, and
//! a negative one counts from the end (`-1` appends), padding the front with
//! NULLs when it reaches past it. Position 0 is `INVALID_INDEX_OF_ZERO`. A
//! NULL array or position is NULL, and a NULL item is inserted as an element.
//! (`spark.sql.legacy.negativeIndexInArrayInsert` is not read: a negative
//! position always follows the default rule.)

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, ListArray, UInt32Array};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::DataType;
use arrow_select::concat::concat;
use arrow_select::take::take;
use krishiv_common::error::{
    CollectionSizeLimitExceededFunction, InvalidIndexOfZero, KrishivError,
};

use super::args::{Inputs, internal};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "array_append",
    "array_compact",
    "array_insert",
    "array_prepend",
];

/// Spark's `ByteArrayMethods.MAX_ROUNDED_ARRAY_LENGTH`.
const MAX_ARRAY_LENGTH: usize = 2_147_483_632;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Edit {
    Append,
    Prepend,
    Insert,
    Compact,
}

/// Binds the array-editing function `name` to `args`.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let edit = match listed {
        "array_append" => Edit::Append,
        "array_prepend" => Edit::Prepend,
        "array_insert" => Edit::Insert,
        _ => Edit::Compact,
    };
    let item_nullable = |i: usize| args.get(i).is_some_and(BindArg::nullable);
    let typed = match (edit, types.as_slice()) {
        (Edit::Compact, [DataType::List(f)]) => Some(compact_type(&DataType::List(Arc::clone(f)))),
        (Edit::Append | Edit::Prepend, [DataType::List(f), item]) if f.data_type() == item => {
            Some(grown_type(f, item_nullable(1)))
        }
        (Edit::Insert, [DataType::List(f), DataType::Int32, item]) if f.data_type() == item => {
            Some(grown_type(f, item_nullable(2)))
        }
        _ => None,
    };
    let binding = match typed.map(|ty| element_field(&ty).map(|field| (ty, field))) {
        Some(Ok((ty, field))) if edit == Edit::Compact => {
            Ok(Binding::new(ty, move |x: &Inputs<'_>| compact(x, &field)))
        }
        Some(Ok((ty, field))) => Ok(Binding::new(ty, move |x: &Inputs<'_>| {
            edit_rows(x, edit, listed, &field)
        })),
        Some(Err(e)) => Err(e),
        None => Err(bad_types(listed, args)),
    };
    Some((listed, binding))
}

/// The result type of an append, prepend or insert: the array's element
/// type, which may hold NULL when the array or the item may.
fn grown_type(f: &arrow_schema::FieldRef, item_nullable: bool) -> DataType {
    DataType::List(Arc::new(arrow_schema::Field::new(
        "element",
        f.data_type().clone(),
        f.is_nullable() || item_nullable,
    )))
}

/// The result type of `array_compact`: the same element type, no NULLs.
fn compact_type(t: &DataType) -> DataType {
    match t {
        DataType::List(f) => DataType::List(Arc::new(arrow_schema::Field::new(
            "element",
            f.data_type().clone(),
            false,
        ))),
        other => other.clone(),
    }
}

/// The element field of a bound list type.
fn element_field(t: &DataType) -> Result<arrow_schema::FieldRef, KrishivError> {
    match t {
        DataType::List(f) => Ok(Arc::clone(f)),
        other => Err(internal(format!("an array edit typed {other}"))),
    }
}

fn list_of<'a>(c: &'a ArrayRef, name: &str) -> Result<&'a ListArray, KrishivError> {
    c.as_list_opt::<i32>()
        .ok_or_else(|| internal(format!("{name} of {}", c.data_type())))
}

/// The first child index and the length of row `row`.
fn span(list: &ListArray, row: usize) -> (usize, usize) {
    (
        usize::try_from(list.value_offsets()[row]).unwrap_or(0),
        usize::try_from(list.value_length(row)).unwrap_or(0),
    )
}

/// `COLLECTION_SIZE_LIMIT_EXCEEDED.FUNCTION` for a result of `elements`.
fn too_long(name: &str, elements: impl std::fmt::Display) -> KrishivError {
    CollectionSizeLimitExceededFunction {
        function_name: format!("`{name}`"),
        number_of_elements: elements.to_string(),
        max_rounded_array_length: MAX_ARRAY_LENGTH.to_string(),
    }
    .into()
}

fn index(i: usize) -> Result<u32, KrishivError> {
    u32::try_from(i).map_err(|_| internal("an array past 2^32 elements".to_owned()))
}

/// The list made of `pieces` (child indexes, `None` for a NULL element) per
/// row, taken from `values`.
fn rebuild(
    field: &arrow_schema::FieldRef,
    values: &ArrayRef,
    rows: &[Option<Vec<Option<u32>>>],
    name: &str,
    limit: usize,
) -> Result<ArrayRef, KrishivError> {
    let mut offsets: Vec<i32> = Vec::with_capacity(rows.len() + 1);
    offsets.push(0);
    let mut picked: Vec<Option<u32>> = Vec::new();
    let mut valid = Vec::with_capacity(rows.len());
    for row in rows {
        if let Some(pieces) = row {
            if pieces.len() > limit {
                return Err(too_long(name, pieces.len()));
            }
            picked.extend_from_slice(pieces);
        }
        valid.push(row.is_some());
        offsets.push(
            i32::try_from(picked.len())
                .map_err(|_| internal("an array past 2^31 elements".to_owned()))?,
        );
    }
    let child = take(values.as_ref(), &UInt32Array::from(picked), None)
        .map_err(|e| internal(e.to_string()))?;
    let nulls = valid.iter().any(|v| !v).then(|| NullBuffer::from(valid));
    Ok(Arc::new(
        ListArray::try_new(
            Arc::clone(field),
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            child,
            nulls,
        )
        .map_err(|e| internal(e.to_string()))?,
    ))
}

/// `array_append`, `array_prepend` and `array_insert` over the rows.
fn edit_rows(
    x: &Inputs<'_>,
    edit: Edit,
    name: &str,
    field: &arrow_schema::FieldRef,
) -> Result<ArrayRef, KrishivError> {
    let c = x.column(0)?;
    let list = list_of(&c, name)?;
    let item = x.column(if edit == Edit::Insert { 2 } else { 1 })?;
    let pos = if edit == Edit::Insert {
        Some(x.column(1)?)
    } else {
        None
    };
    let pos = pos.as_ref().map(AsArray::as_primitive::<Int32Type>);
    // The array's values followed by the item column: an item of row `r`
    // is the child index `values.len() + r`.
    let values =
        concat(&[list.values().as_ref(), item.as_ref()]).map_err(|e| internal(e.to_string()))?;
    let base = list.values().len();
    let mut rows = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if list.is_null(row) || pos.is_some_and(|p| p.is_null(row)) {
            rows.push(None);
            continue;
        }
        let (start, len) = span(list, row);
        let existing = (start..start + len)
            .map(|i| index(i).map(Some))
            .collect::<Result<Vec<_>, _>>()?;
        let new = Some(index(base + row)?);
        let pieces = match edit {
            Edit::Append => [existing, vec![new]].concat(),
            Edit::Prepend => [vec![new], existing].concat(),
            _ => {
                let p = i64::from(pos.map_or(0, |p| p.value(row)));
                if p == 0 {
                    return Err(InvalidIndexOfZero {}.into());
                }
                let len = i64::try_from(len).unwrap_or(i64::MAX);
                let at = if p > 0 { p - 1 } else { len + p + 1 };
                // The result length, known before any padding is allocated.
                let total = if at < 0 {
                    len + 1 - at
                } else {
                    len.max(at) + 1
                };
                if usize::try_from(total).map_or(true, |t| t > MAX_ARRAY_LENGTH) {
                    return Err(too_long(name, total));
                }
                if at < 0 {
                    // Reaches past the front: the item, NULL padding, the array.
                    let pad = usize::try_from(-at).unwrap_or(0);
                    [vec![new], vec![None; pad], existing].concat()
                } else if at <= len {
                    let at = usize::try_from(at).unwrap_or(0);
                    [existing[..at].to_vec(), vec![new], existing[at..].to_vec()].concat()
                } else {
                    let pad = usize::try_from(at - len).unwrap_or(0);
                    [existing, vec![None; pad], vec![new]].concat()
                }
            }
        };
        rows.push(Some(pieces));
    }
    rebuild(field, &values, &rows, name, MAX_ARRAY_LENGTH)
}

/// `array_compact(arr)`: each array without its NULL elements.
fn compact(x: &Inputs<'_>, field: &arrow_schema::FieldRef) -> Result<ArrayRef, KrishivError> {
    let c = x.column(0)?;
    let list = list_of(&c, "array_compact")?;
    let values = list.values();
    // An untyped NULL child has no validity buffer: all its slots are NULL.
    let nulls = values.logical_nulls();
    let mut rows = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if list.is_null(row) {
            rows.push(None);
            continue;
        }
        let (start, len) = span(list, row);
        let mut kept = Vec::new();
        for i in start..start + len {
            if nulls.as_ref().is_none_or(|n| n.is_valid(i)) {
                kept.push(Some(index(i)?));
            }
        }
        rows.push(Some(kept));
    }
    let out = rebuild(field, values, &rows, "array_compact", MAX_ARRAY_LENGTH)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use arrow_array::Int32Array;

    use super::*;

    /// A result longer than the limit is `COLLECTION_SIZE_LIMIT_EXCEEDED.FUNCTION`
    /// naming the function, the length and the limit.
    #[test]
    fn an_array_past_the_limit_is_refused() {
        let field = Arc::new(arrow_schema::Field::new("element", DataType::Int32, true));
        let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let rows = vec![Some(vec![Some(0), Some(1), Some(2)])];
        let err = rebuild(&field, &values, &rows, "array_insert", 2).unwrap_err();
        assert_eq!(
            err.error_class().name(),
            "COLLECTION_SIZE_LIMIT_EXCEEDED.FUNCTION"
        );
        let message = err.to_string();
        assert!(
            message.contains("array_insert") && message.contains('3') && message.contains('2'),
            "{message}"
        );
        assert!(rebuild(&field, &values, &rows, "array_insert", 3).is_ok());
    }
}
