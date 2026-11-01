// SPDX-License-Identifier: Apache-2.0
//! Array reshaping: `flatten`, `slice` and `array_join`.
//!
//! `flatten` is NULL for a NULL array or when any inner array is NULL.
//! `slice(arr, start, length)` counts `start` from 1, or from the end when
//! negative; a start before the first element or past the last gives an
//! empty array, `start = 0` and a negative length are errors. `array_join`
//! skips NULL elements unless a replacement is given, and is NULL when the
//! array, delimiter or replacement is NULL (Spark 4.2.0 run).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int32Array, StringArray};
use arrow_schema::{DataType, Field};
use krishiv_common::error::{
    InvalidParameterValueLength, InvalidParameterValueStart, KrishivError,
};

use super::args::{Inputs, internal, to_utf8};
use super::array_edit::{MAX_ARRAY_LENGTH, index, list_of, rebuild, span, too_long};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["array_join", "flatten", "slice"];

/// Binds the function `name` to `args`.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match (listed, types.as_slice()) {
        ("flatten", [DataType::List(outer)]) => match outer.data_type() {
            DataType::List(inner) => {
                let field = Arc::new(Field::new(
                    inner.name(),
                    inner.data_type().clone(),
                    outer.is_nullable() || inner.is_nullable(),
                ));
                Ok(Binding::new(
                    DataType::List(Arc::clone(&field)),
                    move |x: &Inputs<'_>| flatten(x, &field),
                ))
            }
            _ => Err(bad_types(listed, args)),
        },
        ("slice", [t @ DataType::List(f), DataType::Int32, DataType::Int32]) => {
            let field = Arc::clone(f);
            Ok(Binding::new(t.clone(), move |x: &Inputs<'_>| {
                slice(x, &field)
            }))
        }
        ("array_join", [DataType::List(f), d, ..])
            if is_text(f.data_type()) && is_text(d) && types.len() <= 3 =>
        {
            let replaced = types.len() == 3;
            Ok(Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
                join(x, replaced)
            }))
        }
        _ => Err(bad_types(listed, args)),
    };
    Some((listed, binding))
}

/// Whether `t` is a string type.
fn is_text(t: &DataType) -> bool {
    matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

/// `flatten(arr)`.
fn flatten(x: &Inputs<'_>, field: &Arc<Field>) -> Result<ArrayRef, KrishivError> {
    let c = x.column(0)?;
    let outer = list_of(&c, "flatten")?;
    let inner = outer
        .values()
        .as_list_opt::<i32>()
        .ok_or_else(|| internal("flatten of a non-nested array".to_owned()))?;
    let mut rows = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if outer.is_null(row) {
            rows.push(None);
            continue;
        }
        let (start, len) = span(outer, row);
        let mut picks: Option<Vec<Option<u32>>> = Some(Vec::new());
        for k in start..start + len {
            if inner.is_null(k) {
                picks = None;
                break;
            }
            let (first, n) = span(inner, k);
            if let Some(p) = picks.as_mut() {
                if p.len() + n > MAX_ARRAY_LENGTH {
                    return Err(too_long("flatten", p.len() + n));
                }
                for at in first..first + n {
                    p.push(Some(index(at)?));
                }
            }
        }
        rows.push(picks);
    }
    rebuild(field, inner.values(), &rows, "flatten", MAX_ARRAY_LENGTH)
}

/// The `INT` column `i`, one value per row.
fn ints(x: &Inputs<'_>, i: usize) -> Result<Int32Array, KrishivError> {
    Ok(x.column(i)?
        .as_primitive_opt::<arrow_array::types::Int32Type>()
        .ok_or_else(|| internal("slice with a non-INT bound".to_owned()))?
        .clone())
}

/// `slice(arr, start, length)`.
fn slice(x: &Inputs<'_>, field: &Arc<Field>) -> Result<ArrayRef, KrishivError> {
    let c = x.column(0)?;
    let list = list_of(&c, "slice")?;
    let (starts, lengths) = (ints(x, 1)?, ints(x, 2)?);
    let mut rows = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if list.is_null(row) || starts.is_null(row) || lengths.is_null(row) {
            rows.push(None);
            continue;
        }
        let (start, length) = (starts.value(row), lengths.value(row));
        if start == 0 {
            return Err(InvalidParameterValueStart {
                function_name: "`slice`".to_owned(),
                parameter: "`start`".to_owned(),
            }
            .into());
        }
        if length < 0 {
            return Err(InvalidParameterValueLength {
                function_name: "`slice`".to_owned(),
                parameter: "`length`".to_owned(),
                length: length.to_string(),
            }
            .into());
        }
        let (first, size) = span(list, row);
        let size_i = i64::try_from(size).map_err(|_| internal("array past 2^63".to_owned()))?;
        let from = if start > 0 {
            i64::from(start) - 1
        } else {
            size_i + i64::from(start)
        };
        if from < 0 || from >= size_i {
            rows.push(Some(Vec::new()));
            continue;
        }
        let to = size_i.min(from + i64::from(length));
        let (from, to) = (
            usize::try_from(from).unwrap_or(0),
            usize::try_from(to).unwrap_or(0),
        );
        let mut picks = Vec::with_capacity(to - from);
        for at in first + from..first + to {
            picks.push(Some(index(at)?));
        }
        rows.push(Some(picks));
    }
    rebuild(field, list.values(), &rows, "slice", MAX_ARRAY_LENGTH)
}

/// `array_join(arr, delimiter[, replacement])`.
fn join(x: &Inputs<'_>, replaced: bool) -> Result<ArrayRef, KrishivError> {
    let c = x.column(0)?;
    let list = list_of(&c, "array_join")?;
    let elements = to_utf8(list.values().as_ref())?;
    let delimiters = x.strings(1)?;
    let replacements: Option<StringArray> = replaced.then(|| x.strings(2)).transpose()?;
    let mut out: Vec<Option<String>> = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        let replacement = replacements
            .as_ref()
            .map(|r| (r.is_valid(row)).then(|| r.value(row)));
        if list.is_null(row) || delimiters.is_null(row) || replacement == Some(None) {
            out.push(None);
            continue;
        }
        let (start, len) = span(list, row);
        let delimiter = delimiters.value(row);
        let mut text = String::new();
        let mut first = true;
        for at in start..start + len {
            let piece = if elements.is_valid(at) {
                elements.value(at)
            } else if let Some(Some(r)) = replacement {
                r
            } else {
                continue;
            };
            if !first {
                text.push_str(delimiter);
            }
            first = false;
            text.push_str(piece);
            if i32::try_from(text.len()).is_err() {
                return Err(too_long("array_join", text.len()));
            }
        }
        out.push(Some(text));
    }
    Ok(Arc::new(StringArray::from(out)))
}
