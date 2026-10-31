// SPDX-License-Identifier: Apache-2.0
//! Element access: `array_size`, `get`, `element_at` and `try_element_at`.
//!
//! `element_at` counts from 1, negative indexes from the end; index 0 is
//! `INVALID_INDEX_OF_ZERO` (also for `try_element_at`), and an index past
//! the array is `INVALID_ARRAY_INDEX_IN_ELEMENT_AT` with ANSI on and NULL
//! otherwise (always NULL for `try_element_at`). `get` counts from 0 and is
//! NULL out of range. On a map, a missing key is NULL. A NULL array, map,
//! index or key is NULL.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, Int32Array, UInt32Array};
use arrow_schema::DataType;
use arrow_select::take::take;
use krishiv_common::error::{InvalidArrayIndexInElementAt, InvalidIndexOfZero, KrishivError};
use krishiv_kernels::map as kh;

use super::args::{Inputs, internal};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["array_size", "element_at", "get", "try_element_at"];

/// Binds the element-access function `name` to `args`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let ansi = opts.ansi;
    let binding = match (listed, types.as_slice()) {
        ("array_size", [DataType::List(_) | DataType::Null]) => {
            Ok(Binding::new(DataType::Int32, array_size))
        }
        ("get", [DataType::List(f), DataType::Int32]) => {
            let ty = f.data_type().clone();
            Ok(Binding::new(ty, move |x: &Inputs<'_>| {
                by_index(x, Access::Get, ansi)
            }))
        }
        ("element_at" | "try_element_at", [DataType::List(f), DataType::Int32]) => {
            let access = if listed == "try_element_at" {
                Access::Try
            } else {
                Access::ElementAt
            };
            let ty = f.data_type().clone();
            Ok(Binding::new(ty, move |x: &Inputs<'_>| {
                by_index(x, access, ansi)
            }))
        }
        ("element_at" | "try_element_at", [DataType::Map(entries, _), k]) => {
            match entries.data_type() {
                DataType::Struct(fields) if fields.len() == 2 && fields[0].data_type() == k => {
                    let ty = fields[1].data_type().clone();
                    Ok(Binding::new(ty, by_key))
                }
                _ => Err(bad_types(listed, args)),
            }
        }
        _ => Err(bad_types(listed, args)),
    };
    Some((listed, binding))
}

/// `array_size(arr)`: the length, NULL for a NULL array.
fn array_size(x: &Inputs<'_>) -> Result<ArrayRef, KrishivError> {
    let c = x.column(0)?;
    let mut out = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if c.data_type() == &DataType::Null || c.is_null(row) {
            out.push(None);
        } else if let Some(list) = c.as_list_opt::<i32>() {
            out.push(Some(list.value_length(row)));
        } else {
            return Err(internal(format!("array_size of {}", c.data_type())));
        }
    }
    Ok(Arc::new(Int32Array::from(out)) as ArrayRef)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    /// `get`: from 0, NULL out of range.
    Get,
    /// `element_at`: from 1 or -1, an error out of range with ANSI on.
    ElementAt,
    /// `try_element_at`: as `element_at`, NULL out of range.
    Try,
}

/// The element of each row's array at its index.
fn by_index(x: &Inputs<'_>, access: Access, ansi: bool) -> Result<ArrayRef, KrishivError> {
    let (c, idx) = (x.column(0)?, x.column(1)?);
    let list = c
        .as_list_opt::<i32>()
        .ok_or_else(|| internal(format!("an element access of {}", c.data_type())))?;
    let idx = idx.as_primitive::<Int32Type>();
    let offsets = list.value_offsets();
    let mut picked: Vec<Option<u32>> = Vec::with_capacity(x.rows());
    for (row, &start) in offsets.iter().take(x.rows()).enumerate() {
        if list.is_null(row) || idx.is_null(row) {
            picked.push(None);
            continue;
        }
        let (i, len) = (i64::from(idx.value(row)), i64::from(list.value_length(row)));
        let pos = match access {
            Access::Get => i,
            _ if i == 0 => return Err(InvalidIndexOfZero {}.into()),
            _ if i > 0 => i - 1,
            _ => len + i,
        };
        if pos < 0 || pos >= len {
            if access == Access::ElementAt && ansi {
                return Err(InvalidArrayIndexInElementAt {
                    index_value: i.to_string(),
                    array_size: len.to_string(),
                }
                .into());
            }
            picked.push(None);
            continue;
        }
        let at = i64::from(start) + pos;
        picked.push(Some(
            u32::try_from(at).map_err(|_| internal("an array past 2^32 elements".to_owned()))?,
        ));
    }
    take(list.values().as_ref(), &UInt32Array::from(picked), None)
        .map_err(|e| internal(e.to_string()))
}

/// The value of each row's map at its key (NULL when absent).
fn by_key(x: &Inputs<'_>) -> Result<ArrayRef, KrishivError> {
    let (c, key) = (x.column(0)?, x.column(1)?);
    let map = c
        .as_map_opt()
        .ok_or_else(|| internal(format!("an element access of {}", c.data_type())))?;
    let offsets = map.value_offsets();
    let mut picked: Vec<Option<u32>> = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if map.is_null(row) || key.is_null(row) {
            picked.push(None);
            continue;
        }
        let bounds = |o: i32| usize::try_from(o).unwrap_or(0);
        let (from, to) = (bounds(offsets[row]), bounds(offsets[row + 1]));
        let mut found = None;
        for p in from..to {
            if kh::keys_equal(map.keys().as_ref(), p, key.as_ref(), row)? {
                found = Some(
                    u32::try_from(p).map_err(|_| internal("a map past 2^32 entries".to_owned()))?,
                );
                break;
            }
        }
        picked.push(found);
    }
    take(map.values().as_ref(), &UInt32Array::from(picked), None)
        .map_err(|e| internal(e.to_string()))
}
