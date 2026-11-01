// SPDX-License-Identifier: Apache-2.0
//! Array search functions: `array_max`, `array_min`, `array_position`,
//! `array_remove` and `array_repeat`.
//!
//! Elements compare as Spark's `RowOrdering` does
//! ([`krishiv_kernels::order::cmp_rows`]): NaN equals NaN and is the
//! greatest value, `-0.0` equals `0.0`. `array_max` and `array_min` skip
//! NULL elements and keep the first of equal ones (`array_min` of
//! `[-0.0, 0.0]` is `-0.0`, checked on a Spark 4.2.0 run). `array_position`
//! and `array_remove` never match a NULL element.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, UInt32Array, new_null_array};
use arrow_schema::{DataType, Field, FieldRef};
use arrow_select::take::take;
use krishiv_common::error::KrishivError;
use krishiv_kernels::order::cmp_rows;

use super::args::{Inputs, Prim, internal};
use super::array_edit::{MAX_ARRAY_LENGTH, index, list_of, no_memory, rebuild, span, too_long};
use super::array_set::{kernel, orders, same_value};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "array_max",
    "array_min",
    "array_position",
    "array_remove",
    "array_repeat",
];

/// Binds the function `name` to `args`.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match (listed, types.as_slice()) {
        ("array_max" | "array_min", [DataType::List(f)]) if orders(f.data_type()) => {
            let largest = listed == "array_max";
            Ok(Binding::new(
                f.data_type().clone(),
                move |x: &Inputs<'_>| extreme(x, largest),
            ))
        }
        ("array_position", [DataType::List(f), v]) if f.data_type() == v && orders(v) => {
            Ok(Binding::new(DataType::Int64, position))
        }
        ("array_remove", [t @ DataType::List(f), v]) if f.data_type() == v && orders(v) => {
            let field = Arc::clone(f);
            Ok(Binding::new(t.clone(), move |x: &Inputs<'_>| {
                remove(x, &field)
            }))
        }
        ("array_repeat", [element, DataType::Int32]) => {
            let field = Arc::new(Field::new("element", element.clone(), true));
            Ok(Binding::new(
                DataType::List(Arc::clone(&field)),
                move |x: &Inputs<'_>| repeat(x, &field),
            ))
        }
        _ => Err(bad_types(listed, args)),
    };
    Some((listed, binding))
}

/// `array_max(arr)` / `array_min(arr)`.
fn extreme(x: &Inputs<'_>, largest: bool) -> Result<ArrayRef, KrishivError> {
    let c = x.column(0)?;
    if *c.data_type() == DataType::Null {
        return Ok(new_null_array(&DataType::Null, x.rows()));
    }
    let list = list_of(&c, if largest { "array_max" } else { "array_min" })?;
    let values = list.values();
    let nulls = values.logical_nulls();
    let mut picks: Vec<Option<u32>> = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if list.is_null(row) {
            picks.push(None);
            continue;
        }
        let (start, len) = span(list, row);
        let mut best: Option<usize> = None;
        for i in start..start + len {
            if nulls.as_ref().is_some_and(|n| n.is_null(i)) {
                continue;
            }
            let replace = match best {
                None => true,
                Some(b) => {
                    let ord = cmp_rows(values.as_ref(), i, values.as_ref(), b).map_err(kernel)?;
                    if largest {
                        ord == Ordering::Greater
                    } else {
                        ord == Ordering::Less
                    }
                }
            };
            if replace {
                best = Some(i);
            }
        }
        picks.push(best.map(index).transpose()?);
    }
    if *values.data_type() == DataType::Null {
        return Ok(new_null_array(&DataType::Null, x.rows()));
    }
    take(values.as_ref(), &UInt32Array::from(picks), None).map_err(kernel)
}

/// `array_position(arr, value)`.
fn position(x: &Inputs<'_>) -> Result<ArrayRef, KrishivError> {
    let c = x.column(0)?;
    let list = list_of(&c, "array_position")?;
    let value = x.column(1)?;
    let (elements, wanted) = (list.values().logical_nulls(), value.logical_nulls());
    let values = list.values();
    let mut out: Vec<Option<i64>> = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if list.is_null(row) || wanted.as_ref().is_some_and(|n| n.is_null(row)) {
            out.push(None);
            continue;
        }
        let (start, len) = span(list, row);
        let mut found = 0_i64;
        for i in 0..len {
            let at = start + i;
            if elements.as_ref().is_some_and(|n| n.is_null(at)) {
                continue;
            }
            if same_value(values.as_ref(), at, value.as_ref(), row)? {
                found = i64::try_from(i + 1).map_err(|_| internal("index past 2^63".to_owned()))?;
                break;
            }
        }
        out.push(Some(found));
    }
    Ok(Arc::new(Int64Array::from(out)))
}

/// `array_remove(arr, value)`.
fn remove(x: &Inputs<'_>, field: &FieldRef) -> Result<ArrayRef, KrishivError> {
    let c = x.column(0)?;
    let list = list_of(&c, "array_remove")?;
    let value = x.column(1)?;
    let (elements, wanted) = (list.values().logical_nulls(), value.logical_nulls());
    let values = list.values();
    let mut rows = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if list.is_null(row) || wanted.as_ref().is_some_and(|n| n.is_null(row)) {
            rows.push(None);
            continue;
        }
        let (start, len) = span(list, row);
        let mut kept = Vec::with_capacity(len);
        for at in start..start + len {
            let is_null = elements.as_ref().is_some_and(|n| n.is_null(at));
            if is_null || !same_value(values.as_ref(), at, value.as_ref(), row)? {
                kept.push(Some(index(at)?));
            }
        }
        rows.push(Some(kept));
    }
    rebuild(field, values, &rows, "array_remove", MAX_ARRAY_LENGTH)
}

/// `array_repeat(element, count)`.
fn repeat(x: &Inputs<'_>, field: &FieldRef) -> Result<ArrayRef, KrishivError> {
    let element = x.column(0)?;
    let count = x.int(1)?;
    let count_at = |row: usize| match &count {
        Prim::Array(a) => a.is_valid(row).then(|| a.value(row)),
        Prim::Scalar(v) => *v,
    };
    let mut rows = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        let Some(n) = count_at(row) else {
            rows.push(None);
            continue;
        };
        let n = usize::try_from(n).unwrap_or(0);
        if n > MAX_ARRAY_LENGTH {
            return Err(too_long("array_repeat", n));
        }
        let mut picks = Vec::new();
        picks.try_reserve_exact(n).map_err(|_| no_memory(n))?;
        picks.resize(n, Some(index(row)?));
        rows.push(Some(picks));
    }
    if *element.data_type() == DataType::Null {
        let nulls = new_null_array(&DataType::Null, x.rows());
        return rebuild(field, &nulls, &rows, "array_repeat", MAX_ARRAY_LENGTH);
    }
    rebuild(field, &element, &rows, "array_repeat", MAX_ARRAY_LENGTH)
}
