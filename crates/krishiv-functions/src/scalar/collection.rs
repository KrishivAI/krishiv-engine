// SPDX-License-Identifier: Apache-2.0
//! The collection size function: `size` counts the elements of an array or
//! map (Spark's `Size`). The `map` constructor lives in the map family.
//!
//! `split` lives with the regular-expression family, next to the kernel it
//! shares (see `regex`).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int32Array};
use arrow_schema::DataType;

use super::args::{Inputs, internal};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["size"];

/// Binds the collection function `name` to `args`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match listed {
        "size" => {
            if types.len() != 1
                || !matches!(
                    types[0],
                    DataType::List(_) | DataType::Map(_, _) | DataType::Null
                )
            {
                return Some((listed, Err(bad_types(listed, args))));
            }
            let ansi = opts.ansi;
            Binding::new(DataType::Int32, move |x: &Inputs<'_>| size(x, ansi))
        }
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, Ok(binding)))
}

/// `size(arg)`: the number of elements of the array or map, NULL for NULL
/// input with ANSI on and `-1` with ANSI off (Spark's `legacySizeOfNull`
/// default under ANSI gating).
fn size(x: &Inputs<'_>, ansi: bool) -> Result<ArrayRef, krishiv_common::error::KrishivError> {
    let c = x.column(0)?;
    let mut out = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        // An untyped NULL has no validity buffer: every row is NULL.
        if c.data_type() == &DataType::Null || c.is_null(row) {
            out.push(if ansi { None } else { Some(-1) });
            continue;
        }
        let n = if let Some(list) = c.as_list_opt::<i32>() {
            list.value_length(row)
        } else if let Some(list) = c.as_list_opt::<i64>() {
            let len = list.value_length(row);
            i32::try_from(len).unwrap_or(i32::MAX)
        } else if let Some(map) = c.as_map_opt() {
            map.value_length(row)
        } else {
            return Err(internal(format!("size of {}", c.data_type())));
        };
        out.push(Some(n));
    }
    Ok(Arc::new(Int32Array::from(out)) as ArrayRef)
}

#[cfg(test)]
mod tests {

    use super::*;

    /// `size` of a non-collection does not bind.
    #[test]
    fn size_of_an_int_does_not_bind() {
        let args = vec![BindArg::from(DataType::Int32)];
        let (listed, binding) = bind("size", &args, &BindOptions::new()).unwrap();
        assert_eq!(listed, "size");
        assert!(binding.is_err());
    }

    /// An unlisted name is not this family's to bind.
    #[test]
    fn other_names_do_not_bind() {
        assert!(
            bind(
                "transform",
                &[BindArg::from(DataType::Int32)],
                &BindOptions::new()
            )
            .is_none()
        );
    }
}
