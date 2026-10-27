// SPDX-License-Identifier: Apache-2.0
//! The collection constructor and size functions: `map` packs its key and
//! value arguments into a map per row (Spark's `CreateMap`), and `size`
//! counts the elements of an array or map (Spark's `Size`).
//!
//! `split` lives with the regular-expression family, next to the kernel it
//! shares (see `regex`).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int32Array};
use arrow_schema::DataType;
use krishiv_kernels::map;
use krishiv_kernels::map::entries_field;

use super::args::{Inputs, internal};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["map", "size"];

/// Binds the collection function `name` to `args`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match listed {
        "map" => {
            if !args.len().is_multiple_of(2) {
                return Some((listed, Err(bad_types(listed, args))));
            }
            let first_key = types.first().cloned().unwrap_or(DataType::Null);
            let first_value = types.get(1).cloned().unwrap_or(DataType::Null);
            // Keys share one type and values share one type (the analyzer
            // checks this too, as Spark's `CreateMap` takes no implicit
            // casts); the kernel re-checks at run time.
            if types.iter().step_by(2).any(|t| t != &first_key)
                || types.iter().skip(1).step_by(2).any(|t| t != &first_value)
            {
                return Some((listed, Err(bad_types(listed, args))));
            }
            let value_nullable = args.iter().skip(1).step_by(2).any(BindArg::nullable);
            let last_win = opts.map_key_dedup_last_win;
            let pairs = args.len() / 2;
            let return_type = DataType::Map(
                Arc::new(entries_field(&first_key, &first_value, value_nullable)),
                false,
            );
            Binding::new(return_type, move |x: &Inputs<'_>| {
                let mut key_cols = Vec::with_capacity(pairs);
                let mut value_cols = Vec::with_capacity(pairs);
                for i in 0..pairs {
                    key_cols.push(x.column(2 * i)?);
                    value_cols.push(x.column(2 * i + 1)?);
                }
                Ok(Arc::new(map::create(
                    &key_cols,
                    &value_cols,
                    x.rows(),
                    value_nullable,
                    last_win,
                )?) as ArrayRef)
            })
        }
        "size" => {
            if types.len() != 1 || !matches!(types[0], DataType::List(_) | DataType::Map(_, _)) {
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
        if c.is_null(row) {
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
    use krishiv_common::error::ErrorClass;

    use super::*;

    /// Arguments that do not belong together do not bind.
    #[test]
    fn mismatched_map_types_do_not_bind() {
        let args = vec![
            BindArg::from(DataType::Utf8),
            BindArg::from(DataType::Int32),
            BindArg::from(DataType::Int32),
            BindArg::from(DataType::Int32),
        ];
        let (listed, binding) = bind("map", &args, &BindOptions::new()).unwrap();
        assert_eq!(listed, "map");
        let Err(err) = binding else {
            panic!("mismatched map key types bound");
        };
        assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
        assert!(err.to_string().contains("map("), "{err}");
    }

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
