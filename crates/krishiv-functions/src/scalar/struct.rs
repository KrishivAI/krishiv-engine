// SPDX-License-Identifier: Apache-2.0
//! Struct functions (P2-T030 kernels): `named_struct` and `struct` pack their
//! arguments into an Arrow `StructArray`.
//!
//! `named_struct` reads field names from its folded `STRING` name arguments
//! (literals at bind time; the analyzer rewrites foldable names to literals,
//! so the binding always sees them); `struct` names fields `col1`, `col2`,
//! .... Field types and nullability are the bound argument types exactly, as
//! the analyzer typed them, so the kernel's fields always match. The struct
//! itself is never null.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, Field, Fields};

use super::args::{Inputs, to_utf8};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["named_struct", "struct"];

/// Binds the struct function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let nulls: Vec<bool> = args.iter().map(BindArg::nullable).collect();
    let n = types.len();
    let binding = match listed {
        "named_struct" => {
            if !n.is_multiple_of(2) {
                return Some((listed, Err(bad_types(listed, args))));
            }
            let mut names = Vec::with_capacity(n / 2);
            for i in (0..n).step_by(2) {
                let BindArg::Literal(a) = &args[i] else {
                    return Some((listed, Err(bad_types(listed, args))));
                };
                let text = match to_utf8(a.as_ref()) {
                    Ok(t) => t,
                    Err(e) => return Some((listed, Err(e))),
                };
                if text.is_null(0) {
                    return Some((listed, Err(bad_types(listed, args))));
                }
                names.push(text.value(0).to_owned());
            }
            let value_types: Vec<(DataType, bool)> = (0..n / 2)
                .map(|k| (types[2 * k + 1].clone(), nulls[2 * k + 1]))
                .collect();
            let fields: Fields = names
                .into_iter()
                .zip(value_types)
                .map(|(name, (t, nullable))| Field::new(name, t, nullable))
                .collect();
            // Field types are the bound argument types exactly (no carrier
            // conversion): the analyzer typed the same Arrow types, so the
            // kernel's fields always match.
            Binding::new(DataType::Struct(fields.clone()), move |x: &Inputs<'_>| {
                let mut children = Vec::with_capacity(n / 2);
                for k in 0..n / 2 {
                    children.push(x.column(2 * k + 1)?);
                }
                Ok(Arc::new(krishiv_kernels::r#struct::build(
                    fields.clone(),
                    children,
                    x.rows(),
                )?) as ArrayRef)
            })
        }
        "struct" => {
            let fields: Fields = types
                .iter()
                .zip(nulls.iter())
                .enumerate()
                .map(|(i, (t, nullable))| Field::new(format!("col{}", i + 1), t.clone(), *nullable))
                .collect();
            Binding::new(DataType::Struct(fields.clone()), move |x: &Inputs<'_>| {
                let mut children = Vec::with_capacity(n);
                for i in 0..n {
                    children.push(x.column(i)?);
                }
                Ok(Arc::new(krishiv_kernels::r#struct::build(
                    fields.clone(),
                    children,
                    x.rows(),
                )?) as ArrayRef)
            })
        }
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, Ok(binding)))
}
