// SPDX-License-Identifier: Apache-2.0
//! The `array(...)` constructor: one list per row holding each argument's
//! value (Spark's `CreateArray`). The analyzer casts the arguments to
//! their common type first; the kernel interleaves them (see
//! `krishiv_kernels::list::create`).

use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use krishiv_kernels::list;

use super::args::Inputs;
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["array"];

/// `array(a, b, ...)`: the arguments share one type (checked here, as the
/// analyzer's common cast may leave an untyped `NULL` they all share).
/// No arguments is an empty list per row (Spark's `array()`).
fn array(args: &[BindArg]) -> Option<Binding> {
    let first = args
        .first()
        .map(BindArg::data_type)
        .unwrap_or(DataType::Null);
    if args.iter().any(|a| a.data_type() != first) {
        return None;
    }
    let contains_null = args.iter().any(BindArg::nullable);
    let return_type = DataType::List(Arc::new(arrow_schema::Field::new(
        "element",
        first,
        contains_null,
    )));
    let n = args.len();
    Some(Binding::new(return_type, move |x: &Inputs<'_>| {
        let mut args = Vec::with_capacity(n);
        for i in 0..n {
            args.push(x.column(i)?);
        }
        Ok(Arc::new(list::create(&args, x.rows(), contains_null)?) as ArrayRef)
    }))
}

/// Binds the constructor.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let _ = opts;
    let listed = listed(NAMES, name)?;
    match array(args) {
        Some(b) => Some((listed, Ok(b))),
        None => Some((listed, Err(bad_types(listed, args)))),
    }
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::ErrorClass;

    use super::*;

    /// Arguments of different types do not bind: the analyzer casts to a
    /// common type first, so a mismatch here is a type error.
    #[test]
    fn mismatched_argument_types_do_not_bind() {
        let args = vec![
            BindArg::from(arrow_schema::DataType::Int32),
            BindArg::from(arrow_schema::DataType::Utf8),
        ];
        let (listed_name, binding) = bind("array", &args, &BindOptions::new()).unwrap();
        assert_eq!(listed_name, "array");
        let Err(err) = binding else {
            panic!("mismatched array argument types bound");
        };
        assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
        assert!(err.to_string().contains("array("), "{err}");
    }

    /// An unlisted name is not this family's to bind.
    #[test]
    fn other_names_do_not_bind() {
        assert!(
            bind(
                "transform",
                &[BindArg::from(arrow_schema::DataType::Int32)],
                &BindOptions::new()
            )
            .is_none()
        );
    }
}
