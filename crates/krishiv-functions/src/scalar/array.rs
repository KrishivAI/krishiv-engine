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
