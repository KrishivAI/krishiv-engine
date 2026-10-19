// SPDX-License-Identifier: Apache-2.0
//! Functions of the session: `current_user()` and its aliases `user()` and
//! `session_user()`, the principal the session was opened for.

use std::sync::Arc;

use arrow_array::{ArrayRef, StringArray};
use arrow_schema::DataType;

use super::args::Inputs;
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["current_user", "session_user", "user"];

/// Binds the session function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    if !args.is_empty() {
        return Some((listed, Err(bad_types(listed, args))));
    }
    let user = Arc::clone(&opts.user);
    let binding = Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
        Ok(Arc::new(StringArray::from(vec![&*user; x.rows()])) as ArrayRef)
    });
    Some((listed, Ok(binding)))
}
