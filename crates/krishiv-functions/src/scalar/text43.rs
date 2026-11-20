// SPDX-License-Identifier: Apache-2.0
//! The text functions new in Spark 4.3: `to_base32`, `from_base32` and `jaro_winkler_similarity`, on
//! `krishiv_kernels::string`. The operands are plain strings or binaries after the analyzer's casts; none depends on
//! a conf, a collation (the comparison is binary) or the session.

use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use krishiv_kernels::string;

use super::string::{shape, utf8};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["to_base32", "from_base32", "jaro_winkler_similarity"];

/// Binds the 4.3 text function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let binding = match (listed, shape(args).as_str()) {
        ("to_base32", "B") => Binding::new(DataType::Utf8, |x| {
            Ok(utf8(string::to_base32(&x.binaries(0)?)?))
        }),
        ("from_base32", "S") => Binding::new(DataType::Binary, |x| {
            Ok(Arc::new(string::from_base32(&x.strings(0)?)?) as ArrayRef)
        }),
        ("jaro_winkler_similarity", "SS") => Binding::new(DataType::Float64, |x| {
            let (l, r) = (x.str(0)?, x.str(1)?);
            Ok(Arc::new(string::jaro_winkler_similarity(l.operand(), r.operand())?) as ArrayRef)
        }),
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, Ok(binding)))
}
