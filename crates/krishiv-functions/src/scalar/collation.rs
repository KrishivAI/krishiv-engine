// SPDX-License-Identifier: Apache-2.0
//! `__collation_key(value, type)`: the bytes operators compare in place of a collated value
//! (ADR 0043, design `P2-collations.md` section 3.3). Only the analyzer and the optimizer build
//! it: `type` is the Spark type of `value` as a literal, which names the collation of every
//! string leaf (the Arrow carrier is a plain string).

use std::sync::Arc;

use arrow_array::builder::StringBuilder;
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use krishiv_common::error::KrishivError;
use krishiv_kernels::string::{
    collation_key, collation_key_type, keyed_map, keyed_map_type, push_collation_key,
};
use krishiv_types::Collation;

use super::args::{Inputs, internal, to_utf8};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "__collation_key",
    "__collation_map_keys",
    "__collation_greatest",
    "__collation_least",
];

/// The first literal argument as a Spark type.
fn spark_type(t: &BindArg) -> Result<krishiv_types::DataType, KrishivError> {
    let BindArg::Literal(t) = t else {
        return Err(internal("a collation key's type is no literal".to_owned()));
    };
    let text = to_utf8(t.as_ref())?;
    if text.len() != 1 || text.is_null(0) {
        return Err(internal("a collation key's type is no string".to_owned()));
    }
    krishiv_types::parse_data_type(text.value(0))
}

/// The row-wise pick of `greatest` / `least` under a collation: NULLs are skipped and the first
/// of equal keys stays (Spark's `Greatest` / `Least` compare with `>` / `<` only).
fn pick(x: &Inputs<'_>, collation: Collation, greatest: bool) -> Result<ArrayRef, KrishivError> {
    let (a, b) = (x.strings(1)?, x.strings(2)?);
    let mut out = StringBuilder::new();
    let (mut ka, mut kb) = (Vec::new(), Vec::new());
    for row in 0..x.rows() {
        let chosen = match (a.is_null(row), b.is_null(row)) {
            (true, true) => None,
            (true, false) => Some(b.value(row)),
            (false, true) => Some(a.value(row)),
            (false, false) => {
                ka.clear();
                kb.clear();
                push_collation_key(collation, a.value(row), &mut ka)
                    .map_err(|e| e.error().clone())?;
                push_collation_key(collation, b.value(row), &mut kb)
                    .map_err(|e| e.error().clone())?;
                let takes_b = if greatest { kb > ka } else { kb < ka };
                Some(if takes_b { b.value(row) } else { a.value(row) })
            }
        };
        out.append_option(chosen);
    }
    Ok(Arc::new(out.finish()))
}

pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let bound = (|| match (listed, args) {
        ("__collation_key", [value, t]) => {
            let spark = spark_type(t)?;
            let ty: DataType = collation_key_type(&value.data_type());
            Ok(Binding::new(ty, move |x: &Inputs<'_>| {
                collation_key(&spark, &x.column(0)?).map_err(|e| e.error().clone())
            }))
        }
        ("__collation_map_keys", [value, t]) => {
            let spark = spark_type(t)?;
            let ty = keyed_map_type(&value.data_type());
            Ok(Binding::new(ty, move |x: &Inputs<'_>| {
                keyed_map(&spark, &x.column(0)?).map_err(|e| e.error().clone())
            }))
        }
        ("__collation_greatest" | "__collation_least", [t, _, _]) => {
            let krishiv_types::DataType::CollatedString(c) = spark_type(t)? else {
                return Err(bad_types(listed, args));
            };
            let greatest = listed == "__collation_greatest";
            Ok(Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
                pick(x, c, greatest)
            }))
        }
        _ => Err(bad_types(listed, args)),
    })();
    Some((listed, bound))
}
