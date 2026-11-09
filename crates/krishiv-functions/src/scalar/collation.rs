// SPDX-License-Identifier: Apache-2.0
//! `__collation_key(value, type)`: the bytes operators compare in place of a collated value
//! (ADR 0043, design `P2-collations.md` section 3.3). Only the analyzer and the optimizer build
//! it: `type` is the Spark type of `value` as a literal, which names the collation of every
//! string leaf (the Arrow carrier is a plain string).

use arrow_array::Array;
use arrow_schema::DataType;
use krishiv_kernels::string::{collation_key, collation_key_type};

use super::args::{Inputs, internal};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["__collation_key"];

pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    if name != "__collation_key" {
        return None;
    }
    let bound = (|| {
        let [value, BindArg::Literal(t)] = args else {
            return Err(bad_types(name, args));
        };
        let spark = t
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .filter(|s| s.len() == 1 && !s.is_null(0))
            .map(|s| s.value(0).to_owned())
            .ok_or_else(|| internal("__collation_key: the type is no literal".to_owned()))?;
        let spark = krishiv_types::parse_data_type(&spark)?;
        let ty: DataType = collation_key_type(&value.data_type());
        Ok(Binding::new(ty, move |x: &Inputs<'_>| {
            collation_key(&spark, &x.column(0)?).map_err(|e| e.error().clone())
        }))
    })();
    Some(("__collation_key", bound))
}
