// SPDX-License-Identifier: Apache-2.0
//! The HLL scalar functions (P3-T086): `hll_sketch_estimate(binary)` is a `bigint` and
//! `hll_union(a, b, allowDifferentLgConfigK)` a `binary` (the analyzer fills the omitted flag); each is NULL
//! when an argument is. The aggregates (`hll_sketch_agg`, `hll_union_agg`) live in `defs/aggregates.rs`.

use krishiv_common::error::KrishivError;
use krishiv_types::DataType;

use super::defs::{Spec, scalar};
use crate::signature::{ArgType, CallTypes, InputType, Inputs};

fn estimate(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Long, c.any_nullable()))
}

fn union(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Binary, c.any_nullable()))
}

/// The signatures of this family.
pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(
        &["hll_sketch_estimate"],
        &Inputs::Expects(vec![InputType::Strict(DataType::Binary)]),
        estimate,
    );
    out.extend(scalar(
        &["hll_union"],
        &Inputs::Expects(vec![
            InputType::Strict(DataType::Binary),
            InputType::Strict(DataType::Binary),
            InputType::Strict(DataType::Boolean),
        ]),
        union,
    ));
    out
}
