// SPDX-License-Identifier: Apache-2.0
//! The Theta sketch scalar functions (P3-T086, Spark 4.1+): `theta_sketch_estimate(binary)` is a `bigint`,
//! `theta_union(a, b, lgNomEntries)` (the analyzer fills the omitted size), `theta_intersection(a, b)` and
//! `theta_difference(a, b)` are a `binary`; each is NULL when an argument is. The aggregates (`theta_sketch_agg`,
//! `theta_union_agg`, `theta_intersection_agg`) live in `defs/aggregates.rs`.

use krishiv_common::error::KrishivError;
use krishiv_types::DataType;

use super::defs::{Spec, scalar};
use crate::signature::{ArgType, CallTypes, InputType, Inputs};

fn estimate(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Long, c.any_nullable()))
}

fn binary(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Binary, c.any_nullable()))
}

/// The signatures of this family.
pub(super) fn specs() -> Vec<Spec> {
    let bin = || InputType::Strict(DataType::Binary);
    let mut out = scalar(
        &["theta_sketch_estimate"],
        &Inputs::Expects(vec![bin()]),
        estimate,
    );
    out.extend(scalar(
        &["theta_union"],
        &Inputs::Expects(vec![bin(), bin(), InputType::Strict(DataType::Integer)]),
        binary,
    ));
    out.extend(scalar(
        &["theta_intersection", "theta_difference"],
        &Inputs::Expects(vec![bin(), bin()]),
        binary,
    ));
    out
}
