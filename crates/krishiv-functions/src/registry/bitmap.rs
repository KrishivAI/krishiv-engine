// SPDX-License-Identifier: Apache-2.0
//! The bitmap scalar functions (P3-T086): `bitmap_bucket_number(bigint)`, `bitmap_bit_position(bigint)` and
//! `bitmap_count(binary)`. Each returns a `bigint` that is NULL when its argument is. The aggregates
//! (`bitmap_construct_agg`, `bitmap_or_agg`, `bitmap_and_agg`) live in `krishiv-exec`.

use krishiv_common::error::KrishivError;
use krishiv_types::DataType;

use super::defs::{Spec, exact, scalar};
use crate::signature::{ArgType, CallTypes, InputType, Inputs};

fn result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Long, c.any_nullable()))
}

/// The signatures of this family.
pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(
        &["bitmap_bucket_number", "bitmap_bit_position"],
        &Inputs::Expects(vec![exact(DataType::Long)]),
        result,
    );
    out.extend(scalar(
        &["bitmap_count"],
        &Inputs::Expects(vec![InputType::Strict(DataType::Binary)]),
        result,
    ));
    out
}
