// SPDX-License-Identifier: Apache-2.0
//! The URL functions' analysis signatures: `parse_url` and `try_parse_url`
//! take strict strings (no implicit cast: an `int` URL is refused, as
//! Spark's `ExpectsInputTypes` refuses it), while `url_encode`,
//! `url_decode` and `try_url_decode` cast any argument to a string (as
//! Spark's `ImplicitCastInputTypes` do). Every one is a nullable string,
//! whatever its arguments (Spark's `nullable = true`).

use krishiv_common::error::KrishivError;
use krishiv_types::DataType;

use super::defs::{Spec, exact, scalar};
use crate::signature::{ArgType, CallTypes, InputType, Inputs};

/// A nullable `string`, whatever the arguments.
fn string(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::String, true))
}

pub(super) fn specs() -> Vec<Spec> {
    let strict = Inputs::Expects(vec![
        InputType::Strict(DataType::String),
        InputType::Strict(DataType::String),
        InputType::Strict(DataType::String),
    ]);
    let mut out = scalar(&["parse_url", "try_parse_url"], &strict, string);
    out.extend(scalar(
        &["url_encode", "url_decode", "try_url_decode"],
        &Inputs::Expects(vec![exact(DataType::String)]),
        string,
    ));
    out
}
