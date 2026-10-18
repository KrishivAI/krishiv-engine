// SPDX-License-Identifier: Apache-2.0
//! The hash functions' analysis signatures: `hash` and `xxhash64` take any
//! columns and are never NULL; the digests take `binary` (a string is cast
//! to it).

use krishiv_common::error::KrishivError;
use krishiv_types::DataType;

use super::defs::{Spec, exact, scalar, typed};
use crate::signature::{ArgType, CallTypes, Inputs};

fn hash(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Integer, false))
}

fn xxhash64(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Long, false))
}

fn digest(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::String, c)
}

fn crc32(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Long, c)
}

/// `sha2` is NULL for a bit length it does not know.
fn sha2(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::String, true))
}

pub(super) fn specs() -> Vec<Spec> {
    let binary = Inputs::Expects(vec![exact(DataType::Binary)]);
    let mut out = scalar(&["hash"], &Inputs::Any, hash);
    out.extend(scalar(&["xxhash64"], &Inputs::Any, xxhash64));
    out.extend(scalar(&["md5", "sha", "sha1"], &binary, digest));
    out.extend(scalar(&["crc32"], &binary, crc32));
    out.extend(scalar(
        &["sha2"],
        &Inputs::Expects(vec![exact(DataType::Binary), exact(DataType::Integer)]),
        sha2,
    ));
    out
}
