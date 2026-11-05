// SPDX-License-Identifier: Apache-2.0
//! `aes_encrypt`, `aes_decrypt` and `try_aes_decrypt` (ADR 0042). Spark takes
//! `(expr, key[, mode[, padding[, iv[, aad]]]])` for encryption and
//! `(expr, key[, mode[, padding[, aad]]])` for decryption: `expr`, `key`, `iv`
//! and `aad` are `binary` (a string is cast), `mode` and `padding` are
//! strings. The result is a nullable `binary`.

use krishiv_common::error::KrishivError;
use krishiv_types::DataType;

use super::defs::{Spec, scalar};
use crate::signature::{ArgType, CallTypes, InputBuilder, InputType, Inputs};

fn binary() -> InputType {
    InputType::Exact(DataType::Binary)
}

fn string() -> InputType {
    InputType::Exact(DataType::String)
}

fn encrypt_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    [binary(), binary(), string(), string(), binary(), binary()]
        .into_iter()
        .take(c.args().len())
        .collect()
}

fn decrypt_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    [binary(), binary(), string(), string(), binary()]
        .into_iter()
        .take(c.args().len())
        .collect()
}

fn result(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Binary, true))
}

/// The signatures of this family.
pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(
        &["aes_encrypt"],
        &Inputs::Built(InputBuilder(encrypt_inputs)),
        result,
    );
    out.extend(scalar(
        &["aes_decrypt", "try_aes_decrypt"],
        &Inputs::Built(InputBuilder(decrypt_inputs)),
        result,
    ));
    out
}
