// SPDX-License-Identifier: Apache-2.0
//! The generator functions `explode`, `posexplode`, `inline` and their
//! `_outer` forms.
//!
//! Lowering turns a generator in a `SELECT` item, a `LATERAL VIEW` or a
//! `FROM` clause into a `Generate` node and types it there, so these
//! signatures are reached only by a generator the lowering did not take: one
//! nested in an expression, or outside the select list. The analyzer rejects
//! those as Spark's `UNSUPPORTED_GENERATOR.*`; the result type here only has
//! to be total.

use krishiv_common::error::KrishivError;
use krishiv_types::{DataType, StructField, StructType};

use super::Execution;
use super::defs::Spec;
use crate::signature::{ArgType, CallTypes, FunctionKind, Inputs};

/// The generator functions, one spec per name.
pub(super) fn specs() -> Vec<Spec> {
    [
        "explode",
        "explode_outer",
        "posexplode",
        "posexplode_outer",
        "inline",
        "inline_outer",
    ]
    .into_iter()
    .map(|name| Spec {
        name,
        kind: FunctionKind::Generator,
        inputs: Inputs::Any,
        result: generated,
        execution: Execution::Generator,
    })
    .collect()
}

/// One generated row: the element of an array, or the key and value of a
/// map; any other argument stands for itself.
fn generated(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let data_type = match c.data_type(0) {
        Some(DataType::Array(a)) => a.element_type().clone(),
        Some(DataType::Map(m)) => DataType::Struct(StructType::new(vec![
            StructField::new("key", m.key_type().clone(), false),
            StructField::new("value", m.value_type().clone(), true),
        ])),
        Some(other) => other.clone(),
        None => DataType::Null,
    };
    Ok(ArgType::new(data_type, true))
}
