// SPDX-License-Identifier: Apache-2.0
//! The conditionals' analysis signatures. They run in the expression
//! evaluator, which evaluates only the arguments a row needs
//! ([`Execution::Evaluator`]); the registry types them as Spark rewrites
//! them: `ifnull`/`nvl`/`zeroifnull` as `coalesce`, `nvl2` as `if`,
//! `nullif`/`nullifzero` as `if(a = b, NULL, a)`.

use krishiv_common::error::KrishivError;
use krishiv_types::DataType;

use super::Execution;
use super::defs::{Spec, exact};
use crate::signature::{ArgType, CallTypes, FunctionKind, InputBuilder, InputType, Inputs};

/// `coalesce`: the arguments share one type after their casts; NULL only
/// when every argument may be.
fn coalesce(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType {
        data_type: c.args().iter().find_map(|t| t.data_type.clone()),
        nullable: c.args().iter().all(|t| t.nullable),
    })
}

/// `if(c, a, b)`, `nvl2(x, a, b)`: the branches' common type; NULL when a
/// branch may be.
fn branches(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let branches = c.args().get(1..).unwrap_or_default();
    Ok(ArgType {
        data_type: branches.iter().find_map(|t| t.data_type.clone()),
        nullable: branches.iter().any(|t| t.nullable),
    })
}

/// `nullif(a, b)`: `a`'s type, NULL where `a = b`.
fn nullif(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType {
        data_type: c.data_type(0).cloned(),
        nullable: true,
    })
}

/// `zeroifnull(x)`: `x`'s type (met with `int`), never NULL.
fn zeroifnull(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType {
        data_type: c.data_type(0).cloned(),
        nullable: false,
    })
}

/// `nanvl(a, b)`: `float` for two floats, else `double`.
fn nanvl_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let floats = c
        .args()
        .iter()
        .all(|a| a.data_type == Some(DataType::Float));
    vec![exact(if floats {
        DataType::Float
    } else {
        DataType::Double
    })]
}

fn nanvl(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType {
        data_type: c.data_type(0).cloned(),
        nullable: c.any_nullable(),
    })
}

fn spec(
    name: &'static str,
    inputs: Inputs,
    result: fn(&CallTypes<'_>) -> Result<ArgType, KrishivError>,
) -> Spec {
    Spec {
        name,
        kind: FunctionKind::Scalar,
        inputs,
        result,
        execution: Execution::Evaluator,
    }
}

pub(super) fn specs() -> Vec<Spec> {
    vec![
        spec("coalesce", Inputs::Common, coalesce),
        spec("ifnull", Inputs::Common, coalesce),
        spec("nvl", Inputs::Common, coalesce),
        spec(
            "if",
            Inputs::Leading(vec![InputType::Strict(DataType::Boolean)]),
            branches,
        ),
        spec("nvl2", Inputs::Leading(vec![InputType::Any]), branches),
        spec("nullif", Inputs::Comparable { zero: false }, nullif),
        spec("nullifzero", Inputs::Comparable { zero: true }, nullif),
        spec("zeroifnull", Inputs::CommonWithZero, zeroifnull),
        spec("nanvl", Inputs::Built(InputBuilder(nanvl_inputs)), nanvl),
    ]
}
