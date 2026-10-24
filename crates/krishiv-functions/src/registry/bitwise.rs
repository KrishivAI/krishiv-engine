// SPDX-License-Identifier: Apache-2.0
//! The bitwise functions' analysis signatures: the `&` `|` `^` operators
//! (Spark's `BitwiseAnd`/`Or`/`Xor`, widened like the arithmetic operators
//! but integral-only), `~` (the type of its child), the shifts (an `int`
//! or `bigint` shifted by an `int`), `bit_count` (an integral type or a
//! boolean) and `bit_get`/`getbit` (an integral type by an `int`, a
//! `tinyint`).
//!
//! Ground truth: Spark 4.2's `bitwiseExpressions.scala` and `mathExpressions.scala`
//! (`BitShiftOperation`), the `bitwise.sql` goldens and the `bitwise`
//! function-matrix family (590 cases).

use krishiv_common::error::{
    DatatypeMismatchBinaryOpDiffTypes, DatatypeMismatchUnexpectedInputType, KrishivError,
};
use krishiv_types::DataType;

use super::defs::{Spec, exact, scalar};
use crate::signature::{ArgType, CallTypes, InputBuilder, InputType, Inputs};

/// The integral rank for widening: `Byte < Short < Integer < Long`.
fn rank(t: &DataType) -> Option<u8> {
    match t {
        DataType::Byte => Some(0),
        DataType::Short => Some(1),
        DataType::Integer => Some(2),
        DataType::Long => Some(3),
        _ => None,
    }
}

fn wider(a: &DataType, b: &DataType) -> DataType {
    if rank(a) >= rank(b) {
        a.clone()
    } else {
        b.clone()
    }
}

/// Whether `t` is an integral type.
fn is_integral(t: Option<&DataType>) -> bool {
    t.is_some_and(|t| rank(t).is_some())
}

/// A type as `BINARY_OP_DIFF_TYPES` quotes it (`"VOID"` for untyped `NULL`).
fn quoted(t: Option<&DataType>) -> String {
    format!("\"{}\"", t.map_or_else(|| "VOID".to_owned(), DataType::sql))
}

/// Spark's `ordinalNumber`.
fn ordinal(i: usize) -> String {
    match i {
        0 => "first".to_owned(),
        1 => "second".to_owned(),
        2 => "third".to_owned(),
        n => format!("{}th", n + 1),
    }
}

/// `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` for argument `i` of `function`.
fn unexpected(c: &CallTypes<'_>, function: &str, i: usize, required: &str) -> KrishivError {
    let args: Vec<String> = (0..c.args().len())
        .map(|n| c.sql(n).unwrap_or_default())
        .collect();
    DatatypeMismatchUnexpectedInputType {
        sql_expr: format!("\"{function}({})\"", args.join(", ")),
        param_index: ordinal(i),
        required_type: required.to_owned(),
        input_sql: format!("\"{}\"", c.sql(i).unwrap_or_default()),
        input_type: quoted(c.data_type(i)),
    }
    .into()
}

/// `DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES` for `l symbol r`.
fn diff_types(c: &CallTypes<'_>, symbol: &str) -> KrishivError {
    DatatypeMismatchBinaryOpDiffTypes {
        sql_expr: format!(
            "\"({} {symbol} {})\"",
            c.sql(0).unwrap_or_default(),
            c.sql(1).unwrap_or_default()
        ),
        left: quoted(c.data_type(0)),
        right: quoted(c.data_type(1)),
    }
    .into()
}

// ---- `&`, `|`, `^` ----

/// The operands' common integral type (`None` when incompatible): integrals
/// widen; with ANSI a string meets anything as `bigint`; an untyped `NULL`
/// takes the other side (`int` for two nulls).
fn common_bitwise(c: &CallTypes<'_>) -> Option<DataType> {
    let (a, b) = (c.data_type(0), c.data_type(1));
    match (a, b) {
        (Some(x), Some(y)) if is_integral(Some(x)) && is_integral(Some(y)) => Some(wider(x, y)),
        (None, None) => Some(DataType::Integer),
        (None, Some(t)) | (Some(t), None) if is_integral(Some(t)) => Some(t.clone()),
        (Some(DataType::String), Some(other)) | (Some(other), Some(DataType::String))
            if c.ansi() =>
        {
            // A string meets integrals (and booleans, which the declared
            // cast then refuses) as `bigint`; a non-integral number would
            // narrow through `Exact(Long)`, so it is incompatible.
            if matches!(
                other,
                DataType::Float | DataType::Double | DataType::Decimal(_)
            ) {
                None
            } else {
                Some(DataType::Long)
            }
        }
        (Some(DataType::String), _) | (_, Some(DataType::String)) if c.ansi() => {
            Some(DataType::Long)
        }
        _ => None,
    }
}

/// `a & b`: both operands to their common integral type. Strings do not
/// meet without ANSI (and are refused without casts, so the result rule
/// still sees them); anything else is refused without casts, so the
/// result rule sees the original types.
fn bitwise_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let (a, b) = (c.data_type(0), c.data_type(1));
    let strings = matches!(a, Some(DataType::String)) || matches!(b, Some(DataType::String));
    if !c.ansi() && strings {
        return vec![InputType::Any, InputType::Any];
    }
    match common_bitwise(c) {
        Some(w) => vec![exact(w.clone()), exact(w)],
        None => vec![InputType::Any, InputType::Any],
    }
}

/// The `&`/`|`/`^` result: the operands' common integral type, or
/// `BINARY_OP_DIFF_TYPES`. The rule sees post-cast types, except pairs the
/// inputs refused without casts (legacy strings, incompatible types).
macro_rules! bitwise_result {
    ($name:ident, $symbol:literal) => {
        fn $name(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
            let (a, b) = (c.data_type(0), c.data_type(1));
            match (a, b) {
                (Some(x), Some(y)) if is_integral(Some(x)) && is_integral(Some(y)) => {
                    Ok(ArgType::new(wider(x, y), c.any_nullable()))
                }
                _ => Err(diff_types(c, $symbol)),
            }
        }
    };
}

bitwise_result!(bitwise_and, "&");
bitwise_result!(bitwise_or, "|");
bitwise_result!(bitwise_xor, "^");

// ---- `~` ----

/// `~x`: the child stays as it is; an untyped `NULL` becomes an `int`
/// (Spark's default concrete integral type); anything else is refused
/// without casts.
fn not_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    match c.data_type(0) {
        None => vec![exact(DataType::Integer)],
        Some(t) if is_integral(Some(t)) => vec![exact(t.clone())],
        _ => vec![InputType::Any],
    }
}

/// `~x` keeps its child's type; an untyped `NULL` is an `int` (what the
/// inputs cast it to).
fn not_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    match c.data_type(0) {
        None => Ok(ArgType::new(DataType::Integer, c.any_nullable())),
        Some(t) if is_integral(Some(t)) => Ok(ArgType::new(t.clone(), c.any_nullable())),
        _ => Err(unexpected(c, "~", 0, "\"INTEGRAL\"")),
    }
}

// ---- the shifts ----

/// The shift target of `x`: `bigint` for a `bigint`, `int` for any other
/// value Spark casts (any numeric type, a string or an untyped `NULL`).
fn shift_target(t: Option<&DataType>) -> Option<DataType> {
    match t {
        Some(DataType::Long) => Some(DataType::Long),
        None
        | Some(
            DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Float
            | DataType::Double
            | DataType::Decimal(_)
            | DataType::String,
        ) => Some(DataType::Integer),
        _ => None,
    }
}

/// `shift(x, n)`: `x` to its shift target, `n` to `int`. A targetless `x`
/// is refused with `("INT" or "BIGINT")` (what Spark's `TypeCollection`
/// prints); a bad `n` fails as `"INT"`.
fn shift_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let left = match shift_target(c.data_type(0)) {
        Some(t) => exact(t),
        None => InputType::OneOf(vec![DataType::Integer, DataType::Long]),
    };
    vec![left, exact(DataType::Integer)]
}

/// A shift keeps its target's type; the casts may fail, so the analyzer's
/// post-cast nullability (not just the arguments') decides.
fn shift_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(
        c.data_type(0).cloned().unwrap_or(DataType::Integer),
        c.any_nullable(),
    ))
}

// ---- `bit_count` ----

/// `bit_count(x)`: an integral type or a boolean stays; an untyped
/// `NULL` becomes an `int`; anything else is refused without casts.
fn bit_count_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    match c.data_type(0) {
        None => vec![exact(DataType::Integer)],
        Some(DataType::Boolean) => vec![InputType::Any],
        Some(t) if is_integral(Some(t)) => vec![exact(t.clone())],
        _ => vec![InputType::Any],
    }
}

/// `bit_count(x)` is an `int`; an untyped `NULL` is one too (what the
/// inputs cast it to).
fn bit_count_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    match c.data_type(0) {
        Some(DataType::Boolean) | None => Ok(ArgType::new(DataType::Integer, c.any_nullable())),
        Some(t) if is_integral(Some(t)) => Ok(ArgType::new(DataType::Integer, c.any_nullable())),
        _ => Err(unexpected(
            c,
            "bit_count",
            0,
            "(\"INTEGRAL\" or \"BOOLEAN\")",
        )),
    }
}

// ---- `bit_get`, `getbit` ----

/// `bit_get(x, pos)`: `x` stays integral (an untyped `NULL` becomes an
/// `int`, whose 32 positions are checked); `pos` is an `int`.
fn bit_get_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let left = match c.data_type(0) {
        None => exact(DataType::Integer),
        Some(t) if is_integral(Some(t)) => exact(t.clone()),
        _ => InputType::Any,
    };
    vec![left, exact(DataType::Integer)]
}

/// `bit_get`/`getbit` is a `tinyint`.
macro_rules! bit_get_result {
    ($name:ident, $function:literal) => {
        fn $name(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
            match c.data_type(0) {
                Some(t) if is_integral(Some(t)) => {
                    Ok(ArgType::new(DataType::Byte, c.any_nullable()))
                }
                _ => Err(unexpected(c, $function, 0, "\"INTEGRAL\"")),
            }
        }
    };
}

bit_get_result!(bit_get, "bit_get");
bit_get_result!(getbit, "getbit");

pub(super) fn specs() -> Vec<Spec> {
    let bitwise = Inputs::Built(InputBuilder(bitwise_inputs));
    let mut out = scalar(&["&"], &bitwise, bitwise_and);
    out.extend(scalar(&["|"], &bitwise, bitwise_or));
    out.extend(scalar(&["^"], &bitwise, bitwise_xor));
    out.extend(scalar(
        &["~"],
        &Inputs::Built(InputBuilder(not_inputs)),
        not_result,
    ));
    let shift = Inputs::Built(InputBuilder(shift_inputs));
    out.extend(scalar(&["<<", "shiftleft"], &shift, shift_result));
    out.extend(scalar(&[">>", "shiftright"], &shift, shift_result));
    out.extend(scalar(&["shiftrightunsigned", ">>>"], &shift, shift_result));
    out.extend(scalar(
        &["bit_count"],
        &Inputs::Built(InputBuilder(bit_count_inputs)),
        bit_count_result,
    ));
    let bit_get_inputs = Inputs::Built(InputBuilder(bit_get_inputs));
    out.extend(scalar(&["bit_get"], &bit_get_inputs, bit_get));
    out.extend(scalar(&["getbit"], &bit_get_inputs, getbit));
    out
}
