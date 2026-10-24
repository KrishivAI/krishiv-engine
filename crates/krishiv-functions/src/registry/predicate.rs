// SPDX-License-Identifier: Apache-2.0
//! The predicate functions' analysis signatures: `equal_null` (typed as
//! the null-safe `<=>`, never NULL), `isnan` (a float or double, never
//! NULL), `isnull`/`isnotnull` (anything, never NULL) and `assert_true` /
//! `raise_error` (a boolean condition / a message, `void`).
//!
//! Ground truth: Spark 4.2's `predicates.scala` (`EqualNull` as
//! `EqualNullSafe`), `nullExpressions.scala` (`IsNaN`), `misc.scala`
//! (`AssertTrue` as `If`, `RaiseError`), the `misc-functions.sql` and
//! `null-handling.sql` goldens and the `predicate`/`misc` matrix families.

use krishiv_common::error::KrishivError;
use krishiv_types::DataType;

use super::defs::{Spec, exact, scalar};
use crate::signature::{ArgType, CallTypes, InputBuilder, InputType, Inputs};

// ---- `equal_null` ----

/// `equal_null(a, b)`: like `<=>`, a boolean that is never NULL.
fn equal_null(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let _ = c;
    Ok(ArgType::new(DataType::Boolean, false))
}

// ---- `isnan` ----

/// `isnan(x)`: a float stays a float, anything else Spark casts becomes a
/// double; anything else is refused with `("DOUBLE" or "FLOAT")`.
fn isnan_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    match c.data_type(0) {
        Some(DataType::Float) => vec![exact(DataType::Float)],
        None
        | Some(
            DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::Double
            | DataType::Decimal(_)
            | DataType::String,
        ) => vec![exact(DataType::Double)],
        _ => vec![InputType::OneOf(vec![DataType::Double, DataType::Float])],
    }
}

/// `isnan(x)` is a boolean, never NULL (a NULL input is simply not NaN).
fn isnan(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let _ = c;
    Ok(ArgType::new(DataType::Boolean, false))
}

// ---- `isnull`, `isnotnull` ----

/// `isnull(x)` / `isnotnull(x)`: a boolean, never NULL. (SQL calls reach
/// these through the canonical `isnull`/`isnotnull` operators; the
/// registry entry keeps the catalog complete.)
fn is_null(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let _ = c;
    Ok(ArgType::new(DataType::Boolean, false))
}

// ---- `assert_true`, `raise_error` ----

/// `assert_true(c)` / `assert_true(c, message)`: `void`, always nullable.
/// The analyzer materializes the default message (`'<cond>' is not
/// true!`) as the second argument, as Spark's constructor does.
fn assert_true(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let _ = c;
    Ok(ArgType {
        data_type: None,
        nullable: true,
    })
}

/// `raise_error(message)`: `void`, always nullable.
fn raise_error(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let _ = c;
    Ok(ArgType {
        data_type: None,
        nullable: true,
    })
}

pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(&["equal_null"], &Inputs::NullSafeEqual, equal_null);
    out.extend(scalar(
        &["isnan"],
        &Inputs::Built(InputBuilder(isnan_inputs)),
        isnan,
    ));
    out.extend(scalar(&["isnull", "isnotnull"], &Inputs::Any, is_null));
    // The condition is strictly boolean (an untyped `NULL` becomes one);
    // the message is a string.
    out.extend(scalar(
        &["assert_true"],
        &Inputs::Expects(vec![
            InputType::Strict(DataType::Boolean),
            exact(DataType::String),
        ]),
        assert_true,
    ));
    out.extend(scalar(
        &["raise_error"],
        &Inputs::Expects(vec![exact(DataType::String)]),
        raise_error,
    ));
    out
}
