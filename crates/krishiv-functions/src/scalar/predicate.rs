// SPDX-License-Identifier: Apache-2.0
//! Predicate functions: `equal_null` (null-safe equality, never NULL),
//! `isnan` (never NULL: a NULL input is simply not NaN), `isnull` /
//! `isnotnull` (never NULL) and `assert_true` / `raise_error`
//! (`USER_RAISED_EXCEPTION`, a `void` NULL when they pass).
//!
//! `assert_true` and `raise_error` evaluate eagerly per batch; under the
//! lazy `CASE WHEN` they still raise only for taken rows, because each
//! branch evaluates over its own rows.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef, BooleanArray};
use arrow_schema::DataType;
use krishiv_common::error::{KrishivError, UserRaisedException};
use krishiv_kernels::Operand;
use krishiv_kernels::compare::{self, BytesOperand, Equal, Validity};

use super::args::{Inputs, is_binary, is_string};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "equal_null",
    "isnan",
    "isnull",
    "isnotnull",
    "assert_true",
    "raise_error",
];

/// The validity of an operand: an untyped NULL is all NULL.
fn validity<A: Array + ?Sized>(a: &A) -> Validity<'_> {
    if a.data_type() == &DataType::Null {
        Validity::Scalar { null: true }
    } else {
        Validity::Array(a.nulls())
    }
}

/// `equal_null(a, b)` over two columns of one type: the equality with
/// NULLs as equal values, never NULL itself.
fn equal_null(t: &DataType, ctx: &BindOptions) -> Option<Binding> {
    let ctx = ctx.kernel;
    macro_rules! prim {
        ($t:ty) => {
            Binding::new(DataType::Boolean, move |x: &Inputs<'_>| {
                let (l, r) = (x.column(0)?, x.column(1)?);
                let (l, r) = (
                    l.as_primitive::<$t>().clone(),
                    r.as_primitive::<$t>().clone(),
                );
                let eq =
                    compare::compare::<$t, Equal>(Operand::Array(&l), Operand::Array(&r), &ctx)?;
                Ok(Arc::new(compare::null_safe(&eq, validity(&l), validity(&r))?) as ArrayRef)
            })
        };
    }
    macro_rules! bytes {
        ($array:ident) => {
            Binding::new(DataType::Boolean, move |x: &Inputs<'_>| {
                let (l, r) = (x.$array(0)?, x.$array(1)?);
                let eq = compare::compare_bytes::<_, Equal>(
                    BytesOperand::Array(&l),
                    BytesOperand::Array(&r),
                )?;
                Ok(Arc::new(compare::null_safe(&eq, validity(&l), validity(&r))?) as ArrayRef)
            })
        };
    }
    Some(match t {
        DataType::Null => Binding::new(DataType::Boolean, move |x: &Inputs<'_>| {
            let (l, r) = (x.column(0)?, x.column(1)?);
            let eq = BooleanArray::new_null(x.rows());
            Ok(Arc::new(compare::null_safe(&eq, validity(&l), validity(&r))?) as ArrayRef)
        }),
        DataType::Boolean => Binding::new(DataType::Boolean, move |x: &Inputs<'_>| {
            let (l, r) = (x.booleans(0)?, x.booleans(1)?);
            let eq = compare::compare_bool::<Equal>(
                compare::BoolOperand::Array(&l),
                compare::BoolOperand::Array(&r),
            )?;
            Ok(Arc::new(compare::null_safe(&eq, validity(&l), validity(&r))?) as ArrayRef)
        }),
        DataType::Int8 => prim!(Int8Type),
        DataType::Int16 => prim!(Int16Type),
        DataType::Int32 => prim!(Int32Type),
        DataType::Int64 => prim!(Int64Type),
        DataType::Float32 => prim!(Float32Type),
        DataType::Float64 => prim!(Float64Type),
        DataType::Decimal128(..) => prim!(Decimal128Type),
        DataType::Date32 => prim!(Date32Type),
        DataType::Timestamp(..) => prim!(TimestampMicrosecondType),
        t if is_string(t) => bytes!(strings),
        t if is_binary(t) => bytes!(binaries),
        _ => return None,
    })
}

/// `isnan(x)`: true for NaN, false for anything else (including NULL).
fn isnan(t: &DataType) -> Option<Binding> {
    macro_rules! nan {
        ($t:ty) => {
            Binding::new(DataType::Boolean, move |x: &Inputs<'_>| {
                let a = x.column(0)?.as_primitive::<$t>().clone();
                Ok(Arc::new(BooleanArray::from(
                    (0..a.len())
                        .map(|i| Some(a.is_valid(i) && a.value(i).is_nan()))
                        .collect::<Vec<_>>(),
                )) as ArrayRef)
            })
        };
    }
    Some(match t {
        DataType::Float32 => nan!(Float32Type),
        DataType::Float64 => nan!(Float64Type),
        _ => return None,
    })
}

/// `isnull(x)` / `isnotnull(x)`.
fn is_null(negated: bool) -> Binding {
    Binding::new(DataType::Boolean, move |x: &Inputs<'_>| {
        let a = x.column(0)?;
        Ok(Arc::new(BooleanArray::from(
            (0..a.len())
                .map(|i| Some(a.is_null(i) != negated))
                .collect::<Vec<_>>(),
        )) as ArrayRef)
    })
}

/// The `USER_RAISED_EXCEPTION` for `message` (a NULL message is `"null"`).
fn raised(message: Option<&str>) -> KrishivError {
    UserRaisedException {
        error_message: message.unwrap_or("null").to_owned(),
    }
    .into()
}

/// `assert_true(c, message)`: NULL when the condition holds, else
/// `USER_RAISED_EXCEPTION`.
fn assert_true() -> Binding {
    Binding::new(DataType::Null, move |x: &Inputs<'_>| {
        let (cond, msg) = (x.booleans(0)?, x.strings(1)?);
        for i in 0..cond.len() {
            if !cond.is_valid(i) || !cond.value(i) {
                return Err(raised(msg.is_valid(i).then(|| msg.value(i))));
            }
        }
        Ok(arrow_array::new_null_array(&DataType::Null, x.rows()))
    })
}

/// `raise_error(message)`: always `USER_RAISED_EXCEPTION`.
fn raise_error() -> Binding {
    Binding::new(DataType::Null, move |x: &Inputs<'_>| {
        // Every batch has at least one row (`invoke` returns early for
        // zero rows), so row 0 always has the message.
        let msg = x.strings(0)?;
        Err(raised(msg.is_valid(0).then(|| msg.value(0))))
    })
}

/// Binds the predicate function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match (listed, types.as_slice()) {
        ("equal_null", [l, r])
            if l == r || (is_string(l) && is_string(r)) || (is_binary(l) && is_binary(r)) =>
        {
            match equal_null(l, opts) {
                Some(b) => Ok(b),
                None => return Some((listed, Err(bad_types(listed, args)))),
            }
        }
        ("isnan", [DataType::Float32 | DataType::Float64]) => match isnan(&types[0]) {
            Some(b) => Ok(b),
            None => return Some((listed, Err(bad_types(listed, args)))),
        },
        ("isnull", [_]) => Ok(is_null(false)),
        ("isnotnull", [_]) => Ok(is_null(true)),
        ("assert_true", [DataType::Boolean, t]) if is_string(t) => Ok(assert_true()),
        ("raise_error", [t]) if is_string(t) => Ok(raise_error()),
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, binding))
}
