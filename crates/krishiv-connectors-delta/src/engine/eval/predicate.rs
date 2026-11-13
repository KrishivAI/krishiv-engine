// SPDX-License-Identifier: Apache-2.0
//! Predicate evaluation (port of the kernel's `evaluate_predicate`, `delta_kernel` 0.30.1
//! `engine/arrow_expression/evaluate_expression.rs`, Apache-2.0).
//!
//! - `NOT` is pushed down: each node is evaluated inverted instead of negating its result
//!   (`<` becomes `>=`, `IS NULL` becomes `IS NOT NULL`, `DISTINCT` becomes `NOT DISTINCT`,
//!   `AND` becomes `OR` over inverted children, De Morgan).
//! - Comparisons are SQL's: a NULL side gives NULL. `DISTINCT` is null-safe and never NULL.
//! - `AND`/`OR` are Kleene: `false AND NULL` is false, `true OR NULL` is true; an empty `AND`
//!   is true and an empty `OR` false.
//! - `IN` (a value in a list column, or a literal in a literal array) never gives NULL: a NULL
//!   needle or a NULL element never matches.
//! - Sides of different string or binary encodings (views and plain) are compared as views.
//! - `Opaque` and `Unknown` predicates are unsupported (an engine must never read an unknown
//!   predicate as NULL in a filter; the kernel itself treats it as NULL for data skipping).

use std::sync::Arc;

use arrow_arith::boolean::{and_kleene, is_not_null, is_null, not, or_kleene};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow_ord::cmp::{distinct, eq, gt, gt_eq, lt, lt_eq, neq, not_distinct};
use arrow_ord::comparison::{in_list, in_list_utf8};
use arrow_schema::{DataType as ArrowType, TimeUnit};
use delta_kernel::expressions::{
    BinaryPredicate, BinaryPredicateOp, Expression, JunctionPredicate, JunctionPredicateOp,
    Predicate, Scalar, UnaryPredicate, UnaryPredicateOp,
};
use delta_kernel::schema::DataType;
use delta_kernel::{KernelError, KernelResult};

use super::expr::{evaluate_expression, repeat_bool};
use super::types::cast_to;
use super::{Ctx, arrow_err};

/// Evaluates `predicate` over `batch`, inverted when `inverted`.
///
/// # Errors
/// The kernel's evaluation errors (see the module docs).
pub(crate) fn evaluate_predicate(
    ctx: &Ctx,
    predicate: &Predicate,
    batch: &RecordBatch,
    inverted: bool,
) -> KernelResult<BooleanArray> {
    let maybe_not = |b: BooleanArray| -> KernelResult<BooleanArray> {
        if inverted {
            not(&b).map_err(arrow_err)
        } else {
            Ok(b)
        }
    };
    match predicate {
        Predicate::BooleanExpression(expr) => {
            let a = evaluate_expression(ctx, expr, batch, Some(&DataType::BOOLEAN))?;
            let b = a
                .as_boolean_opt()
                .ok_or_else(|| KernelError::generic("expected boolean array"))?
                .clone();
            maybe_not(b)
        }
        Predicate::Not(p) => evaluate_predicate(ctx, p, batch, !inverted),
        Predicate::Unary(UnaryPredicate {
            op: UnaryPredicateOp::IsNull,
            expr,
        }) => {
            let a = evaluate_expression(ctx, expr, batch, None)?;
            if inverted {
                is_not_null(&a)
            } else {
                is_null(&a)
            }
            .map_err(arrow_err)
        }
        Predicate::Binary(BinaryPredicate { op, left, right }) => {
            if *op == BinaryPredicateOp::In {
                return maybe_not(evaluate_in(ctx, left, right, batch)?);
            }
            let l = evaluate_expression(ctx, left, batch, None)?;
            let r = evaluate_expression(ctx, right, batch, None)?;
            let (l, r) = if l.data_type() == r.data_type() {
                (l, r)
            } else {
                (to_view(&l)?, to_view(&r)?)
            };
            match (op, inverted) {
                (BinaryPredicateOp::LessThan, false) => lt(&l, &r),
                (BinaryPredicateOp::LessThan, true) => gt_eq(&l, &r),
                (BinaryPredicateOp::GreaterThan, false) => gt(&l, &r),
                (BinaryPredicateOp::GreaterThan, true) => lt_eq(&l, &r),
                (BinaryPredicateOp::Equal, false) => eq(&l, &r),
                (BinaryPredicateOp::Equal, true) => neq(&l, &r),
                (BinaryPredicateOp::Distinct, false) => distinct(&l, &r),
                (BinaryPredicateOp::Distinct, true) => not_distinct(&l, &r),
                (BinaryPredicateOp::In, _) => {
                    return Err(KernelError::internal_error("IN handled above"));
                }
            }
            .map_err(arrow_err)
        }
        Predicate::Junction(JunctionPredicate { op, preds }) => {
            let and = matches!(
                (op, inverted),
                (JunctionPredicateOp::And, false) | (JunctionPredicateOp::Or, true)
            );
            let mut acc: Option<BooleanArray> = None;
            for p in preds {
                let b = evaluate_predicate(ctx, p, batch, inverted)?;
                acc = Some(match acc {
                    None => b,
                    Some(a) if and => and_kleene(&a, &b).map_err(arrow_err)?,
                    Some(a) => or_kleene(&a, &b).map_err(arrow_err)?,
                });
            }
            Ok(acc.unwrap_or_else(|| repeat_bool(and, batch.num_rows())))
        }
        Predicate::Opaque(p) => Err(KernelError::unsupported(format!(
            "Unsupported opaque predicate: {:?}",
            p.op
        ))),
        Predicate::Unknown(name) => Err(KernelError::unsupported(format!(
            "Unknown predicate: {name:?}"
        ))),
    }
}

/// The view form of a string or binary column (and of a list of them); others unchanged.
fn to_view(a: &ArrayRef) -> KernelResult<ArrayRef> {
    let target = match a.data_type() {
        ArrowType::Utf8 | ArrowType::LargeUtf8 => ArrowType::Utf8View,
        ArrowType::Binary | ArrowType::LargeBinary => ArrowType::BinaryView,
        _ => return Ok(Arc::clone(a)),
    };
    cast_to(a, &target, false)
}

/// The plain form of a string view column, or of a list of string views.
fn to_plain(a: &ArrayRef) -> KernelResult<ArrayRef> {
    match a.data_type() {
        ArrowType::Utf8View | ArrowType::LargeUtf8 => cast_to(a, &ArrowType::Utf8, false),
        ArrowType::BinaryView | ArrowType::LargeBinary => cast_to(a, &ArrowType::Binary, false),
        ArrowType::List(f) | ArrowType::LargeList(f) | ArrowType::ListView(f)
            if matches!(
                f.data_type(),
                ArrowType::Utf8View | ArrowType::LargeUtf8 | ArrowType::BinaryView
            ) || !matches!(a.data_type(), ArrowType::List(_)) =>
        {
            let element = match f.data_type() {
                ArrowType::Utf8View | ArrowType::LargeUtf8 => ArrowType::Utf8,
                ArrowType::BinaryView => ArrowType::Binary,
                other => other.clone(),
            };
            let target = ArrowType::List(Arc::new(f.as_ref().clone().with_data_type(element)));
            cast_to(a, &target, false)
        }
        _ => Ok(Arc::clone(a)),
    }
}

fn evaluate_in(
    ctx: &Ctx,
    left: &Expression,
    right: &Expression,
    batch: &RecordBatch,
) -> KernelResult<BooleanArray> {
    match (left, right) {
        (Expression::Literal(_), Expression::Column(_)) => {
            let l = to_plain(&evaluate_expression(ctx, left, batch, None)?)?;
            let r = to_plain(&evaluate_expression(ctx, right, batch, None)?)?;
            let list = r.as_list_opt::<i32>().ok_or_else(|| {
                KernelError::invalid_expression(format!(
                    "Cannot cast to list array: {}",
                    r.data_type()
                ))
            })?;
            if let Some(s) = l.as_string_opt::<i32>()
                && list.value_type() == ArrowType::Utf8
            {
                return in_list_utf8(s, list).map_err(arrow_err);
            }
            in_primitive_list(&l, list)
        }
        (Expression::Literal(needle), Expression::Literal(Scalar::Array(items))) => {
            // SQL equality: a NULL never matches, and nested values never match.
            let found = items.array_elements().iter().any(|e| needle.logical_eq(e));
            Ok(repeat_bool(found, batch.num_rows()))
        }
        (l, r) => Err(KernelError::invalid_expression(format!(
            "Invalid right value for (NOT) IN comparison, left is: {l} right is: {r}"
        ))),
    }
}

fn in_primitive_list(l: &ArrayRef, list: &arrow_array::ListArray) -> KernelResult<BooleanArray> {
    macro_rules! typed {
        ($t:ty) => {
            in_list(
                l.as_primitive_opt::<$t>().ok_or_else(|| {
                    KernelError::invalid_expression(format!(
                        "Cannot cast to primitive array: {}",
                        l.data_type()
                    ))
                })?,
                list,
            )
            .map_err(arrow_err)
        };
    }
    match l.data_type() {
        ArrowType::Int8 => typed!(Int8Type),
        ArrowType::Int16 => typed!(Int16Type),
        ArrowType::Int32 => typed!(Int32Type),
        ArrowType::Int64 => typed!(Int64Type),
        ArrowType::Float32 => typed!(Float32Type),
        ArrowType::Float64 => typed!(Float64Type),
        ArrowType::Date32 => typed!(Date32Type),
        ArrowType::Timestamp(TimeUnit::Microsecond, _) => typed!(TimestampMicrosecondType),
        ArrowType::Decimal128(..) => typed!(Decimal128Type),
        other => Err(KernelError::generic(format!(
            "Bad Comparison between: {other:?} and {:?}",
            list.data_type()
        ))),
    }
}
