// SPDX-License-Identifier: Apache-2.0
//! Spark 4.2's type coercion (`AnsiTypeCoercion`, `DecimalPrecision`) for
//! the canonical operators and the declared inputs of registry functions:
//! the implicit cast each argument needs and the call's result type. Every
//! rule here was checked against a Spark 4.2 run (`typeof`, analyzed plans
//! and error classes over a table with a column of each type).
//!
//! - Numbers widen: integral types to the wider one; `float` stays `float`
//!   only against `float`, otherwise `double`; an integral type meets a
//!   decimal as `decimal(digits, 0)` (an integral literal as the decimal of
//!   its own digits in arithmetic and comparisons); decimals combine by
//!   `DecimalPrecision`; anything with `double` or `float` and a decimal is
//!   `double`.
//! - A string meets an integral type as `bigint`, other numbers as
//!   `double`, and becomes the boolean, date, timestamp or binary it is
//!   compared with. A date meets a timestamp as that timestamp.
//! - An untyped `NULL` takes the type of what it meets.
//!
//! With `spark.sql.ansi.enabled` off Spark promotes strings differently
//! (string to `double` in arithmetic, to the other side's type in
//! comparisons); those operator coercions are not implemented and fail with
//! `KRISHIV_UNSUPPORTED_FEATURE`. Arithmetic on dates and timestamps
//! (`date_add`, intervals) is not implemented either.

use krishiv_types::{DataType, DecimalType};

use super::catalog::{CallTypes, InputType, Inputs, Operator};
use crate::{Builtin, CastMode};

/// An argument as coercion sees it.
#[derive(Clone, Copy)]
pub(super) struct Arg<'a> {
    /// Its type; `None` for an untyped `NULL`.
    pub(super) ty: Option<&'a DataType>,
    /// Whether it may be null.
    pub(super) nullable: bool,
    /// The value of an integral literal.
    pub(super) literal: Option<i128>,
}

/// An operator call, typed.
#[derive(Debug)]
pub(super) struct Typed {
    /// The type each argument is cast to, if it changes.
    pub(super) casts: Vec<Option<DataType>>,
    /// The result type.
    pub(super) data_type: DataType,
    /// Whether the result may be null.
    pub(super) nullable: bool,
}

/// Why a call does not type, as Spark's `DATATYPE_MISMATCH` subclasses say.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Mismatch {
    /// `BINARY_OP_DIFF_TYPES`.
    DiffTypes(Option<DataType>, Option<DataType>),
    /// `BINARY_OP_WRONG_TYPE`: the operands' type and what the operator
    /// takes.
    WrongType(Option<DataType>, &'static str),
    /// `UNEXPECTED_INPUT_TYPE`.
    Unexpected {
        /// The argument (0-based).
        arg: usize,
        /// The position Spark reports (0-based; the condition's, for
        /// `CASE`).
        param: usize,
        /// Its type.
        ty: Option<DataType>,
        /// The required type, as Spark prints it.
        required: String,
    },
    /// `DATA_DIFF_TYPES`: the argument types; whether Spark lists them as
    /// `["A", "B"]` (else `("A" or "B")`).
    DataDiff(Vec<Option<DataType>>, bool),
    /// A coercion Krishiv does not implement.
    Unsupported(&'static str),
}

const ADDITIVE: &str =
    r#"("NUMERIC" or "INTERVAL DAY TO SECOND" or "INTERVAL YEAR TO MONTH" or "INTERVAL")"#;
const NUMERIC: &str = r#""NUMERIC""#;
const DIVISIBLE: &str = r#"("DOUBLE" or "DECIMAL")"#;
const BOOLEAN: &str = r#""BOOLEAN""#;

/// The digits of an integral type as a decimal (`DecimalType.forType`).
fn integral_digits(t: &DataType) -> Option<u8> {
    match t {
        DataType::Byte => Some(3),
        DataType::Short => Some(5),
        DataType::Integer => Some(10),
        DataType::Long => Some(20),
        _ => None,
    }
}

fn is_numeric(t: &DataType) -> bool {
    integral_digits(t).is_some()
        || matches!(t, DataType::Float | DataType::Double | DataType::Decimal(_))
}

fn is_datetime(t: Option<&DataType>) -> bool {
    matches!(
        t,
        Some(DataType::Date | DataType::Timestamp | DataType::TimestampNtz)
    )
}

fn decimal(precision: u32, scale: u32) -> DecimalType {
    let p = precision.min(u32::from(DecimalType::MAX_PRECISION));
    let s = scale.min(p);
    // Bounded: 1 <= p <= 38 and s <= p always hold.
    DecimalType::try_new(p.max(1), s).unwrap_or(DecimalType::SYSTEM_DEFAULT)
}

/// The decimal a decimal or integral argument of type `t` meets a decimal
/// as: itself, or the integral type's digits (an integral literal's own
/// digits when `literal`).
fn as_decimal(a: &Arg<'_>, t: &DataType, literal: bool) -> DecimalType {
    match t {
        DataType::Decimal(d) => *d,
        t => {
            let digits = match a.literal.filter(|_| literal) {
                Some(v) => v.unsigned_abs().checked_ilog10().unwrap_or(0) + 1,
                None => u32::from(integral_digits(t).unwrap_or(DecimalType::MAX_PRECISION)),
            };
            decimal(digits, 0)
        }
    }
}

/// `DecimalPrecision.widerDecimalType`: the larger integer range and scale;
/// past 38 digits the fraction gives way and the range stays (checked
/// against Spark 4.2: `decimal(38,0)` and `decimal(10,5)` meet at
/// `decimal(38,0)`).
fn wider_decimal(a: DecimalType, b: DecimalType) -> DecimalType {
    let scale = u32::from(a.scale().max(b.scale()));
    let range = u32::from((a.precision() - a.scale()).max(b.precision() - b.scale()));
    let max = u32::from(DecimalType::MAX_PRECISION);
    if range + scale <= max {
        decimal(range + scale, scale)
    } else {
        decimal(max, max.saturating_sub(range))
    }
}

/// The type `a` and `b` meet at (ANSI `findWiderTypeForTwo` with string
/// promotion, as comparisons and `IN` use it); integral literals size as
/// their digits against a decimal when `literal`.
fn common(a: &Arg<'_>, b: &Arg<'_>, literal: bool) -> Option<Option<DataType>> {
    common_in(a, b, literal, true)
}

/// [`common`] under `spark.sql.ansi.enabled = ansi`: without ANSI a `float`
/// meeting an integral type stays `float`.
fn common_in(a: &Arg<'_>, b: &Arg<'_>, literal: bool, ansi: bool) -> Option<Option<DataType>> {
    use DataType as T;
    let (x, y) = match (a.ty, b.ty) {
        (None, None) => return Some(None),
        (Some(t), None) | (None, Some(t)) => return Some(Some(t.clone())),
        (Some(x), Some(y)) => (x, y),
    };
    if x == y {
        return Some(Some(x.clone()));
    }
    let t = match (x, y) {
        (T::Decimal(_), _) | (_, T::Decimal(_))
            if [x, y]
                .iter()
                .all(|t| matches!(t, T::Decimal(_)) || integral_digits(t).is_some()) =>
        {
            T::Decimal(wider_decimal(
                as_decimal(a, x, literal),
                as_decimal(b, y, literal),
            ))
        }
        _ if integral_digits(x).is_some() && integral_digits(y).is_some() => {
            if integral_digits(x) > integral_digits(y) {
                x.clone()
            } else {
                y.clone()
            }
        }
        (T::Float, n) | (n, T::Float) if !ansi && integral_digits(n).is_some() => T::Float,
        _ if is_numeric(x) && is_numeric(y) => T::Double,
        (T::String, n) | (n, T::String) if integral_digits(n).is_some() => T::Long,
        (T::String, n) | (n, T::String) if is_numeric(n) => T::Double,
        (T::String, t @ (T::Boolean | T::Date | T::Timestamp | T::TimestampNtz | T::Binary))
        | (t @ (T::Boolean | T::Date | T::Timestamp | T::TimestampNtz | T::Binary), T::String) => {
            t.clone()
        }
        (T::Date, t @ (T::Timestamp | T::TimestampNtz))
        | (t @ (T::Timestamp | T::TimestampNtz), T::Date) => t.clone(),
        (T::Timestamp, T::TimestampNtz) | (T::TimestampNtz, T::Timestamp) => T::Timestamp,
        _ => return None,
    };
    Some(Some(t))
}

/// The common type of all of `args` (`findWiderCommonType`: strings
/// first), `Some(None)` when every one is an untyped `NULL`.
fn common_all(args: &[Arg<'_>], ansi: bool) -> Option<Option<DataType>> {
    if !ansi && let Some(t) = legacy_string_type(args) {
        return t;
    }
    let strings = args
        .iter()
        .filter(|a| matches!(a.ty, Some(DataType::String)));
    let others = args
        .iter()
        .filter(|a| !matches!(a.ty, Some(DataType::String)));
    let mut acc: Option<DataType> = None;
    for a in strings.chain(others) {
        let cur = Arg {
            ty: acc.as_ref(),
            nullable: false,
            literal: None,
        };
        acc = common_in(&cur, a, false, ansi)?;
    }
    Some(acc)
}

/// Without ANSI a string meets numbers, dates and timestamps as a string
/// (and no other type): `Some(Some(string))` or `Some(None)` for no common
/// type; `None` when no argument is a string.
fn legacy_string_type(args: &[Arg<'_>]) -> Option<Option<Option<DataType>>> {
    use DataType as T;
    let typed: Vec<&DataType> = args.iter().filter_map(|a| a.ty).collect();
    if !typed.contains(&&T::String) {
        return None;
    }
    let promotes = |t: &&DataType| {
        is_numeric(t) || matches!(t, T::String | T::Date | T::Timestamp | T::TimestampNtz)
    };
    Some(typed.iter().all(promotes).then_some(Some(T::String)))
}

/// The type a value and a subquery column of `(values) IN (subquery)`
/// meet at.
pub(super) fn common_type(a: &Arg<'_>, b: &Arg<'_>) -> Option<Option<DataType>> {
    common(a, b, false)
}

/// The type two set-operation columns meet at (`findWiderTypeForTwo`, the
/// legacy rules with ANSI off).
pub(super) fn wider_type(a: &Arg<'_>, b: &Arg<'_>, ansi: bool) -> Option<Option<DataType>> {
    common_all(&[*a, *b], ansi)
}

/// The type two inline-table values meet at without ANSI
/// (`findWiderTypeWithoutStringPromotion`): the set-operation rules, except
/// that a string never meets another type.
pub(super) fn wider_type_without_string_promotion(
    a: &Arg<'_>,
    b: &Arg<'_>,
) -> Option<Option<DataType>> {
    match (a.ty, b.ty) {
        (Some(x), Some(y)) if (x == &DataType::String) != (y == &DataType::String) => None,
        _ => common_in(a, b, false, false),
    }
}

/// The cast to `to` an argument of type `from` needs, if any.
pub(super) fn cast_to(from: Option<&DataType>, to: Option<&DataType>) -> Option<DataType> {
    match to {
        Some(t) if from != Some(t) => Some(t.clone()),
        _ => None,
    }
}

fn unexpected(arg: usize, param: usize, ty: Option<&DataType>, required: &str) -> Mismatch {
    Mismatch::Unexpected {
        arg,
        param,
        ty: ty.cloned(),
        required: required.to_owned(),
    }
}

/// A non-literal argument of type `t`.
fn of_type(t: &DataType) -> Arg<'_> {
    Arg {
        ty: Some(t),
        nullable: false,
        literal: None,
    }
}

fn any_nullable(args: &[Arg<'_>]) -> bool {
    args.iter().any(|a| a.nullable)
}

/// Strings meet other types differently with ANSI off.
fn legacy_strings(args: &[Arg<'_>], ansi: bool) -> Result<(), Mismatch> {
    let strings = args
        .iter()
        .filter(|a| matches!(a.ty, Some(DataType::String)))
        .count();
    if !ansi && strings > 0 && strings < args.iter().filter(|a| a.ty.is_some()).count() {
        return Err(Mismatch::Unsupported(
            "implicit string casts with spark.sql.ansi.enabled off",
        ));
    }
    Ok(())
}

/// Types the call of operator `op` on `args`.
///
/// `ansi` is `spark.sql.ansi.enabled`; `loss` is
/// `spark.sql.decimalOperations.allowPrecisionLoss`.
pub(super) fn operator(
    op: Builtin,
    args: &[Arg<'_>],
    ansi: bool,
    loss: bool,
) -> Result<Typed, Mismatch> {
    let mut typed = operator_types(op, args, ansi, loss)?;
    // A cast that may yield NULL makes the result nullable, as the cast
    // argument is.
    if !matches!(
        op,
        Builtin::EqNullSafe | Builtin::IsNull | Builtin::IsNotNull
    ) {
        typed.nullable |= args.iter().zip(&typed.casts).any(|(a, c)| {
            a.ty.zip(c.as_ref())
                .is_some_and(|(f, t)| force_nullable(f, t))
        });
    }
    Ok(typed)
}

fn operator_types(
    op: Builtin,
    args: &[Arg<'_>],
    ansi: bool,
    loss: bool,
) -> Result<Typed, Mismatch> {
    use Builtin as B;
    let boolean = |casts: Vec<Option<DataType>>, nullable: bool| Typed {
        casts,
        data_type: DataType::Boolean,
        nullable,
    };
    match (op, args) {
        (B::Eq | B::EqNullSafe | B::Lt | B::LtEq | B::Gt | B::GtEq, [a, b]) => {
            legacy_strings(args, ansi)?;
            let t = common(a, b, true)
                .ok_or_else(|| Mismatch::DiffTypes(a.ty.cloned(), b.ty.cloned()))?;
            let casts = vec![cast_to(a.ty, t.as_ref()), cast_to(b.ty, t.as_ref())];
            Ok(boolean(casts, op != B::EqNullSafe && any_nullable(args)))
        }
        (B::Add | B::Subtract | B::Multiply | B::Divide | B::Remainder, [a, b]) => {
            arithmetic(op, a, b, ansi, loss)
        }
        (B::IntegralDivide, [a, b]) => integral_divide(a, b, ansi),
        (B::Negative, [a]) => {
            let t = match a.ty {
                None | Some(DataType::String) => DataType::Double,
                Some(t) if is_numeric(t) => t.clone(),
                ty => return Err(unexpected(0, 0, ty, ADDITIVE)),
            };
            Ok(Typed {
                casts: vec![cast_to(a.ty, Some(&t))],
                data_type: t,
                nullable: a.nullable,
            })
        }
        (B::Not, [a]) => match a.ty {
            None | Some(DataType::Boolean | DataType::String) => Ok(boolean(
                vec![cast_to(a.ty, Some(&DataType::Boolean))],
                a.nullable,
            )),
            ty => Err(unexpected(0, 0, ty, BOOLEAN)),
        },
        (B::And | B::Or, [a, b]) => {
            if let (Some(x), Some(y)) = (a.ty, b.ty)
                && x != y
                && common(a, b, false).is_none()
            {
                return Err(Mismatch::DiffTypes(Some(x.clone()), Some(y.clone())));
            }
            let mut casts = Vec::new();
            for x in args {
                match x.ty {
                    None | Some(DataType::Boolean | DataType::String) => {
                        casts.push(cast_to(x.ty, Some(&DataType::Boolean)));
                    }
                    ty => return Err(Mismatch::WrongType(ty.cloned(), BOOLEAN)),
                }
            }
            Ok(boolean(casts, any_nullable(args)))
        }
        (B::IsNull | B::IsNotNull, [_]) => Ok(boolean(vec![None], false)),
        (B::In, [_, ..]) => {
            legacy_strings(args, ansi)?;
            let t = common_all(args, ansi).ok_or_else(|| {
                Mismatch::DataDiff(args.iter().map(|a| a.ty.cloned()).collect(), true)
            })?;
            let casts = args.iter().map(|a| cast_to(a.ty, t.as_ref())).collect();
            Ok(boolean(casts, any_nullable(args)))
        }
        (B::Like | B::ILike | B::RLike, _) => {
            let casts = args
                .iter()
                .map(|a| cast_to(a.ty, Some(&DataType::String)))
                .collect();
            Ok(boolean(casts, any_nullable(args)))
        }
        (B::Concat, _) => {
            let binary = !args.is_empty() && args.iter().all(|a| a.ty == Some(&DataType::Binary));
            let t = if binary {
                DataType::Binary
            } else {
                DataType::String
            };
            Ok(Typed {
                casts: args.iter().map(|a| cast_to(a.ty, Some(&t))).collect(),
                data_type: t,
                nullable: any_nullable(args),
            })
        }
        (B::When, _) => when(args, ansi),
        _ => Err(Mismatch::Unsupported("this operator's arguments")),
    }
}

/// `CASE WHEN c THEN v ... ELSE e END`: conditions are booleans, values
/// meet at their common type.
fn when(args: &[Arg<'_>], ansi: bool) -> Result<Typed, Mismatch> {
    let mut values = Vec::new();
    let mut casts = vec![None; args.len()];
    for (i, pair) in args.chunks(2).enumerate() {
        match pair.get(1) {
            Some(v) => {
                let c = &pair[0];
                if c.ty != Some(&DataType::Boolean) {
                    return Err(unexpected(2 * i, i, c.ty, BOOLEAN));
                }
                values.push((2 * i + 1, v));
            }
            None => values.push((2 * i, &pair[0])),
        }
    }
    let vals: Vec<Arg<'_>> = values
        .iter()
        .map(|(_, v)| Arg {
            ty: v.ty,
            nullable: v.nullable,
            literal: v.literal,
        })
        .collect();
    legacy_strings(&vals, ansi)?;
    let t = common_all(&vals, ansi)
        .ok_or_else(|| Mismatch::DataDiff(vals.iter().map(|a| a.ty.cloned()).collect(), true))?
        .unwrap_or(DataType::Null);
    for (i, v) in &values {
        casts[*i] = cast_to(v.ty, Some(&t));
    }
    Ok(Typed {
        casts,
        data_type: t,
        nullable: any_nullable(&vals) || args.len().is_multiple_of(2),
    })
}

/// The operand types `div` takes, as Spark's errors print them.
const INTEGRAL_DIVIDABLE: &str =
    r#"("BIGINT" or "DECIMAL" or "INTERVAL YEAR TO MONTH" or "INTERVAL DAY TO SECOND")"#;

/// `a div b` (`IntegralDivide`): integral operands widen to `bigint`,
/// decimals meet as decimals, and the result is a `bigint` that is NULL
/// for a zero divisor. A string is read as `bigint` under ANSI and as a
/// `double` otherwise, which `div` does not take.
fn integral_divide(a: &Arg<'_>, b: &Arg<'_>, ansi: bool) -> Result<Typed, Mismatch> {
    use DataType as T;
    let typed = |casts: Vec<Option<DataType>>| Typed {
        casts,
        data_type: T::Long,
        nullable: true,
    };
    // What an operand is after Spark's implicit casts to `bigint`.
    let seen = |x: &Arg<'_>, other: &Arg<'_>| match x.ty {
        Some(t) if integral_digits(t).is_some() => Some(T::Long),
        Some(T::String) if !ansi => Some(T::Double),
        Some(T::String) if other.ty.is_some_and(|o| integral_digits(o).is_some()) => Some(T::Long),
        t => t.cloned(),
    };
    let (x, y) = (seen(a, b), seen(b, a));
    let ok = |t: &Option<DataType>| matches!(t, None | Some(T::Long | T::Decimal(_)));
    if !ok(&x) || !ok(&y) {
        return Err(if x == y {
            Mismatch::WrongType(x, INTEGRAL_DIVIDABLE)
        } else if x.as_ref().is_some_and(|t| *t == T::String)
            || y.as_ref().is_some_and(|t| *t == T::String)
        {
            Mismatch::Unsupported("div with a string and a non-integral operand")
        } else {
            Mismatch::DiffTypes(x, y)
        });
    }
    match (a.ty, b.ty) {
        (Some(T::Decimal(_)), _) | (_, Some(T::Decimal(_))) => {
            let wider = match (x, y) {
                (Some(T::Decimal(p)), Some(T::Decimal(q))) => wider_decimal(p, q),
                (Some(T::Decimal(p)), _) => wider_decimal(p, as_decimal(b, &T::Long, true)),
                (_, Some(T::Decimal(q))) => wider_decimal(as_decimal(a, &T::Long, true), q),
                _ => DecimalType::SYSTEM_DEFAULT,
            };
            let to = T::Decimal(wider);
            Ok(typed(vec![
                cast_to(a.ty, Some(&to)),
                cast_to(b.ty, Some(&to)),
            ]))
        }
        _ => Ok(typed(vec![
            cast_to(a.ty, Some(&T::Long)),
            cast_to(b.ty, Some(&T::Long)),
        ])),
    }
}

/// `+ - * / %`.
fn arithmetic(
    op: Builtin,
    a: &Arg<'_>,
    b: &Arg<'_>,
    ansi: bool,
    loss: bool,
) -> Result<Typed, Mismatch> {
    use DataType as T;
    let takes = match op {
        Builtin::Add | Builtin::Subtract => ADDITIVE,
        Builtin::Divide => DIVISIBLE,
        _ => NUMERIC,
    };
    if matches!(op, Builtin::Add | Builtin::Subtract) && (is_datetime(a.ty) || is_datetime(b.ty)) {
        return Err(Mismatch::Unsupported("date and timestamp arithmetic"));
    }
    legacy_strings(&[*a, *b], ansi)?;
    let (x, y) = match (a.ty, b.ty) {
        // `NULL / x` divides doubles (`x / NULL` takes x's type).
        (None, None) => (&T::Double, &T::Double),
        (None, Some(t)) if op == Builtin::Divide && is_numeric(t) => (&T::Double, &T::Double),
        (Some(t), None) | (None, Some(t)) if is_numeric(t) || *t == T::String => (t, t),
        (Some(_), None) | (None, Some(_)) => {
            return Err(Mismatch::DiffTypes(a.ty.cloned(), b.ty.cloned()));
        }
        (Some(x), Some(y)) => (x, y),
    };
    let nullable = a.nullable || b.nullable || matches!(op, Builtin::Divide | Builtin::Remainder);
    let numeric = |t: T, nullable: bool| Typed {
        casts: vec![cast_to(a.ty, Some(&t)), cast_to(b.ty, Some(&t))],
        data_type: t,
        nullable,
    };
    if !is_numeric(x) || !is_numeric(y) {
        // Operands of one type the operator does not take, or a string
        // promoted to one; other types do not meet.
        let promoted = (x == y || *x == T::String || *y == T::String)
            .then(|| common(&of_type(x), &of_type(y), false))
            .flatten();
        let Some(t) = promoted else {
            return Err(Mismatch::DiffTypes(Some(x.clone()), Some(y.clone())));
        };
        if let Some(t) = t.filter(|t| !is_numeric(t)) {
            return Err(Mismatch::WrongType(Some(t), takes));
        }
    }
    match (x, y) {
        (T::String, n) | (n, T::String) if is_numeric(n) => {
            let t = if integral_digits(n).is_some() && op != Builtin::Divide {
                T::Long
            } else {
                T::Double
            };
            Ok(numeric(t, nullable))
        }
        (T::Decimal(_), T::Float | T::Double) | (T::Float | T::Double, T::Decimal(_)) => {
            Ok(numeric(T::Double, nullable))
        }
        (T::Decimal(_), _) | (_, T::Decimal(_)) => {
            let (ld, rd) = (as_decimal(a, x, true), as_decimal(b, y, true));
            let kop = match op {
                Builtin::Add => krishiv_kernels::decimal::DecimalOp::Add,
                Builtin::Subtract => krishiv_kernels::decimal::DecimalOp::Subtract,
                Builtin::Multiply => krishiv_kernels::decimal::DecimalOp::Multiply,
                Builtin::Divide => krishiv_kernels::decimal::DecimalOp::Divide,
                _ => krishiv_kernels::decimal::DecimalOp::Remainder,
            };
            let pair = |d: DecimalType| (d.precision(), i8::try_from(d.scale()).unwrap_or(i8::MAX));
            let (p, s) = krishiv_kernels::decimal::result_type(kop, pair(ld), pair(rd), loss)
                .map_err(|_| Mismatch::Unsupported("this decimal arithmetic"))?;
            let out = decimal(u32::from(p), u32::try_from(s).unwrap_or(0));
            Ok(Typed {
                casts: vec![
                    cast_to(a.ty, Some(&T::Decimal(ld))),
                    cast_to(b.ty, Some(&T::Decimal(rd))),
                ],
                data_type: T::Decimal(out),
                // Without ANSI a decimal overflow is NULL.
                nullable: nullable || !ansi,
            })
        }
        _ if op == Builtin::Divide => Ok(numeric(T::Double, true)),
        (T::Float, T::Float) => Ok(numeric(T::Float, nullable)),
        // Without ANSI a `float` meeting an integral type stays `float`.
        (T::Float, n) | (n, T::Float) if !ansi && integral_digits(n).is_some() => {
            Ok(numeric(T::Float, nullable))
        }
        _ if integral_digits(x).is_some() && integral_digits(y).is_some() => {
            let t = if integral_digits(x) >= integral_digits(y) {
                x.clone()
            } else {
                y.clone()
            };
            Ok(numeric(t, nullable))
        }
        _ => Ok(numeric(T::Double, nullable)),
    }
}

/// The casts a registry function's arguments need for its declared
/// `inputs` (ANSI `implicitCast`); `call` is the arguments as the function's
/// input builder sees them.
pub(super) fn inputs(
    inputs: &Inputs,
    args: &[Arg<'_>],
    call: &CallTypes<'_>,
) -> Result<Vec<Option<DataType>>, Mismatch> {
    match inputs {
        Inputs::Common => common_casts(args, args, call.ansi()),
        Inputs::CommonNoStrings => {
            let typed = args.iter().filter(|a| a.ty.is_some());
            let strings = typed
                .clone()
                .filter(|a| a.ty == Some(&DataType::String))
                .count();
            if strings > 0 && strings < typed.count() {
                return Err(Mismatch::DataDiff(
                    args.iter().map(|a| a.ty.cloned()).collect(),
                    false,
                ));
            }
            common_casts(args, args, call.ansi())
        }
        Inputs::CommonWithZero => {
            let all: Vec<Arg<'_>> = args.iter().copied().chain([hidden_zero()]).collect();
            common_casts(args, &all, call.ansi())
        }
        Inputs::Leading(types) => {
            let n = types.len().min(args.len());
            let mut casts = declared(types, &args[..n])?;
            casts.extend(common_casts(&args[n..], &args[n..], call.ansi())?);
            Ok(casts)
        }
        Inputs::Comparable { zero } => {
            let hidden = hidden_zero();
            let other = if *zero { Some(&hidden) } else { args.get(1) };
            if let (Some(a), Some(b)) = (args.first(), other) {
                // Without ANSI a boolean compares with a number.
                let legacy_boolean = !call.ansi()
                    && matches!(
                        (a.ty, b.ty),
                        (Some(DataType::Boolean), Some(n)) | (Some(n), Some(DataType::Boolean))
                            if is_numeric(n)
                    );
                if !legacy_boolean {
                    common_in(a, b, true, call.ansi())
                        .ok_or_else(|| Mismatch::DiffTypes(a.ty.cloned(), b.ty.cloned()))?;
                }
            }
            Ok(vec![None; args.len()])
        }
        Inputs::Operator(op) | Inputs::TryOperator(op) => {
            let builtin = match op {
                Operator::Add => Builtin::Add,
                Operator::Subtract => Builtin::Subtract,
                Operator::Multiply => Builtin::Multiply,
                Operator::Divide => Builtin::Divide,
                Operator::Remainder => Builtin::Remainder,
            };
            operator_types(builtin, args, call.ansi(), call.decimal_loss()).map(|t| t.casts)
        }
        // `equal_null`: the casts of the null-safe `<=>` operator.
        Inputs::NullSafeEqual => {
            operator_types(Builtin::EqNullSafe, args, call.ansi(), call.decimal_loss())
                .map(|t| t.casts)
        }
        _ => match inputs.expected(call) {
            Some(types) => {
                let casts = declared(&types, args)?;
                legacy_string_counts(&types, args, call)?;
                Ok(casts)
            }
            None => Ok(vec![None; args.len()]),
        },
    }
}

/// Without ANSI, a string takes the place of a day count (`date_add(d,
/// '1')`, the collection `INT`, `SMALLINT`, `TINYINT`) only when it is
/// foldable: `date_add(d, s)` over a column is `UNEXPECTED_INPUT_TYPE`
/// (checked against a Spark 4.2 run, with `'1' || '2'` and `upper('1')`
/// accepted).
fn legacy_string_counts(
    types: &[InputType],
    args: &[Arg<'_>],
    call: &CallTypes<'_>,
) -> Result<(), Mismatch> {
    if call.ansi() {
        return Ok(());
    }
    for (i, a) in args.iter().enumerate() {
        let want = types.get(i).or(types.last()).unwrap_or(&InputType::Any);
        let days = matches!(
            want,
            InputType::OneOf(t) if t == &[DataType::Integer, DataType::Short, DataType::Byte]
        );
        if days && a.ty == Some(&DataType::String) && call.constant(i).is_none() {
            return Err(Mismatch::Unexpected {
                arg: i,
                param: i,
                ty: a.ty.cloned(),
                required: want.sql(),
            });
        }
    }
    Ok(())
}

/// The casts of `args` to the type all of `meet` (which includes `args`)
/// share.
fn common_casts(
    args: &[Arg<'_>],
    meet: &[Arg<'_>],
    ansi: bool,
) -> Result<Vec<Option<DataType>>, Mismatch> {
    let t = common_all(meet, ansi)
        .ok_or_else(|| Mismatch::DataDiff(meet.iter().map(|a| a.ty.cloned()).collect(), false))?;
    Ok(args.iter().map(|a| cast_to(a.ty, t.as_ref())).collect())
}

/// The literal `0` of `zeroifnull` and `nullifzero`.
fn hidden_zero() -> Arg<'static> {
    Arg {
        ty: Some(&DataType::Integer),
        nullable: false,
        literal: Some(0),
    }
}

/// The casts for arguments with declared input types (the last type
/// repeats for further arguments).
fn declared(types: &[InputType], args: &[Arg<'_>]) -> Result<Vec<Option<DataType>>, Mismatch> {
    let mut casts = Vec::new();
    for (i, a) in args.iter().enumerate() {
        let want = types.get(i).or(types.last()).unwrap_or(&InputType::Any);
        let to = implicit(a.ty, want).ok_or_else(|| Mismatch::Unexpected {
            arg: i,
            param: i,
            ty: a.ty.cloned(),
            required: want.sql(),
        })?;
        casts.push(cast_to(a.ty, to.as_ref()));
    }
    Ok(casts)
}

/// The type an argument of type `from` takes for `want`, if it may.
fn implicit(from: Option<&DataType>, want: &InputType) -> Option<Option<DataType>> {
    use DataType as T;
    match (from, want) {
        (_, InputType::Any) => Some(from.cloned()),
        (Some(t), InputType::Numeric { .. }) if is_numeric(t) => Some(Some(t.clone())),
        (None | Some(T::String), InputType::Numeric { .. }) => Some(Some(T::Double)),
        // A type of the collection stays; another takes the first member
        // it may be cast to (Spark's `TypeCollection`), except that a number
        // is not narrowed or widened into one (`date_add(d, 1L)` and
        // `date_add(d, 1.5D)` fail, while `substring(s, 1L)` casts).
        (Some(f), InputType::OneOf(types)) if types.contains(f) => Some(Some(f.clone())),
        (_, InputType::OneOf(types)) => types.iter().find_map(|t| {
            if from.is_some_and(is_numeric) && is_numeric(t) {
                return None;
            }
            implicit(from, &InputType::Exact(t.clone()))
        }),
        (None, InputType::Exact(t) | InputType::Strict(t)) => Some(Some(t.clone())),
        (Some(f), InputType::Strict(t)) => (f == t).then(|| Some(t.clone())),
        (Some(f), InputType::Exact(t)) => {
            let from_string = is_numeric(t)
                || matches!(
                    t,
                    T::Boolean | T::Date | T::Timestamp | T::TimestampNtz | T::Binary
                );
            let ok = f == t
                || (is_numeric(f) && is_numeric(t))
                || *t == T::String
                || (*f == T::String && from_string)
                || (*f == T::Date && matches!(t, T::Timestamp | T::TimestampNtz))
                || (matches!(f, T::Timestamp | T::TimestampNtz) && *t == T::Date);
            ok.then(|| Some(t.clone()))
        }
        _ => None,
    }
}

/// Why Spark refuses a cast (`DATATYPE_MISMATCH.CAST_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BadCast {
    /// `CAST_WITHOUT_SUGGESTION`.
    Without,
    /// `CAST_WITH_FUNC_SUGGESTION`: the function to use instead.
    Func(&'static str),
    /// `CAST_WITH_CONF_SUGGESTION`: allowed with ANSI off.
    Conf,
}

/// Whether Spark casts `from` to `to` in `mode` (`Cast.canCast`,
/// `canAnsiCast`, `canTryCast`), checked against a Spark 4.2 run of every
/// pair of the types here in each mode.
pub(super) fn can_cast(
    from: Option<&DataType>,
    to: &DataType,
    mode: CastMode,
) -> Result<(), BadCast> {
    use DataType as T;
    let Some(from) = from else {
        return Ok(());
    };
    let legacy = mode == CastMode::Legacy;
    // Allowed without ANSI, suggested as a conf under ANSI.
    let conf = || match mode {
        CastMode::Legacy => Ok(()),
        CastMode::Try => Err(BadCast::Without),
        _ => Err(BadCast::Conf),
    };
    if from == to {
        return Ok(());
    }
    // Nothing but a NULL casts to VOID (an untyped NULL returned above).
    if *to == T::Null {
        return Err(BadCast::Without);
    }
    if *from == T::String || *to == T::String {
        return Ok(());
    }
    match (from, to) {
        (f, T::Binary) if integral_digits(f).is_some() => conf(),
        (T::Binary, _) | (_, T::Binary) => Err(BadCast::Without),
        (T::TimestampNtz, t) | (t, T::TimestampNtz) => {
            if matches!(t, T::Date | T::Timestamp) {
                Ok(())
            } else {
                Err(BadCast::Without)
            }
        }
        (T::Timestamp, T::Date) | (T::Date, T::Timestamp) => Ok(()),
        (f, T::Date) => {
            if is_numeric(f) && !legacy {
                Err(BadCast::Func("`DATE_FROM_UNIX_DATE`"))
            } else {
                Err(BadCast::Without)
            }
        }
        (T::Date, t) if is_numeric(t) && !legacy => Err(BadCast::Func("`UNIX_DATE`")),
        (T::Date, _) | (T::Boolean, T::Timestamp) | (T::Timestamp, T::Boolean) => conf(),
        _ => Ok(()),
    }
}

/// Whether a cast from `from` to `to` may yield NULL from a non-NULL
/// value (Spark's `Cast.forceNullable`, in every mode).
pub(super) fn force_nullable(from: &DataType, to: &DataType) -> bool {
    use DataType as T;
    if from == to {
        return false;
    }
    match (from, to) {
        (T::String, T::Binary)
        | (_, T::String)
        | (T::Timestamp, T::Date)
        | (T::Date, T::Timestamp) => false,
        (T::String | T::Date, _) | (T::Float | T::Double, T::Timestamp) | (_, T::Date) => true,
        (_, T::Decimal(d)) => !null_safe_decimal(from, *d),
        (T::Float | T::Double | T::Decimal(_), t) => integral_digits(t).is_some(),
        (T::Timestamp, t) => integral_digits(t).is_some_and(|p| p < 20),
        _ => false,
    }
}

/// Spark's `canNullSafeCastToDecimal`: every value of `from` fits `to`.
fn null_safe_decimal(from: &DataType, to: DecimalType) -> bool {
    let digits = to.precision() - to.scale();
    match from {
        DataType::Boolean => digits >= 1,
        // Rounding to fewer decimals may carry into a new digit.
        DataType::Decimal(d) => digits > d.precision() - d.scale(),
        t => integral_digits(t).is_some_and(|p| digits >= p),
    }
}

/// Whether `TRY_CAST` from `from` to `to` may yield NULL from a non-NULL
/// value: unless it is Spark's up-cast (`Cast.canUpCast`). A STRING is
/// valid UTF-8, so binary bytes that are not become NULL (ADR 0023;
/// Spark keeps them, and its result is not nullable).
pub(super) fn try_nullable(from: &DataType, to: &DataType) -> bool {
    use DataType as T;
    if (from, to) == (&T::Binary, &T::String) {
        return true;
    }
    // Integral and floating types widen in this order.
    let rank = |t: &T| match t {
        T::Byte => Some(1),
        T::Short => Some(2),
        T::Integer => Some(3),
        T::Long => Some(4),
        T::Float => Some(5),
        T::Double => Some(6),
        _ => None,
    };
    let up = match (from, to) {
        _ if from == to || *to == T::String => true,
        (T::String, T::Binary)
        | (T::Date | T::TimestampNtz | T::Long, T::Timestamp)
        | (T::Date | T::Timestamp, T::TimestampNtz)
        | (T::Timestamp, T::Long) => true,
        (T::Decimal(a), T::Decimal(b)) => {
            b.precision() - b.scale() >= a.precision() - a.scale() && b.scale() >= a.scale()
        }
        (f, T::Decimal(d)) => integral_digits(f).is_some_and(|p| d.precision() - d.scale() >= p),
        (T::Decimal(d), t) => {
            d.scale() == 0 && integral_digits(t).is_some_and(|p| d.precision() <= p)
        }
        (f, t) => rank(f).zip(rank(t)).is_some_and(|(f, t)| f <= t),
    };
    !up
}

/// Spark's `ordinalNumber`: `first`, `second`, `third`, `4th`, ...
pub(super) fn ordinal(i: usize) -> String {
    match i {
        0 => "first".to_owned(),
        1 => "second".to_owned(),
        2 => "third".to_owned(),
        n => format!("{}th", n + 1),
    }
}

/// A type as Spark's errors quote it: `"INT"`, `"VOID"` for an untyped
/// `NULL`.
pub(super) fn quoted_type(t: Option<&DataType>) -> String {
    format!("\"{}\"", t.map_or_else(|| "VOID".to_owned(), DataType::sql))
}

#[cfg(test)]
mod tests {
    use krishiv_types::DataType as T;

    use super::super::catalog::ArgType;
    use super::*;

    /// ANSI implicit casts of function arguments (`AnsiTypeCoercion
    /// .implicitCast`), as Spark 4.2 inserts them, e.g. `date_add('2020-01-01',
    /// 1)` casts its string to a date and `upper(dt)` its date to a string.
    #[test]
    fn implicit_casts_follow_ansi() {
        let exact = |t: T| InputType::Exact(t);
        let numeric = InputType::Numeric { intervals: false };
        assert_eq!(
            implicit(Some(&T::String), &exact(T::Date)),
            Some(Some(T::Date))
        );
        assert_eq!(
            implicit(Some(&T::Date), &exact(T::Timestamp)),
            Some(Some(T::Timestamp))
        );
        assert_eq!(
            implicit(Some(&T::Date), &exact(T::String)),
            Some(Some(T::String))
        );
        assert_eq!(
            implicit(Some(&T::Long), &exact(T::Integer)),
            Some(Some(T::Integer))
        );
        assert_eq!(implicit(None, &exact(T::Integer)), Some(Some(T::Integer)));
        assert_eq!(implicit(Some(&T::Boolean), &exact(T::Integer)), None);
        assert_eq!(implicit(Some(&T::Date), &exact(T::Integer)), None);
        assert_eq!(implicit(Some(&T::String), &numeric), Some(Some(T::Double)));
        assert_eq!(implicit(Some(&T::Boolean), &numeric), None);
        assert_eq!(
            implicit(Some(&T::Binary), &InputType::Any),
            Some(Some(T::Binary))
        );
        assert_eq!(numeric.sql(), r#""NUMERIC""#);
        assert_eq!(
            InputType::Numeric { intervals: true }.sql(),
            r#""NUMERIC" or "ANSI INTERVAL""#
        );
        assert_eq!(InputType::Any.sql(), r#""ANY""#);
        assert_eq!(exact(T::Integer).sql(), r#""INT""#);
        // Spark's `TypeCollection`: members stay, others take the first
        // member they may be cast to, numbers excepted.
        let days = InputType::OneOf(vec![T::Integer, T::Short, T::Byte]);
        assert_eq!(implicit(Some(&T::Byte), &days), Some(Some(T::Byte)));
        assert_eq!(implicit(Some(&T::String), &days), Some(Some(T::Integer)));
        assert_eq!(implicit(None, &days), Some(Some(T::Integer)));
        assert_eq!(implicit(Some(&T::Long), &days), None);
        assert_eq!(implicit(Some(&T::Double), &days), None);
        let text = InputType::OneOf(vec![T::String, T::Binary]);
        assert_eq!(implicit(Some(&T::Binary), &text), Some(Some(T::Binary)));
        assert_eq!(implicit(Some(&T::Boolean), &text), Some(Some(T::String)));
        assert_eq!(implicit(Some(&T::Long), &text), Some(Some(T::String)));
        // `year(ts)` reads the timestamp's date.
        assert_eq!(
            implicit(Some(&T::TimestampNtz), &exact(T::Date)),
            Some(Some(T::Date))
        );
    }

    /// A canonical operator called with the wrong number of arguments (no
    /// front end builds one) is reported, never typed.
    #[test]
    fn malformed_operator_calls_are_unsupported() {
        let int = Arg {
            ty: Some(&T::Integer),
            nullable: false,
            literal: None,
        };
        assert_eq!(
            operator(Builtin::Not, &[int, int], true, true).err(),
            Some(Mismatch::Unsupported("this operator's arguments"))
        );
        assert_eq!(ordinal(3), "4th");
    }

    fn arg(t: Option<&T>) -> Arg<'_> {
        Arg {
            ty: t,
            nullable: false,
            literal: None,
        }
    }

    /// The casts a registry function's `Inputs` ask for.
    fn casts(inputs: &Inputs, args: &[Arg<'_>], ansi: bool) -> Result<Vec<Option<T>>, Mismatch> {
        let types: Vec<ArgType> = args
            .iter()
            .map(|a| ArgType {
                data_type: a.ty.cloned(),
                nullable: a.nullable,
            })
            .collect();
        super::inputs(inputs, args, &CallTypes::new(&types, &[]).with_ansi(ansi))
    }

    #[test]
    fn conditional_inputs_meet_at_a_common_type() {
        let (int, long, string, boolean) = (T::Integer, T::Long, T::String, T::Boolean);
        let leading = Inputs::Leading(vec![InputType::Strict(T::Boolean)]);
        // `if(cond, 1, 2L)`: the branches meet at `bigint`, the condition
        // stays.
        let args = [arg(Some(&boolean)), arg(Some(&int)), arg(Some(&long))];
        assert_eq!(
            casts(&leading, &args, true),
            Ok(vec![None, Some(T::Long), None])
        );
        // An untyped NULL condition is a boolean; a string is no condition.
        assert_eq!(
            casts(
                &leading,
                &[arg(None), arg(Some(&int)), arg(Some(&int))],
                true
            ),
            Ok(vec![Some(T::Boolean), None, None])
        );
        let bad = casts(
            &leading,
            &[arg(Some(&string)), arg(Some(&int)), arg(None)],
            true,
        );
        assert!(
            matches!(bad, Err(Mismatch::Unexpected { arg: 0, .. })),
            "{bad:?}"
        );
        // `coalesce('a', 1)`: ANSI promotes the string to `bigint`, legacy
        // makes both strings; a boolean and a string never meet in legacy.
        let mixed = [arg(Some(&string)), arg(Some(&int))];
        assert_eq!(
            casts(&Inputs::Common, &mixed, true),
            Ok(vec![Some(T::Long), Some(T::Long)])
        );
        assert_eq!(
            casts(&Inputs::Common, &mixed, false),
            Ok(vec![None, Some(T::String)])
        );
        let flag = [arg(Some(&string)), arg(Some(&boolean))];
        assert!(casts(&Inputs::Common, &flag, false).is_err());
        // `greatest('a', 1)` is an error in both modes.
        for ansi in [true, false] {
            assert!(casts(&Inputs::CommonNoStrings, &mixed, ansi).is_err());
        }
        assert!(
            casts(
                &Inputs::CommonNoStrings,
                &[arg(Some(&string)), arg(None)],
                true
            )
            .is_ok()
        );
        // `zeroifnull(x)` meets `x` with `0`: a decimal(10,2) grows to (12,2).
        let dec = T::Decimal(DecimalType::try_new(10, 2).unwrap());
        assert_eq!(
            casts(&Inputs::CommonWithZero, &[arg(Some(&dec))], true),
            Ok(vec![Some(T::Decimal(DecimalType::try_new(12, 2).unwrap()))])
        );
        assert_eq!(
            casts(&Inputs::CommonWithZero, &[arg(None)], true),
            Ok(vec![Some(T::Integer)])
        );
    }

    #[test]
    fn comparable_inputs_are_checked_not_cast() {
        let (int, boolean, string) = (T::Integer, T::Boolean, T::String);
        let eq = Inputs::Comparable { zero: false };
        assert_eq!(
            casts(&eq, &[arg(Some(&int)), arg(Some(&T::Float))], true),
            Ok(vec![None, None])
        );
        let bad = casts(&eq, &[arg(Some(&boolean)), arg(Some(&int))], true);
        assert!(matches!(bad, Err(Mismatch::DiffTypes(..))), "{bad:?}");
        // Without ANSI a boolean compares with a number.
        assert!(casts(&eq, &[arg(Some(&boolean)), arg(Some(&int))], false).is_ok());
        assert!(casts(&eq, &[arg(Some(&string)), arg(Some(&int))], true).is_ok());
        let zero = Inputs::Comparable { zero: true };
        assert!(casts(&zero, &[arg(Some(&int))], true).is_ok());
        assert!(casts(&zero, &[arg(Some(&boolean))], true).is_err());
    }

    #[test]
    fn operator_inputs_cast_as_the_operator_does() {
        let (int, long, float, string) = (T::Integer, T::Long, T::Float, T::String);
        let rem = Inputs::Operator(Operator::Remainder);
        assert_eq!(
            casts(&rem, &[arg(Some(&int)), arg(Some(&long))], true),
            Ok(vec![Some(T::Long), None])
        );
        // `try_divide` of integers divides doubles.
        let div = Inputs::Operator(Operator::Divide);
        assert_eq!(
            casts(&div, &[arg(Some(&int)), arg(Some(&int))], true),
            Ok(vec![Some(T::Double), Some(T::Double)])
        );
        // A `float` meeting an integral type is `double`, or `float` without
        // ANSI.
        let add = Inputs::Operator(Operator::Add);
        let args = [arg(Some(&float)), arg(Some(&int))];
        assert_eq!(
            casts(&add, &args, true),
            Ok(vec![Some(T::Double), Some(T::Double)])
        );
        assert_eq!(casts(&add, &args, false), Ok(vec![None, Some(T::Float)]));
        let bad = casts(&add, &[arg(Some(&T::Boolean)), arg(Some(&string))], true);
        assert!(bad.is_err());
    }

    #[test]
    fn strict_inputs_take_the_exact_type_only() {
        let strict = InputType::Strict(T::Boolean);
        assert_eq!(implicit(Some(&T::Boolean), &strict), Some(Some(T::Boolean)));
        assert_eq!(implicit(None, &strict), Some(Some(T::Boolean)));
        assert_eq!(implicit(Some(&T::String), &strict), None);
        assert_eq!(implicit(Some(&T::Integer), &strict), None);
        assert_eq!(strict.sql(), r#""BOOLEAN""#);
    }
}
