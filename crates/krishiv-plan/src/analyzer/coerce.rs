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

use super::catalog::{CallTypes, InputType, Inputs};
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

/// `DecimalPrecision.widerDecimalType`.
fn wider_decimal(a: DecimalType, b: DecimalType) -> DecimalType {
    let scale = a.scale().max(b.scale());
    let range = (a.precision() - a.scale()).max(b.precision() - b.scale());
    decimal(u32::from(range) + u32::from(scale), u32::from(scale))
}

/// The type `a` and `b` meet at (ANSI `findWiderTypeForTwo` with string
/// promotion, as comparisons and `IN` use it); integral literals size as
/// their digits against a decimal when `literal`.
fn common(a: &Arg<'_>, b: &Arg<'_>, literal: bool) -> Option<Option<DataType>> {
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
fn common_all(args: &[Arg<'_>]) -> Option<Option<DataType>> {
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
        acc = common(&cur, a, false)?;
    }
    Some(acc)
}

/// The type a value and a subquery column of `(values) IN (subquery)`
/// meet at.
pub(super) fn common_type(a: &Arg<'_>, b: &Arg<'_>) -> Option<Option<DataType>> {
    common(a, b, false)
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
            let t = common_all(args).ok_or_else(|| {
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
    let t = common_all(&vals)
        .ok_or_else(|| Mismatch::DataDiff(vals.iter().map(|a| a.ty.cloned()).collect(), true))?
        .ok_or(Mismatch::Unsupported("values of the NULL type"))?;
    for (i, v) in &values {
        casts[*i] = cast_to(v.ty, Some(&t));
    }
    Ok(Typed {
        casts,
        data_type: t,
        nullable: any_nullable(&vals) || args.len().is_multiple_of(2),
    })
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
    if *inputs == Inputs::Common {
        let t = common_all(args).ok_or_else(|| {
            Mismatch::DataDiff(args.iter().map(|a| a.ty.cloned()).collect(), false)
        })?;
        return Ok(args.iter().map(|a| cast_to(a.ty, t.as_ref())).collect());
    }
    let Some(types) = inputs.expected(call) else {
        return Ok(vec![None; args.len()]);
    };
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
        (None, InputType::Exact(t)) => Some(Some(t.clone())),
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
    if from == to || *from == T::String || *to == T::String {
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
}
