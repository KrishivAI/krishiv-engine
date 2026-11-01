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
//! - Arrays, maps and structs meet element by element
//!   (`TypeCoercionHelper.findTypeForComplex`): in a conditional or set
//!   operation by the same rules recursively, in a comparison by the
//!   tightest common type only (no decimal widening, no string promotion).
//!   An element, value or field becomes nullable when either side's is or
//!   when its cast may yield NULL (`Cast.forceNullable`); map keys must not
//!   need such a cast; struct fields meet by position, with names equal
//!   ignoring case (exactly, with `spark.sql.caseSensitive`), keep the first
//!   side's name and lose their metadata.
//!
//! With `spark.sql.ansi.enabled` off Spark promotes strings differently
//! (string to `double` in arithmetic, to the other side's type in
//! comparisons); those operator coercions are not implemented and fail with
//! `KRISHIV_UNSUPPORTED_FEATURE`. Arithmetic on dates and timestamps
//! (`date_add`, intervals) is not implemented either.

use krishiv_types::{ArrayType, DataType, DecimalType, MapType, StructField, StructType};

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

/// The session settings coercion depends on.
#[derive(Clone, Copy, Debug)]
pub(super) struct Rules {
    /// `spark.sql.ansi.enabled`.
    pub(super) ansi: bool,
    /// `spark.sql.caseSensitive`: struct field names match by case when
    /// nested types meet.
    pub(super) case_sensitive: bool,
}

impl Rules {
    /// The rules of `call`.
    fn of(call: &CallTypes<'_>) -> Self {
        Self {
            ansi: call.ansi(),
            case_sensitive: call.case_sensitive(),
        }
    }
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
    /// `UNEXPECTED_INPUT_TYPE` of the day argument of `date_add` /
    /// `date_sub`, which Spark names in place of `date + n` / `date - n`.
    DateDays {
        /// Whether the call is `date_sub`.
        sub: bool,
        /// The date argument (0-based); the day count is the other one.
        date: usize,
        /// The day count's type.
        ty: Option<DataType>,
    },
    /// `date + 'text'` / `'text' + date`: Spark reads it as `date_add(a, b)`
    /// and finds the second argument is not an integral day count (a string
    /// operand has been read as a DATE).
    DateAddText,
    /// `DATA_DIFF_TYPES`: the argument types; whether Spark lists them as
    /// `["A", "B"]` (else `("A" or "B")`).
    DataDiff(Vec<Option<DataType>>, bool),
    /// `CREATE_MAP_KEY_DIFF_TYPES`: the key types of `map`.
    CreateMapKeyDiff(Vec<Option<DataType>>),
    /// `CREATE_MAP_VALUE_DIFF_TYPES`: the value types of `map`.
    CreateMapValueDiff(Vec<Option<DataType>>),
    /// `MAP_CONCAT_DIFF_TYPES`: the argument types of `map_concat`.
    MapConcatDiffTypes(Vec<Option<DataType>>),
    /// `INVALID_ORDERING_TYPE`: a type that does not order (a map).
    InvalidOrdering(DataType),
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
fn common(
    a: &Arg<'_>,
    b: &Arg<'_>,
    literal: bool,
    case_sensitive: bool,
) -> Option<Option<DataType>> {
    let rules = Rules {
        ansi: true,
        case_sensitive,
    };
    common_in(a, b, literal, rules, Nested::Tightest)
}

/// How the elements of two nested types meet.
#[derive(Clone, Copy)]
enum Nested {
    /// As the nested types do (`findWiderTypeForTwo`, conditionals and set
    /// operations).
    Wider,
    /// By the tightest common type (`findTightestCommonType`, comparisons).
    Tightest,
}

/// The type two arrays, two maps or two structs meet at
/// (`findTypeForComplex`), if any.
fn complex(x: &DataType, y: &DataType, rules: Rules, how: Nested) -> Option<DataType> {
    use DataType as T;
    let meet = |p: &DataType, q: &DataType| -> Option<DataType> {
        let met = match how {
            Nested::Wider => common_all(&[elem(p), elem(q)], rules)?,
            Nested::Tightest => tightest(p, q, rules)?,
        };
        Some(met.unwrap_or(T::Null))
    };
    // Whether a value of `from` may become NULL when cast to `to`.
    let forced = |from: &DataType, to: &DataType| *from != T::Null && force_nullable(from, to);
    match (x, y) {
        (T::Array(a), T::Array(b)) => {
            let (ea, eb) = (a.element_type(), b.element_type());
            let e = meet(ea, eb)?;
            let contains_null =
                a.contains_null() || b.contains_null() || forced(ea, &e) || forced(eb, &e);
            Some(ArrayType::new(e, contains_null).into())
        }
        (T::Map(a), T::Map(b)) => {
            let k = meet(a.key_type(), b.key_type())?;
            if forced(a.key_type(), &k) || forced(b.key_type(), &k) {
                return None;
            }
            let (va, vb) = (a.value_type(), b.value_type());
            let v = meet(va, vb)?;
            let value_null = a.value_contains_null()
                || b.value_contains_null()
                || forced(va, &v)
                || forced(vb, &v);
            Some(MapType::new(k, v, value_null).into())
        }
        (T::Struct(a), T::Struct(b)) if a.fields().len() == b.fields().len() => {
            let mut fields = Vec::with_capacity(a.fields().len());
            for (f, g) in a.fields().iter().zip(b.fields()) {
                if !super::bind::same(f.name(), g.name(), rules.case_sensitive) {
                    return None;
                }
                let t = meet(f.data_type(), g.data_type())?;
                let nullable = f.nullable()
                    || g.nullable()
                    || forced(f.data_type(), &t)
                    || forced(g.data_type(), &t);
                fields.push(StructField::new(f.name(), t, nullable));
            }
            Some(StructType::new(fields).into())
        }
        _ => None,
    }
}

/// An untyped `NULL` as an array: `array<null>`.
fn null_array() -> DataType {
    ArrayType::new(DataType::Null, true).into()
}

/// An untyped `NULL` as a map: `map<null, null>`.
fn null_map() -> DataType {
    MapType::new(DataType::Null, DataType::Null, true).into()
}

/// An element of a nested type as an argument: `void` is an untyped NULL.
fn elem(t: &DataType) -> Arg<'_> {
    Arg {
        ty: (*t != DataType::Null).then_some(t),
        nullable: true,
        literal: None,
    }
}

/// `findTightestCommonType` of two element types: equal types, NULL with
/// anything, numbers without decimals, an integral type with a decimal that
/// holds it, dates with timestamps, and nested types element by element.
fn tightest(x: &DataType, y: &DataType, rules: Rules) -> Option<Option<DataType>> {
    use DataType as T;
    match (x, y) {
        _ if x == y => Some(Some(x.clone())),
        (T::Null, t) | (t, T::Null) => Some(Some(t.clone())),
        (T::Decimal(d), i) | (i, T::Decimal(d)) => integral_digits(i)
            .filter(|digits| d.precision() - d.scale() >= *digits)
            .map(|_| Some(T::Decimal(*d))),
        (T::String | T::Binary | T::Boolean, _) | (_, T::String | T::Binary | T::Boolean) => None,
        _ => common_in(&elem(x), &elem(y), false, rules, Nested::Tightest),
    }
}

/// [`common`] under `spark.sql.ansi.enabled = ansi`: without ANSI a `float`
/// meeting an integral type stays `float`.
fn common_in(
    a: &Arg<'_>,
    b: &Arg<'_>,
    literal: bool,
    rules: Rules,
    nested: Nested,
) -> Option<Option<DataType>> {
    use DataType as T;
    let ansi = rules.ansi;
    let (x, y) = match (a.ty, b.ty) {
        (None, None) => return Some(None),
        (Some(t), None) | (None, Some(t)) => return Some(Some(t.clone())),
        (Some(x), Some(y)) => (x, y),
    };
    if x == y {
        return Some(Some(x.clone()));
    }
    if x.is_nested() || y.is_nested() {
        return complex(x, y, rules, nested).map(Some);
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
        (
            T::String,
            t
            @ (T::Boolean | T::Date | T::Timestamp | T::TimestampNtz | T::Binary | T::Time { .. }),
        )
        | (
            t
            @ (T::Boolean | T::Date | T::Timestamp | T::TimestampNtz | T::Binary | T::Time { .. }),
            T::String,
        )
        | (T::Date, t @ (T::Timestamp | T::TimestampNtz))
        | (t @ (T::Timestamp | T::TimestampNtz), T::Date) => t.clone(),
        (T::Time { precision: p }, T::Time { precision: q }) => T::Time {
            precision: *p.max(q),
        },
        // Two spatial types of one kind meet at `ANY` (their SRIDs differ).
        (T::Geometry { .. }, T::Geometry { .. }) => T::Geometry {
            srid: krishiv_types::Srid::Any,
        },
        (T::Geography { .. }, T::Geography { .. }) => T::Geography {
            srid: krishiv_types::Srid::Any,
        },
        (T::Timestamp, T::TimestampNtz) | (T::TimestampNtz, T::Timestamp) => T::Timestamp,
        _ => return None,
    };
    Some(Some(t))
}

/// The common type of all of `args` (`findWiderCommonType`: strings
/// first), `Some(None)` when every one is an untyped `NULL`.
fn common_all(args: &[Arg<'_>], rules: Rules) -> Option<Option<DataType>> {
    if !rules.ansi
        && let Some(t) = legacy_string_type(args)
    {
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
        acc = common_in(&cur, a, false, rules, Nested::Wider)?;
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
pub(super) fn common_type(
    a: &Arg<'_>,
    b: &Arg<'_>,
    case_sensitive: bool,
) -> Option<Option<DataType>> {
    common(a, b, false, case_sensitive)
}

/// The type two set-operation columns meet at (`findWiderTypeForTwo`, the
/// legacy rules with ANSI off).
pub(super) fn wider_type(a: &Arg<'_>, b: &Arg<'_>, rules: Rules) -> Option<Option<DataType>> {
    common_all(&[*a, *b], rules)
}

/// The type two inline-table values meet at without ANSI
/// (`findWiderTypeWithoutStringPromotion`): the set-operation rules, except
/// that a string never meets another type.
pub(super) fn wider_type_without_string_promotion(
    a: &Arg<'_>,
    b: &Arg<'_>,
    case_sensitive: bool,
) -> Option<Option<DataType>> {
    match (a.ty, b.ty) {
        (Some(x), Some(y)) if (x == &DataType::String) != (y == &DataType::String) => None,
        _ => {
            let rules = Rules {
                ansi: false,
                case_sensitive,
            };
            common_in(a, b, false, rules, Nested::Wider)
        }
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

/// Whether a comparison of `args` may be null: an argument may be. A null
/// struct field never makes the result null (nulls order smallest); only a
/// null struct operand does, which `nullable` already records.
fn comparison_nullable(args: &[Arg<'_>]) -> bool {
    any_nullable(args)
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
    rules: Rules,
    loss: bool,
) -> Result<Typed, Mismatch> {
    let mut typed = operator_types(op, args, rules, loss)?;
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
    rules: Rules,
    loss: bool,
) -> Result<Typed, Mismatch> {
    use Builtin as B;
    let ansi = rules.ansi;
    let boolean = |casts: Vec<Option<DataType>>, nullable: bool| Typed {
        casts,
        data_type: DataType::Boolean,
        nullable,
    };
    match (op, args) {
        (B::Eq | B::EqNullSafe | B::Lt | B::LtEq | B::Gt | B::GtEq, [a, b]) => {
            legacy_strings(args, ansi)?;
            // Types equal but for field names and nullability compare as
            // they are (Spark's `DataTypeUtils.sameType` check passes).
            let casts = if same_shape(args) {
                vec![None, None]
            } else {
                let t = common(a, b, true, rules.case_sensitive)
                    .ok_or_else(|| Mismatch::DiffTypes(a.ty.cloned(), b.ty.cloned()))?;
                vec![cast_to(a.ty, t.as_ref()), cast_to(b.ty, t.as_ref())]
            };
            if let Some(t) = a.ty.filter(|t| !orderable(t)) {
                return Err(Mismatch::InvalidOrdering(t.clone()));
            }
            Ok(boolean(
                casts,
                op != B::EqNullSafe && comparison_nullable(args),
            ))
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
                && common(a, b, false, rules.case_sensitive).is_none()
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
        (B::In, [first, ..]) => {
            legacy_strings(args, ansi)?;
            let casts = if same_shape(args) {
                vec![None; args.len()]
            } else {
                let t = common_all(args, rules).ok_or_else(|| {
                    Mismatch::DataDiff(args.iter().map(|a| a.ty.cloned()).collect(), true)
                })?;
                args.iter().map(|a| cast_to(a.ty, t.as_ref())).collect()
            };
            if let Some(t) = first.ty.filter(|t| !orderable(t)) {
                return Err(Mismatch::InvalidOrdering(t.clone()));
            }
            Ok(boolean(casts, comparison_nullable(args)))
        }
        (B::Like | B::ILike | B::RLike, _) => {
            // Only atomic types become a string implicitly.
            if let Some((i, a)) = args
                .iter()
                .enumerate()
                .find(|(_, a)| a.ty.is_some_and(DataType::is_nested))
            {
                return Err(unexpected(i, i, a.ty, r#""STRING""#));
            }
            let casts = args
                .iter()
                .map(|a| cast_to(a.ty, Some(&DataType::String)))
                .collect();
            Ok(boolean(casts, any_nullable(args)))
        }
        (B::Concat, _) => concat(args, rules),
        (B::When, _) => when(args, rules),
        _ => Err(Mismatch::Unsupported("this operator's arguments")),
    }
}

/// `CASE WHEN c THEN v ... ELSE e END`: conditions are booleans, values
/// meet at their common type.
/// Whether every typed argument is equal to the first but for field names
/// and nullability (and at least one is nested): such arguments compare as
/// they are, with no cast.
fn same_shape(args: &[Arg<'_>]) -> bool {
    let mut typed = args.iter().filter_map(|a| a.ty);
    let Some(first) = typed.next() else {
        return false;
    };
    first.is_nested() && typed.all(|t| t.equals_structurally(first, true))
}

/// Spark's `TypeUtils.isOrderable`: a map orders nowhere, and neither does
/// an array or struct holding one; a geometry or geography does not either.
pub(super) fn orderable(t: &DataType) -> bool {
    match t {
        // A geometry or geography neither orders nor compares.
        DataType::Map(_) | DataType::Geometry { .. } | DataType::Geography { .. } => false,
        DataType::Array(a) => orderable(a.element_type()),
        DataType::Struct(s) => s.fields().iter().all(|f| orderable(f.data_type())),
        _ => true,
    }
}

/// `concat`: strings (anything atomic becomes one), binaries (when every
/// argument is one), or arrays (meeting at their common type). Any other
/// nested argument is refused, as is a mix of arrays and other types.
fn concat(args: &[Arg<'_>], rules: Rules) -> Result<Typed, Mismatch> {
    const TAKES: &str = r#"("STRING" or "BINARY" or "ARRAY")"#;
    if let Some((i, a)) = args
        .iter()
        .enumerate()
        .find(|(_, a)| matches!(a.ty, Some(DataType::Map(_) | DataType::Struct(_))))
    {
        return Err(unexpected(i, i, a.ty, TAKES));
    }
    let typed = || args.iter().filter_map(|a| a.ty);
    let t = if typed().any(DataType::is_nested) {
        if !typed().all(DataType::is_nested) {
            return Err(Mismatch::DataDiff(
                args.iter().map(|a| a.ty.cloned()).collect(),
                false,
            ));
        }
        common_all(args, rules)
            .ok_or_else(|| Mismatch::DataDiff(args.iter().map(|a| a.ty.cloned()).collect(), false))?
            .unwrap_or(DataType::Null)
    } else if !args.is_empty() && args.iter().all(|a| a.ty == Some(&DataType::Binary)) {
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

fn when(args: &[Arg<'_>], rules: Rules) -> Result<Typed, Mismatch> {
    let ansi = rules.ansi;
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
    let t = common_all(&vals, rules)
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

/// What Spark says of `+` between a date or timestamp and a string: a string
/// operand is read as a number, which a timestamp does not add; a date reads it
/// as a day count.
///
/// Only with ANSI on: without it Spark reads the string as DOUBLE first and
/// reports a different error (`BINARY_OP_DIFF_TYPES`), which stays refused.
fn datetime_text_sum(a: &Arg<'_>, b: &Arg<'_>, ansi: bool) -> Option<Mismatch> {
    use DataType as T;
    if !ansi {
        return None;
    }
    let text = |t: Option<&DataType>| matches!(t, Some(T::String));
    let other = match (a.ty, b.ty) {
        (t, Some(d)) if text(t) => d,
        (Some(d), t) if text(t) => d,
        _ => return None,
    };
    match other {
        T::Date => Some(Mismatch::DateAddText),
        T::Timestamp | T::TimestampNtz => Some(Mismatch::WrongType(Some(other.clone()), ADDITIVE)),
        _ => None,
    }
}

/// What Spark says of `+` / `-` between a date or timestamp and a DECIMAL
/// (`DateTimeOperations` only takes integral days and intervals).
fn datetime_decimal(op: Builtin, a: &Arg<'_>, b: &Arg<'_>) -> Option<Mismatch> {
    use DataType as T;
    let (date_side, dec) = match (a.ty, b.ty) {
        (Some(d), Some(T::Decimal(_))) if is_datetime(Some(d)) => (0, 1),
        (Some(T::Decimal(_)), Some(d)) if is_datetime(Some(d)) => (1, 0),
        _ => return None,
    };
    let dt = [a.ty, b.ty][date_side]?;
    let is_date = *dt == T::Date;
    let dec_ty = [a.ty, b.ty][dec].cloned();
    let unexpected = |required: &str| Mismatch::Unexpected {
        arg: dec,
        param: dec,
        ty: dec_ty.clone(),
        required: required.to_owned(),
    };
    Some(match (op, is_date) {
        (Builtin::Add, false) => Mismatch::DiffTypes(a.ty.cloned(), b.ty.cloned()),
        (Builtin::Add, true) => Mismatch::DateDays {
            sub: false,
            date: date_side,
            ty: dec_ty,
        },
        (_, false) => unexpected(r#""(TIMESTAMP OR TIMESTAMP WITHOUT TIME ZONE)""#),
        (_, true) if date_side == 0 => Mismatch::DateDays {
            sub: true,
            date: 0,
            ty: dec_ty,
        },
        (_, true) => unexpected(r#""DATE""#),
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
        let text = if op == Builtin::Add {
            datetime_text_sum(a, b, ansi)
        } else {
            None
        };
        return Err(text
            .or_else(|| datetime_decimal(op, a, b))
            .unwrap_or(Mismatch::Unsupported("date and timestamp arithmetic")));
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
            .then(|| common(&of_type(x), &of_type(y), false, false))
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
        Inputs::Common => common_casts(args, args, Rules::of(call)),
        Inputs::Map => map_casts(args, Rules::of(call)),
        Inputs::MapConcat => map_concat_casts(args, Rules::of(call)),
        Inputs::MapContainsKey => map_contains_key_casts(args, call),
        Inputs::ArrayItem => Ok(array_item_casts(args, Rules::of(call))),
        Inputs::ArrayContains => Ok(array_contains_casts(args, call)),
        Inputs::ArrayPair => Ok(array_pair_casts(args, Rules::of(call))),
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
            common_casts(args, args, Rules::of(call))
        }
        Inputs::CommonWithZero => {
            let all: Vec<Arg<'_>> = args.iter().copied().chain([hidden_zero()]).collect();
            common_casts(args, &all, Rules::of(call))
        }
        Inputs::Leading(types) => {
            let n = types.len().min(args.len());
            let mut casts = declared(types, &args[..n], call.ansi())?;
            casts.extend(common_casts(&args[n..], &args[n..], Rules::of(call))?);
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
                // Nested types that differ only in field names (or
                // nullability) compare as they are, as in Spark.
                let structural =
                    matches!((a.ty, b.ty), (Some(x), Some(y)) if x.equals_structurally(y, true));
                if !legacy_boolean && !structural {
                    common_in(a, b, true, Rules::of(call), Nested::Tightest)
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
            operator_types(builtin, args, Rules::of(call), call.decimal_loss()).map(|t| t.casts)
        }
        // `equal_null`: the casts of the null-safe `<=>` operator.
        Inputs::NullSafeEqual => operator_types(
            Builtin::EqNullSafe,
            args,
            Rules::of(call),
            call.decimal_loss(),
        )
        .map(|t| t.casts),
        _ => match inputs.expected(call) {
            Some(types) => {
                let casts = declared(&types, args, call.ansi())?;
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
    rules: Rules,
) -> Result<Vec<Option<DataType>>, Mismatch> {
    let t = common_all(meet, rules)
        .ok_or_else(|| Mismatch::DataDiff(meet.iter().map(|a| a.ty.cloned()).collect(), false))?;
    Ok(args.iter().map(|a| cast_to(a.ty, t.as_ref())).collect())
}

/// `map(k1, v1, ...)`: the keys (even positions) and the values (odd) each
/// meet at their common type (`CREATE_MAP_KEY_DIFF_TYPES` and
/// `CREATE_MAP_VALUE_DIFF_TYPES` when either side has none).
fn map_casts(args: &[Arg<'_>], rules: Rules) -> Result<Vec<Option<DataType>>, Mismatch> {
    let keys: Vec<Arg<'_>> = args.iter().step_by(2).copied().collect();
    let values: Vec<Arg<'_>> = args.iter().skip(1).step_by(2).copied().collect();
    let key_type = common_all(&keys, rules)
        .ok_or_else(|| Mismatch::CreateMapKeyDiff(keys.iter().map(|a| a.ty.cloned()).collect()))?;
    let value_type = common_all(&values, rules).ok_or_else(|| {
        Mismatch::CreateMapValueDiff(values.iter().map(|a| a.ty.cloned()).collect())
    })?;
    Ok(args
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let to = if i % 2 == 0 { &key_type } else { &value_type };
            cast_to(a.ty, to.as_ref())
        })
        .collect())
}

/// `map_concat(m1, ...)`: every argument must be a map
/// (`MAP_CONCAT_DIFF_TYPES` otherwise, an untyped `NULL` included). With
/// ANSI the maps must share the type ignoring nullability; without it they
/// meet at their common type (`DATA_DIFF_TYPES` either way).
fn map_concat_casts(args: &[Arg<'_>], rules: Rules) -> Result<Vec<Option<DataType>>, Mismatch> {
    if args.iter().any(|a| !matches!(a.ty, Some(DataType::Map(_)))) {
        return Err(Mismatch::MapConcatDiffTypes(
            args.iter().map(|a| a.ty.cloned()).collect(),
        ));
    }
    if rules.ansi {
        let same = match args.first() {
            None => true,
            Some(first) => args.iter().all(|a| match (&first.ty, &a.ty) {
                (Some(x), Some(y)) => x.equals_ignore_nullability(y),
                _ => false,
            }),
        };
        if !same {
            return Err(Mismatch::DataDiff(
                args.iter().map(|a| a.ty.cloned()).collect(),
                false,
            ));
        }
        return Ok(vec![None; args.len()]);
    }
    common_casts(args, args, rules)
}

/// `map_contains_key(map, key)`: the key meets the map's key type without
/// string promotion, and both are cast to the wider key type; no casts when
/// there is none (the result rule reports it).
fn map_contains_key_casts(
    args: &[Arg<'_>],
    call: &CallTypes<'_>,
) -> Result<Vec<Option<DataType>>, Mismatch> {
    let [map, key] = args else {
        return Ok(vec![None; args.len()]);
    };
    let wider = match (&map.ty, &key.ty) {
        (Some(DataType::Map(m)), Some(k)) => {
            let wider = wider_type_without_string_promotion(
                &elem(m.key_type()),
                &elem(k),
                call.case_sensitive(),
            );
            match wider {
                Some(Some(w)) => {
                    if m.key_type().equals_ignore_nullability(&w) && !orderable(m.key_type()) {
                        return Err(Mismatch::InvalidOrdering(m.key_type().clone()));
                    }
                    Some((m.value_type().clone(), m.value_contains_null(), w))
                }
                _ => None,
            }
        }
        _ => None,
    };
    let Some((value_type, value_contains_null, wider)) = wider else {
        return Ok(vec![None; args.len()]);
    };
    let map_type = MapType::new(wider.clone(), value_type, value_contains_null);
    Ok(vec![
        cast_to(map.ty, Some(&map_type.into())),
        cast_to(key.ty, Some(&wider)),
    ])
}

/// `array_append(arr, item)` and the like: the array's element type and the
/// item meet at their tightest common type, both are cast to it, and the
/// position of `array_insert` to `INT`; no casts when they do not meet.
fn array_item_casts(args: &[Arg<'_>], rules: Rules) -> Vec<Option<DataType>> {
    let none = vec![None; args.len()];
    let (arr, pos, item) = match args {
        [a, i] => (a, None, i),
        [a, p, i] => (a, Some(p), i),
        _ => return none,
    };
    let Some(DataType::Array(a)) = arr.ty else {
        return none;
    };
    let item_ty = item.ty.unwrap_or(&DataType::Null);
    // Spark takes the tightest common type here whatever ANSI says, so an
    // `INT` array and a `FLOAT` item meet at `FLOAT`.
    let rules = Rules {
        ansi: false,
        ..rules
    };
    let Some(Some(meet)) = tightest(a.element_type(), item_ty, rules) else {
        return none;
    };
    let array: DataType = ArrayType::new(meet.clone(), a.contains_null()).into();
    let mut casts = vec![cast_to(arr.ty, Some(&array))];
    if let Some(p) = pos {
        // Only the narrower integers widen to the `INT` position; any other
        // type is left for the result rule to refuse.
        let widens = matches!(p.ty, Some(DataType::Byte | DataType::Short));
        casts.push(widens.then_some(DataType::Integer));
    }
    casts.push(cast_to(item.ty, Some(&meet)));
    casts
}

/// `array_contains(arr, value)`: the value and the element type meet at
/// their wider type without string promotion, and both are cast to it; no
/// casts when either side is an untyped `NULL` or they do not meet (the
/// result rule reports those).
fn array_contains_casts(args: &[Arg<'_>], call: &CallTypes<'_>) -> Vec<Option<DataType>> {
    let none = vec![None; args.len()];
    let [arr, value] = args else { return none };
    let (Some(DataType::Array(a)), Some(v)) = (arr.ty, value.ty) else {
        return none;
    };
    let Some(Some(wider)) = wider_type_without_string_promotion(
        &elem(a.element_type()),
        &elem(v),
        call.case_sensitive(),
    ) else {
        return none;
    };
    let array: DataType = ArrayType::new(wider.clone(), a.contains_null()).into();
    vec![
        cast_to(arr.ty, Some(&array)),
        cast_to(value.ty, Some(&wider)),
    ]
}

/// `array_union(a, b)`: both arrays take the wider of their types, holding
/// NULL when either did; no casts when they do not meet.
fn array_pair_casts(args: &[Arg<'_>], rules: Rules) -> Vec<Option<DataType>> {
    let none = vec![None; args.len()];
    let [x, y] = args else { return none };
    let (Some(DataType::Array(l)), Some(DataType::Array(r))) = (x.ty, y.ty) else {
        return none;
    };
    let Some(Some(meet)) = tightest(l.element_type(), r.element_type(), rules) else {
        return none;
    };
    let array: DataType = ArrayType::new(meet, l.contains_null() || r.contains_null()).into();
    vec![cast_to(x.ty, Some(&array)), cast_to(y.ty, Some(&array))]
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
fn declared(
    types: &[InputType],
    args: &[Arg<'_>],
    ansi: bool,
) -> Result<Vec<Option<DataType>>, Mismatch> {
    let mut casts = Vec::new();
    for (i, a) in args.iter().enumerate() {
        let want = types.get(i).or(types.last()).unwrap_or(&InputType::Any);
        let to = implicit(a.ty, want, ansi).ok_or_else(|| Mismatch::Unexpected {
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
fn implicit(from: Option<&DataType>, want: &InputType, ansi: bool) -> Option<Option<DataType>> {
    use DataType as T;
    match (from, want) {
        (_, InputType::Any)
        | (Some(DataType::Map(_)), InputType::Map)
        | (Some(DataType::Array(_)), InputType::Array) => Some(from.cloned()),
        (Some(t), InputType::Numeric { .. } | InputType::StrictNumeric) if is_numeric(t) => {
            Some(Some(t.clone()))
        }
        (None, InputType::Numeric { .. } | InputType::StrictNumeric)
        | (Some(T::String), InputType::Numeric { .. }) => Some(Some(T::Double)),
        // A type of the collection stays; another takes the first member
        // it may be cast to (Spark's `TypeCollection`), except that a number
        // is not narrowed or widened into one (`date_add(d, 1L)` and
        // `date_add(d, 1.5D)` fail, while `substring(s, 1L)` casts).
        (Some(f), InputType::OneOf(types)) if types.contains(f) => Some(Some(f.clone())),
        (_, InputType::OneOf(types)) => types.iter().find_map(|t| {
            if from.is_some_and(is_numeric) && is_numeric(t) {
                return None;
            }
            implicit(from, &InputType::Exact(t.clone()), ansi)
        }),
        // A member of the strict collection stays, as does an untyped
        // `NULL` (which becomes the first member); anything else is
        // refused without implicit casts.
        (Some(f), InputType::StrictOneOf(types)) if types.contains(f) => Some(Some(f.clone())),
        (None, InputType::StrictOneOf(types)) => types.first().map(|t| Some(t.clone())),
        (None, InputType::Exact(t) | InputType::Strict(t)) => Some(Some(t.clone())),
        (Some(f), InputType::Strict(t)) => (f == t).then(|| Some(t.clone())),
        (None, InputType::Map) => Some(Some(null_map())),
        (None, InputType::Array) => Some(Some(null_array())),
        // A nested type casts to another only element by element, where a
        // target that may not hold NULL needs a source that has none.
        (Some(f), InputType::Exact(t)) if f.is_nested() && t.is_nested() => {
            nested_cast(f, t, CastMode::Ansi).then(|| Some(t.clone()))
        }
        (Some(f), InputType::Exact(t)) => {
            // A string becomes a BOOLEAN parameter only under ANSI
            // (`months_between(a, b, 'abc')` is a `CAST_INVALID_INPUT` there and a
            // `DATATYPE_MISMATCH` without); `AND`, `OR`, `NOT` and `IF` cast it themselves.
            let from_string = is_numeric(t)
                || matches!(t, T::Date | T::Timestamp | T::TimestampNtz | T::Binary)
                || (ansi && *t == T::Boolean);
            // Only an atomic type becomes a string implicitly.
            let ok = f == t
                || (is_numeric(f) && is_numeric(t))
                || (*t == T::String && !f.is_nested())
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
    // A NULL of type `void` (an `array(NULL)` element) casts to anything.
    if *from == T::Null {
        return Ok(());
    }
    if from.is_nested() || to.is_nested() {
        // `TRY_CAST` of a nested type follows the legacy rules (Spark 4.2:
        // `TRY_CAST(array(true) AS array<timestamp>)` resolves), and Spark
        // suggests `try_cast` when only ANSI refuses the cast.
        let mode = if mode == CastMode::Try {
            CastMode::Legacy
        } else {
            mode
        };
        return if nested_cast(from, to, mode) {
            Ok(())
        } else if mode == CastMode::Ansi && nested_cast(from, to, CastMode::Legacy) {
            Err(BadCast::Func("`try_cast`"))
        } else {
            Err(BadCast::Without)
        };
    }
    // A geometry or geography casts only to a spatial type: to its own kind
    // with `ANY` (or the same SRID), and between kinds when both SRIDs are
    // the same fixed one. Nothing else (a string, binary) casts to or from it.
    if is_spatial(from) || is_spatial(to) {
        return if spatial_cast(from, to) {
            Ok(())
        } else {
            Err(BadCast::Without)
        };
    }
    if *from == T::String || *to == T::String {
        return Ok(());
    }
    // A time casts to another time, to the integral types (whole seconds)
    // and to `decimal` (seconds with the fraction).
    if matches!(from, T::Time { .. }) || matches!(to, T::Time { .. }) {
        return match (from, to) {
            (
                T::Time { .. },
                T::Time { .. } | T::Byte | T::Short | T::Integer | T::Long | T::Decimal(_),
            ) => Ok(()),
            _ => Err(BadCast::Without),
        };
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

fn is_spatial(t: &DataType) -> bool {
    matches!(t, DataType::Geometry { .. } | DataType::Geography { .. })
}

/// Spark's `Cast.canCast` between `geometry` and `geography` types.
fn spatial_cast(from: &DataType, to: &DataType) -> bool {
    use DataType as T;
    use krishiv_types::Srid;
    match (from, to) {
        // Into the same kind: to `ANY` always, otherwise only the same SRID.
        (T::Geometry { srid: f }, T::Geometry { srid: t })
        | (T::Geography { srid: f }, T::Geography { srid: t }) => *t == Srid::Any || f == t,
        // Between kinds: only equal fixed SRIDs.
        (
            T::Geometry {
                srid: Srid::Fixed(f),
            },
            T::Geography {
                srid: Srid::Fixed(t),
            },
        )
        | (
            T::Geography {
                srid: Srid::Fixed(f),
            },
            T::Geometry {
                srid: Srid::Fixed(t),
            },
        ) => f == t,
        _ => false,
    }
}

/// Spark's `Cast.canCast` / `canAnsiCast` for a cast involving a nested
/// type: to `STRING` always; between arrays, maps or structs (of as many
/// fields) element by element, where a target that may not hold NULL needs
/// a source that has none. Without ANSI a cast that may make a NULL counts
/// as one too, and a map key cast must never make one (checked against
/// Spark 4.2: `CAST(map('1', 1) AS map<int,int>)` resolves only with ANSI).
fn nested_cast(from: &DataType, to: &DataType, mode: CastMode) -> bool {
    use DataType as T;
    // `resolvableNullability`.
    let fits = |from_null: bool, to_null: bool| !from_null || to_null;
    let ok = |f: &DataType, t: &DataType| can_cast(Some(f), t, mode).is_ok();
    let legacy = mode == CastMode::Legacy;
    let force_nullable = |f: &DataType, t: &DataType| legacy && force_nullable(f, t);
    match (from, to) {
        (_, T::String) => true,
        (T::Array(a), T::Array(b)) => {
            let (f, t) = (a.element_type(), b.element_type());
            ok(f, t) && fits(a.contains_null() || force_nullable(f, t), b.contains_null())
        }
        (T::Map(a), T::Map(b)) => {
            let (fk, tk) = (a.key_type(), b.key_type());
            let (fv, tv) = (a.value_type(), b.value_type());
            ok(fk, tk)
                && !force_nullable(fk, tk)
                && ok(fv, tv)
                && fits(
                    a.value_contains_null() || force_nullable(fv, tv),
                    b.value_contains_null(),
                )
        }
        (T::Struct(a), T::Struct(b)) => {
            a.fields().len() == b.fields().len()
                && a.fields().iter().zip(b.fields()).all(|(f, t)| {
                    ok(f.data_type(), t.data_type())
                        && fits(
                            f.nullable() || force_nullable(f.data_type(), t.data_type()),
                            t.nullable(),
                        )
                })
        }
        _ => false,
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
        // The seconds of a time may not fit a `tinyint` or `smallint`.
        (T::String | T::Date, _)
        | (T::Float | T::Double, T::Timestamp)
        | (_, T::Date)
        | (T::Time { .. }, T::Byte | T::Short) => true,
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
        // A target that keeps every integer digit and every decimal loses
        // nothing (Spark types `array(0.25, 0.5)` with `containsNull =
        // false`); otherwise rounding to fewer decimals may carry into a
        // new digit, so it needs a spare integer digit.
        DataType::Decimal(d) => {
            let from_digits = d.precision() - d.scale();
            digits > from_digits || (digits >= from_digits && to.scale() >= d.scale())
        }
        t => integral_digits(t).is_some_and(|p| digits >= p),
    }
}

/// Whether `TRY_CAST` from `from` to `to` may yield NULL from a non-NULL
/// value: unless it is Spark's up-cast (`Cast.canUpCast`). A STRING is
/// valid UTF-8, so binary bytes that are not become NULL (ADR 0023;
/// Spark keeps them, and its result is not nullable).
pub(super) fn try_nullable(from: &DataType, to: &DataType) -> bool {
    use DataType as T;
    if from.is_nested() || to.is_nested() {
        return !nested_up_cast(from, to);
    }
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

/// Spark's `Cast.canUpCast` for nested types: element by element, where a
/// target that may not hold NULL needs a source that has none (to `STRING`
/// as for any type).
fn nested_up_cast(from: &DataType, to: &DataType) -> bool {
    use DataType as T;
    let fits = |from_null: bool, to_null: bool| !from_null || to_null;
    let up = |f: &DataType, t: &DataType| !try_nullable(f, t);
    match (from, to) {
        _ if from == to || *to == T::String => true,
        (T::Array(a), T::Array(b)) => {
            up(a.element_type(), b.element_type()) && fits(a.contains_null(), b.contains_null())
        }
        (T::Map(a), T::Map(b)) => {
            up(a.key_type(), b.key_type())
                && up(a.value_type(), b.value_type())
                && fits(a.value_contains_null(), b.value_contains_null())
        }
        (T::Struct(a), T::Struct(b)) => {
            a.fields().len() == b.fields().len()
                && a.fields().iter().zip(b.fields()).all(|(f, t)| {
                    up(f.data_type(), t.data_type()) && fits(f.nullable(), t.nullable())
                })
        }
        _ => false,
    }
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

/// Types as Spark's `CREATE_MAP_*_DIFF_TYPES` and `MAP_CONCAT_DIFF_TYPES`
/// list them: `["INT", "STRING"]`.
pub(super) fn list_types(types: &[Option<DataType>]) -> String {
    let types: Vec<String> = types.iter().map(|t| quoted_type(t.as_ref())).collect();
    format!("[{}]", types.join(", "))
}

#[cfg(test)]
mod tests {
    use krishiv_types::DataType as T;

    use super::super::catalog::ArgType;
    use super::*;

    /// `implicit` under ANSI, where a string may reach a BOOLEAN parameter.
    fn implicit(from: Option<&DataType>, want: &InputType) -> Option<Option<DataType>> {
        super::implicit(from, want, true)
    }

    #[test]
    fn a_string_reaches_a_boolean_parameter_only_under_ansi() {
        let want = InputType::Exact(T::Boolean);
        assert_eq!(
            super::implicit(Some(&T::String), &want, true),
            Some(Some(T::Boolean))
        );
        assert_eq!(super::implicit(Some(&T::String), &want, false), None);
    }

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
            operator(Builtin::Not, &[int, int], rules(true), true).err(),
            Some(Mismatch::Unsupported("this operator's arguments"))
        );
        assert_eq!(ordinal(3), "4th");
    }

    fn rules(ansi: bool) -> Rules {
        Rules {
            ansi,
            case_sensitive: false,
        }
    }

    fn arg(t: Option<&T>) -> Arg<'_> {
        Arg {
            ty: t,
            nullable: false,
            literal: None,
        }
    }

    fn nullable_arg(t: Option<&T>) -> Arg<'_> {
        Arg {
            ty: t,
            nullable: true,
            literal: None,
        }
    }

    /// `+` and `-` between a date or timestamp and a DECIMAL fail with the
    /// class Spark's `DateTimeOperations` gives each shape.
    #[test]
    fn datetime_and_decimal_arithmetic_mismatches() {
        let dec = T::Decimal(DecimalType::SYSTEM_DEFAULT);
        let shapes = |op, l: &T, r: &T| datetime_decimal(op, &arg(Some(l)), &arg(Some(r)));
        for ts in [T::Timestamp, T::TimestampNtz] {
            assert!(matches!(
                shapes(Builtin::Add, &ts, &dec),
                Some(Mismatch::DiffTypes(..))
            ));
            assert!(matches!(
                shapes(Builtin::Subtract, &dec, &ts),
                Some(Mismatch::Unexpected {
                    arg: 0,
                    param: 0,
                    ..
                })
            ));
        }
        assert!(matches!(
            shapes(Builtin::Add, &dec, &T::Date),
            Some(Mismatch::DateDays {
                sub: false,
                date: 1,
                ..
            })
        ));
        assert!(matches!(
            shapes(Builtin::Subtract, &T::Date, &dec),
            Some(Mismatch::DateDays {
                sub: true,
                date: 0,
                ..
            })
        ));
        assert!(shapes(Builtin::Add, &T::Date, &T::Integer).is_none());
    }

    /// `+` between a string and a date or timestamp: a timestamp rejects the
    /// number the string is read as, a date reads it as `date_add`'s days.
    #[test]
    fn datetime_and_string_sums() {
        let text = T::String;
        let sum = |l: &T, r: &T| datetime_text_sum(&arg(Some(l)), &arg(Some(r)), true);
        for ts in [T::Timestamp, T::TimestampNtz] {
            assert_eq!(
                sum(&text, &ts),
                Some(Mismatch::WrongType(Some(ts.clone()), ADDITIVE))
            );
            assert_eq!(
                sum(&ts, &text),
                Some(Mismatch::WrongType(Some(ts.clone()), ADDITIVE))
            );
        }
        assert_eq!(sum(&T::Date, &text), Some(Mismatch::DateAddText));
        assert_eq!(sum(&text, &T::Date), Some(Mismatch::DateAddText));
        assert!(sum(&T::Date, &T::Integer).is_none());
        assert!(sum(&text, &T::Integer).is_none());
        // Without ANSI, Spark's error differs: it stays refused.
        let off = |l: &T, r: &T| datetime_text_sum(&arg(Some(l)), &arg(Some(r)), false);
        assert!(off(&T::Date, &text).is_none());
        assert!(off(&text, &T::Timestamp).is_none());
    }

    /// Struct comparisons are NULL only when a whole struct operand is NULL:
    /// nullable fields never make `=`, `<=>`, `<` and friends nullable.
    #[test]
    fn struct_comparison_nullability_ignores_fields() {
        let struct_ty = T::Struct(StructType::new([
            StructField::new("a", T::Integer, true),
            StructField::new("b", T::Integer, true),
        ]));
        let non_null = [arg(Some(&struct_ty)), arg(Some(&struct_ty))];
        for op in [
            Builtin::Eq,
            Builtin::Lt,
            Builtin::LtEq,
            Builtin::Gt,
            Builtin::GtEq,
        ] {
            let typed = operator(op, &non_null, rules(true), true).expect("struct comparison");
            assert!(!typed.nullable, "{op:?} with non-null structs");
        }
        // `<=>` is never NULL, even for nullable structs.
        let nullable = [
            nullable_arg(Some(&struct_ty)),
            nullable_arg(Some(&struct_ty)),
        ];
        let typed = operator(Builtin::EqNullSafe, &nullable, rules(true), true)
            .expect("null-safe struct comparison");
        assert!(!typed.nullable);
        // Other comparisons are NULL when a whole struct operand is.
        let typed =
            operator(Builtin::Eq, &nullable, rules(true), true).expect("nullable struct eq");
        assert!(typed.nullable);
        // A nullable primitive operand still makes the comparison nullable.
        let ints = [arg(Some(&T::Integer)), nullable_arg(Some(&T::Integer))];
        let typed = operator(Builtin::Eq, &ints, rules(true), true).expect("nullable int eq");
        assert!(typed.nullable);
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

    /// Nested types meet as Spark 4.2 typed `coalesce` (and `UNION ALL`) of
    /// `array(..)`, `map(..)` and `named_struct(..)` values on the oracle,
    /// with `spark.sql.ansi.enabled` on and off (types as Spark's JSON).
    #[test]
    fn nested_types_meet_element_by_element() {
        let t = |json: &str| T::from_json(json).unwrap_or_else(|e| panic!("{json}: {e}"));
        let arr = |e: &str, n: bool| {
            format!(r#"{{"type":"array","elementType":{e},"containsNull":{n}}}"#)
        };
        let map = |k: &str, v: &str, n: bool| {
            format!(r#"{{"type":"map","keyType":{k},"valueType":{v},"valueContainsNull":{n}}}"#)
        };
        let st = |fields: &[(&str, &str, bool, &str)]| {
            let fields: Vec<String> = fields
                .iter()
                .map(|(name, ty, n, meta)| {
                    format!(r#"{{"name":"{name}","type":{ty},"nullable":{n},"metadata":{meta}}}"#)
                })
                .collect();
            format!(r#"{{"type":"struct","fields":[{}]}}"#, fields.join(","))
        };
        let wider = |a: &str, b: &str, ansi: bool| {
            let (a, b) = (t(a), t(b));
            wider_type(&arg(Some(&a)), &arg(Some(&b)), rules(ansi)).map(|r| r.map(|t| t.json()))
        };
        let int = r#""integer""#;
        let long = r#""long""#;
        let string = r#""string""#;
        let cases: &[(String, String, bool, Option<String>)] = &[
            (
                arr(int, false),
                arr(long, false),
                true,
                Some(arr(long, false)),
            ),
            (arr(int, false), arr(int, true), true, Some(arr(int, true))),
            // A string element cast to bigint may fail, so may be NULL.
            (
                arr(int, false),
                arr(string, false),
                true,
                Some(arr(long, true)),
            ),
            (
                arr(int, false),
                arr(string, false),
                false,
                Some(arr(string, false)),
            ),
            (
                arr(int, false),
                arr(r#""decimal(2,1)""#, false),
                true,
                Some(arr(r#""decimal(11,1)""#, false)),
            ),
            (
                arr(int, false),
                arr(r#""float""#, false),
                true,
                Some(arr(r#""double""#, false)),
            ),
            (
                arr(int, false),
                arr(r#""float""#, false),
                false,
                Some(arr(r#""float""#, false)),
            ),
            (arr(int, false), arr(r#""date""#, false), true, None),
            (
                arr(r#""date""#, false),
                arr(r#""timestamp""#, false),
                true,
                Some(arr(r#""timestamp""#, false)),
            ),
            (
                arr(r#""void""#, true),
                arr(int, false),
                true,
                Some(arr(int, true)),
            ),
            (arr(int, false), int.to_owned(), true, None),
            (
                arr(&arr(int, false), false),
                arr(&arr(long, false), false),
                true,
                Some(arr(&arr(long, false), false)),
            ),
            (
                map(int, string, false),
                map(long, string, false),
                true,
                Some(map(long, string, false)),
            ),
            (
                map(int, string, false),
                map(int, int, false),
                true,
                Some(map(int, long, true)),
            ),
            (
                map(int, string, false),
                map(int, int, false),
                false,
                Some(map(int, string, false)),
            ),
            // A key that may become NULL is no key.
            (map(string, int, false), map(int, int, false), true, None),
            (
                st(&[("a", int, false, "{}")]),
                st(&[("a", long, false, "{}")]),
                true,
                Some(st(&[("a", long, false, "{}")])),
            ),
            (
                st(&[("a", int, false, "{}")]),
                st(&[("A", long, false, "{}")]),
                true,
                Some(st(&[("a", long, false, "{}")])),
            ),
            (
                st(&[("A", int, false, "{}")]),
                st(&[("a", long, false, "{}")]),
                true,
                Some(st(&[("A", long, false, "{}")])),
            ),
            (
                st(&[("a", int, false, "{}")]),
                st(&[("a", int, true, "{}")]),
                true,
                Some(st(&[("a", int, true, "{}")])),
            ),
            (
                st(&[("a", int, false, "{}")]),
                st(&[("b", long, false, "{}")]),
                true,
                None,
            ),
            (
                st(&[("a", int, false, "{}"), ("b", int, false, "{}")]),
                st(&[("a", int, false, "{}")]),
                true,
                None,
            ),
            // Metadata survives only when the types are equal.
            (
                st(&[("a", int, true, r#"{"comment":"x"}"#)]),
                st(&[("a", int, true, r#"{"comment":"y"}"#)]),
                true,
                Some(st(&[("a", int, true, "{}")])),
            ),
            (
                st(&[("a", int, true, r#"{"comment":"x"}"#)]),
                st(&[("a", int, true, r#"{"comment":"x"}"#)]),
                true,
                Some(st(&[("a", int, true, r#"{"comment":"x"}"#)])),
            ),
            (
                st(&[("a", int, true, r#"{"comment":"x"}"#)]),
                st(&[("a", long, true, "{}")]),
                true,
                Some(st(&[("a", long, true, "{}")])),
            ),
        ];
        for (a, b, ansi, want) in cases {
            let want = want.as_ref().map(|w| Some(t(w).json()));
            assert_eq!(wider(a, b, *ansi), want, "{a} | {b} (ansi {ansi})");
        }
        // Comparisons meet by the tightest common type only.
        let compare = |a: &str, b: &str| {
            let (a, b) = (t(a), t(b));
            common(&arg(Some(&a)), &arg(Some(&b)), false, false).map(|r| r.map(|t| t.json()))
        };
        assert_eq!(
            compare(&arr(int, false), &arr(long, false)),
            Some(Some(arr(long, false)))
        );
        assert_eq!(compare(&arr(int, false), &arr(string, false)), None);
        assert_eq!(
            compare(&arr(int, false), &arr(r#""decimal(2,1)""#, false)),
            None
        );
        assert_eq!(
            compare(&arr(int, false), &arr(r#""decimal(20,0)""#, false)),
            Some(Some(arr(r#""decimal(20,0)""#, false)))
        );
    }

    /// Casts with nested types as Spark 4.2 resolves them on the oracle
    /// (`SELECT CAST(...)`, ANSI on and off).
    #[test]
    fn nested_casts_follow_spark() {
        let t = |s: &str| krishiv_types::parse_data_type(s).unwrap();
        let not_null_a =
            |ty: T| -> T { StructType::new([StructField::new("a", ty, false)]).into() };
        type Outcome = Result<(), BadCast>;
        let cases: Vec<(T, T, Outcome, Outcome)> = vec![
            // (from, to, ANSI, legacy)
            (t("array<string>"), t("array<int>"), Ok(()), Ok(())),
            (
                T::String,
                t("array<int>"),
                Err(BadCast::Without),
                Err(BadCast::Without),
            ),
            (t("array<int>"), T::String, Ok(()), Ok(())),
            (
                t("map<string,int>"),
                t("map<int,int>"),
                Ok(()),
                Err(BadCast::Without),
            ),
            (t("map<int,string>"), t("map<bigint,int>"), Ok(()), Ok(())),
            (t("struct<a:int>"), t("struct<b:bigint>"), Ok(()), Ok(())),
            (
                t("struct<a:int>"),
                t("struct<a:int,b:int>"),
                Err(BadCast::Without),
                Err(BadCast::Without),
            ),
            (
                t("struct<a:int>"),
                not_null_a(T::Integer),
                Err(BadCast::Without),
                Err(BadCast::Without),
            ),
            (
                not_null_a(T::Integer),
                not_null_a(T::Integer),
                Ok(()),
                Ok(()),
            ),
            (
                not_null_a(T::String),
                not_null_a(T::Integer),
                Ok(()),
                Err(BadCast::Without),
            ),
            (
                t("array<int>"),
                t("array<date>"),
                Err(BadCast::Without),
                Err(BadCast::Without),
            ),
            (
                t("array<boolean>"),
                t("array<timestamp>"),
                Err(BadCast::Func("`try_cast`")),
                Ok(()),
            ),
            (
                t("array<int>"),
                t("array<binary>"),
                Err(BadCast::Func("`try_cast`")),
                Ok(()),
            ),
            (
                t("array<date>"),
                t("array<int>"),
                Err(BadCast::Func("`try_cast`")),
                Ok(()),
            ),
            (
                t("array<int>"),
                T::Integer,
                Err(BadCast::Without),
                Err(BadCast::Without),
            ),
            (
                t("array<int>"),
                t("map<int,int>"),
                Err(BadCast::Without),
                Err(BadCast::Without),
            ),
            (t("array<void>"), t("array<date>"), Ok(()), Ok(())),
            (t("array<void>"), t("array<array<int>>"), Ok(()), Ok(())),
        ];
        for (from, to, ansi, legacy) in cases {
            assert_eq!(
                can_cast(Some(&from), &to, CastMode::Ansi),
                ansi,
                "{from} -> {to} (ANSI)"
            );
            assert_eq!(
                can_cast(Some(&from), &to, CastMode::Legacy),
                legacy,
                "{from} -> {to}"
            );
        }
        assert_eq!(
            can_cast(
                Some(&t("array<boolean>")),
                &t("array<timestamp>"),
                CastMode::Try
            ),
            Ok(())
        );
    }
}
