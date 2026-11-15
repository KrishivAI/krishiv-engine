// SPDX-License-Identifier: Apache-2.0
//! Connect literals to IR literals, following Spark's
//! `LiteralExpressionProtoConverter` and `LiteralValueProtoConverter`.
//!
//! A literal's type is the `data_type` field when set, else the type its
//! value implies; a value that does not fit a declared type is
//! `INCOMPATIBLE_LITERAL_DATA_TYPE`. `NULL` is compatible with any type.

use krishiv_common::error::{KrishivError, NumericValueOutOfRangeWithSuggestion};
use krishiv_plan::{DayTimeField, Literal, YearMonthField};
use krishiv_types::{ArrayType, DataType, DecimalType, MapType};

use super::errors::{
    assertion, incompatible_literal, not_set, unsupported_literal_type, unsupported_variant,
};
use super::types::{data_type, decimal_type};
use crate::proto::v4_3::{DataType as ProtoType, data_type::Kind, expression};
use expression::literal::{Decimal, LiteralType};

/// Spark's default for `DayTimeIntervalType()`: `DAY TO SECOND`.
pub(super) const DAY_TIME_DEFAULT: (i32, i32) = (0, 3);
/// Spark's default for `YearMonthIntervalType()`: `YEAR TO MONTH`.
pub(super) const YEAR_MONTH_DEFAULT: (i32, i32) = (0, 1);

/// Translates a literal.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a calendar interval, `TIME`, array,
/// map or struct literal (the IR has none yet), Spark's
/// `CONNECT_INVALID_PLAN.*` for a literal with no value or a value that is
/// out of range or does not fit its declared type.
pub(crate) fn literal(lit: &expression::Literal) -> Result<Literal, KrishivError> {
    literal_at(lit, None, 0)
}

/// A literal inside a nested one: its type is the one the enclosing type
/// gives it (`expected`), since Spark 4.1+ clients carry the whole type once, in the root's
/// `data_type`, and send no type on the members. `depth` counts nested literals.
fn literal_at(
    lit: &expression::Literal,
    expected: Option<&DataType>,
    depth: usize,
) -> Result<Literal, KrishivError> {
    let Some(value) = &lit.literal_type else {
        return Err(unsupported_literal_type("LITERALTYPE_NOT_SET(0)"));
    };
    let declared = lit.data_type.as_ref();
    let declared_kind = match declared {
        Some(d) => Some(
            d.kind
                .as_ref()
                .ok_or_else(|| not_set("spark.connect.DataType", "KIND_NOT_SET"))?,
        ),
        None => None,
    };
    let field = value.variant().field;
    // A specialized array is the array kind's compact form.
    let kind_field = declared_kind.map(|k| k.variant().field);
    if let Some(kind_field) = kind_field
        && field != "null"
        && kind_field != field
        && !(field == "specialized_array" && kind_field == "array")
    {
        return Err(incompatible_literal(kind_field, field));
    }
    Ok(match value {
        LiteralType::Null(own) => match (declared, expected) {
            (None, Some(t)) => Literal::Null(Some(t.clone()).filter(|t| *t != DataType::Null)),
            _ => null(declared.unwrap_or(own))?,
        },
        LiteralType::Boolean(v) => Literal::Boolean(*v),
        LiteralType::Byte(v) => Literal::Byte(
            i8::try_from(*v).map_err(|_| assertion("a byte literal is outside -128..=127"))?,
        ),
        LiteralType::Short(v) => Literal::Short(
            i16::try_from(*v)
                .map_err(|_| assertion("a short literal is outside the 16-bit range"))?,
        ),
        LiteralType::Integer(v) => Literal::Integer(*v),
        LiteralType::Long(v) => Literal::Long(*v),
        LiteralType::Float(v) => Literal::Float(*v),
        LiteralType::Double(v) => Literal::Double(*v),
        LiteralType::Decimal(d) => {
            let declared = match (declared_kind, expected) {
                (Some(Kind::Decimal(t)), _) => Some(decimal_type(t.precision, t.scale)?),
                (None, Some(DataType::Decimal(t))) => Some(*t),
                _ => None,
            };
            decimal(d, declared)?
        }
        LiteralType::String(v) => {
            if let Some(Kind::String(s)) = declared_kind {
                // A string literal declared with a collation is refused until Connect types are gated
                // by profile and backed by a golden (Opus review of #531): `data_type` only knows
                // the collation, and the analyzer would see the wrapped value as explicit.
                data_type(&ProtoType {
                    kind: Some(Kind::String(s.clone())),
                })?
                .refuse_collations()?;
            }
            Literal::String(v.clone())
        }
        LiteralType::Binary(v) => Literal::Binary(v.clone()),
        LiteralType::Date(v) => Literal::Date(*v),
        LiteralType::Timestamp(v) => Literal::Timestamp(*v),
        LiteralType::TimestampNtz(v) => Literal::TimestampNtz(*v),
        LiteralType::YearMonthInterval(months) => {
            let (start, end) = match (declared_kind, expected) {
                (Some(Kind::YearMonthInterval(t)), _) => {
                    let (s, e) = fields(t.start_field, t.end_field, YEAR_MONTH_DEFAULT);
                    (year_month_field(s)?, year_month_field(e)?)
                }
                (None, Some(DataType::YearMonthInterval { start, end })) => (*start, *end),
                _ => (
                    year_month_field(YEAR_MONTH_DEFAULT.0)?,
                    year_month_field(YEAR_MONTH_DEFAULT.1)?,
                ),
            };
            if start > end {
                return Err(assertion("an interval literal starts after it ends"));
            }
            Literal::YearMonthInterval {
                months: *months,
                start,
                end,
            }
        }
        LiteralType::DayTimeInterval(micros) => {
            let (start, end) = match (declared_kind, expected) {
                (Some(Kind::DayTimeInterval(t)), _) => {
                    let (s, e) = fields(t.start_field, t.end_field, DAY_TIME_DEFAULT);
                    (day_time_field(s)?, day_time_field(e)?)
                }
                (None, Some(DataType::DayTimeInterval { start, end })) => (*start, *end),
                _ => (
                    day_time_field(DAY_TIME_DEFAULT.0)?,
                    day_time_field(DAY_TIME_DEFAULT.1)?,
                ),
            };
            if start > end {
                return Err(assertion("an interval literal starts after it ends"));
            }
            Literal::DayTimeInterval {
                micros: *micros,
                start,
                end,
            }
        }
        LiteralType::CalendarInterval(i) => Literal::CalendarInterval {
            months: i.months,
            days: i.days,
            micros: i.microseconds,
        },
        LiteralType::Array(a) => {
            let root = root_type(lit, expected)?;
            array(a, root.as_ref(), depth)?
        }
        LiteralType::Map(m) => {
            let root = root_type(lit, expected)?;
            map(m, root.as_ref(), depth)?
        }
        LiteralType::Struct(s) => {
            let root = root_type(lit, expected)?;
            struct_literal(s, root.as_ref(), depth)?
        }
        LiteralType::SpecializedArray(a) => {
            let root = root_type(lit, expected)?;
            specialized(a, root.as_ref())?
        }
        LiteralType::Time(_)
        | LiteralType::TimestampNtzNanos(_)
        | LiteralType::TimestampLtzNanos(_) => {
            return Err(unsupported_variant(
                "spark.connect.Expression.Literal",
                value.variant(),
            ));
        }
    })
}

/// The type a nested literal is declared with: its own `data_type` (the root's, from a 4.1+
/// client), else the type its parent gives it.
fn root_type(
    lit: &expression::Literal,
    expected: Option<&DataType>,
) -> Result<Option<DataType>, KrishivError> {
    match &lit.data_type {
        Some(t) => Ok(Some(data_type(t)?)),
        None => Ok(expected.cloned()),
    }
}

fn too_deep(depth: usize) -> Result<(), KrishivError> {
    if depth >= DataType::MAX_NESTING_DEPTH {
        return Err(krishiv_common::error::FailedToParseTooComplex {}.into());
    }
    Ok(())
}

/// A type from a message's own type field when the root carries none: an unset one is the
/// message's missing type (golden-pending: Spark infers it from the elements).
fn own_type(t: Option<&ProtoType>, message: &str) -> Result<DataType, KrishivError> {
    match t {
        Some(t) => data_type(t),
        None => Err(assertion(format!(
            "a {message} literal needs a type (the literal's data_type or its own)"
        ))),
    }
}

/// `Literal.Array`: the element type from the declared array type, else the message's
/// `element_type` (`containsNull` true, as `ArrayType(t)`).
#[expect(
    deprecated,
    reason = "clients before Spark 4.1 type a nested literal by the message's own type fields"
)]
fn array(
    a: &expression::literal::Array,
    declared: Option<&DataType>,
    depth: usize,
) -> Result<Literal, KrishivError> {
    too_deep(depth)?;
    let ty = match declared {
        Some(DataType::Array(t)) => (**t).clone(),
        Some(other) => {
            return Err(incompatible_literal(&other.type_name(), "array"));
        }
        None => ArrayType::new(own_type(a.element_type.as_ref(), "array")?, true),
    };
    let elements = a
        .elements
        .iter()
        .map(|e| literal_at(e, Some(ty.element_type()), depth + 1))
        .collect::<Result<Vec<_>, _>>()?;
    Literal::array(ty, elements)
}

/// `Literal.Map`: keys and values are parallel lists of one length.
#[expect(
    deprecated,
    reason = "clients before Spark 4.1 type a nested literal by the message's own type fields"
)]
fn map(
    m: &expression::literal::Map,
    declared: Option<&DataType>,
    depth: usize,
) -> Result<Literal, KrishivError> {
    too_deep(depth)?;
    let ty = match declared {
        Some(DataType::Map(t)) => (**t).clone(),
        Some(other) => {
            return Err(incompatible_literal(&other.type_name(), "map"));
        }
        None => MapType::new(
            own_type(m.key_type.as_ref(), "map")?,
            own_type(m.value_type.as_ref(), "map")?,
            true,
        ),
    };
    if m.keys.len() != m.values.len() {
        return Err(assertion(
            "a map literal has a different number of keys and values",
        ));
    }
    let entries = m
        .keys
        .iter()
        .zip(&m.values)
        .map(|(k, v)| {
            Ok((
                literal_at(k, Some(ty.key_type()), depth + 1)?,
                literal_at(v, Some(ty.value_type()), depth + 1)?,
            ))
        })
        .collect::<Result<Vec<_>, KrishivError>>()?;
    Literal::key_value_map(ty, entries)
}

/// `Literal.Struct`: one element per field of the struct type.
#[expect(
    deprecated,
    reason = "clients before Spark 4.1 type a nested literal by the message's own type fields"
)]
fn struct_literal(
    s: &expression::literal::Struct,
    declared: Option<&DataType>,
    depth: usize,
) -> Result<Literal, KrishivError> {
    too_deep(depth)?;
    let ty = match declared {
        Some(DataType::Struct(t)) => t.clone(),
        Some(other) => {
            return Err(incompatible_literal(&other.type_name(), "struct"));
        }
        None => match own_type(s.struct_type.as_ref(), "struct")? {
            DataType::Struct(t) => t,
            other => return Err(incompatible_literal(&other.type_name(), "struct")),
        },
    };
    if ty.fields().len() != s.elements.len() {
        return Err(assertion(
            "a struct literal has a different number of elements and fields",
        ));
    }
    let fields = ty
        .fields()
        .iter()
        .zip(&s.elements)
        .map(|(f, e)| literal_at(e, Some(f.data_type()), depth + 1))
        .collect::<Result<Vec<_>, _>>()?;
    Literal::struct_value(ty, fields)
}

/// `Literal.SpecializedArray`: a primitive array, as an array literal of non-null elements.
/// The root type, when there is one, must be the array of that primitive.
fn specialized(
    a: &expression::literal::SpecializedArray,
    declared: Option<&DataType>,
) -> Result<Literal, KrishivError> {
    use expression::literal::specialized_array::ValueType as V;
    let Some(values) = &a.value_type else {
        return Err(not_set(
            "spark.connect.Expression.Literal.SpecializedArray",
            "VALUETYPE_NOT_SET",
        ));
    };
    let (element, elements): (DataType, Vec<Literal>) = match values {
        V::Bools(v) => (
            DataType::Boolean,
            v.values.iter().map(|x| Literal::Boolean(*x)).collect(),
        ),
        V::Ints(v) => (
            DataType::Integer,
            v.values.iter().map(|x| Literal::Integer(*x)).collect(),
        ),
        V::Longs(v) => (
            DataType::Long,
            v.values.iter().map(|x| Literal::Long(*x)).collect(),
        ),
        V::Floats(v) => (
            DataType::Float,
            v.values.iter().map(|x| Literal::Float(*x)).collect(),
        ),
        V::Doubles(v) => (
            DataType::Double,
            v.values.iter().map(|x| Literal::Double(*x)).collect(),
        ),
        V::Strings(v) => (
            DataType::String,
            v.values
                .iter()
                .map(|x| Literal::String(x.clone()))
                .collect(),
        ),
    };
    let ty = match declared {
        // The declared type may allow nulls the values do not have.
        Some(DataType::Array(t)) => (**t).clone(),
        Some(other) => {
            return Err(incompatible_literal(
                &other.type_name(),
                "specialized_array",
            ));
        }
        None => ArrayType::new(element, false),
    };
    Literal::array(ty, elements)
}

/// A null of type `of`; the `NULL` type is Spark's `void`.
fn null(of: &ProtoType) -> Result<Literal, KrishivError> {
    match &of.kind {
        Some(Kind::Null(_)) => Ok(Literal::Null(None)),
        _ => Ok(Literal::Null(Some(data_type(of)?.refuse_collations()?))),
    }
}

/// The start and end field of an interval type: both when both are given,
/// the start alone as a one-field type, the default otherwise (Spark's
/// `toCatalyst*IntervalType`).
pub(super) fn fields(start: Option<i32>, end: Option<i32>, default: (i32, i32)) -> (i32, i32) {
    match (start, end) {
        (Some(s), Some(e)) => (s, e),
        (Some(s), None) => (s, s),
        _ => default,
    }
}

pub(super) fn year_month_field(n: i32) -> Result<YearMonthField, KrishivError> {
    match n {
        0 => Ok(YearMonthField::Year),
        1 => Ok(YearMonthField::Month),
        _ => Err(assertion(
            "a year-month interval field is neither YEAR (0) nor MONTH (1)",
        )),
    }
}

pub(super) fn day_time_field(n: i32) -> Result<DayTimeField, KrishivError> {
    match n {
        0 => Ok(DayTimeField::Day),
        1 => Ok(DayTimeField::Hour),
        2 => Ok(DayTimeField::Minute),
        3 => Ok(DayTimeField::Second),
        _ => Err(assertion(
            "a day-time interval field is not DAY, HOUR, MINUTE or SECOND (0..=3)",
        )),
    }
}

/// A decimal literal. Its type is the declared `decimal(p,s)` when there is
/// one; otherwise the type the digits of `value` need, widened by the
/// precision and scale the message gives (Spark's `getProtoDataTypeDefault`).
/// The unscaled value is `value` at that scale, rounded half up if the
/// declared scale is smaller than the value's own.
fn decimal(d: &Decimal, declared: Option<DecimalType>) -> Result<Literal, KrishivError> {
    let parsed =
        parse_decimal(&d.value).ok_or_else(|| assertion("a decimal literal is not a number"))?;
    let data_type = match declared {
        Some(t) => t,
        None => {
            let widest =
                |own: i64, given: Option<i32>| given.map_or(own, |g| own.max(i64::from(g)));
            let scale = widest(parsed.scale, d.scale);
            let precision = widest(i64::from(parsed.precision), d.precision).max(scale);
            match (u32::try_from(precision), u32::try_from(scale)) {
                (Ok(p), Ok(s)) => DecimalType::try_new(p, s)?,
                _ => return Err(assertion("a decimal literal has a negative scale")),
            }
        }
    };
    let out_of_range = || {
        // The value is user input of any length: show its start.
        let shown: String = d.value.chars().take(40).collect();
        KrishivError::from(NumericValueOutOfRangeWithSuggestion {
            value: shown,
            config: "\"spark.sql.ansi.enabled\"".to_owned(),
            precision: data_type.precision().to_string(),
            scale: data_type.scale().to_string(),
        })
    };
    let magnitude = parsed.magnitude.ok_or_else(out_of_range)?;
    let target = i64::from(data_type.scale());
    let scaled = if target >= parsed.scale {
        let shift = u32::try_from(target - parsed.scale).map_err(|_| out_of_range())?;
        magnitude.checked_mul(pow10(shift).ok_or_else(out_of_range)?)
    } else {
        let shift = u32::try_from(parsed.scale - target).ok().and_then(pow10);
        // A divisor past 10^38 divides every magnitude to zero.
        Some(shift.map_or(0, |p| (magnitude + p / 2) / p))
    }
    .ok_or_else(out_of_range)?;
    // The value must fit the precision (10^p is at most 10^38).
    let limit = pow10(u32::from(data_type.precision())).ok_or_else(out_of_range)?;
    if scaled >= limit {
        return Err(out_of_range());
    }
    let unscaled = i128::try_from(scaled).map_err(|_| out_of_range())?;
    Ok(Literal::Decimal {
        unscaled: if parsed.negative { -unscaled } else { unscaled },
        data_type,
    })
}

/// `10^n`, if it fits a `u128` (up to `n = 38`).
fn pow10(n: u32) -> Option<u128> {
    10_u128.checked_pow(n)
}

/// A decimal number as `java.math.BigDecimal` reads it.
#[derive(Debug, PartialEq, Eq)]
struct Parsed {
    negative: bool,
    /// The digits as a number; `None` when there are more than 38 of them.
    magnitude: Option<u128>,
    /// `BigDecimal.precision`: the digits without leading zeros, at least 1.
    precision: u32,
    /// `BigDecimal.scale`: fraction digits less the exponent, at least 0.
    scale: i64,
}

/// Reads `[+-] digits [. digits] [e|E [+-] digits]` (or `.digits`), the
/// grammar of `new BigDecimal(String)`. `None` for anything else.
fn parse_decimal(text: &str) -> Option<Parsed> {
    let bytes = text.as_bytes();
    let (negative, mut i) = match bytes.first()? {
        b'-' => (true, 1),
        b'+' => (false, 1),
        _ => (false, 0),
    };
    // The digits are read in place, so a hostile literal of any length costs
    // no memory: `significant` counts the digits after the leading zeros and
    // `magnitude` holds them while they fit 38.
    let (mut fraction, mut seen_dot, mut any_digit) = (0_i64, false, false);
    let (mut significant, mut magnitude) = (0_usize, Some(0_u128));
    while let Some(&c) = bytes.get(i) {
        match c {
            b'0'..=b'9' => {
                any_digit = true;
                fraction += i64::from(seen_dot);
                if significant > 0 || c != b'0' {
                    significant += 1;
                    magnitude = if significant > 38 {
                        None
                    } else {
                        magnitude.map(|m| m * 10 + u128::from(c - b'0'))
                    };
                }
            }
            b'.' if !seen_dot => seen_dot = true,
            _ => break,
        }
        i += 1;
    }
    if !any_digit {
        return None;
    }
    let mut exponent = 0_i64;
    if matches!(bytes.get(i), Some(b'e' | b'E')) {
        i += 1;
        let negative_exponent = match bytes.get(i) {
            Some(b'-') => {
                i += 1;
                true
            }
            Some(b'+') => {
                i += 1;
                false
            }
            _ => false,
        };
        let start = i;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            exponent = exponent
                .checked_mul(10)?
                .checked_add(i64::from(bytes[i] - b'0'))?;
            // Java caps the exponent at an int; so does this, without overflow.
            if exponent > i64::from(i32::MAX) {
                return None;
            }
            i += 1;
        }
        if i == start {
            return None;
        }
        if negative_exponent {
            exponent = -exponent;
        }
    }
    if i != bytes.len() {
        return None;
    }
    let scale = fraction.checked_sub(exponent)?;
    let precision = u32::try_from(significant.max(1)).unwrap_or(u32::MAX);
    // Spark's `Decimal` has no negative scale: `1E+3` is `1000`, of scale 0.
    let (magnitude, precision, scale) = if scale < 0 {
        let shift = u32::try_from(scale.checked_neg()?).ok()?;
        match magnitude {
            Some(0) => (Some(0), precision, 0),
            m => (
                m.and_then(|m| m.checked_mul(pow10(shift)?)),
                precision.saturating_add(shift),
                0,
            ),
        }
    } else {
        (magnitude, precision, scale)
    };
    Some(Parsed {
        negative,
        magnitude,
        precision,
        scale,
    })
}

#[cfg(test)]
mod tests;
