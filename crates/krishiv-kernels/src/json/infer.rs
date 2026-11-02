// SPDX-License-Identifier: Apache-2.0
//! `schema_of_json` type inference: the Spark type of one JSON text per
//! row, as Spark's `JsonInferSchema` infers it.
//!
//! Integral numbers infer `BIGINT` (past `i64` a `DECIMAL(p, 0)`, past
//! precision 38 a `DOUBLE`), floating-point numbers `DOUBLE` (or
//! `DECIMAL(p, s)` with `prefersDecimal`), strings `STRING` (everything
//! with `primitivesAsString`), an empty string JSON `null`, booleans
//! `BOOLEAN`, JSON `null` `VOID`, arrays the merge of their elements
//! (always nullable) and objects structs of their fields (all nullable,
//! sorted by name as Spark's `JsonInferSchema` sorts them). Merging absorbs
//! `NULL`, widens integral pairs to `BIGINT`, mixes floating-point in as
//! `DOUBLE` (or the wider decimal with `prefersDecimal`) and falls back to
//! `STRING` for anything incompatible. An empty array infers `ARRAY<VOID>`
//! and empty input `VOID`. Nesting past 1000 (Jackson's default
//! `MAX_JSON_DEPTH`) infers `NULL` (no DDL string could name it back);
//! malformed text fails the query with `KRISHIV_MALFORMED_JSON`, as Spark
//! raises a parse error for `schema_of_json('not json')` instead of
//! returning `NULL`.

use arrow_array::Array;
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::{KrishivError, KrishivMalformedJson};

use super::JsonOptions;
use super::parse::{Error as ParseError, Lexer, RawKey, Token};

/// The inferred Arrow type of each row's JSON text (`None` for SQL `NULL`
/// input, JSON `null` infers `VOID`, and nesting past 1000 infers `NULL`;
/// malformed text is an error).
pub fn infer_json(
    json: &arrow_array::StringArray,
    opts: &JsonOptions,
) -> Result<Vec<Option<ArrowType>>, KrishivError> {
    let mut out = Vec::with_capacity(json.len());
    for i in 0..json.len() {
        if json.is_null(i) {
            out.push(None);
            continue;
        }
        let text = json.value(i);
        let mut lexer = Lexer::new(text, opts.lexer());
        match infer_top(&mut lexer, opts) {
            Ok(ty) => out.push(Some(ty)),
            Err(ParseError::TooDeep) => out.push(None),
            Err(ParseError::Malformed) => {
                return Err(KrishivMalformedJson {
                    json: text.to_owned(),
                }
                .into());
            }
        }
    }
    Ok(out)
}

/// The inferred type of one text: empty input infers `VOID`.
fn infer_top(lexer: &mut Lexer<'_>, opts: &JsonOptions) -> Result<ArrowType, ParseError> {
    lexer.skip_trivia()?;
    if lexer.peek()?.is_none() {
        return Ok(ArrowType::Null);
    }
    infer_value(lexer, 0).map(|inferred| inferred.finish(opts))
}

/// An inferred type, still merging.
#[derive(Debug, Clone)]
enum Inferred {
    /// JSON `null`.
    Null,
    /// A boolean.
    Bool,
    /// An integral number.
    Int,
    /// An integral number past `i64`: its digit precision (past 38 a
    /// `DOUBLE` finishes it instead of `DECIMAL(p, 0)`).
    BigInt(u32),
    /// A floating-point number as written (for `prefersDecimal`).
    Float(String),
    /// A string.
    Str,
    /// An array: the merged element type (`None` when empty).
    Array(Option<Box<Inferred>>),
    /// An object: the fields in order.
    Object(Vec<(String, Inferred)>),
}

impl Inferred {
    /// Merges `other` in, as Spark's `compatibleType` does.
    fn merge(&mut self, other: Self) {
        match (&mut *self, other) {
            (Self::Null, o) => *self = o,
            (_, Self::Null)
            | (Self::Int | Self::Float(_) | Self::BigInt(_), Self::Int)
            // A floating-point number absorbs an out-of-range integer, as
            // Spark's `compatibleType` mixes either into `DOUBLE` (or the
            // wider decimal with `prefersDecimal`).
            | (Self::Float(_), Self::BigInt(_)) => {}
            (Self::Int | Self::BigInt(_), Self::BigInt(p)) => {
                let wider = match &*self {
                    Self::BigInt(q) => (*q).max(p),
                    _ => p,
                };
                *self = Self::BigInt(wider);
            }
            (Self::Int | Self::BigInt(_), Self::Float(raw)) => {
                *self = Self::Float(raw);
            }
            (Self::Float(a), Self::Float(b)) => {
                // The wider decimal wins with `prefersDecimal`.
                if decimal_shape(&b) > decimal_shape(a) {
                    *a = b;
                }
            }
            (Self::Array(a), Self::Array(b)) => match (a.take(), b) {
                (None, b) => *a = b,
                (taken, None) => *a = taken,
                (Some(mut x), Some(y)) => {
                    x.merge(*y);
                    *a = Some(x);
                }
            },
            (Self::Object(a), Self::Object(b)) => {
                for (name, ty) in b {
                    match a.iter_mut().find(|(n, _)| n == &name) {
                        Some((_, slot)) => slot.merge(ty),
                        None => a.push((name, ty)),
                    }
                }
                // Spark sorts the inferred fields by name
                // (`JsonInferSchema`, `structFieldComparator`); other code
                // relies on the order, so a merge that appends fields sorts
                // again. Rust's byte order and Java's UTF-16 code-unit
                // order agree on every valid string.
                a.sort_by(|x, y| x.0.cmp(&y.0));
            }
            (this, other) => {
                if !same_shape(this, &other) {
                    *this = Self::Str;
                }
            }
        }
    }

    /// Finishes the Arrow type.
    fn finish(self, opts: &JsonOptions) -> ArrowType {
        // `primitivesAsString` reads every primitive as a string but keeps
        // the structure; `NULL` stays `VOID` either way.
        let prim = opts.primitives_as_string;
        match self {
            Self::Null => ArrowType::Null,
            Self::Bool | Self::Int | Self::BigInt(_) | Self::Float(_) | Self::Str if prim => {
                ArrowType::Utf8View
            }
            Self::Bool => ArrowType::Boolean,
            Self::Int => ArrowType::Int64,
            Self::BigInt(p) if p > 38 => ArrowType::Float64,
            Self::BigInt(p) => ArrowType::Decimal128(u8::try_from(p).unwrap_or(u8::MAX), 0),
            Self::Float(raw) => decimal_or_double(&raw, opts),
            Self::Str => ArrowType::Utf8View,
            Self::Array(None) => list_of(ArrowType::Null),
            Self::Array(Some(e)) => list_of(e.finish(opts)),
            Self::Object(fields) => struct_of(
                fields
                    .into_iter()
                    .map(|(n, t)| (n, t.finish(opts)))
                    .collect(),
            ),
        }
    }
}

/// Whether two inferred types have the same shape (merging keeps the
/// first).
fn same_shape(a: &Inferred, b: &Inferred) -> bool {
    matches!(
        (a, b),
        (Inferred::Bool, Inferred::Bool)
            | (Inferred::Int, Inferred::Int)
            | (Inferred::BigInt(_), Inferred::BigInt(_))
            | (Inferred::Float(_), Inferred::Float(_))
            | (Inferred::Str, Inferred::Str)
            | (Inferred::Array(_), Inferred::Array(_))
            | (Inferred::Object(_), Inferred::Object(_))
    )
}

/// An array of `element` (always nullable, as Spark infers).
fn list_of(element: ArrowType) -> ArrowType {
    ArrowType::List(arrow_schema::Field::new("element", element, true).into())
}

/// A struct of `fields` (all nullable, as Spark infers).
fn struct_of(fields: Vec<(String, ArrowType)>) -> ArrowType {
    ArrowType::Struct(
        fields
            .into_iter()
            .map(|(name, ty)| arrow_schema::Field::new(name, ty, true))
            .collect::<Vec<_>>()
            .into(),
    )
}

/// A floating-point literal as `DECIMAL(p, s)` (`prefersDecimal`) or
/// `DOUBLE`. Over-wide decimals fall back to `DOUBLE`.
fn decimal_or_double(raw: &str, opts: &JsonOptions) -> ArrowType {
    if super::parse::is_non_numeric_number(raw) {
        // `NaN` and the infinities have no decimal form.
        return ArrowType::Float64;
    }
    if !opts.prefers_decimal || raw.is_empty() {
        return ArrowType::Float64;
    }
    let (p, s) = decimal_shape(raw);
    let s = if s < 0 {
        // A negative scale (`1e3`) widens the precision instead.
        let widen = u32::try_from(s.checked_neg().unwrap_or(0)).unwrap_or(0);
        match p.checked_add(widen) {
            Some(p) => {
                return decimal_or_double_sized(p, 0);
            }
            None => return ArrowType::Float64,
        }
    } else {
        u32::try_from(s).unwrap_or(u32::MAX)
    };
    // Spark returns `DecimalType(max(precision, scale), scale)`.
    decimal_or_double_sized(p.max(s), s)
}

/// A decimal of precision `p` and scale `s`, or `DOUBLE` past the decimal
/// bounds.
fn decimal_or_double_sized(p: u32, s: u32) -> ArrowType {
    if p == 0 || p > 38 || s > 38 {
        ArrowType::Float64
    } else {
        ArrowType::Decimal128(
            u8::try_from(p).unwrap_or(u8::MAX),
            i8::try_from(s).unwrap_or(i8::MAX),
        )
    }
}

/// The `(precision, scale)` of a floating-point literal: the significant
/// digits and the fraction digits minus the exponent. Leading zeros never
/// count toward the precision, as Spark's `BigDecimal.precision` ignores
/// them (`0.01` has precision 1); a zero of any shape has precision 1.
fn decimal_shape(raw: &str) -> (u32, i32) {
    let bytes = raw.as_bytes();
    let digits = if bytes.first() == Some(&b'-') {
        bytes.get(1..).unwrap_or_default()
    } else {
        bytes
    };
    let (mantissa, exp) = match digits.iter().position(|&b| b == b'e' || b == b'E') {
        None => (digits, 0_i32),
        Some(e) => {
            let magnitude: i32 =
                core::str::from_utf8(digits.get(e.saturating_add(1)..).unwrap_or_default())
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            (digits.get(..e).unwrap_or_default(), magnitude)
        }
    };
    let (int, frac) = match mantissa.iter().position(|&b| b == b'.') {
        None => (mantissa, bytes.get(..0).unwrap_or_default()),
        Some(dot) => (
            mantissa.get(..dot).unwrap_or_default(),
            mantissa.get(dot.saturating_add(1)..).unwrap_or_default(),
        ),
    };
    let significant = int
        .iter()
        .chain(frac.iter())
        .filter(|&&b| b != b'.')
        .skip_while(|&&b| b == b'0')
        .count();
    // All zeros: precision 1, like `BigDecimal("0")`.
    let p = u32::try_from(significant.max(1)).unwrap_or(u32::MAX);
    let s = i32::try_from(frac.len())
        .unwrap_or(i32::MAX)
        .checked_sub(exp)
        .unwrap_or(i32::MIN);
    (p, s)
}

/// One integral literal's inferred type: `BIGINT` in range, else a
/// `DECIMAL(p, 0)` when the digit precision fits and `DOUBLE` past 38, as
/// Spark infers for a number past `Long`.
fn infer_int(raw: &str) -> Inferred {
    if raw.parse::<i64>().is_ok() {
        return Inferred::Int;
    }
    let digits = raw.strip_prefix('-').unwrap_or(raw);
    let precision =
        u32::try_from(digits.bytes().skip_while(|b| *b == b'0').count().max(1)).unwrap_or(u32::MAX);
    Inferred::BigInt(precision)
}

/// One JSON value's inferred type, nested `depth` deep.
fn infer_value(lexer: &mut Lexer<'_>, depth: usize) -> Result<Inferred, ParseError> {
    if depth > super::parse::MAX_JSON_DEPTH {
        return Err(ParseError::TooDeep);
    }
    let deeper = depth.saturating_add(1);
    match lexer.next()? {
        Some(Token::Null) => Ok(Inferred::Null),
        Some(Token::Bool(_)) => Ok(Inferred::Bool),
        Some(Token::Num(raw, false)) => Ok(infer_int(raw)),
        Some(Token::Num(raw, true)) => Ok(Inferred::Float(raw.to_owned())),
        Some(Token::Str(raw)) => {
            // An empty string infers as JSON `null`, as Spark's
            // `JsonInferSchema` maps it to `NullType`.
            if raw.raw.len() == 2 {
                Ok(Inferred::Null)
            } else {
                Ok(Inferred::Str)
            }
        }
        Some(Token::OpenBracket) => infer_array(lexer, deeper),
        Some(Token::OpenBrace) => infer_object(lexer, deeper),
        None | Some(_) => Err(ParseError::Malformed),
    }
}

/// The elements of a JSON array after its `[`, nested `deeper` deep. Its own function so that
/// [`infer_value`]'s frame stays small: the walk recurses up to `MAX_JSON_DEPTH` (1000) deep and
/// must fit the 2 MiB stack of a test or worker thread, also in sanitizer and debug builds.
#[inline(never)]
fn infer_array(lexer: &mut Lexer<'_>, deeper: usize) -> Result<Inferred, ParseError> {
    let mut merged: Option<Inferred> = None;
    if lexer.eat(Token::CloseBracket)? {
        return Ok(Inferred::Array(None));
    }
    loop {
        let element = infer_value(lexer, deeper)?;
        match merged.as_mut() {
            None => merged = Some(element),
            Some(slot) => slot.merge(element),
        }
        lexer.skip_trivia()?;
        if lexer.eat(Token::Comma)? {
            continue;
        }
        if lexer.eat(Token::CloseBracket)? {
            break;
        }
        return Err(ParseError::Malformed);
    }
    Ok(Inferred::Array(merged.map(Box::new)))
}

/// The fields of a JSON object after its `{`, nested `deeper` deep (see [`infer_array`]).
#[inline(never)]
fn infer_object(lexer: &mut Lexer<'_>, deeper: usize) -> Result<Inferred, ParseError> {
    let mut fields = Vec::new();
    if lexer.eat_byte(b'}')? {
        return Ok(Inferred::Object(fields));
    }
    loop {
        let name = match lexer.raw_key()? {
            RawKey::Quoted(raw) => lexer.unescape(raw).ok_or(ParseError::Malformed)?,
            RawKey::Bare(word) => word.to_owned(),
        };
        lexer.skip_trivia()?;
        if !lexer.eat(Token::Colon)? {
            return Err(ParseError::Malformed);
        }
        let ty = infer_value(lexer, deeper)?;
        match fields.iter_mut().find(|(n, _)| *n == name) {
            Some((_, slot)) => slot.merge(ty),
            None => fields.push((name, ty)),
        }
        lexer.skip_trivia()?;
        if lexer.eat(Token::Comma)? {
            continue;
        }
        if lexer.eat(Token::CloseBrace)? {
            break;
        }
        return Err(ParseError::Malformed);
    }
    // Spark sorts the inferred fields by name (`JsonInferSchema`,
    // `structFieldComparator`); other code relies on the order.
    // Rust's byte order and Java's UTF-16 code-unit order agree on
    // every valid string.
    fields.sort_by(|x, y| x.0.cmp(&y.0));
    Ok(Inferred::Object(fields))
}
