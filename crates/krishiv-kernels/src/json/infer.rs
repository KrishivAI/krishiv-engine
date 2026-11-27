// SPDX-License-Identifier: Apache-2.0
//! `schema_of_json` type inference: the Spark type of one JSON text per
//! row, as Spark's `JsonInferSchema` infers it.
//!
//! Integral numbers infer `BIGINT` (past `i64` a `DECIMAL(p, 0)`, past
//! precision 38 a `DOUBLE`), floating-point numbers `DOUBLE` (or
//! `DECIMAL(max(p, s), s)` with `prefersDecimal`, where a negative scale
//! fails with `NEGATIVE_SCALE_DISALLOWED`), strings `STRING`
//! (`primitivesAsString` makes numbers and booleans `STRING` too), an empty string JSON `null`, booleans
//! `BOOLEAN`, JSON `null` `VOID`, arrays the merge of their elements
//! (always nullable) and objects structs of their fields (all nullable,
//! sorted by name as Spark's `JsonInferSchema` sorts them). Merging is
//! Spark's `compatibleType`: it absorbs `NULL`, `BIGINT` with `DOUBLE` is
//! `DOUBLE`, a decimal with `DOUBLE` is `DOUBLE`, two decimals (a `BIGINT`
//! as `DECIMAL(20, 0)`) the wider range plus the wider scale (`DOUBLE` past
//! 38 digits, [`super::decimal`]), and anything incompatible `STRING`. An empty array infers `ARRAY<VOID>`
//! and empty input `VOID`. Nesting past 1000 (Jackson's default
//! `MAX_JSON_DEPTH`) infers `NULL` (no DDL string could name it back);
//! malformed text fails the query with `KRISHIV_MALFORMED_JSON`, as Spark
//! raises a parse error for `schema_of_json('not json')` instead of
//! returning `NULL`.
//!
//! A string is typed by its text as Spark's `VALUE_STRING` arm does: under
//! `prefersDecimal` a `DECIMAL(p, s)` when it reads as a decimal (commas
//! dropped, [`super::decimal::string_decimal`]; this arm also applies under
//! `primitivesAsString`), then under `inferTimestamp` a `TIME(6)`
//! (only with [`InferSession::time_type`], Spark 4.3.0), a `TIMESTAMP_NTZ`
//! (only when the session's default timestamp type is NTZ), a `TIMESTAMP`,
//! and under `timeParserPolicy=LEGACY` a `TIMESTAMP` for any zone-less
//! timestamp text (SPARK-46769). `TIMESTAMP_NTZ` and `TIMESTAMP` merge to
//! `TIMESTAMP`; a `TIME` and any other type to `STRING`. Goldens: the
//! `schema_of_json` section of `tests/fixtures/spark-{4.2.1,4.3.0}/time-text-profiles.json`
//! and `tests/fixtures/spark-{4.2.1,4.3.0}/json-prefers-decimal.json`.

use arrow_array::Array;
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::{KrishivError, KrishivMalformedJson};

use super::JsonOptions;
use super::decimal::{LONG_DECIMAL, NumberDecimal, merge_decimals, number_decimal, string_decimal};
use super::parse::{Error as ParseError, Lexer, RawKey, RawStr, Token};
use crate::cast::datetime::{FixedOffset, TimeZoneRules};

/// The session settings string inference reads besides the call's options
/// (Spark's `JsonInferSchema` reads them from `SQLConf`).
#[derive(Debug, Clone, Copy)]
pub struct InferSession<'a> {
    /// `spark.sql.timeType.enabled` under a profile whose inference knows
    /// `TIME` (Spark 4.3 and later): a time text infers `TIME(6)`.
    pub time_type: bool,
    /// `spark.sql.timestampType = TIMESTAMP_NTZ`: a zone-less timestamp text
    /// infers `TIMESTAMP_NTZ`.
    pub default_ntz: bool,
    /// The zone a zone-less timestamp text is read in (the options'
    /// `timeZone`, else the session zone).
    pub zone: &'a dyn TimeZoneRules,
    /// Today in that zone, as days since the epoch: a time-only text reads
    /// as a timestamp of today.
    pub today: i32,
}

static UTC: FixedOffset = FixedOffset(0);

impl Default for InferSession<'_> {
    /// No `TIME`, the `TIMESTAMP` default, UTC, day 0.
    fn default() -> Self {
        Self {
            time_type: false,
            default_ntz: false,
            zone: &UTC,
            today: 0,
        }
    }
}

/// The inferred Arrow type of each row's JSON text (`None` for SQL `NULL`
/// input, JSON `null` infers `VOID`, and nesting past 1000 infers `NULL`;
/// malformed text is an error), under the default [`InferSession`].
///
/// # Errors
/// `KRISHIV_MALFORMED_JSON` for malformed text.
pub fn infer_json(
    json: &arrow_array::StringArray,
    opts: &JsonOptions,
) -> Result<Vec<Option<ArrowType>>, KrishivError> {
    infer_json_with(json, opts, &InferSession::default())
}

/// [`infer_json`] under the session settings `session`.
///
/// # Errors
/// `KRISHIV_MALFORMED_JSON` for malformed text.
pub fn infer_json_with(
    json: &arrow_array::StringArray,
    opts: &JsonOptions,
    session: &InferSession<'_>,
) -> Result<Vec<Option<ArrowType>>, KrishivError> {
    let cx = Cx { opts, session };
    let mut out = Vec::with_capacity(json.len());
    for i in 0..json.len() {
        if json.is_null(i) {
            out.push(None);
            continue;
        }
        let text = json.value(i);
        let mut lexer = Lexer::new(text, opts.lexer());
        match infer_top(&mut lexer, &cx) {
            Ok(ty) => out.push(Some(ty)),
            Err(WalkError::Parse(ParseError::TooDeep)) => out.push(None),
            Err(WalkError::Parse(ParseError::Malformed)) => {
                return Err(KrishivMalformedJson {
                    json: text.to_owned(),
                }
                .into());
            }
            Err(WalkError::Type(e)) => return Err(e),
        }
    }
    Ok(out)
}

/// Why a walk stops: the text does not parse, or a value has no Spark type.
#[derive(Debug)]
enum WalkError {
    /// Malformed or too deep.
    Parse(ParseError),
    /// A type Spark refuses (`NEGATIVE_SCALE_DISALLOWED`).
    Type(KrishivError),
}

impl From<ParseError> for WalkError {
    fn from(e: ParseError) -> Self {
        Self::Parse(e)
    }
}

/// The inferred type of one text: empty input infers `VOID`.
fn infer_top(lexer: &mut Lexer<'_>, cx: &Cx<'_>) -> Result<ArrowType, WalkError> {
    lexer.skip_trivia()?;
    if lexer.peek()?.is_none() {
        return Ok(ArrowType::Null);
    }
    infer_value(lexer, 0, cx).map(Inferred::finish)
}

/// What a walk reads: the options and the session settings.
struct Cx<'a> {
    opts: &'a JsonOptions,
    session: &'a InferSession<'a>,
}

/// An inferred type, still merging.
#[derive(Debug, Clone)]
enum Inferred {
    /// JSON `null`.
    Null,
    /// A boolean.
    Bool,
    /// `BIGINT`.
    Long,
    /// `DOUBLE`.
    Double,
    /// `DECIMAL(p, s)`.
    Decimal(u8, i8),
    /// A string.
    Str,
    /// A string that parses as a time of day (`TIME(6)`).
    Time,
    /// A string that parses as a zone-less timestamp (`TIMESTAMP_NTZ`).
    Ntz,
    /// A string that parses as a timestamp (`TIMESTAMP`).
    Timestamp,
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
            // `DOUBLE` absorbs `BIGINT` and any decimal; `TIMESTAMP`
            // absorbs a `TIMESTAMP_NTZ` (`compatibleType`).
            | (Self::Double, Self::Long | Self::Decimal(..))
            | (Self::Timestamp, Self::Ntz) => {}
            (Self::Long | Self::Decimal(..), Self::Double) => *self = Self::Double,
            (Self::Long, Self::Decimal(p, s)) => *self = decimal_or_double(LONG_DECIMAL, (p, s)),
            (Self::Decimal(p, s), Self::Long) => {
                *self = decimal_or_double((*p, *s), LONG_DECIMAL);
            }
            (Self::Decimal(p, s), Self::Decimal(q, r)) => {
                *self = decimal_or_double((*p, *s), (q, r));
            }
            // Spark's `compatibleType`: either timestamp with the other is
            // `TIMESTAMP` (the reverse pair is absorbed above).
            (Self::Ntz, Self::Timestamp) => *self = Self::Timestamp,
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
    fn finish(self) -> ArrowType {
        match self {
            Self::Null => ArrowType::Null,
            Self::Bool => ArrowType::Boolean,
            Self::Long => ArrowType::Int64,
            Self::Double => ArrowType::Float64,
            Self::Decimal(p, s) => ArrowType::Decimal128(p, s),
            Self::Str => ArrowType::Utf8View,
            Self::Time => ArrowType::Time64(arrow_schema::TimeUnit::Nanosecond),
            Self::Ntz => ArrowType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
            Self::Timestamp => {
                ArrowType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into()))
            }
            Self::Array(None) => list_of(ArrowType::Null),
            Self::Array(Some(e)) => list_of(e.finish()),
            Self::Object(fields) => {
                struct_of(fields.into_iter().map(|(n, t)| (n, t.finish())).collect())
            }
        }
    }
}

/// Whether two inferred types have the same shape (merging keeps the
/// first).
fn same_shape(a: &Inferred, b: &Inferred) -> bool {
    matches!(
        (a, b),
        (Inferred::Bool, Inferred::Bool)
            | (Inferred::Long, Inferred::Long)
            | (Inferred::Double, Inferred::Double)
            | (Inferred::Str, Inferred::Str)
            | (Inferred::Time, Inferred::Time)
            | (Inferred::Ntz, Inferred::Ntz)
            | (Inferred::Timestamp, Inferred::Timestamp)
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

/// Two decimals merged ([`merge_decimals`]), `DOUBLE` past 38 digits.
fn decimal_or_double(a: (u8, i8), b: (u8, i8)) -> Inferred {
    merge_decimals(a, b).map_or(Inferred::Double, |(p, s)| Inferred::Decimal(p, s))
}

/// A JSON number's inferred type: `BIGINT` in range, else (an integral past
/// `Long`, or a floating-point number under `prefersDecimal`) the decimal
/// of [`number_decimal`]; other floating-point numbers are `DOUBLE`.
fn infer_number(raw: &str, floating: bool, opts: &JsonOptions) -> Result<Inferred, WalkError> {
    if !floating && raw.parse::<i64>().is_ok() {
        return Ok(Inferred::Long);
    }
    if floating && (!opts.prefers_decimal || super::parse::is_non_numeric_number(raw)) {
        return Ok(Inferred::Double);
    }
    match number_decimal(raw).map_err(WalkError::Type)? {
        NumberDecimal::Decimal(p, s) => Ok(Inferred::Decimal(p, s)),
        NumberDecimal::Double => Ok(Inferred::Double),
    }
}

/// One JSON value's inferred type, nested `depth` deep.
fn infer_value(lexer: &mut Lexer<'_>, depth: usize, cx: &Cx<'_>) -> Result<Inferred, WalkError> {
    if depth > super::parse::MAX_JSON_DEPTH {
        return Err(ParseError::TooDeep.into());
    }
    let deeper = depth.saturating_add(1);
    // `primitivesAsString` types numbers and booleans as strings (Spark's
    // guards before its number and boolean arms; strings keep their arm).
    let prim = cx.opts.primitives_as_string;
    match lexer.next()? {
        Some(Token::Null) => Ok(Inferred::Null),
        Some(Token::Bool(_) | Token::Num(..)) if prim => Ok(Inferred::Str),
        Some(Token::Bool(_)) => Ok(Inferred::Bool),
        Some(Token::Num(raw, floating)) => infer_number(raw, floating, cx.opts),
        Some(Token::Str(raw)) => {
            // An empty string infers as JSON `null`, as Spark's
            // `JsonInferSchema` maps it to `NullType`.
            if raw.raw.len() == 2 {
                Ok(Inferred::Null)
            } else {
                Ok(infer_string(lexer, raw, cx))
            }
        }
        Some(Token::OpenBracket) => infer_array(lexer, deeper, cx),
        Some(Token::OpenBrace) => infer_object(lexer, deeper, cx),
        None | Some(_) => Err(ParseError::Malformed.into()),
    }
}

/// The elements of a JSON array after its `[`, nested `deeper` deep. Its own function so that
/// [`infer_value`]'s frame stays small: the walk recurses up to `MAX_JSON_DEPTH` (1000) deep and
/// must fit the 2 MiB stack of a test or worker thread, also in sanitizer and debug builds.
#[inline(never)]
fn infer_array(lexer: &mut Lexer<'_>, deeper: usize, cx: &Cx<'_>) -> Result<Inferred, WalkError> {
    let mut merged: Option<Inferred> = None;
    if lexer.eat(Token::CloseBracket)? {
        return Ok(Inferred::Array(None));
    }
    loop {
        let element = infer_value(lexer, deeper, cx)?;
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
        return Err(ParseError::Malformed.into());
    }
    Ok(Inferred::Array(merged.map(Box::new)))
}

/// The fields of a JSON object after its `{`, nested `deeper` deep (see [`infer_array`]).
#[inline(never)]
fn infer_object(lexer: &mut Lexer<'_>, deeper: usize, cx: &Cx<'_>) -> Result<Inferred, WalkError> {
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
            return Err(ParseError::Malformed.into());
        }
        let ty = infer_value(lexer, deeper, cx)?;
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
        return Err(ParseError::Malformed.into());
    }
    // Spark sorts the inferred fields by name (`JsonInferSchema`,
    // `structFieldComparator`); other code relies on the order.
    // Rust's byte order and Java's UTF-16 code-unit order agree on
    // every valid string.
    fields.sort_by(|x, y| x.0.cmp(&y.0));
    Ok(Inferred::Object(fields))
}

/// A non-empty string's inferred type, as Spark's `VALUE_STRING` arm types
/// it (see the module docs). Its own function so that [`infer_value`]'s
/// frame stays small.
#[inline(never)]
fn infer_string(lexer: &Lexer<'_>, raw: RawStr<'_>, cx: &Cx<'_>) -> Inferred {
    let opts = cx.opts;
    if !opts.prefers_decimal && !opts.infer_timestamp {
        return Inferred::Str;
    }
    let Some(text) = lexer.unescape(raw) else {
        return Inferred::Str;
    };
    // `prefersDecimal` tries a decimal first.
    if opts.prefers_decimal
        && let Some((p, s)) = string_decimal(&text)
    {
        return Inferred::Decimal(p, s);
    }
    if !opts.infer_timestamp {
        return Inferred::Str;
    }
    let session = cx.session;
    // `allCatch opt timeFormatter.parse`: a failure of any kind is no time.
    if session.time_type && matches!(parse_time(&text, opts), Ok(Some(_))) {
        return Inferred::Time;
    }
    if session.default_ntz {
        let ntz = match &opts.timestamp_ntz_format {
            Some(p) => p.parse_local(&text, false).ok().flatten(),
            None => crate::text::spark_string_to_timestamp_ntz(&text),
        };
        if ntz.is_some() {
            return Inferred::Ntz;
        }
    }
    let ts = match &opts.timestamp_format {
        Some(p) => p.parse_timestamp(&text, session.zone),
        None => crate::text::spark_string_to_timestamp(&text, session.zone, session.today),
    };
    if matches!(ts, Ok(Some(_))) {
        return Inferred::Timestamp;
    }
    // SPARK-46769: under LEGACY the zone-less grammar decides `TIMESTAMP`.
    if opts.legacy_parser && crate::text::spark_string_to_timestamp_ntz(&text).is_some() {
        return Inferred::Timestamp;
    }
    Inferred::Str
}

/// A time of day under `timeFormat`, else Spark's default time grammar.
fn parse_time(text: &str, opts: &JsonOptions) -> Result<Option<i64>, KrishivError> {
    match &opts.time_format {
        Some(p) => p.parse_time_of_day_nanos(text),
        None => Ok(crate::cast::parse_time_nanos(text)),
    }
}
