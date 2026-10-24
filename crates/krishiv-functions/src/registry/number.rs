// SPDX-License-Identifier: Apache-2.0
//! The number-formatting functions' analysis signatures: `hex`/`unhex`
//! (long/binary/string over bytes), `bin` (a long), `conv` (a string by
//! two ints), `format_number` (a number by an int or a `DecimalFormat`
//! pattern), `to_number`/`try_to_number` (a string by an Oracle pattern,
//! a `decimal`) and `to_char`/`to_varchar` (a decimal by an Oracle
//! pattern).
//!
//! Ground truth: Spark 4.2's `mathExpressions.scala` (`Conv`, `Bin`,
//! `Hex`, `Unhex`), `stringExpressions.scala` (`FormatNumber`),
//! `numberFormatExpressions.scala` (`ToNumberBase`, `ToCharacter`,
//! `ToCharacterBuilder`), the `math.sql`, `string-functions.sql`,
//! `charvarchar.sql` and `collations-basic.sql` goldens and the
//! `math`/`string` matrix families.

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int32Type};
use arrow_array::{Array, ArrayRef};
use std::sync::Arc;

use krishiv_common::error::{
    DatatypeMismatchNonFoldableInput, DatatypeMismatchUnexpectedInputType,
    InvalidParameterValueBinaryFormat, KrishivError,
};
use krishiv_kernels::cast::{CastMode, CastOptions, TimeZoneRules, cast};
use krishiv_kernels::number_format::parse_format;
use krishiv_types::{DataType, DecimalType};

use super::defs::{Spec, exact, scalar, typed};
use crate::signature::{ArgType, CallTypes, Constant, InputBuilder, InputType, Inputs};

/// A type as errors quote it (`"VOID"` for untyped `NULL`).
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

/// `DATATYPE_MISMATCH.NON_FOLDABLE_INPUT` for the `format` of `function`.
fn non_foldable_format(c: &CallTypes<'_>, function: &str) -> KrishivError {
    let args: Vec<String> = (0..c.args().len())
        .map(|n| c.sql(n).unwrap_or_default())
        .collect();
    DatatypeMismatchNonFoldableInput {
        sql_expr: format!("\"{function}({})\"", args.join(", ")),
        input_name: "`format`".to_owned(),
        input_type: quoted(c.data_type(1)),
        input_expr: format!("\"{}\"", c.sql(1).unwrap_or_default()),
    }
    .into()
}

// ---- `hex`, `unhex`, `bin`, `conv` ----

/// `hex(x)`: numbers become a `bigint`, strings and binaries stay,
/// anything else Spark renders as a string.
fn hex_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    match c.data_type(0) {
        None
        | Some(
            DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::Float
            | DataType::Double
            | DataType::Decimal(_),
        ) => vec![exact(DataType::Long)],
        Some(DataType::String | DataType::Binary) => {
            vec![exact(c.data_type(0).cloned().unwrap_or(DataType::String))]
        }
        // `DataType` is non-exhaustive: booleans, dates and timestamps
        // (and any future type) render as strings.
        _ => vec![exact(DataType::String)],
    }
}

/// `hex(x)` is a string.
fn hex_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::String, c)
}

/// `unhex(x)` is binary, always nullable (invalid input is NULL).
fn unhex(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Binary, true))
}

/// `bin(x)` is a string; the `bigint` cast may fail.
fn bin(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::String, c)
}

/// `conv(n, from, to)` is a string, always nullable (bad bases are NULL).
fn conv(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::String, true))
}

// ---- `format_number` ----

/// Whether `t` is a numeric type.
fn is_numeric(t: Option<&DataType>) -> bool {
    matches!(
        t,
        Some(
            DataType::Byte
                | DataType::Short
                | DataType::Integer
                | DataType::Long
                | DataType::Float
                | DataType::Double
                | DataType::Decimal(_)
        )
    )
}

/// `format_number(x, d)` is a string, always nullable.
fn format_number_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if !is_numeric(c.data_type(0)) && c.data_type(0).is_some() {
        return Err(unexpected(c, "format_number", 0, "\"NUMERIC\""));
    }
    match c.data_type(1) {
        None | Some(DataType::Integer | DataType::String) => {}
        _ => return Err(unexpected(c, "format_number", 1, "(\"INT\" or \"STRING\")")),
    }
    Ok(ArgType::new(DataType::String, true))
}

// ---- `to_number`, `try_to_number`, `to_char`, `to_varchar` ----

/// Unquotes our pretty-printed string literal (`'a\'b'` to `a'b`).
fn unquote(text: &str) -> Option<String> {
    let inner = text.strip_prefix('\'')?.strip_suffix('\'')?;
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next()? {
            '\\' => out.push('\\'),
            '\'' => out.push('\''),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            'u' => {
                let hex: String = chars.by_ref().take(4).collect();
                out.push(char::from_u32(u32::from_str_radix(&hex, 16).ok()?)?);
            }
            _ => return None,
        }
    }
    Some(out)
}

/// A plain numeric literal, without our type suffix (`1L` to `1`).
fn plain_number(text: &str) -> Option<&str> {
    let body = text.strip_suffix(['Y', 'S', 'L', 'F', 'D']).unwrap_or(text);
    if body.is_empty() {
        return None;
    }
    let digits = body.strip_prefix(['+', '-']).unwrap_or(body);
    if digits.is_empty()
        || !digits.bytes().all(|b| b.is_ascii_digit() || b == b'.')
        || digits == "."
    {
        return None;
    }
    Some(body)
}

/// The string value of a `DATE '...'` literal rendering.
fn date_inner(text: &str) -> Option<&str> {
    let inner = text.strip_prefix("DATE '")?.strip_suffix('\'')?;
    Some(inner)
}

/// The lossy UTF-8 text of an `X'...'` literal rendering.
fn hex_inner(text: &str) -> Option<String> {
    let hex = text.strip_prefix("X'")?.strip_suffix('\'')?;
    if !hex.len().is_multiple_of(2) || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let bytes: Vec<u8> = hex
        .as_bytes()
        .chunks(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok())
        .collect::<Option<_>>()?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

/// The string value of the format argument: `None` for a NULL format.
/// Rendering follows Spark's implicit cast to string (`1L` to `"1"`,
/// `true` to `"true"`, dates and binaries as their text). `function` is
/// the canonical name for `NON_FOLDABLE_INPUT` errors.
fn format_string(c: &CallTypes<'_>, function: &str) -> Result<Option<String>, KrishivError> {
    // Literals (and casts of literals the folder folds) first.
    match c.constant(1) {
        Some(Constant::Null) => return Ok(None),
        Some(Constant::String(s)) => return Ok(Some(s.clone())),
        Some(Constant::Integral(n)) => return Ok(Some(n.to_string())),
        Some(Constant::Boolean(b)) => {
            return Ok(Some(if *b { "true" } else { "false" }.to_owned()));
        }
        Some(Constant::Error(e)) => {
            // A cast that failed while folding may still render (a date or
            // binary our folder does not fold); otherwise raise it.
            if let Some(rendered) = rendered_format(c) {
                return Ok(Some(rendered));
            }
            return Err(e.clone());
        }
        // `Fractional` lost the fraction while folding for `round`'s
        // scale; the rendering below recovers it.
        Some(Constant::Fractional(_)) | None => {}
    }
    if let Some(rendered) = rendered_format(c) {
        return Ok(Some(rendered));
    }
    // `CAST(NULL AS <type>)` folds to nothing our folder keeps, but is a
    // NULL format all the same.
    if c.sql(1)
        .is_some_and(|sql| sql.starts_with("CAST(NULL AS ") && sql.ends_with(')'))
    {
        return Ok(None);
    }
    Err(non_foldable_format(c, function))
}

/// Renders the format argument from our pretty SQL text.
fn rendered_format(c: &CallTypes<'_>) -> Option<String> {
    let text = c.sql(1)?;
    if let Some(body) = plain_number(&text) {
        return Some(body.to_owned());
    }
    if text.starts_with('\'') {
        return unquote(&text);
    }
    if let Some(inner) = date_inner(&text) {
        return Some(inner.to_owned());
    }
    if let Some(text) = hex_inner(&text) {
        return Some(text);
    }
    // `CAST(<literal> AS <type>)`: the cast result, through our own cast
    // kernels (dates, binaries and numbers render as Spark renders them).
    cast_render(&text)
}

/// Renders `CAST(<literal> AS <type>)` via the cast kernels.
fn cast_render(text: &str) -> Option<String> {
    let inner = text.strip_prefix("CAST(")?.strip_suffix(')')?;
    let at = inner.rfind(" AS ")?;
    let (value, to) = (&inner[..at], inner[at + 4..].trim());
    let source: ArrayRef = if value.starts_with('\'') {
        let s = unquote(value)?;
        Arc::new(arrow_array::StringArray::from(vec![s.as_str()]))
    } else if let Some(body) = plain_number(value) {
        if body.contains('.') {
            Arc::new(arrow_array::Float64Array::from(vec![
                body.parse::<f64>().ok()?,
            ]))
        } else {
            Arc::new(arrow_array::Int64Array::from(vec![
                body.parse::<i64>().ok()?,
            ]))
        }
    } else {
        Arc::new(arrow_array::StringArray::from(vec![
            hex_inner(value)?.as_str(),
        ]))
    };
    let target = match to {
        "TINYINT" => arrow_schema::DataType::Int8,
        "SMALLINT" => arrow_schema::DataType::Int16,
        "INT" => arrow_schema::DataType::Int32,
        "BIGINT" => arrow_schema::DataType::Int64,
        "FLOAT" => arrow_schema::DataType::Float32,
        "DOUBLE" => arrow_schema::DataType::Float64,
        "STRING" => arrow_schema::DataType::Utf8,
        t if t.starts_with("DECIMAL(") => {
            let (p, s) = t
                .strip_prefix("DECIMAL(")?
                .strip_suffix(')')?
                .split_once(',')?;
            arrow_schema::DataType::Decimal128(p.trim().parse().ok()?, s.trim().parse().ok()?)
        }
        _ => return None,
    };
    let zone = krishiv_kernels::cast::FixedOffset(0);
    let out = cast(
        source.as_ref(),
        &target,
        &CastOptions {
            mode: CastMode::Ansi,
            time_zone: &zone as &dyn TimeZoneRules,
        },
    )
    .ok()?;
    if out.null_count() > 0 {
        return None;
    }
    Some(render_literal(&out, to))
}

/// Renders a cast result as Spark's implicit cast to string does.
fn render_literal(out: &ArrayRef, to: &str) -> String {
    use arrow_schema::DataType as Arrow;
    match out.data_type() {
        Arrow::Int8 => out
            .as_primitive::<arrow_array::types::Int8Type>()
            .value(0)
            .to_string(),
        Arrow::Int16 => out
            .as_primitive::<arrow_array::types::Int16Type>()
            .value(0)
            .to_string(),
        Arrow::Int32 => out.as_primitive::<Int32Type>().value(0).to_string(),
        Arrow::Int64 => out
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(0)
            .to_string(),
        Arrow::Float32 => {
            let v = out.as_primitive::<Float32Type>().value(0);
            render_float(f64::from(v), to)
        }
        Arrow::Float64 => render_float(out.as_primitive::<Float64Type>().value(0), to),
        Arrow::Decimal128(_, s) => {
            let v = out
                .as_primitive::<arrow_array::types::Decimal128Type>()
                .value(0);
            render_decimal(v, *s)
        }
        Arrow::Utf8 => out.as_string::<i32>().value(0).to_owned(),
        _ => String::new(),
    }
}

/// A float in Spark's shortest rendering (`1.5`, never `1.5F`).
fn render_float(v: f64, _to: &str) -> String {
    if v.fract() == 0.0 && v.abs() < 1e21 {
        format!("{v:.1}")
    } else {
        format!("{v:?}")
    }
}

/// A decimal as plain text (`1.50`, `-0.5`).
fn render_decimal(unscaled: i128, scale: i8) -> String {
    let (neg, abs) = if unscaled < 0 {
        (true, unscaled.unsigned_abs())
    } else {
        (false, unscaled.cast_unsigned())
    };
    let digits = abs.to_string();
    let s = usize::try_from(scale.max(0)).unwrap_or(0);
    let text = if s == 0 {
        digits
    } else if digits.len() > s {
        let (i, f) = digits.split_at(digits.len() - s);
        format!("{i}.{f}")
    } else {
        format!("0.{:0>width$}", digits, width = s)
    };
    format!("{}{text}", if neg { "-" } else { "" })
}

/// The decimal type of `to_number`/`try_to_number` for `format`, or the
/// pattern's analysis error. `None` (a NULL format) is Spark's
/// `DecimalType.USER_DEFAULT`, `decimal(10,0)`.
fn number_type(format: Option<&str>) -> Result<DecimalType, KrishivError> {
    match format {
        None => Ok(DecimalType::USER_DEFAULT),
        Some(pattern) => {
            let parsed = parse_format(pattern);
            parsed.validate(pattern)?;
            let (p, s) = parsed.decimal_type();
            Ok(DecimalType::try_new(p, s)?)
        }
    }
}

/// `to_number(x, format)`: the pattern's decimal type. NULL only for NULL
/// input (a mismatch raises).
fn to_number(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let format = format_string(c, "to_number")?;
    let decimal = number_type(format.as_deref())?;
    Ok(ArgType::new(DataType::Decimal(decimal), c.any_nullable()))
}

/// `try_to_number(x, format)`: the pattern's decimal type, always
/// nullable (a mismatch is NULL).
fn try_to_number(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let format = format_string(c, "try_to_number")?;
    let decimal = number_type(format.as_deref())?;
    Ok(ArgType::new(DataType::Decimal(decimal), true))
}

/// The decimal type `to_char` casts its first argument to (Spark's
/// `DecimalType.forType`): integrals to their digits, floats to
/// `decimal(14,7)`, doubles to `decimal(30,15)`, strings and untyped
/// `NULL`s to `decimal(38,18)`.
fn to_char_decimal(t: Option<&DataType>) -> Option<DataType> {
    let (p, s) = match t {
        None | Some(DataType::String) => (38, 18),
        Some(DataType::Float) => (14, 7),
        Some(DataType::Double) => (30, 15),
        Some(DataType::Byte) => (3, 0),
        Some(DataType::Short) => (5, 0),
        Some(DataType::Integer) => (10, 0),
        Some(DataType::Long) => (20, 0),
        // Decimals, dates, timestamps and binaries stay as they are.
        _ => return None,
    };
    DecimalType::try_new(p, s).ok().map(DataType::Decimal)
}

/// `to_char(x, format)` / `to_varchar`: numbers become decimals (dates,
/// timestamps and binaries stay for the overloads Krishiv does not run
/// yet); anything else is refused as `"DECIMAL"`.
fn to_char_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let first = match c.data_type(0) {
        t @ (None
        | Some(
            DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::Float
            | DataType::Double
            | DataType::Decimal(_)
            | DataType::String,
        )) => match to_char_decimal(t) {
            // A decimal stays one; anything else takes its `forType`.
            None => InputType::Any,
            Some(t) => exact(t),
        },
        Some(DataType::Date | DataType::Timestamp | DataType::TimestampNtz | DataType::Binary) => {
            InputType::Any
        }
        _ => InputType::Any,
    };
    vec![first, exact(DataType::String)]
}

/// `to_char(x, format)` is a string. The casts that may fail (strings,
/// floats, doubles) make it nullable; dates, timestamps and binaries pass
/// through for the unimplemented overloads (binaries still check their
/// `base64`/`hex`/`utf-8` format, as Spark does).
fn to_char_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let bad_first = !matches!(
        c.data_type(0),
        None | Some(
            DataType::Byte
                | DataType::Short
                | DataType::Integer
                | DataType::Long
                | DataType::Float
                | DataType::Double
                | DataType::Decimal(_)
                | DataType::String
                | DataType::Date
                | DataType::Timestamp
                | DataType::TimestampNtz
                | DataType::Binary
        )
    );
    if bad_first {
        return Err(unexpected(c, "to_char", 0, "\"DECIMAL\""));
    }
    // Dates and timestamps take datetime patterns (another task's
    // overload); only their presence is checked here.
    if matches!(
        c.data_type(0),
        Some(DataType::Date | DataType::Timestamp | DataType::TimestampNtz)
    ) {
        return typed(DataType::String, c);
    }
    let format = format_string(c, "to_char")?;
    // Binaries take `base64`/`hex`/`utf-8` (matched lower-cased, like
    // Spark's `toLowerCase.trim`).
    if matches!(c.data_type(0), Some(DataType::Binary)) {
        if let Some(pattern) = format.as_deref()
            && !matches!(
                pattern.to_lowercase().trim_matches(|c: char| c <= ' '),
                "base64" | "hex" | "utf-8"
            )
        {
            return Err(InvalidParameterValueBinaryFormat {
                parameter: "`format`".to_owned(),
                function_name: "`to_char`".to_owned(),
                invalid_format: quoted_format(pattern),
            }
            .into());
        }
        return typed(DataType::String, c);
    }
    // Numbers take an Oracle pattern (its value is read again at bind
    // time).
    if let Some(pattern) = format.as_deref() {
        parse_format(pattern).validate(pattern)?;
    }
    typed(DataType::String, c)
}

/// A format as `BINARY_FORMAT` quotes it: the pattern single-quoted.
fn quoted_format(pattern: &str) -> String {
    format!("'{pattern}'")
}

pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(
        &["hex"],
        &Inputs::Built(InputBuilder(hex_inputs)),
        hex_result,
    );
    out.extend(scalar(
        &["unhex"],
        &Inputs::Expects(vec![exact(DataType::String)]),
        unhex,
    ));
    out.extend(scalar(
        &["bin"],
        &Inputs::Expects(vec![exact(DataType::Long)]),
        bin,
    ));
    out.extend(scalar(
        &["conv"],
        &Inputs::Expects(vec![
            exact(DataType::String),
            exact(DataType::Integer),
            exact(DataType::Integer),
        ]),
        conv,
    ));
    out.extend(scalar(
        &["format_number"],
        &Inputs::Built(InputBuilder(|_| vec![InputType::Any, InputType::Any])),
        format_number_result,
    ));
    // The value casts to a string; the format stays one.
    let string_format = Inputs::Expects(vec![exact(DataType::String), exact(DataType::String)]);
    out.extend(scalar(&["to_number"], &string_format, to_number));
    out.extend(scalar(&["try_to_number"], &string_format, try_to_number));
    out.extend(scalar(
        &["to_char", "to_varchar"],
        &Inputs::Built(InputBuilder(to_char_inputs)),
        to_char_result,
    ));
    out
}
