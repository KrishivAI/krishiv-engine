// SPDX-License-Identifier: Apache-2.0
//! The streaming `from_json` reader: one JSON text per row straight into
//! Arrow builders, following the target Arrow type.
//!
//! A conversion failure poisons only the value that holds it: a struct
//! field catches it and is `NULL`, while an array element or map value
//! nulls the whole array or map (as Spark's `JacksonParser` does; the
//! goldens pin `[{"a": 1}, 2]` to `NULL` but `{"d": "02-29"}` to a null
//! field). A value of the wrong shape for a struct (anything but an
//! object, including JSON `null`) is a struct of `NULL`s; for an array
//! only an object wraps to a single element and anything else is `NULL`;
//! for a map anything but an object is `NULL`. Empty input and JSON `null`
//! are `NULL`; anything after the first JSON value is ignored.
//!
//! Scalar conversions are strict except the documented ones: strings parse
//! to dates, timestamps and timestamp-ntz values (in the session zone, as
//! the CSV and JSON readers do) and to binary through base64; a `STRING`
//! field accepts any JSON token and yields its JSON text (numbers,
//! booleans and strings as written, nested arrays and objects as compact
//! JSON), as Spark's `JacksonParser` does for `StringType`. A string is
//! never read as a number or boolean, and a number is never read as a
//! numeric, date or boolean (a float into an integral stays `NULL`, as does
//! a string like `"1"` into an integral). Duplicate object keys: the last
//! value wins for structs; map entries keep document order (duplicates
//! kept).

use std::sync::Arc;

use arrow_array::builder::ArrayBuilder;
use arrow_array::builder::{
    BinaryViewBuilder, BooleanBuilder, Date32Builder, Decimal128Builder, Float32Builder,
    Float64Builder, Int8Builder, Int16Builder, Int32Builder, Int64Builder, StringViewBuilder,
    TimestampMicrosecondBuilder,
};
use arrow_array::{Array, ArrayRef, NullArray, StringArray};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType as ArrowType, FieldRef, Fields};
use krishiv_common::error::{
    InternalError, KrishivError, MalformedRecordInParsingWithoutSuggestion,
};

use super::JsonOptions;
use super::base64;
use super::options::ParseMode;
use super::parse::{Error as ParseError, Lexer, RawKey, RawStr, Token, non_numeric_value};
use crate::cast::TimeZoneRules;
use crate::text::{
    java_parse_double, spark_string_to_date, spark_string_to_timestamp,
    spark_string_to_timestamp_ntz, today_in,
};

/// Reads `json` into `target`, one row per string.
///
/// `PERMISSIVE` (the default) maps malformed input to a struct of `NULL`s
/// for a struct target and to `NULL` otherwise (a struct field catches a
/// failure and is `NULL` instead); `FAILFAST` fails the whole read with
/// `MALFORMED_RECORD_IN_PARSING.WITHOUT_SUGGESTION`.
pub fn read_json(
    json: &StringArray,
    target: &ArrowType,
    opts: &JsonOptions,
    zone: &dyn TimeZoneRules,
    query_start_micros: i64,
) -> Result<ArrayRef, KrishivError> {
    let mut out = Builder::for_type(target)?;
    let today = today_in(query_start_micros, zone);
    let ctx = Ctx { opts, zone, today };
    let fail_fast = matches!(opts.mode, ParseMode::FailFast);
    for i in 0..json.len() {
        if json.is_null(i) {
            out.append_null().map_err(|e| match e {
                Fault::Internal(e) => e,
                // Appending `NULL` never reads a value, so a row can neither
                // poison nor seal here.
                Fault::Poison | Fault::Sealed => InternalError {
                    message: "appending a NULL JSON row poisoned it".to_owned(),
                }
                .into(),
            })?;
            continue;
        }
        let text = json.value(i);
        let mut lexer = Lexer::new(text, opts.lexer());
        match read_top(&mut lexer, &mut out, &ctx) {
            Err(Fault::Poison | Fault::Sealed) if fail_fast => {
                return Err(MalformedRecordInParsingWithoutSuggestion {
                    bad_record: text.to_owned(),
                    fail_fast_mode: "FAILFAST".to_owned(),
                }
                .into());
            }
            // A clean row, or one that sealed itself as `NULL`: complete.
            Ok(()) | Err(Fault::Sealed) => {}
            Err(Fault::Poison) => {
                append_malformed(&mut out).map_err(|e| match e {
                    Fault::Internal(e) => e,
                    // A poisoned row appends nothing, so the fallback
                    // cannot seal either.
                    Fault::Poison | Fault::Sealed => InternalError {
                        message: "a poisoned JSON row poisoned its fallback".to_owned(),
                    }
                    .into(),
                })?;
            }
            Err(Fault::Internal(e)) => return Err(e),
        }
    }
    out.finish(target)
}

/// A malformed record in `PERMISSIVE` mode: a struct target reads as a struct
/// of `NULL`s, as Spark does (`'nope'`, a truncated document and a bad escape
/// all read as `{null, null}`), anything else as `NULL`. Nothing of the row
/// is appended before it poisons, so the columns stay aligned.
fn append_malformed(out: &mut Builder) -> Result<(), Fault> {
    let Builder::Struct {
        children, valid, ..
    } = out
    else {
        return out.append_null();
    };
    for child in children.iter_mut() {
        child.append_null()?;
    }
    valid.push(true);
    Ok(())
}

/// What reading one value can report.
enum Fault {
    /// A conversion failure with nothing appended: the enclosing array, map
    /// or row is `NULL` (a struct field catches it and is `NULL` instead).
    Poison,
    /// A failed list or map sealed itself as `NULL`: the junk it appended
    /// before failing lies inside its own `NULL` range, so the row is
    /// complete and the caller must not append again. A struct field, a
    /// list element and the top level all treat this as a finished `NULL`;
    /// only `FAILFAST` still fails the row.
    Sealed,
    /// An unreachable state (a kernel called with a type it does not
    /// read).
    Internal(KrishivError),
}

impl From<ParseError> for Fault {
    fn from(_: ParseError) -> Self {
        Self::Poison
    }
}

impl From<Poison> for Fault {
    fn from(_: Poison) -> Self {
        Self::Poison
    }
}

/// A conversion failure (maps to [`Fault::Poison`]).
struct Poison;

/// Unreachable states (a kernel called with a type it does not read).
fn internal(message: String) -> Fault {
    Fault::Internal(InternalError { message }.into())
}

/// The parsing context: options, session zone and today.
struct Ctx<'a> {
    opts: &'a JsonOptions,
    zone: &'a dyn TimeZoneRules,
    today: i32,
}

/// One row's value at the top level: empty input and JSON `null` are
/// `NULL`, anything else is read against `target`. Unparseable text poisons
/// the row; [`append_malformed`] decides what a poisoned row reads as.
fn read_top(lexer: &mut Lexer<'_>, out: &mut Builder, ctx: &Ctx<'_>) -> Result<(), Fault> {
    lexer.skip_trivia()?;
    match lexer.peek() {
        Ok(None) => {
            out.append_null()?;
            Ok(())
        }
        Err(_) => Err(Poison.into()),
        Ok(Some(Token::Null)) => {
            lexer.next()?;
            out.append_null()?;
            Ok(())
        }
        Ok(Some(_)) => read_value(lexer, out, ctx),
    }
}

/// One JSON value into `out` (the cursor is before its first token).
fn read_value(lexer: &mut Lexer<'_>, out: &mut Builder, ctx: &Ctx<'_>) -> Result<(), Fault> {
    match out {
        Builder::Null { .. } => {
            lexer.skip_value()?;
            out.append_null()?;
            Ok(())
        }
        Builder::Struct { .. } => read_struct(lexer, out, ctx),
        Builder::List { .. } | Builder::Map { .. } => read_nested(lexer, out, ctx),
        Builder::Scalar(scalar) => read_scalar_token(lexer, scalar, ctx),
    }
}

/// A struct: an object fills its fields by name (missing is `NULL`, a
/// failing value is `NULL` in `PERMISSIVE` mode and poisons the row in
/// `FAILFAST` mode, the last duplicate wins); JSON `null` is a `NULL`
/// struct, anything else a struct of `NULL`s.
fn read_struct(lexer: &mut Lexer<'_>, out: &mut Builder, ctx: &Ctx<'_>) -> Result<(), Fault> {
    let Builder::Struct {
        fields,
        children,
        valid,
    } = out
    else {
        return Err(internal("a struct value for a struct builder".to_owned()));
    };
    if matches!(lexer.peek()?, Some(Token::Null)) {
        // A JSON `null` is a `NULL` struct, as an array element or a field
        // (Spark's `JacksonParser` returns `null` for a `VALUE_NULL` token
        // before any converter looks at it).
        lexer.skip_value()?;
        return out.append_null();
    }
    if !matches!(lexer.peek()?, Some(Token::OpenBrace)) {
        // Not an object: skip the value and leave a struct of NULLs (the
        // goldens pin `'1'` and `'abc'` to `{null, null}`) in PERMISSIVE
        // mode; in FAILFAST mode this poisons the row.
        if matches!(ctx.opts.mode, ParseMode::FailFast) {
            lexer.skip_value()?;
            return Err(Poison.into());
        }
        lexer.skip_value()?;
        for child in children.iter_mut() {
            child.append_null()?;
        }
        valid.push(true);
        return Ok(());
    }
    lexer.next()?;
    // The value spans of this object, in order; the last span wins a
    // duplicate key.
    let mut spans: Vec<(String, usize, usize)> = Vec::new();
    loop {
        if lexer.eat_byte(b'}')? {
            break;
        }
        let key = match lexer.raw_key()? {
            RawKey::Quoted(raw) => lexer.unescape(raw).ok_or(Poison)?,
            RawKey::Bare(word) => word.to_owned(),
        };
        lexer.skip_trivia()?;
        if !lexer.eat(Token::Colon)? {
            return Err(Poison.into());
        }
        let start = lexer.pos;
        lexer.skip_value()?;
        spans.push((key, start, lexer.pos));
        lexer.skip_trivia()?;
        if lexer.eat(Token::Comma)? {
            continue;
        }
        if lexer.eat(Token::CloseBrace)? {
            break;
        }
        return Err(Poison.into());
    }
    let text = lexer.text();
    let fail_fast = matches!(ctx.opts.mode, ParseMode::FailFast);
    for (field, child) in fields.iter().zip(children.iter_mut()) {
        let mut span = None;
        for (key, start, end) in spans.iter() {
            if key == field.name() {
                span = Some((*start, *end));
            }
        }
        match span {
            None => child.append_null()?,
            Some((start, end)) => {
                let body = text.get(start..end).ok_or(Poison)?;
                let mut sub = Lexer::new(body, ctx.opts.lexer());
                match read_value(&mut sub, child, ctx) {
                    Err(Fault::Poison | Fault::Sealed) if fail_fast => {
                        return Err(Poison.into());
                    }
                    // The field sealed itself as `NULL`: it is complete.
                    Ok(()) | Err(Fault::Sealed) => {}
                    Err(Fault::Poison) => child.append_null()?,
                    Err(Fault::Internal(e)) => return Err(Fault::Internal(e)),
                }
            }
        }
    }
    valid.push(true);
    Ok(())
}

/// An array or map: the shape must match, else the value is `NULL` in
/// `PERMISSIVE` mode and poisons the row in `FAILFAST` mode.
fn read_nested(lexer: &mut Lexer<'_>, out: &mut Builder, ctx: &Ctx<'_>) -> Result<(), Fault> {
    match lexer.peek()? {
        Some(Token::OpenBracket) if matches!(out, Builder::List { .. }) => {
            read_list(lexer, out, ctx)
        }
        Some(Token::OpenBrace) if matches!(out, Builder::List { .. }) => {
            // A single object wraps to a one-element array.
            let Builder::List {
                offsets, values, ..
            } = out
            else {
                return Err(internal("a list value for a list builder".to_owned()));
            };
            match read_value(lexer, values, ctx) {
                Err(Fault::Poison | Fault::Sealed)
                    if matches!(ctx.opts.mode, ParseMode::FailFast) =>
                {
                    Err(Poison.into())
                }
                // The wrapped value sealed itself as `NULL`: it is the element.
                Ok(()) | Err(Fault::Sealed) => {
                    offsets.push(values.len_total()?);
                    out.set_valid(true);
                    Ok(())
                }
                Err(Fault::Poison) => {
                    offsets.push(values.len_total()?);
                    out.set_valid(false);
                    Ok(())
                }
                Err(Fault::Internal(e)) => Err(Fault::Internal(e)),
            }
        }
        Some(Token::OpenBrace) if matches!(out, Builder::Map { .. }) => read_map(lexer, out, ctx),
        Some(Token::Null) => {
            lexer.next()?;
            out.append_null()?;
            Ok(())
        }
        _ => Err(Poison.into()),
    }
}

/// An array's elements; a failing element nulls the whole array in
/// `PERMISSIVE` mode and poisons the row in `FAILFAST` mode.
fn read_list(lexer: &mut Lexer<'_>, out: &mut Builder, ctx: &Ctx<'_>) -> Result<(), Fault> {
    // Borrow `values` for the base length first, then re-borrow `out` for
    // offsets: the borrow checker needs the split.
    let base = match out {
        Builder::List { values, .. } => values.len_total()?,
        _ => return Err(internal("a list value for a list builder".to_owned())),
    };
    let Builder::List {
        offsets, values, ..
    } = out
    else {
        return Err(internal("a list value for a list builder".to_owned()));
    };
    lexer.next()?;
    let mut ok = true;
    if lexer.eat(Token::CloseBracket)? {
        offsets.push(base);
        out.set_valid(true);
        return Ok(());
    }
    loop {
        match read_value(lexer, values, ctx) {
            Ok(()) => {}
            // A sealed element is already a `NULL` inside this row's range:
            // the row still fails, without appending again.
            Err(Fault::Sealed | Fault::Poison) => {
                ok = false;
                break;
            }
            Err(Fault::Internal(e)) => return Err(Fault::Internal(e)),
        }
        if lexer.skip_trivia().is_err() {
            ok = false;
            break;
        }
        match lexer.eat(Token::Comma) {
            Ok(true) => continue,
            Ok(false) => {}
            Err(_) => {
                ok = false;
                break;
            }
        }
        match lexer.eat(Token::CloseBracket) {
            Ok(true) => break,
            Ok(false) => {}
            Err(_) => {
                ok = false;
                break;
            }
        }
        ok = false;
        break;
    }
    if !ok {
        if matches!(ctx.opts.mode, ParseMode::FailFast) {
            return Err(Poison.into());
        }
        // Seal the row as `NULL`: values appended before the failure fall
        // inside its range, which is `NULL` and so never read. Draining is
        // best-effort (the caller aborts, so the cursor is never used).
        offsets.push(values.len_total()?);
        out.set_valid(false);
        let _ = drain_to_close(lexer, b']');
        return Err(Fault::Sealed);
    }
    offsets.push(values.len_total()?);
    out.set_valid(true);
    Ok(())
}

/// Drains the rest of an array (`]`) or object (`}`) after a failure.
fn drain_to_close(lexer: &mut Lexer<'_>, close: u8) -> Result<(), Fault> {
    let mut depth = 1_usize;
    loop {
        match lexer.next()? {
            None => return Err(Poison.into()),
            Some(Token::OpenBrace) if close == b'}' => depth = depth.saturating_add(1),
            Some(Token::OpenBracket) if close == b']' => depth = depth.saturating_add(1),
            Some(Token::CloseBrace) if close == b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Ok(());
                }
            }
            Some(Token::CloseBracket) if close == b']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Ok(());
                }
            }
            Some(_) => {}
        }
    }
}

/// A map's entries; a failing key or value nulls the whole map in
/// `PERMISSIVE` mode and poisons the row in `FAILFAST` mode. Duplicate
/// keys are kept, in document order (as Spark does).
fn read_map(lexer: &mut Lexer<'_>, out: &mut Builder, ctx: &Ctx<'_>) -> Result<(), Fault> {
    let base = match out {
        Builder::Map { values, .. } => values.len_total()?,
        _ => return Err(internal("a map value for a map builder".to_owned())),
    };
    let Builder::Map {
        offsets,
        keys,
        values,
        ..
    } = out
    else {
        return Err(internal("a map value for a map builder".to_owned()));
    };
    lexer.next()?;
    let mut ok = true;
    if lexer.eat_byte(b'}')? {
        offsets.push(base);
        out.set_valid(true);
        return Ok(());
    }
    loop {
        let key = match lexer.raw_key() {
            Ok(RawKey::Quoted(raw)) => match lexer.unescape(raw) {
                Some(key) => key,
                None => {
                    ok = false;
                    break;
                }
            },
            Ok(RawKey::Bare(word)) => word.to_owned(),
            Err(_) => {
                ok = false;
                break;
            }
        };
        if lexer.skip_trivia().is_err() {
            ok = false;
            break;
        }
        match lexer.eat(Token::Colon) {
            Ok(true) => {}
            Ok(false) | Err(_) => {
                ok = false;
                break;
            }
        }
        // The key converts first (a bad key fails before the value is
        // read); the converted key is appended whatever the value does, so
        // keys and values stay aligned and a failing value only needs a
        // `NULL` slot to keep them there.
        let Ok(key) = convert_key(&key, keys) else {
            ok = false;
            break;
        };
        append_key(keys, key);
        match read_value(lexer, values, ctx) {
            Ok(()) => {}
            // A sealed value is already a `NULL` slot: the row still
            // fails, without appending again.
            Err(Fault::Sealed) => {
                ok = false;
                break;
            }
            Err(Fault::Poison) => {
                values.append_null()?;
                ok = false;
                break;
            }
            Err(Fault::Internal(e)) => return Err(Fault::Internal(e)),
        }
        if lexer.skip_trivia().is_err() {
            ok = false;
            break;
        }
        match lexer.eat(Token::Comma) {
            Ok(true) => continue,
            Ok(false) => {}
            Err(_) => {
                ok = false;
                break;
            }
        }
        match lexer.eat(Token::CloseBrace) {
            Ok(true) => break,
            Ok(false) => {}
            Err(_) => {
                ok = false;
                break;
            }
        }
        ok = false;
        break;
    }
    if !ok {
        if matches!(ctx.opts.mode, ParseMode::FailFast) {
            return Err(Poison.into());
        }
        // Seal the row as `NULL`: entries appended before the failure fall
        // inside its range, which is `NULL` and so never read. Draining is
        // best-effort (the caller aborts, so the cursor is never used).
        offsets.push(values.len_total()?);
        out.set_valid(false);
        let _ = drain_to_close(lexer, b'}');
        return Err(Fault::Sealed);
    }
    offsets.push(values.len_total()?);
    out.set_valid(true);
    Ok(())
}

/// One converted map key.
enum Key {
    /// A boolean.
    Bool(bool),
    /// An integral key.
    I64(i64),
    /// A floating-point key.
    F64(f64),
    /// A decimal key (already fitted).
    Decimal(i128),
    /// A string key.
    Str(String),
    /// A date key.
    Date(i32),
    /// A binary key.
    Binary(Vec<u8>),
}

/// Converts one map key (always a JSON string) for the key builder. A key
/// is never a timestamp or nested value in Spark's JSON maps.
fn convert_key(text: &str, keys: &Builder) -> Result<Key, Poison> {
    let Builder::Scalar(scalar) = keys else {
        return Err(Poison);
    };
    match scalar {
        Scalar::Str(_) => Ok(Key::Str(text.to_owned())),
        Scalar::Bool(_) => match text {
            "true" => Ok(Key::Bool(true)),
            "false" => Ok(Key::Bool(false)),
            _ => Err(Poison),
        },
        Scalar::I8(_) => text
            .parse::<i8>()
            .map(|n| Key::I64(i64::from(n)))
            .map_err(|_| Poison),
        Scalar::I16(_) => text
            .parse::<i16>()
            .map(|n| Key::I64(i64::from(n)))
            .map_err(|_| Poison),
        Scalar::I32(_) => text
            .parse::<i32>()
            .map(|n| Key::I64(i64::from(n)))
            .map_err(|_| Poison),
        Scalar::I64(_) => text.parse::<i64>().map(Key::I64).map_err(|_| Poison),
        Scalar::F32(_) | Scalar::F64(_) => java_parse_double(text).map(Key::F64).ok_or(Poison),
        Scalar::Decimal(_, p, s) => decimal_from_text(text, *p, *s)
            .map(Key::Decimal)
            .ok_or(Poison),
        Scalar::Date(_) => spark_string_to_date(text).map(Key::Date).ok_or(Poison),
        Scalar::Binary(_) => base64::decode(text).map(Key::Binary).ok_or(Poison),
        Scalar::Timestamp { .. } => Err(Poison),
    }
}

/// Appends a converted key (ranges were validated at conversion).
fn append_key(keys: &mut Builder, key: Key) {
    let Builder::Scalar(scalar) = keys else {
        return;
    };
    match (scalar, key) {
        (Scalar::Str(b), Key::Str(v)) => b.append_value(&v),
        (Scalar::Bool(b), Key::Bool(v)) => b.append_value(v),
        (Scalar::I8(b), Key::I64(v)) => b.append_value(i8::try_from(v).unwrap_or(i8::MIN)),
        (Scalar::I16(b), Key::I64(v)) => b.append_value(i16::try_from(v).unwrap_or(i16::MIN)),
        (Scalar::I32(b), Key::I64(v)) => b.append_value(i32::try_from(v).unwrap_or(i32::MIN)),
        (Scalar::I64(b), Key::I64(v)) => b.append_value(v),
        (Scalar::F32(b), Key::F64(v)) => b.append_value(f32_from_f64(v)),
        (Scalar::F64(b), Key::F64(v)) => b.append_value(v),
        (Scalar::Decimal(b, _, _), Key::Decimal(v)) => b.append_value(v),
        (Scalar::Date(b), Key::Date(v)) => b.append_value(v),
        (Scalar::Binary(b), Key::Binary(v)) => b.append_value(&v),
        _ => {}
    }
}

/// One JSON value at the cursor as compact JSON text (for a `STRING`
/// field holding a nested value): numbers and booleans as written,
/// strings re-quoted, arrays and objects without whitespace, `null` as
/// `null`. Past [`super::parse::MAX_JSON_DEPTH`] (Jackson's default, 1000)
/// is a conversion failure.
fn render_compact_value(
    lexer: &mut Lexer<'_>,
    text: &mut String,
    depth: usize,
) -> Result<(), Fault> {
    if depth > super::parse::MAX_JSON_DEPTH {
        return Err(Poison.into());
    }
    let deeper = depth.saturating_add(1);
    match lexer.next()? {
        Some(Token::Null) => text.push_str("null"),
        Some(Token::Bool(true)) => text.push_str("true"),
        Some(Token::Bool(false)) => text.push_str("false"),
        Some(Token::Num(raw, _)) => text.push_str(raw),
        Some(Token::Str(raw)) => {
            let s = lexer.unescape(raw).ok_or(Poison)?;
            super::write::push_quoted(&s, text);
        }
        Some(Token::OpenBracket) => {
            text.push('[');
            if lexer.eat(Token::CloseBracket)? {
                text.push(']');
                return Ok(());
            }
            loop {
                render_compact_value(lexer, text, deeper)?;
                lexer.skip_trivia()?;
                if lexer.eat(Token::Comma)? {
                    text.push(',');
                    continue;
                }
                if lexer.eat(Token::CloseBracket)? {
                    break;
                }
                return Err(Poison.into());
            }
            text.push(']');
        }
        Some(Token::OpenBrace) => {
            text.push('{');
            if lexer.eat_byte(b'}')? {
                text.push('}');
                return Ok(());
            }
            loop {
                let key = match lexer.raw_key()? {
                    RawKey::Quoted(raw) => lexer.unescape(raw).ok_or(Poison)?,
                    RawKey::Bare(word) => word.to_owned(),
                };
                lexer.skip_trivia()?;
                if !lexer.eat(Token::Colon)? {
                    return Err(Poison.into());
                }
                super::write::push_quoted(&key, text);
                text.push(':');
                render_compact_value(lexer, text, deeper)?;
                lexer.skip_trivia()?;
                if lexer.eat(Token::Comma)? {
                    text.push(',');
                    continue;
                }
                if lexer.eat(Token::CloseBrace)? {
                    break;
                }
                return Err(Poison.into());
            }
            text.push('}');
        }
        None | Some(_) => return Err(Poison.into()),
    }
    Ok(())
}

/// One scalar token into `out`.
fn read_scalar_token(lexer: &mut Lexer<'_>, out: &mut Scalar, ctx: &Ctx<'_>) -> Result<(), Fault> {
    match lexer.next()? {
        Some(Token::Null) => {
            out.append_null();
            Ok(())
        }
        Some(Token::Bool(b)) => match out {
            Scalar::Bool(o) => {
                o.append_value(b);
                Ok(())
            }
            Scalar::Str(o) => {
                o.append_value(if b { "true" } else { "false" });
                Ok(())
            }
            _ => Err(Poison.into()),
        },
        Some(Token::Str(raw)) => read_string(raw, out, ctx),
        Some(Token::Num(text, is_float)) => read_number(text, is_float, out),
        Some(Token::OpenBracket) if matches!(out, Scalar::Str(_)) => {
            let Scalar::Str(b) = out else {
                return Err(internal("a string builder for a JSON array".to_owned()));
            };
            let mut text = String::from("[");
            if lexer.eat(Token::CloseBracket)? {
                text.push(']');
                b.append_value(&text);
                return Ok(());
            }
            loop {
                render_compact_value(lexer, &mut text, 0)?;
                lexer.skip_trivia()?;
                if lexer.eat(Token::Comma)? {
                    text.push(',');
                    continue;
                }
                if lexer.eat(Token::CloseBracket)? {
                    break;
                }
                return Err(Poison.into());
            }
            text.push(']');
            b.append_value(&text);
            Ok(())
        }
        Some(Token::OpenBrace) if matches!(out, Scalar::Str(_)) => {
            let Scalar::Str(b) = out else {
                return Err(internal("a string builder for a JSON object".to_owned()));
            };
            let mut text = String::from("{");
            if lexer.eat_byte(b'}')? {
                text.push('}');
                b.append_value(&text);
                return Ok(());
            }
            loop {
                let key = match lexer.raw_key()? {
                    RawKey::Quoted(raw) => lexer.unescape(raw).ok_or(Poison)?,
                    RawKey::Bare(word) => word.to_owned(),
                };
                lexer.skip_trivia()?;
                if !lexer.eat(Token::Colon)? {
                    return Err(Poison.into());
                }
                super::write::push_quoted(&key, &mut text);
                text.push(':');
                render_compact_value(lexer, &mut text, 0)?;
                lexer.skip_trivia()?;
                if lexer.eat(Token::Comma)? {
                    text.push(',');
                    continue;
                }
                if lexer.eat(Token::CloseBrace)? {
                    break;
                }
                return Err(Poison.into());
            }
            text.push('}');
            b.append_value(&text);
            Ok(())
        }
        None | Some(_) => Err(Poison.into()),
    }
}

/// One JSON string into `out`.
fn read_string(raw: RawStr<'_>, out: &mut Scalar, ctx: &Ctx<'_>) -> Result<(), Fault> {
    let text = raw
        .unescape(ctx.opts.allow_backslash_escaping_any)
        .ok_or(Poison)?;
    match out {
        Scalar::Str(b) => {
            b.append_value(&text);
            Ok(())
        }
        Scalar::Date(b) => {
            let parsed = match &ctx.opts.date_format {
                Some(pattern) => pattern.parse_date(&text).map_err(|_| Poison)?,
                None => spark_string_to_date(&text),
            };
            match parsed {
                Some(v) => {
                    b.append_value(v);
                    Ok(())
                }
                None => Err(Poison.into()),
            }
        }
        Scalar::Timestamp { builder, ntz } => {
            let parsed = if *ntz {
                match &ctx.opts.timestamp_ntz_format {
                    Some(pattern) => pattern.parse_local(&text, false).map_err(|_| Poison)?,
                    None => spark_string_to_timestamp_ntz(&text),
                }
            } else {
                match &ctx.opts.timestamp_format {
                    Some(pattern) => pattern
                        .parse_timestamp(&text, ctx.zone)
                        .map_err(|_| Poison)?,
                    None => {
                        spark_string_to_timestamp(&text, ctx.zone, ctx.today).map_err(|_| Poison)?
                    }
                }
            };
            match parsed {
                Some(v) => {
                    builder.append_value(v);
                    Ok(())
                }
                None => Err(Poison.into()),
            }
        }
        Scalar::Binary(b) => match base64::decode(&text) {
            Some(bytes) => {
                b.append_value(&bytes);
                Ok(())
            }
            None => Err(Poison.into()),
        },
        Scalar::F32(b) => match quoted_special(&text, ctx) {
            Some(v) => {
                b.append_value(f32_from_f64(v));
                Ok(())
            }
            None => Err(Poison.into()),
        },
        Scalar::F64(b) => match quoted_special(&text, ctx) {
            Some(v) => {
                b.append_value(v);
                Ok(())
            }
            None => Err(Poison.into()),
        },
        _ => Err(Poison.into()),
    }
}

/// A quoted special numeric string (`"NaN"`) for a float field: its value
/// with `allowNonNumericNumbers`, else `None` (a string is never read as a
/// number otherwise).
fn quoted_special(text: &str, ctx: &Ctx<'_>) -> Option<f64> {
    ctx.opts
        .allow_non_numeric_numbers
        .then(|| non_numeric_value(text))
        .flatten()
}

/// Narrows a double to a float, as Java's cast does.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Spark reads JSON numbers for `float` through Java's narrowing cast"
)]
fn f32_from_f64(v: f64) -> f32 {
    v as f32
}

/// One JSON number into `out`: a `STRING` field accepts any number as
/// its JSON text (as Spark does); a float into an integral stays `NULL`,
/// as does a string like `"1"` into an integral.
fn read_number(text: &str, is_float: bool, out: &mut Scalar) -> Result<(), Fault> {
    if is_float {
        // The lexer only produces the non-numeric spellings with
        // `allowNonNumericNumbers`, so no flag check is needed here.
        return match out {
            Scalar::Str(b) => {
                b.append_value(text);
                Ok(())
            }
            Scalar::F32(b) => match non_numeric_value(text).or_else(|| java_parse_double(text)) {
                Some(v) => {
                    b.append_value(f32_from_f64(v));
                    Ok(())
                }
                None => Err(Poison.into()),
            },
            Scalar::F64(b) => match non_numeric_value(text).or_else(|| java_parse_double(text)) {
                Some(v) => {
                    b.append_value(v);
                    Ok(())
                }
                None => Err(Poison.into()),
            },
            Scalar::Decimal(b, p, s) => match decimal_from_text(text, *p, *s) {
                Some(v) => {
                    b.append_value(v);
                    Ok(())
                }
                None => Err(Poison.into()),
            },
            _ => Err(Poison.into()),
        };
    }
    // Integral text: the string and float targets never read the integer
    // value, so a huge mantissa cannot poison them (Spark returns the text
    // for `STRING` and the double for `DOUBLE`); the integral targets
    // parse at their own width.
    match out {
        Scalar::Str(b) => {
            b.append_value(text);
            Ok(())
        }
        Scalar::I8(b) => match text.parse::<i8>() {
            Ok(v) => {
                b.append_value(v);
                Ok(())
            }
            Err(_) => Err(Poison.into()),
        },
        Scalar::I16(b) => match text.parse::<i16>() {
            Ok(v) => {
                b.append_value(v);
                Ok(())
            }
            Err(_) => Err(Poison.into()),
        },
        Scalar::I32(b) => match text.parse::<i32>() {
            Ok(v) => {
                b.append_value(v);
                Ok(())
            }
            Err(_) => Err(Poison.into()),
        },
        Scalar::I64(b) => match text.parse::<i64>() {
            Ok(v) => {
                b.append_value(v);
                Ok(())
            }
            Err(_) => Err(Poison.into()),
        },
        Scalar::F32(b) => match java_parse_double(text) {
            Some(v) => {
                b.append_value(f32_from_f64(v));
                Ok(())
            }
            None => Err(Poison.into()),
        },
        Scalar::F64(b) => match java_parse_double(text) {
            Some(v) => {
                b.append_value(v);
                Ok(())
            }
            None => Err(Poison.into()),
        },
        Scalar::Decimal(b, p, s) => match decimal_from_text(text, *p, *s) {
            Some(v) => {
                b.append_value(v);
                Ok(())
            }
            None => Err(Poison.into()),
        },
        _ => Err(Poison.into()),
    }
}

/// The decimal `text` (integral or floating) rounded to `(p, s)` with
/// `HALF_UP`; `None` past the precision or for malformed text.
///
/// Only the most significant discarded digit decides the rounding: the
/// rest truncate, so a mantissa longer than the precision rounds as
/// Spark's does. Every loop count here is bounded by the input text or by
/// the precision, so no untrusted exponent can set a loop count.
fn decimal_from_text(text: &str, p: u8, s: i8) -> Option<i128> {
    let bytes = text.as_bytes();
    let (negative, digits) = match bytes.first() {
        Some(b'-') => (true, bytes.get(1..)?),
        _ => (false, bytes),
    };
    if digits.is_empty() {
        return None;
    }
    // Split off the exponent first.
    let (mantissa, exp) = match digits.iter().position(|&b| b == b'e' || b == b'E') {
        None => (digits, 0_i32),
        Some(e) => {
            let exp_text = digits.get(e.saturating_add(1)..)?;
            let (sign, mag) = match exp_text.first() {
                Some(b'+') => (1_i32, exp_text.get(1..)?),
                Some(b'-') => (-1_i32, exp_text.get(1..)?),
                Some(_) => (1_i32, exp_text),
                None => return None,
            };
            if mag.is_empty() || mag.iter().any(|&b| !b.is_ascii_digit()) {
                return None;
            }
            let magnitude: i32 = core::str::from_utf8(mag).ok()?.parse().ok()?;
            let exp = if sign < 0 {
                magnitude.checked_neg()?
            } else {
                magnitude
            };
            (digits.get(..e)?, exp)
        }
    };
    let (int, frac) = match mantissa.iter().position(|&b| b == b'.') {
        None => (mantissa, bytes.get(..0)?),
        Some(dot) => (mantissa.get(..dot)?, mantissa.get(dot.saturating_add(1)..)?),
    };
    if int.iter().any(|&b| !b.is_ascii_digit())
        || frac.iter().any(|&b| !b.is_ascii_digit())
        || (int.is_empty() && frac.is_empty())
    {
        return None;
    }
    // The significant digits, past the leading zeros. A zero of any shape
    // reads as zero whatever the exponent says (`0e2000000000` is `0.00`,
    // where `1e-2000000000` is out of range below).
    let total = int.len().saturating_add(frac.len());
    let leading = int
        .iter()
        .chain(frac.iter())
        .take_while(|b| **b == b'0')
        .count();
    if leading == total {
        return Some(0);
    }
    let significant_len = total.saturating_sub(leading);
    let digits = |i: usize| -> u8 {
        let at = leading.saturating_add(i);
        if at < int.len() {
            int.get(at).copied().unwrap_or(b'0')
        } else {
            frac.get(at.saturating_sub(int.len()))
                .copied()
                .unwrap_or(b'0')
        }
    };
    let significant = i64::try_from(significant_len).ok()?;
    // Unscaled = significant digits, scale = frac.len() - exp.
    let scale = i64::try_from(frac.len())
        .ok()?
        .checked_sub(i64::from(exp))?;
    let target = i64::from(s);
    let drop = scale.checked_sub(target)?;
    // Spark rescales with `BigDecimal.setScale`, which raises (so the value
    // is `NULL`) once `10^drop` overflows a `BigInteger`; smaller exponents
    // round to zero. No loop below runs `drop` times.
    if drop > MAX_RESCALE_DIGITS {
        return None;
    }
    let value: i128 = if drop < 0 {
        // Widen by `-drop`: more than `p` digits never fits.
        let widen = drop.checked_neg()?;
        if significant.checked_add(widen)? > i64::from(p) {
            return None;
        }
        let mut value: i128 = 0;
        for i in 0..significant_len {
            value = value
                .checked_mul(10)?
                .checked_add(i128::from(digits(i).checked_sub(b'0')?))?;
        }
        value.checked_mul(10_i128.checked_pow(u32::try_from(widen).ok()?)?)?
    } else if drop > significant {
        // The first discarded digit is a leading zero: it rounds to zero.
        0
    } else {
        // Truncate every discarded digit but the most significant, then
        // round once on that one (`HALF_UP`).
        let keep = usize::try_from(significant.checked_sub(drop)?).ok()?;
        let mut value: i128 = 0;
        for i in 0..keep {
            value = value
                .checked_mul(10)?
                .checked_add(i128::from(digits(i).checked_sub(b'0')?))?;
        }
        if digits(keep) >= b'5' {
            value = value.checked_add(1)?;
        }
        value
    };
    let limit = 10_i128.checked_pow(u32::from(p))?;
    if value.checked_abs()? >= limit {
        return None;
    }
    Some(if negative {
        value.checked_neg()?
    } else {
        value
    })
}

/// The most digits a decimal is rescaled down by before Java's
/// `BigInteger.pow` (behind `BigDecimal.setScale`) overflows: 2^29.
const MAX_RESCALE_DIGITS: i64 = 1 << 29;

/// A scalar builder.
enum Scalar {
    /// `boolean`.
    Bool(BooleanBuilder),
    /// `tinyint`.
    I8(Int8Builder),
    /// `smallint`.
    I16(Int16Builder),
    /// `int`.
    I32(Int32Builder),
    /// `bigint`.
    I64(Int64Builder),
    /// `float`.
    F32(Float32Builder),
    /// `double`.
    F64(Float64Builder),
    /// `decimal(p, s)`.
    Decimal(Decimal128Builder, u8, i8),
    /// `string`.
    Str(StringViewBuilder),
    /// `binary`.
    Binary(BinaryViewBuilder),
    /// `date`.
    Date(Date32Builder),
    /// `timestamp` (`ntz = false`) or `timestamp_ntz`.
    Timestamp {
        /// The values.
        builder: TimestampMicrosecondBuilder,
        /// Whether zone-less.
        ntz: bool,
    },
}

impl Scalar {
    /// Appends a SQL `NULL`.
    fn append_null(&mut self) {
        match self {
            Self::Bool(b) => b.append_null(),
            Self::I8(b) => b.append_null(),
            Self::I16(b) => b.append_null(),
            Self::I32(b) => b.append_null(),
            Self::I64(b) => b.append_null(),
            Self::F32(b) => b.append_null(),
            Self::F64(b) => b.append_null(),
            Self::Decimal(b, _, _) => b.append_null(),
            Self::Str(b) => b.append_null(),
            Self::Binary(b) => b.append_null(),
            Self::Date(b) => b.append_null(),
            Self::Timestamp { builder, .. } => builder.append_null(),
        }
    }

    /// Finishes the column, which must have `target`'s type.
    fn finish(self, target: &ArrowType) -> Result<ArrayRef, KrishivError> {
        let array: ArrayRef = match self {
            Self::Bool(mut b) => Arc::new(b.finish()),
            Self::I8(mut b) => Arc::new(b.finish()),
            Self::I16(mut b) => Arc::new(b.finish()),
            Self::I32(mut b) => Arc::new(b.finish()),
            Self::I64(mut b) => Arc::new(b.finish()),
            Self::F32(mut b) => Arc::new(b.finish()),
            Self::F64(mut b) => Arc::new(b.finish()),
            Self::Decimal(mut b, p, s) => {
                let array =
                    b.finish()
                        .with_precision_and_scale(p, s)
                        .map_err(|e| InternalError {
                            message: format!("a decimal column past decimal({p},{s}): {e}"),
                        })?;
                Arc::new(array)
            }
            Self::Str(mut b) => Arc::new(b.finish()),
            Self::Binary(mut b) => Arc::new(b.finish()),
            Self::Date(mut b) => Arc::new(b.finish()),
            Self::Timestamp { mut builder, ntz } => {
                let mut array = builder.finish();
                if !ntz {
                    array = array.with_timezone("UTC");
                }
                Arc::new(array)
            }
        };
        if array.data_type() != target {
            return Err(InternalError {
                message: format!("a {} column for {target}", array.data_type()),
            }
            .into());
        }
        Ok(array)
    }
}

/// The length of a builder's column.
fn len_of<B: ArrayBuilder>(builder: &B) -> Result<i32, Fault> {
    i32::try_from(builder.len())
        .map_err(|_| internal("a JSON builder length past the 32-bit offset limit".to_owned()))
}

/// A builder for one Arrow type.
enum Builder {
    /// A `NULL` column (for a `void` schema).
    Null {
        /// Rows appended.
        len: usize,
    },
    /// A scalar column.
    Scalar(Scalar),
    /// A list column: `offsets` has one more entry than rows, and `valid`
    /// holds one entry per row.
    List {
        /// The list field (names the element).
        field: FieldRef,
        /// The row offsets into `values`.
        offsets: Vec<i32>,
        /// The elements.
        values: Box<Builder>,
        /// One per row.
        valid: Vec<bool>,
    },
    /// A map column.
    Map {
        /// The entries field.
        field: FieldRef,
        /// The row offsets into `keys`/`values`.
        offsets: Vec<i32>,
        /// The keys.
        keys: Box<Builder>,
        /// The values.
        values: Box<Builder>,
        /// One per row.
        valid: Vec<bool>,
    },
    /// A struct column.
    Struct {
        /// The struct fields.
        fields: Fields,
        /// One builder per field.
        children: Vec<Builder>,
        /// One per row.
        valid: Vec<bool>,
    },
}

impl Builder {
    /// A builder reading `target`.
    fn for_type(target: &ArrowType) -> Result<Self, KrishivError> {
        Ok(match target {
            ArrowType::Null => Self::Null { len: 0 },
            ArrowType::Boolean => Self::Scalar(Scalar::Bool(BooleanBuilder::new())),
            ArrowType::Int8 => Self::Scalar(Scalar::I8(Int8Builder::new())),
            ArrowType::Int16 => Self::Scalar(Scalar::I16(Int16Builder::new())),
            ArrowType::Int32 => Self::Scalar(Scalar::I32(Int32Builder::new())),
            ArrowType::Int64 => Self::Scalar(Scalar::I64(Int64Builder::new())),
            ArrowType::Float32 => Self::Scalar(Scalar::F32(Float32Builder::new())),
            ArrowType::Float64 => Self::Scalar(Scalar::F64(Float64Builder::new())),
            ArrowType::Decimal128(p, s) => {
                Self::Scalar(Scalar::Decimal(Decimal128Builder::new(), *p, *s))
            }
            ArrowType::Utf8View | ArrowType::Utf8 | ArrowType::LargeUtf8 => {
                Self::Scalar(Scalar::Str(StringViewBuilder::new()))
            }
            ArrowType::BinaryView | ArrowType::Binary | ArrowType::LargeBinary => {
                Self::Scalar(Scalar::Binary(BinaryViewBuilder::new()))
            }
            ArrowType::Date32 => Self::Scalar(Scalar::Date(Date32Builder::new())),
            ArrowType::Timestamp(_, tz) => Self::Scalar(Scalar::Timestamp {
                builder: TimestampMicrosecondBuilder::new(),
                ntz: tz.is_none(),
            }),
            ArrowType::List(field) => Self::List {
                field: Arc::clone(field),
                offsets: vec![0],
                values: Box::new(Self::for_type(field.data_type())?),
                valid: Vec::new(),
            },
            ArrowType::LargeList(_) => {
                return Err(InternalError {
                    message: "a JSON value for a large list (from_json builds lists)".to_owned(),
                }
                .into());
            }
            ArrowType::Map(field, _) => {
                let (keys, values) = match field.data_type() {
                    ArrowType::Struct(entries) if entries.len() == 2 => (
                        Self::for_type(entries[0].data_type())?,
                        Self::for_type(entries[1].data_type())?,
                    ),
                    other => {
                        return Err(InternalError {
                            message: format!("a map value with entries {other}"),
                        }
                        .into());
                    }
                };
                Self::Map {
                    field: Arc::clone(field),
                    offsets: vec![0],
                    keys: Box::new(keys),
                    values: Box::new(values),
                    valid: Vec::new(),
                }
            }
            ArrowType::Struct(fields) => Self::Struct {
                children: fields
                    .iter()
                    .map(|f| Self::for_type(f.data_type()))
                    .collect::<Result<Vec<_>, _>>()?,
                fields: fields.clone(),
                valid: Vec::new(),
            },
            other => {
                return Err(InternalError {
                    message: format!("a JSON value for Arrow type {other}"),
                }
                .into());
            }
        })
    }

    /// The number of values appended: the elements of a list, the
    /// entries of a map, the rows of anything else.
    fn len_total(&self) -> Result<i32, Fault> {
        match self {
            Self::Null { len } => i32::try_from(*len).map_err(|_| {
                internal("a JSON null column past the 32-bit offset limit".to_owned())
            }),
            Self::List { valid, .. } | Self::Map { valid, .. } | Self::Struct { valid, .. } => {
                i32::try_from(valid.len()).map_err(|_| {
                    internal("a JSON nested column past the 32-bit offset limit".to_owned())
                })
            }
            Self::Scalar(scalar) => match scalar {
                Scalar::Bool(b) => len_of(b),
                Scalar::I8(b) => len_of(b),
                Scalar::I16(b) => len_of(b),
                Scalar::I32(b) => len_of(b),
                Scalar::I64(b) => len_of(b),
                Scalar::F32(b) => len_of(b),
                Scalar::F64(b) => len_of(b),
                Scalar::Decimal(b, _, _) => len_of(b),
                Scalar::Str(b) => len_of(b),
                Scalar::Binary(b) => len_of(b),
                Scalar::Date(b) => len_of(b),
                Scalar::Timestamp { builder, .. } => len_of(builder),
            },
        }
    }

    /// Appends a SQL `NULL`.
    fn append_null(&mut self) -> Result<(), Fault> {
        match self {
            Self::Null { len } => {
                *len = len.saturating_add(1);
                Ok(())
            }
            Self::Scalar(scalar) => {
                scalar.append_null();
                Ok(())
            }
            Self::List { offsets, valid, .. } | Self::Map { offsets, valid, .. } => {
                let last = offsets.last().copied().ok_or_else(|| {
                    internal("a JSON list or map without a base offset".to_owned())
                })?;
                offsets.push(last);
                valid.push(false);
                Ok(())
            }
            Self::Struct {
                children, valid, ..
            } => {
                for child in children.iter_mut() {
                    child.append_null()?;
                }
                valid.push(false);
                Ok(())
            }
        }
    }

    /// Marks the open list, map or struct row valid (`valid = true`) or
    /// `NULL` (`valid = false`).
    fn set_valid(&mut self, valid: bool) {
        match self {
            Self::Null { .. } | Self::Scalar(_) => {}
            Self::List { valid: v, .. }
            | Self::Map { valid: v, .. }
            | Self::Struct { valid: v, .. } => {
                v.push(valid);
            }
        }
    }

    /// Finishes the column, which must have `target`'s type.
    fn finish(self, target: &ArrowType) -> Result<ArrayRef, KrishivError> {
        match self {
            Self::Null { len } => {
                if !matches!(target, ArrowType::Null) {
                    return Err(InternalError {
                        message: format!("a null column for {target}"),
                    }
                    .into());
                }
                Ok(Arc::new(NullArray::new(len)))
            }
            Self::Scalar(scalar) => scalar.finish(target),
            Self::List {
                field,
                offsets,
                values,
                valid,
            } => {
                let values = values.finish(field.data_type())?;
                let array = arrow_array::ListArray::try_new(
                    field,
                    OffsetBuffer::new(ScalarBuffer::from(offsets)),
                    values,
                    nulls(valid),
                )
                .map_err(|e| InternalError {
                    message: e.to_string(),
                })?;
                check_type(Arc::new(array), target)
            }
            Self::Map {
                field,
                offsets,
                keys,
                values,
                valid,
            } => {
                let (entries_fields, key_type, value_type) = match field.data_type() {
                    ArrowType::Struct(entries) if entries.len() == 2 => (
                        entries.clone(),
                        entries[0].data_type(),
                        entries[1].data_type(),
                    ),
                    other => {
                        return Err(InternalError {
                            message: format!("a map value with entries {other}"),
                        }
                        .into());
                    }
                };
                let keys = keys.finish(key_type)?;
                let values = values.finish(value_type)?;
                let entries =
                    arrow_array::StructArray::try_new(entries_fields, vec![keys, values], None)
                        .map_err(|e| InternalError {
                            message: e.to_string(),
                        })?;
                let array = arrow_array::MapArray::try_new(
                    field,
                    OffsetBuffer::new(ScalarBuffer::from(offsets)),
                    entries,
                    nulls(valid),
                    false,
                )
                .map_err(|e| InternalError {
                    message: e.to_string(),
                })?;
                check_type(Arc::new(array), target)
            }
            Self::Struct {
                fields,
                children,
                valid,
            } => {
                let arrays = fields
                    .iter()
                    .zip(children)
                    .map(|(f, child)| child.finish(f.data_type()))
                    .collect::<Result<Vec<_>, _>>()?;
                let array = arrow_array::StructArray::try_new(fields, arrays, nulls(valid))
                    .map_err(|e| InternalError {
                        message: e.to_string(),
                    })?;
                check_type(Arc::new(array), target)
            }
        }
    }
}

/// The null buffer for `valid` (`None` when every row is valid).
fn nulls(valid: Vec<bool>) -> Option<NullBuffer> {
    if valid.iter().all(|v| *v) {
        None
    } else {
        Some(NullBuffer::from(valid))
    }
}

/// Checks a finished array against its target type.
fn check_type(array: ArrayRef, target: &ArrowType) -> Result<ArrayRef, KrishivError> {
    if array.data_type() != target {
        return Err(InternalError {
            message: format!("a {} column for {target}", array.data_type()),
        }
        .into());
    }
    Ok(array)
}
