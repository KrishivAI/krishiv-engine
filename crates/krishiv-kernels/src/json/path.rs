// SPDX-License-Identifier: Apache-2.0
//! The navigation queries: `get_json_object` (JSONPath lookup),
//! `json_array_length` (top-level array length) and `json_object_keys`
//! (top-level object keys).
//!
//! Paths are Spark's JSONPath: `$`, `.field`, `['quoted']`, `["quoted"]`,
//! `[index]` and the `[*]` wildcard. Anything else (including a path that
//! does not start with `$`) matches nothing, and no match is a SQL `NULL`,
//! as Spark returns `NULL` for a bad path. One match renders as the value
//! (strings bare, anything else compact JSON); several matches (only a
//! `[*]` step can branch) render as a JSON array of the matches.
//!
//! The structural queries need no path: a top-level array's length, or a
//! top-level object's keys in document order. Malformed text and a value
//! of the wrong shape are `NULL`. Like `from_json`, only the first JSON
//! value is read; anything after it is ignored.

use std::sync::Arc;

use arrow_array::builder::{ArrayBuilder, Int32Builder, StringViewBuilder};
use arrow_array::{Array, ArrayRef, StringArray};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType as ArrowType, Field};
use krishiv_common::error::KrishivError;

use super::parse::{Error as ParseError, Lexer, LexerFlags, RawKey, RawStr, Token};
use super::write::push_quoted;

/// Looks `path` up in each row's JSON text.
pub fn get_json_object(json: &StringArray, path: &StringArray) -> ArrayRef {
    let flags = LexerFlags::strict();
    let mut out = Vec::with_capacity(json.len());
    for i in 0..json.len() {
        let value = if json.is_null(i) || path.is_null(i) {
            None
        } else {
            let steps = parse_path(path.value(i));
            match steps {
                None => None,
                Some(steps) => {
                    let mut lexer = Lexer::new(json.value(i), flags);
                    match select(&mut lexer, &steps) {
                        Ok(matches) => render_matches(&matches),
                        Err(_) => None,
                    }
                }
            }
        };
        out.push(value);
    }
    Arc::new(StringArray::from(out))
}

/// The length of each row's top-level JSON array (`NULL` otherwise).
pub fn json_array_length(json: &StringArray) -> ArrayRef {
    let flags = LexerFlags::strict();
    let mut out = Int32Builder::new();
    for i in 0..json.len() {
        if json.is_null(i) {
            out.append_null();
            continue;
        }
        let mut lexer = Lexer::new(json.value(i), flags);
        match array_length(&mut lexer) {
            Ok(n) => out.append_value(n),
            Err(_) => out.append_null(),
        }
    }
    Arc::new(out.finish())
}

/// The keys of each row's top-level JSON object (`NULL` otherwise).
pub fn json_object_keys(json: &StringArray) -> Result<ArrayRef, KrishivError> {
    use krishiv_common::error::InternalError;
    let flags = LexerFlags::strict();
    let mut items = StringViewBuilder::new();
    let mut offsets = vec![0_i32];
    let mut valid = Vec::with_capacity(json.len());
    for i in 0..json.len() {
        if json.is_null(i) {
            let last = offsets.last().copied().ok_or_else(|| InternalError {
                message: "a JSON object-keys column without a base offset".to_owned(),
            })?;
            offsets.push(last);
            valid.push(false);
            continue;
        }
        let mut lexer = Lexer::new(json.value(i), flags);
        match object_keys(&mut lexer) {
            Ok(keys) => {
                for key in keys {
                    items.append_value(&key);
                }
                let next = i32::try_from(items.len()).map_err(|_| InternalError {
                    message: "a JSON object-keys column past the 32-bit offset limit".to_owned(),
                })?;
                offsets.push(next);
                valid.push(true);
            }
            Err(_) => {
                let last = offsets.last().copied().ok_or_else(|| InternalError {
                    message: "a JSON object-keys column without a base offset".to_owned(),
                })?;
                offsets.push(last);
                valid.push(false);
            }
        }
    }
    let field = Arc::new(Field::new("element", ArrowType::Utf8View, false));
    let nulls = if valid.iter().all(|v| *v) {
        None
    } else {
        Some(NullBuffer::from(valid))
    };
    arrow_array::ListArray::try_new(
        field,
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(items.finish()),
        nulls,
    )
    .map(|array| Arc::new(array) as ArrayRef)
    .map_err(|e| {
        krishiv_common::error::InternalError {
            message: e.to_string(),
        }
        .into()
    })
}

/// One path step.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Step {
    /// `.name` or `['name']`.
    Field(String),
    /// `[index]`.
    Index(usize),
    /// `[*]`.
    Wildcard,
}

/// Parses a JSONPath; `None` for a bad path.
fn parse_path(path: &str) -> Option<Vec<Step>> {
    let bytes = path.as_bytes();
    if bytes.first() != Some(&b'$') {
        return None;
    }
    let mut steps = Vec::new();
    let mut i = 1_usize;
    while i < bytes.len() {
        match bytes[i] {
            b'.' => {
                i = i.saturating_add(1);
                let start = i;
                while i < bytes.len() && !matches!(bytes[i], b'.' | b'[' | b']') {
                    i = i.saturating_add(1);
                }
                if start == i {
                    return None;
                }
                let name = core::str::from_utf8(bytes.get(start..i)?).ok()?;
                steps.push(Step::Field(name.to_owned()));
            }
            b'[' => {
                i = i.saturating_add(1);
                if bytes.get(i) == Some(&b'*') {
                    i = i.saturating_add(1);
                    if bytes.get(i) != Some(&b']') {
                        return None;
                    }
                    i = i.saturating_add(1);
                    steps.push(Step::Wildcard);
                    continue;
                }
                let quote = bytes.get(i).copied()?;
                if quote == b'\'' || quote == b'"' {
                    let raw = quoted(bytes, i, quote)?;
                    let value = RawStr { raw, quote }.unescape(false)?;
                    i = i.saturating_add(raw.len());
                    if bytes.get(i) != Some(&b']') {
                        return None;
                    }
                    i = i.saturating_add(1);
                    steps.push(Step::Field(value));
                } else {
                    let start = i;
                    while bytes.get(i).copied().is_some_and(|b| b.is_ascii_digit()) {
                        i = i.saturating_add(1);
                    }
                    if start == i || bytes.get(i) != Some(&b']') {
                        return None;
                    }
                    let index: usize = core::str::from_utf8(bytes.get(start..i)?)
                        .ok()?
                        .parse()
                        .ok()?;
                    i = i.saturating_add(1);
                    steps.push(Step::Index(index));
                }
            }
            _ => return None,
        }
    }
    Some(steps)
}

/// The quoted string at `i` (which is its quote), with its quotes.
fn quoted(bytes: &[u8], i: usize, quote: u8) -> Option<&str> {
    let mut j = i.saturating_add(1);
    loop {
        match bytes.get(j).copied()? {
            b if b == quote => {
                let end = j.saturating_add(1);
                return core::str::from_utf8(bytes.get(i..end)?).ok();
            }
            b'\\' => {
                j = j.saturating_add(2);
            }
            _ => j = j.saturating_add(1),
        }
    }
}

/// One matched value, owned (only matches are materialized).
#[derive(Debug, Clone, PartialEq)]
enum Owned {
    /// JSON `null`.
    Null,
    /// A boolean.
    Bool(bool),
    /// A number as written.
    Num(String),
    /// A string value.
    Str(String),
    /// An array.
    Array(Vec<Owned>),
    /// An object.
    Object(Vec<(String, Owned)>),
}

impl Owned {
    /// Renders the value (strings bare, anything else compact JSON).
    fn render_bare(&self) -> String {
        match self {
            Self::Str(s) => s.clone(),
            other => {
                let mut text = String::new();
                other.render(&mut text);
                text
            }
        }
    }

    /// Renders the value as compact JSON. Numbers render as Jackson
    /// renders the parsed value: a floating-point number as Java's
    /// `Double.toString` (`1.0e2` reads `100.0`), an integral number as
    /// written.
    fn render(&self, text: &mut String) {
        match self {
            Self::Null => text.push_str("null"),
            Self::Bool(true) => text.push_str("true"),
            Self::Bool(false) => text.push_str("false"),
            Self::Num(raw) => text.push_str(&render_number(raw)),
            Self::Str(s) => push_quoted(s, text),
            Self::Array(items) => {
                text.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        text.push(',');
                    }
                    item.render(text);
                }
                text.push(']');
            }
            Self::Object(fields) => {
                text.push('{');
                for (i, (name, value)) in fields.iter().enumerate() {
                    if i > 0 {
                        text.push(',');
                    }
                    push_quoted(name, text);
                    text.push(':');
                    value.render(text);
                }
                text.push('}');
            }
        }
    }
}

/// A captured number as Jackson renders it: floating-point numbers as
/// Java's `Double.toString`, integral numbers as written (a huge integral
/// stays exact, as Jackson's `BigInteger` does).
fn render_number(raw: &str) -> String {
    if raw.bytes().any(|b| b == b'.' || b == b'e' || b == b'E')
        && let Some(v) = crate::text::java_parse_double(raw)
    {
        return crate::text::java_double_string(v);
    }
    raw.to_owned()
}

/// Renders matches: none is `NULL`, one is the value (a matched JSON
/// `null` is a SQL `NULL`), several are a JSON array of the non-null
/// matches.
fn render_matches(matches: &[Owned]) -> Option<String> {
    let mut live = matches.iter().filter(|m| !matches!(m, Owned::Null));
    match (live.next(), live.next()) {
        (None, _) => None,
        (Some(one), None) => Some(one.render_bare()),
        (Some(first), Some(second)) => {
            let mut text = String::from("[");
            first.render(&mut text);
            text.push(',');
            second.render(&mut text);
            for m in live {
                text.push(',');
                m.render(&mut text);
            }
            text.push(']');
            Some(text)
        }
    }
}

/// The values `steps` selects in the value at the cursor.
fn select(lexer: &mut Lexer<'_>, steps: &[Step]) -> Result<Vec<Owned>, ParseError> {
    if steps.is_empty() {
        return Ok(vec![capture(lexer, 0)?]);
    }
    let (step, rest) = (&steps[0], &steps[1..]);
    match lexer.peek()? {
        None => Err(ParseError::Malformed),
        Some(Token::Null) => {
            lexer.next()?;
            Ok(Vec::new())
        }
        Some(Token::OpenBrace) => select_object(lexer, step, rest),
        Some(Token::OpenBracket) => select_array(lexer, step, rest),
        Some(_) => {
            lexer.skip_value()?;
            Ok(Vec::new())
        }
    }
}

/// The values `step` selects in the object at the cursor.
fn select_object(
    lexer: &mut Lexer<'_>,
    step: &Step,
    rest: &[Step],
) -> Result<Vec<Owned>, ParseError> {
    lexer.next()?;
    let mut out = Vec::new();
    if lexer.eat(Token::CloseBrace)? {
        return Ok(out);
    }
    // A named field matches only its first occurrence, as Jackson's field
    // lookup does; a wildcard matches every value.
    let mut matched = false;
    loop {
        let name = match lexer.raw_key()? {
            RawKey::Quoted(raw) => lexer.unescape(raw).ok_or(ParseError::Malformed)?,
            RawKey::Bare(word) => word.to_owned(),
        };
        lexer.skip_trivia()?;
        if !lexer.eat(Token::Colon)? {
            return Err(ParseError::Malformed);
        }
        let hit = match step {
            Step::Field(want) => name == *want,
            Step::Wildcard => true,
            Step::Index(_) => false,
        };
        if hit && (matches!(step, Step::Wildcard) || !matched) {
            matched = true;
            out.extend(select(lexer, rest)?);
        } else {
            lexer.skip_value()?;
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
    Ok(out)
}

/// The values `step` selects in the array at the cursor.
fn select_array(
    lexer: &mut Lexer<'_>,
    step: &Step,
    rest: &[Step],
) -> Result<Vec<Owned>, ParseError> {
    lexer.next()?;
    let mut out = Vec::new();
    if lexer.eat(Token::CloseBracket)? {
        return Ok(out);
    }
    let mut index = 0_usize;
    loop {
        let hit = match step {
            Step::Index(want) => index == *want,
            Step::Wildcard => true,
            Step::Field(_) => false,
        };
        if hit {
            out.extend(select(lexer, rest)?);
        } else {
            lexer.skip_value()?;
        }
        index = index.saturating_add(1);
        lexer.skip_trivia()?;
        if lexer.eat(Token::Comma)? {
            continue;
        }
        if lexer.eat(Token::CloseBracket)? {
            break;
        }
        return Err(ParseError::Malformed);
    }
    Ok(out)
}

/// The value at the cursor, materialized, nested `depth` deep.
fn capture(lexer: &mut Lexer<'_>, depth: usize) -> Result<Owned, ParseError> {
    if depth > super::parse::MAX_JSON_DEPTH {
        return Err(ParseError::TooDeep);
    }
    let deeper = depth.saturating_add(1);
    match lexer.next()? {
        Some(Token::Null) => Ok(Owned::Null),
        Some(Token::Bool(b)) => Ok(Owned::Bool(b)),
        Some(Token::Num(raw, _)) => Ok(Owned::Num(raw.to_owned())),
        Some(Token::Str(raw)) => Ok(Owned::Str(
            lexer.unescape(raw).ok_or(ParseError::Malformed)?,
        )),
        Some(Token::OpenBracket) => {
            let mut items = Vec::new();
            if lexer.eat(Token::CloseBracket)? {
                return Ok(Owned::Array(items));
            }
            loop {
                items.push(capture(lexer, deeper)?);
                lexer.skip_trivia()?;
                if lexer.eat(Token::Comma)? {
                    continue;
                }
                if lexer.eat(Token::CloseBracket)? {
                    break;
                }
                return Err(ParseError::Malformed);
            }
            Ok(Owned::Array(items))
        }
        Some(Token::OpenBrace) => {
            let mut fields = Vec::new();
            if lexer.eat(Token::CloseBrace)? {
                return Ok(Owned::Object(fields));
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
                fields.push((name, capture(lexer, deeper)?));
                lexer.skip_trivia()?;
                if lexer.eat(Token::Comma)? {
                    continue;
                }
                if lexer.eat(Token::CloseBrace)? {
                    break;
                }
                return Err(ParseError::Malformed);
            }
            Ok(Owned::Object(fields))
        }
        None | Some(_) => Err(ParseError::Malformed),
    }
}

/// The length of the top-level array: `Err` for anything else.
fn array_length(lexer: &mut Lexer<'_>) -> Result<i32, ParseError> {
    if !matches!(lexer.next()?, Some(Token::OpenBracket)) {
        return Err(ParseError::Malformed);
    }
    let mut count = 0_i32;
    if lexer.eat(Token::CloseBracket)? {
        return Ok(count);
    }
    loop {
        lexer.skip_value()?;
        count = count.checked_add(1).ok_or(ParseError::Malformed)?;
        lexer.skip_trivia()?;
        if lexer.eat(Token::Comma)? {
            continue;
        }
        if lexer.eat(Token::CloseBracket)? {
            break;
        }
        return Err(ParseError::Malformed);
    }
    Ok(count)
}

/// The keys of the top-level object: `Err` for anything else.
fn object_keys(lexer: &mut Lexer<'_>) -> Result<Vec<String>, ParseError> {
    if !matches!(lexer.next()?, Some(Token::OpenBrace)) {
        return Err(ParseError::Malformed);
    }
    let mut keys = Vec::new();
    if lexer.eat(Token::CloseBrace)? {
        return Ok(keys);
    }
    loop {
        let key = match lexer.raw_key()? {
            RawKey::Quoted(raw) => lexer.unescape(raw).ok_or(ParseError::Malformed)?,
            RawKey::Bare(word) => word.to_owned(),
        };
        keys.push(key);
        lexer.skip_trivia()?;
        if !lexer.eat(Token::Colon)? {
            return Err(ParseError::Malformed);
        }
        lexer.skip_value()?;
        lexer.skip_trivia()?;
        if lexer.eat(Token::Comma)? {
            continue;
        }
        if lexer.eat(Token::CloseBrace)? {
            break;
        }
        return Err(ParseError::Malformed);
    }
    Ok(keys)
}
