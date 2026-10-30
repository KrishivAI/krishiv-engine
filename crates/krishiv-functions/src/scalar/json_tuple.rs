// SPDX-License-Identifier: Apache-2.0
//! `json_tuple`: one row in, N string columns out.
//!
//! The binding returns a struct of N nullable `STRING`s (named `c0` ...
//! `c(N-1)`); the analyzer expands a `json_tuple` select item into field
//! accesses of that struct, so the JSON is parsed once per row. The kernel
//! itself is a plain scalar over Arrow columns.
//!
//! Value semantics (Spark 4.2's `JsonTuple`): the JSON text must be an
//! object; only top-level keys are looked up, by exact (case-sensitive)
//! name. A string value gives its decoded text; an integer gives its
//! digits; a number with a fraction or an exponent is read as a `double`
//! and printed as Java prints one (`1.0E5` is `100000.0`, `1.10` is `1.1`),
//! as Jackson's generator does; a boolean gives its text; `null` gives
//! NULL; an object or array value gives its compact JSON text (no spaces
//! between tokens, numbers normalised the same way, strings escaped as a
//! JSON generator writes them). A missing key gives NULL; a duplicate key
//! resolves to the last occurrence. Invalid JSON, a non-object top level
//! or a NULL input gives one row of N NULLs; a NULL field name gives a NULL
//! column.
//!
//! The scanner reads what Jackson reads with Spark's parser options: single
//! quoted strings and raw control characters inside strings are accepted;
//! unquoted names, `NaN`, leading zeros, a leading `+` or `.`, bad escapes
//! and trailing commas are not. It stops after the first value: text after
//! the closing `}` is ignored (`{"a":1} x` reads as `{"a":1}`), while a
//! malformed document before that point yields the all-NULL row. It
//! recurses with a depth cap (nesting past [`MAX_DEPTH`] is invalid, so
//! deeply nested input cannot overflow the stack).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::StringViewBuilder;
use arrow_array::{Array, ArrayRef, StringArray};
use arrow_schema::{DataType, Field, Fields};

use super::args::{Inputs, is_string};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["json_tuple"];

/// Nesting past this depth is invalid (Spark's parser would also fail the
/// document; the cap keeps the recursive scan stack-safe).
const MAX_DEPTH: usize = 1000;

/// Binds the `json_tuple` function: the JSON string and the N field names
/// (all `STRING` after the analyzer's checks; no casts are applied) to a
/// struct of N nullable `STRING`s named `c0` ... `c(N-1)`.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    if args.len() < 2 || args.iter().any(|a| !is_string(&a.data_type())) {
        return Some((listed, Err(bad_types(listed, args))));
    }
    let n = args.len() - 1;
    let fields: Fields = (0..n)
        .map(|i| Field::new(format!("c{i}"), DataType::Utf8View, true))
        .collect();
    Some((
        listed,
        Ok(Binding::new(
            DataType::Struct(fields.clone()),
            move |x: &Inputs<'_>| {
                let rows = x.rows();
                let json = x.strings(0)?;
                let mut names: Vec<StringArray> = Vec::with_capacity(n);
                for k in 0..n {
                    names.push(x.strings(k + 1)?);
                }
                let mut builders: Vec<StringViewBuilder> = (0..n)
                    .map(|_| StringViewBuilder::with_capacity(rows))
                    .collect();
                // The key index and the value buffer are reused across rows.
                // Field names may be columns, so the index is rebuilt only
                // when a row's names differ from the previous row's (literal
                // names build it once); a NULL name matches nothing.
                let mut wanted: HashMap<&str, Vec<usize>> = HashMap::new();
                let mut built_for: Option<usize> = None;
                let mut values: Vec<Option<String>> = vec![None; n];
                for row in 0..rows {
                    let same = built_for.is_some_and(|prev| {
                        names.iter().all(|c| {
                            c.is_null(prev) == c.is_null(row)
                                && (c.is_null(row) || c.value(prev) == c.value(row))
                        })
                    });
                    if !same {
                        wanted.clear();
                        for (k, column) in names.iter().enumerate() {
                            if !column.is_null(row) {
                                wanted.entry(column.value(row)).or_default().push(k);
                            }
                        }
                        built_for = Some(row);
                    }
                    values.iter_mut().for_each(|v| *v = None);
                    if !json.is_null(row)
                        && !wanted.is_empty()
                        && let Some(found) = extract(json.value(row), &wanted)
                    {
                        for (k, v) in found {
                            values[k] = v;
                        }
                    }
                    for (builder, value) in builders.iter_mut().zip(&values) {
                        match value {
                            Some(v) => builder.append_value(v),
                            None => builder.append_null(),
                        }
                    }
                }
                let children: Vec<ArrayRef> = builders
                    .into_iter()
                    .map(|mut b| Arc::new(b.finish()) as ArrayRef)
                    .collect();
                Ok(Arc::new(krishiv_kernels::r#struct::build(
                    fields.clone(),
                    children,
                    rows,
                )?) as ArrayRef)
            },
        )),
    ))
}

/// The values of the wanted top-level keys of `doc`: `None` when `doc` is
/// not a strictly valid JSON object (invalid JSON anywhere, or a non-object
/// top level, yields the all-NULL row).
fn extract(doc: &str, wanted: &HashMap<&str, Vec<usize>>) -> Option<Vec<(usize, Option<String>)>> {
    let bytes = doc.as_bytes();
    let mut pos = 0;
    skip_ws(bytes, &mut pos);
    if bytes.get(pos) != Some(&b'{') {
        return None;
    }
    pos += 1;
    let mut out = Vec::new();
    skip_ws(bytes, &mut pos);
    if bytes.get(pos) == Some(&b'}') {
        return Some(out);
    }
    loop {
        skip_ws(bytes, &mut pos);
        let key = parse_string(doc, &mut pos).ok()?;
        skip_ws(bytes, &mut pos);
        if bytes.get(pos) != Some(&b':') {
            return None;
        }
        pos += 1;
        skip_ws(bytes, &mut pos);
        if let Some(indexes) = wanted.get(key.as_str()) {
            let value = parse_value(doc, &mut pos, 0).ok()?;
            for &k in indexes {
                out.push((k, value.clone()));
            }
        } else {
            skip_value(doc, &mut pos, 0).ok()?;
        }
        skip_ws(bytes, &mut pos);
        match bytes.get(pos) {
            Some(b',') => pos += 1,
            Some(b'}') => return Some(out),
            _ => return None,
        }
    }
}

/// A parsed top-level value: decoded text for strings, the raw token for
/// numbers and booleans, `None` for `null`, and the compact text for
/// objects and arrays.
fn parse_value(doc: &str, pos: &mut usize, depth: usize) -> Result<Option<String>, Invalid> {
    let bytes = doc.as_bytes();
    skip_ws(bytes, pos);
    match bytes.get(*pos) {
        Some(b'"' | b'\'') => Ok(Some(parse_string(doc, pos)?)),
        Some(b'{' | b'[') => {
            if depth >= MAX_DEPTH {
                return Err(Invalid);
            }
            let mut out = String::new();
            append_compact(doc, pos, depth + 1, &mut out)?;
            Ok(Some(out))
        }
        Some(b't') => {
            consume(doc, pos, "true")?;
            Ok(Some("true".to_owned()))
        }
        Some(b'f') => {
            consume(doc, pos, "false")?;
            Ok(Some("false".to_owned()))
        }
        Some(b'n') => {
            consume(doc, pos, "null")?;
            Ok(None)
        }
        Some(c) if *c == b'-' || c.is_ascii_digit() => {
            let (start, end) = parse_number(doc, pos)?;
            Ok(Some(normal_number(&doc[start..end])))
        }
        _ => Err(Invalid),
    }
}

/// A number token as Jackson's generator prints it: an integer keeps its
/// digits (`-0` is `0`), anything with a fraction or an exponent goes
/// through a `double` and prints as Java's `Double.toString`.
fn normal_number(token: &str) -> String {
    if token.bytes().any(|b| matches!(b, b'.' | b'e' | b'E')) {
        return token.parse::<f64>().map_or_else(
            |_| token.to_owned(),
            krishiv_kernels::text::java_double_string,
        );
    }
    if token == "-0" {
        return "0".to_owned();
    }
    token.to_owned()
}

/// Validates the value at `pos` without keeping it (for keys nobody asked
/// for; the whole document still has to be valid).
fn skip_value(doc: &str, pos: &mut usize, depth: usize) -> Result<(), Invalid> {
    let bytes = doc.as_bytes();
    skip_ws(bytes, pos);
    match bytes.get(*pos) {
        Some(b'"' | b'\'') => {
            parse_string(doc, pos)?;
            Ok(())
        }
        Some(b'{') => {
            if depth >= MAX_DEPTH {
                return Err(Invalid);
            }
            *pos += 1;
            skip_ws(bytes, pos);
            if bytes.get(*pos) == Some(&b'}') {
                *pos += 1;
                return Ok(());
            }
            loop {
                skip_ws(bytes, pos);
                parse_string(doc, pos)?;
                skip_ws(bytes, pos);
                if bytes.get(*pos) != Some(&b':') {
                    return Err(Invalid);
                }
                *pos += 1;
                skip_value(doc, pos, depth + 1)?;
                skip_ws(bytes, pos);
                match bytes.get(*pos) {
                    Some(b',') => *pos += 1,
                    Some(b'}') => {
                        *pos += 1;
                        return Ok(());
                    }
                    _ => return Err(Invalid),
                }
            }
        }
        Some(b'[') => {
            if depth >= MAX_DEPTH {
                return Err(Invalid);
            }
            *pos += 1;
            skip_ws(bytes, pos);
            if bytes.get(*pos) == Some(&b']') {
                *pos += 1;
                return Ok(());
            }
            loop {
                skip_value(doc, pos, depth + 1)?;
                skip_ws(bytes, pos);
                match bytes.get(*pos) {
                    Some(b',') => *pos += 1,
                    Some(b']') => {
                        *pos += 1;
                        return Ok(());
                    }
                    _ => return Err(Invalid),
                }
            }
        }
        Some(b't') => consume(doc, pos, "true"),
        Some(b'f') => consume(doc, pos, "false"),
        Some(b'n') => consume(doc, pos, "null"),
        Some(c) if *c == b'-' || c.is_ascii_digit() => {
            parse_number(doc, pos)?;
            Ok(())
        }
        _ => Err(Invalid),
    }
}

/// Appends the compact text of the object or array at `pos` (no spaces
/// between tokens; number tokens verbatim; strings escaped as a JSON
/// generator writes them).
fn append_compact(
    doc: &str,
    pos: &mut usize,
    depth: usize,
    out: &mut String,
) -> Result<(), Invalid> {
    let bytes = doc.as_bytes();
    skip_ws(bytes, pos);
    match bytes.get(*pos) {
        Some(b'{') => {
            if depth > MAX_DEPTH {
                return Err(Invalid);
            }
            *pos += 1;
            out.push('{');
            skip_ws(bytes, pos);
            if bytes.get(*pos) == Some(&b'}') {
                *pos += 1;
                out.push('}');
                return Ok(());
            }
            loop {
                skip_ws(bytes, pos);
                let key = parse_string(doc, pos)?;
                skip_ws(bytes, pos);
                if bytes.get(*pos) != Some(&b':') {
                    return Err(Invalid);
                }
                *pos += 1;
                append_quoted(out, &key);
                out.push(':');
                append_value_compact(doc, pos, depth, out)?;
                skip_ws(bytes, pos);
                match bytes.get(*pos) {
                    Some(b',') => {
                        *pos += 1;
                        out.push(',');
                    }
                    Some(b'}') => {
                        *pos += 1;
                        out.push('}');
                        return Ok(());
                    }
                    _ => return Err(Invalid),
                }
            }
        }
        Some(b'[') => {
            if depth > MAX_DEPTH {
                return Err(Invalid);
            }
            *pos += 1;
            out.push('[');
            skip_ws(bytes, pos);
            if bytes.get(*pos) == Some(&b']') {
                *pos += 1;
                out.push(']');
                return Ok(());
            }
            loop {
                append_value_compact(doc, pos, depth, out)?;
                skip_ws(bytes, pos);
                match bytes.get(*pos) {
                    Some(b',') => {
                        *pos += 1;
                        out.push(',');
                    }
                    Some(b']') => {
                        *pos += 1;
                        out.push(']');
                        return Ok(());
                    }
                    _ => return Err(Invalid),
                }
            }
        }
        _ => Err(Invalid),
    }
}

/// Appends the compact text of any value at `pos`.
fn append_value_compact(
    doc: &str,
    pos: &mut usize,
    depth: usize,
    out: &mut String,
) -> Result<(), Invalid> {
    let bytes = doc.as_bytes();
    skip_ws(bytes, pos);
    match bytes.get(*pos) {
        Some(b'"' | b'\'') => {
            let text = parse_string(doc, pos)?;
            append_quoted(out, &text);
            Ok(())
        }
        Some(b'{' | b'[') => append_compact(doc, pos, depth + 1, out),
        Some(b't') => consume_into(doc, pos, "true", out),
        Some(b'f') => consume_into(doc, pos, "false", out),
        Some(b'n') => consume_into(doc, pos, "null", out),
        Some(c) if *c == b'-' || c.is_ascii_digit() => {
            let (start, end) = parse_number(doc, pos)?;
            out.push_str(&normal_number(&doc[start..end]));
            Ok(())
        }
        _ => Err(Invalid),
    }
}

/// Appends `text` quoted, escaping as a JSON generator writes strings:
/// short escapes where they exist, `\u00XX` for other control characters,
/// non-ASCII text as is.
fn append_quoted(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{09}' => out.push_str("\\t"),
            '\u{0A}' => out.push_str("\\n"),
            '\u{0C}' => out.push_str("\\f"),
            '\u{0D}' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u00{:02X}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// The string at `pos` (which points at its opening quote, `"` or `'`),
/// decoded. Raw control characters are kept, surrogate pairs combine and
/// lone surrogates are invalid.
fn parse_string(doc: &str, pos: &mut usize) -> Result<String, Invalid> {
    let bytes = doc.as_bytes();
    let quote = match bytes.get(*pos) {
        Some(q @ (b'"' | b'\'')) => *q,
        _ => return Err(Invalid),
    };
    *pos += 1;
    let mut out = String::new();
    loop {
        let start = *pos;
        let mut i = start;
        while i < bytes.len() && bytes[i] != quote && bytes[i] != b'\\' {
            i += 1;
        }
        out.push_str(doc.get(start..i).ok_or(Invalid)?);
        *pos = i;
        match bytes.get(i) {
            Some(q) if *q == quote => {
                *pos = i + 1;
                return Ok(out);
            }
            Some(b'\\') => push_escape(doc, pos, &mut out)?,
            _ => return Err(Invalid),
        }
    }
}

/// Pushes the escape at `pos` (which points at the backslash).
fn push_escape(doc: &str, pos: &mut usize, out: &mut String) -> Result<(), Invalid> {
    let bytes = doc.as_bytes();
    let e = *bytes.get(*pos + 1).ok_or(Invalid)?;
    *pos += 2;
    match e {
        b'"' => out.push('"'),
        b'\'' => out.push('\''),
        b'\\' => out.push('\\'),
        b'/' => out.push('/'),
        b'b' => out.push('\u{08}'),
        b'f' => out.push('\u{0C}'),
        b'n' => out.push('\n'),
        b'r' => out.push('\r'),
        b't' => out.push('\t'),
        b'u' => {
            let high = hex4(doc, pos)?;
            if (0xD800..0xDC00).contains(&high) {
                if bytes.get(*pos) == Some(&b'\\') && bytes.get(*pos + 1) == Some(&b'u') {
                    let low = hex4_at(doc, *pos + 2)?;
                    if !(0xDC00..0xE000).contains(&low) {
                        return Err(Invalid);
                    }
                    *pos += 6;
                    let c = 0x1_0000 + ((high - 0xD800) << 10) + (low - 0xDC00);
                    out.push(char::from_u32(c).ok_or(Invalid)?);
                } else {
                    return Err(Invalid);
                }
            } else if (0xDC00..0xE000).contains(&high) {
                return Err(Invalid);
            } else {
                out.push(char::from_u32(high).ok_or(Invalid)?);
            }
        }
        _ => return Err(Invalid),
    }
    Ok(())
}

/// The four hex digits at `pos`, advancing past them.
fn hex4(doc: &str, pos: &mut usize) -> Result<u32, Invalid> {
    let v = hex4_at(doc, *pos)?;
    *pos += 4;
    Ok(v)
}

/// The four hex digits at `at`, without advancing.
fn hex4_at(doc: &str, at: usize) -> Result<u32, Invalid> {
    let bytes = doc.as_bytes();
    if at.checked_add(4).is_none_or(|end| end > bytes.len()) {
        return Err(Invalid);
    }
    let mut v: u32 = 0;
    for &b in &bytes[at..at + 4] {
        v = v
            .checked_mul(16)
            .and_then(|v| v.checked_add(hex_val(b)?))
            .ok_or(Invalid)?;
    }
    Ok(v)
}

/// The value of a hex digit.
fn hex_val(b: u8) -> Option<u32> {
    match b {
        b'0'..=b'9' => Some(u32::from(b - b'0')),
        b'a'..=b'f' => Some(u32::from(b - b'a') + 10),
        b'A'..=b'F' => Some(u32::from(b - b'A') + 10),
        _ => None,
    }
}

/// The strictly-valid number token at `pos` (which points at `-` or a
/// digit): `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`.
fn parse_number(doc: &str, pos: &mut usize) -> Result<(usize, usize), Invalid> {
    let bytes = doc.as_bytes();
    let start = *pos;
    let mut i = start;
    if bytes.get(i) == Some(&b'-') {
        i += 1;
    }
    match bytes.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => {
            i += 1;
            while bytes.get(i).is_some_and(u8::is_ascii_digit) {
                i += 1;
            }
        }
        _ => return Err(Invalid),
    }
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        if !bytes.get(i).is_some_and(u8::is_ascii_digit) {
            return Err(Invalid);
        }
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
    }
    if matches!(bytes.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(bytes.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        if !bytes.get(i).is_some_and(u8::is_ascii_digit) {
            return Err(Invalid);
        }
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
    }
    *pos = i;
    Ok((start, i))
}

/// Consumes `word` at `pos`.
fn consume(doc: &str, pos: &mut usize, word: &str) -> Result<(), Invalid> {
    if doc.get(*pos..*pos + word.len()) == Some(word) {
        *pos += word.len();
        Ok(())
    } else {
        Err(Invalid)
    }
}

/// Consumes `word` at `pos`, appending it to `out`.
fn consume_into(doc: &str, pos: &mut usize, word: &str, out: &mut String) -> Result<(), Invalid> {
    consume(doc, pos, word)?;
    out.push_str(word);
    Ok(())
}

/// Skips JSON whitespace at `pos`.
fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while matches!(bytes.get(*pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        *pos += 1;
    }
}

/// Strict validation failed: the row is all NULLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Invalid;
