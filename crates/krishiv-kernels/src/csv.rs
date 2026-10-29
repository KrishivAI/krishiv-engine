// SPDX-License-Identifier: Apache-2.0
//! CSV record parsing and formatting for `from_csv` and `to_csv`.
//!
//! The tokenizer reproduces Univocity's handling of quotes, escapes,
//! `charToEscapeQuoteEscaping`, white space and unescaped quotes (the same
//! rules the CSV file format in `krishiv-io` uses; that format uses this
//! module). Conversion builds Arrow `StructArray`s directly from column
//! builders: the schema is resolved once at bind, then each batch fills one
//! builder per field.

#![allow(
    clippy::arithmetic_side_effects,
    reason = "tokenizer ported from krishiv-io, where positions are bounded by the input length"
)]
#![allow(
    clippy::too_many_lines,
    reason = "tokenizer state machine, as reviewed in krishiv-io"
)]

use std::sync::Arc;

use arrow_array::builder::{
    BinaryBuilder, BinaryViewBuilder, BooleanBuilder, Date32Builder, Decimal128Builder,
    Float32Builder, Float64Builder, Int8Builder, Int16Builder, Int32Builder, Int64Builder,
    LargeBinaryBuilder, LargeStringBuilder, StringBuilder, StringViewBuilder,
    TimestampMicrosecondBuilder,
};
use arrow_array::{Array, ArrayRef, StringArray, StructArray, new_null_array};
use arrow_schema::{DataType, Fields, TimeUnit};

use crate::datetime_pattern::DateTimePattern;
use crate::error::KernelError;
use crate::text::{
    java_decimal_string, java_double_string, java_float_string32, java_parse_decimal,
    java_parse_double, java_parse_float, java_parse_integral,
};

// ---------------------------------------------------------------------------
// Syntax and tokenizer (shared with the CSV file format).
// ---------------------------------------------------------------------------

/// What happens to a quote inside a quoted value that is followed by other
/// text (`unescapedQuoteHandling`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unescaped {
    /// Stop at the closing quote.
    StopAtClosingQuote,
    /// Back to the delimiter.
    BackToDelimiter,
    /// Stop at the delimiter.
    StopAtDelimiter,
    /// Skip the value.
    SkipValue,
    /// Raise an error.
    RaiseError,
}

/// Where a record ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Eol {
    /// The end of the text (the lines were split already).
    EndOfText,
    /// `\n`, `\r` or `\r\n` outside quotes.
    Default,
    /// This separator outside quotes.
    Custom(String),
}

/// A record that cannot be tokenized at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizeError {
    /// More columns than `maxColumns`.
    TooManyColumns,
    /// A value longer than `maxCharsPerColumn`.
    ValueTooLong,
    /// `unescapedQuoteHandling` is `RAISE_ERROR` and a quote was unescaped.
    UnescapedQuote,
}

/// A record of a multi-line text: its raw text, its values (or why it has
/// none) and the position after it.
pub type Record<'t> = (&'t str, Result<Vec<Token>, TokenizeError>, usize);

/// One value: `None` is Univocity's null (an empty unquoted value).
pub type Token = Option<String>;

/// The tokenizer's settings.
#[derive(Debug, Clone)]
pub struct Syntax {
    /// Field separator (may be several characters).
    pub sep: String,
    /// Quote character (`None` disables quoting).
    pub quote: Option<char>,
    /// Escape character in quoted values.
    pub escape: Option<char>,
    /// Character that escapes the escape character.
    pub escape_escape: Option<char>,
    /// Comment character.
    pub comment: Option<char>,
    /// Trim leading white space.
    pub ltrim: bool,
    /// Trim trailing white space.
    pub rtrim: bool,
    /// Text of an empty quoted value.
    pub empty_value: String,
    /// How a quote inside a value is handled.
    pub unescaped: Unescaped,
    /// Most columns of a record.
    pub max_columns: usize,
    /// Most characters of a value (`None`: any).
    pub max_chars: Option<usize>,
    /// Where a record ends.
    pub eol: Eol,
}

impl Default for Syntax {
    fn default() -> Self {
        Self {
            sep: ",".to_owned(),
            quote: Some('"'),
            escape: Some('\\'),
            escape_escape: None,
            comment: None,
            ltrim: false,
            rtrim: false,
            empty_value: String::new(),
            unescaped: Unescaped::StopAtDelimiter,
            max_columns: 20480,
            max_chars: None,
            eol: Eol::EndOfText,
        }
    }
}

/// One CSV record of text into values.
#[derive(Debug)]
pub struct Tokenizer {
    s: Syntax,
}

/// Position in the text.
struct Cursor<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn rest(&self) -> &'a str {
        &self.text[self.pos..]
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }
}

impl Tokenizer {
    /// Tokenizer with these settings.
    #[must_use]
    pub fn new(s: Syntax) -> Self {
        Self { s }
    }

    /// The next record of a multi-line `text` from `pos`, skipping empty and
    /// comment lines: its raw text, its values (or why it has none) and the
    /// position after it. `None` at the end of the text.
    pub fn next_record<'t>(
        &self,
        text: &'t str,
        mut pos: usize,
        comment: Option<char>,
    ) -> Option<Record<'t>> {
        loop {
            let rest = text.get(pos..).filter(|r| !r.is_empty())?;
            let mut c = Cursor { text, pos };
            let empty = self.eol_len(&c);
            if empty > 0 || comment.is_some_and(|ch| rest.starts_with(ch)) {
                while !self.at_eol(&c) {
                    c.bump();
                }
                pos = c.pos.wrapping_add(self.eol_len(&c));
                continue;
            }
            let mut tokens = Vec::new();
            let (res, end) = match self.record(text, pos, &mut tokens) {
                Ok(end) => (Ok(tokens), end),
                Err(e) => (Err(e), text.len()),
            };
            let raw = &text[pos..end];
            let raw = match &self.s.eol {
                Eol::Custom(sep) => raw.strip_suffix(sep.as_str()).unwrap_or(raw),
                _ => raw.trim_end_matches(['\r', '\n']),
            };
            return Some((raw, res, end));
        }
    }

    fn at_sep(&self, c: &Cursor<'_>) -> bool {
        c.rest().starts_with(self.s.sep.as_str())
    }

    /// The length of the record terminator at the cursor (0: none).
    fn eol_len(&self, c: &Cursor<'_>) -> usize {
        match &self.s.eol {
            Eol::EndOfText => 0,
            Eol::Default => match c.rest().as_bytes() {
                [b'\r', b'\n', ..] => 2,
                [b'\r' | b'\n', ..] => 1,
                _ => 0,
            },
            Eol::Custom(sep) => {
                if c.rest().starts_with(sep.as_str()) {
                    sep.len()
                } else {
                    0
                }
            }
        }
    }

    fn at_end(&self, c: &Cursor<'_>) -> bool {
        c.rest().is_empty() || self.eol_len(c) > 0
    }

    fn check_len(&self, value: &str) -> Result<(), TokenizeError> {
        match self.s.max_chars {
            Some(max) if value.chars().count() > max => Err(TokenizeError::ValueTooLong),
            _ => Ok(()),
        }
    }

    fn push(&self, out: &mut Vec<Token>, t: Token) -> Result<(), TokenizeError> {
        if let Some(v) = &t {
            self.check_len(v)?;
        }
        if out.len() >= self.s.max_columns {
            return Err(TokenizeError::TooManyColumns);
        }
        out.push(t);
        Ok(())
    }

    /// Skips white space (`<= ' '`) before a value, but not a separator or
    /// quote that is itself white space.
    fn skip_white(&self, c: &mut Cursor<'_>) {
        while let Some(ch) = c.peek() {
            if ch > ' ' || self.at_sep(c) || Some(ch) == self.s.quote || self.eol_len(c) > 0 {
                break;
            }
            c.bump();
        }
    }

    /// The record at `text[pos..]`: its values appended to `out`; returns
    /// the position after the record and its terminator.
    ///
    /// # Errors
    /// [`TokenizeError`] for a record the settings cannot represent.
    pub fn record(
        &self,
        text: &str,
        pos: usize,
        out: &mut Vec<Token>,
    ) -> Result<usize, TokenizeError> {
        let mut c = Cursor { text, pos };
        loop {
            if self.s.ltrim {
                self.skip_white(&mut c);
            }
            if self.at_sep(&c) || self.at_end(&c) {
                self.push(out, None)?;
            } else if Some(c.peek().unwrap_or('\0')) == self.s.quote && self.s.quote.is_some() {
                let t = self.quoted(&mut c, out)?;
                self.push(out, t)?;
            } else {
                let t = self.unquoted(&mut c);
                self.push(out, t)?;
            }
            if self.at_sep(&c) {
                c.pos += self.s.sep.len();
                if self.at_end(&c) {
                    self.push(out, None)?;
                    break;
                }
            } else {
                break;
            }
        }
        let n = self.eol_len(&c);
        Ok(c.pos + n)
    }

    /// With a separator of several characters Univocity doubles a quote that
    /// ends the last value of a record.
    fn double_closing_quote(&self, c: &Cursor<'_>, v: &mut String) {
        if self.s.sep.chars().nth(1).is_some()
            && !self.at_sep(c)
            && let Some(q) = self.s.quote
            && v.ends_with(q)
        {
            v.push(q);
        }
    }

    /// An unquoted value: up to the next separator.
    fn unquoted(&self, c: &mut Cursor<'_>) -> Token {
        let mut v = String::new();
        while !self.at_sep(c) && !self.at_end(c) {
            let Some(ch) = c.bump() else { break };
            v.push(ch);
        }
        self.double_closing_quote(c, &mut v);
        if self.s.rtrim {
            let t = v.trim_end_matches(|ch: char| ch <= ' ').len();
            v.truncate(t);
        }
        if v.is_empty() { None } else { Some(v) }
    }

    /// Whether a record ends at the cursor (or the text does).
    fn at_eol(&self, c: &Cursor<'_>) -> bool {
        self.eol_len(c) > 0 || c.rest().is_empty()
    }

    /// Consumes the rest of a value, up to the next separator or the end of
    /// the record, appending to `out`.
    fn rest_of_value(&self, c: &mut Cursor<'_>, out: &mut String) {
        while !self.at_sep(c) && !self.at_eol(c) {
            let Some(ch) = c.bump() else { break };
            out.push(ch);
        }
    }

    /// The rest of a value up to the next separator, with the escape
    /// character applied.
    fn escaped_rest(&self, c: &mut Cursor<'_>, out: &mut String) {
        let q = self.s.quote.unwrap_or('"');
        let mut plain_quote_last = false;
        while !self.at_sep(c) && !self.at_eol(c) {
            let Some(ch) = c.bump() else { break };
            plain_quote_last = false;
            if Some(ch) == self.s.escape && (self.at_sep(c) || self.at_eol(c)) {
                // An escape character before the separator escapes nothing.
            } else if Some(ch) == self.s.escape
                && let Some(n) = c.peek()
                && (n == q || Some(n) == self.s.escape_escape)
            {
                c.bump();
                out.push(n);
            } else {
                plain_quote_last = ch == q;
                out.push(ch);
            }
        }
        if plain_quote_last {
            out.pop();
        }
    }

    /// Skips to the next separator or the end of the record.
    fn skip_value(&self, c: &mut Cursor<'_>) {
        while !self.at_sep(c) && !self.at_eol(c) {
            c.bump();
        }
    }

    /// A quoted value; the cursor is on the opening quote.
    fn quoted(&self, c: &mut Cursor<'_>, out: &mut Vec<Token>) -> Result<Token, TokenizeError> {
        if self.s.escape.is_none()
            || self
                .s
                .escape_escape
                .is_some_and(|ee| Some(ee) != self.s.escape && Some(ee) != self.s.quote)
        {
            let mut probe = Cursor {
                text: c.text,
                pos: c.pos,
            };
            let mut scratch = Vec::new();
            if let Ok((token, true)) = self.quoted_machine(&mut probe, &mut scratch, false)
                && scratch.is_empty()
            {
                c.pos = probe.pos;
                return Ok(token);
            }
        }
        self.quoted_machine(c, out, true).map(|r| r.0)
    }

    /// The quoted-value state machine.
    fn quoted_machine(
        &self,
        c: &mut Cursor<'_>,
        out: &mut Vec<Token>,
        slow: bool,
    ) -> Result<(Token, bool), TokenizeError> {
        let q = self.s.quote.unwrap_or('"');
        let esc = self.s.escape;
        let esc_esc = self.s.escape_escape;
        let is_esc = |ch: char| Some(ch) == esc;
        c.bump();
        let mut v = String::new();
        let mut prev: Option<char> = None;
        let mut began = false;
        let mut unescaped = false;
        let mut skipped = false;
        loop {
            let ch = c.peek();
            if prev == Some(q) && (ch.is_none_or(|x| x <= ' ') || self.at_sep(c) || self.at_eol(c))
            {
                break;
            }
            let Some(ch) = ch else {
                if !began && v.is_empty() {
                    v.push(q);
                }
                self.check_len(&v)?;
                return Ok(((!skipped).then_some(v), false));
            };
            began = true;
            if ch != q && !is_esc(ch) {
                if prev == Some(q) {
                    match self.s.unescaped {
                        Unescaped::StopAtDelimiter => {
                            return Ok((self.stop_at_delimiter(c, &v, "")?, false));
                        }
                        Unescaped::StopAtClosingQuote => {
                            v.push(q);
                            prev = None;
                            unescaped = true;
                        }
                        Unescaped::BackToDelimiter => {
                            self.split_at_last_sep(&mut v, out)?;
                            prev = None;
                            unescaped = true;
                        }
                        Unescaped::SkipValue => {
                            self.skip_value(c);
                            return Ok((None, false));
                        }
                        Unescaped::RaiseError => return Err(TokenizeError::UnescapedQuote),
                    }
                }
                if prev.is_some() && prev == esc {
                    v.push(prev.unwrap_or_default());
                }
                while let Some(x) = c.peek() {
                    if x == q || is_esc(x) || (slow && Some(x) == esc_esc) {
                        break;
                    }
                    if x == '\r' && self.s.eol == Eol::Default {
                        v.push('\n');
                        c.bump();
                        if c.peek() == Some('\n') {
                            c.bump();
                        }
                        continue;
                    }
                    v.push(x);
                    c.bump();
                }
                prev = c.bump();
                continue;
            }
            let mut seen = Some(ch);
            if is_esc(ch) && prev.is_some() && prev == esc_esc {
                v.extend(esc);
                seen = None;
            } else if prev == esc && (esc.is_some() || slow) {
                if ch == q {
                    v.push(q);
                    seen = None;
                } else {
                    v.extend(prev);
                }
            } else if ch == q && prev == Some(q) {
                v.push(q);
            } else if prev == Some(q) {
                match self.s.unescaped {
                    Unescaped::SkipValue => {
                        self.skip_value(c);
                        if self.at_sep(c) {
                            c.pos += self.s.sep.len();
                        }
                        skipped = true;
                        prev = None;
                        began = true;
                        continue;
                    }
                    Unescaped::RaiseError => return Err(TokenizeError::UnescapedQuote),
                    _ => {
                        unescaped = true;
                        v.push(q);
                        v.push(ch);
                        c.bump();
                        self.escaped_rest(c, &mut v);
                        if self.at_sep(c) {
                            c.pos += self.s.sep.len();
                        }
                        prev = None;
                        began = true;
                        continue;
                    }
                }
            }
            c.bump();
            prev = seen;
            if unescaped && (self.at_sep(c) || self.at_eol(c)) {
                break;
            }
        }
        let mut white = String::new();
        while let Some(x) = c.peek() {
            if x > ' ' || self.at_sep(c) || self.at_eol(c) {
                break;
            }
            white.push(x);
            c.bump();
        }
        if white.is_empty() || self.at_sep(c) || self.at_eol(c) {
            self.check_len(&v)?;
            if skipped {
                return Ok((None, false));
            }
            let clean = white.is_empty() && !unescaped && self.at_sep(c);
            let token = Some(if v.is_empty() {
                self.s.empty_value.clone()
            } else {
                v
            });
            return Ok((token, clean));
        }
        let token = match self.s.unescaped {
            Unescaped::StopAtDelimiter | Unescaped::BackToDelimiter => {
                self.stop_at_delimiter(c, &v, &white)?
            }
            Unescaped::StopAtClosingQuote => {
                v.push(q);
                v.push_str(&white);
                self.quoted_rest(c, v)?
            }
            Unescaped::SkipValue => {
                self.skip_value(c);
                None
            }
            Unescaped::RaiseError => return Err(TokenizeError::UnescapedQuote),
        };
        Ok((token, false))
    }

    /// `BACK_TO_DELIMITER` split.
    fn split_at_last_sep(&self, v: &mut String, out: &mut Vec<Token>) -> Result<(), TokenizeError> {
        let sep = self.s.sep.as_str();
        let from = self
            .s
            .quote
            .and_then(|q| v.rfind(q))
            .map_or(0, |i| i + self.s.quote.map_or(0, char::len_utf8));
        if !v[from..].contains(sep) {
            return Ok(());
        }
        let text = std::mem::take(v);
        let (head, tail) = text.split_at(from);
        let mut parts: Vec<&str> = tail.split(sep).collect();
        let rest = parts.pop().unwrap_or_default();
        for (i, part) in parts.iter().enumerate() {
            let value = if i == 0 {
                format!("{head}{part}")
            } else {
                (*part).to_owned()
            };
            self.push(out, if value.is_empty() { None } else { Some(value) })?;
        }
        *v = rest.to_owned();
        Ok(())
    }

    /// `STOP_AT_DELIMITER` value.
    fn stop_at_delimiter(
        &self,
        c: &mut Cursor<'_>,
        content: &str,
        white: &str,
    ) -> Result<Token, TokenizeError> {
        let q = self.s.quote.unwrap_or('"');
        let mut v = String::with_capacity(content.len() + 2);
        v.push(q);
        v.push_str(content);
        v.push(q);
        v.push_str(white);
        self.rest_of_value(c, &mut v);
        if !white.is_empty() {
            self.double_closing_quote(c, &mut v);
        }
        if self.s.rtrim {
            let t = v.trim_end_matches(|ch: char| ch <= ' ').len();
            v.truncate(t);
        }
        self.check_len(&v)?;
        Ok(Some(v))
    }

    /// `STOP_AT_CLOSING_QUOTE` continuation.
    fn quoted_rest(&self, c: &mut Cursor<'_>, mut v: String) -> Result<Token, TokenizeError> {
        let q = self.s.quote.unwrap_or('"');
        while let Some(x) = c.bump() {
            if x == q && (self.at_sep(c) || self.at_eol(c)) {
                break;
            }
            v.push(x);
        }
        self.check_len(&v)?;
        Ok(Some(v))
    }
}

// ---------------------------------------------------------------------------
// Parse mode and field conversion.
// ---------------------------------------------------------------------------

/// How malformed records are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParseMode {
    /// Malformed fields are NULL; the corrupt record goes to its column.
    #[default]
    Permissive,
    /// Malformed records are NULL.
    DropMalformed,
    /// Malformed records raise.
    FailFast,
}

/// The CSV type of a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    /// `BOOLEAN`.
    Bool,
    /// Integral types by bits.
    Int(u32),
    /// `FLOAT`.
    F32,
    /// `DOUBLE`.
    F64,
    /// `DECIMAL(p, s)`.
    Decimal(u8, i8),
    /// `STRING`.
    Str,
    /// `BINARY`.
    Bin,
    /// `DATE`.
    Date,
    /// `TIMESTAMP`.
    Timestamp,
    /// `TIMESTAMP_NTZ`.
    TimestampNtz,
    /// `VOID`.
    Null,
    /// An array, map or struct (written only: `to_csv` shows it the way
    /// Spark prints a nested value; `from_csv` has no such column).
    Nested,
}

impl FieldKind {
    /// The kind for a column `to_csv` writes: as [`Self::of`], and the
    /// nested types too.
    ///
    /// # Errors
    /// `KernelError` for a type CSV cannot hold.
    pub fn of_write(dt: &DataType) -> Result<Self, KernelError> {
        match dt {
            DataType::List(_)
            | DataType::LargeList(_)
            | DataType::Map(..)
            | DataType::Struct(_) => Ok(Self::Nested),
            other => Self::of(other),
        }
    }

    /// The kind for an Arrow type.
    ///
    /// # Errors
    /// `KernelError` for a type CSV cannot hold.
    pub fn of(dt: &DataType) -> Result<Self, KernelError> {
        Ok(match dt {
            DataType::Boolean => Self::Bool,
            DataType::Int8 => Self::Int(8),
            DataType::Int16 => Self::Int(16),
            DataType::Int32 => Self::Int(32),
            DataType::Int64 => Self::Int(64),
            DataType::Float32 => Self::F32,
            DataType::Float64 => Self::F64,
            DataType::Decimal128(p, s) => Self::Decimal(*p, *s),
            DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 => Self::Str,
            DataType::Binary | DataType::BinaryView | DataType::LargeBinary => Self::Bin,
            DataType::Date32 => Self::Date,
            DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => Self::Timestamp,
            DataType::Timestamp(TimeUnit::Microsecond, None) => Self::TimestampNtz,
            DataType::Null => Self::Null,
            other => {
                return Err(KernelError::unsupported(format!(
                    "CSV with a column of type {other}"
                )));
            }
        })
    }
}

/// One converted value.
#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    /// NULL.
    Null,
    /// Boolean.
    Bool(bool),
    /// Integral.
    I64(i64),
    /// Decimal unscaled.
    I128(i128),
    /// Float.
    F32(f32),
    /// Double.
    F64(f64),
    /// Text.
    Text(String),
    /// Binary bytes.
    Bytes(Vec<u8>),
}

/// Why a value did not convert.
#[derive(Debug)]
pub enum ConvError {
    /// The text is not a value of the type.
    Malformed,
    /// The conversion cannot be done at all.
    Fatal(KernelError),
}

impl From<KernelError> for ConvError {
    fn from(e: KernelError) -> Self {
        Self::Fatal(e)
    }
}

/// Converts tokens with one set of options.
#[derive(Debug)]
pub struct Converter<'a> {
    /// Text reading as NULL.
    pub null_value: &'a str,
    /// NaN spelling.
    pub nan: &'a str,
    /// Positive infinity spelling.
    pub pos_inf: &'a str,
    /// Negative infinity spelling.
    pub neg_inf: &'a str,
    /// Strict date pattern.
    pub date: Option<&'a DateTimePattern>,
    /// Whether to fall back to the lenient grammar.
    pub fallback: bool,
}

impl Converter<'_> {
    /// `token` as a value of `kind` (dates via `parse_date`; timestamps are
    /// handled by the caller with its zone).
    ///
    /// # Errors
    /// [`ConvError::Malformed`] for text that is not a value of the type.
    pub fn convert(&self, kind: &FieldKind, token: Option<&str>) -> Result<Val, ConvError> {
        let Some(s) = token else {
            return Ok(Val::Null);
        };
        if s == self.null_value {
            return Ok(Val::Null);
        }
        let bad = || ConvError::Malformed;
        Ok(match kind {
            FieldKind::Null => Val::Null,
            FieldKind::Str => Val::Text(s.to_owned()),
            FieldKind::Bin => Val::Bytes(s.as_bytes().to_vec()),
            FieldKind::Bool => {
                if s.eq_ignore_ascii_case("true") {
                    Val::Bool(true)
                } else if s.eq_ignore_ascii_case("false") {
                    Val::Bool(false)
                } else {
                    return Err(bad());
                }
            }
            FieldKind::Int(bits) => Val::I64(java_parse_integral(s, *bits).ok_or_else(bad)?),
            FieldKind::F32 => Val::F32(if s == self.nan {
                f32::NAN
            } else if s == self.neg_inf {
                f32::NEG_INFINITY
            } else if s == self.pos_inf {
                f32::INFINITY
            } else {
                java_parse_float(s).ok_or_else(bad)?
            }),
            FieldKind::F64 => Val::F64(if s == self.nan {
                f64::NAN
            } else if s == self.neg_inf {
                f64::NEG_INFINITY
            } else if s == self.pos_inf {
                f64::INFINITY
            } else {
                java_parse_double(s).ok_or_else(bad)?
            }),
            FieldKind::Decimal(p, sc) => {
                let sc = u8::try_from(*sc).map_err(|_| bad())?;
                let v = java_parse_decimal(&s.replace(',', ""), *p, sc).ok_or_else(bad)?;
                Val::I128(v)
            }
            FieldKind::Date => {
                let parsed = match self.date {
                    Some(p) => p
                        .parse_date(s)
                        .map_err(|e| ConvError::Fatal(KernelError::with_row(None, e)))?,
                    None => crate::text::spark_string_to_date(s),
                };
                match parsed {
                    Some(d) => Val::I64(i64::from(d)),
                    None if self.fallback && self.date.is_some() => {
                        match crate::text::spark_string_to_date(s) {
                            Some(d) => Val::I64(i64::from(d)),
                            None => return Err(bad()),
                        }
                    }
                    None => return Err(bad()),
                }
            }
            FieldKind::Timestamp | FieldKind::TimestampNtz | FieldKind::Nested => {
                return Err(bad());
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Column builders.
// ---------------------------------------------------------------------------

/// A string column of any Arrow string type.
#[derive(Debug)]
enum StrBuilder {
    /// View.
    View(StringViewBuilder),
    /// Plain.
    Utf8(StringBuilder),
    /// Large.
    Large(LargeStringBuilder),
}

impl StrBuilder {
    /// Builder for `dt`, if it is a string type.
    fn for_type(dt: &DataType) -> Option<Self> {
        match dt {
            DataType::Utf8View => Some(Self::View(StringViewBuilder::new())),
            DataType::Utf8 => Some(Self::Utf8(StringBuilder::new())),
            DataType::LargeUtf8 => Some(Self::Large(LargeStringBuilder::new())),
            _ => None,
        }
    }

    fn append(&mut self, s: Option<&str>) {
        match (self, s) {
            (Self::View(b), s) => b.append_option(s),
            (Self::Utf8(b), s) => b.append_option(s),
            (Self::Large(b), s) => b.append_option(s),
        }
    }

    fn finish(&mut self) -> ArrayRef {
        match self {
            Self::View(b) => Arc::new(b.finish()),
            Self::Utf8(b) => Arc::new(b.finish()),
            Self::Large(b) => Arc::new(b.finish()),
        }
    }
}

/// A binary column of any Arrow binary type.
#[derive(Debug)]
enum BinBuilder {
    /// Plain.
    Plain(BinaryBuilder),
    /// View.
    View(BinaryViewBuilder),
    /// Large.
    Large(LargeBinaryBuilder),
}

impl BinBuilder {
    /// Builder for `dt`, if it is a binary type.
    fn for_type(dt: &DataType) -> Option<Self> {
        match dt {
            DataType::Binary => Some(Self::Plain(BinaryBuilder::new())),
            DataType::BinaryView => Some(Self::View(BinaryViewBuilder::new())),
            DataType::LargeBinary => Some(Self::Large(LargeBinaryBuilder::new())),
            _ => None,
        }
    }

    fn append(&mut self, v: Option<&[u8]>) {
        match self {
            Self::Plain(b) => match v {
                Some(x) => b.append_value(x),
                None => b.append_null(),
            },
            Self::View(b) => match v {
                Some(x) => b.append_value(x),
                None => b.append_null(),
            },
            Self::Large(b) => match v {
                Some(x) => b.append_value(x),
                None => b.append_null(),
            },
        }
    }

    fn finish(&mut self) -> ArrayRef {
        match self {
            Self::Plain(b) => Arc::new(b.finish()),
            Self::View(b) => Arc::new(b.finish()),
            Self::Large(b) => Arc::new(b.finish()),
        }
    }
}

/// Builds one output column.
#[derive(Debug)]
enum Column {
    /// Boolean.
    Bool(BooleanBuilder),
    /// Int8.
    I8(Int8Builder),
    /// Int16.
    I16(Int16Builder),
    /// Int32.
    I32(Int32Builder),
    /// Int64.
    I64(Int64Builder),
    /// Float32.
    F32(Float32Builder),
    /// Float64.
    F64(Float64Builder),
    /// Decimal.
    Decimal(Decimal128Builder),
    /// String.
    Str(StrBuilder),
    /// Binary.
    Bin(BinBuilder),
    /// Date.
    Date(Date32Builder),
    /// Timestamp.
    Timestamp(TimestampMicrosecondBuilder),
    /// Null.
    Null(usize, DataType),
}

impl Column {
    /// A builder for the Arrow type `dt`.
    ///
    /// # Errors
    /// `KernelError` for a type CSV cannot hold.
    fn new(dt: &DataType) -> Result<Self, KernelError> {
        FieldKind::of(dt)?;
        Ok(match dt {
            DataType::Boolean => Self::Bool(BooleanBuilder::new()),
            DataType::Int8 => Self::I8(Int8Builder::new()),
            DataType::Int16 => Self::I16(Int16Builder::new()),
            DataType::Int32 => Self::I32(Int32Builder::new()),
            DataType::Int64 => Self::I64(Int64Builder::new()),
            DataType::Float32 => Self::F32(Float32Builder::new()),
            DataType::Float64 => Self::F64(Float64Builder::new()),
            DataType::Decimal128(p, s) => Self::Decimal(
                Decimal128Builder::new()
                    .with_precision_and_scale(*p, *s)
                    .map_err(|e| KernelError::internal(e.to_string()))?,
            ),
            DataType::Date32 => Self::Date(Date32Builder::new()),
            DataType::Timestamp(_, tz) => {
                let b = TimestampMicrosecondBuilder::new();
                Self::Timestamp(match tz {
                    Some(tz) => b.with_timezone(Arc::clone(tz)),
                    None => b,
                })
            }
            DataType::Null => Self::Null(0, DataType::Null),
            other => match StrBuilder::for_type(other) {
                Some(b) => Self::Str(b),
                None => match BinBuilder::for_type(other) {
                    Some(b) => Self::Bin(b),
                    None => {
                        return Err(KernelError::unsupported(format!(
                            "CSV column of type {other}"
                        )));
                    }
                },
            },
        })
    }

    /// Appends a value (a mismatched value is null).
    fn append(&mut self, v: &Val) {
        match (self, v) {
            (Self::Bool(b), Val::Bool(x)) => b.append_value(*x),
            (Self::I8(b), Val::I64(x)) => b.append_option(i8::try_from(*x).ok()),
            (Self::I16(b), Val::I64(x)) => b.append_option(i16::try_from(*x).ok()),
            (Self::I32(b), Val::I64(x)) => b.append_option(i32::try_from(*x).ok()),
            (Self::I64(b), Val::I64(x)) => b.append_value(*x),
            (Self::F32(b), Val::F32(x)) => b.append_value(*x),
            (Self::F64(b), Val::F64(x)) => b.append_value(*x),
            (Self::Decimal(b), Val::I128(x)) => b.append_value(*x),
            (Self::Str(b), Val::Text(x)) => b.append(Some(x)),
            (Self::Bin(b), Val::Bytes(x)) => b.append(Some(x)),
            (Self::Date(b), Val::I64(x)) => b.append_option(i32::try_from(*x).ok()),
            (Self::Timestamp(b), Val::I64(x)) => b.append_value(*x),
            (Self::Null(n, _), _) => *n += 1,
            (c, _) => c.append_null(),
        }
    }

    /// Appends a NULL.
    fn append_null(&mut self) {
        match self {
            Self::Bool(b) => b.append_null(),
            Self::I8(b) => b.append_null(),
            Self::I16(b) => b.append_null(),
            Self::I32(b) => b.append_null(),
            Self::I64(b) => b.append_null(),
            Self::F32(b) => b.append_null(),
            Self::F64(b) => b.append_null(),
            Self::Decimal(b) => b.append_null(),
            Self::Str(b) => b.append(None),
            Self::Bin(b) => b.append(None),
            Self::Date(b) => b.append_null(),
            Self::Timestamp(b) => b.append_null(),
            Self::Null(n, _) => *n += 1,
        }
    }

    /// The column built so far (the builder starts over).
    fn finish(&mut self) -> ArrayRef {
        match self {
            Self::Bool(b) => Arc::new(b.finish()),
            Self::I8(b) => Arc::new(b.finish()),
            Self::I16(b) => Arc::new(b.finish()),
            Self::I32(b) => Arc::new(b.finish()),
            Self::I64(b) => Arc::new(b.finish()),
            Self::F32(b) => Arc::new(b.finish()),
            Self::F64(b) => Arc::new(b.finish()),
            Self::Decimal(b) => Arc::new(b.finish()),
            Self::Str(b) => b.finish(),
            Self::Bin(b) => b.finish(),
            Self::Date(b) => Arc::new(b.finish()),
            Self::Timestamp(b) => Arc::new(b.finish()),
            Self::Null(n, dt) => {
                let rows = std::mem::take(n);
                new_null_array(dt, rows)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// `from_csv`.
// ---------------------------------------------------------------------------

/// Options of `from_csv` resolved once at bind.
#[derive(Debug, Clone)]
pub struct FromCsvOptions {
    /// Tokenizer settings.
    pub syntax: Syntax,
    /// How malformed records are handled.
    pub mode: ParseMode,
    /// Text reading as NULL.
    pub null_value: String,
    /// NaN spelling.
    pub nan: String,
    /// Positive infinity spelling.
    pub pos_inf: String,
    /// Negative infinity spelling.
    pub neg_inf: String,
    /// Strict date pattern.
    pub date_pattern: Option<DateTimePattern>,
    /// Strict timestamp pattern.
    pub timestamp_pattern: Option<DateTimePattern>,
    /// Strict timestamp_ntz pattern.
    pub ntz_pattern: Option<DateTimePattern>,
    /// Fall back to the lenient grammar when the pattern fails.
    pub fallback: bool,
}

/// A parsed row as Spark prints a failed record: `[1,null,abc]`.
fn row_text(vals: &[Val]) -> String {
    let mut out = String::from("[");
    for (i, v) in vals.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        match v {
            Val::Null => out.push_str("null"),
            Val::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Val::I64(n) => out.push_str(&n.to_string()),
            Val::I128(n) => out.push_str(&n.to_string()),
            Val::F32(f) => out.push_str(&java_float_string32(*f)),
            Val::F64(f) => out.push_str(&java_double_string(*f)),
            Val::Text(t) => out.push_str(t),
            Val::Bytes(b) => binary_text(b, &mut out),
        }
    }
    out.push(']');
    out
}

/// Parses `input` rows into the struct `fields`.
///
/// `parse_timestamp` parses a `TIMESTAMP` (an instant, micros since the
/// epoch) with the session zone; `parse_ntz` a `TIMESTAMP_NTZ` (local
/// micros). Timestamps with a strict pattern are parsed here with the
/// pattern and `zone`; the closures only run for the lenient grammar.
///
/// # Errors
/// `KernelError` for a `FAILFAST` malformed record.
#[expect(
    clippy::too_many_arguments,
    reason = "options, zone and parsers travel together for one batch"
)]
pub fn from_csv(
    input: &StringArray,
    fields: &Fields,
    kinds: &[FieldKind],
    corrupt: Option<usize>,
    o: &FromCsvOptions,
    zone: &dyn crate::cast::TimeZoneRules,
    parse_timestamp: &impl Fn(&str) -> Result<Option<i64>, KernelError>,
    parse_ntz: &impl Fn(&str) -> Result<Option<i64>, KernelError>,
) -> Result<StructArray, KernelError> {
    let tok = Tokenizer::new(o.syntax.clone());
    let conv = Converter {
        null_value: &o.null_value,
        nan: &o.nan,
        pos_inf: &o.pos_inf,
        neg_inf: &o.neg_inf,
        date: o.date_pattern.as_ref(),
        fallback: o.fallback,
    };
    let mut cols: Vec<Column> = fields
        .iter()
        .map(|f| Column::new(f.data_type()))
        .collect::<Result<_, _>>()?;
    let data: Vec<usize> = (0..fields.len()).filter(|i| Some(*i) != corrupt).collect();
    // The token position of each field (the corrupt-record column has none).
    let mut slot: Vec<Option<usize>> = vec![None; fields.len()];
    for (k, i) in data.iter().enumerate() {
        slot[*i] = Some(k);
    }
    let mut vals: Vec<Val> = Vec::with_capacity(fields.len());
    let mut tokens: Vec<Token> = Vec::new();
    let mut valid: Vec<bool> = Vec::with_capacity(input.len());
    for row in 0..input.len() {
        if input.is_null(row) {
            for c in &mut cols {
                c.append_null();
            }
            valid.push(false);
            continue;
        }
        let text = input.value(row);
        tokens.clear();
        if tok.record(text, 0, &mut tokens).is_err() {
            match o.mode {
                ParseMode::DropMalformed => {
                    for c in &mut cols {
                        c.append_null();
                    }
                    valid.push(false);
                    continue;
                }
                ParseMode::FailFast => {
                    return Err(KernelError::malformed_record(text));
                }
                ParseMode::Permissive => {
                    for (i, c) in cols.iter_mut().enumerate() {
                        if Some(i) == corrupt {
                            c.append(&Val::Text(text.to_owned()));
                        } else {
                            c.append_null();
                        }
                    }
                    valid.push(true);
                    continue;
                }
            }
        }
        let mut bad = tokens.len() != data.len();
        vals.clear();
        for (i, kind) in kinds.iter().enumerate() {
            if Some(i) == corrupt {
                vals.push(Val::Null);
                continue;
            }
            let t = slot[i].and_then(|k| tokens.get(k).and_then(|x| x.as_deref()));
            let v = match kind {
                FieldKind::Timestamp => match t {
                    None => Val::Null,
                    Some(s) if s == o.null_value => Val::Null,
                    Some(s) => {
                        let parsed = match o.timestamp_pattern.as_ref() {
                            Some(p) => p
                                .parse_timestamp(s, zone)
                                .map_err(|e| KernelError::with_row(None, e))?,
                            None => parse_timestamp(s)?,
                        };
                        match parsed {
                            Some(m) => Val::I64(m),
                            None if o.fallback && o.timestamp_pattern.is_some() => {
                                match parse_timestamp(s)? {
                                    Some(m) => Val::I64(m),
                                    None => {
                                        bad = true;
                                        Val::Null
                                    }
                                }
                            }
                            None => {
                                bad = true;
                                Val::Null
                            }
                        }
                    }
                },
                FieldKind::TimestampNtz => match t {
                    None => Val::Null,
                    Some(s) if s == o.null_value => Val::Null,
                    Some(s) => {
                        let parsed = match o.ntz_pattern.as_ref() {
                            Some(p) => p
                                .parse_local(s, false)
                                .map_err(|e| KernelError::with_row(None, e))?,
                            None => parse_ntz(s)?,
                        };
                        match parsed {
                            Some(m) => Val::I64(m),
                            None => {
                                bad = true;
                                Val::Null
                            }
                        }
                    }
                },
                _ => match conv.convert(kind, t) {
                    Ok(v) => v,
                    Err(ConvError::Malformed) => {
                        bad = true;
                        Val::Null
                    }
                    Err(ConvError::Fatal(e)) => return Err(e),
                },
            };
            vals.push(v);
        }
        if bad {
            match o.mode {
                ParseMode::DropMalformed => {
                    for c in &mut cols {
                        c.append_null();
                    }
                    valid.push(false);
                    continue;
                }
                ParseMode::FailFast => {
                    // Spark shows the parsed row (`[1,null]`), not the text.
                    return Err(KernelError::malformed_record(&row_text(&vals)));
                }
                ParseMode::Permissive => {
                    for (v, i) in vals.iter_mut().zip(0..) {
                        if Some(i) == corrupt {
                            *v = Val::Text(text.to_owned());
                        }
                    }
                }
            }
        }
        for (c, v) in cols.iter_mut().zip(&vals) {
            c.append(v);
        }
        valid.push(true);
    }
    let arrays: Vec<ArrayRef> = cols.iter_mut().map(Column::finish).collect();
    let nulls = arrow_buffer::NullBuffer::from_iter(valid.iter().copied());
    let nulls = (nulls.null_count() > 0).then_some(nulls);
    Ok(StructArray::new(fields.clone(), arrays, nulls))
}

// ---------------------------------------------------------------------------
// `to_csv`.
// ---------------------------------------------------------------------------

/// Options of `to_csv` resolved once at bind.
#[derive(Debug, Clone)]
pub struct ToCsvOptions {
    /// Quoting settings (`sep`, `quote`, `escape`, `escape_escape`,
    /// `empty_value`).
    pub syntax: Syntax,
    /// Quote every value.
    pub quote_all: bool,
    /// Quote values holding a quote.
    pub escape_quotes: bool,
    /// Custom line separator inside values (`None`: `\n`).
    pub line_sep: Option<String>,
    /// Text written for NULL.
    pub null_value: String,
    /// Date pattern.
    pub date_pattern: Option<DateTimePattern>,
    /// Timestamp pattern.
    pub timestamp_pattern: Option<DateTimePattern>,
    /// Timestamp_ntz pattern.
    pub ntz_pattern: Option<DateTimePattern>,
}

/// Spark's text of a binary value: `[AB 01 FF]`.
fn binary_text(b: &[u8], out: &mut String) {
    use std::fmt::Write as _;
    out.push('[');
    for (i, byte) in b.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        let _ = write!(out, "{byte:02X}");
    }
    out.push(']');
}

/// One value as the file shows it. Whitespace at either end is dropped when
/// the options ask for it (`ignoreLeadingWhiteSpace` and
/// `ignoreTrailingWhiteSpace`, both on by default when writing), and a value
/// left empty is written as the empty value.
fn push_quoted(out: &mut String, value: Option<&str>, o: &ToCsvOptions) {
    let s = &o.syntax;
    let quote = s.quote.unwrap_or('\0');
    let (mut text, is_null) = match value {
        None => (o.null_value.as_str(), true),
        Some(v) => (v, false),
    };
    if !is_null {
        if s.ltrim {
            text = text.trim_start_matches(|c: char| c <= ' ');
        }
        if s.rtrim {
            text = text.trim_end_matches(|c: char| c <= ' ');
        }
    }
    if text.is_empty() {
        if is_null {
            if o.quote_all {
                out.push(quote);
                out.push(quote);
            }
        } else {
            out.push_str(&s.empty_value);
        }
        return;
    }
    let needs = o.quote_all
        || text.starts_with(quote)
        || text.contains(s.sep.as_str())
        || text.contains(['\n', '\r'])
        || (o.escape_quotes && text.contains(quote))
        || o.line_sep
            .as_ref()
            .is_some_and(|e| text.contains(e.as_str()));
    if !needs {
        out.push_str(text);
        return;
    }
    out.push(quote);
    for c in text.chars() {
        if c == quote {
            if let Some(e) = s.escape {
                out.push(e);
            }
        } else if Some(c) == s.escape
            && let Some(ee) = s.escape_escape
        {
            out.push(ee);
        }
        out.push(c);
    }
    out.push(quote);
}

/// Spark's text of a nested value: a struct `{a, b}`, an array `[a, b]`, a
/// map `{k -> v}`, a null element `null`, and scalars as their plain text
/// (dates and timestamps in the CSV formats).
fn nested_text(
    a: &ArrayRef,
    row: usize,
    out: &mut String,
    format_date: &impl Fn(i32) -> Result<String, KernelError>,
    format_timestamp: &impl Fn(i64) -> Result<String, KernelError>,
    format_ntz: &impl Fn(i64) -> Result<String, KernelError>,
) -> Result<(), KernelError> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{
        Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
        Int64Type, TimestampMicrosecondType,
    };
    use std::fmt::Write as _;
    if a.is_null(row) {
        out.push_str("null");
        return Ok(());
    }
    let each = |child: &ArrayRef, range: std::ops::Range<usize>, out: &mut String| {
        for (n, i) in range.enumerate() {
            if n > 0 {
                out.push_str(", ");
            }
            nested_text(child, i, out, format_date, format_timestamp, format_ntz)?;
        }
        Ok::<(), KernelError>(())
    };
    match a.data_type() {
        DataType::Boolean => out.push_str(if a.as_boolean().value(row) {
            "true"
        } else {
            "false"
        }),
        DataType::Int8 => {
            let _ = write!(out, "{}", a.as_primitive::<Int8Type>().value(row));
        }
        DataType::Int16 => {
            let _ = write!(out, "{}", a.as_primitive::<Int16Type>().value(row));
        }
        DataType::Int32 => {
            let _ = write!(out, "{}", a.as_primitive::<Int32Type>().value(row));
        }
        DataType::Int64 => {
            let _ = write!(out, "{}", a.as_primitive::<Int64Type>().value(row));
        }
        DataType::Float32 => out.push_str(&java_float_string32(
            a.as_primitive::<Float32Type>().value(row),
        )),
        DataType::Float64 => out.push_str(&java_double_string(
            a.as_primitive::<Float64Type>().value(row),
        )),
        DataType::Decimal128(_, sc) => out.push_str(&java_decimal_string(
            a.as_primitive::<Decimal128Type>().value(row),
            *sc,
        )),
        DataType::Utf8View => out.push_str(a.as_string_view().value(row)),
        DataType::Utf8 => out.push_str(a.as_string::<i32>().value(row)),
        DataType::LargeUtf8 => out.push_str(a.as_string::<i64>().value(row)),
        DataType::Date32 => out.push_str(&format_date(a.as_primitive::<Date32Type>().value(row))?),
        DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => out.push_str(&format_timestamp(
            a.as_primitive::<TimestampMicrosecondType>().value(row),
        )?),
        DataType::Timestamp(TimeUnit::Microsecond, None) => out.push_str(&format_ntz(
            a.as_primitive::<TimestampMicrosecondType>().value(row),
        )?),
        DataType::List(_) => {
            let list = a.as_list::<i32>();
            let o = list.value_offsets();
            let (lo, hi) = (
                usize::try_from(o[row]).unwrap_or(0),
                usize::try_from(o[row + 1]).unwrap_or(0),
            );
            out.push('[');
            each(list.values(), lo..hi, out)?;
            out.push(']');
        }
        DataType::LargeList(_) => {
            let list = a.as_list::<i64>();
            let o = list.value_offsets();
            let (lo, hi) = (
                usize::try_from(o[row]).unwrap_or(0),
                usize::try_from(o[row + 1]).unwrap_or(0),
            );
            out.push('[');
            each(list.values(), lo..hi, out)?;
            out.push(']');
        }
        DataType::Struct(_) => {
            let st = a.as_struct();
            out.push('{');
            for (n, child) in st.columns().iter().enumerate() {
                if n > 0 {
                    out.push_str(", ");
                }
                nested_text(child, row, out, format_date, format_timestamp, format_ntz)?;
            }
            out.push('}');
        }
        DataType::Map(..) => {
            let m = a.as_map();
            let o = m.value_offsets();
            let (lo, hi) = (
                usize::try_from(o[row]).unwrap_or(0),
                usize::try_from(o[row + 1]).unwrap_or(0),
            );
            out.push('{');
            for (n, i) in (lo..hi).enumerate() {
                if n > 0 {
                    out.push_str(", ");
                }
                nested_text(m.keys(), i, out, format_date, format_timestamp, format_ntz)?;
                out.push_str(" -> ");
                nested_text(
                    m.values(),
                    i,
                    out,
                    format_date,
                    format_timestamp,
                    format_ntz,
                )?;
            }
            out.push('}');
        }
        other => {
            return Err(KernelError::unsupported(format!(
                "CSV with a nested value of type {other}"
            )));
        }
    }
    Ok(())
}

/// Renders `col` rows into CSV lines.
///
/// # Errors
/// `KernelError` for a value the pattern cannot show.
pub fn struct_to_csv(
    col: &StructArray,
    kinds: &[FieldKind],
    o: &ToCsvOptions,
    format_date: &impl Fn(i32) -> Result<String, KernelError>,
    format_timestamp: &impl Fn(i64) -> Result<String, KernelError>,
    format_ntz: &impl Fn(i64) -> Result<String, KernelError>,
) -> Result<StringArray, KernelError> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{
        Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
        Int64Type, TimestampMicrosecondType,
    };
    use std::fmt::Write as _;
    let mut out = StringBuilder::new();
    let mut line = String::new();
    let mut cell = String::new();
    for row in 0..col.len() {
        line.clear();
        if col.is_null(row) {
            out.append_null();
            continue;
        }
        for (k, kind) in kinds.iter().enumerate() {
            if k > 0 {
                line.push_str(&o.syntax.sep);
            }
            let a = col.column(k);
            cell.clear();
            let present = if a.is_null(row) {
                false
            } else {
                match (kind, a.data_type()) {
                    (FieldKind::Bool, _) => {
                        cell.push_str(if a.as_boolean().value(row) {
                            "true"
                        } else {
                            "false"
                        });
                        true
                    }
                    (FieldKind::Int(8), _) => {
                        let _ = write!(cell, "{}", a.as_primitive::<Int8Type>().value(row));
                        true
                    }
                    (FieldKind::Int(16), _) => {
                        let _ = write!(cell, "{}", a.as_primitive::<Int16Type>().value(row));
                        true
                    }
                    (FieldKind::Int(32), _) => {
                        let _ = write!(cell, "{}", a.as_primitive::<Int32Type>().value(row));
                        true
                    }
                    (FieldKind::Int(_), _) => {
                        let _ = write!(cell, "{}", a.as_primitive::<Int64Type>().value(row));
                        true
                    }
                    (FieldKind::F32, _) => {
                        cell.push_str(&java_float_string32(
                            a.as_primitive::<Float32Type>().value(row),
                        ));
                        true
                    }
                    (FieldKind::F64, _) => {
                        cell.push_str(&java_double_string(
                            a.as_primitive::<Float64Type>().value(row),
                        ));
                        true
                    }
                    (FieldKind::Decimal(_, s), _) => {
                        cell.push_str(&java_decimal_string(
                            a.as_primitive::<Decimal128Type>().value(row),
                            *s,
                        ));
                        true
                    }
                    (FieldKind::Str, DataType::Utf8View) => {
                        cell.push_str(a.as_string_view().value(row));
                        true
                    }
                    (FieldKind::Str, DataType::Utf8) => {
                        cell.push_str(a.as_string::<i32>().value(row));
                        true
                    }
                    (FieldKind::Str, _) => {
                        cell.push_str(a.as_string::<i64>().value(row));
                        true
                    }
                    (FieldKind::Bin, DataType::BinaryView) => {
                        binary_text(a.as_binary_view().value(row), &mut cell);
                        true
                    }
                    (FieldKind::Bin, DataType::Binary) => {
                        binary_text(a.as_binary::<i32>().value(row), &mut cell);
                        true
                    }
                    (FieldKind::Bin, _) => {
                        binary_text(a.as_binary::<i64>().value(row), &mut cell);
                        true
                    }
                    (FieldKind::Date, _) => {
                        cell.push_str(&format_date(a.as_primitive::<Date32Type>().value(row))?);
                        true
                    }
                    (FieldKind::Timestamp, _) => {
                        cell.push_str(&format_timestamp(
                            a.as_primitive::<TimestampMicrosecondType>().value(row),
                        )?);
                        true
                    }
                    (FieldKind::TimestampNtz, _) => {
                        cell.push_str(&format_ntz(
                            a.as_primitive::<TimestampMicrosecondType>().value(row),
                        )?);
                        true
                    }
                    (FieldKind::Nested, _) => {
                        nested_text(a, row, &mut cell, format_date, format_timestamp, format_ntz)?;
                        true
                    }
                    (FieldKind::Null, _) => false,
                }
            };
            push_quoted(&mut line, present.then_some(cell.as_str()), o);
        }
        out.append_value(&line);
    }
    Ok(out.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;
    use arrow_array::cast::AsArray;
    use arrow_schema::{Field, Fields};

    /// Tokenizes one record.
    fn tokens_of(text: &str, syntax: Syntax) -> Vec<Token> {
        let tok = Tokenizer::new(syntax);
        let mut out = Vec::new();
        tok.record(text, 0, &mut out).unwrap();
        out
    }

    #[test]
    fn tokenizer_quotes_and_escapes() {
        let s = Syntax::default();
        assert_eq!(
            tokens_of("a,\"b,c\",d", s.clone()),
            [
                Some("a".to_owned()),
                Some("b,c".to_owned()),
                Some("d".to_owned())
            ]
        );
        // An escaped quote inside a quoted value.
        assert_eq!(
            tokens_of("\"a\\\"b\",c", s.clone()),
            [Some("a\"b".to_owned()), Some("c".to_owned())]
        );
        // Empty unquoted is null; quoted empty uses `empty_value`.
        assert_eq!(tokens_of(",", s.clone()), [None, None]);
        assert_eq!(tokens_of("\"\"", s), [Some(String::new())]);
    }

    #[test]
    fn tokenizer_sep_and_newlines_in_quotes() {
        let s = Syntax {
            sep: "|".to_owned(),
            ..Syntax::default()
        };
        assert_eq!(
            tokens_of("a|b|c", s),
            [
                Some("a".to_owned()),
                Some("b".to_owned()),
                Some("c".to_owned())
            ]
        );
        let s = Syntax::default();
        assert_eq!(
            tokens_of("\"a\nb\",c", s),
            [Some("a\nb".to_owned()), Some("c".to_owned())]
        );
    }

    #[test]
    fn from_csv_builds_structs() {
        let fields: Fields = vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8View, true),
        ]
        .into();
        let kinds = vec![FieldKind::Int(32), FieldKind::Str];
        let input = StringArray::from(vec![Some("1,hi"), None]);
        let o = FromCsvOptions {
            syntax: Syntax::default(),
            mode: ParseMode::Permissive,
            null_value: String::new(),
            nan: "NaN".to_owned(),
            pos_inf: "Inf".to_owned(),
            neg_inf: "-Inf".to_owned(),
            date_pattern: None,
            timestamp_pattern: None,
            ntz_pattern: None,
            fallback: false,
        };
        let zone = crate::cast::FixedOffset(0);
        let out = from_csv(
            &input,
            &fields,
            &kinds,
            None,
            &o,
            &zone,
            &|_| Ok(None),
            &|_| Ok(None),
        )
        .unwrap();
        assert_eq!(out.len(), 2);
        assert!(!out.is_null(0));
        assert!(out.is_null(1));
        assert_eq!(
            out.column(0)
                .as_primitive::<arrow_array::types::Int32Type>()
                .value(0),
            1
        );
    }

    #[test]
    fn struct_to_csv_quotes() {
        use arrow_array::StringViewArray;
        let fields: Fields = vec![Field::new("v", DataType::Utf8View, true)].into();
        let col = StructArray::new(
            fields,
            vec![Arc::new(StringViewArray::from(vec![Some("a,b"), None])) as ArrayRef],
            None,
        );
        let o = ToCsvOptions {
            syntax: Syntax::default(),
            quote_all: false,
            escape_quotes: true,
            line_sep: None,
            null_value: String::new(),
            date_pattern: None,
            timestamp_pattern: None,
            ntz_pattern: None,
        };
        let out = struct_to_csv(
            &col,
            &[FieldKind::Str],
            &o,
            &|_| Err(KernelError::unsupported("no date")),
            &|_| Err(KernelError::unsupported("no ts")),
            &|_| Err(KernelError::unsupported("no ntz")),
        )
        .unwrap();
        assert_eq!(out.value(0), "\"a,b\"");
        assert_eq!(out.value(1), "");
    }
}
