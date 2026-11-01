// SPDX-License-Identifier: Apache-2.0
//! The JSON tokenizer and decoder: flat objects (one per line, or spanning
//! lines with `multiLine`) into batches, with Spark's parse modes,
//! `columnNameOfCorruptRecord` and option handling.
//!
//! The tokenizer is a small flat-object scanner over the record's text. A
//! parsed record owns its field names and string values (like the CSV
//! decoder's owned tokens); numbers borrow nothing either, so a huge line
//! holds only its own bytes plus one batch. Nested values are recognised
//! iteratively (never recursed into) and treated as malformed records:
//! nulls plus the raw record in PERMISSIVE, dropped in DROPMALFORMED,
//! failed in FAILFAST. Only a nested *schema* is unsupported at open.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::SchemaRef;
use bytes::Bytes;
use krishiv_common::error::{
    InvalidJsonRootField, KrishivError, MalformedRecordInParsingCannotParseJsonArraysAsStructs,
    MalformedRecordInParsingWithoutSuggestion,
};
use krishiv_kernels::datetime_pattern::Purpose;

use super::convert::kind_of;
use super::options::{JsonOptions, Mode, Syntax};
use crate::csv::convert::{Column, Conv, Kind, Val};
use crate::error::{FormatError, FormatResult, internal};
use crate::format::{Decoder, ScanUnitMetrics, Step, UnitPlan};
use crate::io::ByteRange;
use crate::request::{ROW_INDEX_COLUMN, ScanRequest};
use crate::text::reader::{LineReader, ReadChunk, WholeReader};
use crate::text::{BATCH_ROWS, Datetimes, TextMetadata, spark_type_name};

/// Why a record is not a flat object: always the record's defect, never a
/// bug, so every arm maps to malformed-record handling.
pub(crate) type Reason = &'static str;

/// One scalar of a flat JSON object.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum JVal {
    Null,
    Bool(bool),
    /// The raw number text and whether it is fractional (`1.5`, `1e3`,
    /// `NaN`): fractional numbers never read as integers or dates.
    Num(String, bool),
    /// An unescaped string.
    Str(String),
    /// An object or array value, as compact JSON text: only a string column
    /// can hold it (Spark reads a nested value as its raw JSON text there).
    Nested(String),
}

/// A flat JSON object: field names in order with their values. Later
/// duplicates already won in the parser, as in Spark.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct JObject {
    pub fields: Vec<(String, JVal)>,
}

/// A top-level JSON value: a flat object, or anything else (an array, a
/// scalar) which the malformed-record path handles.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Top {
    Object(JObject),
    /// The raw text of the non-object value.
    Other(String),
}

/// The scanner's settings live in `super::options` ([`Syntax`]): the
/// tokenizer, decoder and options share the one struct.
/// The bytes of a record, with a cursor.
struct Cur<'t> {
    b: &'t [u8],
    pos: usize,
    syntax: Syntax,
}

impl Cur<'_> {
    fn at(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    /// Skips white space and, when allowed, comments.
    fn skip_gap(&mut self) {
        loop {
            match self.at() {
                Some(b' ' | b'\t' | b'\n' | b'\r') => self.pos += 1,
                Some(b'/') if self.syntax.comments => match self.b.get(self.pos + 1).copied() {
                    Some(b'/') => {
                        self.pos += 2;
                        while !matches!(self.at(), None | Some(b'\n')) {
                            self.pos += 1;
                        }
                    }
                    Some(b'*') => {
                        self.pos += 2;
                        loop {
                            match self.at() {
                                None => return,
                                Some(b'*') if self.b.get(self.pos + 1).copied() == Some(b'/') => {
                                    self.pos += 2;
                                    break;
                                }
                                Some(_) => self.pos += 1,
                            }
                        }
                    }
                    _ => return,
                },
                _ => return,
            }
        }
    }
}

/// Whether `c` may start a bare field name.
fn name_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_' || c == b'$'
}

/// Whether `c` may continue a bare field name.
fn name_char(c: u8) -> bool {
    name_start(c) || c.is_ascii_digit()
}

/// `####` as a `u32`, for `\u` escapes.
fn hex4(b: &[u8]) -> Option<u32> {
    if b.len() < 4 {
        return None;
    }
    let mut v = 0_u32;
    for c in &b[..4] {
        v = v.checked_mul(16)?.checked_add(u32::from(match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => return None,
        }))?;
    }
    Some(v)
}

/// Appends the `\uXXXX` escape at `c.pos` (c is on the `u`).
fn push_unicode(c: &mut Cur<'_>, out: &mut String) -> Result<(), Reason> {
    let Some(v) = c.b.get(c.pos + 1..c.pos + 5).and_then(hex4) else {
        return Err("bad unicode escape");
    };
    c.pos += 5;
    let v = if (0xD800..0xDC00).contains(&v) {
        // A high surrogate needs its low half.
        if c.b.get(c.pos).copied() != Some(b'\\') || c.b.get(c.pos + 1).copied() != Some(b'u') {
            return Err("lone surrogate");
        }
        let Some(lo) = c.b.get(c.pos + 2..c.pos + 6).and_then(hex4) else {
            return Err("bad unicode escape");
        };
        if !(0xDC00..0xE000).contains(&lo) {
            return Err("lone surrogate");
        }
        c.pos += 6;
        0x1_0000 + ((v - 0xD800) << 10) + (lo - 0xDC00)
    } else if (0xDC00..0xE000).contains(&v) {
        return Err("lone surrogate");
    } else {
        v
    };
    let Some(ch) = char::from_u32(v) else {
        return Err("bad unicode escape");
    };
    out.push(ch);
    Ok(())
}

/// A quoted string starting at `c.pos` (on the quote); ends after the
/// closing quote.
fn parse_quoted(c: &mut Cur<'_>, text: &str) -> Result<String, Reason> {
    let quote = c.at().ok_or("truncated string")?;
    if quote != b'"' && !(quote == b'\'' && c.syntax.single_quotes) {
        return Err("expected a string");
    }
    c.pos += 1;
    let mut out = String::new();
    loop {
        let b = c.at().ok_or("truncated string")?;
        if b == quote {
            c.pos += 1;
            return Ok(out);
        }
        if b == b'\\' {
            c.pos += 1;
            match c.at().ok_or("truncated string")? {
                b'"' => {
                    out.push('"');
                    c.pos += 1;
                }
                b'\'' => {
                    out.push('\'');
                    c.pos += 1;
                }
                b'\\' => {
                    out.push('\\');
                    c.pos += 1;
                }
                b'/' => {
                    out.push('/');
                    c.pos += 1;
                }
                b'b' => {
                    out.push('\u{8}');
                    c.pos += 1;
                }
                b'f' => {
                    out.push('\u{c}');
                    c.pos += 1;
                }
                b'n' => {
                    out.push('\n');
                    c.pos += 1;
                }
                b'r' => {
                    out.push('\r');
                    c.pos += 1;
                }
                b't' => {
                    out.push('\t');
                    c.pos += 1;
                }
                b'u' => push_unicode(c, &mut out)?,
                other => {
                    if c.syntax.backslash_any {
                        if other < 0x80 {
                            out.push(char::from(other));
                            c.pos += 1;
                        } else {
                            let rest = text.get(c.pos..).ok_or("truncated string")?;
                            let mut chars = rest.chars();
                            let Some(ch) = chars.next() else {
                                return Err("truncated string");
                            };
                            out.push(ch);
                            c.pos += ch.len_utf8();
                        }
                    } else {
                        return Err("bad escape");
                    }
                }
            }
            continue;
        }
        if b < 0x20 && !c.syntax.unquoted_controls {
            return Err("unescaped control character");
        }
        if b < 0x80 {
            out.push(char::from(b));
            c.pos += 1;
        } else {
            let rest = text.get(c.pos..).ok_or("truncated string")?;
            let mut chars = rest.chars();
            let Some(ch) = chars.next() else {
                return Err("truncated string");
            };
            out.push(ch);
            c.pos += ch.len_utf8();
        }
    }
}

/// A number starting at `c.pos`: its end and whether it is fractional.
/// `NaN` and `Infinity` (Spark's default non-numeric tokens) count as
/// fractional.
fn parse_number(c: &mut Cur<'_>) -> Result<(usize, bool), Reason> {
    let start = c.pos;
    if c.at() == Some(b'-') {
        c.pos += 1;
    }
    // Non-numeric tokens.
    for word in ["NaN", "Infinity"] {
        if let Some(rest) = c.b.get(c.pos..)
            && rest.len() >= word.len()
            && &rest[..word.len()] == word.as_bytes()
        {
            c.pos += word.len();
            return Ok((c.pos, true));
        }
    }
    match c.at() {
        Some(b'0') => {
            c.pos += 1;
            if c.at().is_some_and(|d| d.is_ascii_digit()) && !c.syntax.leading_zeros {
                return Err("leading zeros");
            }
            while c.at().is_some_and(|d| d.is_ascii_digit()) {
                c.pos += 1;
            }
        }
        Some(d) if d.is_ascii_digit() => {
            while c.at().is_some_and(|d| d.is_ascii_digit()) {
                c.pos += 1;
            }
        }
        _ => return Err("bad number"),
    }
    let mut float = false;
    if c.at() == Some(b'.') {
        float = true;
        c.pos += 1;
        if !c.at().is_some_and(|d| d.is_ascii_digit()) {
            return Err("bad number");
        }
        while c.at().is_some_and(|d| d.is_ascii_digit()) {
            c.pos += 1;
        }
    }
    if matches!(c.at(), Some(b'e' | b'E')) {
        float = true;
        c.pos += 1;
        if matches!(c.at(), Some(b'+' | b'-')) {
            c.pos += 1;
        }
        if !c.at().is_some_and(|d| d.is_ascii_digit()) {
            return Err("bad number");
        }
        while c.at().is_some_and(|d| d.is_ascii_digit()) {
            c.pos += 1;
        }
    }
    if c.pos == start || (c.pos == start + 1 && c.b.get(start).copied() == Some(b'-')) {
        return Err("bad number");
    }
    Ok((c.pos, float))
}

/// A literal (`true`, `false`, `null`) starting at `c.pos`.
fn parse_literal(c: &mut Cur<'_>) -> Result<JVal, Reason> {
    for (word, v) in [
        ("true", JVal::Bool(true)),
        ("false", JVal::Bool(false)),
        ("null", JVal::Null),
    ] {
        if let Some(rest) = c.b.get(c.pos..)
            && rest.len() >= word.len()
            && &rest[..word.len()] == word.as_bytes()
        {
            c.pos += word.len();
            return Ok(v);
        }
    }
    Err("bad literal")
}

/// Whether the scalar at the cursor ends here (white space, a structural
/// character, a comment start, or the end of the text).
fn scalar_end(c: &Cur<'_>) -> bool {
    match c.at() {
        None | Some(b' ' | b'\t' | b'\n' | b'\r' | b',' | b'}' | b']' | b':') => true,
        Some(b'/') if c.syntax.comments => true,
        _ => false,
    }
}

/// Skips any JSON value starting at `c.pos` (for nested values and
/// non-object records): iterative, so no depth limit to overflow.
fn skip_value(c: &mut Cur<'_>, text: &str) -> Result<(), Reason> {
    c.skip_gap();
    let mut depth = 0_u32;
    loop {
        c.skip_gap();
        match c.at() {
            None => return Err("truncated value"),
            Some(b'{' | b'[') => {
                depth = depth.checked_add(1).ok_or("too deeply nested")?;
                c.pos += 1;
            }
            Some(b'}' | b']') => {
                if depth == 0 {
                    return Err("unexpected bracket");
                }
                depth -= 1;
                c.pos += 1;
                if depth == 0 {
                    return Ok(());
                }
            }
            Some(b'"') => {
                parse_quoted(c, text)?;
                if depth == 0 {
                    return Ok(());
                }
                expect_sep(c)?;
            }
            Some(b'\'') => {
                if !c.syntax.single_quotes {
                    return Err("unexpected character");
                }
                parse_quoted(c, text)?;
                if depth == 0 {
                    return Ok(());
                }
                expect_sep(c)?;
            }
            Some(b'-' | b'0'..=b'9' | b'N' | b'I') => {
                parse_number(c)?;
                if !scalar_end(c) {
                    return Err("bad number");
                }
                if depth == 0 {
                    return Ok(());
                }
                expect_sep(c)?;
            }
            Some(b't' | b'f' | b'n') => {
                parse_literal(c)?;
                if !scalar_end(c) {
                    return Err("bad literal");
                }
                if depth == 0 {
                    return Ok(());
                }
                expect_sep(c)?;
            }
            Some(_) => return Err("unexpected character"),
        }
    }
}

/// After a scalar inside a container: the separator (consumed) or closing
/// bracket (left for the main loop) that must follow it.
fn expect_sep(c: &mut Cur<'_>) -> Result<(), Reason> {
    c.skip_gap();
    match c.at() {
        Some(b',' | b':') => {
            c.pos += 1;
            Ok(())
        }
        Some(b'}' | b']') => Ok(()),
        None => Err("truncated value"),
        Some(_) => Err("unexpected character"),
    }
}

/// `raw` without the whitespace between tokens (string contents are kept
/// as written).
fn compact_json(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for ch in raw.chars() {
        match quote {
            Some(q) => {
                out.push(ch);
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == q {
                    quote = None;
                }
            }
            None if ch == '"' || ch == '\'' => {
                quote = Some(ch);
                out.push(ch);
            }
            None if ch.is_whitespace() => {}
            None => out.push(ch),
        }
    }
    out
}

/// One field value starting at `c.pos`.
fn parse_value(c: &mut Cur<'_>, text: &str) -> Result<JVal, Reason> {
    c.skip_gap();
    match c.at() {
        None => Err("truncated value"),
        Some(b'{' | b'[') => {
            let start = c.pos;
            skip_value(c, text)?;
            let raw = text.get(start..c.pos).ok_or("truncated value")?;
            Ok(JVal::Nested(compact_json(raw)))
        }
        Some(b'"') => Ok(JVal::Str(parse_quoted(c, text)?)),
        Some(b'\'') => {
            if !c.syntax.single_quotes {
                return Err("unexpected character");
            }
            Ok(JVal::Str(parse_quoted(c, text)?))
        }
        Some(b'-' | b'0'..=b'9' | b'N' | b'I') => {
            let start = c.pos;
            let (end, float) = parse_number(c)?;
            if !scalar_end(c) {
                return Err("bad number");
            }
            let raw = text.get(start..end).ok_or("truncated value")?;
            Ok(JVal::Num(raw.to_owned(), float))
        }
        Some(b't' | b'f' | b'n') => {
            let v = parse_literal(c)?;
            if !scalar_end(c) {
                return Err("bad literal");
            }
            Ok(v)
        }
        Some(_) => Err("unexpected character"),
    }
}

/// A field name starting at `c.pos`.
fn parse_name(c: &mut Cur<'_>, text: &str) -> Result<String, Reason> {
    c.skip_gap();
    match c.at() {
        Some(b'"') => parse_quoted(c, text),
        Some(b'\'') => {
            if !c.syntax.single_quotes {
                return Err("expected a field name");
            }
            parse_quoted(c, text)
        }
        Some(b) if c.syntax.unquoted_names && name_start(b) => {
            let start = c.pos;
            while c.at().is_some_and(name_char) {
                c.pos += 1;
            }
            text.get(start..c.pos)
                .map(str::to_owned)
                .ok_or("truncated name")
        }
        Some(_) => Err("expected a field name"),
        None => Err("truncated object"),
    }
}

/// The members of the `{` at `c.pos` (the cursor is on `{`); ends after
/// `}`.
fn parse_members(c: &mut Cur<'_>, text: &str) -> Result<JObject, Reason> {
    c.pos += 1;
    let mut fields: Vec<(String, JVal)> = Vec::new();
    c.skip_gap();
    if c.at() == Some(b'}') {
        c.pos += 1;
        return Ok(JObject { fields });
    }
    loop {
        let name = parse_name(c, text)?;
        c.skip_gap();
        if c.at() != Some(b':') {
            return Err("expected a colon");
        }
        c.pos += 1;
        let value = parse_value(c, text)?;
        if let Some(slot) = fields.iter_mut().find(|(n, _)| *n == name) {
            slot.1 = value;
        } else {
            fields.push((name, value));
        }
        c.skip_gap();
        match c.at() {
            Some(b',') => c.pos += 1,
            Some(b'}') => {
                c.pos += 1;
                return Ok(JObject { fields });
            }
            Some(_) => return Err("expected a comma"),
            None => return Err("truncated object"),
        }
    }
}

/// The top-level value at `c.pos`: an object or anything else.
fn parse_top(c: &mut Cur<'_>, text: &str) -> Result<Top, Reason> {
    c.skip_gap();
    let start = c.pos;
    match c.at() {
        None => Err("empty record"),
        Some(b'{') => Ok(Top::Object(parse_members(c, text)?)),
        Some(_) => {
            skip_value(c, text)?;
            let raw = text.get(start..c.pos).ok_or("truncated value")?;
            Ok(Top::Other(raw.to_owned()))
        }
    }
}

/// One line of JSON-lines text: exactly one top-level value.
pub(crate) fn parse_line(text: &str, syntax: Syntax) -> Result<Top, Reason> {
    let mut c = Cur {
        b: text.as_bytes(),
        pos: 0,
        syntax,
    };
    let top = parse_top(&mut c, text)?;
    c.skip_gap();
    if c.pos != c.b.len() {
        return Err("trailing text");
    }
    Ok(top)
}

/// The next top-level value of multi-line text from `pos`: the value and
/// the position after it, or `None` at the end of the text.
pub(crate) fn parse_next(
    text: &str,
    pos: usize,
    syntax: Syntax,
) -> Result<Option<(Top, usize)>, Reason> {
    let mut c = Cur {
        b: text.as_bytes(),
        pos,
        syntax,
    };
    c.skip_gap();
    if c.pos >= c.b.len() {
        return Ok(None);
    }
    let top = parse_top(&mut c, text)?;
    Ok(Some((top, c.pos)))
}

/// A malformed record in `FAILFAST` mode: `MALFORMED_RECORD_IN_PARSING`
/// caused by the classified error, like the CSV decoder.
fn fail_fast(raw: &str, cause: Option<KrishivError>) -> FormatError {
    let e: KrishivError = MalformedRecordInParsingWithoutSuggestion {
        bad_record: raw.to_owned(),
        fail_fast_mode: "FAILFAST".to_owned(),
    }
    .into();
    FormatError::Failed(match cause {
        Some(c) => e.with_cause(c),
        None => e,
    })
}

/// What one output column is.
enum Out {
    /// The column of malformed records.
    Corrupt,
    /// A value from the record's field `slot`.
    Data { slot: usize, kind: Kind },
    /// `_metadata.row_index`: the record's position among the file's rows.
    RowIndex,
}

enum Source {
    Lines(LineReader),
    Whole {
        reader: WholeReader,
        text: Option<String>,
        pos: usize,
    },
}

/// The next record of a source: its top-level value (or why it has none)
/// with the raw text of the record.
enum Next {
    Record(Result<Top, Reason>, String),
    NeedData(Range<u64>),
    End,
}

pub(crate) struct JsonDecoder {
    source: Source,
    syntax: Syntax,
    o: JsonOptions,
    dt: Datetimes,
    outs: Vec<Out>,
    columns: Vec<Column>,
    schema: SchemaRef,
    /// Names of the data columns, for field lookup.
    data_names: Vec<String>,
    /// Spark type names of the data columns, for conversion errors.
    data_types: Vec<String>,
    /// Field name to data-column index.
    index: HashMap<String, usize>,
    /// Reads the lines before a split that starts after byte 0, to count
    /// the rows `_metadata.row_index` continues from (line mode only).
    prefix: Option<LineReader>,
    /// Rows accepted so far in the file, prefix included.
    row_no: i64,
    rows: usize,
    held_text: usize,
    done: bool,
    metrics: ScanUnitMetrics,
}

impl JsonDecoder {
    pub(crate) fn new(
        m: &Arc<dyn crate::format::FileMetadata>,
        units: &UnitPlan,
        req: &ScanRequest,
        o: &crate::options::FormatOptions,
    ) -> FormatResult<Self> {
        let m: TextMetadata = crate::text::downcast(m)?;
        let opts = JsonOptions::new(o, false)?;
        let dt = Datetimes::new(o, Purpose::Parse)?;
        let schema = req.output_schema()?;
        let table = &req.table_schema;
        let data: Vec<&str> = table
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .filter(|n| *n != opts.corrupt_column)
            .collect();
        let mut outs = Vec::with_capacity(schema.fields().len());
        let mut columns = Vec::with_capacity(schema.fields().len());
        for f in schema.fields() {
            if f.name() == ROW_INDEX_COLUMN {
                outs.push(Out::RowIndex);
            } else if f.name() == &opts.corrupt_column {
                outs.push(Out::Corrupt);
            } else {
                let slot = data
                    .iter()
                    .position(|n| *n == f.name())
                    .ok_or_else(|| internal(format!("no column {}", f.name())))?;
                let kind = kind_of(f.data_type())?;
                columns.push(Column::new(f.data_type())?);
                outs.push(Out::Data { slot, kind });
                continue;
            }
            columns.push(Column::new(f.data_type())?);
        }
        let data_names: Vec<String> = data.iter().map(|s| (*s).to_owned()).collect();
        let data_types: Vec<String> = table
            .fields()
            .iter()
            .filter(|f| f.name() != &opts.corrupt_column)
            .map(|f| spark_type_name(f.data_type()))
            .collect();
        let index = data_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let split = units.units.first().map_or(ByteRange::new(0, 0), |u| {
            ByteRange::new(u.bytes.start, u.bytes.end)
        });
        let source = if opts.multi_line {
            Source::Whole {
                reader: WholeReader::new(m.size, split).compressed(m.codec),
                text: None,
                pos: 0,
            }
        } else {
            Source::Lines(
                LineReader::new(m.size, split, true, opts.line_sep.clone()).compressed(m.codec),
            )
        };
        let metrics = ScanUnitMetrics {
            units: units.units.len() as u64,
            ..ScanUnitMetrics::default()
        };
        let prefix =
            (req.metadata_cols.row_index && split.start > 0 && !opts.multi_line).then(|| {
                LineReader::new(
                    m.size,
                    ByteRange::new(0, split.start),
                    true,
                    opts.line_sep.clone(),
                )
            });
        Ok(Self {
            syntax: Syntax::from(&opts),
            o: opts,
            dt,
            outs,
            columns,
            schema,
            data_names,
            data_types,
            index,
            source,
            prefix,
            row_no: 0,
            rows: 0,
            held_text: 0,
            done: units.units.is_empty(),
            metrics,
        })
    }

    fn next(&mut self) -> FormatResult<Next> {
        loop {
            if self.prefix.is_some() {
                let prefix = self.prefix.as_mut().ok_or_else(|| {
                    FormatError::Other(internal("a JSON prefix reader is missing"))
                })?;
                match prefix.next_line()? {
                    ReadChunk::NeedData(range) => return Ok(Next::NeedData(range)),
                    ReadChunk::End => {
                        self.prefix = None;
                        continue;
                    }
                    ReadChunk::Line(bytes, offset) => {
                        // Prefix rows are counted exactly as the main loop
                        // counts them: blank lines carry no record.
                        let bytes = if offset == 0 {
                            bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&bytes)
                        } else {
                            &bytes[..]
                        };
                        let line = self.o.charset.decode(bytes).into_owned();
                        if line.trim_matches(|c: char| c <= ' ').is_empty() {
                            continue;
                        }
                        self.row_no += 1;
                        continue;
                    }
                }
            }
            match &mut self.source {
                Source::Lines(main) => match main.next_line()? {
                    ReadChunk::NeedData(range) => return Ok(Next::NeedData(range)),
                    ReadChunk::End => return Ok(Next::End),
                    ReadChunk::Line(bytes, offset) => {
                        let bytes = if offset == 0 {
                            bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&bytes)
                        } else {
                            &bytes[..]
                        };
                        let line = self.o.charset.decode(bytes).into_owned();
                        if line.trim_matches(|c: char| c <= ' ').is_empty() {
                            continue;
                        }
                        let parsed = parse_line(&line, self.syntax);
                        return Ok(Next::Record(parsed, line));
                    }
                },
                Source::Whole { reader, text, pos } => {
                    if text.is_none() {
                        if let Some(r) = reader.need() {
                            return Ok(Next::NeedData(r));
                        }
                        match reader.take() {
                            Some(bytes) => {
                                let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&bytes);
                                *text = Some(self.o.charset.decode(bytes).into_owned());
                                *pos = 0;
                            }
                            None => return Ok(Next::End),
                        }
                    }
                    let Some(t) = text.as_deref() else {
                        return Ok(Next::End);
                    };
                    let (top, raw, next) = match parse_next(t, *pos, self.syntax) {
                        // A truncated tail is one malformed record, then
                        // the end: no bytes are skipped silently.
                        Err(reason) => {
                            let raw = t.get(*pos..).unwrap_or("").to_owned();
                            *pos = t.len();
                            return Ok(Next::Record(Err(reason), raw));
                        }
                        Ok(None) => return Ok(Next::End),
                        Ok(Some((top, end))) => {
                            let raw = t.get(*pos..end).unwrap_or("").to_owned();
                            (top, raw, end)
                        }
                    };
                    *pos = next;
                    return Ok(Next::Record(Ok(top), raw));
                }
            }
        }
    }

    /// One record into the builders.
    fn add(&mut self, raw: &str, parsed: &Result<Top, Reason>) -> FormatResult<()> {
        let Ok(top) = parsed else {
            return self.malformed(raw, None);
        };
        let obj = match top {
            Top::Object(o) => o,
            Top::Other(other) => {
                let cause: KrishivError = if other.trim_start().starts_with('[') {
                    MalformedRecordInParsingCannotParseJsonArraysAsStructs {
                        bad_record: raw.to_owned(),
                        fail_fast_mode: "FAILFAST".to_owned(),
                    }
                    .into()
                } else {
                    InvalidJsonRootField {}.into()
                };
                return self.malformed(raw, Some(cause));
            }
        };
        let mut vals: Vec<Val> = Vec::with_capacity(self.outs.len());
        let conv = super::convert::JsonConverter {
            primitives_as_string: self.o.primitives_as_string,
            dt: &self.dt,
        };
        // Each record field to its data-column slot (later duplicates
        // already won in the parser).
        let mut slots: Vec<Option<&JVal>> = vec![None; self.data_names.len()];
        for (name, v) in &obj.fields {
            if let Some(i) = self.index.get(name) {
                slots[*i] = Some(v);
            }
        }
        let mut bad: Option<Option<KrishivError>> = None;
        for out in &self.outs {
            vals.push(match out {
                Out::Corrupt => Val::Null,
                Out::RowIndex => Val::I64(self.row_no),
                Out::Data { slot, kind } => {
                    match conv.convert(
                        kind,
                        &self.data_names[*slot],
                        &self.data_types[*slot],
                        slots[*slot],
                    ) {
                        Ok(val) => val,
                        Err(Conv::Malformed) => {
                            if bad.is_none() {
                                bad = Some(None);
                            }
                            Val::Null
                        }
                        Err(Conv::Fatal(e)) => {
                            // A conversion that cannot be done at all (a
                            // region time zone) fails the scan even in
                            // PERMISSIVE. Nested values never reach here:
                            // they are `Malformed` above.
                            if e.error_class().name() == "KRISHIV_UNSUPPORTED_FEATURE" {
                                return Err(FormatError::Other(e));
                            }
                            if bad.is_none() {
                                bad = Some(Some(e));
                            }
                            Val::Null
                        }
                    }
                }
            });
        }
        if let Some(cause) = bad {
            return self.malformed_with(raw, cause, vals);
        }
        self.accept(&vals);
        Ok(())
    }

    /// Accepted rows: counted and appended (rows before the split are
    /// counted by the prefix reader and never get here).
    fn accept(&mut self, vals: &[Val]) {
        self.row_no += 1;
        for (col, v) in self.columns.iter_mut().zip(vals) {
            if let Val::Text(s) = v {
                self.held_text += s.len();
            }
            col.append(v);
        }
        self.rows += 1;
    }

    /// A malformed record without a conversion error.
    fn malformed(&mut self, raw: &str, cause: Option<KrishivError>) -> FormatResult<()> {
        let vals = self
            .outs
            .iter()
            .map(|out| match out {
                Out::RowIndex => Val::I64(self.row_no),
                _ => Val::Null,
            })
            .collect();
        self.malformed_with(raw, cause, vals)
    }

    /// A malformed record with its (mostly null) values.
    fn malformed_with(
        &mut self,
        raw: &str,
        cause: Option<KrishivError>,
        mut vals: Vec<Val>,
    ) -> FormatResult<()> {
        match self.o.mode {
            // A dropped record still holds its place in the file's row
            // numbering (the prefix count above counts every record).
            Mode::DropMalformed => {
                self.row_no += 1;
                Ok(())
            }
            Mode::FailFast => Err(fail_fast(raw, cause)),
            Mode::Permissive => {
                for (out, v) in self.outs.iter().zip(&mut vals) {
                    if matches!(out, Out::Corrupt) {
                        *v = Val::Text(raw.to_owned());
                    }
                }
                self.accept(&vals);
                Ok(())
            }
        }
    }

    fn batch(&mut self) -> FormatResult<RecordBatch> {
        let arrays: Vec<ArrayRef> = self.columns.iter_mut().map(Column::finish).collect();
        let rows = std::mem::take(&mut self.rows);
        self.held_text = 0;
        self.metrics.rows += rows as u64;
        RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            arrays,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .map_err(|e| FormatError::Corrupt(e.to_string()))
    }
}

impl Decoder for JsonDecoder {
    fn step(&mut self) -> FormatResult<Step> {
        if self.done {
            return Ok(Step::Done);
        }
        loop {
            match self.next()? {
                Next::NeedData(r) => return Ok(Step::NeedData(vec![r])),
                Next::Record(parsed, raw) => {
                    self.add(&raw, &parsed)?;
                    if self.rows >= BATCH_ROWS {
                        return Ok(Step::Batch(self.batch()?));
                    }
                }
                Next::End => {
                    self.done = true;
                    return if self.rows > 0 {
                        Ok(Step::Batch(self.batch()?))
                    } else {
                        Ok(Step::Done)
                    };
                }
            }
        }
    }

    fn push(&mut self, data: Vec<(Range<u64>, Bytes)>) -> FormatResult<()> {
        for (range, bytes) in data {
            self.metrics.bytes_pushed += bytes.len() as u64;
            match (&mut self.prefix, &mut self.source) {
                (Some(p), _) => p.push(&range, &bytes)?,
                (None, Source::Lines(r)) => r.push(&range, &bytes)?,
                (None, Source::Whole { reader, .. }) => reader.push(&range, &bytes)?,
            }
        }
        Ok(())
    }

    fn upcoming(&self, max_bytes: u64) -> Vec<Range<u64>> {
        match (&self.prefix, &self.source) {
            (Some(p), _) => p.upcoming(max_bytes),
            (None, Source::Lines(r)) => r.upcoming(max_bytes),
            (None, Source::Whole { reader, .. }) => reader.upcoming(max_bytes),
        }
    }

    fn buffered_bytes(&self) -> usize {
        let source = match &self.source {
            Source::Lines(r) => r.buffered_bytes(),
            Source::Whole { reader, text, .. } => {
                reader.buffered_bytes() + text.as_ref().map_or(0, String::capacity)
            }
        };
        source
            + self.prefix.as_ref().map_or(0, LineReader::buffered_bytes)
            + self.held_text
            + self.rows * 8 * self.columns.len()
    }

    fn metrics(&self) -> ScanUnitMetrics {
        self.metrics
    }
}
