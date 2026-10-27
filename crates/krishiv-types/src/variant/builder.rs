// SPDX-License-Identifier: Apache-2.0
//! JSON-to-Variant builder emitting Spark 4.2.0 `parse_json` bytes.
//!
//! The parser is a hand-written single-pass tokenizer (no new dependencies;
//! number lexemes are kept exact) driving an explicit frame stack, so
//! arbitrarily deep documents cannot overflow the call stack. Nesting past
//! [`MAX_NESTING_DEPTH`] levels fails like Spark 4.2.0, which caps nesting at
//! 1000. Values are appended to one buffer; closing a container splices its
//! header, id and offset tables in front of its data. Memory stays
//! proportional to the input, and the value buffer is capped at
//! [`VARIANT_SIZE_LIMIT`].

use std::collections::{HashMap, HashSet};

use krishiv_common::error::{
    KrishivError, MalformedRecordInParsingWithoutSuggestion, VariantSizeLimit,
};

use super::VARIANT_SIZE_LIMIT;
use crate::error::Result;

/// Maximum nesting depth of arrays and objects, matching Spark 4.2.0's
/// `parse_json` (depth 1000 succeeds, depth 1001 fails).
const MAX_NESTING_DEPTH: usize = 1000;

/// Builds the builder's malformed-input error:
/// `MALFORMED_RECORD_IN_PARSING.WITHOUT_SUGGESTION` with Spark's parameters
/// (`badRecord` is the whole input, `failFastMode` is `FAILFAST`).
fn malformed(text: &str) -> KrishivError {
    MalformedRecordInParsingWithoutSuggestion {
        bad_record: text.to_owned(),
        fail_fast_mode: "FAILFAST".to_owned(),
    }
    .into()
}

/// Builds the size-limit error: `VARIANT_SIZE_LIMIT` with Spark's parameters
/// (the limit rendered as `128.0 MiB`, the function as `` `parse_json` ``).
fn size_limit() -> KrishivError {
    VariantSizeLimit {
        size_limit: bytes_to_string(VARIANT_SIZE_LIMIT),
        function_name: "`parse_json`".to_owned(),
    }
    .into()
}

/// Renders a byte count the way Spark's `Utils.bytesToString` does
/// (`"128.0 MiB"` for the size-limit constant); other magnitudes use the
/// same one-decimal units and are truncated, not rounded.
fn bytes_to_string(size: usize) -> String {
    const UNITS: [(u64, &str); 7] = [
        (1 << 60, "EiB"),
        (1 << 50, "PiB"),
        (1 << 40, "TiB"),
        (1 << 30, "GiB"),
        (1 << 20, "MiB"),
        (1 << 10, "KiB"),
        (1, "B"),
    ];
    let size = u64::try_from(size).unwrap_or(u64::MAX);
    let (unit, name) = UNITS
        .into_iter()
        .find(|(unit, _)| size >= *unit)
        .unwrap_or((1, "B"));
    format!("{}.{} {}", size / unit, (size % unit) * 10 / unit, name)
}

/// Smallest offset width in bytes (1-4) holding `max` (a `u32` range).
fn width_for_u32(max: u32) -> usize {
    if max <= 0xFF {
        1
    } else if max <= 0xFFFF {
        2
    } else if max <= 0xFF_FFFF {
        3
    } else {
        4
    }
}

/// Smallest offset width in bytes (1-4) holding `max`, or the size-limit
/// error when the dictionary exceeds 4 GiB.
fn width_for_usize(max: usize) -> Result<usize, KrishivError> {
    if max <= 0xFF {
        Ok(1)
    } else if max <= 0xFFFF {
        Ok(2)
    } else if max <= 0xFF_FFFF {
        Ok(3)
    } else if max <= 0xFFFF_FFFF {
        Ok(4)
    } else {
        Err(size_limit())
    }
}

/// JSON whitespace: space, tab, LF, CR.
fn is_ws(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
}

/// ASCII letters, digits, underscore and dollar sign: characters that continue
/// a JSON literal token (`nullx` is malformed, `null,` is a literal plus
/// junk). This is exactly Java's `isJavaIdentifierPart` over ASCII, which is
/// what Jackson checks after a literal.
fn is_literal_continue(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'$'
}

/// Non-ASCII characters Java treats as identifier parts for the literal
/// terminator rule (`Character.isJavaIdentifierPart` approximated as
/// alphanumeric plus the connector-punctuation and format characters below;
/// currency symbols and other number forms fall back to "terminates", and
/// U+2028/U+2029 terminate as in Java).
fn is_non_ascii_literal_continue(c: char) -> bool {
    const EXTRA: &[char] = &[
        '\u{00AD}', '\u{200B}', '\u{200C}', '\u{200D}', '\u{200E}', '\u{200F}', '\u{203F}',
        '\u{2040}', '\u{2054}', '\u{FE33}', '\u{FE34}', '\u{FE4D}', '\u{FE4E}', '\u{FE4F}',
        '\u{FEFF}', '\u{FF3F}',
    ];
    c.is_alphanumeric() || EXTRA.contains(&c)
}

/// Builds Variant `(metadata, value)` bytes from one JSON document.
///
/// Byte-identical to Spark 4.2.0 `parse_json`: keys in first-seen order in
/// the dictionary (duplicates in one object are malformed), object fields
/// ordered by key bytes with values in input order, the smallest widths that
/// fit, short strings below 64 bytes, and Spark's integer/decimal/double
/// number rules (see the module docs).
#[derive(Debug, Clone, Copy, Default)]
pub struct VariantBuilder;

impl VariantBuilder {
    /// Parses `text` into `(metadata, value)` bytes.
    ///
    /// Leading and trailing whitespace is allowed; text after the first
    /// complete value is ignored, except that a number must end at
    /// whitespace or end of input and a literal must not be followed by an
    /// identifier character, as Spark's tokenizer requires.
    ///
    /// # Errors
    ///
    /// `MALFORMED_RECORD_IN_PARSING.WITHOUT_SUGGESTION` for malformed input
    /// (including nesting past 1000 levels),
    /// `VARIANT_SIZE_LIMIT` when the value would exceed 128 MiB.
    pub fn from_json(text: &str) -> Result<(Vec<u8>, Vec<u8>), KrishivError> {
        let mut parser = Parser {
            text,
            bytes: text.as_bytes(),
            pos: 0,
            dict: Vec::new(),
            dict_index: HashMap::new(),
            value: Vec::new(),
            stack: Vec::new(),
        };
        parser.run_root()?;
        parser.finish()
    }
}

/// How a root scalar constrains the bytes that may follow it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootClass {
    /// A number must end at whitespace or end of input.
    Number,
    /// A literal must not be followed by an identifier character.
    Literal,
    /// Strings and containers are self-delimiting; the rest is ignored.
    Compound,
}

/// One pending object field: key bytes, dictionary id and value start.
struct Field {
    /// Decoded key bytes.
    key: Vec<u8>,
    /// Dictionary id.
    id: u32,
    /// Value start relative to the object's data region.
    start: u32,
}

/// Array parsing state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArrayState {
    /// Just opened: a value or `]` follows.
    Open,
    /// After a comma: a value follows (`]` is a trailing comma).
    AfterComma,
    /// After a value: `,` or `]` follows.
    AfterValue,
}

/// Object parsing state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjectState {
    /// Just opened: a key or `}` follows.
    Open,
    /// After a comma: a key follows (`}` is a trailing comma).
    AfterComma,
    /// After a value: `,` or `}` follows.
    AfterValue,
}

/// One open container frame.
enum Frame {
    /// An open array: header position, element starts, state.
    Array {
        /// Value-buffer position of the placeholder header byte.
        hdr: usize,
        /// Element starts relative to the data region.
        starts: Vec<u32>,
        /// What the frame expects next.
        state: ArrayState,
    },
    /// An open object: header position, finished fields, live key ids, state.
    Object {
        /// Value-buffer position of the placeholder header byte.
        hdr: usize,
        /// Finished fields (key, id, value start).
        fields: Vec<Field>,
        /// Dictionary ids already present in this object (duplicate check).
        seen: HashSet<u32>,
        /// What the frame expects next.
        state: ObjectState,
    },
}

/// The single-pass parser: a cursor over the input plus the dictionary, the
/// value buffer and the explicit container stack.
struct Parser<'t> {
    /// Whole input (for `badRecord`).
    text: &'t str,
    /// Input bytes.
    bytes: &'t [u8],
    /// Cursor into [`Self::bytes`].
    pos: usize,
    /// Dictionary keys in first-seen order.
    dict: Vec<Vec<u8>>,
    /// Dictionary key to id.
    dict_index: HashMap<Vec<u8>, u32>,
    /// Value buffer under construction.
    value: Vec<u8>,
    /// Open containers, innermost last.
    stack: Vec<Frame>,
}

impl<'t> Parser<'t> {
    /// Next input byte, if any.
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    /// Advances past one byte known to be present.
    fn bump(&mut self) -> Result<(), KrishivError> {
        self.pos = self
            .pos
            .checked_add(1)
            .ok_or_else(|| malformed(self.text))?;
        Ok(())
    }

    /// Skips JSON whitespace.
    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b) if is_ws(b)) {
            // `peek` returned `Some`, so `pos < len` and `pos + 1` is safe.
            self.pos += 1;
        }
    }

    /// Ensures the value buffer can grow by `n` bytes without passing the
    /// Spark variant size limit.
    fn grow(&self, n: usize) -> Result<(), KrishivError> {
        let end = self.value.len().checked_add(n).ok_or_else(size_limit)?;
        if end > VARIANT_SIZE_LIMIT {
            return Err(size_limit());
        }
        Ok(())
    }

    /// Appends raw value bytes after a limit check.
    fn emit(&mut self, bytes: &[u8]) -> Result<(), KrishivError> {
        self.grow(bytes.len())?;
        self.value.extend_from_slice(bytes);
        Ok(())
    }

    /// Parses the whole document, including the root trailing-text rule.
    fn run_root(&mut self) -> Result<(), KrishivError> {
        self.skip_ws();
        let class = match self.peek() {
            Some(b'{') => {
                self.bump()?;
                self.open_frame(true)?;
                self.run_stack()?;
                RootClass::Compound
            }
            Some(b'[') => {
                self.bump()?;
                self.open_frame(false)?;
                self.run_stack()?;
                RootClass::Compound
            }
            Some(_) => self.parse_scalar()?,
            None => return Err(malformed(self.text)),
        };
        match class {
            RootClass::Number => match self.peek() {
                None => Ok(()),
                Some(b) if is_ws(b) => Ok(()),
                Some(_) => Err(malformed(self.text)),
            },
            RootClass::Literal => {
                if self.literal_continues()? {
                    return Err(malformed(self.text));
                }
                Ok(())
            }
            RootClass::Compound => Ok(()),
        }
    }

    /// Drives open containers to completion; returns when the root container
    /// closes (the stack is empty).
    fn run_stack(&mut self) -> Result<(), KrishivError> {
        loop {
            if self.stack.is_empty() {
                return Ok(());
            }
            // `len() - 1` is safe: the stack is non-empty.
            let top = self.stack.len() - 1;
            match &self.stack[top] {
                Frame::Array { state, .. } => match state {
                    ArrayState::Open => self.parse_array_open(true)?,
                    ArrayState::AfterComma => self.parse_array_open(false)?,
                    ArrayState::AfterValue => self.parse_array_sep()?,
                },
                Frame::Object { state, .. } => match state {
                    ObjectState::Open => self.parse_object_key(true)?,
                    ObjectState::AfterComma => self.parse_object_key(false)?,
                    ObjectState::AfterValue => self.parse_object_sep()?,
                },
            }
        }
    }

    /// Whether the byte at the cursor continues a literal token.
    fn literal_continues(&self) -> Result<bool, KrishivError> {
        match self.peek() {
            None => Ok(false),
            Some(b) if b < 0x80 => Ok(is_literal_continue(b)),
            Some(_) => {
                // The cursor only ever rests on ASCII structural bytes or
                // just past a complete token, so it is always on a char
                // boundary here.
                let rest = self
                    .text
                    .get(self.pos..)
                    .ok_or_else(|| malformed(self.text))?;
                let c = rest.chars().next().ok_or_else(|| malformed(self.text))?;
                Ok(is_non_ascii_literal_continue(c))
            }
        }
    }

    /// Opens a container frame, emitting a one-byte placeholder header that
    /// [`Self::close_frame`] replaces with the full tables. Offsets recorded
    /// while the frame is open stay valid because every splice happens at or
    /// after the positions recorded so far. Nesting past
    /// [`MAX_NESTING_DEPTH`] levels is malformed, matching Spark 4.2.0.
    fn open_frame(&mut self, is_object: bool) -> Result<(), KrishivError> {
        if self.stack.len() >= MAX_NESTING_DEPTH {
            return Err(malformed(self.text));
        }
        self.grow(1)?;
        let hdr = self.value.len();
        self.value.push(0x00);
        if is_object {
            self.stack.push(Frame::Object {
                hdr,
                fields: Vec::new(),
                seen: HashSet::new(),
                state: ObjectState::Open,
            });
        } else {
            self.stack.push(Frame::Array {
                hdr,
                starts: Vec::new(),
                state: ArrayState::Open,
            });
        }
        Ok(())
    }

    /// Data-region start of frame `hdr` (just past its placeholder).
    fn data_start(hdr: usize, text: &str) -> Result<usize, KrishivError> {
        hdr.checked_add(1).ok_or_else(|| malformed(text))
    }

    /// Current offset relative to frame `hdr`'s data region, as `u32`.
    fn relative_pos(&self, hdr: usize) -> Result<u32, KrishivError> {
        let data = Self::data_start(hdr, self.text)?;
        let len = self
            .value
            .len()
            .checked_sub(data)
            .ok_or_else(|| malformed(self.text))?;
        u32::try_from(len).map_err(|_| size_limit())
    }

    /// Dictionary id for `key`, appending first-seen keys.
    fn dict_id(&mut self, key: &[u8]) -> Result<u32, KrishivError> {
        if let Some(&id) = self.dict_index.get(key) {
            return Ok(id);
        }
        let id = u32::try_from(self.dict.len()).map_err(|_| size_limit())?;
        self.dict.push(key.to_vec());
        self.dict_index.insert(key.to_vec(), id);
        Ok(id)
    }

    /// Array in `Open` (empty allowed) or `AfterComma` state: a value or the
    /// closing bracket follows.
    fn parse_array_open(&mut self, allow_close: bool) -> Result<(), KrishivError> {
        self.skip_ws();
        // Copy the header position out; the frame borrow must end before
        // parsing the element value.
        let hdr = match self.stack.last() {
            Some(Frame::Array { hdr, .. }) => *hdr,
            Some(_) | None => return Err(malformed(self.text)),
        };
        if allow_close && self.peek() == Some(b']') {
            self.bump()?;
            self.close_frame()?;
            return Ok(());
        }
        let start = self.relative_pos(hdr)?;
        // The parent frame index: `parse_value_or_open` may push a nested
        // frame, so the parent must be updated by index, not `last_mut`.
        let parent = self
            .stack
            .len()
            .checked_sub(1)
            .ok_or_else(|| malformed(self.text))?;
        self.parse_value_or_open()?;
        match self.stack.get_mut(parent) {
            Some(Frame::Array { starts, state, .. }) => {
                starts.push(start);
                *state = ArrayState::AfterValue;
                Ok(())
            }
            Some(_) | None => Err(malformed(self.text)),
        }
    }

    /// Array in `AfterValue` state: `,` continues, `]` closes.
    fn parse_array_sep(&mut self) -> Result<(), KrishivError> {
        self.skip_ws();
        match self.peek() {
            Some(b',') => {
                self.bump()?;
                match self.stack.last_mut() {
                    Some(Frame::Array { state, .. }) => {
                        *state = ArrayState::AfterComma;
                        Ok(())
                    }
                    Some(_) | None => Err(malformed(self.text)),
                }
            }
            Some(b']') => {
                self.bump()?;
                self.close_frame()
            }
            Some(_) | None => Err(malformed(self.text)),
        }
    }

    /// Object in `Open` (empty allowed) or `AfterComma` state: a key or the
    /// closing brace follows; the key, colon and value are parsed as one
    /// step, so no pending state is needed.
    fn parse_object_key(&mut self, allow_close: bool) -> Result<(), KrishivError> {
        self.skip_ws();
        let hdr = match self.stack.last() {
            Some(Frame::Object { hdr, .. }) => *hdr,
            Some(_) | None => return Err(malformed(self.text)),
        };
        if allow_close && self.peek() == Some(b'}') {
            self.bump()?;
            self.close_frame()?;
            return Ok(());
        }
        if self.peek() != Some(b'"') {
            return Err(malformed(self.text));
        }
        let key = self.parse_string()?;
        self.skip_ws();
        if self.peek() != Some(b':') {
            return Err(malformed(self.text));
        }
        self.bump()?;
        let id = self.dict_id(&key)?;
        let start = self.relative_pos(hdr)?;
        // The parent frame index: `parse_value_or_open` may push a nested
        // frame, so the parent must be updated by index, not `last_mut`.
        let parent = self
            .stack
            .len()
            .checked_sub(1)
            .ok_or_else(|| malformed(self.text))?;
        self.parse_value_or_open()?;
        match self.stack.get_mut(parent) {
            Some(Frame::Object {
                fields,
                seen,
                state,
                ..
            }) => {
                if !seen.insert(id) {
                    return Err(malformed(self.text));
                }
                fields.push(Field { key, id, start });
                *state = ObjectState::AfterValue;
                Ok(())
            }
            Some(_) | None => Err(malformed(self.text)),
        }
    }

    /// Object in `AfterValue` state: `,` continues, `}` closes.
    fn parse_object_sep(&mut self) -> Result<(), KrishivError> {
        self.skip_ws();
        match self.peek() {
            Some(b',') => {
                self.bump()?;
                match self.stack.last_mut() {
                    Some(Frame::Object { state, .. }) => {
                        *state = ObjectState::AfterComma;
                        Ok(())
                    }
                    Some(_) | None => Err(malformed(self.text)),
                }
            }
            Some(b'}') => {
                self.bump()?;
                self.close_frame()
            }
            Some(_) | None => Err(malformed(self.text)),
        }
    }

    /// Parses one scalar value (emitting it) or opens one container at the
    /// cursor. Returns the root-text class for the trailing-text rule;
    /// nested callers ignore it.
    fn parse_value_or_open(&mut self) -> Result<RootClass, KrishivError> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => {
                self.bump()?;
                self.open_frame(true)?;
                Ok(RootClass::Compound)
            }
            Some(b'[') => {
                self.bump()?;
                self.open_frame(false)?;
                Ok(RootClass::Compound)
            }
            Some(_) => self.parse_scalar(),
            None => Err(malformed(self.text)),
        }
    }

    /// Parses one scalar (string, number or literal) and emits it.
    fn parse_scalar(&mut self) -> Result<RootClass, KrishivError> {
        match self.peek() {
            Some(b'"') => {
                let string = self.parse_string()?;
                self.emit_string(&string)?;
                Ok(RootClass::Compound)
            }
            Some(b't') => {
                self.parse_literal("true")?;
                self.emit(&[0x04])?;
                Ok(RootClass::Literal)
            }
            Some(b'f') => {
                self.parse_literal("false")?;
                self.emit(&[0x08])?;
                Ok(RootClass::Literal)
            }
            Some(b'n') => {
                self.parse_literal("null")?;
                self.emit(&[0x00])?;
                Ok(RootClass::Literal)
            }
            Some(b'-' | b'0'..=b'9') => {
                self.parse_number()?;
                Ok(RootClass::Number)
            }
            Some(_) | None => Err(malformed(self.text)),
        }
    }

    /// Matches `expected` exactly, then rejects identifier-continuation bytes
    /// (Jackson's tokenizer rule, applied at every level).
    fn parse_literal(&mut self, expected: &str) -> Result<(), KrishivError> {
        let end = self
            .pos
            .checked_add(expected.len())
            .ok_or_else(|| malformed(self.text))?;
        if self.bytes.get(self.pos..end) != Some(expected.as_bytes()) {
            return Err(malformed(self.text));
        }
        self.pos = end;
        if self.literal_continues()? {
            return Err(malformed(self.text));
        }
        Ok(())
    }

    /// Parses a JSON string, returning the decoded UTF-8 bytes. Rejects raw
    /// control characters, truncated input, bad escapes and lone surrogates,
    /// mirroring Jackson; `/` and `\"` style escapes are accepted.
    fn parse_string(&mut self) -> Result<Vec<u8>, KrishivError> {
        // Caller guarantees the opening quote.
        self.bump()?;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                None => return Err(malformed(self.text)),
                Some(b'"') => {
                    self.bump()?;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.bump()?;
                    self.parse_escape(&mut out)?;
                }
                Some(b) if b < 0x20 => return Err(malformed(self.text)),
                Some(b) => {
                    out.push(b);
                    self.bump()?;
                }
            }
        }
    }

    /// Parses one escape (after the backslash) into `out`.
    fn parse_escape(&mut self, out: &mut Vec<u8>) -> Result<(), KrishivError> {
        let escaped = match self.peek() {
            Some(b'"') => b'"',
            Some(b'\\') => b'\\',
            Some(b'/') => b'/',
            Some(b'b') => 0x08,
            Some(b'f') => 0x0C,
            Some(b'n') => b'\n',
            Some(b'r') => b'\r',
            Some(b't') => b'\t',
            Some(b'u') => {
                self.bump()?;
                let unit = self.parse_hex4()?;
                // Surrogate pairs: a high surrogate must be followed by
                // `\u` and a low surrogate; lone halves are malformed.
                if (0xD800..0xDC00).contains(&unit) {
                    if self.peek() != Some(b'\\') {
                        return Err(malformed(self.text));
                    }
                    self.bump()?;
                    if self.peek() != Some(b'u') {
                        return Err(malformed(self.text));
                    }
                    self.bump()?;
                    let low = self.parse_hex4()?;
                    if !(0xDC00..0xE000).contains(&low) {
                        return Err(malformed(self.text));
                    }
                    let scalar = 0x1_0000 + ((unit - 0xD800) << 10) + (low - 0xDC00);
                    let c = char::from_u32(scalar).ok_or_else(|| malformed(self.text))?;
                    let mut encoded = [0_u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut encoded).as_bytes());
                    return Ok(());
                }
                if (0xDC00..0xE000).contains(&unit) {
                    return Err(malformed(self.text));
                }
                let c = char::from_u32(unit).ok_or_else(|| malformed(self.text))?;
                let mut encoded = [0_u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut encoded).as_bytes());
                return Ok(());
            }
            Some(_) | None => return Err(malformed(self.text)),
        };
        self.bump()?;
        out.push(escaped);
        Ok(())
    }

    /// Parses exactly four ASCII hex digits into a `u32`.
    fn parse_hex4(&mut self) -> Result<u32, KrishivError> {
        let mut value: u32 = 0;
        for _ in 0..4 {
            let digit = self
                .peek()
                .and_then(hex_value)
                .ok_or_else(|| malformed(self.text))?;
            value = value * 16 + digit;
            self.bump()?;
        }
        Ok(value)
    }
}

/// Numeric value of one ASCII hex digit.
fn hex_value(byte: u8) -> Option<u32> {
    match byte {
        b'0'..=b'9' => Some(u32::from(byte - b'0')),
        b'a'..=b'f' => Some(u32::from(byte - b'a') + 10),
        b'A'..=b'F' => Some(u32::from(byte - b'A') + 10),
        _ => None,
    }
}

impl<'t> Parser<'t> {
    /// Parses one JSON number at the cursor and emits its Variant encoding.
    ///
    /// Shape errors (leading zeros, missing digits, lone signs) are
    /// malformed; the value rules follow Spark: pure integers take the
    /// smallest int8/16/32/64 that fits, then decimal (scale 0) up to 38
    /// digits, then double; fractions take decimal when `precision <= 38`
    /// and `scale <= 38` (width by `max(precision, scale)`), else double;
    /// any exponent means double, with overflow to infinity.
    fn parse_number(&mut self) -> Result<(), KrishivError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.bump()?;
        }
        // Integer part: `0` alone, or a non-zero digit plus digits.
        match self.peek() {
            Some(b'0') => {
                self.bump()?;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(malformed(self.text));
                }
            }
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.bump()?;
                }
            }
            Some(_) | None => return Err(malformed(self.text)),
        }
        let mut has_fraction = false;
        if self.peek() == Some(b'.') {
            has_fraction = true;
            self.bump()?;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(malformed(self.text));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.bump()?;
            }
        }
        let mut has_exponent = false;
        if matches!(self.peek(), Some(b'e' | b'E')) {
            has_exponent = true;
            self.bump()?;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.bump()?;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(malformed(self.text));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.bump()?;
            }
        }
        let lexeme = self
            .bytes
            .get(start..self.pos)
            .ok_or_else(|| malformed(self.text))?;
        if has_exponent {
            self.emit_double_lexeme(lexeme)
        } else if has_fraction {
            self.emit_decimal_lexeme(lexeme)
        } else {
            self.emit_integer_lexeme(lexeme)
        }
    }

    /// Emits a pure-integer lexeme: smallest fitting int, else decimal with
    /// scale 0 (up to 38 digits), else double.
    fn emit_integer_lexeme(&mut self, lexeme: &[u8]) -> Result<(), KrishivError> {
        let negative = lexeme.first() == Some(&b'-');
        let digits = if negative { &lexeme[1..] } else { lexeme };
        // Magnitude in `u64` (any `i64`-fitting value fits); overflow falls
        // through to the decimal/double path.
        let mut magnitude: u64 = 0;
        let mut fits_u64 = true;
        for &digit in digits {
            let digit = u64::from(digit - b'0');
            match magnitude.checked_mul(10).and_then(|m| m.checked_add(digit)) {
                Some(next) => magnitude = next,
                None => {
                    fits_u64 = false;
                    break;
                }
            }
        }
        let as_i64 = if fits_u64 {
            if negative {
                if magnitude == (1_u64 << 63) {
                    Some(i64::MIN)
                } else {
                    i64::try_from(magnitude).ok().and_then(i64::checked_neg)
                }
            } else {
                i64::try_from(magnitude).ok()
            }
        } else {
            None
        };
        if let Some(value) = as_i64 {
            return self.emit_int(value);
        }
        if digits.len() <= 38 {
            let mut unscaled: i128 = 0;
            for &digit in digits {
                let digit = i128::from(digit - b'0');
                match unscaled.checked_mul(10).and_then(|u| u.checked_add(digit)) {
                    Some(next) => unscaled = next,
                    None => return self.emit_double_lexeme(lexeme),
                }
            }
            let unscaled = if negative {
                match unscaled.checked_neg() {
                    Some(negated) => negated,
                    None => return self.emit_double_lexeme(lexeme),
                }
            } else {
                unscaled
            };
            // Scale 0: width by precision (the digit count).
            let width = decimal_width(digits.len(), 0);
            return self.emit_decimal(unscaled, 0, width);
        }
        self.emit_double_lexeme(lexeme)
    }

    /// Emits a fraction lexeme (no exponent): decimal when `precision <= 38`
    /// and `scale <= 38`, else double.
    fn emit_decimal_lexeme(&mut self, lexeme: &[u8]) -> Result<(), KrishivError> {
        let negative = lexeme.first() == Some(&b'-');
        let digits = if negative { &lexeme[1..] } else { lexeme };
        let dot = digits
            .iter()
            .position(|&b| b == b'.')
            .ok_or_else(|| malformed(self.text))?;
        let (int_digits, frac_with_dot) = digits.split_at(dot);
        let frac_digits = &frac_with_dot[1..];
        // Significant digits: leading zeros don't count; all-zero counts 1.
        let mut significant = 0_usize;
        let mut seen_nonzero = false;
        for &digit in int_digits.iter().chain(frac_digits.iter()) {
            if digit != b'0' {
                seen_nonzero = true;
            }
            if seen_nonzero {
                significant += 1;
            }
        }
        if !seen_nonzero {
            significant = 1;
        }
        let scale = frac_digits.len();
        if significant > 38 || scale > 38 {
            return self.emit_double_lexeme(lexeme);
        }
        let mut unscaled: i128 = 0;
        for &digit in int_digits.iter().chain(frac_digits.iter()) {
            let digit = i128::from(digit - b'0');
            match unscaled.checked_mul(10).and_then(|u| u.checked_add(digit)) {
                Some(next) => unscaled = next,
                None => return self.emit_double_lexeme(lexeme),
            }
        }
        let unscaled = if negative {
            match unscaled.checked_neg() {
                Some(negated) => negated,
                None => return self.emit_double_lexeme(lexeme),
            }
        } else {
            unscaled
        };
        let scale_u8 = u8::try_from(scale).map_err(|_| malformed(self.text))?;
        let width = decimal_width(significant, scale);
        self.emit_decimal(unscaled, scale_u8, width)
    }

    /// Emits any exponent lexeme as a double; overflow yields infinity and
    /// underflow yields zero, exactly like Java's `Double.parseDouble`.
    fn emit_double_lexeme(&mut self, lexeme: &[u8]) -> Result<(), KrishivError> {
        let text = core::str::from_utf8(lexeme).map_err(|_| malformed(self.text))?;
        let value: f64 = text.parse().map_err(|_| malformed(self.text))?;
        self.emit_double(value)
    }

    /// Emits an integer in the smallest width that holds it.
    fn emit_int(&mut self, value: i64) -> Result<(), KrishivError> {
        let bytes = value.to_le_bytes();
        if (-128..=127).contains(&value) {
            self.emit(&[0x0C, bytes[0]])
        } else if (-32_768..=32_767).contains(&value) {
            self.grow(3)?;
            self.value.push(0x10);
            self.value.extend_from_slice(&bytes[..2]);
            Ok(())
        } else if (-2_147_483_648..=2_147_483_647).contains(&value) {
            self.grow(5)?;
            self.value.push(0x14);
            self.value.extend_from_slice(&bytes[..4]);
            Ok(())
        } else {
            self.grow(9)?;
            self.value.push(0x18);
            self.value.extend_from_slice(&bytes[..8]);
            Ok(())
        }
    }

    /// Emits an IEEE 754 binary64.
    fn emit_double(&mut self, value: f64) -> Result<(), KrishivError> {
        self.grow(9)?;
        self.value.push(0x1C);
        self.value.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    /// Emits a decimal with the given unscaled value, scale and byte width
    /// (4, 8 or 16).
    fn emit_decimal(
        &mut self,
        unscaled: i128,
        scale: u8,
        width: usize,
    ) -> Result<(), KrishivError> {
        let header = match width {
            4 => 0x20,
            8 => 0x24,
            _ => 0x28,
        };
        let bytes = unscaled.to_le_bytes();
        let body = bytes.get(..width).ok_or_else(|| malformed(self.text))?;
        let end = width.checked_add(2).ok_or_else(size_limit)?;
        self.grow(end)?;
        self.value.push(header);
        self.value.push(scale);
        self.value.extend_from_slice(body);
        Ok(())
    }

    /// Emits a string: short form below 64 bytes, otherwise primitive id 16
    /// with a 4-byte length.
    fn emit_string(&mut self, string: &[u8]) -> Result<(), KrishivError> {
        if string.len() < 64 {
            let len = u8::try_from(string.len()).map_err(|_| malformed(self.text))?;
            let end = string.len().checked_add(1).ok_or_else(size_limit)?;
            self.grow(end)?;
            self.value.push(len << 2 | 0x01);
            self.value.extend_from_slice(string);
            Ok(())
        } else {
            let len = u32::try_from(string.len()).map_err(|_| size_limit())?;
            let end = string.len().checked_add(5).ok_or_else(size_limit)?;
            self.grow(end)?;
            self.value.push(0x40);
            self.value.extend_from_slice(&len.to_le_bytes());
            self.value.extend_from_slice(string);
            Ok(())
        }
    }

    /// Closes the innermost open container, splicing its header, id and
    /// offset tables over its placeholder byte.
    fn close_frame(&mut self) -> Result<(), KrishivError> {
        match self.stack.pop() {
            Some(Frame::Array { hdr, starts, .. }) => self.finish_array(hdr, &starts),
            Some(Frame::Object { hdr, fields, .. }) => self.finish_object(hdr, fields),
            None => Err(malformed(self.text)),
        }
    }

    /// Assembles an array prefix and splices it over the placeholder.
    fn finish_array(&mut self, hdr: usize, starts: &[u32]) -> Result<(), KrishivError> {
        let large = starts.len() > 255;
        let data = Self::data_start(hdr, self.text)?;
        let total = self
            .value
            .len()
            .checked_sub(data)
            .ok_or_else(|| malformed(self.text))?;
        let total_u32 = u32::try_from(total).map_err(|_| size_limit())?;
        let off_size = width_for_u32(total_u32);
        let header = 0x03 | width_bits(off_size, 2, self.text)? | large_bit(large, 4);
        let count = starts.len();
        let mut prefix = Vec::new();
        prefix.push(header);
        if large {
            let count_u32 = u32::try_from(count).map_err(|_| size_limit())?;
            prefix.extend_from_slice(&count_u32.to_le_bytes());
        } else {
            prefix.push(u8::try_from(count).map_err(|_| size_limit())?);
        }
        for start in starts {
            prefix.extend_from_slice(&start.to_le_bytes()[..off_size]);
        }
        let total_bytes = total_u32.to_le_bytes();
        prefix.extend_from_slice(&total_bytes[..off_size]);
        self.splice_prefix(hdr, &prefix)
    }

    /// Assembles an object prefix (fields sorted by key bytes, values already
    /// in input order) and splices it over the placeholder.
    fn finish_object(&mut self, hdr: usize, mut fields: Vec<Field>) -> Result<(), KrishivError> {
        fields.sort_by(|a, b| a.key.cmp(&b.key));
        let large = fields.len() > 255;
        let data = Self::data_start(hdr, self.text)?;
        let total = self
            .value
            .len()
            .checked_sub(data)
            .ok_or_else(|| malformed(self.text))?;
        let total_u32 = u32::try_from(total).map_err(|_| size_limit())?;
        let off_size = width_for_u32(total_u32);
        let max_id = fields.iter().map(|f| f.id).max().unwrap_or(0);
        let id_size = width_for_u32(max_id);
        let header = 0x02
            | width_bits(off_size, 2, self.text)?
            | width_bits(id_size, 4, self.text)?
            | large_bit(large, 6);
        let count = fields.len();
        let mut prefix = Vec::new();
        prefix.push(header);
        if large {
            let count_u32 = u32::try_from(count).map_err(|_| size_limit())?;
            prefix.extend_from_slice(&count_u32.to_le_bytes());
        } else {
            prefix.push(u8::try_from(count).map_err(|_| size_limit())?);
        }
        for field in &fields {
            prefix.extend_from_slice(&field.id.to_le_bytes()[..id_size]);
        }
        for field in &fields {
            prefix.extend_from_slice(&field.start.to_le_bytes()[..off_size]);
        }
        let total_bytes = total_u32.to_le_bytes();
        prefix.extend_from_slice(&total_bytes[..off_size]);
        self.splice_prefix(hdr, &prefix)
    }

    /// Replaces the one-byte placeholder at `hdr` with `prefix`, after a
    /// size-limit check on the net growth.
    fn splice_prefix(&mut self, hdr: usize, prefix: &[u8]) -> Result<(), KrishivError> {
        let net = prefix
            .len()
            .checked_sub(1)
            .ok_or_else(|| malformed(self.text))?;
        self.grow(net)?;
        let end = hdr.checked_add(1).ok_or_else(|| malformed(self.text))?;
        // `hdr` was recorded as `value.len()` when the frame opened and the
        // buffer only grows, so `hdr..hdr + 1` is in bounds.
        self.value.splice(hdr..end, prefix.iter().copied());
        Ok(())
    }

    /// Assembles the metadata dictionary and returns `(metadata, value)`.
    /// The metadata buffer is not size-capped, mirroring Spark.
    fn finish(self) -> Result<(Vec<u8>, Vec<u8>), KrishivError> {
        let text = self.text;
        // Decoded keys never exceed the input length (escapes contract), so
        // the sum fits `usize`.
        let total_keys: usize = self.dict.iter().map(Vec::len).sum();
        let width = width_for_usize(total_keys)?;
        // Distinct keys are non-empty, so the count fits the offset width.
        let dict_size = u32::try_from(self.dict.len()).map_err(|_| size_limit())?;
        let width_u8 = u8::try_from(width).map_err(|_| malformed(text))?;
        let header = 0x01_u8 | (width_u8.checked_sub(1).ok_or_else(|| malformed(text))? << 6);
        let mut metadata = Vec::new();
        metadata.push(header);
        metadata.extend_from_slice(&dict_size.to_le_bytes()[..width]);
        // Offsets: `offsets[0] = 0`, then the running total after each key.
        let mut offset: u32 = 0;
        metadata.extend_from_slice(&offset.to_le_bytes()[..width]);
        for key in &self.dict {
            let key_len = u32::try_from(key.len()).map_err(|_| size_limit())?;
            offset = offset.checked_add(key_len).ok_or_else(size_limit)?;
            metadata.extend_from_slice(&offset.to_le_bytes()[..width]);
        }
        for key in &self.dict {
            metadata.extend_from_slice(key);
        }
        Ok((metadata, self.value))
    }
}

/// Decimal byte width (4, 8 or 16) from `max(precision, scale)`.
fn decimal_width(precision: usize, scale: usize) -> usize {
    if precision.max(scale) <= 9 {
        4
    } else if precision.max(scale) <= 18 {
        8
    } else {
        16
    }
}

/// `(width - 1)` shifted to bit `at` for a container header byte.
fn width_bits(width: usize, at: u8, text: &str) -> Result<u8, KrishivError> {
    let minus_one = width.checked_sub(1).ok_or_else(|| malformed(text))?;
    let bits = u8::try_from(minus_one).map_err(|_| malformed(text))?;
    Ok(bits << at)
}

/// The `is_large` bit for a container header byte.
fn large_bit(large: bool, at: u8) -> u8 {
    if large { 1 << at } else { 0 }
}
