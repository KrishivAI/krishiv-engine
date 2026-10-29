// SPDX-License-Identifier: Apache-2.0
//! Parser for Spark DDL type strings (`CatalystSqlParser.parseDataType`).
//!
//! Follows Spark's `dataType` grammar for the types in [`DataType`]: a type
//! name, optionally followed by `(` integer `,` ... `)`, plus the multi-word
//! forms `TIMESTAMP WITH LOCAL TIME ZONE`, `TIMESTAMP WITHOUT TIME ZONE` and
//! `TIME[(p)] WITHOUT TIME ZONE`, and `GEOMETRY(ANY)` / `GEOGRAPHY(ANY)`.
//! The lexer follows Spark's: case-insensitive keywords, Unicode whitespace,
//! `--` and nested `/* */` comments, backquoted identifiers and numeric
//! literal tokens, string literals (for field comments), and Spark's
//! multi-character operators (`<>`; `>>` only outside a complex type).
//!
//! Nested types: `ARRAY<t>`, `MAP<k, v>` and `STRUCT<name [:] t [NOT NULL]
//! [COMMENT 'c'], ...>` (also `STRUCT<>`), nested at most
//! [`DataType::MAX_NESTING_DEPTH`] deep. As Spark's ANTLR parser does, a
//! syntax error anywhere inside a complex type is reported at the
//! outermost `<`, a bare `ARRAY` / `MAP` / `STRUCT` is
//! `INCOMPLETE_TYPE_DEFINITION.*`, and a single token after a complete
//! complex type is "extra input".
//!
//! Interval, char/varchar, variant and collated string types are
//! recognized and rejected with `KRISHIV_UNSUPPORTED_FEATURE`.

use krishiv_common::error::ErrorClass;

use crate::error::{self, Result};
use crate::nested::{ArrayType, MapType, Metadata, StructField, StructType};
use crate::{DataType, DecimalType, Srid};

/// Parses a Spark DDL type string such as `int`, `DECIMAL(10, 2)` or
/// `timestamp_ntz`.
///
/// ```
/// use krishiv_types::{parse_data_type, DataType, DecimalType};
///
/// let t = parse_data_type("DECIMAL(10, 2)").unwrap();
/// assert_eq!(t, DataType::Decimal(DecimalType::try_new(10, 2).unwrap()));
/// assert_eq!(t.simple_string(), "decimal(10,2)");
/// assert!(parse_data_type("decimal(39, 0)").is_err());
/// ```
///
/// # Errors
/// `PARSE_SYNTAX_ERROR` / `PARSE_EMPTY_STATEMENT` /
/// `UNCLOSED_BRACKETED_COMMENT` / `INVALID_IDENTIFIER` for a malformed
/// string, the Spark decimal bound errors, `UNSUPPORTED_DATATYPE` for an
/// unknown type and `KRISHIV_UNSUPPORTED_FEATURE` for a Spark type not
/// supported yet (see [`crate::error`]).
pub fn parse_data_type(input: &str) -> Result<DataType> {
    let tokens = Lexer {
        src: input,
        pos: 0,
        complex_level: 0,
    }
    .tokens()?;
    if tokens.is_empty() {
        return Err(error::empty_statement());
    }
    let mut p = Parser {
        tokens,
        pos: 0,
        deferred: None,
    };
    let data_type = p.data_type(0)?;
    let left = p.tokens.len().saturating_sub(p.pos);
    match p.next() {
        // Spark checks names and arguments only once the whole string
        // parses (`AstBuilder` runs after the parser).
        None => p.deferred.map_or(Ok(data_type), Err),
        // ANTLR deletes a single extra token after a complex type.
        Some(tok) if data_type.is_nested() && left == 1 => {
            Err(error::extra_input(&format!("'{}'", tok.raw)))
        }
        Some(tok) => Err(tok.syntax_error()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A word; `quoted` for a backquoted identifier, which never matches a
    /// keyword.
    Word { quoted: bool },
    /// Digits only (Spark's `INTEGER_VALUE`).
    Integer,
    /// A single punctuation character.
    Punct(char),
    /// A multi-character operator Spark lexes as one token (`<>`, `>>`).
    Op,
    /// A string literal (`'...'`, `"..."`, `r'...'`); `text` is its value.
    Str,
    /// Any other token (a numeric literal, an unknown character).
    Other,
}

#[derive(Debug, Clone)]
struct Token<'a> {
    kind: Kind,
    /// The token's text (for a quoted identifier, unescaped and without the
    /// backquotes).
    text: String,
    /// The token as written, for error messages.
    raw: &'a str,
}

impl Token<'_> {
    /// Spark folds keywords with `Character.toUpperCase` per character, so
    /// `ınt` (dotless i) is `INT`.
    fn folded(&self) -> String {
        self.text.chars().map(fold_char).collect()
    }

    fn is_keyword(&self, keyword: &str) -> bool {
        self.kind == (Kind::Word { quoted: false }) && self.folded().eq_ignore_ascii_case(keyword)
    }

    fn syntax_error(&self) -> krishiv_common::error::KrishivError {
        error::syntax_error(&format!("'{}'", self.raw))
    }
}

fn fold_char(c: char) -> char {
    let mut up = c.to_uppercase();
    match (up.next(), up.next()) {
        (Some(u), None) => u,
        _ => c,
    }
}

/// Spark's `WS` token: ASCII whitespace plus the Unicode space separators
/// it lists (U+2028 but not U+2029).
fn is_spark_whitespace(c: char) -> bool {
    c.is_ascii_whitespace()
        || matches!(
            c,
            '\u{0B}' | '\u{A0}' | '\u{1680}' | '\u{2000}'..='\u{200A}'
        )
        || matches!(c, '\u{2028}' | '\u{202F}' | '\u{205F}' | '\u{3000}')
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

struct Lexer<'a> {
    src: &'a str,
    pos: usize,
    /// Spark's `complex_type_level_counter`: raised by each `ARRAY`, `MAP`
    /// and `STRUCT` keyword, lowered by each `>`; while it is above zero,
    /// `>>` is two tokens (closing two levels), not a shift operator.
    complex_level: usize,
}

impl<'a> Lexer<'a> {
    fn rest(&self) -> &'a str {
        self.src.get(self.pos..).unwrap_or_default()
    }

    /// Length in bytes of the longest prefix of `rest()[from..]` whose chars
    /// satisfy `pred`.
    fn span(&self, from: usize, pred: impl Fn(char) -> bool) -> usize {
        let tail = self.rest().get(from..).unwrap_or_default();
        tail.find(|c| !pred(c)).unwrap_or(tail.len())
    }

    fn take(&mut self, len: usize, kind: Kind) -> Token<'a> {
        let raw = self.rest().get(..len).unwrap_or_default();
        self.pos += len;
        Token {
            kind,
            text: raw.to_owned(),
            raw,
        }
    }

    fn tokens(mut self) -> Result<Vec<Token<'a>>> {
        let mut out = Vec::new();
        while self.skip_trivia()? {
            out.push(self.token());
        }
        Ok(out)
    }

    /// Skips whitespace and comments; returns whether a token follows.
    fn skip_trivia(&mut self) -> Result<bool> {
        loop {
            self.pos += self.span(0, is_spark_whitespace);
            let rest = self.rest();
            if rest.starts_with("--") {
                self.pos += rest.find('\n').unwrap_or(rest.len());
            } else if rest.starts_with("/*") {
                // Bracketed comments nest, as in Spark.
                let mut depth = 0_usize;
                let mut i = 0;
                loop {
                    let tail = rest.get(i..).unwrap_or_default();
                    if tail.is_empty() {
                        return Err(error::unclosed_comment());
                    } else if tail.starts_with("/*") {
                        depth += 1;
                        i += 2;
                    } else if tail.starts_with("*/") {
                        depth -= 1;
                        i += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        i += tail.chars().next().map_or(1, char::len_utf8);
                    }
                }
                self.pos += i;
            } else {
                return Ok(!rest.is_empty());
            }
        }
    }

    fn token(&mut self) -> Token<'a> {
        let rest = self.rest();
        let mut chars = rest.chars();
        let c = chars.next().unwrap_or(' ');
        let second = chars.next();
        let starts_number =
            c.is_ascii_digit() || (c == '.' && second.is_some_and(|d| d.is_ascii_digit()));
        if starts_number {
            self.number()
        } else if matches!(c, 'r' | 'R') && matches!(second, Some('\'' | '"')) {
            self.string(1)
        } else if is_word_char(c) {
            let t = self.take(self.span(0, is_word_char), Kind::Word { quoted: false });
            if ["array", "map", "struct"]
                .iter()
                .any(|k| t.text.eq_ignore_ascii_case(k))
            {
                self.complex_level += 1;
            }
            t
        } else if c == '`' {
            self.quoted()
        } else if matches!(c, '\'' | '"') {
            self.string(0)
        } else if let Some(op) = self.operator() {
            self.take(op, Kind::Op)
        } else {
            let kind = if c.is_ascii_punctuation() {
                Kind::Punct(c)
            } else {
                Kind::Other
            };
            if c == '>' {
                self.complex_level = self.complex_level.saturating_sub(1);
            }
            self.take(c.len_utf8(), kind)
        }
    }

    /// The length of a multi-character operator at the cursor, if any:
    /// the ones Spark's lexer has, `>>` and `>>>` only outside a complex
    /// type.
    fn operator(&self) -> Option<usize> {
        let rest = self.rest();
        if self.complex_level == 0 {
            for op in [">>>", ">>"] {
                if rest.starts_with(op) {
                    return Some(op.len());
                }
            }
        }
        [
            "<=>", "<>", "<=", ">=", "!=", "==", "<<", "::", "->", "=>", "||", "|>",
        ]
        .into_iter()
        .find(|op| rest.starts_with(op))
        .map(str::len)
    }

    /// A string literal `prefix` bytes after the cursor (1 for a raw
    /// string's `r`). Unterminated, its quote is a token of its own.
    fn string(&mut self, prefix: usize) -> Token<'a> {
        let rest = self.rest();
        let bytes = rest.as_bytes();
        let quote = bytes.get(prefix).copied().unwrap_or(b'\'');
        let raw = prefix > 0;
        let mut i = prefix + 1;
        loop {
            match bytes.get(i) {
                None => return self.take(prefix + 1, Kind::Other),
                Some(&b) if b == quote && !raw && bytes.get(i + 1) == Some(&quote) => i += 2,
                Some(&b) if b == quote => break,
                Some(b'\\') if !raw => i += 2,
                Some(_) => i += 1,
            }
        }
        let mut t = self.take(i + 1, Kind::Str);
        t.text = unescape_string(t.raw);
        t
    }

    /// Spark's numeric tokens: `INTEGER_VALUE`, decimal and exponent
    /// literals, and suffixed literals (`10L`, `1.5BD`). Digits followed by
    /// other word characters form an identifier (`1a`).
    fn number(&mut self) -> Token<'a> {
        let digit = |c: char| c.is_ascii_digit();
        let rest = self.rest().as_bytes();
        let mut end = self.span(0, digit);
        let mut literal = false;
        if rest.get(end) == Some(&b'.') {
            end += 1 + self.span(end + 1, digit);
            literal = true;
        }
        if matches!(rest.get(end), Some(b'e' | b'E')) {
            let sign = usize::from(matches!(rest.get(end + 1), Some(b'+' | b'-')));
            let exp = self.span(end + 1 + sign, digit);
            if exp > 0 {
                end += 1 + sign + exp;
                literal = true;
            }
        }
        let tail = self.span(end, is_word_char);
        let suffix = self.rest().get(end..end + tail).unwrap_or_default();
        let suffixes: &[&str] = if literal {
            &["D", "F", "BD"]
        } else {
            &["L", "S", "Y", "D", "F", "BD"]
        };
        if suffixes.iter().any(|s| s.eq_ignore_ascii_case(suffix)) {
            self.take(end + tail, Kind::Other)
        } else if literal {
            self.take(end, Kind::Other)
        } else if tail == 0 {
            self.take(end, Kind::Integer)
        } else {
            self.take(end + tail, Kind::Word { quoted: false })
        }
    }

    /// A backquoted identifier; ``` `` ``` inside it is a literal backquote.
    /// An unterminated backquote is a token of its own.
    fn quoted(&mut self) -> Token<'a> {
        let rest = self.rest();
        let mut text = String::new();
        let mut i = 1;
        while let Some(off) = rest.get(i..).and_then(|t| t.find('`')) {
            text.push_str(rest.get(i..i + off).unwrap_or_default());
            i += off + 1;
            if rest.get(i..).is_some_and(|t| t.starts_with('`')) {
                text.push('`');
                i += 1;
            } else {
                let mut t = self.take(i, Kind::Word { quoted: true });
                t.text = text;
                return t;
            }
        }
        self.take(1, Kind::Other)
    }
}

struct Parser<'a> {
    tokens: Vec<Token<'a>>,
    pos: usize,
    /// The first error that is not a syntax error, raised once the whole
    /// string has parsed.
    deferred: Option<krishiv_common::error::KrishivError>,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Token<'a>> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token<'a>> {
        let t = self.tokens.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    /// The error at `tok`, the next token: Spark drops a last character
    /// that starts no token as "extra input".
    fn unexpected(&self, tok: &Token<'a>) -> krishiv_common::error::KrishivError {
        let last = self.pos + 1 == self.tokens.len();
        // Spark's lexer has single-character tokens only for these; any other
        // character (`#`, `@`, `$`, `{`, ...) is UNRECOGNIZED.
        const SPARK_PUNCT: &str = "&*!:,.=>^[(<-%|+?]);/~";
        let unrecognized = match tok.kind {
            Kind::Other => !tok
                .raw
                .starts_with(|c: char| c.is_ascii_digit() || c == '.'),
            Kind::Punct(c) => !SPARK_PUNCT.contains(c),
            _ => false,
        };
        if last && unrecognized {
            error::extra_input(&format!("'{}'", tok.raw))
        } else {
            tok.syntax_error()
        }
    }

    /// Consumes the next token, which must satisfy `ok`.
    fn expect(&mut self, ok: impl Fn(&Token<'a>) -> bool) -> Result<Token<'a>> {
        match self.next() {
            Some(t) if ok(&t) => Ok(t),
            Some(t) => Err(t.syntax_error()),
            None => Err(error::syntax_error("end of input")),
        }
    }

    /// Consumes `keywords` after `lead`; a mismatch is reported at `lead`,
    /// as Spark does for `TIMESTAMP WITH TIME ZONE`.
    fn expect_keywords(&mut self, lead: &Token<'a>, keywords: &[&str]) -> Result<()> {
        for k in keywords {
            if !self.next().is_some_and(|t| t.is_keyword(k)) {
                return Err(lead.syntax_error());
            }
        }
        Ok(())
    }

    /// A type inside `depth` enclosing complex types.
    fn data_type(&mut self, depth: usize) -> Result<DataType> {
        let name = self.expect(|t| matches!(t.kind, Kind::Word { .. }))?;
        let quoted = name.kind == (Kind::Word { quoted: true });
        let lower = if quoted {
            name.text
        } else {
            let folded = name.folded();
            if !folded.is_ascii() {
                return Err(error::invalid_identifier(name.raw));
            }
            folded.to_ascii_lowercase()
        };
        if !quoted {
            if matches!(lower.as_str(), "array" | "map" | "struct") {
                match self.peek() {
                    Some(t) if matches!(t.kind, Kind::Punct('<')) || t.raw == "<>" => {
                        return self.complex(&lower, depth);
                    }
                    // `ARRAY(...)`: the arguments of no type.
                    Some(t) if t.kind == Kind::Punct('(') => return Err(t.syntax_error()),
                    _ => {}
                }
            }
            if let Some(feature) = self.unsupported_type(&lower) {
                self.skip_type_tail();
                if depth == 0
                    && let Some(tok) = self.peek()
                {
                    return Err(self.unexpected(tok));
                }
                return Ok(self.defer(error::unsupported(feature)));
            }
            if lower == "timestamp" {
                let lead = self.peek().cloned();
                if let Some(lead) = lead.filter(|t| t.is_keyword("with")) {
                    self.next();
                    self.expect_keywords(&lead, &["local", "time", "zone"])?;
                    return Ok(DataType::Timestamp);
                }
                let lead = self.peek().cloned();
                if let Some(lead) = lead.filter(|t| t.is_keyword("without")) {
                    self.next();
                    self.expect_keywords(&lead, &["time", "zone"])?;
                    return Ok(DataType::TimestampNtz);
                }
            }
            if matches!(lower.as_str(), "geometry" | "geography") && self.any_srid()? {
                // Inside a complex type the next token is legitimate (`>`,
                // `,`, `NOT NULL`, `COMMENT`); only a top-level type ends here.
                if depth == 0
                    && let Some(tok) = self.peek()
                {
                    return Err(error::extra_input(&format!("'{}'", tok.raw)));
                }
                return Ok(if lower == "geometry" {
                    DataType::Geometry { srid: Srid::Any }
                } else {
                    DataType::Geography { srid: Srid::Any }
                });
            }
        }
        let args = self.type_arguments()?;
        if !quoted && lower == "time" && self.peek().is_some_and(|t| t.is_keyword("without")) {
            self.time_without_time_zone()?;
        }
        if depth == 0
            && let Some(tok) = self.peek()
        {
            return Err(self.unexpected(tok));
        }
        let resolved = if quoted {
            None
        } else if let Some(e) = error::incomplete_type(&lower) {
            Some(Err(e))
        } else {
            resolve(&lower, &args)
        };
        Ok(
            match resolved.unwrap_or_else(|| Err(unsupported_data_type(&lower, &args))) {
                Ok(t) => t,
                Err(e) => self.defer(e),
            },
        )
    }

    /// Records `e` unless an earlier error is recorded; a placeholder type.
    fn defer(&mut self, e: krishiv_common::error::KrishivError) -> DataType {
        self.deferred.get_or_insert(e);
        DataType::Null
    }

    /// Skips the rest of a type Krishiv does not support (its words and a
    /// parenthesized argument list), so the type around it still parses.
    fn skip_type_tail(&mut self) {
        while let Some(t) = self.peek() {
            match t.kind {
                Kind::Word { .. } if !t.is_keyword("not") && !t.is_keyword("comment") => {
                    self.next();
                }
                Kind::Punct('(') => {
                    while let Some(t) = self.next() {
                        if t.kind == Kind::Punct(')') {
                            break;
                        }
                    }
                }
                _ => break,
            }
        }
    }

    /// `ARRAY<...>`, `MAP<...>` or `STRUCT<...>` (`kind`) at the `<` (or
    /// `<>`) after the keyword. A syntax error inside is reported at that
    /// `<`; an enclosing complex type moves it to its own.
    fn complex(&mut self, kind: &str, depth: usize) -> Result<DataType> {
        if depth >= DataType::MAX_NESTING_DEPTH {
            return Err(error::too_deep());
        }
        let Some(lt) = self.next() else {
            return Err(error::syntax_error("end of input"));
        };
        if lt.raw == "<>" {
            return if kind == "struct" {
                Ok(StructType::new([]).into())
            } else {
                Err(lt.syntax_error())
            };
        }
        let body = match kind {
            "array" => self
                .data_type(depth + 1)
                .map(|e| ArrayType::new(e, true).into()),
            "map" => self.map_body(depth),
            _ => self.struct_body(depth),
        };
        let body = body.and_then(|t| self.expect(|t| t.kind == Kind::Punct('>')).map(|_| t));
        body.map_err(|e| {
            if e.error_class() == ErrorClass::ParseSyntaxError {
                lt.syntax_error()
            } else {
                e
            }
        })
    }

    /// `k , v` of a map.
    fn map_body(&mut self, depth: usize) -> Result<DataType> {
        let key = self.data_type(depth + 1)?;
        self.expect(|t| t.kind == Kind::Punct(','))?;
        let value = self.data_type(depth + 1)?;
        Ok(MapType::new(key, value, true).into())
    }

    /// The fields of a struct, up to (not including) its `>`.
    fn struct_body(&mut self, depth: usize) -> Result<DataType> {
        let mut fields = Vec::new();
        if self.peek().is_some_and(|t| t.kind == Kind::Punct('>')) {
            return Ok(StructType::new(fields).into());
        }
        loop {
            fields.push(self.struct_field(depth)?);
            if !self.peek().is_some_and(|t| t.kind == Kind::Punct(',')) {
                return Ok(StructType::new(fields).into());
            }
            self.next();
        }
    }

    /// `name [:] type [NOT NULL] [COMMENT 'c' ...]`.
    fn struct_field(&mut self, depth: usize) -> Result<StructField> {
        let name = self.expect(|t| matches!(t.kind, Kind::Word { .. }))?;
        if name.kind == (Kind::Word { quoted: false }) && !name.text.is_ascii() {
            return Err(error::invalid_identifier(name.raw));
        }
        // Spark's `errorCapturingIdentifier`: `a-b` is an invalid name.
        let mut dashed = name.text.clone();
        while self.peek().is_some_and(|t| t.kind == Kind::Punct('-'))
            && self
                .tokens
                .get(self.pos + 1)
                .is_some_and(|t| matches!(t.kind, Kind::Word { .. }))
        {
            self.next();
            if let Some(part) = self.next() {
                dashed.push('-');
                dashed.push_str(&part.text);
            }
        }
        if dashed != name.text {
            self.defer(error::invalid_identifier(&dashed));
        }
        if self.peek().is_some_and(|t| t.kind == Kind::Punct(':')) {
            self.next();
        }
        let data_type = self.data_type(depth + 1)?;
        let mut nullable = true;
        if self.peek().is_some_and(|t| t.is_keyword("not")) {
            self.next();
            self.expect(|t| t.is_keyword("null"))?;
            nullable = false;
        }
        let mut field = StructField::new(name.text, data_type, nullable);
        if self.peek().is_some_and(|t| t.is_keyword("comment")) {
            self.next();
            let mut comment = self.expect(|t| t.kind == Kind::Str)?.text;
            while self.peek().is_some_and(|t| t.kind == Kind::Str) {
                if let Some(t) = self.next() {
                    comment.push_str(&t.text);
                }
            }
            field = field.with_metadata(Metadata::with_comment(comment));
        }
        Ok(field)
    }

    /// Recognized types that are not represented yet.
    fn unsupported_type(&self, lower: &str) -> Option<String> {
        let what = match lower {
            "interval" => "interval data types",
            "char" | "character" | "varchar" => "char/varchar data types",
            "variant" => "the variant data type",
            "string" if self.peek().is_some_and(|t| t.is_keyword("collate")) => "collated strings",
            _ => return None,
        };
        Some(format!("{what} (\"{}\")", lower.to_ascii_uppercase()))
    }

    /// `WITHOUT TIME ZONE` after `TIME[(p)]`, at `WITHOUT`. As in Spark, a
    /// `WITHOUT` that no `TIME` follows is extra input, and a missing
    /// `ZONE` is reported as missing.
    fn time_without_time_zone(&mut self) -> Result<()> {
        let without = self.next();
        if !self.peek().is_some_and(|t| t.is_keyword("time")) {
            let raw = without.map_or("", |t| t.raw);
            return Err(error::extra_input(&format!("'{raw}'")));
        }
        self.next();
        match self.next() {
            Some(t) if t.is_keyword("zone") => Ok(()),
            Some(t) => Err(t.syntax_error()),
            None => Err(error::missing("end of input", "'ZONE'")),
        }
    }

    /// After `GEOMETRY` or `GEOGRAPHY`: consumes `( ANY )` and returns
    /// true, or consumes nothing when an integer list follows. Spark
    /// reports any other argument at the `(`.
    fn any_srid(&mut self) -> Result<bool> {
        let Some(open) = self.peek().filter(|t| t.kind == Kind::Punct('(')).cloned() else {
            return Ok(false);
        };
        match self.tokens.get(self.pos + 1) {
            Some(t) if t.kind == Kind::Integer => Ok(false),
            Some(t)
                if t.is_keyword("any")
                    && self
                        .tokens
                        .get(self.pos + 2)
                        .is_some_and(|t| t.kind == Kind::Punct(')')) =>
            {
                self.pos += 3;
                Ok(true)
            }
            _ => Err(open.syntax_error()),
        }
    }

    /// `( INTEGER (, INTEGER)* )`, if present.
    fn type_arguments(&mut self) -> Result<Vec<String>> {
        let mut args = Vec::new();
        if self.peek().is_some_and(|t| t.kind == Kind::Punct('(')) {
            self.next();
            loop {
                args.push(self.expect(|t| t.kind == Kind::Integer)?.text);
                if self
                    .expect(|t| matches!(t.kind, Kind::Punct(',' | ')')))?
                    .kind
                    == Kind::Punct(')')
                {
                    break;
                }
            }
        }
        Ok(args)
    }
}

/// The type named `lower` with `args`, or `None` if there is no such type.
fn resolve(lower: &str, args: &[String]) -> Option<Result<DataType>> {
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let t = match (lower, args.as_slice()) {
        ("void", []) => DataType::Null,
        ("boolean", []) => DataType::Boolean,
        ("tinyint" | "byte", []) => DataType::Byte,
        ("smallint" | "short", []) => DataType::Short,
        ("int" | "integer", []) => DataType::Integer,
        ("bigint" | "long", []) => DataType::Long,
        ("float" | "real", []) => DataType::Float,
        ("double", []) => DataType::Double,
        ("string", []) => DataType::String,
        ("binary", []) => DataType::Binary,
        ("date", []) => DataType::Date,
        ("timestamp" | "timestamp_ltz", []) => DataType::Timestamp,
        ("timestamp_ntz", []) => DataType::TimestampNtz,
        ("decimal" | "dec" | "numeric", []) => DataType::Decimal(DecimalType::USER_DEFAULT),
        ("decimal" | "dec" | "numeric", [p]) => return Some(decimal(p, "0")),
        ("decimal" | "dec" | "numeric", [p, s]) => return Some(decimal(p, s)),
        ("time", []) => DataType::Time {
            precision: DataType::TIME_DEFAULT_PRECISION,
        },
        ("time", [p]) => {
            return Some(match p.parse::<u32>() {
                Ok(p) => DataType::time(p),
                Err(_) => Err(error::time_precision(&digits(p))),
            });
        }
        ("geometry" | "geography", [srid]) => {
            return Some(match srid.parse::<i32>() {
                Ok(s) if lower == "geometry" => DataType::geometry(Srid::Fixed(s)),
                Ok(s) => DataType::geography(Srid::Fixed(s)),
                Err(_) => Err(error::invalid_srid(&digits(srid))),
            });
        }
        _ => return None,
    };
    Some(Ok(t))
}

/// Digits as written, without leading zeros.
fn digits(s: &str) -> String {
    match s.trim_start_matches('0') {
        "" => "0".to_owned(),
        d => d.to_owned(),
    }
}

/// Values too large for `u32` (Spark: an unclassified number format error)
/// are reported with the digits as written, without leading zeros.
fn decimal(precision: &str, scale: &str) -> Result<DataType> {
    match (precision.parse::<u32>(), scale.parse::<u32>()) {
        (Ok(p), Ok(s)) => DecimalType::try_new(p, s).map(DataType::Decimal),
        (Ok(_), Err(_)) => Err(error::scale_exceeds_precision(
            &digits(precision),
            &digits(scale),
        )),
        (Err(_), _) => Err(error::precision_exceeds_max(&digits(precision))),
    }
}

/// Spark renders the rejected type as `NAME(arg,...)`, upper-cased.
fn unsupported_data_type(name: &str, args: &[String]) -> krishiv_common::error::KrishivError {
    let mut s = name.to_ascii_uppercase();
    if !args.is_empty() {
        s.push('(');
        s.push_str(&args.join(","));
        s.push(')');
    }
    error::unsupported_data_type(&s)
}

/// The value of a string literal token, as Spark's `unescapeSQLString`
/// decodes it: a raw string as written; otherwise backslash escapes
/// (`\n`, `\uXXXX`, octal `\101`, ...) and doubled quotes.
fn unescape_string(raw: &str) -> String {
    if let Some(body) = raw
        .strip_prefix(['r', 'R'])
        .and_then(|r| r.get(1..r.len().saturating_sub(1)))
    {
        return body.to_owned();
    }
    let quote = raw.chars().next().unwrap_or('\'');
    let body: Vec<char> = raw
        .get(1..raw.len().saturating_sub(1))
        .unwrap_or_default()
        .chars()
        .collect();
    // Java strings are UTF-16: an escaped lone surrogate becomes U+FFFD.
    let mut out: Vec<u16> = Vec::with_capacity(body.len());
    let mut buf = [0_u16; 2];
    let push = |out: &mut Vec<u16>, c: char, buf: &mut [u16; 2]| {
        out.extend_from_slice(c.encode_utf16(buf));
    };
    let hex = |from: usize, n: usize| -> Option<u32> {
        let s: String = body.get(from..from + n)?.iter().collect();
        s.chars()
            .all(|c| c.is_ascii_hexdigit())
            .then(|| u32::from_str_radix(&s, 16).ok())?
    };
    let octal = |at: usize| -> Option<u16> {
        let d = |i: usize| body.get(at + i).and_then(|c| c.to_digit(8));
        let (a, b, c) = (d(0)?, d(1)?, d(2)?);
        (a <= 1).then(|| u16::try_from(a * 64 + b * 8 + c).unwrap_or(0xFFFD))
    };
    let mut i = 0;
    while let Some(&c) = body.get(i) {
        let next = body.get(i + 1).copied();
        if c == quote && next == Some(quote) {
            push(&mut out, c, &mut buf);
            i += 2;
        } else if c != '\\' || next.is_none() {
            push(&mut out, c, &mut buf);
            i += 1;
        } else if let (Some('u'), Some(v)) = (next, hex(i + 2, 4)) {
            out.push(u16::try_from(v).unwrap_or(0xFFFD));
            i += 6;
        } else if let (Some('U'), Some(v)) = (next, hex(i + 2, 8)) {
            match char::from_u32(v) {
                Some(ch) => push(&mut out, ch, &mut buf),
                None => out.push(0xFFFD),
            }
            i += 10;
        } else if let Some(v) = octal(i + 1) {
            out.push(v);
            i += 4;
        } else {
            match next.unwrap_or('\\') {
                '0' => out.push(0),
                'b' => out.push(8),
                'n' => push(&mut out, '\n', &mut buf),
                'r' => push(&mut out, '\r', &mut buf),
                't' => push(&mut out, '\t', &mut buf),
                'Z' => out.push(0x1A),
                n @ ('%' | '_') => {
                    push(&mut out, '\\', &mut buf);
                    push(&mut out, n, &mut buf);
                }
                n => push(&mut out, n, &mut buf),
            }
            i += 2;
        }
    }
    String::from_utf16_lossy(&out)
}
