// SPDX-License-Identifier: Apache-2.0
//! Parser for Spark DDL type strings (`CatalystSqlParser.parseDataType`).
//!
//! Follows Spark's `dataType` grammar for the types in [`DataType`]: a type
//! name, optionally followed by `(` integer `,` ... `)`, plus the multi-word
//! forms `TIMESTAMP WITH LOCAL TIME ZONE` and `TIMESTAMP WITHOUT TIME ZONE`.
//! The lexer follows Spark's: case-insensitive keywords, Unicode whitespace,
//! `--` and nested `/* */` comments, backquoted identifiers and numeric
//! literal tokens. Nested, interval, char/varchar, variant, void, time and
//! collated string types are recognized and rejected with
//! `KRISHIV_UNSUPPORTED_FEATURE`.

use crate::error::{self, Result};
use crate::{DataType, DecimalType};

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
    let tokens = Lexer { src: input, pos: 0 }.tokens()?;
    if tokens.is_empty() {
        return Err(error::empty_statement());
    }
    let mut p = Parser { tokens, pos: 0 };
    let data_type = p.data_type()?;
    match p.next() {
        None => Ok(data_type),
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

/// Spark's `WS` token: ASCII whitespace plus the Unicode space separators.
fn is_spark_whitespace(c: char) -> bool {
    c.is_ascii_whitespace()
        || matches!(
            c,
            '\u{0B}' | '\u{A0}' | '\u{1680}' | '\u{2000}'..='\u{200A}'
        )
        || matches!(
            c,
            '\u{2028}' | '\u{2029}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
        )
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

struct Lexer<'a> {
    src: &'a str,
    pos: usize,
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
        let starts_number =
            c.is_ascii_digit() || (c == '.' && chars.next().is_some_and(|d| d.is_ascii_digit()));
        if starts_number {
            self.number()
        } else if is_word_char(c) {
            self.take(self.span(0, is_word_char), Kind::Word { quoted: false })
        } else if c == '`' {
            self.quoted()
        } else {
            let kind = if c.is_ascii_punctuation() {
                Kind::Punct(c)
            } else {
                Kind::Other
            };
            self.take(c.len_utf8(), kind)
        }
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

    fn data_type(&mut self) -> Result<DataType> {
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
            if let Some(feature) = self.unsupported_type(&lower) {
                return Err(error::unsupported(feature));
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
        }
        let args = self.type_arguments()?;
        if let Some(tok) = self.peek() {
            return Err(tok.syntax_error());
        }
        let unsupported = || Err(unsupported_data_type(&lower, &args));
        if quoted {
            return unsupported();
        }
        resolve(&lower, &args).unwrap_or_else(unsupported)
    }

    /// Recognized types that are not represented yet.
    fn unsupported_type(&self, lower: &str) -> Option<String> {
        let what = match lower {
            "array" | "map" | "struct" => "nested data types",
            "interval" => "interval data types",
            "char" | "character" | "varchar" => "char/varchar data types",
            "variant" => "the variant data type",
            "void" => "the void data type",
            "time" => "the time data type",
            "string" if self.peek().is_some_and(|t| t.is_keyword("collate")) => "collated strings",
            _ => return None,
        };
        Some(format!("{what} (\"{}\")", lower.to_ascii_uppercase()))
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
        _ => return None,
    };
    Some(Ok(t))
}

/// Values too large for `u32` (Spark: an unclassified number format error)
/// are reported with the digits as written, without leading zeros.
fn decimal(precision: &str, scale: &str) -> Result<DataType> {
    let digits = |s: &'_ str| match s.trim_start_matches('0') {
        "" => "0".to_owned(),
        d => d.to_owned(),
    };
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
