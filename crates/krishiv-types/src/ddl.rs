// SPDX-License-Identifier: Apache-2.0
//! Parser for Spark DDL type strings (`CatalystSqlParser.parseDataType`).
//!
//! Follows Spark's `dataType` grammar for the types in [`DataType`]: a type
//! name, optionally followed by `(` integer `,` ... `)`, plus the multi-word
//! forms `TIMESTAMP WITH LOCAL TIME ZONE` and `TIMESTAMP WITHOUT TIME ZONE`.
//! Keywords are case-insensitive. Nested, interval, char/varchar, variant,
//! void, time and collated string types are recognized and rejected with
//! `KRISHIV_UNSUPPORTED_FEATURE`. SQL comments inside a type string are not
//! supported.

use std::iter::Peekable;
use std::str::CharIndices;

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
/// `PARSE_SYNTAX_ERROR` / `PARSE_EMPTY_STATEMENT` for a malformed string,
/// the Spark decimal bound errors, `UNSUPPORTED_DATATYPE` for an unknown type
/// and `KRISHIV_UNSUPPORTED_FEATURE` for a Spark type not supported yet (see
/// [`crate::error`]).
pub fn parse_data_type(input: &str) -> Result<DataType> {
    let mut p = Parser {
        lexer: Lexer {
            src: input,
            chars: input.char_indices().peekable(),
        },
        peeked: None,
    };
    if p.peek().is_none() {
        return Err(error::empty_statement());
    }
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
    /// Any other token (a non-integer number, an unknown character).
    Other,
}

#[derive(Debug, Clone, Copy)]
struct Token<'a> {
    kind: Kind,
    /// The token's text (for a quoted identifier, without the backquotes).
    text: &'a str,
    /// The token as written, for error messages.
    raw: &'a str,
}

impl Token<'_> {
    fn is_keyword(&self, keyword: &str) -> bool {
        self.kind == (Kind::Word { quoted: false }) && self.text.eq_ignore_ascii_case(keyword)
    }

    fn syntax_error(&self) -> krishiv_common::error::KrishivError {
        error::syntax_error(&format!("'{}'", self.raw))
    }
}

struct Lexer<'a> {
    src: &'a str,
    chars: Peekable<CharIndices<'a>>,
}

impl<'a> Lexer<'a> {
    fn slice(&self, start: usize, end: usize) -> &'a str {
        self.src.get(start..end).unwrap_or_default()
    }

    /// Consumes characters while `pred` holds; returns the end offset.
    fn eat_while(&mut self, pred: impl Fn(char) -> bool) -> usize {
        while let Some(&(_, c)) = self.chars.peek() {
            if !pred(c) {
                break;
            }
            self.chars.next();
        }
        self.chars.peek().map_or(self.src.len(), |&(i, _)| i)
    }

    fn next_token(&mut self) -> Option<Token<'a>> {
        self.eat_while(|c| c.is_ascii_whitespace() || c == '\u{0B}');
        let (start, c) = self.chars.next()?;
        let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
        let unquoted = |kind, raw| Token {
            kind,
            text: raw,
            raw,
        };
        let token = if is_word(c) {
            let end = self.eat_while(is_word);
            let text = self.slice(start, end);
            if !text.bytes().all(|b| b.is_ascii_digit()) {
                unquoted(Kind::Word { quoted: false }, text)
            } else if self.chars.peek().is_some_and(|&(_, c)| c == '.') {
                // A decimal literal such as `1.5` is a single token.
                self.chars.next();
                let end = self.eat_while(is_word);
                unquoted(Kind::Other, self.slice(start, end))
            } else {
                unquoted(Kind::Integer, text)
            }
        } else if c == '`' {
            let end = self.eat_while(|c| c != '`');
            let close = self.chars.next().map_or(end, |(i, _)| i + 1);
            Token {
                kind: Kind::Word { quoted: true },
                text: self.slice(start + 1, end),
                raw: self.slice(start, close),
            }
        } else {
            let kind = if c.is_ascii_punctuation() {
                Kind::Punct(c)
            } else {
                Kind::Other
            };
            unquoted(kind, self.slice(start, start + c.len_utf8()))
        };
        Some(token)
    }
}

struct Parser<'a> {
    lexer: Lexer<'a>,
    peeked: Option<Option<Token<'a>>>,
}

impl<'a> Parser<'a> {
    fn peek(&mut self) -> Option<Token<'a>> {
        *self.peeked.get_or_insert_with(|| self.lexer.next_token())
    }

    fn next(&mut self) -> Option<Token<'a>> {
        match self.peeked.take() {
            Some(t) => t,
            None => self.lexer.next_token(),
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

    fn expect_keywords(&mut self, keywords: &[&str]) -> Result<()> {
        for k in keywords {
            self.expect(|t| t.is_keyword(k))?;
        }
        Ok(())
    }

    fn data_type(&mut self) -> Result<DataType> {
        let name = self.expect(|t| matches!(t.kind, Kind::Word { .. }))?;
        let quoted = name.kind == (Kind::Word { quoted: true });
        let lower = name.text.to_ascii_lowercase();
        if !quoted {
            if let Some(feature) = self.unsupported_type(&lower) {
                return Err(error::unsupported(feature));
            }
            if lower == "timestamp" {
                if self.peek().is_some_and(|t| t.is_keyword("with")) {
                    self.next();
                    self.expect_keywords(&["local", "time", "zone"])?;
                    return Ok(DataType::Timestamp);
                }
                if self.peek().is_some_and(|t| t.is_keyword("without")) {
                    self.next();
                    self.expect_keywords(&["time", "zone"])?;
                    return Ok(DataType::TimestampNtz);
                }
            }
        }
        let args = self.type_arguments()?;
        if let Some(tok) = self.peek() {
            return Err(tok.syntax_error());
        }
        if quoted {
            return Err(unsupported_data_type(name.text, &args));
        }
        resolve(&lower, &args).unwrap_or_else(|| Err(unsupported_data_type(name.text, &args)))
    }

    /// Recognized types that are not represented yet.
    fn unsupported_type(&mut self, lower: &str) -> Option<String> {
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
    fn type_arguments(&mut self) -> Result<Vec<&'a str>> {
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
fn resolve(lower: &str, args: &[&str]) -> Option<Result<DataType>> {
    let t = match (lower, args) {
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

fn decimal(precision: &str, scale: &str) -> Result<DataType> {
    let Ok(p) = precision.parse::<u32>() else {
        let digits = precision.trim_start_matches('0');
        return Err(error::precision_exceeds_max(digits));
    };
    let s = scale.parse::<u32>().unwrap_or(u32::MAX);
    DecimalType::try_new(p, s).map(DataType::Decimal)
}

/// Spark renders the rejected type as `NAME(arg,...)`, upper-cased.
fn unsupported_data_type(name: &str, args: &[&str]) -> krishiv_common::error::KrishivError {
    let mut s = name.to_ascii_uppercase();
    if !args.is_empty() {
        s.push('(');
        s.push_str(&args.join(","));
        s.push(')');
    }
    error::unsupported_data_type(&s)
}
