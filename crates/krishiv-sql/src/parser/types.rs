// SPDX-License-Identifier: Apache-2.0
//! Data types (`dataType`), and the type-string entry points Spark parses
//! with the same grammar: `singleDataType` (`CatalystSqlParser.parseDataType`)
//! and `singleTableSchema` (`parseTableSchema`).
//!
//! Spark's errors for malformed types come from ANTLR's prediction, which
//! this parser reproduces in two places:
//!
//! * A complex type whose `<...>` does not parse falls back to its bare form
//!   (`ARRAY`), when the bare alternative is still viable where the complex
//!   one fails ([`Parser::bare_type_viable_until`]); the error then surfaces
//!   at the `<`, in the rule that called `dataType`.
//! * Optional parts of a type (arguments, `WITH LOCAL TIME ZONE`, interval
//!   units) are taken only when complete, as an LL(*) prediction takes them;
//!   after a bare type keyword, a token that cannot continue the type is a
//!   syntax error without the `extra input` hint
//!   ([`Parser::extra_after_type`]).

use super::{Fail, PR, Parser};
use crate::ast::{DataTypeKind, DataTypeSpec, NumKind, StructField};
use crate::keywords::is_kw;
use crate::lexer::{Sym, TokenKind, unquote_ident};

/// Spark's type keywords: after one of them alone, a token that cannot
/// continue the type is reported without a hint.
const TYPE_KEYWORDS: &[&str] = &[
    "BOOLEAN",
    "TINYINT",
    "BYTE",
    "SMALLINT",
    "SHORT",
    "INT",
    "INTEGER",
    "BIGINT",
    "LONG",
    "FLOAT",
    "REAL",
    "DOUBLE",
    "DATE",
    "TIMESTAMP",
    "TIMESTAMP_NTZ",
    "TIMESTAMP_LTZ",
    "STRING",
    "CHARACTER",
    "CHAR",
    "VARCHAR",
    "BINARY",
    "DECIMAL",
    "DEC",
    "NUMERIC",
    "VOID",
    "INTERVAL",
    "VARIANT",
    "TIME",
    "ARRAY",
    "MAP",
    "STRUCT",
];

/// Types whose `(` always starts their arguments: an error inside them is
/// reported there, not at the `(`.
const COMMITTED_ARGUMENTS: &[&str] = &[
    "CHAR",
    "CHARACTER",
    "VARCHAR",
    "DECIMAL",
    "DEC",
    "NUMERIC",
    "TIME",
];

/// Interval units, in order.
const UNITS: &[&str] = &["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"];

/// Words that follow `NOT` in a predicate (`a NOT IN (...)`).
const NOT_PREDICATES: &[&str] = &["IN", "LIKE", "ILIKE", "RLIKE", "REGEXP", "BETWEEN"];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Complex {
    Array,
    Map,
    Struct,
}

impl Complex {
    fn bare(self) -> DataTypeKind {
        match self {
            Self::Array => DataTypeKind::Array(None),
            Self::Map => DataTypeKind::Map(None),
            Self::Struct => DataTypeKind::Struct(None),
        }
    }
}

/// How a type takes `( arguments )`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arguments {
    /// A `(` starts them; an error inside is reported there.
    Committed,
    /// Integers, taken only if the whole list is there.
    IfComplete,
    /// One integer or `ANY` (geometry types), only if complete.
    Srid,
}

impl DataTypeSpec {
    /// Whether this is `ARRAY<...>`, `MAP<...>` or `STRUCT<...>`.
    pub(crate) fn is_complex(&self) -> bool {
        matches!(
            self.kind,
            DataTypeKind::Array(Some(_))
                | DataTypeKind::Map(Some(_))
                | DataTypeKind::Struct(Some(_))
        )
    }
}

/// A state of [`Parser::bare_type_viable_until`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Follow {
    Operand,
    AfterOperand,
    AfterAlias,
}

impl Parser<'_> {
    /// `singleDataType`: one data type and the end of the input.
    pub(super) fn single_data_type(&mut self) -> PR<DataTypeSpec> {
        let t = self.data_type()?;
        if self.pos < self.toks.len() {
            self.extra_after_type(&t, &[]);
            return self.fail();
        }
        Ok(t)
    }

    /// `singleTableSchema`: `name type [NOT NULL] [COMMENT '...'], ...` and
    /// the end of the input.
    pub(super) fn single_table_schema(&mut self) -> PR<Vec<StructField>> {
        let columns = self.comma_sep(Self::col_type)?;
        if self.pos < self.toks.len() {
            return self.fail();
        }
        Ok(columns)
    }

    /// `colType`: a column of a table schema.
    fn col_type(&mut self) -> PR<StructField> {
        let start = self.start();
        let name = self.name_part(true)?;
        let data_type = self.data_type()?;
        let continues = self.kind(0).is_none()
            || self.sym_at(0, Sym::Comma)
            || self.kw_at(0, "NOT")
            || self.kw_at(0, "COMMENT");
        if !continues {
            self.extra_after_type(&data_type, &[]);
            return self.fail();
        }
        let not_null = self.eat_kw("NOT");
        if not_null && !self.eat_kw("NULL") {
            // ANTLR inserts the missing `NULL` when what follows fits after it.
            if self.kind(0).is_none() || self.sym_at(0, Sym::Comma) || self.kw_at(0, "COMMENT") {
                self.missing = Some((self.pos, "'NULL'"));
            }
            return self.fail();
        }
        let comment = if self.eat_kw("COMMENT") {
            match self.string_lit() {
                Some(s) => Some(s),
                None => return self.fail(),
            }
        } else {
            None
        };
        Ok(StructField {
            name,
            data_type,
            not_null,
            comment,
            span: self.span_from(start),
        })
    }

    /// Notes Spark's `extra input` hint for the token after the type `t`,
    /// where the rule expected one of `follow` or the end of the input:
    /// ANTLR drops that token when the one after it fits. After a type
    /// keyword alone, only a token that could continue the type (an
    /// argument list, `WITH`, a unit, ...) gets the hint.
    fn extra_after_type(&mut self, t: &DataTypeSpec, follow: &[Sym]) {
        // A character no token starts with is dropped like any other.
        let unrecognized = self.kind(0) == Some(TokenKind::Unrecognized);
        if self.bare_keyword_type(t) && !self.continues_type(t) && !unrecognized {
            return;
        }
        let fits = match self.kind(1) {
            None => true,
            Some(TokenKind::Sym(s)) => follow.contains(&s),
            Some(_) => false,
        };
        if fits {
            self.extra = Some(self.pos);
        }
    }

    /// Whether `t` is a type keyword with nothing after it.
    pub(super) fn bare_keyword_type(&self, t: &DataTypeSpec) -> bool {
        let text = t.span.text(self.src);
        TYPE_KEYWORDS.iter().any(|k| is_kw(text, k))
    }

    /// Whether the current token could continue the bare type keyword `t`.
    pub(super) fn continues_type(&self, t: &DataTypeSpec) -> bool {
        let text = t.span.text(self.src);
        let is = |kw: &str| is_kw(text, kw);
        self.sym_at(0, Sym::LParen)
            || (matches!(
                t.kind,
                DataTypeKind::Array(None) | DataTypeKind::Map(None) | DataTypeKind::Struct(None)
            ) && self.sym_at(0, Sym::Lt))
            || ((is("TIMESTAMP") || is("TIME")) && self.any_kw_at(0, &["WITH", "WITHOUT"]))
            || (is("STRING") && self.kw_at(0, "COLLATE"))
            || (is("INTERVAL") && self.any_kw_at(0, UNITS))
    }

    /// `dataType`.
    pub(crate) fn data_type(&mut self) -> PR<DataTypeSpec> {
        self.nested(|p| {
            let start = p.start();
            let quoted = match p.kind(0) {
                Some(TokenKind::Word) => false,
                Some(TokenKind::QuotedIdent) => true,
                Some(TokenKind::DoubleQuoted) if p.opts.double_quoted_identifiers => true,
                _ => return p.fail(),
            };
            let keyword = p.pos;
            let word = p.text(0);
            // Spark's `exitUnquotedIdentifier`: a type name that is not a
            // keyword is an identifier.
            let keyword_type = TYPE_KEYWORDS.iter().any(|k| is_kw(word, k));
            if !quoted
                && !keyword_type
                && !word.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
            {
                let span = p.toks[keyword].span;
                return Self::fatal(crate::error::invalid_identifier(word, span));
            }
            p.bump();
            let is = |kw: &str| !quoted && is_kw(word, kw);
            let complex = if is("ARRAY") {
                Some(Complex::Array)
            } else if is("MAP") {
                Some(Complex::Map)
            } else if is("STRUCT") {
                Some(Complex::Struct)
            } else {
                None
            };
            let kind = match complex {
                Some(Complex::Struct) if p.force_bare != Some(keyword) && p.eat_sym(Sym::Neq) => {
                    DataTypeKind::Struct(Some(vec![]))
                }
                Some(c) if p.sym_at(0, Sym::Lt) && p.force_bare != Some(keyword) => {
                    p.complex_type(c, keyword)?
                }
                Some(c) => {
                    let args = p.type_arguments(Arguments::IfComplete)?;
                    if args.is_empty() {
                        c.bare()
                    } else {
                        DataTypeKind::Primitive {
                            name: word.to_ascii_lowercase(),
                            args,
                        }
                    }
                }
                None if quoted => DataTypeKind::Quoted {
                    name: quoted_name(word),
                    args: p.type_arguments(Arguments::IfComplete)?,
                },
                None => p.primitive_type(word)?,
            };
            Ok(DataTypeSpec {
                kind,
                span: p.span_from(start),
            })
        })
    }

    /// `ARRAY<...>`, `MAP<...>` or `STRUCT<...>` after the keyword at token
    /// `keyword`, or the bare type when ANTLR's prediction would fall back
    /// to it.
    fn complex_type(&mut self, c: Complex, keyword: usize) -> PR<DataTypeKind> {
        let (far, extra, missing) = (self.far, self.extra, self.missing);
        self.far = 0;
        let r = self.complex_body(c);
        let failed_at = self.far;
        self.far = self.far.max(far);
        match r {
            Err(Fail::Syntax) if self.bare_type_viable_until(keyword) >= failed_at => {
                self.pos = keyword + 1;
                (self.far, self.extra, self.missing) = (far, extra, missing);
                Ok(c.bare())
            }
            r => r,
        }
    }

    fn complex_body(&mut self, c: Complex) -> PR<DataTypeKind> {
        self.expect_sym(Sym::Lt)?;
        let kind = match c {
            Complex::Array => DataTypeKind::Array(Some(Box::new(self.data_type()?))),
            Complex::Map => {
                let k = self.data_type()?;
                self.expect_sym(Sym::Comma)?;
                let v = self.data_type()?;
                DataTypeKind::Map(Some(Box::new((k, v))))
            }
            Complex::Struct => {
                let fields = if self.sym_at(0, Sym::Gt) {
                    vec![]
                } else {
                    self.comma_sep(Self::struct_field)?
                };
                if !self.sym_at(0, Sym::Gt)
                    && (self.sym_at(1, Sym::Comma) || self.sym_at(1, Sym::Gt))
                {
                    // ANTLR drops one extraneous token before `,` or `>`.
                    self.extra = Some(self.pos);
                }
                DataTypeKind::Struct(Some(fields))
            }
        };
        self.expect_sym(Sym::Gt)?;
        Ok(kind)
    }

    /// The token at which the bare form of the complex type keyword at token
    /// `keyword` stops being viable for ANTLR's SLL prediction, which
    /// ignores the calling rule: a bare `ARRAY` may end a cast inside any
    /// expression, so `ARRAY < int` reads on as a comparison, `, x` as the
    /// next item and `x` as an alias. The token count when it stays viable
    /// to the end.
    fn bare_type_viable_until(&self, keyword: usize) -> usize {
        let (mut state, mut depth) = (Follow::AfterOperand, 0_usize);
        let mut i = keyword + 1;
        while let Some(t) = self.toks.get(i) {
            let word = |kw: &str| t.kind == TokenKind::Word && is_kw(t.span.text(self.src), kw);
            let next_word = |kw: &str| {
                self.toks
                    .get(i + 1)
                    .is_some_and(|n| n.kind == TokenKind::Word && is_kw(n.span.text(self.src), kw))
            };
            state = match (state, t.kind) {
                (Follow::Operand, TokenKind::Word | TokenKind::QuotedIdent)
                    if !word("NOT")
                        && self
                            .toks
                            .get(i + 1)
                            .is_some_and(|n| n.kind == TokenKind::Sym(Sym::LParen)) =>
                {
                    depth += 1;
                    i += 1;
                    Follow::Operand
                }
                (Follow::Operand, TokenKind::Word) if word("NOT") => Follow::Operand,
                (
                    Follow::Operand,
                    TokenKind::Word
                    | TokenKind::QuotedIdent
                    | TokenKind::Number(_)
                    | TokenKind::String
                    | TokenKind::DoubleQuoted,
                ) => Follow::AfterOperand,
                (Follow::Operand, TokenKind::Sym(Sym::LParen)) => {
                    depth += 1;
                    Follow::Operand
                }
                // `f()`.
                (Follow::Operand, TokenKind::Sym(Sym::RParen))
                    if depth > 0
                        && self
                            .toks
                            .get(i - 1)
                            .is_some_and(|p| p.kind == TokenKind::Sym(Sym::LParen)) =>
                {
                    depth -= 1;
                    Follow::AfterOperand
                }
                (Follow::Operand, TokenKind::Sym(Sym::Minus | Sym::Plus | Sym::Tilde))
                | (Follow::AfterAlias, TokenKind::Sym(Sym::Comma))
                | (
                    Follow::AfterOperand,
                    TokenKind::Sym(
                        Sym::Lt
                        | Sym::Gt
                        | Sym::Le
                        | Sym::Ge
                        | Sym::Eq
                        | Sym::Neq
                        | Sym::NullSafeEq
                        | Sym::Plus
                        | Sym::Minus
                        | Sym::Star
                        | Sym::Slash
                        | Sym::Percent
                        | Sym::Concat
                        | Sym::Amp
                        | Sym::Pipe
                        | Sym::Caret
                        | Sym::Comma,
                    ),
                ) => Follow::Operand,
                (Follow::AfterOperand, TokenKind::Word) if word("AND") || word("OR") => {
                    Follow::Operand
                }
                (Follow::AfterOperand, TokenKind::Word)
                    if word("NOT") && NOT_PREDICATES.iter().any(|k| next_word(k)) =>
                {
                    i += 1;
                    Follow::Operand
                }
                (Follow::AfterOperand, TokenKind::Word) if word("NOT") => return i + 1,
                // A variant path or a qualified name: `a:b`, `a.b`.
                (Follow::AfterOperand, TokenKind::Sym(Sym::Colon | Sym::Dot))
                    if self.toks.get(i + 1).is_some_and(|n| {
                        matches!(n.kind, TokenKind::Word | TokenKind::QuotedIdent)
                    }) =>
                {
                    i += 1;
                    Follow::AfterOperand
                }
                (Follow::AfterOperand, TokenKind::Sym(Sym::RParen)) if depth > 0 => {
                    depth -= 1;
                    Follow::AfterOperand
                }
                (Follow::AfterOperand, TokenKind::Word | TokenKind::QuotedIdent) if depth == 0 => {
                    Follow::AfterAlias
                }
                _ => return i,
            };
            i += 1;
        }
        // The end of the input ends the expression unless an operand is due.
        if state == Follow::Operand || depth > 0 {
            self.toks.len()
        } else {
            self.toks.len() + 1
        }
    }

    #[inline(never)]
    fn struct_field(&mut self) -> PR<StructField> {
        let start = self.start();
        let name = self.name_part(true)?;
        self.eat_sym(Sym::Colon);
        let data_type = self.data_type()?;
        let not_null = self.kw_at(0, "NOT") && self.kw_at(1, "NULL");
        self.pos += 2 * usize::from(not_null);
        let comment = if self.eat_kw("COMMENT") {
            match self.string_lit() {
                Some(s) => Some(s),
                None => return self.fail(),
            }
        } else {
            None
        };
        Ok(StructField {
            name,
            data_type,
            not_null,
            comment,
            span: self.span_from(start),
        })
    }

    /// `( arguments )` of a type, taken as `how` says; empty if absent.
    fn type_arguments(&mut self, how: Arguments) -> PR<Vec<String>> {
        if !self.sym_at(0, Sym::LParen) {
            return Ok(vec![]);
        }
        let int_at =
            |p: &Self, n: usize| matches!(p.kind(n), Some(TokenKind::Number(NumKind::Int)));
        match how {
            Arguments::Committed => {
                self.bump();
                let args = self.comma_sep(|p| {
                    if !int_at(p, 0) {
                        return p.fail();
                    }
                    let a = p.text(0).to_owned();
                    p.bump();
                    Ok(a)
                })?;
                self.expect_sym(Sym::RParen)?;
                Ok(args)
            }
            Arguments::IfComplete => {
                let mut n = 1;
                while int_at(self, n) && self.sym_at(n + 1, Sym::Comma) {
                    n += 2;
                }
                if !(int_at(self, n) && self.sym_at(n + 1, Sym::RParen)) {
                    return Ok(vec![]);
                }
                let args = (0..=n / 2)
                    .map(|k| self.text(1 + 2 * k).to_owned())
                    .collect();
                self.pos += n + 2;
                Ok(args)
            }
            Arguments::Srid => {
                if !((int_at(self, 1) || self.kw_at(1, "ANY")) && self.sym_at(2, Sym::RParen)) {
                    return Ok(vec![]);
                }
                let srid = self.text(1);
                let arg = if int_at(self, 1) {
                    srid.to_owned()
                } else {
                    srid.to_ascii_uppercase()
                };
                self.pos += 3;
                Ok(vec![arg])
            }
        }
    }

    /// `primitiveType` after its first word.
    #[inline(never)]
    fn primitive_type(&mut self, word: &str) -> PR<DataTypeKind> {
        let is = |kw: &str| is_kw(word, kw);
        // The keyword's own spelling: Spark folds case per character, so
        // `ınt` is `INT`.
        let mut name = TYPE_KEYWORDS
            .iter()
            .find(|k| is(k))
            .map_or_else(|| word.to_ascii_lowercase(), |k| k.to_ascii_lowercase());
        let mut push = |p: &mut Self, words: &[&str]| {
            for w in words {
                name.push(' ');
                name.push_str(&w.to_ascii_lowercase());
            }
            p.pos += words.len();
        };
        if is("INTERVAL")
            && let Some(from) = UNITS.iter().position(|u| self.kw_at(0, u))
        {
            {
                let unit = UNITS[from];
                push(self, &[unit]);
                // A range runs to a smaller unit of the same family.
                let to = if from < 2 {
                    &UNITS[from + 1..2]
                } else {
                    &UNITS[from + 1..]
                };
                if self.kw_at(0, "TO")
                    && let Some(end) = to.iter().find(|u| self.kw_at(1, u))
                {
                    push(self, &["TO", end]);
                }
                return Ok(DataTypeKind::Primitive { name, args: vec![] });
            }
        }
        let how = if COMMITTED_ARGUMENTS.iter().any(|k| is(k)) {
            Arguments::Committed
        } else if is("GEOMETRY") || is("GEOGRAPHY") {
            Arguments::Srid
        } else {
            Arguments::IfComplete
        };
        let args = self.type_arguments(how)?;
        let zone = |p: &Self, words: &[&str]| words.iter().enumerate().all(|(n, w)| p.kw_at(n, w));
        if (is("TIMESTAMP") || is("TIME")) && zone(self, &["WITHOUT", "TIME", "ZONE"]) {
            push(self, &["WITHOUT", "TIME", "ZONE"]);
        } else if is("TIMESTAMP") && zone(self, &["WITH", "LOCAL", "TIME", "ZONE"]) {
            push(self, &["WITH", "LOCAL", "TIME", "ZONE"]);
        }
        Ok(DataTypeKind::Primitive { name, args })
    }
}

/// The name of a quoted type identifier, unquoted.
fn quoted_name(text: &str) -> String {
    if text.starts_with('`') {
        unquote_ident(text)
    } else {
        text.get(1..text.len().saturating_sub(1))
            .unwrap_or_default()
            .replace("\"\"", "\"")
    }
}
