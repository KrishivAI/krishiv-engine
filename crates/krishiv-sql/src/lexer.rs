// SPDX-License-Identifier: Apache-2.0
//! The Spark SQL lexer (`SqlBaseLexer.g4`, Spark 4.2).
//!
//! [`tokenize`] never fails and loses nothing but whitespace: every
//! character of the input is in exactly one token, comments included
//! ([`TokenKind::Comment`]), and malformed input becomes typed tokens
//! ([`TokenKind::Unterminated`] for a quote without its closing quote,
//! [`TokenKind::UnclosedComment`] for `/*` without `*/`), each running to the
//! end of the input. So a redactor can mask every literal, even in a broken
//! statement, from the tokens alone. Tokens borrow nothing: each is a kind
//! plus the [`Span`] of its source text. Keywords are lexed as
//! [`TokenKind::Word`]; whether a word is a keyword depends on where it
//! appears ([`is_keyword`] compares one the way Spark does).
//!
//! Spark's lexing rules that are reproduced here: case-insensitive numeric
//! suffixes and longest-match against identifiers (`10L` is a literal, `10Lx`
//! and `1a` are identifiers), the `isValidDecimal` check (a decimal must not
//! be followed by a letter, digit or `_`), identifiers of `\p{L}` letters,
//! ASCII digits and `_`, `--` comments continued by a backslash-newline,
//! nested `/* */` comments, `/*+` hint starts and `*/` hint ends, URI-path
//! identifiers (`file://b/k`; a digit in the scheme, as in `s3`, disables
//! them), raw strings `R'...'`, dollar-quoted strings `$tag$...$tag$`, and
//! the complex-type counter that makes `>>` two `>` tokens after `ARRAY`,
//! `MAP` or `STRUCT`. Spark lexes an unterminated quote as one unrecognized
//! character; the parser reports `PARSE_SYNTAX_ERROR` at that character for
//! an [`TokenKind::Unterminated`] token, so errors are the same.

use crate::ast::{NumKind, Span};
use crate::generated::letters::LETTERS;

/// A punctuation token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(missing_docs, reason = "each variant is the symbol named")]
pub enum Sym {
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Dot,
    Semi,
    Star,
    Plus,
    Minus,
    Slash,
    Percent,
    Tilde,
    Amp,
    Pipe,
    Concat,
    PipeGt,
    Caret,
    Eq,
    NullSafeEq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
    Shl,
    Shr,
    Shru,
    Colon,
    DoubleColon,
    Arrow,
    FatArrow,
    Question,
    Bang,
}

/// What a token is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TokenKind {
    /// An unquoted identifier or keyword.
    Word,
    /// A backquoted identifier.
    QuotedIdent,
    /// A single-quoted or raw (`R'..'`) string literal.
    String,
    /// A double-quoted string: a string literal, or an identifier when
    /// `spark.sql.ansi.doubleQuotedIdentifiers` is on.
    DoubleQuoted,
    /// A dollar-quoted string, `$tag$...$tag$`.
    DollarString,
    /// A quote (`'`, `"`, `` ` ``, `$tag$`) without its closing quote,
    /// running to the end of the input.
    Unterminated,
    /// A numeric literal.
    Number(NumKind),
    /// `/*+`, the start of a hint.
    HintStart,
    /// `*/`, the end of a hint.
    HintEnd,
    /// Punctuation.
    Sym(Sym),
    /// A `--` or `/* */` comment.
    Comment,
    /// A `/*` comment without its `*/`, running to the end of the input.
    UnclosedComment,
    /// A character no Spark token starts with.
    Unrecognized,
}

impl TokenKind {
    /// Whether the token is a comment, which the grammar ignores.
    #[must_use]
    pub const fn is_trivia(self) -> bool {
        matches!(self, Self::Comment)
    }

    /// Whether the token holds a string literal's text (complete or not),
    /// which error messages and logs must not echo.
    #[must_use]
    pub const fn is_string_like(self) -> bool {
        matches!(
            self,
            Self::String | Self::DoubleQuoted | Self::DollarString | Self::Unterminated
        )
    }
}

/// A token and its source text.
#[derive(Debug, Clone, Copy, Eq)]
pub struct Token {
    /// The kind.
    pub kind: TokenKind,
    /// Where it is.
    pub span: Span,
}

impl PartialEq for Token {
    /// Kind and position ([`Span`] equality alone ignores positions).
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.span.range() == other.span.range()
    }
}

/// Whether `word` is the keyword `kw` (upper-case), folding case per
/// character the way Spark's case-insensitive stream does (Java
/// `Character.toUpperCase`, so `ı` is `I`).
#[must_use]
pub fn is_keyword(word: &str, kw: &str) -> bool {
    crate::keywords::is_kw(word, kw)
}

/// Spark's `WS` rule.
fn is_space(c: char) -> bool {
    matches!(
        c,
        ' ' | '\t' | '\n' | '\u{0C}' | '\r' | '\u{0B}' | '\u{A0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200A}' | '\u{2028}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
    )
}

/// `UNICODE_LETTER` (`\p{L}`).
fn is_letter(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    let c = u32::from(c);
    LETTERS
        .binary_search_by(|&(lo, hi)| {
            if hi < c {
                std::cmp::Ordering::Less
            } else if lo > c {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// `UNICODE_LETTER | DIGIT | '_'` (digits are ASCII only).
fn is_ident_char(c: char) -> bool {
    is_letter(c) || c.is_ascii_digit() || c == '_'
}

/// Characters of the URI part of an `IDENTIFIER` (`scheme://...`).
fn is_uri_char(c: char) -> bool {
    is_ident_char(c) || "/-.?=&#%".contains(c)
}

/// Lexes `src` into tokens; only whitespace is dropped.
#[must_use]
pub fn tokenize(src: &str) -> Vec<Token> {
    let mut lexer = Lexer {
        src,
        pos: 0,
        tokens: Vec::with_capacity(src.len() / 4 + 1),
        complex_level: 0,
    };
    lexer.run();
    lexer.tokens
}

struct Lexer<'a> {
    src: &'a str,
    pos: usize,
    tokens: Vec<Token>,
    /// Spark's `complex_type_level_counter`.
    complex_level: u32,
}

impl Lexer<'_> {
    fn rest(&self) -> &str {
        self.src.get(self.pos..).unwrap_or_default()
    }

    fn byte(&self, at: usize) -> u8 {
        self.src.as_bytes().get(at).copied().unwrap_or(0)
    }

    fn push(&mut self, kind: TokenKind, len: usize) {
        let span = Span::new(self.pos, self.pos + len);
        self.tokens.push(Token { kind, span });
        self.pos += len;
    }

    /// Length of the longest prefix of `rest()[from..]` satisfying `pred`.
    fn run_len(&self, from: usize, pred: impl Fn(char) -> bool) -> usize {
        let tail = self.rest().get(from..).unwrap_or_default();
        tail.find(|c| !pred(c)).unwrap_or(tail.len())
    }

    fn run(&mut self) {
        while let Some(c) = self.rest().chars().next() {
            let next = self.byte(self.pos + 1);
            match c {
                c if is_space(c) => self.pos += c.len_utf8(),
                '-' if next == b'-' => self.line_comment(),
                '/' if next == b'*' && self.byte(self.pos + 2) == b'+' => {
                    self.push(TokenKind::HintStart, 3);
                }
                '/' if next == b'*' => self.block_comment(),
                '*' if next == b'/' => self.push(TokenKind::HintEnd, 2),
                '\'' => {
                    self.quoted(b'\'', 0, TokenKind::String, true);
                }
                '"' => {
                    self.quoted(b'"', 0, TokenKind::DoubleQuoted, true);
                }
                '`' => {
                    self.quoted(b'`', 0, TokenKind::QuotedIdent, false);
                }
                'r' | 'R' if matches!(next, b'\'' | b'"') => {
                    // Unterminated, `r` is an identifier and the quote
                    // starts an ordinary string, as Spark lexes it.
                    if !self.quoted(next, 1, TokenKind::String, false) {
                        self.push(TokenKind::Word, 1);
                    }
                }
                '$' => self.dollar(),
                '0'..='9' => self.number(),
                '.' if next.is_ascii_digit() => self.number(),
                c if is_ident_char(c) => self.word(),
                _ => self.symbol(c),
            }
        }
    }

    /// The rest of the input as one token.
    fn push_rest(&mut self, kind: TokenKind) {
        let len = self.src.len() - self.pos;
        self.push(kind, len);
    }

    /// `--` up to the end of the line; `\` before the newline continues it.
    fn line_comment(&mut self) {
        let bytes = self.src.as_bytes();
        let mut i = self.pos + 2;
        while let Some(&b) = bytes.get(i) {
            if b == b'\\' && bytes.get(i + 1) == Some(&b'\n') {
                i += 2;
            } else if b == b'\n' || b == b'\r' {
                break;
            } else {
                i += 1;
            }
        }
        self.push(TokenKind::Comment, i - self.pos);
    }

    /// A nested `/* */` comment; an inner `/*+` does not nest.
    fn block_comment(&mut self) {
        let bytes = self.src.as_bytes();
        let (mut i, mut depth) = (self.pos + 2, 1_u32);
        while depth > 0 {
            match (bytes.get(i), bytes.get(i + 1)) {
                (None, _) => return self.push_rest(TokenKind::UnclosedComment),
                (Some(b'/'), Some(b'*')) if bytes.get(i + 2) != Some(&b'+') => {
                    depth += 1;
                    i += 2;
                }
                (Some(b'*'), Some(b'/')) => {
                    depth -= 1;
                    i += 2;
                }
                _ => i += 1,
            }
        }
        self.push(TokenKind::Comment, i - self.pos);
    }

    /// A quoted token starting `prefix` bytes after `pos`. Backslash escapes
    /// (when `escapes`) and doubled quotes (except in raw strings) stay
    /// inside it. Unterminated, a raw string pushes nothing and returns
    /// `false`; any other quote runs to the end of the input.
    fn quoted(&mut self, quote: u8, prefix: usize, kind: TokenKind, escapes: bool) -> bool {
        let doubling = escapes || quote == b'`';
        let bytes = self.src.as_bytes();
        let mut i = self.pos + prefix + 1;
        loop {
            match bytes.get(i) {
                None if prefix > 0 => return false,
                None => {
                    self.push_rest(TokenKind::Unterminated);
                    return true;
                }
                Some(&b) if b == quote && doubling && bytes.get(i + 1) == Some(&quote) => i += 2,
                Some(&b) if b == quote => {
                    self.push(kind, i + 1 - self.pos);
                    return true;
                }
                Some(b'\\') if escapes => i += 2,
                Some(_) => i += 1,
            }
        }
    }

    /// `$tag$ ... $tag$` (`tag` is ASCII letters, maybe none); a lone `$`
    /// is unrecognized.
    fn dollar(&mut self) {
        let letters = self.run_len(1, |c| c.is_ascii_alphabetic());
        if self.byte(self.pos + 1 + letters) != b'$' {
            return self.push(TokenKind::Unrecognized, 1);
        }
        let tag_len = letters + 2;
        let tag = self.rest().get(..tag_len).unwrap_or_default();
        match self.rest().get(tag_len..).and_then(|body| body.find(tag)) {
            Some(end) => self.push(TokenKind::DollarString, tag_len + end + tag_len),
            None => self.push_rest(TokenKind::Unterminated),
        }
    }

    fn word(&mut self) {
        let mut len = self.run_len(0, is_ident_char);
        let word = self.rest().get(..len).unwrap_or_default();
        if word.chars().all(is_letter)
            && self.rest().get(len..).is_some_and(|r| r.starts_with("://"))
        {
            let uri = self.run_len(len + 3, is_uri_char);
            if uri > 0 {
                len += 3 + uri;
            }
        }
        let word = self.rest().get(..len).unwrap_or_default();
        if ["ARRAY", "MAP", "STRUCT"]
            .iter()
            .any(|k| k.eq_ignore_ascii_case(word))
        {
            self.complex_level += 1;
        }
        self.push(TokenKind::Word, len);
    }

    /// Spark's numeric tokens, by longest match against `IDENTIFIER`.
    fn number(&mut self) {
        let digits = self.run_len(0, |c| c.is_ascii_digit());
        let after_ident = self.run_len(0, is_ident_char);
        if self.byte(self.pos + digits) == b'.' {
            let frac = self.run_len(digits + 1, |c| c.is_ascii_digit());
            if digits + frac > 0
                && let Some((len, kind)) = self.number_tail(digits + 1 + frac, true)
            {
                return self.push(TokenKind::Number(kind), len);
            }
            // `isValidDecimal` failed: Spark falls back to the integer part.
            if digits == 0 {
                return self.push(TokenKind::Sym(Sym::Dot), 1);
            }
            return self.push(TokenKind::Number(NumKind::Int), digits);
        }
        match self.number_tail(digits, false) {
            Some((len, kind)) if len >= after_ident => self.push(TokenKind::Number(kind), len),
            _ => self.push(TokenKind::Word, after_ident),
        }
    }

    /// The exponent and type suffix after `len` bytes of digits; `None`
    /// when a decimal (`dotted`) is followed by a letter, digit or `_`.
    fn number_tail(&self, mut len: usize, dotted: bool) -> Option<(usize, NumKind)> {
        let mut kind = if dotted {
            NumKind::Decimal
        } else {
            NumKind::Int
        };
        if matches!(self.byte(self.pos + len), b'e' | b'E') {
            let sign = usize::from(matches!(self.byte(self.pos + len + 1), b'+' | b'-'));
            let exp = self.run_len(len + 1 + sign, |c| c.is_ascii_digit());
            if exp > 0 {
                len += 1 + sign + exp;
                kind = NumKind::Exponent;
            }
        }
        let suffix = |s: &str| {
            self.rest()
                .get(len..len + s.len())
                .is_some_and(|t| t.eq_ignore_ascii_case(s))
        };
        let plain = kind != NumKind::Exponent && !dotted;
        let (kind, len) = if suffix("BD") {
            (NumKind::BigDecimal, len + 2)
        } else if suffix("D") {
            (NumKind::Double, len + 1)
        } else if suffix("F") {
            (NumKind::Float, len + 1)
        } else if plain && suffix("L") {
            (NumKind::Long, len + 1)
        } else if plain && suffix("S") {
            (NumKind::Short, len + 1)
        } else if plain && suffix("Y") {
            (NumKind::Byte, len + 1)
        } else {
            (kind, len)
        };
        let next = self.byte(self.pos + len);
        let valid = !(next.is_ascii_alphanumeric() || next == b'_');
        (valid || !dotted).then_some((len, kind))
    }

    fn symbol(&mut self, c: char) {
        use Sym::{
            Amp, Arrow, Bang, Caret, Colon, Comma, Concat, Dot, DoubleColon, Eq, FatArrow, Ge, Gt,
            LBracket, LParen, Le, Lt, Minus, Neq, NullSafeEq, Percent, Pipe, PipeGt, Plus,
            Question, RBracket, RParen, Semi, Shl, Shr, Shru, Slash, Star, Tilde,
        };
        let (b1, b2) = (self.byte(self.pos + 1), self.byte(self.pos + 2));
        let shift_ok = self.complex_level == 0;
        let (sym, len) = match (c, b1, b2) {
            ('(', ..) => (LParen, 1),
            (')', ..) => (RParen, 1),
            ('[', ..) => (LBracket, 1),
            (']', ..) => (RBracket, 1),
            (',', ..) => (Comma, 1),
            ('.', ..) => (Dot, 1),
            (';', ..) => (Semi, 1),
            ('*', ..) => (Star, 1),
            ('+', ..) => (Plus, 1),
            ('-', b'>', _) => (Arrow, 2),
            ('-', ..) => (Minus, 1),
            ('/', ..) => (Slash, 1),
            ('%', ..) => (Percent, 1),
            ('~', ..) => (Tilde, 1),
            ('&', ..) => (Amp, 1),
            ('|', b'|', _) => (Concat, 2),
            ('|', b'>', _) => (PipeGt, 2),
            ('|', ..) => (Pipe, 1),
            ('^', ..) => (Caret, 1),
            ('=', b'=', _) => (Eq, 2),
            ('=', b'>', _) => (FatArrow, 2),
            ('=', ..) => (Eq, 1),
            ('!', b'=', _) | ('<', b'>', _) => (Neq, 2),
            ('!', b'>', _) => (Le, 2),
            ('!', b'<', _) | ('>', b'=', _) => (Ge, 2),
            ('!', ..) => (Bang, 1),
            ('<', b'=', b'>') => (NullSafeEq, 3),
            ('<', b'=', _) => (Le, 2),
            ('<', b'<', _) => (Shl, 2),
            ('<', ..) => (Lt, 1),
            ('>', b'>', b'>') if shift_ok => (Shru, 3),
            ('>', b'>', _) if shift_ok => (Shr, 2),
            ('>', ..) => (Gt, 1),
            (':', b':', _) => (DoubleColon, 2),
            (':', ..) => (Colon, 1),
            ('?', ..) => (Question, 1),
            _ => return self.push(TokenKind::Unrecognized, c.len_utf8()),
        };
        match sym {
            Gt => self.complex_level = self.complex_level.saturating_sub(1),
            // `STRUCT<>`: the parser's `decComplexTypeLevelCounter()` action.
            Neq if self.tokens.last().is_some_and(|t| {
                t.kind == TokenKind::Word && t.span.text(self.src).eq_ignore_ascii_case("STRUCT")
            }) =>
            {
                self.complex_level = self.complex_level.saturating_sub(1);
            }
            _ => {}
        }
        self.push(TokenKind::Sym(sym), len);
    }
}

/// Decodes a [`TokenKind::String`] or [`TokenKind::DoubleQuoted`] token the
/// way Spark's `unescapeSQLString` does (`escapes` false keeps backslashes,
/// as with `spark.sql.parser.escapedStringLiterals`).
pub(crate) fn unescape_string(raw: &str, escapes: bool) -> String {
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
    if !escapes {
        return body.iter().collect();
    }
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
        } else if let Some(v) = octal(&body, i + 1) {
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
    // Java strings are UTF-16: `😀` is one character; a lone
    // surrogate cannot be represented and becomes U+FFFD.
    String::from_utf16_lossy(&out)
}

/// `\[01][0-7][0-7]` at `at`.
fn octal(body: &[char], at: usize) -> Option<u16> {
    let d = |i: usize| body.get(at + i).and_then(|c| c.to_digit(8));
    let (a, b, c) = (d(0)?, d(1)?, d(2)?);
    (a <= 1).then(|| u16::try_from(a * 64 + b * 8 + c).unwrap_or(0xFFFD))
}

/// Decodes a backquoted identifier (`` `a``b` `` is ``a`b``).
pub(crate) fn unquote_ident(raw: &str) -> String {
    raw.get(1..raw.len().saturating_sub(1))
        .unwrap_or_default()
        .replace("``", "`")
}
