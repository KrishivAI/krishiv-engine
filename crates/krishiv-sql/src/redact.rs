// SPDX-License-Identifier: Apache-2.0
//! Query-text redaction from lexer tokens (ADR 0038 update).
//!
//! [`redact_sql`] walks the tokens (the lexer never fails, so broken text
//! is redacted too) and marks exactly:
//!
//! - in `OPTIONS (...)`, `WITH (...)` and `TBLPROPERTIES (...)` lists, the
//!   value of every key that [`is_sensitive_key`] classifies as secret,
//!   whatever its token kind (the rule of `OptionValue::from_user`), except
//!   a `secret('scope', 'key')` reference;
//! - anywhere, the credential parts of a URL in a string literal or a
//!   quoted identifier (`parquet.`s3://k:secret@b/p``), as `SecretUrl`
//!   splits them;
//! - comments, which are free text, through the pattern `redact` (defence
//!   in depth only).
//!
//! Nothing else is guessed. The result is a [`RedactedSql`], the only form
//! of query text that sinks (EXPLAIN, profiles, history, `system.*`, error
//! contexts) accept.

use std::ops::Range;

use krishiv_common::redact::{Mark, RedactedSql};
use krishiv_common::secret::{SecretUrl, UrlPart, is_sensitive_key};

use crate::lexer::{Sym, Token, TokenKind, is_keyword, tokenize, unescape_string, unquote_ident};

/// `sql` with its secrets replaced (see the module docs).
#[must_use]
pub fn redact_sql(sql: &str) -> RedactedSql {
    RedactedSql::from_marks_unchecked(sql, marks(sql))
}

/// The byte ranges of `sql` that [`redact_sql`] hides as secrets.
pub(crate) fn secret_ranges(sql: &str) -> Vec<Range<usize>> {
    marks(sql)
        .into_iter()
        .filter_map(|m| match m {
            Mark::Secret(r) => Some(r),
            Mark::FreeText(_) => None,
        })
        .collect()
}

/// What an option list is reading.
enum State {
    /// The key, parts so far, once per reading (see [`key_text`]).
    Key([String; 4]),
    /// The value of a key; `Some(start)` once a value token was seen.
    Value {
        sensitive: bool,
        start: Option<usize>,
        end: usize,
        reference: bool,
    },
}

fn marks(sql: &str) -> Vec<Mark> {
    let toks: Vec<Token> = tokenize(sql);
    let mut marks = Vec::new();
    let code: Vec<&Token> = toks
        .iter()
        .filter(|t| {
            let comment = matches!(t.kind, TokenKind::Comment | TokenKind::UnclosedComment);
            if comment {
                marks.push(Mark::FreeText(t.span.range()));
            }
            !comment
        })
        .collect();
    let text = |t: &Token| t.span.text(sql);
    for t in &code {
        url_marks(sql, t, &mut marks);
    }
    set_value(sql, &code, &mut marks);
    aes_key_marks(sql, &code, &mut marks);
    let mut depth = 0_usize;
    // The depth of the option list being read, and what it is reading.
    let mut list: Option<usize> = None;
    let mut state = State::Key(Default::default());
    let finish = |state: &State, marks: &mut Vec<Mark>| {
        if let State::Value {
            sensitive: true,
            start: Some(start),
            end,
            reference: false,
        } = *state
        {
            marks.push(Mark::Secret(start..end));
        }
    };
    for (i, t) in code.iter().enumerate() {
        let at_list = list == Some(depth);
        match t.kind {
            TokenKind::Sym(Sym::LParen) => {
                let opener = i
                    .checked_sub(1)
                    .and_then(|j| code.get(j))
                    .filter(|p| p.kind == TokenKind::Word)
                    .is_some_and(|p| {
                        ["OPTIONS", "WITH", "TBLPROPERTIES"]
                            .iter()
                            .any(|k| is_keyword(text(p), k))
                    });
                depth += 1;
                if list.is_none() && opener {
                    list = Some(depth);
                    state = State::Key(Default::default());
                    continue;
                }
            }
            TokenKind::Sym(Sym::RParen) if at_list => {
                finish(&state, &mut marks);
                list = None;
                depth = depth.saturating_sub(1);
                continue;
            }
            TokenKind::Sym(Sym::RParen) => depth = depth.saturating_sub(1),
            TokenKind::Sym(Sym::Comma) if at_list => {
                finish(&state, &mut marks);
                state = State::Key(Default::default());
                continue;
            }
            _ => {}
        }
        if list.is_none() {
            continue;
        }
        let key_part = matches!(t.kind, TokenKind::Word | TokenKind::QuotedIdent)
            || t.kind.is_string_like()
            || t.kind == TokenKind::Sym(Sym::Dot);
        match &mut state {
            State::Key(key) if at_list && t.kind == TokenKind::Sym(Sym::Eq) => {
                state = State::Value {
                    sensitive: sensitive(&key[..]),
                    start: None,
                    end: 0,
                    reference: false,
                };
            }
            State::Key(key) if at_list && key_part && !value_starts(&code, i, &key[0]) => {
                for (k, part) in key.iter_mut().zip(key_text(sql, t)) {
                    k.push_str(&part);
                }
            }
            State::Key(key) if at_list => {
                // A value written without `=`.
                state = State::Value {
                    sensitive: sensitive(&key[..]),
                    start: Some(t.span.start),
                    end: t.span.end,
                    reference: is_reference(sql, &code, i),
                };
            }
            State::Value {
                start,
                end,
                reference,
                ..
            } => {
                if start.is_none() {
                    *start = Some(t.span.start);
                    *reference = at_list && is_reference(sql, &code, i);
                }
                *end = t.span.end;
            }
            State::Key(_) => {}
        }
    }
    // An unterminated list: its last value runs to the end.
    finish(&state, &mut marks);
    marks
}

/// The key argument (the second) of `aes_encrypt`, `aes_decrypt` and
/// `try_aes_decrypt` calls, wherever they stand: a key is a secret in the
/// error context as much as in the error message. An unterminated call is
/// marked to the end (fail-safe).
fn aes_key_marks(sql: &str, code: &[&Token], marks: &mut Vec<Mark>) {
    for (i, t) in code.iter().enumerate() {
        if !matches!(
            t.kind,
            TokenKind::Word | TokenKind::QuotedIdent | TokenKind::DoubleQuoted
        ) {
            continue;
        }
        let name = if t.kind == TokenKind::QuotedIdent {
            unquote_ident(t.span.text(sql))
        } else if t.kind == TokenKind::DoubleQuoted {
            // `"aes_encrypt"(...)` under doubleQuotedIdentifiers.
            let raw = t.span.text(sql);
            raw.get(1..raw.len().saturating_sub(1))
                .unwrap_or_default()
                .replace("\"\"", "\"")
        } else {
            t.span.text(sql).to_owned()
        };
        let aes = ["aes_encrypt", "aes_decrypt", "try_aes_decrypt"]
            .iter()
            .any(|f| name.eq_ignore_ascii_case(f));
        if !aes
            || code
                .get(i + 1)
                .is_none_or(|n| n.kind != TokenKind::Sym(Sym::LParen))
        {
            continue;
        }
        let (mut depth, mut arg) = (0_usize, 0_usize);
        // Angle brackets of a complex type (`MAP<STRING,STRING>`) open only
        // after a type word, so a comparison `a < b` never hides a comma.
        let mut angle = 0_usize;
        let mut prev_word = String::new();
        let mut range: Option<(usize, usize)> = None;
        for (j, u) in code[i + 2..].iter().enumerate() {
            let word = std::mem::take(&mut prev_word);
            if u.kind == TokenKind::Word {
                prev_word = u.span.text(sql).to_owned();
            }
            // A `<` opens a type only after a type word and before a type
            // (a word that is a type name, or a `field:` of a struct), so
            // `map < 1` and `array < col` stay comparisons.
            let opens_type = ["map", "array", "struct"]
                .iter()
                .any(|w| word.eq_ignore_ascii_case(w))
                && code[i + 2..].get(j + 1).is_some_and(|n| {
                    let after = code[i + 2..].get(j + 2);
                    match n.kind {
                        TokenKind::Word => {
                            is_type_word(n.span.text(sql))
                                || after.is_some_and(|m| {
                                    m.kind == TokenKind::Sym(Sym::Colon)
                                        // `field TYPE`: the colon may be omitted.
                                        || (m.kind == TokenKind::Word
                                            && is_type_word(m.span.text(sql)))
                                })
                        }
                        // A quoted struct field: `STRUCT<`a`: INT>` or
                        // `STRUCT<`a` INT>`.
                        TokenKind::QuotedIdent | TokenKind::DoubleQuoted => {
                            after.is_some_and(|m| {
                                m.kind == TokenKind::Sym(Sym::Colon)
                                    || (m.kind == TokenKind::Word && is_type_word(m.span.text(sql)))
                            })
                        }
                        _ => false,
                    }
                });
            match u.kind {
                TokenKind::Sym(Sym::Lt) if angle > 0 || opens_type => {
                    angle += 1;
                }
                TokenKind::Sym(Sym::Gt) if angle > 0 => angle -= 1,
                TokenKind::Sym(Sym::Comma) if angle > 0 => {}
                TokenKind::Sym(Sym::LParen) => depth += 1,
                TokenKind::Sym(Sym::RParen) if depth == 0 => break,
                TokenKind::Sym(Sym::RParen) => depth = depth.saturating_sub(1),
                TokenKind::Sym(Sym::Comma) if depth == 0 => {
                    arg += 1;
                    if arg > 1 {
                        break;
                    }
                    continue;
                }
                _ => {}
            }
            if arg == 1 {
                let r = range.get_or_insert((u.span.start, u.span.end));
                r.1 = u.span.end;
            }
        }
        if let Some((a, b)) = range {
            marks.push(Mark::Secret(a..b));
        }
    }
}

/// Whether `w` names a SQL data type (the start of `MAP<STRING, INT>`).
fn is_type_word(w: &str) -> bool {
    [
        "map",
        "array",
        "struct",
        "string",
        "varchar",
        "char",
        "int",
        "integer",
        "bigint",
        "smallint",
        "tinyint",
        "long",
        "short",
        "byte",
        "float",
        "double",
        "real",
        "decimal",
        "dec",
        "numeric",
        "boolean",
        "binary",
        "date",
        "timestamp",
        "timestamp_ntz",
        "timestamp_ltz",
        "interval",
        "variant",
        "time",
        "void",
    ]
    .iter()
    .any(|t| w.eq_ignore_ascii_case(t))
}

/// `SET key = value` with a sensitive key: the value (raw text to the end
/// of the statement) is a secret.
fn set_value(sql: &str, code: &[&Token], marks: &mut Vec<Mark>) {
    let Some(first) = code.first() else {
        return;
    };
    if first.kind != TokenKind::Word || !is_keyword(first.span.text(sql), "SET") {
        return;
    }
    let Some(eq) = code.iter().position(|t| t.kind == TokenKind::Sym(Sym::Eq)) else {
        return;
    };
    let mut key: [String; 4] = Default::default();
    for t in &code[1..eq] {
        for (k, part) in key.iter_mut().zip(key_text(sql, t)) {
            k.push_str(&part);
        }
    }
    let end = code
        .iter()
        .rposition(|t| t.kind != TokenKind::Sym(Sym::Semi))
        .map_or(sql.len(), |j| code[j].span.end);
    let start = code.get(eq + 1).map_or(end, |t| t.span.start);
    if sensitive(&key[..]) && start < end {
        marks.push(Mark::Secret(start..end));
    }
}

/// Whether any reading of a key names a secret, its escapes decoded or not
/// (fail-safe).
fn sensitive(readings: &[String]) -> bool {
    readings.iter().any(|k| is_sensitive_key(k))
}

/// A key part's text, once per reading: identifiers unquoted, strings
/// unescaped with and without escapes and with and without `consecutive`
/// (so an escaped key cannot hide). `redact_sql` has no parser options (it
/// works from tokens alone), so the readings stay separate and a key is
/// sensitive when any complete reading is, fail-safe in either mode.
fn key_text(sql: &str, t: &Token) -> [String; 4] {
    let raw = t.span.text(sql);
    match t.kind {
        TokenKind::QuotedIdent => std::array::from_fn(|_| unquote_ident(raw)),
        TokenKind::Sym(Sym::Dot) => std::array::from_fn(|_| ".".to_owned()),
        k if k.is_string_like() => [
            unescape_string(raw, true, false),
            unescape_string(raw, true, true),
            unescape_string(raw, false, false),
            unescape_string(raw, false, true),
        ],
        _ => std::array::from_fn(|_| raw.to_owned()),
    }
}

/// Whether token `i` starts a value written without `=`: any token but a
/// dot right after a complete key (`password 'x'`, `password x`).
fn value_starts(code: &[&Token], i: usize, key: &str) -> bool {
    let dot = |j: usize| code[j].kind == TokenKind::Sym(Sym::Dot);
    !key.is_empty() && !dot(i) && i.checked_sub(1).is_some_and(|j| !dot(j))
}

/// Whether the value at token `i` is a `secret('scope', 'key')` reference.
fn is_reference(sql: &str, code: &[&Token], i: usize) -> bool {
    matches!(code[i].kind, TokenKind::Word | TokenKind::QuotedIdent)
        && is_keyword(code[i].span.text(sql), "SECRET")
        && code
            .get(i + 1)
            .is_some_and(|n| n.kind == TokenKind::Sym(Sym::LParen))
}

/// The range of a token's content: inside its quotes for strings and
/// quoted identifiers, the whole token otherwise.
fn content(sql: &str, t: &Token) -> Range<usize> {
    let r = t.span.range();
    if !(t.kind.is_string_like() || t.kind == TokenKind::QuotedIdent) {
        return r;
    }
    let text = &sql[r.clone()];
    let (lead, trail) = match text.as_bytes() {
        [b'R' | b'r', b'\'' | b'"', ..] => (2, 1),
        [b'$', rest @ ..] => {
            let tag = rest.iter().position(|&b| b == b'$').map_or(1, |p| p + 2);
            (tag, tag)
        }
        _ => (1, 1),
    };
    let trail = if t.kind == TokenKind::Unterminated {
        0
    } else {
        trail
    };
    if lead + trail > text.len() {
        return r.start..r.start;
    }
    r.start + lead..r.end - trail
}

/// Marks the credentials of a URL in a string literal or quoted identifier.
fn url_marks(sql: &str, t: &Token, marks: &mut Vec<Mark>) {
    if !(t.kind.is_string_like() || t.kind == TokenKind::QuotedIdent) {
        return;
    }
    let r = content(sql, t);
    let value = &sql[r.clone()];
    let escaped = value.contains('\\')
        || value.contains("''")
        || value.contains("\"\"")
        || value.contains("``");
    if escaped {
        // Positions in the value are not positions in the text, and an
        // escape (`\u0040`, a doubled quote) can make the decoded value a
        // URL the written one is not: hide all when any reading is a URL.
        // No parser options here (see `key_text`): every escapes and
        // `consecutive` reading counts.
        let raw = t.span.text(sql);
        let readings = match t.kind {
            TokenKind::QuotedIdent => vec![unquote_ident(raw)],
            _ => vec![
                unescape_string(raw, true, false),
                unescape_string(raw, true, true),
                unescape_string(raw, false, false),
                unescape_string(raw, false, true),
            ],
        };
        if SecretUrl::parse(value).is_some()
            || readings.iter().any(|d| SecretUrl::parse(d).is_some())
        {
            marks.push(Mark::Secret(r));
        }
        return;
    }
    let Some(url) = SecretUrl::parse(value) else {
        return;
    };
    let mut at = r.start;
    for part in url.parts() {
        let len = match part {
            UrlPart::Text(s) => s.len(),
            UrlPart::Secret(s) => {
                let len = s.expose_secret().len();
                marks.push(Mark::Secret(at..at + len));
                len
            }
        };
        at += len;
    }
}
