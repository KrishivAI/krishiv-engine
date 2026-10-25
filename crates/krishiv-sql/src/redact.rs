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
    /// The key, parts so far.
    Key(String),
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
    let mut depth = 0_usize;
    // The depth of the option list being read, and what it is reading.
    let mut list: Option<usize> = None;
    let mut state = State::Key(String::new());
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
                    state = State::Key(String::new());
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
                state = State::Key(String::new());
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
                    sensitive: sensitive(key),
                    start: None,
                    end: 0,
                    reference: false,
                };
            }
            State::Key(key) if at_list && key_part && !value_starts(&code, i, key) => {
                key.push_str(&key_text(sql, t));
            }
            State::Key(key) if at_list => {
                // A value written without `=`.
                state = State::Value {
                    sensitive: sensitive(key),
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
    let key: String = code[1..eq].iter().map(|t| key_text(sql, t)).collect();
    let end = code
        .iter()
        .rposition(|t| t.kind != TokenKind::Sym(Sym::Semi))
        .map_or(sql.len(), |j| code[j].span.end);
    let start = code.get(eq + 1).map_or(end, |t| t.span.start);
    if sensitive(&key) && start < end {
        marks.push(Mark::Secret(start..end));
    }
}

/// Whether a key names a secret, its escapes decoded or not (fail-safe).
fn sensitive(key: &str) -> bool {
    is_sensitive_key(key)
}

/// A key part's text: identifiers unquoted, strings unescaped (both the
/// decoded and the written form, so an escaped key cannot hide).
fn key_text(sql: &str, t: &Token) -> String {
    let raw = t.span.text(sql);
    match t.kind {
        TokenKind::QuotedIdent => unquote_ident(raw),
        TokenKind::Sym(Sym::Dot) => ".".to_owned(),
        k if k.is_string_like() => {
            let decoded = unescape_string(raw, true, false);
            let plain = unescape_string(raw, false, false);
            if decoded == plain {
                decoded
            } else {
                format!("{decoded}{plain}")
            }
        }
        _ => raw.to_owned(),
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
        let raw = t.span.text(sql);
        let readings = match t.kind {
            TokenKind::QuotedIdent => vec![unquote_ident(raw)],
            _ => vec![
                unescape_string(raw, true, false),
                unescape_string(raw, false, false),
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
