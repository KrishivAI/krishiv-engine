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

use crate::lexer::{Sym, Token, TokenKind, is_keyword, tokenize};

/// `sql` with its secrets replaced (see the module docs).
#[must_use]
pub fn redact_sql(sql: &str) -> RedactedSql {
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
    let mut depth = 0_usize;
    // The depth inside an option list, the key read so far, whether a
    // value comes next, and the depth of a `secret(...)` to skip.
    let mut list: Option<usize> = None;
    let mut key = String::new();
    let mut skip: Option<usize> = None;
    for (i, t) in code.iter().enumerate() {
        url_marks(sql, t, &mut marks);
        match t.kind {
            TokenKind::Sym(Sym::LParen) => {
                depth += 1;
                let opener = i
                    .checked_sub(1)
                    .and_then(|j| code.get(j))
                    .filter(|p| p.kind == TokenKind::Word)
                    .is_some_and(|p| {
                        ["OPTIONS", "WITH", "TBLPROPERTIES"]
                            .iter()
                            .any(|k| is_keyword(text(p), k))
                    });
                if list.is_none() && opener {
                    list = Some(depth);
                    key.clear();
                }
                continue;
            }
            TokenKind::Sym(Sym::RParen) => {
                if list == Some(depth) {
                    list = None;
                }
                if skip == Some(depth) {
                    skip = None;
                }
                depth = depth.saturating_sub(1);
                continue;
            }
            _ => {}
        }
        if list != Some(depth) || skip.is_some() {
            continue;
        }
        match t.kind {
            TokenKind::Sym(Sym::Comma) => key.clear(),
            TokenKind::Sym(Sym::Eq) => {}
            TokenKind::Sym(Sym::Dot) => key.push('.'),
            TokenKind::Word | TokenKind::QuotedIdent
                if is_keyword(text(t), "SECRET")
                    && code
                        .get(i + 1)
                        .is_some_and(|n| n.kind == TokenKind::Sym(Sym::LParen)) =>
            {
                skip = Some(depth + 1);
            }
            _ if value_follows(&code, i, &key) => {
                if is_sensitive_key(&key) {
                    marks.push(Mark::Secret(content(sql, t)));
                }
                key.clear();
            }
            TokenKind::Word | TokenKind::QuotedIdent => key.push_str(&unquote(sql, t)),
            _ if t.kind.is_string_like() => key.push_str(&unquote(sql, t)),
            _ => {}
        }
    }
    RedactedSql::from_marks(sql, marks)
}

/// Whether token `i` is an option's value: the key is complete and `i`
/// follows `=`, or is a literal after the key.
fn value_follows(code: &[&Token], i: usize, key: &str) -> bool {
    if key.is_empty() {
        return false;
    }
    let after_eq = i
        .checked_sub(1)
        .is_some_and(|j| code[j].kind == TokenKind::Sym(Sym::Eq));
    let literal = matches!(code[i].kind, TokenKind::Number(_)) || code[i].kind.is_string_like();
    let key_ended = i
        .checked_sub(1)
        .is_some_and(|j| !matches!(code[j].kind, TokenKind::Sym(Sym::Dot)));
    after_eq || (literal && key_ended)
}

/// The quoted text of a string or identifier token, without its quotes.
fn unquote(sql: &str, t: &Token) -> String {
    sql[content(sql, t)].to_owned()
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
    let Some(url) = SecretUrl::parse(value) else {
        return;
    };
    let escaped = value.contains('\\') || value.contains("''") || value.contains("``");
    if escaped {
        // Positions in the value are not positions in the text: hide all.
        marks.push(Mark::Secret(r));
        return;
    }
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
