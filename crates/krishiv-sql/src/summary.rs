// SPDX-License-Identifier: Apache-2.0
//! Query text for telemetry (ADR 0035 section 3, P2-T116): the sanitized
//! text of OpenTelemetry's `db.query.text` and the low-cardinality
//! `db.query.summary`.
//!
//! Both read the lexer's tokens, which never fail, so broken text is
//! handled too. Neither echoes a literal: [`sanitize_query_text`] replaces
//! every string and number with `?`, drops comments and collapses white
//! space; [`query_summary`] keeps only the operation and the names of the
//! tables the statement reads or writes.

use crate::lexer::{Sym, Token, TokenKind, tokenize, unquote_ident};

/// The most table names a summary lists.
const MAX_TARGETS: usize = 5;
/// The longest sanitized text, in bytes (cut at a token boundary).
const MAX_TEXT: usize = 4096;

/// `sql` with every literal replaced by `?`, comments removed and white
/// space collapsed to one space: safe to attach to a span or a log record.
///
/// Quoted identifiers stay (they are names). A double-quoted string may be
/// either a literal or, with `spark.sql.ansi.doubleQuotedIdentifiers`, an
/// identifier; the text does not know the setting, so it is treated as a
/// literal. The result is cut at 4 KiB (`MAX_TEXT`) with a trailing `...`.
#[must_use]
pub fn sanitize_query_text(sql: &str) -> String {
    sanitized(sql, MAX_TEXT)
}

/// [`sanitize_query_text`] without the cut: every string and number literal is `?` and comments
/// are dropped, whatever the length (`system.queries.query_text` applies its own limit).
#[must_use]
pub fn mask_literals(sql: &str) -> String {
    sanitized(sql, usize::MAX)
}

fn sanitized(sql: &str, limit: usize) -> String {
    let mut out = String::with_capacity(sql.len().min(limit));
    let mut pending_space = false;
    for t in tokenize(sql) {
        let text = t.span.text(sql);
        let piece = match t.kind {
            TokenKind::Comment | TokenKind::UnclosedComment => {
                pending_space = true;
                continue;
            }
            kind if kind.is_string_like() => "?",
            TokenKind::Number(_) => "?",
            _ => text,
        };
        if piece.trim().is_empty() {
            pending_space = true;
            continue;
        }
        if out.len().saturating_add(piece.len()).saturating_add(1) > limit {
            out.push_str("...");
            return out;
        }
        if pending_space && !out.is_empty() {
            out.push(' ');
        }
        out.push_str(piece);
        pending_space = true;
    }
    out
}

/// OpenTelemetry's `db.query.summary`: the operation and the tables, e.g.
/// `SELECT orders customers`. Empty text gives an empty summary.
///
/// The operation is the statement's first word, upper-cased (`WITH` looks
/// through to the statement it introduces). The tables are the names after
/// `FROM`, `JOIN`, `INTO`, `UPDATE` and `TABLE`, in order of appearance
/// without repeats, at most five (`MAX_TARGETS`); a subquery's own tables count,
/// and a name keeps its qualifier (`db.t`).
#[must_use]
pub fn query_summary(sql: &str) -> String {
    let toks: Vec<Token> = tokenize(sql)
        .into_iter()
        .filter(|t| !t.kind.is_trivia() && t.kind != TokenKind::UnclosedComment)
        .collect();
    let word = |t: &Token| -> Option<String> {
        match t.kind {
            TokenKind::Word => Some(t.span.text(sql).to_owned()),
            TokenKind::QuotedIdent => Some(unquote_ident(t.span.text(sql))),
            _ => None,
        }
    };
    let mut operation = String::new();
    let mut targets: Vec<String> = Vec::new();
    // `WITH` and `EXPLAIN` introduce the statement; the operation is the
    // first statement keyword outside parentheses (a CTE name is not one).
    let mut leading: Option<String> = None;
    let mut depth = 0_usize;
    let mut i = 0;
    while i < toks.len() {
        let t = &toks[i];
        match t.kind {
            TokenKind::Sym(Sym::LParen) => depth += 1,
            TokenKind::Sym(Sym::RParen) => depth = depth.saturating_sub(1),
            _ => {}
        }
        if t.kind == TokenKind::Word {
            let upper = t.span.text(sql).to_ascii_uppercase();
            if operation.is_empty() && depth == 0 {
                if leading.is_none() {
                    leading = Some(upper.clone());
                }
                let introducer = matches!(leading.as_deref(), Some("WITH" | "EXPLAIN"));
                let statement = matches!(
                    upper.as_str(),
                    "SELECT"
                        | "INSERT"
                        | "UPDATE"
                        | "DELETE"
                        | "MERGE"
                        | "VALUES"
                        | "FROM"
                        | "TABLE"
                );
                if !introducer || statement {
                    operation = upper.clone();
                }
            }
            if matches!(
                upper.as_str(),
                "FROM" | "JOIN" | "INTO" | "UPDATE" | "TABLE"
            ) {
                // A (possibly qualified) name; `(` starts a subquery and
                // `IF NOT EXISTS`/`OVERWRITE` style words are skipped.
                let mut j = i + 1;
                while let Some(n) = toks.get(j) {
                    let w = n.span.text(sql).to_ascii_uppercase();
                    if n.kind == TokenKind::Word
                        && matches!(w.as_str(), "IF" | "NOT" | "EXISTS" | "OVERWRITE" | "TABLE")
                    {
                        j += 1;
                    } else {
                        break;
                    }
                }
                let mut name = String::new();
                while let Some(part) = toks.get(j).and_then(word) {
                    name.push_str(&part);
                    if toks
                        .get(j + 1)
                        .is_some_and(|d| d.kind == TokenKind::Sym(Sym::Dot))
                    {
                        name.push('.');
                        j += 2;
                    } else {
                        break;
                    }
                }
                if !name.is_empty()
                    && !name.ends_with('.')
                    && !targets.contains(&name)
                    && targets.len() < MAX_TARGETS
                {
                    targets.push(name);
                }
            }
        }
        i += 1;
    }
    if targets.is_empty() {
        operation
    } else {
        format!("{operation} {}", targets.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals_are_removed() {
        let text = sanitize_query_text(
            "SELECT * FROM t WHERE name = 'bob' AND ssn = \"123-45\" AND n > 42.5e3 -- secret\n \
             AND k = $q$x$q$ /* hint */",
        );
        assert_eq!(
            text,
            "SELECT * FROM t WHERE name = ? AND ssn = ? AND n > ? AND k = ?"
        );
        for leak in ["bob", "123-45", "42", "secret", "hint"] {
            assert!(!text.contains(leak), "{text}");
        }
    }

    #[test]
    fn mask_literals_does_not_cut_a_long_text() {
        let long = format!("SELECT {}", vec!["'x'"; 3000].join(", "));
        let masked = mask_literals(&long);
        assert_eq!(masked.matches('?').count(), 3000);
        assert!(!masked.ends_with("..."), "{}", &masked[masked.len() - 10..]);
        assert!(sanitize_query_text(&long).ends_with("..."));
    }

    #[test]
    fn broken_text_is_sanitized_too() {
        assert_eq!(sanitize_query_text("SELECT 'unterminated"), "SELECT ?");
        assert_eq!(sanitize_query_text(""), "");
    }

    #[test]
    fn quoted_identifiers_stay() {
        assert_eq!(
            sanitize_query_text("SELECT `a b` FROM `my t`"),
            "SELECT `a b` FROM `my t`"
        );
    }

    #[test]
    fn long_text_is_cut() {
        let sql = format!("SELECT {}", "a, ".repeat(5000));
        let text = sanitize_query_text(&sql);
        assert!(text.len() <= MAX_TEXT + 3, "{}", text.len());
        assert!(text.ends_with("..."));
    }

    #[test]
    fn summary_has_the_operation_and_tables() {
        assert_eq!(
            query_summary("select * from orders o join db.customers c on o.k = c.k where x = 'y'"),
            "SELECT orders db.customers"
        );
        assert_eq!(
            query_summary("WITH a AS (SELECT 1 FROM t) SELECT * FROM a JOIN t ON true"),
            "SELECT t a"
        );
        assert_eq!(
            query_summary("INSERT INTO sink SELECT * FROM src"),
            "INSERT sink src"
        );
        assert_eq!(query_summary("UPDATE t SET a = 1"), "UPDATE t");
        assert_eq!(query_summary("SELECT 1"), "SELECT");
        assert_eq!(query_summary("SELECT * FROM (SELECT 1) x"), "SELECT");
        assert_eq!(query_summary(""), "");
    }

    #[test]
    fn summary_is_bounded_and_has_no_literals() {
        let sql = "SELECT 1 FROM a, b JOIN c JOIN d JOIN e JOIN f JOIN g WHERE s = 'secret'";
        let s = query_summary(sql);
        assert_eq!(s, "SELECT a c d e f");
        assert!(!s.contains("secret"));
    }
}
