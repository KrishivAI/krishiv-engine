// SPDX-License-Identifier: Apache-2.0
//! Parameter markers (`?` and `:name`) in statement text, found from the
//! lexer's tokens so that markers inside string literals and comments are
//! never seen.

use crate::keywords::{CLAUSE_WORD, RESERVED_WORD, classify};
use crate::lexer::{Sym, TokenKind, tokenize};

/// Words that are not reserved but take an operand after them, so a `:` that follows is a marker.
const OPERATOR_WORDS: &[&str] = &[
    "DIV", "RLIKE", "REGEXP", "LIKE", "ILIKE", "IN", "IS", "NOT", "BETWEEN", "AND", "OR", "ESCAPE",
    "BY", "THEN", "ELSE", "WHEN", "USING", "LIMIT", "OFFSET", "VALUES", "SET", "AS", "DISTINCT",
    "ALL", "ANY", "SOME", "INTERVAL", "DEFAULT", "SELECT", "WHERE", "HAVING", "ON", "FROM",
];

/// One parameter marker in a statement's text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterMarker {
    /// Byte range of the whole marker (`?` or `:name`).
    pub start: usize,
    /// Byte offset after the marker.
    pub end: usize,
    /// The marker's name, `None` for a positional `?`.
    pub name: Option<String>,
}

/// The parameter markers of `sql`, in source order.
///
/// A `:` is a named marker when a name follows it and what precedes it
/// cannot end an expression; after an expression it is the variant path
/// operator (`col:field`). After a string literal, or a marker that
/// continues one, it is a marker again: Spark 4.2's `stringLit` is a run of
/// string literals and markers (`'a' '.' :b`), and the run is greedy.
/// A double-quoted token is a string only when `double_quoted_identifiers`
/// (`spark.sql.ansi.doubleQuotedIdentifiers`) is off; otherwise it is an
/// identifier, so `"v":field` is a variant path.
#[must_use]
pub fn parameter_markers(sql: &str, double_quoted_identifiers: bool) -> Vec<ParameterMarker> {
    let toks: Vec<_> = tokenize(sql)
        .into_iter()
        .filter(|t| !t.kind.is_trivia())
        .collect();
    let mut out = Vec::new();
    // Index of the name token of the last marker that continued a string run.
    let mut run_marker_end: Option<usize> = None;
    for (i, t) in toks.iter().enumerate() {
        match t.kind {
            TokenKind::Sym(Sym::Question) => out.push(ParameterMarker {
                start: t.span.start,
                end: t.span.end,
                name: None,
            }),
            TokenKind::Sym(Sym::Colon) => {
                let Some(next) = toks.get(i + 1) else {
                    continue;
                };
                if !matches!(next.kind, TokenKind::Word | TokenKind::QuotedIdent) {
                    continue;
                }
                let prev = i.checked_sub(1);
                let continues_string = prev.is_some_and(|p| {
                    run_marker_end == Some(p)
                        || toks.get(p).is_some_and(|t| match t.kind {
                            TokenKind::String => true,
                            TokenKind::DoubleQuoted => !double_quoted_identifiers,
                            _ => false,
                        })
                });
                if continues_string {
                    run_marker_end = Some(i + 1);
                }
                let ends_expression = !continues_string
                    && prev
                        .and_then(|p| toks.get(p))
                        .is_some_and(|p| match p.kind {
                            TokenKind::Word => {
                                let text = sql.get(p.span.start..p.span.end).unwrap_or_default();
                                text.eq_ignore_ascii_case("END")
                                    || (classify(text) & (RESERVED_WORD | CLAUSE_WORD) == 0
                                        && !OPERATOR_WORDS
                                            .iter()
                                            .any(|w| text.eq_ignore_ascii_case(w)))
                            }
                            TokenKind::QuotedIdent
                            | TokenKind::String
                            | TokenKind::DoubleQuoted
                            | TokenKind::Number(_)
                            | TokenKind::Sym(Sym::RParen | Sym::RBracket) => true,
                            _ => false,
                        });
                if ends_expression {
                    continue;
                }
                let text = sql.get(next.span.start..next.span.end).unwrap_or_default();
                let name = if next.kind == TokenKind::QuotedIdent {
                    text.trim_matches('`').replace("``", "`")
                } else {
                    text.to_owned()
                };
                out.push(ParameterMarker {
                    start: t.span.start,
                    end: next.span.end,
                    name: Some(name),
                });
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(sql: &str) -> Vec<(usize, usize, Option<String>)> {
        parameter_markers(sql, false)
            .into_iter()
            .map(|m| (m.start, m.end, m.name))
            .collect()
    }

    #[test]
    fn positional_and_named_markers_are_found_in_order() {
        assert_eq!(
            found("SELECT ?, :a + 1 FROM t WHERE x = :b AND y IN (?, :`c d`)"),
            vec![
                (7, 8, None),
                (10, 12, Some("a".into())),
                (34, 36, Some("b".into())),
                (47, 48, None),
                (50, 56, Some("c d".into())),
            ]
        );
    }

    #[test]
    fn a_double_quoted_token_ends_an_expression_so_a_colon_after_it_is_a_path() {
        // A string literal or (with double-quoted identifiers on) an identifier: either way
        // `:a` is the variant path operator, never a marker.
        assert!(found("SELECT \"v\":a FROM t").is_empty());
        assert_eq!(found("SELECT \"v\" = :a").len(), 1);
    }

    #[test]
    fn markers_in_strings_and_comments_are_text() {
        assert!(found("SELECT '?', ':a' /* ? :b */ -- :c ?\n, \"?\"").is_empty());
    }

    #[test]
    fn a_colon_after_an_expression_is_a_variant_path() {
        assert!(found("SELECT v:field, t.v:a.b, f(x):y, c[0]:z FROM t").is_empty());
        assert_eq!(
            found("SELECT v:field, :m"),
            vec![(16, 18, Some("m".into()))]
        );
    }

    #[test]
    fn a_marker_after_a_string_continues_the_string_run() {
        // Spark 4.2: `stringLit : singleStringLit+`, where a marker is a `singleStringLit`; the
        // loop is greedy, so `'.' :t` is two parts of one literal, not a variant path.
        assert_eq!(
            found("SELECT IDENTIFIER(:s '.' :t), '{}' :u :v"),
            vec![
                (18, 20, Some("s".into())),
                (25, 27, Some("t".into())),
                (35, 37, Some("u".into())),
                (38, 40, Some("v".into())),
            ]
        );
    }

    #[test]
    fn a_double_quoted_identifier_ends_an_expression() {
        // With doubleQuotedIdentifiers, `"v"` is a column, so `:field` is a variant path.
        assert!(parameter_markers(r#"SELECT "v":field FROM t"#, true).is_empty());
        // Without it, `"v"` is a string literal and the run takes the marker.
        assert_eq!(
            found(r#"SELECT "v":field FROM t"#),
            vec![(10, 16, Some("field".into()))]
        );
    }

    #[test]
    fn a_name_after_a_gap_is_a_marker_only_where_no_expression_ends() {
        assert_eq!(found("SELECT : a, :"), vec![(7, 10, Some("a".into()))]);
        assert!(found("SELECT v : a FROM t").is_empty());
    }

    #[test]
    fn operator_words_and_end_decide_what_a_colon_is() {
        assert_eq!(found("SELECT 7 DIV :a, x RLIKE :b").len(), 2);
        assert!(found("SELECT CASE WHEN x THEN 1 END:f FROM t").is_empty());
    }
}
