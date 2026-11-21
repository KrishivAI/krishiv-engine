// SPDX-License-Identifier: Apache-2.0
//! Source-text helpers of the validator: the text and the span of a clause or a property list.

use crate::ast::Span;
use crate::lexer::{Sym, TokenKind, tokenize};

/// The clause at `span` as Spark's `validateRowFormatFileFormat` prints a context: its children's
/// `getText` joined by spaces. The first `tokens` tokens are children of their own, as is each
/// token after them outside brackets (`ROW FORMAT SERDE 'x'`, `FIELDS TERMINATED BY ','`); a
/// bracketed list (`SERDEPROPERTIES (...)`'s), or with `tokens` set everything after the first
/// `tokens` (a `STORED BY` storage handler), is one child whose text drops the whitespace.
pub(super) fn context_children_text(sql: &str, span: Span, tokens: usize) -> String {
    let text = span.text(sql);
    let mut children: Vec<String> = Vec::new();
    let mut depth = 0_usize;
    for (i, t) in tokenize(text)
        .iter()
        .filter(|t| !t.kind.is_trivia())
        .enumerate()
    {
        let token = t.span.text(text);
        let glued = i > tokens || depth > 0;
        match t.kind {
            TokenKind::Sym(Sym::LParen) => depth += 1,
            TokenKind::Sym(Sym::RParen) => depth = depth.saturating_sub(1),
            _ => {}
        }
        match children.last_mut() {
            Some(last) if glued => last.push_str(token),
            _ => children.push(token.to_owned()),
        }
    }
    children.join(" ")
}

/// The parenthesized list from `first` to `last` (the first and last options of a property list):
/// the `(` before the first through the `)` after the last, where Spark reports an error raised
/// inside `visitPropertyList`. The options themselves when a comment hides a parenthesis.
pub(super) fn property_list_span(sql: &str, first: Span, last: Span) -> Span {
    let paren = |t: &crate::lexer::Token, sym| t.kind == TokenKind::Sym(sym);
    let start = sql
        .get(..first.start)
        .and_then(|before| {
            tokenize(before)
                .into_iter()
                .rev()
                .find(|t| !t.kind.is_trivia())
        })
        .filter(|t| paren(t, Sym::LParen))
        .map_or(first.start, |t| t.span.start);
    let end = sql
        .get(last.end..)
        .and_then(|after| tokenize(after).into_iter().find(|t| !t.kind.is_trivia()))
        .filter(|t| paren(t, Sym::RParen))
        .map_or(last.end, |t| last.end + t.span.end);
    Span::new(start, end)
}
