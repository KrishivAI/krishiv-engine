// SPDX-License-Identifier: Apache-2.0
//! `CREATE [OR REPLACE] VIEW ... WITH METRICS LANGUAGE YAML AS $$...$$`: Spark 4.2's
//! `createMetricView` rule (`SqlBaseParser.g4` 330-337), read after the view's name and column list.
//!
//! ANTLR predicts between `createView` and `createMetricView` by reading on until one alternative
//! is left, so the reading that gets further decides ([`Parser::create_full_view_command`]). Once
//! a metric-only clause (`WITH METRICS`, `LANGUAGE`) is read the prediction has picked this rule:
//! a stray token in the clause loop is then ANTLR's loop-back recovery, "extra input".

use super::{Fail, PR, Parser};
use crate::ast::{
    CreateMetricView, DdlName, Ident, PermanentViewColumn, Span, Statement, TableOption,
};
use crate::error;
use crate::keywords::SQL_KEYWORDS;
use crate::lexer::TokenKind;

/// The view head both readings share.
pub(super) struct ViewHead {
    pub(super) start: usize,
    pub(super) or_replace: bool,
    pub(super) if_not_exists: bool,
    pub(super) name: DdlName,
    pub(super) columns: Vec<PermanentViewColumn>,
}

impl Parser<'_> {
    /// The clauses and the `AS $$body$$` of a metric view, after its head.
    pub(super) fn metric_view_tail(&mut self, head: ViewHead) -> PR<Statement> {
        // Spark's `checkDuplicateClauses` order.
        let mut seen = [
            ("COMMENT", 0u8),
            ("TBLPROPERTIES", 0),
            ("LANGUAGE", 0),
            ("WITH METRICS", 0),
        ];
        let mut comment = None;
        let mut tblproperties: Option<Vec<TableOption>> = None;
        let mut language = None;
        let mut language_keyword = None;
        let mut with_metrics = None;
        // A metric-only clause was read: only this rule is left.
        let mut predicted = false;
        loop {
            let clause = if self.eat_kw("COMMENT") {
                let c = self.string_after()?;
                comment.get_or_insert(c);
                0
            } else if self.eat_kw("TBLPROPERTIES") {
                let props = self.property_list(false)?;
                tblproperties.get_or_insert(props);
                1
            } else if self.kw_at(0, "LANGUAGE") {
                let keyword = self.tok(0).map(|t| t.span);
                if language_keyword.is_none() {
                    language_keyword = keyword;
                }
                self.bump();
                predicted = true;
                let name = self.routine_language_name()?;
                language.get_or_insert(name);
                2
            } else if self.kw_at(0, "WITH") {
                let with = self.tok(0).map(|t| t.span);
                self.bump();
                let metrics = self.tok(0).map(|t| t.span);
                self.expect_kw("METRICS")?;
                predicted = true;
                if let (Some(w), Some(m)) = (with, metrics) {
                    with_metrics.get_or_insert([w, m]);
                }
                3
            } else {
                break;
            };
            if let Some(entry) = seen.get_mut(clause) {
                entry.1 = entry.1.saturating_add(1);
            }
        }
        if !self.kw_at(0, "AS") {
            if predicted && self.tok(0).is_some() {
                self.extra = Some(self.pos);
            }
            return self.fail();
        }
        self.bump();
        let (body, body_span) = self.code_literal()?;
        let span = self.span_from(head.start);
        self.command_end()?;
        Ok(Statement::CreateMetricView(Box::new(CreateMetricView {
            or_replace: head.or_replace,
            if_not_exists: head.if_not_exists,
            name: head.name,
            columns: head.columns,
            comment,
            tblproperties: tblproperties.unwrap_or_default(),
            language,
            with_metrics,
            language_keyword,
            duplicate_clause: seen.iter().find(|(_, n)| *n > 1).map(|(c, _)| *c),
            body,
            body_span,
            span,
        })))
    }

    /// `routineLanguage`'s name: `SQL`, or an `IDENTIFIER` token (an unquoted word that is no
    /// keyword; a backquoted name is not one).
    fn routine_language_name(&mut self) -> PR<Ident> {
        let Some(t) = self.tok(0) else {
            return self.fail();
        };
        let text = t.span.text(self.src);
        let keyword = SQL_KEYWORDS
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case(text));
        if t.kind != TokenKind::Word || (keyword && !text.eq_ignore_ascii_case("SQL")) {
            return self.fail();
        }
        self.bump();
        Ok(Ident {
            value: text.to_owned(),
            quoted: false,
            span: t.span,
        })
    }

    /// `codeLiteral`: a dollar-quoted string with a non-empty body; the body and the literal's
    /// span. An empty body fails at the closing tag, as ANTLR does (the tags are its own tokens).
    fn code_literal(&mut self) -> PR<(String, Span)> {
        let Some(t) = self.tok(0).filter(|t| t.kind == TokenKind::DollarString) else {
            return self.fail();
        };
        let text = t.span.text(self.src);
        let tag = text.get(1..).and_then(|r| r.find('$')).map_or(2, |i| i + 2);
        let body = text.get(tag..text.len().saturating_sub(tag)).unwrap_or("");
        if body.is_empty() {
            let close = Span::new(t.span.start + tag, t.span.end);
            let shown = format!("'{}'", text.get(..tag).unwrap_or("$$"));
            return Err(Fail::Fatal(Box::new(error::syntax_error(
                shown,
                String::new(),
                close,
            ))));
        }
        self.bump();
        Ok((body.to_owned(), t.span))
    }
}
