// SPDX-License-Identifier: Apache-2.0
//! Syntax errors: the message, the recovery and the hints for common mistakes.

use super::*;

impl Parser<'_> {
    #[inline(never)]
    pub(super) fn syntax_error(&self) -> ParseError {
        let found = self.toks.get(self.far);
        let (display, span) = found.map_or_else(
            || {
                (
                    "end of input".to_owned(),
                    Span::new(self.src.len(), self.src.len()),
                )
            },
            |t| {
                // An unterminated quote is reported at the quote, as Spark does.
                let span = if t.kind == TokenKind::Unterminated {
                    let len = t
                        .span
                        .text(self.src)
                        .chars()
                        .next()
                        .map_or(1, char::len_utf8);
                    Span::new(t.span.start, t.span.start + len)
                } else {
                    t.span
                };
                (self.token_display(*t), span)
            },
        );
        let hint = match self.missing {
            _ if self.extra == Some(self.far) => format!(": extra input {display}"),
            Some((at, token)) if at == self.far => format!(": missing {token}"),
            _ => String::new(),
        };
        error::syntax_error(display, hint, span)
    }

    /// The syntax error at the furthest failure, with the hint of ANTLR's
    /// inline recovery when the failure is at a `match` ([`expect_closing`]):
    /// ANTLR first drops the token in the way if the one after it fits
    /// ("extra input"), else inserts the wanted token if the token in the way
    /// fits after it ("missing"). Whether a token fits is decided by parsing
    /// the statement again with the edit made: it fits when the parse
    /// consumes it, and for a dropped token only when the wanted token is next.
    ///
    /// [`expect_closing`]: Parser::expect_closing
    #[inline(never)]
    pub(super) fn error_with_recovery<R: TopRule>(&mut self, opts: ParserOptions) -> ParseError {
        self.dangling_clause_word::<R>(opts);
        self.organization_clause_failure::<R>(opts);
        self.group_by_with_rollup_failure();
        self.filter_without_where_failure();
        self.extract_identifier_failure();
        self.pipe_extend_distinct_failure();
        if let Some((at, sym, how)) = self.matched
            && at == self.far
            && self.extra != Some(at)
            && self.missing.is_none_or(|(m, _)| m != at)
        {
            let fits = |edited: Vec<Token>, past: usize| {
                let mut p = Parser::new(self.src, edited, opts, self.force_bare);
                match R::parse(&mut p) {
                    Ok(_) => true,
                    Err(Fail::Syntax) => p.far > past,
                    Err(Fail::Fatal(_)) => false,
                }
            };
            let mut dropped = self.toks.clone();
            if at < dropped.len() {
                dropped.remove(at);
                // The token after the dropped one must be the wanted one: it
                // is what the state expects (operators that continue the
                // element aside, which Spark's hints do not show either).
                let next = self.toks.get(at + 1).map(|t| t.kind);
                if next == Some(TokenKind::Sym(sym)) && fits(dropped, at) {
                    self.extra = Some(at);
                }
            } else if matches!(how, Close::List(n) if n >= 2) {
                self.extra = Some(at);
            }
            if self.extra != Some(at) && matches!(how, Close::DropOrInsert) {
                let text = match sym {
                    Sym::RBracket => Some("']'"),
                    Sym::RParen => Some("')'"),
                    _ => None,
                };
                let mut inserted = self.toks.clone();
                let start = self.toks.get(at).map_or(self.src.len(), |t| t.span.start);
                let token = Token {
                    kind: TokenKind::Sym(sym),
                    span: Span::new(start, start),
                };
                inserted.insert(at, token);
                if let Some(text) = text
                    && fits(inserted, at + 1)
                {
                    self.missing = Some((at, text));
                }
            }
        }
        self.syntax_error()
    }

    /// A statement that ends with `BY` or `HAVING` right after text that is a
    /// statement without it (`... t ORDER BY`, `... GROUP BY a HAVING`):
    /// ANTLR's prediction for the optional clause fails at the end of the
    /// input and takes the exit, so the error is the clause's last token
    /// dropped as "extra input" instead of the end of the input.
    pub(super) fn dangling_clause_word<R: TopRule>(&mut self, opts: ParserOptions) {
        let n = self.toks.len();
        let last_is = |p: &Self, kw: &str| {
            p.toks
                .last()
                .is_some_and(|t| t.kind == TokenKind::Word && is_kw(t.span.text(p.src), kw))
        };
        if n >= 2
            && self.far == n
            && (last_is(self, "BY") || last_is(self, "HAVING"))
            && self.extra.is_none()
        {
            let mut p = Parser::new(self.src, self.toks[..n - 1].to_vec(), opts, self.force_bare);
            if R::parse(&mut p).is_ok() {
                self.far = n - 1;
                self.extra = Some(n - 1);
            }
        }
    }

    /// A failure at the end of the input inside an `ORDER BY`, `SORT BY` or
    /// `LIMIT` clause that follows a complete query (`... t ORDER BY a,`; `(SELECT 1 ORDER BY a,` also gets its `missing` hint):
    /// ANTLR's prediction takes the keyword as an alias and reports it, with
    /// no hint, instead of the end of the input. Failing right after the
    /// `BY` reports the `BY`.
    pub(super) fn organization_clause_failure<R: TopRule>(&mut self, opts: ParserOptions) {
        let n = self.toks.len();
        if self.far != n || self.extra.is_some() {
            return;
        }
        let word = |p: &Self, i: usize, kw: &str| {
            p.toks
                .get(i)
                .is_some_and(|t| t.kind == TokenKind::Word && is_kw(t.span.text(p.src), kw))
        };
        let mut depth = 0usize;
        let mut clause = None;
        for i in (0..n).rev() {
            match self.toks[i].kind {
                TokenKind::Sym(Sym::RParen) => depth += 1,
                TokenKind::Sym(Sym::LParen) => depth = depth.saturating_sub(1),
                _ if depth == 0
                    && ((word(self, i, "ORDER") || word(self, i, "SORT"))
                        && word(self, i + 1, "BY")
                        && i + 2 < n
                        || word(self, i, "LIMIT") && i + 1 < n) =>
                {
                    clause = Some(i);
                    break;
                }
                _ => {}
            }
        }
        let Some(k) = clause.filter(|&k| k > 0) else {
            return;
        };
        let mut p = Parser::new(self.src, self.toks[..k].to_vec(), opts, self.force_bare);
        // The query before the clause is complete, or ends where the clause
        // starts (an unclosed `(`), which then gets its recovery hint.
        match R::parse(&mut p) {
            Ok(_) => self.far = k,
            Err(Fail::Syntax) if p.far == k => {
                self.far = k;
                self.matched = p.matched;
            }
            Err(_) => {}
        }
    }

    /// `GROUP BY WITH ROLLUP` / `WITH CUBE` with no grouping expressions
    /// (`SELECT a, b FROM grouping GROUP BY WITH ROLLUP`): ANTLR reads the
    /// `WITH` as part of the grouping and reports the `ROLLUP` / `CUBE` as
    /// extra input. This parser's `GROUP BY` items fail at the `WITH`, so the
    /// error is moved to the next token with the extra-input hint.
    pub(super) fn group_by_with_rollup_failure(&mut self) {
        if self.extra.is_some() || self.missing.is_some() {
            return;
        }
        let at = self.far;
        let is_with = self
            .toks
            .get(at)
            .is_some_and(|t| t.kind == TokenKind::Word && is_kw(t.span.text(self.src), "WITH"));
        let next_is_rollup_cube = self.toks.get(at + 1).is_some_and(|t| {
            t.kind == TokenKind::Word
                && (is_kw(t.span.text(self.src), "ROLLUP") || is_kw(t.span.text(self.src), "CUBE"))
        });
        let prev_is_group_by = at >= 2
            && self.toks.get(at - 2).is_some_and(|t| {
                t.kind == TokenKind::Word && is_kw(t.span.text(self.src), "GROUP")
            })
            && self
                .toks
                .get(at - 1)
                .is_some_and(|t| t.kind == TokenKind::Word && is_kw(t.span.text(self.src), "BY"));
        if is_with && next_is_rollup_cube && prev_is_group_by {
            self.far = at + 1;
            self.extra = Some(at + 1);
        }
    }

    /// `FILTER (expr)` without `WHERE` (`SELECT sum(c1) FILTER (c1 != var1)`):
    /// Spark reports the opening `(`. This parser's `expect_kw("WHERE")`
    /// fails at the first token of the expression, so the error is moved back
    /// to the `(` with no hint.
    pub(super) fn filter_without_where_failure(&mut self) {
        if self.extra.is_some() || self.missing.is_some() {
            return;
        }
        let far = self.far;
        for (i, t) in self.toks.iter().enumerate() {
            if t.kind != TokenKind::Word || !is_kw(t.span.text(self.src), "FILTER") {
                continue;
            }
            let Some(open) = self.toks.get(i + 1) else {
                continue;
            };
            if open.kind != TokenKind::Sym(Sym::LParen) {
                continue;
            }
            // An aggregate `FILTER` follows the `)` that closes the call;
            // the higher-order function `filter(array, x -> p)` does not.
            if i == 0 || self.toks[i - 1].kind != TokenKind::Sym(Sym::RParen) {
                continue;
            }
            // The `(` whose matching `)` (or the end of input) contains the
            // failure, and whose first token inside is not `WHERE`.
            let close = self.closers.get(i + 1).copied().unwrap_or(usize::MAX);
            let inside_end = if close == usize::MAX {
                self.toks.len()
            } else {
                close
            };
            if far <= i + 1 || far >= inside_end {
                continue;
            }
            let first_inside = self.toks.get(i + 2);
            let has_where = first_inside.is_some_and(|t| {
                t.kind == TokenKind::Word && is_kw(t.span.text(self.src), "WHERE")
            });
            if !has_where {
                self.far = i + 1;
                self.matched = None;
                return;
            }
        }
    }

    /// `EXTRACT(IDENTIFIER('...') FROM ...)` (`SELECT EXTRACT(IDENTIFIER('YEAR')
    /// FROM ...)`): Spark inserts the missing `)` before the `FROM`. This
    /// parser reads `EXTRACT` as an ordinary call (the field is not a plain
    /// word) and its `expect_closing` only drops, so the missing hint is added
    /// here when the failure is at that `FROM`.
    pub(super) fn extract_identifier_failure(&mut self) {
        if self.extra.is_some() || self.missing.is_some() {
            return;
        }
        let at = self.far;
        let is_from = self
            .toks
            .get(at)
            .is_some_and(|t| t.kind == TokenKind::Word && is_kw(t.span.text(self.src), "FROM"));
        if !is_from {
            return;
        }
        // The sequence `EXTRACT ( IDENTIFIER (`: the field position of an
        // `EXTRACT` holds an `IDENTIFIER(...)` call.
        let word = |i: usize, kw: &str| {
            self.toks
                .get(i)
                .is_some_and(|t| t.kind == TokenKind::Word && is_kw(t.span.text(self.src), kw))
        };
        let sym = |i: usize| {
            self.toks
                .get(i)
                .is_some_and(|t| t.kind == TokenKind::Sym(Sym::LParen))
        };
        let has_extract_identifier = (0..self.toks.len())
            .any(|i| word(i, "EXTRACT") && sym(i + 1) && word(i + 2, "IDENTIFIER") && sym(i + 3));
        if has_extract_identifier {
            self.missing = Some((at, "')'"));
        }
    }

    /// `TABLE t |> EXTEND DISTINCT x AS z`: Spark reads `extend distinct x`
    /// and reports the `AS`. This parser rejects the `DISTINCT`, so the error
    /// is moved forward to the `AS` with no hint.
    pub(super) fn pipe_extend_distinct_failure(&mut self) {
        if self.extra.is_some() || self.missing.is_some() {
            return;
        }
        let at = self.far;
        let is_distinct = self
            .toks
            .get(at)
            .is_some_and(|t| t.kind == TokenKind::Word && is_kw(t.span.text(self.src), "DISTINCT"));
        if !is_distinct {
            return;
        }
        let prev_is_extend = at >= 1
            && self.toks.get(at - 1).is_some_and(|t| {
                t.kind == TokenKind::Word && is_kw(t.span.text(self.src), "EXTEND")
            });
        // `EXTEND DISTINCT <item> AS ...`: the failure belongs at the `AS`.
        let next_next_is_as = self
            .toks
            .get(at + 2)
            .is_some_and(|t| t.kind == TokenKind::Word && is_kw(t.span.text(self.src), "AS"));
        if prev_is_extend && next_next_is_as {
            self.far = at + 2;
        }
    }

    /// A token as an error message shows it: quoted text like Spark's, except
    /// that the text of a string literal, which may be a secret, is never
    /// echoed (an unterminated one shows its quote, as Spark does).
    pub(super) fn token_display(&self, t: Token) -> String {
        let text = t.span.text(self.src);
        match t.kind {
            TokenKind::Unterminated => {
                format!("'{}'", text.chars().next().unwrap_or('\''))
            }
            TokenKind::DoubleQuoted if self.opts.double_quoted_identifiers => format!("'{text}'"),
            // Spark shows the opening tag of a dollar-quoted string.
            TokenKind::DollarString => {
                let tag = text.get(1..).and_then(|t| t.find('$')).map_or(1, |i| i + 2);
                format!("'{}'", text.get(..tag).unwrap_or("$"))
            }
            k if k.is_string_like() => "a string literal".to_owned(),
            _ => {
                let shown = text
                    .replace('\n', "\\n")
                    .replace('\r', "\\r")
                    .replace('\t', "\\t");
                format!("'{shown}'")
            }
        }
    }
}
