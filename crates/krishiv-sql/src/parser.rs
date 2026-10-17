// SPDX-License-Identifier: Apache-2.0
//! Recursive-descent parser for queries, relations and clauses; expressions
//! are in [`expr`] (precedence climbing over explicit stacks).
//!
//! Failures are [`Fail::Syntax`] (the furthest token reached is reported as
//! `PARSE_SYNTAX_ERROR`; a caller may try another alternative) or
//! [`Fail::Fatal`] (an error Spark raises at once, never retried). The only
//! retried alternative is a parenthesized subquery, and failed query
//! positions are remembered, so parsing stays near-linear on any input.
//!
//! Depth: every recursive rule runs inside [`Parser::nested`], and loops that
//! build left-deep trees (operator chains, set operations, postfix chains)
//! account for the height they add with [`Parser::measured`] and
//! [`Parser::grow`], so no tree nests deeper than [`MAX_DEPTH`]. A statement
//! nested deeper than [`INLINE_NESTING`] brackets parses on a thread whose
//! stack fits that depth.

mod expr;

use std::collections::HashMap;

use crate::ast::{
    Alias, Cte, Expr, GroupBy, GroupItem, GroupItemKind, GroupWith, GroupingSet, Hint, Ident, Join,
    JoinConstraint, JoinKind, LateralView, Limit, NamedWindow, ObjectName, Pivot, Query, Relation,
    RelationExt, Select, SelectItem, SetExpr, SetExprKind, SetOperator, SetQuantifier, SortItem,
    Span, TableAlias, TableFactor, TableFactorKind, Unpivot, UnpivotColumns, Values, With,
};
use crate::error::{self, ParseError};
use crate::keywords::{CLAUSE_WORD, RESERVED_WORD, STRICT_WORD, classify, is_kw};
use crate::lexer::{Sym, Token, TokenKind, tokenize, unquote_ident};
use crate::{MAX_DEPTH, ParserOptions};

/// Why a rule did not match.
pub(crate) enum Fail {
    /// The tokens do not fit; the caller may try another alternative.
    Syntax,
    /// An error to report as is.
    Fatal(Box<ParseError>),
}

pub(crate) type PR<T> = Result<T, Fail>;

pub(crate) struct Parser<'s> {
    src: &'s str,
    toks: Vec<Token>,
    /// [`classify`] bits per token.
    classes: Vec<u8>,
    pos: usize,
    opts: ParserOptions,
    /// Furthest token a rule failed at.
    far: usize,
    /// Token after a complete statement, when it is the last one (Spark's
    /// "extra input" hint).
    extra: Option<usize>,
    /// Current recursion depth and the deepest tree level reached.
    depth: usize,
    peak: usize,
    /// Positions where a query failed to parse, with the furthest token
    /// reached.
    failed_queries: HashMap<usize, usize>,
    /// For each token, the first token at or after it that is not `(`.
    after_parens: Vec<usize>,
}

/// Parses one query statement (`query ;* EOF`).
pub(crate) fn parse_query(src: &str, opts: ParserOptions) -> Result<Query, ParseError> {
    // Java's `String.trim`: Spark reports an empty statement only for these.
    if src.trim_matches(|c| c <= ' ').is_empty() {
        return Err(error::empty_statement(Span::new(0, src.len())));
    }
    let toks = tokenize(src)?;
    if nesting_bound(src, &toks) <= INLINE_NESTING {
        return run(src, toks, opts);
    }
    // A deeply nested statement parses on its own thread, whose stack fits
    // `MAX_DEPTH` levels whatever the caller's stack is.
    let stack = if cfg!(debug_assertions) {
        64 << 20
    } else {
        16 << 20
    };
    let at = Span::new(0, 0);
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("krishiv-sql-parse".to_owned())
            .stack_size(stack)
            .spawn_scoped(scope, || run(src, toks, opts))
            .map_or_else(
                |_| Err(error::too_complex(at)),
                |h| {
                    h.join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                },
            )
    })
}

/// Bracket nesting (`(`, `[`, `CASE`, `<` of a complex type) parsed on the
/// caller's stack; each level takes a few recursion levels.
const INLINE_NESTING: usize = 16;

/// An upper bound of the bracket nesting of `toks`, plus one per lambda
/// arrow: every recursion of the parser is inside one of them.
fn nesting_bound(src: &str, toks: &[Token]) -> usize {
    let (mut depth, mut angles, mut max, mut arrows) = (0_usize, 0_usize, 0_usize, 0_usize);
    let mut prev_complex = false;
    for t in toks {
        let word = |kw: &str| t.kind == TokenKind::Word && is_kw(t.span.text(src), kw);
        match t.kind {
            TokenKind::Sym(Sym::LParen | Sym::LBracket) => depth += 1,
            TokenKind::Sym(Sym::Lt) if prev_complex => {
                depth += 1;
                angles += 1;
            }
            TokenKind::Sym(Sym::RParen | Sym::RBracket) => depth = depth.saturating_sub(1),
            TokenKind::Sym(Sym::Gt) if angles > 0 => {
                angles -= 1;
                depth = depth.saturating_sub(1);
            }
            TokenKind::Sym(Sym::Arrow) => arrows += 1,
            _ if word("CASE") => depth += 1,
            _ if word("END") => depth = depth.saturating_sub(1),
            _ => {}
        }
        max = max.max(depth);
        prev_complex = word("ARRAY") || word("MAP") || word("STRUCT");
    }
    max + arrows
}

fn run(src: &str, toks: Vec<Token>, opts: ParserOptions) -> Result<Query, ParseError> {
    let classes = toks
        .iter()
        .map(|t| {
            if t.kind == TokenKind::Word {
                classify(t.span.text(src))
            } else {
                0
            }
        })
        .collect();
    let mut after_parens = vec![0; toks.len()];
    let mut next = toks.len();
    for (i, t) in toks.iter().enumerate().rev() {
        if t.kind != TokenKind::Sym(Sym::LParen) {
            next = i;
        }
        after_parens[i] = next;
    }
    let mut p = Parser {
        src,
        toks,
        classes,
        pos: 0,
        opts,
        far: 0,
        extra: None,
        depth: 0,
        peak: 0,
        failed_queries: HashMap::new(),
        after_parens,
    };
    match p.statement() {
        Ok(q) => Ok(q),
        Err(Fail::Fatal(e)) => Err(*e),
        Err(Fail::Syntax) => Err(p.syntax_error()),
    }
}

/// Words after which a parenthesis opens a query rather than an expression.
const QUERY_START: &[&str] = &["SELECT", "WITH", "VALUES", "TABLE", "FROM"];

impl<'s> Parser<'s> {
    // ---- tokens -------------------------------------------------------

    fn tok(&self, n: usize) -> Option<Token> {
        self.toks.get(self.pos + n).copied()
    }

    fn kind(&self, n: usize) -> Option<TokenKind> {
        self.tok(n).map(|t| t.kind)
    }

    /// The [`classify`] bits of token `n` (0 for a non-word).
    fn class(&self, n: usize) -> u8 {
        self.classes.get(self.pos + n).copied().unwrap_or(0)
    }

    fn text(&self, n: usize) -> &'s str {
        self.tok(n).map_or("", |t| t.span.text(self.src))
    }

    fn bump(&mut self) {
        self.pos += 1;
    }

    /// Start of the current token (end of input if none).
    fn start(&self) -> usize {
        self.tok(0).map_or(self.src.len(), |t| t.span.start)
    }

    /// The span from `start` to the end of the last consumed token.
    fn span_from(&self, start: usize) -> Span {
        let end = self
            .pos
            .checked_sub(1)
            .and_then(|i| self.toks.get(i))
            .map_or(start, |t| t.span.end);
        Span::new(start, end.max(start))
    }

    fn kw_at(&self, n: usize, kw: &str) -> bool {
        self.kind(n) == Some(TokenKind::Word) && is_kw(self.text(n), kw)
    }

    fn any_kw_at(&self, n: usize, kws: &[&str]) -> bool {
        kws.iter().any(|k| self.kw_at(n, k))
    }

    fn sym_at(&self, n: usize, s: Sym) -> bool {
        self.kind(n) == Some(TokenKind::Sym(s))
    }

    /// Records a failure at the current token.
    fn fail<T>(&mut self) -> PR<T> {
        self.far = self.far.max(self.pos);
        Err(Fail::Syntax)
    }

    fn fatal<T>(e: ParseError) -> PR<T> {
        Err(Fail::Fatal(Box::new(e)))
    }

    fn eat_kw(&mut self, kw: &str) -> bool {
        let hit = self.kw_at(0, kw);
        self.pos += usize::from(hit);
        hit
    }

    fn expect_kw(&mut self, kw: &str) -> PR<()> {
        if self.eat_kw(kw) { Ok(()) } else { self.fail() }
    }

    fn eat_sym(&mut self, s: Sym) -> bool {
        let hit = self.sym_at(0, s);
        self.pos += usize::from(hit);
        hit
    }

    fn expect_sym(&mut self, s: Sym) -> PR<()> {
        if self.eat_sym(s) { Ok(()) } else { self.fail() }
    }

    fn comma_sep<T>(&mut self, mut f: impl FnMut(&mut Self) -> PR<T>) -> PR<Vec<T>> {
        let mut v = vec![f(self)?];
        while self.eat_sym(Sym::Comma) {
            v.push(f(self)?);
        }
        Ok(v)
    }

    /// `( f, ... )`.
    fn paren_list<T>(&mut self, f: impl FnMut(&mut Self) -> PR<T>) -> PR<Vec<T>> {
        self.expect_sym(Sym::LParen)?;
        let v = self.comma_sep(f)?;
        self.expect_sym(Sym::RParen)?;
        Ok(v)
    }

    #[inline(never)]
    fn syntax_error(&self) -> ParseError {
        let found = self.toks.get(self.far);
        let (display, span) = found.map_or_else(
            || {
                (
                    "end of input".to_owned(),
                    Span::new(self.src.len(), self.src.len()),
                )
            },
            |t| {
                let text = t.span.text(self.src);
                let shown = text
                    .replace('\n', "\\n")
                    .replace('\r', "\\r")
                    .replace('\t', "\\t");
                (format!("'{shown}'"), t.span)
            },
        );
        let hint = if self.extra == Some(self.far) {
            format!(": extra input {display}")
        } else {
            String::new()
        };
        error::syntax_error(display, hint, span)
    }

    // ---- depth ----------------------------------------------------------

    /// Goes one level deeper (the caller decrements `depth` after).
    fn enter(&mut self) -> PR<()> {
        self.depth += 1;
        let r = self.grow(0);
        if r.is_err() {
            self.depth -= 1;
        }
        r
    }

    /// Runs `f` one level deeper.
    fn nested<T>(&mut self, f: impl FnOnce(&mut Self) -> PR<T>) -> PR<T> {
        self.enter()?;
        let r = f(self);
        self.depth -= 1;
        r
    }

    /// Runs `f` and returns how many levels its tree reaches below the
    /// current depth.
    fn measured<T>(&mut self, f: impl FnOnce(&mut Self) -> PR<T>) -> PR<(T, usize)> {
        let outer = self.peak;
        self.peak = self.depth;
        let r = f(self);
        let height = self.peak - self.depth;
        self.peak = self.peak.max(outer);
        r.map(|t| (t, height))
    }

    /// Records a tree reaching `height` levels below the current depth.
    fn grow(&mut self, height: usize) -> PR<()> {
        let level = self.depth + height;
        if level > MAX_DEPTH {
            return Self::fatal(error::too_complex(Span::new(self.start(), self.start())));
        }
        self.peak = self.peak.max(level);
        Ok(())
    }

    // ---- identifiers ----------------------------------------------------

    /// The identifier at token `n`, if it is one; `strong` accepts every
    /// keyword (a position where only an identifier fits) unless keywords
    /// are enforced.
    fn ident_at(&self, n: usize, strong: bool) -> Option<Ident> {
        let t = self.tok(n)?;
        let text = t.span.text(self.src);
        let (value, quoted) = match t.kind {
            TokenKind::Word => {
                let keyword_ok = strong && !self.opts.enforce_reserved_keywords;
                if !keyword_ok && self.class(n) & RESERVED_WORD != 0 {
                    return None;
                }
                (text.to_owned(), false)
            }
            TokenKind::QuotedIdent => (unquote_ident(text), true),
            TokenKind::DoubleQuoted if self.opts.double_quoted_identifiers => (
                text.get(1..text.len() - 1)
                    .unwrap_or_default()
                    .replace("\"\"", "\""),
                true,
            ),
            _ => return None,
        };
        Some(Ident {
            value,
            quoted,
            span: t.span,
        })
    }

    /// Whether token `n` is an identifier ([`Self::ident_at`] without
    /// building it).
    fn is_ident_at(&self, n: usize, strong: bool) -> bool {
        match self.kind(n) {
            Some(TokenKind::Word) => {
                (strong && !self.opts.enforce_reserved_keywords)
                    || self.class(n) & RESERVED_WORD == 0
            }
            Some(TokenKind::QuotedIdent) => true,
            Some(TokenKind::DoubleQuoted) => self.opts.double_quoted_identifiers,
            _ => false,
        }
    }

    /// Consumes an identifier (`strong` as for [`Self::ident_at`]).
    fn ident_with(&mut self, strong: bool) -> PR<Ident> {
        let Some(id) = self.ident_at(0, strong) else {
            return self.fail();
        };
        // Spark's `exitUnquotedIdentifier`: only ASCII letters, digits, `_`.
        if !id.quoted
            && !id
                .value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            return Self::fatal(error::invalid_identifier(&id.value, id.span));
        }
        self.bump();
        Ok(id)
    }

    fn ident(&mut self) -> PR<Ident> {
        self.ident_with(false)
    }

    fn any_ident(&mut self) -> PR<Ident> {
        self.ident_with(true)
    }

    /// An identifier that is not followed by `- word` (Spark's
    /// `errorCapturingIdentifier`: `test-table` is `INVALID_IDENTIFIER`).
    fn name_part(&mut self, strong: bool) -> PR<Ident> {
        let id = self.ident_with(strong)?;
        if self.sym_at(0, Sym::Minus) && self.is_ident_at(1, true) {
            let start = id.span.start;
            while self.sym_at(0, Sym::Minus) && self.is_ident_at(1, true) {
                self.pos += 2;
            }
            let span = self.span_from(start);
            return Self::fatal(error::invalid_identifier(span.text(self.src), span));
        }
        Ok(id)
    }

    fn object_name(&mut self) -> PR<ObjectName> {
        let start = self.start();
        let mut parts = vec![self.name_part(false)?];
        while self.sym_at(0, Sym::Dot) && self.is_ident_at(1, true) {
            self.bump();
            parts.push(self.name_part(true)?);
        }
        Ok(ObjectName {
            parts,
            span: self.span_from(start),
        })
    }

    /// A table name; a reserved word followed by another name is one too
    /// (`FROM from t` reads table `from` with alias `t`, as Spark does).
    fn table_name(&mut self) -> PR<ObjectName> {
        let named = matches!(self.kind(1), Some(TokenKind::Word | TokenKind::QuotedIdent));
        if !named || self.opts.enforce_reserved_keywords || self.is_ident_at(0, false) {
            return self.object_name();
        }
        let first = self.any_ident()?;
        Ok(ObjectName {
            span: first.span,
            parts: vec![first],
        })
    }

    fn ident_list(&mut self) -> PR<Vec<Ident>> {
        self.paren_list(Self::any_ident)
    }

    /// Whether token `n` ends a list item: end of input, `,`, `)` or `;`.
    fn item_end_at(&self, n: usize) -> bool {
        matches!(
            self.kind(n),
            None | Some(TokenKind::Sym(Sym::Comma | Sym::RParen | Sym::Semi))
        )
    }

    /// An alias without `AS`. Unless keywords are enforced, a keyword that
    /// ends the item is one too (`SELECT a FROM` names `a` `FROM`, as
    /// ANTLR reads it); `strict` admits the strict-non-reserved words
    /// (`UNION`, `JOIN`, ...), which only a column alias may be.
    fn implicit_alias(&mut self, strict: bool) -> PR<Option<Ident>> {
        let keyword = !self.opts.enforce_reserved_keywords
            && self.kind(0) == Some(TokenKind::Word)
            && self.item_end_at(1)
            && (strict || self.class(0) & STRICT_WORD == 0);
        let clause = self.kind(0) == Some(TokenKind::Word) && self.class(0) & CLAUSE_WORD != 0;
        if !self.is_ident_at(0, keyword) || (clause && !keyword) {
            return Ok(None);
        }
        self.ident_with(keyword).map(Some)
    }

    /// `[AS] name`.
    fn opt_alias_name(&mut self) -> PR<Option<Ident>> {
        if !self.item_end_at(1) && self.eat_kw("AS") {
            return self.any_ident().map(Some);
        }
        self.implicit_alias(false)
    }

    // ---- statements and queries ------------------------------------------

    fn statement(&mut self) -> PR<Query> {
        let q = self.query()?;
        let mut semi = false;
        while self.eat_sym(Sym::Semi) {
            semi = true;
        }
        if self.pos < self.toks.len() {
            // ANTLR drops one extraneous token before the end, or any after
            // `;` (its `SEMICOLON*` loop), as "extra input".
            if semi || self.pos + 1 == self.toks.len() {
                self.extra = Some(self.pos);
            }
            return self.fail();
        }
        Ok(q)
    }

    /// A query; a failure at a position is remembered, so the subquery
    /// attempts of nested parentheses never repeat work.
    pub(crate) fn query(&mut self) -> PR<Query> {
        let pos = self.pos;
        if let Some(&far) = self.failed_queries.get(&pos) {
            self.far = self.far.max(far);
            return Err(Fail::Syntax);
        }
        let r = self.query_at();
        if matches!(r, Err(Fail::Syntax)) {
            self.failed_queries.insert(pos, self.far);
        }
        r
    }

    fn query_at(&mut self) -> PR<Query> {
        self.nested(|p| {
            let start = p.start();
            let with = if p.kw_at(0, "WITH") {
                Some(p.with_clause()?)
            } else {
                None
            };
            let body = p.query_term()?;
            let mut q = Self::bare_query(body, start);
            q.with = with;
            p.organization(&mut q)?;
            q.span = p.span_from(start);
            Ok(q)
        })
    }

    fn bare_query(body: SetExpr, start: usize) -> Query {
        Query {
            with: None,
            span: Span::new(start, body.span.end),
            body,
            order_by: vec![],
            cluster_by: vec![],
            distribute_by: vec![],
            sort_by: vec![],
            windows: vec![],
            limit: None,
            offset: None,
        }
    }

    /// `queryOrganization`.
    #[inline(never)]
    fn organization(&mut self, q: &mut Query) -> PR<()> {
        if self.kw_at(0, "ORDER") && self.kw_at(1, "BY") {
            self.pos += 2;
            q.order_by = self.comma_sep(|p| p.sort_item(true))?;
        }
        for (kw, list) in [
            ("CLUSTER", &mut q.cluster_by),
            ("DISTRIBUTE", &mut q.distribute_by),
        ] {
            if self.kw_at(0, kw) && self.kw_at(1, "BY") {
                self.pos += 2;
                *list = self.comma_sep(Self::expr)?;
            }
        }
        if self.kw_at(0, "SORT") && self.kw_at(1, "BY") {
            self.pos += 2;
            q.sort_by = self.comma_sep(|p| p.sort_item(false))?;
        }
        q.windows = self.window_clause()?;
        let start = self.start();
        if self.eat_kw("LIMIT") {
            let value = if self.eat_kw("ALL") {
                None
            } else {
                Some(self.expr()?)
            };
            q.limit = Some(Limit {
                value,
                span: self.span_from(start),
            });
        }
        if self.eat_kw("OFFSET") {
            q.offset = Some(self.expr()?);
        }
        Ok(())
    }

    #[inline(never)]
    fn with_clause(&mut self) -> PR<With> {
        let start = self.start();
        self.expect_kw("WITH")?;
        let recursive = self.eat_kw("RECURSIVE");
        let ctes = self.comma_sep(|p| {
            let start = p.start();
            let name = p.name_part(false)?;
            let columns = if p.sym_at(0, Sym::LParen) {
                p.ident_list()?
            } else {
                vec![]
            };
            let mut max_recursion_level = None;
            if p.kw_at(0, "MAX") && p.kw_at(1, "RECURSION") && p.kw_at(2, "LEVEL") {
                p.pos += 3;
                if p.kind(0) != Some(TokenKind::Number(crate::ast::NumKind::Int)) {
                    return p.fail();
                }
                max_recursion_level = Some(p.text(0).to_owned());
                p.bump();
            }
            p.eat_kw("AS");
            p.expect_sym(Sym::LParen)?;
            let query = Box::new(p.query()?);
            p.expect_sym(Sym::RParen)?;
            Ok(Cte {
                name,
                columns,
                max_recursion_level,
                query,
                span: p.span_from(start),
            })
        })?;
        Ok(With {
            recursive,
            ctes,
            span: self.span_from(start),
        })
    }

    /// Set operations: `INTERSECT` binds tighter than `UNION` / `EXCEPT`
    /// unless `spark.sql.legacy.setopsPrecedence.enabled`; all are
    /// left-associative.
    fn query_term(&mut self) -> PR<SetExpr> {
        self.set_ops(0)
    }

    fn set_ops(&mut self, min_prec: u8) -> PR<SetExpr> {
        let (mut left, mut height) = self.measured(Self::query_primary)?;
        loop {
            let (op, prec) = if self.kw_at(0, "INTERSECT") {
                (SetOperator::Intersect, 2)
            } else if self.kw_at(0, "UNION") {
                (SetOperator::Union, 1)
            } else if self.any_kw_at(0, &["EXCEPT", "MINUS"]) {
                (SetOperator::Except, 1)
            } else {
                break;
            };
            let prec = if self.opts.legacy_setops_precedence {
                1
            } else {
                prec
            };
            if prec < min_prec {
                break;
            }
            self.bump();
            let quantifier = self.set_quantifier();
            let (right, right_height) = self.measured(|p| p.nested(|p| p.set_ops(prec + 1)))?;
            height = (height + 1).max(right_height);
            self.grow(height)?;
            let span = left.span.to(right.span);
            left = SetExpr {
                kind: SetExprKind::SetOp {
                    op,
                    quantifier,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn set_quantifier(&mut self) -> Option<SetQuantifier> {
        if self.eat_kw("DISTINCT") {
            Some(SetQuantifier::Distinct)
        } else if self.eat_kw("ALL") {
            Some(SetQuantifier::All)
        } else {
            None
        }
    }

    fn query_primary(&mut self) -> PR<SetExpr> {
        let start = self.start();
        let kind = if self.kw_at(0, "SELECT") {
            SetExprKind::Select(Box::new(self.select(start, None)?))
        } else if self.kw_at(0, "FROM") {
            return self.query_from_first();
        } else if self.kw_at(0, "VALUES") {
            SetExprKind::Values(self.values()?)
        } else if self.eat_kw("TABLE") {
            SetExprKind::Table(self.object_name()?)
        } else if self.eat_sym(Sym::LParen) {
            let q = self.query()?;
            self.expect_sym(Sym::RParen)?;
            SetExprKind::Query(Box::new(q))
        } else {
            return self.fail();
        };
        Ok(SetExpr {
            kind,
            span: self.span_from(start),
        })
    }

    #[inline(never)]
    fn values(&mut self) -> PR<Values> {
        let start = self.start();
        self.expect_kw("VALUES")?;
        let rows = self.comma_sep(Self::expr)?;
        let alias = self.table_alias()?;
        Ok(Values {
            rows,
            alias,
            span: self.span_from(start),
        })
    }

    /// `FROM relations SELECT ... [SELECT ...]`: several bodies are a
    /// `UNION ALL` over the same `FROM`, as in Spark.
    #[inline(never)]
    fn query_from_first(&mut self) -> PR<SetExpr> {
        let start = self.start();
        let from = self.clause_from()?;
        let mut out: Option<SetExpr> = None;
        loop {
            if !self.kw_at(0, "SELECT") {
                break;
            }
            // The `FROM` clause comes first, so every body's span starts there.
            let select = self.select(start, Some(from.clone()))?;
            let span = self.span_from(start);
            let mut q = Self::bare_query(
                SetExpr {
                    kind: SetExprKind::Select(Box::new(select)),
                    span,
                },
                start,
            );
            self.organization(&mut q)?;
            q.span = self.span_from(start);
            let organized = !(q.order_by.is_empty()
                && q.cluster_by.is_empty()
                && q.distribute_by.is_empty()
                && q.sort_by.is_empty()
                && q.windows.is_empty()
                && q.limit.is_none()
                && q.offset.is_none());
            let body = if !organized {
                q.body
            } else {
                SetExpr {
                    span: q.span,
                    kind: SetExprKind::Query(Box::new(q)),
                }
            };
            out = Some(match out {
                None => body,
                Some(left) => {
                    self.grow(1)?;
                    SetExpr {
                        span: left.span.to(body.span),
                        kind: SetExprKind::SetOp {
                            op: SetOperator::Union,
                            quantifier: Some(SetQuantifier::All),
                            left: Box::new(left),
                            right: Box::new(body),
                        },
                    }
                }
            });
        }
        match out {
            Some(mut e) => {
                e.span = self.span_from(start);
                Ok(e)
            }
            // `FROM t` alone selects every column (an empty projection).
            None => Ok(SetExpr {
                span: self.span_from(start),
                kind: SetExprKind::Select(Box::new(Select {
                    from_first: true,
                    hints: vec![],
                    quantifier: None,
                    projection: vec![],
                    from: from.relations,
                    lateral_views: from.lateral_views,
                    pivot: from.pivot,
                    unpivot: from.unpivot,
                    selection: None,
                    group_by: None,
                    having: None,
                    windows: vec![],
                    qualify: None,
                    span: self.span_from(start),
                })),
            }),
        }
    }

    /// `FROM relation, ... lateralView* pivot? unpivot?`.
    #[inline(never)]
    fn clause_from(&mut self) -> PR<FromClause> {
        self.expect_kw("FROM")?;
        let relations = self.comma_sep(Self::relation)?;
        let lateral_views = self.lateral_views()?;
        let pivot = if self.kw_at(0, "PIVOT") {
            Some(self.pivot()?)
        } else {
            None
        };
        let unpivot = if self.kw_at(0, "UNPIVOT") {
            Some(self.unpivot()?)
        } else {
            None
        };
        Ok(FromClause {
            relations,
            lateral_views,
            pivot,
            unpivot,
        })
    }

    /// A query specification from `SELECT` on; `from` is the clause of a
    /// `FROM ... SELECT` statement.
    fn select(&mut self, start: usize, from: Option<FromClause>) -> PR<Select> {
        self.expect_kw("SELECT")?;
        let mut hints = vec![];
        while self.kind(0) == Some(TokenKind::HintStart) {
            self.hints(&mut hints)?;
        }
        // `SELECT all, b`: `ALL` / `DISTINCT` before `,` or `FROM` is a column.
        let column_next = !self.opts.enforce_reserved_keywords
            && (self.sym_at(1, Sym::Comma) || self.kw_at(1, "FROM") || self.tok(1).is_none());
        let quantifier = if column_next {
            None
        } else {
            self.set_quantifier()
        };
        let projection = self.comma_sep(Self::select_item)?;
        let from_first = from.is_some();
        let from = match from {
            Some(f) => f,
            None if self.kw_at(0, "FROM") => self.clause_from()?,
            None => FromClause::default(),
        };
        let mut lateral_views = from.lateral_views;
        lateral_views.extend(self.lateral_views()?);
        let selection = self.opt_clause("WHERE")?;
        let group_by = if self.kw_at(0, "GROUP") && self.kw_at(1, "BY") {
            Some(self.group_by()?)
        } else {
            None
        };
        let having = self.opt_clause("HAVING")?;
        let windows = self.window_clause()?;
        let qualify = self.opt_clause("QUALIFY")?;
        Ok(Select {
            from_first,
            hints,
            quantifier,
            projection,
            from: from.relations,
            lateral_views,
            pivot: from.pivot,
            unpivot: from.unpivot,
            selection,
            group_by,
            having,
            windows,
            qualify,
            span: self.span_from(start),
        })
    }

    fn opt_clause(&mut self, kw: &str) -> PR<Option<Expr>> {
        if self.eat_kw(kw) {
            self.expr().map(Some)
        } else {
            Ok(None)
        }
    }

    /// `/*+ name[(params)] [,] ... */`.
    #[inline(never)]
    fn hints(&mut self, out: &mut Vec<Hint>) -> PR<()> {
        self.bump();
        loop {
            let start = self.start();
            let name = self.any_ident()?;
            let params = if self.sym_at(0, Sym::LParen) {
                self.paren_list(Self::primary_expr)?
            } else {
                vec![]
            };
            out.push(Hint {
                name,
                params,
                span: self.span_from(start),
            });
            self.eat_sym(Sym::Comma);
            if self.kind(0) == Some(TokenKind::HintEnd) {
                self.bump();
                return Ok(());
            }
        }
    }

    fn select_item(&mut self) -> PR<SelectItem> {
        let start = self.start();
        let expr = self.expr()?;
        let alias = if !self.item_end_at(1) && self.eat_kw("AS") {
            Some(if self.sym_at(0, Sym::LParen) {
                Alias::List(self.ident_list()?)
            } else {
                Alias::Name(self.name_part(true)?)
            })
        } else if self.sym_at(0, Sym::LParen) {
            Some(Alias::List(self.ident_list()?))
        } else {
            self.implicit_alias(true)?.map(Alias::Name)
        };
        Ok(SelectItem {
            expr,
            alias,
            span: self.span_from(start),
        })
    }

    fn sort_item(&mut self, allow_all: bool) -> PR<SortItem> {
        let start = self.start();
        let expr = if allow_all
            && self.kw_at(0, "ALL")
            && !self.sym_at(1, Sym::Dot)
            && !self.sym_at(1, Sym::LParen)
        {
            self.column_named_all()
        } else {
            self.expr()?
        };
        let asc = if self.eat_kw("ASC") {
            Some(true)
        } else if self.eat_kw("DESC") {
            Some(false)
        } else {
            None
        };
        let mut nulls_first = None;
        if self.eat_kw("NULLS") {
            nulls_first = Some(self.eat_kw("FIRST"));
            if nulls_first == Some(false) {
                self.expect_kw("LAST")?;
            }
        }
        Ok(SortItem {
            expr,
            asc,
            nulls_first,
            span: self.span_from(start),
        })
    }

    /// `ALL` in `ORDER BY ALL` / `GROUP BY ALL`: a column named `ALL`.
    fn column_named_all(&mut self) -> Expr {
        let span = self.tok(0).map_or_else(Span::default, |t| t.span);
        self.bump();
        Expr {
            kind: crate::ast::ExprKind::Column(vec![Ident {
                value: span.text(self.src).to_owned(),
                quoted: false,
                span,
            }]),
            span,
        }
    }

    // ---- GROUP BY -------------------------------------------------------

    #[inline(never)]
    fn group_by(&mut self) -> PR<GroupBy> {
        let start = self.start();
        self.pos += 2;
        let items = self.comma_sep(Self::group_item)?;
        let with = if self.kw_at(0, "WITH") && self.any_kw_at(1, &["ROLLUP", "CUBE"]) {
            let rollup = self.kw_at(1, "ROLLUP");
            self.pos += 2;
            Some(if rollup {
                GroupWith::Rollup
            } else {
                GroupWith::Cube
            })
        } else if self.kw_at(0, "GROUPING") && self.kw_at(1, "SETS") {
            self.pos += 2;
            Some(GroupWith::GroupingSets(
                self.paren_list(Self::grouping_set)?,
            ))
        } else {
            None
        };
        Ok(GroupBy {
            items,
            with,
            span: self.span_from(start),
        })
    }

    fn group_item(&mut self) -> PR<GroupItem> {
        let start = self.start();
        let kind = if self.any_kw_at(0, &["ROLLUP", "CUBE"]) && self.sym_at(1, Sym::LParen) {
            let rollup = self.kw_at(0, "ROLLUP");
            self.bump();
            let sets = self.paren_list(Self::grouping_set)?;
            if rollup {
                GroupItemKind::Rollup(sets)
            } else {
                GroupItemKind::Cube(sets)
            }
        } else if self.kw_at(0, "GROUPING") && self.kw_at(1, "SETS") {
            self.pos += 2;
            GroupItemKind::GroupingSets(self.paren_list(|p| {
                let analytics = p.any_kw_at(0, &["ROLLUP", "CUBE"]) && p.sym_at(1, Sym::LParen)
                    || p.kw_at(0, "GROUPING") && p.kw_at(1, "SETS");
                if analytics {
                    p.nested(Self::group_item)
                } else {
                    let set = p.grouping_set()?;
                    Ok(GroupItem {
                        span: set.span,
                        kind: GroupItemKind::Set(set),
                    })
                }
            })?)
        } else if self.kw_at(0, "ALL") && !self.sym_at(1, Sym::Dot) && !self.sym_at(1, Sym::LParen)
        {
            GroupItemKind::Expr(self.column_named_all())
        } else {
            GroupItemKind::Expr(self.expr()?)
        };
        Ok(GroupItem {
            kind,
            span: self.span_from(start),
        })
    }

    /// `(a, b)`, `()` or `a`. A parenthesized list is a set; `(a + b) * 2`
    /// is an expression, continued from `(a + b)` without re-parsing it.
    fn grouping_set(&mut self) -> PR<GroupingSet> {
        let start = self.start();
        if !self.sym_at(0, Sym::LParen) || self.query_follows(1) {
            let e = self.expr()?;
            return Ok(self.grouping_set_of(vec![e], start));
        }
        self.bump();
        if self.eat_sym(Sym::RParen) {
            return Ok(self.grouping_set_of(vec![], start));
        }
        let (first, height) = self.measured(Self::expr)?;
        if self.eat_sym(Sym::Comma) {
            let mut exprs = vec![first];
            exprs.extend(self.comma_sep(Self::expr)?);
            self.expect_sym(Sym::RParen)?;
            return Ok(self.grouping_set_of(exprs, start));
        }
        self.expect_sym(Sym::RParen)?;
        if self.item_end_at(0) {
            return Ok(self.grouping_set_of(vec![first], start));
        }
        let nested = Expr {
            span: self.span_from(start),
            kind: crate::ast::ExprKind::Nested(Box::new(first)),
        };
        let e = self.expr_from(nested, height)?;
        Ok(self.grouping_set_of(vec![e], start))
    }

    fn grouping_set_of(&self, exprs: Vec<Expr>, start: usize) -> GroupingSet {
        GroupingSet {
            exprs,
            span: self.span_from(start),
        }
    }

    // ---- windows ----------------------------------------------------------

    fn window_clause(&mut self) -> PR<Vec<NamedWindow>> {
        if !self.eat_kw("WINDOW") {
            return Ok(vec![]);
        }
        self.comma_sep(|p| {
            let start = p.start();
            let name = p.name_part(false)?;
            p.expect_kw("AS")?;
            let window = p.window_ref()?;
            Ok(NamedWindow {
                name,
                window,
                span: p.span_from(start),
            })
        })
    }

    // ---- relations ----------------------------------------------------------

    fn relation(&mut self) -> PR<Relation> {
        self.nested(|p| {
            let start = p.start();
            let lateral = p.eat_kw("LATERAL");
            let primary = p.table_factor()?;
            let mut extensions = vec![];
            loop {
                if p.kw_at(0, "PIVOT") && p.sym_at(1, Sym::LParen) {
                    extensions.push(RelationExt::Pivot(p.pivot()?));
                } else if p.kw_at(0, "UNPIVOT")
                    && (p.sym_at(1, Sym::LParen) || p.any_kw_at(1, &["INCLUDE", "EXCLUDE"]))
                {
                    extensions.push(RelationExt::Unpivot(p.unpivot()?));
                } else if let Some(j) = p.join()? {
                    extensions.push(RelationExt::Join(j));
                } else {
                    break;
                }
            }
            Ok(Relation {
                lateral,
                primary,
                extensions,
                span: p.span_from(start),
            })
        })
    }

    /// Whether token `n` and the parentheses after it open a query.
    fn query_follows(&self, n: usize) -> bool {
        let at = self.pos + n;
        let after = self.after_parens.get(at).copied().unwrap_or(at);
        self.any_kw_at(after - self.pos, QUERY_START)
    }

    fn table_alias(&mut self) -> PR<Option<TableAlias>> {
        let start = self.start();
        let Some(name) = self.opt_alias_name()? else {
            return Ok(None);
        };
        let columns = if self.sym_at(0, Sym::LParen) {
            self.ident_list()?
        } else {
            vec![]
        };
        Ok(Some(TableAlias {
            name,
            columns,
            span: self.span_from(start),
        }))
    }

    fn table_factor(&mut self) -> PR<TableFactor> {
        let start = self.start();
        let kind = if self.kw_at(0, "VALUES") && !self.item_end_at(1) {
            let values = self.values()?;
            return Ok(TableFactor {
                kind: TableFactorKind::Values(values.rows),
                alias: values.alias,
                span: values.span,
            });
        } else if self.sym_at(0, Sym::LParen) {
            let save = self.pos;
            let mut sub = None;
            if self.query_follows(1) {
                self.bump();
                match self.query() {
                    Ok(q) if self.eat_sym(Sym::RParen) => sub = Some(q),
                    Err(Fail::Fatal(e)) => return Err(Fail::Fatal(e)),
                    _ => self.pos = save,
                }
            }
            match sub {
                Some(q) => TableFactorKind::Subquery(Box::new(q)),
                None => {
                    self.expect_sym(Sym::LParen)?;
                    let r = self.relation()?;
                    self.expect_sym(Sym::RParen)?;
                    TableFactorKind::Nested(Box::new(r))
                }
            }
        } else {
            let name = self.table_name()?;
            if self.sym_at(0, Sym::LParen) {
                self.bump();
                let args = self.function_args()?;
                self.expect_sym(Sym::RParen)?;
                TableFactorKind::Function { name, args }
            } else {
                TableFactorKind::Table(name)
            }
        };
        let alias = self.table_alias()?;
        Ok(TableFactor {
            kind,
            alias,
            span: self.span_from(start),
        })
    }

    /// A join, if one starts here.
    #[inline(never)]
    fn join(&mut self) -> PR<Option<Join>> {
        let start = self.start();
        let natural = self.kw_at(0, "NATURAL");
        let n = usize::from(natural);
        let at = |p: &Self, i: usize, kw: &str| p.kw_at(n + i, kw);
        let (kind, words) = if at(self, 0, "JOIN") {
            (JoinKind::Inner, 0)
        } else if at(self, 0, "INNER") {
            (JoinKind::Inner, 1)
        } else if at(self, 0, "CROSS") {
            (JoinKind::Cross, 1)
        } else if at(self, 0, "LEFT") && at(self, 1, "SEMI") || at(self, 0, "SEMI") {
            (JoinKind::LeftSemi, 1 + usize::from(at(self, 0, "LEFT")))
        } else if at(self, 0, "LEFT") && at(self, 1, "ANTI") || at(self, 0, "ANTI") {
            (JoinKind::LeftAnti, 1 + usize::from(at(self, 0, "LEFT")))
        } else if let Some(kind) = [
            ("LEFT", JoinKind::Left),
            ("RIGHT", JoinKind::Right),
            ("FULL", JoinKind::Full),
        ]
        .into_iter()
        .find_map(|(kw, k)| at(self, 0, kw).then_some(k))
        {
            (kind, 1 + usize::from(at(self, 1, "OUTER")))
        } else {
            if natural {
                self.bump();
                return self.fail();
            }
            return Ok(None);
        };
        self.pos += n + words;
        self.expect_kw("JOIN")?;
        let lateral = self.eat_kw("LATERAL");
        let right = Box::new(self.table_factor()?);
        let constraint = if natural {
            JoinConstraint::None
        } else if self.eat_kw("ON") {
            JoinConstraint::On(self.expr()?)
        } else if self.eat_kw("USING") {
            JoinConstraint::Using(self.ident_list()?)
        } else {
            JoinConstraint::None
        };
        Ok(Some(Join {
            kind,
            natural,
            lateral,
            right,
            constraint,
            span: self.span_from(start),
        }))
    }

    #[inline(never)]
    fn pivot(&mut self) -> PR<Pivot> {
        let start = self.start();
        self.expect_kw("PIVOT")?;
        self.expect_sym(Sym::LParen)?;
        let aggregates = self.comma_sep(Self::select_item)?;
        self.expect_kw("FOR")?;
        let columns = if self.sym_at(0, Sym::LParen) {
            self.paren_list(|p| p.name_part(false))?
        } else {
            vec![self.name_part(false)?]
        };
        self.expect_kw("IN")?;
        let values = self.paren_list(|p| {
            let start = p.start();
            let expr = p.expr()?;
            let alias = p.opt_alias_name()?.map(Alias::Name);
            Ok(SelectItem {
                expr,
                alias,
                span: p.span_from(start),
            })
        })?;
        self.expect_sym(Sym::RParen)?;
        let alias = self.opt_alias_name()?;
        Ok(Pivot {
            aggregates,
            columns,
            values,
            alias,
            span: self.span_from(start),
        })
    }

    #[inline(never)]
    fn unpivot(&mut self) -> PR<Unpivot> {
        let start = self.start();
        self.expect_kw("UNPIVOT")?;
        let mut include_nulls = None;
        if self.any_kw_at(0, &["INCLUDE", "EXCLUDE"]) {
            include_nulls = Some(self.kw_at(0, "INCLUDE"));
            self.bump();
            self.expect_kw("NULLS")?;
        }
        self.expect_sym(Sym::LParen)?;
        let multi = self.sym_at(0, Sym::LParen);
        let value_columns = if multi {
            self.ident_list()?
        } else {
            vec![self.ident()?]
        };
        self.expect_kw("FOR")?;
        let name_column = self.ident()?;
        self.expect_kw("IN")?;
        let columns = self.paren_list(|p| {
            let start = p.start();
            let columns = if multi {
                p.paren_list(Self::object_name)?
            } else {
                vec![p.object_name()?]
            };
            let alias = p.opt_alias_name()?;
            Ok(UnpivotColumns {
                columns,
                alias,
                span: p.span_from(start),
            })
        })?;
        self.expect_sym(Sym::RParen)?;
        let alias = self.opt_alias_name()?;
        Ok(Unpivot {
            include_nulls,
            value_columns,
            name_column,
            columns,
            alias,
            span: self.span_from(start),
        })
    }

    /// `LATERAL VIEW [OUTER] f(args) table [AS] col, ...`*.
    #[inline(never)]
    fn lateral_views(&mut self) -> PR<Vec<LateralView>> {
        let mut out = vec![];
        while self.kw_at(0, "LATERAL") && self.kw_at(1, "VIEW") {
            let start = self.start();
            self.pos += 2;
            let outer = self.eat_kw("OUTER");
            let function = self.object_name()?;
            self.expect_sym(Sym::LParen)?;
            let args = if self.sym_at(0, Sym::RParen) {
                vec![]
            } else {
                self.comma_sep(Self::expr)?
            };
            self.expect_sym(Sym::RParen)?;
            // Spark's grammar needs the table name: `AS t` names it `AS`.
            let table = self.any_ident()?;
            let columns = if self.eat_kw("AS") {
                self.comma_sep(Self::any_ident)?
            } else if let Some(first) = self.implicit_alias(false)? {
                let mut columns = vec![first];
                while self.eat_sym(Sym::Comma) {
                    columns.push(self.ident()?);
                }
                columns
            } else {
                vec![]
            };
            out.push(LateralView {
                outer,
                function,
                args,
                table,
                columns,
                span: self.span_from(start),
            });
        }
        Ok(out)
    }
}

/// The parts of a `FROM` clause.
#[derive(Clone, Default)]
struct FromClause {
    relations: Vec<Relation>,
    lateral_views: Vec<LateralView>,
    pivot: Option<Pivot>,
    unpivot: Option<Unpivot>,
}
