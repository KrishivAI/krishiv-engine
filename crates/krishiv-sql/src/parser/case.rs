// SPDX-License-Identifier: Apache-2.0
//! Keywords read as columns and the `CASE` expression (`impl Parser`, split
//! from `expr.rs`): whether `NOT`, `CASE` or another keyword at the cursor is
//! the keyword or a column, as ANTLR's prediction takes it, and the `CASE`
//! expression with its per-position memo.

use super::expr::{BRANCH_END, boxed};
use super::{CaseMemo, CasePart, Checkpoint, Fail, PR, Parser};
use crate::ast::{Expr, ExprKind, WhenClause};
use crate::keywords::CLAUSE_WORD;
use crate::lexer::{Sym, TokenKind};

impl Parser<'_> {
    /// Whether the `NOT` here is a column, as Spark reads `SELECT NOT`,
    /// `NOT = 1`, `NOT.x`, `NOT::INT` and `NOT[0]`: no operand can follow
    /// it (a postfix operator does instead). `+` and `-` do
    /// (`NOT + 1` is `NOT (+1)`). Nor can `IS`, `AND`, `OR` or the end of a
    /// `CASE` branch (`NOT IS NULL`, `WHEN y < not THEN`),
    /// and before `->` it is a lambda parameter.
    pub(super) fn not_is_column(&self) -> bool {
        let operand_sign = self.sym_at(1, Sym::Plus) || self.sym_at(1, Sym::Minus);
        !operand_sign
            && (self.reserved_as_column()
                || self.sym_at(1, Sym::Dot)
                || self.sym_at(1, Sym::DoubleColon)
                || self.sym_at(1, Sym::LBracket)
                || self.sym_at(1, Sym::Arrow)
                || (!self.opts.enforce_reserved_keywords
                    && self.any_kw_at(1, BRANCH_END)
                    && !self.not_takes_operand(1)))
    }

    /// Whether the `NOT` before the word at offset `w` (one of
    /// [`BRANCH_END`]) is the operator with that word as a column operand,
    /// as ANTLR takes it when the operator reading can go on after the
    /// word; otherwise `NOT` is the column and the word continues it.
    /// - A word that closes the current `CASE` part does so unless the
    ///   operator reading goes on: a second closer or a continuation after
    ///   it (`ELSE not END` is the column, `THEN not END END` and `THEN NOT
    ///   end IS NULL END` the operator).
    /// - At a select item's top level the item may end after the word or
    ///   after a bare alias (`SELECT NOT END`, `SELECT NOT AND x`).
    /// - Elsewhere the operator when an expression may end or go on after
    ///   the word (`WHERE NOT END`, `x -> NOT AND`, `WHEN NOT END THEN`,
    ///   `NOT IS IN (1)`, `NOT END[0]`), the column when only the word's own
    ///   reading continues (`WHERE NOT IS NULL`, `WHERE NOT AND x`).
    pub(super) fn not_takes_operand(&self, w: usize) -> bool {
        let part = self
            .case_part
            .filter(|(start, _)| self.depth_since(*start) == 0);
        if let Some((_, part)) = part {
            if self.any_kw_at(w, part.closers()) {
                return self.any_kw_at(w + 1, part.closers())
                    || self.expr_continues_at(w + 1)
                    || (self.binary_op_at(w + 1) && self.part_closer_after(w + 1, part));
            }
        } else if self
            .item_start
            .is_some_and(|start| self.depth_since(start) == 0)
            && self.item_ends_after(w + 1)
        {
            return true;
        }
        self.expr_may_end_at(w + 1) || self.expr_continues_at(w + 1)
    }

    /// Whether the token at offset `n` is a binary operator, `AND` or `OR`.
    pub(super) fn binary_op_at(&self, n: usize) -> bool {
        use Sym::{
            Amp, Caret, Concat, Eq, Ge, Gt, Le, Lt, Minus, Neq, NullSafeEq, Percent, Pipe, Plus,
            Slash, Star,
        };
        self.any_kw_at(n, &["AND", "OR"])
            || matches!(self.kind(n), Some(TokenKind::Sym(s)) if [
                Amp, Caret, Concat, Eq, Ge, Gt, Le, Lt, Minus, Neq, NullSafeEq, Percent, Pipe,
                Plus, Slash, Star,
            ]
            .contains(&s))
    }

    /// Whether a word closing `part` follows offset `n` in the same brackets
    /// and outside any `CASE` that starts after `n`: the operator reading of
    /// `NOT` can then end the part (`THEN NOT end = 1 END`, Spark
    /// `NOT ('end = 1)`), where `ELSE not END AND x FROM t` has no closer
    /// left, so `NOT` is the column there.
    pub(super) fn part_closer_after(&self, n: usize, part: CasePart) -> bool {
        let (mut depth, mut cases) = (0i64, 0usize);
        let mut i = n;
        while let Some(kind) = self.kind(i) {
            match kind {
                TokenKind::Sym(Sym::LParen | Sym::LBracket) => depth += 1,
                TokenKind::Sym(Sym::RParen | Sym::RBracket) => {
                    depth -= 1;
                    if depth < 0 {
                        return false;
                    }
                }
                TokenKind::Sym(Sym::Semi) if depth == 0 => return false,
                TokenKind::Word if depth == 0 && self.kw_at(i, "CASE") => cases += 1,
                TokenKind::Word if depth == 0 && cases > 0 && self.kw_at(i, "END") => cases -= 1,
                TokenKind::Word
                    if depth == 0 && cases == 0 && self.any_kw_at(i, part.closers()) =>
                {
                    return true;
                }
                _ => {}
            }
            i += 1;
        }
        false
    }

    /// Whether the token at offset `n` continues an expression before it: a
    /// field or element access, call arguments, a string (a typed literal),
    /// `::`, `COLLATE`, `DIV` or a predicate (`IS NULL`, `IN`, `BETWEEN`,
    /// `LIKE`, ..., also after `NOT`).
    pub(super) fn expr_continues_at(&self, n: usize) -> bool {
        const PREDICATES: &[&str] = &["IN", "BETWEEN", "LIKE", "ILIKE", "RLIKE", "REGEXP"];
        const IS_TAIL: &[&str] = &["NULL", "NOT", "TRUE", "FALSE", "UNKNOWN", "DISTINCT"];
        self.sym_at(n, Sym::Dot)
            || self.sym_at(n, Sym::LBracket)
            // The word is then a call or a typed literal (`NOT and(x)`).
            || self.sym_at(n, Sym::LParen)
            || self.single_string_at(n)
            || self.sym_at(n, Sym::DoubleColon)
            || self.any_kw_at(n, PREDICATES)
            || self.any_kw_at(n, &["COLLATE", "DIV"])
            || (self.kw_at(n, "IS") && self.any_kw_at(n + 1, IS_TAIL))
            || (self.kw_at(n, "NOT") && self.any_kw_at(n + 1, PREDICATES))
    }

    /// The bracket depth of the cursor relative to the token `start`.
    pub(super) fn depth_since(&self, start: usize) -> i64 {
        let mut depth = 0i64;
        for t in self.toks.get(start..self.pos).unwrap_or_default() {
            match t.kind {
                TokenKind::Sym(Sym::LParen | Sym::LBracket) => depth += 1,
                TokenKind::Sym(Sym::RParen | Sym::RBracket) => depth -= 1,
                _ => {}
            }
        }
        depth
    }

    /// Whether the token at offset `n` may follow a complete expression:
    /// the end of a list or statement, a clause, a `CASE` keyword, `AND`,
    /// `OR`, `AS` or a binary operator.
    pub(super) fn expr_may_end_at(&self, n: usize) -> bool {
        use Sym::{
            Amp, Caret, Concat, Eq, Ge, Gt, Le, Lt, Minus, Neq, NullSafeEq, Percent, Pipe, Plus,
            RBracket, Slash, Star,
        };
        const WORDS: &[&str] = &["FROM", "THEN", "ELSE", "END", "WHEN", "AND", "OR", "AS"];
        self.item_end_at(n)
            || self.any_kw_at(n, WORDS)
            || (self.kind(n) == Some(TokenKind::Word) && self.class(n) & CLAUSE_WORD != 0)
            || matches!(self.kind(n), Some(TokenKind::Sym(s)) if [
                Amp, Caret, Concat, Eq, Ge, Gt, Le, Lt, Minus, Neq, NullSafeEq, Percent, Pipe,
                Plus, RBracket, Slash, Star,
            ]
            .contains(&s))
    }

    /// Whether the select item can end at offset `n`, or after a bare alias
    /// there.
    pub(super) fn item_ends_after(&self, n: usize) -> bool {
        let ends = |n: usize| {
            self.item_end_at(n)
                || self.kw_at(n, "FROM")
                || (self.kind(n) == Some(TokenKind::Word) && self.class(n) & CLAUSE_WORD != 0)
        };
        let alias = matches!(self.kind(n), Some(TokenKind::Word | TokenKind::QuotedIdent));
        ends(n) || (alias && ends(n + 1))
    }

    /// Whether `CASE` here is a column, as Spark reads `SELECT CASE` and
    /// `SELECT CASE END` (a column `CASE` aliased `END`) unless keywords are
    /// enforced: no `WHEN` or operand can follow it.
    pub(super) fn case_is_column(&self) -> bool {
        let alias_then_end =
            self.kind(1) == Some(TokenKind::Word) && !self.kw_at(1, "WHEN") && self.item_end_at(2);
        !self.opts.enforce_reserved_keywords && (self.reserved_as_column() || alias_then_end)
    }

    /// Whether `CASE` here may be either a `CASE` expression or a column:
    /// the next token continues both (`CASE - x WHEN ...` against
    /// `case - y`, `CASE from - 1 WHEN ...` against `case FROM t`), so
    /// `case_or_column` tries the expression first.
    pub(super) fn case_may_be_column(&self) -> bool {
        !self.kw_at(1, "WHEN") && self.reserved_column_here()
    }

    /// `CASE` that may be a column: the `CASE` expression if it parses, as
    /// ANTLR's prediction takes it, else `None` and `CASE` is a column
    /// (`(case, y) -> case - y`, `CASE WHEN case IS NULL THEN ...`, `ON
    /// t.id = case - 1 WHEN MATCHED`).
    pub(super) fn case_or_column(&mut self) -> PR<Option<ExprKind>> {
        if self.case_column_at == Some(self.pos) {
            return Ok(None);
        }
        let at = self.checkpoint();
        match self.case_expr() {
            Ok(kind) => {
                self.commit(at);
                Ok(Some(kind))
            }
            Err(Fail::Fatal(e)) => Err(Fail::Fatal(e)),
            Err(Fail::Syntax) => {
                self.restore_keeping_far(at);
                Ok(None)
            }
        }
    }

    /// Rewinds to `at` after a failed speculative read, keeping its furthest
    /// failure when that is further than any before: Spark reports the
    /// furthest token any reading reached.
    pub(super) fn restore_keeping_far(&mut self, at: Checkpoint) {
        let (far, matched, missing) = (self.far, self.matched, self.missing);
        self.restore(at);
        if far > self.far {
            self.far = far;
            self.matched = matched;
            self.missing = missing;
        }
    }

    /// Whether the reserved word here is a column, because no keyword
    /// reading can continue with the next token (`all, b`, `all * 2`,
    /// `all FROM t`).
    /// Spark accepts every keyword as an identifier unless keywords are
    /// enforced.
    pub(super) fn reserved_as_column(&self) -> bool {
        use Sym::{
            Amp, Caret, Comma, Concat, Eq, Ge, Gt, Le, Lt, Neq, Percent, Pipe, Plus, RParen, Semi,
            Slash, Star,
        };
        if self.opts.enforce_reserved_keywords || self.kind(0) != Some(TokenKind::Word) {
            return false;
        }
        match self.kind(1) {
            None => true,
            Some(TokenKind::Word) => self.kw_at(1, "FROM"),
            Some(TokenKind::Sym(s)) => [
                Amp, Caret, Comma, Concat, Eq, Ge, Gt, Le, Lt, Neq, Percent, Pipe, Plus, RParen,
                Semi, Slash, Star,
            ]
            .contains(&s),
            Some(_) => false,
        }
    }

    /// Whether the reserved word starting a primary is a column, as Spark
    /// reads every keyword unless keywords are enforced: no keyword reading
    /// of it continues with the next token, which here may also be `-`, a
    /// field or element access, `]`, an infix keyword or a `CASE` branch
    /// keyword (`left - right`, `all.a`, `on[0]`, `a[from]`, `from IS NULL`,
    /// `WHEN x < right THEN`: the bodies of lambdas whose parameters are
    /// keywords).
    pub(super) fn reserved_column_here(&self) -> bool {
        const FOLLOW: &[&str] = &[
            "AND", "OR", "IS", "IN", "NOT", "BETWEEN", "LIKE", "ILIKE", "RLIKE", "REGEXP", "DIV",
            "WHEN", "THEN", "ELSE", "END",
        ];
        if self.reserved_as_column() {
            return true;
        }
        if self.opts.enforce_reserved_keywords || self.kind(0) != Some(TokenKind::Word) {
            return false;
        }
        // The operand of a `NOT` that takes it (`SELECT NOT AND x` is `NOT
        // and AS x`, `WHERE NOT END`; see `not_takes_operand`).
        if self.kw_before("NOT") && self.any_kw_at(0, BRANCH_END) && self.not_takes_operand(0) {
            return true;
        }
        self.sym_at(1, Sym::Minus)
            || self.sym_at(1, Sym::Dot)
            || self.sym_at(1, Sym::LBracket)
            || self.sym_at(1, Sym::DoubleColon)
            || self.sym_at(1, Sym::RBracket)
            || self.sym_at(1, Sym::NullSafeEq)
            || self.any_kw_at(1, FOLLOW)
    }

    #[inline(never)]
    pub(super) fn case_expr(&mut self) -> PR<ExprKind> {
        let key = self.pos;
        match self.case_memo.get(&key) {
            Some(CaseMemo::Parsed(kind, end, height)) => {
                let (kind, end, height) = (kind.clone(), *end, *height);
                self.grow(height)?;
                self.pos = end;
                return Ok(kind);
            }
            Some(&CaseMemo::Failed {
                far,
                matched,
                missing,
            }) => {
                if far > self.far {
                    self.far = far;
                    self.matched = matched;
                    self.missing = missing;
                }
                return Err(Fail::Syntax);
            }
            Some(CaseMemo::Seen) | None => {}
        }
        let deferred = self.deferred.is_some();
        let outer = self.case_part.take();
        let done = self.measured(Self::case_expr_uncached);
        self.case_part = outer;
        match &done {
            Ok((kind, height)) if self.deferred.is_some() == deferred => {
                let memo = if self.case_memo.contains_key(&key) {
                    CaseMemo::Parsed(kind.clone(), self.pos, *height)
                } else {
                    CaseMemo::Seen
                };
                self.case_memo.insert(key, memo);
            }
            Err(Fail::Syntax) => {
                let memo = CaseMemo::Failed {
                    far: self.far,
                    matched: self.matched,
                    missing: self.missing,
                };
                self.case_memo.insert(key, memo);
            }
            _ => {}
        }
        done.map(|(kind, _)| kind)
    }

    /// Runs `f` with `part` as the current `CASE` part.
    pub(super) fn in_case_part<T>(&mut self, part: CasePart, f: impl FnOnce(&mut Self) -> T) -> T {
        let outer = self.case_part.replace((self.pos, part));
        let out = f(self);
        self.case_part = outer;
        out
    }

    pub(super) fn case_expr_uncached(&mut self) -> PR<ExprKind> {
        self.bump();
        // `CASE when - 1 WHEN 0 THEN ...`: a `WHEN` that may be a column
        // starts the operand when the searched form does not parse (the
        // searched form's error stands when neither does).
        if self.kw_at(0, "WHEN") && self.reserved_column_here() {
            let at = self.checkpoint();
            match self.case_branches(None) {
                Err(Fail::Syntax) => {}
                done => {
                    self.commit(at);
                    return done;
                }
            }
            let failed = self.checkpoint();
            self.restore(at);
            let at = self.checkpoint();
            let operand = match self.in_case_part(CasePart::Operand, Self::expr) {
                Ok(e) => Some(boxed(e)),
                Err(Fail::Fatal(e)) => return Err(Fail::Fatal(e)),
                Err(Fail::Syntax) => None,
            };
            if operand.is_some()
                && let Ok(kind) = self.case_branches(operand)
            {
                self.commit(at);
                return Ok(kind);
            }
            self.restore(failed);
            return self.fail();
        }
        if self.kw_at(0, "WHEN") {
            return self.case_branches(None);
        }
        // A `CASE` that starts the operand is read as one; when that leaves
        // no `WHEN` (`CASE case - 1 WHEN 0 THEN 1 END` in a lambda over
        // `case`), it is the column instead.
        let operand_start = self.pos;
        let at = self.checkpoint();
        let operand = self.in_case_part(CasePart::Operand, Self::expr)?;
        let case_first = self.toks.get(operand_start).is_some_and(|t| {
            t.kind == TokenKind::Word && crate::keywords::is_kw(t.span.text(self.src), "CASE")
        });
        if !self.kw_at(0, "WHEN") && case_first {
            let first = self.checkpoint();
            self.restore(at);
            let outer = self.case_column_at.replace(operand_start);
            let again = self.in_case_part(CasePart::Operand, Self::expr);
            self.case_column_at = outer;
            if let Ok(operand) = again
                && self.kw_at(0, "WHEN")
            {
                return self.case_branches(Some(boxed(operand)));
            }
            self.restore(first);
            return self.case_branches(Some(boxed(operand)));
        }
        self.commit(at);
        self.case_branches(Some(boxed(operand)))
    }

    /// The `WHEN ... THEN ...` branches, `ELSE` and `END` of a `CASE`.
    pub(super) fn case_branches(&mut self, operand: Option<Box<Expr>>) -> PR<ExprKind> {
        let mut branches = vec![];
        while self.kw_at(0, "WHEN") {
            let start = self.start();
            self.bump();
            let condition = self.in_case_part(CasePart::Condition, Self::expr)?;
            self.expect_kw("THEN")?;
            let result = self.in_case_part(CasePart::Result, Self::expr)?;
            branches.push(WhenClause {
                condition,
                result,
                span: self.span_from(start),
            });
        }
        if branches.is_empty() {
            return self.fail();
        }
        let else_result = if self.eat_kw("ELSE") {
            Some(boxed(self.in_case_part(CasePart::Else, Self::expr)?))
        } else {
            None
        };
        self.expect_kw("END")?;
        Ok(ExprKind::Case {
            operand,
            branches,
            else_result,
        })
    }
}
