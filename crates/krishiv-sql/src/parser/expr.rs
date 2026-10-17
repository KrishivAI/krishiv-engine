// SPDX-License-Identifier: Apache-2.0
//! Expressions (`booleanExpression` / `valueExpression` / `primaryExpression`),
//! literals, data types and windows.
//!
//! Precedence, loosest first: `OR`, `AND`, `NOT`, predicates (`BETWEEN`,
//! `IN`, `LIKE`, `IS`, ...), comparison, `|`, `^`, `&`, shifts, `+ - ||`,
//! `* / % DIV`, unary `- + ~`, postfix `[i]`, `.field`, `::type`.

use super::{Fail, PR, Parser};
use crate::ast::{
    BinaryOp, DataTypeKind, DataTypeSpec, Expr, ExprKind, FrameBound, FrameBoundKind, FrameUnits,
    Function, FunctionArg, Ident, Interval, IntervalValue, LikeQuantifier, Literal, NullTreatment,
    NumKind, ObjectName, SelectItem, Span, StructField, Truth, UnaryOp, WhenClause, WindowFrame,
    WindowRef, WindowSpec,
};
use crate::keywords::{NOT_FUNCTION_WORD, PRIMARY_WORD, RESERVED_WORD};
use crate::lexer::{Sym, TokenKind, unescape_string};

/// Binding powers, loosest first; prefix `NOT` and unary operators bind as
/// tightly as their level, and predicates sit between `NOT` and the
/// comparisons.
const OR: u8 = 1;
const AND: u8 = 2;
const NOT: u8 = 3;
const PREDICATE: u8 = 4;
/// The loosest operator of a `valueExpression` (comparison).
const VALUE: u8 = 5;
const UNARY: u8 = 12;

/// A pending operator of [`Parser::climb`].
#[derive(Clone, Copy)]
enum Op {
    Binary(BinaryOp, u8),
    /// `NOT` at a source position.
    Not(usize),
    Unary(UnaryOp, usize),
}

impl Op {
    fn bp(self) -> u8 {
        match self {
            Self::Binary(_, bp) => bp,
            Self::Not(_) => NOT,
            Self::Unary(..) => UNARY,
        }
    }
}

const CURRENT_LIKE: &[&str] = &[
    "CURRENT_DATE",
    "CURRENT_TIMESTAMP",
    "CURRENT_USER",
    "USER",
    "SESSION_USER",
    "CURRENT_TIME",
    "CURRENT_PATH",
];

const DATETIME_UNITS: &[&str] = &[
    "YEAR",
    "QUARTER",
    "MONTH",
    "WEEK",
    "DAY",
    "DAYOFYEAR",
    "HOUR",
    "MINUTE",
    "SECOND",
    "MILLISECOND",
    "MICROSECOND",
];

/// Functions whose first argument is a `datetimeUnit` keyword.
const TIMESTAMP_UNIT_FUNCTIONS: &[&str] = &[
    "TIMESTAMPADD",
    "DATEADD",
    "DATE_ADD",
    "TIMESTAMPDIFF",
    "DATEDIFF",
    "DATE_DIFF",
    "TIMEDIFF",
];

const INTERVAL_UNITS: &[&str] = &[
    "NANOSECOND",
    "NANOSECONDS",
    "MICROSECOND",
    "MICROSECONDS",
    "MILLISECOND",
    "MILLISECONDS",
    "SECOND",
    "SECONDS",
    "MINUTE",
    "MINUTES",
    "HOUR",
    "HOURS",
    "DAY",
    "DAYS",
    "WEEK",
    "WEEKS",
    "MONTH",
    "MONTHS",
    "YEAR",
    "YEARS",
];

/// Spark's `AstBuilder.visitLogicalBinary`: a balanced tree over the
/// operands of a chain of one operator.
fn balance(mut operands: Vec<Expr>, op: BinaryOp) -> Option<Expr> {
    if operands.len() <= 1 {
        return operands.pop();
    }
    let right = operands.split_off((operands.len() - 1) / 2 + 1);
    let (left, right) = (balance(operands, op)?, balance(right, op)?);
    Some(Expr {
        span: left.span.to(right.span),
        kind: ExprKind::Binary(Box::new(left), op, Box::new(right)),
    })
}

fn boxed(e: Expr) -> Box<Expr> {
    Box::new(e)
}

impl Parser<'_> {
    fn mk(&self, kind: ExprKind, start: usize) -> Expr {
        Expr {
            kind,
            span: self.span_from(start),
        }
    }

    /// A full `expression`, one level deeper.
    pub(crate) fn expr(&mut self) -> PR<Expr> {
        self.enter()?;
        let r = self.climb(OR);
        self.depth -= 1;
        r
    }

    /// A `valueExpression` (no `NOT`, predicates, `AND` or `OR`), one
    /// level deeper, with the levels its tree reaches below the current
    /// depth.
    fn value_expr(&mut self) -> PR<(Expr, usize)> {
        self.enter()?;
        let outer = self.peak;
        self.peak = self.depth;
        let r = self.climb(VALUE);
        let height = self.peak + 1 - self.depth;
        self.peak = self.peak.max(outer);
        self.depth -= 1;
        r.map(|e| (e, height))
    }

    fn value(&mut self) -> PR<Expr> {
        self.value_expr().map(|(e, _)| e)
    }

    /// Operators binding at least `min_bp`, by precedence climbing over
    /// explicit operator and operand stacks: only operands recurse, so
    /// operator chains and prefix runs cost no stack.
    fn climb(&mut self, min_bp: u8) -> PR<Expr> {
        self.climb_from(min_bp, None)
    }

    /// [`Self::climb`] with an already parsed first operand (and its height).
    fn climb_from(&mut self, min_bp: u8, mut first: Option<(Expr, usize)>) -> PR<Expr> {
        let mut operands: Vec<(Expr, usize)> = vec![];
        let mut ops: Vec<Op> = vec![];
        loop {
            if let Some(operand) = first.take() {
                operands.push(operand);
            } else {
                self.prefixes(min_bp, &mut ops);
                operands.push(self.operand()?);
            }
            let mut predicated = false;
            if min_bp <= PREDICATE && self.predicate_follows() {
                self.reduce(&mut operands, &mut ops, VALUE)?;
                let (e, height) = operands.pop().ok_or(Fail::Syntax)?;
                operands.push(self.predicate(e, height)?);
                predicated = true;
            }
            match self.infix_op() {
                Some((op, bp)) if bp >= min_bp && (!predicated || bp <= AND) => {
                    let floor = if matches!(op, BinaryOp::And | BinaryOp::Or) {
                        bp + 1
                    } else {
                        bp
                    };
                    self.reduce(&mut operands, &mut ops, floor)?;
                    self.bump();
                    ops.push(Op::Binary(op, bp));
                }
                _ => break,
            }
        }
        self.reduce(&mut operands, &mut ops, 0)?;
        // With every operator applied, exactly one operand is left.
        operands.pop().map(|(e, _)| e).ok_or(Fail::Syntax)
    }

    /// Pushes the prefix operators at the current token: `NOT` / `!` (in a
    /// boolean context) and unary `-` (unless a number follows), `+`, `~`.
    fn prefixes(&mut self, min_bp: u8, ops: &mut Vec<Op>) {
        loop {
            let start = self.start();
            let op = if min_bp <= NOT && (self.kw_at(0, "NOT") || self.sym_at(0, Sym::Bang)) {
                Op::Not(start)
            } else if self.sym_at(0, Sym::Minus)
                && !matches!(self.kind(1), Some(TokenKind::Number(_)))
            {
                Op::Unary(UnaryOp::Minus, start)
            } else if self.sym_at(0, Sym::Plus) {
                Op::Unary(UnaryOp::Plus, start)
            } else if self.sym_at(0, Sym::Tilde) {
                Op::Unary(UnaryOp::BitNot, start)
            } else {
                return;
            };
            self.bump();
            ops.push(op);
        }
    }

    /// A call argument that may be followed by one of `keywords` (Spark's
    /// special call syntax, `TRIM(c FROM s)`): a `valueExpression` and
    /// `true` when one follows, else the whole argument expression and
    /// `false`. One parse decides, so nested calls never backtrack.
    fn first_arg(&mut self, keywords: &[&str]) -> PR<(Expr, bool)> {
        self.enter()?;
        let outer = self.peak;
        self.peak = self.depth;
        let r = self.climb(VALUE).and_then(|e| {
            if self.any_kw_at(0, keywords) {
                Ok((e, true))
            } else {
                let height = self.peak - self.depth;
                self.climb_from(OR, Some((e, height))).map(|e| (e, false))
            }
        });
        self.peak = self.peak.max(outer);
        self.depth -= 1;
        r
    }

    /// The expression that starts with the parsed primary `first` (of
    /// `height` levels below the current depth), one level deeper.
    pub(super) fn expr_from(&mut self, first: Expr, height: usize) -> PR<Expr> {
        self.enter()?;
        let outer = self.peak;
        self.peak = self.depth + height;
        let start = first.span.start;
        let r = self.postfix(first, start).and_then(|e| {
            let height = self.peak - self.depth;
            self.climb_from(OR, Some((e, height)))
        });
        self.peak = self.peak.max(outer);
        self.depth -= 1;
        r
    }

    /// Applies the operators on `ops` binding at least `floor`; a run of
    /// `AND` (or `OR`) becomes one balanced tree, as Spark builds it.
    #[inline(never)]
    fn reduce(
        &mut self,
        operands: &mut Vec<(Expr, usize)>,
        ops: &mut Vec<Op>,
        floor: u8,
    ) -> PR<()> {
        while let Some(&op) = ops.last().filter(|op| op.bp() >= floor) {
            ops.pop();
            let (expr, height) = match op {
                Op::Not(start) | Op::Unary(_, start) => {
                    let (e, h) = operands.pop().ok_or(Fail::Syntax)?;
                    let span = Span::new(start, e.span.end);
                    let kind = match op {
                        Op::Unary(u, _) => ExprKind::Unary(u, boxed(e)),
                        _ => ExprKind::Not(boxed(e)),
                    };
                    (Expr { kind, span }, h + 1)
                }
                Op::Binary(b @ (BinaryOp::And | BinaryOp::Or), _) => {
                    let mut n = 2;
                    while ops
                        .last()
                        .is_some_and(|o| matches!(o, Op::Binary(x, _) if *x == b))
                    {
                        ops.pop();
                        n += 1;
                    }
                    let chain = operands.split_off(operands.len().saturating_sub(n));
                    let height = chain.iter().map(|c| c.1).max().unwrap_or(0);
                    let levels = usize::BITS - (chain.len().max(1) - 1).leading_zeros();
                    let exprs = chain.into_iter().map(|c| c.0).collect();
                    let e = balance(exprs, b).ok_or(Fail::Syntax)?;
                    (e, height + levels as usize)
                }
                Op::Binary(b, _) => {
                    let (right, hr) = operands.pop().ok_or(Fail::Syntax)?;
                    let (left, hl) = operands.pop().ok_or(Fail::Syntax)?;
                    let span = left.span.to(right.span);
                    let kind = ExprKind::Binary(boxed(left), b, boxed(right));
                    (Expr { kind, span }, hl.max(hr) + 1)
                }
            };
            self.grow(height)?;
            operands.push((expr, height));
        }
        Ok(())
    }

    fn predicate_follows(&self) -> bool {
        let negated = self.kw_at(0, "NOT") || self.sym_at(0, Sym::Bang);
        self.any_kw_at(
            usize::from(negated),
            &["BETWEEN", "IN", "LIKE", "ILIKE", "RLIKE", "REGEXP"],
        ) || (!negated && self.kw_at(0, "IS"))
    }

    /// The predicate after `expr` (of `height`).
    #[inline(never)]
    fn predicate(&mut self, expr: Expr, mut height: usize) -> PR<(Expr, usize)> {
        let start = expr.span.start;
        let negated = !self.kw_at(0, "IS") && (self.kw_at(0, "NOT") || self.sym_at(0, Sym::Bang));
        self.pos += usize::from(negated);
        let expr = boxed(expr);
        let mut value = |p: &mut Self| -> PR<Box<Expr>> {
            let (e, h) = p.value_expr()?;
            height = height.max(h);
            Ok(boxed(e))
        };
        let kind = if self.eat_kw("BETWEEN") {
            let low = value(self)?;
            self.expect_kw("AND")?;
            let high = value(self)?;
            ExprKind::Between {
                expr,
                low,
                high,
                negated,
            }
        } else if self.eat_kw("IN") {
            self.expect_sym(Sym::LParen)?;
            let kind = if self.query_follows(0) {
                let query = Box::new(self.query()?);
                ExprKind::InSubquery {
                    expr,
                    query,
                    negated,
                }
            } else {
                let list = self.opt_expr_list()?;
                ExprKind::InList {
                    expr,
                    list,
                    negated,
                }
            };
            self.expect_sym(Sym::RParen)?;
            kind
        } else if self.eat_kw("RLIKE") || self.eat_kw("REGEXP") {
            let pattern = value(self)?;
            ExprKind::RLike {
                expr,
                pattern,
                negated,
            }
        } else if self.any_kw_at(0, &["LIKE", "ILIKE"]) {
            let case_insensitive = self.kw_at(0, "ILIKE");
            self.bump();
            let quantifier = [
                ("ANY", LikeQuantifier::Any),
                ("SOME", LikeQuantifier::Some),
                ("ALL", LikeQuantifier::All),
            ]
            .into_iter()
            .find_map(|(kw, q)| self.kw_at(0, kw).then_some(q));
            if let Some(quantifier) = quantifier {
                self.bump();
                self.expect_sym(Sym::LParen)?;
                let patterns = self.opt_expr_list()?;
                self.expect_sym(Sym::RParen)?;
                ExprKind::LikeQuantified {
                    expr,
                    quantifier,
                    patterns,
                    negated,
                    case_insensitive,
                }
            } else {
                let pattern = value(self)?;
                let escape = if self.eat_kw("ESCAPE") {
                    Some(self.string_lit().ok_or(Fail::Syntax)?)
                } else {
                    None
                };
                ExprKind::Like {
                    expr,
                    pattern,
                    escape,
                    negated,
                    case_insensitive,
                }
            }
        } else {
            self.expect_kw("IS")?;
            let negated = self.eat_kw("NOT") || self.eat_sym(Sym::Bang);
            let truth = [
                ("TRUE", Truth::True),
                ("FALSE", Truth::False),
                ("UNKNOWN", Truth::Unknown),
            ]
            .into_iter()
            .find_map(|(kw, t)| self.kw_at(0, kw).then_some(t));
            if self.eat_kw("NULL") {
                ExprKind::IsNull { expr, negated }
            } else if let Some(truth) = truth {
                self.bump();
                ExprKind::IsTruth {
                    expr,
                    truth,
                    negated,
                }
            } else {
                self.expect_kw("DISTINCT")?;
                self.expect_kw("FROM")?;
                let right = value(self)?;
                ExprKind::IsDistinctFrom {
                    left: expr,
                    right,
                    negated,
                }
            }
        };
        height += 1;
        self.grow(height)?;
        Ok((self.mk(kind, start), height))
    }

    /// `[expr, ...]` before a `)`.
    fn opt_expr_list(&mut self) -> PR<Vec<Expr>> {
        if self.sym_at(0, Sym::RParen) {
            Ok(vec![])
        } else {
            self.comma_sep(Self::expr)
        }
    }

    fn infix_op(&self) -> Option<(BinaryOp, u8)> {
        use BinaryOp as B;
        Some(match self.kind(0)? {
            TokenKind::Sym(s) => match s {
                Sym::Eq => (B::Eq, VALUE),
                Sym::Neq => (B::NotEq, VALUE),
                Sym::Lt => (B::Lt, VALUE),
                Sym::Le => (B::LtEq, VALUE),
                Sym::Gt => (B::Gt, VALUE),
                Sym::Ge => (B::GtEq, VALUE),
                Sym::NullSafeEq => (B::NullSafeEq, VALUE),
                Sym::Pipe => (B::BitOr, 6),
                Sym::Caret => (B::BitXor, 7),
                Sym::Amp => (B::BitAnd, 8),
                Sym::Shl => (B::ShiftLeft, 9),
                Sym::Shr => (B::ShiftRight, 9),
                Sym::Shru => (B::ShiftRightUnsigned, 9),
                Sym::Plus => (B::Plus, 10),
                Sym::Minus => (B::Minus, 10),
                Sym::Concat => (B::Concat, 10),
                Sym::Star => (B::Multiply, 11),
                Sym::Slash => (B::Divide, 11),
                Sym::Percent => (B::Modulo, 11),
                _ => return None,
            },
            TokenKind::Word if self.kw_at(0, "DIV") => (B::IntegerDivide, 11),
            TokenKind::Word if self.kw_at(0, "AND") => (B::And, AND),
            TokenKind::Word if self.kw_at(0, "OR") => (B::Or, OR),
            _ => return None,
        })
    }

    /// A primary with its postfix operators, and the levels its tree
    /// reaches below the current depth.
    fn operand(&mut self) -> PR<(Expr, usize)> {
        let outer = self.peak;
        self.peak = self.depth;
        let r = self.primary_expr();
        let height = self.peak - self.depth;
        self.peak = self.peak.max(outer);
        r.map(|e| (e, height))
    }

    /// `primaryExpression` (a negative number literal included) with
    /// `[index]`, `.field` and `::type` after it.
    pub(super) fn primary_expr(&mut self) -> PR<Expr> {
        let start = self.start();
        let e = match self.kind(1) {
            Some(TokenKind::Number(kind)) if self.sym_at(0, Sym::Minus) => {
                // `number: MINUS? INTEGER_VALUE | ...`: a negative literal.
                self.bump();
                let lit = self.number(kind, true);
                self.mk(ExprKind::Literal(lit), start)
            }
            _ => self.primary()?,
        };
        self.postfix(e, start)
    }

    #[inline(never)]
    fn postfix(&mut self, mut e: Expr, start: usize) -> PR<Expr> {
        let mut height = self.peak - self.depth;
        loop {
            let kind = if self.eat_sym(Sym::LBracket) {
                let (index, h) = self.value_expr()?;
                self.expect_sym(Sym::RBracket)?;
                height = height.max(h);
                ExprKind::Subscript(boxed(e), boxed(index))
            } else if self.sym_at(0, Sym::Dot) && self.is_ident_at(1, true) {
                self.bump();
                ExprKind::Field(boxed(e), self.any_ident()?)
            } else if self.eat_sym(Sym::DoubleColon) {
                let (data_type, h) = self.measured(Self::data_type)?;
                height = height.max(h);
                ExprKind::Cast {
                    expr: boxed(e),
                    data_type: Box::new(data_type),
                    try_cast: false,
                }
            } else {
                return Ok(e);
            };
            height += 1;
            self.grow(height)?;
            e = self.mk(kind, start);
        }
    }

    fn primary(&mut self) -> PR<Expr> {
        let start = self.start();
        let kind = match self.kind(0) {
            Some(TokenKind::Number(kind)) => self.number(kind, false),
            Some(TokenKind::String) => Literal::String(self.string_lit().unwrap_or_default()),
            Some(TokenKind::DoubleQuoted) if !self.opts.double_quoted_identifiers => {
                Literal::String(self.string_lit().unwrap_or_default())
            }
            Some(TokenKind::Sym(Sym::Question)) => {
                self.bump();
                return Ok(self.mk(ExprKind::Parameter, start));
            }
            Some(TokenKind::Sym(Sym::Star)) => {
                self.bump();
                let except = self.star_except()?;
                return Ok(self.mk(
                    ExprKind::Wildcard {
                        qualifier: vec![],
                        except,
                    },
                    start,
                ));
            }
            Some(TokenKind::Sym(Sym::LParen)) => return self.paren_expr(),
            Some(TokenKind::Word | TokenKind::QuotedIdent | TokenKind::DoubleQuoted) => {
                return self.word_primary();
            }
            _ => return self.fail(),
        };
        Ok(self.mk(ExprKind::Literal(kind), start))
    }

    /// The number at the current token, consumed.
    #[inline(never)]
    fn number(&mut self, kind: NumKind, negative: bool) -> Literal {
        let raw = self.text(0);
        let suffix = match kind {
            NumKind::Long | NumKind::Short | NumKind::Byte | NumKind::Double | NumKind::Float => 1,
            NumKind::BigDecimal => 2,
            NumKind::Int | NumKind::Decimal | NumKind::Exponent => 0,
        };
        let digits = raw.get(..raw.len() - suffix).unwrap_or(raw);
        self.bump();
        Literal::Number {
            text: if negative {
                format!("-{digits}")
            } else {
                digits.to_owned()
            },
            kind,
        }
    }

    /// Adjacent string literals, concatenated (`stringLit`).
    #[inline(never)]
    fn string_lit(&mut self) -> Option<String> {
        let mut out: Option<String> = None;
        while let Some(k @ (TokenKind::String | TokenKind::DoubleQuoted)) = self.kind(0) {
            if k == TokenKind::DoubleQuoted && self.opts.double_quoted_identifiers {
                break;
            }
            let s = unescape_string(self.text(0), !self.opts.escaped_string_literals);
            out.get_or_insert_with(String::new).push_str(&s);
            self.bump();
        }
        out
    }

    /// One string token (`singleStringLitWithoutMarker`) at offset `n`.
    fn single_string_at(&self, n: usize) -> bool {
        match self.kind(n) {
            Some(TokenKind::String) => true,
            Some(TokenKind::DoubleQuoted) => !self.opts.double_quoted_identifiers,
            _ => false,
        }
    }

    fn star_except(&mut self) -> PR<Vec<ObjectName>> {
        if self.kw_at(0, "EXCEPT") && self.sym_at(1, Sym::LParen) && !self.query_follows(1) {
            self.bump();
            self.paren_list(Self::object_name)
        } else {
            Ok(vec![])
        }
    }

    fn paren_expr(&mut self) -> PR<Expr> {
        let start = self.start();
        if let Some(e) = self.paren_lambda_or_query(start)? {
            return Ok(e);
        }
        self.expect_sym(Sym::LParen)?;
        let first = self.select_item()?;
        self.paren_tail(first, start)
    }

    /// `(x, y) -> body` or `(query)`, if one starts here.
    #[inline(never)]
    fn paren_lambda_or_query(&mut self, start: usize) -> PR<Option<Expr>> {
        let mut n = 1;
        while self.is_ident_at(n, true) && self.sym_at(n + 1, Sym::Comma) {
            n += 2;
        }
        if n > 1
            && self.is_ident_at(n, true)
            && self.sym_at(n + 1, Sym::RParen)
            && self.sym_at(n + 2, Sym::Arrow)
        {
            let params = self.paren_list(Self::any_ident)?;
            self.bump();
            let body = boxed(self.expr()?);
            return Ok(Some(self.mk(ExprKind::Lambda { params, body }, start)));
        }
        if self.query_follows(1) {
            let save = self.pos;
            self.bump();
            match self.query() {
                Ok(q) if self.eat_sym(Sym::RParen) => {
                    return Ok(Some(self.mk(ExprKind::Subquery(Box::new(q)), start)));
                }
                Err(Fail::Fatal(e)) => return Err(Fail::Fatal(e)),
                _ => self.pos = save,
            }
        }
        Ok(None)
    }

    /// `(expr)` or a row constructor `(a, b AS c)` after its first item.
    #[inline(never)]
    fn paren_tail(&mut self, first: SelectItem, start: usize) -> PR<Expr> {
        if !self.sym_at(0, Sym::Comma) {
            // `(a AS b)` is no expression: a row needs two items.
            if first.alias.is_some() {
                return self.fail();
            }
            self.expect_sym(Sym::RParen)?;
            return Ok(self.mk(ExprKind::Nested(boxed(first.expr)), start));
        }
        let mut items = vec![first];
        while self.eat_sym(Sym::Comma) {
            items.push(self.select_item()?);
        }
        self.expect_sym(Sym::RParen)?;
        Ok(self.mk(ExprKind::Row(items), start))
    }

    fn word_primary(&mut self) -> PR<Expr> {
        let start = self.start();
        let special = self.class(0) & PRIMARY_WORD != 0 || self.single_string_at(1);
        if self.kind(0) == Some(TokenKind::Word)
            && special
            && let Some(e) = self.keyword_primary(start)?
        {
            return Ok(e);
        }
        // `x -> body`
        if self.sym_at(1, Sym::Arrow) && self.is_ident_at(0, true) {
            let params = vec![self.any_ident()?];
            self.bump();
            let body = boxed(self.expr()?);
            return Ok(self.mk(ExprKind::Lambda { params, body }, start));
        }
        let mut parts = vec![];
        let reserved_call = self.kind(0) == Some(TokenKind::Word)
            && self.sym_at(1, Sym::LParen)
            && self.class(0) & NOT_FUNCTION_WORD == 0;
        if (reserved_call || self.reserved_as_column()) && !self.is_ident_at(0, false) {
            parts.push(self.keyword_ident());
        } else {
            parts.push(self.ident()?);
        }
        while self.sym_at(0, Sym::Dot) {
            if self.sym_at(1, Sym::Star) {
                self.pos += 2;
                let except = self.star_except()?;
                let kind = ExprKind::Wildcard {
                    qualifier: parts,
                    except,
                };
                return Ok(self.mk(kind, start));
            }
            if !self.is_ident_at(1, true) {
                break;
            }
            self.bump();
            parts.push(self.any_ident()?);
        }
        if self.sym_at(0, Sym::LParen) {
            let name = ObjectName {
                span: self.span_from(start),
                parts,
            };
            return self.function_call(name, start);
        }
        Ok(self.mk(ExprKind::Column(parts), start))
    }

    /// Whether the reserved word here is a column, because no keyword
    /// reading can continue with the next token (`all, b`, `all * 2`,
    /// `all FROM t`).
    /// Spark accepts every keyword as an identifier unless keywords are
    /// enforced.
    fn reserved_as_column(&self) -> bool {
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

    /// The current word as an identifier, whatever it is.
    fn keyword_ident(&mut self) -> Ident {
        let span = self.tok(0).map_or_else(Span::default, |t| t.span);
        self.bump();
        Ident {
            value: span.text(self.src).to_owned(),
            quoted: false,
            span,
        }
    }

    /// Primaries that start with a keyword; `None` if the word is not one
    /// of them here.
    #[inline(never)]
    fn keyword_primary(&mut self, start: usize) -> PR<Option<Expr>> {
        let word = self.text(0);
        let call = self.sym_at(1, Sym::LParen);
        let is = |kw: &str| crate::keywords::is_kw(word, kw);
        let kind = if is("NULL") {
            self.bump();
            ExprKind::Literal(Literal::Null)
        } else if is("TRUE") || is("FALSE") {
            self.bump();
            ExprKind::Literal(Literal::Boolean(is("TRUE")))
        } else if is("CASE") {
            self.case_expr()?
        } else if (is("CAST") || is("TRY_CAST")) && call {
            self.pos += 2;
            let expr = boxed(self.expr()?);
            self.expect_kw("AS")?;
            let data_type = Box::new(self.data_type()?);
            self.expect_sym(Sym::RParen)?;
            ExprKind::Cast {
                expr,
                data_type,
                try_cast: is("TRY_CAST"),
            }
        } else if is("EXISTS") && call && self.query_follows(2) {
            self.pos += 2;
            let q = self.query()?;
            self.expect_sym(Sym::RParen)?;
            ExprKind::Exists(Box::new(q))
        } else if is("INTERVAL") && self.interval_follows() {
            ExprKind::Literal(self.interval()?)
        } else if self.single_string_at(1) && (self.class(0) & RESERVED_WORD == 0 || is("TIME")) {
            self.bump();
            let raw = self.text(0);
            self.bump();
            ExprKind::Literal(Literal::Typed {
                type_name: word.to_ascii_lowercase(),
                value: unescape_string(raw, !self.opts.escaped_string_literals),
            })
        } else if is("STRUCT") && call {
            self.pos += 2;
            let items = if self.sym_at(0, Sym::RParen) {
                vec![]
            } else {
                self.comma_sep(Self::select_item)?
            };
            self.expect_sym(Sym::RParen)?;
            ExprKind::Struct(items)
        } else if CURRENT_LIKE.iter().any(|k| is(k)) && !call {
            let id = self.keyword_ident();
            if self.opts.enforce_reserved_keywords || is("CURRENT_PATH") {
                let name = ObjectName {
                    span: id.span,
                    parts: vec![id],
                };
                ExprKind::Function(Box::new(Self::call(name, vec![], self.span_from(start))))
            } else {
                ExprKind::Column(vec![id])
            }
        } else if call {
            match self.special_call(start)? {
                Some(f) => ExprKind::Function(Box::new(f)),
                None => return Ok(None),
            }
        } else {
            return Ok(None);
        };
        Ok(Some(self.mk(kind, start)))
    }

    fn call(name: ObjectName, args: Vec<Expr>, span: Span) -> Function {
        Function {
            name,
            quantifier: None,
            args: args
                .into_iter()
                .map(|expr| FunctionArg {
                    name: None,
                    span: expr.span,
                    expr,
                })
                .collect(),
            within_group: vec![],
            filter: None,
            null_treatment: None,
            over: None,
            span,
        }
    }

    /// Spark's special call syntax: `EXTRACT(f FROM x)`, `SUBSTRING(s FROM
    /// p [FOR l])`, `TRIM([BOTH | LEADING | TRAILING] [c] FROM s)`,
    /// `POSITION(a IN b)` and `OVERLAY(s PLACING r FROM p [FOR l])`, decided
    /// without backtracking: once the first argument is parsed, the token
    /// after it tells the special form from an ordinary call (which is then
    /// finished here). `None` (nothing consumed) is an ordinary call.
    #[inline(never)]
    fn special_call(&mut self, start: usize) -> PR<Option<Function>> {
        let word = self.text(0);
        let is = |kw: &str| crate::keywords::is_kw(word, kw);
        // `f()`, `f(x => 1)` and `f(DISTINCT ...)` are ordinary calls.
        if self.sym_at(2, Sym::RParen) || self.sym_at(3, Sym::FatArrow) || self.kw_at(2, "DISTINCT")
        {
            return Ok(None);
        }
        let extract = is("EXTRACT") && self.is_ident_at(2, true) && self.kw_at(3, "FROM");
        let trim_spec = is("TRIM")
            && (self.kw_at(2, "FROM")
                || self.any_kw_at(2, &["BOTH", "LEADING", "TRAILING"]) && !self.item_end_at(3));
        let keywords: &[&str] = if extract || trim_spec {
            &[]
        } else if is("TRIM") || is("SUBSTR") || is("SUBSTRING") {
            &["FROM"]
        } else if is("POSITION") {
            &["IN"]
        } else if is("OVERLAY") {
            &["PLACING"]
        } else {
            return Ok(None);
        };
        let (trim, overlay, position) = (is("TRIM"), is("OVERLAY"), is("POSITION"));
        let word = self.keyword_ident();
        self.bump();
        let mut name = word.value.to_ascii_lowercase();
        let mut args = vec![];
        if extract {
            let field = self.any_ident()?;
            args.push(Expr {
                span: field.span,
                kind: ExprKind::Literal(Literal::String(field.value)),
            });
            self.bump();
            args.push(self.value()?);
        } else if trim_spec {
            if self.eat_kw("LEADING") {
                name = "ltrim".to_owned();
            } else if self.eat_kw("TRAILING") {
                name = "rtrim".to_owned();
            } else {
                self.eat_kw("BOTH");
            }
            let trim = if self.kw_at(0, "FROM") {
                None
            } else {
                Some(self.value()?)
            };
            self.expect_kw("FROM")?;
            args.push(self.value()?);
            args.extend(trim);
        } else {
            let (first, special) = self.first_arg(keywords)?;
            let name = ObjectName {
                span: word.span,
                parts: vec![word],
            };
            if !special {
                return self.call_rest(name, first, start).map(Some);
            }
            self.bump();
            let second = self.value()?;
            if trim {
                // `TRIM(c FROM s)` is `trim(s, c)`.
                args.extend([second, first]);
            } else {
                args.extend([first, second]);
                if overlay {
                    self.expect_kw("FROM")?;
                    args.push(self.value()?);
                }
                if !position && self.eat_kw("FOR") {
                    args.push(self.value()?);
                }
            }
            self.expect_sym(Sym::RParen)?;
            return Ok(Some(Self::call(name, args, self.span_from(start))));
        }
        self.expect_sym(Sym::RParen)?;
        let name = ObjectName {
            parts: vec![Ident {
                value: name,
                ..word
            }],
            span: word.span,
        };
        Ok(Some(Self::call(name, args, self.span_from(start))))
    }

    /// An ordinary call whose first argument `first` is parsed.
    fn call_rest(&mut self, name: ObjectName, first: Expr, start: usize) -> PR<Function> {
        let mut args = vec![FunctionArg {
            name: None,
            span: first.span,
            expr: first,
        }];
        if self.eat_sym(Sym::Comma) {
            args.extend(self.function_args()?);
        }
        self.call_tail(name, None, args, start)
    }

    #[inline(never)]
    fn case_expr(&mut self) -> PR<ExprKind> {
        self.bump();
        let operand = if self.kw_at(0, "WHEN") {
            None
        } else {
            Some(boxed(self.expr()?))
        };
        let mut branches = vec![];
        while self.kw_at(0, "WHEN") {
            let start = self.start();
            self.bump();
            let condition = self.expr()?;
            self.expect_kw("THEN")?;
            let result = self.expr()?;
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
            Some(boxed(self.expr()?))
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

    // ---- function calls -------------------------------------------------

    pub(super) fn function_args(&mut self) -> PR<Vec<FunctionArg>> {
        if self.sym_at(0, Sym::RParen) {
            return Ok(vec![]);
        }
        self.comma_sep(|p| {
            let start = p.start();
            let name = if p.sym_at(1, Sym::FatArrow) && p.is_ident_at(0, true) {
                let name = p.any_ident()?;
                p.bump();
                Some(name)
            } else {
                None
            };
            let expr = p.expr()?;
            Ok(FunctionArg {
                name,
                expr,
                span: p.span_from(start),
            })
        })
    }

    #[inline(never)]
    fn function_call(&mut self, name: ObjectName, start: usize) -> PR<Expr> {
        self.expect_sym(Sym::LParen)?;
        let quantifier = if self.kw_at(0, "ALL") && self.sym_at(1, Sym::RParen) {
            None
        } else {
            self.set_quantifier()
        };
        let args = self.function_args()?;
        let f = self.call_tail(name, quantifier, args, start)?;
        Ok(self.mk(ExprKind::Function(Box::new(f)), start))
    }

    /// A call from `[IGNORE NULLS] )` after its arguments on.
    #[inline(never)]
    fn call_tail(
        &mut self,
        name: ObjectName,
        quantifier: Option<crate::ast::SetQuantifier>,
        mut args: Vec<FunctionArg>,
        start: usize,
    ) -> PR<Function> {
        let mut null_treatment = self.null_treatment();
        self.expect_sym(Sym::RParen)?;
        let mut within_group = vec![];
        if self.kw_at(0, "WITHIN") && self.kw_at(1, "GROUP") {
            self.pos += 2;
            self.expect_sym(Sym::LParen)?;
            self.expect_kw("ORDER")?;
            self.expect_kw("BY")?;
            within_group = self.comma_sep(|p| p.sort_item(false))?;
            self.expect_sym(Sym::RParen)?;
        }
        let mut filter = None;
        if self.kw_at(0, "FILTER") && self.sym_at(1, Sym::LParen) {
            self.pos += 2;
            self.expect_kw("WHERE")?;
            filter = Some(boxed(self.expr()?));
            self.expect_sym(Sym::RParen)?;
        }
        null_treatment = null_treatment.or_else(|| self.null_treatment());
        let over = if self.eat_kw("OVER") {
            Some(self.window_ref()?)
        } else {
            None
        };
        // `TIMESTAMPADD(unit, n, t)` / `TIMESTAMPDIFF(unit, a, b)`: the unit
        // is a keyword, not a column.
        let unit_call = name.parts.len() == 1
            && TIMESTAMP_UNIT_FUNCTIONS
                .iter()
                .any(|f| crate::keywords::is_kw(&name.parts[0].value, f));
        if let [unit, _, _] = args.as_mut_slice()
            && unit_call
            && unit.name.is_none()
            && let ExprKind::Column(parts) = &unit.expr.kind
            && let [id] = parts.as_slice()
            && !id.quoted
            && DATETIME_UNITS
                .iter()
                .any(|u| crate::keywords::is_kw(&id.value, u))
        {
            unit.expr.kind = ExprKind::Literal(Literal::String(id.value.to_ascii_uppercase()));
        }
        Ok(Function {
            name,
            quantifier,
            args,
            within_group,
            filter,
            null_treatment,
            over,
            span: self.span_from(start),
        })
    }

    fn null_treatment(&mut self) -> Option<NullTreatment> {
        if !self.kw_at(1, "NULLS") {
            return None;
        }
        let t = if self.kw_at(0, "IGNORE") {
            NullTreatment::Ignore
        } else if self.kw_at(0, "RESPECT") {
            NullTreatment::Respect
        } else {
            return None;
        };
        self.pos += 2;
        Some(t)
    }

    // ---- windows ----------------------------------------------------------

    /// `windowSpec`: `name`, `(name)` or `(definition)`.
    #[inline(never)]
    pub(super) fn window_ref(&mut self) -> PR<WindowRef> {
        if !self.sym_at(0, Sym::LParen) {
            return self.name_part(false).map(WindowRef::Named);
        }
        if self.is_ident_at(1, false) && self.sym_at(2, Sym::RParen) {
            self.bump();
            let name = self.name_part(false)?;
            self.bump();
            return Ok(WindowRef::Named(name));
        }
        let start = self.start();
        self.bump();
        let mut partition_by = vec![];
        if self.any_kw_at(0, &["PARTITION", "DISTRIBUTE", "CLUSTER"]) && self.kw_at(1, "BY") {
            self.pos += 2;
            partition_by = self.comma_sep(Self::expr)?;
        }
        let mut order_by = vec![];
        if self.any_kw_at(0, &["ORDER", "SORT"]) && self.kw_at(1, "BY") {
            self.pos += 2;
            order_by = self.comma_sep(|p| p.sort_item(false))?;
        }
        let frame = self.window_frame()?;
        self.expect_sym(Sym::RParen)?;
        Ok(WindowRef::Spec(Box::new(WindowSpec {
            partition_by,
            order_by,
            frame,
            span: self.span_from(start),
        })))
    }

    fn window_frame(&mut self) -> PR<Option<WindowFrame>> {
        let start = self.start();
        let units = if self.eat_kw("ROWS") {
            FrameUnits::Rows
        } else if self.eat_kw("RANGE") {
            FrameUnits::Range
        } else {
            return Ok(None);
        };
        let between = self.eat_kw("BETWEEN");
        let first = self.frame_bound()?;
        let end = if between {
            self.expect_kw("AND")?;
            Some(self.frame_bound()?)
        } else {
            None
        };
        Ok(Some(WindowFrame {
            units,
            start: first,
            end,
            span: self.span_from(start),
        }))
    }

    fn frame_bound(&mut self) -> PR<FrameBound> {
        let start = self.start();
        let kind = if self.eat_kw("UNBOUNDED") {
            if self.eat_kw("PRECEDING") {
                FrameBoundKind::UnboundedPreceding
            } else {
                self.expect_kw("FOLLOWING")?;
                FrameBoundKind::UnboundedFollowing
            }
        } else if self.kw_at(0, "CURRENT") && self.kw_at(1, "ROW") {
            self.pos += 2;
            FrameBoundKind::CurrentRow
        } else {
            let e = self.expr()?;
            if self.eat_kw("PRECEDING") {
                FrameBoundKind::Preceding(e)
            } else {
                self.expect_kw("FOLLOWING")?;
                FrameBoundKind::Following(e)
            }
        };
        Ok(FrameBound {
            kind,
            span: self.span_from(start),
        })
    }

    // ---- data types ---------------------------------------------------------

    /// `dataType`.
    pub(crate) fn data_type(&mut self) -> PR<DataTypeSpec> {
        self.nested(|p| {
            let start = p.start();
            if p.kind(0) != Some(TokenKind::Word) {
                return p.fail();
            }
            let word = p.text(0);
            p.bump();
            let is = |kw: &str| crate::keywords::is_kw(word, kw);
            let kind = if is("ARRAY") {
                let element = if p.eat_sym(Sym::Lt) {
                    let t = p.data_type()?;
                    p.expect_sym(Sym::Gt)?;
                    Some(Box::new(t))
                } else {
                    None
                };
                DataTypeKind::Array(element)
            } else if is("MAP") {
                let entry = if p.eat_sym(Sym::Lt) {
                    let k = p.data_type()?;
                    p.expect_sym(Sym::Comma)?;
                    let v = p.data_type()?;
                    p.expect_sym(Sym::Gt)?;
                    Some(Box::new((k, v)))
                } else {
                    None
                };
                DataTypeKind::Map(entry)
            } else if is("STRUCT") {
                DataTypeKind::Struct(if p.eat_sym(Sym::Neq) {
                    Some(vec![])
                } else if p.eat_sym(Sym::Lt) {
                    let fields = if p.sym_at(0, Sym::Gt) {
                        vec![]
                    } else {
                        p.comma_sep(Self::struct_field)?
                    };
                    p.expect_sym(Sym::Gt)?;
                    Some(fields)
                } else {
                    None
                })
            } else {
                p.primitive_type(word)?
            };
            Ok(DataTypeSpec {
                kind,
                span: p.span_from(start),
            })
        })
    }

    #[inline(never)]
    fn struct_field(&mut self) -> PR<StructField> {
        let start = self.start();
        let name = self.name_part(true)?;
        self.eat_sym(Sym::Colon);
        let data_type = self.data_type()?;
        let not_null = self.kw_at(0, "NOT") && self.kw_at(1, "NULL");
        self.pos += 2 * usize::from(not_null);
        let comment = if self.eat_kw("COMMENT") {
            match self.string_lit() {
                Some(s) => Some(s),
                None => return self.fail(),
            }
        } else {
            None
        };
        Ok(StructField {
            name,
            data_type,
            not_null,
            comment,
            span: self.span_from(start),
        })
    }

    /// `primitiveType` after its first word.
    #[inline(never)]
    fn primitive_type(&mut self, word: &str) -> PR<DataTypeKind> {
        let is = |kw: &str| crate::keywords::is_kw(word, kw);
        let mut name = word.to_ascii_lowercase();
        let mut push = |p: &mut Self, words: &[&str]| -> PR<()> {
            for w in words {
                p.expect_kw(w)?;
                name.push(' ');
                name.push_str(&w.to_ascii_lowercase());
            }
            Ok(())
        };
        let mut args = vec![];
        if is("INTERVAL") {
            let units: &[&str] = &["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"];
            if let Some(from) = units.iter().position(|u| self.kw_at(0, u)) {
                push(self, &units[from..=from])?;
                let to: &[&str] = if from < 2 {
                    &["MONTH"]
                } else {
                    &["HOUR", "MINUTE", "SECOND"]
                };
                if self.kw_at(0, "TO") && to.iter().any(|u| self.kw_at(1, u)) {
                    push(self, &["TO"])?;
                    let unit = self.text(0).to_ascii_uppercase();
                    push(self, &[unit.as_str()])?;
                }
            }
            return Ok(DataTypeKind::Primitive { name, args });
        }
        if self.sym_at(0, Sym::LParen)
            && (matches!(self.kind(1), Some(TokenKind::Number(NumKind::Int)))
                || self.kw_at(1, "ANY"))
        {
            self.bump();
            args = self.comma_sep(|p| {
                if !matches!(p.kind(0), Some(TokenKind::Number(NumKind::Int))) && !p.kw_at(0, "ANY")
                {
                    return p.fail();
                }
                let a = p.text(0).to_owned();
                p.bump();
                Ok(a)
            })?;
            self.expect_sym(Sym::RParen)?;
        }
        if (is("TIMESTAMP") || is("TIME")) && self.kw_at(0, "WITHOUT") {
            push(self, &["WITHOUT", "TIME", "ZONE"])?;
        } else if is("TIMESTAMP") && self.kw_at(0, "WITH") && self.kw_at(1, "LOCAL") {
            push(self, &["WITH", "LOCAL", "TIME", "ZONE"])?;
        }
        Ok(DataTypeKind::Primitive { name, args })
    }

    // ---- intervals ----------------------------------------------------------

    /// `INTERVAL` followed by a value (`intervalValue`: an integer, a
    /// decimal or a string).
    fn interval_follows(&self) -> bool {
        self.interval_value_len(1).is_some()
    }

    /// `[+|-] (integer | decimal | string)` at offset `n`: its length in
    /// tokens.
    fn interval_value_len(&self, n: usize) -> Option<usize> {
        let sign = usize::from(self.sym_at(n, Sym::Plus) || self.sym_at(n, Sym::Minus));
        let number = matches!(
            self.kind(n + sign),
            Some(TokenKind::Number(NumKind::Int | NumKind::Decimal))
        );
        let mut len = sign + usize::from(number);
        if !number {
            while self.single_string_at(n + len) {
                len += 1;
            }
        }
        (len > sign).then_some(len)
    }

    fn interval_value(&mut self) -> PR<IntervalValue> {
        let start = self.start();
        let negative = self.sym_at(0, Sym::Minus);
        self.pos += usize::from(negative || self.sym_at(0, Sym::Plus));
        let (value, is_string) = match self.kind(0) {
            Some(TokenKind::Number(_)) => {
                let v = self.text(0).to_owned();
                self.bump();
                (v, false)
            }
            // `interval_follows` / `interval_value_len` checked the shape.
            _ => (self.string_lit().ok_or(Fail::Syntax)?, true),
        };
        Ok(IntervalValue {
            negative,
            value,
            is_string,
            span: self.span_from(start),
        })
    }

    fn interval_unit(&mut self) -> PR<String> {
        if !self.any_kw_at(0, INTERVAL_UNITS) {
            return self.fail();
        }
        let unit = self.text(0).to_ascii_lowercase();
        self.bump();
        Ok(unit)
    }

    /// `INTERVAL value unit [value unit ...]`, `INTERVAL value unit TO
    /// unit`, or the typed literal `INTERVAL 'text'`.
    #[inline(never)]
    fn interval(&mut self) -> PR<Literal> {
        let start = self.start();
        self.bump();
        let single = self.interval_value_len(0) == Some(1) && self.single_string_at(0);
        if single && !self.any_kw_at(1, INTERVAL_UNITS) {
            let value = unescape_string(self.text(0), !self.opts.escaped_string_literals);
            self.bump();
            return Ok(Literal::Typed {
                type_name: "interval".to_owned(),
                value,
            });
        }
        let mut units = vec![];
        let mut range = None;
        loop {
            let value = self.interval_value()?;
            let unit = self.interval_unit()?;
            if self.kw_at(0, "TO") && self.any_kw_at(1, INTERVAL_UNITS) {
                self.bump();
                range = Some((value, unit, self.interval_unit()?));
                break;
            }
            units.push((value, unit));
            match self.interval_value_len(0) {
                Some(n) if self.any_kw_at(n, INTERVAL_UNITS) => {}
                _ => break,
            }
        }
        Ok(Literal::Interval(Box::new(Interval {
            units,
            range,
            span: self.span_from(start),
        })))
    }
}
