// SPDX-License-Identifier: Apache-2.0
//! The checks Spark's `AstBuilder` makes while it builds a plan from a
//! parse tree: `ParseException`s (and a few other errors) for statements
//! the grammar accepts. The parser keeps the facts they need in the tree;
//! these checks walk the finished tree in the order `AstBuilder` visits it
//! (a query's body before its `WITH` clause; `FROM`, the select list,
//! `WHERE`, `GROUP BY`, `HAVING` and `WINDOW` in that order; a cast's type
//! before its operand) and report the first error.
//!
//! | Check | Error class |
//! |---|---|
//! | data types | see [`types`] |
//! | numeric literals out of range / too precise | `INVALID_NUMERIC_LITERAL_RANGE`, `DECIMAL_PRECISION_EXCEEDS_MAX_PRECISION` |
//! | typed literals (`DATE '...'`, ...) | `INVALID_TYPED_LITERAL`, `UNSUPPORTED_TYPED_LITERAL` |
//! | interval literals | see [`interval`] |
//! | a string unit in `TIMESTAMPADD` and friends | `INVALID_PARAMETER_VALUE.DATETIME_UNIT` |
//! | `IN ()` | `INVALID_SQL_SYNTAX.EMPTY_IN_PREDICATE` |
//! | `LIKE ANY ()`, a frame bound that is not a literal | `_LEGACY_ERROR_TEMP_0064` |
//! | duplicate CTE names, duplicate window names | `DUPLICATED_CTE_NAMES`, `INVALID_SQL_SYNTAX.REPETITIVE_WINDOW_DEFINITION` |
//! | `MAX RECURSION LEVEL` without `RECURSIVE`, a table option without a value | `_LEGACY_ERROR_TEMP_0035` |
//! | `NATURAL JOIN LATERAL`, `JOIN LATERAL ... USING` | `INCOMPATIBLE_JOIN_TYPES`, `UNSUPPORTED_FEATURE.LATERAL_JOIN_USING` |
//! | `VALUES` rows of different widths | `INVALID_INLINE_TABLE.NUM_COLUMNS_MISMATCH` |
//! | `()` in `ROLLUP` / `CUBE` | `_LEGACY_ERROR_TEMP_0051` |
//!
//! Not checked here: an inline table whose expressions fail when Spark
//! evaluates them while parsing (`INVALID_INLINE_TABLE.FAILED_SQL_EXPRESSION_EVALUATION`),
//! which needs expression evaluation.

mod interval;
pub(crate) mod literal;
mod types;

use std::collections::HashSet;

use krishiv_common::error::{
    DuplicatedCteNames, IncompatibleJoinTypes, InvalidInlineTableNumColumnsMismatch,
    InvalidParameterValueDatetimeUnit, InvalidSqlSyntaxEmptyInPredicate,
    InvalidSqlSyntaxRepetitiveWindowDefinition, KrishivError, LegacyErrorTemp0035,
    LegacyErrorTemp0051, LegacyErrorTemp0064, UnsupportedFeatureLateralJoinUsing,
};

pub(crate) use self::types::data_type;
use crate::ParserOptions;
use crate::ast::{
    Expr, ExprKind, FrameBoundKind, FromClause, Function, GroupBy, GroupItem, GroupItemKind,
    GroupWith, GroupingSet, Join, JoinConstraint, Literal, NamedWindow, NumKind, Query, Relation,
    RelationExt, Select, SelectItem, SetExpr, SetExprKind, SortItem, Span, Statement, StructField,
    TableAlias, TableFactor, TableFactorKind, WindowRef, WindowSpec, With,
};
use crate::error::ParseError;
use crate::lexer::tokenize;

type R = Result<(), ParseError>;

fn at(error: impl Into<KrishivError>, span: Span) -> ParseError {
    ParseError {
        error: error.into(),
        span,
    }
}

/// Checks a statement; `sql` is its text.
pub(crate) fn statement(s: &Statement, sql: &str, opts: ParserOptions) -> R {
    match s {
        Statement::Query(q) => query(q, sql, opts),
        Statement::Explain(e) => statement(&e.statement, sql, opts),
    }
}

/// Checks a query statement; `sql` is its text.
pub(crate) fn query(q: &Query, sql: &str, opts: ParserOptions) -> R {
    Checker { sql, opts }.query(q)
}

/// Checks the column types of a table schema.
pub(crate) fn columns(columns: &[StructField]) -> R {
    columns.iter().try_for_each(|c| data_type(&c.data_type))
}

/// Functions whose first argument is a `datetimeUnit`: a string there is
/// an error (`TIMESTAMPADD('DAY', ...)`).
const UNIT_FUNCTIONS: &[&str] = &[
    "TIMESTAMPADD",
    "DATEADD",
    "DATE_ADD",
    "TIMESTAMPDIFF",
    "DATEDIFF",
    "DATE_DIFF",
    "TIMEDIFF",
];

struct Checker<'a> {
    sql: &'a str,
    opts: ParserOptions,
}

impl Checker<'_> {
    fn query(&self, q: &Query) -> R {
        self.set_expr(&q.body)?;
        self.sort_items(&q.order_by)?;
        q.cluster_by.iter().try_for_each(|e| self.expr(e))?;
        q.distribute_by.iter().try_for_each(|e| self.expr(e))?;
        self.sort_items(&q.sort_by)?;
        self.windows(&q.windows)?;
        if let Some(e) = q.limit.as_ref().and_then(|l| l.value.as_ref()) {
            self.expr(e)?;
        }
        if let Some(e) = &q.offset {
            self.expr(e)?;
        }
        if let Some(with) = &q.with {
            self.with(with, q.span)?;
        }
        Ok(())
    }

    /// The CTEs of a query (`query` is its span), then their names.
    fn with(&self, with: &With, query: Span) -> R {
        for cte in &with.ctes {
            self.query(&cte.query)?;
            if cte.max_recursion_level.is_some() && !with.recursive {
                return Err(at(
                    LegacyErrorTemp0035 {
                        message: "Cannot specify MAX RECURSION LEVEL when the CTE is not marked \
                                  as RECURSIVE"
                            .to_owned(),
                    },
                    query,
                ));
            }
        }
        let mut seen = HashSet::new();
        let mut duplicates: Vec<&str> = vec![];
        for cte in &with.ctes {
            let name = cte.name.value.to_lowercase();
            if !seen.insert(name.clone()) && !duplicates.iter().any(|d| d.to_lowercase() == name) {
                let first = with
                    .ctes
                    .iter()
                    .find(|c| c.name.value.to_lowercase() == name)
                    .map_or(cte.name.value.as_str(), |c| c.name.value.as_str());
                duplicates.push(first);
            }
        }
        if duplicates.is_empty() {
            return Ok(());
        }
        let names: Vec<String> = duplicates.iter().map(|n| quote(n)).collect();
        Err(at(
            DuplicatedCteNames {
                duplicate_names: names.join(", "),
            },
            query,
        ))
    }

    fn set_expr(&self, s: &SetExpr) -> R {
        // A left-deep chain of set operations is walked as a loop.
        let mut rights = vec![];
        let mut left = s;
        while let SetExprKind::SetOp { left: l, right, .. } = &left.kind {
            rights.push(right.as_ref());
            left = l;
        }
        self.set_operand(left)?;
        rights
            .iter()
            .rev()
            .try_for_each(|right| self.set_expr(right))
    }

    fn set_operand(&self, s: &SetExpr) -> R {
        match &s.kind {
            SetExprKind::Select(select) => self.select(select),
            SetExprKind::FromStatement(f) => {
                self.from(&f.from)?;
                f.bodies.iter().try_for_each(|q| self.query(q))
            }
            SetExprKind::Query(q) => self.query(q),
            SetExprKind::Values(v) => self.inline_table(&v.rows, v.alias.as_ref(), v.span),
            _ => Ok(()),
        }
    }

    fn select(&self, s: &Select) -> R {
        if let Some(from) = &s.from {
            self.from(from)?;
        }
        self.select_items(&s.projection)?;
        for view in &s.lateral_views {
            view.args.iter().try_for_each(|e| self.expr(e))?;
        }
        if let Some(e) = &s.selection {
            self.expr(e)?;
        }
        if let Some(g) = &s.group_by {
            self.group_by(g)?;
        }
        if let Some(e) = &s.having {
            self.expr(e)?;
        }
        self.windows(&s.windows)?;
        if let Some(e) = &s.qualify {
            self.expr(e)?;
        }
        for hint in &s.hints {
            hint.params.iter().try_for_each(|e| self.expr(e))?;
        }
        Ok(())
    }

    fn select_items(&self, items: &[SelectItem]) -> R {
        items.iter().try_for_each(|i| self.expr(&i.expr))
    }

    fn sort_items(&self, items: &[SortItem]) -> R {
        items.iter().try_for_each(|i| self.expr(&i.expr))
    }

    fn from(&self, f: &FromClause) -> R {
        f.relations.iter().try_for_each(|r| self.relation(r))?;
        for view in &f.lateral_views {
            view.args.iter().try_for_each(|e| self.expr(e))?;
        }
        if let Some(p) = &f.pivot {
            self.select_items(&p.aggregates)?;
            self.select_items(&p.values)?;
        }
        Ok(())
    }

    fn relation(&self, r: &Relation) -> R {
        self.factor(&r.primary)?;
        for ext in &r.extensions {
            match ext {
                RelationExt::Join(j) => self.join(j)?,
                RelationExt::Pivot(p) => {
                    self.select_items(&p.aggregates)?;
                    self.select_items(&p.values)?;
                }
                RelationExt::Unpivot(_) => {}
            }
        }
        Ok(())
    }

    fn join(&self, j: &Join) -> R {
        if j.lateral {
            if j.natural {
                return Err(at(
                    IncompatibleJoinTypes {
                        join_type1: "LATERAL".to_owned(),
                        join_type2: "NATURAL".to_owned(),
                    },
                    j.span,
                ));
            }
            if matches!(j.constraint, JoinConstraint::Using(_)) {
                return Err(at(UnsupportedFeatureLateralJoinUsing {}, j.span));
            }
        }
        self.factor(&j.right)?;
        match &j.constraint {
            JoinConstraint::On(e) => self.expr(e),
            _ => Ok(()),
        }
    }

    fn factor(&self, f: &TableFactor) -> R {
        match &f.kind {
            TableFactorKind::Table {
                name,
                temporal,
                options,
            } => {
                if let Some(t) = temporal {
                    self.expr(&t.value)?;
                }
                let missing: Vec<String> = options
                    .iter()
                    .filter(|o| o.value.is_none())
                    .map(|o| {
                        o.key
                            .parts
                            .iter()
                            .map(|p| p.value.as_str())
                            .collect::<Vec<_>>()
                            .join(".")
                    })
                    .collect();
                if !missing.is_empty() {
                    let error: KrishivError = LegacyErrorTemp0035 {
                        message: format!(
                            "Values must be specified for key(s): [{}]",
                            missing.join(",")
                        ),
                    }
                    .into();
                    // A key written as a string literal is never echoed (it
                    // may hold a secret, like `'password' 'x'` read as one
                    // key).
                    let string_key = options.iter().any(|o| o.value.is_none() && o.key.is_string);
                    let error = if string_key { error.redacted() } else { error };
                    return Err(at(error, name.span));
                }
                options
                    .iter()
                    .filter_map(|o| o.value.as_ref())
                    .try_for_each(|e| self.expr(e))
            }
            TableFactorKind::Subquery(q) => self.query(q),
            TableFactorKind::Nested(r) => self.relation(r),
            TableFactorKind::Values(rows) => self.inline_table(rows, f.alias.as_ref(), f.span),
            TableFactorKind::TableFunction { args, .. } => {
                args.iter().try_for_each(|a| self.expr(&a.expr))
            }
        }
    }

    /// `VALUES rows [AS alias(columns)]` at `span`: every row has as many
    /// columns as the alias names, or as the first row.
    fn inline_table(&self, rows: &[Expr], alias: Option<&TableAlias>, span: Span) -> R {
        rows.iter().try_for_each(|e| self.expr(e))?;
        let width = |e: &Expr| match &e.kind {
            ExprKind::Row(items) => items.len(),
            _ => 1,
        };
        let expected = alias
            .filter(|a| !a.columns.is_empty())
            .map_or_else(|| rows.first().map_or(0, width), |a| a.columns.len());
        match rows.iter().position(|r| width(r) != expected) {
            Some(row) => Err(at(
                InvalidInlineTableNumColumnsMismatch {
                    expected_num_cols: expected.to_string(),
                    actual_num_cols: width(&rows[row]).to_string(),
                    row_index: row.to_string(),
                },
                span,
            )),
            None => Ok(()),
        }
    }

    fn group_by(&self, g: &GroupBy) -> R {
        g.items
            .iter()
            .try_for_each(|i| self.group_item(i, g.span))?;
        if let Some(GroupWith::GroupingSets(sets)) = &g.with {
            sets.iter()
                .try_for_each(|s| s.exprs.iter().try_for_each(|e| self.expr(e)))?;
        }
        Ok(())
    }

    /// One item of a `GROUP BY` clause at `clause`.
    fn group_item(&self, item: &GroupItem, clause: Span) -> R {
        let sets = |sets: &[GroupingSet], element: &str| -> R {
            sets.iter()
                .try_for_each(|s| s.exprs.iter().try_for_each(|e| self.expr(e)))?;
            if sets.iter().any(|s| s.exprs.is_empty()) {
                return Err(at(
                    LegacyErrorTemp0051 {
                        element: element.to_owned(),
                    },
                    clause,
                ));
            }
            Ok(())
        };
        match &item.kind {
            GroupItemKind::Expr(e) => self.expr(e),
            GroupItemKind::Rollup(s) => sets(s, "ROLLUP"),
            GroupItemKind::Cube(s) => sets(s, "CUBE"),
            GroupItemKind::GroupingSets(items) => {
                items.iter().try_for_each(|i| self.group_item(i, clause))
            }
            GroupItemKind::Set(s) => s.exprs.iter().try_for_each(|e| self.expr(e)),
        }
    }

    /// A `WINDOW` clause: its specifications, then its names.
    fn windows(&self, windows: &[NamedWindow]) -> R {
        for w in windows {
            if let WindowRef::Spec(spec) = &w.window {
                self.window_spec(spec)?;
            }
        }
        let mut seen = HashSet::new();
        for w in windows {
            if !seen.insert(w.name.value.as_str()) {
                let start = self.token_before(windows[0].span.start);
                let end = windows.last().map_or(w.span.end, |l| l.span.end);
                return Err(at(
                    InvalidSqlSyntaxRepetitiveWindowDefinition {
                        window_name: quote(&w.name.value),
                    },
                    Span::new(start, end),
                ));
            }
        }
        Ok(())
    }

    fn window_spec(&self, w: &WindowSpec) -> R {
        w.partition_by.iter().try_for_each(|e| self.expr(e))?;
        self.sort_items(&w.order_by)?;
        let Some(frame) = &w.frame else {
            return Ok(());
        };
        for bound in std::iter::once(&frame.start).chain(&frame.end) {
            if let FrameBoundKind::Preceding(e) | FrameBoundKind::Following(e) = &bound.kind {
                self.expr(e)?;
                if !foldable(e) {
                    return Err(at(
                        LegacyErrorTemp0064 {
                            msg: "Frame bound value must be a literal.".to_owned(),
                        },
                        bound.span,
                    ));
                }
            }
        }
        Ok(())
    }

    fn function(&self, f: &Function) -> R {
        self.datetime_unit(f)?;
        f.args.iter().try_for_each(|a| self.expr(&a.expr))?;
        self.sort_items(&f.within_group)?;
        if let Some(e) = &f.filter {
            self.expr(e)?;
        }
        if let Some(WindowRef::Spec(spec)) = &f.over {
            self.window_spec(spec)?;
        }
        Ok(())
    }

    /// `TIMESTAMPADD('DAY', ...)`: the unit of these functions is a
    /// keyword, never a string.
    fn datetime_unit(&self, f: &Function) -> R {
        let [name] = f.name.parts.as_slice() else {
            return Ok(());
        };
        let unit_function = !name.quoted
            && UNIT_FUNCTIONS
                .iter()
                .any(|u| crate::keywords::is_kw(&name.value, u));
        if let [unit, _, _] = f.args.as_slice()
            && unit_function
            && unit.name.is_none()
            && matches!(unit.expr.kind, ExprKind::Literal(Literal::String(_)))
        {
            return Err(at(
                InvalidParameterValueDatetimeUnit {
                    parameter: "`unit`".to_owned(),
                    function_name: quote(&name.value),
                    invalid_value: unit.expr.span.text(self.sql).to_owned(),
                },
                f.span,
            ));
        }
        Ok(())
    }

    fn literal(&self, l: &Literal, span: Span) -> R {
        let legacy = self.opts.legacy_interval;
        match l {
            // Spark checks a plain integer or decimal's precision when it
            // analyzes the plan, not here.
            Literal::Number {
                kind: NumKind::Int | NumKind::Decimal,
                ..
            } => Ok(()),
            Literal::Number { text, kind } => literal::number(text, *kind)
                .map(drop)
                .map_err(|e| at(e, span)),
            Literal::Typed { type_name, value } => {
                let kind = literal::TypedKind::of(&type_name.value, type_name.quoted)
                    .map_err(|e| at(e, span))?;
                if kind == literal::TypedKind::Interval {
                    interval::typed(value, span, self.sql, legacy)
                } else {
                    literal::typed(kind, value, self.opts.timestamp_ntz).map_err(|e| at(e, span))
                }
            }
            Literal::Interval(iv) => interval::literal(iv, self.sql, legacy),
            _ => Ok(()),
        }
    }

    fn expr(&self, e: &Expr) -> R {
        // A left-deep chain of binary operators is walked as a loop.
        let mut rights = vec![];
        let mut left = e;
        while let ExprKind::Binary(l, _, r) = &left.kind {
            rights.push(r.as_ref());
            left = l;
        }
        self.expr_node(left)?;
        rights.iter().rev().try_for_each(|r| self.expr(r))
    }

    fn expr_node(&self, e: &Expr) -> R {
        match &e.kind {
            ExprKind::Literal(l) => self.literal(l, e.span),
            ExprKind::Field(inner, _) | ExprKind::Unary(_, inner) | ExprKind::Not(inner) => {
                self.expr(inner)
            }
            ExprKind::Nested(inner)
            | ExprKind::IsNull { expr: inner, .. }
            | ExprKind::IsTruth { expr: inner, .. } => self.expr(inner),
            ExprKind::Subscript(a, b) | ExprKind::Binary(a, _, b) => {
                self.expr(a)?;
                self.expr(b)
            }
            ExprKind::IsDistinctFrom { left, right, .. } => {
                self.expr(left)?;
                self.expr(right)
            }
            ExprKind::Between {
                expr, low, high, ..
            } => {
                self.expr(expr)?;
                self.expr(low)?;
                self.expr(high)
            }
            ExprKind::InList { expr, list, op, .. } => {
                self.expr(expr)?;
                if list.is_empty() {
                    return Err(at(InvalidSqlSyntaxEmptyInPredicate {}, *op));
                }
                list.iter().try_for_each(|e| self.expr(e))
            }
            ExprKind::InSubquery { expr, query, .. } => {
                self.expr(expr)?;
                self.query(query)
            }
            ExprKind::Like { expr, pattern, .. } | ExprKind::RLike { expr, pattern, .. } => {
                self.expr(expr)?;
                self.expr(pattern)
            }
            ExprKind::LikeQuantified { expr, patterns, .. } => {
                self.expr(expr)?;
                if patterns.is_empty() {
                    let start = self.token_after(expr.span.end);
                    return Err(at(
                        LegacyErrorTemp0064 {
                            msg: "Expected something between '(' and ')'.".to_owned(),
                        },
                        Span::new(start, e.span.end),
                    ));
                }
                patterns.iter().try_for_each(|p| self.expr(p))
            }
            ExprKind::Exists(q) | ExprKind::Subquery(q) => self.query(q),
            ExprKind::Cast {
                expr, data_type: t, ..
            } => {
                data_type(t)?;
                self.expr(expr)
            }
            ExprKind::Case {
                operand,
                branches,
                else_result,
            } => {
                if let Some(o) = operand {
                    self.expr(o)?;
                }
                for b in branches {
                    self.expr(&b.condition)?;
                    self.expr(&b.result)?;
                }
                else_result.as_ref().map_or(Ok(()), |e| self.expr(e))
            }
            ExprKind::Function(f) => self.function(f),
            ExprKind::Lambda { body, .. } => self.expr(body),
            ExprKind::Row(items) | ExprKind::Struct(items) => self.select_items(items),
            _ => Ok(()),
        }
    }

    /// The start of the last token before `pos`.
    fn token_before(&self, pos: usize) -> usize {
        tokenize(self.sql)
            .iter()
            .rev()
            .find(|t| !t.kind.is_trivia() && t.span.end <= pos)
            .map_or(pos, |t| t.span.start)
    }

    /// The start of the first token at or after `pos`.
    fn token_after(&self, pos: usize) -> usize {
        tokenize(self.sql)
            .iter()
            .find(|t| !t.kind.is_trivia() && t.span.start >= pos)
            .map_or(pos, |t| t.span.start)
    }
}

/// Whether a frame bound is a value Spark can fold: literals, and
/// operators and casts over them (no columns, calls or subqueries).
fn foldable(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Literal(_) => true,
        ExprKind::Nested(inner)
        | ExprKind::Unary(_, inner)
        | ExprKind::Not(inner)
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::IsNull { expr: inner, .. } => foldable(inner),
        ExprKind::Binary(a, _, b) => foldable(a) && foldable(b),
        ExprKind::Case {
            operand,
            branches,
            else_result,
        } => {
            operand.as_deref().is_none_or(foldable)
                && branches
                    .iter()
                    .all(|b| foldable(&b.condition) && foldable(&b.result))
                && else_result.as_deref().is_none_or(foldable)
        }
        _ => false,
    }
}

/// A name as Spark quotes it in errors: `` `name` ``.
fn quote(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}
