// SPDX-License-Identifier: Apache-2.0
//! `CHAR(n)` / `VARCHAR(n)` columns (design P2-T001b). The column's type is `string`; the declared type travels
//! beside it, as Spark carries it in a field's metadata (`__CHAR_VARCHAR_TYPE_STRING`): [`Resolver::declared`]
//! maps the attributes that hold such a column to their catalog string (`char(5)`).
//!
//! Read-side padding (`spark.sql.readSideCharPadding`): a scan of a table with `char(n)` columns is read through
//! `rpad(c, n, ' ')`, so a `CHAR(5)` holding `'N'` reads as `'N    '`. The padded column is a new attribute with
//! the column's name that carries the same declared type.

use std::sync::Arc;

use super::catalog::FunctionLookup;
use super::exprs::Bound;
use super::resolve::{Attr, Rel, Resolver};
use crate::{BoundColumn, Expr, ExprNode, Literal, Origin, ResolvedExpr, ResolvedFunction};
use krishiv_types::DataType;

/// `n` of a declared `char(n)`; `None` for `varchar(n)`, nested types and other strings.
pub(super) fn char_length(declared: &str) -> Option<i32> {
    declared
        .strip_prefix("char(")?
        .strip_suffix(')')?
        .trim()
        .parse()
        .ok()
}

impl Resolver<'_> {
    /// `rel`, a scan, read through `rpad(c, n, ' ')` for each of its `char(n)` columns; `rel` itself when it has
    /// none, when `spark.sql.legacy.charVarcharAsString` is on (the padding rule is off) or when the registry has
    /// no `rpad`.
    pub(super) fn pad_char_columns(&mut self, rel: Rel, origin: Origin) -> Rel {
        if self.ctx.conf.char_varchar_as_string {
            return rel;
        }
        let padded: Vec<(usize, i32)> = rel
            .out
            .iter()
            .enumerate()
            .filter_map(|(i, a)| {
                let n = char_length(self.declared.get(&a.col.id)?)?;
                Some((i, n))
            })
            .collect();
        if padded.is_empty() {
            return rel;
        }
        if self
            .rpad_call(Arc::new(Expr::literal(Literal::Null(None))), 0, false)
            .is_none()
        {
            return rel;
        }
        let mut exprs = Vec::with_capacity(rel.out.len());
        let mut out: Vec<Attr> = Vec::with_capacity(rel.out.len());
        for (i, a) in rel.out.iter().enumerate() {
            let Some(&(_, n)) = padded.iter().find(|(p, _)| *p == i) else {
                exprs.push(super::ops::col_expr(&a.col, Origin::UNKNOWN));
                out.push(a.clone());
                continue;
            };
            let Some(call) = self.rpad_call(
                super::ops::col_expr(&a.col, Origin::UNKNOWN),
                n,
                a.col.nullable,
            ) else {
                return rel;
            };
            let col = BoundColumn {
                id: self.fresh(),
                name: a.col.name.clone(),
                data_type: DataType::String,
                nullable: a.col.nullable,
            };
            if let Some(declared) = self.declared.get(&a.col.id).cloned() {
                self.declared.insert(col.id, declared);
            }
            exprs.push(Arc::new(Expr::new(ExprNode::Resolved(
                ResolvedExpr::Alias {
                    expr: call,
                    output: col.clone(),
                },
            ))));
            out.push(Attr {
                col,
                qual: Arc::clone(&a.qual),
                source: a.source.clone(),
            });
        }
        self.projection(origin, &rel, exprs, out)
    }
}

/// The declared length `n` of a bare `char(n)` attribute operand; `None` for any other operand (an expression,
/// a nested field, a `varchar` column).
fn bare_char(
    declared: &std::collections::HashMap<crate::AttrId, Arc<str>>,
    b: &Bound,
) -> Option<i32> {
    if b.shown.is_some() {
        return None;
    }
    match &b.expr.node {
        ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) => {
            char_length(declared.get(&c.id)?)
        }
        _ => None,
    }
}

/// A written string literal operand.
fn string_literal(b: &Bound) -> Option<&str> {
    match &b.expr.node {
        ExprNode::Literal(Literal::String(s)) if b.shown.is_none() => Some(s),
        _ => None,
    }
}

impl Resolver<'_> {
    /// `rpad(arg, n, ' ')`, `None` when the registry has no `rpad`.
    fn rpad_call(&self, arg: Arc<Expr>, n: i32, nullable: bool) -> Option<Arc<Expr>> {
        let FunctionLookup::Found(entry) = self
            .ctx
            .functions
            .lookup(&["rpad".to_owned()], self.ctx.profile)
        else {
            return None;
        };
        Some(Arc::new(Expr::new(ExprNode::Resolved(
            ResolvedExpr::Function(ResolvedFunction {
                id: entry.id.clone(),
                kind: entry.kind,
                args: vec![
                    arg,
                    Arc::new(Expr::literal(Literal::Integer(n))),
                    Arc::new(Expr::literal(Literal::String(" ".to_owned()))),
                ],
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type: DataType::String,
                nullable,
                deterministic: true,
            }),
        ))))
    }

    /// Spark's `ApplyCharTypePadding` for the comparison of `kids` (`=`, `<=>`, `<`, `<=`, `>`, `>=`): only a bare
    /// `char(n)` attribute is padded. Two of them pad the shorter to the longer; against a string literal of
    /// length L the column pads up to L when n < L, and the literal pads up to n when n > L. The operands as
    /// Spark shows them (`shown`) are kept, so names do not change. `None` when nothing pads.
    pub(super) fn pad_char_comparison(&self, kids: &[Bound]) -> Option<Vec<Bound>> {
        if self.ctx.conf.char_varchar_as_string || self.declared.is_empty() {
            return None;
        }
        let [a, b] = kids else { return None };
        let pad = |k: &Bound, to: i32| -> Option<Bound> {
            let e = self.rpad_call(Arc::clone(&k.expr), to, k.ty.nullable)?;
            Some(Bound {
                expr: e,
                shown: Some(k.shown()),
                ..k.clone()
            })
        };
        let literal = |k: &Bound, to: i32| -> Option<Bound> {
            let text = string_literal(k)?;
            let width = i32::try_from(text.chars().count()).ok()?;
            let padded = format!("{text}{}", " ".repeat(usize::try_from(to - width).ok()?));
            Some(Bound {
                expr: Arc::new(Expr::literal(Literal::String(padded)).with_origin(k.expr.origin)),
                shown: Some(k.shown()),
                ..k.clone()
            })
        };
        let length = |s: &str| i32::try_from(s.chars().count()).ok();
        match (bare_char(&self.declared, a), bare_char(&self.declared, b)) {
            (Some(la), Some(lb)) if la < lb => Some(vec![pad(a, lb)?, b.clone()]),
            (Some(la), Some(lb)) if lb < la => Some(vec![a.clone(), pad(b, la)?]),
            (Some(n), None) => {
                let l = length(string_literal(b)?)?;
                match n.cmp(&l) {
                    std::cmp::Ordering::Less => Some(vec![pad(a, l)?, b.clone()]),
                    std::cmp::Ordering::Greater => Some(vec![a.clone(), literal(b, n)?]),
                    std::cmp::Ordering::Equal => None,
                }
            }
            (None, Some(n)) => {
                let l = length(string_literal(a)?)?;
                match n.cmp(&l) {
                    std::cmp::Ordering::Less => Some(vec![a.clone(), pad(b, l)?]),
                    std::cmp::Ordering::Greater => Some(vec![literal(a, n)?, b.clone()]),
                    std::cmp::Ordering::Equal => None,
                }
            }
            _ => None,
        }
    }
}
