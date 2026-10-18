// SPDX-License-Identifier: Apache-2.0
//! Expression binding: names, stars, functions, casts, subqueries, output
//! names (Spark's `toPrettySQL`) and semantic keys.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::error::{
    AmbiguousColumnReference, AmbiguousLateralColumnAlias, AmbiguousReference,
    CannotResolveDataframeColumn, CannotResolveStarExpand, InvalidExtractBaseFieldType,
    InvalidSubqueryExpressionScalarSubqueryReturnMoreThanOneOutputColumn,
    InvalidUsageOfStarOrRegex, NestedAggregateFunction, UnresolvedColumnWithSuggestion,
    UnresolvedColumnWithoutSuggestion, UnresolvedRoutine,
};
use krishiv_types::DataType;
use twox_hash::XxHash3_128;

use super::bind::{addr, same, unsupported};
use super::catalog::quoted;
use super::resolve::{Attr, Env, Rel, Resolver};
use super::{AnalysisError, ArgType, FunctionLookup};
use crate::{
    AttrId, BoundColumn, Builtin, CastMode, ColumnRef, Expr, ExprNode, FunctionCall, FunctionId,
    FunctionKind, Literal, Origin, ResolvedExpr, ResolvedFunction, Star, SubqueryKind,
};

/// Where an operator's expressions resolve.
#[derive(Clone, Copy)]
pub(super) struct Cx<'s> {
    /// The operator's input attributes.
    pub(super) scope: &'s [Attr],
    /// The operator's children, whose `plan_id` carriers `df["a"]` uses.
    pub(super) rels: &'s [&'s Rel],
    /// Aliases defined earlier in the same select list (lateral column
    /// aliases), searched after the input; `None` where Spark has none.
    pub(super) lateral: Option<&'s [Attr]>,
    /// Select-list aliases a `GROUP BY` may name: (name, unresolved expr).
    pub(super) aliases: &'s [(String, Arc<Expr>)],
    /// Attributes searched after `scope`: an aggregate's grouping
    /// attributes (`HAVING`), a projection's input (`ORDER BY`).
    pub(super) extra: Option<&'s [Attr]>,
    /// Over an aggregate: its input, where aggregate-function arguments
    /// resolve.
    pub(super) agg_scope: Option<&'s [Attr]>,
    /// A name `scope` holds several times is looked up further rather
    /// than ambiguous (Spark's `ORDER BY` over a select list).
    pub(super) lenient: bool,
    /// Whether names may resolve in enclosing queries.
    pub(super) outer: bool,
    pub(super) env: &'s Arc<Env>,
    /// The operator, as Spark names it in errors.
    pub(super) op: &'static str,
}

/// A bound expression.
#[derive(Clone)]
pub(super) struct Bound {
    pub(super) expr: Arc<Expr>,
    pub(super) ty: ArgType,
    /// The first aggregate call inside (outside subqueries).
    pub(super) agg: Option<Origin>,
    /// The outermost query level referenced, for outer references.
    pub(super) outer: Option<usize>,
    /// The lateral column aliases referenced (their attribute ids).
    pub(super) lateral: Vec<AttrId>,
}

fn column(c: &BoundColumn) -> ArgType {
    ArgType {
        data_type: Some(c.data_type.clone()),
        nullable: c.nullable,
    }
}

pub(super) fn literal_type(v: &Literal) -> Option<DataType> {
    Some(match v {
        Literal::Null(t) => return t.clone(),
        Literal::Boolean(_) => DataType::Boolean,
        Literal::Byte(_) => DataType::Byte,
        Literal::Short(_) => DataType::Short,
        Literal::Integer(_) => DataType::Integer,
        Literal::Long(_) => DataType::Long,
        Literal::Float(_) => DataType::Float,
        Literal::Double(_) => DataType::Double,
        Literal::Decimal { data_type, .. } => DataType::Decimal(*data_type),
        Literal::String(_) => DataType::String,
        Literal::Binary(_) => DataType::Binary,
        Literal::Date(_) => DataType::Date,
        Literal::Timestamp(_) => DataType::Timestamp,
        Literal::TimestampNtz(_) => DataType::TimestampNtz,
        Literal::YearMonthInterval { .. } | Literal::DayTimeInterval { .. } => return None,
    })
}

enum Step {
    Enter(Arc<Expr>, bool),
    Exit(Arc<Expr>, bool),
    /// The first expression's result is the second's (a rewritten call).
    Link(usize, Arc<Expr>, bool),
}

impl Resolver<'_> {
    /// Binds `e` in `cx`, post-order with an explicit stack; a node shared
    /// within the expression is bound once.
    pub(super) fn expr(&mut self, e: &Arc<Expr>, cx: &Cx<'_>) -> Result<Bound, AnalysisError> {
        let mut done: HashMap<(usize, bool), Bound> = HashMap::new();
        // Rewritten calls stay alive for the pass: `done` is keyed by
        // address.
        let mut rewritten_calls: Vec<Arc<Expr>> = Vec::new();
        let mut stack = vec![Step::Enter(Arc::clone(e), false)];
        while let Some(step) = stack.pop() {
            match step {
                Step::Enter(x, in_agg) => {
                    if done.contains_key(&(addr(&*x), in_agg)) {
                        continue;
                    }
                    if let Some(rewritten) = self.star_args(&x, cx)? {
                        rewritten_calls.push(Arc::clone(&rewritten));
                        stack.push(Step::Link(addr(&*x), Arc::clone(&rewritten), in_agg));
                        stack.push(Step::Enter(rewritten, in_agg));
                        continue;
                    }
                    let inner = in_agg || self.is_aggregate(&x);
                    stack.push(Step::Exit(Arc::clone(&x), in_agg));
                    let mut kids = Vec::new();
                    x.for_each_child(&mut |c| kids.push(Arc::clone(c)));
                    stack.extend(kids.into_iter().rev().map(|c| Step::Enter(c, inner)));
                }
                Step::Link(from, to, in_agg) => {
                    if let Some(b) = done.get(&(addr(&*to), in_agg)).cloned() {
                        done.insert((from, in_agg), b);
                    }
                }
                Step::Exit(x, in_agg) => {
                    let inner = in_agg || self.is_aggregate(&x);
                    let mut kids = Vec::new();
                    x.for_each_child(&mut |c| {
                        if let Some(b) = done.get(&(addr(&**c), inner)) {
                            kids.push(b.clone());
                        }
                    });
                    let b = self.expr_node(&x, &kids, in_agg, cx)?;
                    done.insert((addr(&*x), in_agg), b);
                }
            }
        }
        done.remove(&(addr(&**e), false))
            .ok_or_else(|| unsupported("an expression shape", e.origin))
    }

    fn is_aggregate(&self, e: &Expr) -> bool {
        match &e.node {
            ExprNode::Function(f) => matches!(
                self.ctx.functions.lookup(std::slice::from_ref(&f.name), self.ctx.profile),
                FunctionLookup::Found(entry) if entry.kind == FunctionKind::Aggregate
            ),
            ExprNode::Resolved(ResolvedExpr::Function(f)) => f.kind == FunctionKind::Aggregate,
            _ => false,
        }
    }

    /// A call with `*` arguments, rewritten: `count(*)` is `count(1)`, any
    /// other call takes the input columns.
    fn star_args(&mut self, e: &Expr, cx: &Cx<'_>) -> Result<Option<Arc<Expr>>, AnalysisError> {
        let ExprNode::Function(f) = &e.node else {
            return Ok(None);
        };
        if !f.args.iter().any(|a| matches!(a.node, ExprNode::Star(_))) {
            return Ok(None);
        }
        let count_star = f.name.eq_ignore_ascii_case("count")
            && matches!(&f.args[..], [a] if matches!(&a.node, ExprNode::Star(s) if s.qualifier.is_empty() && s.plan_id.is_none()));
        let mut args = Vec::new();
        for a in &f.args {
            match &a.node {
                _ if count_star => args.push(Arc::new(
                    Expr::literal(Literal::Integer(1)).with_origin(a.origin),
                )),
                ExprNode::Star(s) => {
                    for attr in self.expand(s, cx, a.origin)? {
                        args.push(Arc::new(
                            Expr::new(ExprNode::Resolved(ResolvedExpr::Column(attr.col)))
                                .with_origin(a.origin),
                        ));
                    }
                }
                _ => args.push(Arc::clone(a)),
            }
        }
        let call = FunctionCall { args, ..f.clone() };
        Ok(Some(Arc::new(
            Expr::new(ExprNode::Function(call)).with_origin(e.origin),
        )))
    }

    /// The attributes `*`, `t.*` or `df["*"]` stands for.
    pub(super) fn expand(
        &mut self,
        s: &Star,
        cx: &Cx<'_>,
        origin: Origin,
    ) -> Result<Vec<Attr>, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        let out: Vec<Attr> = match s.plan_id {
            Some(id) => carriers(cx, id)
                .into_iter()
                .filter(|a| cx.scope.iter().any(|b| b.col.id == a.col.id))
                .collect(),
            None => cx
                .scope
                .iter()
                .filter(|a| {
                    let q = s.qualifier.len();
                    a.qual.len() >= q
                        && a.qual[a.qual.len() - q..]
                            .iter()
                            .zip(&s.qualifier)
                            .all(|(x, y)| same(x, y, cs))
                })
                .cloned()
                .collect(),
        };
        if out.is_empty() && !s.qualifier.is_empty() {
            return Err(AnalysisError::new(
                CannotResolveStarExpand {
                    target_string: quoted(&s.qualifier),
                    columns: names(cx.scope, false),
                },
                origin,
            ));
        }
        out.iter().for_each(|a| self.used(a));
        Ok(out)
    }

    /// Records a column of a securable as referenced.
    pub(super) fn used(&mut self, a: &Attr) {
        if let Some((i, name)) = &a.source
            && let Some(set) = self.columns.get_mut(*i)
        {
            set.insert(name.to_string());
        }
    }

    fn expr_node(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
        in_agg: bool,
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        let outer = kids.iter().filter_map(|k| k.outer).min();
        let agg = kids.iter().find_map(|k| k.agg);
        let mut lateral: Vec<AttrId> = Vec::new();
        for id in kids.iter().flat_map(|k| k.lateral.iter()) {
            if !lateral.contains(id) {
                lateral.push(*id);
            }
        }
        let keep = |ty| Bound {
            expr: Arc::clone(e),
            ty,
            agg,
            outer,
            lateral: lateral.clone(),
        };
        let rebuilt = |node: ExprNode| Arc::new(Expr::new(node).with_origin(e.origin));
        let args: Vec<Arc<Expr>> = kids.iter().map(|k| Arc::clone(&k.expr)).collect();
        Ok(match &e.node {
            ExprNode::Column(c) => self.column(c, e.origin, in_agg, cx)?,
            ExprNode::Literal(v) => keep(ArgType {
                nullable: matches!(v, Literal::Null(_)),
                data_type: literal_type(v),
            }),
            ExprNode::Star(_) => {
                return Err(AnalysisError::new(
                    InvalidUsageOfStarOrRegex {
                        elem: "'*'".to_owned(),
                        pretty_name: cx.op.to_owned(),
                    },
                    e.origin,
                ));
            }
            ExprNode::Function(f) => {
                let entry = match self
                    .ctx
                    .functions
                    .lookup(std::slice::from_ref(&f.name), self.ctx.profile)
                {
                    FunctionLookup::Found(entry) => entry,
                    FunctionLookup::KnownUnsupported => {
                        return Err(unsupported(format!("function {}", f.name), e.origin));
                    }
                    FunctionLookup::NotFound => {
                        return Err(AnalysisError::new(
                            UnresolvedRoutine {
                                routine_name: quoted(std::slice::from_ref(&f.name)),
                                search_path: self.b.search_path.clone(),
                            },
                            e.origin,
                        ));
                    }
                };
                let n = f.args.len();
                if entry.kind == FunctionKind::Aggregate
                    && let Some(inner) = kids.iter().find_map(|k| k.agg)
                {
                    return Err(AnalysisError::new(NestedAggregateFunction {}, inner));
                }
                let types: Vec<ArgType> = kids[..n].iter().map(|k| k.ty.clone()).collect();
                let ty = (entry.result)(&types).map_err(|err| AnalysisError::new(err, e.origin))?;
                let data_type = ty.data_type.clone().ok_or_else(|| {
                    unsupported("values of the NULL and interval types", e.origin)
                })?;
                let call = ResolvedFunction {
                    id: entry.id,
                    kind: entry.kind,
                    args: args[..n].to_vec(),
                    distinct: f.distinct,
                    filter: args.get(n).cloned(),
                    data_type,
                    nullable: ty.nullable,
                };
                Bound {
                    agg: if entry.kind == FunctionKind::Aggregate {
                        Some(e.origin)
                    } else {
                        agg
                    },
                    expr: rebuilt(ExprNode::Resolved(ResolvedExpr::Function(call))),
                    ..keep(ty)
                }
            }
            ExprNode::Cast { to, mode, .. } => {
                let mode = match mode {
                    CastMode::Default if self.ctx.conf.ansi => CastMode::Ansi,
                    CastMode::Default => CastMode::Legacy,
                    m => *m,
                };
                let child = &kids[0].ty;
                let changes = child.data_type.as_ref() != Some(to);
                let nullable = child.nullable
                    || mode == CastMode::Try
                    || (mode == CastMode::Legacy && changes);
                Bound {
                    expr: rebuilt(ExprNode::Cast {
                        expr: Arc::clone(&args[0]),
                        to: to.clone(),
                        mode,
                    }),
                    ..keep(ArgType {
                        data_type: Some(to.clone()),
                        nullable,
                    })
                }
            }
            ExprNode::Alias { name, .. } => {
                let output = self.attr(name, &kids[0].ty, e.origin)?;
                Bound {
                    expr: rebuilt(ExprNode::Resolved(ResolvedExpr::Alias {
                        expr: Arc::clone(&args[0]),
                        output: output.clone(),
                    })),
                    ..keep(column(&output))
                }
            }
            ExprNode::Subquery { kind, plan } => {
                let env = self.sub_env(cx);
                let rel = self.rel(plan, &env)?;
                let (data_type, nullable) = match kind {
                    SubqueryKind::Scalar => {
                        if rel.out.len() != 1 {
                            return Err(AnalysisError::new(
                                InvalidSubqueryExpressionScalarSubqueryReturnMoreThanOneOutputColumn {
                                    number: rel.out.len().to_string(),
                                },
                                e.origin,
                            ));
                        }
                        (Some(rel.out[0].col.data_type.clone()), true)
                    }
                    SubqueryKind::Exists => (Some(DataType::Boolean), false),
                    SubqueryKind::In(_) => (Some(DataType::Boolean), true),
                };
                let kind = match kind {
                    SubqueryKind::In(_) => SubqueryKind::In(args),
                    k => k.clone(),
                };
                let level = cx.env.outer.len();
                Bound {
                    expr: rebuilt(ExprNode::Subquery {
                        kind,
                        plan: Arc::clone(&rel.plan),
                    }),
                    outer: [outer, rel.outer.filter(|&l| l < level)]
                        .into_iter()
                        .flatten()
                        .min(),
                    ..keep(ArgType {
                        data_type,
                        nullable,
                    })
                }
            }
            ExprNode::Resolved(r) => keep(match r {
                ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c) => column(c),
                ResolvedExpr::Alias { output, .. } => column(output),
                ResolvedExpr::Function(f) => ArgType {
                    data_type: Some(f.data_type.clone()),
                    nullable: f.nullable,
                },
            }),
        })
    }

    /// A new attribute named `name` of type `ty`.
    pub(super) fn attr(
        &mut self,
        name: &str,
        ty: &ArgType,
        origin: Origin,
    ) -> Result<BoundColumn, AnalysisError> {
        let data_type = ty
            .data_type
            .clone()
            .ok_or_else(|| unsupported("values of the NULL and interval types", origin))?;
        Ok(BoundColumn {
            id: self.fresh(),
            name: name.to_owned(),
            data_type,
            nullable: ty.nullable,
        })
    }

    /// A column reference, in Spark's order: the input, lateral column
    /// aliases (a select list), select-list aliases (a `GROUP BY`),
    /// enclosing queries.
    fn column(
        &mut self,
        c: &ColumnRef,
        origin: Origin,
        in_agg: bool,
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        let bound = |expr: ExprNode, col: &BoundColumn, outer| Bound {
            expr: Arc::new(Expr::new(expr).with_origin(origin)),
            ty: column(col),
            agg: None,
            outer,
            lateral: Vec::new(),
        };
        let renamed = |a: &Attr, parts: &[String], used: usize| {
            // A reference keeps the spelling it was written with.
            let mut col = a.col.clone();
            col.name.clone_from(&parts[used - 1]);
            col
        };
        if let Some(id) = c.plan_id {
            let a = self.plan_column(id, &c.name, cx, origin)?;
            self.used(&a);
            let col = renamed(&a, &c.name, c.name.len());
            return Ok(bound(
                ExprNode::Resolved(ResolvedExpr::Column(col.clone())),
                &col,
                None,
            ));
        }
        let (scope, extra) = match cx.agg_scope {
            Some(input) if in_agg => (input, None),
            _ => (cx.scope, cx.extra),
        };
        for (i, level) in [Some(scope), extra].into_iter().enumerate() {
            let Some(level) = level else { continue };
            let lenient = cx.lenient && i == 0;
            if let Some((a, used)) = self.find(level, &c.name, origin, lenient)? {
                self.used(&a);
                let col = renamed(&a, &c.name, used);
                return Ok(bound(
                    ExprNode::Resolved(ResolvedExpr::Column(col.clone())),
                    &col,
                    None,
                ));
            }
        }
        if let ([name], Some(lateral)) = (&c.name[..], cx.lateral) {
            let cs = self.ctx.conf.case_sensitive;
            let hits: Vec<&Attr> = lateral
                .iter()
                .filter(|a| same(&a.col.name, name, cs))
                .collect();
            match hits[..] {
                [] => {}
                [a] => {
                    let mut col = a.col.clone();
                    col.name.clone_from(name);
                    let mut b = bound(
                        ExprNode::Resolved(ResolvedExpr::Column(col.clone())),
                        &col,
                        None,
                    );
                    b.lateral.push(col.id);
                    return Ok(b);
                }
                _ => {
                    return Err(AnalysisError::new(
                        AmbiguousLateralColumnAlias {
                            name: quoted(std::slice::from_ref(name)),
                            n: hits.len().to_string(),
                        },
                        Origin::UNKNOWN,
                    ));
                }
            }
        }
        if let [name] = &c.name[..]
            && let Some((_, alias)) = cx
                .aliases
                .iter()
                .find(|(n, _)| same(n, name, self.ctx.conf.case_sensitive))
        {
            let inner = Cx {
                aliases: &[],
                ..*cx
            };
            return self.expr(alias, &inner);
        }
        let levels = cx.env.outer.iter().enumerate().rev();
        for (level, outer) in levels.filter(|_| cx.outer) {
            if let Some((a, used)) = self.find(&outer.scope, &c.name, origin, false)? {
                self.used(&a);
                let col = renamed(&a, &c.name, used);
                return Ok(bound(
                    ExprNode::Resolved(ResolvedExpr::OuterColumn(col.clone())),
                    &col,
                    Some(level),
                ));
            }
        }
        Err(unresolved(&c.name, cx.scope, origin))
    }

    /// `df["a"]`: the column `name` of the relations carrying `id` in the
    /// operator's subtree.
    fn plan_column(
        &self,
        id: crate::PlanId,
        name: &[String],
        cx: &Cx<'_>,
        origin: Origin,
    ) -> Result<Attr, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        let pick = |carried: Vec<Attr>, scope: &[Attr]| {
            let mut hits: Vec<Attr> = Vec::new();
            for a in carried {
                let visible = scope.iter().any(|s| s.col.id == a.col.id);
                if visible
                    && name.len() == 1
                    && same(&a.col.name, &name[0], cs)
                    && !hits.iter().any(|h| h.col.id == a.col.id)
                {
                    hits.push(a);
                }
            }
            hits
        };
        let sql_name = format!("\"{}\"", name.join("."));
        // The error names the DataFrame the reference belongs to.
        let origin = match origin.plan_id {
            Some(_) => origin,
            None => origin.with_plan_id(id),
        };
        // Only the operator's own subtree: Spark 4.2 does not resolve a
        // DataFrame reference against an enclosing query (golden case
        // `outer`).
        let mut hits = pick(carriers(cx, id), cx.scope);
        match hits.len() {
            1 => Ok(hits.remove(0)),
            0 => Err(AnalysisError::new(
                CannotResolveDataframeColumn { name: sql_name },
                origin,
            )),
            _ => Err(AnalysisError::new(
                AmbiguousColumnReference { name: sql_name },
                origin,
            )),
        }
    }

    /// The attribute `parts` names in `scope` and how many parts it used:
    /// with a qualifier of 3, 2 or 1 parts (a suffix of the attribute's),
    /// then by name alone; the first form with any match decides. Several
    /// matches are `AMBIGUOUS_REFERENCE`, or no match when `lenient`.
    fn find(
        &self,
        scope: &[Attr],
        parts: &[String],
        origin: Origin,
        lenient: bool,
    ) -> Result<Option<(Attr, usize)>, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        for q in (0..parts.len().min(4)).rev() {
            let mut hits: Vec<&Attr> = Vec::new();
            for a in scope {
                let qualified = a.qual.len() >= q
                    && a.qual[a.qual.len() - q..]
                        .iter()
                        .zip(&parts[..q])
                        .all(|(x, y)| same(x, y, cs));
                if qualified
                    && same(&a.col.name, &parts[q], cs)
                    && !hits.iter().any(|h| h.col.id == a.col.id)
                {
                    hits.push(a);
                }
            }
            match hits[..] {
                [] => continue,
                [a] if parts.len() > q + 1 => {
                    // Struct fields: no nested types exist yet, so any
                    // field access is on a non-struct column.
                    return Err(AnalysisError::new(
                        InvalidExtractBaseFieldType {
                            base: format!("\"{}\"", parts[q]),
                            other: format!("\"{}\"", a.col.data_type.sql()),
                        },
                        origin,
                    ));
                }
                [a] => return Ok(Some((a.clone(), q + 1))),
                _ if lenient => return Ok(None),
                _ => {
                    let mut names: Vec<String> = hits
                        .iter()
                        .map(|a| {
                            let mut p = a.qual.to_vec();
                            p.push(a.col.name.clone());
                            quoted(&p)
                        })
                        .collect();
                    names.sort();
                    return Err(AnalysisError::new(
                        AmbiguousReference {
                            name: quoted(parts),
                            reference_names: format!("[{}]", names.join(", ")),
                        },
                        origin,
                    ));
                }
            }
        }
        Ok(None)
    }

    pub(super) fn sub_env(&mut self, cx: &Cx<'_>) -> Arc<Env> {
        let mut outer = cx.env.outer.clone();
        outer.push(super::resolve::Level {
            scope: cx.scope.to_vec().into(),
        });
        self.env(cx.env.ctes.clone(), cx.env.ctes_id, outer)
    }
}

/// The attributes carrying `id` below the operator.
fn carriers(cx: &Cx<'_>, id: crate::PlanId) -> Vec<Attr> {
    cx.rels
        .iter()
        .filter_map(|r| r.carriers.get(&id))
        .flatten()
        .flat_map(|out| out.iter().cloned())
        .collect()
}

/// `UNRESOLVED_COLUMN`, with up to five of `scope`'s columns ranked by edit
/// distance (qualified by their last qualifier part when `parts` is).
pub(super) fn unresolved(parts: &[String], scope: &[Attr], origin: Origin) -> AnalysisError {
    let target = parts.join(".");
    let qualified = parts.len() > 1;
    let mut ranked: Vec<(usize, Vec<String>)> = scope
        .iter()
        .map(|a| {
            let mut p: Vec<String> = a
                .qual
                .last()
                .filter(|_| qualified)
                .cloned()
                .into_iter()
                .collect();
            p.push(a.col.name.clone());
            (levenshtein(&p.join("."), &target), p)
        })
        .collect();
    ranked.sort_by_key(|(d, _)| *d);
    let object_name = quoted(parts);
    if ranked.is_empty() {
        return AnalysisError::new(UnresolvedColumnWithoutSuggestion { object_name }, origin);
    }
    let proposal = ranked
        .iter()
        .take(5)
        .map(|(_, p)| quoted(p))
        .collect::<Vec<_>>()
        .join(", ");
    AnalysisError::new(
        UnresolvedColumnWithSuggestion {
            object_name,
            proposal,
        },
        origin,
    )
}

/// The quoted names of `attrs`, comma-separated.
pub(super) fn names(attrs: &[Attr], _qualified: bool) -> String {
    attrs
        .iter()
        .map(|a| quoted(std::slice::from_ref(&a.col.name)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn levenshtein(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.chars().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let cur = row[j + 1];
            row[j + 1] = (prev + usize::from(x != *y)).min(row[j] + 1).min(cur + 1);
            prev = cur;
        }
    }
    row[b.len()]
}

/// The longest generated name, in bytes.
const MAX_NAME: usize = 64 << 10;

/// Spark's generated name for an unaliased expression (`toPrettySQL`):
/// `(a + 1)`, `count(1)`, `CASE WHEN ... END`. Built bottom-up with an
/// explicit stack, once per distinct node.
pub(super) fn pretty(e: &Expr) -> String {
    let mut done: HashMap<usize, String> = HashMap::new();
    let mut stack = vec![(e, false)];
    while let Some((x, expanded)) = stack.pop() {
        if done.contains_key(&addr(x)) {
            continue;
        }
        let mut kids = Vec::new();
        x.for_each_child(&mut |c| kids.push(&**c));
        if !expanded {
            stack.push((x, true));
            stack.extend(kids.into_iter().rev().map(|c| (c, false)));
            continue;
        }
        let k: Vec<&str> = kids
            .iter()
            .map(|c| done.get(&addr(*c)).map_or("", String::as_str))
            .collect();
        let mut s = pretty_node(x, &k);
        if s.len() > MAX_NAME {
            // A shared subexpression repeated in a name doubles it per level:
            // names stay bounded (Spark's would not).
            let mut end = MAX_NAME;
            while !s.is_char_boundary(end) {
                end -= 1;
            }
            s.truncate(end);
            s.push_str("...");
        }
        done.insert(addr(x), s);
    }
    done.remove(&addr(e)).unwrap_or_default()
}

fn pretty_node(x: &Expr, k: &[&str]) -> String {
    let list = |v: &[&str]| v.join(", ");
    match &x.node {
        ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) => {
            c.name.clone()
        }
        ExprNode::Column(c) => c.name.last().cloned().unwrap_or_default(),
        ExprNode::Literal(v) => pretty_literal(v),
        ExprNode::Cast { to, .. } => format!("CAST({} AS {})", k[0], to.sql()),
        ExprNode::Alias { .. } | ExprNode::Resolved(ResolvedExpr::Alias { .. }) => k[0].to_owned(),
        ExprNode::Subquery { kind, .. } => match kind {
            SubqueryKind::Scalar => "scalarsubquery()".to_owned(),
            SubqueryKind::Exists => "exists()".to_owned(),
            SubqueryKind::In(_) => format!("({} IN (listquery()))", list(k)),
        },
        ExprNode::Star(_) => "*".to_owned(),
        ExprNode::Function(FunctionCall { name, distinct, .. }) => call(name, None, *distinct, k),
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            let op = match &f.id {
                FunctionId::Builtin(b) => Some(*b),
                FunctionId::Registry { name, .. } => Builtin::from_name(name),
            };
            let n = f.args.len().min(k.len());
            let text = call(f.id.name(), op, f.distinct, &k[..n]);
            match k.get(n) {
                Some(filter) => format!("{text} FILTER (WHERE {filter})"),
                None => text,
            }
        }
    }
}

fn call(name: &str, op: Option<Builtin>, distinct: bool, k: &[&str]) -> String {
    let infix = |sym: &str| format!("({} {sym} {})", k[0], k[1]);
    match (op, k.len()) {
        (Some(Builtin::Eq), 2) => infix("="),
        (Some(Builtin::EqNullSafe), 2) => infix("<=>"),
        (Some(Builtin::Lt), 2) => infix("<"),
        (Some(Builtin::LtEq), 2) => infix("<="),
        (Some(Builtin::Gt), 2) => infix(">"),
        (Some(Builtin::GtEq), 2) => infix(">="),
        (Some(Builtin::And), 2) => infix("AND"),
        (Some(Builtin::Or), 2) => infix("OR"),
        (Some(Builtin::Add), 2) => infix("+"),
        (Some(Builtin::Subtract), 2) => infix("-"),
        (Some(Builtin::Multiply), 2) => infix("*"),
        (Some(Builtin::Divide), 2) => infix("/"),
        (Some(Builtin::Remainder), 2) => infix("%"),
        (Some(Builtin::Like), 2) => format!("{} LIKE {}", k[0], k[1]),
        (Some(Builtin::Not), 1) => format!("(NOT {})", k[0]),
        (Some(Builtin::Negative), 1) => format!("(- {})", k[0]),
        (Some(Builtin::IsNull), 1) => format!("({} IS NULL)", k[0]),
        (Some(Builtin::IsNotNull), 1) => format!("({} IS NOT NULL)", k[0]),
        (Some(Builtin::In), n) if n > 0 => format!("({} IN ({}))", k[0], k[1..].join(", ")),
        (Some(Builtin::When), _) => {
            let mut s = "CASE".to_owned();
            for pair in k.chunks(2) {
                match pair {
                    [c, v] => s.push_str(&format!(" WHEN {c} THEN {v}")),
                    [e] => s.push_str(&format!(" ELSE {e}")),
                    _ => {}
                }
            }
            s + " END"
        }
        _ if name.eq_ignore_ascii_case("extract") && k.len() == 2 => {
            format!("extract({} FROM {})", k[0], k[1])
        }
        _ => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{})", name.to_lowercase(), k.join(", "))
        }
    }
}

fn pretty_literal(v: &Literal) -> String {
    let text = v.to_string();
    match v {
        // A URL's credentials never reach a name (ADR 0038).
        Literal::String(s) => krishiv_common::secret::SecretUrl::parse(s)
            .map_or_else(|| s.clone(), |url| url.to_string()),
        Literal::Null(_) => "NULL".to_owned(),
        Literal::Byte(_) | Literal::Short(_) | Literal::Long(_) => {
            text.trim_end_matches(['Y', 'S', 'L']).to_owned()
        }
        Literal::Float(x) => format!("{x:?}"),
        Literal::Double(x) => format!("{x:?}"),
        Literal::Decimal { .. } => text.trim_end_matches("BD").to_owned(),
        _ => text,
    }
}

/// A semantic key: equal for expressions that compute the same value,
/// whatever their origins, column spellings or (top-level) alias.
pub(super) fn semantic_key(e: &Expr, memo: &mut HashMap<usize, u128>) -> u128 {
    let e = match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => &**expr,
        _ => e,
    };
    let mut stack = vec![(e, false)];
    while let Some((x, expanded)) = stack.pop() {
        if memo.contains_key(&addr(x)) {
            continue;
        }
        let mut kids = Vec::new();
        x.for_each_child(&mut |c| kids.push(&**c));
        if !expanded {
            stack.push((x, true));
            stack.extend(kids.into_iter().map(|c| (c, false)));
            continue;
        }
        let mut b = match &x.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => format!("c{}", c.id.0),
            ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => format!("o{}", c.id.0),
            ExprNode::Resolved(ResolvedExpr::Function(f)) => {
                format!("f{}:{:?}:{}", f.id.name(), f.kind, f.distinct)
            }
            ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => format!("a{}", output.id.0),
            ExprNode::Literal(v) => format!("l{v:?}"),
            ExprNode::Cast { to, mode, .. } => format!("x{}:{mode:?}", to.simple_string()),
            ExprNode::Subquery { plan, .. } => format!("s{}", addr(&**plan)),
            other => format!("u{other:?}"),
        }
        .into_bytes();
        for c in kids {
            b.extend_from_slice(&memo.get(&addr(c)).copied().unwrap_or(0).to_le_bytes());
        }
        memo.insert(addr(x), XxHash3_128::oneshot(&b));
    }
    memo.get(&addr(e)).copied().unwrap_or(0)
}
