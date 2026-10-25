// SPDX-License-Identifier: Apache-2.0
//! The relational operators of the Resolve pass and their rewrites.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::convert::Infallible;
use std::sync::Arc;

use krishiv_common::error::{
    AmbiguousReference, AssignmentArityMismatch, DatatypeMismatchFilterNotBoolean,
    FailedToParseTooComplex, GroupByAggregate, GroupByPosAggregate, GroupByPosOutOfRange,
    InvalidLimitLikeExpressionDataType, InvalidWhereCondition, JoinConditionIsNotBooleanType,
    MissingAggregation, MissingGroupBy, OrderByPosOutOfRange, QualifyAggregateNotAllowed,
    QualifyRequiresWindowFunction, StarGroupByPos, SumOfLimitAndOffsetExceedsMaxInt,
    TableOrViewNotFound, UnresolvedAllInGroupBy, UnresolvedUsingColumnForJoin,
    UnsupportedFeatureLateralColumnAliasInAggregateFunc,
    UnsupportedFeatureLateralColumnAliasInGroupBy,
};

use super::analytics::Analytics;
use super::bind::{BoundTable, same, unsupported};
use super::catalog::quoted;
use super::coerce::quoted_type;
use super::exprs::{Bound, Cx, WindowUse, names, pretty, semantic_key, unresolved};
use super::resolve::{AggInfo, Attr, Env, Rel, Resolver, Widen};
use super::{AnalysisError, ArgType, TableKind};
use crate::{
    AttrId, BoundColumn, CteRef, Expr, ExprNode, FunctionCall, FunctionKind, Grouping,
    JoinCondition, JoinKind, Literal, LogicalPlan, Origin, PlanNode, ReadSource, ResolvedExpr,
    ResolvedFunction, Scan, SortOrder,
};

/// What [`Resolver::group_substitute`] reads and grows: the aggregate, the
/// attributes it shows, and the output it adds for grouping expressions the
/// select list does not name.
struct Grouped<'a> {
    info: &'a AggInfo,
    env: &'a Arc<Env>,
    op: &'static str,
    visible: &'a HashSet<AttrId>,
    memo: &'a mut HashMap<usize, u128>,
    known: &'a mut Vec<(u128, Attr)>,
    grown: &'a mut AggInfo,
    pass: &'a mut Vec<Attr>,
}

/// The two attributes of a `USING` column.
type Keys = (Attr, Attr);

/// The largest attribute id: ids are numbered densely from 0, and a
/// self-join instance reserves its subtree's span; past this (about 2^62),
/// a plan is too complex.
pub(super) const MAX_ID: u64 = 1 << 62;

pub(super) fn col_expr(c: &BoundColumn, origin: Origin) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(c.clone()))).with_origin(origin))
}

pub(super) fn column_bound(a: &Attr, origin: Origin) -> Bound {
    Bound {
        expr: col_expr(&a.col, origin),
        ty: ArgType {
            data_type: Some(a.col.data_type.clone()),
            nullable: a.col.nullable,
        },
        agg: None,
        win: None,
        outer: None,
        lateral: Vec::new(),
        shown: None,
    }
}

pub(super) fn no_qual() -> Arc<[String]> {
    Arc::from(Vec::new())
}

impl Resolver<'_> {
    /// A table, view or CTE by name: CTEs shadow tables, inner ones outer.
    pub(super) fn read(
        &mut self,
        p: &LogicalPlan,
        name: &[String],
        options: &BTreeMap<String, krishiv_common::secret::OptionValue>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        if let [n] = name
            && let Some((cte, def)) = env.ctes.iter().rev().find(|(c, _)| same(c, n, cs))
        {
            let output: Vec<BoundColumn> = def
                .out
                .iter()
                .map(|a| BoundColumn {
                    id: self.fresh(),
                    ..a.col.clone()
                })
                .collect();
            let qual: Arc<[String]> = Arc::from(vec![n.clone()]);
            let out = output
                .iter()
                .map(|c| Attr {
                    col: c.clone(),
                    qual: Arc::clone(&qual),
                    source: None,
                })
                .collect();
            let node = PlanNode::CteRef(CteRef {
                name: cte.clone(),
                source: def.out.iter().map(|a| a.col.id).collect(),
                output,
            });
            return Ok(self.make(p.origin, node, out, &[]));
        }
        if let Some(&BoundTable::Files(i)) = self.b.tables.get(name) {
            return Ok(self.files(p, i));
        }
        let Some(BoundTable::Found { meta, securable }) = self.b.tables.get(name) else {
            return Err(AnalysisError::new(
                TableOrViewNotFound {
                    relation_name: quoted(name),
                    search_path: self.b.search_path.clone(),
                },
                p.origin,
            ));
        };
        if !matches!(meta.kind, TableKind::Table) {
            return Err(unsupported("views", p.origin));
        }
        let parts = meta.ident.parts();
        let qual: Arc<[String]> = Arc::from(parts.clone());
        let mut out = Vec::new();
        for c in &meta.columns {
            let col = BoundColumn {
                id: self.fresh(),
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                nullable: c.nullable,
            };
            out.push(Attr {
                col,
                qual: Arc::clone(&qual),
                source: Some((*securable, Arc::from(c.name.as_str()))),
            });
        }
        let node = PlanNode::Scan(Scan {
            source: ReadSource::Table {
                name: parts,
                options: options.clone(),
            },
            output: out.iter().map(|a| a.col.clone()).collect(),
            version: meta.version,
        });
        Ok(self.make(p.origin, node, out, &[]))
    }

    /// The files of [`super::bind::Bindings::files`] entry `i`, read by
    /// `p`: a scan without a qualifier (Spark names neither the path nor
    /// the format), with fresh attributes.
    pub(super) fn files(&mut self, p: &LogicalPlan, i: usize) -> Rel {
        let bound = &self.b.files[i];
        let out: Vec<Attr> = bound
            .columns
            .iter()
            .map(|c| Attr {
                col: BoundColumn {
                    id: self.fresh(),
                    name: c.name.clone(),
                    data_type: c.data_type.clone(),
                    nullable: c.nullable,
                },
                qual: no_qual(),
                source: None,
            })
            .collect();
        let node = PlanNode::Scan(Scan {
            source: bound.read.clone(),
            output: out.iter().map(|a| a.col.clone()).collect(),
            version: None,
        });
        self.make(p.origin, node, out, &[])
    }

    /// `c` as an instance: every id inside shifted by `offset`; `plan` is
    /// the `Shift` node when it exists already. Ids past `MAX_ID` are an
    /// error, never an overflow.
    pub(super) fn shifted(
        &self,
        c: &Rel,
        offset: u64,
        plan: Option<Arc<LogicalPlan>>,
    ) -> Result<Rel, AnalysisError> {
        let (_, hi) = c.ids.unwrap_or((0, 0));
        if hi.checked_add(offset).is_none_or(|top| top > MAX_ID) {
            return Err(AnalysisError::new(
                FailedToParseTooComplex {},
                c.plan.origin,
            ));
        }
        let sh = |a: &Attr| Attr {
            col: BoundColumn {
                id: AttrId(a.col.id.0 + offset),
                ..a.col.clone()
            },
            ..a.clone()
        };
        let carriers = c
            .carriers
            .iter()
            .map(|(id, outs)| {
                let outs = outs
                    .iter()
                    .map(|o| o.iter().map(sh).collect::<Vec<_>>().into())
                    .collect();
                (*id, outs)
            })
            .collect();
        let plan = plan.unwrap_or_else(|| {
            Arc::new(LogicalPlan::new(PlanNode::Shift {
                input: Arc::clone(&c.plan),
                offset,
            }))
        });
        Ok(Rel {
            plan,
            out: c.out.iter().map(sh).collect::<Vec<_>>().into(),
            ids: c.ids.map(|(lo, hi)| (lo + offset, hi + offset)),
            outer: None,
            carriers: Arc::new(carriers),
            agg: None,
            widen: None,
        })
    }

    /// A fresh instance of `c`, for the second side of a self-join.
    pub(super) fn instance(&mut self, c: &Rel) -> Result<Rel, AnalysisError> {
        let (lo, hi) = c.ids.unwrap_or((0, 0));
        // `next_id` is past every id of the input plan, so the instance's
        // ids are fresh.
        let offset = self.next_id.saturating_sub(lo);
        let r = self.shifted(c, offset, None)?;
        self.next_id += hi - lo + 1;
        Ok(r)
    }

    pub(super) fn project(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        self.project_items(p, &c, exprs, env)
    }

    pub(super) fn project_items(
        &mut self,
        p: &LogicalPlan,
        c: &Rel,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let items = self.select_items(c, exprs, env, "Aggregate")?;
        if items.iter().any(|b| b.agg.is_some()) {
            return self.build_aggregate(p.origin, c, Vec::new(), &items, true, None);
        }
        if items.iter().any(|b| b.win.is_some()) {
            return self.window_project(p.origin, c, &items, &[]);
        }
        Ok(self.lateral_project(p.origin, c, &items, &[]))
    }

    /// A select list: stars expanded, every item named. A name the input
    /// lacks may be an alias defined earlier in the list (a lateral column
    /// alias), when `spark.sql.lateralColumnAlias.enableImplicitResolution`.
    pub(super) fn select_items(
        &mut self,
        c: &Rel,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
        op: &'static str,
    ) -> Result<Vec<Bound>, AnalysisError> {
        self.select_items_in(c, exprs, env, op, None)
    }

    /// [`Self::select_items`] of an aggregate over grouping sets, which
    /// may call `grouping` and `grouping_id` and name `grouping__id`
    /// (`extra`, searched after the input).
    pub(super) fn select_items_in(
        &mut self,
        c: &Rel,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
        op: &'static str,
        extra: Option<&[Attr]>,
    ) -> Result<Vec<Bound>, AnalysisError> {
        let rels = [c];
        let mut items = Vec::new();
        let mut lateral: Vec<Attr> = Vec::new();
        let lca = self.ctx.conf.lateral_column_alias;
        for e in exprs {
            if let ExprNode::Star(s) = &e.node {
                let cx = Self::cx(&c.out, &rels, env, op);
                for a in self.expand(s, &cx, e.origin)? {
                    items.push(column_bound(&a, e.origin));
                }
                continue;
            }
            let b = {
                let mut cx = Self::cx(&c.out, &rels, env, op);
                cx.lateral = lca.then_some(&lateral[..]);
                cx.extra = extra;
                cx.grouping = extra.is_some();
                cx.windows = WindowUse::Allowed;
                self.bind(e, &cx)?
            };
            let b = self.named(b)?;
            if let ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) = &b.expr.node {
                lateral.push(Attr {
                    col: output.clone(),
                    qual: no_qual(),
                    source: None,
                });
            }
            items.push(b);
        }
        Ok(items)
    }

    /// `b` as a named expression: an attribute, or aliased by Spark's
    /// generated name (a cast of a column keeps the column's name, a
    /// lateral alias otherwise reads `lateralAliasReference(name)`).
    fn named(&mut self, b: Bound) -> Result<Bound, AnalysisError> {
        if let ExprNode::Resolved(ResolvedExpr::Column(_) | ResolvedExpr::Alias { .. }) =
            &b.expr.node
        {
            return Ok(b);
        }
        let shown = b.shown();
        let zone = self.ctx.conf.session_zone;
        let name = match &shown.node {
            ExprNode::Cast { expr, .. }
                if matches!(expr.node, ExprNode::Resolved(ResolvedExpr::Column(_))) =>
            {
                pretty(expr, zone)
            }
            _ => pretty(&lateral_refs(&shown, &b.lateral), zone),
        };
        let output = self.attr(&name, &b.ty);
        let origin = b.expr.origin;
        Ok(Bound {
            expr: Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                    expr: b.expr,
                    output,
                }))
                .with_origin(origin),
            ),
            ..b
        })
    }

    /// Select `items` over `base`, then the attributes `pass`. Items using
    /// lateral column aliases are planned as Spark plans them: an alias
    /// another item reads is computed by a projection below the items that
    /// read it (one projection per level of the chain), unless `base`
    /// outputs it already.
    pub(super) fn lateral_project(
        &self,
        origin: Origin,
        base: &Rel,
        items: &[Bound],
        pass: &[Attr],
    ) -> Rel {
        if pass.is_empty() && items.iter().all(|b| b.lateral.is_empty()) {
            let out = items.iter().map(|b| out_attr(b, &base.out)).collect();
            let exprs = items.iter().map(|b| Arc::clone(&b.expr)).collect();
            return self.projection(origin, base, exprs, out);
        }
        let present: HashSet<AttrId> = base.out.iter().map(|a| a.col.id).collect();
        let outs: Vec<Attr> = items.iter().map(|b| out_attr(b, &base.out)).collect();
        let read: HashSet<AttrId> = items.iter().flat_map(|b| b.lateral.clone()).collect();
        // The level an item is computed at: 0 for what `base` outputs, else
        // one above the highest alias it reads.
        let mut level = vec![0_usize; items.len()];
        for (i, b) in items.iter().enumerate() {
            if present.contains(&outs[i].col.id) {
                continue;
            }
            let reads = b.lateral.iter().filter_map(|id| {
                outs[..i]
                    .iter()
                    .rposition(|a| a.col.id == *id)
                    .map(|j| level[j])
            });
            level[i] = 1 + reads.max().unwrap_or(0);
        }
        let below = |i: usize| level[i] > 0 && read.contains(&outs[i].col.id);
        let top = (0..items.len())
            .filter(|&i| below(i))
            .map(|i| level[i])
            .max()
            .unwrap_or(0);
        let mut cur = base.clone();
        for l in 1..=top {
            let mut exprs: Vec<Arc<Expr>> = cur
                .out
                .iter()
                .map(|a| col_expr(&a.col, Origin::UNKNOWN))
                .collect();
            let mut out = cur.out.to_vec();
            for i in (0..items.len()).filter(|&i| level[i] == l && below(i)) {
                exprs.push(Arc::clone(&items[i].expr));
                out.push(outs[i].clone());
            }
            cur = self.projection(Origin::UNKNOWN, &cur, exprs, out);
        }
        let mut exprs = Vec::new();
        for (i, b) in items.iter().enumerate() {
            exprs.push(if below(i) {
                col_expr(&outs[i].col, b.expr.origin)
            } else {
                Arc::clone(&b.expr)
            });
        }
        exprs.extend(pass.iter().map(|a| col_expr(&a.col, Origin::UNKNOWN)));
        let mut out = outs;
        out.extend(pass.iter().cloned());
        self.projection(origin, &cur, exprs, out)
    }

    /// A projection of `exprs` (with output `out`) over `input`.
    pub(super) fn projection(
        &self,
        origin: Origin,
        input: &Rel,
        exprs: Vec<Arc<Expr>>,
        out: Vec<Attr>,
    ) -> Rel {
        let node = PlanNode::Project {
            input: Arc::clone(&input.plan),
            exprs: exprs.clone(),
        };
        let mut r = self.make(origin, node, out, &[input]);
        r.widen = Some(Arc::new(Widen::Project {
            input: input.clone(),
            exprs,
        }));
        r
    }

    pub(super) fn filter(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        predicate: &Arc<Expr>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        // A `QUALIFY` filter: lowering wraps its predicate (see
        // `lower.rs`), so the analyzer tells it apart from `WHERE` and
        // `HAVING`. Only SQL lowers one: a DataFrame filter has a plan id
        // and no span, and never sits directly over its query's own
        // projection the way a `QUALIFY` does.
        if p.origin.sql.is_some()
            && let Some(q) = qualify_expression(predicate)
            && let Some(r) = self.qualify(p, input, &c, q, env)?
        {
            return Ok(r);
        }
        // A `HAVING` over an aggregate with windows filters below them
        // (Spark's `UnresolvedHaving`: it cannot name a window's alias). Only
        // SQL has a `HAVING`: lowering stamps the filter with the span of
        // its text, as every node it builds, and a DataFrame filter has a
        // plan id and no span. A DataFrame filter, and any further filter,
        // filters the output of the windows (checked against Spark 4.2.0:
        // `df.filter(col("r") > 1)` over `... rank() OVER (...) AS r` works,
        // `HAVING r > 1` is UNRESOLVED_COLUMN).
        if let Some(info) = c.agg.clone()
            && info.has_windows()
            && info.having.is_none()
            && p.origin.sql.is_some()
        {
            return self.having_below_windows(p, &info, predicate, env);
        }
        if c.agg.as_ref().is_some_and(|i| !i.has_windows()) {
            let (bounds, agg, extended) =
                self.over_agg(&c, std::slice::from_ref(predicate), env, "Filter")?;
            boolean_filter(&bounds[0], p.origin)?;
            let node = PlanNode::Filter {
                input: Arc::clone(&agg.plan),
                predicate: Arc::clone(&bounds[0].expr),
            };
            let mut f = self.make(p.origin, node, agg.out.to_vec(), &[&agg]);
            f.widen = Some(Arc::new(Widen::Through { child: agg }));
            return Ok(if extended {
                self.restore(&f, &c.out)
            } else {
                f
            });
        }
        let (mut bounds, wide) =
            self.bind_missing(&c, std::slice::from_ref(predicate), env, "Filter", false)?;
        let b = bounds.remove(0);
        if b.agg.is_some() {
            return Err(AnalysisError::new(
                InvalidWhereCondition {
                    condition: format!("\"{}\"", pretty(&b.shown(), self.ctx.conf.session_zone)),
                    expression_list: aggregate_sql(&b.expr, &wide.out),
                },
                p.origin,
            ));
        }
        boolean_filter(&b, p.origin)?;
        let node = PlanNode::Filter {
            input: Arc::clone(&wide.plan),
            predicate: b.expr,
        };
        let mut f = self.make(p.origin, node, wide.out.to_vec(), &[&wide]);
        let extended = wide.out.len() > c.out.len();
        f.widen = Some(Arc::new(Widen::Through { child: wide }));
        Ok(if extended {
            self.restore(&f, &c.out)
        } else {
            f
        })
    }

    /// A projection of `out` over `r`: drops what a rewrite added.
    fn restore(&self, r: &Rel, out: &[Attr]) -> Rel {
        let exprs = out
            .iter()
            .map(|a| col_expr(&a.col, Origin::UNKNOWN))
            .collect();
        self.projection(Origin::UNKNOWN, r, exprs, out.to_vec())
    }

    /// A `QUALIFY` filter over the resolved select list `c`: `q` is the
    /// wrapped predicate, `input` its unresolved `Filter` node. The
    /// predicate reads the select output (aliases included) and the query
    /// input's other columns, as Spark's does; window functions in either
    /// go through [`Resolver::windowed`], the same extraction a `HAVING`
    /// with window functions uses. `None` when the shape is not a
    /// `QUALIFY`'s own (a wrapped predicate over anything but its query's
    /// projection or aggregate), so the caller falls back to the
    /// `WHERE`/`HAVING` paths.
    fn qualify(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        c: &Rel,
        q: &Arc<Expr>,
        env: &Arc<Env>,
    ) -> Result<Option<Rel>, AnalysisError> {
        match &input.node {
            PlanNode::Project { input: base, exprs } => {
                let base = self.child(base, env)?;
                let items = self.select_items(&base, exprs, env, "Project")?;
                if items.iter().any(|b| b.agg.is_some()) {
                    // A global aggregate (a `Project` lowering whose items
                    // aggregate): qualify over the aggregate, as below.
                    let agg =
                        self.build_aggregate(p.origin, &base, Vec::new(), &items, true, None)?;
                    return Ok(Some(self.qualify_agg(p, &agg, q, None, &agg.out, env)?));
                }
                Ok(Some(self.qualify_project(p, &base, &items, q, env)?))
            }
            PlanNode::Aggregate { .. } => {
                // `GROUP BY` without `HAVING`: `c` is the aggregate.
                if c.agg.is_none() {
                    return Ok(None);
                }
                let out = c.out.to_vec();
                Ok(Some(self.qualify_agg(p, c, q, None, &out, env)?))
            }
            PlanNode::Filter {
                input: mid,
                predicate,
            } => {
                // `HAVING` below `QUALIFY`: qualify over the aggregate the
                // inner filter resolved.
                if qualify_expression(predicate).is_some() {
                    return Ok(None);
                }
                let inner = self.child(mid, env)?;
                let Some(info) = inner.agg.clone() else {
                    return Ok(None);
                };
                let having = info.having.clone().or_else(|| match &c.plan.node {
                    PlanNode::Filter { predicate: h, .. } => Some(Arc::clone(h)),
                    _ => None,
                });
                let Some(having) = having else {
                    return Ok(None);
                };
                let out = c.out.to_vec();
                Ok(Some(self.qualify_agg(
                    p,
                    &inner,
                    q,
                    Some(&having),
                    &out,
                    env,
                )?))
            }
            _ => Ok(None),
        }
    }

    /// `QUALIFY` over a projection: the predicate reads the select output
    /// and the input's other columns. `c` is the resolved projection (its
    /// output is the select list); the filter is rebuilt below so the
    /// predicate's windows extract beside the select list's.
    fn qualify_project(
        &mut self,
        p: &LogicalPlan,
        base: &Rel,
        items: &[Bound],
        q: &Arc<Expr>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        Self::check_qualify_windows(q, items.iter().any(|b| b.win.is_some()), self)?;
        // The select list's windows over the input, as
        // [`Resolver::window_project`] computes them.
        let exprs: Vec<Arc<Expr>> = items.iter().map(|b| Arc::clone(&b.expr)).collect();
        let (below, exprs) = self.windowed(base, &exprs)?;
        let items: Vec<Bound> = items
            .iter()
            .zip(exprs)
            .map(|(b, expr)| Bound { expr, ..b.clone() })
            .collect();
        // The input's columns the select list drops pass through (the
        // predicate and its windows may name them); names the select list
        // outputs shadow the input's, as Spark resolves the output first.
        let cs = self.ctx.conf.case_sensitive;
        let outs: Vec<Attr> = items.iter().map(|b| out_attr(b, &base.out)).collect();
        let pass: Vec<Attr> = base
            .out
            .iter()
            .filter(|a| !outs.iter().any(|o| same(&o.col.name, &a.col.name, cs)))
            .cloned()
            .collect();
        let wide = self.lateral_project(Origin::UNKNOWN, &below, &items, &pass);
        let rels = [&wide];
        let mut cx = Self::cx(&wide.out, &rels, env, "Filter");
        cx.windows = WindowUse::Allowed;
        let qb = self.bind(q, &cx)?;
        boolean_filter(&qb, p.origin)?;
        let (wrel, exprs) = self.windowed(&wide, std::slice::from_ref(&qb.expr))?;
        let pred = exprs.into_iter().next().ok_or_else(|| {
            AnalysisError::new(krishiv_common::error::FailedToParseTooComplex {}, p.origin)
        })?;
        let node = PlanNode::Filter {
            input: Arc::clone(&wrel.plan),
            predicate: pred,
        };
        let mut f = self.make(p.origin, node, wrel.out.to_vec(), &[&wrel]);
        f.widen = Some(Arc::new(Widen::Through { child: wrel }));
        Ok(self.restore(&f, &wide.out[..outs.len()]))
    }

    /// `QUALIFY` over an aggregate `agg` (its output is the select list):
    /// the predicate resolves as a `HAVING` does (names read the output,
    /// then grouping attributes, then the input; missing aggregates and
    /// keys grow it), except windows are allowed and extracted above it.
    /// `having` re-applies a `HAVING` below the windows when one filtered
    /// the aggregate; `select_out` is the select list's output.
    fn qualify_agg(
        &mut self,
        p: &LogicalPlan,
        agg: &Rel,
        q: &Arc<Expr>,
        having: Option<&Arc<Expr>>,
        select_out: &[Attr],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let windows = agg.agg.as_ref().is_some_and(|i| i.has_windows());
        Self::check_qualify_windows(q, windows, self)?;
        let (bounds, grown, _) = self.over_agg(agg, std::slice::from_ref(q), env, "Qualify")?;
        let b = bounds.into_iter().next().ok_or_else(|| {
            AnalysisError::new(krishiv_common::error::FailedToParseTooComplex {}, p.origin)
        })?;
        boolean_filter(&b, p.origin)?;
        // With a `HAVING`, it filters the grown aggregate below the
        // predicate's windows (as [`Resolver::having_below_windows`] does
        // for the select list's).
        let base = match having {
            None => grown,
            Some(h) => {
                let info = grown.agg.as_ref().ok_or_else(|| {
                    AnalysisError::new(krishiv_common::error::FailedToParseTooComplex {}, p.origin)
                })?;
                let info = AggInfo {
                    having: Some(Arc::clone(h)),
                    ..(**info).clone()
                };
                self.aggregate_rel(Origin::UNKNOWN, info, &[])?
            }
        };
        let (wrel, exprs) = self.windowed(&base, std::slice::from_ref(&b.expr))?;
        let pred = exprs.into_iter().next().ok_or_else(|| {
            AnalysisError::new(krishiv_common::error::FailedToParseTooComplex {}, p.origin)
        })?;
        let node = PlanNode::Filter {
            input: Arc::clone(&wrel.plan),
            predicate: pred,
        };
        let mut f = self.make(p.origin, node, wrel.out.to_vec(), &[&wrel]);
        f.widen = Some(Arc::new(Widen::Through { child: wrel }));
        Ok(self.restore(&f, select_out))
    }

    /// Spark's `QUALIFY` checks on the predicate `q`: aggregates are not
    /// allowed in it, and at least one window function must appear in it
    /// or in the select list (`select_windows`).
    fn check_qualify_windows(
        q: &Arc<Expr>,
        select_windows: bool,
        r: &Resolver<'_>,
    ) -> Result<(), AnalysisError> {
        if let Some(a) = first_qualify_aggregate(q, r) {
            let zone = r.ctx.conf.session_zone;
            return Err(AnalysisError::new(
                QualifyAggregateNotAllowed {
                    aggregate_expr: format!("\"{}\"", pretty(&a, zone)),
                },
                a.origin,
            ));
        }
        if !select_windows && !has_qualify_window(q) {
            return Err(AnalysisError::new(
                QualifyRequiresWindowFunction {},
                q.origin,
            ));
        }
        Ok(())
    }

    /// Binds the top-level expressions of a filter or sort over `c`. Names
    /// `c` does not output, but a relation below it dropped (through
    /// projections, filters, sorts and limits, down to an aggregate's
    /// grouping attributes), are Spark's missing references: they are
    /// added back up to `c` and resolve. Enclosing queries come last, as
    /// in Spark: a dropped column hides an outer one of its name. Returns
    /// the bound expressions and the relation they read, wider than `c`
    /// when something was added. `lenient`: a name `c` outputs several
    /// times is looked up below.
    pub(super) fn bind_missing(
        &mut self,
        c: &Rel,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
        op: &'static str,
        lenient: bool,
    ) -> Result<(Vec<Bound>, Rel), AnalysisError> {
        let rels = [c];
        let mut cx = Self::cx(&c.out, &rels, env, op);
        cx.lenient = lenient;
        if op == "Filter" {
            cx.windows = WindowUse::Where;
        }
        // Without the enclosing queries when there is something below.
        let below = Self::below(c);
        let lookup = Self::below_of(c, true);
        cx.outer = below.is_empty();
        let unresolved = |e: &AnalysisError| {
            e.error
                .error_class()
                .name()
                .starts_with("UNRESOLVED_COLUMN")
        };
        match self.bind_all(exprs, &cx) {
            Ok(bounds) => return Ok((bounds, c.clone())),
            Err(e) if !unresolved(&e) || lookup.is_empty() => return Err(e),
            Err(_) => {}
        }
        cx.extra = Some(&lookup);
        let bounds = match self.bind_all(exprs, &cx) {
            Err(e) if unresolved(&e) => {
                cx.outer = true;
                self.bind_all(exprs, &cx)?
            }
            other => other?,
        };
        let have: HashSet<AttrId> = c.out.iter().map(|a| a.col.id).collect();
        let mut missing: Vec<Attr> = Vec::new();
        let mut seen = HashSet::new();
        for b in &bounds {
            let mut stack = vec![Arc::clone(&b.expr)];
            while let Some(x) = stack.pop() {
                if !seen.insert(super::bind::addr(&*x)) {
                    continue;
                }
                if let ExprNode::Resolved(ResolvedExpr::Column(col)) = &x.node
                    && !have.contains(&col.id)
                    && !missing.iter().any(|m| m.col.id == col.id)
                    && let Some(a) = below.iter().find(|a| a.col.id == col.id)
                {
                    missing.push(a.clone());
                }
                x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
            }
        }
        // The retry resolved a name only `below` holds: `missing` is not
        // empty.
        Ok((bounds, self.widened(c, &missing)?))
    }

    /// The aggregate under `c`'s filters, sorts and limits (`HAVING`), and
    /// those layers, top first; `None` when `c` is not such a stack. A
    /// `HAVING` that had to grow the aggregate left a projection over its
    /// filter (`restore`), and that is not such a stack: Spark does not
    /// resolve a sort through it (`GROUP BY x + 1 HAVING sum(l) > 1 ORDER BY
    /// x + 1` is UNRESOLVED_COLUMN, checked against 4.2.0).
    fn through_to_agg(c: &Rel) -> Option<(Vec<Rel>, Rel)> {
        let mut layers = Vec::new();
        let mut cur = c.clone();
        loop {
            if cur.agg.is_some() {
                return Some((layers, cur));
            }
            let child = match cur.widen.as_deref() {
                Some(Widen::Through { child }) => child.clone(),
                _ => return None,
            };
            layers.push(cur);
            cur = child;
        }
    }

    /// `layers` (from [`Self::through_to_agg`]) rebuilt over `inner`, a
    /// grown version of the aggregate they were over.
    fn rewrapped(&self, layers: &[Rel], inner: Rel) -> Rel {
        let mut built = inner;
        for rel in layers.iter().rev() {
            let child = built;
            let plan = Arc::clone(&rel.plan)
                .map_children(|_| Ok::<_, Infallible>(Arc::clone(&child.plan)))
                .unwrap_or_else(|never| match never {});
            let mut wide = self.wrap(plan, rel.plan.origin.plan_id, child.out.to_vec(), &[&child]);
            wide.widen = Some(Arc::new(Widen::Through { child }));
            built = wide;
        }
        built
    }

    /// Binds each of `exprs` in `cx`.
    fn bind_all(&mut self, exprs: &[Arc<Expr>], cx: &Cx<'_>) -> Result<Vec<Bound>, AnalysisError> {
        exprs.iter().map(|e| self.bind(e, cx)).collect()
    }

    /// The attributes relations below `r` dropped and could add back
    /// (see [`Self::bind_missing`]), nearest first.
    fn below(r: &Rel) -> Vec<Attr> {
        Self::below_of(r, false)
    }

    /// [`Self::below`]; with `keep_output`, attributes `r` already outputs
    /// are kept too (Spark looks a name the select list outputs twice up
    /// among them: `SELECT id, CAST(id AS STRING) AS id ... ORDER BY id`
    /// sorts by the first).
    fn below_of(r: &Rel, keep_output: bool) -> Vec<Attr> {
        let mut out: Vec<Attr> = Vec::new();
        let mut seen: HashSet<AttrId> = if keep_output {
            HashSet::new()
        } else {
            r.out.iter().map(|a| a.col.id).collect()
        };
        let mut add = |a: &Attr, out: &mut Vec<Attr>| {
            if seen.insert(a.col.id) {
                out.push(a.clone());
            }
        };
        let mut cur = r.clone();
        while let Some(w) = cur.widen.clone() {
            cur = match &*w {
                Widen::Aggregate(info) => {
                    info.keys.iter().for_each(|k| add(k, &mut out));
                    break;
                }
                Widen::Through { child } => child.clone(),
                Widen::Project { input, .. } => {
                    input.out.iter().for_each(|a| add(a, &mut out));
                    input.clone()
                }
            };
        }
        out
    }

    /// `r` outputting `missing` too (attributes [`Self::below`] found),
    /// each relation on the way down adding what the ones above need.
    fn widened(&mut self, r: &Rel, missing: &[Attr]) -> Result<Rel, AnalysisError> {
        // Down: each relation, how it widens and what it must add.
        let mut path: Vec<(Rel, Arc<Widen>, Vec<Attr>)> = Vec::new();
        let mut cur = r.clone();
        let mut need = missing.to_vec();
        while let Some(w) = cur.widen.clone() {
            let here = need.clone();
            let next = match &*w {
                Widen::Aggregate(_) => None,
                Widen::Through { child } => Some(child.clone()),
                Widen::Project { input, .. } => {
                    need.retain(|a| !input.out.iter().any(|o| o.col.id == a.col.id));
                    (!need.is_empty()).then(|| input.clone())
                }
            };
            path.push((cur, w, here));
            match next {
                Some(next) => cur = next,
                None => break,
            }
        }
        // Up: rebuild each over the wider relation below it.
        let mut built: Option<Rel> = None;
        while let Some((rel, w, need)) = path.pop() {
            let mut out = rel.out.to_vec();
            out.extend(need.iter().cloned());
            built = Some(match &*w {
                Widen::Aggregate(info) => {
                    let mut grown = (**info).clone();
                    for a in &need {
                        if !grown.out.iter().any(|o| o.col.id == a.col.id) {
                            grown.aggregates.push(col_expr(&a.col, Origin::UNKNOWN));
                            grown.out.push(a.clone());
                        }
                    }
                    self.aggregate_rel(rel.plan.origin, grown, &need)?
                }
                Widen::Project { input, exprs } => {
                    let input = built.take().unwrap_or_else(|| input.clone());
                    let mut exprs = exprs.clone();
                    exprs.extend(need.iter().map(|a| col_expr(&a.col, Origin::UNKNOWN)));
                    self.projection(rel.plan.origin, &input, exprs, out)
                }
                Widen::Through { child } => {
                    let child = built.take().unwrap_or_else(|| child.clone());
                    let plan = Arc::clone(&rel.plan)
                        .map_children(|_| Ok::<_, Infallible>(Arc::clone(&child.plan)))
                        .unwrap_or_else(|never| match never {});
                    let mut wide = self.wrap(plan, rel.plan.origin.plan_id, out, &[&child]);
                    wide.widen = Some(Arc::new(Widen::Through { child }));
                    wide
                }
            });
        }
        Ok(built.unwrap_or_else(|| r.clone()))
    }

    /// Expressions over an aggregate (`HAVING`, `ORDER BY`): names resolve
    /// against its output, then its grouping attributes; aggregate-function
    /// arguments against its input. Aggregate calls the aggregate does not
    /// compute yet, and grouping attributes it does not output, are added
    /// to it (and passed up through the projections of lateral aliases
    /// above it); the flag says whether its output grew.
    pub(super) fn over_agg(
        &mut self,
        agg: &Rel,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
        op: &'static str,
    ) -> Result<(Vec<Bound>, Rel, bool), AnalysisError> {
        let Some(info) = agg.agg.clone() else {
            return Err(unsupported("a plan shape", agg.plan.origin));
        };
        let rels = [agg];
        let mut cx = Self::cx(&agg.out, &rels, env, op);
        cx.extra = Some(&info.keys);
        cx.agg_scope = Some(&info.input.out);
        cx.grouping = info.analytics.is_some();
        if op == "Filter" {
            cx.windows = WindowUse::Having;
        } else if op == "Qualify" {
            // `QUALIFY` resolves like `HAVING`, but windows are allowed
            // and extracted above the aggregate.
            cx.windows = WindowUse::Allowed;
        }
        let mut memo = HashMap::new();
        let mut known: Vec<(u128, Attr)> = info
            .aggregates
            .iter()
            .zip(info.out.iter())
            .map(|(e, a)| (semantic_key(e, &mut memo), a.clone()))
            .collect();
        let visible: HashSet<AttrId> = agg.out.iter().map(|a| a.col.id).collect();
        let mut grown = (*info).clone();
        // What the aggregate adds to the output it shows.
        let mut pass: Vec<Attr> = Vec::new();
        let mut bounds = Vec::new();
        for e in exprs {
            match self.bind(e, &cx) {
                Ok(b) => bounds.push(b),
                // A grouping expression written out in full (`GROUP BY x + 1
                // ORDER BY x + 1`) is the aggregate's output, wherever it
                // appears in the expression; names it does not cover stay
                // unresolved, as in Spark.
                Err(err)
                    if err
                        .error
                        .error_class()
                        .name()
                        .starts_with("UNRESOLVED_COLUMN") =>
                {
                    let mut st = Grouped {
                        info: &info,
                        env,
                        op,
                        visible: &visible,
                        memo: &mut memo,
                        known: &mut known,
                        grown: &mut grown,
                        pass: &mut pass,
                    };
                    let Some(sub) = self.group_substitute(e, &mut st) else {
                        return Err(err);
                    };
                    bounds.push(self.bind(&sub, &cx).map_err(|_| err)?);
                }
                Err(err) => return Err(err),
            }
        }
        if let Some(an) = &info.analytics {
            for b in &mut bounds {
                an.check_late_call(&b.expr)?;
                b.expr = self.resolve_grouping(&b.expr, an, env)?;
            }
        }
        for b in &mut bounds {
            b.expr = Arc::clone(&b.expr).transform_down(&mut |x: Arc<Expr>| match &x.node {
                ExprNode::Resolved(ResolvedExpr::Function(f))
                    if f.kind == FunctionKind::Aggregate =>
                {
                    let k = semantic_key(&x, &mut memo);
                    let a = match known.iter().find(|(kk, _)| *kk == k) {
                        Some((_, a)) => a.clone(),
                        None => {
                            let ty = ArgType {
                                data_type: Some(f.data_type.clone()),
                                nullable: f.nullable,
                            };
                            let output = self.attr(&pretty(&x, self.ctx.conf.session_zone), &ty);
                            grown.aggregates.push(Arc::new(Expr::new(ExprNode::Resolved(
                                ResolvedExpr::Alias {
                                    expr: Arc::clone(&x),
                                    output: output.clone(),
                                },
                            ))));
                            let a = Attr {
                                col: output,
                                qual: no_qual(),
                                source: None,
                            };
                            grown.out.push(a.clone());
                            known.push((k, a.clone()));
                            a
                        }
                    };
                    if !visible.contains(&a.col.id) && !pass.iter().any(|p| p.col.id == a.col.id) {
                        pass.push(a.clone());
                    }
                    Ok(col_expr(&a.col, x.origin))
                }
                ExprNode::Resolved(ResolvedExpr::Column(c))
                    if !visible.contains(&c.id) && !pass.iter().any(|a| a.col.id == c.id) =>
                {
                    if let Some(k) = info.keys.iter().find(|k| k.col.id == c.id) {
                        // An output that already computes the key (`x AS
                        // m`) stands for it, as in Spark: nothing grows.
                        let key = semantic_key(&x, &mut memo);
                        if let Some((_, a)) = known.iter().find(|(kk, _)| *kk == key) {
                            return Ok(col_expr(&a.col, x.origin));
                        }
                        if !grown.out.iter().any(|o| o.col.id == c.id) {
                            grown.aggregates.push(col_expr(&k.col, Origin::UNKNOWN));
                            grown.out.push(k.clone());
                        }
                        pass.push(k.clone());
                    }
                    Ok(x)
                }
                _ => Ok(x),
            })?;
        }
        // Grouping attributes a subquery reads from here.
        for b in &bounds {
            let _ = crate::walk::visit_distinct(crate::walk::Node::Expr(&b.expr), |n| {
                if let crate::walk::Node::Expr(e) = n
                    && let ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) = &e.node
                    && !visible.contains(&c.id)
                    && !pass.iter().any(|a| a.col.id == c.id)
                    && let Some(k) = info.keys.iter().find(|k| k.col.id == c.id)
                {
                    if !grown.out.iter().any(|o| o.col.id == c.id) {
                        grown.aggregates.push(col_expr(&k.col, Origin::UNKNOWN));
                        grown.out.push(k.clone());
                    }
                    pass.push(k.clone());
                }
                std::ops::ControlFlow::Continue(())
            });
        }
        if pass.is_empty() {
            return Ok((bounds, agg.clone(), false));
        }
        let r = self.aggregate_rel(agg.plan.origin, grown, &pass)?;
        Ok((bounds, r, true))
    }

    /// `e` with every subexpression that is one of the aggregate's grouping
    /// expressions or output expressions (equal after canonicalisation, as Spark's
    /// `semanticEquals`) replaced by the attribute the aggregate computes
    /// it as, added to the aggregate when it does not output it. `None`
    /// when nothing matches, or `e` holds a subquery.
    #[inline(never)]
    fn group_substitute(&mut self, e: &Arc<Expr>, st: &mut Grouped<'_>) -> Option<Arc<Expr>> {
        let info = st.info;
        let rels = [&info.input];
        let probe = Self::cx(&info.input.out, &rels, st.env, st.op);
        let grouping: HashSet<u128> = info
            .grouping
            .iter()
            .map(|g| semantic_key(g, st.memo))
            .collect();
        let mut subquery = false;
        let _ = crate::walk::visit_distinct(crate::walk::Node::Expr(e), |n| {
            if let crate::walk::Node::Expr(x) = n
                && matches!(x.node, ExprNode::Subquery { .. })
            {
                subquery = true;
            }
            std::ops::ControlFlow::Continue(())
        });
        if subquery {
            return None;
        }
        let mut changed = false;
        let out = Arc::clone(e)
            .transform_down(&mut |x: Arc<Expr>| -> Result<Arc<Expr>, Infallible> {
                // Leaves resolve by name; aggregate calls and the rest of
                // the output are bound by the caller.
                if matches!(
                    x.node,
                    ExprNode::Column(_) | ExprNode::Literal(_) | ExprNode::Star(_)
                ) {
                    return Ok(x);
                }
                let Ok(b) = self.expr(&x, &probe) else {
                    return Ok(x);
                };
                if b.agg.is_some() || b.outer.is_some() || !b.lateral.is_empty() {
                    return Ok(x);
                }
                // The probe's expressions are dropped on return: their
                // addresses must not outlive them in a memo.
                let key = semantic_key(&b.expr, &mut HashMap::new());
                if !grouping.contains(&key) && !st.known.iter().any(|(k, _)| *k == key) {
                    return Ok(x);
                }
                let a = match st.known.iter().find(|(k, _)| *k == key) {
                    Some((_, a)) => a.clone(),
                    None => {
                        let output = self.attr(&pretty(&b.expr, self.ctx.conf.session_zone), &b.ty);
                        st.grown
                            .aggregates
                            .push(Arc::new(Expr::new(ExprNode::Resolved(
                                ResolvedExpr::Alias {
                                    expr: Arc::clone(&b.expr),
                                    output: output.clone(),
                                },
                            ))));
                        let a = Attr {
                            col: output,
                            qual: no_qual(),
                            source: None,
                        };
                        st.grown.out.push(a.clone());
                        st.known.push((key, a.clone()));
                        a
                    }
                };
                if !st.visible.contains(&a.col.id) && !st.pass.iter().any(|p| p.col.id == a.col.id)
                {
                    st.pass.push(a.clone());
                }
                changed = true;
                Ok(col_expr(&a.col, x.origin))
            })
            .unwrap_or_else(|never| match never {});
        changed.then_some(out)
    }

    /// The aggregate of `info`, under its `HAVING` (with windows), its
    /// windows and the projections of its lateral aliases if it has any;
    /// `pass` (attributes of the aggregate) are output after the select
    /// items.
    pub(super) fn aggregate_rel(
        &mut self,
        origin: Origin,
        info: AggInfo,
        pass: &[Attr],
    ) -> Result<Rel, AnalysisError> {
        // Over grouping sets, the aggregate reads their expansion; the
        // attributes it defines are part of the aggregate's id span.
        let (input, defined) = match info.analytics.as_deref().and_then(|a| a.expand.as_ref()) {
            Some(e) => (Arc::clone(&e.plan), e.ids),
            None => (Arc::clone(&info.input.plan), None),
        };
        let node = PlanNode::Aggregate {
            input,
            grouping: Grouping::GroupBy(info.grouping.clone()),
            aggregates: info.aggregates.clone(),
        };
        let mut r = match &info.top {
            None => self.make(origin, node, info.out.clone(), &[&info.input]),
            Some(top) => {
                let mut a = self.make(Origin::UNKNOWN, node, info.out.clone(), &[&info.input]);
                if let Some(h) = &info.having {
                    let f = PlanNode::Filter {
                        input: Arc::clone(&a.plan),
                        predicate: Arc::clone(h),
                    };
                    a = self.make(Origin::UNKNOWN, f, a.out.to_vec(), &[&a]);
                }
                if info.has_windows() {
                    self.window_project(origin, &a, top, pass)?
                } else {
                    self.lateral_project(origin, &a, top, pass)
                }
            }
        };
        r.ids = super::resolve::hull(r.ids, defined);
        let info = Arc::new(info);
        // A column the select list dropped comes back from the aggregate
        // (rebuilt with its windows and projections above it).
        r.widen = Some(Arc::new(Widen::Aggregate(Arc::clone(&info))));
        r.agg = Some(info);
        Ok(r)
    }

    pub(super) fn aggregate(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        grouping: &Grouping,
        aggregates: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let Grouping::GroupBy(keys) = grouping else {
            return self.grouping_sets(p, input, grouping, aggregates, env);
        };
        let c = self.child(input, env)?;
        self.check_star_ordinals(keys, aggregates)?;
        // Spark resolves the select list first, then ordinals, then the
        // other keys.
        let items = self.select_items(&c, aggregates, env, "Aggregate")?;
        // `GROUP BY ALL`: a lone unqualified `all` in SQL groups by the select
        // list, unless a column or a select-list alias named `all` takes it
        // (Spark resolves names first). A DataFrame `groupBy("all")` has no
        // SQL span and never expands (`UNRESOLVED_COLUMN`, as in Spark).
        let group = if p.origin.sql.is_some() && is_bare_all(keys) && !self.names_all(&c, &items) {
            self.group_by_all(&items, keys[0].origin)?
        } else {
            self.group_keys(&c, keys, &items, env)?
        };
        // No grouping (`HAVING` without `GROUP BY`): Spark's
        // `MISSING_GROUP_BY` for an ungrouped column.
        let global = group.is_empty();
        self.build_aggregate(p.origin, &c, group, &items, global, None)
    }

    /// Whether the input has a column, or (with `group_by_aliases`) the
    /// select list an output, named `all`: then `GROUP BY ALL` names it.
    fn names_all(&self, c: &Rel, items: &[Bound]) -> bool {
        let cs = self.ctx.conf.case_sensitive;
        c.out.iter().any(|a| same(&a.col.name, "all", cs))
            || (self.ctx.conf.group_by_aliases
                && items
                    .iter()
                    .any(|b| same(&out_attr(b, &c.out).col.name, "all", cs)))
    }

    /// `GROUP BY ALL` over the select list `items`, as Spark's
    /// `ResolveReferencesInAggregate` infers it: every item without an
    /// aggregate function groups (scalar subqueries count as groupable:
    /// Spark's check does not look inside them); when nothing groups, an
    /// item naming a column outside any aggregate cannot be inferred:
    /// `UNRESOLVED_ALL_IN_GROUP_BY`. Select items stay named; grouping
    /// uses the expression computing them. `origin` is the `GROUP BY ALL`
    /// clause's, as lowering stamps it on the key.
    fn group_by_all(
        &mut self,
        items: &[Bound],
        origin: Origin,
    ) -> Result<Vec<Bound>, AnalysisError> {
        let mut group = Vec::new();
        let mut bare = false;
        for b in items {
            let expr = match &b.expr.node {
                ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => Arc::clone(expr),
                _ => Arc::clone(&b.expr),
            };
            if contains_aggregate(&expr) {
                bare = bare || contains_bare(&expr);
            } else {
                group.push(Bound { expr, ..b.clone() });
            }
        }
        if group.is_empty() && bare {
            return Err(AnalysisError::new(UnresolvedAllInGroupBy {}, origin));
        }
        Ok(group)
    }

    /// `SELECT *` with `GROUP BY n` is an error (Spark's
    /// `STAR_GROUP_BY_POS`).
    pub(super) fn check_star_ordinals(
        &self,
        keys: &[Arc<Expr>],
        aggregates: &[Arc<Expr>],
    ) -> Result<(), AnalysisError> {
        let star = aggregates
            .iter()
            .any(|e| matches!(e.node, ExprNode::Star(_)));
        if star && keys.iter().any(|k| self.ordinal(k).is_some()) {
            return Err(AnalysisError::new(StarGroupByPos {}, Origin::UNKNOWN));
        }
        Ok(())
    }

    /// `GROUP BY n` with ordinals on: `n`.
    fn ordinal(&self, k: &Expr) -> Option<i32> {
        match &k.node {
            ExprNode::Literal(Literal::Integer(n)) if self.ctx.conf.group_by_ordinal => Some(*n),
            _ => None,
        }
    }

    /// The grouping expressions `keys` over `c`: an ordinal names a select
    /// item, other names may be select-list aliases.
    pub(super) fn group_keys(
        &mut self,
        c: &Rel,
        keys: &[Arc<Expr>],
        items: &[Bound],
        env: &Arc<Env>,
    ) -> Result<Vec<Bound>, AnalysisError> {
        let mut group: Vec<Option<Bound>> = vec![None; keys.len()];
        for (slot, k) in group.iter_mut().zip(keys) {
            if let Some(n) = self.ordinal(k) {
                *slot = Some(self.group_ordinal(n, k.origin, items, c)?);
            }
        }
        // Spark resolves the select list first, then ordinals, then the
        // other keys. A `GROUP BY` may name any select-list alias, explicit
        // or implicit (`SELECT 1 GROUP BY `1`` groups by the literal): the
        // aliases are the items' output names over the expressions that
        // compute them, searched after the input.
        let aliases: Vec<(String, Arc<Expr>)> = if self.ctx.conf.group_by_aliases {
            items
                .iter()
                .map(|b| {
                    let name = out_attr(b, &c.out).col.name;
                    let expr = match &b.expr.node {
                        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => Arc::clone(expr),
                        _ => Arc::clone(&b.expr),
                    };
                    (name, expr)
                })
                .collect()
        } else {
            Vec::new()
        };
        let rels = [c];
        for (slot, k) in group.iter_mut().zip(keys) {
            if slot.is_none() {
                let mut cx = Self::cx(&c.out, &rels, env, "Aggregate");
                cx.aliases = &aliases;
                *slot = Some(self.bind(k, &cx)?);
            }
        }
        let group: Vec<Bound> = group.into_iter().flatten().collect();
        self.check_grouping(&group, items)?;
        Ok(group)
    }

    /// Spark's `GROUP_BY_AGGREGATE` and
    /// `UNSUPPORTED_FEATURE.LATERAL_COLUMN_ALIAS_IN_GROUP_BY` for named
    /// keys: a grouping expression may not contain an aggregate call,
    /// however it was written (directly, or through a select-list alias
    /// resolving to one, which otherwise hides in the grouping-key set
    /// `check_aggregation` skips), and it may not compute what a lateral
    /// alias reader computes (as `group_ordinal` enforces for ordinals).
    fn check_grouping(&self, group: &[Bound], items: &[Bound]) -> Result<(), AnalysisError> {
        let mut memo = HashMap::new();
        let mut lateral: Vec<u128> = Vec::new();
        for b in items {
            if !b.lateral.is_empty() {
                lateral.push(semantic_key(&b.expr, &mut memo));
            }
        }
        for b in group {
            if has_aggregate(&b.expr) {
                return Err(AnalysisError::new(
                    GroupByAggregate {
                        sql_expr: pretty(&b.expr, self.ctx.conf.session_zone),
                    },
                    b.expr.origin,
                ));
            }
            if lateral.contains(&semantic_key(&b.expr, &mut memo)) {
                return Err(AnalysisError::new(
                    UnsupportedFeatureLateralColumnAliasInGroupBy {},
                    Origin::UNKNOWN,
                ));
            }
        }
        Ok(())
    }

    /// `GROUP BY n`: the `n`-th select item, unaliased.
    fn group_ordinal(
        &self,
        n: i32,
        origin: Origin,
        items: &[Bound],
        c: &Rel,
    ) -> Result<Bound, AnalysisError> {
        let index = usize::try_from(n)
            .ok()
            .filter(|i| (1..=items.len()).contains(i));
        let Some(i) = index else {
            return Err(AnalysisError::new(
                GroupByPosOutOfRange {
                    index: n.to_string(),
                    size: items.len().to_string(),
                },
                origin,
            ));
        };
        let item = &items[i - 1];
        if item.win.is_some() {
            return Err(super::windows::misplaced(WindowUse::Other, item, origin));
        }
        if !item.lateral.is_empty() {
            return Err(AnalysisError::new(
                UnsupportedFeatureLateralColumnAliasInGroupBy {},
                Origin::UNKNOWN,
            ));
        }
        // `named` aliases every item with an aggregate.
        if let (Some(_), ExprNode::Resolved(ResolvedExpr::Alias { expr, output })) =
            (item.agg, &item.expr.node)
        {
            let text = pretty(&qualified(expr, &c.out), self.ctx.conf.session_zone);
            return Err(AnalysisError::new(
                GroupByPosAggregate {
                    index: n.to_string(),
                    agg_expr: format!("{text} AS {}", quote_if_needed(&output.name)),
                },
                origin,
            ));
        }
        let expr = match &item.expr.node {
            ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => Arc::clone(expr),
            _ => Arc::clone(&item.expr),
        };
        Ok(Bound {
            expr,
            ..item.clone()
        })
    }

    /// An aggregate of `items` grouped by `group`, validated. Items that
    /// read lateral column aliases are computed by projections above it,
    /// the aggregate calls and grouping expressions they use computed by
    /// the aggregate, as Spark plans them.
    pub(super) fn build_aggregate(
        &mut self,
        origin: Origin,
        c: &Rel,
        group: Vec<Bound>,
        items: &[Bound],
        global: bool,
        analytics: Option<Arc<Analytics>>,
    ) -> Result<Rel, AnalysisError> {
        let mut memo = HashMap::new();
        let gkeys: Vec<(u128, ArgType)> = group
            .iter()
            .map(|b| (semantic_key(&b.expr, &mut memo), b.ty.clone()))
            .collect();
        let key_set: HashSet<u128> = gkeys.iter().map(|(k, _)| *k).collect();
        // Items with lateral aliases or windows are computed above the
        // aggregate, over its output.
        let base: Vec<Bound> = items
            .iter()
            .filter(|b| b.lateral.is_empty() && b.win.is_none())
            .cloned()
            .collect();
        self.check_aggregation(&base, &key_set, &mut memo, origin, global)?;
        // Over grouping sets, the grouping attributes are the expansion's.
        let (keys, scope): (Vec<Attr>, Vec<Attr>) = match &analytics {
            Some(a) => (
                a.keys(),
                c.out
                    .iter()
                    .chain(a.exprs.iter().map(|g| &g.attr))
                    .cloned()
                    .collect(),
            ),
            None => (
                group
                    .iter()
                    .filter_map(|b| match &b.expr.node {
                        ExprNode::Resolved(ResolvedExpr::Column(col)) => {
                            c.out.iter().find(|a| a.col.id == col.id).cloned()
                        }
                        _ => None,
                    })
                    .collect(),
                c.out.to_vec(),
            ),
        };
        let mut info = AggInfo {
            input: c.clone(),
            grouping: group.into_iter().map(|b| b.expr).collect(),
            aggregates: base.iter().map(|b| Arc::clone(&b.expr)).collect(),
            out: base.iter().map(|b| out_attr(b, &scope)).collect(),
            keys,
            top: None,
            analytics,
            having: None,
        };
        if base.len() == items.len() {
            return self.aggregate_rel(origin, info, &[]);
        }
        for b in items.iter().filter(|b| !b.lateral.is_empty()) {
            in_aggregate_call(b)?;
        }
        let lateral: HashSet<AttrId> = items.iter().filter_map(|b| alias_id(&b.expr)).collect();
        let mut top = Vec::new();
        for b in items {
            if b.lateral.is_empty() && b.win.is_none() {
                let a = out_attr(b, &scope);
                top.push(Bound {
                    expr: col_expr(&a.col, b.expr.origin),
                    ..b.clone()
                });
                continue;
            }
            let expr = self.lift(b, &mut info, &gkeys, &lateral, &mut memo, global, origin)?;
            top.push(Bound { expr, ..b.clone() });
        }
        info.top = Some(top.into());
        self.aggregate_rel(origin, info, &[])
    }

    /// A select item that reads lateral aliases, bound over the
    /// aggregate's input, over its output instead: aggregate calls and
    /// grouping expressions read attributes of the aggregate (added to it
    /// when new); lateral aliases stay. Any other input column is
    /// `MISSING_AGGREGATION` (`MISSING_GROUP_BY` without `GROUP BY`).
    #[expect(
        clippy::too_many_arguments,
        reason = "one call site, all state of the rewrite"
    )]
    fn lift(
        &mut self,
        b: &Bound,
        info: &mut AggInfo,
        keys: &[(u128, ArgType)],
        lateral: &HashSet<AttrId>,
        memo: &mut HashMap<usize, u128>,
        global: bool,
        origin: Origin,
    ) -> Result<Arc<Expr>, AnalysisError> {
        let mut cx = Lift {
            info,
            keys,
            lateral,
            memo,
            global,
            origin,
        };
        self.lift_expr(Arc::clone(&b.expr), &mut cx)
    }

    /// [`Self::lift`] of one node, pre-order. A window's function stays a
    /// window call (its arguments are lifted), so `sum(sum(x)) OVER w`
    /// reads the aggregate's `sum(x)`.
    fn lift_expr(&mut self, x: Arc<Expr>, l: &mut Lift<'_>) -> Result<Arc<Expr>, AnalysisError> {
        if let ExprNode::Window(_) = &x.node {
            let mut first = true;
            return x.map_children(|c| {
                if std::mem::take(&mut first) {
                    c.map_children(|a| self.lift_expr(a, l))
                } else {
                    self.lift_expr(c, l)
                }
            });
        }
        if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &x.node
            && l.lateral.contains(&c.id)
        {
            return Ok(x);
        }
        let k = semantic_key(&x, l.memo);
        let ty = match (l.keys.iter().find(|(kk, _)| *kk == k), &x.node) {
            (Some((_, ty)), _) => ty.clone(),
            (None, ExprNode::Resolved(ResolvedExpr::Function(f)))
                if f.kind == FunctionKind::Aggregate =>
            {
                ArgType {
                    data_type: Some(f.data_type.clone()),
                    nullable: f.nullable,
                }
            }
            (None, ExprNode::Resolved(ResolvedExpr::Column(_))) if l.global => {
                return Err(AnalysisError::new(MissingGroupBy {}, l.origin));
            }
            (None, ExprNode::Resolved(ResolvedExpr::Column(_))) => {
                let text = pretty(&x, self.ctx.conf.session_zone);
                return Err(AnalysisError::new(
                    MissingAggregation {
                        expression: format!("\"{text}\""),
                        expression_any_value: format!("\"any_value({text})\""),
                    },
                    Origin::UNKNOWN,
                ));
            }
            (None, _) => return x.map_children(|c| self.lift_expr(c, l)),
        };
        // An attribute of the aggregate computing `x`, added if new.
        let info = &mut *l.info;
        let found = info
            .aggregates
            .iter()
            .zip(&info.out)
            .find(|(e, _)| semantic_key(e, l.memo) == k)
            .map(|(_, a)| a.clone());
        let key_attr = match &x.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                info.keys.iter().find(|a| a.col.id == c.id).cloned()
            }
            _ => None,
        };
        let a = match (found, key_attr) {
            (Some(a), _) => a,
            (None, Some(a)) => {
                info.aggregates.push(col_expr(&a.col, Origin::UNKNOWN));
                info.out.push(a.clone());
                a
            }
            (None, None) => {
                let output = self.attr(&pretty(&x, self.ctx.conf.session_zone), &ty);
                info.aggregates.push(Arc::new(Expr::new(ExprNode::Resolved(
                    ResolvedExpr::Alias {
                        expr: Arc::clone(&x),
                        output: output.clone(),
                    },
                ))));
                let a = Attr {
                    col: output,
                    qual: no_qual(),
                    source: None,
                };
                info.out.push(a.clone());
                a
            }
        };
        Ok(col_expr(&a.col, x.origin))
    }

    /// Spark's `MISSING_AGGREGATION` / `MISSING_GROUP_BY`: outside
    /// aggregate calls, an item may use only grouping expressions.
    fn check_aggregation(
        &self,
        items: &[Bound],
        keys: &HashSet<u128>,
        memo: &mut HashMap<usize, u128>,
        origin: Origin,
        global: bool,
    ) -> Result<(), AnalysisError> {
        let mut seen = HashSet::new();
        for b in items {
            let mut stack = vec![Arc::clone(&b.expr)];
            while let Some(x) = stack.pop() {
                if !seen.insert(super::bind::addr(&*x)) || keys.contains(&semantic_key(&x, memo)) {
                    continue;
                }
                match &x.node {
                    ExprNode::Resolved(ResolvedExpr::Function(f))
                        if f.kind == FunctionKind::Aggregate => {}
                    ExprNode::Subquery { .. } => {}
                    ExprNode::Resolved(ResolvedExpr::Column(_)) if global => {
                        return Err(AnalysisError::new(MissingGroupBy {}, origin));
                    }
                    ExprNode::Resolved(ResolvedExpr::Column(_)) => {
                        let text = pretty(&x, self.ctx.conf.session_zone);
                        return Err(AnalysisError::new(
                            MissingAggregation {
                                expression: format!("\"{text}\""),
                                expression_any_value: format!("\"any_value({text})\""),
                            },
                            Origin::UNKNOWN,
                        ));
                    }
                    _ => x.for_each_child(&mut |c| stack.push(Arc::clone(c))),
                }
            }
        }
        Ok(())
    }

    pub(super) fn join(
        &mut self,
        p: &LogicalPlan,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        kind: JoinKind,
        condition: &JoinCondition,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let l = self.child(left, env)?;
        let mut r = self.child(right, env)?;
        if let (Some(a), Some(b)) = (l.ids, r.ids)
            && a.0 <= b.1
            && b.0 <= a.1
        {
            if r.outer.is_some_and(|k| k < env.outer.len()) {
                return Err(unsupported("a self-join of a correlated subtree", p.origin));
            }
            r = self.instance(&r)?;
        }
        let nullable = |a: &Attr, yes: bool| Attr {
            col: BoundColumn {
                nullable: a.col.nullable || yes,
                ..a.col.clone()
            },
            ..a.clone()
        };
        let (ln, rn) = match kind {
            JoinKind::LeftOuter => (false, true),
            JoinKind::RightOuter => (true, false),
            JoinKind::FullOuter => (true, true),
            _ => (false, false),
        };
        let mut out: Vec<Attr> = l.out.iter().map(|a| nullable(a, ln)).collect();
        if !matches!(kind, JoinKind::LeftSemi | JoinKind::LeftAnti) {
            out.extend(r.out.iter().map(|a| nullable(a, rn)));
        }
        let rels = [&l, &r];
        let mut keys = None;
        let mut kind = kind;
        let condition = match condition {
            JoinCondition::None => JoinCondition::None,
            JoinCondition::Natural => {
                // `NATURAL JOIN`: the common column names of the two
                // inputs, then the `USING` path, so output order and
                // outer-join coalescing are exactly the `USING` ones. With
                // no common columns the join is a cross join, as Spark's
                // is.
                if matches!(kind, JoinKind::LeftSemi | JoinKind::LeftAnti) {
                    // Spark fails here with a bare `requirement failed:
                    // Unsupported natural join type LeftSemi` (no error class).
                    return Err(unsupported(
                        "NATURAL LEFT SEMI and LEFT ANTI joins",
                        p.origin,
                    ));
                }
                let commons = Self::common_columns(&l, &r, self.ctx.conf.case_sensitive);
                if commons.is_empty() {
                    if kind == JoinKind::Inner {
                        kind = JoinKind::Cross;
                    }
                    JoinCondition::None
                } else {
                    let (cond, k) = self.using(&commons, &l, &r, env, p.origin, true)?;
                    keys = Some(k);
                    JoinCondition::On(cond)
                }
            }
            JoinCondition::On(e) => {
                let scope: Vec<Attr> = l.out.iter().chain(r.out.iter()).cloned().collect();
                let b = self.bind(e, &Self::cx(&scope, &rels, env, "Join"))?;
                if b.ty.data_type != Some(krishiv_types::DataType::Boolean) {
                    return Err(AnalysisError::new(
                        JoinConditionIsNotBooleanType {
                            join_condition: format!(
                                "\"{}\"",
                                pretty(&b.shown(), self.ctx.conf.session_zone)
                            ),
                            condition_type: quoted_type(b.ty.data_type.as_ref()),
                        },
                        p.origin,
                    ));
                }
                JoinCondition::On(b.expr)
            }
            JoinCondition::Using(cols) => {
                let (cond, k) = self.using(cols, &l, &r, env, p.origin, false)?;
                keys = Some(k);
                JoinCondition::On(cond)
            }
        };
        let node = PlanNode::Join {
            left: Arc::clone(&l.plan),
            right: Arc::clone(&r.plan),
            kind,
            condition,
        };
        let Some(keys) = keys else {
            return Ok(self.make(p.origin, node, out, &[&l, &r]));
        };
        let j = self.make(Origin::UNKNOWN, node, out, &[&l, &r]);
        self.using_project(p.origin, &j, kind, &keys, env)
    }

    /// The common column names of a `NATURAL JOIN`'s inputs, as Spark's
    /// `canonicalizedIntersect` computes them: in the left input's order,
    /// each name as many times as both sides output it (a name the left
    /// outputs twice and the right once is common once; resolving a name
    /// always takes the first match, as Spark's key resolution does).
    /// `case_sensitive` applies.
    fn common_columns(l: &Rel, r: &Rel, case_sensitive: bool) -> Vec<String> {
        let mut right: Vec<&str> = r.out.iter().map(|a| a.col.name.as_str()).collect();
        let mut commons: Vec<String> = Vec::new();
        for a in l.out.iter() {
            if let Some(i) = right
                .iter()
                .position(|n| same(n, &a.col.name, case_sensitive))
            {
                right.remove(i);
                commons.push(a.col.name.clone());
            }
        }
        commons
    }

    /// The condition of `USING (cols)`, and each column's two attributes.
    /// A name resolves to the first match on each side when `lenient`
    /// (a repeated `NATURAL JOIN` name pairs the first matches again, as
    /// Spark's key resolution does); otherwise several matches are
    /// ambiguous.
    fn using(
        &mut self,
        cols: &[String],
        l: &Rel,
        r: &Rel,
        env: &Arc<Env>,
        origin: Origin,
        lenient: bool,
    ) -> Result<(Arc<Expr>, Vec<Keys>), AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        let side = |rel: &Rel, which: &str, name: &String| {
            let hits: Vec<&Attr> = rel
                .out
                .iter()
                .filter(|a| same(&a.col.name, name, cs))
                .collect();
            match hits[..] {
                [a] => Ok(a.clone()),
                [a, ..] if lenient => Ok(a.clone()),
                [] => Err(AnalysisError::new(
                    UnresolvedUsingColumnForJoin {
                        col_name: quoted(std::slice::from_ref(name)),
                        side: which.to_owned(),
                        suggestion: names(&rel.out, false),
                    },
                    Origin::UNKNOWN,
                )),
                _ => Err(AnalysisError::new(
                    AmbiguousReference {
                        name: quoted(std::slice::from_ref(name)),
                        reference_names: format!("[{}]", names(&rel.out, false)),
                    },
                    Origin::UNKNOWN,
                )),
            }
        };
        let mut keys = Vec::new();
        let mut cond: Option<Arc<Expr>> = None;
        for name in cols {
            let (la, ra) = (side(l, "left", name)?, side(r, "right", name)?);
            // Spark's errors cite the join for its keys' comparison.
            let eq = self.call(
                "==",
                vec![
                    col_expr(&la.col, Origin::UNKNOWN),
                    col_expr(&ra.col, Origin::UNKNOWN),
                ],
                env,
                origin,
            )?;
            cond = Some(match cond {
                None => eq,
                Some(c) => self.call("and", vec![c, eq], env, Origin::UNKNOWN)?,
            });
            keys.push((la, ra));
        }
        let cond = cond.ok_or_else(|| unsupported("USING ()", Origin::UNKNOWN))?;
        Ok((cond, keys))
    }

    /// A resolved call of `name` on resolved arguments.
    pub(super) fn call(
        &mut self,
        name: &str,
        args: Vec<Arc<Expr>>,
        env: &Arc<Env>,
        origin: Origin,
    ) -> Result<Arc<Expr>, AnalysisError> {
        let call = Arc::new(
            Expr::new(ExprNode::Function(FunctionCall {
                name: name.to_owned(),
                qualifier: Vec::new(),
                arg_names: Vec::new(),
                args,
                distinct: false,
                filter: None,
            }))
            .with_origin(origin),
        );
        Ok(self.bind(&call, &Self::cx(&[], &[], env, "Join"))?.expr)
    }

    /// The output of a `USING` join: the keys (from the side that keeps
    /// all rows; `coalesce` for a full join), then the other columns.
    fn using_project(
        &mut self,
        origin: Origin,
        j: &Rel,
        kind: JoinKind,
        keys: &[(Attr, Attr)],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        // A key as the join outputs it (nullable as the join made it). A
        // semi or anti join outputs the left side only, its keys included
        // (Spark), so only the left key is looked up there.
        let find = |a: &Attr| {
            j.out
                .iter()
                .find(|x| x.col.id == a.col.id)
                .cloned()
                .unwrap_or_else(|| a.clone())
        };
        let mut exprs = Vec::new();
        let mut out = Vec::new();
        let mut key_ids = HashSet::new();
        for (la, ra) in keys {
            key_ids.extend([la.col.id, ra.col.id]);
            let l = find(la);
            match kind {
                JoinKind::RightOuter => {
                    let r = find(ra);
                    exprs.push(col_expr(&r.col, Origin::UNKNOWN));
                    out.push(r);
                }
                JoinKind::FullOuter => {
                    let r = find(ra);
                    let both = vec![
                        col_expr(&l.col, Origin::UNKNOWN),
                        col_expr(&r.col, Origin::UNKNOWN),
                    ];
                    let value = self.call("coalesce", both, env, Origin::UNKNOWN)?;
                    let named = Arc::new(Expr::new(ExprNode::Alias {
                        expr: value,
                        name: l.col.name.clone(),
                    }));
                    let b = self.bind(&named, &Self::cx(&[], &[], env, "Project"))?;
                    out.push(out_attr(&b, &[]));
                    exprs.push(b.expr);
                }
                _ => {
                    exprs.push(col_expr(&l.col, Origin::UNKNOWN));
                    out.push(l);
                }
            }
        }
        for a in j.out.iter().filter(|a| !key_ids.contains(&a.col.id)) {
            exprs.push(col_expr(&a.col, Origin::UNKNOWN));
            out.push(a.clone());
        }
        let node = PlanNode::Project {
            input: Arc::clone(&j.plan),
            exprs,
        };
        Ok(self.make(origin, node, out, &[j]))
    }

    /// A `LIMIT` (`offset` false) or `OFFSET` over `input`. Kept out of line:
    /// `resolve` recurses per plan level.
    #[inline(never)]
    pub(super) fn limit_node(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        count: &Arc<Expr>,
        offset: bool,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        let name = if offset { "offset" } else { "limit" };
        let expr = self.limit_like(count, name, &c, env)?;
        // `LIMIT l OFFSET o` is a limit over an offset; Spark bounds `l + o`.
        if let (
            false,
            PlanNode::Offset { offset: below, .. },
            ExprNode::Literal(Literal::Integer(l)),
        ) = (offset, &c.plan.node, &expr.node)
            && let ExprNode::Literal(Literal::Integer(o)) = &below.node
            && i64::from(*l) + i64::from(*o) > i64::from(i32::MAX)
        {
            return Err(AnalysisError::new(
                SumOfLimitAndOffsetExceedsMaxInt {
                    limit: l.to_string(),
                    offset: o.to_string(),
                },
                count.origin,
            ));
        }
        let input = Arc::clone(&c.plan);
        let node = if offset {
            PlanNode::Offset {
                input,
                offset: expr,
            }
        } else {
            PlanNode::Limit { input, limit: expr }
        };
        let mut r = self.make(p.origin, node, c.out.to_vec(), &[&c]);
        r.widen = Some(Arc::new(Widen::Through { child: c }));
        Ok(r)
    }

    /// The count of a `LIMIT` or `OFFSET` (`name`), checked as Spark does:
    /// it must be a constant, an `INT`, not NULL and not negative.
    pub(super) fn limit_like(
        &mut self,
        e: &Arc<Expr>,
        name: &str,
        input: &Rel,
        env: &Arc<Env>,
    ) -> Result<Arc<Expr>, AnalysisError> {
        // Columns resolve against the input (so `OFFSET id` is "not
        // constant", as in Spark, not an unresolved column).
        let rels = [input];
        let cx = Self::cx(
            &input.out,
            &rels,
            env,
            if name == "limit" { "Limit" } else { "Offset" },
        );
        let b = self.bind(e, &cx)?;
        // Spark's order: foldable, then the type, then the value.
        let unfoldable = not_constant(&b.expr);
        if !unfoldable && b.ty.data_type != Some(krishiv_types::DataType::Integer) {
            return Err(AnalysisError::new(
                InvalidLimitLikeExpressionDataType {
                    expr: format!("\"{}\"", pretty(&b.shown(), self.ctx.conf.session_zone)),
                    name: name.to_owned(),
                    data_type: quoted_type(b.ty.data_type.as_ref()),
                },
                e.origin,
            ));
        }
        let folded = self.check_limit(&b, name, unfoldable, e.origin, input.plan.origin)?;
        match folded {
            // `CAST('3' AS INT)` and the like: the planner reads a literal.
            Some(v) if !matches!(b.expr.node, ExprNode::Literal(_)) => {
                let v = i32::try_from(v)
                    .map_err(|_| unsupported("a LIMIT or OFFSET out of range", e.origin))?;
                Ok(Arc::new(Expr::literal(Literal::Integer(v))))
            }
            _ => Ok(b.expr),
        }
    }

    pub(super) fn sort(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        order: &[SortOrder],
        global: bool,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        // `ORDER BY ALL [ASC|DESC] [NULLS ...]`: every output column in
        // order, with that direction (a column named `all` takes
        // precedence, as Spark resolves it first). Only a global sort: in
        // `SORT BY ALL` and `sortWithinPartitions("all")` it is an unknown
        // column (`UNRESOLVED_COLUMN`, checked against Spark 4.2.0). Inside a window
        // specification `all` stays a column: only the sort's own key
        // expands here.
        let expanded;
        let order = if global
            && order.len() == 1
            && is_all(&order[0].expr)
            && !c
                .out
                .iter()
                .any(|a| same(&a.col.name, "all", self.ctx.conf.case_sensitive))
            && !c.agg.as_ref().is_some_and(|agg| {
                agg.keys
                    .iter()
                    .any(|a| same(&a.col.name, "all", self.ctx.conf.case_sensitive))
                    || agg
                        .input
                        .out
                        .iter()
                        .any(|a| same(&a.col.name, "all", self.ctx.conf.case_sensitive))
            }) {
            let template = &order[0];
            expanded = c
                .out
                .iter()
                .map(|a| SortOrder {
                    expr: col_expr(&a.col, template.expr.origin),
                    ..template.clone()
                })
                .collect::<Vec<_>>();
            &expanded[..]
        } else {
            order
        };
        let mut exprs = Vec::new();
        for o in order {
            exprs.push(match &o.expr.node {
                // `ORDER BY n`: the `n`-th output column.
                ExprNode::Literal(Literal::Integer(n)) if self.ctx.conf.order_by_ordinal => {
                    let i = usize::try_from(*n)
                        .ok()
                        .filter(|i| (1..=c.out.len()).contains(i));
                    let Some(i) = i else {
                        return Err(AnalysisError::new(
                            OrderByPosOutOfRange {
                                index: n.to_string(),
                                size: c.out.len().to_string(),
                            },
                            o.expr.origin,
                        ));
                    };
                    col_expr(&c.out[i - 1].col, o.expr.origin)
                }
                _ => Arc::clone(&o.expr),
            });
        }
        let sorted = |exprs: Vec<Arc<Expr>>| -> Vec<SortOrder> {
            order
                .iter()
                .zip(exprs)
                .map(|(o, expr)| SortOrder { expr, ..o.clone() })
                .collect()
        };
        // Over no grouping sets, `ORDER BY grouping(..)` is rejected before
        // its argument resolves (`HAVING` reports the argument first).
        if c.agg.as_ref().is_none_or(|a| a.analytics.is_none()) {
            super::analytics::reject_grouping_calls(&exprs)?;
        }
        // Over an aggregate with windows, Spark's `ORDER BY` reads the
        // select list only (nothing resolves through the windows).
        // A name the aggregate outputs twice is `UNRESOLVED_COLUMN` in
        // `ORDER BY` (not `AMBIGUOUS_REFERENCE`): Spark lists the
        // duplicates as proposals (checked against Spark 4.2.0).
        let (bounds, below, extended) = if c.agg.as_ref().is_some_and(|i| !i.has_windows()) {
            for e in &exprs {
                if let ExprNode::Column(col) = &e.node
                    && col.plan_id.is_none()
                    && col.name.len() == 1
                    && c.out
                        .iter()
                        .filter(|a| same(&a.col.name, &col.name[0], self.ctx.conf.case_sensitive))
                        .count()
                        > 1
                {
                    return Err(unresolved(&col.name, &c.out, e.origin));
                }
            }
            self.over_agg(&c, &exprs, env, "Sort")?
        } else {
            // A name the select list outputs twice is looked up below it,
            // as Spark does.
            match self.bind_missing(&c, &exprs, env, "Sort", true) {
                Ok((bounds, wide)) => {
                    let extended = wide.out.len() > c.out.len();
                    (bounds, wide, extended)
                }
                // `HAVING ... ORDER BY`: the sort reads the aggregate below
                // the filter, which grows what the sort needs.
                Err(e)
                    if e.error
                        .error_class()
                        .name()
                        .starts_with("UNRESOLVED_COLUMN") =>
                {
                    let Some((layers, inner)) = Self::through_to_agg(&c) else {
                        return Err(e);
                    };
                    // Over windows, Spark's `ORDER BY` reads the select list
                    // only: nothing resolves through them to the aggregate.
                    if inner.agg.as_ref().is_some_and(|i| i.has_windows()) {
                        return Err(e);
                    }
                    let Ok((bounds, grown, extended)) = self.over_agg(&inner, &exprs, env, "Sort")
                    else {
                        return Err(e);
                    };
                    let top = if extended {
                        self.rewrapped(&layers, grown)
                    } else {
                        c.clone()
                    };
                    (bounds, top, extended)
                }
                Err(e) => return Err(e),
            }
        };
        let node = PlanNode::Sort {
            input: Arc::clone(&below.plan),
            order: sorted(bounds.into_iter().map(|b| b.expr).collect()),
            global,
        };
        let mut s = self.make(p.origin, node, below.out.to_vec(), &[&below]);
        s.widen = Some(Arc::new(Widen::Through { child: below }));
        Ok(if extended {
            self.restore(&s, &c.out)
        } else {
            s
        })
    }

    pub(super) fn subquery_alias(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        alias: &str,
        columns: &[String],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        let inner = if columns.is_empty() {
            c
        } else {
            if columns.len() != c.out.len() {
                return Err(AnalysisError::new(
                    AssignmentArityMismatch {
                        num_target: c.out.len().to_string(),
                        num_expr: columns.len().to_string(),
                    },
                    p.origin,
                ));
            }
            let mut exprs = Vec::new();
            let mut out = Vec::new();
            for (a, name) in c.out.iter().zip(columns) {
                let output = BoundColumn {
                    id: self.fresh(),
                    name: name.clone(),
                    ..a.col.clone()
                };
                exprs.push(Arc::new(Expr::new(ExprNode::Resolved(
                    ResolvedExpr::Alias {
                        expr: col_expr(&a.col, Origin::UNKNOWN),
                        output: output.clone(),
                    },
                ))));
                out.push(Attr {
                    col: output,
                    qual: no_qual(),
                    source: a.source.clone(),
                });
            }
            let node = PlanNode::Project {
                input: Arc::clone(&c.plan),
                exprs,
            };
            self.make(Origin::UNKNOWN, node, out, &[&c])
        };
        let qual: Arc<[String]> = Arc::from(vec![alias.to_owned()]);
        let out = inner
            .out
            .iter()
            .map(|a| Attr {
                qual: Arc::clone(&qual),
                ..a.clone()
            })
            .collect();
        let node = PlanNode::SubqueryAlias {
            input: Arc::clone(&inner.plan),
            alias: alias.to_owned(),
            columns: Vec::new(),
        };
        Ok(self.make(p.origin, node, out, &[&inner]))
    }
}

/// The state of [`Resolver::lift`]: the aggregate it reads (and grows),
/// its grouping keys and the lateral aliases of the select list.
struct Lift<'a> {
    info: &'a mut AggInfo,
    keys: &'a [(u128, ArgType)],
    lateral: &'a HashSet<AttrId>,
    memo: &'a mut HashMap<usize, u128>,
    global: bool,
    origin: Origin,
}

/// The attribute a named item outputs; a column keeps its input
/// attribute's qualifier and source.
fn out_attr(b: &Bound, scope: &[Attr]) -> Attr {
    match &b.expr.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => {
            match scope.iter().find(|a| a.col.id == c.id) {
                Some(a) => Attr {
                    col: c.clone(),
                    ..a.clone()
                },
                None => Attr {
                    col: c.clone(),
                    qual: no_qual(),
                    source: None,
                },
            }
        }
        ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => Attr {
            col: output.clone(),
            qual: no_qual(),
            source: None,
        },
        _ => Attr {
            col: BoundColumn {
                id: AttrId(u64::MAX),
                name: pretty(&b.expr, None),
                data_type: krishiv_types::DataType::Boolean,
                nullable: true,
            },
            qual: no_qual(),
            source: None,
        },
    }
}

/// Spark's `DATATYPE_MISMATCH.FILTER_NOT_BOOLEAN`.
pub(super) fn boolean_filter(b: &Bound, origin: Origin) -> Result<(), AnalysisError> {
    if b.ty.data_type == Some(krishiv_types::DataType::Boolean) {
        return Ok(());
    }
    let text = format!("\"{}\"", pretty(&b.shown(), None));
    Err(AnalysisError::new(
        DatatypeMismatchFilterNotBoolean {
            sql_expr: text.clone(),
            filter: text,
            r#type: quoted_type(b.ty.data_type.as_ref()),
        },
        origin,
    ))
}

/// `e` with every column renamed by `rename` (an attribute's name in
/// generated text only; ids and types stay).
fn renamed_columns(e: &Arc<Expr>, rename: &dyn Fn(&BoundColumn) -> Option<String>) -> Arc<Expr> {
    let out = Arc::clone(e).transform_up(&mut |n: Arc<Expr>| {
        Ok::<_, std::convert::Infallible>(match &n.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => match rename(c) {
                Some(name) => col_expr(&BoundColumn { name, ..c.clone() }, n.origin),
                None => n,
            },
            _ => n,
        })
    });
    out.unwrap_or_else(|never| match never {})
}

/// `e` with its columns fully qualified, as Spark's `sql` prints them.
pub(super) fn qualified(e: &Arc<Expr>, scope: &[Attr]) -> Arc<Expr> {
    renamed_columns(e, &|c| {
        let mut parts: Vec<String> = scope
            .iter()
            .find(|a| a.col.id == c.id)
            .map(|a| a.qual.to_vec())
            .unwrap_or_default();
        parts.push(c.name.clone());
        Some(parts.join("."))
    })
}

/// `e` with each reference to a lateral alias in `ids` read as Spark names
/// it: `lateralAliasReference(name)`.
pub(super) fn lateral_refs(e: &Arc<Expr>, ids: &[AttrId]) -> Arc<Expr> {
    if ids.is_empty() {
        return Arc::clone(e);
    }
    renamed_columns(e, &|c| {
        ids.contains(&c.id)
            .then(|| format!("lateralAliasReference({})", c.name))
    })
}

/// Spark's `quoteIfNeeded`: a name of letters, digits and `_` (not all
/// digits) stays bare.
fn quote_if_needed(name: &str) -> String {
    let bare = !name.is_empty()
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.chars().all(|c| c.is_ascii_digit());
    if bare {
        name.to_owned()
    } else {
        quoted(&[name.to_owned()])
    }
}

/// The attribute an aliased item defines.
fn alias_id(e: &Expr) -> Option<AttrId> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => Some(output.id),
        _ => None,
    }
}

/// Whether `keys` is a lone `GROUP BY ALL` / `ORDER BY ALL`: one
/// unqualified, unresolvable-by-shape column named `all` (case-insensitive;
/// the analyzer resolves a real column first, so this only gates the
/// expansion).
fn is_bare_all(keys: &[Arc<Expr>]) -> bool {
    matches!(keys, [k] if is_all(k))
}

/// Whether `e` is an unqualified column named `all` (case-insensitive)
/// without a DataFrame owner.
fn is_all(e: &Expr) -> bool {
    match &e.node {
        ExprNode::Column(c) => {
            c.plan_id.is_none() && matches!(&c.name[..], [n] if n.eq_ignore_ascii_case("all"))
        }
        _ => false,
    }
}

/// Whether `e` calls an aggregate function outside any subquery: a scalar
/// subquery groups like any other value (checked against Spark 4.2.0),
/// whatever it computes inside.
fn contains_aggregate(e: &Arc<Expr>) -> bool {
    let mut found = false;
    contains_walk(e, false, &mut found, &mut |x, _| {
        if matches!(&x.node, ExprNode::Subquery { .. }) {
            Contains::Skip
        } else if matches!(&x.node, ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate)
        {
            Contains::Yes
        } else {
            Contains::No
        }
    });
    found
}

/// Whether `e` names a column outside any aggregate call's arguments: an
/// attribute, or a correlated (`OuterColumn`) reference inside a subquery
/// (columns a subquery reads from its own scope do not count).
fn contains_bare(e: &Arc<Expr>) -> bool {
    let mut found = false;
    contains_walk(e, false, &mut found, &mut |x, in_subquery| match &x.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate => {
            Contains::Skip
        }
        ExprNode::Resolved(ResolvedExpr::Column(_)) if !in_subquery => Contains::Yes,
        ExprNode::Resolved(ResolvedExpr::OuterColumn(_)) => Contains::Yes,
        _ => Contains::No,
    });
    found
}

/// What a [`contains_walk`] visitor reports for a node.
enum Contains {
    /// Counts: stop the walk.
    Yes,
    /// Does not count: visit children.
    No,
    /// Does not count: skip the subtree (aggregate arguments).
    Skip,
}

/// Walks `e` and the plans of the subqueries it holds, outside in;
/// `in_subquery` says whether the node is inside a subquery plan. A
/// visitor answering [`Contains::Yes`] stops the walk.
fn contains_walk(
    e: &Arc<Expr>,
    in_subquery: bool,
    found: &mut bool,
    visit: &mut impl FnMut(&Arc<Expr>, bool) -> Contains,
) {
    let mut stack = vec![(Arc::clone(e), in_subquery)];
    while let Some((x, inner)) = stack.pop() {
        if *found {
            return;
        }
        match visit(&x, inner) {
            Contains::Yes => {
                *found = true;
                return;
            }
            Contains::Skip => continue,
            Contains::No => {}
        }
        if let ExprNode::Subquery { plan, .. } = &x.node {
            let mut plans = vec![Arc::clone(plan)];
            while let Some(p) = plans.pop() {
                let mut exprs = Vec::new();
                p.for_each_expr(&mut |e| exprs.push(Arc::clone(e)));
                stack.extend(exprs.into_iter().map(|e| (e, true)));
                p.for_each_child(&mut |k| plans.push(Arc::clone(k)));
            }
            continue;
        }
        x.for_each_child(&mut |k| stack.push((Arc::clone(k), inner)));
    }
}
/// The `QUALIFY` predicate of a wrapped filter predicate, if `e` is the
/// wrapper lowering stamps (see `lower.rs`).
fn qualify_expression(e: &Arc<Expr>) -> Option<&Arc<Expr>> {
    match &e.node {
        ExprNode::Function(f) if is_qualify_wrapper(&f.name) && f.args.len() == 1 => {
            Some(&f.args[0])
        }
        _ => None,
    }
}

/// Whether `name` is lowering's `QUALIFY` marker (see `lower.rs`): the
/// analyzer resolves and strips it, so it is never a routine call.
pub(super) fn is_qualify_wrapper(name: &str) -> bool {
    name == "qualify"
}

/// The first aggregate call in a `QUALIFY` predicate, outside window
/// expressions and subqueries (neither counts, as for the select list's
/// own aggregates): `QUALIFY` forbids aggregates.
fn first_qualify_aggregate(q: &Arc<Expr>, r: &Resolver<'_>) -> Option<Arc<Expr>> {
    let mut stack = vec![Arc::clone(q)];
    while let Some(x) = stack.pop() {
        match &x.node {
            ExprNode::Window(_) | ExprNode::Subquery { .. } => continue,
            _ => {}
        }
        if r.is_aggregate(&x) {
            return Some(x);
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    None
}

/// Whether a `QUALIFY` predicate holds a window expression, outside
/// subqueries.
fn has_qualify_window(q: &Arc<Expr>) -> bool {
    let mut stack = vec![Arc::clone(q)];
    while let Some(x) = stack.pop() {
        match &x.node {
            ExprNode::Subquery { .. } => continue,
            ExprNode::Window(_) => return true,
            _ => {}
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    false
}

/// Whether `e` calls an aggregate function. Subquery plans are not
/// expressions and are not visited, so their own aggregates never count.
fn has_aggregate(e: &Arc<Expr>) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if !seen.insert(super::bind::addr(&*x)) {
            continue;
        }
        if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node
            && f.kind == FunctionKind::Aggregate
        {
            return true;
        }
        x.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    false
}

/// Spark's `UNSUPPORTED_FEATURE.LATERAL_COLUMN_ALIAS_IN_AGGREGATE_FUNC`:
/// an aggregate call (its `FILTER` included) of `b` reads a lateral alias.
fn in_aggregate_call(b: &Bound) -> Result<(), AnalysisError> {
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(&b.expr)];
    while let Some(x) = stack.pop() {
        if !seen.insert(super::bind::addr(&*x)) {
            continue;
        }
        if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node
            && f.kind == FunctionKind::Aggregate
        {
            let mut inner = vec![Arc::clone(&x)];
            while let Some(y) = inner.pop() {
                if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &y.node
                    && b.lateral.contains(&c.id)
                {
                    return Err(AnalysisError::new(
                        UnsupportedFeatureLateralColumnAliasInAggregateFunc {
                            agg_func: format!(
                                "\"{}\"",
                                pretty(&lateral_refs(&x, &b.lateral), None)
                            ),
                            lca: quoted(std::slice::from_ref(&c.name)),
                        },
                        Origin::UNKNOWN,
                    ));
                }
                y.for_each_child(&mut |k| inner.push(Arc::clone(k)));
            }
            continue;
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    Ok(())
}

/// The aggregate calls in `e` as Spark's `sql` prints them, columns fully
/// qualified (for `INVALID_WHERE_CONDITION`).
fn aggregate_sql(e: &Arc<Expr>, scope: &[Attr]) -> String {
    let mut found = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if !seen.insert(super::bind::addr(&*x)) {
            continue;
        }
        if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node
            && f.kind == FunctionKind::Aggregate
        {
            found.push(pretty(&qualified(&x, scope), None));
            continue;
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    found.join(", ")
}

/// Whether `e` reads a column or calls a non-deterministic function, so it
/// cannot be folded to a constant.
fn not_constant(e: &Expr) -> bool {
    if matches!(
        &e.node,
        // A subquery is not foldable, whatever it selects.
        ExprNode::Subquery { .. }
            | ExprNode::Resolved(
                ResolvedExpr::Column(_)
                    | ResolvedExpr::OuterColumn(_)
                    | ResolvedExpr::Function(ResolvedFunction {
                        deterministic: false,
                        ..
                    })
            )
    ) {
        return true;
    }
    let mut found = false;
    e.for_each_child(&mut |c| found = found || not_constant(c));
    found
}
