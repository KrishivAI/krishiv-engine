// SPDX-License-Identifier: Apache-2.0
//! The relational operators of the Resolve pass and their rewrites.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::convert::Infallible;
use std::sync::Arc;

use krishiv_common::error::{
    AmbiguousReference, AssignmentArityMismatch, FailedToParseTooComplex, GroupByPosAggregate,
    GroupByPosOutOfRange, InvalidWhereCondition, MissingAggregation, MissingGroupBy,
    OrderByPosOutOfRange, StarGroupByPos, TableOrViewNotFound, UnresolvedUsingColumnForJoin,
    UnsupportedFeatureLateralColumnAliasInAggregateFunc,
    UnsupportedFeatureLateralColumnAliasInGroupBy,
};

use super::bind::{BoundTable, same, unsupported};
use super::catalog::quoted;
use super::exprs::{Bound, Cx, names, pretty, semantic_key};
use super::resolve::{AggInfo, Attr, Env, Rel, Resolver, Widen};
use super::{AnalysisError, ArgType, TableKind};
use crate::{
    AttrId, BoundColumn, CteRef, Expr, ExprNode, FunctionCall, FunctionKind, Grouping,
    JoinCondition, JoinKind, Literal, LogicalPlan, Origin, PlanNode, ReadSource, ResolvedExpr,
    Scan, SortOrder,
};

/// The two attributes of a `USING` column.
type Keys = (Attr, Attr);

/// The largest attribute id: ids are numbered densely from 0, and a
/// self-join instance reserves its subtree's span; past this (about 2^62),
/// a plan is too complex.
pub(super) const MAX_ID: u64 = 1 << 62;

pub(super) fn col_expr(c: &BoundColumn, origin: Origin) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(c.clone()))).with_origin(origin))
}

fn column_bound(a: &Attr, origin: Origin) -> Bound {
    Bound {
        expr: col_expr(&a.col, origin),
        ty: ArgType {
            data_type: Some(a.col.data_type.clone()),
            nullable: a.col.nullable,
        },
        agg: None,
        outer: None,
        lateral: Vec::new(),
    }
}

fn no_qual() -> Arc<[String]> {
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
            return self.build_aggregate(p.origin, c, Vec::new(), &items, true);
        }
        Ok(self.lateral_project(p.origin, c, &items, &[]))
    }

    /// A select list: stars expanded, every item named. A name the input
    /// lacks may be an alias defined earlier in the list (a lateral column
    /// alias), when `spark.sql.lateralColumnAlias.enableImplicitResolution`.
    fn select_items(
        &mut self,
        c: &Rel,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
        op: &'static str,
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
        let name = match &b.expr.node {
            ExprNode::Resolved(ResolvedExpr::Column(_) | ResolvedExpr::Alias { .. }) => {
                return Ok(b);
            }
            ExprNode::Cast { expr, .. }
                if matches!(expr.node, ExprNode::Resolved(ResolvedExpr::Column(_))) =>
            {
                pretty(expr)
            }
            _ => pretty(&lateral_refs(&b.expr, &b.lateral)),
        };
        let output = self.attr(&name, &b.ty, b.expr.origin)?;
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
    fn lateral_project(&self, origin: Origin, base: &Rel, items: &[Bound], pass: &[Attr]) -> Rel {
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
    fn projection(
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
        if c.agg.is_some() {
            let (bounds, agg, extended) =
                self.over_agg(&c, std::slice::from_ref(predicate), env, "Filter")?;
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
                    condition: format!("\"{}\"", pretty(&b.expr)),
                    expression_list: aggregate_sql(&b.expr, &wide.out),
                },
                p.origin,
            ));
        }
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

    /// Binds the top-level expressions of a filter or sort over `c`. Names
    /// `c` does not output, but a relation below it dropped (through
    /// projections, filters, sorts and limits, down to an aggregate's
    /// grouping attributes), are Spark's missing references: they are
    /// added back up to `c` and resolve. Enclosing queries come last, as
    /// in Spark: a dropped column hides an outer one of its name. Returns
    /// the bound expressions and the relation they read, wider than `c`
    /// when something was added. `lenient`: a name `c` outputs several
    /// times is looked up below.
    fn bind_missing(
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
        // Without the enclosing queries when there is something below.
        let below = Self::below(c);
        cx.outer = below.is_empty();
        let unresolved = |e: &AnalysisError| {
            e.error
                .error_class()
                .name()
                .starts_with("UNRESOLVED_COLUMN")
        };
        match self.bind_all(exprs, &cx) {
            Ok(bounds) => return Ok((bounds, c.clone())),
            Err(e) if !unresolved(&e) || below.is_empty() => return Err(e),
            Err(_) => {}
        }
        cx.extra = Some(&below);
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
        Ok((bounds, self.widened(c, &missing)))
    }

    /// Binds each of `exprs` in `cx`.
    fn bind_all(&mut self, exprs: &[Arc<Expr>], cx: &Cx<'_>) -> Result<Vec<Bound>, AnalysisError> {
        exprs.iter().map(|e| self.bind(e, cx)).collect()
    }

    /// The attributes relations below `r` dropped and could add back
    /// (see [`Self::bind_missing`]), nearest first.
    fn below(r: &Rel) -> Vec<Attr> {
        let mut out: Vec<Attr> = Vec::new();
        let mut seen: HashSet<AttrId> = r.out.iter().map(|a| a.col.id).collect();
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
    fn widened(&self, r: &Rel, missing: &[Attr]) -> Rel {
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
                    self.aggregate_rel(rel.plan.origin, grown, &need)
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
        built.unwrap_or_else(|| r.clone())
    }

    /// Expressions over an aggregate (`HAVING`, `ORDER BY`): names resolve
    /// against its output, then its grouping attributes; aggregate-function
    /// arguments against its input. Aggregate calls the aggregate does not
    /// compute yet, and grouping attributes it does not output, are added
    /// to it (and passed up through the projections of lateral aliases
    /// above it); the flag says whether its output grew.
    fn over_agg(
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
        let mut bounds = Vec::new();
        for e in exprs {
            bounds.push(self.bind(e, &cx)?);
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
                            let output = self.attr(&pretty(&x), &ty, x.origin)?;
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
        if pass.is_empty() {
            return Ok((bounds, agg.clone(), false));
        }
        let r = self.aggregate_rel(agg.plan.origin, grown, &pass);
        Ok((bounds, r, true))
    }

    /// The aggregate of `info`, under the projections of its lateral
    /// aliases if it has any; `pass` (attributes of the aggregate) are
    /// output after the select items.
    fn aggregate_rel(&self, origin: Origin, info: AggInfo, pass: &[Attr]) -> Rel {
        let node = PlanNode::Aggregate {
            input: Arc::clone(&info.input.plan),
            grouping: Grouping::GroupBy(info.grouping.clone()),
            aggregates: info.aggregates.clone(),
        };
        let mut r = match &info.top {
            None => self.make(origin, node, info.out.clone(), &[&info.input]),
            Some(top) => {
                let a = self.make(Origin::UNKNOWN, node, info.out.clone(), &[&info.input]);
                self.lateral_project(origin, &a, top, pass)
            }
        };
        let info = Arc::new(info);
        r.widen = Some(Arc::new(Widen::Aggregate(Arc::clone(&info))));
        r.agg = Some(info);
        r
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
            return Err(unsupported("ROLLUP, CUBE and GROUPING SETS", p.origin));
        };
        let c = self.child(input, env)?;
        let conf = self.ctx.conf;
        let ordinal = |k: &Expr| match &k.node {
            ExprNode::Literal(Literal::Integer(n)) if conf.group_by_ordinal => Some(*n),
            _ => None,
        };
        let star = aggregates
            .iter()
            .any(|e| matches!(e.node, ExprNode::Star(_)));
        if star && keys.iter().any(|k| ordinal(k).is_some()) {
            return Err(AnalysisError::new(StarGroupByPos {}, Origin::UNKNOWN));
        }
        // Spark resolves the select list first, then ordinals, then the
        // other keys.
        let items = self.select_items(&c, aggregates, env, "Aggregate")?;
        let mut group: Vec<Option<Bound>> = vec![None; keys.len()];
        for (slot, k) in group.iter_mut().zip(keys) {
            if let Some(n) = ordinal(k) {
                *slot = Some(self.group_ordinal(n, k.origin, &items, &c)?);
            }
        }
        let aliases: Vec<(String, Arc<Expr>)> = aggregates
            .iter()
            .filter(|_| conf.group_by_aliases)
            .filter_map(|e| match &e.node {
                ExprNode::Alias { expr, name } => Some((name.clone(), Arc::clone(expr))),
                _ => None,
            })
            .collect();
        let rels = [&c];
        for (slot, k) in group.iter_mut().zip(keys) {
            if slot.is_none() {
                let mut cx = Self::cx(&c.out, &rels, env, "Aggregate");
                cx.aliases = &aliases;
                *slot = Some(self.bind(k, &cx)?);
            }
        }
        let group = group.into_iter().flatten().collect();
        self.build_aggregate(p.origin, &c, group, &items, false)
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
            let text = pretty(&qualified(expr, &c.out));
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
    fn build_aggregate(
        &mut self,
        origin: Origin,
        c: &Rel,
        group: Vec<Bound>,
        items: &[Bound],
        global: bool,
    ) -> Result<Rel, AnalysisError> {
        let mut memo = HashMap::new();
        let gkeys: Vec<(u128, ArgType)> = group
            .iter()
            .map(|b| (semantic_key(&b.expr, &mut memo), b.ty.clone()))
            .collect();
        let key_set: HashSet<u128> = gkeys.iter().map(|(k, _)| *k).collect();
        let base: Vec<Bound> = items
            .iter()
            .filter(|b| b.lateral.is_empty())
            .cloned()
            .collect();
        self.check_aggregation(&base, &key_set, &mut memo, origin, global)?;
        let keys: Vec<Attr> = group
            .iter()
            .filter_map(|b| match &b.expr.node {
                ExprNode::Resolved(ResolvedExpr::Column(col)) => {
                    c.out.iter().find(|a| a.col.id == col.id).cloned()
                }
                _ => None,
            })
            .collect();
        let mut info = AggInfo {
            input: c.clone(),
            grouping: group.into_iter().map(|b| b.expr).collect(),
            aggregates: base.iter().map(|b| Arc::clone(&b.expr)).collect(),
            out: base.iter().map(|b| out_attr(b, &c.out)).collect(),
            keys,
            top: None,
        };
        if base.len() == items.len() {
            return Ok(self.aggregate_rel(origin, info, &[]));
        }
        for b in items.iter().filter(|b| !b.lateral.is_empty()) {
            in_aggregate_call(b)?;
        }
        let lateral: HashSet<AttrId> = items.iter().filter_map(|b| alias_id(&b.expr)).collect();
        let mut top = Vec::new();
        for b in items {
            if b.lateral.is_empty() {
                let a = out_attr(b, &c.out);
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
        Ok(self.aggregate_rel(origin, info, &[]))
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
        Arc::clone(&b.expr).transform_down(&mut |x: Arc<Expr>| {
            if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &x.node
                && lateral.contains(&c.id)
            {
                return Ok(x);
            }
            let k = semantic_key(&x, memo);
            let ty = match (keys.iter().find(|(kk, _)| *kk == k), &x.node) {
                (Some((_, ty)), _) => ty.clone(),
                (None, ExprNode::Resolved(ResolvedExpr::Function(f)))
                    if f.kind == FunctionKind::Aggregate =>
                {
                    ArgType {
                        data_type: Some(f.data_type.clone()),
                        nullable: f.nullable,
                    }
                }
                (None, node) => {
                    return match node {
                        ExprNode::Resolved(ResolvedExpr::Column(_)) if global => {
                            Err(AnalysisError::new(MissingGroupBy {}, origin))
                        }
                        ExprNode::Resolved(ResolvedExpr::Column(_)) => {
                            let text = pretty(&x);
                            Err(AnalysisError::new(
                                MissingAggregation {
                                    expression: format!("\"{text}\""),
                                    expression_any_value: format!("\"any_value({text})\""),
                                },
                                Origin::UNKNOWN,
                            ))
                        }
                        _ => Ok(x),
                    };
                }
            };
            // An attribute of the aggregate computing `x`, added if new.
            let found = info
                .aggregates
                .iter()
                .zip(&info.out)
                .find(|(e, _)| semantic_key(e, memo) == k)
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
                    let output = self.attr(&pretty(&x), &ty, x.origin)?;
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
        })
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
                        let text = pretty(&x);
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
        let condition = match condition {
            JoinCondition::None => JoinCondition::None,
            JoinCondition::On(e) => {
                let scope: Vec<Attr> = l.out.iter().chain(r.out.iter()).cloned().collect();
                JoinCondition::On(self.bind(e, &Self::cx(&scope, &rels, env, "Join"))?.expr)
            }
            JoinCondition::Using(cols) => {
                let (cond, k) = self.using(cols, &l, &r, env)?;
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

    /// The condition of `USING (cols)`, and each column's two attributes.
    fn using(
        &mut self,
        cols: &[String],
        l: &Rel,
        r: &Rel,
        env: &Arc<Env>,
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
            let eq = self.call(
                "==",
                vec![
                    col_expr(&la.col, Origin::UNKNOWN),
                    col_expr(&ra.col, Origin::UNKNOWN),
                ],
                env,
            )?;
            cond = Some(match cond {
                None => eq,
                Some(c) => self.call("and", vec![c, eq], env)?,
            });
            keys.push((la, ra));
        }
        let cond = cond.ok_or_else(|| unsupported("USING ()", Origin::UNKNOWN))?;
        Ok((cond, keys))
    }

    /// A resolved call of `name` on resolved arguments.
    fn call(
        &mut self,
        name: &str,
        args: Vec<Arc<Expr>>,
        env: &Arc<Env>,
    ) -> Result<Arc<Expr>, AnalysisError> {
        let call = Arc::new(Expr::new(ExprNode::Function(FunctionCall {
            name: name.to_owned(),
            args,
            distinct: false,
            filter: None,
        })));
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
        let find = |a: &Attr| j.out.iter().find(|x| x.col.id == a.col.id).cloned();
        let mut exprs = Vec::new();
        let mut out = Vec::new();
        let mut key_ids = HashSet::new();
        for (la, ra) in keys {
            key_ids.extend([la.col.id, ra.col.id]);
            let (Some(l), Some(r)) = (find(la), find(ra)) else {
                return Err(unsupported("a USING join shape", origin));
            };
            match kind {
                JoinKind::RightOuter => {
                    exprs.push(col_expr(&r.col, Origin::UNKNOWN));
                    out.push(r);
                }
                JoinKind::FullOuter => {
                    let both = vec![
                        col_expr(&l.col, Origin::UNKNOWN),
                        col_expr(&r.col, Origin::UNKNOWN),
                    ];
                    let value = self.call("coalesce", both, env)?;
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

    pub(super) fn sort(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        order: &[SortOrder],
        global: bool,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
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
        let (bounds, below, extended) = if c.agg.is_some() {
            self.over_agg(&c, &exprs, env, "Sort")?
        } else {
            // A name the select list outputs twice is looked up below it,
            // as Spark does.
            let (bounds, wide) = self.bind_missing(&c, &exprs, env, "Sort", true)?;
            let extended = wide.out.len() > c.out.len();
            (bounds, wide, extended)
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
                name: pretty(&b.expr),
                data_type: krishiv_types::DataType::Boolean,
                nullable: true,
            },
            qual: no_qual(),
            source: None,
        },
    }
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
fn qualified(e: &Arc<Expr>, scope: &[Attr]) -> Arc<Expr> {
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
fn lateral_refs(e: &Arc<Expr>, ids: &[AttrId]) -> Arc<Expr> {
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
                            agg_func: format!("\"{}\"", pretty(&lateral_refs(&x, &b.lateral))),
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
            found.push(pretty(&qualified(&x, scope)));
            continue;
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    found.join(", ")
}
