// SPDX-License-Identifier: Apache-2.0
//! The relational operators of the Resolve pass and their rewrites.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use krishiv_common::error::{
    AmbiguousReference, AssignmentArityMismatch, FailedToParseTooComplex, InvalidWhereCondition,
    MissingAggregation, MissingGroupBy, TableOrViewNotFound, UnresolvedUsingColumnForJoin,
};

use super::bind::{BoundTable, same, unsupported};
use super::catalog::quoted;
use super::exprs::{Bound, names, pretty, semantic_key};
use super::resolve::{AggInfo, Attr, Env, Rel, Resolver};
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
const MAX_ID: u64 = 1 << 62;

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
            proj: None,
        })
    }

    /// A fresh instance of `c`, for the second side of a self-join.
    fn instance(&mut self, c: &Rel) -> Result<Rel, AnalysisError> {
        let (lo, hi) = c.ids.unwrap_or((0, 0));
        let offset = self.next_id - lo;
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
            return self.build_aggregate(p.origin, c, Vec::new(), items, true);
        }
        let out = items.iter().map(|b| out_attr(b, &c.out)).collect();
        let node = PlanNode::Project {
            input: Arc::clone(&c.plan),
            exprs: items.into_iter().map(|b| b.expr).collect(),
        };
        let mut r = self.make(p.origin, node, out, &[c]);
        r.proj = Some(Arc::new(c.clone()));
        Ok(r)
    }

    /// A select list: stars expanded, every item named.
    fn select_items(
        &mut self,
        c: &Rel,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
        op: &'static str,
    ) -> Result<Vec<Bound>, AnalysisError> {
        let rels = [c];
        let mut items = Vec::new();
        let cx = Self::cx(&c.out, &rels, env, op);
        for e in exprs {
            if let ExprNode::Star(s) = &e.node {
                for a in self.expand(s, &cx, e.origin)? {
                    items.push(column_bound(&a, e.origin));
                }
                continue;
            }
            let b = self.bind(e, &cx)?;
            items.push(self.named(b)?);
        }
        Ok(items)
    }

    /// `b` as a named expression: an attribute, or aliased by Spark's
    /// generated name (a cast of a column keeps the column's name).
    fn named(&mut self, b: Bound) -> Result<Bound, AnalysisError> {
        let name = match &b.expr.node {
            ExprNode::Resolved(ResolvedExpr::Column(_) | ResolvedExpr::Alias { .. }) => {
                return Ok(b);
            }
            ExprNode::Cast { expr, .. } => match &expr.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) => c.name.clone(),
                _ => pretty(&b.expr),
            },
            _ => pretty(&b.expr),
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
            let f = self.make(p.origin, node, agg.out.to_vec(), &[&agg]);
            return Ok(if extended {
                self.restore(&f, &c.out)
            } else {
                f
            });
        }
        let rels = [&c];
        let cx = Self::cx(&c.out, &rels, env, "Filter");
        let b = self.bind(predicate, &cx)?;
        if b.agg.is_some() {
            return Err(AnalysisError::new(
                InvalidWhereCondition {
                    condition: format!("\"{}\"", pretty(&b.expr)),
                    expression_list: aggregate_sql(&b.expr, &c.out),
                },
                p.origin,
            ));
        }
        let node = PlanNode::Filter {
            input: Arc::clone(&c.plan),
            predicate: b.expr,
        };
        Ok(self.make(p.origin, node, c.out.to_vec(), &[&c]))
    }

    /// A projection of `out` over `r`: drops what a rewrite added.
    fn restore(&self, r: &Rel, out: &[Attr]) -> Rel {
        let exprs = out
            .iter()
            .map(|a| col_expr(&a.col, Origin::UNKNOWN))
            .collect();
        let node = PlanNode::Project {
            input: Arc::clone(&r.plan),
            exprs,
        };
        self.make(Origin::UNKNOWN, node, out.to_vec(), &[r])
    }

    /// Expressions over an aggregate (`HAVING`, `ORDER BY`): names resolve
    /// against its output, then its grouping attributes; aggregate-function
    /// arguments against its input. Aggregate calls the aggregate does not
    /// compute yet, and grouping attributes it does not output, are added
    /// to it; the flag says whether it grew.
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
            .zip(agg.out.iter())
            .map(|(e, a)| (semantic_key(e, &mut memo), a.clone()))
            .collect();
        let out_ids: HashSet<AttrId> = agg.out.iter().map(|a| a.col.id).collect();
        let mut added: Vec<(Arc<Expr>, Attr)> = Vec::new();
        for b in &mut bounds {
            b.expr = Arc::clone(&b.expr).transform_down(&mut |x: Arc<Expr>| match &x.node {
                ExprNode::Resolved(ResolvedExpr::Function(f))
                    if f.kind == FunctionKind::Aggregate =>
                {
                    let k = semantic_key(&x, &mut memo);
                    if let Some((_, a)) = known.iter().find(|(kk, _)| *kk == k) {
                        return Ok(col_expr(&a.col, x.origin));
                    }
                    let ty = ArgType {
                        data_type: Some(f.data_type.clone()),
                        nullable: f.nullable,
                    };
                    let output = self.attr(&pretty(&x), &ty, x.origin)?;
                    let alias = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                        expr: Arc::clone(&x),
                        output: output.clone(),
                    })));
                    let a = Attr {
                        col: output,
                        qual: no_qual(),
                        source: None,
                    };
                    known.push((k, a.clone()));
                    added.push((alias, a.clone()));
                    Ok(col_expr(&a.col, x.origin))
                }
                ExprNode::Resolved(ResolvedExpr::Column(c))
                    if !out_ids.contains(&c.id) && !added.iter().any(|(_, a)| a.col.id == c.id) =>
                {
                    if let Some(k) = info.keys.iter().find(|k| k.col.id == c.id) {
                        added.push((col_expr(&k.col, Origin::UNKNOWN), k.clone()));
                    }
                    Ok(x)
                }
                _ => Ok(x),
            })?;
        }
        if added.is_empty() {
            return Ok((bounds, agg.clone(), false));
        }
        let mut aggregates = info.aggregates.clone();
        let mut out = agg.out.to_vec();
        for (e, a) in added {
            aggregates.push(e);
            out.push(a);
        }
        let grown = AggInfo {
            aggregates,
            ..(*info).clone()
        };
        let r = self.aggregate_rel(agg.plan.origin, grown, out);
        Ok((bounds, r, true))
    }

    fn aggregate_rel(&self, origin: Origin, info: AggInfo, out: Vec<Attr>) -> Rel {
        let node = PlanNode::Aggregate {
            input: Arc::clone(&info.input.plan),
            grouping: Grouping::GroupBy(info.grouping.clone()),
            aggregates: info.aggregates.clone(),
        };
        let mut r = self.make(origin, node, out, &[&info.input]);
        r.agg = Some(Arc::new(info));
        r
    }

    fn finish_aggregate(
        &self,
        origin: Origin,
        c: &Rel,
        grouping: Vec<Arc<Expr>>,
        items: Vec<Bound>,
        keys: Vec<Attr>,
    ) -> Rel {
        let out = items.iter().map(|b| out_attr(b, &c.out)).collect();
        let info = AggInfo {
            input: c.clone(),
            grouping,
            aggregates: items.into_iter().map(|b| b.expr).collect(),
            keys,
        };
        self.aggregate_rel(origin, info, out)
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
        let rels = [&c];
        let aliases: Vec<(String, Arc<Expr>)> = aggregates
            .iter()
            .filter(|_| conf.group_by_aliases)
            .filter_map(|e| match &e.node {
                ExprNode::Alias { expr, name } => Some((name.clone(), Arc::clone(expr))),
                _ => None,
            })
            .collect();
        let mut group = Vec::new();
        for k in keys {
            let b = match &k.node {
                ExprNode::Literal(Literal::Integer(_)) if conf.group_by_ordinal => {
                    return Err(unsupported("GROUP BY ordinals", k.origin));
                }
                _ => {
                    let mut cx = Self::cx(&c.out, &rels, env, "Aggregate");
                    cx.aliases = &aliases;
                    self.bind(k, &cx)?
                }
            };
            group.push(b);
        }
        let items = self.select_items(&c, aggregates, env, "Aggregate")?;
        self.build_aggregate(p.origin, &c, group, items, false)
    }

    /// An aggregate of `items` grouped by `group`, validated.
    fn build_aggregate(
        &mut self,
        origin: Origin,
        c: &Rel,
        group: Vec<Bound>,
        items: Vec<Bound>,
        global: bool,
    ) -> Result<Rel, AnalysisError> {
        let mut memo = HashMap::new();
        let gkeys: HashSet<u128> = group
            .iter()
            .map(|b| semantic_key(&b.expr, &mut memo))
            .collect();
        self.check_aggregation(&items, &gkeys, &mut memo, origin, global)?;
        let key_attrs = group
            .iter()
            .filter_map(|b| match &b.expr.node {
                ExprNode::Resolved(ResolvedExpr::Column(col)) => {
                    c.out.iter().find(|a| a.col.id == col.id).cloned()
                }
                _ => None,
            })
            .collect();
        let grouping = group.into_iter().map(|b| b.expr).collect();
        Ok(self.finish_aggregate(origin, c, grouping, items, key_attrs))
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
                ExprNode::Literal(Literal::Integer(_)) if self.ctx.conf.order_by_ordinal => {
                    return Err(unsupported("ORDER BY ordinals", o.expr.origin));
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
        if c.agg.is_some() {
            let (bounds, agg, extended) = self.over_agg(&c, &exprs, env, "Sort")?;
            let node = PlanNode::Sort {
                input: Arc::clone(&agg.plan),
                order: sorted(bounds.into_iter().map(|b| b.expr).collect()),
                global,
            };
            let s = self.make(p.origin, node, agg.out.to_vec(), &[&agg]);
            return Ok(if extended {
                self.restore(&s, &c.out)
            } else {
                s
            });
        }
        let rels = [&c];
        let mut cx = Self::cx(&c.out, &rels, env, "Sort");
        let below = c.proj.as_ref().map(|i| Arc::clone(&i.out));
        cx.extra = below.as_deref();
        let mut bound = Vec::new();
        for e in &exprs {
            bound.push(self.bind(e, &cx)?.expr);
        }
        // `ORDER BY` columns the select list dropped (Spark projects them
        // below the sort): not yet.
        let have: HashSet<AttrId> = c.out.iter().map(|a| a.col.id).collect();
        let mut seen = HashSet::new();
        for e in &bound {
            let mut stack = vec![Arc::clone(e)];
            while let Some(x) = stack.pop() {
                if !seen.insert(super::bind::addr(&*x)) {
                    continue;
                }
                if let ExprNode::Resolved(ResolvedExpr::Column(col)) = &x.node
                    && !have.contains(&col.id)
                    && below
                        .iter()
                        .flat_map(|b| b.iter())
                        .any(|a| a.col.id == col.id)
                {
                    return Err(unsupported(
                        "ORDER BY columns outside the select list",
                        x.origin,
                    ));
                }
                x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
            }
        }
        let node = PlanNode::Sort {
            input: Arc::clone(&c.plan),
            order: sorted(bound),
            global,
        };
        Ok(self.make(p.origin, node, c.out.to_vec(), &[&c]))
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
            let qualified = Arc::clone(&x).transform_up(&mut |n: Arc<Expr>| {
                Ok::<_, std::convert::Infallible>(match &n.node {
                    ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                        let mut parts: Vec<String> = scope
                            .iter()
                            .find(|a| a.col.id == c.id)
                            .map(|a| a.qual.to_vec())
                            .unwrap_or_default();
                        parts.push(c.name.clone());
                        col_expr(
                            &BoundColumn {
                                name: parts.join("."),
                                ..c.clone()
                            },
                            n.origin,
                        )
                    }
                    _ => n,
                })
            });
            found.push(qualified.map_or_else(|_| String::new(), |q| pretty(&q)));
            continue;
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    found.join(", ")
}
