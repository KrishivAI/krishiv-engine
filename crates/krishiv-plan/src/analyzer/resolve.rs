// SPDX-License-Identifier: Apache-2.0
//! Stage 2, Resolve: one bottom-up pass over the relations.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use krishiv_common::error::FailedToParseTooComplex;
use krishiv_common::stack::{STEP, on_big_stack};

use super::bind::{Bindings, addr, same, unsupported};
use super::exprs::{Bound, Cx};
use super::ops::col_expr;
use super::{AnalysisContext, AnalysisError, SecurableUse};
use crate::{
    AttrId, BoundColumn, Cte, Expr, ExprNode, LogicalPlan, Origin, Plan, PlanId, PlanNode,
    ReadSource,
};

/// An attribute in scope: the column, its qualifier and, for a table
/// column, the securable it belongs to and its catalog name.
#[derive(Debug, Clone)]
pub(super) struct Attr {
    pub(super) col: BoundColumn,
    pub(super) qual: Arc<[String]>,
    pub(super) source: Option<(usize, Arc<str>)>,
}

/// What an aggregate needs to resolve `HAVING` / `ORDER BY` over it.
#[derive(Clone)]
pub(super) struct AggInfo {
    pub(super) input: Rel,
    pub(super) grouping: Vec<Arc<Expr>>,
    /// The aggregate's own expressions and output, aligned.
    pub(super) aggregates: Vec<Arc<Expr>>,
    pub(super) out: Vec<Attr>,
    /// The grouping attributes, as attributes of the input.
    pub(super) keys: Vec<Attr>,
    /// Select items that use lateral column aliases make projections
    /// above the aggregate: every select item, over the aggregate's output
    /// (`None` when no item does).
    pub(super) top: Option<Arc<[Bound]>>,
}

/// How a relation gets back columns it dropped: Spark adds the attributes
/// an `ORDER BY` or a DataFrame filter names, but a projection below
/// dropped, to that projection (`ResolveMissingReferences`).
pub(super) enum Widen {
    /// A projection of `exprs` over `input`.
    Project { input: Rel, exprs: Vec<Arc<Expr>> },
    /// A filter, sort or limit over `child`: its columns pass through.
    Through { child: Rel },
    /// An aggregate (under the projections of its lateral aliases): it
    /// adds grouping attributes to its output.
    Aggregate(Arc<AggInfo>),
}

/// A resolved relation and its output scope.
#[derive(Clone)]
pub(super) struct Rel {
    pub(super) plan: Arc<LogicalPlan>,
    pub(super) out: Arc<[Attr]>,
    /// The span of attribute ids defined in the subtree.
    pub(super) ids: Option<(u64, u64)>,
    /// The outermost query level the subtree references, if any.
    pub(super) outer: Option<usize>,
    /// The outputs of the relations carrying each `plan_id` below.
    pub(super) carriers: Arc<BTreeMap<PlanId, Vec<Arc<[Attr]>>>>,
    pub(super) agg: Option<Arc<AggInfo>>,
    /// Where dropped columns come back from, if anywhere.
    pub(super) widen: Option<Arc<Widen>>,
}

/// An enclosing query's operator, as a subquery sees it.
#[derive(Clone)]
pub(super) struct Level {
    pub(super) scope: Arc<[Attr]>,
}

/// A scope environment: the CTEs visible and the enclosing queries.
pub(super) struct Env {
    id: usize,
    pub(super) ctes: Vec<(String, Rel)>,
    /// Identifies `ctes`: environments with one CTE scope resolve a
    /// relation without outer references alike.
    pub(super) ctes_id: usize,
    pub(super) outer: Vec<Level>,
    /// The span of the ids the enclosing queries define.
    pub(super) outer_ids: Option<(u64, u64)>,
}

pub(super) struct Resolver<'a> {
    pub(super) ctx: &'a AnalysisContext,
    pub(super) b: &'a Bindings,
    pub(super) next_id: u64,
    next_env: usize,
    memo: HashMap<(usize, usize), Rel>,
    /// Relations without outer references, by (relation, CTE scope): they
    /// resolve alike in every environment with that CTE scope, so a plan
    /// reached both as a child and inside a subquery is resolved once.
    closed: HashMap<(usize, usize), Rel>,
    /// Each relation's representative among the relations equal to it
    /// (`encode::identity_keys`): resolution does not depend on how the
    /// input shares subtrees, so equal input keys give equal plans.
    canon: HashMap<usize, Arc<LogicalPlan>>,
    pub(super) columns: Vec<BTreeSet<String>>,
    depth: usize,
    op_outer: Option<usize>,
    root_env: Option<Arc<Env>>,
}

type Resolved = (
    Arc<LogicalPlan>,
    Vec<BoundColumn>,
    Vec<SecurableUse>,
    Vec<String>,
);

pub(super) fn resolve(
    plan: &Plan,
    ctx: &AnalysisContext,
    b: &Bindings,
) -> Result<Resolved, AnalysisError> {
    let mut r = Resolver {
        ctx,
        b,
        next_id: 0,
        next_env: 0,
        memo: HashMap::new(),
        closed: HashMap::new(),
        canon: HashMap::new(),
        columns: vec![BTreeSet::new(); b.securables.len()],
        depth: 0,
        op_outer: None,
        root_env: None,
    };
    // Fresh ids start past every id the input already holds (a plan that
    // is resolved in part, or re-analysed).
    r.next_id = first_free_id(plan.root())?;
    let env = r.env(Vec::new(), 0, Vec::new());
    r.root_env = Some(Arc::clone(&env));
    let rel = r.rel(plan.root(), &env)?;
    // Hints name relations of their resolved subtree.
    let (root, warnings) = super::hints::resolve(&rel.plan, ctx.conf.case_sensitive)?;
    // Subqueries to joins, after everything resolved (P2-T071c).
    let root = super::subquery::rewrite(&root, &mut r.next_id)?;
    let output = rel.out.iter().map(|a| a.col.clone()).collect();
    let uses = b
        .securables
        .iter()
        .zip(r.columns)
        .map(|((securable, privilege), columns)| SecurableUse {
            securable: securable.clone(),
            privilege: *privilege,
            columns,
        })
        .collect();
    Ok((root, output, uses, warnings))
}

/// One past the largest attribute id in `root` (a `Shift` counted at its
/// offset).
fn first_free_id(root: &LogicalPlan) -> Result<u64, AnalysisError> {
    use crate::walk::{Node, visit_distinct};
    use crate::{ExprNode, ResolvedExpr};
    let mut top = 0_u64;
    let mut shifts = 0_u64;
    let _ = visit_distinct(Node::Plan(root), |node| {
        let mut bump = |c: &BoundColumn| top = top.max(c.id.0.saturating_add(1));
        match node {
            Node::Plan(p) => match &p.node {
                PlanNode::Scan(s) => s.output.iter().for_each(&mut bump),
                PlanNode::CteRef(r) => r.output.iter().for_each(&mut bump),
                PlanNode::Shift { offset, .. } => shifts = shifts.saturating_add(*offset),
                _ => {}
            },
            Node::Expr(e) => {
                if let ExprNode::Resolved(
                    ResolvedExpr::Column(c)
                    | ResolvedExpr::OuterColumn(c)
                    | ResolvedExpr::Alias { output: c, .. },
                ) = &e.node
                {
                    bump(c);
                }
            }
        }
        std::ops::ControlFlow::Continue(())
    });
    let next = top.saturating_add(shifts);
    if next > super::ops::MAX_ID {
        return Err(AnalysisError::new(FailedToParseTooComplex {}, root.origin));
    }
    Ok(next)
}

fn hull(a: Option<(u64, u64)>, b: Option<(u64, u64)>) -> Option<(u64, u64)> {
    match (a, b) {
        (Some(x), Some(y)) => Some((x.0.min(y.0), x.1.max(y.1))),
        (x, None) | (None, x) => x,
    }
}

impl Resolver<'_> {
    pub(super) fn fresh(&mut self) -> AttrId {
        self.next_id += 1;
        AttrId(self.next_id - 1)
    }

    pub(super) fn env(
        &mut self,
        ctes: Vec<(String, Rel)>,
        ctes_id: usize,
        outer: Vec<Level>,
    ) -> Arc<Env> {
        self.next_env += 1;
        let outer_ids = outer
            .iter()
            .flat_map(|l| l.scope.iter())
            .fold(None, |h, a| hull(h, Some((a.col.id.0, a.col.id.0))));
        Arc::new(Env {
            id: self.next_env,
            ctes,
            ctes_id,
            outer,
            outer_ids,
        })
    }

    /// A new CTE scope id.
    pub(super) fn ctes_id(&mut self) -> usize {
        self.next_env += 1;
        self.next_env
    }

    /// Records the representative of every relation under `root`.
    fn canonicalize(&mut self, root: &Arc<LogicalPlan>) {
        if self.canon.contains_key(&addr(&**root)) {
            return;
        }
        let keys = crate::encode::identity_keys(root);
        let mut first: HashMap<u128, Arc<LogicalPlan>> = HashMap::new();
        let mut stack = vec![Arc::clone(root)];
        let mut exprs: Vec<Arc<crate::Expr>> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        while let Some(p) = stack.pop() {
            if !seen.insert(addr(&*p)) {
                continue;
            }
            let key = keys.get(&std::ptr::from_ref(&*p)).copied();
            let rep = match key {
                Some(k) => Arc::clone(first.entry(k).or_insert_with(|| Arc::clone(&p))),
                None => Arc::clone(&p),
            };
            self.canon.insert(addr(&*p), rep);
            p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
            p.for_each_expr(&mut |e| exprs.push(Arc::clone(e)));
            while let Some(e) = exprs.pop() {
                if !seen.insert(addr(&*e)) {
                    continue;
                }
                if let crate::ExprNode::Subquery { plan, .. } = &e.node {
                    stack.push(Arc::clone(plan));
                }
                e.for_each_child(&mut |c| exprs.push(Arc::clone(c)));
            }
        }
    }

    fn rep(&self, p: &Arc<LogicalPlan>) -> Arc<LogicalPlan> {
        self.canon
            .get(&addr(&**p))
            .cloned()
            .unwrap_or_else(|| Arc::clone(p))
    }

    /// Resolves the relation `root` in `env`, every `STEP` nested calls on
    /// a fresh stack.
    pub(super) fn rel(
        &mut self,
        root: &Arc<LogicalPlan>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        self.depth += 1;
        let r = if self.depth.is_multiple_of(STEP) {
            on_big_stack(|| self.rel_on_stack(root, env))
                .unwrap_or_else(|| Err(AnalysisError::new(FailedToParseTooComplex {}, root.origin)))
        } else {
            self.rel_on_stack(root, env)
        };
        self.depth -= 1;
        r
    }

    /// Children first, with an explicit stack; one resolution per
    /// (relation, environment).
    fn rel_on_stack(
        &mut self,
        root: &Arc<LogicalPlan>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        self.canonicalize(root);
        let root = self.rep(root);
        let mut stack = vec![(Arc::clone(&root), false)];
        while let Some((p, expanded)) = stack.pop() {
            let key = (addr(&*p), env.id);
            if self.memo.contains_key(&key) {
                continue;
            }
            if !expanded {
                if let Some(r) = self.closed.get(&(key.0, env.ctes_id)).cloned() {
                    // Resolved before in another environment: the same
                    // relation, as a fresh instance where its ids would
                    // meet an enclosing query's.
                    let overlaps = matches!((r.ids, env.outer_ids),
                        (Some(a), Some(b)) if a.0 <= b.1 && b.0 <= a.1);
                    let r = if overlaps { self.instance(&r)? } else { r };
                    self.memo.insert(key, r);
                    continue;
                }
                stack.push((Arc::clone(&p), true));
                if !matches!(p.node, PlanNode::WithCte { .. }) {
                    let mut kids = Vec::new();
                    p.for_each_child(&mut |c| kids.push(self.rep(c)));
                    if matches!(p.node, PlanNode::Shift { .. }) {
                        // A shift reads its input's own resolution (see
                        // `shift_input`): resolved before, nothing to do.
                        kids.retain(|c| !self.closed.contains_key(&(addr(&**c), env.ctes_id)));
                    }
                    stack.extend(kids.into_iter().rev().map(|c| (c, false)));
                }
                continue;
            }
            let (lo, saved) = (self.next_id, self.op_outer.take());
            let mut r = self.node(&p, env)?;
            if self.next_id > lo {
                r.ids = hull(r.ids, Some((lo, self.next_id - 1)));
            }
            r.outer = [r.outer, self.op_outer].into_iter().flatten().min();
            self.op_outer = saved;
            if r.outer.is_none() {
                self.closed.insert((key.0, env.ctes_id), r.clone());
            }
            self.memo.insert(key, r);
        }
        self.memo
            .get(&(addr(&*root), env.id))
            .cloned()
            .ok_or_else(|| unsupported("a plan shape", root.origin))
    }

    pub(super) fn child(&self, c: &Arc<LogicalPlan>, env: &Env) -> Result<Rel, AnalysisError> {
        self.memo
            .get(&(addr(&*self.rep(c)), env.id))
            .cloned()
            .ok_or_else(|| unsupported("a plan shape", c.origin))
    }

    /// The input of a `Shift`, resolved as itself: the shift is the fresh
    /// instance, so it stands over the input's own resolution (the ids the
    /// input states, which it reads as `id + offset`), never over an
    /// instance made because those ids meet an enclosing query's. The
    /// input has no outer references, so its resolution is closed and
    /// shared by every environment with this CTE scope.
    fn shift_input(&self, input: &Arc<LogicalPlan>, env: &Env) -> Result<Rel, AnalysisError> {
        let rep = self.rep(input);
        match self.closed.get(&(addr(&*rep), env.ctes_id)) {
            Some(r) => Ok(r.clone()),
            None => {
                let r = self.child(&rep, env)?;
                if r.outer.is_some() {
                    return Err(unsupported("a shift over a correlated plan", input.origin));
                }
                Ok(r)
            }
        }
    }

    /// A relation over `kids` with output `out`, at `origin` (whose
    /// `plan_id` it carries).
    pub(super) fn make(
        &self,
        origin: Origin,
        node: PlanNode,
        out: Vec<Attr>,
        kids: &[&Rel],
    ) -> Rel {
        let plan = Arc::new(LogicalPlan::new(node).with_origin(origin));
        self.wrap(plan, origin.plan_id, out, kids)
    }

    pub(super) fn wrap(
        &self,
        plan: Arc<LogicalPlan>,
        id: Option<PlanId>,
        out: Vec<Attr>,
        kids: &[&Rel],
    ) -> Rel {
        let out: Arc<[Attr]> = out.into();
        let own = out
            .iter()
            .fold(None, |h, a| hull(h, Some((a.col.id.0, a.col.id.0))));
        let ids = kids.iter().fold(own, |h, k| hull(h, k.ids));
        let mut carriers = match kids {
            [k] => Arc::clone(&k.carriers),
            _ => {
                let mut m: BTreeMap<PlanId, Vec<Arc<[Attr]>>> = BTreeMap::new();
                for k in kids {
                    for (id, v) in k.carriers.iter() {
                        m.entry(*id).or_default().extend(v.iter().cloned());
                    }
                }
                Arc::new(m)
            }
        };
        if let Some(id) = id {
            Arc::make_mut(&mut carriers).insert(id, vec![Arc::clone(&out)]);
        }
        Rel {
            plan,
            out,
            ids,
            outer: kids.iter().filter_map(|k| k.outer).min(),
            carriers,
            agg: None,
            widen: None,
        }
    }

    /// Binds a top-level expression of the current operator.
    pub(super) fn bind(&mut self, e: &Arc<Expr>, cx: &Cx<'_>) -> Result<Bound, AnalysisError> {
        let b = self.expr(e, cx)?;
        self.op_outer = [self.op_outer, b.outer].into_iter().flatten().min();
        Ok(b)
    }

    pub(super) fn cx<'s>(
        scope: &'s [Attr],
        rels: &'s [&'s Rel],
        env: &'s Arc<Env>,
        op: &'static str,
    ) -> Cx<'s> {
        Cx {
            scope,
            rels,
            lateral: None,
            aliases: &[],
            extra: None,
            agg_scope: None,
            lenient: false,
            outer: true,
            env,
            op,
        }
    }

    fn node(&mut self, p: &Arc<LogicalPlan>, env: &Arc<Env>) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        match &p.node {
            PlanNode::Read(ReadSource::Table { name, options }) => self.read(p, name, options, env),
            PlanNode::Read(source @ ReadSource::Files { .. }) => {
                match self.b.files.iter().position(|f| f.source == *source) {
                    Some(i) => Ok(self.files(p, i)),
                    None => Err(unsupported("reading files by path", o)),
                }
            }
            PlanNode::Sql { .. } => {
                let root = self.b.sql.get(&addr(&**p)).cloned();
                let env0 = self.root_env.clone();
                match (root, env0) {
                    (Some(root), Some(env0)) => self.rel(&root, &env0),
                    _ => Err(unsupported("SQL text", o)),
                }
            }
            PlanNode::OneRowRelation => Ok(self.wrap(Arc::clone(p), o.plan_id, Vec::new(), &[])),
            PlanNode::Scan(s) => {
                let name = match &s.source {
                    ReadSource::Table { name, .. } => name.clone(),
                    ReadSource::Files { .. } => Vec::new(),
                };
                let source = self.b.scans.get(&name).copied();
                // Bind checked every scan of a table; one it did not bind
                // never reaches a plan.
                if matches!(s.source, ReadSource::Table { .. }) && source.is_none() {
                    return Err(unsupported("a scan of an unchecked table", o));
                }
                let qual: Arc<[String]> = name.into();
                let out = s
                    .output
                    .iter()
                    .map(|c| Attr {
                        col: c.clone(),
                        qual: Arc::clone(&qual),
                        source: source.map(|i| (i, Arc::from(c.name.as_str()))),
                    })
                    .collect();
                Ok(self.wrap(Arc::clone(p), o.plan_id, out, &[]))
            }
            PlanNode::CteRef(r) => {
                let qual: Arc<[String]> = vec![r.name.clone()].into();
                let out = r
                    .output
                    .iter()
                    .map(|c| Attr {
                        col: c.clone(),
                        qual: Arc::clone(&qual),
                        source: None,
                    })
                    .collect();
                Ok(self.wrap(Arc::clone(p), o.plan_id, out, &[]))
            }
            PlanNode::Shift { input, offset } => {
                let c = self.shift_input(input, env)?;
                // The shifted node keeps its input: it may stand for the
                // resolved plan only when resolving left the input as it
                // was, with the attributes it states.
                if !Arc::ptr_eq(&c.plan, input) {
                    let same = input.is_resolved()
                        && input.resolved_output().is_some_and(|out| {
                            out.iter().map(|b| b.id).eq(c.out.iter().map(|a| a.col.id))
                        });
                    if !same {
                        return Err(unsupported("a shift over an unresolved plan", o));
                    }
                }
                self.shifted(&c, *offset, Some(Arc::clone(p)))
            }
            PlanNode::Project { input, exprs } => self.project(p, input, exprs, env),
            PlanNode::WithColumns { input, columns } => {
                let c = self.child(input, env)?;
                let rels = [&c];
                let cx = Self::cx(&c.out, &rels, env, "Project");
                let mut news: Vec<(String, Bound)> = Vec::new();
                for (name, e) in columns {
                    news.push((name.clone(), self.bind(e, &cx)?));
                }
                let cs = self.ctx.conf.case_sensitive;
                let mut exprs = Vec::new();
                let mut taken = HashSet::new();
                for a in c.out.iter() {
                    match news.iter().rposition(|(n, _)| same(n, &a.col.name, cs)) {
                        Some(i) => {
                            taken.insert(i);
                            exprs.push(Arc::new(Expr::new(ExprNode::Alias {
                                expr: Arc::clone(&news[i].1.expr),
                                name: news[i].0.clone(),
                            })));
                        }
                        None => exprs.push(col_expr(&a.col, Origin::UNKNOWN)),
                    }
                }
                for (i, (name, b)) in news.iter().enumerate() {
                    let later = news[i + 1..].iter().any(|(n, _)| same(n, name, cs));
                    if !taken.contains(&i) && !later {
                        exprs.push(Arc::new(Expr::new(ExprNode::Alias {
                            expr: Arc::clone(&b.expr),
                            name: name.clone(),
                        })));
                    }
                }
                self.project_items(p, &c, &exprs, env)
            }
            PlanNode::Filter { input, predicate } => self.filter(p, input, predicate, env),
            PlanNode::Aggregate {
                input,
                grouping,
                aggregates,
            } => self.aggregate(p, input, grouping, aggregates, env),
            PlanNode::Join {
                left,
                right,
                kind,
                condition,
            } => self.join(p, left, right, *kind, condition, env),
            PlanNode::Sort {
                input,
                order,
                global,
            } => self.sort(p, input, order, *global, env),
            PlanNode::Limit { input, limit } => {
                let c = self.child(input, env)?;
                let cx = Self::cx(&[], &[], env, "Limit");
                let b = self.bind(limit, &cx)?;
                if b.ty.data_type != Some(krishiv_types::DataType::Integer) {
                    return Err(AnalysisError::new(
                        krishiv_common::error::InvalidLimitLikeExpressionDataType {
                            expr: format!("\"{}\"", super::exprs::pretty(&b.shown())),
                            name: "limit".to_owned(),
                            data_type: super::coerce::quoted_type(b.ty.data_type.as_ref()),
                        },
                        limit.origin,
                    ));
                }
                let node = PlanNode::Limit {
                    input: Arc::clone(&c.plan),
                    limit: b.expr,
                };
                let mut r = self.make(o, node, c.out.to_vec(), &[&c]);
                r.widen = Some(Arc::new(Widen::Through { child: c }));
                Ok(r)
            }
            PlanNode::SubqueryAlias {
                input,
                alias,
                columns,
            } => self.subquery_alias(p, input, alias, columns, env),
            PlanNode::Union {
                inputs,
                by_name,
                allow_missing_columns,
            } => self.union(p, inputs, *by_name, *allow_missing_columns, env),
            PlanNode::SetOp {
                left,
                right,
                op,
                all,
            } => self.set_op(p, left, right, *op, *all, env),
            PlanNode::ReplicateRows { input, count } => {
                let c = self.child(input, env)?;
                let rels = [&c];
                let b = self.bind(count, &Self::cx(&c.out, &rels, env, "Generate"))?;
                let node = PlanNode::ReplicateRows {
                    input: Arc::clone(&c.plan),
                    count: b.expr,
                };
                Ok(self.make(o, node, c.out.to_vec(), &[&c]))
            }
            PlanNode::Distinct { input } => self.distinct(p, input, env),
            PlanNode::Hint {
                input,
                name,
                params,
            } => self.hint(p, input, name, params, env),
            PlanNode::JoinHint { input, strategy } => {
                let c = self.child(input, env)?;
                let node = PlanNode::JoinHint {
                    input: Arc::clone(&c.plan),
                    strategy: *strategy,
                };
                let mut r = self.make(o, node, c.out.to_vec(), &[&c]);
                r.widen = Some(Arc::new(Widen::Through { child: c }));
                Ok(r)
            }
            PlanNode::WithCte { input, ctes } => {
                let mut visible = env.ctes.clone();
                let mut scope = env.ctes_id;
                let mut defs = Vec::new();
                let mut rels = Vec::new();
                for c in ctes {
                    let e = self.env(visible.clone(), scope, env.outer.clone());
                    let r = self.rel(&c.plan, &e)?;
                    defs.push(Cte {
                        name: c.name.clone(),
                        plan: Arc::clone(&r.plan),
                    });
                    visible.push((c.name.clone(), r.clone()));
                    scope = self.ctes_id();
                    rels.push(r);
                }
                let e = self.env(visible, scope, env.outer.clone());
                let i = self.rel(input, &e)?;
                let node = PlanNode::WithCte {
                    input: Arc::clone(&i.plan),
                    ctes: defs,
                };
                let mut kids: Vec<&Rel> = rels.iter().collect();
                kids.push(&i);
                let mut r = self.make(o, node, i.out.to_vec(), &kids);
                r.carriers = Arc::clone(&i.carriers);
                Ok(r)
            }
        }
    }
}
