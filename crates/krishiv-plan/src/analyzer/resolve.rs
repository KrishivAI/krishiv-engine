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
    pub(super) aggregates: Vec<Arc<Expr>>,
    /// The grouping attributes, as attributes of the input.
    pub(super) keys: Vec<Attr>,
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
    /// For a projection: its input.
    pub(super) proj: Option<Arc<Rel>>,
}

/// An enclosing query's operator, as a subquery sees it.
#[derive(Clone)]
pub(super) struct Level {
    pub(super) scope: Arc<[Attr]>,
    pub(super) rels: Vec<Rel>,
}

/// A scope environment: the CTEs visible and the enclosing queries.
pub(super) struct Env {
    id: usize,
    pub(super) ctes: Vec<(String, Rel)>,
    pub(super) outer: Vec<Level>,
}

pub(super) struct Resolver<'a> {
    pub(super) ctx: &'a AnalysisContext,
    pub(super) b: &'a Bindings,
    pub(super) next_id: u64,
    next_env: usize,
    memo: HashMap<(usize, usize), Rel>,
    pub(super) columns: Vec<BTreeSet<String>>,
    depth: usize,
    op_outer: Option<usize>,
    root_env: Option<Arc<Env>>,
}

type Resolved = (Arc<LogicalPlan>, Vec<BoundColumn>, Vec<SecurableUse>);

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
        columns: vec![BTreeSet::new(); b.securables.len()],
        depth: 0,
        op_outer: None,
        root_env: None,
    };
    let env = r.env(Vec::new(), Vec::new());
    r.root_env = Some(Arc::clone(&env));
    let rel = r.rel(plan.root(), &env)?;
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
    Ok((rel.plan, output, uses))
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

    pub(super) fn env(&mut self, ctes: Vec<(String, Rel)>, outer: Vec<Level>) -> Arc<Env> {
        self.next_env += 1;
        Arc::new(Env {
            id: self.next_env,
            ctes,
            outer,
        })
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
        let mut stack = vec![(Arc::clone(root), false)];
        while let Some((p, expanded)) = stack.pop() {
            let key = (addr(&*p), env.id);
            if self.memo.contains_key(&key) {
                continue;
            }
            if !expanded {
                stack.push((Arc::clone(&p), true));
                if !matches!(p.node, PlanNode::WithCte { .. }) {
                    let mut kids = Vec::new();
                    p.for_each_child(&mut |c| kids.push(Arc::clone(c)));
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
            self.memo.insert(key, r);
        }
        self.memo
            .get(&(addr(&**root), env.id))
            .cloned()
            .ok_or_else(|| unsupported("a plan shape", root.origin))
    }

    pub(super) fn child(&self, c: &Arc<LogicalPlan>, env: &Env) -> Result<Rel, AnalysisError> {
        self.memo
            .get(&(addr(&**c), env.id))
            .cloned()
            .ok_or_else(|| unsupported("a plan shape", c.origin))
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

    fn wrap(
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
            proj: None,
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
            aliases: &[],
            extra: None,
            agg_scope: None,
            env,
            op,
        }
    }

    fn node(&mut self, p: &Arc<LogicalPlan>, env: &Arc<Env>) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        match &p.node {
            PlanNode::Read(ReadSource::Table { name, options }) => self.read(p, name, options, env),
            PlanNode::Read(ReadSource::Files { .. }) => {
                Err(unsupported("reading files by path", o))
            }
            PlanNode::Sql { .. } => {
                let root = self.b.sql.get(&addr(&**p)).cloned();
                let env0 = self.root_env.clone();
                match (root, env0) {
                    (Some(root), Some(env0)) => self.rel(&root, &env0),
                    _ => Err(unsupported("SQL text", o)),
                }
            }
            PlanNode::Scan(s) => {
                let name = match &s.source {
                    ReadSource::Table { name, .. } => name.clone(),
                    ReadSource::Files { .. } => Vec::new(),
                };
                let source = self.b.scans.get(&name).copied();
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
                let c = self.child(input, env)?;
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
                let node = PlanNode::Limit {
                    input: Arc::clone(&c.plan),
                    limit: b.expr,
                };
                Ok(self.make(o, node, c.out.to_vec(), &[&c]))
            }
            PlanNode::SubqueryAlias {
                input,
                alias,
                columns,
            } => self.subquery_alias(p, input, alias, columns, env),
            PlanNode::WithCte { input, ctes } => {
                let mut visible = env.ctes.clone();
                let mut defs = Vec::new();
                let mut rels = Vec::new();
                for c in ctes {
                    let e = self.env(visible.clone(), env.outer.clone());
                    let r = self.rel(&c.plan, &e)?;
                    defs.push(Cte {
                        name: c.name.clone(),
                        plan: Arc::clone(&r.plan),
                    });
                    visible.push((c.name.clone(), r.clone()));
                    rels.push(r);
                }
                let e = self.env(visible, env.outer.clone());
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
