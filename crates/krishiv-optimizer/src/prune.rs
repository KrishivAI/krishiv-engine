// SPDX-License-Identifier: Apache-2.0
//! Column pruning (projection pushdown).
//!
//! One pass over the plan as a DAG, subquery plans included: parents before
//! children, each relation learns which of its output attributes any of
//! its consumers reads (the union over all consumers of a shared subtree);
//! then, children first, scans keep only those columns, projections and
//! aggregates drop unused outputs, and join and aggregate inputs that carry
//! extra columns get a projection. The root's output never changes; CTE
//! definitions and scalar and `IN` subquery plans keep their whole output.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use krishiv_plan::{
    BoundColumn, Expr, ExprNode, Grouping, JoinCondition, LogicalPlan, PlanNode, Scan, SubqueryKind,
};

use crate::expr::{Attrs, column, output_of, references_into};

/// `root` with unused columns pruned.
pub(crate) fn prune(root: &Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    let graph = Graph::new(root);
    let req = graph.requirements();
    let mut pass = Rewrite {
        graph: &graph,
        req: &req,
        done: HashMap::new(),
    };
    pass.node(0)
}

/// An edge to a child relation, with what the parent reads of it (`None`:
/// everything).
type Demand = (usize, Option<Attrs>);

struct Graph {
    nodes: Vec<Arc<LogicalPlan>>,
    index: HashMap<*const LogicalPlan, usize>,
}

impl Graph {
    fn new(root: &Arc<LogicalPlan>) -> Self {
        let mut g = Self {
            nodes: Vec::new(),
            index: HashMap::new(),
        };
        let mut stack = vec![Arc::clone(root)];
        while let Some(p) = stack.pop() {
            if g.index.contains_key(&Arc::as_ptr(&p)) {
                continue;
            }
            g.index.insert(Arc::as_ptr(&p), g.nodes.len());
            p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
            subqueries(&p, &mut |_, s| stack.push(Arc::clone(s)));
            g.nodes.push(p);
        }
        g
    }

    fn id(&self, p: &Arc<LogicalPlan>) -> usize {
        self.index.get(&Arc::as_ptr(p)).copied().unwrap_or(0)
    }

    /// Every edge of node `i` with its demand, given what is read of `i`.
    fn demands(&self, i: usize, req: &Attrs) -> Vec<Demand> {
        let p = &self.nodes[i];
        let mut out = Vec::new();
        let mut child = |c: &Arc<LogicalPlan>, d: Option<Attrs>| out.push((self.id(c), d));
        match &p.node {
            PlanNode::Project { input, exprs } => child(input, Some(refs(&keep(exprs, req)))),
            PlanNode::Aggregate {
                input, aggregates, ..
            } => {
                let mut need = refs(&keep(aggregates, req));
                grouping_refs(p, &mut need);
                child(input, Some(need));
            }
            PlanNode::SubqueryAlias { input, columns, .. } => {
                // A renaming alias keeps its input whole.
                child(input, columns.is_empty().then(|| req.clone()));
            }
            PlanNode::Filter { input, .. }
            | PlanNode::Sort { input, .. }
            | PlanNode::Limit { input, .. }
            | PlanNode::Offset { input, .. }
            | PlanNode::JoinHint { input, .. }
            | PlanNode::ReplicateRows { input, .. } => {
                let mut need = req.clone();
                p.for_each_expr(&mut |e| references_into(e, &mut need));
                child(input, Some(need));
            }
            PlanNode::Join { left, right, .. } => {
                let need = join_need(p, req);
                child(left, Some(need.clone()));
                child(right, Some(need));
            }
            PlanNode::WithCte { input, ctes } => {
                ctes.iter().for_each(|c| child(&c.plan, None));
                child(input, Some(req.clone()));
            }
            PlanNode::Shift { input, offset } => {
                let need = req.iter().filter_map(|id| id.0.checked_sub(*offset));
                child(input, Some(need.map(krishiv_plan::AttrId).collect()));
            }
            // Inputs match by position: each is read whole.
            PlanNode::Union { inputs, .. } => inputs.iter().for_each(|i| child(i, None)),
            _ => {}
        }
        subqueries(p, &mut |kind, s| {
            // `EXISTS` reads no column of its plan.
            let demand = matches!(kind, SubqueryKind::Exists).then(Attrs::new);
            out.push((self.id(s), demand));
        });
        out
    }

    /// What is read of each node: parents first (Kahn's order), so a shared
    /// node sees the union of its consumers' demands.
    fn requirements(&self) -> Vec<Attrs> {
        let n = self.nodes.len();
        let empty = Attrs::new();
        let mut indegree = vec![0_usize; n];
        for i in 0..n {
            for (c, _) in self.demands(i, &empty) {
                indegree[c] += 1;
            }
        }
        let mut req = vec![Attrs::new(); n];
        if let Some(root) = self.nodes.first() {
            req[0] = output_ids(root);
        }
        let mut queue = VecDeque::from([0]);
        while let Some(i) = queue.pop_front() {
            let mine = std::mem::take(&mut req[i]);
            for (c, demand) in self.demands(i, &mine) {
                let own = output_ids(&self.nodes[c]);
                match demand {
                    Some(d) => req[c].extend(d.into_iter().filter(|id| own.contains(id))),
                    None => req[c].extend(own),
                }
                indegree[c] -= 1;
                if indegree[c] == 0 {
                    queue.push_back(c);
                }
            }
            req[i] = mine;
        }
        req
    }
}

struct Rewrite<'a> {
    graph: &'a Graph,
    req: &'a [Attrs],
    done: HashMap<usize, Arc<LogicalPlan>>,
}

impl Rewrite<'_> {
    fn node(&mut self, i: usize) -> Arc<LogicalPlan> {
        if let Some(out) = self.done.get(&i) {
            return Arc::clone(out);
        }
        let original = Arc::clone(&self.graph.nodes[i]);
        let req = &self.req[i];
        let rebuilt = Arc::clone(&original)
            .map_children(|c| Ok::<_, std::convert::Infallible>(self.node(self.graph.id(&c))))
            .and_then(|p| p.map_expressions(|e| Ok(self.subqueries(e))));
        let rebuilt = match rebuilt {
            Ok(p) => p,
            Err(never) => match never {},
        };
        let out = match &rebuilt.node {
            PlanNode::Scan(scan) if scan.output.iter().any(|c| !req.contains(&c.id)) => {
                let mut output: Vec<BoundColumn> = scan
                    .output
                    .iter()
                    .filter(|c| req.contains(&c.id))
                    .cloned()
                    .collect();
                if output.is_empty() {
                    output.extend(scan.output.first().cloned());
                }
                replace(
                    &rebuilt,
                    PlanNode::Scan(Scan {
                        output,
                        ..scan.clone()
                    }),
                )
            }
            PlanNode::Project { input, exprs } => {
                let kept = keep(exprs, req);
                if kept.len() == exprs.len() {
                    rebuilt
                } else {
                    let input = Arc::clone(input);
                    replace(&rebuilt, PlanNode::Project { input, exprs: kept })
                }
            }
            PlanNode::Aggregate {
                input,
                grouping,
                aggregates,
            } => {
                let kept = keep(aggregates, req);
                let mut need = refs(&kept);
                grouping_refs(&rebuilt, &mut need);
                let aggregates = kept;
                let node = PlanNode::Aggregate {
                    input: narrow(input, &need),
                    grouping: grouping.clone(),
                    aggregates,
                };
                replace(&rebuilt, node)
            }
            PlanNode::Join {
                left,
                right,
                kind,
                condition,
            } => {
                let need = join_need(&rebuilt, req);
                let node = PlanNode::Join {
                    left: narrow(left, &need),
                    right: narrow(right, &need),
                    kind: *kind,
                    condition: condition.clone(),
                };
                replace(&rebuilt, node)
            }
            _ => rebuilt,
        };
        self.done.insert(i, Arc::clone(&out));
        out
    }

    fn subqueries(&mut self, e: Arc<Expr>) -> Arc<Expr> {
        if !crate::expr::has_subquery(&e) {
            return e;
        }
        let out = e.transform_up(&mut |x| {
            Ok::<_, std::convert::Infallible>(match &x.node {
                ExprNode::Subquery { kind, plan } => {
                    let plan = self.node(self.graph.id(plan));
                    Arc::new(Expr {
                        node: ExprNode::Subquery {
                            kind: kind.clone(),
                            plan,
                        },
                        origin: x.origin,
                    })
                }
                _ => x,
            })
        });
        match out {
            Ok(v) => v,
            Err(never) => match never {},
        }
    }
}

/// `node` with a new node kind, its origin kept.
fn replace(like: &LogicalPlan, node: PlanNode) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan {
        node,
        origin: like.origin,
    })
}

/// `input`, under a projection onto `need` when it outputs more.
fn narrow(input: &Arc<LogicalPlan>, need: &Attrs) -> Arc<LogicalPlan> {
    let Some(out) = input.resolved_output() else {
        return Arc::clone(input);
    };
    if out.iter().all(|c| need.contains(&c.id)) {
        return Arc::clone(input);
    }
    let mut exprs: Vec<Arc<Expr>> = out
        .iter()
        .filter(|c| need.contains(&c.id))
        .map(column)
        .collect();
    if exprs.is_empty() {
        exprs.extend(out.first().map(column));
    }
    Arc::new(LogicalPlan::new(PlanNode::Project {
        input: Arc::clone(input),
        exprs,
    }))
}

/// The output entries of a projection or aggregate that are read, keeping
/// at least one so the row count survives.
fn keep(exprs: &[Arc<Expr>], req: &Attrs) -> Vec<Arc<Expr>> {
    let mut kept: Vec<Arc<Expr>> = exprs
        .iter()
        .filter(|e| output_of(e).is_none_or(|c| req.contains(&c.id)))
        .cloned()
        .collect();
    if kept.is_empty() {
        kept.extend(exprs.first().cloned());
    }
    kept
}

fn refs(exprs: &[Arc<Expr>]) -> Attrs {
    let mut out = Attrs::new();
    exprs.iter().for_each(|e| references_into(e, &mut out));
    out
}

/// Adds what an aggregate's grouping reads (for a grouping kind unknown
/// here, everything the aggregate reads).
fn grouping_refs(p: &LogicalPlan, out: &mut Attrs) {
    let PlanNode::Aggregate { grouping, .. } = &p.node else {
        return;
    };
    match grouping {
        Grouping::GroupBy(v) | Grouping::Rollup(v) | Grouping::Cube(v) => {
            v.iter().for_each(|e| references_into(e, out));
        }
        Grouping::GroupingSets { exprs, sets } => exprs
            .iter()
            .chain(sets.iter().flatten())
            .for_each(|e| references_into(e, out)),
        _ => p.for_each_expr(&mut |e| references_into(e, out)),
    }
}

/// What a join reads of its inputs: what is read of it, and its condition.
fn join_need(p: &LogicalPlan, req: &Attrs) -> Attrs {
    let mut need = req.clone();
    if let PlanNode::Join {
        condition: JoinCondition::On(c),
        ..
    } = &p.node
    {
        references_into(c, &mut need);
    }
    need
}

fn output_ids(p: &LogicalPlan) -> Attrs {
    p.resolved_output()
        .unwrap_or_default()
        .into_iter()
        .map(|c| c.id)
        .collect()
}

/// Calls `f` on each subquery plan in `p`'s own expressions (not inside
/// those plans).
fn subqueries<'a>(p: &'a LogicalPlan, f: &mut impl FnMut(&'a SubqueryKind, &'a Arc<LogicalPlan>)) {
    let mut stack: Vec<&'a Arc<Expr>> = Vec::new();
    p.for_each_expr(&mut |e| stack.push(e));
    let mut seen = std::collections::HashSet::new();
    while let Some(e) = stack.pop() {
        if !seen.insert(Arc::as_ptr(e)) {
            continue;
        }
        if let ExprNode::Subquery { kind, plan } = &e.node {
            f(kind, plan);
        }
        e.for_each_child(&mut |c| stack.push(c));
    }
}
