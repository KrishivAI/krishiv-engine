// SPDX-License-Identifier: Apache-2.0
//! Join ordering (ADR 0012): each maximal tree of inner joins is
//! re-planned as a whole.
//!
//! Its inputs and the conjuncts of its conditions form a query (hyper)graph.
//! With up to [`DP_LIMIT`] inputs, dynamic programming over connected
//! subsets finds the order with the fewest intermediate rows (`C_out`);
//! above it, a greedy pass repeatedly joins the connected pair with the
//! smallest result. Only connected subsets are joined, so no cross product
//! is planned unless the query itself has one (then its connected parts
//! are joined last, smallest first). Each conjunct sits at the lowest join
//! that has its inputs; the larger input of each join is on the left, the
//! smaller (the build side of a hash join) on the right. Where the
//! consumer reads columns by position, a projection on top restores the
//! original columns.

use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use std::sync::Arc;

use krishiv_plan::{
    AttrId, Expr, ExprNode, JoinCondition, JoinKind, LogicalPlan, PlanNode, ResolvedExpr,
};

use crate::expr::{and_all, column, conjuncts, has_subquery, is_deterministic, references};
use crate::stats::{Est, Estimator, Statistics, equi_columns, selectivity};

/// An original relation (kept alive) and its rewrite.
type Rewritten = (Arc<LogicalPlan>, Arc<LogicalPlan>);

/// The most join inputs planned exactly; larger trees are planned greedily.
pub const DP_LIMIT: usize = 10;

/// `root` with every inner-join tree reordered.
pub(crate) fn reorder(root: &Arc<LogicalPlan>, stats: &dyn Statistics) -> Arc<LogicalPlan> {
    Reorder {
        est: Estimator::new(stats),
        done: HashMap::new(),
    }
    .positional(root)
}

struct Reorder<'a> {
    est: Estimator<'a>,
    /// The original is kept alive so its address stays unique.
    done: HashMap<*const LogicalPlan, Rewritten>,
}

fn inner(p: &LogicalPlan) -> bool {
    matches!(
        p.node,
        PlanNode::Join {
            kind: JoinKind::Inner | JoinKind::Cross,
            ..
        }
    )
}

/// The input of a projection that only selects columns: column pruning
/// put it there, and the join tree looks through it.
fn selection(p: &LogicalPlan) -> Option<&Arc<LogicalPlan>> {
    match &p.node {
        PlanNode::Project { input, exprs }
            if exprs
                .iter()
                .all(|e| matches!(e.node, ExprNode::Resolved(ResolvedExpr::Column(_)))) =>
        {
            Some(input)
        }
        _ => None,
    }
}

/// Whether `p` tops an inner-join tree: the join itself, or a deterministic
/// filter over one (through column selections). A column selection on top
/// is left in place, so a join tree shared by several consumers is
/// re-planned once and stays shared.
fn tops_tree(p: &LogicalPlan) -> bool {
    let mut p = p;
    if let PlanNode::Filter { input, predicate } = &p.node {
        if !is_deterministic(predicate) {
            return false;
        }
        p = input;
        while let Some(input) = selection(p) {
            p = input;
        }
        return tops_tree(p);
    }
    inner(p)
}

impl Reorder<'_> {
    /// `p` reordered for a consumer that reads its columns by position
    /// (the query result, a subquery, a CTE definition, a renaming alias):
    /// a reordered tree may output more columns, in another order, so a
    /// projection restores the original ones.
    fn positional(&mut self, p: &Arc<LogicalPlan>) -> Arc<LogicalPlan> {
        let new = self.node(p);
        let ids = |q: &LogicalPlan| -> Vec<AttrId> {
            q.resolved_output()
                .unwrap_or_default()
                .iter()
                .map(|c| c.id)
                .collect()
        };
        if Arc::ptr_eq(&new, p) || ids(&new) == ids(p) {
            return new;
        }
        let Some(out) = p.resolved_output() else {
            return new;
        };
        Arc::new(LogicalPlan {
            node: PlanNode::Project {
                input: new,
                exprs: out.iter().map(column).collect(),
            },
            origin: p.origin,
        })
    }

    /// `p` reordered for a consumer that reads columns by id; its output
    /// holds at least the original attributes. A relation shared by several
    /// consumers is re-planned once and stays shared.
    fn node(&mut self, p: &Arc<LogicalPlan>) -> Arc<LogicalPlan> {
        if let Some((_, out)) = self.done.get(&Arc::as_ptr(p)) {
            return Arc::clone(out);
        }
        let out = if tops_tree(p) {
            self.tree(p)
        } else {
            self.children(p)
        };
        self.done
            .insert(Arc::as_ptr(p), (Arc::clone(p), Arc::clone(&out)));
        out
    }

    /// `p` with its inputs and subquery plans reordered.
    fn children(&mut self, p: &Arc<LogicalPlan>) -> Arc<LogicalPlan> {
        // CTE definitions (named by their exact columns) and the input of a
        // renaming alias are read by position; every other input by id.
        let (by_position, ctes) = match &p.node {
            PlanNode::SubqueryAlias { columns, .. } => (!columns.is_empty(), 0),
            PlanNode::WithCte { ctes, .. } => (false, ctes.len()),
            _ => (false, 0),
        };
        let mut i = 0;
        let out = Arc::clone(p)
            .map_children(|c| {
                i += 1;
                Ok::<_, std::convert::Infallible>(if by_position || i <= ctes {
                    self.positional(&c)
                } else {
                    self.node(&c)
                })
            })
            .and_then(|q| q.map_expressions(|e| Ok(self.subqueries(e))));
        match out {
            Ok(v) => v,
            Err(never) => match never {},
        }
    }

    fn subqueries(&mut self, e: Arc<Expr>) -> Arc<Expr> {
        if !has_subquery(&e) {
            return e;
        }
        let out = e.transform_up(&mut |x| {
            Ok::<_, std::convert::Infallible>(match &x.node {
                ExprNode::Subquery { kind, plan } => {
                    let new = self.positional(plan);
                    if Arc::ptr_eq(&new, plan) {
                        x
                    } else {
                        Arc::new(Expr {
                            node: ExprNode::Subquery {
                                kind: kind.clone(),
                                plan: new,
                            },
                            origin: x.origin,
                        })
                    }
                }
                _ => x,
            })
        });
        match out {
            Ok(v) => v,
            Err(never) => match never {},
        }
    }

    /// Re-plans the inner-join tree under `root`, with the filters and
    /// column selections in it. A join with a nondeterministic condition
    /// is an input of the tree, not part of it.
    fn tree(&mut self, root: &Arc<LogicalPlan>) -> Arc<LogicalPlan> {
        let mut leaves = Vec::new();
        let mut preds = Vec::new();
        let mut stack = vec![root];
        while let Some(n) = stack.pop() {
            match &n.node {
                PlanNode::Join {
                    left,
                    right,
                    kind: JoinKind::Inner | JoinKind::Cross,
                    condition,
                } if deterministic_condition(condition) => {
                    if let JoinCondition::On(c) = condition {
                        preds.extend(conjuncts(c));
                    }
                    stack.push(right);
                    stack.push(left);
                }
                PlanNode::Filter { input, predicate } if is_deterministic(predicate) => {
                    preds.extend(conjuncts(predicate));
                    stack.push(input);
                }
                _ if Arc::ptr_eq(n, root) => return self.children(root),
                _ => match selection(n) {
                    Some(input) => stack.push(input),
                    None => leaves.push(self.node(n)),
                },
            }
        }
        if leaves.len() < 2 {
            return self.children(root);
        }
        let ests: Vec<Rc<Est>> = leaves.iter().map(|l| self.est.plan(l)).collect();
        Graph::new(&leaves, &ests, preds).plan()
    }
}

fn deterministic_condition(c: &JoinCondition) -> bool {
    match c {
        JoinCondition::On(e) => is_deterministic(e),
        _ => true,
    }
}

/// Several column equalities between the same two inputs form one
/// composite key. The product of the columns' selectivities would assume
/// the columns independent, but a foreign key to a composite primary key
/// matches every row. Instead the selectivity is `1 / NDV` of the composite
/// key on the side with fewer keys (an NDV is the product of its distinct
/// columns' NDVs, at most the input's estimated rows): composite keys are
/// taken to be foreign keys into the side with fewer keys. This can
/// overestimate a many-to-many composite join; it never underestimates a
/// key join, the costlier mistake for ordering.
fn composite_keys(
    preds: &mut [Pred],
    all: &Est,
    owner: &HashMap<AttrId, usize>,
    at: &HashMap<usize, usize>,
    rows: &[f64],
) {
    let mut groups: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (i, p) in preds.iter().enumerate() {
        if let ([x, y], Some(_)) = (p.on.as_slice(), equi_columns(&p.expr, all)) {
            groups.entry((*x, *y)).or_default().push(i);
        }
    }
    for ((x, y), members) in groups {
        if members.len() < 2 {
            continue;
        }
        // Each side's distinct key columns (a column repeated in several
        // equalities counts once).
        let mut cols: [HashMap<AttrId, f64>; 2] = [HashMap::new(), HashMap::new()];
        for &i in &members {
            for (id, n) in equi_columns(&preds[i].expr, all).into_iter().flatten() {
                let leaf = owner.get(&id).and_then(|o| at.get(o)).copied();
                cols[usize::from(leaf != Some(x))].insert(id, n);
            }
        }
        let side = |k: usize, leaf: usize| cols[k].values().product::<f64>().min(rows[leaf]);
        let sel = 1.0 / side(0, x).min(side(1, y)).max(1.0);
        for (k, &i) in members.iter().enumerate() {
            preds[i].sel = if k == 0 { sel } else { 1.0 };
        }
    }
}

/// Row counts stay finite and at least one: estimates overflow on large
/// trees, and NaN would compare as anything.
fn rows(x: f64) -> f64 {
    if x.is_nan() {
        f64::MAX
    } else {
        x.clamp(1.0, f64::MAX)
    }
}

/// A set of inputs, as bits (exact planning only).
type Set = u64;

/// The cheapest plan of a subset: the halves it joins (none for an input).
#[derive(Clone, Copy)]
struct Best {
    cost: f64,
    left: Set,
    right: Set,
}

/// A conjunct of the tree: the inputs it reads, its selectivity, and
/// whether it can be a join condition (one with a subquery stays a filter
/// and does not connect inputs).
struct Pred {
    expr: Arc<Expr>,
    on: Vec<usize>,
    sel: f64,
    edge: bool,
}

struct Graph {
    leaves: Vec<Arc<LogicalPlan>>,
    rows: Vec<f64>,
    /// The conjuncts over more than one input.
    preds: Vec<Pred>,
}

impl Graph {
    fn new(leaves: &[Arc<LogicalPlan>], ests: &[Rc<Est>], preds: Vec<Arc<Expr>>) -> Self {
        let mut owner: HashMap<AttrId, usize> = HashMap::new();
        for (i, l) in leaves.iter().enumerate() {
            for c in l.resolved_output().unwrap_or_default() {
                owner.insert(c.id, i);
            }
        }
        let refs: Vec<_> = preds.iter().map(|p| references(p)).collect();
        // A canonical input order, so the plan depends neither on how the
        // tree was written nor on columns pruned since: inputs by the
        // smallest attribute the conjuncts read of them.
        let key = |i: usize| {
            let read = refs.iter().flatten().filter(|id| owner.get(id) == Some(&i));
            let all = leaves[i].resolved_output().unwrap_or_default();
            (read.min().copied(), all.iter().map(|c| c.id).min())
        };
        let mut order: Vec<usize> = (0..leaves.len()).collect();
        order.sort_by_key(|&i| key(i));
        let at: HashMap<usize, usize> = order
            .iter()
            .enumerate()
            .map(|(new, &old)| (old, new))
            .collect();
        let mut rows: Vec<f64> = order.iter().map(|&i| ests[i].rows).collect();
        let mut sorted: Vec<Arc<LogicalPlan>> =
            order.iter().map(|&i| Arc::clone(&leaves[i])).collect();
        let mut all = Est::default();
        for e in ests {
            all.cols.extend(e.cols.iter().map(|(k, v)| (*k, v.clone())));
        }
        let mut kept = Vec::new();
        for (p, r) in preds.into_iter().zip(&refs) {
            let mut on: Vec<usize> = r
                .iter()
                .filter_map(|id| owner.get(id).map(|i| at[i]))
                .collect();
            on.sort_unstable();
            on.dedup();
            if let [] | [_] = on.as_slice() {
                // A conjunct over one input (or none) filters that input.
                let i = on.first().copied().unwrap_or(0);
                rows[i] = (rows[i] * selectivity(&p, &ests[order[i]])).max(1.0);
                sorted[i] = Arc::new(LogicalPlan::new(PlanNode::Filter {
                    input: Arc::clone(&sorted[i]),
                    predicate: p,
                }));
                continue;
            }
            let sel = selectivity(&p, &all);
            let edge = !has_subquery(&p);
            kept.push(Pred {
                expr: p,
                on,
                sel,
                edge,
            });
        }
        composite_keys(&mut kept, &all, &owner, &at, &rows);
        Self {
            leaves: sorted,
            rows,
            preds: kept,
        }
    }

    fn n(&self) -> usize {
        self.leaves.len()
    }

    /// The conjuncts that join `a` and `b`: they read both and nothing
    /// outside `a ∪ b`.
    fn joining(&self, a: &dyn Fn(usize) -> bool, b: &dyn Fn(usize) -> bool) -> Vec<usize> {
        (0..self.preds.len())
            .filter(|&i| {
                let on = &self.preds[i].on;
                on.iter().all(|&x| a(x) || b(x))
                    && on.iter().any(|&x| a(x))
                    && on.iter().any(|&x| b(x))
            })
            .collect()
    }

    /// The rows of the join of the inputs in `set`.
    fn rows_of(&self, set: &dyn Fn(usize) -> bool) -> f64 {
        let base: f64 = (0..self.n())
            .filter(|&i| set(i))
            .map(|i| self.rows[i])
            .product();
        let sel: f64 = self
            .preds
            .iter()
            .filter(|p| p.on.iter().all(|&x| set(x)))
            .map(|p| p.sel)
            .product();
        rows(base * sel)
    }

    fn plan(&self) -> Arc<LogicalPlan> {
        if self.n() <= DP_LIMIT {
            self.dp()
        } else {
            self.greedy()
        }
    }

    /// Exact dynamic programming over connected subsets (`C_out` cost).
    fn dp(&self) -> Arc<LogicalPlan> {
        let n = self.n();
        let full: Set = (1 << n) - 1;
        let masks: Vec<Set> = self
            .preds
            .iter()
            .filter(|p| p.edge)
            .map(|p| p.on.iter().fold(0, |m, &x| m | (1 << x)))
            .collect();
        let bit = |s: Set| move |i: usize| s & (1 << i) != 0;
        let mut best: HashMap<Set, Best> = (0..n)
            .map(|i| {
                (
                    1 << i,
                    Best {
                        cost: 0.0,
                        left: 0,
                        right: 0,
                    },
                )
            })
            .collect();
        for s in 1..=full {
            if s.count_ones() < 2 {
                continue;
            }
            let mut found: Option<Best> = None;
            let low = s.isolate_lowest_one();
            let mut a = (s - 1) & s;
            while a > 0 {
                let b = s & !a;
                // Each split once (the half holding the lowest input is `a`),
                // both halves planned, and joined by some conjunct.
                if a & low != 0
                    && let (Some(x), Some(y)) = (best.get(&a), best.get(&b))
                    && masks
                        .iter()
                        .any(|&m| m & !s == 0 && m & a != 0 && m & b != 0)
                {
                    let cost = x.cost + y.cost;
                    if found.is_none_or(|f| cost < f.cost) {
                        found = Some(Best {
                            cost,
                            left: a,
                            right: b,
                        });
                    }
                }
                a = (a - 1) & s;
            }
            if let Some(mut f) = found {
                f.cost = rows(f.cost + self.rows_of(&bit(s)));
                best.insert(s, f);
            }
        }
        // The connected parts of the graph (one when it is connected).
        let mut part: Vec<Set> = (0..n).map(|i| 1 << i).collect();
        for &m in &masks {
            let joined = part
                .iter()
                .filter(|&&p| p & m != 0)
                .fold(0, |acc, p| acc | p);
            part.retain(|&p| p & m == 0);
            part.push(joined);
        }
        // A part held together only by a conjunct over three or more inputs
        // has no exact plan (no split is connected): plan greedily, which
        // joins such parts last.
        if part.iter().any(|p| !best.contains_key(p)) {
            return self.greedy();
        }
        let plans = part
            .into_iter()
            .map(|p| {
                let set = (0..n).map(bit(p)).collect();
                (set, self.build(p, &best), self.rows_of(&bit(p)))
            })
            .collect();
        self.cross(plans)
    }

    /// The plan of `s` from the table.
    fn build(&self, s: Set, best: &HashMap<Set, Best>) -> Arc<LogicalPlan> {
        let bit = |s: Set| move |i: usize| s & (1 << i) != 0;
        match best.get(&s) {
            Some(b) if b.left != 0 => {
                let (l, r) = (self.build(b.left, best), self.build(b.right, best));
                let on = self.joining(&bit(b.left), &bit(b.right));
                let (lr, rr) = (self.rows_of(&bit(b.left)), self.rows_of(&bit(b.right)));
                self.join((l, lr), (r, rr), &on)
            }
            _ => Arc::clone(&self.leaves[s.trailing_zeros() as usize]),
        }
    }

    /// Greedy ordering: join the connected pair with the fewest result rows
    /// until no pair is connected. Each round reads every conjunct once.
    fn greedy(&self) -> Arc<LogicalPlan> {
        let n = self.n();
        let mut part_of: Vec<usize> = (0..n).collect();
        let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut parts: Vec<Option<(Arc<LogicalPlan>, f64)>> = (0..n)
            .map(|i| Some((Arc::clone(&self.leaves[i]), self.rows[i])))
            .collect();
        loop {
            // The conjuncts between exactly two parts, by pair.
            let mut pairs: BTreeMap<(usize, usize), (f64, bool, Vec<usize>)> = BTreeMap::new();
            for (k, p) in self.preds.iter().enumerate() {
                let mut on: Vec<usize> = p.on.iter().map(|&x| part_of[x]).collect();
                on.sort_unstable();
                on.dedup();
                if let [a, b] = on[..] {
                    let e = pairs.entry((a, b)).or_insert((1.0, false, Vec::new()));
                    e.0 *= p.sel;
                    e.1 |= p.edge;
                    e.2.push(k);
                }
            }
            let mut pick: Option<((usize, usize), f64)> = None;
            for (&(a, b), (sel, edge, _)) in &pairs {
                let (Some((_, ra)), Some((_, rb))) = (&parts[a], &parts[b]) else {
                    continue;
                };
                let r = rows(ra * rb * sel);
                if *edge && pick.is_none_or(|(_, best)| r < best) {
                    pick = Some(((a, b), r));
                }
            }
            let Some(((a, b), r)) = pick else {
                break;
            };
            let (Some(pa), Some(pb)) = (parts[a].take(), parts[b].take()) else {
                break;
            };
            let on = pairs
                .remove(&(a, b))
                .map(|(_, _, on)| on)
                .unwrap_or_default();
            parts[a] = Some((self.join(pa, pb, &on), r));
            let moved = std::mem::take(&mut members[b]);
            for &x in &moved {
                part_of[x] = a;
            }
            members[a].extend(moved);
        }
        let left = parts
            .into_iter()
            .zip(members)
            .filter_map(|(p, m)| {
                let (plan, r) = p?;
                let mut set = vec![false; n];
                m.iter().for_each(|&x| set[x] = true);
                Some((set, plan, r))
            })
            .collect();
        self.cross(left)
    }

    /// Joins disconnected parts by cross products, largest on the left.
    fn cross(&self, mut parts: Vec<(Vec<bool>, Arc<LogicalPlan>, f64)>) -> Arc<LogicalPlan> {
        parts.sort_by(|a, b| b.2.total_cmp(&a.2));
        let mut it = parts.into_iter();
        let Some((mut set, mut acc, mut rows)) = it.next() else {
            return Arc::clone(&self.leaves[0]);
        };
        for (s, p, r) in it {
            let on = self.joining(&|x| set[x], &|x| s[x]);
            acc = self.join((acc, rows), (p, r), &on);
            rows = self::rows(rows * r);
            set = set.iter().zip(&s).map(|(a, b)| *a || *b).collect();
        }
        acc
    }

    /// An inner join of two plans on conjuncts `on`, the larger on the left;
    /// conjuncts that cannot be join conditions filter its result.
    fn join(
        &self,
        a: (Arc<LogicalPlan>, f64),
        b: (Arc<LogicalPlan>, f64),
        on: &[usize],
    ) -> Arc<LogicalPlan> {
        let ((left, _), (right, _)) = if b.1 > a.1 { (b, a) } else { (a, b) };
        let pick = |edge: bool| {
            on.iter()
                .filter(move |&&i| self.preds[i].edge == edge)
                .map(|&i| Arc::clone(&self.preds[i].expr))
        };
        let condition = and_all(pick(true)).map_or(JoinCondition::None, JoinCondition::On);
        let join = Arc::new(LogicalPlan::new(PlanNode::Join {
            left,
            right,
            kind: JoinKind::Inner,
            condition,
        }));
        match and_all(pick(false)) {
            Some(predicate) => Arc::new(LogicalPlan::new(PlanNode::Filter {
                input: join,
                predicate,
            })),
            None => join,
        }
    }
}
