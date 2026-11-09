// SPDX-License-Identifier: Apache-2.0
//! Join ordering (ADR 0012): each maximal tree of inner joins is
//! re-planned as a whole.
//!
//! Its inputs and the conjuncts of its conditions form a query (hyper)graph.
//! With up to [`DP_LIMIT`] inputs (the default limit), dynamic programming
//! over connected subsets finds the order with the fewest intermediate rows
//! (`C_out`; [`dp::dpsub`], which P2-T113 kept over DPhyp); above it, a
//! greedy pass repeatedly joins the connected pair with the smallest result. Session
//! configuration `krishiv.optimizer.joinEnumerator` can choose `greedy` for
//! every tree, and `krishiv.optimizer.joinDpLimit` moves the limit. Both
//! plan the same tree with the same costs, so they differ only in which
//! orders they consider; each planned tree is described in `EXPLAIN
//! VERBOSE` ([`TreeStats`]). Only connected subsets are joined, so no cross product
//! is planned unless the query itself has one (then its connected parts
//! are joined last, smallest first). Each conjunct sits at the lowest join
//! that has its inputs; the larger input of each join is on the left, the
//! smaller (the build side of a hash join) on the right. Where the
//! consumer reads columns by position, a projection on top restores the
//! original columns.
//!
//! Column equalities (`a = b`) between inputs form equivalence classes:
//! inputs with columns of one class are joined on it even where the query
//! relates them only through a third input (TPC-H q02's partsupp and its
//! decorrelated minimum share a partkey only through part), and a join
//! takes one equality per class it connects, never one that the joins
//! below already imply. Estimates count each class's equalities the same
//! way, so a cycle of them is not counted twice.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, JoinCondition, JoinKind, LogicalPlan, PlanNode,
    ResolvedExpr,
};
use krishiv_types::DataType;

use crate::estimate::Estimator;
use crate::expr::{
    Attrs, and_all, builtin, call, column, common_equalities, conjuncts, has_subquery, implied,
    is_deterministic, references, substitute,
};
use crate::narrow::{conjunct_selectivities, equi_columns, selectivity};
use crate::stats::{RelStats, Statistics, cse_definitions, reads_cse, resolve_cse};

mod dp;

/// An original relation (kept alive) and its rewrite.
type Rewritten = (Arc<LogicalPlan>, Arc<LogicalPlan>);

/// The default of the most join inputs planned exactly (`krishiv.optimizer.joinDpLimit`); larger trees
/// are planned greedily.
pub const DP_LIMIT: usize = 10;

/// The highest `krishiv.optimizer.joinDpLimit` honoured: `dpsub` considers
/// about 3^n splits.
pub(crate) const MAX_DP_LIMIT: usize = 14;

/// How each inner-join tree is enumerated (`krishiv.optimizer.joinEnumerator`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Enumerator {
    /// Every split of every subset, up to the limit; greedy above it.
    DpSub,
    /// Always greedy.
    Greedy,
}

impl Enumerator {
    /// The configuration value naming it.
    fn name(self) -> &'static str {
        match self {
            Self::DpSub => "dpsub",
            Self::Greedy => "greedy",
        }
    }

    /// The enumerator a configuration value names; `dpsub` for anything else.
    pub(crate) fn parse(name: &str) -> Self {
        match name {
            "greedy" => Self::Greedy,
            _ => Self::DpSub,
        }
    }
}

/// The join-ordering choices a session can make.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Settings {
    pub(crate) enumerator: Enumerator,
    /// The most inputs `dpsub` plans exactly (at most [`MAX_DP_LIMIT`]).
    pub(crate) dp_limit: usize,
    /// Whether to describe each planned tree ([`TreeStats`]).
    pub(crate) trace: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            enumerator: Enumerator::DpSub,
            dp_limit: DP_LIMIT,
            trace: false,
        }
    }
}

/// What planning one inner-join tree did (`EXPLAIN VERBOSE`, one note per
/// tree): its size, the enumerator, why it fell back to greedy, the pairs
/// of halves it costed, the estimated `C_out` of the plan, and the plan's
/// shape as a fingerprint (inputs numbered in their canonical order; the
/// left of each pair is the probe side).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TreeStats {
    inputs: usize,
    /// Pairs of inputs some join condition over exactly those two joins.
    edges: usize,
    /// Join conditions over three or more inputs.
    hyperedges: usize,
    enumerator: Enumerator,
    /// Why the tree was planned greedily, if it was: `inputs` (too many for
    /// the enumerator), `hyperedge`
    /// (some part is held together only by a condition over three or more
    /// inputs) or `forced` (the `greedy` enumerator).
    fallback: Option<&'static str>,
    /// Pairs of planned, connected halves costed (csg-cmp pairs).
    pairs: u64,
    /// Candidate pairs of halves looked at to find them.
    considered: u64,
    cost: f64,
    shape: String,
}

impl std::fmt::Display for TreeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tree inputs={} edges={} hyperedges={} enumerator={} fallback={} pairs={} considered={} cost={:.6e} shape={}",
            self.inputs,
            self.edges,
            self.hyperedges,
            self.enumerator.name(),
            self.fallback.unwrap_or("none"),
            self.pairs,
            self.considered,
            self.cost,
            self.shape
        )
    }
}

/// `root` with every inner-join tree reordered, and a note per tree planned
/// (when `settings.trace`).
pub(crate) fn reorder(
    root: &Arc<LogicalPlan>,
    stats: &dyn Statistics,
    propagate: bool,
    settings: Settings,
) -> (Arc<LogicalPlan>, Vec<String>) {
    let mut r = Reorder {
        est: Estimator::with_propagation(stats, propagate),
        propagate,
        done: HashMap::new(),
        settings,
        notes: Vec::new(),
    };
    let out = r.positional(root);
    (out, r.notes)
}

struct Reorder<'a> {
    est: Estimator<'a>,
    /// Whether estimates narrow ranges within a conjunct
    /// ([`crate::Options::stats_propagation`]).
    propagate: bool,
    /// The original is kept alive so its address stays unique.
    done: HashMap<*const LogicalPlan, Rewritten>,
    settings: Settings,
    /// A [`TreeStats`] per tree planned, when tracing.
    notes: Vec<String>,
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

/// The outputs of `p`'s projection that compute an expression rather than
/// pass a column through, by attribute. Empty when `p` is no projection, or
/// one the filter pushdown does not cross (a nondeterministic expression, a
/// common-subexpression output): a conjunct over it stays above, so it is
/// estimated as written.
fn computed_outputs(p: &LogicalPlan) -> HashMap<AttrId, Arc<Expr>> {
    let PlanNode::Project { exprs, .. } = &p.node else {
        return HashMap::new();
    };
    let crossed = exprs.iter().all(|e| is_deterministic(e))
        && !exprs.iter().any(|e| match &e.node {
            ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => {
                crate::cse::is_cse_column(output)
            }
            _ => false,
        });
    if !crossed {
        return HashMap::new();
    }
    exprs
        .iter()
        .filter_map(|e| match &e.node {
            ExprNode::Resolved(ResolvedExpr::Alias { expr, output })
                if !matches!(expr.node, ExprNode::Resolved(ResolvedExpr::Column(_))) =>
            {
                Some((output.id, Arc::clone(expr)))
            }
            _ => None,
        })
        .collect()
}

/// Whether the column selection `p` over `input` drops only columns common-
/// subexpression elimination extracted. CSE runs after join ordering, so only a
/// second optimization meets one: pruning left it when the columns it was made
/// to drop went after CSE, and it would not rebuild it (it never narrows for a
/// `cse_*` column alone). Looking through it would lose it and change the plan,
/// so the tree takes it as an input.
fn drops_only_cse(p: &LogicalPlan, input: &LogicalPlan) -> bool {
    let (Some(out), Some(below)) = (p.resolved_output(), input.resolved_output()) else {
        return false;
    };
    let kept = |c: &BoundColumn| out.iter().any(|o| o.id == c.id);
    below
        .iter()
        .all(|c| crate::cse::is_cse_column(c) || kept(c))
        && below.iter().any(|c| !kept(c))
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
    /// (the query result, a subquery, a CTE definition, a renaming alias,
    /// the inputs of a union and the terms of a recursive CTE):
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
        // CTE definitions (named by their exact columns), the input of a
        // renaming alias and the inputs of a union or the terms of a
        // recursive CTE (their output is the first input's columns, and the
        // others are matched to them in order) are read by position; every
        // other input by id. The analyzer rewrites every `SetOp` before
        // optimization; it is listed in case a plan ever keeps one.
        let (by_position, ctes) = match &p.node {
            PlanNode::SubqueryAlias { columns, .. } => (!columns.is_empty(), 0),
            PlanNode::Union { .. } | PlanNode::SetOp { .. } | PlanNode::UnionLoop { .. } => {
                (true, 0)
            }
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
                    Some(input) if !drops_only_cse(n, input) => stack.push(input),
                    _ => leaves.push(self.node(n)),
                },
            }
        }
        if leaves.len() < 2 {
            return self.children(root);
        }
        let ests: Vec<Arc<RelStats>> = leaves.iter().map(|l| self.est.plan(l)).collect();
        let mut graph = Graph::new(&leaves, &ests, preds.clone(), self.propagate);
        if graph.binary_parts() > graph.parts(true) {
            // Some inputs are joined only by a condition over three or more
            // of them, so two of them would meet in a cross product. Take
            // the column equalities every disjunct of such a condition holds
            // as conditions of their own (TPC-DS q13). Only then: elsewhere
            // they would add join keys the inputs do not need.
            let preds = preds.iter().flat_map(common_equalities).collect();
            graph = Graph::new(&leaves, &ests, preds, self.propagate);
        }
        if graph.parts(false) < graph.parts(true) {
            // Inputs related only through a conjunct with a subquery: as
            // written they meet in a join on it; re-planned they would meet
            // in a cross product. Keep the tree as written.
            return self.children(root);
        }
        graph.trace = self.settings.trace;
        let (plan, stats) = graph.plan(self.settings);
        if self.settings.trace {
            self.notes.push(stats.to_string());
        }
        plan
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
    all: &RelStats,
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
        // The key's selectivity goes on one conjunct, the same however the
        // query wrote the key (so planning a plan again gives it again):
        // one outside a column class where there is one, else the class
        // equality of the smallest columns. `Graph::independent` takes a
        // carrier before the other equalities, so it is not the one left out.
        let carrier = members
            .iter()
            .copied()
            .find(|&i| preds[i].class.is_none())
            .or_else(|| {
                members
                    .iter()
                    .copied()
                    .min_by_key(|&i| preds[i].class.map(|[a, b]| (a.min(b), a.max(b))))
            });
        for &i in &members {
            let carries = Some(i) == carrier;
            preds[i].sel = if carries { sel } else { 1.0 };
            preds[i].carrier = Some(carries);
        }
    }
}

/// The selectivity `sel` of a conjunct over the inputs `on`, given the
/// filters it implies on each of them. Filter pushdown adds, below the
/// join, what a disjunction implies on each input alone (`(a1 and b1) or
/// (a2 and b2)` implies `a1 or a2` on `a`'s input; [`implied`]), so the
/// inputs' estimates already hold those factors: counting them again in
/// the conjunct would square them. TPC-DS q85 is the case in point: its
/// `ca_state`/`ws_net_profit` disjunction keeps about 60% of the pairs
/// that pass the implied filters, but the unconditional product estimated
/// a tiny fraction, so a nested-loop join of `web_sales` and
/// `customer_address` was planned first and produced 148M rows at SF1. Dividing
/// by the implied filters' selectivities gives the conditional
/// selectivity (at most 1); if pushdown did not add such a filter, the
/// estimate errs high, the cheaper mistake for ordering.
fn conditional<'e>(
    pred: &Arc<Expr>,
    sel: f64,
    on: &[usize],
    attrs: &[Attrs],
    est: impl Fn(usize) -> &'e RelStats,
    narrow: bool,
) -> f64 {
    let implied: f64 = on
        .iter()
        .filter_map(|&i| implied(pred, &attrs[i]).map(|f| selectivity(&f, est(i), narrow)))
        .product();
    // An implied filter estimated to keep nothing leaves its input at one
    // row (estimates are at least one), and the conjunct at most 1.
    (sel / implied.max(f64::MIN_POSITIVE)).min(1.0)
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
#[derive(Debug, Clone, Copy, PartialEq)]
struct Best {
    cost: f64,
    left: Set,
    right: Set,
}

/// A plan of some inputs with its estimated rows, its `C_out` (the rows of
/// every join in it, summed) and its shape (when tracing).
struct Built {
    plan: Arc<LogicalPlan>,
    rows: f64,
    cost: f64,
    shape: String,
}

/// A conjunct of the tree: the inputs it reads, its selectivity, and
/// whether it can be a join condition (one with a subquery stays a filter
/// and does not connect inputs). A column equality between two inputs also
/// names its columns (`class`): it belongs to their equivalence class.
struct Pred {
    expr: Arc<Expr>,
    on: Vec<usize>,
    sel: f64,
    edge: bool,
    class: Option<[AttrId; 2]>,
    /// For a member of a composite key ([`composite_keys`]): whether it
    /// carries the key's selectivity (the others count 1).
    carrier: Option<bool>,
}

/// The columns of `a = b` over two plain columns of the same type, which
/// is transitive (`=`, not `<=>`: a null never matches, so with `a = b`
/// and `b = c`, `a = c` holds of every row that passes both). Array, map
/// and struct columns are left out: a class can turn a join key into a
/// filter of one input, and the executor's scalar `=` does not compare
/// every nested type its join keys do (a struct holding an array), so
/// their equalities stay as the query wrote them.
fn equality(e: &Expr) -> Option<[&BoundColumn; 2]> {
    let Some((Builtin::Eq, [a, b], _)) = builtin(e) else {
        return None;
    };
    fn col(e: &Expr) -> Option<&BoundColumn> {
        match &e.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c),
            _ => None,
        }
    }
    let (a, b) = (col(a)?, col(b)?);
    (a.id != b.id && a.data_type == b.data_type && !a.data_type.is_nested()).then_some([a, b])
}

/// Union-find over attributes, for the equalities that a set of joined
/// inputs already holds.
#[derive(Default)]
struct Classes(HashMap<AttrId, AttrId>);

impl Classes {
    fn find(&mut self, mut x: AttrId) -> AttrId {
        while let Some(&p) = self.0.get(&x) {
            if p == x {
                break;
            }
            let gp = self.0.get(&p).copied().unwrap_or(p);
            self.0.insert(x, gp);
            x = gp;
        }
        x
    }

    /// Joins the classes of `a` and `b`; false when they were one already.
    fn union(&mut self, [a, b]: [AttrId; 2]) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        self.0.insert(ra, rb);
        self.0.entry(rb).or_insert(rb);
        true
    }
}

struct Graph {
    /// Whether plans record their shape ([`Settings::trace`]).
    trace: bool,
    leaves: Vec<Arc<LogicalPlan>>,
    rows: Vec<f64>,
    /// The conjuncts over more than one input.
    preds: Vec<Pred>,
}

impl Graph {
    fn new(
        leaves: &[Arc<LogicalPlan>],
        ests: &[Arc<RelStats>],
        preds: Vec<Arc<Expr>>,
        narrow: bool,
    ) -> Self {
        let mut owner: HashMap<AttrId, usize> = HashMap::new();
        for (i, l) in leaves.iter().enumerate() {
            for c in l.resolved_output().unwrap_or_default() {
                owner.insert(c.id, i);
            }
        }
        let preds = canonical_equalities(preds, &owner);
        let refs: Vec<_> = preds.iter().map(|p| references(p)).collect();
        // Estimates see through common-subexpression projections: a `cse_*`
        // column holds its definition's values, so the same rows are
        // estimated before and after that pass runs. The plans keep the
        // columns; only the estimates read the definitions.
        let any_cse = preds.iter().any(|p| reads_cse(p));
        let mut all_defs: HashMap<AttrId, Arc<Expr>> = HashMap::new();
        if any_cse {
            for l in leaves {
                all_defs.extend(cse_definitions(l));
            }
        }
        let resolved_refs: Vec<Attrs> = refs
            .iter()
            .map(|r| {
                r.iter()
                    .flat_map(|id| match all_defs.get(id) {
                        Some(def) => references(def),
                        None => Attrs::from([*id]),
                    })
                    .collect()
            })
            .collect();
        // A canonical input order, so the plan depends neither on how the
        // tree was written nor on columns pruned since: inputs by the
        // smallest attribute the conjuncts read of them.
        let key = |i: usize| {
            let read = resolved_refs
                .iter()
                .flatten()
                .filter(|id| owner.get(id) == Some(&i));
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
        let mut rows: Vec<f64> = order.iter().map(|&i| ests[i].rows_for_costing()).collect();
        let mut sorted: Vec<Arc<LogicalPlan>> =
            order.iter().map(|&i| Arc::clone(&leaves[i])).collect();
        let mut all = RelStats::default();
        for e in ests {
            all.cols.extend(e.cols.iter().map(|(k, v)| (*k, v.clone())));
        }
        // Each input's attributes, by its position in `sorted`.
        let attrs: Vec<Attrs> = order
            .iter()
            .map(|&i| {
                leaves[i]
                    .resolved_output()
                    .unwrap_or_default()
                    .iter()
                    .map(|c| c.id)
                    .collect()
            })
            .collect();
        let mut joins = Vec::new();
        // Each input's projection outputs that compute an expression (not a
        // plain column), by attribute.
        let projected: Vec<HashMap<AttrId, Arc<Expr>>> =
            sorted.iter().map(|l| computed_outputs(l)).collect();
        // Definitions per sorted input, so estimates read through
        // common-subexpression projections (built only when a conjunct
        // needs them).
        let leaf_defs: Vec<HashMap<AttrId, Arc<Expr>>> = if any_cse {
            sorted.iter().map(|l| cse_definitions(l)).collect()
        } else {
            Vec::new()
        };
        // The conjuncts over each input alone, estimated together after the
        // loop: each narrows the columns the next reads.
        let mut local: Vec<Vec<Arc<Expr>>> = vec![Vec::new(); sorted.len()];
        for (p, r) in preds.into_iter().zip(&refs) {
            let mut on: Vec<usize> = r
                .iter()
                .filter_map(|id| owner.get(id).map(|i| at[i]))
                .collect();
            on.sort_unstable();
            on.dedup();
            if let [] | [_] = on.as_slice() {
                // A conjunct over one input filters that input. One over none
                // (a constant `false`) filters the first input without inputs
                // of its own (a scan): pushdown cannot move it further, so it
                // is where the next optimization finds and places it again.
                // On an input with a plan under it, a later pushdown would
                // carry it into a join tree ordered here without it.
                let i = on.first().copied().unwrap_or_else(|| {
                    sorted
                        .iter()
                        .position(|l| {
                            let mut leaf = true;
                            l.for_each_child(&mut |_| leaf = false);
                            leaf
                        })
                        .unwrap_or(0)
                });
                let resolved = leaf_defs
                    .get(i)
                    .map_or_else(|| Arc::clone(&p), |defs| resolve_cse(&p, defs));
                // Read through the input's projection to what it computes, as
                // the estimate sees the conjunct once pushed below it: a class
                // equality join ordering derives over `x + 1 AS c0, x AS c2` is
                // `x + 1 = x` there, so both places estimate it alike.
                let resolved = match projected.get(i) {
                    // A conjunct with a subquery stays above the projection.
                    Some(defs) if !defs.is_empty() && !has_subquery(&resolved) => {
                        substitute(&resolved, defs)
                    }
                    _ => resolved,
                };
                local[i].push(resolved);
                sorted[i] = Arc::new(LogicalPlan::new(PlanNode::Filter {
                    input: Arc::clone(&sorted[i]),
                    predicate: p,
                }));
                continue;
            }
            joins.push((p, on));
        }
        // In a canonical order (each conjunct narrows the view the next reads,
        // and a null-rejecting or `IN` conjunct changes what a range after it
        // keeps), so the estimate does not depend on how the query wrote them.
        // Multiplied in one at a time, as each conjunct's rows always were,
        // so inputs whose conjuncts do not narrow each other estimate as before.
        for preds in &mut local {
            preds.sort_by_cached_key(ToString::to_string);
        }
        for (i, preds) in local.iter().enumerate() {
            for sel in
                conjunct_selectivities(preds.iter().map(AsRef::as_ref), &ests[order[i]], narrow)
            {
                rows[i] = (rows[i] * sel).max(1.0);
            }
        }
        // A filtered input keeps at most as many distinct values of a column
        // as it keeps rows, as the estimate of a filter does: a join with a
        // filtered input then counts the distinct keys that input keeps
        // (containment), not its whole table's. Without it, a dimension
        // filtered to a year (365 of 73,049 dates) joined to a fact table
        // holding five years of dates was estimated 40x too small, which
        // made the fact side look like the smaller input to build.
        for (i, ids) in attrs.iter().enumerate() {
            for id in ids {
                if let Some(c) = all.cols.get_mut(id) {
                    c.ndv = c.ndv.min(rows[i]);
                }
            }
        }
        let mut kept = Vec::new();
        for (p, on) in joins {
            let resolved = if leaf_defs.is_empty() {
                Arc::clone(&p)
            } else {
                let mut env = HashMap::new();
                for o in &on {
                    if let Some(defs) = leaf_defs.get(*o) {
                        env.extend(defs.iter().map(|(k, v)| (*k, Arc::clone(v))));
                    }
                }
                resolve_cse(&p, &env)
            };
            let sel = conditional(
                &resolved,
                selectivity(&resolved, &all, narrow),
                &on,
                &attrs,
                |i| &ests[order[i]],
                narrow,
            );
            let edge = !has_subquery(&p);
            let class = equality(&p).map(|[a, b]| [a.id, b.id]);
            kept.push(Pred {
                expr: p,
                on,
                sel,
                edge,
                class,
                carrier: None,
            });
        }
        composite_keys(&mut kept, &all, &owner, &at, &rows);
        Self {
            trace: false,
            leaves: sorted,
            rows,
            preds: kept,
        }
    }

    /// The connected parts of the graph, through join conditions only
    /// (`edges_only`) or through every conjunct.
    fn parts(&self, edges_only: bool) -> usize {
        let mut root: Vec<usize> = (0..self.n()).collect();
        fn find(root: &mut [usize], mut x: usize) -> usize {
            while root[x] != x {
                root[x] = root[root[x]];
                x = root[x];
            }
            x
        }
        for p in self.preds.iter().filter(|p| p.edge || !edges_only) {
            for w in p.on.windows(2) {
                let (a, b) = (find(&mut root, w[0]), find(&mut root, w[1]));
                root[a] = b;
            }
        }
        (0..self.n()).filter(|&x| find(&mut root, x) == x).count()
    }

    /// The connected parts through join conditions over exactly two inputs.
    fn binary_parts(&self) -> usize {
        let mut root: Vec<usize> = (0..self.n()).collect();
        fn find(root: &mut [usize], mut x: usize) -> usize {
            while root[x] != x {
                root[x] = root[root[x]];
                x = root[x];
            }
            x
        }
        for p in self.preds.iter().filter(|p| p.edge && p.on.len() == 2) {
            let (a, b) = (find(&mut root, p.on[0]), find(&mut root, p.on[1]));
            root[a] = b;
        }
        (0..self.n()).filter(|&x| find(&mut root, x) == x).count()
    }

    fn n(&self) -> usize {
        self.leaves.len()
    }

    /// The conjuncts that join `a` and `b`: they read both and nothing
    /// outside `a ∪ b`.
    /// A column equality the joins of `a` and of `b` already imply is left
    /// out, as is a second equality of one class.
    fn joining(&self, a: &dyn Fn(usize) -> bool, b: &dyn Fn(usize) -> bool) -> Vec<usize> {
        let crossing: Vec<usize> = (0..self.preds.len())
            .filter(|&i| {
                let on = &self.preds[i].on;
                on.iter().all(|&x| a(x) || b(x))
                    && on.iter().any(|&x| a(x))
                    && on.iter().any(|&x| b(x))
            })
            .collect();
        let mut held = self.held(a);
        held.extend(self.held(b));
        self.independent(&held, &crossing)
    }

    /// The rows of the join of the inputs in `set`.
    fn rows_of(&self, set: &dyn Fn(usize) -> bool) -> f64 {
        let base: f64 = (0..self.n())
            .filter(|&i| set(i))
            .map(|i| self.rows[i])
            .product();
        let inside: Vec<usize> = (0..self.preds.len())
            .filter(|&i| self.preds[i].on.iter().all(|&x| set(x)))
            .collect();
        let sel: f64 = self
            .independent(&[], &inside)
            .into_iter()
            .map(|i| self.preds[i].sel)
            .product();
        rows(base * sel)
    }

    /// Of the conjuncts `cands` (in order), those not implied by the
    /// equalities `held` and the ones kept before them: every conjunct but
    /// a column equality whose columns are already one class.
    fn independent(&self, held: &[usize], cands: &[usize]) -> Vec<usize> {
        let mut classes = Classes::default();
        for c in held.iter().filter_map(|&i| self.preds[i].class) {
            classes.union(c);
        }
        // Equalities are chosen in the order of their columns, not of the
        // conjuncts, so the choice does not depend on how the query (or a
        // previous planning) wrote them. A composite key's members come
        // first (the one carrying its selectivity, then the others): the
        // key's selectivity already counts them, so a class joined through
        // a composite key counts it there and leaves out the other
        // equalities it implies (q02: partsupp's partkey equality with the
        // minimum's is in the cost key, so part joins on one of the two
        // partkeys, not both).
        let mut eqs: Vec<(u8, AttrId, AttrId, usize)> = cands
            .iter()
            .filter_map(|&i| {
                let p = &self.preds[i];
                let rank = match p.carrier {
                    Some(true) => 0,
                    Some(false) => 1,
                    None => 2,
                };
                p.class.map(|[a, b]| (rank, a.min(b), a.max(b), i))
            })
            .collect();
        if eqs.is_empty() {
            return cands.to_vec();
        }
        eqs.sort_unstable();
        let kept: Vec<usize> = eqs
            .into_iter()
            .filter(|&(_, a, b, _)| classes.union([a, b]))
            .map(|(_, _, _, i)| i)
            .collect();
        cands
            .iter()
            .copied()
            .filter(|i| self.preds[*i].class.is_none() || kept.contains(i))
            .collect()
    }

    /// The column equalities over the inputs `set` alone.
    fn held(&self, set: &dyn Fn(usize) -> bool) -> Vec<usize> {
        (0..self.preds.len())
            .filter(|&i| self.preds[i].class.is_some() && self.preds[i].on.iter().all(|&x| set(x)))
            .collect()
    }

    /// The join conditions as input masks (inputs as bits; empty past 63
    /// inputs, which only the greedy pass plans).
    fn masks(&self) -> Vec<Set> {
        if self.n() >= 64 {
            return Vec::new();
        }
        self.preds
            .iter()
            .filter(|p| p.edge)
            .map(|p| p.on.iter().fold(0, |m, &x| m | (1 << x)))
            .collect()
    }

    /// The tree planned by `settings`' enumerator, and what it did.
    fn plan(&self, settings: Settings) -> (Arc<LogicalPlan>, TreeStats) {
        let n = self.n();
        let masks = self.masks();
        let enumerator = settings.enumerator;
        let (table, mut fallback, work) = match enumerator {
            Enumerator::Greedy => (None, Some("forced"), dp::Work::default()),
            Enumerator::DpSub if n <= settings.dp_limit.min(MAX_DP_LIMIT) => {
                let rows_of = |s: Set| self.rows_of(&|i| s & (1 << i) != 0);
                let (table, work) = dp::dpsub(n, &masks, &rows_of);
                (Some(table), None, work)
            }
            Enumerator::DpSub => (None, Some("inputs"), dp::Work::default()),
        };
        let built = match table.and_then(|t| self.table_plan(&t, &masks)) {
            Some(b) => b,
            None => {
                // A part held together only by a conjunct over three or more
                // inputs has no exact plan (no split is connected).
                fallback = fallback.or(Some("hyperedge"));
                self.greedy()
            }
        };
        let mut pairs_seen = std::collections::BTreeSet::new();
        let mut hyperedges = 0;
        for p in self.preds.iter().filter(|p| p.edge) {
            match p.on.as_slice() {
                [a, b] => {
                    pairs_seen.insert((*a, *b));
                }
                [_, _, _, ..] => hyperedges += 1,
                _ => {}
            }
        }
        let stats = TreeStats {
            inputs: n,
            edges: pairs_seen.len(),
            hyperedges,
            enumerator,
            fallback,
            pairs: work.pairs,
            considered: work.considered,
            cost: built.cost,
            shape: built.shape,
        };
        (built.plan, stats)
    }

    /// The plan of an exact table: each connected part from the table,
    /// joined last by cross products; `None` when some part has no plan.
    fn table_plan(&self, best: &HashMap<Set, Best>, masks: &[Set]) -> Option<Built> {
        let n = self.n();
        let bit = |s: Set| move |i: usize| s & (1 << i) != 0;
        // The connected parts of the graph (one when it is connected).
        let mut part: Vec<Set> = (0..n).map(|i| 1 << i).collect();
        for &m in masks {
            let joined = part
                .iter()
                .filter(|&&p| p & m != 0)
                .fold(0, |acc, p| acc | p);
            part.retain(|&p| p & m == 0);
            part.push(joined);
        }
        if part.iter().any(|p| !best.contains_key(p)) {
            return None;
        }
        let plans = part
            .into_iter()
            .map(|p| ((0..n).map(bit(p)).collect(), self.build(p, best)))
            .collect();
        Some(self.cross(plans))
    }

    /// The plan of `s` from the table.
    fn build(&self, s: Set, best: &HashMap<Set, Best>) -> Built {
        let bit = |s: Set| move |i: usize| s & (1 << i) != 0;
        let rows = self.rows_of(&bit(s));
        match best.get(&s) {
            Some(b) if b.left != 0 => {
                let (l, r) = (self.build(b.left, best), self.build(b.right, best));
                let on = self.joining(&bit(b.left), &bit(b.right));
                self.join(l, r, &on, rows)
            }
            _ => self.leaf(s.trailing_zeros() as usize, rows),
        }
    }

    /// Input `i` as a plan of `rows` rows.
    fn leaf(&self, i: usize, rows: f64) -> Built {
        Built {
            plan: Arc::clone(&self.leaves[i]),
            rows,
            cost: 0.0,
            shape: if self.trace {
                i.to_string()
            } else {
                String::new()
            },
        }
    }

    /// Greedy ordering: join the connected pair with the fewest result rows
    /// until no pair is connected. Each round reads every conjunct once.
    fn greedy(&self) -> Built {
        let n = self.n();
        let mut part_of: Vec<usize> = (0..n).collect();
        let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut parts: Vec<Option<Built>> =
            (0..n).map(|i| Some(self.leaf(i, self.rows[i]))).collect();
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
            // Each pair counts and joins on the equalities its parts do
            // not already hold (each part's, found once a round).
            let mut held: HashMap<usize, Vec<usize>> = HashMap::new();
            for (k, p) in self.preds.iter().enumerate() {
                if p.class.is_some()
                    && let Some((&first, rest)) = p.on.split_first()
                    && rest.iter().all(|&x| part_of[x] == part_of[first])
                {
                    held.entry(part_of[first]).or_default().push(k);
                }
            }
            for (&(a, b), (sel, _, on)) in &mut pairs {
                if !on.iter().any(|&k| self.preds[k].class.is_some()) {
                    continue;
                }
                let both: Vec<usize> = [a, b]
                    .iter()
                    .flat_map(|x| held.get(x).into_iter().flatten().copied())
                    .collect();
                *on = self.independent(&both, on);
                *sel = on.iter().map(|&k| self.preds[k].sel).product();
            }
            let mut pick: Option<((usize, usize), f64)> = None;
            for (&(a, b), (sel, edge, _)) in &pairs {
                let (Some(pa), Some(pb)) = (&parts[a], &parts[b]) else {
                    continue;
                };
                let r = rows(pa.rows * pb.rows * sel);
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
            parts[a] = Some(self.join(pa, pb, &on, r));
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
                let built = p?;
                let mut set = vec![false; n];
                m.iter().for_each(|&x| set[x] = true);
                Some((set, built))
            })
            .collect();
        self.cross(left)
    }

    /// Joins disconnected parts by cross products, largest on the left.
    fn cross(&self, mut parts: Vec<(Vec<bool>, Built)>) -> Built {
        parts.sort_by(|a, b| b.1.rows.total_cmp(&a.1.rows));
        let mut it = parts.into_iter();
        let Some((mut set, mut acc)) = it.next() else {
            return self.leaf(0, self.rows.first().copied().unwrap_or(1.0));
        };
        for (s, p) in it {
            let on = self.joining(&|x| set[x], &|x| s[x]);
            let r = self::rows(acc.rows * p.rows);
            acc = self.join(acc, p, &on, r);
            set = set.iter().zip(&s).map(|(a, b)| *a || *b).collect();
        }
        acc
    }

    /// An inner join of two plans on conjuncts `on`, of `rows` rows, the
    /// larger on the left; conjuncts that cannot be join conditions filter
    /// its result.
    fn join(&self, a: Built, b: Built, on: &[usize], rows: f64) -> Built {
        let (left, right) = if b.rows > a.rows { (b, a) } else { (a, b) };
        let pick = |edge: bool| {
            on.iter()
                .filter(move |&&i| self.preds[i].edge == edge)
                .map(|&i| Arc::clone(&self.preds[i].expr))
        };
        let condition = and_all(pick(true)).map_or(JoinCondition::None, JoinCondition::On);
        let shape = if self.trace {
            format!("({} {})", left.shape, right.shape)
        } else {
            String::new()
        };
        let cost = self::rows(left.cost + right.cost + rows);
        let join = Arc::new(LogicalPlan::new(PlanNode::Join {
            left: left.plan,
            right: right.plan,
            kind: JoinKind::Inner,
            condition,
        }));
        let plan = match and_all(pick(false)) {
            Some(predicate) => Arc::new(LogicalPlan::new(PlanNode::Filter {
                input: join,
                predicate,
            })),
            None => join,
        };
        Built {
            plan,
            rows,
            cost,
            shape,
        }
    }
}

/// `preds` with the column equalities between inputs of the tree replaced
/// by their equivalence classes' canonical equalities. Each class's
/// representative on an input is its smallest column there; the class
/// becomes one equality of the representatives of each pair of its inputs
/// (an input pair the query relates only through a third input included),
/// and one of the representative with each other column of the class on
/// the same input (a filter of that input). Which of them a join takes is
/// [`Graph::independent`]'s choice. The result does not depend on how the
/// query wrote the class, so re-planning a planned tree gives the same
/// plan. Other conjuncts and the written canonical equalities keep their
/// places; a written equality that is not canonical is left out (the
/// canonical ones imply it), and the derived ones follow in the order of
/// their columns.
fn canonical_equalities(preds: Vec<Arc<Expr>>, owner: &HashMap<AttrId, usize>) -> Vec<Arc<Expr>> {
    let mut classes = Classes::default();
    let mut cols: BTreeMap<AttrId, BoundColumn> = BTreeMap::new();
    // The equalities as written, by their columns (smaller id first).
    let mut written: HashMap<(AttrId, AttrId), Arc<Expr>> = HashMap::new();
    // The conjuncts in order; an equality as its columns.
    let mut order: Vec<Result<Arc<Expr>, (AttrId, AttrId)>> = Vec::new();
    for p in preds {
        match equality(&p) {
            Some([a, b]) if owner.contains_key(&a.id) && owner.contains_key(&b.id) => {
                classes.union([a.id, b.id]);
                cols.insert(a.id, a.clone());
                cols.insert(b.id, b.clone());
                let key = (a.id.min(b.id), a.id.max(b.id));
                if let std::collections::hash_map::Entry::Vacant(e) = written.entry(key) {
                    e.insert(Arc::clone(&p));
                    order.push(Err(key));
                }
            }
            _ => order.push(Ok(p)),
        }
    }
    if written.is_empty() {
        return order.into_iter().filter_map(Result::ok).collect();
    }
    // Per class (by its smallest column), its columns per input, smallest
    // first. `cols` is in id order, so each list is too.
    let mut members: BTreeMap<AttrId, BTreeMap<usize, Vec<&BoundColumn>>> = BTreeMap::new();
    let mut first: HashMap<AttrId, AttrId> = HashMap::new();
    for (id, c) in &cols {
        let root = classes.find(*id);
        let key = *first.entry(root).or_insert(*id);
        if let Some(&leaf) = owner.get(id) {
            members
                .entry(key)
                .or_default()
                .entry(leaf)
                .or_default()
                .push(c);
        }
    }
    // The canonical equalities, by their columns (smaller id first).
    let mut canonical: Vec<(AttrId, AttrId)> = Vec::new();
    for inputs in members.values() {
        let mut reps: Vec<&BoundColumn> = Vec::new();
        for list in inputs.values() {
            if let [rep, others @ ..] = list.as_slice() {
                canonical.extend(others.iter().map(|o| (rep.id, o.id)));
                reps.push(rep);
            }
        }
        reps.sort_by_key(|c| c.id);
        for (k, a) in reps.iter().enumerate() {
            canonical.extend(reps[k + 1..].iter().map(|b| (a.id, b.id)));
        }
    }
    // Written ones stay where and as the query wrote them; the others
    // follow.
    let keep: std::collections::HashSet<(AttrId, AttrId)> = canonical.iter().copied().collect();
    let mut out: Vec<Arc<Expr>> = order
        .into_iter()
        .filter_map(|p| match p {
            Ok(p) => Some(p),
            Err(key) => keep
                .contains(&key)
                .then(|| written.get(&key).map(Arc::clone))
                .flatten(),
        })
        .collect();
    for (a, b) in canonical {
        if !written.contains_key(&(a, b))
            && let (Some(a), Some(b)) = (cols.get(&a), cols.get(&b))
        {
            out.push(call(
                Builtin::Eq,
                vec![column(a), column(b)],
                DataType::Boolean,
            ));
        }
    }
    out
}
