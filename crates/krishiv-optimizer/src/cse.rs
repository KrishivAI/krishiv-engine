// SPDX-License-Identifier: Apache-2.0
//! Common-subexpression elimination within one operator.
//!
//! A repeated subexpression evaluated once per row at each use is computed
//! again at every use. Within one [`PlanNode::Project`], [`PlanNode::Filter`]
//! or [`PlanNode::Aggregate`] (aggregate arguments and grouping keys), this
//! pass finds structurally equal subexpressions (compared with
//! [`crate::expr::same`], the structural equality the canonical encoding
//! implements: equal apart from origins, with attribute ids) and evaluates
//! each once in a projection directly below the operator, with fresh
//! attribute ids, replacing the uses by column references.
//!
//! A subexpression qualifies when it is evaluated unconditionally at least
//! twice, or unconditionally once and conditionally elsewhere. Conditional
//! positions are: the right operand of `and`/`or`, `coalesce`/`if`/`nanvl`
//! arguments after the first, every `WHEN` condition after the first and
//! every `WHEN` value and `ELSE`, aggregate arguments under
//! `FILTER (WHERE ...)`, and lambda bodies. A subexpression that appears
//! only conditionally is never hoisted: no row may be evaluated that Spark
//! would not evaluate, and ANSI errors stay where they are.
//!
//! Never extracted: columns, literals, aliases as such, aggregate functions
//! themselves (their arguments do count), nondeterministic calls, anything
//! holding a subquery or a lambda variable, and anything under `try_*` (a
//! `try_*` call itself included, and a `TRY_CAST`).
//!
//! `nullif`/`nvl`/`ifnull`/`nvl2` are first resolved to their `CASE`/
//! `coalesce` forms (`nullif(a, b)` to `WHEN a = b THEN NULL ELSE a`,
//! `nvl`/`ifnull` to `coalesce`, `nvl2(x, a, b)` to
//! `WHEN x IS NOT NULL THEN a ELSE b`), so spellings of the same
//! computation meet. The existing projection-collapse rule leaves the new
//! projection in place: every extracted producer is deterministic,
//! non-trivial and used more than once, which blocks its inlining.
//!
//! Hoisting a surely-evaluated subexpression can change which error is
//! raised first when two expressions on one row would both raise; Spark
//! specifies no order there. That divergence is recorded with the ADR 0023
//! divergences (see `docs/known-differences.md`).
//!
//! Plans are pure data; the rewrite is deterministic and changes no result.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use krishiv_plan::{
    AttrId, BoundColumn, Builtin, CastMode, Expr, ExprNode, FunctionId, FunctionKind, Literal,
    LogicalPlan, Origin, PlanNode, ResolvedExpr, ResolvedFunction,
};

use crate::expr::{
    Attrs, builtin, column, data_type, has_subquery, is_deterministic, nullable, references_into,
    same,
};
use crate::restrict::{MAX_ID, max_id};

/// Eliminates common subexpressions within one operator, everywhere in
/// `node` (subquery plans included; shared subtrees once).
pub(crate) fn eliminate(node: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    // Extracting a larger expression can leave a smaller one that now
    // qualifies in the operators around it (`(a + b) * 2` extracted, and
    // `a + b` still repeated above): run again until the plan stops
    // changing, so re-optimising returns the same plan. Each round extracts
    // at least one expression, each new column is read, and the rounds are
    // capped.
    let mut current = node;
    for _ in 0..MAX_ROUNDS {
        let next = pass(Arc::clone(&current));
        if Arc::ptr_eq(&next, &current) || next.to_string() == current.to_string() {
            return next;
        }
        current = next;
    }
    current
}

/// The most rounds of [`eliminate`]: nesting deeper than this is not
/// worth the plan growth.
const MAX_ROUNDS: usize = 4;

/// One round of [`eliminate`].
fn pass(node: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    // Fresh attribute ids come from one plan-wide counter: ids are unique
    // across the whole plan (self-join [`PlanNode::Shift`] ranges
    // included), so an extracted column can never share an id with an
    // existing one. The steps never fail; an error would come from an
    // allocation the caller already paid for, so the input plan is
    // returned as is.
    let mut next = max_id(&node, &mut HashMap::new()).saturating_add(1);
    let Ok(rewritten) =
        crate::dag::rewrite_up(Arc::clone(&node), &mut |p| Ok(rewrite_node(p, &mut next)))
    else {
        return node;
    };
    // A rewritten projection that only renames extracted columns merges
    // into its parent, which the next run's projection collapse would do:
    // without this the plan would change again on re-optimising. The
    // extraction projections themselves stay: every producer is used more
    // than once, which blocks the inlining.
    let mut out = rewritten;
    loop {
        match crate::dag::rewrite_up(Arc::clone(&out), &mut |p| Ok(crate::project::collapse(p))) {
            Ok(next) if !Arc::ptr_eq(&next, &out) => out = next,
            _ => return out,
        }
    }
}

/// One operator rewritten, or returned unchanged.
fn rewrite_node(node: Arc<LogicalPlan>, next: &mut u64) -> Arc<LogicalPlan> {
    match &node.node {
        PlanNode::Project { input, exprs } => rewrite_project(&node, input, exprs, next),
        PlanNode::Filter { input, predicate } => rewrite_filter(&node, input, predicate, next),
        PlanNode::Aggregate { .. } => rewrite_aggregate(&node, next),
        PlanNode::Union { .. } | PlanNode::SetOp { .. } | PlanNode::WithCte { .. } => {
            narrow_positional(node)
        }
        _ => node,
    }
}

/// Whether `c` is a column CSE extracted (see `fresh_columns`).
fn is_cse_column(c: &BoundColumn) -> bool {
    c.name == format!("cse_{}", c.id.0)
}

/// `p` without the columns CSE extracted into its output: a filter passes
/// its input through, so a rewritten filter's output grew by them.
fn narrowed(p: &Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    let Some(out) = p.resolved_output() else {
        return Arc::clone(p);
    };
    let keep: Vec<&BoundColumn> = out.iter().filter(|c| !is_cse_column(c)).collect();
    if keep.len() == out.len() || keep.is_empty() {
        return Arc::clone(p);
    }
    Arc::new(LogicalPlan {
        node: PlanNode::Project {
            input: Arc::clone(p),
            exprs: keep.into_iter().map(column).collect(),
        },
        origin: p.origin,
    })
}

/// A union, set operation or `WITH` reads its inputs by position (or, for
/// a CTE reference, by the definition's attributes): their outputs must be
/// the ones the query wrote, without extracted columns.
fn narrow_positional(node: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    let mut out = LogicalPlan::clone(&node);
    match &mut out.node {
        PlanNode::Union { inputs, .. } => {
            for i in inputs.iter_mut() {
                *i = narrowed(i);
            }
        }
        PlanNode::SetOp { left, right, .. } => {
            *left = narrowed(left);
            *right = narrowed(right);
        }
        PlanNode::WithCte { ctes, .. } => {
            for c in ctes.iter_mut() {
                c.plan = narrowed(&c.plan);
            }
        }
        _ => return node,
    }
    let changed = match (&node.node, &out.node) {
        (PlanNode::Union { inputs: a, .. }, PlanNode::Union { inputs: b, .. }) => {
            a.iter().zip(b).any(|(x, y)| !Arc::ptr_eq(x, y))
        }
        (
            PlanNode::SetOp {
                left: a, right: b, ..
            },
            PlanNode::SetOp {
                left: c, right: d, ..
            },
        ) => !Arc::ptr_eq(a, c) || !Arc::ptr_eq(b, d),
        (PlanNode::WithCte { ctes: a, .. }, PlanNode::WithCte { ctes: b, .. }) => {
            a.iter().zip(b).any(|(x, y)| !Arc::ptr_eq(&x.plan, &y.plan))
        }
        _ => false,
    };
    if changed { Arc::new(out) } else { node }
}

/// Whether `e` is a `try_*` call or a `TRY_CAST`.
fn is_try(e: &Expr) -> bool {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            f.id.name().starts_with("try_") || f.id.name() == "try_cast"
        }
        ExprNode::Cast { mode, .. } => matches!(mode, CastMode::Try),
        _ => false,
    }
}

/// Whether `e` holds a lambda variable.
fn has_lambda_var(e: &Expr) -> bool {
    let mut found = false;
    let mut stack = vec![e];
    let mut seen: HashSet<*const Expr> = HashSet::new();
    while let Some(x) = stack.pop() {
        if !seen.insert(std::ptr::from_ref(x)) {
            continue;
        }
        if matches!(&x.node, ExprNode::Resolved(ResolvedExpr::LambdaVar(_))) {
            found = true;
            break;
        }
        x.for_each_child(&mut |c| stack.push(c));
    }
    found
}

/// The registry's `coalesce`, so a resolved `nvl`/`ifnull` meets a written
/// `coalesce` structurally (the numeric id is the definition's position in
/// the process-wide registry, as the analyzer's own).
fn coalesce_function() -> Option<FunctionId> {
    use krishiv_functions::registry::{Lookup, builtins};
    use krishiv_functions::signature::SparkProfile;
    let profile = SparkProfile::ALL.into_iter().max()?;
    match builtins().lookup("coalesce", profile) {
        Lookup::Found(def) => Some(FunctionId::Registry {
            name: def.name.to_owned(),
            id: def.id,
        }),
        _ => None,
    }
}

/// `e` with `nullif`/`nvl`/`ifnull`/`nvl2` resolved to `WHEN`/`coalesce`.
fn normalize(e: &Arc<Expr>) -> Arc<Expr> {
    let out = Arc::clone(e).transform_up(&mut |x| {
        Ok::<_, std::convert::Infallible>(match &x.node {
            ExprNode::Resolved(ResolvedExpr::Function(f))
                if f.kind == FunctionKind::Scalar && f.filter.is_none() =>
            {
                // Only the registry's own `nvl` and friends: a session UDF
                // (or SQL-bodied function) of the same name is another
                // function and must stay a call.
                let name = match &f.id {
                    FunctionId::Registry { name, .. } => name.as_str(),
                    _ => "",
                };
                match (name, f.args.as_slice()) {
                    // `nullif` with arguments of different (or unknown) types
                    // would need the analyzer's implicit casts around the
                    // equality to stay well-typed: leave those calls alone
                    // (both arguments still count as unconditional).
                    ("nullif", [a, b])
                        if matches!((data_type(a), data_type(b)), (Some(x), Some(y)) if x == y) =>
                    {
                        let eq = Arc::new(Expr::new(ExprNode::Resolved(
                            ResolvedExpr::Function(crate::expr::call_builtin(
                                Builtin::Eq,
                                vec![Arc::clone(a), Arc::clone(b)],
                                krishiv_types::DataType::Boolean,
                            )),
                        )));
                        let null =
                            Arc::new(Expr::literal(Literal::Null(data_type(a))));
                        Arc::new(Expr::new(ExprNode::Resolved(
                            ResolvedExpr::Function(crate::expr::call_builtin(
                                Builtin::When,
                                vec![eq, null, Arc::clone(a)],
                                f.data_type.clone(),
                            )),
                        )))
                    }
                    ("nvl" | "ifnull", [a, b]) => {
                        let Some(coalesce) = coalesce_function() else {
                            return Ok(x);
                        };
                        Arc::new(Expr::new(ExprNode::Resolved(
                            ResolvedExpr::Function(ResolvedFunction {
                                id: coalesce,
                                kind: FunctionKind::Scalar,
                                args: vec![Arc::clone(a), Arc::clone(b)],
                                distinct: false,
                                filter: None,
                                data_type: f.data_type.clone(),
                                nullable: f.nullable,
                                deterministic: f.deterministic,
                            }),
                        )))
                    }
                    ("nvl2", [x, a, b]) => {
                        let cond = Arc::new(Expr::new(ExprNode::Resolved(
                            ResolvedExpr::Function(crate::expr::call_builtin(
                                Builtin::IsNotNull,
                                vec![Arc::clone(x)],
                                krishiv_types::DataType::Boolean,
                            )),
                        )));
                        Arc::new(Expr::new(ExprNode::Resolved(
                            ResolvedExpr::Function(crate::expr::call_builtin(
                                Builtin::When,
                                vec![cond, Arc::clone(a), Arc::clone(b)],
                                f.data_type.clone(),
                            )),
                        )))
                    }
                    _ => x,
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

/// Direct children of `e` with whether each is evaluated conditionally.
fn children(e: &Expr) -> Vec<(Arc<Expr>, bool)> {
    let mut out = Vec::new();
    let mut push = |c: &Arc<Expr>, cond: bool| out.push((Arc::clone(c), cond));
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) | ExprNode::Cast { expr, .. } => {
            push(expr, false);
        }
        ExprNode::Field { base, .. } | ExprNode::Resolved(ResolvedExpr::Field { base, .. }) => {
            push(base, false);
        }
        ExprNode::Lambda { body, .. } => push(body, true),
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            if f.kind == FunctionKind::Aggregate {
                let conditional = f.filter.is_some();
                for a in &f.args {
                    push(a, conditional);
                }
                if let Some(filter) = &f.filter {
                    push(filter, false);
                }
                return out;
            }
            if let Some((op, args, _)) = builtin(e) {
                match (op, args) {
                    (Builtin::And | Builtin::Or, [a, b]) => {
                        push(a, false);
                        push(b, true);
                        return out;
                    }
                    // `v IN (a, b, ...)` returns on the first match or on a
                    // NULL `v`, and `BETWEEN` is `v >= lo AND v <= hi`: only
                    // `v` (and `lo`) are surely evaluated.
                    (Builtin::In, [v, list @ ..]) => {
                        push(v, false);
                        for a in list {
                            push(a, true);
                        }
                        return out;
                    }
                    (Builtin::Between, [v, lo, hi]) => {
                        push(v, false);
                        push(lo, false);
                        push(hi, true);
                        return out;
                    }
                    (Builtin::When, branches) => {
                        for (i, b) in branches.iter().enumerate() {
                            push(b, i != 0);
                        }
                        return out;
                    }
                    _ => {}
                }
            }
            match f.id.name() {
                // `coalesce` and friends evaluate their first argument for
                // every row and the rest only when needed; `nanvl(a, b)`
                // evaluates `b` only when `a` is NaN; `elt(n, ...)` only the
                // selected argument.
                "coalesce" | "nvl" | "ifnull" | "zeroifnull" | "if" | "nvl2" | "nanvl" | "elt" => {
                    for (i, a) in f.args.iter().enumerate() {
                        push(a, i != 0);
                    }
                }
                _ => {
                    for a in &f.args {
                        push(a, false);
                    }
                }
            }
            if let Some(filter) = &f.filter {
                push(filter, false);
            }
        }
        ExprNode::Subquery { kind, .. } => {
            if let krishiv_plan::SubqueryKind::In(values) = kind {
                for v in values {
                    push(v, false);
                }
            }
        }
        _ => e.for_each_child(&mut |c| push(c, false)),
    }
    out
}

/// Whether `e` may be extracted (node-local part; `try_*` descendants are
/// marked ineligible separately).
fn eligible(e: &Expr) -> bool {
    match &e.node {
        ExprNode::Literal(_)
        | ExprNode::Resolved(
            ResolvedExpr::Column(_)
            | ResolvedExpr::OuterColumn(_)
            | ResolvedExpr::LambdaVar(_)
            | ResolvedExpr::Alias { .. },
        )
        | ExprNode::Alias { .. }
        | ExprNode::Subquery { .. } => return false,
        _ => {}
    }
    if is_try(e) {
        return false;
    }
    if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node
        && f.kind == FunctionKind::Aggregate
    {
        return false;
    }
    if !is_deterministic(e) || has_subquery(e) || has_lambda_var(e) {
        return false;
    }
    data_type(e).is_some()
}

/// The size of `e` in distinct nodes, capped at `cap`.
fn size(e: &Expr, cap: usize) -> usize {
    let mut seen: HashSet<*const Expr> = HashSet::new();
    let mut stack = vec![e];
    let mut n = 0;
    while let Some(x) = stack.pop() {
        if !seen.insert(std::ptr::from_ref(x)) {
            continue;
        }
        n += 1;
        if n >= cap {
            return cap;
        }
        x.for_each_child(&mut |c| stack.push(c));
    }
    n
}

/// One parent-to-child edge: the child's index, whether the child is
/// evaluated conditionally, and how many times the child occurs among the
/// parent's children with that flag (a subtree shared through `Arc` and
/// used twice counts twice).
type Edge = (usize, bool, u8);

/// The distinct-node graph of an operator's row-evaluated expressions:
/// post-order nodes, address index, edges, the nodes under a `try_*`
/// ancestor, and per-node eligibility.
type Graph = (
    Vec<Arc<Expr>>,
    HashMap<*const Expr, usize>,
    Vec<Vec<Edge>>,
    HashSet<*const Expr>,
    HashMap<*const Expr, bool>,
);

/// Builds the distinct-node graph of `roots`: post-order nodes, address
/// index, parent-to-child edges with conditional flags, the set of nodes
/// under a `try_*` ancestor, and per-node eligibility. `None` when the
/// graph is too large (the plan is left unchanged).
fn graph(roots: &[Arc<Expr>]) -> Option<Graph> {
    const LIMIT: usize = 10_000;
    let mut order = Vec::new();
    let mut index: HashMap<*const Expr, usize> = HashMap::new();
    let mut visiting: HashSet<*const Expr> = HashSet::new();
    let mut pending: Vec<Arc<Expr>> = roots.to_vec();
    // Iterative post-order over distinct nodes.
    let mut stack: Vec<(Arc<Expr>, bool)> = pending.drain(..).map(|e| (e, false)).collect();
    while let Some((e, expanded)) = stack.pop() {
        let ptr = Arc::as_ptr(&e);
        if index.contains_key(&ptr) {
            continue;
        }
        if expanded {
            if order.len() >= LIMIT {
                return None;
            }
            index.insert(ptr, order.len());
            order.push(e);
            continue;
        }
        if !visiting.insert(ptr) {
            continue;
        }
        // Cycle safety: an expression DAG never cycles, but a repeated
        // shared node waits for its first visit to finish.
        stack.push((Arc::clone(&e), true));
        for (c, _) in children(&e) {
            if !index.contains_key(&Arc::as_ptr(&c)) {
                stack.push((c, false));
            }
        }
        if visiting.len() + stack.len() > LIMIT * 2 {
            return None;
        }
    }
    let mut edges: Vec<Vec<Edge>> = vec![Vec::new(); order.len()];
    for (i, e) in order.iter().enumerate() {
        // Fold repeated children (a shared subtree used twice) with their
        // multiplicity, capped: counts cap at two anyway.
        let mut folded: Vec<(*const Expr, bool, u8)> = Vec::new();
        for (c, cond) in children(e) {
            let ptr = Arc::as_ptr(&c);
            if let Some(entry) = folded.iter_mut().find(|(p, f, _)| *p == ptr && *f == cond) {
                entry.2 = entry.2.saturating_add(1).min(2);
            } else {
                folded.push((ptr, cond, 1));
            }
        }
        for (ptr, cond, mult) in folded {
            if let Some(&j) = index.get(&ptr) {
                edges[i].push((j, cond, mult));
            }
        }
    }
    // Nodes under a `try_*` ancestor (the `try_*` node included).
    let mut tainted: HashSet<*const Expr> = HashSet::new();
    for e in &order {
        if is_try(e) {
            let mut stack = vec![Arc::clone(e)];
            let mut seen: HashSet<*const Expr> = HashSet::new();
            while let Some(x) = stack.pop() {
                let ptr = Arc::as_ptr(&x);
                if !seen.insert(ptr) {
                    continue;
                }
                tainted.insert(ptr);
                for (c, _) in children(&x) {
                    stack.push(c);
                }
            }
        }
    }
    let eligible_map: HashMap<*const Expr, bool> = order
        .iter()
        .map(|e| (Arc::as_ptr(e), eligible(e)))
        .collect();
    Some((order, index, edges, tainted, eligible_map))
}

/// Deterministic 64-bit FNV-1a: group keys must not vary between
/// processes, so the standard library's random-seeded hasher is out.
struct Fnv(u64);

impl Fnv {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn push(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    fn byte(&mut self, v: u8) {
        self.push(&[v]);
    }

    fn int(&mut self, v: u64) {
        self.push(&v.to_le_bytes());
    }

    fn text(&mut self, s: &str) {
        self.int(s.len() as u64);
        self.push(s.as_bytes());
    }
}

/// The node-local part of the structural fingerprint: exactly what
/// [`same`] compares at one node (literals wholly; columns by id;
/// functions by name, kind, distinctness and result type; casts by target
/// and mode), so structurally equal expressions share a fingerprint.
/// Anything else gets its discriminant only: [`same`] never equates those,
/// and verification below splits collisions.
fn node_tag(e: &Expr, h: &mut Fnv) {
    match &e.node {
        ExprNode::Literal(l) => {
            h.byte(1);
            h.text(&format!("{l:?}"));
        }
        ExprNode::Resolved(ResolvedExpr::Column(c)) => {
            h.byte(2);
            h.int(c.id.0);
        }
        ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => {
            h.byte(3);
            h.int(c.id.0);
        }
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            h.byte(4);
            h.text(f.id.name());
            h.text(&format!("{:?}", f.kind));
            h.byte(u8::from(f.distinct));
            h.text(&f.data_type.simple_string());
            h.byte(u8::from(f.filter.is_some()));
        }
        ExprNode::Cast { to, mode, .. } => {
            h.byte(5);
            h.text(&to.simple_string());
            h.byte(match mode {
                CastMode::Default => 1,
                CastMode::Legacy => 2,
                CastMode::Ansi => 3,
                CastMode::Try => 4,
                _ => 0,
            });
        }
        ExprNode::Resolved(ResolvedExpr::Alias { .. }) | ExprNode::Alias { .. } => h.byte(6),
        ExprNode::Subquery { .. } => h.byte(7),
        _ => h.byte(8),
    }
}

/// The structural fingerprint of every node, bottom-up: the node tag with
/// the children's fingerprints. `order` is post-order, so children (lower
/// indices) come first. Only the structure is hashed, never where or how
/// often a node occurs: structurally equal expressions share a
/// fingerprint however they are used.
fn fingerprints(order: &[Arc<Expr>], edges: &[Vec<Edge>]) -> Vec<u64> {
    let mut out = vec![0u64; order.len()];
    for (i, e) in order.iter().enumerate() {
        let mut h = Fnv::new();
        node_tag(e, &mut h);
        h.int(edges[i].len() as u64);
        for (child, _, _) in &edges[i] {
            h.int(out[*child]);
        }
        out[i] = h.0;
    }
    out
}

/// One structurally equal group that qualifies (surely evaluated twice, or
/// once and conditionally once): its representative and its distinct-node
/// size.
struct Group {
    expr: Arc<Expr>,
    size: usize,
}

/// Capped occurrence counts of every structurally equal group. Counts
/// propagate parent to child (a shared subtree used twice counts twice);
/// groups merge address-distinct nodes by fingerprint, verified with
/// [`same`]. Totals cap at two; taint is the union over members (an
/// occurrence under `try_*` anywhere forbids the extraction).
fn analyze(
    roots: &[Arc<Expr>],
    conditional_roots: &[Arc<Expr>],
    order: &[Arc<Expr>],
    index: &HashMap<*const Expr, usize>,
    edges: &[Vec<Edge>],
    tainted: &HashSet<*const Expr>,
    eligible_map: &HashMap<*const Expr, bool>,
) -> Vec<Group> {
    let mut uncond = vec![0u8; order.len()];
    let mut cond = vec![0u8; order.len()];
    for r in roots {
        if let Some(&i) = index.get(&Arc::as_ptr(r)) {
            uncond[i] = uncond[i].saturating_add(1).min(2);
        }
    }
    for r in conditional_roots {
        if let Some(&i) = index.get(&Arc::as_ptr(r)) {
            cond[i] = cond[i].saturating_add(1).min(2);
        }
    }
    // Parents before children: `order` is post-order, so iterate reversed.
    for i in (0..order.len()).rev() {
        let (cu, cc) = (uncond[i], cond[i]);
        if cu == 0 && cc == 0 {
            continue;
        }
        for (child, is_cond, mult) in &edges[i] {
            if *is_cond {
                let add = cu.saturating_add(cc).saturating_mul(*mult).min(2);
                cond[*child] = cond[*child].saturating_add(add).min(2);
            } else {
                let au = cu.saturating_mul(*mult).min(2);
                let ac = cc.saturating_mul(*mult).min(2);
                uncond[*child] = uncond[*child].saturating_add(au).min(2);
                cond[*child] = cond[*child].saturating_add(ac).min(2);
            }
        }
    }
    let fps = fingerprints(order, edges);
    let mut by_fp: std::collections::BTreeMap<u64, Vec<usize>> = std::collections::BTreeMap::new();
    for (i, fp) in fps.iter().enumerate() {
        by_fp.entry(*fp).or_default().push(i);
    }
    let mut groups = Vec::new();
    for (_, members) in by_fp {
        // Split fingerprint collisions: first-fit in index order with
        // `same` (deterministic, as the traversal is).
        let mut verified: Vec<Vec<usize>> = Vec::new();
        for i in members {
            if let Some(group) = verified
                .iter_mut()
                .find(|group: &&mut Vec<usize>| same(&order[group[0]], &order[i]))
            {
                group.push(i);
            } else {
                verified.push(vec![i]);
            }
        }
        for members in verified {
            let eligible = eligible_map
                .get(&Arc::as_ptr(&order[members[0]]))
                .copied()
                .unwrap_or(false);
            let tainted = members
                .iter()
                .any(|i| tainted.contains(&Arc::as_ptr(&order[*i])));
            if !eligible || tainted {
                continue;
            }
            let size = members
                .iter()
                .map(|i| size(&order[*i], 10_000))
                .max()
                .unwrap_or(0);
            if size < 2 {
                continue;
            }
            let uncond = members.iter().map(|i| uncond[*i]).sum::<u8>().min(2);
            let cond = members.iter().map(|i| cond[*i]).sum::<u8>().min(2);
            if uncond >= 2 || (uncond >= 1 && cond >= 1) {
                groups.push(Group {
                    expr: Arc::clone(&order[members[0]]),
                    size,
                });
            }
        }
    }
    groups
}

/// The occurrences of `pat` in `rep` by conditionality, through the same
/// linear machinery (a naive walk would blow up on shared subtrees).
/// `None` (a graph too large) assumes the worst: full cover.
fn occurrences_in(rep: &Arc<Expr>, pat: &Arc<Expr>) -> Option<(u8, u8)> {
    let roots = [Arc::clone(rep)];
    let (order, index, edges, _, _) = graph(&roots)?;
    let mut uncond = vec![0u8; order.len()];
    let mut cond = vec![0u8; order.len()];
    if let Some(&i) = index.get(&Arc::as_ptr(rep)) {
        uncond[i] = 1;
    }
    for i in (0..order.len()).rev() {
        let (cu, cc) = (uncond[i], cond[i]);
        if cu == 0 && cc == 0 {
            continue;
        }
        for (child, is_cond, mult) in &edges[i] {
            if *is_cond {
                let add = cu.saturating_add(cc).saturating_mul(*mult).min(2);
                cond[*child] = cond[*child].saturating_add(add).min(2);
            } else {
                let au = cu.saturating_mul(*mult).min(2);
                let ac = cc.saturating_mul(*mult).min(2);
                uncond[*child] = uncond[*child].saturating_add(au).min(2);
                cond[*child] = cond[*child].saturating_add(ac).min(2);
            }
        }
    }
    let mut out = (0u8, 0u8);
    for (i, e) in order.iter().enumerate() {
        if same(e, pat) {
            out.0 = out.0.saturating_add(uncond[i]).min(2);
            out.1 = out.1.saturating_add(cond[i]).min(2);
        }
    }
    Some(out)
}

/// Replaces every subtree structurally equal to a candidate by its column.
fn replace(e: &Arc<Expr>, map: &[(Arc<Expr>, BoundColumn)]) -> Arc<Expr> {
    if map.is_empty() {
        return Arc::clone(e);
    }
    let out = Arc::clone(e).transform_up(&mut |x| {
        Ok::<_, std::convert::Infallible>(match map.iter().find(|(c, _)| same(&x, c)) {
            Some((_, col)) => column(col),
            None => x,
        })
    });
    match out {
        Ok(v) => v,
        Err(never) => match never {},
    }
}

/// `e` with `map` applied inside its aggregate calls only (their arguments
/// and `FILTER`), and each grouping key in `keys` (old, rewritten) swapped
/// wherever `e` spells it. Anything else above the aggregate stays: a
/// post-aggregation scalar is evaluated over the grouping keys, where a
/// `cse_*` column of the input is not available, and the executor matches
/// such an output to the computed key by its text.
fn replace_above(
    e: &Arc<Expr>,
    map: &[(Arc<Expr>, BoundColumn)],
    keys: &[(Arc<Expr>, Arc<Expr>)],
) -> Arc<Expr> {
    if map.is_empty() {
        return Arc::clone(e);
    }
    let out = Arc::clone(e).transform_up(&mut |x| {
        if let Some((_, new)) = keys.iter().find(|(old, _)| same(&x, old)) {
            return Ok::<_, std::convert::Infallible>(Arc::clone(new));
        }
        Ok(match &x.node {
            ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate => {
                let mut g = f.clone();
                g.args = f.args.iter().map(|a| replace(a, map)).collect();
                g.filter = f.filter.as_ref().map(|a| replace(a, map));
                Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(g))))
            }
            _ => x,
        })
    });
    match out {
        Ok(v) => v,
        Err(never) => match never {},
    }
}

/// Fresh output columns for `extracted`, or `None` when the id space is
/// exhausted.
fn fresh_columns(extracted: &[Arc<Expr>], next: &mut u64) -> Option<Vec<BoundColumn>> {
    let mut cols = Vec::with_capacity(extracted.len());
    for e in extracted {
        let (data_type, nullable) = (data_type(e)?, nullable(e));
        if *next > MAX_ID {
            return None;
        }
        let id = AttrId(*next);
        *next += 1;
        cols.push(BoundColumn {
            id,
            name: format!("cse_{}", id.0),
            data_type,
            nullable,
        });
    }
    Some(cols)
}

/// The input columns of `input` that `exprs` still read, in input order.
fn passthrough(exprs: &[Arc<Expr>], input: &Arc<LogicalPlan>) -> Vec<BoundColumn> {
    let mut refs = Attrs::new();
    for e in exprs {
        references_into(e, &mut refs);
    }
    input
        .resolved_output()
        .unwrap_or_default()
        .into_iter()
        .filter(|c| refs.contains(&c.id))
        .collect()
}

/// The projection below `input` computing `extracted` as `cols` (see
/// [`fresh_columns`]) over the input: the pass-through columns in
/// `passthrough` first, then one alias per extraction. `passthrough` must
/// be exactly the input columns the operator above still reads (see
/// [`passthrough`]; a filter passes everything through): anything wider is
/// narrowed by a later column-pruning pass, so re-optimising would return
/// a different plan.
fn project_below(
    input: &Arc<LogicalPlan>,
    extracted: &[Arc<Expr>],
    cols: &[BoundColumn],
    passthrough: &[BoundColumn],
    origin: Origin,
) -> Arc<LogicalPlan> {
    let mut exprs: Vec<Arc<Expr>> = passthrough.iter().map(column).collect();
    for (e, c) in extracted.iter().zip(cols) {
        exprs.push(Arc::new(Expr::new(ExprNode::Resolved(
            ResolvedExpr::Alias {
                expr: Arc::clone(e),
                output: c.clone(),
            },
        ))));
    }
    Arc::new(LogicalPlan {
        node: PlanNode::Project {
            input: Arc::clone(input),
            exprs,
        },
        origin,
    })
}

/// The base for a new projection below `input`: an input that only
/// passes columns through adds a layer a later column-pruning and
/// projection-collapse pass would merge away, so re-optimising would
/// return a different plan. Following it down keeps one projection.
/// Only bare columns (never aliases or renames) are absorbed, and only
/// while every column in `keep` is still an output below.
fn absorb(input: &Arc<LogicalPlan>, keep: &[BoundColumn]) -> Arc<LogicalPlan> {
    let want: HashSet<AttrId> = keep.iter().map(|c| c.id).collect();
    let mut base = Arc::clone(input);
    while let PlanNode::Project {
        input: below,
        exprs,
    } = &base.node
    {
        if !exprs
            .iter()
            .all(|e| matches!(&e.node, ExprNode::Resolved(ResolvedExpr::Column(_))))
        {
            break;
        }
        let below_out = below.resolved_output().unwrap_or_default();
        if !want.iter().all(|id| below_out.iter().any(|c| c.id == *id)) {
            break;
        }
        base = Arc::clone(below);
    }
    base
}

/// Greedily keeps qualifying groups, largest first, dropping those that
/// occur inside a kept larger extraction at all (a graph too large to
/// count assumes it does). Once the larger is extracted, an occurrence of
/// the smaller inside its definition is gone; keeping the smaller as well
/// would need a second extraction level, and replacing bottom-up would
/// leave the larger one's producer unread. Dropping it loses a little
/// sharing, never correctness, and keeps the plan a fixed point.
fn select(mut qualified: Vec<Group>) -> Vec<Arc<Expr>> {
    qualified.sort_by_key(|g| std::cmp::Reverse(g.size));
    let mut kept: Vec<Group> = Vec::new();
    for g in qualified {
        if kept.iter().any(|k: &Group| same(&k.expr, &g.expr)) {
            continue;
        }
        let inside = kept.iter().any(|k| {
            occurrences_in(&k.expr, &g.expr)
                .is_none_or(|(in_u, in_c)| in_u.saturating_add(in_c) > 0)
        });
        if !inside {
            kept.push(g);
        }
    }
    kept.into_iter().map(|g| g.expr).collect()
}

fn rewrite_project(
    node: &Arc<LogicalPlan>,
    input: &Arc<LogicalPlan>,
    exprs: &[Arc<Expr>],
    next: &mut u64,
) -> Arc<LogicalPlan> {
    let normalized: Vec<Arc<Expr>> = exprs.iter().map(normalize).collect();
    // Row-evaluated roots: each output's expression without its alias.
    let inners: Vec<Arc<Expr>> = normalized
        .iter()
        .map(|e| match &e.node {
            ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => Arc::clone(expr),
            _ => Arc::clone(e),
        })
        .collect();
    let Some((order, index, edges, tainted, eligible_map)) = graph(&inners) else {
        return Arc::clone(node);
    };
    let qualified = analyze(
        &inners,
        &[],
        &order,
        &index,
        &edges,
        &tainted,
        &eligible_map,
    );
    if qualified.is_empty() {
        return Arc::clone(node);
    }
    let unique = select(qualified);
    if unique.is_empty() {
        return Arc::clone(node);
    }
    let Some(cols) = fresh_columns(&unique, &mut *next) else {
        return Arc::clone(node);
    };
    let map: Vec<(Arc<Expr>, BoundColumn)> = unique.iter().cloned().zip(cols).collect();
    let rebuilt: Vec<Arc<Expr>> = normalized
        .iter()
        .map(|e| match &e.node {
            ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => {
                Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                    expr: replace(expr, &map),
                    output: output.clone(),
                })))
            }
            _ => replace(e, &map),
        })
        .collect();
    let keep = passthrough(&rebuilt, input);
    let cols: Vec<BoundColumn> = map.into_iter().map(|(_, c)| c).collect();
    let base = absorb(input, &keep);
    let new_input = project_below(&base, &unique, &cols, &keep, node.origin);
    Arc::new(LogicalPlan {
        node: PlanNode::Project {
            input: new_input,
            exprs: rebuilt,
        },
        origin: node.origin,
    })
}

fn rewrite_filter(
    node: &Arc<LogicalPlan>,
    input: &Arc<LogicalPlan>,
    predicate: &Arc<Expr>,
    next: &mut u64,
) -> Arc<LogicalPlan> {
    let normalized = normalize(predicate);
    let roots = vec![Arc::clone(&normalized)];
    let Some((order, index, edges, tainted, eligible_map)) = graph(&roots) else {
        return Arc::clone(node);
    };
    let qualified = analyze(&roots, &[], &order, &index, &edges, &tainted, &eligible_map);
    if qualified.is_empty() {
        return Arc::clone(node);
    }
    let unique = select(qualified);
    if unique.is_empty() {
        return Arc::clone(node);
    }
    let Some(cols) = fresh_columns(&unique, &mut *next) else {
        return Arc::clone(node);
    };
    let map: Vec<(Arc<Expr>, BoundColumn)> = unique.iter().cloned().zip(cols).collect();
    let predicate = replace(&normalized, &map);
    let keep: Vec<BoundColumn> = input.resolved_output().unwrap_or_default();
    let cols: Vec<BoundColumn> = map.into_iter().map(|(_, c)| c).collect();
    let base = absorb(input, &keep);
    let new_input = project_below(&base, &unique, &cols, &keep, node.origin);
    Arc::new(LogicalPlan {
        node: PlanNode::Filter {
            input: new_input,
            predicate,
        },
        origin: node.origin,
    })
}

/// All aggregate function nodes inside `e` (subquery plans excluded).
fn aggregates_in(e: &Arc<Expr>) -> Vec<Arc<Expr>> {
    let mut out = Vec::new();
    let mut stack = vec![Arc::clone(e)];
    let mut seen: HashSet<*const Expr> = HashSet::new();
    while let Some(x) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&x)) {
            continue;
        }
        if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node
            && f.kind == FunctionKind::Aggregate
        {
            out.push(Arc::clone(&x));
            continue;
        }
        if matches!(&x.node, ExprNode::Subquery { .. }) {
            for (c, _) in children(&x) {
                stack.push(c);
            }
            continue;
        }
        x.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    out
}

fn rewrite_aggregate(node: &Arc<LogicalPlan>, next: &mut u64) -> Arc<LogicalPlan> {
    let PlanNode::Aggregate {
        input,
        grouping,
        aggregates,
    } = &node.node
    else {
        return Arc::clone(node);
    };
    let krishiv_plan::Grouping::GroupBy(keys) = grouping else {
        return Arc::clone(node);
    };
    let normalized_keys: Vec<Arc<Expr>> = keys.iter().map(normalize).collect();
    let normalized_aggs: Vec<Arc<Expr>> = aggregates.iter().map(normalize).collect();
    // Row-evaluated roots: grouping keys plus aggregate args and filters.
    let mut roots: Vec<Arc<Expr>> = normalized_keys.clone();
    let mut conditional_roots: Vec<Arc<Expr>> = Vec::new();
    for a in &normalized_aggs {
        for agg in aggregates_in(a) {
            if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &agg.node {
                for arg in &f.args {
                    if f.filter.is_some() {
                        conditional_roots.push(Arc::clone(arg));
                    } else {
                        roots.push(Arc::clone(arg));
                    }
                }
                if let Some(filter) = &f.filter {
                    roots.push(Arc::clone(filter));
                }
            }
        }
    }
    let mut all_roots = roots.clone();
    all_roots.extend(conditional_roots.iter().cloned());
    if all_roots.is_empty() {
        return Arc::clone(node);
    }
    let Some((order, index, edges, tainted, eligible_map)) = graph(&all_roots) else {
        return Arc::clone(node);
    };
    let qualified = analyze(
        &roots,
        &conditional_roots,
        &order,
        &index,
        &edges,
        &tainted,
        &eligible_map,
    );
    if qualified.is_empty() {
        return Arc::clone(node);
    }
    let unique = select(qualified);
    if unique.is_empty() {
        return Arc::clone(node);
    }
    let Some(cols) = fresh_columns(&unique, &mut *next) else {
        return Arc::clone(node);
    };
    let map: Vec<(Arc<Expr>, BoundColumn)> = unique.iter().cloned().zip(cols).collect();
    let keys_out: Vec<Arc<Expr>> = normalized_keys.iter().map(|k| replace(k, &map)).collect();
    let rewritten_keys: Vec<(Arc<Expr>, Arc<Expr>)> = normalized_keys
        .iter()
        .zip(&keys_out)
        .filter(|(old, new)| !Arc::ptr_eq(old, new))
        .map(|(old, new)| (Arc::clone(old), Arc::clone(new)))
        .collect();
    let aggs_out: Vec<Arc<Expr>> = normalized_aggs
        .iter()
        .map(|a| replace_above(a, &map, &rewritten_keys))
        .collect();
    let mut read: Vec<Arc<Expr>> = keys_out.clone();
    read.extend(aggs_out.iter().cloned());
    let keep = passthrough(&read, input);
    let cols: Vec<BoundColumn> = map.into_iter().map(|(_, c)| c).collect();
    let base = absorb(input, &keep);
    let new_input = project_below(&base, &unique, &cols, &keep, node.origin);
    Arc::new(LogicalPlan {
        node: PlanNode::Aggregate {
            input: new_input,
            grouping: krishiv_plan::Grouping::GroupBy(keys_out),
            aggregates: aggs_out,
        },
        origin: node.origin,
    })
}

#[cfg(test)]
mod tests {
    use krishiv_plan::FunctionId;
    use krishiv_types::DataType;

    use super::*;

    fn col(id: u64) -> Arc<Expr> {
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
            BoundColumn {
                id: AttrId(id),
                name: format!("c{id}"),
                data_type: DataType::Integer,
                nullable: true,
            },
        ))))
    }

    fn scalar(name: &str, args: Vec<Arc<Expr>>) -> Arc<Expr> {
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: FunctionId::Registry {
                    name: name.to_owned(),
                    id: 7,
                },
                kind: FunctionKind::Scalar,
                args,
                distinct: false,
                filter: None,
                data_type: DataType::Integer,
                nullable: true,
                deterministic: true,
            },
        ))))
    }

    /// `children` marks exactly the conditional positions: the right side
    /// of `and`/`or`, non-first `coalesce`/`if`/`nanvl` arguments, every
    /// `WHEN` arm but the first condition, lambda bodies, and aggregate
    /// arguments under `FILTER (WHERE ...)`.
    #[test]
    fn conditional_positions() {
        let (a, b, c) = (col(0), col(1), col(2));
        let flags = |e: &Expr| {
            children(e)
                .iter()
                .map(|(_, cond)| *cond)
                .collect::<Vec<_>>()
        };
        let and = crate::expr::call(
            Builtin::And,
            vec![Arc::clone(&a), Arc::clone(&b)],
            DataType::Boolean,
        );
        assert_eq!(flags(&and), vec![false, true]);
        let coalesce = scalar(
            "coalesce",
            vec![Arc::clone(&a), Arc::clone(&b), Arc::clone(&c)],
        );
        assert_eq!(flags(&coalesce), vec![false, true, true]);
        let nanvl = scalar("nanvl", vec![Arc::clone(&a), Arc::clone(&b)]);
        assert_eq!(flags(&nanvl), vec![false, true]);
        let when = crate::expr::call(
            Builtin::When,
            vec![Arc::clone(&a), Arc::clone(&b), Arc::clone(&c)],
            DataType::Integer,
        );
        assert_eq!(flags(&when), vec![false, true, true]);
        let lambda = Arc::new(Expr::new(ExprNode::Lambda {
            params: vec![],
            body: Arc::clone(&a),
        }));
        assert_eq!(flags(&lambda), vec![true]);
        let agg = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: FunctionId::Registry {
                    name: "sum".to_owned(),
                    id: 9,
                },
                kind: FunctionKind::Aggregate,
                args: vec![Arc::clone(&a)],
                distinct: false,
                filter: Some(Arc::clone(&b)),
                data_type: DataType::Long,
                nullable: true,
                deterministic: true,
            },
        ))));
        assert_eq!(flags(&agg), vec![true, false]);
    }

    /// `nullif`/`nvl`/`ifnull`/`nvl2` resolve to their `when`/`coalesce`
    /// forms with the same arguments.
    #[test]
    fn null_forms_resolve() {
        let (a, b, c) = (col(0), col(1), col(2));
        let top = |e: &Arc<Expr>| match &normalize(e).node {
            ExprNode::Resolved(ResolvedExpr::Function(f)) => (f.id.name().to_owned(), f.args.len()),
            other => panic!("not a call: {other:?}"),
        };
        assert_eq!(
            top(&scalar("nullif", vec![Arc::clone(&a), Arc::clone(&b)])),
            ("when".to_owned(), 3)
        );
        for name in ["nvl", "ifnull"] {
            assert_eq!(
                top(&scalar(name, vec![Arc::clone(&a), Arc::clone(&b)])),
                ("coalesce".to_owned(), 2)
            );
        }
        assert_eq!(
            top(&scalar(
                "nvl2",
                vec![Arc::clone(&a), Arc::clone(&b), Arc::clone(&c)]
            )),
            ("when".to_owned(), 3)
        );
        // Anything else is left alone (the same node back).
        let coalesce = scalar("coalesce", vec![Arc::clone(&a)]);
        assert!(Arc::ptr_eq(&normalize(&coalesce), &coalesce));
    }

    /// `try_*` calls, `TRY_CAST`s and their subtrees never extract.
    #[test]
    fn try_marks() {
        let a = col(0);
        let try_add = scalar("try_add", vec![Arc::clone(&a)]);
        assert!(is_try(&try_add));
        assert!(is_try(&Arc::new(Expr::new(ExprNode::Cast {
            expr: Arc::clone(&a),
            to: DataType::Integer,
            mode: CastMode::Try,
        }))));
        assert!(!is_try(&a));
        assert!(!eligible(&try_add));
        // A `cse_*` alias is never extracted itself.
        let alias = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
            expr: Arc::clone(&a),
            output: BoundColumn {
                id: AttrId(99),
                name: "cse_99".to_owned(),
                data_type: DataType::Integer,
                nullable: true,
            },
        })));
        assert!(!eligible(&alias));
        assert!(!eligible(&a));
    }

    /// Fingerprints ignore where and how often a node occurs: equal
    /// structures share one, and the encoding order is deterministic.
    #[test]
    fn fingerprints_are_structural() {
        let (a, b) = (col(0), col(1));
        let x = crate::expr::call(
            Builtin::Add,
            vec![Arc::clone(&a), Arc::clone(&b)],
            DataType::Integer,
        );
        let y = crate::expr::call(
            Builtin::Add,
            vec![Arc::clone(&a), Arc::clone(&b)],
            DataType::Integer,
        );
        let g = |e: &Arc<Expr>| {
            let (order, _, edges, _, _) = graph(std::slice::from_ref(e)).unwrap();
            fingerprints(&order, &edges)[order.len() - 1]
        };
        assert_eq!(g(&x), g(&y));
        let z = crate::expr::call(
            Builtin::Add,
            vec![Arc::clone(&b), Arc::clone(&a)],
            DataType::Integer,
        );
        assert_ne!(g(&x), g(&z));
    }
}
