// SPDX-License-Identifier: Apache-2.0
//! `SubqueryAsWindow` (P3 W1 shape (a), form M; `design/P2b-W1-subquery-one-scan.md` section 2):
//! a decorrelated scalar aggregate subquery over the outer block's own relations becomes a
//! window over the outer rows.
//!
//! ```text
//! Join Inner (k = e AND R)                        Project [J's output]
//! :- X                                       =>   +- Filter R'
//! +- Aggregate [k] (calls F) over S                  +- Window [w := f(args) OVER (PARTITION BY P)]
//!                                                       +- X
//! ```
//!
//! where `S` reads the same relations as the part `X_S` of `X`, under the same filters, and `k`
//! is `X`'s partition key `P`. Every `S` row of a group `p` then appears in `X` exactly `m_p`
//! times (form M: the rest of `X`, `X_D`, joins `X_S` only through equalities on the partition
//! key's class), and `min`, `max` and exact decimal `avg` do not change under uniform
//! duplication. This is TPC-H q17 and q02: the second scan and its join disappear.
//!
//! Anything the rule cannot prove equal leaves the plan as it is. The refusals are numbered as in
//! the design note (section 2.2); form J (`sum`, `count`) is not part of this rule yet.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FrameUnits, FunctionKind, Grouping,
    JoinCondition, JoinKind, LogicalPlan, PlanNode, ResolvedExpr, ResolvedFunction, WindowExpr,
    WindowFrame, WindowSpec, WindowSpecRef,
};
use krishiv_types::DataType;

use crate::dag;
use crate::eager::refusal_of_call;
use crate::estimate::Estimator;
use crate::expr::{
    Classes, and_all, builtin, column, conjuncts, data_type, equality, equality_classes,
    has_outer_reference, has_subquery, is_deterministic, references, same, strip_alias, substitute,
};
use crate::plan_stats::default_size;
use crate::restrict::{MAX_ID, max_id};
use crate::stats::{RelStats, Statistics};

/// Rewrites the joins of `root` that are a subquery-as-window (see the module docs).
pub(crate) fn rewrite(
    root: Arc<LogicalPlan>,
    stats: &dyn Statistics,
    propagate: bool,
) -> Result<(Arc<LogicalPlan>, Vec<String>), KrishivError> {
    let mut r = Rule {
        est: Estimator::with_propagation(stats, propagate),
        root: Arc::clone(&root),
        next: None,
        notes: Vec::new(),
    };
    let out = dag::rewrite_up(root, &mut |p| Ok(r.join(&p).unwrap_or(p)))?;
    Ok((out, r.notes))
}

struct Rule<'a> {
    est: Estimator<'a>,
    root: Arc<LogicalPlan>,
    /// The next unused attribute id, computed at the first firing.
    next: Option<u64>,
    /// One line per firing and per cost-guard refusal, for `EXPLAIN VERBOSE`.
    notes: Vec<String>,
}

/// A relation both blocks read: the body under its `Shift`s, and what those add to every id.
struct Leaf {
    body: Arc<LogicalPlan>,
    offset: u64,
}

/// An inner-join block flattened: its leaves, and its conjuncts in the ids of the block's output.
#[derive(Default)]
struct Block {
    leaves: Vec<Leaf>,
    conjuncts: Vec<Arc<Expr>>,
}

impl Rule<'_> {
    fn join(&mut self, p: &Arc<LogicalPlan>) -> Option<Arc<LogicalPlan>> {
        let PlanNode::Join {
            left,
            right,
            kind: JoinKind::Inner,
            condition: JoinCondition::On(cond),
        } = &p.node
        else {
            return None;
        };
        self.fire(p, left, right, cond)
            .or_else(|| self.fire(p, right, left, cond))
    }

    /// `x` the outer block, `a` the aggregate side.
    fn fire(
        &mut self,
        join: &Arc<LogicalPlan>,
        x: &Arc<LogicalPlan>,
        a: &Arc<LogicalPlan>,
        cond: &Arc<Expr>,
    ) -> Option<Arc<LogicalPlan>> {
        // The aggregate, under filters that read only its own outputs (they join `R`).
        let mut above = Vec::new();
        let mut node = a;
        while let PlanNode::Filter { input, predicate } = &node.node {
            if !is_deterministic(predicate) || has_subquery(predicate) {
                return None;
            }
            above.extend(conjuncts(predicate));
            node = input;
        }
        let PlanNode::Aggregate {
            input: s,
            grouping: Grouping::GroupBy(keys),
            aggregates,
        } = &node.node
        else {
            return None;
        };
        if keys.is_empty() {
            return None;
        }
        // Group keys are plain columns of `S`.
        let keys: Vec<&BoundColumn> = keys.iter().map(|k| plain(k)).collect::<Option<_>>()?;
        let key_ids: HashSet<AttrId> = keys.iter().map(|k| k.id).collect();
        // The outputs: a key as it is, or an expression over calls and keys.
        let agg_cols = a.resolved_output()?;
        let a_ids: HashSet<AttrId> = agg_cols.iter().map(|c| c.id).collect();
        let mut outputs: Vec<(BoundColumn, Arc<Expr>)> = Vec::new();
        let mut calls: Vec<Arc<Expr>> = Vec::new();
        for e in aggregates {
            let (out, inner) = match &e.node {
                ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => {
                    (output.clone(), Arc::clone(expr))
                }
                ExprNode::Resolved(ResolvedExpr::Column(c)) => (c.clone(), Arc::clone(e)),
                _ => return None,
            };
            collect_calls(&inner, &mut calls)?;
            outputs.push((out, inner));
        }
        if calls.is_empty() {
            return None;
        }
        // Calls: min, max, decimal avg only (form M), order free and unable to raise.
        let none = RelStats::default();
        for c in &calls {
            let f = function(c)?;
            if f.distinct || f.filter.is_some() || !f.deterministic {
                return None;
            }
            let ok = match f.id.name() {
                "min" | "max" => true,
                "avg" | "mean" => matches!(
                    f.args.first().and_then(|a| data_type(a)),
                    Some(DataType::Decimal(_))
                ),
                _ => false,
            };
            if !ok || refusal_of_call(f, &none, false).is_err() {
                return None;
            }
        }
        // The condition: `k = e` for every key once, the rest is `R`.
        let x_cols = x.resolved_output()?;
        let x_ids: HashSet<AttrId> = x_cols.iter().map(|c| c.id).collect();
        let mut pairs: HashMap<AttrId, &BoundColumn> = HashMap::new();
        let mut rest: Vec<Arc<Expr>> = above;
        let cond_parts = conjuncts(cond);
        let eqs: Vec<Option<[&BoundColumn; 2]>> = cond_parts.iter().map(|c| equality(c)).collect();
        for (part, eq) in cond_parts.iter().zip(&eqs) {
            let pair =
                eq.and_then(
                    |[l, r]| match (key_ids.contains(&l.id), key_ids.contains(&r.id)) {
                        (true, false) if x_ids.contains(&r.id) => Some((l.id, r)),
                        (false, true) if x_ids.contains(&l.id) => Some((r.id, l)),
                        _ => None,
                    },
                );
            match pair {
                Some((k, e)) if !pairs.contains_key(&k) => {
                    pairs.insert(k, e);
                }
                _ => rest.push(Arc::clone(part)),
            }
        }
        if pairs.len() != keys.len() {
            return None;
        }
        // `R` reads `X`'s columns and the aggregate's non-key outputs, never a key.
        for r in &rest {
            if !is_deterministic(r) || has_subquery(r) || has_outer_reference(r) {
                return None;
            }
            let refs = references(r);
            if refs.iter().any(|id| key_ids.contains(id))
                || !refs
                    .iter()
                    .all(|id| x_ids.contains(id) || a_ids.contains(id))
            {
                return None;
            }
        }
        // The blocks.
        let sb = flatten_block(s)?;
        let xb = flatten_block(x)?;
        // Leaves: every leaf of `S` is one leaf of `X`, injectively.
        let mut taken: HashSet<usize> = HashSet::new();
        let mut to_x: HashMap<AttrId, Arc<Expr>> = HashMap::new();
        let mut s_leaf_x: Vec<usize> = Vec::new();
        for sl in &sb.leaves {
            let found: Vec<usize> = (0..xb.leaves.len())
                .filter(|i| equal(&xb.leaves[*i].body, &sl.body))
                .collect();
            let [i] = found[..] else { return None };
            if !taken.insert(i) {
                return None;
            }
            s_leaf_x.push(i);
            let xl = &xb.leaves[i];
            for c in sl.body.resolved_output()? {
                let from = c.id.0.checked_add(sl.offset)?;
                let to = BoundColumn {
                    id: AttrId(c.id.0.checked_add(xl.offset)?),
                    ..c
                };
                to_x.insert(AttrId(from), column(&to));
            }
        }
        // `X_S`'s columns.
        let mut xs_ids: HashSet<AttrId> = HashSet::new();
        for i in &taken {
            let l = &xb.leaves[*i];
            for c in l.body.resolved_output()? {
                xs_ids.insert(AttrId(c.id.0.checked_add(l.offset)?));
            }
        }
        // `S`'s conjuncts and the partition key in `X`'s ids.
        let s_conj: Vec<Arc<Expr>> = sb.conjuncts.iter().map(|c| substitute(c, &to_x)).collect();
        let p_cols: Vec<BoundColumn> = keys
            .iter()
            .map(|k| plain(&substitute(&column(k), &to_x)).cloned())
            .collect::<Option<_>>()?;
        for p in &p_cols {
            if !partition_type(&p.data_type) {
                return None;
            }
        }
        // Each `e` is `P_k` or equal to it within `X`.
        let mut x_classes = equality_classes(&xb.conjuncts);
        for (k, p) in keys.iter().zip(&p_cols) {
            let e = pairs.get(&k.id)?;
            if e.id != p.id && x_classes.find(e.id) != x_classes.find(p.id) {
                return None;
            }
        }
        // `Cls(P)`: the class of the partition key over `S`'s own equalities.
        let mut s_classes = equality_classes(&s_conj);
        let in_cls =
            |c: &mut Classes, id: AttrId| p_cols.iter().any(|p| c.find(p.id) == c.find(id));
        // Condition 4: equal row sets per group.
        let null_ignoring: Option<HashSet<AttrId>> = {
            // `isnotnull(c)` is harmless when every call's sole argument is `c`.
            let mut set = HashSet::new();
            let mut all = true;
            for c in &calls {
                match function(c).map(|f| f.args.as_slice()) {
                    Some([arg]) => match plain(&substitute(arg, &to_x)) {
                        Some(col) => {
                            set.insert(col.id);
                        }
                        None => all = false,
                    },
                    _ => all = false,
                }
            }
            (all && set.len() == 1).then_some(set)
        };
        let mut matched_s: HashSet<usize> = HashSet::new();
        let (mut only_d, mut mixed): (Vec<&Arc<Expr>>, Vec<&Arc<Expr>>) = (Vec::new(), Vec::new());
        for xc in &xb.conjuncts {
            let refs = references(xc);
            if !has_subquery(xc) && !refs.is_empty() && refs.iter().all(|id| xs_ids.contains(id)) {
                if let Some(i) = s_conj.iter().position(|s| same(s, xc)) {
                    matched_s.insert(i);
                    continue;
                }
                let is_notnull_of = matches!(
                    builtin(xc),
                    Some((Builtin::IsNotNull, [c], _))
                        if plain(c).is_some_and(|c| null_ignoring.as_ref().is_some_and(|s| s.contains(&c.id)))
                );
                if is_notnull_of || refs.iter().all(|id| in_cls(&mut s_classes, *id)) {
                    continue;
                }
                return None;
            }
            if refs.iter().any(|id| xs_ids.contains(id)) {
                mixed.push(xc);
            } else {
                only_d.push(xc);
            }
        }
        if matched_s.len() != s_conj.len() {
            // A conjunct of `S` that `X` lacks (equal conjuncts count once).
            let missing = s_conj
                .iter()
                .enumerate()
                .any(|(i, s)| !matched_s.contains(&i) && !xb.conjuncts.iter().any(|x| same(x, s)));
            if missing {
                return None;
            }
        }
        // Form M: `X_D` meets `X_S` only through `d = p` with `p` in `Cls(P)`.
        for m in mixed {
            let [l, r] = equality(m)?;
            let (s_side, d_side) = if xs_ids.contains(&l.id) {
                (l, r)
            } else {
                (r, l)
            };
            if !xs_ids.contains(&s_side.id)
                || xs_ids.contains(&d_side.id)
                || !in_cls(&mut s_classes, s_side.id)
            {
                return None;
            }
        }
        let _ = only_d;
        // The guard (design 2.4): the bytes the window sorts must not exceed the bytes the second
        // scan reads. An estimate without statistics is no basis: no firing.
        let outs = join.resolved_output()?;
        let x_est = self.est.plan(x);
        if !x_est.from_statistics {
            return None;
        }
        let window_bytes = x_est.rows_for_costing()
            * outs
                .iter()
                .map(|c| default_size(&c.data_type) as f64)
                .sum::<f64>();
        let mut used: HashSet<AttrId> = HashSet::new();
        for c in &sb.conjuncts {
            used.extend(references(c));
        }
        used.extend(keys.iter().map(|k| k.id));
        for c in &calls {
            used.extend(references(c));
        }
        let mut scan_bytes = 0.0;
        for sl in &sb.leaves {
            let est = self.est.plan(&sl.body);
            let width: f64 = sl
                .body
                .resolved_output()?
                .iter()
                .filter(|c| used.contains(&AttrId(c.id.0 + sl.offset)))
                .map(|c| default_size(&c.data_type) as f64)
                .sum();
            scan_bytes += est.rows_for_costing() * width;
        }
        if window_bytes > scan_bytes {
            self.notes.push(format!(
                "SubqueryAsWindow: refused by the cost guard (window ~{window_bytes:.0} bytes > second scan ~{scan_bytes:.0} bytes)"
            ));
            return None;
        }
        self.notes.push(format!(
            "SubqueryAsWindow: form M, window over the outer block (window ~{window_bytes:.0} bytes <= second scan ~{scan_bytes:.0} bytes)"
        ));
        // The plan.
        let first = match self.next {
            Some(n) => n,
            None => max_id(&self.root, &mut Default::default()).checked_add(1)?,
        };
        let mut next = first;
        let mut fresh = || {
            let id = next;
            next += 1;
            (id <= MAX_ID).then_some(AttrId(id))
        };
        let spec_partition: Vec<Arc<Expr>> = p_cols.iter().map(column).collect();
        let mut window_exprs = Vec::new();
        let mut call_cols: Vec<(Arc<Expr>, Arc<Expr>)> = Vec::new();
        for c in &calls {
            let f = function(c)?;
            let translated = substitute(c, &to_x);
            let w = BoundColumn {
                id: fresh()?,
                name: f.id.name().to_owned(),
                data_type: f.data_type.clone(),
                nullable: f.nullable,
            };
            window_exprs.push(Arc::new(Expr::new(ExprNode::Resolved(
                ResolvedExpr::Alias {
                    expr: Arc::new(Expr::new(ExprNode::Window(WindowExpr {
                        function: translated,
                        window: WindowSpecRef::Spec(WindowSpec {
                            partition_by: spec_partition.clone(),
                            order_by: Vec::new(),
                            frame: Some(WindowFrame::unbounded(FrameUnits::Rows)),
                        }),
                    }))),
                    output: w.clone(),
                },
            ))));
            call_cols.push((Arc::clone(c), column(&w)));
        }
        self.next = Some(next);
        // Each aggregate output as an expression over the window columns.
        let mut value: HashMap<AttrId, Arc<Expr>> = HashMap::new();
        for (out, inner) in &outputs {
            let over = replace_calls(inner, &call_cols);
            let over = substitute(&over, &to_x);
            value.insert(out.id, over);
        }
        for (k, p) in keys.iter().zip(&p_cols) {
            value.insert(k.id, column(p));
        }
        let windowed = Arc::new(LogicalPlan::new(PlanNode::Window {
            input: Arc::clone(x),
            exprs: window_exprs,
            partition_by: spec_partition,
            order_by: Vec::new(),
        }));
        let kept: Vec<Arc<Expr>> = rest.iter().map(|r| substitute(r, &value)).collect();
        let filtered = match and_all(kept) {
            Some(predicate) => Arc::new(LogicalPlan::new(PlanNode::Filter {
                input: windowed,
                predicate,
            })),
            None => windowed,
        };
        let exprs: Vec<Arc<Expr>> = outs
            .iter()
            .map(|o| match value.get(&o.id) {
                Some(v) => Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                    expr: Arc::clone(strip_alias(v)),
                    output: o.clone(),
                }))),
                None => column(o),
            })
            .collect();
        Some(Arc::new(LogicalPlan::new(PlanNode::Project {
            input: filtered,
            exprs,
        })))
    }
}

/// Key types whose grouping and equality agree with the row's value (integral, decimal, date,
/// timestamp, string under the binary collation).
fn partition_type(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::Decimal(_)
            | DataType::Date
            | DataType::Timestamp
            | DataType::TimestampNtz
            | DataType::String
    )
}

fn equal(a: &Arc<LogicalPlan>, b: &Arc<LogicalPlan>) -> bool {
    Arc::ptr_eq(a, b) || **a == **b
}

fn plain(e: &Expr) -> Option<&BoundColumn> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c),
        _ => None,
    }
}

fn function(e: &Expr) -> Option<&ResolvedFunction> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate => {
            Some(f)
        }
        _ => None,
    }
}

/// The aggregate calls under `e`, once each; `None` for a call nested in a call or a window.
fn collect_calls(e: &Arc<Expr>, out: &mut Vec<Arc<Expr>>) -> Option<()> {
    if function(e).is_some() {
        if !out.iter().any(|c| same(c, e)) {
            out.push(Arc::clone(e));
        }
        return Some(());
    }
    if matches!(e.node, ExprNode::Window(_)) {
        return None;
    }
    let mut ok = true;
    e.for_each_child(&mut |c| ok &= collect_calls(c, out).is_some());
    ok.then_some(())
}

fn replace_calls(e: &Arc<Expr>, calls: &[(Arc<Expr>, Arc<Expr>)]) -> Arc<Expr> {
    let out = Arc::clone(e).transform_up(&mut |x| {
        Ok::<_, std::convert::Infallible>(
            calls
                .iter()
                .find(|(c, _)| function(&x).is_some() && same(c, &x))
                .map_or(x, |(_, w)| Arc::clone(w)),
        )
    });
    match out {
        Ok(v) => v,
        Err(never) => match never {},
    }
}

/// `e` with every column id raised by `offset`.
fn shift_up(e: &Arc<Expr>, offset: u64) -> Option<Arc<Expr>> {
    if offset == 0 {
        return Some(Arc::clone(e));
    }
    let out: Result<_, ()> = Arc::clone(e).transform_up(&mut |x| match &x.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => {
            let id = AttrId(c.id.0.checked_add(offset).ok_or(())?);
            Ok(column(&BoundColumn { id, ..c.clone() }))
        }
        _ => Ok(x),
    });
    out.ok()
}

/// `p` as leaves and conjuncts of an inner-join block (through filters, inner joins, column-only
/// projections and shifts); `None` when anything else the rule must not look through is in it.
fn flatten_block(p: &Arc<LogicalPlan>) -> Option<Block> {
    let mut block = Block::default();
    walk(p, 0, &mut block)?;
    Some(block)
}

fn walk(p: &Arc<LogicalPlan>, ctx: u64, out: &mut Block) -> Option<()> {
    let add = |pred: &Arc<Expr>, out: &mut Block| -> Option<()> {
        if !is_deterministic(pred) || has_subquery(pred) || has_outer_reference(pred) {
            return None;
        }
        for c in conjuncts(pred) {
            out.conjuncts.push(shift_up(&c, ctx)?);
        }
        Some(())
    };
    match &p.node {
        PlanNode::Filter { input, predicate } => {
            add(predicate, out)?;
            walk(input, ctx, out)
        }
        PlanNode::Join {
            left,
            right,
            kind: JoinKind::Inner,
            condition,
        } => {
            match condition {
                JoinCondition::On(c) => add(c, out)?,
                JoinCondition::None => {}
                _ => return None,
            }
            walk(left, ctx, out)?;
            walk(right, ctx, out)
        }
        PlanNode::Project { input, exprs } if exprs.iter().all(|e| plain(e).is_some()) => {
            walk(input, ctx, out)
        }
        PlanNode::Shift { input, offset } => walk(input, ctx.checked_add(*offset)?, out),
        PlanNode::Limit { .. } | PlanNode::Aggregate { .. } | PlanNode::Window { .. } => None,
        _ => {
            // A leaf: a scan or another relation the block reads whole.
            let mut det = true;
            p.for_each_expr(&mut |e| det &= is_deterministic(e));
            det.then(|| {
                out.leaves.push(Leaf {
                    body: Arc::clone(p),
                    offset: ctx,
                });
            })
        }
    }
}
