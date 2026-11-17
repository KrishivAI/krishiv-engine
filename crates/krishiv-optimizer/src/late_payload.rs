// SPDX-License-Identifier: Apache-2.0
//! `LatePayloadJoin`: a dimension's payload columns join above the semi and
//! anti joins that never read them (P2b S5 lever (2); design
//! `design/P2b-S5-dimension-payload-late.md` in krishiv-internal).
//!
//! ```text
//! Consumer(Path(X ⋈[x = d] D))
//!   =>  Consumer(Project[out(Path)](Path'(X ⋉[x = d'] Shift(π_d D)) ⋈[x = d] D))
//! ```
//!
//! `D` is one side of an inner equi-join whose other columns (the payload,
//! TPC-H q21's `s_name`) ride through every build of the path above it and are
//! first read by the consumer. Each operator on the path commutes with an
//! inner join on columns it does not read: `(X ⋈ D) ⋉ L = (X ⋉ L) ⋈ D` when
//! `L`'s condition reads only `X`, the same for an anti join, an existence
//! join and the preserved side of an outer join, and filters and projections
//! trivially. So the join moves to the top of the path with the same rows,
//! each `X` row joined to every matching `D` row whatever the multiplicity;
//! no key needs to be unique. The semi join left in its place drops only the
//! rows the top join drops (no match, NULL keys included), so the path keeps
//! the reduction the dimension gave it. `D` is computed twice, the semi
//! join's instance under a [`PlanNode::Shift`] as in `restrict`, so it must
//! be repeatable (`restrict::repeatable`).
//!
//! # When it applies
//!
//! - The join is inner, its condition only plain `=` of a column of each side,
//!   of one type whose equality and grouping agree;
//! - the path from the join to its consumer holds only filters, projections
//!   and joins that do not read the payload or `D`'s key columns, with the path
//!   on the side whose rows they keep (inner either side; left semi, anti,
//!   existence and outer on the left; right outer on the right), and every
//!   expression on it is deterministic (a filter evaluated once per `X` row
//!   instead of once per joined copy would sample differently);
//! - at least one join on the path is not inner: join ordering never moves an
//!   inner join across one, so it cannot undo the rule, and once applied the
//!   rule finds no such join between the moved join and its consumer: the
//!   plan stays a fixed point. An all-inner path is join ordering's business;
//! - the join and the path's nodes are read by no other parent (rebuilding
//!   them would copy a shared subtree);
//! - gates, from statistics only (both sides' estimates `from_statistics`,
//!   never default row counts): the payload the path carries holds a
//!   variable-width column or at least [`MIN_PAYLOAD_BYTES`] a row; `D`
//!   estimates no more rows than `X`, and its scans read no more rows than
//!   `X` holds (it is computed twice); and some join on the path builds the
//!   path's side as the physical planner orients it (`builds_path`: an inner
//!   join's right input, or the left input of a semi, anti or left outer join
//!   whose right input costs more than twice as much and is not broadcast). A
//!   payload on a probe side only streams, and moving it would only add work.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, JoinCondition, JoinKind, LogicalPlan, PlanNode,
};
use krishiv_types::DataType;

use crate::dag;
use crate::estimate::Estimator;
use crate::expr::{
    Attrs, builtin, column, conjuncts, is_deterministic, output_of, recall, references,
};
use crate::restrict::{MAX_ID, key_type, max_id, repeatable, shift};
use crate::stats::Statistics;

/// The least fixed width, in bytes a row, of a payload worth moving.
pub(crate) const MIN_PAYLOAD_BYTES: usize = 16;

/// The most joins moved below one consumer.
const MAX_MOVES: usize = 16;

/// The most plan nodes one walk down from a consumer visits.
const MAX_WALK: usize = 256;

/// Moves dimension payload joins above the non-inner joins of their path,
/// entering subquery plans. Also returns one line per move, and per refused
/// candidate the gate that refused it (`EXPLAIN VERBOSE`).
///
/// `broadcast_max` is the physical planner's broadcast threshold
/// (`spark.sql.autoBroadcastJoinThreshold`, `None` when off): a join that
/// broadcasts its other side never builds the path.
pub(crate) fn rewrite(
    root: Arc<LogicalPlan>,
    stats: &dyn Statistics,
    propagate: bool,
    broadcast_max: Option<f64>,
) -> Result<(Arc<LogicalPlan>, Vec<String>), KrishivError> {
    let mut ctx = Context {
        next: max_id(&root, &mut crate::expr::AddrMap::default()).saturating_add(1),
        notes: Vec::new(),
        shared: shared_nodes(&root),
        broadcast_max,
    };
    let mut est = Estimator::with_propagation(stats, propagate);
    let out = dag::rewrite_up(root, &mut |p| {
        // Several dimensions may carry payloads to one consumer: move each,
        // so one pass reaches the fixed point.
        let mut at = p;
        for _ in 0..MAX_MOVES {
            match at_consumer(&at, &mut est, &mut ctx) {
                Some(moved) => at = moved,
                None => break,
            }
        }
        Ok(at)
    })?;
    Ok((out, ctx.notes))
}

/// What one pass carries between consumers.
struct Context {
    /// The next free attribute id.
    next: u64,
    /// `EXPLAIN VERBOSE` lines, each once.
    notes: Vec<String>,
    /// Nodes read by more than one parent in the input plan.
    shared: HashSet<*const LogicalPlan>,
    broadcast_max: Option<f64>,
}

/// Appends `line`; a refusal only once (the moves below one consumer walk
/// the same refused candidates again), every move.
fn note(notes: &mut Vec<String>, line: String) {
    if line.starts_with("moved") || !notes.contains(&line) {
        notes.push(line);
    }
}

/// The nodes of `root` that more than one parent reads (subquery plans not
/// entered: the rule rewrites inside them as their own roots).
fn shared_nodes(root: &Arc<LogicalPlan>) -> HashSet<*const LogicalPlan> {
    let mut parents: HashMap<*const LogicalPlan, usize> = HashMap::new();
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(root)];
    while let Some(p) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&p)) {
            continue;
        }
        p.for_each_child(&mut |c| {
            *parents.entry(Arc::as_ptr(c)).or_default() += 1;
            stack.push(Arc::clone(c));
        });
    }
    parents
        .into_iter()
        .filter(|(_, n)| *n > 1)
        .map(|(p, _)| p)
        .collect()
}

/// Whether the rule can put a new child under `p` (`with_child`).
fn replaceable(p: &LogicalPlan) -> bool {
    matches!(
        p.node,
        PlanNode::Join { .. }
            | PlanNode::Project { .. }
            | PlanNode::Filter { .. }
            | PlanNode::Aggregate { .. }
            | PlanNode::Sort { .. }
            | PlanNode::Limit { .. }
            | PlanNode::Offset { .. }
            | PlanNode::SubqueryAlias { .. }
            | PlanNode::WithCte { .. }
            | PlanNode::Shift { .. }
            | PlanNode::JoinHint { .. }
            | PlanNode::Repartition { .. }
            | PlanNode::Window { .. }
            | PlanNode::Generate { .. }
            | PlanNode::ReplicateRows { .. }
    )
}

/// How much larger the other side's table must be before the physical
/// planner builds a semi, anti or left outer join's left input
/// (`krishiv-exec` `planner/orient.rs` `STATS_FACTOR`, both sides from
/// statistics; `DEFAULT_FACTOR` otherwise).
const STATS_FACTOR: f64 = 2.0;
const DEFAULT_FACTOR: f64 = 8.0;

/// Whether the physical planner builds the hash table of `s`'s join from the
/// path's input (`krishiv-exec` `planner/join.rs` and `planner/orient.rs`):
/// an inner join's right input, or the left input of a semi, anti (not
/// null-aware) or left outer join whose right input costs more than the
/// factor times as much and is not broadcast. Existence, null-aware anti and
/// right outer joins never build their left input.
fn builds_path(s: &Step, est: &mut Estimator<'_>, broadcast_max: Option<f64>) -> bool {
    let PlanNode::Join {
        kind,
        condition: JoinCondition::On(on),
        ..
    } = &s.node.node
    else {
        return false;
    };
    let kids = children(&s.node);
    let (Some(path), Some(other)) = (kids.get(s.child), kids.get(usize::from(s.child == 0))) else {
        return false;
    };
    match kind {
        JoinKind::Inner => s.child == 1,
        JoinKind::LeftSemi | JoinKind::LeftOuter => {
            s.child == 0 && flips(path, other, est, broadcast_max)
        }
        JoinKind::LeftAnti => {
            s.child == 0 && !null_aware(on) && flips(path, other, est, broadcast_max)
        }
        _ => false,
    }
}

/// `orient.rs`'s decision to build the left input over `right`.
fn flips(
    left: &Arc<LogicalPlan>,
    right: &Arc<LogicalPlan>,
    est: &mut Estimator<'_>,
    broadcast_max: Option<f64>,
) -> bool {
    let (l, r) = (est.plan(left), est.plan(right));
    let (lb, rb) = (bytes(left, l.rows), bytes(right, r.rows));
    if broadcast_max.is_some_and(|max| rb <= max) {
        return false;
    }
    let factor = if l.from_statistics && r.from_statistics {
        STATS_FACTOR
    } else {
        DEFAULT_FACTOR
    };
    rb > factor * lb
}

/// Spark's size estimate of `rows` rows of `p`'s output.
fn bytes(p: &LogicalPlan, rows: f64) -> f64 {
    let width = p
        .resolved_output()
        .unwrap_or_default()
        .iter()
        .map(|c| crate::plan_stats::default_size(&c.data_type))
        .fold(8, u64::saturating_add);
    #[expect(clippy::cast_precision_loss, reason = "an estimate")]
    let width = width as f64;
    rows.max(0.0) * width
}

/// Whether `on` is Spark's null-aware anti-join key `(a = b) OR isnull(a = b)`.
fn null_aware(on: &Arc<Expr>) -> bool {
    let conj = conjuncts(on);
    let [only] = conj.as_slice() else {
        return false;
    };
    matches!(builtin(only), Some((Builtin::Or, [_, is_null], _))
        if matches!(builtin(is_null), Some((Builtin::IsNull, _, _))))
}

/// The rows the scans of `p` read, as estimated.
fn scanned_rows(p: &Arc<LogicalPlan>, est: &mut Estimator<'_>) -> f64 {
    let mut total = 0.0;
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(p)];
    while let Some(x) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&x)) {
            continue;
        }
        if matches!(x.node, PlanNode::Scan(_)) {
            total += est.plan(&x).rows;
        }
        x.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    total
}

/// One step of a path: a node and the index of the child the path came from.
#[derive(Clone)]
struct Step {
    node: Arc<LogicalPlan>,
    child: usize,
}

/// A join that may move: the path from just above it to the consumer's child
/// (bottom first), the join, and which side is the dimension.
struct Candidate {
    path: Vec<Step>,
    join: Arc<LogicalPlan>,
    dim_left: bool,
}

/// `consumer` with one payload join below it moved up, if any qualifies.
fn at_consumer(
    consumer: &Arc<LogicalPlan>,
    est: &mut Estimator<'_>,
    ctx: &mut Context,
) -> Option<Arc<LogicalPlan>> {
    let kids = children(consumer);
    if !replaceable(consumer) {
        return None;
    }
    let (next, notes) = (&mut ctx.next, &mut ctx.notes);
    for (i, kid) in kids.iter().enumerate() {
        let mut found = Vec::new();
        let mut budget = MAX_WALK;
        walk(kid, &mut Vec::new(), false, &mut found, &mut budget);
        for cand in found {
            if cand
                .path
                .iter()
                .any(|s| ctx.shared.contains(&Arc::as_ptr(&s.node)))
                || ctx.shared.contains(&Arc::as_ptr(&cand.join))
            {
                // Rebuilding a node other parents read would copy it.
                continue;
            }
            let Some(shape) = Shape::of(&cand, ctx.broadcast_max) else {
                continue;
            };
            // The consumer must be where the path ends for this payload: a
            // node the payload could pass is not the consumer (its parent is).
            if passes(consumer, i, &shape.blocked) {
                continue;
            }
            // Nothing of the dimension above the join: nothing to move.
            if shape.carried(&cand).is_empty() {
                continue;
            }
            if let Err(why) = shape.gates(&cand, est) {
                note(
                    notes,
                    format!("not applied to the join on [{}]: {why}", shape.keys_text()),
                );
                continue;
            }
            let Some(moved) = shape.moved(&cand, kid, next) else {
                note(
                    notes,
                    "not applied: the plan has no attribute ids left".to_owned(),
                );
                continue;
            };
            let out = with_child(consumer, i, moved);
            note(
                notes,
                format!(
                    "moved the join on [{}] above {} non-inner join(s), payload [{}]",
                    shape.keys_text(),
                    cand.path
                        .iter()
                        .filter(|s| is_non_inner_join(&s.node))
                        .count(),
                    shape.payload_text(),
                ),
            );
            return Some(out);
        }
    }
    None
}

/// Every inner join reachable from `at` through pass-through nodes with a
/// non-inner join on the path above it (nothing else can move), with that
/// path. `path` holds the nodes from the consumer's child down to `at`'s
/// parent (top first while walking); `non_inner` says it holds a non-inner
/// join, so a plan of inner joins only yields no candidate to read.
fn walk(
    at: &Arc<LogicalPlan>,
    path: &mut Vec<Step>,
    non_inner: bool,
    found: &mut Vec<Candidate>,
    budget: &mut usize,
) {
    if *budget == 0 {
        return;
    }
    *budget -= 1;
    if let PlanNode::Join {
        kind: JoinKind::Inner,
        ..
    } = &at.node
        && non_inner
    {
        for dim_left in [true, false] {
            found.push(Candidate {
                path: path.iter().rev().cloned().collect(),
                join: Arc::clone(at),
                dim_left,
            });
        }
    }
    // Continue down through the children a path may come from (the payload is
    // not known yet, so `passes` is checked once it is).
    let kids = children(at);
    for (i, kid) in kids.iter().enumerate() {
        if may_pass(at, i) {
            path.push(Step {
                node: Arc::clone(at),
                child: i,
            });
            walk(kid, path, non_inner || is_non_inner_join(at), found, budget);
            path.pop();
        }
    }
}

/// Whether a path may come up through `node` from child `i`, whatever it
/// carries: the node kinds and join sides of the module docs.
fn may_pass(node: &LogicalPlan, i: usize) -> bool {
    match &node.node {
        PlanNode::Project { .. } | PlanNode::Filter { .. } => i == 0,
        PlanNode::Join {
            kind, condition, ..
        } => {
            let det = match condition {
                JoinCondition::On(on) => is_deterministic(on),
                _ => false,
            };
            det && match kind {
                JoinKind::Inner => true,
                JoinKind::LeftSemi
                | JoinKind::LeftAnti
                | JoinKind::LeftOuter
                | JoinKind::Existence(_) => i == 0,
                JoinKind::RightOuter => i == 1,
                _ => false,
            }
        }
        _ => false,
    }
}

/// Whether a path carrying the columns `blocked` (the payload and the
/// dimension's keys) may come up through `node` from child `i`: the node does
/// not read them and its expressions are deterministic. A projection may pass
/// a blocked column through as a bare column.
fn passes(node: &LogicalPlan, i: usize, blocked: &Attrs) -> bool {
    if !may_pass(node, i) {
        return false;
    }
    match &node.node {
        PlanNode::Project { exprs, .. } => exprs.iter().all(|e| {
            is_deterministic(e) && (plain_column(e) || references(e).is_disjoint(blocked))
        }),
        PlanNode::Filter { predicate, .. } => {
            is_deterministic(predicate) && references(predicate).is_disjoint(blocked)
        }
        PlanNode::Join {
            condition: JoinCondition::On(on),
            ..
        } => references(on).is_disjoint(blocked),
        _ => false,
    }
}

fn is_non_inner_join(p: &LogicalPlan) -> bool {
    matches!(&p.node, PlanNode::Join { kind, .. } if *kind != JoinKind::Inner)
}

/// A candidate's join, read: the equalities, the dimension's columns.
struct Shape {
    /// `(x, d)` per equality: the other side's column, the dimension's.
    eqs: Vec<(BoundColumn, BoundColumn)>,
    /// The conjuncts, as written.
    conj: Vec<Arc<Expr>>,
    /// The dimension's output columns other than its keys.
    payload: Vec<BoundColumn>,
    /// The payload and the dimension's keys.
    blocked: Attrs,
    /// The broadcast threshold the physical planner uses, `None` when off.
    broadcast_max: Option<f64>,
}

impl Shape {
    fn of(c: &Candidate, broadcast_max: Option<f64>) -> Option<Self> {
        let PlanNode::Join {
            left,
            right,
            condition: JoinCondition::On(on),
            ..
        } = &c.join.node
        else {
            return None;
        };
        let (dim, other) = if c.dim_left {
            (left, right)
        } else {
            (right, left)
        };
        let dim_cols = dim.resolved_output()?;
        let other_cols = other.resolved_output()?;
        let in_dim = |id: AttrId| dim_cols.iter().any(|c| c.id == id);
        let in_other = |id: AttrId| other_cols.iter().any(|c| c.id == id);
        let conj = conjuncts(on);
        let mut eqs = Vec::new();
        for e in &conj {
            let (op, args, _) = builtin(e)?;
            if op != Builtin::Eq || args.len() != 2 {
                return None;
            }
            let (a, b) = (bare_column(&args[0])?, bare_column(&args[1])?);
            if a.data_type != b.data_type || !key_type(&a.data_type) {
                return None;
            }
            let pair = if in_other(a.id) && in_dim(b.id) {
                (a.clone(), b.clone())
            } else if in_dim(a.id) && in_other(b.id) {
                (b.clone(), a.clone())
            } else {
                return None;
            };
            eqs.push(pair);
        }
        if eqs.is_empty() || !repeatable(dim) {
            return None;
        }
        let keys: Attrs = eqs.iter().map(|(_, d)| d.id).collect();
        let payload: Vec<BoundColumn> = dim_cols
            .iter()
            .filter(|c| !keys.contains(&c.id))
            .cloned()
            .collect();
        let mut blocked = keys;
        blocked.extend(payload.iter().map(|c| c.id));
        let shape = Self {
            eqs,
            conj,
            payload,
            blocked,
            broadcast_max,
        };
        // Every node on the path must pass what the shape blocks.
        c.path
            .iter()
            .all(|s| passes(&s.node, s.child, &shape.blocked))
            .then_some(shape)
    }

    /// The statistics gates, or why the move is not worth it.
    fn gates(&self, c: &Candidate, est: &mut Estimator<'_>) -> Result<(), String> {
        let carried = self.carried(c);
        let width: Option<usize> = carried.iter().map(|c| fixed_width(&c.data_type)).sum();
        if width.is_some_and(|w| w < MIN_PAYLOAD_BYTES) {
            return Err(format!(
                "a payload of {} B a row is under {MIN_PAYLOAD_BYTES} B",
                width.unwrap_or(0)
            ));
        }
        let PlanNode::Join { left, right, .. } = &c.join.node else {
            return Err("not a join".to_owned());
        };
        let (dim, other) = if c.dim_left {
            (left, right)
        } else {
            (right, left)
        };
        let (de, xe) = (est.plan(dim), est.plan(other));
        // Never blind: a default row count says nothing of which side builds.
        if !de.from_statistics || !xe.from_statistics {
            return Err("the estimates rest on default row counts".to_owned());
        }
        let (d, x) = (de.rows, xe.rows);
        if d > x {
            return Err(format!(
                "the dimension estimates more rows ({d:.0}) than the other side ({x:.0})"
            ));
        }
        // Computing D a second time must cost less than the rows it narrows.
        let scanned = scanned_rows(dim, est);
        if scanned > x {
            return Err(format!(
                "the dimension scans more rows ({scanned:.0}) than the other side holds ({x:.0})"
            ));
        }
        // A payload on a join's probe side only streams through it; one on
        // its build side is held in the hash table. Some join on the path must
        // build the path's side, as the physical planner orients it.
        if !c
            .path
            .iter()
            .any(|s| builds_path(s, est, self.broadcast_max))
        {
            return Err("no join on the path builds it: the payload only streams".to_owned());
        }
        Ok(())
    }

    /// The payload columns the path's top still outputs.
    fn carried(&self, c: &Candidate) -> Vec<BoundColumn> {
        let top = c.path.last().map_or(&c.join, |s| &s.node);
        let out = top.resolved_output().unwrap_or_default();
        self.payload
            .iter()
            .filter(|p| out.iter().any(|o| o.id == p.id))
            .cloned()
            .collect()
    }

    /// The path rebuilt over the semi join, the moved join over it, and a
    /// projection giving the path's original output; `None` when attribute
    /// ids run out or a plan has no resolved output.
    fn moved(
        &self,
        c: &Candidate,
        top: &Arc<LogicalPlan>,
        next: &mut u64,
    ) -> Option<Arc<LogicalPlan>> {
        let PlanNode::Join {
            left,
            right,
            kind,
            condition,
            ..
        } = &c.join.node
        else {
            return None;
        };
        let (dim, other) = if c.dim_left {
            (left, right)
        } else {
            (right, left)
        };

        // X ⋉ Shift(π_d D) on x = shifted d.
        let offset = *next;
        let high = max_id(dim, &mut crate::expr::AddrMap::default());
        *next = offset
            .checked_add(high)?
            .checked_add(1)
            .filter(|n| *n <= MAX_ID)?;
        let mut key_cols: Vec<BoundColumn> = Vec::new();
        for (_, d) in &self.eqs {
            if !key_cols.iter().any(|k| k.id == d.id) {
                key_cols.push(d.clone());
            }
        }
        let keys_only = Arc::new(LogicalPlan::new(PlanNode::Project {
            input: Arc::clone(dim),
            exprs: key_cols.iter().map(column).collect(),
        }));
        let shifted = Arc::new(LogicalPlan::new(PlanNode::Shift {
            input: keys_only,
            offset,
        }));
        let semi_cond = self
            .conj
            .iter()
            .map(|e| {
                let (op, args, f) = builtin(e)?;
                let args = args
                    .iter()
                    .map(|a| {
                        if bare_column(a).is_some_and(|b| self.blocked.contains(&b.id)) {
                            shift(a, offset)
                        } else {
                            Arc::clone(a)
                        }
                    })
                    .collect();
                Some(recall(f, op, args))
            })
            .collect::<Option<Vec<_>>>()?;
        let semi_cond = crate::expr::and_all(semi_cond)?;
        let mut below = Arc::new(
            LogicalPlan::new(PlanNode::Join {
                left: Arc::clone(other),
                right: shifted,
                kind: JoinKind::LeftSemi,
                condition: JoinCondition::On(semi_cond),
            })
            .with_origin(c.join.origin),
        );

        // The path, bottom up, without the dimension's columns and with the
        // other side's key columns kept for the moved join.
        let xs: Vec<BoundColumn> = self.eqs.iter().map(|(x, _)| x.clone()).collect();
        for s in &c.path {
            below = match &s.node.node {
                PlanNode::Project { exprs, .. } => {
                    let mut kept: Vec<Arc<Expr>> = exprs
                        .iter()
                        .filter(|e| output_of(e).is_none_or(|o| !self.blocked.contains(&o.id)))
                        .cloned()
                        .collect();
                    for x in &xs {
                        if !kept
                            .iter()
                            .any(|e| output_of(e).is_some_and(|o| o.id == x.id))
                        {
                            kept.push(column(x));
                        }
                    }
                    let mut p = LogicalPlan::clone(&s.node);
                    if let PlanNode::Project { input, exprs } = &mut p.node {
                        *input = below;
                        *exprs = kept;
                    }
                    Arc::new(p)
                }
                _ => with_child(&s.node, s.child, below),
            };
        }

        // The moved join, then the path's original output.
        let joined = Arc::new(
            LogicalPlan::new(PlanNode::Join {
                left: below,
                right: Arc::clone(dim),
                kind: *kind,
                condition: condition.clone(),
            })
            .with_origin(c.join.origin),
        );
        let out = top.resolved_output()?;
        Some(Arc::new(
            LogicalPlan::new(PlanNode::Project {
                input: joined,
                exprs: out.iter().map(column).collect(),
            })
            .with_origin(top.origin),
        ))
    }

    fn keys_text(&self) -> String {
        let v: Vec<String> = self
            .eqs
            .iter()
            .map(|(x, d)| format!("{} = {}", x.name, d.name))
            .collect();
        v.join(", ")
    }

    fn payload_text(&self) -> String {
        let v: Vec<&str> = self.payload.iter().map(|c| c.name.as_str()).collect();
        v.join(", ")
    }
}

/// Whether `e` is a column, not renamed.
fn plain_column(e: &Expr) -> bool {
    matches!(
        &e.node,
        krishiv_plan::ExprNode::Resolved(krishiv_plan::ResolvedExpr::Column(_))
    )
}

/// The column `e` is, under aliases only.
fn bare_column(e: &Arc<Expr>) -> Option<&BoundColumn> {
    match &crate::expr::strip_alias(e).node {
        krishiv_plan::ExprNode::Resolved(krishiv_plan::ResolvedExpr::Column(c)) => Some(c),
        _ => None,
    }
}

/// Bytes a row of a fixed-width type; `None` for a variable-width one.
fn fixed_width(t: &DataType) -> Option<usize> {
    Some(match t {
        DataType::Boolean | DataType::Byte => 1,
        DataType::Short => 2,
        DataType::Integer | DataType::Float | DataType::Date => 4,
        DataType::Long | DataType::Double | DataType::Timestamp | DataType::TimestampNtz => 8,
        DataType::Decimal(d) if d.precision() <= 18 => 8,
        DataType::Decimal(_) => 16,
        _ => return None,
    })
}

fn children(p: &LogicalPlan) -> Vec<Arc<LogicalPlan>> {
    let mut v = Vec::new();
    p.for_each_child(&mut |c| v.push(Arc::clone(c)));
    v
}

/// `p` with its child `i` replaced (a relation with one input, or a join).
fn with_child(p: &Arc<LogicalPlan>, i: usize, child: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    match &p.node {
        PlanNode::Join { .. } => {
            let mut j = LogicalPlan::clone(p);
            if let PlanNode::Join { left, right, .. } = &mut j.node {
                if i == 0 {
                    *left = child;
                } else {
                    *right = child;
                }
            }
            Arc::new(j)
        }
        _ => dag::with_input(p, child),
    }
}
