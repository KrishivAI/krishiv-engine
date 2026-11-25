// SPDX-License-Identifier: Apache-2.0
//! `RollupFromParent` (perf lever F-C2, study 54 Finding 2, TPC-DS q67): the
//! grouping sets of a pre-aggregated expansion are each computed from an
//! already-computed parent set instead of from every input row.
//!
//! ```text
//! Aggregate(merge)(Project*(Expand[p_1 .. p_n](P)))      P = PartialAggregate(X) by K
//!   =>  Aggregate(merge)(Project*(Union(Project p_1 (S_1), .., Project p_n (S_n))))
//!       S_j = P                                  when K_j = K
//!           = S_parent                           when K_j = ()
//!           = PartialAggregate(merge)(S_parent)  by K_j, S_parent the newest
//!                                                computed set with K_parent ⊇ K_j
//! ```
//!
//! `eager/expand.rs` puts the partial aggregate `P` below the expansion, so
//! the expansion repeats `P`'s groups instead of the input rows. Where the
//! finest set is nearly unique (q67: eight keys over 5.4M rows), `P` does not
//! reduce and its adaptive bypass passes the rows on, so the expansion still
//! makes `n` copies of every row for the aggregate above (48.5M from 5.4M).
//! Here each projection `p_j` reads the states of `P` merged by the keys it
//! keeps (`K_j`), computed from the states of a finer set (Gray et al. 1997,
//! super-aggregates from their parent; Agarwal et al. VLDB 1996,
//! smallest-parent; DuckDB's grouping-sets chain). A `ROLLUP` level reads the
//! level above it, which is already reduced.
//!
//! # Why the result is the same
//!
//! The aggregate above merges states only (every call's single argument is a
//! `StateInput`), and its keys and the projections between it and the
//! expansion read no state column except as bare columns. A projection `p_j`
//! reads, besides the state columns it passes through, only keys of `K_j`
//! (deterministic expressions) and literals, so two rows of `P` with equal
//! `K_j` values give equal values in every column but the states: they fall
//! into one group of the aggregate above, whose merge of their states equals
//! merging the merged state of their `K_j` group (the calls are those
//! `eager/expand.rs` already moved: order- and grouping-independent states).
//! Every set `S_j` holds each row of `X` once, so nothing is lost or counted
//! twice. A partial aggregate keeps its key attributes, so a set's keys are
//! the parent's columns. The grand total `()` is never a keyless partial
//! aggregate (over empty input it would yield a state row, and the grouped
//! aggregate above a row where Spark yields none): its branch reads the
//! chosen parent's rows as they are, which the merge above folds into
//! one group. The output keeps the expansion's attributes (the
//! first branch's, a union's output); nullability is the expansion's.
//!
//! # When it applies
//!
//! - an `Aggregate` over projections over an `Expand` of two or more
//!   projections, over column-only projections over a `PartialAggregate` of
//!   raw calls with at least one key;
//! - every state column of `P` passes through every projection, each at one
//!   output position, and nothing else reads a state column except the
//!   aggregate's `StateInput` arguments;
//! - statistics say `P` does not reduce: its estimated groups are more than
//!   `1 / `[`MIN_REDUCTION`] of `X`'s estimated rows (where it reduces, the
//!   expansion of its groups is already small: q22).
//!
//! Sets read twice (by their own branch and a finer set's child, or by two
//! children) become `WITH` definitions named [`ROLLUP_PREFIX`], which starts
//! with `ShareCte`'s prefix, so `InlineCTE` keeps them and a second
//! optimization finds no `Expand` to rewrite: the plan stays a fixed point.
//! Runs once, after `ShareFilteredScan`, only with a session configuration and
//! `krishiv.optimizer.rollupFromParent` on (`Options::rollup_from_parent`).

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{
    AttrId, BoundColumn, Cte, CteRef, Expr, ExprNode, Grouping, LogicalPlan, PlanNode,
    ResolvedExpr, ResolvedFunction, partial_state_columns,
};
use krishiv_types::DataType;

use crate::dag;
use crate::eager::MIN_REDUCTION;
use crate::estimate::Estimator;
use crate::expr::{Attrs, column, data_type, is_deterministic, references, substitute};
use crate::restrict::{MAX_ID, max_id};
use crate::stats::Statistics;

/// The name prefix of the definitions this rule adds (a `ShareCte` name, so
/// `InlineCTE` keeps them).
pub(crate) const ROLLUP_PREFIX: &str = "__shared_rollup_";

/// `root` with each qualifying grouping-sets expansion computed from parents.
pub(crate) fn chain(
    root: Arc<LogicalPlan>,
    stats: &dyn Statistics,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    let mut est = Estimator::new(stats);
    let mut next = max_id(&root, &mut Default::default()).saturating_add(1);
    let mut ctes: Vec<Cte> = Vec::new();
    let out = dag::rewrite_up(Arc::clone(&root), &mut |p| {
        if next >= MAX_ID {
            return Ok(p);
        }
        Ok(rewrite(&p, &mut est, &mut next, &mut ctes).unwrap_or(p))
    })?;
    if ctes.is_empty() {
        return Ok(root);
    }
    // Into the root's `WITH` when one is there (`ShareCte`'s, kept by
    // `InlineCTE`); these go first. `P`'s input may read a
    // `ShareFilteredScan` definition, which then comes after them: the
    // executor registers every definition before it plans the input
    // (`krishiv-exec` `planner/build.rs`).
    if let PlanNode::WithCte {
        input,
        ctes: theirs,
    } = &out.node
    {
        let mut all = ctes;
        all.extend(theirs.iter().cloned());
        return Ok(Arc::new(LogicalPlan::new(PlanNode::WithCte {
            input: Arc::clone(input),
            ctes: all,
        })));
    }
    Ok(Arc::new(LogicalPlan::new(PlanNode::WithCte {
        input: out,
        ctes,
    })))
}

/// A grouping set's relation: its plan, its keys (ids of `P`'s keys) and, per
/// output attribute of `P`, the column holding it.
struct Set {
    plan: Arc<LogicalPlan>,
    keys: Vec<AttrId>,
    cols: HashMap<AttrId, BoundColumn>,
    readers: usize,
    parent: Option<usize>,
    /// The definition's name once it is shared.
    name: Option<String>,
}

/// The aggregate `p` with its expansion computed from parents, or `None`.
fn rewrite(
    p: &Arc<LogicalPlan>,
    est: &mut Estimator<'_>,
    next: &mut u64,
    ctes: &mut Vec<Cte>,
) -> Option<Arc<LogicalPlan>> {
    let PlanNode::Aggregate {
        input,
        grouping: Grouping::GroupBy(group),
        aggregates,
    } = &p.node
    else {
        return None;
    };
    // The spine of projections down to the expansion.
    let mut spine = Vec::new();
    let mut at = input;
    let (expand_input, projections, output) = loop {
        match &at.node {
            PlanNode::Project { input, .. } => {
                spine.push(Arc::clone(at));
                at = input;
            }
            PlanNode::Expand {
                input,
                projections,
                output,
            } if projections.len() >= 2 => break (input, projections, output),
            _ => return None,
        }
    };
    let expand = Arc::clone(at);
    // Column-only projections between the expansion and `P`: each id read
    // above them names an output of `P`.
    let mut rename: HashMap<AttrId, AttrId> = HashMap::new();
    let mut below = expand_input;
    let mut renames: Vec<HashMap<AttrId, AttrId>> = Vec::new();
    while let PlanNode::Project { input, exprs } = &below.node {
        let mut step = HashMap::new();
        for e in exprs {
            let (out, from) = match &e.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) => (c.id, c.id),
                ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => match &expr.node {
                    ExprNode::Resolved(ResolvedExpr::Column(c)) => (output.id, c.id),
                    _ => return None,
                },
                _ => return None,
            };
            step.insert(out, from);
        }
        renames.push(step);
        below = input;
    }
    let PlanNode::PartialAggregate {
        input: x,
        keys: p_keys,
        aggregates: p_calls,
        states,
    } = &below.node
    else {
        return None;
    };
    let p = Arc::clone(below);
    let key_cols: Vec<BoundColumn> = p_keys
        .iter()
        .map(|k| match &k.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.clone()),
            _ => None,
        })
        .collect::<Option<_>>()?;
    if key_cols.is_empty() || states.is_empty() {
        return None;
    }
    // Ids above the column-only projections, composed down to `P`'s.
    let p_out: Vec<BoundColumn> = key_cols.iter().chain(states.iter()).cloned().collect();
    for c in &p_out {
        rename.insert(c.id, c.id);
    }
    for step in renames.iter().rev() {
        let mut composed = HashMap::new();
        for (out, from) in step {
            composed.insert(*out, *rename.get(from)?);
        }
        rename = composed;
    }
    let is_state = |id: AttrId| states.iter().any(|s| s.id == id);
    let is_key = |id: AttrId| key_cols.iter().any(|k| k.id == id);

    // The calls of `P` and their state counts, to build the merges.
    let mut per_call: Vec<(ResolvedFunction, DataType, usize)> = Vec::new();
    let mut counted = 0usize;
    for c in p_calls {
        let ExprNode::Resolved(ResolvedExpr::Function(f)) = &c.node else {
            return None;
        };
        if f.args.iter().any(|a| crate::eager::is_state(a)) {
            return None;
        }
        let raw = f
            .args
            .first()
            .and_then(|a| data_type(a))
            .unwrap_or(DataType::Long);
        let n = partial_state_columns(f, &raw, AttrId(0))?.len();
        counted += n;
        per_call.push((f.clone(), raw, n));
    }
    if counted != states.len() {
        return None;
    }

    // Each projection: its items (in `P`'s ids) and the keys it keeps.
    let mut items: Vec<Vec<Arc<Expr>>> = Vec::with_capacity(projections.len());
    let mut sets: Vec<Vec<AttrId>> = Vec::with_capacity(projections.len());
    let to_p: HashMap<AttrId, Arc<Expr>> = rename
        .iter()
        .filter_map(|(above, id)| {
            p_out
                .iter()
                .find(|c| c.id == *id)
                .map(|c| (*above, column(c)))
        })
        .collect();
    let mut state_at: Vec<Option<AttrId>> = vec![None; output.len()];
    for (j, proj) in projections.iter().enumerate() {
        if proj.len() != output.len() {
            return None;
        }
        let mut keys: Vec<AttrId> = Vec::new();
        let mut row = Vec::with_capacity(proj.len());
        for (i, e) in proj.iter().enumerate() {
            let e = substitute(e, &to_p);
            let refs: Attrs = references(&e);
            let reads_state = refs.iter().any(|a| is_state(*a));
            if reads_state {
                // A state passes through bare, at one position in every projection.
                let ExprNode::Resolved(ResolvedExpr::Column(c)) = &e.node else {
                    return None;
                };
                if j == 0 {
                    state_at[i] = Some(c.id);
                } else if state_at[i] != Some(c.id) {
                    return None;
                }
            } else {
                if state_at[i].is_some() || !is_deterministic(&e) {
                    return None;
                }
                for a in refs.iter() {
                    if !is_key(*a) {
                        return None;
                    }
                    if !keys.contains(a) {
                        keys.push(*a);
                    }
                }
            }
            row.push(e);
        }
        keys.sort_unstable();
        items.push(row);
        sets.push(keys);
    }
    if states.iter().any(|s| !state_at.contains(&Some(s.id))) {
        return None;
    }
    // Above the expansion, states are read only as bare columns of the
    // projections and as the aggregate's `StateInput` arguments.
    let state_out: Attrs = output
        .iter()
        .zip(&state_at)
        .filter(|(_, s)| s.is_some())
        .map(|(o, _)| o.id)
        .collect();
    if !reads_states_only_as_merges(&spine, group, aggregates, state_out) {
        return None;
    }

    // `P` must not reduce: then its groups are as many as its rows.
    let x_est = est.plan(x);
    let p_est = est.plan(&p);
    if !x_est.from_statistics || p_est.rows * MIN_REDUCTION <= x_est.rows {
        return None;
    }

    // The sets, finest first; each reads the newest computed superset. The
    // estimate below orders `P` before the sets built so far, but a set not
    // yet built reads a placeholder of its parent's plan and so has its
    // parent's estimate: among those the newest wins. For a `ROLLUP` that is
    // the level above (the smallest parent); for a `CUBE` it is a superset,
    // not always the smallest one.
    let mut all_keys: Vec<AttrId> = key_cols.iter().map(|k| k.id).collect();
    all_keys.sort_unstable();
    let mut built: Vec<Set> = vec![Set {
        plan: Arc::clone(&p),
        keys: all_keys,
        cols: p_out.iter().map(|c| (c.id, c.clone())).collect(),
        readers: 0,
        parent: None,
        name: None,
    }];
    let mut order: Vec<usize> = (0..sets.len()).collect();
    order.sort_by_key(|j| std::cmp::Reverse(sets[*j].len()));
    let mut branch_set: Vec<usize> = vec![0; sets.len()];
    for j in order {
        let want = &sets[j];
        let parent = (0..built.len())
            .filter(|b| want.iter().all(|k| built[*b].keys.contains(k)))
            .min_by(|a, b| {
                let ra = est.plan(&built[*a].plan).rows;
                let rb = est.plan(&built[*b].plan).rows;
                ra.total_cmp(&rb).then(b.cmp(a))
            })?;
        // The grand total reads its parent's rows as they are: a keyless
        // partial aggregate yields a state row over empty input, where the
        // grouped aggregate above yields none; the merge above folds the
        // parent's groups into the one row.
        if want.is_empty() || built[parent].keys == *want {
            built[parent].readers += 1;
            branch_set[j] = parent;
            continue;
        }
        // Built over a placeholder read of the parent; replaced below once
        // the readers are known.
        built[parent].readers += 1;
        built.push(Set {
            plan: Arc::clone(&built[parent].plan),
            keys: want.clone(),
            cols: HashMap::new(),
            readers: 1,
            parent: Some(parent),
            name: None,
        });
        branch_set[j] = built.len() - 1;
    }
    // Build each set over a read of its parent, in order (parents first).
    for s in 0..built.len() {
        if let Some(parent) = built[s].parent {
            let (read, cols) = read_of(&mut built, parent, next, ctes)?;
            let mut keys = Vec::with_capacity(built[s].keys.len());
            let mut out_cols: HashMap<AttrId, BoundColumn> = HashMap::new();
            for k in &built[s].keys {
                let c = cols.get(k)?.clone();
                keys.push(column(&c));
                out_cols.insert(*k, c);
            }
            let mut calls = Vec::with_capacity(per_call.len());
            let mut new_states = Vec::with_capacity(states.len());
            let mut at = 0usize;
            for (f, raw, n) in &per_call {
                let mut merge = f.clone();
                merge.args = vec![Arc::new(Expr::new(ExprNode::Resolved(
                    ResolvedExpr::StateInput {
                        columns: states[at..at + n]
                            .iter()
                            .map(|st| cols.get(&st.id).map(column))
                            .collect::<Option<_>>()?,
                        raw_type: raw.clone(),
                    },
                )))];
                calls.push(Arc::new(Expr::new(ExprNode::Resolved(
                    ResolvedExpr::Function(merge),
                ))));
                for st in &states[at..at + n] {
                    let fresh = BoundColumn {
                        id: AttrId(fresh_id(next)?),
                        ..st.clone()
                    };
                    out_cols.insert(st.id, fresh.clone());
                    new_states.push(fresh);
                }
                at += n;
            }
            built[s].plan = Arc::new(
                LogicalPlan::new(PlanNode::PartialAggregate {
                    input: read,
                    keys,
                    aggregates: calls,
                    states: new_states,
                })
                .with_origin(p.origin),
            );
            built[s].cols = out_cols;
        }
    }
    // One branch per projection, over a read of its set.
    let mut branches = Vec::with_capacity(items.len());
    for (j, row) in items.iter().enumerate() {
        let (read, cols) = read_of(&mut built, branch_set[j], next, ctes)?;
        let map: HashMap<AttrId, Arc<Expr>> = cols.iter().map(|(id, c)| (*id, column(c))).collect();
        let mut exprs = Vec::with_capacity(row.len());
        for (e, o) in row.iter().zip(output) {
            let id = if j == 0 {
                o.id
            } else {
                AttrId(fresh_id(next)?)
            };
            exprs.push(Arc::new(Expr::new(ExprNode::Resolved(
                ResolvedExpr::Alias {
                    expr: substitute(e, &map),
                    output: BoundColumn { id, ..o.clone() },
                },
            ))));
        }
        branches.push(Arc::new(
            LogicalPlan::new(PlanNode::Project { input: read, exprs }).with_origin(expand.origin),
        ));
    }
    let mut rebuilt = Arc::new(
        LogicalPlan::new(PlanNode::Union {
            inputs: branches,
            by_name: false,
            allow_missing_columns: false,
        })
        .with_origin(expand.origin),
    );
    for node in spine.iter().rev() {
        let PlanNode::Project { exprs, .. } = &node.node else {
            return None;
        };
        rebuilt = Arc::new(
            LogicalPlan::new(PlanNode::Project {
                input: rebuilt,
                exprs: exprs.clone(),
            })
            .with_origin(node.origin),
        );
    }
    Some(Arc::new(
        LogicalPlan::new(PlanNode::Aggregate {
            input: rebuilt,
            grouping: Grouping::GroupBy(group.clone()),
            aggregates: aggregates.clone(),
        })
        .with_origin(p.origin),
    ))
}

/// A read of set `s`: the set itself when one reader reads it, else a
/// reference to its definition (added on the first read) with fresh columns.
fn read_of(
    built: &mut [Set],
    s: usize,
    next: &mut u64,
    ctes: &mut Vec<Cte>,
) -> Option<(Arc<LogicalPlan>, HashMap<AttrId, BoundColumn>)> {
    let set = &mut built[s];
    if set.readers <= 1 {
        return Some((Arc::clone(&set.plan), set.cols.clone()));
    }
    let source = set.plan.resolved_output()?;
    let name = match &set.name {
        Some(n) => n.clone(),
        None => {
            let n = format!("{ROLLUP_PREFIX}{}", ctes.len());
            ctes.push(Cte {
                name: n.clone(),
                plan: Arc::clone(&set.plan),
            });
            set.name = Some(n.clone());
            n
        }
    };
    let mut fresh: HashMap<AttrId, BoundColumn> = HashMap::new();
    let mut output = Vec::with_capacity(source.len());
    for c in &source {
        let f = BoundColumn {
            id: AttrId(fresh_id(next)?),
            ..c.clone()
        };
        fresh.insert(c.id, f.clone());
        output.push(f);
    }
    let cols = set
        .cols
        .iter()
        .map(|(id, c)| fresh.get(&c.id).map(|f| (*id, f.clone())))
        .collect::<Option<_>>()?;
    let reference = Arc::new(LogicalPlan::new(PlanNode::CteRef(CteRef {
        name,
        source: source.iter().map(|c| c.id).collect(),
        output,
    })));
    Some((reference, cols))
}

/// The next fresh attribute id, or `None` past [`MAX_ID`].
fn fresh_id(next: &mut u64) -> Option<u64> {
    if *next >= MAX_ID {
        return None;
    }
    let id = *next;
    *next += 1;
    Some(id)
}

/// Whether the projections `spine` (top first) and the aggregate read the
/// state columns `states` (the expansion's outputs) only as bare columns
/// passed through and, at the top, as `StateInput` arguments of calls that
/// have no other argument and no filter.
fn reads_states_only_as_merges(
    spine: &[Arc<LogicalPlan>],
    group: &[Arc<Expr>],
    aggregates: &[Arc<Expr>],
    mut states: Attrs,
) -> bool {
    for node in spine.iter().rev() {
        let PlanNode::Project { exprs, .. } = &node.node else {
            return false;
        };
        let mut above = Attrs::default();
        for e in exprs {
            let refs = references(e);
            if !refs.iter().any(|a| states.contains(a)) {
                continue;
            }
            match &e.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                    above.insert(c.id);
                }
                ExprNode::Resolved(ResolvedExpr::Alias { expr, output })
                    if matches!(expr.node, ExprNode::Resolved(ResolvedExpr::Column(_))) =>
                {
                    above.insert(output.id);
                }
                _ => return false,
            }
        }
        states = above;
    }
    if group
        .iter()
        .any(|g| references(g).iter().any(|a| states.contains(a)))
    {
        return false;
    }
    aggregates.iter().all(|e| merges_only(e, &states))
}

/// Whether `e` reads `states` only inside the `StateInput` argument of an
/// aggregate call without a filter.
fn merges_only(e: &Arc<Expr>, states: &Attrs) -> bool {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(f))
            if matches!(
                f.args.as_slice(),
                [a] if matches!(a.node, ExprNode::Resolved(ResolvedExpr::StateInput { .. }))
            ) =>
        {
            f.filter.is_none()
        }
        ExprNode::Resolved(ResolvedExpr::Column(c)) => !states.contains(&c.id),
        _ => {
            let mut ok = true;
            e.for_each_child(&mut |c| ok &= merges_only(c, states));
            ok
        }
    }
}
