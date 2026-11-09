// SPDX-License-Identifier: Apache-2.0
//! A filter and a scan's row stage fused into the hash probe above them
//! (design P2-E3 U2, [`ProbeSelect`]): the probe selects the rows it joins
//! by id instead of joining a filtered copy of each batch.
//!
//! - **The filter** ([`Planner::fuse_probe_filter`]) is fused when the
//!   probe's input is, in the same pipeline, a `Filter` under column-only
//!   projections, and neither the predicate nor the probe's keys nor its
//!   residual can raise on a row (`krishiv_plan::may_raise`): the fused
//!   predicate then sees every row the `Filter` saw, and the keys and the
//!   residual only the rows it kept (ADR 0008 §6). The projections are
//!   fused with it (the probe reads its columns out of the filter's input,
//!   [`ProbeSelect::columns`]), so a predicate may read columns they drop.
//!   A predicate with a subquery or a nondeterministic call stays a
//!   `Filter`. `EXPLAIN` prints it on the join (`probe filter: ...`) instead
//!   of as a node, and the projections not at all.
//! - **The row stage** ([`fuse_row_stage`]) of a runtime-filtered scan under
//!   column-only projections moves to the probe when every filter it
//!   evaluates (a top-level column target) reaches the probe's input. The
//!   scan still builds and restarts it (instantiation hands it over); the
//!   probe evaluates it over the same rows.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_plan::{Expr, is_deterministic, may_raise};

use super::build::{Planned, Planner, has_subquery};
use super::explain::ExplainNode;
use crate::physical::{
    PhysNode, ProbeColumn, ProbeFilter, ProbeFilterColumn, ProbeRowStage, ProbeSelect,
    ProjectItemPlan, ScanId, ScanRuntimeFilter,
};

/// The input columns of a column-only projection `n` (each output column's
/// input column), or `None` for any other node.
fn column_only(n: &PhysNode) -> Option<(&PhysNode, Vec<usize>)> {
    let PhysNode::Project {
        input,
        items,
        nullable: None,
        subqueries,
        ..
    } = n
    else {
        return None;
    };
    if !subqueries.is_empty() {
        return None;
    }
    let columns = items
        .iter()
        .map(|i| match i {
            ProjectItemPlan::Column { index, .. } => Some(*index),
            ProjectItemPlan::Expr { .. } | ProjectItemPlan::NullOf { .. } => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Some((input, columns))
}

/// Below `n`'s column-only projections (`depth` of them): the node, and
/// the column of it each column of `n` reads.
fn below_projections(n: &PhysNode, width: usize) -> (&PhysNode, Vec<usize>, usize) {
    let mut map: Vec<usize> = (0..width).collect();
    let mut n = n;
    let mut depth = 0;
    while let Some((input, columns)) = column_only(n) {
        let Some(next) = map
            .iter()
            .map(|&c| columns.get(c).copied())
            .collect::<Option<Vec<_>>>()
        else {
            break;
        };
        map = next;
        n = input;
        depth += 1;
    }
    (n, map, depth)
}

impl Planner<'_> {
    /// Fuses the `Filter` below `probe`'s column-only projections, and those
    /// projections, into the hash probe over it (see the module docs): the
    /// filter's input becomes `probe`'s node and printed tree; `None` (and
    /// `probe` unchanged) when the rules do not allow it. `keys` are the
    /// probe's key expressions and `residual` the join's other conditions.
    /// Also returns the predicate as `EXPLAIN` printed the `Filter`.
    pub(super) fn fuse_probe_filter(
        &self,
        probe: &mut Planned,
        keys: &[Arc<Expr>],
        residual: &[&Expr],
    ) -> Option<(ProbeSelect, String)> {
        let ansi = self.env.ansi;
        if keys.iter().any(|k| may_raise(k, ansi)) || residual.iter().any(|r| may_raise(r, ansi)) {
            return None;
        }
        let (n, map, depth) = below_projections(&probe.node, probe.layout.len());
        let PhysNode::Filter {
            layout,
            predicate,
            subqueries,
            ..
        } = n
        else {
            return None;
        };
        if !subqueries.is_empty()
            || has_subquery(predicate)
            || !is_deterministic(predicate)
            || may_raise(predicate, ansi)
            || !explained_as(&probe.explain, depth)
        {
            return None;
        }
        let filter = ProbeFilter {
            layout: layout.clone(),
            predicate: Arc::clone(predicate),
        };
        let identity = map.len() == layout.len() && map.iter().enumerate().all(|(i, &c)| i == c);
        // The names the projections gave the probe's columns.
        let columns = (!identity || depth > 0).then(|| {
            map.iter()
                .zip(probe.schema.fields())
                .map(|(&index, f)| ProbeColumn {
                    index,
                    name: f.name().clone(),
                })
                .collect()
        });
        let detail = take_filter_input(&mut probe.node, &mut probe.explain, depth)?;
        Some((
            ProbeSelect {
                filter: Some(filter),
                row_stage: None,
                columns,
            },
            detail,
        ))
    }
}

/// Whether `explain` prints `depth` projections over a filter, each with one
/// input (the shape of the plan nodes it prints).
fn explained_as(explain: &ExplainNode, depth: usize) -> bool {
    let mut e = explain;
    for _ in 0..depth {
        match e.children.as_slice() {
            [only] if e.name == "Project" => e = only,
            _ => return false,
        }
    }
    e.name == "Filter" && e.children.len() == 1
}

/// Replaces `node` (`depth` column-only projections over a `Filter`) with
/// the filter's input, and `explain` (the same shape, [`explained_as`])
/// with its input's line; returns the filter's line's detail.
fn take_filter_input(
    node: &mut PhysNode,
    explain: &mut ExplainNode,
    depth: usize,
) -> Option<String> {
    let input = {
        let mut n = &mut *node;
        for _ in 0..depth {
            let PhysNode::Project { input, .. } = n else {
                return None;
            };
            n = input;
        }
        let PhysNode::Filter { input, .. } = n else {
            return None;
        };
        std::mem::replace(&mut **input, PhysNode::OneRow)
    };
    *node = input;
    let (child, detail) = {
        let mut e = &mut *explain;
        for _ in 0..depth {
            e = e.children.first_mut()?;
        }
        (e.children.pop()?, std::mem::take(&mut e.detail))
    };
    *explain = child;
    Some(detail)
}

/// Hands the row stage of the runtime-filtered scan under `node`'s probe
/// (a hash join's, through column-only projections) to the probe, when
/// every filter it evaluates reaches the probe's input (see the module
/// docs). `filters` are the runtime filters each scan is about to receive
/// (planning attaches them after the joins are planned).
pub(super) fn fuse_row_stage(
    node: &mut PhysNode,
    filters: &HashMap<ScanId, Vec<ScanRuntimeFilter>>,
) {
    let PhysNode::HashJoin(j) = node else {
        return;
    };
    let width = probe_width(&j.probe);
    let Some(width) = width else {
        return;
    };
    let (n, map, _) = below_projections(&j.probe, width);
    let PhysNode::Scan(s) = n else {
        return;
    };
    let Some(list) = filters.get(&s.id).filter(|l| !l.is_empty()) else {
        return;
    };
    if s.row_id {
        return;
    }
    let mut columns = Vec::new();
    for f in list {
        if f.column.0.len() != 1 {
            // The row stage evaluates top-level columns only.
            continue;
        }
        let Some(k) = s.projection.iter().position(|p| *p == f.column) else {
            return;
        };
        let Some(column) = map.iter().position(|&c| c == k) else {
            return;
        };
        columns.push(ProbeFilterColumn { id: f.id, column });
    }
    if columns.is_empty() {
        return;
    }
    let id = s.id;
    j.select
        .get_or_insert(ProbeSelect {
            filter: None,
            row_stage: None,
            columns: None,
        })
        .row_stage = Some(ProbeRowStage { scan: id, columns });
}

/// The number of columns `n` outputs, through column-only projections down
/// to a scan (`None` for any other shape).
fn probe_width(n: &PhysNode) -> Option<usize> {
    match n {
        PhysNode::Project { items, .. } => column_only(n).map(|_| items.len()),
        PhysNode::Scan(s) => Some(s.projection.len()),
        _ => None,
    }
}
