// SPDX-License-Identifier: Apache-2.0
//! The columns each table read of a query reads, derived from the final resolved plan (P3-T062
//! review B1): the column pass of a policy that decides columns asks exactly these.
//!
//! A scan column is read when any expression anywhere in the plan references it (filters, join
//! conditions, `USING`/`NATURAL` keys, grouping, sort and window keys, generators, subqueries
//! and their correlated references), when it reaches the root's output, or when a node consumes
//! it by position (a union's later inputs, a CTE reference's definition, a recursive CTE's
//! terms). Projections and expansions (`ROLLUP`, `CUBE`, `GROUPING SETS`) pass a bare column
//! through, read only if something above reads it. This over-approximates (a projected-away
//! expression still counts), never under: a plan node this walk does not prove (an unresolved
//! form) makes every table read whole rows.
//! Each scan is its own appearance: one that reads no column (as `count(*)` does) reads whole
//! rows and asks the table, whatever other appearances of the same table read. A subtree shared
//! through `Shift` instances is one appearance (the union of what its instances read), so the
//! walk stays linear in the distinct nodes.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::{ExprNode, JoinCondition, LogicalPlan, PlanNode, ReadSource, ResolvedExpr};

use super::bind::{Bindings, BoundTable};

/// What a query reads of one bound securable.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Reads {
    /// Some appearance reads whole rows (no column, or a plan the walk could not prove): the
    /// table itself is asked.
    pub(super) whole: bool,
    /// The columns the appearances read, in catalog spelling.
    pub(super) columns: BTreeSet<String>,
}

/// The reads of `root` per [`Bindings::securables`] entry (see the module docs). `hinted` is
/// what the plan's hints read before they were removed ([`hint_reads`]). The inlined queries
/// of definer views are walked too: their reads are attributed by name, so a table both the
/// principal and a definer text read is asked for the union of the columns (over-denies, never
/// under; the definer-only uses themselves are skipped by the column pass).
pub(super) fn column_reads(
    root: &LogicalPlan,
    b: &Bindings,
    hinted: &(HashSet<u64>, bool),
) -> Vec<Reads> {
    let mut by_name: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
    for bound in b.tables.values() {
        if let BoundTable::Found { meta, securable } = bound {
            by_name
                .entry(meta.ident.parts())
                .or_default()
                .push(*securable);
        }
    }
    for (name, securable) in &b.scans {
        by_name.entry(name.clone()).or_default().push(*securable);
    }
    let mut walk = Walk::default();
    walk.refs.extend(hinted.0.iter().copied());
    walk.unproven |= hinted.1;
    walk.plan(root);
    match root.resolved_output() {
        Some(out) => walk.refs.extend(out.iter().map(|c| c.id.0)),
        None => walk.unproven = true,
    }
    walk.close();
    // Every use starts as a whole-row read and narrows only when a scan of it is attributed
    // (review r-m3): a use no scan accounts for asks the table.
    let mut reads = vec![
        Reads {
            whole: true,
            columns: BTreeSet::new(),
        };
        b.securables.len()
    ];
    let mut attributed = vec![false; b.securables.len()];
    for scan in &walk.scans {
        let Some(securables) = by_name.get(&scan.name) else {
            // A table scan Bind did not bind: no column can be attributed.
            walk.unproven = true;
            continue;
        };
        let read: Vec<&String> = scan
            .columns
            .iter()
            .filter(|(id, _)| walk.refs.contains(id))
            .map(|(_, name)| name)
            .collect();
        for &i in securables {
            if let (Some(r), Some(seen)) = (reads.get_mut(i), attributed.get_mut(i)) {
                if !*seen {
                    *seen = true;
                    r.whole = false;
                }
                r.whole |= read.is_empty();
                r.columns.extend(read.iter().map(|c| (*c).clone()));
            }
        }
    }
    if walk.unproven {
        for r in &mut reads {
            r.whole = true;
        }
    }
    reads
}

/// The attributes the partitioning hints of `plan` name (`REPARTITION(3, b)`,
/// `REPARTITION_BY_RANGE(b)`, `df.hint("repartition", b)`), read before hint resolution drops
/// them (review r-m1: as `DISTRIBUTE BY b` asks for b); `true` when a parameter holds an
/// unresolved column (then every read is whole rows).
pub(super) fn hint_reads(plan: &LogicalPlan) -> (HashSet<u64>, bool) {
    let mut ids = HashSet::new();
    let mut unproven = false;
    let _ = crate::walk::visit_distinct(crate::walk::Node::Plan(plan), |n| {
        if let crate::walk::Node::Plan(p) = n
            && let PlanNode::Hint { params, .. } = &p.node
        {
            let mut stack: Vec<&crate::Expr> = params.iter().map(AsRef::as_ref).collect();
            while let Some(e) = stack.pop() {
                match &e.node {
                    ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) => {
                        ids.insert(c.id.0);
                    }
                    ExprNode::Column(_) | ExprNode::Star(_) => unproven = true,
                    _ => {}
                }
                e.for_each_child(&mut |c| stack.push(c));
            }
        }
        std::ops::ControlFlow::Continue(())
    });
    (ids, unproven)
}

/// One scan of a table: its name parts and its columns (attribute id as seen from the root,
/// name).
struct Scan {
    name: Vec<String>,
    columns: Vec<(u64, String)>,
}

#[derive(Default)]
struct Walk {
    /// Attribute ids (as seen from the root) something reads.
    refs: HashSet<u64>,
    /// `(a, b)`: if `a` is read, so is `b` (positional consumption).
    edges: Vec<(u64, u64)>,
    scans: Vec<Scan>,
    /// A node the walk does not prove: every read is whole rows.
    unproven: bool,
    /// Expressions visited, per offset (expressions are DAGs too).
    seen_exprs: HashSet<(*const crate::Expr, u64)>,
}

impl Walk {
    /// Marks every output column of `p` (shifted by `off`) read.
    fn all_of(&mut self, p: &LogicalPlan, off: u64) {
        match p.resolved_output() {
            Some(out) => {
                for c in out {
                    self.shifted(c.id.0, off);
                }
            }
            None => self.unproven = true,
        }
    }

    fn shifted(&mut self, id: u64, off: u64) {
        match id.checked_add(off) {
            Some(id) => {
                self.refs.insert(id);
            }
            None => self.unproven = true,
        }
    }

    /// Walks every relation and expression under `root` once per (node, offset).
    fn plan(&mut self, root: &LogicalPlan) {
        let mut seen: HashSet<(*const LogicalPlan, u64)> = HashSet::new();
        let mut stack: Vec<(&LogicalPlan, u64)> = vec![(root, 0)];
        while let Some((p, off)) = stack.pop() {
            if !seen.insert((std::ptr::from_ref(p), off)) {
                continue;
            }
            let mut subplans: Vec<&LogicalPlan> = Vec::new();
            if let PlanNode::Project { exprs, .. } = &p.node {
                for e in exprs {
                    self.projected(e, off, &mut subplans);
                }
            } else if let PlanNode::Expand {
                projections,
                output,
                ..
            } = &p.node
            {
                self.expanded(projections, output, off, &mut subplans);
            } else {
                p.for_each_expr(&mut |e| self.expr(e, off, &mut subplans));
            }
            for s in subplans {
                // A subquery's output is what its expression consumes.
                self.all_of(s, off);
                stack.push((s, off));
            }
            match &p.node {
                PlanNode::Scan(s) => match &s.source {
                    ReadSource::Table { name, .. } => {
                        let columns = s
                            .output
                            .iter()
                            .filter_map(|c| Some((c.id.0.checked_add(off)?, c.name.clone())))
                            .collect::<Vec<_>>();
                        if columns.len() != s.output.len() {
                            self.unproven = true;
                        }
                        self.scans.push(Scan {
                            name: name.clone(),
                            columns,
                        });
                    }
                    // Paths are checked as path securables; a local relation is data the client
                    // sent with the request. Neither reads a catalog table's columns.
                    ReadSource::Files { .. } | ReadSource::Local(_) => {}
                },
                PlanNode::OneRowRelation
                | PlanNode::Range { .. }
                | PlanNode::UnionLoopRef { .. }
                | PlanNode::Project { .. }
                | PlanNode::Filter { .. }
                // Rows pass through; the metric expressions are read above (`for_each_expr`).
                | PlanNode::CollectMetrics { .. }
                | PlanNode::Aggregate { .. }
                | PlanNode::PartialAggregate { .. }
                | PlanNode::Sort { .. }
                | PlanNode::Limit { .. }
                | PlanNode::Offset { .. }
                | PlanNode::SubqueryAlias { .. }
                | PlanNode::WithCte { .. }
                | PlanNode::JoinHint { .. }
                | PlanNode::Repartition { .. }
                | PlanNode::Sample { .. }
                | PlanNode::Window { .. }
                | PlanNode::ReplicateRows { .. }
                | PlanNode::Expand { .. }
                | PlanNode::Generate { .. } => {}
                PlanNode::Join {
                    left,
                    right,
                    condition,
                    ..
                } => match condition {
                    JoinCondition::On(_) | JoinCondition::None => {}
                    // Resolved joins carry their keys as a condition; a key list still here
                    // compares columns the walk cannot see, so both sides read whole.
                    JoinCondition::Using(_) | JoinCondition::Natural => {
                        self.all_of(left, off);
                        self.all_of(right, off);
                    }
                },
                PlanNode::Union { inputs, .. } => {
                    // The output is the first input's attributes; the others are read by
                    // position.
                    let outs: Option<Vec<_>> = inputs.iter().map(|i| i.resolved_output()).collect();
                    match outs {
                        Some(outs) => {
                            if let Some((first, rest)) = outs.split_first() {
                                for other in rest {
                                    for (a, b) in first.iter().zip(other) {
                                        self.edge(a.id.0, b.id.0, off, off);
                                    }
                                }
                            }
                        }
                        None => self.unproven = true,
                    }
                }
                PlanNode::CteRef(r) => {
                    for (out, source) in r.output.iter().zip(&r.source) {
                        // The source is not shifted by an enclosing `Shift`; the definition
                        // may be: both are kept.
                        self.edge(out.id.0, source.0, off, 0);
                        self.edge(out.id.0, source.0, off, off);
                    }
                }
                PlanNode::UnionLoop {
                    anchor, recursive, ..
                } => {
                    self.all_of(anchor, off);
                    self.all_of(recursive, off);
                }
                PlanNode::Shift { input, offset } => {
                    // The shared subtree is walked once, unshifted (a walk per instance would
                    // be exponential in nested self-joins): an instance's output read means
                    // the shared column is read, so a column read through any instance counts
                    // for every one (an over-approximation, never under).
                    match input.resolved_output() {
                        Some(out) => {
                            for c in out {
                                self.edge(c.id.0, c.id.0, off.saturating_add(*offset), off);
                            }
                        }
                        None => self.unproven = true,
                    }
                }
                // Unresolved forms: nothing proves what they read. A Python UDF over a relation
                // (map, grouped and cogrouped map, table function) sees every column of it.
                PlanNode::Read(_)
                | PlanNode::MapPartitions { .. }
                | PlanNode::GroupMap { .. }
                | PlanNode::CoGroupMap { .. }
                | PlanNode::PythonTableFunction { .. }
                | PlanNode::Sql { .. }
                | PlanNode::WithColumns { .. }
                | PlanNode::Hint { .. }
                | PlanNode::PipeOperator { .. }
                | PlanNode::Distinct { .. }
                | PlanNode::SetOp { .. }
                | PlanNode::Pivot { .. }
                | PlanNode::Unpivot { .. }
                | PlanNode::InlineTable { .. }
                | PlanNode::TableFunction { .. } => self.unproven = true,
            }
            p.for_each_child(&mut |c| stack.push((c, off)));
        }
    }

    fn edge(&mut self, from: u64, to: u64, from_off: u64, to_off: u64) {
        match (from.checked_add(from_off), to.checked_add(to_off)) {
            (Some(a), Some(b)) => self.edges.push((a, b)),
            _ => self.unproven = true,
        }
    }

    /// One output expression of a projection: a bare column passes its attribute through (read
    /// only if something above reads it), `column AS name` reads the column if the alias is read;
    /// anything else counts as read.
    fn projected<'a>(&mut self, e: &'a crate::Expr, off: u64, subplans: &mut Vec<&'a LogicalPlan>) {
        match &e.node {
            ExprNode::Resolved(ResolvedExpr::Column(_)) => {}
            ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => match &expr.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                    self.edge(output.id.0, c.id.0, off, off);
                }
                _ => self.expr(e, off, subplans),
            },
            _ => self.expr(e, off, subplans),
        }
    }

    /// An expansion (`ROLLUP`, `CUBE`, `GROUPING SETS`, and the other rewrites that build one),
    /// column by column: output `j` is `projections[i][j]` for each `i`, so a bare column there
    /// passes its attribute through (read only if something above reads output `j`), a literal
    /// (the null fill of an absent key, the grouping id, the set number) reads nothing, and
    /// anything else counts as read. A projection whose width differs from the output proves
    /// nothing: every expression counts as read and the walk fails closed.
    fn expanded<'a>(
        &mut self,
        projections: &'a [Vec<std::sync::Arc<crate::Expr>>],
        output: &[crate::BoundColumn],
        off: u64,
        subplans: &mut Vec<&'a LogicalPlan>,
    ) {
        for row in projections {
            if row.len() != output.len() {
                self.unproven = true;
                for e in row {
                    self.expr(e, off, subplans);
                }
                continue;
            }
            for (e, out) in row.iter().zip(output) {
                match &e.node {
                    ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                        self.edge(out.id.0, c.id.0, off, off);
                    }
                    _ => self.expr(e, off, subplans),
                }
            }
        }
    }

    /// The attributes `e` references, and its subquery plans.
    fn expr<'a>(&mut self, e: &'a crate::Expr, off: u64, subplans: &mut Vec<&'a LogicalPlan>) {
        let mut stack = vec![e];
        while let Some(e) = stack.pop() {
            if !self.seen_exprs.insert((std::ptr::from_ref(e), off)) {
                continue;
            }
            match &e.node {
                ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) => {
                    self.shifted(c.id.0, off);
                }
                ExprNode::Subquery { plan, .. } => subplans.push(plan),
                // An unresolved reference names a column the walk cannot place.
                ExprNode::Column(_) | ExprNode::Star(_) => self.unproven = true,
                _ => {}
            }
            e.for_each_child(&mut |c| stack.push(c));
        }
    }

    /// Adds every attribute read through a positional edge, to a fixpoint.
    fn close(&mut self) {
        let mut next: HashMap<u64, Vec<u64>> = HashMap::new();
        for (a, b) in &self.edges {
            next.entry(*a).or_default().push(*b);
        }
        let mut todo: Vec<u64> = self.refs.iter().copied().collect();
        while let Some(a) = todo.pop() {
            for b in next.get(&a).into_iter().flatten() {
                if self.refs.insert(*b) {
                    todo.push(*b);
                }
            }
        }
    }
}
