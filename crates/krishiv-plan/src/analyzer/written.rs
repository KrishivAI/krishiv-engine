// SPDX-License-Identifier: Apache-2.0
//! The column references a statement wrote (ADR 0048 §3 Denials, A1 §3 Listings; P3-T062 review r3
//! M1, r4 N1): a refusal of a column read names the column only when the statement wrote a
//! reference to it, so a `*`, a `* EXCEPT`, a `table()`, a view's text, or the same name written
//! for something else (another relation's column, an alias, a lambda variable) never makes a
//! refusal name a table's other columns. Asked only once a column is refused, so the walk costs
//! nothing on a granted query.

use std::collections::{HashMap, HashSet};

use crate::walk::{Node, PtrSet, children};
use crate::{ExprNode, LogicalPlan, Origin, PlanNode, ReadSource, ResolvedExpr, SourceId};

use super::bind::{Bindings, BoundTable};

/// Whether the statement `input` (with the lowered SQL texts of its `PlanNode::Sql` nodes in
/// `bindings`) wrote a reference to `column` of the table read `securable` (an index of
/// [`Bindings::securables`]): a column reference naming it, lowered from the statement's own SQL
/// source or an embedded text's, whose resolved form in `root` (found by its position) reads that
/// column of a scan of that table (the attribute itself, as a bare column or `*` passes it
/// through; not a name the statement wrote for something else, nor a `*` expanded to it). A stored view's text (a source of its own), the
/// read projection the session builds over a view (no source) and DataFrame columns (no source)
/// are not the statement's: such a refusal names the table, failing toward less.
pub(super) fn wrote_column(
    root: &LogicalPlan,
    input: &LogicalPlan,
    bindings: &Bindings,
    securable: usize,
    column: &str,
) -> bool {
    let mut sources: Vec<SourceId> = Vec::new();
    sources.extend(input.origin.sql.map(|(s, _)| s));
    sources.extend(
        bindings
            .sql
            .values()
            .filter_map(|p| p.origin.sql.map(|(s, _)| s)),
    );
    let own = |origin: &Origin| origin.sql.is_some_and(|(s, _)| sources.contains(&s));
    // Where the statement wrote the name: its column references naming it (a `*`'s expansion
    // carries the star's position, never one of these).
    let mut at: Vec<Origin> = Vec::new();
    let mut seen_input: PtrSet<*const ()> = PtrSet::default();
    let mut walk = vec![Node::Plan(input)];
    let mut next = Vec::new();
    while let Some(node) = walk.pop() {
        if !seen_input.insert(node.key()) {
            continue;
        }
        match node {
            Node::Plan(p) => {
                if let PlanNode::Sql { .. } = &p.node
                    && let Some(lowered) = bindings.sql.get(&super::bind::addr(p))
                {
                    walk.push(Node::Plan(lowered));
                }
            }
            Node::Expr(e) => {
                if let ExprNode::Column(c) = &e.node
                    && own(&e.origin)
                    && c.name.iter().any(|n| n.eq_ignore_ascii_case(column))
                {
                    at.push(e.origin);
                }
            }
        }
        children(node, &mut next);
        walk.append(&mut next);
    }
    if at.is_empty() {
        return false;
    }
    // The names a scan of the securable carries.
    let mut names: Vec<Vec<String>> = bindings
        .tables
        .values()
        .filter_map(|bound| match bound {
            BoundTable::Found { meta, securable: i } if *i == securable => Some(meta.ident.parts()),
            _ => None,
        })
        .collect();
    names.extend(
        bindings
            .scans
            .iter()
            .filter(|(_, i)| **i == securable)
            .map(|(name, _)| name.clone()),
    );
    // Attribute ids (as seen from the root, past `Shift` instances) of the column's scans, and
    // those the statement's own column references read.
    let mut column_ids: HashSet<u64> = HashSet::new();
    let mut written: HashSet<u64> = HashSet::new();
    let mut seen: HashSet<(*const LogicalPlan, u64)> = HashSet::new();
    let mut seen_exprs: HashSet<(*const crate::Expr, u64)> = HashSet::new();
    // An expansion's (`ROLLUP`, `CUBE`, `GROUPING SETS`) bare-column entries: output id to the
    // input attribute it passes through, so a key written above it reads the scan's column.
    let mut passes: HashMap<u64, HashSet<u64>> = HashMap::new();
    let mut stack: Vec<(&LogicalPlan, u64)> = vec![(root, 0)];
    while let Some((p, off)) = stack.pop() {
        if !seen.insert((std::ptr::from_ref(p), off)) {
            continue;
        }
        let mut exprs: Vec<&crate::Expr> = Vec::new();
        p.for_each_expr(&mut |e| exprs.push(e));
        while let Some(e) = exprs.pop() {
            if !seen_exprs.insert((std::ptr::from_ref(e), off)) {
                continue;
            }
            match &e.node {
                ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c))
                    if at.contains(&e.origin) =>
                {
                    written.extend(c.id.0.checked_add(off));
                }
                ExprNode::Subquery { plan, .. } => stack.push((plan, off)),
                _ => {}
            }
            e.for_each_child(&mut |c| exprs.push(c));
        }
        match &p.node {
            PlanNode::Scan(s) => {
                if let ReadSource::Table { name, .. } = &s.source
                    && names.contains(name)
                {
                    column_ids.extend(
                        s.output
                            .iter()
                            .filter(|c| c.name.eq_ignore_ascii_case(column))
                            .filter_map(|c| c.id.0.checked_add(off)),
                    );
                }
            }
            PlanNode::Shift { input, offset } => {
                stack.push((input, off.saturating_add(*offset)));
                continue;
            }
            PlanNode::Expand {
                projections,
                output,
                ..
            } => {
                for row in projections.iter().filter(|row| row.len() == output.len()) {
                    for (e, out) in row.iter().zip(output) {
                        if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &e.node
                            && let (Some(from), Some(to)) =
                                (out.id.0.checked_add(off), c.id.0.checked_add(off))
                        {
                            passes.entry(from).or_default().insert(to);
                        }
                    }
                }
            }
            _ => {}
        }
        p.for_each_child(&mut |c| stack.push((c, off)));
    }
    // Close the written attributes over the expansions' pass-throughs.
    let mut todo: Vec<u64> = written.iter().copied().collect();
    while let Some(id) = todo.pop() {
        for &to in passes.get(&id).into_iter().flatten() {
            if written.insert(to) {
                todo.push(to);
            }
        }
    }
    !written.is_disjoint(&column_ids)
}
