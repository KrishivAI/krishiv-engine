// SPDX-License-Identifier: Apache-2.0
//! The column names a statement wrote (ADR 0048 §3 Denials, A1 §3 Listings; P3-T062 review r3
//! M1): a refusal of a column read names the column only when the statement wrote that name, so
//! a `*`, a `* EXCEPT`, a `table()` or a view's text never makes a refusal list a table's other
//! columns. Asked only once a column is refused, so the walk costs nothing on a granted query.

use crate::walk::{Node, PtrSet, children};
use crate::{ExprNode, JoinCondition, LogicalPlan, Origin, PlanNode, SourceId};

use super::bind::Bindings;
use super::views::DefinerTexts;

/// Whether the statement `root` (with the lowered SQL texts of its `PlanNode::Sql` nodes in
/// `bindings`) wrote `column` as a column name: a part of a column reference, a `* EXCEPT` path
/// or a `USING` key, compared ignoring ASCII case (over-names, never under). Only what the
/// statement's own text holds counts: a node lowered from the SQL source of `root` or of an
/// embedded text. A stored view's text (lowered from a source of its own) and the read
/// projection the session builds over it (no source) are not the statement's, nor is a
/// DataFrame column (no source: such a refusal names the table, failing toward less); the
/// inlined queries of definer views (`definer_texts`) are skipped whole.
pub(super) fn wrote_column(
    root: &LogicalPlan,
    bindings: &Bindings,
    definer_texts: &DefinerTexts,
    column: &str,
) -> bool {
    let same = |name: &String| name.eq_ignore_ascii_case(column);
    let mut sources: Vec<SourceId> = Vec::new();
    sources.extend(root.origin.sql.map(|(s, _)| s));
    sources.extend(
        bindings
            .sql
            .values()
            .filter_map(|p| p.origin.sql.map(|(s, _)| s)),
    );
    let own = |origin: &Origin| origin.sql.is_some_and(|(s, _)| sources.contains(&s));
    let mut seen: PtrSet<*const ()> = PtrSet::default();
    let mut stack = vec![Node::Plan(root)];
    let mut next = Vec::new();
    while let Some(node) = stack.pop() {
        if !seen.insert(node.key()) {
            continue;
        }
        match node {
            Node::Plan(p) => {
                if definer_texts.contains(p) {
                    continue;
                }
                match &p.node {
                    PlanNode::Sql { .. } => {
                        if let Some(lowered) = bindings.sql.get(&super::bind::addr(p)) {
                            stack.push(Node::Plan(lowered));
                        }
                    }
                    PlanNode::Join {
                        condition: JoinCondition::Using(keys),
                        ..
                    } if own(&p.origin) && keys.iter().any(same) => return true,
                    _ => {}
                }
            }
            Node::Expr(e) if own(&e.origin) => match &e.node {
                ExprNode::Column(c) if c.name.iter().any(same) => return true,
                ExprNode::Star(s) if s.except.iter().flatten().any(same) => return true,
                _ => {}
            },
            Node::Expr(_) => {}
        }
        children(node, &mut next);
        stack.append(&mut next);
    }
    false
}
