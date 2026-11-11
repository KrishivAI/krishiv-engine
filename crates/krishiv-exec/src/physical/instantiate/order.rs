// SPDX-License-Identifier: Apache-2.0
//! Which rows keep a `range`'s order: the nodes a `LIMIT` or the result read
//! a range through, and the `PythonEval` nodes among them (design P2-T059
//! §6.3), which keep it only where something above needs it.

use std::collections::HashSet;

use crate::physical::PhysNode;

/// Whether `n` streams a `range` through row-wise nodes only (filters,
/// projections, limits and Python UDF calls), so its rows reach a `LIMIT` or
/// the result in a defined order.
pub(super) fn reads_range_in_order(n: &PhysNode) -> bool {
    match n {
        PhysNode::Range(_) => true,
        PhysNode::Filter { input, .. }
        | PhysNode::Project { input, .. }
        | PhysNode::Limit { input, .. }
        | PhysNode::PythonEval { input, .. } => reads_range_in_order(input),
        _ => false,
    }
}

/// The `PythonEval` nodes (by address) whose output must keep the order of
/// the `range` below them: those a `LIMIT`, or `result` (the plan's root,
/// when given), reads the range through. Any other one reads its
/// results with several drivers, since nothing above needs the order. The
/// addresses are valid only for the plan borrowed while it is instantiated
/// (`Detached` computes its own per round).
pub(super) fn range_ordered_evals<'a>(
    roots: impl IntoIterator<Item = &'a PhysNode>,
    result: Option<&PhysNode>,
) -> HashSet<usize> {
    fn chain(n: &PhysNode, out: &mut HashSet<usize>) {
        match n {
            PhysNode::PythonEval { input, .. } => {
                out.insert(std::ptr::from_ref(n).addr());
                chain(input, out);
            }
            PhysNode::Filter { input, .. }
            | PhysNode::Project { input, .. }
            | PhysNode::Limit { input, .. } => chain(input, out),
            _ => {}
        }
    }
    let mut out = HashSet::new();
    if let Some(r) = result.filter(|r| reads_range_in_order(r)) {
        chain(r, &mut out);
    }
    for root in roots {
        let mut stack = vec![root];
        while let Some(n) = stack.pop() {
            if let PhysNode::Limit { input, .. } = n
                && reads_range_in_order(input)
            {
                chain(input, &mut out);
            }
            stack.extend(n.children());
        }
    }
    out
}
