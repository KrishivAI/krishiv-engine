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
        | PhysNode::Sample { input, .. }
        | PhysNode::PythonEval { input, .. } => reads_range_in_order(input),
        _ => false,
    }
}

/// The most rows (`offset + limit`) a `LIMIT` over a filterless scan reads on one driver
/// ([`reads_scan_in_order`]): the first morsel holds that many rows or the read moves on to the next in order, so
/// a larger limit would turn a parallel scan into a serial one for little of the benefit.
pub(super) const SCAN_LIMIT_IN_ORDER_ROWS: u64 = 1 << 20;

/// Whether `n` streams a file or table scan with no filter of its own (pushed or runtime) through projections and
/// limits only, to a `LIMIT`. Spark's `executeTake` reads the scan's partitions in index order, so a `LIMIT` over
/// one file returns the file's first rows; here several drivers would each read a morsel and the limit would keep
/// whichever batch came first (PY-067). One driver claims the morsels in order, and the limit needs only the first
/// batch, so no parallelism is lost. A filter between (which may make the limit read much more) keeps its drivers.
pub(super) fn reads_scan_in_order(n: &PhysNode) -> bool {
    match n {
        PhysNode::Scan(s) => {
            s.filter.is_none()
                && s.runtime_filters.is_empty()
                && !s.row_id
                && !matches!(s.scan.source, krishiv_plan::ReadSource::Local(_))
        }
        PhysNode::Project { input, .. } | PhysNode::Limit { input, .. } => {
            reads_scan_in_order(input)
        }
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
            | PhysNode::Sample { input, .. }
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
