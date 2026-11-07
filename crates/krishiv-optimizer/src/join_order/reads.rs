// SPDX-License-Identifier: Apache-2.0
//! Which columns are read above a join tree (E5b, `krishiv.optimizer.joinCost`
//! = `bytes`).
//!
//! Join ordering runs before column pruning, so a join input's output still
//! carries columns nothing reads. Costing a tree by bytes counts only the
//! columns its consumers read: pruning's own requirements
//! ([`crate::prune::requirements`]), computed once over the query, parents
//! first, so a projection that passes a column nobody reads above does not
//! keep it, positional consumers read every column, and a tree shared by
//! several consumers sees the union of what they read. The same plan
//! after pruning has the same requirements, so the cost, and the plan, is
//! the same when the optimized plan is optimized again.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_plan::LogicalPlan;

use crate::expr::Attrs;

/// What the query reads of each of its relations, by address.
pub(super) struct Reads {
    req: HashMap<*const LogicalPlan, Attrs>,
}

impl Reads {
    /// The reads of the query `root`; valid while `root` is alive.
    pub(super) fn of_query(root: &Arc<LogicalPlan>) -> Self {
        Self {
            req: crate::prune::requirements(root),
        }
    }

    /// The columns read of `tree`'s output; all of them for a relation the
    /// query does not hold.
    pub(super) fn above(&self, tree: &Arc<LogicalPlan>) -> Attrs {
        self.req
            .get(&Arc::as_ptr(tree))
            .cloned()
            .unwrap_or_else(|| {
                tree.resolved_output()
                    .unwrap_or_default()
                    .iter()
                    .map(|c| c.id)
                    .collect()
            })
    }
}
