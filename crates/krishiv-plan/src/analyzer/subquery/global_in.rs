// SPDX-License-Identifier: Apache-2.0
//! `[NOT] IN (SELECT .. WHERE k = o [HAVING ..])` over a global aggregate.
//!
//! A global aggregate has one row for every outer row, matches or not, and a `HAVING` can reject that row
//! (`c1 IN (SELECT 1 FROM t2 WHERE t2.c1 = t1.c1 HAVING count(*) = 0)` is true for an outer row without matches).
//! Turning the `IN` into `v = (scalar subquery)` cannot tell a rejected row from a NULL value, which matters for
//! `NOT IN` (an empty set is true, a NULL value is not). So the subquery keeps its rows per outer row through the
//! domain, as a lateral join's right side does, and the `IN` becomes a semi or anti join on them.

use super::{Lifted, LogicalPlan, Pred, R, Rewriter, on_path};
use crate::{Grouping, PlanNode};
use std::sync::Arc;

/// A global aggregate: grouped by nothing.
fn global_aggregate(p: &LogicalPlan) -> bool {
    matches!(&p.node, PlanNode::Aggregate { grouping: Grouping::GroupBy(keys), .. } if keys.is_empty())
}

impl Rewriter {
    /// Whether `pred` is an `IN` whose subquery holds a global aggregate on its correlation path. (Under the
    /// legacy count handling the aggregate is grouped by the correlation instead, see `pull`.)
    pub(super) fn global_in(&self, pred: &Pred) -> bool {
        !self.legacy_exists_count
            && !pred.values.is_empty()
            && on_path(&pred.plan, global_aggregate)
    }

    /// [`Self::domain`] for an `IN`: a global aggregate on the way keeps its one row per domain value.
    pub(super) fn domain_for_in(
        &mut self,
        base: &Arc<LogicalPlan>,
        input: &Arc<LogicalPlan>,
    ) -> R<Lifted> {
        let was = self.lateral;
        self.lateral = was || on_path(input, global_aggregate);
        let lifted = self.domain(base, input);
        self.lateral = was;
        lifted
    }
}
