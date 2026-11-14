// SPDX-License-Identifier: Apache-2.0
//! Generators in DataFrame select lists: the lifting itself is [`krishiv_plan::select_with_generators`]
//! and [`krishiv_plan::with_column_generator`], shared with the Connect translation.

use std::sync::Arc;

use krishiv_plan::{Expr, Origin, PlanNode};

use super::DataFrame;

impl DataFrame {
    fn next_origin(&self) -> Origin {
        Origin::plan(self.session.next_plan_id())
    }

    /// `SELECT items`, the generator calls among them lifted into `Generate` nodes.
    pub(super) fn select_lifting_generators(&self, items: Vec<Arc<Expr>>) -> Self {
        self.derive(krishiv_plan::select_with_generators(
            Arc::clone(&self.plan),
            items,
            || self.next_origin(),
        ))
    }

    /// `withColumn(name, value)` for a generator `value`, `None` for any other.
    pub(super) fn with_column_generator(&self, name: &str, value: &Arc<Expr>) -> Option<Self> {
        let node: PlanNode =
            krishiv_plan::with_column_generator(&self.plan, name, value, || self.next_origin())?;
        Some(self.derive(node))
    }
}
