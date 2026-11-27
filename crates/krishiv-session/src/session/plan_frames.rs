// SPDX-License-Identifier: Apache-2.0
//! Frames over a caller's plan (`from_plan` and its variants): the statement such a frame lists its runs under.

use super::*;

impl Session {
    /// A frame of `root` on `session`. A plan over a `spark.sql` relation (`PlanNode::Sql`, what
    /// Spark Connect sends for a query it analyzed earlier, alone for `collect()` or under the
    /// client's own nodes for `count()`, `filter(...)` or `first()`) lists its runs in
    /// `system.queries` with that statement's text (secrets masked) and class, as a frame derived
    /// from `Session::sql` does; with several such relations it is the first one, as there. A plan
    /// with none stays without text. A statement that does not parse keeps none: its analysis
    /// reports the error.
    pub(super) fn plan_frame(session: Session, root: Arc<LogicalPlan>) -> DataFrame {
        let statement = session.plan_statement(&root);
        let df = DataFrame::new(session, root);
        match statement {
            Some(env) => df.with_first_statement(env),
            None => df,
        }
    }

    /// The statement a frame of `root` lists its runs under (see [`Self::plan_frame`]): the text
    /// and class of the first `spark.sql` relation in `root`, in this session's environment.
    pub(super) fn plan_statement(&self, root: &LogicalPlan) -> Option<QueryEnv> {
        fn first_sql(plan: &LogicalPlan) -> Option<&str> {
            if let PlanNode::Sql { query, .. } = &plan.node {
                return Some(query);
            }
            let mut found = None;
            plan.for_each_child(&mut |child| {
                if found.is_none() {
                    found = first_sql(child);
                }
            });
            found
        }
        let query = first_sql(root)?;
        let mut env = self.query_env();
        let redacted = krishiv_sql::redact_sql(query);
        let parsed = krishiv_sql::parse_statement(query, &env.parser).ok()?;
        env.set_text(&redacted);
        env.statement_class = Some(self.statement_class(&parsed, &env).name());
        Some(env)
    }
}
