// SPDX-License-Identifier: Apache-2.0
//! A `SecurityBarrier` hides what its input's projection dropped: a parent
//! resolves only the barrier's output columns, so `ORDER BY secret` above a
//! barrier over `SELECT id` fails instead of sorting by the hidden column.

use std::sync::Arc;

use krishiv_common::PrincipalKey;
use krishiv_common::error::ErrorClass;

use super::dsl::{c, int};
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AnalysisError, AnalyzedPlan, Analyzer as _, DefaultAnalyzer, MemoryCatalog, Principal,
    PrincipalKind, ResolutionConf, Securable,
};
use crate::{LogicalPlan, Origin, Plan, PlanNode, PolicyRef, SortOrder};

fn analyze(plan: LogicalPlan) -> Result<AnalyzedPlan, AnalysisError> {
    let ctx = context(
        MemoryCatalog::new("spark_catalog"),
        ResolutionConf::default(),
    );
    let plan = Plan::try_new(plan).map_err(|error| AnalysisError {
        error,
        origin: Origin::UNKNOWN,
    })?;
    block_on(DefaultAnalyzer.analyze(&plan, &ctx))
}

/// `SELECT id FROM VALUES (1, 2) AS (id, secret)`.
fn visible_id() -> LogicalPlan {
    LogicalPlan::new(PlanNode::InlineTable {
        rows: vec![vec![Arc::new(int(1)), Arc::new(int(2))]],
        columns: vec!["id".to_owned(), "secret".to_owned()],
    })
    .project(vec![c("id")])
}

fn barrier(input: LogicalPlan) -> LogicalPlan {
    let principal = Principal::new("alice".to_owned(), PrincipalKind::User);
    LogicalPlan::new(PlanNode::SecurityBarrier {
        input: Arc::new(input),
        policy: PolicyRef::new(
            Securable::System("t".to_owned()),
            PrincipalKey::of(&principal),
            1,
        ),
    })
}

#[test]
fn a_sort_above_a_barrier_cannot_reach_a_dropped_column() {
    // Control: without the barrier the sort widens through the projection.
    let open = analyze(visible_id().sort(vec![SortOrder::asc(c("secret"))]))
        .expect("the sort widens the projection to secret");
    assert_eq!(open.output.len(), 1);

    let err = analyze(barrier(visible_id()).sort(vec![SortOrder::asc(c("secret"))]))
        .expect_err("secret is not an output of the barrier");
    assert_eq!(
        err.error.error_class(),
        ErrorClass::UnresolvedColumnWithSuggestion
    );
}

#[test]
fn a_sort_above_a_barrier_resolves_its_output_columns() {
    let ok = analyze(barrier(visible_id()).sort(vec![SortOrder::asc(c("id"))]))
        .expect("id is an output of the barrier");
    assert_eq!(ok.output.len(), 1);
}
