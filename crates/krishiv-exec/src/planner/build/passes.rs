// SPDX-License-Identifier: Apache-2.0
//! The planning entry point and its passes: the plan with its shared definitions inlined, then shared.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{AttrId, BoundColumn, LogicalPlan, Plan};

use super::super::probe_select::attach_runtime_filters;
use super::super::{PhysicalPlanner, PlanningContext};
use super::{Copy, Planner, internal, select};
use crate::error::Result;
use crate::physical::PhysicalPlan;

/// The embedded physical planner.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultPlanner;

impl PhysicalPlanner for DefaultPlanner {
    fn plan(&self, plan: &Plan, cx: &PlanningContext) -> Result<PhysicalPlan> {
        let root = plan.root();
        if !root.is_resolved() {
            return Err(internal("the physical planner needs a resolved plan"));
        }
        let output = root
            .resolved_output()
            .ok_or_else(|| internal("the plan has no resolved output"))?;
        // Planning recurses once per relation; plans nest up to
        // `krishiv_plan::MAX_DEPTH` levels.
        let run = || {
            // First as a plan with its deterministic shared definitions
            // inlined per reference (as before); the groups no runtime
            // filter reaches are then planned shared (design P2-T198 3.1).
            //
            // A source that decides once runs user code to decide, so the first pass leaves it
            // undecided and the pass that is kept decides (once per plan, not once per pass).
            let (plan, shareable, deferred) = plan_once(cx, root, &output, None, false)?;
            if shareable.is_empty() && !deferred {
                return Ok::<_, KrishivError>(plan);
            }
            Ok(plan_once(
                cx,
                root,
                &output,
                (!shareable.is_empty()).then_some(shareable),
                true,
            )?
            .0)
        };
        krishiv_common::stack::on_big_stack(run)
            .ok_or_else(|| internal("cannot start a planner thread"))?
    }
}

/// One planning pass; the definitions of `share` are shared. Also returns the
/// deterministic groups a second pass may share: two or more references whose
/// copies no runtime filter reaches.
fn plan_once(
    cx: &PlanningContext,
    root: &Arc<LogicalPlan>,
    output: &[BoundColumn],
    share: Option<HashSet<Vec<AttrId>>>,
    decide: bool,
) -> Result<(PhysicalPlan, HashSet<Vec<AttrId>>, bool)> {
    let mut planner = Planner::new(cx, root);
    planner.share_groups = share;
    planner.decide = decide && cx.decisions;
    let planned = planner.plan(root)?;
    let mut planned = select(planned, output, "Project")?;
    // A group is shareable when no copy's scan receives a filter from outside the copy: a join inside the body
    // keeps filtering the body's own scans when the body is shared (design 3.1, 6.1).
    let reached = |copy: &Copy| {
        planner.scan_filters.iter().any(|(scan, filters)| {
            copy.scans.contains(&scan.0) && filters.iter().any(|f| !copy.filters.contains(&f.id.0))
        })
    };
    let shareable: HashSet<Vec<AttrId>> = planner
        .copies
        .iter()
        .filter(|(_, copies)| copies.len() >= 2 && !copies.iter().any(reached))
        .map(|(key, _)| key.clone())
        .collect();
    let mut shared = std::mem::take(&mut planner.shared);
    planner.shared_reads.strip_dead(&mut shared);
    let fuse = planner.fuse_probe_select();
    attach_runtime_filters(
        &mut planner.scan_filters,
        fuse,
        shared
            .iter_mut()
            .map(|s| &mut s.plan)
            .chain(std::iter::once(&mut planned.node)),
    );
    let plan = PhysicalPlan::new(
        planned.node,
        planned.schema,
        planned.explain,
        planner.env.to_plan(),
    )
    .with_shared(shared)?
    .with_warnings(planner.warnings)
    .with_guarded(planner.guarded.into_iter().collect());
    Ok((plan, shareable, planner.deferred))
}
