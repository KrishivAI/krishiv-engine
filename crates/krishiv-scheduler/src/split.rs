// SPDX-License-Identifier: Apache-2.0
//! The N-task split of nested-loop joins that return build rows (design
//! P2-T052b, change log 2026-11-06): right outer, full outer, right semi
//! and right anti over a broadcast build. Unsplit, such a join's stage runs
//! as one task ([`Stage::single`](crate::Stage::single)): each of several
//! tasks would emit the build rows. Split, the probe stays partitioned:
//!
//! ```text
//! NLJ(probe, Exchange(Broadcast)(B))
//!   => Union[ Exchange(RoundRobin)(NLJ{marks}(probe, Exchange(Broadcast)(B))),
//!             Exchange(RoundRobin)(UnmatchedBuild{marks, B}) ]
//! ```
//!
//! The local half emits the matched pairs (and, full outer, the unmatched
//! probe rows) and publishes its build-match bitmap; [`fragment()`]
//! cuts the `UnmatchedBuild` branch after it (union inputs in order), as a
//! one-task stage that reads `B` again in exchange order and the OR of the
//! bitmaps of every task of the local stage (task data). The parent
//! continues above the union in a stage of its own. The planner marks
//! these joins' output placement unknown, so nothing above relies on where
//! the rows are.
//!
//! [`fragment()`]: crate::fragment()

use arrow_schema::SchemaRef;
use krishiv_common::error::KrishivError;
use krishiv_exec::exchange::round_robin_routes;
use krishiv_exec::physical::{
    ExchangeId, ExchangePlan, MarksId, Partitioning, PhysNode, PhysicalPlan, PlanSchemas,
    UnmatchedBuildPlan,
};

type Result<T> = std::result::Result<T, KrishivError>;

/// `plan` with each splittable nested-loop join split (see the module
/// docs), and `schemas` with the columns of the exchanges it adds. Joins
/// in shared plans and recursive CTEs are not split: they run whole.
/// A join whose output the round-robin hash cannot route (a nested
/// column) stays whole too.
///
/// # Errors
/// `INTERNAL_ERROR` for a join whose output columns cannot be derived.
pub fn split_nested_loops(plan: &PhysicalPlan, schemas: &mut PlanSchemas) -> Result<PhysicalPlan> {
    let mut next_exchange = next_exchange(plan);
    let mut next_marks = 0_u64;
    let mut root = plan.root().clone();
    let mut split = 0;
    walk(
        &mut root,
        schemas,
        &mut next_exchange,
        &mut next_marks,
        &mut split,
    )?;
    if split == 0 {
        return Ok(plan.clone());
    }
    PhysicalPlan::new(
        root,
        SchemaRef::clone(plan.schema()),
        plan.explain().clone(),
        plan.env().clone(),
    )
    .with_shared(plan.shared().to_vec())
}

/// Splits the joins below and at `n` (children first: a split adds only
/// exchanges and leaves, so the order does not matter).
fn walk(
    n: &mut PhysNode,
    schemas: &mut PlanSchemas,
    next_exchange: &mut u64,
    next_marks: &mut u64,
    split: &mut usize,
) -> Result<()> {
    if matches!(n, PhysNode::UnionLoop { .. }) {
        return Ok(());
    }
    for c in n.children_mut() {
        walk(c, schemas, next_exchange, next_marks, split)?;
    }
    let PhysNode::NestedLoopJoin(j) = n else {
        return Ok(());
    };
    if j.marks.is_some() || !j.join_type.builds_preserved() {
        return Ok(());
    }
    let PhysNode::Exchange(build) = &*j.build else {
        return Ok(());
    };
    if build.partitioning != Partitioning::Broadcast {
        return Ok(());
    }
    let (Some(probe), Some(build_schema)) = (
        schemas.nested_loop_probes.get(&build.id).cloned(),
        schemas.exchanges.get(&build.id).cloned(),
    ) else {
        return Ok(());
    };
    let marks = MarksId(*next_marks);
    let unmatched = UnmatchedBuildPlan {
        marks,
        join_type: j.join_type,
        exchange: build.id,
        probe_schema: probe,
        build_schema,
    };
    let schema = unmatched.schema()?;
    if !round_robin_routes(&schema) {
        return Ok(());
    }
    *next_marks += 1;
    let (local, rest) = (ExchangeId(*next_exchange), ExchangeId(*next_exchange + 1));
    *next_exchange += 2;
    let mut join = j.clone();
    join.marks = Some(marks);
    schemas.exchanges.insert(local, SchemaRef::clone(&schema));
    schemas.exchanges.insert(rest, schema);
    let exchange = |id, input| {
        PhysNode::Exchange(ExchangePlan {
            id,
            input: Box::new(input),
            partitioning: Partitioning::RoundRobin,
            filters: Vec::new(),
        })
    };
    *n = PhysNode::Union {
        inputs: vec![
            exchange(local, PhysNode::NestedLoopJoin(join)),
            exchange(rest, PhysNode::UnmatchedBuild(unmatched)),
        ],
    };
    *split += 1;
    Ok(())
}

/// One past the largest exchange id `plan` uses.
fn next_exchange(plan: &PhysicalPlan) -> u64 {
    let mut next = 0_u64;
    let mut stack: Vec<&PhysNode> = plan.roots().collect();
    while let Some(n) = stack.pop() {
        let id = match n {
            PhysNode::Exchange(e) => Some(e.id),
            PhysNode::ExchangeSource { id, .. } => Some(*id),
            PhysNode::UnmatchedBuild(u) => Some(u.exchange),
            _ => None,
        };
        if let Some(id) = id {
            next = next.max(id.0.saturating_add(1));
        }
        stack.extend(n.children());
    }
    next
}

/// The ids of the bitmaps the split joins of `fragment` publish.
#[must_use]
pub fn published_marks(fragment: &PhysicalPlan) -> Vec<MarksId> {
    let mut out = Vec::new();
    let mut stack: Vec<&PhysNode> = fragment.roots().collect();
    while let Some(n) = stack.pop() {
        if let PhysNode::NestedLoopJoin(j) = n
            && let Some(m) = j.marks
        {
            out.push(m);
        }
        stack.extend(n.children());
    }
    out.sort_unstable();
    out
}
