// SPDX-License-Identifier: Apache-2.0
//! The query profile: `EXPLAIN ANALYZE`'s data.
//!
//! A [`QueryProfile`] is the physical plan tree (as `EXPLAIN` prints it)
//! with the executor's [`QueryMetrics`]: every operator names the plan node
//! it implements by the node's pre-order index, so the profile prints each
//! node with its operators' counters ([`QueryProfile::to_text`]) and
//! serializes to a versioned JSON document ([`QueryProfile::to_json`],
//! format [`PROFILE_FORMAT`] version [`PROFILE_VERSION`]; the schema is
//! `docs/reference/query-profile-v1.schema.json`). The document is
//! experimental: it may change before 1.0.
//!
//! A profile is plain data owned by the caller. Profiles of one plan run by
//! several workers merge ([`QueryProfile::merge`]) as their metrics do.

mod json;
mod text;

use krishiv_common::error::{KrishivError, KrishivInvalidProfileDocument};

use crate::metrics::QueryMetrics;
use crate::physical::{PhysNode, PhysicalPlan};
use crate::planner::ExplainNode;

pub use text::TextOptions;

/// The `format` member of a profile document.
pub const PROFILE_FORMAT: &str = "krishiv.query-profile";
/// The version of the profile document this build writes and reads. A
/// change that removes or retypes a member bumps it.
pub const PROFILE_VERSION: u64 = 1;

/// A node of the profiled plan: `EXPLAIN`'s node with its id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileNode {
    /// The node's pre-order index in the physical plan, which operators
    /// name ([`OperatorEntry::node`](crate::OperatorEntry::node)). `None`
    /// for a node `EXPLAIN` adds that is not a plan node (a subquery's
    /// heading) or one that could not be matched.
    pub id: Option<u32>,
    /// The node, e.g. `HashJoin`.
    pub name: String,
    /// Its arguments, as `EXPLAIN` prints them.
    pub detail: String,
    /// Its inputs.
    pub children: Vec<ProfileNode>,
}

impl ProfileNode {
    /// The tree of `plan` with node ids.
    #[must_use]
    pub fn of(plan: &PhysicalPlan) -> Self {
        let mut next = 0;
        annotate(plan.root(), plan.explain(), &mut next)
    }

    fn unmatched(e: &ExplainNode) -> Self {
        Self {
            id: None,
            name: e.name.clone(),
            detail: e.detail.clone(),
            children: e.children.iter().map(Self::unmatched).collect(),
        }
    }

    /// Every node, depth first.
    pub(crate) fn walk(&self, f: &mut impl FnMut(&Self)) {
        f(self);
        for c in &self.children {
            c.walk(f);
        }
    }
}

/// `EXPLAIN` names a subquery a node reads with a `Subquery` heading over
/// the subquery's plan; the physical plan has no node for the heading.
const SUBQUERY: &str = "Subquery";

/// Numbers `p` and its subtree in pre-order ([`PhysNode::children`] order,
/// as instantiation does) while walking `e`, the same subtree as `EXPLAIN`
/// prints it. Where the two disagree on a node's inputs, the inputs keep
/// their text without ids (and the numbering skips their subtrees).
fn annotate(p: &PhysNode, e: &ExplainNode, next: &mut u32) -> ProfileNode {
    let id = *next;
    *next = next.saturating_add(1);
    let inputs = p.children();
    let children = if inputs.len() == e.children.len() {
        inputs
            .into_iter()
            .zip(&e.children)
            .map(|(p, e)| match e.children.as_slice() {
                [inner] if e.name == SUBQUERY => ProfileNode {
                    id: None,
                    name: e.name.clone(),
                    detail: e.detail.clone(),
                    children: vec![annotate(p, inner, next)],
                },
                _ => annotate(p, e, next),
            })
            .collect()
    } else {
        for p in inputs {
            *next = next.saturating_add(size(p));
        }
        e.children.iter().map(ProfileNode::unmatched).collect()
    };
    ProfileNode {
        id: Some(id),
        name: e.name.clone(),
        detail: e.detail.clone(),
        children,
    }
}

fn size(p: &PhysNode) -> u32 {
    p.children()
        .into_iter()
        .fold(1_u32, |n, c| n.saturating_add(size(c)))
}

/// A profiled query: its plan and its metrics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryProfile {
    /// The physical plan, with node ids.
    pub plan: ProfileNode,
    /// The executor's metrics.
    pub metrics: QueryMetrics,
    /// Whether the costlier detailed metrics were collected
    /// ([`ExecConfig::detailed_metrics`](crate::ExecConfig::detailed_metrics)).
    pub detailed: bool,
}

impl QueryProfile {
    /// The profile of `plan` run with `metrics`.
    #[must_use]
    pub fn new(plan: &PhysicalPlan, metrics: QueryMetrics, detailed: bool) -> Self {
        Self {
            plan: ProfileNode::of(plan),
            metrics,
            detailed,
        }
    }

    /// Adds `o`, another worker's profile of the same plan.
    ///
    /// # Errors
    /// [`ProfileError`] if `o` profiles another plan.
    pub fn merge(&mut self, o: &Self) -> Result<(), ProfileError> {
        if self.plan != o.plan {
            return Err(ProfileError::new(
                "cannot merge profiles of different plans",
            ));
        }
        self.metrics.merge(&o.metrics);
        self.detailed &= o.detailed;
        Ok(())
    }
}

/// A profile document that does not parse, or profiles that do not merge.
/// As a [`KrishivError`] it is `KRISHIV_INVALID_PROFILE_DOCUMENT`.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid query profile: {0}")]
pub struct ProfileError(String);

impl ProfileError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl From<ProfileError> for KrishivError {
    fn from(e: ProfileError) -> Self {
        KrishivInvalidProfileDocument {
            document: PROFILE_FORMAT.to_owned(),
            reason: e.0,
        }
        .into()
    }
}

#[cfg(all(test, not(loom)))]
mod tests;
