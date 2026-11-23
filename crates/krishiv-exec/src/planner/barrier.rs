// SPDX-License-Identifier: Apache-2.0
//! A security barrier at lowering (design `P3-E14-security-barrier.md` §6): no physical node. The
//! input is planned as it is, its scans join the planner's guarded set (which the finished plan
//! carries for `profile_scans`, and which probe fusion skips), and its runtime-filter sites are
//! cleared, so no join's build filters a scan inside it.

use std::sync::Arc;

use krishiv_plan::LogicalPlan;

use super::build::{Planned, Planner};
use crate::error::Result;
use crate::physical::ScanId;

impl Planner<'_> {
    /// The planned input of a barrier over `input`.
    pub(super) fn barrier(&mut self, input: &Arc<LogicalPlan>) -> Result<Planned> {
        let (planned, scans) = self.planned_with_scans(input);
        let mut planned = planned?;
        // Every scan the input reads: those numbered while planning it (a shared definition
        // planned from inside counts too: over-marking only hides more) and those of a shared
        // plan an earlier reference outside the barrier planned first.
        self.guarded.extend(scans);
        planned.sites.clear();
        Ok(planned)
    }

    /// `input` planned, with every scan it reads: the ids numbered while planning it and the
    /// scans of the shared plans it reads that were planned before. They are also counted for
    /// the enclosing barrier or shared plan, if any.
    pub(super) fn planned_with_scans(
        &mut self,
        input: &Arc<LogicalPlan>,
    ) -> (Result<Planned>, Vec<ScanId>) {
        let first = self.next_scan;
        self.scan_reads.push(Vec::new());
        let planned = self.plan(input);
        let mut scans = self.scan_reads.pop().unwrap_or_default();
        scans.extend((first..self.next_scan).map(ScanId));
        // A set: a shared plan read twice (and so on up a chain of shared plans) would otherwise
        // double its ids at every level.
        scans.sort_unstable();
        scans.dedup();
        if let Some(parent) = self.scan_reads.last_mut() {
            parent.extend(scans.iter().copied());
        }
        (planned, scans)
    }
}
