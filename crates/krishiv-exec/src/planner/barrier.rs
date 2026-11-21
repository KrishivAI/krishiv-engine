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
        let first = self.next_scan;
        let mut planned = self.plan(input)?;
        // Every scan numbered while planning the input is one of its scans (a shared definition
        // planned from inside counts too: over-marking only hides more).
        self.guarded.extend((first..self.next_scan).map(ScanId));
        planned.sites.clear();
        Ok(planned)
    }
}
