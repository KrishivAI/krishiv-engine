// SPDX-License-Identifier: Apache-2.0
//! Resource groups (P3-T003, `design/P3-T003-workload-management.md`): an
//! [`AdmissionController`] that routes each query to a leaf group by selector, runs it when the
//! group and its ancestors have room, queues it when they do not (refusing past `maxQueued`),
//! and dispatches queued queries by each parent's scheduling policy.
//!
//! The group file is Trino's `resource-groups.json` shape (`config.rs`); the tree and the
//! dispatcher are in `tree.rs`. Selectors on `source` and `clientTags` read what the client
//! asserts, so they route queries but never limit a tenant: use `user` / `userGroup` for that.

mod config;
mod select;
mod tree;

#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use krishiv_common::error::{
    InternalError, KrishivError, KrishivResourceGroupConfigInvalid, KrishivResourceGroupQueueFull,
};
use krishiv_plan::analyzer::BoxFuture;

use crate::admission::{AdmissionController, Permit, QueryRequest};
use config::Config;
use tree::{Admitted, Refusal, Shared};

pub use select::tag_matches;
pub use tree::{GroupRow, GroupSlot};

/// The admission controller for a resource group tree.
#[derive(Debug)]
pub struct ResourceGroups {
    config: Config,
    shared: Arc<Shared>,
}

impl ResourceGroups {
    /// Builds the controller from the text of a group file. `total_memory` is what a
    /// `"NN%"` memory limit in the file is a share of (`krishiv.memory.limit`).
    ///
    /// # Errors
    ///
    /// `KRISHIV_RESOURCE_GROUP_CONFIG_INVALID`, naming the key, for a malformed file, an
    /// unsupported key (CPU quotas, name templates), a missing default group or a selector that
    /// names no leaf.
    pub fn from_json(json: &str, total_memory: usize) -> Result<Self, KrishivError> {
        Self::with_clock(json, total_memory, Arc::new(wall_clock_ms))
    }

    /// As [`ResourceGroups::from_json`], with the clock (milliseconds since the Unix epoch)
    /// that scan-quota windows use, so a test moves time by hand.
    ///
    /// # Errors
    ///
    /// As [`ResourceGroups::from_json`].
    pub fn with_clock(
        json: &str,
        total_memory: usize,
        now_ms: Arc<dyn Fn() -> u64 + Send + Sync>,
    ) -> Result<Self, KrishivError> {
        let config = config::parse(json, total_memory)?;
        let shared = Shared::new(&config.roots, now_ms);
        Ok(Self { config, shared })
    }

    /// Builds the controller from the group file at `path` (`krishiv.workload.resourceGroups.file`).
    ///
    /// # Errors
    ///
    /// `KRISHIV_RESOURCE_GROUP_CONFIG_INVALID` when the file cannot be read, and as
    /// [`ResourceGroups::from_json`] otherwise.
    pub fn from_file(path: &str, total_memory: usize) -> Result<Self, KrishivError> {
        let json = std::fs::read_to_string(path).map_err(|e| {
            KrishivError::from(KrishivResourceGroupConfigInvalid {
                path: path.to_owned(),
                reason: format!("cannot read the file: {}", e.kind()),
            })
        })?;
        Self::from_json(&json, total_memory)
    }

    /// One row per group of the tree, parents first, with the counters now (the source of
    /// `system.resource_groups`).
    #[must_use]
    pub fn rows(&self) -> Vec<GroupRow> {
        self.shared.rows()
    }

    /// `(running, queued)` queries of the group `id` (a dotted path such as `root.etl`).
    #[must_use]
    pub fn counts(&self, id: &str) -> Option<(usize, usize)> {
        self.shared.counts(id)
    }
}

fn wall_clock_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

impl AdmissionController for ResourceGroups {
    fn admit<'a>(
        &'a self,
        request: &'a QueryRequest,
    ) -> BoxFuture<'a, Result<Permit, KrishivError>> {
        Box::pin(async move {
            let group = select::select(&self.config, request);
            let leaf = self.shared.leaf(group).ok_or_else(|| {
                KrishivError::from(InternalError {
                    message: format!("resource group {group} is not in the tree"),
                })
            })?;
            let budget = self.shared.per_query_memory(leaf);
            let slot = match self.shared.admit(leaf) {
                Err(Refusal::Quota(e)) => return Err(e),
                Err(Refusal::QueueFull { group, max_queued }) => {
                    return Err(KrishivResourceGroupQueueFull {
                        group,
                        max_queued: max_queued.to_string(),
                    }
                    .into());
                }
                Ok(Admitted::Now(slot)) => slot,
                Ok(Admitted::Queued(rx, guard)) => {
                    let granted = rx.await;
                    drop(guard);
                    granted.map_err(|_| {
                        KrishivError::from(InternalError {
                            message: "a queued query lost its resource group".to_owned(),
                        })
                    })??
                }
            };
            let mut permit = Permit::with_slot(slot).with_group(group);
            if let Some((priority, class)) = self.shared.leaf_traits(leaf) {
                permit = permit.with_priority(priority).with_class(class);
            }
            Ok(match budget {
                Some(bytes) => permit.with_memory_budget(bytes),
                None => permit,
            })
        })
    }

    fn uses_estimate(&self) -> bool {
        false
    }
}
