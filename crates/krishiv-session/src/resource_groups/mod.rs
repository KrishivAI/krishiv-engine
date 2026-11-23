// SPDX-License-Identifier: Apache-2.0
//! Resource groups: an [`AdmissionController`] that routes each query to a leaf group by selector, runs it when the
//! group and its ancestors have room, queues it when they do not (refusing past `maxQueued`),
//! and dispatches queued queries by each parent's scheduling policy.
//!
//! The group file is Trino's `resource-groups.json` shape (`config.rs`); the tree and the
//! dispatcher are in `tree.rs`. Selectors on `source` and `clientTags` read what the client
//! asserts, so they route queries but never limit a tenant: use `user` / `userGroup` for that.

mod config;
mod replace;
mod select;
mod tree;

#[cfg(test)]
mod tests;

use std::fmt::Write as _;
use std::sync::{Arc, PoisonError, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use krishiv_common::error::{
    InternalError, KrishivError, KrishivResourceGroupConfigInvalid, KrishivResourceGroupQueueFull,
};
use krishiv_plan::analyzer::BoxFuture;

use crate::admission::{AdmissionController, Permit, QueryRequest};
use config::Config;
use sha2::{Digest as _, Sha256};
use tree::{Admitted, Refusal, Shared};

pub use select::tag_matches;
pub use tree::{GroupRow, GroupSlot};

/// The admission controller for a resource group tree.
#[derive(Debug)]
pub struct ResourceGroups {
    loaded: RwLock<Arc<Loaded>>,
    total_memory: usize,
    shared: Arc<Shared>,
    /// One replace of the group file at a time.
    replacing: std::sync::Mutex<()>,
    /// The group each waiting `admit` queued its request in, keyed by the request's address (it
    /// is borrowed for the whole wait): `queued_group` answers with it, not a fresh selection a
    /// reload may have changed.
    waiting: Arc<std::sync::Mutex<std::collections::HashMap<usize, String>>>,
}

/// Forgets a waiting request when its `admit` ends or is dropped.
struct WaitingEntry {
    map: Arc<std::sync::Mutex<std::collections::HashMap<usize, String>>>,
    key: usize,
}

impl Drop for WaitingEntry {
    fn drop(&mut self) {
        self.map
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&self.key);
    }
}

/// The document the tree was last built from.
#[derive(Debug)]
struct Loaded {
    config: Config,
    /// The document as canonical JSON.
    document: String,
    /// SHA-256 of `document`, in hex.
    etag: String,
    loaded_at_ms: u64,
}

/// A document as canonical JSON and its SHA-256 (hex): the same document always gives the same pair.
fn canonical(json: &str) -> (String, String) {
    // `serde_json` keeps object keys sorted, so the same document always prints the same.
    let document = serde_json::from_str::<serde_json::Value>(json)
        .map_or_else(|_| json.to_owned(), |v| v.to_string());
    let etag = Sha256::digest(document.as_bytes())
        .iter()
        .fold(String::new(), |mut hex, b| {
            let _ = write!(hex, "{b:02x}");
            hex
        });
    (document, etag)
}

impl Loaded {
    fn new(json: &str, config: Config, now_ms: u64) -> Self {
        let (document, etag) = canonical(json);
        Self {
            config,
            document,
            etag,
            loaded_at_ms: now_ms,
        }
    }
}

impl ResourceGroups {
    /// Builds the controller from the text of a group file.
    ///
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
        let loaded = Arc::new(Loaded::new(json, config, shared.now()));
        Ok(Self {
            loaded: RwLock::new(loaded),
            total_memory,
            shared,
            replacing: std::sync::Mutex::new(()),
            waiting: Arc::default(),
        })
    }

    fn current(&self) -> Arc<Loaded> {
        Arc::clone(&self.loaded.read().unwrap_or_else(PoisonError::into_inner))
    }

    /// Checks a candidate document without changing anything (`ValidateResourceGroupConfig`).
    ///
    /// # Errors
    ///
    /// As [`ResourceGroups::from_json`].
    pub fn validate(json: &str, total_memory: usize) -> Result<(), KrishivError> {
        config::parse(json, total_memory).map(drop)
    }

    /// Every problem of `json` (empty when it is valid): one per bad group, selector or
    /// reference, each with its path.
    #[must_use]
    pub fn problems(json: &str, total_memory: usize) -> Vec<KrishivError> {
        config::problems(json, total_memory)
    }

    /// Swaps the group tree for the one `json` describes.
    ///
    /// Swaps the tree for the one `json` describes. A group the new document keeps (by dotted
    /// id) keeps its counters, its queue and its running queries and takes the new limits; a
    /// new group starts empty; a removed group takes no new query and is freed when its last
    /// one ends. Selectors and the default group change at once. An invalid document is
    /// refused and the old tree stays.
    ///
    /// # Errors
    ///
    /// As [`ResourceGroups::from_json`].
    pub fn reload_json(&self, json: &str) -> Result<(), KrishivError> {
        // Not while a replace checks the file and writes it.
        let _one = self
            .replacing
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.swap(json).map(drop)
    }

    /// The swap of [`ResourceGroups::reload_json`], for a caller holding `replacing`. Returns the
    /// document, etag and load time it installed, read under the same lock as the swap.
    fn swap(&self, json: &str) -> Result<(String, String, u64), KrishivError> {
        let config = config::parse(json, self.total_memory)?;
        // Held across the swap, so a query that selects a group after the swap sees the new
        // selectors and one that selected before it asks again when its group is gone.
        let mut guard = self.loaded.write().unwrap_or_else(PoisonError::into_inner);
        self.shared.reload(&config.roots);
        let loaded = Arc::new(Loaded::new(json, config, self.shared.now()));
        let installed = (
            loaded.document.clone(),
            loaded.etag.clone(),
            loaded.loaded_at_ms,
        );
        *guard = loaded;
        Ok(installed)
    }

    /// As [`ResourceGroups::reload_json`], from the file at `path`; returns the document, etag and
    /// load time it installed.
    ///
    /// # Errors
    ///
    /// `KRISHIV_RESOURCE_GROUP_CONFIG_INVALID` when the file cannot be read, and as
    /// [`ResourceGroups::from_json`] otherwise.
    pub fn reload_file(&self, path: &str) -> Result<(String, String, u64), KrishivError> {
        let _one = self
            .replacing
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.swap(&read_file(path)?)
    }

    /// The document, its etag and load time from one snapshot (a reload between three reads
    /// would pair an old document with a new etag).
    pub(crate) fn snapshot(&self) -> (String, String, u64) {
        let l = self.current();
        (l.document.clone(), l.etag.clone(), l.loaded_at_ms)
    }

    /// The loaded document as canonical JSON.
    #[must_use]
    pub fn document(&self) -> String {
        self.current().document.clone()
    }

    /// SHA-256 (hex) of [`ResourceGroups::document`]: what a replace is checked against.
    #[must_use]
    pub fn etag(&self) -> String {
        self.current().etag.clone()
    }

    /// When the loaded document was read, in milliseconds since the Unix epoch.
    #[must_use]
    pub fn loaded_at_ms(&self) -> u64 {
        self.current().loaded_at_ms
    }

    /// Builds the controller from the group file at `path` (`krishiv.workload.resourceGroups.file`).
    ///
    /// # Errors
    ///
    /// `KRISHIV_RESOURCE_GROUP_CONFIG_INVALID` when the file cannot be read, and as
    /// [`ResourceGroups::from_json`] otherwise.
    pub fn from_file(path: &str, total_memory: usize) -> Result<Self, KrishivError> {
        Self::from_json(&read_file(path)?, total_memory)
    }

    /// One row per group of the tree, parents first, with the counters now.
    ///
    /// One row per group of the tree, parents first, with the counters now (the source of
    /// `system.resource_groups`).
    #[must_use]
    pub fn rows(&self) -> Vec<GroupRow> {
        // The loaded read guard is held through the row collection: `swap` takes the write lock
        // before it changes the tree, so the rows and the etag come from one document.
        let loaded = self.loaded.read().unwrap_or_else(PoisonError::into_inner);
        let etag = loaded.etag.clone();
        let loaded_at_ms = i64::try_from(loaded.loaded_at_ms).unwrap_or(i64::MAX);
        let mut rows = self.shared.rows();
        drop(loaded);
        // A removed group came from an earlier document: it carries neither this etag nor this time.
        for row in rows.iter_mut().filter(|r| r.state != "removed") {
            row.config_etag.clone_from(&etag);
            row.loaded_at_ms = loaded_at_ms;
        }
        rows
    }

    /// `(running, queued)` queries of the group `id` (a dotted path such as `root.etl`).
    #[must_use]
    pub fn counts(&self, id: &str) -> Option<(usize, usize)> {
        self.shared.counts(id)
    }
}

fn read_file(path: &str) -> Result<String, KrishivError> {
    std::fs::read_to_string(path).map_err(|e| {
        KrishivError::from(KrishivResourceGroupConfigInvalid {
            path: path.to_owned(),
            reason: format!("cannot read the file: {}", e.kind()),
        })
    })
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
            // A reload between the selector and the admission removes the group the selector
            // chose: select again against the new document.
            let mut tries = 0;
            let (admitted, traits, group) = loop {
                let group = select::select(&self.current().config, request).to_owned();
                match self.shared.admit(&group) {
                    Ok((admitted, traits)) => break (admitted, traits, group),
                    Err(Refusal::Missing) if tries < 3 => tries += 1,
                    Err(Refusal::Missing) => {
                        return Err(KrishivError::from(InternalError {
                            message: format!("resource group {group} is not in the tree"),
                        }));
                    }
                    Err(Refusal::Quota(e)) => return Err(e),
                    Err(Refusal::QueueFull { group, max_queued }) => {
                        return Err(KrishivResourceGroupQueueFull {
                            group,
                            max_queued: max_queued.to_string(),
                        }
                        .into());
                    }
                }
            };
            let slot = match admitted {
                Admitted::Now(slot) => slot,
                Admitted::Queued(rx, guard) => {
                    let key = std::ptr::from_ref(request) as usize;
                    self.waiting
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .insert(key, group.clone());
                    let _entry = WaitingEntry {
                        map: Arc::clone(&self.waiting),
                        key,
                    };
                    let granted = rx.await;
                    drop(guard);
                    granted.map_err(|_| {
                        KrishivError::from(InternalError {
                            message: "a queued query lost its resource group".to_owned(),
                        })
                    })??
                }
            };
            let permit = Permit::with_slot(slot)
                .with_group(&group)
                .with_priority(traits.priority)
                .with_class(traits.class);
            let budget = traits.budget;
            Ok(match budget {
                Some(bytes) => permit.with_memory_budget(bytes),
                None => permit,
            })
        })
    }

    fn uses_estimate(&self) -> bool {
        false
    }

    fn queued_group(&self, request: &QueryRequest) -> Option<String> {
        let key = std::ptr::from_ref(request) as usize;
        let waiting = self
            .waiting
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&key)
            .cloned();
        waiting.or_else(|| Some(select::select(&self.current().config, request).to_owned()))
    }
}
