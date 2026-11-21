// SPDX-License-Identifier: Apache-2.0
//! What the admin API reads and changes of the resource groups (design P3-T066 §5.4): the group
//! rows a caller may see, the loaded document, and validate / replace / reload of the group file.

use krishiv_common::config::{ConfigValue, WORKLOAD_RESOURCE_GROUPS_FILE};
use krishiv_common::error::{KrishivError, KrishivResourceGroupConfigReadOnly};
use krishiv_memory::MemoryConfig;
use krishiv_plan::analyzer::Principal;

use crate::EngineRuntime;
use crate::resource_groups::ResourceGroups;

/// One resource group as `ListResourceGroups` shows it (the columns of `system.resource_groups`).
/// The subtree counters are `None` on an ancestor row of a caller without `Monitor`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceGroupInfo {
    /// Dotted path, `root.etl.nightly`.
    pub group_id: String,
    /// The parent's id; `None` for a root.
    pub parent_id: Option<String>,
    /// `active`, or `removed` (left by a reload, still holding queries).
    pub state: String,
    /// The scheduling policy, inherited values resolved.
    pub scheduling_policy: String,
    /// The scheduling weight.
    pub scheduling_weight: i32,
    /// The priority.
    pub priority: i32,
    /// The workload class.
    pub workload_class: String,
    /// Queries running at once.
    pub hard_concurrency_limit: i32,
    /// Queries that may wait.
    pub max_queued: i32,
    /// Soft memory limit in bytes.
    pub soft_memory_limit_bytes: Option<i64>,
    /// Per-query memory limit in bytes.
    pub per_query_memory_limit_bytes: Option<i64>,
    /// Scan quota in bytes per period.
    pub scan_quota_bytes: Option<i64>,
    /// The scan quota period in seconds.
    pub scan_quota_period_s: Option<i64>,
    /// Running queries in the subtree.
    pub running: Option<i32>,
    /// Queued queries in the subtree.
    pub queued: Option<i32>,
    /// Memory reserved by the subtree's running queries.
    pub memory_reserved_bytes: Option<i64>,
    /// Bytes scanned in the current quota window.
    pub scan_bytes_in_window: Option<i64>,
    /// When the quota window resets, in milliseconds since the epoch.
    pub scan_window_resets_at_ms: Option<i64>,
    /// The etag of the document the row came from.
    pub config_etag: String,
    /// When that document was loaded, in milliseconds since the epoch.
    pub loaded_at_ms: i64,
}

/// The loaded resource group document (`GetResourceGroupConfig`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceGroupDocument {
    /// The canonical JSON; empty when no group file is configured.
    pub document: String,
    /// SHA-256 of the canonical JSON; what `Replace` is checked against.
    pub etag: String,
    /// When the document was loaded, in milliseconds since the epoch.
    pub loaded_at_ms: i64,
    /// The etag of the file as it is on disk now; `None` when it cannot be read.
    pub file_etag: Option<String>,
}

impl EngineRuntime {
    fn group_file(&self) -> Option<&str> {
        self.config
            .get(&WORKLOAD_RESOURCE_GROUPS_FILE)
            .and_then(ConfigValue::as_str)
    }

    /// The group file is unset, so groups are off for this run (design §5.4).
    fn groups_off() -> KrishivError {
        KrishivResourceGroupConfigReadOnly {
            path: String::new(),
            reason: "no resource group file is configured (krishiv.workload.resourceGroups.file); \
                     groups are switched on at startup"
                .to_owned(),
        }
        .into()
    }

    fn group_memory(&self) -> usize {
        MemoryConfig::from_config(&self.config)
            .limit
            .unwrap_or(usize::MAX)
    }

    /// The groups `caller` may see. With `monitor` (Monitor on `system`) every group; without it
    /// the groups holding one of the caller's running queries and their ancestors, whose subtree
    /// counters are `None` (they are totals over other tenants' work).
    #[must_use]
    pub fn resource_group_infos(
        &self,
        caller: &Principal,
        monitor: bool,
    ) -> Vec<ResourceGroupInfo> {
        let Some(groups) = &self.resource_groups else {
            return Vec::new();
        };
        let own: Vec<String> = if monitor {
            Vec::new()
        } else {
            self.query_infos()
                .into_iter()
                .filter(|q| crate::system_groups::holds_a_slot_or_a_place(&q.state))
                .filter(|q| q.is_owned_by(caller))
                .filter_map(|q| q.resource_group)
                .collect()
        };
        let rows = groups.rows();
        let visible: Vec<(_, bool)> = if monitor {
            rows.into_iter().map(|r| (r, true)).collect()
        } else {
            crate::system_groups::visible_to_owner(rows, &own, |r| r.group_id.as_str())
        };
        visible
            .into_iter()
            .map(|(r, counters)| ResourceGroupInfo {
                group_id: r.group_id,
                parent_id: r.parent_id,
                state: r.state.to_owned(),
                scheduling_policy: r.scheduling_policy.to_owned(),
                scheduling_weight: r.scheduling_weight,
                priority: r.priority,
                workload_class: r.workload_class.to_owned(),
                hard_concurrency_limit: r.hard_concurrency_limit,
                max_queued: r.max_queued,
                soft_memory_limit_bytes: r.soft_memory_limit_bytes,
                per_query_memory_limit_bytes: r.per_query_memory_limit_bytes,
                scan_quota_bytes: r.scan_quota_bytes,
                scan_quota_period_s: r.scan_quota_period_s,
                running: counters.then_some(r.running),
                queued: counters.then_some(r.queued),
                memory_reserved_bytes: counters.then_some(r.memory_reserved_bytes),
                scan_bytes_in_window: if counters {
                    r.scan_bytes_in_window
                } else {
                    None
                },
                scan_window_resets_at_ms: if counters {
                    r.scan_window_resets_at_ms
                } else {
                    None
                },
                config_etag: r.config_etag,
                loaded_at_ms: r.loaded_at_ms,
            })
            .collect()
    }

    /// The loaded group document; an empty one when no group file is configured.
    #[must_use]
    pub fn resource_group_document(&self) -> ResourceGroupDocument {
        match (&self.resource_groups, self.group_file()) {
            (Some(groups), Some(path)) => {
                let (document, etag, loaded_at_ms) = groups.snapshot();
                ResourceGroupDocument {
                    document,
                    etag,
                    loaded_at_ms: i64::try_from(loaded_at_ms).unwrap_or(i64::MAX),
                    file_etag: ResourceGroups::file_etag(path),
                }
            }
            _ => ResourceGroupDocument {
                document: String::new(),
                etag: String::new(),
                loaded_at_ms: 0,
                file_etag: None,
            },
        }
    }

    /// Checks a candidate document with the group loader, changing nothing.
    ///
    /// # Errors
    ///
    /// `KRISHIV_RESOURCE_GROUP_CONFIG_INVALID` for an invalid document.
    pub fn validate_resource_group_document(&self, document: &str) -> Result<(), KrishivError> {
        ResourceGroups::validate(document, self.group_memory())
    }

    /// Every problem of `document`, one per bad group or selector (`ValidateResourceGroupConfig`).
    #[must_use]
    pub fn resource_group_problems(&self, document: &str) -> Vec<KrishivError> {
        ResourceGroups::problems(document, self.group_memory())
    }

    /// Writes `document` over the group file and swaps the tree (`ReplaceResourceGroupConfig`).
    ///
    /// # Errors
    ///
    /// As [`ResourceGroups::replace_file`]; `KRISHIV_RESOURCE_GROUP_CONFIG_READ_ONLY` when no group
    /// file is configured.
    pub fn replace_resource_group_document(
        &self,
        document: &str,
        etag: &str,
    ) -> Result<ResourceGroupDocument, KrishivError> {
        match (&self.resource_groups, self.group_file()) {
            (Some(groups), Some(path)) => groups
                .replace_file(path, document, etag)
                .map(|installed| Self::installed(path, installed)),
            _ => Err(Self::groups_off()),
        }
    }

    /// Re-reads the group file and swaps the tree (`ReloadResourceGroupConfig`); an invalid file
    /// is refused and the old tree stays.
    ///
    /// # Errors
    ///
    /// As [`ResourceGroups::reload_file`]; `KRISHIV_RESOURCE_GROUP_CONFIG_READ_ONLY` when no group
    /// file is configured.
    pub fn reload_resource_group_document(&self) -> Result<ResourceGroupDocument, KrishivError> {
        match (&self.resource_groups, self.group_file()) {
            (Some(groups), Some(path)) => groups
                .reload_file(path)
                .map(|installed| Self::installed(path, installed)),
            _ => Err(Self::groups_off()),
        }
    }

    /// The document a replace or reload installed, as that call saw it (a later swap by another
    /// request changes none of it).
    fn installed(
        path: &str,
        (document, etag, loaded_at_ms): (String, String, u64),
    ) -> ResourceGroupDocument {
        ResourceGroupDocument {
            document,
            etag,
            loaded_at_ms: i64::try_from(loaded_at_ms).unwrap_or(i64::MAX),
            file_etag: ResourceGroups::file_etag(path),
        }
    }
}
