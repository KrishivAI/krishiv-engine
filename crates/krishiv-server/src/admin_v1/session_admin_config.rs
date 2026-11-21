// SPDX-License-Identifier: Apache-2.0
//! The resource group and configuration RPCs of [`SessionAdmin`] (design P3-T066 §5.4, §5.5):
//! listing the groups, reading, validating, replacing and reloading the group document, and
//! reading and changing the runtime's configuration defaults.

use krishiv_common::error::KrishivError;
use krishiv_common::redact::is_redacted_setting;
use krishiv_common::secret::is_sensitive_key;
use krishiv_session::{ConfigListing, ConfigSource, Privilege, ResourceGroupInfo};

use super::page::Page;
use super::session_admin::{SessionAdmin, Who, timestamp};
use super::{
    AdminError, AdminResult, Call, ConfigEntry, ConfigProblem, ErrorKind, GetConfigRequest,
    GetConfigResponse, GetResourceGroupConfigResponse, ListResourceGroupsRequest,
    ListResourceGroupsResponse, ReloadResourceGroupConfigResponse,
    ReplaceResourceGroupConfigRequest, ReplaceResourceGroupConfigResponse, ResourceGroup,
    SetConfigDefaultsRequest, SetConfigDefaultsResponse, ValidateResourceGroupConfigRequest,
    ValidateResourceGroupConfigResponse,
};

/// An engine error as the admin API answers it: the class decides the kind of answer.
fn admin_error(e: &KrishivError) -> AdminError {
    let class = e.error_class();
    let name = class.name();
    let kind = match name {
        "KRISHIV_ADMIN_PRECONDITION_FAILED" | "KRISHIV_RESOURCE_GROUP_CONFIG_READ_ONLY" => {
            ErrorKind::FailedPrecondition
        }
        "KRISHIV_RESOURCE_GROUP_CONFIG_INVALID"
        | "CANNOT_MODIFY_STATIC_CONFIG"
        | "KRISHIV_INVALID_CONFIG_VALUE"
        | "KRISHIV_UNKNOWN_CONFIG" => ErrorKind::InvalidArgument,
        n if n.starts_with("INVALID_CONF_VALUE") || n.starts_with("INVALID_") => {
            ErrorKind::InvalidArgument
        }
        _ => ErrorKind::Internal,
    };
    let mut out = AdminError::new(
        kind,
        name,
        class.sql_state().unwrap_or("XX000"),
        e.message(),
    );
    for (k, v) in e.message_parameters() {
        out = out.with_parameter(k, v.clone());
    }
    out
}

fn group_message(g: &ResourceGroupInfo) -> ResourceGroup {
    ResourceGroup {
        group_id: g.group_id.clone(),
        parent_id: g.parent_id.clone().unwrap_or_default(),
        state: g.state.clone(),
        scheduling_policy: g.scheduling_policy.clone(),
        scheduling_weight: g.scheduling_weight,
        priority: g.priority,
        workload_class: g.workload_class.clone(),
        hard_concurrency_limit: g.hard_concurrency_limit,
        max_queued: g.max_queued,
        soft_memory_limit_bytes: g.soft_memory_limit_bytes,
        per_query_memory_limit_bytes: g.per_query_memory_limit_bytes,
        scan_quota_bytes: g.scan_quota_bytes,
        scan_quota_period_s: g.scan_quota_period_s,
        running: g.running,
        queued: g.queued,
        memory_reserved_bytes: g.memory_reserved_bytes,
        scan_bytes_in_window: g.scan_bytes_in_window,
        scan_window_resets_at: g.scan_window_resets_at_ms.map(timestamp),
        config_etag: g.config_etag.clone(),
        loaded_at: (g.loaded_at_ms > 0).then(|| timestamp(g.loaded_at_ms)),
    }
}

fn entry_message(c: &ConfigListing) -> ConfigEntry {
    ConfigEntry {
        key: c.key.clone(),
        value: c.value.clone(),
        scope: c.scope.unwrap_or_default().to_owned(),
        settable_by: c.settable_by.unwrap_or_default().to_owned(),
        source: match c.source {
            ConfigSource::Default => "default",
            ConfigSource::Startup => "startup",
            ConfigSource::Admin => "admin",
        }
        .to_owned(),
        withheld: c.withheld,
    }
}

/// A value as the action record shows it: never one a name marks as a credential.
fn shown(key: &str, value: Option<&str>) -> String {
    match value {
        None => "(unset)".to_owned(),
        Some(v) if is_sensitive_key(key) || is_redacted_setting(key, Some(v)) => {
            "(withheld)".to_owned()
        }
        Some(v) => {
            // A request may carry a value of any length: the record keeps the start of it.
            const MAX: usize = 256;
            if v.len() <= MAX {
                v.to_owned()
            } else {
                let cut = (0..=MAX)
                    .rev()
                    .find(|i| v.is_char_boundary(*i))
                    .unwrap_or(0);
                format!("{}... ({} bytes)", &v[..cut], v.len())
            }
        }
    }
}

impl SessionAdmin {
    pub(super) fn do_list_resource_groups(
        &self,
        call: &Call,
        request: &ListResourceGroupsRequest,
    ) -> AdminResult<ListResourceGroupsResponse> {
        let who = Who::of(call);
        let page = Page::parse(request.page_size, &request.page_token, "resourceGroups")?;
        let groups = self
            .runtime()
            .resource_group_infos(&who.principal, who.monitor);
        let (rows, next) = page.apply(groups, |g| (0, g.group_id.clone()), "resourceGroups");
        Ok(ListResourceGroupsResponse {
            groups: rows.iter().map(group_message).collect(),
            next_page_token: next,
        })
    }

    pub(super) fn do_get_resource_group_config(
        &self,
        call: &Call,
    ) -> AdminResult<GetResourceGroupConfigResponse> {
        let who = Who::of(call);
        self.require(
            call,
            who.monitor,
            Privilege::Monitor,
            "GetResourceGroupConfig",
            "system",
        )?;
        let d = self.runtime().resource_group_document();
        Ok(GetResourceGroupConfigResponse {
            document: d.document,
            etag: d.etag,
            loaded_at: (d.loaded_at_ms > 0).then(|| timestamp(d.loaded_at_ms)),
            file_etag: d.file_etag.unwrap_or_default(),
        })
    }

    pub(super) fn do_validate_resource_group_config(
        &self,
        call: &Call,
        request: &ValidateResourceGroupConfigRequest,
    ) -> AdminResult<ValidateResourceGroupConfigResponse> {
        let who = Who::of(call);
        self.require(
            call,
            who.monitor,
            Privilege::Monitor,
            "ValidateResourceGroupConfig",
            "system",
        )?;
        let problems: Vec<ConfigProblem> = self
            .runtime()
            .resource_group_problems(&request.document)
            .iter()
            .map(|e| ConfigProblem {
                error_class: e.error_class().name().to_owned(),
                message: e.message().to_owned(),
                path: e
                    .message_parameters()
                    .iter()
                    .find(|(k, _)| *k == "path")
                    .map(|(_, v)| v.clone())
                    .unwrap_or_default(),
            })
            .collect();
        Ok(ValidateResourceGroupConfigResponse {
            valid: problems.is_empty(),
            problems,
        })
    }

    pub(super) fn do_replace_resource_group_config(
        &self,
        call: &Call,
        request: &ReplaceResourceGroupConfigRequest,
    ) -> AdminResult<ReplaceResourceGroupConfigResponse> {
        let who = Who::of(call);
        self.require(
            call,
            who.manage,
            Privilege::Manage,
            "ReplaceResourceGroupConfig",
            "cluster",
        )?;
        let old = self.runtime().resource_group_document().etag;
        if let Err(e) = self
            .runtime()
            .replace_resource_group_document(&request.document, &request.etag)
        {
            let e = admin_error(&e);
            let target = format!("etag {} (current {old})", request.etag);
            self.log(
                call,
                "ReplaceResourceGroupConfig",
                &target,
                false,
                "failed",
                &e.error_class,
            );
            return Err(e);
        }
        let d = self.runtime().resource_group_document();
        let target = format!("etag {old} -> {}", d.etag);
        self.log(call, "ReplaceResourceGroupConfig", &target, false, "ok", "");
        Ok(ReplaceResourceGroupConfigResponse {
            etag: d.etag,
            loaded_at: Some(timestamp(d.loaded_at_ms)),
        })
    }

    pub(super) fn do_reload_resource_group_config(
        &self,
        call: &Call,
    ) -> AdminResult<ReloadResourceGroupConfigResponse> {
        let who = Who::of(call);
        self.require(
            call,
            who.manage,
            Privilege::Manage,
            "ReloadResourceGroupConfig",
            "cluster",
        )?;
        let old = self.runtime().resource_group_document().etag;
        if let Err(e) = self.runtime().reload_resource_group_document() {
            let e = admin_error(&e);
            let target = format!("etag {old}");
            self.log(
                call,
                "ReloadResourceGroupConfig",
                &target,
                false,
                "failed",
                &e.error_class,
            );
            return Err(e);
        }
        let d = self.runtime().resource_group_document();
        let target = format!("etag {old} -> {}", d.etag);
        self.log(call, "ReloadResourceGroupConfig", &target, false, "ok", "");
        Ok(ReloadResourceGroupConfigResponse {
            etag: d.etag,
            loaded_at: Some(timestamp(d.loaded_at_ms)),
        })
    }

    pub(super) fn do_get_config(
        &self,
        call: &Call,
        request: &GetConfigRequest,
    ) -> AdminResult<GetConfigResponse> {
        let who = Who::of(call);
        self.require(call, who.monitor, Privilege::Monitor, "GetConfig", "system")?;
        let filter = format!("config{:?}", request.prefix);
        let page = Page::parse(request.page_size, &request.page_token, &filter)?;
        let prefix = (!request.prefix.is_empty()).then_some(request.prefix.as_str());
        let mut rows = self.runtime().config_listing(prefix, who.manage);
        // The listener's own settings never reach the runtime: list them as written (references,
        // never resolved secrets), their values for `Manage` only.
        rows.extend(
            self.startup_auth
                .iter()
                .filter(|(k, _)| prefix.is_none_or(|p| k.starts_with(p)))
                .map(|(k, v)| {
                    // A reference (`env:`/`file:`) is shown as written; an inline value of a
                    // credential-named key (the Spark token may be one) is shown to no one.
                    let is_reference = v.starts_with("env:") || v.starts_with("file:");
                    let show = who.manage && (is_reference || !is_sensitive_key(k));
                    ConfigListing {
                        key: k.clone(),
                        value: show.then(|| v.clone()),
                        scope: Some("cluster"),
                        settable_by: Some("startup"),
                        source: ConfigSource::Startup,
                        withheld: !show,
                    }
                }),
        );
        let (rows, next) = page.apply(rows, |c| (0, c.key.clone()), &filter);
        Ok(GetConfigResponse {
            entries: rows.iter().map(entry_message).collect(),
            next_page_token: next,
        })
    }

    pub(super) fn do_set_config_defaults(
        &self,
        call: &Call,
        request: &SetConfigDefaultsRequest,
    ) -> AdminResult<SetConfigDefaultsResponse> {
        let who = Who::of(call);
        self.require(
            call,
            who.manage,
            Privilege::Manage,
            "SetConfigDefaults",
            "cluster",
        )?;
        let set: Vec<(String, String)> = request
            .set
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let target = self.describe_defaults_change(&set, &request.unset);
        if let Err(e) = self.runtime().set_config_defaults(&set, &request.unset) {
            let e = admin_error(&e);
            self.log(
                call,
                "SetConfigDefaults",
                &target,
                false,
                "failed",
                &e.error_class,
            );
            return Err(e);
        }
        self.log(call, "SetConfigDefaults", &target, false, "ok", "");
        Ok(SetConfigDefaultsResponse {
            set_count: i32::try_from(set.len()).unwrap_or(i32::MAX),
            unset_count: i32::try_from(request.unset.len()).unwrap_or(i32::MAX),
        })
    }

    /// The keys of a defaults change with their old and new values (design §5.6), values of a
    /// credential-named key withheld.
    fn describe_defaults_change(&self, set: &[(String, String)], unset: &[String]) -> String {
        // The record names the canonical key: a Spark alias resolves to its registry entry.
        let canonical = |key: &str| {
            krishiv_common::config::lookup(key).map_or_else(|| key.to_owned(), |e| e.key.to_owned())
        };
        let current = |key: &str| {
            self.runtime()
                .config_listing(Some(key), true)
                .into_iter()
                .find(|r| r.key == key)
                .and_then(|r| r.value)
        };
        let mut parts: Vec<String> = set
            .iter()
            .map(|(k, v)| {
                let k = canonical(k);
                format!(
                    "set {k}: {} -> {}",
                    shown(&k, current(&k).as_deref()),
                    shown(&k, Some(v))
                )
            })
            .collect();
        parts.extend(unset.iter().map(|k| {
            let k = canonical(k);
            format!("unset {k}: {}", shown(&k, current(&k).as_deref()))
        }));
        parts.join("; ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_credential_inside_a_value_is_withheld() {
        assert_eq!(
            shown(
                "spark.datasource.jdbc.url",
                Some("postgres://user:password@host/db")
            ),
            "(withheld)"
        );
        assert_eq!(shown("spark.sql.ansi.enabled", Some("true")), "true");
    }

    #[test]
    fn an_invalid_config_value_is_an_invalid_argument() {
        let entry = krishiv_common::config::lookup("krishiv.sql.ansi.enabled").expect("registered");
        let err = entry
            .parse("krishiv.sql.ansi.enabled", "maybe")
            .expect_err("not a boolean");
        assert!(matches!(admin_error(&err).kind, ErrorKind::InvalidArgument));
    }
}
