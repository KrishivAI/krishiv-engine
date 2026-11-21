// SPDX-License-Identifier: Apache-2.0
//! Session defaults an administrator sets while the runtime runs (P3-T066 `SetConfigDefaults`,
//! design `P3-T066-admin-api.md` §5.5).
//!
//! The defaults are an overlay on the runtime's startup pairs: every session built afterwards
//! starts from them, a session that exists keeps its values, `unset` returns a key to its startup
//! value, and nothing is persisted. Only a registered, session-scope key whose `settable_by` is
//! `Admin` or `User` can be set; the keys that bound what a role can do (`krishiv.auth.*`, the path
//! allowlists) are startup-only, so even an administrator cannot widen them here.

use std::collections::BTreeMap;
use std::sync::{Arc, PoisonError};

use krishiv_common::config::{Config, REGISTRY, Scope, SettableBy, lookup};
use krishiv_common::error::{
    CannotModifyStaticConfig, InvalidConfValueRequirement, KrishivError, KrishivUnknownConfig,
};

use crate::conf::{ConfStore, RuntimeConf, is_modifiable};
use crate::runtime::EngineRuntime;
use krishiv_common::redact::is_redacted_setting;
use krishiv_common::secret::is_sensitive_key;

impl EngineRuntime {
    /// The configuration a new session starts from, and the one its `RESET` returns to: the
    /// startup pairs with the defaults set at run time on top.
    pub(crate) fn current_config(&self) -> Config {
        let admin = Arc::clone(
            &self
                .admin_defaults
                .read()
                .unwrap_or_else(PoisonError::into_inner),
        );
        if admin.is_empty() {
            return Config::clone(&self.config);
        }
        let mut config = Config::clone(&self.config);
        for (key, value) in admin.iter() {
            // Validated when it was set; a failure here keeps the startup value.
            let _ = config.set(key, value);
        }
        config
    }

    /// The session defaults set at run time, as `(canonical key, value)` in key order
    /// (`GetConfig` shows them with source `admin`).
    #[must_use]
    pub fn config_defaults(&self) -> Vec<(String, String)> {
        self.admin_defaults
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Sets and unsets session defaults in one step: every key is checked before any changes,
    /// so a batch with one bad key changes nothing. Sessions built afterwards start from the new
    /// defaults; existing sessions keep their values. The caller has checked `Manage` on the
    /// cluster; nothing here asks the access policy.
    ///
    /// # Errors
    ///
    /// `KRISHIV_UNKNOWN_CONFIG` for a key that is not registered, `CANNOT_MODIFY_STATIC_CONFIG`
    /// for a cluster-scope, startup-only or build-time key, the key's own error for an invalid
    /// value, and `INVALID_CONF_VALUE.REQUIREMENT` when a key is both set and unset.
    pub fn set_config_defaults(
        &self,
        set: &[(String, String)],
        unset: &[String],
    ) -> Result<(), KrishivError> {
        // One write lock for the whole read-modify-write: two concurrent calls never start from the
        // same copy, so neither loses the other's change.
        let mut guard = self
            .admin_defaults
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let mut next: BTreeMap<String, String> = guard.as_ref().clone();
        let mut set_keys = Vec::with_capacity(set.len());
        for (key, value) in set {
            let entry = admin_settable(key)?;
            // The session's own `SET` checks (time zone, SET PATH grammar, unsupported switches).
            RuntimeConf::new(Arc::new(ConfStore::new(Config::new()))).set(key, value)?;
            next.insert(entry.to_owned(), value.clone());
            set_keys.push(entry);
        }
        for key in unset {
            let entry = admin_settable(key)?;
            if set_keys.contains(&entry) {
                return Err(InvalidConfValueRequirement {
                    conf_name: key.clone(),
                    conf_value: String::new(),
                    conf_requirement: "A key cannot be both set and unset in one request."
                        .to_owned(),
                }
                .into());
            }
            next.remove(entry);
        }
        *guard = Arc::new(next);
        Ok(())
    }
}

/// The canonical key of `key` when an administrator may set it as a default.
fn admin_settable(key: &str) -> Result<&'static str, KrishivError> {
    // The authentication settings are read at startup (`AuthConfig::from_settings`) and are not
    // registry entries: refused as static, never as unknown, so the answer does not depend on
    // whether the name is spelled right.
    if key.starts_with("krishiv.auth.") {
        return Err(CannotModifyStaticConfig {
            key: format!("\"{key}\""),
        }
        .into());
    }
    let Some(entry) = lookup(key) else {
        return Err(KrishivUnknownConfig {
            key: format!("\"{key}\""),
        }
        .into());
    };
    if entry.settable_by == SettableBy::Startup || !is_modifiable(key) {
        return Err(CannotModifyStaticConfig {
            key: format!("\"{key}\""),
        }
        .into());
    }
    Ok(entry.key)
}

/// Where a configuration value of [`EngineRuntime::config_listing`] comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigSource {
    /// The registry default: nothing set it.
    Default,
    /// The startup configuration (the builder, the startup file or the environment).
    Startup,
    /// A default set while running, [`EngineRuntime::set_config_defaults`].
    Admin,
}

/// One row of [`EngineRuntime::config_listing`] (design `P3-T066-admin-api.md` §5.5 `GetConfig`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigListing {
    /// The canonical key.
    pub key: String,
    /// The value as text; `None` when it is withheld or the key has no value.
    pub value: Option<String>,
    /// `"session"` or `"cluster"`; `None` for a startup key that is not in the registry.
    pub scope: Option<&'static str>,
    /// `"startup"`, `"admin"` or `"user"`; `None` for a key that is not in the registry.
    pub settable_by: Option<&'static str>,
    /// Where the value comes from.
    pub source: ConfigSource,
    /// The value is hidden because the caller may not see it.
    pub withheld: bool,
}

/// Keys whose value says how to become an administrator or what a role may read (design §4):
/// the authentication settings and the path allowlists.
fn is_sensitive(key: &str) -> bool {
    key.starts_with("krishiv.auth.")
        || key == "krishiv.io.local.allowedRoots"
        || key == "krishiv.io.objectStore.allowedPrefixes"
}

impl EngineRuntime {
    /// The configuration the runtime serves, one row per registered key plus the startup keys the
    /// registry does not know, in key order; `prefix` keeps the keys that start with it. A
    /// sensitive key (the path allowlists) shows its value only when `reveal_sensitive` is set (the
    /// caller holds `Manage` on the cluster), else `withheld`. The `krishiv.auth.*` settings are
    /// listener settings that never reach the runtime's configuration, so they are not listed
    /// here: the server layer lists them (withheld the same way) beside these rows.
    #[must_use]
    pub fn config_listing(
        &self,
        prefix: Option<&str>,
        reveal_sensitive: bool,
    ) -> Vec<ConfigListing> {
        let admin = Arc::clone(
            &self
                .admin_defaults
                .read()
                .unwrap_or_else(PoisonError::into_inner),
        );
        let explicit: BTreeMap<&str, String> =
            self.config.explicit_registered().into_iter().collect();
        let keep = |key: &str| prefix.is_none_or(|p| key.starts_with(p));
        let mut rows: BTreeMap<String, ConfigListing> = BTreeMap::new();
        let mut push = |key: &str,
                        value: Option<String>,
                        scope: Option<&'static str>,
                        settable_by: Option<&'static str>,
                        source| {
            // A key the registry does not know is not vetted: its value is shown only to `Manage`,
            // and never when its name says credential (`spark.hadoop.fs.s3a.secret.key`).
            let unvetted = scope.is_none();
            // A registered value may carry a credential too (`krishiv.io.egress.proxy=http://u:pw@h`).
            let withheld = value.is_some()
                && ((is_sensitive(key) && !reveal_sensitive)
                    || (!reveal_sensitive
                        && value
                            .as_deref()
                            .is_some_and(|v| is_redacted_setting(key, Some(v))))
                    || (unvetted && (!reveal_sensitive || is_sensitive_key(key))));
            rows.insert(
                key.to_owned(),
                ConfigListing {
                    key: key.to_owned(),
                    value: if withheld { None } else { value },
                    scope,
                    settable_by,
                    source,
                    withheld,
                },
            );
        };
        for entry in REGISTRY.iter().filter(|e| keep(e.key)) {
            let (value, source) = if let Some(v) = admin.get(entry.key) {
                (Some(v.clone()), ConfigSource::Admin)
            } else if let Some(v) = explicit.get(entry.key) {
                (Some(v.clone()), ConfigSource::Startup)
            } else {
                (
                    entry.default.as_ref().map(ToString::to_string),
                    ConfigSource::Default,
                )
            };
            let scope = match entry.scope {
                Scope::Session => "session",
                Scope::Cluster => "cluster",
            };
            push(
                entry.key,
                value,
                Some(scope),
                Some(entry.settable_by.as_str()),
                source,
            );
        }
        for (key, value) in self.config.explicit_unregistered() {
            if keep(key) {
                push(
                    key,
                    Some(value.to_owned()),
                    None,
                    None,
                    ConfigSource::Startup,
                );
            }
        }
        rows.into_values().collect()
    }
}
