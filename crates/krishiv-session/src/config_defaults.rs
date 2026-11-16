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

use krishiv_common::config::{Config, SettableBy, lookup};
use krishiv_common::error::{
    CannotModifyStaticConfig, InvalidConfValueRequirement, KrishivError, KrishivUnknownConfig,
};

use crate::conf::{ConfStore, RuntimeConf, is_modifiable};
use crate::runtime::EngineRuntime;

impl EngineRuntime {
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
        let mut next: BTreeMap<String, String> = self
            .admin_defaults
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .clone();
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
        *self
            .admin_defaults
            .write()
            .unwrap_or_else(PoisonError::into_inner) = Arc::new(next);
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
