// SPDX-License-Identifier: Apache-2.0
//! Where the store layer reads its settings from.
//!
//! The Spark-style `fs.*` keys (see the table in the module docs of [`StoreRegistry`](crate::StoreRegistry))
//! come through [`StoreSettings`], so a caller that is not the Krishiv engine needs no engine
//! configuration type: a map is enough. The engine's own [`Config`] implements it too, and every
//! source resolves each key the same way.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_common::secret::OptionValue;

/// A setting's value, typed as ADR 0038 requires: plain text, a secret or a URL with credentials.
pub type SettingValue = OptionValue;

/// A source of `fs.*` settings, by key.
///
/// A value is typed when it is read. A source that holds raw text (the engine's `Config`, a
/// `BTreeMap<String, String>`) types it by its key, with [`OptionValue::from_user`]: a value of
/// a sensitive key (an access key, a secret key, a session token, an account key, a SAS, a
/// service-account key, a bearer token) becomes a secret at once and is exposed only at the
/// object_store builder, and a value that is a URL with credentials becomes a URL. A source that
/// holds typed values (`BTreeMap<String, OptionValue>`) hands them out as they are, so a caller
/// never has to turn a secret into plain text to give it to this layer.
///
/// Keys match ASCII case-insensitively ([`Self::lookup`]), as Spark's `CaseInsensitiveStringMap`
/// does for per-read options; an exact match wins over a case-folded one. A value of the wrong
/// kind for its key (a secret where plain text is required, or a plain value for a credential
/// that must be secret) is `KRISHIV_INVALID_CONFIG_VALUE` naming the key, never the value.
///
/// ```
/// use std::collections::BTreeMap;
/// use krishiv_store::StoreSettings;
///
/// let mut settings = BTreeMap::new();
/// settings.insert("fs.s3a.endpoint".to_owned(), "http://127.0.0.1:9000".to_owned());
/// settings.insert("fs.s3a.secret.key".to_owned(), "s3cr3t".to_owned());
/// let endpoint = settings.lookup("FS.S3A.ENDPOINT").unwrap();
/// assert_eq!(endpoint.as_plain(), Some("http://127.0.0.1:9000"));
/// // A sensitive key is a secret as soon as it is read.
/// assert!(settings.lookup("fs.s3a.secret.key").unwrap().as_plain().is_none());
/// ```
pub trait StoreSettings: Send + Sync {
    /// The value of `key`, matched exactly.
    fn get(&self, key: &str) -> Option<SettingValue>;

    /// Every key this source holds, in a stable order. Used to find a key that differs only in
    /// ASCII case; it never copies a value.
    fn keys(&self) -> Vec<String>;

    /// The value of `key`: the exact match first, else the first key (in [`Self::keys`] order)
    /// equal to `key` ignoring ASCII case.
    fn lookup(&self, key: &str) -> Option<SettingValue> {
        if let Some(value) = self.get(key) {
            return Some(value);
        }
        let folded = self
            .keys()
            .into_iter()
            .find(|k| k.eq_ignore_ascii_case(key))?;
        self.get(&folded)
    }
}

impl StoreSettings for Config {
    fn get(&self, key: &str) -> Option<SettingValue> {
        self.get_unregistered(key)
            .map(|raw| OptionValue::from_user(key, raw))
    }

    fn keys(&self) -> Vec<String> {
        self.explicit_unregistered()
            .into_iter()
            .map(|(k, _)| k.to_owned())
            .collect()
    }
}

impl StoreSettings for BTreeMap<String, String> {
    fn get(&self, key: &str) -> Option<SettingValue> {
        BTreeMap::get(self, key).map(|raw| OptionValue::from_user(key, raw.as_str()))
    }

    fn keys(&self) -> Vec<String> {
        BTreeMap::keys(self).cloned().collect()
    }
}

/// Typed values are handed out as they are.
impl StoreSettings for BTreeMap<String, OptionValue> {
    fn get(&self, key: &str) -> Option<SettingValue> {
        BTreeMap::get(self, key).cloned()
    }

    fn keys(&self) -> Vec<String> {
        BTreeMap::keys(self).cloned().collect()
    }
}

impl StoreSettings for HashMap<String, String> {
    fn get(&self, key: &str) -> Option<SettingValue> {
        HashMap::get(self, key).map(|raw| OptionValue::from_user(key, raw.as_str()))
    }

    /// Sorted, so a case-folded match is deterministic.
    fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = HashMap::keys(self).cloned().collect();
        keys.sort();
        keys
    }
}

/// A shared source is a source.
impl<T: StoreSettings + ?Sized> StoreSettings for Arc<T> {
    fn get(&self, key: &str) -> Option<SettingValue> {
        (**self).get(key)
    }

    fn keys(&self) -> Vec<String> {
        (**self).keys()
    }

    fn lookup(&self, key: &str) -> Option<SettingValue> {
        (**self).lookup(key)
    }
}
