// SPDX-License-Identifier: Apache-2.0
//! Where the store layer reads its settings from.
//!
//! The Spark-style `fs.*` keys (see the table in [`crate::StoreRegistry`]'s module docs) come
//! through [`StoreSettings`], so a caller that is not the Krishiv engine needs no engine
//! configuration type: a `BTreeMap<String, String>` is enough. The engine's own
//! [`Config`] implements it too, and both resolve every key the same way.

use std::collections::{BTreeMap, HashMap};

use krishiv_common::config::Config;

/// A source of `fs.*` settings: raw text values by key, as Hadoop configuration holds them.
///
/// A value is classified when it is read, by its key: a value of a sensitive key (an access key,
/// a secret key, a session token, an account key, a SAS, a service-account key, a bearer token)
/// becomes a [`SecretString`](krishiv_common::secret::SecretString) at once and is exposed only
/// at the object_store builder; every other value stays plain text. A source never marks a value
/// secret itself.
///
/// Keys match ASCII case-insensitively ([`Self::lookup`]), as Spark's `CaseInsensitiveStringMap`
/// does for per-read options; an exact match wins over a case-folded one.
///
/// ```
/// use std::collections::BTreeMap;
/// use krishiv_store::StoreSettings;
///
/// let mut settings = BTreeMap::new();
/// settings.insert("fs.s3a.endpoint".to_owned(), "http://127.0.0.1:9000".to_owned());
/// assert_eq!(settings.lookup("FS.S3A.ENDPOINT").as_deref(), Some("http://127.0.0.1:9000"));
/// ```
pub trait StoreSettings: Send + Sync {
    /// The raw value of `key`, matched exactly.
    fn get(&self, key: &str) -> Option<String>;

    /// Every setting this source holds, as `(key, value)` pairs in a stable order. Used to
    /// find a key that differs only in ASCII case.
    fn entries(&self) -> Vec<(String, String)>;

    /// The value of `key`: the exact match first, else the first entry (in [`Self::entries`]
    /// order) whose key equals `key` ignoring ASCII case.
    fn lookup(&self, key: &str) -> Option<String> {
        if let Some(value) = self.get(key) {
            return Some(value);
        }
        self.entries()
            .into_iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v)
    }
}

impl StoreSettings for Config {
    fn get(&self, key: &str) -> Option<String> {
        self.get_unregistered(key).map(str::to_owned)
    }

    fn entries(&self) -> Vec<(String, String)> {
        self.explicit_unregistered()
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect()
    }
}

impl StoreSettings for BTreeMap<String, String> {
    fn get(&self, key: &str) -> Option<String> {
        BTreeMap::get(self, key).cloned()
    }

    fn entries(&self) -> Vec<(String, String)> {
        self.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}

/// The entries are sorted by key, so a case-folded match is deterministic.
impl StoreSettings for HashMap<String, String> {
    fn get(&self, key: &str) -> Option<String> {
        HashMap::get(self, key).cloned()
    }

    fn entries(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> =
            self.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        out.sort();
        out
    }
}

/// A shared source is a source.
impl<T: StoreSettings + ?Sized> StoreSettings for std::sync::Arc<T> {
    fn get(&self, key: &str) -> Option<String> {
        (**self).get(key)
    }

    fn entries(&self) -> Vec<(String, String)> {
        (**self).entries()
    }

    fn lookup(&self, key: &str) -> Option<String> {
        (**self).lookup(key)
    }
}
