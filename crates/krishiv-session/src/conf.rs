// SPDX-License-Identifier: Apache-2.0
//! Session configuration: Spark's `spark.conf`.

use std::sync::{Arc, PoisonError, RwLock};

use krishiv_common::config::{Config, Scope, lookup};
use krishiv_common::error::{CannotModifyStaticConfig, KrishivError};

/// Prefixes of entries read once, when the session is built (memory pool,
/// spill directory, I/O scheduler): setting them later would not apply,
/// so it fails like a static config.
const BUILD_TIME: &[&str] = &["krishiv.memory.", "krishiv.spill.", "krishiv.io."];

/// The configuration of a session, shared by its clones. Queries read an
/// immutable snapshot ([`Self::snapshot`]).
#[derive(Debug, Default)]
pub(crate) struct ConfStore {
    current: RwLock<Arc<Config>>,
    /// Bumped on every change: analysis results are reused only within one
    /// version.
    version: std::sync::atomic::AtomicU64,
}

impl ConfStore {
    pub(crate) fn new(config: Config) -> Self {
        Self {
            current: RwLock::new(Arc::new(config)),
            version: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub(crate) fn snapshot(&self) -> Arc<Config> {
        Arc::clone(&self.current.read().unwrap_or_else(PoisonError::into_inner))
    }

    pub(crate) fn version(&self) -> u64 {
        self.version.load(std::sync::atomic::Ordering::Acquire)
    }

    fn update(
        &self,
        f: impl FnOnce(&mut Config) -> Result<(), KrishivError>,
    ) -> Result<(), KrishivError> {
        let mut current = self.current.write().unwrap_or_else(PoisonError::into_inner);
        let mut next = Config::clone(&current);
        f(&mut next)?;
        *current = Arc::new(next);
        self.version
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Ok(())
    }
}

/// The value of `key` in `config` as text: a registered entry (by its
/// `krishiv.*` key or Spark alias) with its default, else an unregistered
/// key as set.
pub(crate) fn get(config: &Config, key: &str) -> Option<String> {
    match lookup(key) {
        Some(entry) => config.get(entry).map(ToString::to_string),
        None => config.get_unregistered(key).map(str::to_owned),
    }
}

/// Whether `key` can change after the session is built.
pub(crate) fn is_modifiable(key: &str) -> bool {
    match lookup(key) {
        Some(entry) => {
            entry.scope == Scope::Session && !BUILD_TIME.iter().any(|p| entry.key.starts_with(p))
        }
        None => true,
    }
}

/// The runtime configuration of a session (Spark's `spark.conf`). Changes
/// apply to queries that start afterwards.
///
/// ```
/// let session = krishiv_session::Session::new()?;
/// session.conf().set("spark.sql.caseSensitive", "true")?;
/// assert_eq!(session.conf().get("spark.sql.caseSensitive").as_deref(), Some("true"));
/// # Ok::<(), krishiv_common::error::KrishivError>(())
/// ```
#[derive(Debug, Clone)]
pub struct RuntimeConf {
    store: Arc<ConfStore>,
}

impl RuntimeConf {
    pub(crate) fn new(store: Arc<ConfStore>) -> Self {
        Self { store }
    }

    /// The value of `key`: a set value or a registered entry's default;
    /// `None` for an unset key without a default.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// assert_eq!(session.conf().get("spark.sql.ansi.enabled").as_deref(), Some("true"));
    /// assert_eq!(session.conf().get("my.app.flag"), None);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn get(&self, key: &str) -> Option<String> {
        get(&self.store.snapshot(), key)
    }

    /// Sets `key` to `value` for queries that start afterwards.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// session.conf().set("krishiv.exec.batchSize", "1024")?;
    /// let e = session.conf().set("krishiv.exec.batchSize", "many").unwrap_err();
    /// assert_eq!(e.error_class().name(), "KRISHIV_INVALID_CONFIG_VALUE");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// `CANNOT_MODIFY_STATIC_CONFIG` for an entry fixed when the session
    /// was built ([`Self::is_modifiable`]); `KRISHIV_UNKNOWN_CONFIG` for an
    /// unknown `krishiv.*` key; `KRISHIV_INVALID_CONFIG_VALUE` for a value
    /// of the wrong type. Unknown keys of other prefixes are kept.
    pub fn set(&self, key: &str, value: &str) -> Result<(), KrishivError> {
        if !is_modifiable(key) {
            return Err(CannotModifyStaticConfig {
                key: format!("\"{key}\""),
            }
            .into());
        }
        self.store.update(|c| c.set(key, value).map(drop))
    }

    /// Resets `key` to its default.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// session.conf().set("krishiv.exec.batchSize", "1024")?;
    /// session.conf().unset("krishiv.exec.batchSize")?;
    /// assert_eq!(session.conf().get("krishiv.exec.batchSize").as_deref(), Some("4096"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// `CANNOT_MODIFY_STATIC_CONFIG` as for [`Self::set`].
    pub fn unset(&self, key: &str) -> Result<(), KrishivError> {
        if !is_modifiable(key) {
            return Err(CannotModifyStaticConfig {
                key: format!("\"{key}\""),
            }
            .into());
        }
        self.store.update(|c| {
            c.unset(key);
            Ok(())
        })
    }

    /// Whether `key` can be set at runtime.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// assert!(session.conf().is_modifiable("spark.sql.ansi.enabled"));
    /// assert!(!session.conf().is_modifiable("krishiv.memory.limit"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn is_modifiable(&self, key: &str) -> bool {
        is_modifiable(key)
    }
}
