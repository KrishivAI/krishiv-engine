// SPDX-License-Identifier: Apache-2.0
//! The session's UDF registrations: which implementations a name resolves
//! to on the executing side.
//!
//! A [`UdfLibrary`] lives with the session that registers into it and is
//! passed to the workers that run its plans (`WorkerContext::with_udfs` in
//! `krishiv-exec`). Plans reference a UDF by name and version only, so
//! the library is consulted at execution, never at planning: a worker
//! holding another version (or none) refuses the plan with
//! `KRISHIV_UDF_NOT_AVAILABLE` instead of running the wrong code.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, PoisonError, RwLock};

use super::{ScalarUdf, guard_call};

/// The UDFs registered on one session, by lower-case name.
///
/// Re-registering a name replaces its implementation. Every registration
/// bumps the library [`version`](Self::version): the analyzer's context
/// fingerprint covers it, so a cached plan never outlives the registration
/// it was analyzed with.
pub struct UdfLibrary {
    inner: RwLock<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    udfs: HashMap<String, Arc<dyn ScalarUdf>>,
    /// Bumped on every registration; part of the analysis fingerprint.
    epoch: u64,
}

impl UdfLibrary {
    /// An empty library.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Inner::default()),
        }
    }

    /// Registers `udf` under its lower-case [`name`](ScalarUdf::name),
    /// replacing any UDF of that name, and bumps the library version.
    ///
    /// The `name` call goes through [`guard_call`] like every other trait
    /// call: a panicking `name` is returned as a [`UdfError`](crate::UdfError) instead of
    /// unwinding (there is no name to register the UDF under). An empty
    /// name, or one containing a `.` or a backtick, is also an error,
    /// since such a UDF can never be called by name.
    ///
    /// # Errors
    /// A [`UdfError`](crate::UdfError) when `name` panics, is empty, or contains a `.` or
    /// a backtick.
    pub fn register(&self, udf: Arc<dyn ScalarUdf>) -> Result<(), super::UdfError> {
        let name = guard_call("unknown", "name", || udf.name().to_owned())?;
        if name.is_empty() {
            return Err(super::UdfError::new(
                "UDF name is empty: a UDF name must be non-empty and contain no '.' or '`', so the UDF could never be called by name",
            ));
        }
        if name.contains('.') || name.contains('`') {
            return Err(super::UdfError::new(format!(
                "UDF {name}.name is invalid: a UDF name must contain no '.' or '`', so the UDF could never be called by name"
            )));
        }
        let name = name.to_lowercase();
        let mut inner = self.inner.write().unwrap_or_else(PoisonError::into_inner);
        inner.udfs.insert(name, udf);
        inner.epoch = inner.epoch.wrapping_add(1);
        Ok(())
    }

    /// The UDF named `name` (case-insensitive), if registered.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn ScalarUdf>> {
        let inner = self.inner.read().unwrap_or_else(PoisonError::into_inner);
        inner.udfs.get(&name.to_lowercase()).cloned()
    }

    /// The UDF named `name` at `version` (both case- and
    /// version-sensitive), as a plan references it; `None` when the name is
    /// unregistered or holds another version. A panicking `version`
    /// matches, so that binding surfaces `FAILED_EXECUTE_UDF` rather than
    /// `KRISHIV_UDF_NOT_AVAILABLE`.
    #[must_use]
    pub fn get_versioned(&self, name: &str, version: u32) -> Option<Arc<dyn ScalarUdf>> {
        let udf = self.get(name)?;
        let matches = guard_call(name, "version", || udf.version())
            .map(|v| v == version)
            .unwrap_or(true);
        matches.then_some(udf)
    }

    /// Bumped on every registration; part of the analysis fingerprint.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .epoch
    }

    /// Whether no UDF is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .udfs
            .is_empty()
    }
}

impl Default for UdfLibrary {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for UdfLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.read().unwrap_or_else(PoisonError::into_inner);
        let mut names: Vec<String> = inner
            .udfs
            .values()
            .map(|u| {
                let name = guard_call("unknown", "name", || u.name().to_owned())
                    .unwrap_or_else(|_| "<panicking>".to_owned());
                let version = guard_call(&name, "version", || u.version())
                    .map(|v| v.to_string())
                    .unwrap_or_else(|_| "<panicking>".to_owned());
                format!("{name}@{version}")
            })
            .collect();
        names.sort();
        f.debug_struct("UdfLibrary")
            .field("udfs", &names)
            .field("epoch", &inner.epoch)
            .finish()
    }
}
