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
    /// call: a panicking `name` is returned as a [`UdfError`] instead of
    /// unwinding (there is no name to register the UDF under). An empty
    /// name, or one containing a `.` or a backtick, is also an error,
    /// since such a UDF can never be called by name.
    ///
    /// # Errors
    /// A [`UdfError`] when `name` panics, is empty, or contains a `.` or
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
    /// unregistered or holds another version. A panicking `version` matches
    /// (through [`guard_call`]) so that binding surfaces `FAILED_EXECUTE_UDF`
    /// rather than `KRISHIV_UDF_NOT_AVAILABLE`.
    #[must_use]
    pub fn get_versioned(&self, name: &str, version: u32) -> Option<Arc<dyn ScalarUdf>> {
        let udf = self.get(name)?;
        let matches = match guard_call(name, "version", || udf.version()) {
            Ok(v) => v == version,
            Err(_) => true,
        };
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

#[cfg(test)]
mod tests {
    use super::super::{ArgField, InvokeContext, ReturnField, Signature, UdfArg, UdfError};
    use super::*;

    use arrow_array::ArrayRef;
    use krishiv_types::DataType;

    #[derive(Debug)]
    struct Identity {
        version: u32,
    }

    impl ScalarUdf for Identity {
        fn name(&self) -> &str {
            "my_identity"
        }

        fn version(&self) -> u32 {
            self.version
        }

        fn signature(&self) -> &Signature {
            use std::sync::OnceLock;
            static SIG: OnceLock<Signature> = OnceLock::new();
            SIG.get_or_init(|| Signature::Exact(Vec::new()))
        }

        fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
            Ok(ReturnField {
                data_type: DataType::Integer,
                nullable: true,
            })
        }

        fn invoke(
            &self,
            _: &[UdfArg<'_>],
            _: usize,
            _: &InvokeContext,
        ) -> Result<ArrayRef, UdfError> {
            Err(UdfError::new("unimplemented"))
        }
    }

    #[test]
    fn registration_replaces_and_bumps_the_version() {
        let lib = UdfLibrary::new();
        assert!(lib.is_empty());
        assert_eq!(lib.version(), 0);
        lib.register(Arc::new(Identity { version: 1 })).unwrap();
        assert!(!lib.is_empty());
        assert_eq!(lib.version(), 1);
        assert_eq!(lib.get("MY_IDENTITY").unwrap().version(), 1);
        // Another registration, same or new version: replaced, bumped.
        lib.register(Arc::new(Identity { version: 2 })).unwrap();
        assert_eq!(lib.version(), 2);
        assert_eq!(lib.get("my_identity").unwrap().version(), 2);
        assert!(lib.get("nosuch").is_none());
        // A plan names its version: only the matching one resolves.
        assert!(lib.get_versioned("my_identity", 2).is_some());
        assert!(lib.get_versioned("my_identity", 1).is_none());
        assert!(lib.get_versioned("nosuch", 2).is_none());
        assert_eq!(
            format!("{lib:?}"),
            "UdfLibrary { udfs: [\"my_identity@2\"], epoch: 2 }"
        );
        assert_eq!(format!("{}", UdfLibrary::default().version()), "0");
    }

    /// A UDF whose `name` panics: registration catches it and leaves the
    /// library unchanged instead of unwinding.
    #[derive(Debug)]
    struct PanicName;

    impl ScalarUdf for PanicName {
        fn name(&self) -> &str {
            panic!("boom")
        }

        fn version(&self) -> u32 {
            1
        }

        fn signature(&self) -> &Signature {
            use std::sync::OnceLock;
            static SIG: OnceLock<Signature> = OnceLock::new();
            SIG.get_or_init(|| Signature::Exact(Vec::new()))
        }

        fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
            Ok(ReturnField {
                data_type: DataType::Integer,
                nullable: true,
            })
        }

        fn invoke(
            &self,
            _: &[UdfArg<'_>],
            _: usize,
            _: &InvokeContext,
        ) -> Result<ArrayRef, UdfError> {
            Err(UdfError::new("test only"))
        }
    }

    #[test]
    fn panicking_name_leaves_the_library_unchanged() {
        let lib = UdfLibrary::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            lib.register(Arc::new(PanicName)).unwrap_err()
        }));
        let err = result.expect("a UDF whose name panics must not unwind registration");
        assert!(err.message().contains("boom"), "{err}");
        assert!(lib.is_empty());
        assert_eq!(lib.version(), 0);
    }

    /// A UDF with an empty, dotted or backquoted name: registration
    /// rejects it, since such a UDF can never be called by name.
    #[derive(Debug)]
    struct BadName(&'static str);

    impl ScalarUdf for BadName {
        fn name(&self) -> &str {
            self.0
        }

        fn version(&self) -> u32 {
            1
        }

        fn signature(&self) -> &Signature {
            use std::sync::OnceLock;
            static SIG: OnceLock<Signature> = OnceLock::new();
            SIG.get_or_init(|| Signature::Exact(Vec::new()))
        }

        fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
            Ok(ReturnField {
                data_type: DataType::Integer,
                nullable: true,
            })
        }

        fn invoke(
            &self,
            _: &[UdfArg<'_>],
            _: usize,
            _: &InvokeContext,
        ) -> Result<ArrayRef, UdfError> {
            Err(UdfError::new("test only"))
        }
    }

    #[test]
    fn empty_dotted_or_backquoted_names_are_rejected() {
        for bad in ["", "a.b", "a`b"] {
            let lib = UdfLibrary::new();
            let err = lib.register(Arc::new(BadName(bad))).unwrap_err();
            assert!(!err.message().is_empty(), "{bad}: {err}");
            assert!(lib.is_empty(), "{bad}");
            assert_eq!(lib.version(), 0, "{bad}");
        }
    }
}
