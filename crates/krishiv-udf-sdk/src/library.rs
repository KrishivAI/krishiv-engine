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

use super::{AggregateUdf, ScalarUdf, UdfError, guard_call};

/// The UDFs registered on one session, by lower-case name.
///
/// Scalar and aggregate UDFs share one namespace, as Spark's functions do:
/// registering a name replaces whatever UDF held it, of either kind. Every
/// registration bumps the library [`version`](Self::version): the
/// analyzer's context fingerprint covers it, so a cached plan never
/// outlives the registration it was analyzed with.
pub struct UdfLibrary {
    inner: RwLock<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    udfs: HashMap<String, Arc<dyn ScalarUdf>>,
    udafs: HashMap<String, Arc<dyn AggregateUdf>>,
    /// Bumped on every registration; part of the analysis fingerprint.
    epoch: u64,
}

/// The lower-case name a UDF registers under, from its (guarded) `name`.
fn checked_name(name: Result<String, UdfError>) -> Result<String, UdfError> {
    let name = name?;
    if name.is_empty() {
        return Err(UdfError::new(
            "UDF name is empty: a UDF name must be non-empty and contain no '.' or '`', so the UDF could never be called by name",
        ));
    }
    if name.contains('.') || name.contains('`') {
        return Err(UdfError::new(format!(
            "UDF {name}.name is invalid: a UDF name must contain no '.' or '`', so the UDF could never be called by name"
        )));
    }
    Ok(name.to_lowercase())
}

/// Whether a UDF's (guarded) version is `version`. A panicking `version`
/// matches, so that binding surfaces `FAILED_EXECUTE_UDF` rather than
/// `KRISHIV_UDF_NOT_AVAILABLE`.
fn version_matches(name: &str, version: u32, read: impl FnOnce() -> u32) -> bool {
    match guard_call(name, "version", read) {
        Ok(v) => v == version,
        Err(_) => true,
    }
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
    /// replacing any UDF of that name (scalar or aggregate), and bumps the
    /// library version.
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
    pub fn register(&self, udf: Arc<dyn ScalarUdf>) -> Result<(), UdfError> {
        let name = checked_name(guard_call("unknown", "name", || udf.name().to_owned()))?;
        let mut inner = self.inner.write().unwrap_or_else(PoisonError::into_inner);
        inner.udafs.remove(&name);
        inner.udfs.insert(name, udf);
        inner.epoch = inner.epoch.wrapping_add(1);
        Ok(())
    }

    /// Registers the aggregate `udaf` under its lower-case
    /// [`name`](AggregateUdf::name), replacing any UDF of that name
    /// (scalar or aggregate), and bumps the library version. Names are
    /// checked as for [`register`](Self::register).
    ///
    /// # Errors
    /// A [`UdfError`] when `name` panics, is empty, or contains a `.` or
    /// a backtick.
    pub fn register_aggregate(&self, udaf: Arc<dyn AggregateUdf>) -> Result<(), UdfError> {
        let name = checked_name(guard_call("unknown", "name", || udaf.name().to_owned()))?;
        let mut inner = self.inner.write().unwrap_or_else(PoisonError::into_inner);
        inner.udfs.remove(&name);
        inner.udafs.insert(name, udaf);
        inner.epoch = inner.epoch.wrapping_add(1);
        Ok(())
    }

    /// The scalar UDF named `name` (case-insensitive), if registered.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn ScalarUdf>> {
        let inner = self.inner.read().unwrap_or_else(PoisonError::into_inner);
        inner.udfs.get(&name.to_lowercase()).cloned()
    }

    /// The aggregate UDF named `name` (case-insensitive), if registered.
    #[must_use]
    pub fn get_aggregate(&self, name: &str) -> Option<Arc<dyn AggregateUdf>> {
        let inner = self.inner.read().unwrap_or_else(PoisonError::into_inner);
        inner.udafs.get(&name.to_lowercase()).cloned()
    }

    /// The scalar UDF named `name` at `version` (both case- and
    /// version-sensitive), as a plan references it; `None` when the name is
    /// unregistered, is an aggregate, or holds another version. A panicking
    /// `version` matches (through [`guard_call`]) so that binding surfaces
    /// `FAILED_EXECUTE_UDF` rather than `KRISHIV_UDF_NOT_AVAILABLE`.
    #[must_use]
    pub fn get_versioned(&self, name: &str, version: u32) -> Option<Arc<dyn ScalarUdf>> {
        let udf = self.get(name)?;
        version_matches(name, version, || udf.version()).then_some(udf)
    }

    /// The aggregate UDF named `name` at `version`, as a plan references
    /// it; as [`get_versioned`](Self::get_versioned) for aggregates.
    #[must_use]
    pub fn get_aggregate_versioned(
        &self,
        name: &str,
        version: u32,
    ) -> Option<Arc<dyn AggregateUdf>> {
        let udaf = self.get_aggregate(name)?;
        version_matches(name, version, || udaf.version()).then_some(udaf)
    }

    /// Bumped on every registration; part of the analysis fingerprint.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .epoch
    }

    /// Whether no UDF (scalar or aggregate) is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        let inner = self.inner.read().unwrap_or_else(PoisonError::into_inner);
        inner.udfs.is_empty() && inner.udafs.is_empty()
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
        let label = |name: Result<String, UdfError>, version: &dyn Fn() -> u32| {
            let name = name.unwrap_or_else(|_| "<panicking>".to_owned());
            let version = guard_call(&name, "version", version)
                .map(|v| v.to_string())
                .unwrap_or_else(|_| "<panicking>".to_owned());
            format!("{name}@{version}")
        };
        let mut names: Vec<String> = inner
            .udfs
            .values()
            .map(|u| {
                label(
                    guard_call("unknown", "name", || u.name().to_owned()),
                    &|| u.version(),
                )
            })
            .collect();
        let mut aggregates: Vec<String> = inner
            .udafs
            .values()
            .map(|u| {
                label(
                    guard_call("unknown", "name", || u.name().to_owned()),
                    &|| u.version(),
                )
            })
            .collect();
        aggregates.sort();
        names.sort();
        f.debug_struct("UdfLibrary")
            .field("udfs", &names)
            .field("aggregates", &aggregates)
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
            "UdfLibrary { udfs: [\"my_identity@2\"], aggregates: [], epoch: 2 }"
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

    #[test]
    fn fixtures_expose_their_declared_contracts() {
        let lib = UdfLibrary::new();
        lib.register(Arc::new(Identity { version: 1 })).unwrap();
        let udf = lib.get("my_identity").unwrap();
        assert_eq!(udf.version(), 1);
        assert_eq!(
            udf.signature(),
            &Signature::Exact(Vec::new()),
            "zero-arg signature"
        );
        let field = udf.return_field(&[]).unwrap();
        assert_eq!(field.data_type, krishiv_types::DataType::Integer);
        assert!(field.nullable);
        assert!(
            udf.invoke(
                &[],
                0,
                &InvokeContext {
                    partition: 0,
                    ansi: true,
                    time_zone: None,
                }
            )
            .is_err()
        );
        // The panicking-name and bad-name fixtures expose theirs too.
        let panicking = PanicName;
        assert_eq!(panicking.version(), 1);
        assert_eq!(panicking.signature(), &Signature::Exact(Vec::new()));
        assert!(panicking.return_field(&[]).is_ok());
        assert!(
            panicking
                .invoke(
                    &[],
                    0,
                    &InvokeContext {
                        partition: 0,
                        ansi: true,
                        time_zone: None,
                    }
                )
                .is_err()
        );
        for bad in ["", "a.b", "a`b"] {
            let udf = BadName(bad);
            assert_eq!(udf.version(), 1);
            assert_eq!(udf.signature(), &Signature::Exact(Vec::new()));
            assert!(udf.return_field(&[]).is_ok());
            assert!(
                udf.invoke(
                    &[],
                    0,
                    &InvokeContext {
                        partition: 0,
                        ansi: true,
                        time_zone: None,
                    }
                )
                .is_err()
            );
        }
    }

    /// A UDF whose `version` panics still resolves by name and version: the
    /// match defaults to true so binding surfaces `FAILED_EXECUTE_UDF`
    /// rather than `KRISHIV_UDF_NOT_AVAILABLE`.
    #[derive(Debug)]
    struct PanicVersion;

    impl ScalarUdf for PanicVersion {
        fn name(&self) -> &str {
            "panic_version"
        }

        fn version(&self) -> u32 {
            panic!("version went bang")
        }

        fn signature(&self) -> &Signature {
            use std::sync::OnceLock;
            static SIG: OnceLock<Signature> = OnceLock::new();
            SIG.get_or_init(|| Signature::Exact(Vec::new()))
        }

        fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
            Ok(ReturnField {
                data_type: krishiv_types::DataType::Integer,
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
    fn a_panicking_version_still_matches() {
        let lib = UdfLibrary::new();
        lib.register(Arc::new(PanicVersion)).unwrap();
        assert!(lib.get("panic_version").is_some());
        assert!(lib.get_versioned("panic_version", 1).is_some());
        assert!(lib.get_versioned("panic_version", 999).is_some());
        assert!(lib.get_versioned("nosuch", 1).is_none());
    }

    const CTX: InvokeContext = InvokeContext {
        partition: 0,
        ansi: false,
        time_zone: None,
    };

    /// An aggregate UDF whose accumulator is never built here.
    #[derive(Debug)]
    struct Total {
        name: &'static str,
        version: u32,
    }

    impl super::super::AggregateUdf for Total {
        fn name(&self) -> &str {
            self.name
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
                data_type: DataType::Long,
                nullable: true,
            })
        }

        fn state_fields(&self, _: &[ArgField]) -> Result<Vec<super::super::StateField>, UdfError> {
            Ok(Vec::new())
        }

        fn accumulator(
            &self,
            _: &[ArgField],
            _: &InvokeContext,
        ) -> Result<Box<dyn super::super::GroupsAccumulator>, UdfError> {
            Err(UdfError::new("test only"))
        }
    }

    #[test]
    fn aggregates_register_by_lower_case_name_and_version() {
        let lib = UdfLibrary::new();
        let before = lib.version();
        lib.register_aggregate(Arc::new(Total {
            name: "My_Total",
            version: 2,
        }))
        .unwrap();
        assert_ne!(lib.version(), before, "registering bumps the epoch");
        // Names are case-insensitive.
        let udaf = lib.get_aggregate("my_total").unwrap();
        assert!(lib.get_aggregate("MY_TOTAL").is_some());
        assert!(lib.get_aggregate("nosuch").is_none());
        assert!(lib.get_aggregate_versioned("my_total", 2).is_some());
        assert!(lib.get_aggregate_versioned("my_total", 3).is_none());
        assert!(lib.get_aggregate_versioned("nosuch", 2).is_none());
        // The debug form lists it with its own name.
        let shown = format!("{lib:?}");
        assert!(
            shown.contains("aggregates") && shown.contains("My_Total"),
            "{shown}"
        );
        // The double declares what the contract asks of it.
        assert_eq!(udaf.signature(), &Signature::Exact(Vec::new()));
        assert_eq!(udaf.return_field(&[]).unwrap().data_type, DataType::Long);
        assert!(udaf.state_fields(&[]).unwrap().is_empty());
        assert_eq!(
            udaf.properties(),
            super::super::AggregateProperties::DEFAULT
        );
        assert!(udaf.accumulator(&[], &CTX).is_err());
    }

    #[test]
    fn registering_an_aggregate_replaces_a_scalar_of_the_same_name() {
        let lib = UdfLibrary::new();
        lib.register(Arc::new(Identity { version: 1 })).unwrap();
        assert!(lib.get("my_identity").is_some());
        lib.register_aggregate(Arc::new(Total {
            name: "my_identity",
            version: 1,
        }))
        .unwrap();
        assert!(lib.get("my_identity").is_none(), "one name, one kind");
        assert!(lib.get_aggregate("my_identity").is_some());
    }

    #[test]
    fn the_panicking_version_double_is_otherwise_a_working_udf() {
        let u = PanicVersion;
        assert_eq!(u.signature(), &Signature::Exact(Vec::new()));
        assert_eq!(u.return_field(&[]).unwrap().data_type, DataType::Integer);
        assert!(u.invoke(&[], 0, &CTX).is_err());
        assert_eq!(u.properties(), super::super::UdfProperties::default());
    }
}
