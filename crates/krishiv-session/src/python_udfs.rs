// SPDX-License-Identifier: Apache-2.0
//! The session's Python UDF registrations (design P2-T059 §2.1): what the
//! analyzer types a call by. The callable itself never enters the session:
//! it lives in the UDF runtime the Python package installs
//! ([`Session::set_python_udf_runtime`](crate::Session::set_python_udf_runtime)),
//! keyed by the same `(name, version)` a plan carries
//! ([`FunctionId::PythonUdf`]).
//!
//! Re-registering a name gives it the next version and bumps the library's
//! epoch, which the analysis fingerprint covers: a cached plan never runs a
//! callable it was not analyzed with, and a worker without that version
//! refuses the plan (`KRISHIV_UDF_NOT_AVAILABLE`).

use std::collections::HashMap;
use std::sync::{Arc, PoisonError, RwLock};

use krishiv_plan::analyzer::{ArgType, Arity, CallTypes, FunctionEntry, Inputs};
use krishiv_plan::{FunctionId, FunctionKind, PythonEvalType};
use krishiv_types::DataType;

/// What a Python UDF is registered with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonUdfDef {
    /// The name, as registered (resolved ASCII case-insensitively).
    pub name: String,
    /// How it is converted and called.
    pub eval_type: PythonEvalType,
    /// The declared return type. Results are always nullable, as in Spark.
    pub return_type: DataType,
    /// Spark's `deterministic` (`asNondeterministic()` clears it).
    pub deterministic: bool,
    /// Whether calls may overlap; `false` serialises them in the runtime.
    pub thread_safe: bool,
}

/// One registration: the definition and the version it was given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonUdfEntry {
    /// The definition.
    pub def: PythonUdfDef,
    /// The registration version (1 for a name's first registration).
    pub version: u32,
}

/// The Python UDFs registered on one session, by lower-case name.
#[derive(Debug, Default)]
pub struct PythonUdfLibrary {
    inner: RwLock<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    udfs: HashMap<String, PythonUdfEntry>,
    /// The last version given to each name, kept after a replacement so a
    /// version is never reused.
    versions: HashMap<String, u32>,
    /// Bumped on every registration; part of the analysis fingerprint.
    epoch: u64,
}

impl PythonUdfLibrary {
    /// An empty library.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `def` under its lower-case name, replacing any Python UDF
    /// of that name, and returns the registration with its new version.
    #[must_use]
    pub fn register(&self, def: PythonUdfDef) -> PythonUdfEntry {
        let key = def.name.to_lowercase();
        let mut inner = self.inner.write().unwrap_or_else(PoisonError::into_inner);
        let version = inner
            .versions
            .get(&key)
            .copied()
            .unwrap_or(0)
            .saturating_add(1);
        inner.versions.insert(key.clone(), version);
        let entry = PythonUdfEntry { def, version };
        inner.udfs.insert(key, entry.clone());
        inner.epoch = inner.epoch.wrapping_add(1);
        entry
    }

    /// The registration of `name` (any case).
    #[must_use]
    pub fn get(&self, name: &str) -> Option<PythonUdfEntry> {
        self.inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .udfs
            .get(&name.to_lowercase())
            .cloned()
    }

    /// Bumped on every registration.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .epoch
    }

    /// The analyzer's entry for `name` (already lower-case), if registered.
    /// A Python UDF takes any number of arguments of any type, as in Spark
    /// (the converters check them when it runs), and its result is nullable.
    #[must_use]
    pub fn entry(&self, name: &str) -> Option<FunctionEntry> {
        let PythonUdfEntry { def, version } = self.get(name)?;
        let data_type = def.return_type.clone();
        let result = Arc::new(move |_: &CallTypes<'_>| {
            Ok(ArgType {
                data_type: Some(data_type.clone()),
                nullable: true,
            })
        });
        Some(FunctionEntry {
            id: FunctionId::PythonUdf {
                name: name.to_owned(),
                version,
                eval_type: def.eval_type,
                thread_safe: def.thread_safe,
            },
            kind: FunctionKind::Scalar,
            arity: Arity::any(),
            inputs: Inputs::Any,
            result,
            deterministic: def.deterministic,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def(name: &str) -> PythonUdfDef {
        PythonUdfDef {
            name: name.to_owned(),
            eval_type: PythonEvalType::ArrowBatched,
            return_type: DataType::Integer,
            deterministic: true,
            thread_safe: true,
        }
    }

    #[test]
    fn re_registering_gives_the_next_version_and_bumps_the_epoch() {
        let lib = PythonUdfLibrary::new();
        assert_eq!(lib.register(def("Plus")).version, 1);
        let before = lib.version();
        assert_eq!(lib.register(def("plus")).version, 2);
        assert_ne!(lib.version(), before);
        assert_eq!(lib.get("PLUS").map(|e| e.version), Some(2));
        assert!(lib.get("other").is_none());
    }

    #[test]
    fn the_entry_is_a_nullable_scalar_reference_by_name_and_version() {
        let lib = PythonUdfLibrary::new();
        let mut d = def("f");
        d.deterministic = false;
        d.thread_safe = false;
        let _ = lib.register(d);
        let Some(entry) = lib.entry("f") else {
            panic!("registered");
        };
        assert_eq!(
            entry.id,
            FunctionId::PythonUdf {
                name: "f".to_owned(),
                version: 1,
                eval_type: PythonEvalType::ArrowBatched,
                thread_safe: false,
            }
        );
        assert!(!entry.deterministic);
        assert_eq!(entry.kind, FunctionKind::Scalar);
        assert!(lib.entry("g").is_none());
    }
}
