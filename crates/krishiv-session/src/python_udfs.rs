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
//!
//! An unregistered UDF (a DataFrame-side `udf(...)` call, design unit 2c
//! (a)) is never in a library: its call carries its whole definition in an
//! encoded name ([`anonymous_call_name`]) under the `session` qualifier, which
//! [`SessionFunctions`](crate::SessionFunctions) decodes before resolving a
//! name, so SQL cannot reach it by its display name. Its version has the top
//! bit set ([`ANONYMOUS_VERSION_BIT`]), so it never equals a registration's.

use std::collections::HashMap;
use std::sync::{Arc, PoisonError, RwLock};

use krishiv_common::error::{InternalError, KrishivError};
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

    /// The analyzer's entry for `name` (any case), if registered. Its
    /// identity names the UDF as it was registered, as Spark names the
    /// expression by the registered name.
    #[must_use]
    pub fn entry(&self, name: &str) -> Option<FunctionEntry> {
        let PythonUdfEntry { def, version } = self.get(name)?;
        Some(entry_of(&def, version))
    }
}

/// The analyzer's entry of `def` at `version`: a Python UDF takes any number
/// of arguments of any type, as in Spark (the converters check them when it
/// runs), and its result is nullable.
pub(crate) fn entry_of(def: &PythonUdfDef, version: u32) -> FunctionEntry {
    let data_type = def.return_type.clone();
    let result = Arc::new(move |_: &CallTypes<'_>| {
        Ok(ArgType {
            data_type: Some(data_type.clone()),
            nullable: true,
        })
    });
    FunctionEntry {
        id: FunctionId::PythonUdf {
            name: def.name.clone(),
            version,
            eval_type: def.eval_type,
            thread_safe: def.thread_safe,
        },
        kind: FunctionKind::Scalar,
        arity: Arity::any(),
        inputs: Inputs::Any,
        result,
        deterministic: def.deterministic,
    }
}

/// The bit every unregistered UDF's version has, and no registration's
/// (registrations count up from 1 per name).
pub const ANONYMOUS_VERSION_BIT: u32 = 0x8000_0000;

/// What starts an encoded name; the fields are separated by it too. A NUL
/// cannot be typed in SQL text short of a backquoted identifier holding one.
const ANONYMOUS_MARK: char = '\u{0}';

/// The tag after the first mark.
const ANONYMOUS_TAG: &str = "krishiv.python_udf";

/// The encoded name of an unregistered UDF's call: the definition and
/// `version` (whose [`ANONYMOUS_VERSION_BIT`] must be set). Called under the
/// `session` qualifier it resolves to `def` at `version`, shown as
/// `def.name`.
///
/// # Errors
/// `KRISHIV_INTERNAL_ERROR` when `version` lacks the bit or the name holds
/// a NUL.
pub(crate) fn anonymous_call_name(
    def: &PythonUdfDef,
    version: u32,
) -> Result<String, KrishivError> {
    if version & ANONYMOUS_VERSION_BIT == 0 || def.name.contains(ANONYMOUS_MARK) {
        return Err(InternalError {
            message: format!(
                "an unregistered Python UDF call needs a version with the top bit and a name without NUL (got {version:#x}, {:?})",
                def.name
            ),
        }
        .into());
    }
    let m = ANONYMOUS_MARK;
    Ok(format!(
        "{m}{ANONYMOUS_TAG}{m}{version}{m}{}{m}{}{m}{}{m}{}{m}{}",
        def.eval_type.code(),
        u8::from(def.deterministic),
        u8::from(def.thread_safe),
        def.return_type.sql(),
        def.name
    ))
}

/// The definition and version an encoded name ([`anonymous_call_name`])
/// carries; `None` for any other name.
#[must_use]
pub(crate) fn decode_anonymous(name: &str) -> Option<(PythonUdfDef, u32)> {
    let rest = name.strip_prefix(ANONYMOUS_MARK)?;
    let mut parts = rest.splitn(7, ANONYMOUS_MARK);
    if parts.next()? != ANONYMOUS_TAG {
        return None;
    }
    let version: u32 = parts.next()?.parse().ok()?;
    if version & ANONYMOUS_VERSION_BIT == 0 {
        return None;
    }
    let eval_type = PythonEvalType::from_code(parts.next()?.parse().ok()?)?;
    let flag = |p: &str| match p {
        "0" => Some(false),
        "1" => Some(true),
        _ => None,
    };
    let deterministic = flag(parts.next()?)?;
    let thread_safe = flag(parts.next()?)?;
    let return_type = krishiv_types::parse_data_type(parts.next()?).ok()?;
    let display = parts.next()?;
    Some((
        PythonUdfDef {
            name: display.to_owned(),
            eval_type,
            return_type,
            deterministic,
            thread_safe,
        },
        version,
    ))
}

impl crate::Session {
    /// Registers a Python UDF (P2-T059) under `def.name`, replacing any
    /// Python UDF of that name; the entry carries the new registration
    /// version, which plans name the UDF by. The callable itself stays with
    /// the runtime set by [`set_python_udf_runtime`](Self::set_python_udf_runtime),
    /// keyed by name and version.
    pub fn register_python_udf(&self, def: PythonUdfDef) -> PythonUdfEntry {
        self.state.python_udfs.register(def)
    }

    /// Sets the runtime that evaluates this session's Python UDFs (the
    /// Python facade's embedded interpreter); queries started after this
    /// call use it.
    pub fn set_python_udf_runtime(
        &self,
        runtime: std::sync::Arc<dyn krishiv_exec::python_eval::UdfRuntime>,
    ) {
        *self
            .state
            .python_runtime
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(runtime);
    }

    /// `worker` calling this session's Python UDFs through its runtime, if
    /// it has one (a plan calling one without fails with
    /// `KRISHIV_UDF_NOT_AVAILABLE`).
    pub(crate) fn with_python_runtime(
        &self,
        worker: krishiv_exec::physical::WorkerContext,
    ) -> krishiv_exec::physical::WorkerContext {
        let python = self
            .state
            .python_runtime
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        match python {
            Some(runtime) => worker.with_python_udfs(runtime),
            None => worker,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_encoded_name_round_trips_its_definition() {
        let d = PythonUdfDef {
            name: "<lambda>".to_owned(),
            eval_type: PythonEvalType::ScalarPandas,
            return_type: krishiv_types::parse_data_type("struct<a:int,`b c`:array<string>>")
                .unwrap_or(DataType::Null),
            deterministic: false,
            thread_safe: true,
        };
        let name = anonymous_call_name(&d, ANONYMOUS_VERSION_BIT | 7).unwrap_or_default();
        assert_eq!(
            decode_anonymous(&name),
            Some((d.clone(), ANONYMOUS_VERSION_BIT | 7))
        );
        assert!(anonymous_call_name(&d, 7).is_err());
        assert_eq!(decode_anonymous("<lambda>"), None);
        assert_eq!(decode_anonymous(&name.replace("krishiv", "other")), None);
        let entry = entry_of(&d, ANONYMOUS_VERSION_BIT | 7);
        assert_eq!(entry.id.name(), "<lambda>");
        assert!(!entry.deterministic);
    }

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
