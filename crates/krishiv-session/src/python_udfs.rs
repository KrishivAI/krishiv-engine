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
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use krishiv_common::error::{InternalError, KrishivError};
use krishiv_plan::analyzer::{ArgType, Arity, CallTypes, FunctionEntry, Inputs};
use krishiv_plan::{FunctionId, FunctionKind, PythonEvalType, PythonUdfKind};
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

/// What a Python table function (UDTF, eval types 300, 301, 302) is
/// registered with (Spark's `spark.udtf.register`; Spark Connect's
/// `CommonInlineUserDefinedTableFunction` sends the same fields).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonUdtfDef {
    /// The name, as registered (resolved ASCII case-insensitively).
    pub name: String,
    /// The eval type the client chose (`@udtf` is 300 or 301 by the
    /// client's `spark.sql.execution.pythonUDTF.arrow.enabled`, `@arrow_udtf`
    /// 302): kept as sent, never re-derived from the server's settings.
    pub eval_type: PythonEvalType,
    /// The declared result (a struct), or `None` for a UDTF whose static
    /// `analyze` gives it per call.
    pub return_type: Option<DataType>,
    /// Spark's `deterministic` (`asDeterministic()` sets it; `False` by
    /// default for a UDTF).
    pub deterministic: bool,
}

/// One UDTF registration: the definition and the version it was given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonUdtfEntry {
    /// The definition.
    pub def: PythonUdtfDef,
    /// The registration version: plans and the runtime name the UDTF by
    /// `(name, version)`. Drawn from the same allocator as the scalar
    /// UDFs', so a UDTF and a UDF of one name never share a version.
    pub version: u32,
}

#[derive(Debug, Default, Clone)]
struct Inner {
    udfs: HashMap<String, PythonUdfEntry>,
    /// The session's Python table functions, a namespace of their own (as
    /// Spark's table-function registry is).
    udtfs: HashMap<String, PythonUdtfEntry>,
    /// The last version given to each name, kept after a replacement so a
    /// version is never reused. Shared with the libraries copied from this
    /// one ([`PythonUdfLibrary::copy`]): a clone shares its source's Python
    /// runtime, which keys callables by `(name, version)`, so two sessions
    /// must never give one name the same version.
    versions: Arc<Mutex<HashMap<String, u32>>>,
    /// Bumped on every registration; part of the analysis fingerprint.
    epoch: u64,
}

impl PythonUdfLibrary {
    /// A library holding this one's registrations; later registrations to
    /// either do not reach the other, but both draw versions from one
    /// allocator.
    pub(crate) fn copy(&self) -> Self {
        let inner = self.inner.read().unwrap_or_else(PoisonError::into_inner);
        Self {
            inner: RwLock::new(inner.clone()),
        }
    }

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
        let version = {
            let mut versions = inner
                .versions
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let version = versions.get(&key).copied().unwrap_or(0).saturating_add(1);
            versions.insert(key.clone(), version);
            version
        };
        let entry = PythonUdfEntry { def, version };
        inner.udfs.insert(key, entry.clone());
        inner.epoch = inner.epoch.wrapping_add(1);
        entry
    }

    /// Registers the Python UDTF `def` under its lower-case name, replacing
    /// any UDTF of that name, and returns the registration with its new
    /// version.
    ///
    /// # Errors
    /// `KRISHIV_INTERNAL_ERROR` when `def.eval_type` is not a table kind or
    /// its return type is not a struct.
    pub fn register_udtf(&self, def: PythonUdtfDef) -> Result<PythonUdtfEntry, KrishivError> {
        if def.eval_type.kind() != PythonUdfKind::Udtf
            || def
                .return_type
                .as_ref()
                .is_some_and(|t| !matches!(t, DataType::Struct(_)))
        {
            return Err(InternalError {
                message: format!(
                    "a Python UDTF registration needs a table eval type (300, 301, 302) and a struct result (got {} for {})",
                    def.eval_type.code(),
                    def.name
                ),
            }
            .into());
        }
        let key = def.name.to_lowercase();
        let mut inner = self.inner.write().unwrap_or_else(PoisonError::into_inner);
        let version = {
            let mut versions = inner
                .versions
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let version = versions.get(&key).copied().unwrap_or(0).saturating_add(1);
            versions.insert(key.clone(), version);
            version
        };
        let entry = PythonUdtfEntry { def, version };
        inner.udtfs.insert(key, entry.clone());
        inner.epoch = inner.epoch.wrapping_add(1);
        Ok(entry)
    }

    /// The UDTF registration of `name` (any case).
    #[must_use]
    pub fn get_udtf(&self, name: &str) -> Option<PythonUdtfEntry> {
        self.inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .udtfs
            .get(&name.to_lowercase())
            .cloned()
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
        kind: if def.eval_type.kind() == PythonUdfKind::GroupedAgg {
            FunctionKind::Aggregate
        } else {
            FunctionKind::Scalar
        },
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

/// The encoded name an unregistered UDF's call goes by under the `session`
/// qualifier: `def` at `version` (whose [`ANONYMOUS_VERSION_BIT`] must be
/// set), as [`functions::python_udf_call`](crate::functions::python_udf_call)
/// names it. For servers that build the IR call themselves (Spark Connect's
/// inline Python UDFs).
///
/// # Errors
/// `KRISHIV_INTERNAL_ERROR` when `version` lacks the bit or the name holds
/// a NUL.
#[cfg(feature = "unstable-ir")]
pub fn python_udf_call_name(def: &PythonUdfDef, version: u32) -> Result<String, KrishivError> {
    anonymous_call_name(def, version)
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

/// The table functions a session's queries resolve: the built-ins, then
/// its Python UDTFs, whose `analyze` runs through the session's Python
/// runtime (the one analysis path of SQL and Spark Connect).
pub(crate) struct SessionTableFunctions {
    pub(crate) builtins: krishiv_plan::analyzer::BuiltinTableFunctions,
    pub(crate) library: Arc<PythonUdfLibrary>,
    pub(crate) runtime: Option<Arc<dyn krishiv_exec::python_eval::UdfRuntime>>,
}

impl krishiv_plan::analyzer::TableFunctionRegistry for SessionTableFunctions {
    fn lookup(
        &self,
        name: &str,
        profile: krishiv_plan::analyzer::SparkProfile,
    ) -> krishiv_plan::analyzer::TableFunctionLookup {
        self.builtins.lookup(name, profile)
    }

    fn version(&self) -> u64 {
        self.library.version()
    }

    fn python_udtf(&self, name: &str) -> Option<krishiv_plan::udtf::PythonUdtfEntry> {
        // One part only: `session.f` and catalog names are not UDTF names.
        if name.contains(krishiv_plan::TABLE_FUNCTION_NAME_SEP) {
            return None;
        }
        let PythonUdtfEntry { def, version } = self.library.get_udtf(name)?;
        Some(krishiv_plan::udtf::PythonUdtfEntry {
            udf: krishiv_plan::PythonUdfRef {
                name: def.name.to_lowercase(),
                version,
                eval_type: def.eval_type,
                thread_safe: true,
                deterministic: def.deterministic,
            },
            return_type: def.return_type,
        })
    }

    fn analyze_python_udtf(
        &self,
        spec: &krishiv_plan::udtf::UdtfAnalyzeSpec,
    ) -> Result<krishiv_plan::udtf::UdtfAnalysis, KrishivError> {
        let Some(runtime) = &self.runtime else {
            return Err(krishiv_common::error::KrishivUdfNotAvailable {
                name: spec.name.clone(),
                version: spec.version.to_string(),
            }
            .into());
        };
        // The runtime runs `analyze` on its own threads; this only waits.
        futures::executor::block_on(runtime.analyze_udtf(spec))
    }
}

impl crate::Session {
    /// Registers a Python table function (UDTF, PR-26c-4) under `def.name`,
    /// replacing any UDTF of that name: SQL's `FROM name(..)` and
    /// `LATERAL name(..)`, and Spark Connect's `register_table_function`,
    /// resolve it through this one registration. The callable stays with
    /// the runtime ([`set_python_udf_runtime`](Self::set_python_udf_runtime)),
    /// which the caller gives the payload under the returned
    /// `(name, version)`; a UDTF without a return type runs its `analyze`
    /// through that runtime when a query calling it is analyzed.
    ///
    /// # Errors
    /// `KRISHIV_INTERNAL_ERROR` when `def.eval_type` is not 300, 301 or 302
    /// or the return type is not a struct.
    pub fn register_python_udtf(
        &self,
        def: PythonUdtfDef,
    ) -> Result<PythonUdtfEntry, KrishivError> {
        self.state.python_udfs.register_udtf(def)
    }

    /// The table functions a query of this session resolves.
    pub(crate) fn table_functions(&self) -> Arc<dyn krishiv_plan::analyzer::TableFunctionRegistry> {
        Arc::new(SessionTableFunctions {
            builtins: krishiv_plan::analyzer::BuiltinTableFunctions::new(),
            library: Arc::clone(&self.state.python_udfs),
            runtime: self
                .state
                .python_runtime
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone(),
        })
    }

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

    /// A grouped-aggregate UDF (202, 217, 252, 254) is an aggregate to the analyzer, as Spark's
    /// `PythonUDAF`: it may be grouped by, and it may run over a window (as 203, 253).
    #[test]
    fn a_grouped_aggregate_udf_is_an_aggregate() {
        let lib = PythonUdfLibrary::default();
        for (name, eval_type) in [
            ("p", PythonEvalType::GroupedAggPandas),
            ("pi", PythonEvalType::GroupedAggPandasIter),
            ("a", PythonEvalType::GroupedAggArrow),
            ("ai", PythonEvalType::GroupedAggArrowIter),
        ] {
            let mut d = def(name);
            d.eval_type = eval_type;
            let _ = lib.register(d);
            let entry = lib.entry(name).map(|e| e.kind);
            assert_eq!(entry, Some(FunctionKind::Aggregate), "{eval_type:?}");
        }
    }
}
