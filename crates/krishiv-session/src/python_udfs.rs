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

use futures::future::BoxFuture;
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
#[derive(Clone)]
pub(crate) struct SessionTableFunctions {
    pub(crate) builtins: krishiv_plan::analyzer::BuiltinTableFunctions,
    pub(crate) library: Arc<PythonUdfLibrary>,
    pub(crate) runtime: Option<Arc<dyn krishiv_exec::python_eval::UdfRuntime>>,
    /// The policy the analysis asks (`AnalysisContext::policy`: the session's, or a trial's
    /// `QuietPolicy`), which decides whether `principal` may run Python (`Usage` on `System("udf")`). A
    /// refusal is an analysis error, audited and counted once as the analysis's read denials are
    /// (`access::audit_read`: not in a trial).
    pub(crate) policy: Arc<dyn krishiv_plan::analyzer::AccessPolicy>,
    pub(crate) principal: Arc<krishiv_plan::analyzer::Principal>,
}

impl SessionTableFunctions {
    /// This registry asking `policy` (the analysis's, after `QueryEnv::set_quiet_denials`).
    pub(crate) fn with_policy(
        &self,
        policy: &Arc<dyn krishiv_plan::analyzer::AccessPolicy>,
    ) -> Self {
        Self {
            policy: Arc::clone(policy),
            ..self.clone()
        }
    }
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

    fn python_udtf(&self, name: &str) -> Option<krishiv_plan::udtf::ResolvedUdtf> {
        // `session.<encoded>`: an unregistered (inline) UDTF's call, its definition in the name.
        let (def, version) = match name.split_once(krishiv_plan::TABLE_FUNCTION_NAME_SEP) {
            Some((q, f)) if q.eq_ignore_ascii_case("session") => decode_anonymous_udtf(f)?,
            // Other qualified names (catalog functions) are not UDTF names.
            Some(_) => return None,
            None => {
                let PythonUdtfEntry { def, version } = self.library.get_udtf(name)?;
                (def, version)
            }
        };
        Some(krishiv_plan::udtf::ResolvedUdtf {
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
        // ADR 0046 §3b: an analysis that runs Python asks `Usage` on `System("udf")` every time, before any
        // worker starts (a registration was checked when it was made, a call is checked when it runs).
        let (policy, principal) = (Arc::clone(&self.policy), Arc::clone(&self.principal));
        let runtime = Arc::clone(runtime);
        let spec = spec.clone();
        let checked = async move {
            krishiv_plan::analyzer::authorize(
                &*policy,
                &principal,
                &krishiv_plan::analyzer::Securable::System("udf".to_owned()),
                krishiv_plan::analyzer::Privilege::Usage,
            )
            .await?;
            runtime.analyze_udtf(&spec).await
        };
        wait_for_analysis(Box::pin(checked))
    }
}

/// What starts an unregistered UDTF's encoded name, and separates its fields.
const ANONYMOUS_UDTF_TAG: &str = "krishiv.python_udtf";

/// The encoded name of an unregistered (inline) UDTF's call: `def` at `version` (whose
/// [`ANONYMOUS_VERSION_BIT`] must be set), the return type as Spark's JSON (empty without one).
fn anonymous_udtf_name(def: &PythonUdtfDef, version: u32) -> Result<String, KrishivError> {
    if version & ANONYMOUS_VERSION_BIT == 0
        || def.name.contains(ANONYMOUS_MARK)
        || def.eval_type.kind() != PythonUdfKind::Udtf
        || def
            .return_type
            .as_ref()
            .is_some_and(|t| !matches!(t, DataType::Struct(_)))
    {
        return Err(InternalError {
            message: format!(
                "an unregistered Python UDTF call needs a version with the top bit, a table eval type, a struct \
                 result and a name without NUL (got {version:#x}, {}, {:?})",
                def.eval_type.code(),
                def.name
            ),
        }
        .into());
    }
    let m = ANONYMOUS_MARK;
    Ok(format!(
        "{m}{ANONYMOUS_UDTF_TAG}{m}{version}{m}{}{m}{}{m}{}{m}{}",
        def.eval_type.code(),
        u8::from(def.deterministic),
        def.return_type
            .as_ref()
            .map(DataType::json)
            .unwrap_or_default(),
        def.name
    ))
}

/// The definition and version an encoded UDTF name carries; `None` for any other name.
fn decode_anonymous_udtf(name: &str) -> Option<(PythonUdtfDef, u32)> {
    let rest = name.strip_prefix(ANONYMOUS_MARK)?;
    let mut parts = rest.splitn(6, ANONYMOUS_MARK);
    if parts.next()? != ANONYMOUS_UDTF_TAG {
        return None;
    }
    let version: u32 = parts.next()?.parse().ok()?;
    if version & ANONYMOUS_VERSION_BIT == 0 {
        return None;
    }
    let eval_type = PythonEvalType::from_code(parts.next()?.parse().ok()?)?;
    if eval_type.kind() != PythonUdfKind::Udtf {
        return None;
    }
    let deterministic = match parts.next()? {
        "0" => false,
        "1" => true,
        _ => return None,
    };
    let return_type = match parts.next()? {
        "" => None,
        json => Some(DataType::from_json(json).ok()?),
    };
    let display = parts.next()?;
    Some((
        PythonUdtfDef {
            name: display.to_owned(),
            eval_type,
            return_type,
            deterministic,
        },
        version,
    ))
}

/// The table-function name an unregistered (inline) Python UDTF's call goes by: `def` at `version` (whose
/// [`ANONYMOUS_VERSION_BIT`] must be set) under the `session` qualifier, as a
/// [`PlanNode::TableFunction`](krishiv_plan::PlanNode)'s `name`. It resolves and runs `analyze` as a
/// registered UDTF does (`Usage` asked each time) without an entry in the session's library; the runtime
/// holds its payload under `(def.name` in lower case`, version)`. For Spark Connect's inline UDTFs.
///
/// # Errors
/// `KRISHIV_INTERNAL_ERROR` when `version` lacks the bit, the eval type is not a table kind, the result is
/// not a struct or the name holds a NUL.
#[cfg(feature = "unstable-ir")]
pub fn python_udtf_call_name(def: &PythonUdtfDef, version: u32) -> Result<String, KrishivError> {
    Ok(format!(
        "session{}{}",
        krishiv_plan::TABLE_FUNCTION_NAME_SEP,
        anonymous_udtf_name(def, version)?
    ))
}

/// Waits for `analysis` (analysis is synchronous, `analyze` runs on the runtime's own threads), ending with
/// the statement's cancel: a cancelled action's error comes back at once and `analysis` is dropped, which
/// abandons the runtime's call (the pool kills its worker). On a multi-thread runtime the wait leaves the
/// executor thread to the runtime's other tasks (`block_in_place`). On a current-thread runtime (a caller's
/// own, or the deterministic simulation's) the wait would block the runtime's only thread: no cancel from a task
/// of that runtime could run, and a policy check needing the runtime (a timer, I/O) would never finish. Analysis
/// is synchronous, so that wait cannot yield; the analysis is refused with `KRISHIV_UNSUPPORTED_FEATURE`
/// instead of risking a hang (review R2-1). Production analysis runs on the multi-thread executor.
fn wait_for_analysis(
    analysis: futures::future::BoxFuture<
        'static,
        krishiv_exec::Result<krishiv_plan::udtf::UdtfAnalysis>,
    >,
) -> Result<krishiv_plan::udtf::UdtfAnalysis, KrishivError> {
    wait_for_user_code(
        analysis,
        "analyzing a Python UDTF without a declared result on a current-thread runtime; \
         run the statement on a multi-thread runtime or declare the UDTF's return type",
    )
}

/// The most threads of Python source calls (`plan`) alive in a process: the GIL runs one at a time, so
/// more would only hold memory. A call beyond the bound waits for a permit, cancellably.
pub const MAX_PYTHON_SOURCE_CALLS: usize = 32;

/// Runs `call` (user Python that plans a data source's read) on a thread of its own, named
/// `krishiv-pysrc`, and waits for its result like `wait_for_analysis`: ending with the statement's cancel,
/// leaving the executor thread to its other tasks, and holding no engine pool thread (not the executor's
/// blocking pool, not the I/O runtime's). A cancelled call returns its error at once; its thread runs
/// to its end, then drops its result and frees its permit.
///
/// # Errors
/// `call`'s error; `OPERATION_CANCELED` when the action is cancelled first; `KRISHIV_UNSUPPORTED_FEATURE`
/// on a current-thread runtime.
pub fn run_python_source_call<T: Send + 'static>(
    call: impl FnOnce() -> Result<T, KrishivError> + Send + 'static,
) -> Result<T, KrishivError> {
    wait_for_user_code(
        Box::pin(python_source_call(call)),
        "planning a Python data source on a current-thread runtime; run the statement on a \
         multi-thread runtime",
    )
}

/// The async form of [`run_python_source_call`]: the call runs on a `krishiv-pysrc` thread under the
/// same 32-permit bound; the caller awaits it (and races its own cancel). Dropping the future leaves
/// the thread to finish and free its permit.
///
/// # Errors
/// `call`'s error; `KRISHIV_INTERNAL_ERROR` when the thread does not start.
pub async fn python_source_call<T: Send + 'static>(
    call: impl FnOnce() -> Result<T, KrishivError> + Send + 'static,
) -> Result<T, KrishivError> {
    use std::sync::OnceLock;
    use tokio::sync::{Semaphore, oneshot};
    static PERMITS: OnceLock<Arc<Semaphore>> = OnceLock::new();
    let permits =
        Arc::clone(PERMITS.get_or_init(|| Arc::new(Semaphore::new(MAX_PYTHON_SOURCE_CALLS))));
    let permit = permits
        .acquire_owned()
        .await
        .map_err(|_| internal_error("the Python source call limit closed"))?;
    let (tx, rx) = oneshot::channel();
    std::thread::Builder::new()
        .name("krishiv-pysrc".to_owned())
        .spawn(move || {
            let result = call();
            // The permit goes back when the user code returned, not when the caller gave up.
            drop(permit);
            let _ = tx.send(result);
        })
        .map_err(|e| internal_error(&format!("a Python source call thread did not start: {e}")))?;
    rx.await
        .map_err(|_| internal_error("a Python source call ended without an answer"))?
}

fn internal_error(message: &str) -> KrishivError {
    krishiv_common::error::InternalError {
        message: message.to_owned(),
    }
    .into()
}

fn wait_for_user_code<T>(
    work: futures::future::BoxFuture<'static, Result<T, KrishivError>>,
    current_thread_refusal: &str,
) -> Result<T, KrishivError> {
    if tokio::runtime::Handle::try_current()
        .is_ok_and(|h| h.runtime_flavor() == tokio::runtime::RuntimeFlavor::CurrentThread)
    {
        return Err(krishiv_common::error::KrishivUnsupportedFeature {
            feature: current_thread_refusal.to_owned(),
        }
        .into());
    }
    let cancelled = Box::pin(crate::action::cancelled());
    let both = async move {
        match futures::future::select(work, cancelled).await {
            futures::future::Either::Left((done, _)) => done,
            futures::future::Either::Right((reason, _)) => {
                Err(crate::action::cancelled_error(reason))
            }
        }
    };
    match tokio::runtime::Handle::try_current() {
        Ok(h) if h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(|| futures::executor::block_on(both))
        }
        _ => futures::executor::block_on(both),
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

    /// The rows of a call of the unregistered (inline) Python UDTF `def` at `version` (whose top bit,
    /// [`ANONYMOUS_VERSION_BIT`], must be set) with `args`, lazily: the frame of Spark Connect's
    /// `CommonInlineUserDefinedTableFunction`. Nothing is registered: the call resolves (and runs `analyze`,
    /// asking `Usage` on `System("udf")` each time) as a registered UDTF's does, and the runtime holds its
    /// payload under `(def.name` in lower case`, version)`. A named argument is `col.alias(name)`.
    ///
    /// # Errors
    /// `KRISHIV_INTERNAL_ERROR` when `version` lacks the top bit, `def.eval_type` is not 300, 301 or 302,
    /// the return type is not a struct or the name holds a NUL.
    pub fn python_udtf_frame(
        &self,
        def: &PythonUdtfDef,
        version: u32,
        args: Vec<crate::Column>,
    ) -> Result<crate::DataFrame, KrishivError> {
        let name = format!(
            "session{}{}",
            krishiv_plan::TABLE_FUNCTION_NAME_SEP,
            anonymous_udtf_name(def, version)?
        );
        let plan = krishiv_plan::LogicalPlan::new(krishiv_plan::PlanNode::TableFunction {
            name,
            args: args.into_iter().map(|c| Arc::new(c.expr)).collect(),
        });
        Ok(crate::DataFrame::new(self.clone(), Arc::new(plan)))
    }

    /// The table functions a query of this session resolves.
    pub(crate) fn table_functions(&self) -> Arc<SessionTableFunctions> {
        Arc::new(SessionTableFunctions {
            builtins: krishiv_plan::analyzer::BuiltinTableFunctions::new(),
            library: Arc::clone(&self.state.python_udfs),
            runtime: self
                .state
                .python_runtime
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone(),
            policy: Arc::clone(&self.state.policy),
            principal: Arc::clone(&self.state.principal),
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

/// Where `CREATE FUNCTION ... AS 'python' USING PYTHON 'name'` and `DROP FUNCTION` keep and remove
/// the persisted Python UDFs of a server (P3-T072d): the Connect server's per-tenant function
/// catalog. A session built alone has none, and both statements are refused there.
#[cfg_attr(
    not(feature = "unstable-ir"),
    expect(unreachable_pub, reason = "re-exported only under unstable-ir")
)]
pub trait PersistedFunctions: Send + Sync {
    /// Keeps the session's registered Python UDF `registered` as the stored function `name`
    /// (stored under the registered name: the catalog keys a function by it).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` when the session registered no such UDF; the catalog's failure.
    fn persist<'a>(
        &'a self,
        session: &'a crate::Session,
        registered: &'a str,
    ) -> BoxFuture<'a, Result<(), KrishivError>>;

    /// Whether a function `name` is stored for the session's tenant.
    ///
    /// # Errors
    /// The catalog's failure.
    fn exists<'a>(
        &'a self,
        session: &'a crate::Session,
        name: &'a str,
    ) -> BoxFuture<'a, Result<bool, KrishivError>>;

    /// Removes the stored function `name`.
    ///
    /// # Errors
    /// The catalog's failure; a refusal when another principal stored it.
    fn remove<'a>(
        &'a self,
        session: &'a crate::Session,
        name: &'a str,
    ) -> BoxFuture<'a, Result<(), KrishivError>>;
}

impl crate::Session {
    /// Sets where this session's `CREATE FUNCTION ... USING PYTHON` and `DROP FUNCTION` keep
    /// persisted Python UDFs.
    #[cfg(feature = "unstable-ir")]
    pub fn set_persisted_functions(&self, hook: std::sync::Arc<dyn PersistedFunctions>) {
        *self
            .state
            .persisted_functions
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(hook);
    }

    /// The hook for persisted Python UDFs, if the session has one.
    pub(crate) fn persisted_functions(&self) -> Option<Arc<dyn PersistedFunctions>> {
        self.state
            .persisted_functions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hook over a set of stored names, logging what it was asked to do.
    #[cfg(feature = "unstable-ir")]
    struct Stored(Mutex<Vec<String>>);

    #[cfg(feature = "unstable-ir")]
    impl PersistedFunctions for Stored {
        fn persist<'a>(
            &'a self,
            _: &'a crate::Session,
            registered: &'a str,
        ) -> BoxFuture<'a, Result<(), KrishivError>> {
            Box::pin(async move {
                self.0.lock().unwrap().push(registered.to_owned());
                Ok(())
            })
        }

        fn exists<'a>(
            &'a self,
            _: &'a crate::Session,
            name: &'a str,
        ) -> BoxFuture<'a, Result<bool, KrishivError>> {
            Box::pin(async move { Ok(self.0.lock().unwrap().iter().any(|n| n == name)) })
        }

        fn remove<'a>(
            &'a self,
            _: &'a crate::Session,
            name: &'a str,
        ) -> BoxFuture<'a, Result<(), KrishivError>> {
            Box::pin(async move {
                self.0.lock().unwrap().retain(|n| n != name);
                Ok(())
            })
        }
    }

    /// `CREATE FUNCTION ... USING PYTHON` and `DROP FUNCTION` go through the session's hook: refused without one,
    /// a stored name needs `OR REPLACE` or `IF NOT EXISTS`, and `DROP` removes it.
    #[cfg(feature = "unstable-ir")]
    #[test]
    fn persisted_function_statements_go_through_the_hook() {
        let create = "CREATE FUNCTION f AS 'python' USING PYTHON 'f'";
        let s = crate::Session::new().unwrap();
        let err = futures::executor::block_on(s.sql(create)).unwrap_err();
        assert!(err.to_string().contains("function catalog"), "{err}");

        let hook = Arc::new(Stored(Mutex::new(Vec::new())));
        s.set_persisted_functions(Arc::clone(&hook) as Arc<dyn PersistedFunctions>);
        let run = |q: &str| futures::executor::block_on(s.sql(q));
        run(create).unwrap();
        assert_eq!(*hook.0.lock().unwrap(), ["f"]);
        let err = run(create).unwrap_err();
        assert!(
            err.to_string().contains("ROUTINE_ALREADY_EXISTS"),
            "{err:?}"
        );
        run("CREATE FUNCTION IF NOT EXISTS f AS 'python' USING PYTHON 'f'").unwrap();
        run("CREATE OR REPLACE FUNCTION f AS 'python' USING PYTHON 'f'").unwrap();
        assert_eq!(hook.0.lock().unwrap().len(), 2);
        let err = run("CREATE FUNCTION g AS 'python' USING PYTHON 'f2'").unwrap_err();
        assert!(err.to_string().contains("another name"), "{err}");
        let err = run("CREATE FUNCTION a.g AS 'python' USING PYTHON 'g'").unwrap_err();
        assert!(err.to_string().contains("qualified"), "{err}");

        run("DROP FUNCTION f").unwrap();
        assert!(hook.0.lock().unwrap().is_empty());
        let err = run("DROP FUNCTION f").unwrap_err();
        assert!(err.to_string().contains("ROUTINE_NOT_FOUND"), "{err}");
        // A qualified name never reaches the persisted-function hook.
        let before = hook.0.lock().unwrap().len();
        run("CREATE FUNCTION f AS 'python' USING PYTHON 'f'").unwrap();
        let err = run("DROP FUNCTION other.f").unwrap_err();
        assert!(err.to_string().contains("SCHEMA_NOT_FOUND"), "{err}");
        assert_eq!(hook.0.lock().unwrap().len(), before + 1);
        run("DROP FUNCTION f").unwrap();
        run("DROP FUNCTION IF EXISTS f").unwrap();
    }

    /// An inline UDTF's encoded name carries its definition, return type nullability included; another
    /// version or a scalar kind is no UDTF name.
    #[test]
    fn an_encoded_udtf_name_round_trips_its_definition() {
        let rt = DataType::Struct(krishiv_types::StructType::new([
            krishiv_types::StructField::new("a".to_owned(), DataType::Integer, false),
        ]));
        for return_type in [Some(rt), None] {
            let def = PythonUdtfDef {
                name: "My.F".to_owned(),
                eval_type: PythonEvalType::ArrowTableUdf,
                return_type,
                deterministic: false,
            };
            let name = anonymous_udtf_name(&def, ANONYMOUS_VERSION_BIT | 7).unwrap();
            assert_eq!(
                decode_anonymous_udtf(&name),
                Some((def.clone(), ANONYMOUS_VERSION_BIT | 7))
            );
            assert!(anonymous_udtf_name(&def, 7).is_err());
        }
        let scalar = PythonUdtfDef {
            name: "f".to_owned(),
            eval_type: PythonEvalType::ScalarArrow,
            return_type: None,
            deterministic: true,
        };
        assert!(anonymous_udtf_name(&scalar, ANONYMOUS_VERSION_BIT).is_err());
    }

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
    /// A source call runs on a thread of its own named `krishiv-pysrc` and hands its result back.
    #[test]
    fn a_source_call_runs_on_its_own_named_thread() {
        let name = run_python_source_call(|| Ok(std::thread::current().name().map(str::to_owned)))
            .expect("call");
        assert_eq!(name.as_deref(), Some("krishiv-pysrc"));
    }

    /// The call's error is the caller's; a call that panics is an internal error, not a hang.
    #[test]
    fn a_failed_or_panicking_source_call_reports_an_error() {
        let failed: Result<(), KrishivError> =
            run_python_source_call(|| Err(super::internal_error("boom")));
        assert!(failed.unwrap_err().to_string().contains("boom"));
        let panicked: Result<(), KrishivError> = run_python_source_call(|| panic!("user code"));
        assert!(panicked.is_err());
    }

    /// A cancel ends a call that is still running user code: the caller gets `OPERATION_CANCELED`
    /// at once, the thread runs on to its end and frees its permit then.
    #[test]
    fn a_cancel_ends_a_blocked_source_call_at_once() {
        use crate::action::Actions;
        use crate::cancel::CancelReason;
        use std::sync::mpsc;
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("runtime");
        let actions = Arc::new(Actions::default());
        let guard = actions.begin(1, Some("op"));
        let hold = Arc::new(std::sync::Mutex::new(()));
        let held = hold.lock().expect("lock");
        let (started_tx, started_rx) = mpsc::channel();
        let call_hold = Arc::clone(&hold);
        let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let call_finished = Arc::clone(&finished);
        let canceller = {
            let actions = Arc::clone(&actions);
            std::thread::spawn(move || {
                started_rx.recv().expect("the call started");
                actions.cancel_where(CancelReason::User, &|_, _| true);
            })
        };
        let began = std::time::Instant::now();
        let out: Result<(), KrishivError> = rt.block_on(guard.run(async move {
            // The statement's executor task: the call blocks in `block_in_place`, as at plan time.
            tokio::spawn(async move {
                run_python_source_call(move || {
                    let _ = started_tx.send(());
                    drop(call_hold.lock());
                    call_finished.store(true, std::sync::atomic::Ordering::SeqCst);
                    Ok(())
                })
            })
            .await
            .expect("join")
        }));
        canceller.join().expect("canceller");
        assert!(out.is_err(), "the cancel ends the call");
        // The error is the cancel's own: the user code is still blocked, it has not returned one.
        assert!(
            !finished.load(std::sync::atomic::Ordering::SeqCst),
            "the caller returned while the user code was still running"
        );
        assert!(
            began.elapsed() < std::time::Duration::from_secs(5),
            "the caller did not wait for the user code"
        );
        drop(held);
        // The thread runs on to its end after the cancel.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !finished.load(std::sync::atomic::Ordering::SeqCst) {
            assert!(
                std::time::Instant::now() < deadline,
                "the call never finished"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// Calls beyond the bound wait for a permit: with every permit held by blocked calls, one more
    /// call does not run until a permit returns.
    #[test]
    fn calls_beyond_the_bound_wait() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::mpsc;
        let release = Arc::new(std::sync::Mutex::new(()));
        let guard = release.lock().expect("lock");
        let mut running = Vec::new();
        for _ in 0..MAX_PYTHON_SOURCE_CALLS {
            let release = Arc::clone(&release);
            let (started_tx, started_rx) = mpsc::channel();
            running.push((
                std::thread::spawn(move || {
                    run_python_source_call(move || {
                        let _ = started_tx.send(());
                        drop(release.lock());
                        Ok(())
                    })
                }),
                started_rx,
            ));
        }
        for (_, started) in &running {
            started.recv().expect("started");
        }
        let ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        let extra = std::thread::spawn(move || {
            run_python_source_call(move || {
                flag.store(true, Ordering::SeqCst);
                Ok(())
            })
        });
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(
            !ran.load(Ordering::SeqCst),
            "the 33rd call ran past the bound"
        );
        drop(guard);
        extra.join().expect("join").expect("extra call");
        assert!(ran.load(Ordering::SeqCst));
        for (handle, _) in running {
            handle.join().expect("join").expect("call");
        }
    }
}
