// SPDX-License-Identifier: Apache-2.0
//! Python UDFs over Spark Connect (P3-T014, PR-26d part 1; design `P3-T014-connect-udfs-sandbox.md` §3.2,
//! ADR 0046 §3b and N1): the server's one worker pool ([`UdfHost`], `krishiv_udf_pool::Pool` on Linux) and
//! each Connect session's Python UDFs (`SessionPython`).
//!
//! **Client code runs only after its privilege check.** A registration (`register_function`) is refused
//! by its command class (`auth::check_command`: `Usage` on `System("udf")`) before it runs (it is parsed, never
//! registered); a plan with an inline UDF is
//! checked the same way after translation and before anything is registered or run
//! (`InlineUdfs::grant`). A session gets its UDF runtime only at its first passed check
//! (`SessionPython::grant`): a session that never passed one has no runtime at all, so no plan of it can
//! start a worker, whatever it names (a call then fails with `KRISHIV_UDF_NOT_AVAILABLE`). Workers serve one
//! owner (the session's principal and server-side id); `SessionPython::release` stops them when the
//! session is released, swept or closed.
//!
//! **Inline UDFs** (`common_inline_user_defined_function` in a plan) become calls of the session's encoded
//! name of an unregistered UDF (`krishiv_session::python_udf_call_name`, top-bit versions), as the
//! embedded `udf(...)` path does. A payload is registered once per session: the same UDF sent again (a
//! DataFrame re-run) is found by the digest of its definition and keeps its version, so a long session
//! holds one payload per distinct UDF. `AnalyzePlan` translates them with `InlineUdfs::analysis`: the plan
//! is typed, nothing is registered and no privilege is needed (a scalar UDF runs no code while a plan is
//! analyzed, ADR 0046 §3b).
//!
//! **Relation UDFs** (`mapInPandas`, `applyInPandas`, cogroups and their Arrow forms) take a version the same
//! way and the same privilege check (`InlineUdfs::seen`), and run on the session's workers through the
//! pool's `prepare_relation` (`tests/udf.rs`: map, grouped and cogrouped).
//!
//! **Clones** (`CloneSession`, PR-15) must not share a runtime: its owner is the source session.
//! `SessionPython::copy_for` builds the clone's own runtime with the payloads the source registered.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_connect::translate::{PythonUdf, PythonUdtf, UdfCallName, UdfHost as TranslateHost};
use krishiv_exec::python_eval::UdfRuntime;
use krishiv_plan::PythonUdfRef;
use krishiv_session::{ANONYMOUS_VERSION_BIT, PythonUdfDef, PythonUdtfDef, Session};
use krishiv_types::DataType;

use crate::artifacts::SessionArtifacts;
use sha2::{Digest, Sha256};

/// The session conf a worker gets with each batch (Spark's runner conf; the keys the embedded runtime
/// sends, `krishiv-python`'s `RUNNER_KEYS`), read from the session's live conf when a query's calls are
/// prepared, so each query sends the values current when it starts, as Spark sends them per execution.
const RUNNER_KEYS: [&str; 8] = [
    "spark.sql.legacy.execution.pandas.groupedMap.assignColumnsByName",
    "spark.sql.execution.arrow.useLargeVarTypes",
    "spark.sql.execution.pandas.convertToArrowArraySafely",
    "spark.sql.execution.pythonUDF.pandas.intToDecimalCoercionEnabled",
    "spark.sql.execution.pyspark.binaryAsBytes",
    "spark.sql.session.timeZone",
    "spark.sql.execution.pythonUDF.pandas.preferIntExtensionDtype",
    "spark.sql.execution.pythonUDF.arrow.concurrency.level",
];

/// A session's Python artifacts, as [`SessionUdfRuntime::set_artifacts`] takes them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PythonArtifacts {
    /// The installed `pyfiles/` files.
    pub(crate) pyfiles: Vec<PathBuf>,
    /// The installed `archives/` files.
    pub(crate) archives: Vec<PythonArchive>,
}

/// An installed `archives/` artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonArchive {
    /// The archive file: absolute, no link at or above it.
    pub path: PathBuf,
    /// The name's `#fragment`: Spark unpacks the archive into a directory of that name (the file's
    /// name without one).
    pub fragment: Option<String>,
}

/// Where a session runtime reads its runner conf when a call is prepared.
pub type RunnerConf = Arc<dyn Fn() -> BTreeMap<String, String> + Send + Sync>;

/// The server's Python UDF workers: one per server, shared by every Connect session.
pub trait UdfHost: Send + Sync + fmt::Debug {
    /// A new, empty runtime for session `session` of `principal`: its workers serve that owner only. Each
    /// prepared call reads its runner conf from `conf`.
    fn session_runtime(
        &self,
        principal: &str,
        session: &str,
        conf: RunnerConf,
    ) -> Arc<dyn SessionUdfRuntime>;
}

/// One session's UDF runtime on the server's workers: what the engine calls, plus its payload store.
pub trait SessionUdfRuntime: UdfRuntime {
    /// Registers `udf`'s payload as `(name, version)`, the identity a plan's call names. A Python UDTF's
    /// payload comes the same way ([`udtf_payload`]): its eval type (300, 301, 302), class and Python
    /// version; `name` is then the lower-case name the plan's `PythonUdfRef` carries.
    fn register(&self, name: &str, version: u32, udf: &PythonUdf);
    /// Stops the session's workers (idle ones now, busy ones when their call ends).
    fn release(&self);
    /// The session's Python artifacts, replacing the whole set: the installed `pyfiles/` and
    /// `archives/` files (absolute paths with no link in them, in the order installed; each archive
    /// with its `#fragment`). Calls prepared afterwards see them; workers started for an earlier set are
    /// not reused. Called on an async worker under a session lock: it must not block (no file I/O; a
    /// runtime checks the paths when a call is prepared). Never called after [`Self::release`]. The
    /// default ignores them (a runtime that runs no client files).
    fn set_artifacts(&self, pyfiles: Vec<PathBuf>, archives: Vec<PythonArchive>) {
        let _ = (pyfiles, archives);
    }
}

#[cfg(target_os = "linux")]
impl UdfHost for krishiv_udf_pool::Pool {
    fn session_runtime(
        &self,
        principal: &str,
        session: &str,
        conf: RunnerConf,
    ) -> Arc<dyn SessionUdfRuntime> {
        Arc::new(
            krishiv_udf_pool::SessionUdfs::new(
                self.clone(),
                krishiv_udf_pool::Owner {
                    principal: principal.to_owned(),
                    session: session.to_owned(),
                },
            )
            .with_conf_source(conf),
        )
    }
}

#[cfg(target_os = "linux")]
impl SessionUdfRuntime for krishiv_udf_pool::SessionUdfs {
    fn register(&self, name: &str, version: u32, udf: &PythonUdf) {
        krishiv_udf_pool::SessionUdfs::register(
            self,
            name,
            version,
            krishiv_udf_pool::PythonPayload {
                command: Arc::from(udf.command.as_slice()),
                eval_type: udf.eval_type,
                python_ver: udf.python_ver.clone(),
            },
        );
    }

    fn release(&self) {
        krishiv_udf_pool::SessionUdfs::release(self);
    }
}

/// A Connect session's Python UDFs.
pub(crate) struct SessionPython {
    host: Option<Arc<dyn UdfHost>>,
    /// `None` when the server runs no worker pool: every grant is refused.
    runtime: Option<Arc<dyn SessionUdfRuntime>>,
    state: Mutex<State>,
    /// The last artifact set handed to the runtime; held while a set is read and handed over.
    artifacts: Mutex<Synced>,
}

/// What [`SessionPython::sync_artifacts`] last did.
#[derive(Default)]
struct Synced {
    /// The set handed over last (none yet: an empty set is not handed over).
    last: PythonArtifacts,
    /// The session was released: no set is handed over any more.
    released: bool,
}

#[derive(Default, Clone)]
struct State {
    /// The runtime is the session's (set at the first passed privilege check).
    installed: bool,
    /// Every payload registered, for a clone's runtime.
    payloads: Vec<(String, u32, Arc<PythonUdf>)>,
    /// The version of each unregistered UDF (inline or a relation's) registered, by the digest of its
    /// definition.
    anonymous: HashMap<[u8; 32], u32>,
    /// The last unregistered version handed out (without the top bit).
    last_anonymous: u32,
    /// Each inline UDTF registered in the session's library, by the digest of its definition.
    udtfs: HashMap<[u8; 32], InlineUdtf>,
}

/// An inline UDTF of the session.
#[derive(Clone)]
struct InlineUdtf {
    /// Its hidden name, lower case (the library's key).
    key: String,
    /// Its registration version in the session's library.
    version: u32,
    /// Its payload is registered with the session's workers (only after a passed privilege check).
    payload: bool,
}

impl fmt::Debug for SessionPython {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionPython")
            .field("pool", &self.runtime.is_some())
            .finish_non_exhaustive()
    }
}

impl SessionPython {
    /// The UDFs of `session`, built for `principal`, on `host`'s workers (none: UDFs are refused).
    pub(crate) fn new(host: Option<&Arc<dyn UdfHost>>, principal: &str, session: &Session) -> Self {
        Self {
            host: host.map(Arc::clone),
            runtime: host.map(|h| h.session_runtime(principal, session.id(), runner_conf(session))),
            state: Mutex::default(),
            artifacts: Mutex::default(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The runtime of `session`, installed on it at the first call. Only after the session's principal
    /// passed the `Usage` check on `System("udf")`: this is what lets the session run client code.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` when the server runs no UDF worker pool.
    fn grant(&self, session: &Session) -> Result<&Arc<dyn SessionUdfRuntime>, KrishivError> {
        let Some(runtime) = &self.runtime else {
            return Err(KrishivUnsupportedFeature {
                feature: "Python UDFs on a Spark Connect server without a UDF worker pool"
                    .to_owned(),
            }
            .into());
        };
        let mut state = self.lock();
        if !state.installed {
            session.set_python_udf_runtime(Arc::clone(runtime) as Arc<dyn UdfRuntime>);
            state.installed = true;
        }
        Ok(runtime)
    }

    /// Runs `register_function` for a Python UDF, after its privilege check passed: registers `udf` in
    /// `session`'s registry (its next version) and its payload with the session's workers.
    ///
    /// # Errors
    /// [`Self::grant`]'s.
    pub(crate) fn register(&self, session: &Session, udf: PythonUdf) -> Result<(), KrishivError> {
        let runtime = self.grant(session)?;
        let entry = session.register_python_udf(def_of(&udf));
        runtime.register(&entry.def.name, entry.version, &udf);
        self.lock()
            .payloads
            .push((entry.def.name, entry.version, Arc::new(udf)));
        Ok(())
    }

    /// Runs `register_table_function` for a Python UDTF, after its privilege check passed: registers
    /// `udtf` in `session`'s library (its next version; SQL's `FROM name(..)` resolves it) and its payload
    /// with the session's workers under the same name and version.
    ///
    /// # Errors
    /// [`Self::grant`]'s, and the session's refusal of the definition.
    pub(crate) fn register_udtf(
        &self,
        session: &Session,
        udtf: &PythonUdtf,
    ) -> Result<(), KrishivError> {
        let runtime = self.grant(session)?;
        let entry = session.register_python_udtf(udtf_def(udtf, udtf.name.clone()))?;
        let name = entry.def.name.to_lowercase();
        let payload = udtf_payload(udtf);
        runtime.register(&name, entry.version, &payload);
        let mut state = self.lock();
        // A registration under an inline UDTF's hidden name replaced that library entry: the
        // cached version is no longer the library's, so the next inline call registers its own
        // definition again (a new version) instead of resolving to this one.
        state.udtfs.retain(|_, u| u.key != name);
        state
            .payloads
            .push((name, entry.version, Arc::new(payload)));
        Ok(())
    }

    /// Registers the inline UDTF `udtf` under its hidden `name` in `session`'s library, once per
    /// definition (`digest`), and its payload with `runtime`'s workers when given (and not done yet).
    fn inline_udtf(
        &self,
        session: &Session,
        digest: [u8; 32],
        name: &str,
        udtf: &PythonUdtf,
        runtime: Option<&Arc<dyn SessionUdfRuntime>>,
    ) -> Result<(), KrishivError> {
        let mut state = self.lock();
        // The cached entry is the library's current one: every registration of this session goes
        // through `SessionPython` (`register_udtf` drops an entry whose hidden name it takes), so a
        // cached version that is gone means the library's entry changed and is registered again.
        let mut known = match state.udtfs.get(&digest) {
            Some(known) => known.clone(),
            None => {
                let entry = session.register_python_udtf(udtf_def(udtf, name.to_owned()))?;
                InlineUdtf {
                    key: name.to_lowercase(),
                    version: entry.version,
                    payload: false,
                }
            }
        };
        if let Some(runtime) = runtime
            && !known.payload
        {
            let payload = udtf_payload(udtf);
            let name = name.to_lowercase();
            runtime.register(&name, known.version, &payload);
            state
                .payloads
                .push((name, known.version, Arc::new(payload)));
            known.payload = true;
        }
        state.udtfs.insert(digest, known);
        Ok(())
    }

    /// Hands the session's Python artifacts, read from `artifacts` now, to its workers
    /// (`AddArtifacts`, after every transfer, whether it succeeded or not: one that fails can still
    /// have installed some; `CloneSession`, after the copy): the whole set, which replaces the last
    /// one, when it changed. Read and handed over under one lock, so two transfers ending together
    /// cannot leave the older set in place, and never after [`Self::release`]. Also before the first
    /// granted call: `AddArtifacts` passed the same privilege check (`Usage` on `System("udf")`), a
    /// clone's files are its source's, which passed it, and the runtime runs nothing until a plan of
    /// the session is granted.
    pub(crate) fn sync_artifacts(&self, artifacts: &SessionArtifacts) {
        if let Some(runtime) = &self.runtime {
            let mut synced = self
                .artifacts
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let set = artifacts.python_artifacts();
            if synced.released || set == synced.last {
                return;
            }
            synced.last = set.clone();
            runtime.set_artifacts(set.pyfiles, set.archives);
        }
    }

    /// Stops the session's workers; no artifact set is handed over afterwards.
    pub(crate) fn release(&self) {
        self.artifacts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .released = true;
        if let Some(runtime) = &self.runtime {
            runtime.release();
        }
    }

    /// The UDFs of `clone`, a copy of this session built for `principal` (`CloneSession`): a runtime of
    /// its own (its workers serve the clone only) holding every payload this session registered, so the
    /// registrations the clone copied keep running; installed on the clone when it was on this session.
    pub(crate) fn copy_for(&self, principal: &str, clone: &Session) -> Self {
        let copy = Self::new(self.host.as_ref(), principal, clone);
        let state = self.lock().clone();
        if let Some(runtime) = &copy.runtime {
            for (name, version, udf) in &state.payloads {
                runtime.register(name, *version, udf);
            }
            if state.installed {
                clone.set_python_udf_runtime(Arc::clone(runtime) as Arc<dyn UdfRuntime>);
            }
        }
        *copy.lock() = state;
        copy
    }
}

/// The runner conf of `session`, read live: the conf store, not the session (which holds the runtime that
/// holds this, so a session handle here would never be freed).
fn runner_conf(session: &Session) -> RunnerConf {
    let conf = session.conf();
    Arc::new(move || {
        RUNNER_KEYS
            .iter()
            .filter_map(|k| conf.get(k).map(|v| ((*k).to_owned(), v)))
            .collect()
    })
}

/// The registry definition of `udf`. Calls on pool workers may overlap: each takes a worker of its own.
fn def_of(udf: &PythonUdf) -> PythonUdfDef {
    PythonUdfDef {
        name: udf.name.clone(),
        eval_type: udf.plan_eval_type,
        return_type: udf.return_type.clone(),
        deterministic: udf.deterministic,
        thread_safe: true,
    }
}

/// The library definition of `udtf` under `name`.
fn udtf_def(udtf: &PythonUdtf, name: String) -> PythonUdtfDef {
    PythonUdtfDef {
        name,
        eval_type: udtf.plan_eval_type,
        return_type: udtf.return_type.clone(),
        deterministic: udtf.deterministic,
    }
}

/// `udtf`'s payload as [`SessionUdfRuntime::register`] takes it: the eval type as sent, the class, the
/// Python version (the return type is the library's; `Null` stands for none).
fn udtf_payload(udtf: &PythonUdtf) -> PythonUdf {
    PythonUdf {
        name: udtf.name.clone(),
        deterministic: udtf.deterministic,
        eval_type: udtf.eval_type,
        plan_eval_type: udtf.plan_eval_type,
        return_type: udtf.return_type.clone().unwrap_or(DataType::Null),
        command: udtf.command.clone(),
        python_ver: udtf.python_ver.clone(),
    }
}

/// The digest of everything that tells two inline UDTFs apart.
fn udtf_digest(udtf: &PythonUdtf) -> [u8; 32] {
    let mut h = Sha256::new();
    let return_type = udtf.return_type.as_ref().map(DataType::sql);
    for part in [
        b"udtf".as_slice(),
        udtf.name.as_bytes(),
        udtf.python_ver.as_bytes(),
        return_type.as_deref().unwrap_or("\0").as_bytes(),
        &[u8::from(return_type.is_some())],
        &udtf.eval_type.to_le_bytes(),
        &[u8::from(udtf.deterministic)],
        &udtf.command,
    ] {
        h.update((part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().into()
}

/// The hidden name an inline UDTF of digest `digest` is registered under in its session: the client's
/// name, then `#` and 128 bits of the digest in hex, so each distinct definition has a name of its own.
fn udtf_name(name: &str, digest: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    let mut out = format!("{name}#");
    for b in &digest[..16] {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// The digest of everything that tells two unregistered UDFs apart.
fn digest(udf: &PythonUdf) -> [u8; 32] {
    let mut h = Sha256::new();
    for part in [
        udf.name.as_bytes(),
        udf.python_ver.as_bytes(),
        udf.return_type.sql().as_bytes(),
        &udf.eval_type.to_le_bytes(),
        &[u8::from(udf.deterministic)],
        &udf.command,
    ] {
        h.update((part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().into()
}

/// The translator's host for one plan of a session ([`TranslateHost`]).
pub(crate) struct InlineUdfs<'a> {
    udfs: &'a SessionPython,
    /// Analysis: names only, nothing registered.
    analysis: bool,
    /// Whether the plan holds a Python UDF (an inline call or a relation's).
    seen: bool,
    /// UDFs new to the session, registered by [`Self::grant`].
    pending: Vec<([u8; 32], u32, PythonUdf)>,
    /// The plan's inline UDTFs (digest, hidden name, definition), registered by [`Self::grant`] or
    /// [`Self::prepare_analysis`].
    udtfs: Vec<([u8; 32], String, PythonUdtf)>,
}

impl<'a> InlineUdfs<'a> {
    /// The host of a plan that will run.
    pub(crate) fn execution(udfs: &'a SessionPython) -> Self {
        Self {
            udfs,
            analysis: false,
            seen: false,
            pending: Vec::new(),
            udtfs: Vec::new(),
        }
    }

    /// The host of a plan that is only analyzed.
    pub(crate) fn analysis(udfs: &'a SessionPython) -> Self {
        Self {
            analysis: true,
            ..Self::execution(udfs)
        }
    }

    /// Whether the translated plan calls a Python UDF, inline or in a relation (it needs `Usage` on
    /// `System("udf")`).
    pub(crate) fn seen(&self) -> bool {
        self.seen
    }

    /// Registers the plan's new UDFs with the session's workers. Call only after the plan's privilege
    /// check passed.
    ///
    /// # Errors
    /// [`SessionPython::grant`]'s.
    pub(crate) fn grant(self, session: &Session) -> Result<(), KrishivError> {
        if !self.seen || self.analysis {
            return Ok(());
        }
        let runtime = self.udfs.grant(session)?;
        for (digest, name, udtf) in &self.udtfs {
            self.udfs
                .inline_udtf(session, *digest, name, udtf, Some(runtime))?;
        }
        for (digest, version, udf) in self.pending {
            runtime.register(&udf.name, version, &udf);
            let mut state = self.udfs.lock();
            state.anonymous.insert(digest, version);
            state
                .payloads
                .push((udf.name.clone(), version, Arc::new(udf)));
        }
        Ok(())
    }
}

impl InlineUdfs<'_> {
    /// Whether analyzing the plan runs client code: an inline UDTF without a return type, whose
    /// `analyze` the analyzer calls (ADR 0046 §3b: `Usage` on `System("udf")` before it, then
    /// [`Self::prepare_analysis`] with `granted`).
    pub(crate) fn analysis_runs_code(&self) -> bool {
        self.udtfs.iter().any(|(_, _, u)| u.return_type.is_none())
    }

    /// Makes the inline UDTFs of a plan that is only analyzed resolvable: each is registered in the
    /// session's library under its hidden name (no code runs: its declared result types the call). With
    /// `granted` (the caller passed the `Usage` check because [`Self::analysis_runs_code`]), their payloads
    /// also go to the session's workers, so `analyze` can run.
    ///
    /// # Errors
    /// [`SessionPython::grant`]'s when `granted`, and the session's refusal of a definition.
    pub(crate) fn prepare_analysis(
        &self,
        session: &Session,
        granted: bool,
    ) -> Result<(), KrishivError> {
        let runtime = if granted && !self.udtfs.is_empty() {
            Some(self.udfs.grant(session)?)
        } else {
            None
        };
        for (digest, name, udtf) in &self.udtfs {
            self.udfs
                .inline_udtf(session, *digest, name, udtf, runtime)?;
        }
        Ok(())
    }

    /// The version `udf` runs under in this session: the one it was registered with before, or a new
    /// top-bit version it is registered under once the plan's privilege check passed ([`Self::grant`]).
    /// Marks the plan as calling client code.
    fn version_of(&mut self, udf: &PythonUdf) -> Result<u32, KrishivError> {
        if udf.name.contains('\0') {
            return Err(KrishivUnsupportedFeature {
                feature: "Python UDF names holding a NUL character".to_owned(),
            }
            .into());
        }
        self.seen = true;
        let key = digest(udf);
        if self.analysis {
            // Typed, never run or registered: a version from the definition's digest, so the plans of two
            // UDFs that differ only in their function never compare or hash equal (Spark's `PythonUDF`
            // compares its `command` bytes).
            let [a, b, c, d, ..] = key;
            return Ok(u32::from_le_bytes([a, b, c, d]) | ANONYMOUS_VERSION_BIT);
        }
        let mut state = self.udfs.lock();
        if let Some(v) = state.anonymous.get(&key) {
            return Ok(*v);
        }
        if let Some(&(_, v, _)) = self.pending.iter().find(|(d, _, _)| *d == key) {
            return Ok(v);
        }
        let n = state.last_anonymous.wrapping_add(1);
        if n & ANONYMOUS_VERSION_BIT != 0 {
            return Err(KrishivUnsupportedFeature {
                feature: "more than 2^31 - 1 distinct unregistered Python UDFs (inline or in a relation) in one session".to_owned(),
            }
            .into());
        }
        state.last_anonymous = n;
        drop(state);
        let v = n | ANONYMOUS_VERSION_BIT;
        self.pending.push((key, v, udf.clone()));
        Ok(v)
    }
}

impl TranslateHost for InlineUdfs<'_> {
    fn inline_python_udf(&mut self, udf: &PythonUdf) -> Result<UdfCallName, KrishivError> {
        let version = self.version_of(udf)?;
        Ok(UdfCallName {
            qualifier: vec!["session".to_owned()],
            name: krishiv_session::python_udf_call_name(&def_of(udf), version)?,
        })
    }

    /// A relation's UDF takes a version as an inline call's does, under the same privilege check (the plan
    /// is marked as calling client code, and nothing is registered before [`Self::grant`]); it runs on the
    /// session's workers like an inline call.
    fn relation_python_udf(&mut self, udf: &PythonUdf) -> Result<PythonUdfRef, KrishivError> {
        let version = self.version_of(udf)?;
        Ok(PythonUdfRef {
            name: udf.name.clone(),
            version,
            eval_type: udf.plan_eval_type,
            thread_safe: true,
            deterministic: udf.deterministic,
        })
    }

    /// An inline UDTF goes by a hidden name of the session, one per distinct definition (by digest),
    /// registered once the plan's privilege check passed ([`Self::grant`]), or for analysis by
    /// [`Self::prepare_analysis`]. Marks the plan as calling client code.
    fn inline_python_udtf(&mut self, udtf: &PythonUdtf) -> Result<String, KrishivError> {
        if udtf.name.contains('\0') {
            return Err(KrishivUnsupportedFeature {
                feature: "Python UDTF names holding a NUL character".to_owned(),
            }
            .into());
        }
        self.seen = true;
        let digest = udtf_digest(udtf);
        let name = udtf_name(&udtf.name, &digest);
        if !self.udtfs.iter().any(|(d, _, _)| *d == digest) {
            self.udtfs.push((digest, name.clone(), udtf.clone()));
        }
        Ok(name)
    }
}

#[cfg(test)]
mod tests;
