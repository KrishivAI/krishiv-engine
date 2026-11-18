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
use krishiv_connect::translate::{PythonUdf, UdfCallName, UdfHost as TranslateHost};
use krishiv_exec::python_eval::UdfRuntime;
use krishiv_plan::PythonUdfRef;
use krishiv_session::{ANONYMOUS_VERSION_BIT, PythonUdfDef, Session};

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
    /// Registers `udf`'s payload as `(name, version)`, the identity a plan's call names.
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

    /// An archive's `#fragment` is the pool's alias: the directory it is unpacked into.
    fn set_artifacts(&self, pyfiles: Vec<PathBuf>, archives: Vec<PythonArchive>) {
        krishiv_udf_pool::SessionUdfs::set_artifacts(
            self,
            pyfiles,
            archives
                .into_iter()
                .map(|archive| krishiv_udf_pool::PythonArchive {
                    path: archive.path,
                    alias: archive.fragment,
                })
                .collect(),
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
}

impl<'a> InlineUdfs<'a> {
    /// The host of a plan that will run.
    pub(crate) fn execution(udfs: &'a SessionPython) -> Self {
        Self {
            udfs,
            analysis: false,
            seen: false,
            pending: Vec::new(),
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
}

#[cfg(test)]
mod tests;
