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

use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
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
    /// A new, empty runtime for session `session` of `principal` (of `tenant`, in a multi-tenant server): its
    /// workers serve that owner only (`authenticated`: the server tells principals apart, so the pool caps each), and a tenant's workers and calls are bounded together. Each
    /// prepared call reads its runner conf from `conf`.
    fn session_runtime(
        &self,
        principal: &str,
        tenant: Option<&str>,
        authenticated: bool,
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
        tenant: Option<&str>,
        authenticated: bool,
        session: &str,
        conf: RunnerConf,
    ) -> Arc<dyn SessionUdfRuntime> {
        Arc::new(
            krishiv_udf_pool::SessionUdfs::new(
                self.clone(),
                krishiv_udf_pool::Owner {
                    principal: principal.to_owned(),
                    session: session.to_owned(),
                    tenant: tenant.map(str::to_owned),
                    capped: authenticated,
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

/// The server's catalog of persisted Python UDFs (P3-T072c): what a client keeps with the `PersistPythonFunction`
/// command, per tenant (`None`: the server's one tenant), and what later sessions of the tenant find. The calls
/// may block (a file); the server runs them on a blocking thread.
pub trait FunctionStore: Send + Sync + fmt::Debug {
    /// Every function stored for `tenant`.
    ///
    /// # Errors
    /// The store's failure.
    fn list(&self, tenant: Option<&str>) -> Result<Vec<PythonUdf>, KrishivError>;
    /// Stores `udf` for `tenant`, replacing one of the same name; `owner` is the principal that persisted it.
    /// A function another principal stored is replaced only with `override_owner`, and the store checks that
    /// atomically with the write: [`StoreChange::OwnedByOther`] changes nothing.
    ///
    /// # Errors
    /// The store's failure.
    fn put(
        &self,
        tenant: Option<&str>,
        owner: &str,
        udf: &PythonUdf,
        override_owner: bool,
    ) -> Result<StoreChange, KrishivError>;
    /// The principal that stored `tenant`'s function `name`, if there is one. Only a hint that lets `persist`
    /// check `Manage` before the first write; the safety is the atomic check in `put` and `remove`.
    ///
    /// # Errors
    /// The store's failure.
    fn owner(&self, tenant: Option<&str>, name: &str) -> Result<Option<String>, KrishivError>;
    /// Removes `tenant`'s function `name` for `caller`; another principal's only with `override_owner`
    /// (checked atomically with the removal).
    ///
    /// # Errors
    /// The store's failure.
    fn remove(
        &self,
        tenant: Option<&str>,
        name: &str,
        caller: &str,
        override_owner: bool,
    ) -> Result<StoreChange, KrishivError>;
}

/// What a [`FunctionStore`] change did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreChange {
    /// Written or removed.
    Done,
    /// There was nothing to remove.
    Absent,
    /// Another principal owns the function and the caller may not override it; nothing changed.
    OwnedByOther,
}

/// `Manage` on the cluster: what changing another principal's stored function takes (a refusal is audited).
async fn authorize_manage(session: &Session) -> Result<(), KrishivError> {
    session
        .authorize_privilege(
            &krishiv_session::Securable::System("cluster".to_owned()),
            krishiv_session::Privilege::Manage,
            krishiv_session::StatementClass::Ddl,
        )
        .await
}

/// A Connect session's Python UDFs.
pub(crate) struct SessionPython {
    /// The persisted-function catalog, if the server has one.
    store: Option<Arc<dyn FunctionStore>>,
    /// The session's principal and tenant, for the catalog.
    principal: String,
    tenant: Option<String>,
    /// The server tells principals apart (authentication is on).
    authenticated: bool,
    /// Held while the tenant's persisted functions are listed and registered, so every first request
    /// of the session waits for the one load.
    loading: tokio::sync::Mutex<()>,
    /// The tenant's persisted functions were loaded (or none is to be).
    loaded: std::sync::atomic::AtomicBool,
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
    pub(crate) fn new(
        host: Option<&Arc<dyn UdfHost>>,
        principal: &str,
        tenant: Option<&str>,
        authenticated: bool,
        session: &Session,
    ) -> Self {
        Self {
            store: None,
            principal: principal.to_owned(),
            tenant: tenant.map(str::to_owned),
            authenticated,
            loading: tokio::sync::Mutex::new(()),
            // No store yet: nothing to load (`with_store` changes that).
            loaded: std::sync::atomic::AtomicBool::new(true),
            host: host.map(Arc::clone),
            runtime: host.map(|h| {
                h.session_runtime(
                    principal,
                    tenant,
                    authenticated,
                    session.id(),
                    runner_conf(session),
                )
            }),
            state: Mutex::default(),
            artifacts: Mutex::default(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The session with the server's persisted-function catalog (none: nothing is persisted or loaded).
    pub(crate) fn with_store(mut self, store: Option<&Arc<dyn FunctionStore>>) -> Self {
        self.store = store.map(Arc::clone);
        *self.loaded.get_mut() = self.store.is_none();
        self
    }

    /// Whether the tenant's persisted functions are still to be loaded into this session.
    pub(crate) fn load_pending(&self) -> bool {
        !self.loaded.load(std::sync::atomic::Ordering::Acquire)
    }

    /// The principal may not run client code: nothing is loaded into this session.
    pub(crate) fn skip_load(&self) {
        self.loaded
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Loads the tenant's persisted functions once, as `register_function` of each would. Call only after the
    /// session's principal passed the `Usage` check on `System("udf")`. A catalog failure is logged and the
    /// load is tried again at the next call; a stored function that no longer converts is skipped. A name the
    /// session registered itself is kept.
    pub(crate) async fn load_persisted(&self, session: &Session) {
        let Some(store) = self.store.clone() else {
            return;
        };
        // One load at a time; a request that waited finds it done.
        let _loading = self.loading.lock().await;
        if !self.load_pending() {
            return;
        }
        let key = self.store_key();
        let listed = tokio::task::spawn_blocking(move || store.list(key.as_deref())).await;
        let udfs = match listed {
            Ok(Ok(udfs)) => udfs,
            Ok(Err(e)) => {
                tracing::warn!(target: "krishiv::udf", error = %e, "the persisted Python UDFs could not be read");
                return;
            }
            Err(_) => return,
        };
        for udf in udfs {
            let held = self
                .lock()
                .payloads
                .iter()
                // Only a registered function holds the name: an inline UDF (top-bit version) is not in the registry.
                .any(|(name, v, _)| {
                    v & ANONYMOUS_VERSION_BIT == 0 && name.to_lowercase() == udf.name.to_lowercase()
                });
            if held {
                continue;
            }
            if let Err(e) = self.register(session, udf) {
                tracing::warn!(target: "krishiv::udf", error = %e, "a persisted Python UDF was not loaded");
            }
        }
        self.loaded
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// The catalog key of this session's functions: its tenant, else (when the server tells principals apart)
    /// its principal, so one untenanted principal never reads another's; the auth-off local principal has
    /// the one empty key.
    fn store_key(&self) -> Option<String> {
        match (&self.tenant, self.authenticated) {
            (Some(t), _) => Some(format!("t:{t}")),
            (None, true) => Some(format!("p:{}", self.principal)),
            (None, false) => None,
        }
    }

    /// The principal that stored the tenant's function `name`; `None` when there is none or the server has no
    /// function catalog.
    ///
    /// # Errors
    /// The catalog's failure.
    pub(crate) async fn stored_owner(&self, name: &str) -> Result<Option<String>, KrishivError> {
        let Some(store) = self.store.clone() else {
            return Ok(None);
        };
        let (key, name) = (self.store_key(), name.to_owned());
        blocking(move || store.owner(key.as_deref(), &name)).await
    }

    /// `PersistPythonFunction`: keeps the session's registered UDF `name` for the tenant, or removes the tenant's
    /// stored one (`drop`).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` when the server has no function catalog or the session registered no such
    /// UDF; the catalog's failure.
    pub(crate) async fn persist(
        &self,
        session: &Session,
        name: &str,
        drop: bool,
    ) -> Result<(), KrishivError> {
        let Some(store) = self.store.clone() else {
            return Err(KrishivUnsupportedFeature {
                feature: "persisting Python UDFs on a server without a function catalog".to_owned(),
            }
            .into());
        };
        let key = self.store_key();
        let udf = if drop {
            None
        } else {
            Some(
                self.lock()
                    .payloads
                    .iter()
                    .rev()
                    // A registered function only: an inline UDF of the same name (top-bit version) is not it.
                    .find(|(n, v, _)| {
                        v & ANONYMOUS_VERSION_BIT == 0 && n.to_lowercase() == name.to_lowercase()
                    })
                    .map(|(_, _, udf)| Arc::clone(udf))
                    .ok_or_else(|| {
                        KrishivError::from(KrishivUnsupportedFeature {
                            feature: format!(
                                "persisting `{name}`: this session registered no such Python UDF"
                            ),
                        })
                    })?,
            )
        };
        // Replacing or removing a function another principal stored needs `Manage` on the cluster: what a
        // stored function runs is the owner's to change (a refusal is audited).
        let (owner_key, owner_name) = (key.clone(), name.to_owned());
        let owner_store = Arc::clone(&store);
        let stored_by =
            blocking(move || owner_store.owner(owner_key.as_deref(), &owner_name)).await?;
        let foreign = stored_by.is_some_and(|o| o != self.principal);
        if foreign {
            authorize_manage(session).await?;
        }
        let name = name.to_owned();
        let owner = self.principal.clone();
        let change = |override_owner: bool| {
            let (store, udf, key) = (Arc::clone(&store), udf.clone(), key.clone());
            let (name, owner) = (name.clone(), owner.clone());
            blocking(move || match udf {
                Some(udf) => store.put(key.as_deref(), &owner, &udf, override_owner),
                None => store.remove(key.as_deref(), &name, &owner, override_owner),
            })
        };
        // The store checks the owner inside the write, so a row that appeared after the read above is
        // still refused; the retry runs only once `Manage` passed.
        let mut outcome = change(foreign).await?;
        if outcome == StoreChange::OwnedByOther {
            authorize_manage(session).await?;
            outcome = change(true).await?;
        }
        if outcome == StoreChange::OwnedByOther {
            // A store that ignored `override_owner`: nothing was written, so do not report success.
            return Err(InternalError {
                message: format!(
                    "the function store refused to change `{name}` although the caller may override"
                ),
            }
            .into());
        }
        Ok(())
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
    pub(crate) fn copy_for(
        &self,
        principal: &str,
        tenant: Option<&str>,
        authenticated: bool,
        clone: &Session,
    ) -> Self {
        let mut copy = Self::new(self.host.as_ref(), principal, tenant, authenticated, clone);
        copy.store.clone_from(&self.store);
        // The clone is another owner's: it loads its own tenant's functions, unless the source never needed to.
        *copy.loaded.get_mut() = self.store.is_none();
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

/// Runs a catalog call on a blocking thread.
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, KrishivError> + Send + 'static,
) -> Result<T, KrishivError> {
    tokio::task::spawn_blocking(f).await.map_err(|_| {
        KrishivError::from(krishiv_common::error::InternalError {
            message: "the function catalog call did not finish".to_owned(),
        })
    })?
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

/// The session's persisted-function hook (P3-T072d): SQL `CREATE FUNCTION ... USING PYTHON` and `DROP FUNCTION`
/// over the same catalog and checks as the `PersistPythonFunction` command.
pub(crate) struct SqlPersist(pub(crate) Arc<SessionPython>);

impl krishiv_session::PersistedFunctions for SqlPersist {
    fn persist<'a>(
        &'a self,
        session: &'a Session,
        registered: &'a str,
    ) -> futures::future::BoxFuture<'a, Result<(), KrishivError>> {
        Box::pin(async move {
            crate::auth::enforce(&crate::auth::client_code(), session).await?;
            self.0.persist(session, registered, false).await
        })
    }

    fn exists<'a>(
        &'a self,
        session: &'a Session,
        name: &'a str,
    ) -> futures::future::BoxFuture<'a, Result<bool, KrishivError>> {
        // No privilege check: a name that is no stored function is `ROUTINE_NOT_FOUND` for every caller, and
        // `DROP` was checked by the statement already.
        Box::pin(async move {
            let _ = session;
            self.0.stored_owner(name).await.map(|o| o.is_some())
        })
    }

    fn remove<'a>(
        &'a self,
        session: &'a Session,
        name: &'a str,
    ) -> futures::future::BoxFuture<'a, Result<(), KrishivError>> {
        Box::pin(async move {
            crate::auth::enforce(&crate::auth::client_code(), session).await?;
            self.0.persist(session, name, true).await
        })
    }
}
