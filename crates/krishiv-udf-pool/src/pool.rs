// SPDX-License-Identifier: Apache-2.0
//! The pool: workers keyed by principal, session and interpreter, their caps, reuse and idle stop
//! (design `P3-T014-connect-udfs-sandbox.md` §3.4).

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

use krishiv_common::config::{
    Config, ConfigValue, PYTHON_WORKER_ALLOW_FORK, PYTHON_WORKER_EXECUTABLES,
    PYTHON_WORKER_IDLE_TIMEOUT, PYTHON_WORKER_MAX_PER_SESSION, PYTHON_WORKER_MAX_PER_TENANT,
    PYTHON_WORKER_MAX_TOTAL, PYTHON_WORKER_MEMORY_LIMIT, PYTHON_WORKER_TENANT_MEMORY_LIMIT,
    UDF_SANDBOX,
};
use krishiv_common::error::{
    InternalError, KrishivError, KrishivPythonWorkerLimit, OperationCanceled,
};
use krishiv_udf_sandbox::{Capabilities, Limits, SandboxSpec, probe};
use rustix::net::{AddressFamily, SocketFlags, SocketType};
use rustix::process::{Pid, PidfdFlags, Resource, getrlimit};

use crate::WORKER_SUBCOMMAND;
use crate::artifacts::{self, ArtifactSet, Failure};
use crate::calls::{CallThreads, Job, Refused};
use crate::inspection::Inspection;
use crate::interpreter::{Interpreter, choose};
use crate::package::{self, BootstrapLimits};
use crate::worker::{Started, Worker, kill_via};

/// How the pool starts the sandbox trampoline: `program` with `args`, then the sandbox spec's own
/// arguments. The default re-runs the server's own executable with [`WORKER_SUBCOMMAND`], which the binary
/// must dispatch first thing in `main` ([`crate::run_trampoline_if_requested`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Launcher {
    /// The executable that runs [`krishiv_udf_sandbox::trampoline_main`].
    pub program: PathBuf,
    /// Arguments before the spec's.
    pub args: Vec<OsString>,
}

impl Default for Launcher {
    fn default() -> Self {
        Launcher {
            // The running binary even if its file was replaced since it started.
            program: PathBuf::from("/proc/self/exe"),
            args: vec![OsString::from(WORKER_SUBCOMMAND)],
        }
    }
}

/// The pool's settings, fixed when it starts (the `krishiv.python.worker.*` and `krishiv.udf.sandbox` keys).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PoolConfig {
    /// Run workers in the OS sandbox (`krishiv.udf.sandbox=on`).
    pub sandbox: bool,
    /// The interpreters, in the administrator's order.
    pub executables: Vec<String>,
    /// How long an idle worker is kept.
    pub idle_timeout: Duration,
    /// Most workers of one session.
    pub max_per_session: usize,
    /// Most workers in all.
    pub max_total: usize,
    /// Most workers of one tenant, over all its sessions.
    pub max_per_tenant: usize,
    /// `RLIMIT_AS` of a tenant's workers, when lower than [`Self::memory_limit`].
    pub tenant_memory_limit: Option<u64>,
    /// Each worker's `RLIMIT_AS`.
    pub memory_limit: Option<u64>,
    /// Allow `fork` in sandboxed workers.
    pub allow_fork: bool,
    /// How workers are started.
    pub launcher: Launcher,
    /// What the kernel can enforce; `None` probes it when the pool starts. Tests set it to check the
    /// refusal path.
    pub capabilities: Option<Capabilities>,
    /// The most bytes one session's artifacts take once copied and unpacked (pyfiles and archives,
    /// [`crate::SessionUdfs::set_artifacts`]); more is `KRISHIV_PYTHON_ARTIFACT_REFUSED`.
    pub artifact_limit: u64,
    /// The workers' `TZ`: the server's own `TZ` by default, as a PySpark worker inherits the JVM's environment.
    /// Python renders a `TIMESTAMP` argument as a naive `datetime` in this zone (Spark 4.2.0 does the same with
    /// the executor's zone, whatever `spark.sql.session.timeZone` says); `None` leaves it to the host's
    /// `/etc/localtime`.
    pub time_zone: Option<OsString>,
}

/// [`PoolConfig::artifact_limit`]'s default: 1 GiB.
pub const DEFAULT_ARTIFACT_LIMIT: u64 = 1 << 30;

impl Default for PoolConfig {
    fn default() -> Self {
        Self::from_config(&Config::new())
    }
}

impl PoolConfig {
    /// The settings `config` holds (defaults for keys it does not set).
    pub fn from_config(config: &Config) -> Self {
        let duration = config
            .get(&PYTHON_WORKER_IDLE_TIMEOUT)
            .and_then(ConfigValue::as_duration)
            .unwrap_or(Duration::from_secs(60));
        let count = |entry, default: usize| {
            config
                .get(entry)
                .and_then(ConfigValue::as_int)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n > 0)
                .unwrap_or(default)
        };
        let pool = PoolConfig {
            sandbox: config
                .get(&UDF_SANDBOX)
                .and_then(|v| v.as_str())
                .is_none_or(|v| !v.eq_ignore_ascii_case("off")),
            executables: config
                .get(&PYTHON_WORKER_EXECUTABLES)
                .and_then(|v| v.as_str())
                .unwrap_or("python3")
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect(),
            idle_timeout: duration,
            max_per_session: count(&PYTHON_WORKER_MAX_PER_SESSION, 8),
            max_total: count(&PYTHON_WORKER_MAX_TOTAL, 64),
            max_per_tenant: count(&PYTHON_WORKER_MAX_PER_TENANT, 32),
            tenant_memory_limit: config
                .get(&PYTHON_WORKER_TENANT_MEMORY_LIMIT)
                .and_then(ConfigValue::as_size),
            memory_limit: config
                .get(&PYTHON_WORKER_MEMORY_LIMIT)
                .and_then(ConfigValue::as_size),
            allow_fork: config
                .get(&PYTHON_WORKER_ALLOW_FORK)
                .and_then(ConfigValue::as_bool)
                .unwrap_or(false),
            launcher: Launcher::default(),
            capabilities: None,
            artifact_limit: DEFAULT_ARTIFACT_LIMIT,
            time_zone: std::env::var_os("TZ"),
        };
        if pool.tenant_memory_limit_ignored() {
            tracing::warn!(
                "krishiv.python.worker.tenantMemoryLimit is not below krishiv.python.worker.memoryLimit, so it \
                 changes nothing: it only lowers a tenant's workers' limit"
            );
        }
        pool
    }

    /// Whether `tenant_memory_limit` is set but not below `memory_limit` (the lower of the two applies).
    pub(crate) fn tenant_memory_limit_ignored(&self) -> bool {
        matches!((self.tenant_memory_limit, self.memory_limit), (Some(t), Some(m)) if t >= m)
    }
}

/// Whose workers these are: a worker serves one principal and one session only (design D3), so module state,
/// `sys.path` and files never cross sessions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Owner {
    /// The authenticated principal.
    pub principal: String,
    /// The session id.
    pub session: String,
    /// The principal's tenant (multi-tenant servers). Workers are never shared across tenants, and the
    /// tenant's workers and running calls are bounded together (`krishiv.python.worker.maxPerTenant`).
    pub tenant: Option<String>,
    /// The principal is one of several the server tells apart (authentication on). With authentication off
    /// every session has the one local principal, so there is nothing to cap it against: such an owner
    /// without a tenant is in no cap group, and only `maxTotal` and `maxPerSession` bound it.
    pub capped: bool,
}

impl Owner {
    /// What the per-tenant caps count by: the tenant, or the principal when it has none (so a principal of a
    /// single-tenant server cannot take the whole pool either). Prefixed, so a tenant and a principal of the
    /// same name are two groups.
    pub(crate) fn group(&self) -> Option<String> {
        match &self.tenant {
            Some(t) => Some(format!("t:{t}")),
            None if self.capped => Some(format!("p:{}", self.principal)),
            None => None,
        }
    }

    /// Whether both are in the same cap group (no allocation).
    pub(crate) fn same_group(&self, other: &Owner) -> bool {
        match (&self.tenant, &other.tenant) {
            (Some(a), Some(b)) => a == b,
            (None, None) => self.capped && other.capped && self.principal == other.principal,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    owner: Owner,
    interpreter: usize,
    /// The session's artifact set the worker was started for (its read paths are fixed at start).
    artifacts: u64,
}

struct Idle {
    worker: Worker,
    since: Instant,
}

#[derive(Default)]
struct State {
    idle: HashMap<Key, Vec<Idle>>,
    /// Live workers (idle, busy or starting).
    live: usize,
    per_owner: HashMap<Owner, usize>,
    /// Live workers per cap group ([`Owner::group`]).
    per_group: HashMap<String, usize>,
    /// Owners whose session ended while some of their workers were busy: those are stopped on return.
    released: HashSet<Owner>,
}

struct SpawnRequest {
    command: Command,
    reply: mpsc::SyncSender<std::io::Result<Child>>,
}

pub(crate) struct Inner {
    // Workers first: they are killed before the spawner thread can end.
    state: Mutex<State>,
    freed: Condvar,
    config: PoolConfig,
    /// The interpreters and the pool's refusal, found on a thread of their own (the server listens first).
    inspection: Arc<Inspection>,
    /// Declared before `root`: dropping it makes its directories writable, so `root` can then be removed.
    package: package::Package,
    fd_limit: u64,
    spawner: Mutex<mpsc::Sender<SpawnRequest>>,
    /// The threads calls run on (bounded; see [`crate::calls`]).
    calls: CallThreads,
    /// The package and the workers' private directories; removed with the pool.
    root: tempfile::TempDir,
}

/// The workers of one [`Owner`] in a [`PoolSnapshot`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct OwnerWorkers {
    /// Whose workers these are.
    pub owner: Owner,
    /// Its live workers (idle, busy or starting).
    pub live: usize,
    /// Those of them that wait for a call.
    pub idle: usize,
}

/// The pool at one moment ([`Pool::snapshot`]): what an operator's page lists.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct PoolSnapshot {
    /// Live workers (idle, busy or starting).
    pub live: usize,
    /// Those of them that wait for a call.
    pub idle: usize,
    /// [`PoolConfig::max_total`].
    pub max_total: usize,
    /// [`PoolConfig::max_per_session`].
    pub max_per_session: usize,
    /// [`PoolConfig::max_per_tenant`].
    pub max_per_tenant: usize,
    /// Every owner with a live worker, in no order.
    pub owners: Vec<OwnerWorkers>,
}

impl PoolSnapshot {
    /// The snapshot of the counts `per_owner` (live workers) and `idle` (idle workers, an owner once per key
    /// of its workers) under `config`'s limits.
    fn of<'a>(
        config: &PoolConfig,
        live: usize,
        per_owner: &HashMap<Owner, usize>,
        idle: impl Iterator<Item = (&'a Owner, usize)>,
    ) -> Self {
        let mut waiting: HashMap<&Owner, usize> = HashMap::new();
        for (owner, n) in idle {
            *waiting.entry(owner).or_default() += n;
        }
        let owners: Vec<OwnerWorkers> = per_owner
            .iter()
            .map(|(owner, live)| OwnerWorkers {
                owner: owner.clone(),
                live: *live,
                idle: waiting.get(owner).copied().unwrap_or(0),
            })
            .collect();
        Self {
            live,
            idle: owners.iter().map(|o| o.idle).sum(),
            max_total: config.max_total,
            max_per_session: config.max_per_session,
            max_per_tenant: config.max_per_tenant,
            owners,
        }
    }
}

/// A view of a [`Pool`] that does not keep it running ([`Pool::monitor`]): for a status page, which must not
/// delay the workers' stop when the server drops the pool.
#[derive(Clone)]
pub struct PoolMonitor {
    inner: std::sync::Weak<Inner>,
}

impl std::fmt::Debug for PoolMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoolMonitor").finish_non_exhaustive()
    }
}

impl PoolMonitor {
    /// The pool now; `None` once the pool is gone.
    ///
    /// The pool is held for the length of the read. If the server drops its pool in that moment, this
    /// reference is the last one and the idle workers are stopped and reaped here, on the caller's task: only
    /// at shutdown, and each wait is short (the worker was killed first).
    #[must_use]
    pub fn snapshot(&self) -> Option<PoolSnapshot> {
        self.inner.upgrade().map(|inner| inner.snapshot())
    }
}

/// The worker pool of one server. Cheap to clone; workers stop when the last clone is dropped.
#[derive(Clone)]
pub struct Pool {
    pub(crate) inner: Arc<Inner>,
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field(
                "interpreters",
                &self.inner.inspection.get().map(|i| i.interpreters.len()),
            )
            .field("sandbox", &self.inner.config.sandbox)
            .finish_non_exhaustive()
    }
}

impl Pool {
    /// Starts the pool: writes the worker package, starts the spawner thread, checks the sandbox and starts
    /// the interpreters' inspection (one run each) on a thread of its own, so it delays neither the server's
    /// listeners nor, once it ended, a query. The first worker start waits for it.
    ///
    /// A pool that cannot run workers (no usable interpreter, or the sandbox on and the kernel unable to
    /// enforce it) still starts: its calls fail with the reason, which the startup log also gives.
    ///
    /// # Errors
    /// Only when the pool's own directory or threads cannot be made.
    pub fn new(config: PoolConfig) -> Result<Self, KrishivError> {
        let root = tempfile::Builder::new()
            .prefix("krishiv-udf-")
            .tempdir()
            .map_err(|e| internal(format!("cannot make the UDF worker directory: {e}")))?;
        let package = package::extract(&root.path().join("package"))
            .map_err(|e| internal(format!("cannot write the UDF worker package: {e}")))?;
        let mut sandbox_refusal = None;
        if config.sandbox {
            let caps = config.capabilities.unwrap_or_else(probe);
            let spec = SandboxSpec::new("/", root.path());
            if let Err(e) = caps.require(&spec) {
                sandbox_refusal = Some(KrishivError::from(&e));
            }
        } else {
            tracing::warn!(
                "Python UDF workers run without the OS sandbox (\"krishiv.udf.sandbox\" is off): only \
                 resource limits, a clean environment and closed descriptors apply"
            );
        }
        let inspection = Inspection::start(config.executables.clone(), sandbox_refusal)?;
        let fd_limit = getrlimit(Resource::Nofile).current.unwrap_or(1 << 20);
        let (sender, requests) = mpsc::channel::<SpawnRequest>();
        // The trampoline's parent-death signal follows the thread that started it, so every worker is
        // started by this one long-lived thread (the sandbox crate's rule). It ends with the pool.
        std::thread::Builder::new()
            .name("krishiv-udf-spawner".to_owned())
            .spawn(move || {
                for mut request in requests {
                    let _ = request.reply.send(request.command.spawn());
                }
            })
            .map_err(|e| internal(format!("cannot start the UDF worker spawner: {e}")))?;
        let calls = CallThreads::with_tenant_cap(
            config.max_total,
            config.max_per_session,
            config.max_per_tenant,
            config.idle_timeout,
        );
        let inner = Arc::new(Inner {
            state: Mutex::new(State::default()),
            freed: Condvar::new(),
            config,
            inspection,
            package,
            fd_limit,
            spawner: Mutex::new(sender),
            calls,
            root,
        });
        start_reaper(Arc::downgrade(&inner));
        Ok(Pool { inner })
    }

    /// The usable interpreters, in the listed order. Waits for their inspection to end.
    pub fn interpreters(&self) -> &[Interpreter] {
        &self.inner.inspection.wait().interpreters
    }

    /// Why this pool refuses every call, if it does (no usable interpreter, or the sandbox unavailable).
    /// Waits for the interpreters' inspection to end.
    pub fn refusal(&self) -> Option<&KrishivError> {
        self.inner.inspection.wait().refusal.as_ref()
    }

    /// The largest result a worker may return, in bytes: what it can write at all (its `RLIMIT_FSIZE`
    /// applies to the memfd it fills).
    pub(crate) fn max_result_bytes(&self) -> u64 {
        krishiv_udf_sandbox::DEFAULT_FILE_SIZE
    }

    /// Live workers (idle, busy or starting).
    pub fn live_workers(&self) -> usize {
        self.inner.lock().live
    }

    /// The workers and the limits now. One short hold of the pool's lock: no worker is waited for.
    #[must_use]
    pub fn snapshot(&self) -> PoolSnapshot {
        self.inner.snapshot()
    }

    /// A view of this pool that does not keep its workers running.
    #[must_use]
    pub fn monitor(&self) -> PoolMonitor {
        PoolMonitor {
            inner: Arc::downgrade(&self.inner),
        }
    }

    /// Stops the idle workers of `owner` at once, and its busy ones when they finish (its session ended).
    pub fn release(&self, owner: &Owner) {
        let dropped = {
            let mut state = self.inner.lock();
            let keys: Vec<Key> = state
                .idle
                .keys()
                .filter(|k| &k.owner == owner)
                .cloned()
                .collect();
            let mut dropped = Vec::new();
            for key in keys {
                dropped.extend(state.idle.remove(&key).unwrap_or_default());
            }
            if state.per_owner.contains_key(owner) {
                state.released.insert(owner.clone());
            }
            dropped
        };
        if dropped.is_empty() {
            self.inner.freed.notify_all();
            return;
        }
        // Dropping a worker kills it and waits for its exit: off the caller, which may be an async worker
        // thread (a session closed by an RPC or the idle sweep). The owner's slots are freed only once its
        // workers are gone ([`Released`]), so a new session never runs more than `maxTotal` workers at once.
        let released = Released {
            inner: Arc::clone(&self.inner),
            owner: owner.clone(),
            workers: dropped,
        };
        // If the thread cannot start, `spawn` drops the closure, and with it `released`: the workers are then
        // stopped, and their slots freed, on the caller's thread.
        let spawned = std::thread::Builder::new()
            .name("krishiv-udf-release".to_owned())
            .spawn(move || drop(released));
        if let Err(e) = spawned {
            tracing::warn!("Python UDF workers were stopped on the caller's thread: {e}");
        }
    }

    /// Runs `job`, a scalar call of `owner`, on one of the pool's call threads (now, or when one is free).
    ///
    /// # Errors
    /// `KRISHIV_PYTHON_WORKER_LIMIT` when `owner`'s only call (`maxPerSession` 1) is a relation task's, which
    /// may wait for this call's result; `INTERNAL_ERROR` when no call thread can run.
    pub(crate) fn submit(&self, owner: &Owner, job: Job) -> Result<(), KrishivError> {
        self.inner
            .calls
            .submit(owner.clone(), job)
            .map_err(|r| self.refused(r, owner))
    }

    /// Runs `job`, a relation task of `owner` holding one of its slots, ahead of the queued calls.
    pub(crate) fn submit_task(&self, owner: &Owner, job: Job) -> Result<(), KrishivError> {
        self.inner
            .calls
            .submit_task(owner.clone(), job)
            .map_err(|r| self.refused(r, owner))
    }

    fn refused(&self, refused: Refused, owner: &Owner) -> KrishivError {
        match refused {
            Refused::NoThread(why) => internal(why),
            Refused::Reserve => self.worker_limit(
                "A scalar Python UDF call beside an open Python UDF task",
                2,
                owner,
            ),
        }
    }

    /// `KRISHIV_PYTHON_WORKER_LIMIT` for `operation`, which needs `needed` workers at once; it names the setting
    /// that bounds the slots (`maxTotal` when it is below `maxPerSession`).
    pub(crate) fn worker_limit(
        &self,
        operation: &str,
        needed: usize,
        owner: &Owner,
    ) -> KrishivError {
        let config = &self.inner.config;
        let (setting, value) = if self.inner.calls.tenant_bound(owner) {
            ("krishiv.python.worker.maxPerTenant", config.max_per_tenant)
        } else if self.inner.calls.threads_bound() {
            ("krishiv.python.worker.maxTotal", config.max_total)
        } else {
            (
                "krishiv.python.worker.maxPerSession",
                config.max_per_session,
            )
        };
        KrishivPythonWorkerLimit {
            operation: operation.to_owned(),
            needed: needed.to_string(),
            available: self.inner.calls.slot_limit_for(owner).to_string(),
            setting: setting.to_owned(),
            value: value.to_string(),
        }
        .into()
    }

    /// The pool's call threads (the relation task slots live there).
    pub(crate) fn calls(&self) -> &CallThreads {
        &self.inner.calls
    }

    /// A worker of `owner` for a client on Python `python_ver`: an idle one of the same key, else a new one
    /// within the caps, else waits for one. `watch` is polled while waiting, and told of a worker it starts.
    pub(crate) fn checkout(
        &self,
        owner: &Owner,
        python_ver: &str,
        artifacts: &ArtifactSet,
        watch: &dyn CallWatch,
    ) -> Result<Lease, KrishivError> {
        let Some(inspected) = self.inner.inspection.wait_until(&|| watch.cancelled()) else {
            return Err(OperationCanceled {}.into());
        };
        if let Some(refusal) = &inspected.refusal {
            return Err(refusal.clone());
        }
        if let Some(refused) = artifacts.refused() {
            return Err(refused.clone());
        }
        let Some((index, interpreter)) = choose(&inspected.interpreters, python_ver) else {
            return Err(internal(
                "no usable Python interpreter for UDF workers".to_owned(),
            ));
        };
        let key = Key {
            owner: owner.clone(),
            interpreter: index,
            artifacts: artifacts.id,
        };
        let mut state = self.inner.lock();
        loop {
            if watch.cancelled() {
                return Err(OperationCanceled {}.into());
            }
            if let Some(idle) = state.idle.get_mut(&key).and_then(Vec::pop) {
                return Ok(self.lease(key, idle.worker));
            }
            let mine = state.per_owner.get(owner).copied().unwrap_or(0);
            {
                let group = owner.group();
                let theirs = group
                    .as_ref()
                    .map_or(0, |g| state.per_group.get(g).copied().unwrap_or(0));
                if group.is_some() && theirs >= self.inner.config.max_per_tenant {
                    // At the tenant's cap: stop its longest-idle worker of another key (another session,
                    // interpreter or artifact set) rather than wait for the idle timeout; never another
                    // tenant's.
                    if let Some(victim) =
                        state.take_oldest_idle(|k| k.owner.same_group(owner) && *k != key)
                    {
                        drop(state);
                        state = self.inner.evicted(victim);
                        continue;
                    }
                    state = self
                        .inner
                        .freed
                        .wait_timeout(state, Duration::from_millis(100))
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                    continue;
                }
            }
            if mine < self.inner.config.max_per_session {
                if state.live < self.inner.config.max_total {
                    state.live += 1;
                    *state.per_owner.entry(owner.clone()).or_default() += 1;
                    if let Some(g) = owner.group() {
                        *state.per_group.entry(g).or_default() += 1;
                    }
                    drop(state);
                    return match self.start(interpreter, artifacts, owner, watch) {
                        Ok(worker) => Ok(self.lease(key, worker)),
                        Err(e) => {
                            self.inner.lock().forget(owner);
                            self.inner.freed.notify_all();
                            Err(e)
                        }
                    };
                }
                // At the total cap: stop the longest-idle worker of anyone to make room.
                if let Some(victim) = state.take_oldest_idle(|_| true) {
                    drop(state);
                    state = self.inner.evicted(victim);
                    continue;
                }
            } else if let Some(victim) = state.take_oldest_idle(|k| &k.owner == owner && *k != key)
            {
                // At the session's cap with idle workers of another interpreter (or of an artifact set the
                // session replaced): stop one of those rather than wait for the idle timeout (review
                // 2026-11-13 m2).
                drop(state);
                state = self.inner.evicted(victim);
                continue;
            }
            state = self
                .inner
                .freed
                .wait_timeout(state, Duration::from_millis(100))
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    fn lease(&self, key: Key, worker: Worker) -> Lease {
        Lease {
            pool: Arc::clone(&self.inner),
            key,
            worker: Some(worker),
            reusable: true,
        }
    }

    /// Starts one worker of `interpreter` for the artifact set `artifacts` and waits for its greeting; `watch`
    /// gets its pidfd first, so a cancel kills it rather than wait for the greeting.
    fn start(
        &self,
        interpreter: &Interpreter,
        artifacts: &ArtifactSet,
        owner: &Owner,
        watch: &dyn CallWatch,
    ) -> Result<Worker, KrishivError> {
        let inner = &self.inner;
        let prepared = if artifacts.is_empty() {
            None
        } else {
            Some(artifacts.prepare(
                inner.root.path(),
                inner.config.artifact_limit,
                &|archive, dest, budget| self.unpack(interpreter, archive, dest, budget, watch),
                &|| watch.cancelled(),
            )?)
        };
        let private_dir = tempfile::Builder::new()
            .prefix("w")
            .tempdir_in(inner.root.path())
            .map_err(|e| internal(format!("cannot make a UDF worker directory: {e}")))?;
        if let Some(p) = &prepared {
            link_files(p, private_dir.path())?;
        }
        let (server_end, worker_end) = rustix::net::socketpair(
            AddressFamily::UNIX,
            SocketType::SEQPACKET,
            SocketFlags::CLOEXEC,
            None,
        )
        .map_err(|e| internal(format!("cannot make a UDF worker socket: {e}")))?;
        let no_paths = Vec::new();
        let (paths, read) = prepared
            .as_ref()
            .map_or((&no_paths, &no_paths), |p| (&p.sys_path, &p.read_paths));
        let mut command = self.command(
            interpreter,
            private_dir.path(),
            &package::Entry::Worker { paths },
            read,
            None,
            owner
                .tenant
                .is_some()
                .then_some(inner.config.tenant_memory_limit)
                .flatten(),
        );
        command
            .stdin(Stdio::from(worker_end))
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let child = self.spawn(command, interpreter)?;
        Worker::handshake(
            Started {
                child,
                control: server_end,
                private_dir,
                sandboxed: inner.config.sandbox,
                artifacts: prepared,
            },
            watch,
        )
    }

    /// The command that runs `entry` with `interpreter` in the private directory `dir`: through the sandbox
    /// trampoline (reading the system, the interpreter, the package and `read`; writing only `dir`) when it is
    /// on, else directly with the limits applied by the bootstrap. `file_size` lowers `RLIMIT_FSIZE`.
    fn command(
        &self,
        interpreter: &Interpreter,
        dir: &Path,
        entry: &package::Entry<'_>,
        read: &[PathBuf],
        file_size: Option<u64>,
        tenant_memory: Option<u64>,
    ) -> Command {
        let inner = &self.inner;
        let config = &inner.config;
        let mut env: Vec<(OsString, OsString)> = vec![
            ("HOME".into(), dir.as_os_str().to_owned()),
            ("TMPDIR".into(), dir.as_os_str().to_owned()),
            ("LC_ALL".into(), "C.UTF-8".into()),
        ];
        if let Some(zone) = &config.time_zone {
            env.push(("TZ".into(), zone.clone()));
        }
        let mut limits = Limits::default();
        limits.address_space = match (config.memory_limit, tenant_memory) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        if let Some(n) = file_size {
            limits.file_size = Some(limits.file_size.map_or(n, |m| m.min(n)));
        }
        let mut command = if config.sandbox {
            let mut spec = SandboxSpec::new(&interpreter.path, dir);
            spec.args = vec![
                "-I".into(),
                "-c".into(),
                package::bootstrap_for(
                    inner.package.path(),
                    inner.fd_limit,
                    None,
                    !config.allow_fork,
                    entry,
                )
                .into(),
            ];
            spec.env = env;
            spec.read_paths = SandboxSpec::system_read_paths();
            spec.read_paths
                .extend(interpreter.read_paths.iter().cloned());
            spec.read_paths.push(inner.package.path().to_path_buf());
            spec.read_paths.extend(read.iter().cloned());
            spec.limits = limits;
            spec.allow_fork = config.allow_fork;
            spec.parent_pid = Some(std::process::id());
            // The trampoline's own environment is empty too: the spec carries the worker's.
            let mut command = Command::new(&config.launcher.program);
            command
                .args(&config.launcher.args)
                .args(spec.to_args())
                .env_clear();
            command
        } else {
            let bootstrap = package::bootstrap_for(
                inner.package.path(),
                inner.fd_limit,
                Some(&BootstrapLimits {
                    address_space: limits.address_space,
                    file_size: limits.file_size,
                    open_files: limits.open_files,
                }),
                false,
                entry,
            );
            let mut command = Command::new(&interpreter.path);
            command.args(["-I", "-c", &bootstrap]).env_clear().envs(env);
            command
        };
        command.current_dir(dir);
        command
    }

    /// Starts `command` on the spawner thread (the parent-death signal follows the starting thread).
    fn spawn(&self, command: Command, interpreter: &Interpreter) -> Result<Child, KrishivError> {
        let (reply, answer) = mpsc::sync_channel(1);
        self.inner
            .spawner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .send(SpawnRequest { command, reply })
            .map_err(|_| internal("the UDF worker spawner has stopped".to_owned()))?;
        answer
            .recv()
            .map_err(|_| internal("the UDF worker spawner has stopped".to_owned()))?
            .map_err(|e| {
                internal(format!(
                    "cannot start a Python UDF worker ({}): {e}",
                    interpreter.path.display()
                ))
            })
    }

    /// Unpacks the archive artifact `archive` (the pool's copy) into the empty directory `dest`, within `budget`
    /// bytes, with `interpreter` (`unpack.py`; in the sandbox when it is on: it reads the archive and writes
    /// only `dest`). The bytes unpacked. Runs on a call thread (it blocks until the program ends, at most
    /// [`UNPACK_TIMEOUT`]); a cancel of `watch` kills the program through its pidfd. Only the program's own
    /// refusal or failure is about the archive ([`Failure::lasting`]); a cancel, a timeout (until the set's
    /// [`artifacts::MAX_TIMEOUTS`]th) or a program that could not be started or waited for is not.
    fn unpack(
        &self,
        interpreter: &Interpreter,
        archive: &Path,
        dest: &Path,
        budget: u64,
        watch: &dyn CallWatch,
    ) -> Result<u64, Failure> {
        let mut command = self.command(
            interpreter,
            dest,
            &package::Entry::Unpack {
                archive,
                dest,
                budget,
            },
            &[archive.to_path_buf()],
            // No one file past the whole budget (at least one byte, so an empty file still opens).
            Some(budget.max(1)),
            None,
        );
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = self
            .spawn(command, interpreter)
            .map_err(Failure::transient)?;
        let pid = Pid::from_child(&child);
        // The child is not reaped yet, so `pid` is still its own.
        let pidfd = match rustix::process::pidfd_open(pid, PidfdFlags::empty()) {
            Ok(fd) => fd,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(Failure::transient(internal(format!(
                    "cannot open a pidfd for the UDF archive unpacker: {e}"
                ))));
            }
        };
        let output = match wait_capped(&mut child, &pidfd, UNPACK_TIMEOUT, &|| watch.cancelled()) {
            Ok(Ended::Exited(output)) => output,
            Ok(Ended::Cancelled) => return Err(Failure::transient(OperationCanceled {}.into())),
            Ok(Ended::TimedOut) => {
                return Err(Failure::timed_out(artifacts::refused(
                    archive,
                    &format!("it could not be unpacked in {} s", UNPACK_TIMEOUT.as_secs()),
                )));
            }
            Err(e) => {
                return Err(Failure::transient(artifacts::refused(
                    archive,
                    &format!("it could not be unpacked: {e}"),
                )));
            }
        };
        // The count is the program's last line (a launcher may print before it starts the interpreter).
        if output.status.success()
            && let Some(n) = String::from_utf8_lossy(&output.stdout)
                .lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .and_then(|l| l.trim().parse::<u64>().ok())
        {
            return Ok(n);
        }
        let why = String::from_utf8_lossy(&output.stderr);
        let why = why.trim();
        Err(Failure::lasting(artifacts::refused(
            archive,
            if why.is_empty() {
                "the unpacking program failed"
            } else {
                why
            },
        )))
    }
}

/// Makes each entry of `prepared`'s files directory reachable from a worker's working directory `dir` (its
/// own new directory): a symbolic link `dir/<name>` to it. Spark's worker runs in its files directory; this
/// worker keeps its private directory as its (writable) working directory. Links are made by the server, in a
/// directory only it has written, so a name cannot collide with or follow anything the client made.
fn link_files(prepared: &artifacts::Prepared, dir: &Path) -> Result<(), KrishivError> {
    for name in &prepared.entries {
        std::os::unix::fs::symlink(prepared.files_dir.join(name), dir.join(name)).map_err(|e| {
            internal(format!(
                "cannot link the UDF artifact {name} into a worker directory: {e}"
            ))
        })?;
    }
    Ok(())
}

/// The longest an archive artifact's unpacking may take.
const UNPACK_TIMEOUT: Duration = Duration::from_secs(300);

/// The most bytes of the unpacking program's output kept (its reason, or its byte count).
const UNPACK_OUTPUT: u64 = 16 * 1024;

/// How often [`wait_capped`] looks at the program and the cancel.
const UNPACK_TICK: Duration = Duration::from_millis(20);

/// A thread reading `pipe` to its end, keeping the first [`UNPACK_OUTPUT`] bytes (the rest is read and
/// dropped, so the writer never blocks on a full pipe).
fn read_capped(
    pipe: Option<impl std::io::Read + Send + 'static>,
) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        use std::io::Read as _;
        let mut out = Vec::new();
        if let Some(mut p) = pipe {
            let _ = (&mut p).take(UNPACK_OUTPUT).read_to_end(&mut out);
            let _ = std::io::copy(&mut p, &mut std::io::sink());
        }
        out
    })
}

/// How a program [`wait_capped`] waited for ended.
#[derive(Debug)]
enum Ended {
    /// It exited: what it wrote, each stream cut to [`UNPACK_OUTPUT`] bytes.
    Exited(std::process::Output),
    /// It ran past the timeout and was killed.
    TimedOut,
    /// The call was cancelled and the program killed.
    Cancelled,
}

/// Waits for `child` (whose pidfd is `pidfd`) to exit; killed, through the pidfd, past `timeout` or once
/// `cancelled` says so (and reaped either way).
fn wait_capped(
    child: &mut Child,
    pidfd: &OwnedFd,
    timeout: Duration,
    cancelled: &dyn Fn() -> bool,
) -> std::io::Result<Ended> {
    let stdout = read_capped(child.stdout.take());
    let stderr = read_capped(child.stderr.take());
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        let ended = if cancelled() {
            Ended::Cancelled
        } else if Instant::now() >= deadline {
            Ended::TimedOut
        } else {
            std::thread::sleep(UNPACK_TICK);
            continue;
        };
        kill_via(pidfd);
        let _ = child.wait();
        return Ok(ended);
    };
    Ok(Ended::Exited(std::process::Output {
        status,
        stdout: stdout.join().unwrap_or_default(),
        stderr: stderr.join().unwrap_or_default(),
    }))
}

/// What a call tells the pool while it waits for a worker or starts one.
pub(crate) trait CallWatch {
    /// The call was cancelled.
    fn cancelled(&self) -> bool;
    /// A worker is starting for the call and has `pidfd`: a cancel from now on kills it. False when the call
    /// was already cancelled; the start is then abandoned (the worker killed and reaped).
    fn starting(&self, pidfd: &Arc<OwnedFd>) -> bool;
}

/// Idle workers taken from a released owner: dropped, they stop the workers first (each waits for its exit),
/// then free the owner's slots and wake the waiters, in that order.
struct Released {
    inner: Arc<Inner>,
    owner: Owner,
    workers: Vec<Idle>,
}

impl Drop for Released {
    fn drop(&mut self) {
        let n = self.workers.len();
        drop(std::mem::take(&mut self.workers));
        {
            let mut state = self.inner.lock();
            for _ in 0..n {
                state.forget(&self.owner);
            }
        }
        self.inner.freed.notify_all();
    }
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// [`Pool::snapshot`].
    fn snapshot(&self) -> PoolSnapshot {
        let state = self.lock();
        PoolSnapshot::of(
            &self.config,
            state.live,
            &state.per_owner,
            state
                .idle
                .iter()
                .map(|(key, list)| (&key.owner, list.len())),
        )
    }

    /// Stops `victim` (an idle worker taken with `take_oldest_idle`, the state lock released), then releases
    /// its charge and wakes the waiters; returns the re-taken lock.
    fn evicted(&self, victim: (Worker, Owner)) -> MutexGuard<'_, State> {
        let (worker, owner) = victim;
        drop(worker);
        let mut state = self.lock();
        state.forget(&owner);
        self.freed.notify_all();
        state
    }
}

impl State {
    /// One worker of `owner` is gone.
    fn forget(&mut self, owner: &Owner) {
        self.live = self.live.saturating_sub(1);
        if let Some(group) = owner.group()
            && let Some(n) = self.per_group.get_mut(&group)
        {
            *n = n.saturating_sub(1);
            if *n == 0 {
                self.per_group.remove(&group);
            }
        }
        if let Some(n) = self.per_owner.get_mut(owner) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                self.per_owner.remove(owner);
                self.released.remove(owner);
            }
        }
    }

    /// Removes the worker idle the longest among the keys `which` accepts. Its count stays charged (owner,
    /// group, live) until the caller has stopped it and calls [`Inner::evicted`], so a replacement never
    /// runs beside a victim that is still exiting.
    fn take_oldest_idle(&mut self, which: impl Fn(&Key) -> bool) -> Option<(Worker, Owner)> {
        let key = self
            .idle
            .iter()
            .filter(|(k, _)| which(k))
            .filter_map(|(k, list)| list.first().map(|i| (k, i.since)))
            .min_by_key(|(_, since)| *since)
            .map(|(k, _)| k.clone())?;
        let list = self.idle.get_mut(&key)?;
        let idle = list.remove(0);
        if list.is_empty() {
            self.idle.remove(&key);
        }
        Some((idle.worker, key.owner))
    }
}

/// A worker handed to one call. Dropping it gives the worker back (kept for reuse when it is healthy and
/// its session is still open) or stops it.
pub(crate) struct Lease {
    pool: Arc<Inner>,
    key: Key,
    worker: Option<Worker>,
    reusable: bool,
}

impl Lease {
    pub(crate) fn worker(&mut self) -> Result<&mut Worker, KrishivError> {
        self.worker
            .as_mut()
            .ok_or_else(|| internal("a UDF worker lease without a worker".to_owned()))
    }

    /// The worker is not given back: it was cancelled mid-call or is in an unknown state.
    pub(crate) fn discard(&mut self) {
        self.reusable = false;
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        let Some(worker) = self.worker.take() else {
            return;
        };
        let stopped = {
            let mut state = self.pool.lock();
            if self.reusable && worker.healthy() && !state.released.contains(&self.key.owner) {
                // Most recently used last: a later checkout pops it (warm), the reaper ages the front.
                state.idle.entry(self.key.clone()).or_default().push(Idle {
                    worker,
                    since: Instant::now(),
                });
                None
            } else {
                state.forget(&self.key.owner);
                Some(worker)
            }
        };
        self.pool.freed.notify_all();
        drop(stopped);
    }
}

/// Stops workers idle longer than the idle timeout, until the pool is gone.
fn start_reaper(pool: Weak<Inner>) {
    let Some(inner) = pool.upgrade() else { return };
    let tick =
        (inner.config.idle_timeout / 4).clamp(Duration::from_millis(50), Duration::from_secs(5));
    drop(inner);
    let spawned = std::thread::Builder::new()
        .name("krishiv-udf-reaper".to_owned())
        .spawn(move || {
            loop {
                let Some(inner) = pool.upgrade() else { return };
                let timeout = inner.config.idle_timeout;
                let expired = {
                    let mut state = inner.lock();
                    let mut expired = Vec::new();
                    let now = Instant::now();
                    let mut emptied = Vec::new();
                    let mut gone = Vec::new();
                    for (key, list) in &mut state.idle {
                        let keep = list
                            .iter()
                            .position(|i| now.duration_since(i.since) < timeout)
                            .unwrap_or(list.len());
                        for idle in list.drain(..keep) {
                            gone.push(key.owner.clone());
                            expired.push(idle.worker);
                        }
                        if list.is_empty() {
                            emptied.push(key.clone());
                        }
                    }
                    for key in emptied {
                        state.idle.remove(&key);
                    }
                    for owner in gone {
                        state.forget(&owner);
                    }
                    expired
                };
                if !expired.is_empty() {
                    inner.freed.notify_all();
                }
                drop(expired);
                // Wait without holding the pool, so dropping it stops its workers at once.
                drop(inner);
                std::thread::park_timeout(tick);
            }
        });
    if let Err(e) = spawned {
        tracing::warn!("cannot start the UDF worker idle timer: {e}");
    }
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An interpreter that answers its inspection only after `release` exists, then fails it.
    fn slow_interpreter(dir: &Path, release: &Path) -> String {
        use std::os::unix::fs::PermissionsExt;
        let exe = dir.join("python3");
        let script = format!(
            "#!/bin/sh\nwhile [ ! -e '{}' ]; do sleep 0.05; done\necho broken >&2\nexit 3\n",
            release.display()
        );
        std::fs::write(&exe, script).expect("script");
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        exe.display().to_string()
    }

    struct Never;
    impl CallWatch for Never {
        fn cancelled(&self) -> bool {
            false
        }
        fn starting(&self, _pidfd: &Arc<OwnedFd>) -> bool {
            true
        }
    }

    #[test]
    fn the_pool_starts_before_its_interpreters_answer_and_a_call_waits_for_them() {
        let dir = tempfile::tempdir().expect("dir");
        let release = dir.path().join("release");
        let config = PoolConfig {
            executables: vec![slow_interpreter(dir.path(), &release)],
            sandbox: false,
            ..PoolConfig::default()
        };
        let pool = Pool::new(config).expect("pool");
        assert!(
            pool.inner.inspection.get().is_none(),
            "Pool::new waited for the interpreter"
        );
        let owner = Owner {
            principal: "ana".to_owned(),
            session: "s1".to_owned(),
            tenant: None,
            capped: false,
        };
        let call = {
            let pool = pool.clone();
            std::thread::spawn(move || {
                pool.checkout(
                    &owner,
                    "3.12",
                    &ArtifactSet::new(Vec::new(), Vec::new()),
                    &Never,
                )
                .map(drop)
            })
        };
        std::thread::sleep(Duration::from_millis(200));
        assert!(
            !call.is_finished(),
            "the call did not wait for the inspection"
        );
        std::fs::write(&release, b"").expect("release");
        let why = call.join().expect("call").expect_err("refused").to_string();
        assert!(
            why.contains("no usable Python interpreter") && why.contains("broken"),
            "{why}"
        );
        assert!(pool.interpreters().is_empty());
    }

    #[test]
    fn a_snapshot_counts_each_owners_idle_workers_over_all_its_keys() {
        let owner = |session: &str, tenant: Option<&str>| Owner {
            principal: "ana".to_owned(),
            session: session.to_owned(),
            tenant: tenant.map(str::to_owned),
            capped: true,
        };
        let (a, b) = (owner("s1", Some("acme")), owner("s2", None));
        let per_owner = HashMap::from([(a.clone(), 3), (b.clone(), 1)]);
        let config = PoolConfig {
            max_total: 9,
            max_per_session: 4,
            max_per_tenant: 6,
            ..PoolConfig::default()
        };
        // `a` has idle workers under two keys (two interpreters or artifact sets), `b` none.
        let mut snapshot = PoolSnapshot::of(&config, 4, &per_owner, [(&a, 1), (&a, 1)].into_iter());
        snapshot
            .owners
            .sort_by(|x, y| x.owner.session.cmp(&y.owner.session));
        assert_eq!(
            snapshot,
            PoolSnapshot {
                live: 4,
                idle: 2,
                max_total: 9,
                max_per_session: 4,
                max_per_tenant: 6,
                owners: vec![
                    OwnerWorkers {
                        owner: a,
                        live: 3,
                        idle: 2
                    },
                    OwnerWorkers {
                        owner: b,
                        live: 1,
                        idle: 0
                    },
                ],
            }
        );
    }

    #[test]
    fn a_tenant_memory_limit_not_below_the_general_one_is_flagged() {
        let mut config = PoolConfig {
            memory_limit: Some(1 << 30),
            tenant_memory_limit: Some(512 << 20),
            ..PoolConfig::default()
        };
        assert!(!config.tenant_memory_limit_ignored());
        config.tenant_memory_limit = Some(2 << 30);
        assert!(config.tenant_memory_limit_ignored());
        config.tenant_memory_limit = Some(1 << 30);
        assert!(config.tenant_memory_limit_ignored());
        config.tenant_memory_limit = None;
        assert!(!config.tenant_memory_limit_ignored());
    }

    fn sleeper() -> (Child, OwnedFd) {
        let child = Command::new("sleep")
            .arg("30")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let pidfd =
            rustix::process::pidfd_open(Pid::from_child(&child), PidfdFlags::empty()).unwrap();
        (child, pidfd)
    }

    #[test]
    fn a_cancel_or_the_timeout_kills_the_unpacker() {
        let (mut child, pidfd) = sleeper();
        let start = Instant::now();
        let ended = wait_capped(&mut child, &pidfd, Duration::from_secs(60), &|| true).unwrap();
        assert!(matches!(ended, Ended::Cancelled), "{ended:?}");
        assert!(start.elapsed() < Duration::from_secs(20));
        // Reaped: nothing is left to wait for.
        assert!(child.try_wait().unwrap().is_some());

        let (mut child, pidfd) = sleeper();
        let ended = wait_capped(&mut child, &pidfd, Duration::ZERO, &|| false).unwrap();
        assert!(matches!(ended, Ended::TimedOut), "{ended:?}");

        let mut child = Command::new("echo")
            .arg("7")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let pidfd =
            rustix::process::pidfd_open(Pid::from_child(&child), PidfdFlags::empty()).unwrap();
        match wait_capped(&mut child, &pidfd, Duration::from_secs(60), &|| false).unwrap() {
            Ended::Exited(output) => assert_eq!(output.stdout, b"7\n"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn dropping_the_pool_removes_its_directory_and_package() {
        let config = PoolConfig {
            sandbox: false,
            ..PoolConfig::default()
        };
        let pool = Pool::new(config).unwrap_or_else(|e| panic!("{e}"));
        let root = pool.inner.root.path().to_path_buf();
        assert!(root.join("package/krishiv/_udf/pool_worker.py").is_file());
        drop(pool);
        // The reaper upgrades its weak reference once a tick, so the last reference may be its own.
        let deadline = Instant::now() + Duration::from_secs(10);
        while root.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(!root.exists(), "{} is left behind", root.display());
    }
}
