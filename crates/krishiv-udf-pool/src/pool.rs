// SPDX-License-Identifier: Apache-2.0
//! The pool: workers keyed by principal, session and interpreter, their caps, reuse and idle stop
//! (design `P3-T014-connect-udfs-sandbox.md` §3.4).

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

use krishiv_common::config::{
    Config, ConfigValue, PYTHON_WORKER_ALLOW_FORK, PYTHON_WORKER_EXECUTABLES,
    PYTHON_WORKER_IDLE_TIMEOUT, PYTHON_WORKER_MAX_PER_SESSION, PYTHON_WORKER_MAX_TOTAL,
    PYTHON_WORKER_MEMORY_LIMIT, UDF_SANDBOX,
};
use krishiv_common::error::{InternalError, KrishivError, OperationCanceled};
use krishiv_udf_sandbox::{Capabilities, Limits, SandboxSpec, probe};
use rustix::net::{AddressFamily, SocketFlags, SocketType};
use rustix::process::{Resource, getrlimit};

use crate::WORKER_SUBCOMMAND;
use crate::interpreter::{Interpreter, choose};
use crate::package::{self, BootstrapLimits};
use crate::worker::{Started, Worker};

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
    /// Each worker's `RLIMIT_AS`.
    pub memory_limit: Option<u64>,
    /// Allow `fork` in sandboxed workers.
    pub allow_fork: bool,
    /// How workers are started.
    pub launcher: Launcher,
    /// What the kernel can enforce; `None` probes it when the pool starts. Tests set it to check the
    /// refusal path.
    pub capabilities: Option<Capabilities>,
}

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
        PoolConfig {
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
            memory_limit: config
                .get(&PYTHON_WORKER_MEMORY_LIMIT)
                .and_then(ConfigValue::as_size),
            allow_fork: config
                .get(&PYTHON_WORKER_ALLOW_FORK)
                .and_then(ConfigValue::as_bool)
                .unwrap_or(false),
            launcher: Launcher::default(),
            capabilities: None,
        }
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
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    owner: Owner,
    interpreter: usize,
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
    interpreters: Vec<Interpreter>,
    /// Why every call is refused, when the pool cannot run workers at all.
    refusal: Option<KrishivError>,
    /// Declared before `root`: dropping it makes its directories writable, so `root` can then be removed.
    package: package::Package,
    fd_limit: u64,
    spawner: Mutex<mpsc::Sender<SpawnRequest>>,
    /// The package and the workers' private directories; removed with the pool.
    root: tempfile::TempDir,
}

/// The worker pool of one server. Cheap to clone; workers stop when the last clone is dropped.
#[derive(Clone)]
pub struct Pool {
    pub(crate) inner: Arc<Inner>,
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field("interpreters", &self.inner.interpreters.len())
            .field("sandbox", &self.inner.config.sandbox)
            .finish_non_exhaustive()
    }
}

impl Pool {
    /// Starts the pool: records the interpreters, writes the worker package, starts the spawner thread and
    /// checks the sandbox. Runs at server startup, so the interpreters' inspection (one run each) does not
    /// delay a query.
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
        let mut interpreters = Vec::new();
        let mut rejected = Vec::new();
        for exe in &config.executables {
            match Interpreter::inspect(exe) {
                Ok(i) => match i.unusable_reason() {
                    None => interpreters.push(i),
                    Some(why) => rejected.push(why),
                },
                Err(why) => rejected.push(why),
            }
        }
        for why in &rejected {
            tracing::warn!("Python UDF worker interpreter not used: {why}");
        }
        let mut refusal = interpreters.is_empty().then(|| {
            internal(format!(
                "no usable Python interpreter for UDF workers (\"krishiv.python.worker.executables\"): {}",
                if rejected.is_empty() { "none listed".to_owned() } else { rejected.join("; ") }
            ))
        });
        if config.sandbox {
            let caps = config.capabilities.unwrap_or_else(probe);
            let spec = SandboxSpec::new("/", root.path());
            if let Err(e) = caps.require(&spec) {
                tracing::warn!("Python UDFs are refused: {e}");
                refusal.get_or_insert_with(|| KrishivError::from(&e));
            }
        } else {
            tracing::warn!(
                "Python UDF workers run without the OS sandbox (\"krishiv.udf.sandbox\" is off): only \
                 resource limits, a clean environment and closed descriptors apply"
            );
        }
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
        let inner = Arc::new(Inner {
            state: Mutex::new(State::default()),
            freed: Condvar::new(),
            config,
            interpreters,
            refusal,
            package,
            fd_limit,
            spawner: Mutex::new(sender),
            root,
        });
        start_reaper(Arc::downgrade(&inner));
        Ok(Pool { inner })
    }

    /// The usable interpreters, in the listed order.
    pub fn interpreters(&self) -> &[Interpreter] {
        &self.inner.interpreters
    }

    /// Why this pool refuses every call, if it does (no usable interpreter, or the sandbox unavailable).
    pub fn refusal(&self) -> Option<&KrishivError> {
        self.inner.refusal.as_ref()
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
            for _ in &dropped {
                state.forget(owner);
            }
            if state.per_owner.contains_key(owner) {
                state.released.insert(owner.clone());
            }
            dropped
        };
        self.inner.freed.notify_all();
        drop(dropped);
    }

    /// A worker of `owner` for a client on Python `python_ver`: an idle one of the same key, else a new one
    /// within the caps, else waits for one. `cancelled` is polled while waiting.
    pub(crate) fn checkout(
        &self,
        owner: &Owner,
        python_ver: &str,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Lease, KrishivError> {
        if let Some(refusal) = &self.inner.refusal {
            return Err(refusal.clone());
        }
        let Some((index, interpreter)) = choose(&self.inner.interpreters, python_ver) else {
            return Err(internal(
                "no usable Python interpreter for UDF workers".to_owned(),
            ));
        };
        let key = Key {
            owner: owner.clone(),
            interpreter: index,
        };
        let mut state = self.inner.lock();
        loop {
            if cancelled() {
                return Err(OperationCanceled {}.into());
            }
            if let Some(idle) = state.idle.get_mut(&key).and_then(Vec::pop) {
                return Ok(self.lease(key, idle.worker));
            }
            let mine = state.per_owner.get(owner).copied().unwrap_or(0);
            if mine < self.inner.config.max_per_session {
                if state.live < self.inner.config.max_total {
                    state.live += 1;
                    *state.per_owner.entry(owner.clone()).or_default() += 1;
                    drop(state);
                    return match self.start(interpreter) {
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
                    drop(victim);
                    state = self.inner.lock();
                    continue;
                }
            } else if let Some(victim) =
                state.take_oldest_idle(|k| &k.owner == owner && k.interpreter != key.interpreter)
            {
                // At the session's cap with idle workers of another interpreter: stop one of those rather than
                // wait for the idle timeout (review 2026-11-13 m2).
                drop(state);
                drop(victim);
                state = self.inner.lock();
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

    /// Starts one worker of `interpreter` and waits for its greeting.
    fn start(&self, interpreter: &Interpreter) -> Result<Worker, KrishivError> {
        let inner = &self.inner;
        let private_dir = tempfile::Builder::new()
            .prefix("w")
            .tempdir_in(inner.root.path())
            .map_err(|e| internal(format!("cannot make a UDF worker directory: {e}")))?;
        let (server_end, worker_end) = rustix::net::socketpair(
            AddressFamily::UNIX,
            SocketType::SEQPACKET,
            SocketFlags::CLOEXEC,
            None,
        )
        .map_err(|e| internal(format!("cannot make a UDF worker socket: {e}")))?;
        let dir = private_dir.path().to_path_buf();
        let env: Vec<(OsString, OsString)> = vec![
            ("HOME".into(), dir.clone().into_os_string()),
            ("TMPDIR".into(), dir.clone().into_os_string()),
            ("LC_ALL".into(), "C.UTF-8".into()),
        ];
        let config = &inner.config;
        let mut limits = Limits::default();
        limits.address_space = config.memory_limit;
        let mut command = if config.sandbox {
            let mut spec = SandboxSpec::new(&interpreter.path, &dir);
            spec.args = vec![
                "-I".into(),
                "-c".into(),
                package::bootstrap(
                    inner.package.path(),
                    inner.fd_limit,
                    None,
                    !config.allow_fork,
                )
                .into(),
            ];
            spec.env = env;
            spec.read_paths = SandboxSpec::system_read_paths();
            spec.read_paths
                .extend(interpreter.read_paths.iter().cloned());
            spec.read_paths.push(inner.package.path().to_path_buf());
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
            let bootstrap = package::bootstrap(
                inner.package.path(),
                inner.fd_limit,
                Some(&BootstrapLimits {
                    address_space: limits.address_space,
                    file_size: limits.file_size,
                    open_files: limits.open_files,
                }),
                false,
            );
            let mut command = Command::new(&interpreter.path);
            command.args(["-I", "-c", &bootstrap]).env_clear().envs(env);
            command
        };
        command
            .current_dir(&dir)
            .stdin(Stdio::from(worker_end))
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let (reply, answer) = mpsc::sync_channel(1);
        inner
            .spawner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .send(SpawnRequest { command, reply })
            .map_err(|_| internal("the UDF worker spawner has stopped".to_owned()))?;
        let child = answer
            .recv()
            .map_err(|_| internal("the UDF worker spawner has stopped".to_owned()))?
            .map_err(|e| {
                internal(format!(
                    "cannot start a Python UDF worker ({}): {e}",
                    interpreter.path.display()
                ))
            })?;
        Worker::handshake(Started {
            child,
            control: server_end,
            private_dir,
            sandboxed: config.sandbox,
        })
    }
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl State {
    /// One worker of `owner` is gone.
    fn forget(&mut self, owner: &Owner) {
        self.live = self.live.saturating_sub(1);
        if let Some(n) = self.per_owner.get_mut(owner) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                self.per_owner.remove(owner);
                self.released.remove(owner);
            }
        }
    }

    /// Removes the worker idle the longest among the keys `which` accepts.
    fn take_oldest_idle(&mut self, which: impl Fn(&Key) -> bool) -> Option<Worker> {
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
        self.forget(&key.owner);
        Some(idle.worker)
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
