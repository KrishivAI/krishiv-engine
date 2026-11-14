// SPDX-License-Identifier: Apache-2.0
//! The pool as an executor UDF runtime: one [`SessionUdfs`] per session, implementing
//! `krishiv_exec::python_eval::UdfRuntime` for the scalar eval types (design `P3-T014-connect-udfs-sandbox.md`
//! §3.2). Its payload store is filled by the session's registrations (the Connect `udf` module, PR-26d), so
//! the executor and the session need no change to call pool workers.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::DataType;
use futures::channel::oneshot;
use futures::future::BoxFuture;
use krishiv_common::error::{
    InternalError, KrishivError, KrishivUdfNotAvailable, OperationCanceled,
};
use krishiv_exec::python_eval::{PreparedUdf, UdfCallSpec, UdfRuntime};
use std::os::fd::OwnedFd;

use crate::pool::{CallWatch, Owner, Pool};
use crate::worker::{PythonPayload, kill_via};

/// Ids of loaded payloads, unique in the process: a worker keeps what it has loaded by id.
static NEXT_PAYLOAD: AtomicU64 = AtomicU64::new(1);

/// The Python UDFs of one session, run on the pool's workers of that session.
pub struct SessionUdfs {
    pool: Pool,
    owner: Owner,
    /// The runner conf (the session's SQL conf subset Spark sends its worker), read when a call is prepared.
    conf: RwLock<Arc<BTreeMap<String, String>>>,
    payloads: RwLock<HashMap<(String, u32), Registered>>,
}

#[derive(Clone)]
struct Registered {
    id: u64,
    payload: PythonPayload,
}

impl std::fmt::Debug for SessionUdfs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionUdfs")
            .field("owner", &self.owner)
            .finish_non_exhaustive()
    }
}

impl SessionUdfs {
    /// The UDFs of `owner`'s session, run on `pool`.
    pub fn new(pool: Pool, owner: Owner) -> Self {
        SessionUdfs {
            pool,
            owner,
            conf: RwLock::new(Arc::new(BTreeMap::new())),
            payloads: RwLock::new(HashMap::new()),
        }
    }

    /// Registers `payload` as `(name, version)`; plans name UDFs by both, so an older version stays callable
    /// by the plans that name it.
    pub fn register(&self, name: impl Into<String>, version: u32, payload: PythonPayload) {
        let id = NEXT_PAYLOAD.fetch_add(1, Ordering::Relaxed);
        self.payloads
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert((name.into(), version), Registered { id, payload });
    }

    /// Sets the runner conf that calls prepared from now on send their workers.
    pub fn set_runner_conf(&self, conf: BTreeMap<String, String>) {
        *self.conf.write().unwrap_or_else(PoisonError::into_inner) = Arc::new(conf);
    }

    /// Stops this session's workers (idle ones now, busy ones when their call ends).
    pub fn release(&self) {
        self.pool.release(&self.owner);
    }
}

impl UdfRuntime for SessionUdfs {
    fn prepare(&self, spec: &UdfCallSpec) -> krishiv_exec::Result<Arc<dyn PreparedUdf>> {
        let registered = self
            .payloads
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&(spec.name.clone(), spec.version))
            .cloned()
            .ok_or_else(|| {
                KrishivError::from(KrishivUdfNotAvailable {
                    name: spec.name.clone(),
                    version: spec.version.to_string(),
                })
            })?;
        Ok(Arc::new(Prepared {
            pool: self.pool.clone(),
            owner: self.owner.clone(),
            registered,
            return_type: spec.return_arrow.clone(),
            conf: Arc::clone(&self.conf.read().unwrap_or_else(PoisonError::into_inner)),
        }))
    }
}

struct Prepared {
    pool: Pool,
    owner: Owner,
    registered: Registered,
    return_type: DataType,
    conf: Arc<BTreeMap<String, String>>,
}

/// A call's link to its future: dropping the future (the query was cancelled or failed) kills the worker
/// running it at once (design §3.4 "Cancel").
#[derive(Default)]
struct CallState {
    cancelled: bool,
    /// The pidfd of the worker running the call; cleared, under this lock, before the worker is given back, so
    /// a late cancel can never signal a worker that already serves another call (and a pidfd never names a
    /// reused pid, so a worker that exited on its own is safe too).
    running: Option<Arc<OwnedFd>>,
    /// The cancel killed `running` (a worker still starting, or one mid-call): it is never given back.
    killed: bool,
}

impl CallWatch for Mutex<CallState> {
    fn cancelled(&self) -> bool {
        self.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .cancelled
    }

    fn starting(&self, pidfd: &Arc<OwnedFd>) -> bool {
        let mut state = self.lock().unwrap_or_else(PoisonError::into_inner);
        if state.cancelled {
            return false;
        }
        state.running = Some(Arc::clone(pidfd));
        true
    }
}

struct CancelOnDrop(Arc<Mutex<CallState>>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        state.cancelled = true;
        if let Some(pidfd) = state.running.take() {
            kill_via(&pidfd);
            state.killed = true;
        }
    }
}

impl PreparedUdf for Prepared {
    fn call(&self, args: RecordBatch) -> BoxFuture<'static, krishiv_exec::Result<ArrayRef>> {
        let (done, result) = oneshot::channel();
        let call = Arc::new(Mutex::new(CallState::default()));
        let cancel = CancelOnDrop(Arc::clone(&call));
        let pool = self.pool.clone();
        let owner = self.owner.clone();
        let registered = self.registered.clone();
        let return_type = self.return_type.clone();
        let conf = Arc::clone(&self.conf);
        // Each call blocks on its worker, so it runs on one of the pool's call threads, never on the caller
        // (a compute worker). Those threads are bounded by the worker caps (`crate::calls`).
        let submitted = self.pool.submit(
            &self.owner,
            Box::new(move || {
                let result = run(
                    &pool,
                    &owner,
                    &registered,
                    &args,
                    &conf,
                    &return_type,
                    &call,
                );
                let _ = done.send(result);
            }),
        );
        if let Err(e) = submitted {
            return Box::pin(std::future::ready(Err(e)));
        }
        Box::pin(async move {
            let _cancel = cancel;
            result.await.unwrap_or_else(|_| {
                Err(InternalError {
                    message: "a Python UDF call ended without a result".to_owned(),
                }
                .into())
            })
        })
    }
}

fn run(
    pool: &Pool,
    owner: &Owner,
    registered: &Registered,
    args: &RecordBatch,
    conf: &BTreeMap<String, String>,
    return_type: &DataType,
    call: &Mutex<CallState>,
) -> Result<ArrayRef, KrishivError> {
    let lock = || call.lock().unwrap_or_else(PoisonError::into_inner);
    let mut lease = match pool.checkout(owner, &registered.payload.python_ver, call) {
        Ok(lease) => lease,
        Err(e) => {
            // A start that failed (or was killed by the cancel) left its pidfd behind; that worker is reaped.
            let mut state = lock();
            state.running = None;
            return Err(if state.cancelled {
                OperationCanceled {}.into()
            } else {
                e
            });
        }
    };
    let pidfd = lease.worker()?.pidfd();
    {
        let mut state = lock();
        if state.cancelled {
            state.running = None;
            if state.killed {
                // Killed while it started (or just after): never reused.
                lease.discard();
            }
            return Err(OperationCanceled {}.into());
        }
        state.running = Some(pidfd);
    }
    let worker = lease.worker()?;
    let max_result = pool.max_result_bytes();
    let result = worker
        .ensure_loaded(registered.id, &registered.payload)
        .and_then(|()| worker.run_batch(registered.id, args, conf, return_type, max_result));
    let mut state = lock();
    state.running = None;
    if state.cancelled {
        // Killed mid-call (or just after): never reused.
        lease.discard();
        return Err(OperationCanceled {}.into());
    }
    drop(state);
    result
}
