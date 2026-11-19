// SPDX-License-Identifier: Apache-2.0
//! The pool as an executor UDF runtime: one [`SessionUdfs`] per session, implementing
//! `krishiv_exec::python_eval::UdfRuntime` for the scalar eval types and, through `crate::relation`, the
//! relation kinds (design `P3-T014-connect-udfs-sandbox.md` §3.2). Its payload store is filled by the session's registrations (the Connect `udf` module, PR-26d), so
//! the executor and the session need no change to call pool workers.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::DataType;
use futures::channel::oneshot;
use futures::future::BoxFuture;
use krishiv_common::error::{
    InternalError, KrishivError, KrishivPythonUdfEvalTypeMismatch, KrishivUdfNotAvailable,
    OperationCanceled,
};
use krishiv_exec::python_eval::{PreparedUdf, UdfCallSpec, UdfRuntime};
use krishiv_exec::python_relation::{PreparedRelationUdf, RelationCallSpec};
use krishiv_plan::function::PythonEvalType;
use std::os::fd::OwnedFd;

use crate::artifacts::{ArtifactSet, PythonArchive};
use crate::pool::{CallWatch, Owner, Pool};
use crate::relation::PoolSlots;
use crate::worker::{PythonPayload, Worker, kill_via};

/// Ids of loaded payloads, unique in the process: a worker keeps what it has loaded by id.
static NEXT_PAYLOAD: AtomicU64 = AtomicU64::new(1);

/// Where a session's runner conf is read from when a call is prepared: the session's live conf, so every
/// query sends the values current when it starts, as Spark does per execution.
pub type ConfSource = Arc<dyn Fn() -> BTreeMap<String, String> + Send + Sync>;

/// The Python UDFs of one session, run on the pool's workers of that session.
pub struct SessionUdfs {
    pool: Pool,
    owner: Owner,
    /// The session's relation task slots, shared by its relation call sites.
    slots: Arc<PoolSlots>,
    /// The runner conf (the session's SQL conf subset Spark sends its worker), read when a call is prepared.
    conf: RwLock<Arc<BTreeMap<String, String>>>,
    /// When set, read at each `prepare` instead of `conf`.
    conf_source: Option<ConfSource>,
    payloads: RwLock<HashMap<(String, u32), Registered>>,
    /// The session's pyfiles and archives, read when a call is prepared.
    artifacts: RwLock<Arc<ArtifactSet>>,
}

#[derive(Clone)]
pub(crate) struct Registered {
    pub(crate) id: u64,
    pub(crate) payload: PythonPayload,
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
            slots: Arc::new(PoolSlots::new(pool.clone(), owner.clone())),
            pool,
            owner,
            conf: RwLock::new(Arc::new(BTreeMap::new())),
            conf_source: None,
            payloads: RwLock::new(HashMap::new()),
            artifacts: RwLock::new(Arc::new(ArtifactSet::empty())),
        }
    }

    /// Sets the session's Python artifacts that calls prepared from now on see: `pyfiles` (`.py` files, and
    /// zips or eggs) in the workers' files directory, which is on their `sys.path` with each zip or egg, and
    /// `archives` (zip, jar, tar, tar.gz, tgz) unpacked into that directory under their alias or file name,
    /// each entry reachable from the workers' working directory as in Spark (`artifacts.rs`). The server names
    /// the paths: each must be an absolute path of a regular file with no symbolic link at or above it, and an
    /// alias one plain name (not empty, no `/`, not hidden), or the calls fail with `INVALID_ARTIFACT_PATH`;
    /// two artifacts of one name fail them with `ARTIFACT_ALREADY_EXISTS`; an archive the pool cannot unpack
    /// within its limit fails them with `KRISHIV_PYTHON_ARTIFACT_REFUSED`. Workers started for the previous set
    /// are not reused for the new one. The Spark Connect server does not call this yet (its
    /// `SessionUdfRuntime` forward comes with a later change), so a Connect client's artifacts do not reach
    /// the workers until then.
    pub fn set_artifacts(&self, pyfiles: Vec<PathBuf>, archives: Vec<PythonArchive>) {
        *self
            .artifacts
            .write()
            .unwrap_or_else(PoisonError::into_inner) =
            Arc::new(ArtifactSet::new(pyfiles, archives));
    }

    /// The artifact set calls prepared now see.
    fn artifacts(&self) -> Arc<ArtifactSet> {
        Arc::clone(
            &self
                .artifacts
                .read()
                .unwrap_or_else(PoisonError::into_inner),
        )
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

    /// Reads the runner conf from `source` each time a call is prepared (one query's calls share the conf
    /// current when it was prepared); [`Self::set_runner_conf`] is then unused.
    #[must_use]
    pub fn with_conf_source(mut self, source: ConfSource) -> Self {
        self.conf_source = Some(source);
        self
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

impl SessionUdfs {
    /// The payload registered as `(name, version)`.
    fn registered(&self, name: &str, version: u32) -> Result<Registered, KrishivError> {
        self.payloads
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&(name.to_owned(), version))
            .cloned()
            .ok_or_else(|| {
                KrishivError::from(KrishivUdfNotAvailable {
                    name: name.to_owned(),
                    version: version.to_string(),
                })
            })
    }

    /// The runner conf a call prepared now sends: the [`ConfSource`]'s, else the one
    /// [`Self::set_runner_conf`] set.
    fn runner_conf(&self) -> Arc<BTreeMap<String, String>> {
        match &self.conf_source {
            Some(source) => Arc::new(source()),
            None => Arc::clone(&self.conf.read().unwrap_or_else(PoisonError::into_inner)),
        }
    }
}

/// Spark's `SQL_BATCHED_UDF` (100), which a plan never names (`PythonEvalType` has no such kind): the worker
/// runs it on the Arrow-batched path (`worker.py` `make_wrapper`).
const SQL_BATCHED_UDF: i32 = 100;

/// Whether the pool's workers run a payload of eval type `code` as a scalar call (`pool_worker.py`): Spark's
/// `SQL_BATCHED_UDF`, or a [`PythonEvalType`] that [`PythonEvalType::is_scalar`] (the Arrow-optimized `@udf`,
/// the pandas and the Arrow scalar UDF; review N7d-1: derived, so a new scalar kind cannot drift).
///
/// A scalar call site accepts a payload of any of them **on purpose** (review N7d-2): the worker runs the
/// payload's own eval type, not the call's. Over Connect a plain `@udf` is registered as 100 and planned as the
/// Arrow-optimized 101 (`krishiv-connect`'s `translate/udf.rs`; review R7c-1), and a payload registered as a
/// pandas UDF (200) and called as 101 runs the pandas wrapper, as it did before the check existed. Do not
/// tighten this to the call's exact code.
fn runs_as_scalar(code: i32) -> bool {
    code == SQL_BATCHED_UDF
        || PythonEvalType::from_code(code).is_some_and(PythonEvalType::is_scalar)
}

/// The payload store finds a UDF by `(name, version)` only, so a plan's call and the registration may disagree
/// on its kind (a scalar UDF called as a relation, or the reverse): `KRISHIV_PYTHON_UDF_EVAL_TYPE_MISMATCH`,
/// before any worker starts, rather than a worker running a payload as the wrong kind. A relation call needs
/// the eval type its runner runs (each relation kind's runner differs): `runs(registered, called)` says which
/// payload codes fit, the exact code but for a grouped aggregate over a window (`relation.rs` `runs_as`).
pub(crate) fn check_eval_type(
    name: &str,
    version: u32,
    registered: &Registered,
    eval_type: i32,
    called: &str,
    runs: fn(i32, i32) -> bool,
) -> Result<(), KrishivError> {
    if runs(registered.payload.eval_type, eval_type) {
        return Ok(());
    }
    Err(eval_type_mismatch(
        name, version, registered, eval_type, called,
    ))
}

/// As [`check_eval_type`] for a scalar call: any scalar payload ([`runs_as_scalar`]) fits.
fn check_scalar_eval_type(
    name: &str,
    version: u32,
    registered: &Registered,
    eval_type: i32,
) -> Result<(), KrishivError> {
    if runs_as_scalar(registered.payload.eval_type) {
        return Ok(());
    }
    Err(eval_type_mismatch(
        name,
        version,
        registered,
        eval_type,
        "a scalar function",
    ))
}

fn eval_type_mismatch(
    name: &str,
    version: u32,
    registered: &Registered,
    eval_type: i32,
    called: &str,
) -> KrishivError {
    KrishivError::from(KrishivPythonUdfEvalTypeMismatch {
        name: name.to_owned(),
        version: version.to_string(),
        registered: registered.payload.eval_type.to_string(),
        called: called.to_owned(),
        eval_type: eval_type.to_string(),
    })
}

impl UdfRuntime for SessionUdfs {
    fn prepare(&self, spec: &UdfCallSpec) -> krishiv_exec::Result<Arc<dyn PreparedUdf>> {
        let registered = self.registered(&spec.name, spec.version)?;
        check_scalar_eval_type(&spec.name, spec.version, &registered, spec.eval_type.code())?;
        Ok(Arc::new(Prepared {
            pool: self.pool.clone(),
            owner: self.owner.clone(),
            registered,
            return_type: spec.return_arrow.clone(),
            conf: self.runner_conf(),
            artifacts: self.artifacts(),
        }))
    }

    fn prepare_relation(
        &self,
        spec: &RelationCallSpec,
    ) -> krishiv_exec::Result<Arc<dyn PreparedRelationUdf>> {
        let registered = self.registered(&spec.name, spec.version)?;
        let session = self.runner_conf();
        crate::relation::prepare(
            self.pool.clone(),
            self.owner.clone(),
            Arc::clone(&self.slots),
            registered,
            spec,
            &session,
            self.artifacts(),
        )
    }
    fn analyze_udtf(
        &self,
        spec: &krishiv_exec::python_udtf::UdtfAnalyzeSpec,
    ) -> BoxFuture<'static, krishiv_exec::Result<krishiv_exec::python_udtf::UdtfAnalysis>> {
        crate::udtf::analyze(
            &self.pool,
            &self.owner,
            self.registered(&spec.name, spec.version),
            self.artifacts(),
            spec,
        )
    }
}

struct Prepared {
    pool: Pool,
    owner: Owner,
    registered: Registered,
    return_type: DataType,
    conf: Arc<BTreeMap<String, String>>,
    artifacts: Arc<ArtifactSet>,
}

/// A call's link to its future: dropping the future (the query was cancelled or failed) kills the worker
/// running it at once (design §3.4 "Cancel").
#[derive(Default)]
pub(crate) struct CallState {
    cancelled: bool,
    /// The pidfd of the worker running the call; cleared, under this lock, before the worker is given back, so
    /// a late cancel can never signal a worker that already serves another call (and a pidfd never names a
    /// reused pid, so a worker that exited on its own is safe too).
    running: Option<Arc<OwnedFd>>,
    /// The cancel killed `running` (a worker still starting, or one mid-call): it is never given back.
    killed: bool,
    /// The call's input until a call thread takes it; a cancel drops it, so a call still queued keeps no
    /// batch in memory (review 2026-11-14 m1).
    pub(crate) args: Option<RecordBatch>,
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

pub(crate) struct CancelOnDrop(pub(crate) Arc<Mutex<CallState>>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        state.cancelled = true;
        if let Some(pidfd) = state.running.take() {
            kill_via(&pidfd);
            state.killed = true;
        }
        let args = state.args.take();
        drop(state);
        drop(args);
    }
}

impl PreparedUdf for Prepared {
    fn call(&self, args: RecordBatch) -> BoxFuture<'static, krishiv_exec::Result<ArrayRef>> {
        let (done, result) = oneshot::channel();
        let call = Arc::new(Mutex::new(CallState {
            args: Some(args),
            ..CallState::default()
        }));
        let cancel = CancelOnDrop(Arc::clone(&call));
        let pool = self.pool.clone();
        let owner = self.owner.clone();
        let registered = self.registered.clone();
        let return_type = self.return_type.clone();
        let conf = Arc::clone(&self.conf);
        let artifacts = Arc::clone(&self.artifacts);
        // Each call blocks on its worker, so it runs on one of the pool's call threads, never on the caller
        // (a compute worker). Those threads are bounded by the worker caps (`crate::calls`).
        let submitted = self.pool.submit(
            &self.owner,
            Box::new(move || {
                let args = call
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .args
                    .take();
                let result = match args {
                    Some(args) => run(
                        &pool,
                        &owner,
                        &registered,
                        &artifacts,
                        &args,
                        &conf,
                        &return_type,
                        &call,
                    ),
                    // Cancelled while it was queued.
                    None => Err(OperationCanceled {}.into()),
                };
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

#[expect(
    clippy::too_many_arguments,
    reason = "one scalar call's parts, moved onto its call thread"
)]
fn run(
    pool: &Pool,
    owner: &Owner,
    registered: &Registered,
    artifacts: &ArtifactSet,
    args: &RecordBatch,
    conf: &BTreeMap<String, String>,
    return_type: &DataType,
    call: &Mutex<CallState>,
) -> Result<ArrayRef, KrishivError> {
    let max_result = pool.max_result_bytes();
    with_worker(pool, owner, registered, artifacts, call, |worker| {
        worker.run_batch(registered.id, args, conf, return_type, max_result)
    })
}

/// Runs `body` on a worker of `owner` that has `registered` loaded and reads `artifacts`, for the call `call`: a
/// cancel kills the
/// worker while it starts or while `body` runs, and a worker killed so is never given back (the call then
/// ends with `OPERATION_CANCELED`). The worker is given back (or stopped) before this returns.
pub(crate) fn with_worker<T>(
    pool: &Pool,
    owner: &Owner,
    registered: &Registered,
    artifacts: &ArtifactSet,
    call: &Mutex<CallState>,
    body: impl FnOnce(&mut Worker) -> Result<T, KrishivError>,
) -> Result<T, KrishivError> {
    let lock = || call.lock().unwrap_or_else(PoisonError::into_inner);
    let mut lease = match pool.checkout(owner, &registered.payload.python_ver, artifacts, call) {
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
    let result = worker
        .ensure_loaded(registered.id, &registered.payload)
        .and_then(|()| body(worker));
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

#[cfg(test)]
mod tests {
    use arrow_schema::{Field, Schema};
    use krishiv_common::error::ErrorClass;
    use krishiv_exec::python_relation::RunnerConf;
    use krishiv_types::DataType as SparkType;

    use super::*;
    use crate::pool::PoolConfig;

    /// The session's UDFs over a pool that cannot start workers: the checks run before any would start.
    fn udfs() -> SessionUdfs {
        let pool = Pool::new(PoolConfig {
            sandbox: false,
            executables: vec!["/nonexistent/krishiv-test-python".to_owned()],
            ..PoolConfig::default()
        })
        .unwrap();
        SessionUdfs::new(
            pool,
            Owner {
                principal: "p".to_owned(),
                session: "s".to_owned(),
                tenant: None,
                capped: true,
            },
        )
    }

    fn payload(eval_type: i32) -> PythonPayload {
        PythonPayload {
            command: Arc::from(&b"never loaded"[..]),
            eval_type,
            python_ver: "3.11".to_owned(),
        }
    }

    fn scalar(name: &str) -> UdfCallSpec {
        UdfCallSpec {
            name: name.to_owned(),
            version: 1,
            eval_type: PythonEvalType::ArrowBatched,
            thread_safe: true,
            arg_types: vec![SparkType::Long],
            return_type: SparkType::Long,
            return_arrow: DataType::Int64,
        }
    }

    fn relation(name: &str) -> RelationCallSpec {
        RelationCallSpec {
            name: name.to_owned(),
            version: 1,
            eval_type: PythonEvalType::MapArrowIter,
            thread_safe: true,
            input_types: vec![vec![SparkType::Long]],
            input_names: vec![vec!["v".to_owned()]],
            key_count: 0,
            arity: 1,
            output_names: vec!["v".to_owned()],
            output_types: vec![SparkType::Long],
            output_arrow: Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)])),
            conf: RunnerConf::default(),
            udtf: None,
        }
    }

    /// A scalar-registered UDF called as a relation, and a relation-registered one called as a scalar
    /// function, are refused with the class (the store finds payloads by name and version only).
    #[test]
    fn a_udf_called_as_another_kind_than_registered_is_refused() {
        let udfs = udfs();
        udfs.register("s", 1, payload(PythonEvalType::ArrowBatched.code()));
        udfs.register("m", 1, payload(PythonEvalType::MapArrowIter.code()));
        let e = udfs
            .prepare_relation(&relation("s"))
            .err()
            .expect("refused");
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivPythonUdfEvalTypeMismatch,
            "{e}"
        );
        assert!(e.message().contains("as a relation (eval type 207)"), "{e}");
        let e = udfs.prepare(&scalar("m")).err().expect("refused");
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivPythonUdfEvalTypeMismatch,
            "{e}"
        );
        assert!(
            e.message().contains("registered with eval type 207 but the plan calls it as a scalar function (eval type 101)"),
            "{e}"
        );
        // Each kind as registered prepares.
        assert!(udfs.prepare(&scalar("s")).is_ok());
        assert!(udfs.prepare_relation(&relation("m")).is_ok());
    }

    /// Over Connect a plain `@udf` is registered as `SQL_BATCHED_UDF` (100) and planned as the Arrow-optimized
    /// 101 (review R7c-1): any scalar payload fits a scalar call; a relation payload called as a scalar
    /// function is still refused, and a scalar payload called as a relation too.
    #[test]
    fn a_scalar_call_takes_any_scalar_payload_and_no_relation_payload() {
        let udfs = udfs();
        for (name, code) in [("b", 100), ("a", 101), ("p", 200), ("r", 250)] {
            udfs.register(name, 1, payload(code));
            assert!(udfs.prepare(&scalar(name)).is_ok(), "eval type {code}");
            let e = udfs
                .prepare_relation(&relation(name))
                .err()
                .expect("refused");
            assert_eq!(
                e.error_class(),
                ErrorClass::KrishivPythonUdfEvalTypeMismatch,
                "{e}"
            );
        }
        udfs.register("g", 1, payload(PythonEvalType::GroupedMapPandas.code()));
        let e = udfs.prepare(&scalar("g")).err().expect("refused");
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivPythonUdfEvalTypeMismatch,
            "{e}"
        );
        assert!(e.message().contains("registered with eval type 201"), "{e}");
    }

    /// The scalar set is exactly what `pool_worker.py`'s `make_wrapper` runs (`worker.py` `SQL_BATCHED_UDF`,
    /// `SQL_ARROW_BATCHED_UDF`, `SQL_SCALAR_PANDAS_UDF`, `SQL_SCALAR_ARROW_UDF`): a new scalar kind in
    /// `PythonEvalType` needs a wrapper there too (review N7d-1).
    #[test]
    fn the_scalar_eval_types_are_the_worker_s_wrappers() {
        let worker = include_str!("../worker/krishiv/_udf/worker.py");
        let mut wrapped: Vec<i32> = [
            "SQL_BATCHED_UDF",
            "SQL_ARROW_BATCHED_UDF",
            "SQL_SCALAR_PANDAS_UDF",
            "SQL_SCALAR_ARROW_UDF",
        ]
        .into_iter()
        .map(|name| {
            let prefix = format!("{name} = ");
            worker
                .lines()
                .find_map(|l| l.strip_prefix(prefix.as_str()))
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or_else(|| panic!("worker.py defines {name}"))
        })
        .collect();
        wrapped.sort_unstable();
        let derived: Vec<i32> = (0..400).filter(|c| runs_as_scalar(*c)).collect();
        assert_eq!(derived, wrapped);
        assert_eq!(derived, [100, 101, 200, 250]);
    }
}
