// SPDX-License-Identifier: Apache-2.0
//! The relation kinds on the pool (P3-T014 PR-26c-2; design `P3-T085-embedded-python-udf-kinds.md` §3.2): a
//! prepared relation call site opens one [`UdfTask`] per driver, and each task runs on a pool call thread that
//! holds one worker for the task's life.
//!
//! The task's thread speaks wire `task` (`pool_worker.py`'s module doc): it answers each `want` with the next
//! [`Frame`] the operator sent, and passes each `result` back, checked against the declared schema. The server
//! writes only after a `want`, so the worker and the server never both block writing. The operator's side is
//! two bounded channels: `send` waits while the thread has not taken the previous frame, and the thread waits
//! while the operator has not read its results (back pressure into the worker).
//!
//! **Ends.** `done` ends the results (`next` then gives `None`); a UDF that stopped reading before the last
//! frame leaves frames the operator still sends, which are dropped (`send` gives `Ok`). A failure (a Python
//! error, a crashed worker, a refused result) is what `next` gives after the results read so far, and what
//! `send` gives once the task ended. A task whose thread ended any other way (its call panicked, or the call
//! was dropped unrun) fails with `INTERNAL_ERROR`, never a clean end that would lose rows ([`Ending`]).
//! Dropping the task cancels it as dropping a scalar call's future does: the worker is killed while it holds
//! one, and never reused.
//!
//! **Slots** (design `P3-T085-embedded-python-udf-kinds.md` §3.3a). A task holds a worker for its whole life, so
//! it is opened in one of its owner's relation slots ([`PoolSlots`], the call site's capacity hint), taken by
//! the operator for all of a driver's tasks at once, and goes ahead of the queued calls
//! ([`Pool::submit_task`]): never behind its owner's own tasks.
//!
//! **Memory.** `send` resolves once the task's thread has taken the frame, so the operator's reservation
//! covers it until then; the thread's copy into a memfd, and up to two decoded result batches (one in the
//! channel, one the thread waits to hand over), are outside the operator's accounting.

use std::collections::BTreeMap;
use std::future::poll_fn;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::channel::mpsc;
use futures::executor::block_on;
use futures::future::BoxFuture;
use futures::{SinkExt, StreamExt};
use krishiv_common::error::{InternalError, KrishivError, OperationCanceled};
use krishiv_exec::python_relation::{
    Frame, PreparedRelationUdf, RelationCallSpec, RelationSlot, RelationSlots, RelationTicket,
    UdfTask,
};
use krishiv_types::{StructField, StructType};

use crate::artifacts::ArtifactSet;
use crate::calls::{SlotTake, TaskSlot, TaskTicket};
use crate::pool::{Owner, Pool};
use crate::runtime::{CallState, CancelOnDrop, Registered, with_worker};
use crate::worker::{TaskEvent, Worker};
use krishiv_exec::python_relation::{SchemaCheck, VALIDATE_SCHEMA};

/// The eval types of the relation kinds the worker runs (`relations.py`): map 205, 207; grouped map 201, 209,
/// 216, 215; cogrouped map 206, 210; grouped aggregate 202, 217, 252, 254; window aggregate 203, 253; scalar
/// iterator 204, 251; table functions (UDTFs) 300, 301, 302 (`udtf.py`).
const KINDS: [i32; 19] = [
    205, 207, 201, 209, 216, 215, 206, 210, 202, 217, 252, 254, 203, 253, 204, 251, 300, 301, 302,
];

/// The UDTF eval types: a task of one needs the call's [`RelationCallSpec::udtf`].
const UDTF_KINDS: [i32; 3] = [300, 301, 302];

/// Whether a UDF registered with eval type `registered` runs as `called`: its own kind, or a grouped-aggregate
/// UDF over a window, which Spark runs as the window kind (`ArrowWindowPythonExec`, v4.2.0: 202 as 203, 252 as
/// 253; the client registers the grouped-aggregate code).
fn runs_as(registered: i32, called: i32) -> bool {
    registered == called || matches!((registered, called), (202, 203) | (252, 253))
}

/// The relation slots of one owner on one pool: every call site of a session shares one, so a driver's lanes
/// take theirs together.
pub(crate) struct PoolSlots {
    pool: Pool,
    owner: Owner,
}

impl PoolSlots {
    pub(crate) fn new(pool: Pool, owner: Owner) -> Self {
        PoolSlots { pool, owner }
    }
}

impl RelationSlots for PoolSlots {
    fn limit(&self) -> usize {
        self.pool.calls().slot_limit()
    }

    fn poll_take(
        &self,
        n: usize,
        ticket: &mut Option<RelationTicket>,
        cx: &mut Context<'_>,
    ) -> Poll<krishiv_exec::Result<Vec<RelationSlot>>> {
        // The pool's ticket out of the operator's box and back (moved, never dropped, so it keeps its place).
        let mut own: Option<TaskTicket> = ticket
            .take()
            .and_then(|t| t.into_inner().downcast::<TaskTicket>().ok())
            .map(|t| *t);
        let taken = self
            .pool
            .calls()
            .take_slots(&self.owner, n, &mut own, cx.waker());
        *ticket = own.map(RelationTicket::new);
        match taken {
            (SlotTake::Taken, slots) => {
                Poll::Ready(Ok(slots.into_iter().map(RelationSlot::new).collect()))
            }
            (SlotTake::Wait, _) => Poll::Pending,
            (SlotTake::TooMany, _) => Poll::Ready(Err(self.pool.worker_limit(
                "A Python UDF relation operator (one task per UDF of a driver)",
                n,
            ))),
        }
    }
}

/// The prepared call site of `spec` over `registered`, run on `pool`'s workers of `owner` in `slots` (the
/// owner's). `session` is the session's runner conf; the plan's own settings (`spec.conf`) override it.
pub(crate) fn prepare(
    pool: Pool,
    owner: Owner,
    slots: Arc<PoolSlots>,
    registered: Registered,
    spec: &RelationCallSpec,
    session: &BTreeMap<String, String>,
    artifacts: Arc<ArtifactSet>,
) -> Result<Arc<dyn PreparedRelationUdf>, KrishivError> {
    if !KINDS.contains(&spec.eval_type.code()) {
        // The executor called a kind this runtime does not run as a relation: its own inconsistency.
        return Err(InternalError {
            message: format!(
                "the Python UDF {} v{} is called as a relation of eval type {}, which the pool does not run",
                spec.name,
                spec.version,
                spec.eval_type.code()
            ),
        }
        .into());
    }
    crate::runtime::check_eval_type(
        &spec.name,
        spec.version,
        &registered,
        spec.eval_type.code(),
        "a relation",
        runs_as,
    )?;
    let mut conf = session.clone();
    conf.extend(
        spec.conf
            .settings
            .iter()
            .map(|(k, v)| (k.clone(), v.clone())),
    );
    let validate = VALIDATE_SCHEMA
        .iter()
        .find_map(|k| conf.get(*k))
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("true"));
    let declared = StructType::new(
        spec.output_names
            .iter()
            .zip(&spec.output_types)
            .zip(spec.output_arrow.fields())
            .map(|((n, t), f)| StructField::new(n.clone(), t.clone(), f.is_nullable())),
    );
    let udtf = udtf_message(spec, &declared)?;
    // A lateral UDTF's result leads with the input columns it carries; its own result is the rest.
    let declared = match &spec.udtf {
        Some(call) if !call.passthrough.is_empty() => StructType::new(
            declared
                .fields()
                .iter()
                .skip(call.passthrough.len())
                .cloned(),
        ),
        _ => declared,
    };
    let check = SchemaCheck::new(spec.eval_type.code(), declared, validate);
    Ok(Arc::new(PreparedRelation {
        pool,
        owner,
        slots,
        registered,
        artifacts,
        site: Arc::new(Site {
            eval_type: spec.eval_type.code(),
            key_count: spec.key_count,
            arity: spec.arity,
            max_rows: spec.conf.max_records_per_batch,
            output: Arc::clone(&spec.output_arrow),
            check,
            conf,
            udtf,
        }),
    }))
}

/// The task message's `udtf` part for a UDTF call site (`udtf.py` `UdtfCall`): the arguments, the partition
/// child indexes, the `TABLE` argument offsets, the declared result as Spark's JSON and `analyze`'s pickled
/// result (hex); `None` for every other kind. A UDTF call without its call data is the executor's own
/// inconsistency.
fn udtf_message(
    spec: &RelationCallSpec,
    declared: &StructType,
) -> Result<Option<serde_json::Value>, KrishivError> {
    let is_udtf = UDTF_KINDS.contains(&spec.eval_type.code());
    match (&spec.udtf, is_udtf) {
        (None, false) => Ok(None),
        (Some(call), true) => Ok(Some(serde_json::json!({
            "name": spec.name,
            "return_type": krishiv_types::DataType::Struct(StructType::new(
                declared.fields().iter().skip(call.passthrough.len()).cloned()
            ))
            .json(),
            "passthrough": call.passthrough,
            "args": call
                .args
                .iter()
                .map(|a| serde_json::json!([a.offset, a.name]))
                .collect::<Vec<_>>(),
            "partition_child_indexes": call.partition_child_indexes,
            "table_arg_offsets": call.table_arg_offsets,
            "analyze_result": call.analyze_result.as_deref().map(hex),
        }))),
        _ => Err(InternalError {
            message: format!(
                "the Python UDF {} v{} is called as eval type {} {} a table function's call data",
                spec.name,
                spec.version,
                spec.eval_type.code(),
                if is_udtf { "without" } else { "with" }
            ),
        }
        .into()),
    }
}

/// `bytes` as lower-case hex (the wire's form of a pickled `AnalyzeResult`).
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// What every task of one call site sends its worker.
struct Site {
    /// The kind the task runs (a window kind for a grouped-aggregate UDF over a window).
    eval_type: i32,
    key_count: usize,
    arity: usize,
    max_rows: usize,
    output: SchemaRef,
    /// Spark's result-schema check of a map kind, when the session turned it on.
    check: Option<SchemaCheck>,
    conf: BTreeMap<String, String>,
    /// A UDTF call's part of the task message (`udtf_message`).
    udtf: Option<serde_json::Value>,
}

struct PreparedRelation {
    pool: Pool,
    owner: Owner,
    slots: Arc<PoolSlots>,
    registered: Registered,
    artifacts: Arc<ArtifactSet>,
    site: Arc<Site>,
}

/// What the task's thread passes the operator: its result batches (a failure goes through `Task::failure`).
type Results = RecordBatch;

/// A task's failure, set by its thread before its results end.
type Failure = Arc<Mutex<Option<KrishivError>>>;

/// The thread's ends of a task's channels. Dropped (the job done, panicked, or dropped unrun) it records
/// `INTERNAL_ERROR` unless the task ended well or its failure is set, and only then closes the channels, so
/// the operator never takes a lost end for a clean one.
struct Ending {
    frames: mpsc::Receiver<Frame>,
    results: mpsc::Sender<Results>,
    failure: Failure,
    /// The worker sent `done`.
    done: bool,
}

impl Drop for Ending {
    fn drop(&mut self) {
        if !self.done {
            self.failure
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get_or_insert_with(|| {
                    InternalError {
                        message: "a Python UDF task ended without finishing (its call panicked or never ran)"
                            .to_owned(),
                    }
                    .into()
                });
        }
    }
}

/// A task's two ends: the operator's and its thread's.
fn task_channels(cancel: CancelOnDrop) -> (Task, Ending) {
    // One frame waits for the thread; one result waits for the operator (plus one each in hand).
    let (frames_tx, frames_rx) = mpsc::channel::<Frame>(0);
    let (results_tx, results_rx) = mpsc::channel::<Results>(0);
    let failure: Failure = Arc::new(Mutex::new(None));
    let task = Task {
        _cancel: cancel,
        frames: Mutex::new(frames_tx),
        results: Mutex::new(results_rx),
        failure: Arc::clone(&failure),
    };
    let ending = Ending {
        frames: frames_rx,
        results: results_tx,
        failure,
        done: false,
    };
    (task, ending)
}

impl PreparedRelationUdf for PreparedRelation {
    fn open(&self, slot: Option<RelationSlot>) -> Result<Box<dyn UdfTask>, KrishivError> {
        let slot = self.slot(slot)?;
        let call = Arc::new(Mutex::new(CallState::default()));
        let (task, mut ending) = task_channels(CancelOnDrop(Arc::clone(&call)));
        let pool = self.pool.clone();
        let owner = self.owner.clone();
        let registered = self.registered.clone();
        let artifacts = Arc::clone(&self.artifacts);
        let site = Arc::clone(&self.site);
        self.pool.submit_task(
            &self.owner,
            Box::new(move || {
                let outcome =
                    with_worker(&pool, &owner, &registered, &artifacts, &call, |worker| {
                        drive(worker, &pool, &registered, &site, &mut ending)
                    });
                // The worker is given back: the slot frees before the operator can see the results end, so a
                // scalar call the plan makes next never finds it still held (review R2-5).
                drop(slot);
                // The worker is given back (or stopped) before the results end, so a task dropped once its
                // results ended never kills a worker that finished well.
                match outcome {
                    Ok(()) => ending.done = true,
                    Err(e) => {
                        *ending
                            .failure
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner) = Some(e);
                    }
                }
                // Dropping `ending` ends the results after what the operator has not read yet.
                drop(ending);
            }),
        )?;
        Ok(Box::new(task))
    }

    fn capacity(&self) -> Option<Arc<dyn RelationSlots>> {
        Some(Arc::clone(&self.slots) as Arc<dyn RelationSlots>)
    }
}

impl PreparedRelation {
    /// The task's slot: the operator's, else one taken now if one is free (a caller that did not take one
    /// first, as a test may do).
    fn slot(&self, slot: Option<RelationSlot>) -> Result<TaskSlot, KrishivError> {
        if let Some(slot) = slot
            && let Ok(slot) = slot.into_inner().downcast::<TaskSlot>()
            && slot.is_of(self.pool.calls(), &self.owner)
        {
            return Ok(*slot);
        }
        self.pool.calls().try_take_now(&self.owner).ok_or_else(|| {
            InternalError {
                message: "a Python UDF task was opened without a free relation slot of its session"
                    .to_owned(),
            }
            .into()
        })
    }
}

/// Runs one task on `worker`: frames on request, results as they come, to the worker's `done`.
fn drive(
    worker: &mut Worker,
    pool: &Pool,
    registered: &Registered,
    site: &Site,
    ending: &mut Ending,
) -> Result<(), KrishivError> {
    let max_result = pool.max_result_bytes();
    worker.start_task(
        registered.id,
        site.eval_type,
        site.key_count,
        site.arity,
        site.max_rows,
        &site.conf,
        site.udtf.as_ref(),
    )?;
    // `End` was sent: the worker has no frame left to ask for.
    let mut ended = false;
    loop {
        match worker.task_event(&site.output, site.check.as_ref(), max_result)? {
            TaskEvent::Want if ended => {
                return Err(worker.broke("a frame asked for after the end"));
            }
            TaskEvent::Want => {
                let Some(frame) = block_on(ending.frames.next()) else {
                    // The task was dropped: the worker waits for a frame that never comes.
                    worker.abandon();
                    return Err(OperationCanceled {}.into());
                };
                ended = matches!(frame, Frame::End);
                worker.send_frame(&frame)?;
            }
            TaskEvent::Result(batch) => {
                if block_on(ending.results.send(batch)).is_err() {
                    worker.abandon();
                    return Err(OperationCanceled {}.into());
                }
            }
            TaskEvent::Done => return Ok(()),
        }
    }
}

/// The operator's side of a running task. `_cancel` is declared first, so it drops first: the call is marked
/// cancelled (and its worker killed) before the thread can see the frames end.
struct Task {
    _cancel: CancelOnDrop,
    frames: Mutex<mpsc::Sender<Frame>>,
    results: Mutex<mpsc::Receiver<Results>>,
    /// The task's failure, set by its thread before the results end.
    failure: Failure,
}

impl Task {
    /// A copy of the task's failure, if it failed (`send` and `next` both give it).
    fn failed(&self) -> Option<KrishivError> {
        self.failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl UdfTask for Task {
    /// Resolves once the task's thread has taken the frame (the channel's one slot is free again), so the
    /// operator keeps the frame's reservation until then. Once the task ended, the frame is dropped: `Ok`
    /// after `done` (its UDF stopped reading), else the task's failure.
    fn send(&self, frame: Frame) -> BoxFuture<'_, krishiv_exec::Result<()>> {
        let mut frame = Some(frame);
        Box::pin(poll_fn(move |cx| {
            let mut frames = self.frames.lock().unwrap_or_else(PoisonError::into_inner);
            // Before the frame is queued: room for it; after: the thread took it.
            match frames.poll_ready(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(Err(_)) => Poll::Ready(self.failed().map_or(Ok(()), Err)),
                Poll::Ready(Ok(())) => match frame.take() {
                    Some(f) => {
                        if frames.start_send(f).is_err() {
                            return Poll::Ready(self.failed().map_or(Ok(()), Err));
                        }
                        // Taken already, or wait (`poll_ready` registered the waker again).
                        match frames.poll_ready(cx) {
                            Poll::Pending => Poll::Pending,
                            Poll::Ready(Err(_)) => Poll::Ready(self.failed().map_or(Ok(()), Err)),
                            Poll::Ready(Ok(())) => Poll::Ready(Ok(())),
                        }
                    }
                    None => Poll::Ready(Ok(())),
                },
            }
        }))
    }

    fn next(&self) -> BoxFuture<'_, krishiv_exec::Result<Option<RecordBatch>>> {
        Box::pin(poll_fn(move |cx| {
            let next = self
                .results
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .poll_next_unpin(cx);
            next.map(|item| match item {
                Some(batch) => Ok(Some(batch)),
                None => self.failed().map_or(Ok(None), Err),
            })
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::Waker;
    use std::time::Duration;

    use krishiv_common::error::ErrorClass;

    use super::*;
    use crate::pool::PoolConfig;

    fn cancel() -> CancelOnDrop {
        CancelOnDrop(Arc::new(Mutex::new(CallState::default())))
    }

    #[test]
    fn a_task_whose_thread_ended_without_done_fails_instead_of_ending_clean() {
        // Dropped unrun (or after a panic): never a clean end that loses rows (review finding 2).
        let (task, ending) = task_channels(cancel());
        drop(ending);
        let e = block_on(task.next()).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
        let (task, ending) = task_channels(cancel());
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _ending = ending;
            panic!("arrow-ipc panicked");
        }));
        assert!(panicked.is_err());
        let e = block_on(task.next()).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
        // `send` reports it too.
        let e = block_on(task.send(Frame::End)).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
    }

    #[test]
    fn after_done_frames_are_dropped_and_the_results_end() {
        let (task, mut ending) = task_channels(cancel());
        ending.done = true;
        drop(ending);
        block_on(task.send(Frame::End)).unwrap();
        assert!(block_on(task.next()).unwrap().is_none());
    }

    #[test]
    fn send_resolves_once_the_thread_took_the_frame() {
        let (task, mut ending) = task_channels(cancel());
        let mut send = task.send(Frame::EndGroup);
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        assert!(send.as_mut().poll(&mut cx).is_pending(), "not taken yet");
        assert!(matches!(
            block_on(ending.frames.next()),
            Some(Frame::EndGroup)
        ));
        assert!(matches!(send.as_mut().poll(&mut cx), Poll::Ready(Ok(()))));
    }

    /// A pool that cannot start workers (no interpreter): enough for the slots.
    fn pool(max_per_session: usize) -> Pool {
        Pool::new(PoolConfig {
            sandbox: false,
            executables: vec!["/nonexistent/krishiv-test-python".to_owned()],
            max_per_session,
            ..PoolConfig::default()
        })
        .unwrap()
    }

    fn owner() -> Owner {
        Owner {
            principal: "p".to_owned(),
            session: "s".to_owned(),
        }
    }

    struct Count(AtomicUsize);

    impl std::task::Wake for Count {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn more_tasks_than_the_session_s_slots_fail_at_once_naming_the_setting() {
        for (max, limit) in [(1, 1), (2, 1), (8, 7)] {
            let slots = PoolSlots::new(pool(max), owner());
            assert_eq!(slots.limit(), limit);
            let count = Arc::new(Count(AtomicUsize::new(0)));
            let waker = Waker::from(Arc::clone(&count));
            let mut cx = Context::from_waker(&waker);
            let Poll::Ready(Err(e)) = slots.poll_take(limit + 1, &mut None, &mut cx) else {
                panic!("{} tasks over {limit} slots must fail", limit + 1);
            };
            assert_eq!(e.error_class(), ErrorClass::KrishivPythonWorkerLimit);
            assert!(
                e.message().contains("krishiv.python.worker.maxPerSession")
                    && e.message().contains(&max.to_string()),
                "{}",
                e.message()
            );
            let Poll::Ready(Ok(taken)) = slots.poll_take(limit, &mut None, &mut cx) else {
                panic!("the slots are free");
            };
            let mut ticket = None;
            assert!(
                slots.poll_take(1, &mut ticket, &mut cx).is_pending(),
                "would block until a slot frees"
            );
            drop(taken);
            assert!(
                count.0.load(Ordering::SeqCst) >= 1,
                "a freed slot wakes the waiter"
            );
        }
    }

    #[test]
    fn with_one_worker_per_session_a_scalar_call_beside_an_open_task_fails_fast() {
        let pool = pool(1);
        let slots = PoolSlots::new(pool.clone(), owner());
        let waker = futures::task::noop_waker();
        let Poll::Ready(Ok(_taken)) =
            slots.poll_take(1, &mut None, &mut Context::from_waker(&waker))
        else {
            panic!("the slot is free");
        };
        let (done, ran) = std::sync::mpsc::channel::<()>();
        let e = pool
            .submit(
                &owner(),
                Box::new(move || {
                    let _ = done.send(());
                }),
            )
            .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivPythonWorkerLimit);
        assert!(
            ran.recv_timeout(Duration::from_millis(100)).is_err(),
            "never queued"
        );
    }
}
