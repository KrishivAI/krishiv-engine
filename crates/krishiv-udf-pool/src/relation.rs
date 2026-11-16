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
//! frame leaves frames the operator still sends, which are dropped. A failure (a Python error, a crashed worker,
//! a refused result) is what `next` gives after the results read so far. Dropping the task cancels it as dropping a scalar call's
//! future does: the worker is killed while it holds one, and never reused.

use std::collections::BTreeMap;
use std::future::poll_fn;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::Poll;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::channel::mpsc;
use futures::executor::block_on;
use futures::future::BoxFuture;
use futures::{SinkExt, StreamExt};
use krishiv_common::error::{InternalError, KrishivError, OperationCanceled};
use krishiv_exec::python_relation::{Frame, PreparedRelationUdf, RelationCallSpec, UdfTask};

use crate::pool::{Owner, Pool};
use crate::runtime::{CallState, CancelOnDrop, Registered, with_worker};
use crate::worker::{TaskEvent, Worker};

/// The eval types of the relation kinds the worker runs (`relations.py`): map 205, 207; grouped map 201, 209,
/// 216, 215; cogrouped map 206, 210; grouped aggregate 202, 217, 252, 254. The window kinds and UDTFs are
/// PR-26c-3's.
const KINDS: [i32; 12] = [205, 207, 201, 209, 216, 215, 206, 210, 202, 217, 252, 254];

/// The prepared call site of `spec` over `registered`, run on `pool`'s workers of `owner`. `session` is the
/// session's runner conf; the plan's own settings (`spec.conf`) override it.
pub(crate) fn prepare(
    pool: Pool,
    owner: Owner,
    registered: Registered,
    spec: &RelationCallSpec,
    session: &BTreeMap<String, String>,
) -> Result<Arc<dyn PreparedRelationUdf>, KrishivError> {
    if !KINDS.contains(&spec.eval_type.code())
        || spec.eval_type.code() != registered.payload.eval_type
    {
        return Err(InternalError {
            message: format!(
                "the Python UDF {} v{} is registered with eval type {} and called as a relation of eval type {}",
                spec.name,
                spec.version,
                registered.payload.eval_type,
                spec.eval_type.code()
            ),
        }
        .into());
    }
    let mut conf = session.clone();
    conf.extend(
        spec.conf
            .settings
            .iter()
            .map(|(k, v)| (k.clone(), v.clone())),
    );
    Ok(Arc::new(PreparedRelation {
        pool,
        owner,
        registered,
        site: Arc::new(Site {
            key_count: spec.key_count,
            arity: spec.arity,
            max_rows: spec.conf.max_records_per_batch,
            output: Arc::clone(&spec.output_arrow),
            conf,
        }),
    }))
}

/// What every task of one call site sends its worker.
struct Site {
    key_count: usize,
    arity: usize,
    max_rows: usize,
    output: SchemaRef,
    conf: BTreeMap<String, String>,
}

struct PreparedRelation {
    pool: Pool,
    owner: Owner,
    registered: Registered,
    site: Arc<Site>,
}

/// What the task's thread passes the operator: its result batches (a failure goes through `Task::failure`).
type Results = RecordBatch;

impl PreparedRelationUdf for PreparedRelation {
    fn open(&self) -> Result<Box<dyn UdfTask>, KrishivError> {
        // One frame waits for the thread; one result waits for the operator (plus one each in hand).
        let (frames_tx, frames_rx) = mpsc::channel::<Frame>(0);
        let (results_tx, results_rx) = mpsc::channel::<Results>(0);
        let failure = Arc::new(Mutex::new(None));
        let failed = Arc::clone(&failure);
        let call = Arc::new(Mutex::new(CallState::default()));
        let cancel = CancelOnDrop(Arc::clone(&call));
        let pool = self.pool.clone();
        let owner = self.owner.clone();
        let registered = self.registered.clone();
        let site = Arc::clone(&self.site);
        self.pool.submit(
            &self.owner,
            Box::new(move || {
                let mut frames = frames_rx;
                let mut results = results_tx;
                let outcome = with_worker(&pool, &owner, &registered, &call, |worker| {
                    drive(worker, &pool, &registered, &site, &mut frames, &mut results)
                });
                // The worker is given back (or stopped) before the results end, so a task dropped once its
                // results ended never kills a worker that finished well.
                if let Err(e) = outcome {
                    *failed.lock().unwrap_or_else(PoisonError::into_inner) = Some(e);
                }
                // Dropping `results` ends them after what the operator has not read yet.
                drop(results);
            }),
        )?;
        Ok(Box::new(Task {
            _cancel: cancel,
            frames: Mutex::new(frames_tx),
            results: Mutex::new(results_rx),
            failure,
        }))
    }
}

/// Runs one task on `worker`: frames on request, results as they come, to the worker's `done`.
fn drive(
    worker: &mut Worker,
    pool: &Pool,
    registered: &Registered,
    site: &Site,
    frames: &mut mpsc::Receiver<Frame>,
    results: &mut mpsc::Sender<Results>,
) -> Result<(), KrishivError> {
    let max_result = pool.max_result_bytes();
    worker.start_task(
        registered.id,
        site.key_count,
        site.arity,
        site.max_rows,
        &site.conf,
    )?;
    loop {
        match worker.task_event(&site.output, max_result)? {
            TaskEvent::Want => {
                let Some(frame) = block_on(frames.next()) else {
                    // The task was dropped: the worker waits for a frame that never comes.
                    worker.abandon();
                    return Err(OperationCanceled {}.into());
                };
                worker.send_frame(&frame)?;
            }
            TaskEvent::Result(batch) => {
                if block_on(results.send(batch)).is_err() {
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
    failure: Arc<Mutex<Option<KrishivError>>>,
}

impl UdfTask for Task {
    fn send(&self, frame: Frame) -> BoxFuture<'_, krishiv_exec::Result<()>> {
        let mut frame = Some(frame);
        Box::pin(poll_fn(move |cx| {
            let mut frames = self.frames.lock().unwrap_or_else(PoisonError::into_inner);
            match frames.poll_ready(cx) {
                Poll::Pending => Poll::Pending,
                // The task ended (its UDF stopped reading, or it failed: `next` gives the error): the frame is
                // dropped.
                Poll::Ready(Err(_)) => Poll::Ready(Ok(())),
                Poll::Ready(Ok(())) => {
                    if let Some(frame) = frame.take() {
                        let _ = frames.start_send(frame);
                    }
                    Poll::Ready(Ok(()))
                }
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
                None => match self
                    .failure
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take()
                {
                    Some(e) => Err(e),
                    None => Ok(None),
                },
            })
        }))
    }
}
