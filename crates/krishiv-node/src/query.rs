// SPDX-License-Identifier: Apache-2.0
//! [`NativeQuery`]: one statement's result as a stream of Arrow IPC chunks.
//! The first `next()` analyzes and starts the statement and returns the schema
//! message; each later one a record batch, then the end-of-stream marker, then
//! `null`. `cancel()` ends it at any point: dropping an action's future or
//! result stream cancels its query (krishiv-session invariant).

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::StreamExt;
use futures::future::{Either, select};
use krishiv_common::error::{KrishivError, OperationCanceled};
use krishiv_session::{RecordBatchStream, Session};
use napi::bindgen_prelude::Buffer;
use napi_derive::napi;
use tokio::sync::{Mutex, Notify};

use crate::error::{guarded_async, napi_guarded, to_napi};
use crate::ipc::Encoder;

/// Where a query is.
enum State {
    /// Not analyzed yet: the statement text.
    Pending(String),
    /// Running: the batches still to come and their encoder.
    Streaming(Box<Running>),
    /// Cancelled while no step ran: the next step rejects, then the query is `Done`.
    Cancelled,
    /// Ended (finished, failed, or cancelled and told).
    Done,
}

/// A running query's stream and encoder.
struct Running {
    stream: RecordBatchStream,
    encoder: Encoder,
}

/// A cancellation that wakes the step waiting on it.
#[derive(Debug, Default)]
pub(crate) struct Cancel {
    set: AtomicBool,
    wake: Notify,
}

impl Cancel {
    fn trip(&self) {
        self.set.store(true, Ordering::Release);
        self.wake.notify_waiters();
    }

    fn is_set(&self) -> bool {
        self.set.load(Ordering::Acquire)
    }

    /// Runs `work` unless the query is cancelled first (or meanwhile: `work` is then dropped).
    pub(crate) async fn guard<T>(
        &self,
        work: impl Future<Output = Result<T, KrishivError>>,
    ) -> Result<T, KrishivError> {
        let notified = pin!(self.wake.notified());
        let mut notified = notified;
        // Registered before the flag is read, so a cancel between the two still wakes it.
        notified.as_mut().enable();
        if self.is_set() {
            return Err(OperationCanceled {}.into());
        }
        match select(pin!(work), notified).await {
            Either::Left((result, _)) => result,
            Either::Right(((), _)) => Err(OperationCanceled {}.into()),
        }
    }
}

/// The running statement behind a JavaScript `NativeQuery`.
#[napi]
pub struct NativeQuery {
    session: Session,
    state: Arc<Mutex<State>>,
    cancel: Arc<Cancel>,
}

impl std::fmt::Debug for NativeQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeQuery")
            .field("cancelled", &self.cancel.is_set())
            .finish_non_exhaustive()
    }
}

impl NativeQuery {
    pub(crate) fn new(session: Session, sql: String) -> Self {
        Self {
            session,
            state: Arc::new(Mutex::new(State::Pending(sql))),
            cancel: Arc::default(),
        }
    }

    /// The next chunk, `None` at the end. A failure or a cancel ends the query.
    pub(crate) async fn step(&self) -> Result<Option<Vec<u8>>, KrishivError> {
        let mut state = self.state.lock().await;
        match *state {
            // An ended query answers `null`, even after the cancel that ended it.
            State::Done => return Ok(None),
            State::Cancelled => {
                *state = State::Done;
                return Err(OperationCanceled {}.into());
            }
            State::Pending(_) | State::Streaming(_) => {}
        }
        let result = self.cancel.guard(advance(&self.session, &mut state)).await;
        if result.is_err() {
            // Drops the stream (and with it the query) if the guard stopped the work.
            *state = State::Done;
        }
        result
    }

    pub(crate) fn stop(&self) {
        self.cancel.trip();
        // Not stepping now: end it here. Stepping: the step sees the cancel and ends it.
        if let Ok(mut state) = self.state.try_lock()
            && !matches!(*state, State::Done)
        {
            *state = State::Cancelled;
        }
    }
}

/// One step of `state`.
async fn advance(session: &Session, state: &mut State) -> Result<Option<Vec<u8>>, KrishivError> {
    match std::mem::replace(state, State::Done) {
        State::Pending(sql) => {
            let frame = session.sql_boxed(&sql).await?;
            let schema = frame.schema_boxed().await?;
            let stream = frame.execute_stream_boxed().await?;
            let (encoder, head) = Encoder::start(&schema)?;
            *state = State::Streaming(Box::new(Running { stream, encoder }));
            Ok(Some(head))
        }
        State::Streaming(mut running) => match running.stream.next().await {
            Some(batch) => {
                let chunk = running.encoder.batch(&batch?)?;
                *state = State::Streaming(running);
                Ok(Some(chunk))
            }
            None => Ok(Some(running.encoder.finish()?)),
        },
        State::Cancelled | State::Done => Ok(None),
    }
}

#[napi]
impl NativeQuery {
    /// The next Arrow IPC chunk of the result (schema first, then one per batch, then the
    /// end-of-stream marker), or `null` once the stream has ended.
    #[napi]
    pub async fn next(&self) -> napi::Result<Option<Buffer>> {
        guarded_async(self.step())
            .await
            .map(|chunk| chunk.map(Buffer::from))
            .map_err(|e| to_napi(&e))
    }

    /// Cancels the query, before or while it streams: the next `next()` rejects with
    /// `OPERATION_CANCELED` unless the stream had already ended.
    #[napi]
    pub fn cancel(&self) -> napi::Result<()> {
        napi_guarded(|| {
            self.stop();
            Ok(())
        })
    }
}
