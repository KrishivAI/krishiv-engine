// SPDX-License-Identifier: Apache-2.0
//! One lane of a relation sink's driver (design P3-T085 §3.2): the frames
//! queued for a [`UdfTask`], the results it yielded, and the future (the
//! *pump*) that moves both while the driver, its `Blocked` future or
//! `finalize` polls it.
//!
//! **Memory (I11).** A queued frame holds the reservation of its rows (the
//! input chunk's, shared by every lane's frame of that chunk and released by
//! the last task that takes it) and of the copies made for its client form.
//! Every result batch is reserved in the driver's consumer as the pump
//! receives it; when the pool refuses, the batch waits in [`LaneIo::refused`]
//! and the pump reads no more results until the driver has taken it
//! (`sink` returns `NeedMemory`; `finalize` stores a map or grouped result
//! as the others, written out when the pool is full, and waits for an
//! aggregate value's memory while the other lanes move): back pressure into
//! the task, never unaccounted results.

use std::collections::VecDeque;
use std::future::poll_fn;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_memory::Reservation;

use super::{Frame, PreparedRelationUdf, UdfTask};
use crate::error::Result;
use crate::sync::{Mutex, lock};

/// The memory a queued frame holds until its task takes it.
pub(super) struct FrameMem {
    /// The input chunk's rows, shared by the lanes' frames of the chunk.
    pub(super) _shared: Option<Arc<Mutex<Reservation>>>,
    /// The copies made for this lane's frame (client forms).
    pub(super) _own: Reservation,
}

/// A lane's queues, shared by the driver and the lane's pump.
pub(super) struct LaneIo {
    /// Frames not taken by the task yet, each with its memory.
    pub(super) inbound: VecDeque<(Frame, FrameMem)>,
    /// The memory of the frame the task is taking.
    pub(super) sending: Option<FrameMem>,
    /// Batches the task yielded, each with its reservation, not stored yet.
    pub(super) outputs: VecDeque<(RecordBatch, Reservation)>,
    /// A batch the pool had no memory for: the pump reads no more results
    /// until the driver reserved it and moved it to [`Self::outputs`].
    pub(super) refused: Option<RecordBatch>,
    /// An empty reservation of the driver's consumer, the results' are
    /// split from.
    pub(super) memory: Reservation,
}

impl LaneIo {
    pub(super) fn in_flight(&self) -> usize {
        self.inbound.len() + usize::from(self.sending.is_some())
    }

    /// Whether results wait for the driver (stored or reserved).
    pub(super) fn has_results(&self) -> bool {
        !self.outputs.is_empty() || self.refused.is_some()
    }
}

/// The future that moves a lane's frames to its task and its results back.
enum Pump {
    /// No frame yet: the task is not open.
    Idle,
    Running(BoxFuture<'static, Result<()>>),
    Done(Result<()>),
}

/// One lane of one driver.
#[derive(Clone)]
pub(super) struct Lane {
    pub(super) io: Arc<Mutex<LaneIo>>,
    pump: Arc<Mutex<Pump>>,
}

impl Lane {
    /// A lane whose results reserve in `memory`'s consumer.
    pub(super) fn new(memory: Reservation) -> Self {
        Self {
            io: Arc::new(Mutex::new(LaneIo {
                inbound: VecDeque::new(),
                sending: None,
                outputs: VecDeque::new(),
                refused: None,
                memory,
            })),
            pump: Arc::new(Mutex::new(Pump::Idle)),
        }
    }

    /// Polls the pump; its outcome once it ended (`None` while it runs or
    /// before the task opened).
    pub(super) fn poll(&self, cx: &mut Context<'_>) -> Option<Result<()>> {
        let mut p = lock(&self.pump);
        if let Pump::Running(f) = &mut *p
            && let Poll::Ready(r) = f.as_mut().poll(cx)
        {
            *p = Pump::Done(r);
        }
        match &*p {
            Pump::Done(r) => Some(r.clone()),
            Pump::Idle | Pump::Running(_) => None,
        }
    }

    pub(super) fn opened(&self) -> bool {
        !matches!(*lock(&self.pump), Pump::Idle)
    }

    /// Whether the lane is never opened or its task ended well.
    pub(super) fn finished(&self) -> bool {
        matches!(*lock(&self.pump), Pump::Idle | Pump::Done(Ok(())))
    }

    /// Queues `frame`, opening the task first if it is not.
    pub(super) fn queue(
        &self,
        udf: &dyn PreparedRelationUdf,
        frame: Frame,
        memory: FrameMem,
    ) -> Result<()> {
        {
            let mut p = lock(&self.pump);
            if matches!(*p, Pump::Idle) {
                let task = udf.open()?;
                *p = Pump::Running(pump(task, Arc::clone(&self.io)));
            }
        }
        lock(&self.io).inbound.push_back((frame, memory));
        Ok(())
    }

    /// Reserves the refused result, if any, without waiting, and moves it
    /// to the outputs; the bytes still missing if the pool refuses again.
    pub(super) fn try_take_refused(&self) -> Option<usize> {
        let mut q = lock(&self.io);
        let bytes = q.refused.as_ref()?.get_array_memory_size();
        let mut r = q.memory.split(0);
        if r.try_grow(bytes).is_err() {
            return Some(bytes);
        }
        if let Some(batch) = q.refused.take() {
            q.outputs.push_back((batch, r));
        }
        None
    }

    /// The refused result, if any, taken without a reservation (its
    /// caller stores it at once, writing it out when the pool is full).
    pub(super) fn take_refused_unreserved(&self) -> Option<RecordBatch> {
        lock(&self.io).refused.take()
    }

    /// The future that reserves the refused result, if any, waiting for
    /// the memory, and moves it to the outputs. Cancel-safe.
    pub(super) fn reserve_refused(&self) -> Option<BoxFuture<'static, Result<()>>> {
        let (bytes, mut r) = {
            let mut q = lock(&self.io);
            let bytes = q.refused.as_ref()?.get_array_memory_size();
            (bytes, q.memory.split(0))
        };
        let io = Arc::clone(&self.io);
        Some(Box::pin(async move {
            r.grow(bytes).await.map_err(KrishivError::from)?;
            let mut q = lock(&io);
            if let Some(batch) = q.refused.take() {
                q.outputs.push_back((batch, r));
            }
            Ok(())
        }))
    }
}

/// Sends the queued frames to `task` while it reads the task's results.
fn pump(task: Box<dyn UdfTask>, io: Arc<Mutex<LaneIo>>) -> BoxFuture<'static, Result<()>> {
    Box::pin(async move {
        let task = task;
        let send = async {
            loop {
                // Never woken: whoever queues a frame polls the pump next.
                let frame = poll_fn(|_| {
                    let mut q = lock(&io);
                    match q.inbound.pop_front() {
                        Some((frame, memory)) => {
                            q.sending = Some(memory);
                            Poll::Ready(frame)
                        }
                        None => Poll::Pending,
                    }
                })
                .await;
                let end = matches!(frame, Frame::End);
                task.send(frame).await?;
                lock(&io).sending = None;
                if end {
                    return Ok::<(), KrishivError>(());
                }
            }
        };
        let receive = async {
            loop {
                // Never woken either: whoever reserves the refused batch
                // polls the pump next.
                poll_fn(|_| {
                    if lock(&io).refused.is_some() {
                        Poll::Pending
                    } else {
                        Poll::Ready(())
                    }
                })
                .await;
                let Some(batch) = task.next().await? else {
                    return Ok(());
                };
                let bytes = batch.get_array_memory_size();
                let mut q = lock(&io);
                let mut r = q.memory.split(0);
                if r.try_grow(bytes).is_ok() {
                    q.outputs.push_back((batch, r));
                } else {
                    q.refused = Some(batch);
                }
            }
        };
        futures::future::try_join(send, receive).await.map(drop)
    })
}
