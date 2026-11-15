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
//!
//! **Peers.** An aggregate's lanes yield one value per group and a group is
//! stored only once every lane has its value (`pair.rs`), so a lane's values
//! past its peers' cannot be released, nor spilled. A lane therefore reads
//! no result while it holds more rows than a peer whose task still runs
//! *and still has frames it has not taken*: it waits on its own task,
//! holding at most one result batch past that peer, so the peer's values
//! do not wait for memory this lane pins. A peer that took every frame it
//! was given may need later frames before it yields (a runtime batching
//! values across groups); holding back then would stop this lane's task
//! taking frames, fill its window and so stop the frames to every lane, a
//! hang. So the hold lasts only while the peer has frames to take, and is
//! released (the waiting pump woken) when it took its last one, yielded or
//! stopped; the pool then decides (a spill, or a classed error), never a
//! hang. The lane with the fewest rows is never held, and a lane is held
//! only by a peer with strictly fewer rows, so the holds form no cycle.
//! Only an aggregate's lanes are peers ([`Lane::group`]): a map or grouped
//! shape's results are stored one lane at a time.

use std::collections::VecDeque;
use std::future::poll_fn;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

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
    /// The rows of [`Self::outputs`] and [`Self::refused`], kept as they
    /// change (a peer's hold reads it on every result).
    rows: usize,
    /// The pump ended: the task yielded its last result or failed.
    stopped: bool,
    /// The pump waiting while a peer holds the lane back, woken when a
    /// peer yields, takes its last frame or stops.
    waiting: Option<Waker>,
}

impl LaneIo {
    pub(super) fn in_flight(&self) -> usize {
        self.inbound.len() + usize::from(self.sending.is_some())
    }

    /// Whether results wait for the driver (stored or reserved).
    pub(super) fn has_results(&self) -> bool {
        !self.outputs.is_empty() || self.refused.is_some()
    }

    /// The next result, taken by the driver.
    pub(super) fn pop_output(&mut self) -> Option<(RecordBatch, Reservation)> {
        let next = self.outputs.pop_front();
        if let Some((b, _)) = &next {
            self.took(b.num_rows());
        }
        next
    }

    /// `rows` rows of the results taken by the driver (the front batch
    /// sliced or popped by the caller).
    pub(super) fn took(&mut self, rows: usize) {
        self.rows = self.rows.saturating_sub(rows);
    }
}

/// The queues of a driver's lanes, one per lane.
type Peers = Arc<[Arc<Mutex<LaneIo>>]>;

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
    /// Every lane's queues of the driver, this one's at `index`.
    peers: Peers,
    index: usize,
}

impl Lane {
    /// The lanes of one driver, one per reservation, whose results reserve
    /// in that reservation's consumer. With `peered` (an aggregate's lanes)
    /// each reads no result while it holds more rows than a peer still
    /// running that has frames to take; else every lane is alone.
    pub(super) fn group(
        memories: impl IntoIterator<Item = Reservation>,
        peered: bool,
    ) -> Vec<Self> {
        if !peered {
            return memories
                .into_iter()
                .flat_map(|memory| Self::group([memory], true))
                .collect();
        }
        let peers: Peers = memories
            .into_iter()
            .map(|memory| {
                Arc::new(Mutex::new(LaneIo {
                    inbound: VecDeque::new(),
                    sending: None,
                    outputs: VecDeque::new(),
                    refused: None,
                    memory,
                    rows: 0,
                    stopped: false,
                    waiting: None,
                }))
            })
            .collect();
        peers
            .iter()
            .enumerate()
            .map(|(index, io)| Self {
                io: Arc::clone(io),
                pump: Arc::new(Mutex::new(Pump::Idle)),
                peers: Arc::clone(&peers),
                index,
            })
            .collect()
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
                *p = Pump::Running(pump(task, Arc::clone(&self.peers), self.index));
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
        let mut q = lock(&self.io);
        let batch = q.refused.take()?;
        q.took(batch.num_rows());
        Some(batch)
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

/// Whether lane `index` holds more rows than a peer whose task still runs
/// and has frames it has not taken (a peer waiting for input never holds
/// it back, see the module doc). Each queue is locked alone, so two lanes
/// asking at once never deadlock.
fn held_back(peers: &[Arc<Mutex<LaneIo>>], index: usize) -> bool {
    let Some(own) = peers.get(index).map(|io| lock(io).rows) else {
        return false;
    };
    peers.iter().enumerate().any(|(i, io)| {
        let q = lock(io);
        i != index && !q.stopped && q.in_flight() > 0 && q.rows < own
    })
}

/// Wakes every peer of lane `index` waiting for it to yield, take its last
/// frame or stop; each waker is taken under the peer's lock and woken
/// after it is released.
fn wake_peers(peers: &[Arc<Mutex<LaneIo>>], index: usize) {
    for (i, io) in peers.iter().enumerate() {
        if i == index {
            continue;
        }
        let waker = lock(io).waiting.take();
        if let Some(w) = waker {
            w.wake();
        }
    }
}

/// Sends the queued frames of lane `index` to `task` while it reads the
/// task's results.
fn pump(task: Box<dyn UdfTask>, peers: Peers, index: usize) -> BoxFuture<'static, Result<()>> {
    Box::pin(async move {
        let task = task;
        let Some(io) = peers.get(index).map(Arc::clone) else {
            return Err(crate::error::internal(format!("no lane {index}")));
        };
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
                let idle = {
                    let mut q = lock(&io);
                    q.sending = None;
                    q.in_flight() == 0
                };
                if idle {
                    // A peer held back by this lane's frames moves on.
                    wake_peers(&peers, index);
                }
                if end {
                    return Ok::<(), KrishivError>(());
                }
            }
        };
        let receive = async {
            loop {
                // Not woken while refused: whoever reserves the refused
                // batch polls the pump next. Held back by a peer: woken
                // when a peer yields, takes its last frame or stops (the
                // waker is set before the check, so a peer that moves
                // after it wakes this pump).
                poll_fn(|cx| {
                    if lock(&io).refused.is_some() {
                        return Poll::Pending;
                    }
                    lock(&io).waiting = Some(cx.waker().clone());
                    if held_back(&peers, index) {
                        Poll::Pending
                    } else {
                        lock(&io).waiting = None;
                        Poll::Ready(())
                    }
                })
                .await;
                let Some(batch) = task.next().await? else {
                    return Ok(());
                };
                let bytes = batch.get_array_memory_size();
                {
                    let mut q = lock(&io);
                    q.rows += batch.num_rows();
                    let mut r = q.memory.split(0);
                    if r.try_grow(bytes).is_ok() {
                        q.outputs.push_back((batch, r));
                    } else {
                        q.refused = Some(batch);
                    }
                }
                wake_peers(&peers, index);
            }
        };
        let outcome = futures::future::try_join(send, receive).await.map(drop);
        // Ended or failed: this lane holds no peer back any more.
        lock(&io).stopped = true;
        wake_peers(&peers, index);
        outcome
    })
}
