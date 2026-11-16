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
//!
//! **Wakers.** A pump may be polled by several futures in turn: its
//! driver's `Blocked` future, another driver's (which keeps a combined
//! driver's pumps moving while it waits for their tasks' slots, design
//! P3-T085 §3.3a) and `finalize`. A task keeps only the last waker it was
//! polled with, so the pump is polled with a waker of its own that wakes
//! every future that polled it since its last wake ([`Lane::poll`]): none
//! is lost when another polls the pump after it.

use std::collections::VecDeque;
use std::future::poll_fn;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

use arrow_array::RecordBatch;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_memory::Reservation;

use super::{Frame, PreparedRelationUdf, RelationSlot, UdfTask};
use crate::error::Result;
use crate::sync::{Mutex, lock, register, wake_all};

/// A reservation shared by everything that keeps its buffers alive (the
/// slices of one input batch), freed with the last of them.
pub(super) type SharedMem = Arc<Mutex<Reservation>>;

/// The memory a queued frame holds until its task takes it.
pub(super) struct FrameMem {
    /// The reservations of the input batches (and sized columns) the
    /// frame's rows slice, shared with every other frame and cut of them.
    pub(super) _shared: Vec<SharedMem>,
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
    /// The task's output ended, or the pump ended (its last result or a
    /// failure).
    stopped: bool,
    /// The pump waiting while a peer holds the lane back, woken when a
    /// peer yields, takes its last frame or stops.
    waiting: Option<Waker>,
    /// The pump waiting for a frame or for its refused result to be
    /// reserved: woken by whoever queues the frame or moves the result
    /// ([`wake_pump`]).
    pump: Option<Waker>,
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
        debug_assert!(rows <= self.rows, "took {rows} of {} rows", self.rows);
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

/// The futures that polled a pump since its last wake, woken together.
struct Pollers(Mutex<Vec<Waker>>);

impl Wake for Pollers {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let wakers = std::mem::take(&mut *lock(&self.0));
        wake_all(wakers);
    }
}

/// One lane of one driver.
#[derive(Clone)]
pub(super) struct Lane {
    pub(super) io: Arc<Mutex<LaneIo>>,
    pump: Arc<Mutex<Pump>>,
    /// Who polled the pump since its last wake.
    pollers: Arc<Pollers>,
    /// The waker the pump is polled with: wakes every one of `pollers`.
    waker: Waker,
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
                    pump: None,
                }))
            })
            .collect();
        peers
            .iter()
            .enumerate()
            .map(|(index, io)| {
                let pollers = Arc::new(Pollers(Mutex::new(Vec::new())));
                Self {
                    io: Arc::clone(io),
                    pump: Arc::new(Mutex::new(Pump::Idle)),
                    waker: Waker::from(Arc::clone(&pollers)),
                    pollers,
                    peers: Arc::clone(&peers),
                    index,
                }
            })
            .collect()
    }

    /// Polls the pump; its outcome once it ended (`None` while it runs or
    /// before the task opened). `cx` is woken when the pump can move, even
    /// if another future polls the pump after this one.
    pub(super) fn poll(&self, cx: &mut Context<'_>) -> Option<Result<()>> {
        let mut p = lock(&self.pump);
        if matches!(*p, Pump::Running(_)) {
            register(&mut lock(&self.pollers.0), cx.waker());
        }
        if let Pump::Running(f) = &mut *p
            && let Poll::Ready(r) = f.as_mut().poll(&mut Context::from_waker(&self.waker))
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

    /// Whether the lane's task is open and has not ended (nor failed).
    pub(super) fn running(&self) -> bool {
        matches!(*lock(&self.pump), Pump::Running(_))
    }

    /// Whether the lane is never opened or its task ended well.
    pub(super) fn finished(&self) -> bool {
        matches!(*lock(&self.pump), Pump::Idle | Pump::Done(Ok(())))
    }

    /// Queues `frame`, opening the task first if it is not, in `slot` (the
    /// lane's slot of the driver's admission, `admit.rs`).
    pub(super) fn queue(
        &self,
        udf: &dyn PreparedRelationUdf,
        frame: Frame,
        memory: FrameMem,
        slot: Option<RelationSlot>,
    ) -> Result<()> {
        {
            let mut p = lock(&self.pump);
            if matches!(*p, Pump::Idle) {
                let task = udf.open(slot)?;
                *p = Pump::Running(pump(task, Arc::clone(&self.peers), self.index));
            }
        }
        lock(&self.io).inbound.push_back((frame, memory));
        wake_pump(&self.io);
        Ok(())
    }

    /// The bytes of the refused result, if any.
    pub(super) fn refused_bytes(&self) -> Option<usize> {
        lock(&self.io)
            .refused
            .as_ref()
            .map(RecordBatch::get_array_memory_size)
    }

    /// Grows `memory` (a reservation of any consumer: the store takes it
    /// over) to the refused result, if any, and moves the result to the
    /// outputs, under one lock, so two drivers never reserve for one batch
    /// nor move one under another's size (review R4-3); the bytes the pool
    /// refused. The pump is woken to read on.
    pub(super) fn reserve_refused_with(
        &self,
        mut memory: Reservation,
    ) -> std::result::Result<(), usize> {
        {
            let mut q = lock(&self.io);
            let Some(bytes) = q.refused.as_ref().map(RecordBatch::get_array_memory_size) else {
                return Ok(());
            };
            if memory.try_grow(bytes).is_err() {
                return Err(bytes);
            }
            if let Some(batch) = q.refused.take() {
                q.outputs.push_back((batch, memory));
            }
        }
        wake_pump(&self.io);
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
        drop(q);
        // The pump waits for this: it reads on.
        wake_pump(&self.io);
        None
    }

    /// The refused result, if any, taken without a reservation (its
    /// caller stores it at once, writing it out when the pool is full).
    pub(super) fn take_refused_unreserved(&self) -> Option<RecordBatch> {
        let batch = {
            let mut q = lock(&self.io);
            let batch = q.refused.take()?;
            q.took(batch.num_rows());
            batch
        };
        wake_pump(&self.io);
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
            {
                let mut q = lock(&io);
                if let Some(batch) = q.refused.take() {
                    q.outputs.push_back((batch, r));
                }
            }
            wake_pump(&io);
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

/// Wakes the pump of `io` if it waits for a frame or for its refused result
/// (the waker is taken under the lock and woken after it is released).
fn wake_pump(io: &Mutex<LaneIo>) {
    let waker = lock(io).pump.take();
    if let Some(w) = waker {
        w.wake();
    }
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
                // Woken by whoever queues the next frame (`Lane::queue`).
                let frame = poll_fn(|cx| {
                    let mut q = lock(&io);
                    match q.inbound.pop_front() {
                        Some((frame, memory)) => {
                            q.sending = Some(memory);
                            Poll::Ready(frame)
                        }
                        None => {
                            q.pump = Some(cx.waker().clone());
                            Poll::Pending
                        }
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
                // Refused: woken by whoever reserves or takes the refused
                // batch ([`wake_pump`]). Held back by a peer: woken when a
                // peer yields, takes its last frame or stops (the waker is
                // set before the check, so a peer that moves after it wakes
                // this pump).
                poll_fn(|cx| {
                    {
                        let mut q = lock(&io);
                        if q.refused.is_some() {
                            q.pump = Some(cx.waker().clone());
                            return Poll::Pending;
                        }
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
                    // The output ended: it yields no more rows, so it holds
                    // no peer back, even before its send side finished.
                    // `UdfTask::next` ends only once `End` was processed,
                    // so the send side is about to finish; a runtime that
                    // ended early while a send hangs (a write to a dead
                    // worker that blocks) would otherwise hold every peer
                    // with more rows, a hang instead of the mismatch.
                    lock(&io).stopped = true;
                    wake_peers(&peers, index);
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
