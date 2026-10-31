// SPDX-License-Identifier: Apache-2.0
//! The query's result: a bounded channel from the root pipeline's drivers
//! to the caller's stream. A full channel accepts the batch and returns
//! `SinkStatus::Blocked`, so drivers stop pulling their sources until the
//! caller reads (backpressure); the channel holds at most its capacity plus
//! one batch per driver. Dropping the stream cancels the query.
//!
//! # Order
//! Without `ORDER BY`, Spark returns a query's rows partition by partition:
//! partition 0 first, and a `UNION ALL`'s first child before its second.
//! Drivers that read an ordered source (the runner registers them with
//! [`Sink::expect_ordered`]) report each morsel's claim ordinal
//! ([`Sink::position`]); their batches reach the channel ordered by
//! `(fan-in member, ordinal)`. The driver holding the smallest such key
//! sends straight to the channel; the others hold their batches back, in
//! memory reserved through their consumer (taken, never awaited), at most
//! [`ORDER_BATCHES`] each and at most their share of a quarter of the
//! query's memory limit (so the operators of the driver whose turn it is
//! keep room to work), and wait once over either limit or as soon as a
//! reservation is refused (one batch over, the one in hand). When the
//! smallest key moves on (a driver reports a new morsel, or ends), the
//! batches it passed are sent in key order. The driver with the smallest
//! key is never held back, so the order cannot deadlock.
//!
//! A driver that has not claimed its first morsel holds the smallest key
//! of its member (it may still claim any of them): every batch of the
//! member waits until each of its drivers has started. Once the caller
//! drops the stream, held batches are dropped and their memory returned at
//! once. A released batch keeps its memory reserved while it waits in the
//! channel: the reservation travels with it and is returned when the caller
//! reads the batch (or drops the stream), so a slow caller cannot let
//! released batches pile up unaccounted.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use futures::Stream;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_types::batch::Batch;

use krishiv_memory::Reservation;

use crate::context::{QueryContext, TaskContext};
use crate::error::{Result, internal};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus};
use crate::sync::{Mutex, lock, register, wake_all};

#[derive(Debug, Default)]
struct State {
    /// The batches, each with the memory an ordered release left reserved
    /// for it (returned when it leaves the queue).
    queue: VecDeque<(Batch, Option<Reservation>)>,
    /// Set by the query's coordinator once every task has ended.
    end: Option<Option<KrishivError>>,
    receiver_gone: bool,
    receiver: Option<Waker>,
    senders: Vec<Waker>,
    /// Batches in `queue` that an ordered release sent (they were held,
    /// and bounded, before): they do not count against the capacity, so a
    /// release never stalls the driver whose turn it is.
    released: usize,
}

impl State {
    /// Whether the queue is at capacity, released batches aside.
    fn full(&self, capacity: usize) -> bool {
        self.queue.len().saturating_sub(self.released) >= capacity
    }
}

/// Shared by the result sink and the stream.
#[derive(Debug)]
pub(crate) struct Channel {
    capacity: usize,
    state: Mutex<State>,
}

impl Channel {
    pub(crate) fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            capacity: capacity.max(1),
            state: Mutex::new(State::default()),
        })
    }

    /// Accepts `batch`: `Some(full)`, or `None` if the receiver is gone.
    fn send(&self, batch: Batch) -> Option<bool> {
        self.push(batch, None, false)
    }

    /// As [`send`](Self::send), for a batch an ordered release sends, with
    /// the memory it holds (kept reserved until the caller reads it).
    fn send_released(&self, batch: Batch, memory: Option<Reservation>) -> Option<bool> {
        self.push(batch, memory, true)
    }

    fn push(&self, batch: Batch, memory: Option<Reservation>, released: bool) -> Option<bool> {
        let (full, waker) = {
            let mut s = lock(&self.state);
            if s.receiver_gone {
                return None;
            }
            s.queue.push_back((batch, memory));
            s.released += usize::from(released);
            (s.full(self.capacity), s.receiver.take())
        };
        if let Some(w) = waker {
            w.wake();
        }
        Some(full)
    }

    #[cfg(all(test, loom))]
    pub(crate) fn send_for_test(&self, batch: Batch) -> Option<bool> {
        self.send(batch)
    }

    /// Completes once the queue has room or the receiver is gone.
    pub(crate) fn space(self: Arc<Self>) -> impl Future<Output = ()> + Send + 'static {
        std::future::poll_fn(move |cx| {
            let mut s = lock(&self.state);
            if s.receiver_gone || !s.full(self.capacity) {
                return Poll::Ready(());
            }
            register(&mut s.senders, cx.waker());
            Poll::Pending
        })
    }

    /// The coordinator's last word: no more batches; `error` if the query
    /// failed.
    pub(crate) fn close(&self, error: Option<KrishivError>) {
        let waker = {
            let mut s = lock(&self.state);
            s.end = Some(error);
            s.receiver.take()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }

    pub(crate) fn poll_recv(&self, cx: &mut Context<'_>) -> Poll<Option<Result<Batch>>> {
        let (out, senders, memory) = {
            let mut s = lock(&self.state);
            if let Some((b, memory)) = s.queue.pop_front() {
                // Released batches count down as any batch leaves: either
                // way the queue is bounded by the capacity plus what the
                // releases sent.
                s.released = s.released.saturating_sub(1);
                // Wake every blocked sender, not one: a woken sender may
                // have been cancelled (its waker is stale), and waking only
                // it would strand the others. There are at most as many
                // senders as worker slots.
                (
                    Poll::Ready(Some(Ok(b))),
                    std::mem::take(&mut s.senders),
                    memory,
                )
            } else if let Some(end) = s.end.as_mut() {
                // An error is reported once, then the stream ends.
                (Poll::Ready(end.take().map(Err)), Vec::new(), None)
            } else {
                s.receiver = Some(cx.waker().clone());
                (Poll::Pending, Vec::new(), None)
            }
        };
        // The batch left the queue: its memory goes back to the pool, outside
        // the state lock.
        drop(memory);
        wake_all(senders);
        out
    }

    /// Whether the caller dropped the stream.
    fn receiver_gone(&self) -> bool {
        lock(&self.state).receiver_gone
    }

    /// Wakes `waker` when the caller reads a batch or drops the stream.
    fn register_sender(&self, waker: &Waker) {
        register(&mut lock(&self.state).senders, waker);
    }

    pub(crate) fn drop_receiver(&self) {
        let (senders, queued) = {
            let mut s = lock(&self.state);
            s.receiver_gone = true;
            s.released = 0;
            (std::mem::take(&mut s.senders), std::mem::take(&mut s.queue))
        };
        // Dropping the queue returns the released batches' memory.
        drop(queued);
        wake_all(senders);
    }
}

/// Most batches one driver holds back waiting for its turn (module docs).
/// The memory they reserve is the bound that matters: a driver ahead of
/// the order runs ahead while its pool has room (a small cap serialised
/// the drivers of a large scan, whose morsels are thousands of batches),
/// and waits once it does not.
pub(crate) const ORDER_BATCHES: usize = 4096;

/// A held-back batch: the driver holding it, the batch, the bytes it is
/// charged and whether they are reserved.
type Held = (u32, Batch, usize, bool);

/// The ordered drivers of the result (module docs).
#[derive(Debug, Default)]
struct Order {
    drivers: HashMap<u32, Track>,
    /// Batches held back, by `(member, ordinal)`, each with the driver
    /// that holds it and the bytes it reserved.
    pending: BTreeMap<(u32, u64), VecDeque<Held>>,
}

/// One ordered driver.
#[derive(Debug)]
struct Track {
    member: u32,
    /// The ordinal of the morsel it reads; `None` before its first.
    position: Option<u64>,
    ended: bool,
    /// Batches it holds back, the bytes they are charged, and the memory
    /// reserved for them.
    held: usize,
    bytes: usize,
    memory: Option<Reservation>,
    /// The most bytes it may hold back (module docs).
    cap: usize,
    /// A reservation was refused: it waits until nothing it holds is left.
    short: bool,
    waker: Option<Waker>,
}

impl Track {
    /// The smallest key this driver can still produce.
    fn bound(&self) -> (u32, u64) {
        (self.member, self.position.unwrap_or(0))
    }

    /// Whether it must wait before sinking more.
    fn over(&self) -> bool {
        self.held >= ORDER_BATCHES || self.bytes >= self.cap || (self.short && self.held > 0)
    }
}

impl Order {
    /// The smallest key a live driver can still produce (`None` once every
    /// ordered driver ended).
    fn min(&self) -> Option<(u32, u64)> {
        self.drivers
            .values()
            .filter(|t| !t.ended)
            .map(Track::bound)
            .min()
    }

    /// Sends every held batch whose key is at most the smallest live key,
    /// in key order; returns the wakers of drivers that may go on. Once the
    /// caller dropped the stream, every held batch is dropped instead.
    fn release(&mut self, channel: &Channel) -> Vec<Waker> {
        let min = if channel.receiver_gone() {
            None
        } else {
            self.min()
        };
        let mut wakers = Vec::new();
        while let Some(entry) = self.pending.first_entry() {
            if min.is_some_and(|m| *entry.key() > m) {
                break;
            }
            for (partition, batch, bytes, reserved) in entry.remove() {
                // The batch's memory leaves the driver's reservation with it
                // and stays reserved until the caller reads the batch.
                let memory = if reserved {
                    self.drivers
                        .get_mut(&partition)
                        .and_then(|t| t.memory.as_mut())
                        .map(|m| m.split(bytes.min(m.size())))
                } else {
                    None
                };
                // A receiver that is gone takes nothing: the batch is dropped.
                let _ = channel.send_released(batch, memory);
                if let Some(t) = self.drivers.get_mut(&partition) {
                    t.held = t.held.saturating_sub(1);
                    t.bytes = t.bytes.saturating_sub(bytes);
                    if t.held == 0 {
                        t.short = false;
                        if t.ended {
                            t.memory = None;
                        }
                    }
                    if !t.over()
                        && let Some(w) = t.waker.take()
                    {
                        wakers.push(w);
                    }
                }
            }
        }
        wakers
    }
}

/// The bytes a held-back batch is charged: its rows' bytes, not the whole
/// buffers it shares (a scan's filtered batch shares its decoded pages).
/// A string or binary view column is charged its views and the bytes its
/// rows' longer values take, not the data buffers they point into (whole
/// pages: charging those filled the pool with a fraction of the rows).
#[expect(
    clippy::cast_possible_truncation,
    reason = "a view's low 32 bits are its value's length"
)]
fn held_bytes(batch: &Batch) -> usize {
    use arrow_array::cast::AsArray;
    use arrow_schema::DataType;
    let views = |lens: &mut dyn Iterator<Item = u32>, n: usize| {
        n * 16 + lens.filter(|&l| l > 12).map(|l| l as usize).sum::<usize>()
    };
    batch
        .data()
        .columns()
        .iter()
        .map(|c| match c.data_type() {
            DataType::Utf8View => {
                let a = c.as_string_view();
                views(&mut a.views().iter().map(|v| *v as u32), a.views().len())
            }
            DataType::BinaryView => {
                let a = c.as_binary_view();
                views(&mut a.views().iter().map(|v| *v as u32), a.views().len())
            }
            _ => c
                .to_data()
                .get_slice_memory_size()
                .unwrap_or_else(|_| c.get_array_memory_size()),
        })
        .sum()
}

/// The result sink's state: the ordered drivers.
#[derive(Debug, Default)]
struct ResultGlobal {
    order: Mutex<Order>,
}

/// One driver's result-sink state.
#[derive(Debug)]
struct ResultLocal {
    partition: u32,
    /// The runner registered this driver as ordered.
    ordered: bool,
}

/// The root pipeline's sink.
pub(crate) struct ResultSink {
    pub(crate) channel: Arc<Channel>,
}

fn result_global(global: &GlobalState) -> Result<&ResultGlobal> {
    global
        .downcast_ref::<ResultGlobal>()
        .ok_or_else(|| internal("wrong global state for the result".to_owned()))
}

fn result_local(local: &LocalState) -> Result<&ResultLocal> {
    local
        .downcast_ref::<ResultLocal>()
        .ok_or_else(|| internal("wrong local state for the result".to_owned()))
}

impl ResultSink {
    /// What sending one batch tells the driver.
    fn sent(&self, full: Option<bool>) -> SinkStatus {
        match full {
            None => SinkStatus::Finished,
            Some(false) => SinkStatus::NeedMoreInput,
            Some(true) => SinkStatus::Blocked(Box::pin(Arc::clone(&self.channel).space())),
        }
    }

    /// Completes once driver `partition` may sink again (module docs), or
    /// the caller dropped the stream.
    fn turn(
        &self,
        global: GlobalState,
        partition: u32,
    ) -> impl Future<Output = ()> + Send + 'static {
        let channel = Arc::clone(&self.channel);
        std::future::poll_fn(move |cx| {
            let Ok(g) = result_global(&global) else {
                return Poll::Ready(());
            };
            if channel.receiver_gone() {
                return Poll::Ready(());
            }
            let mut order = lock(&g.order);
            match order.drivers.get_mut(&partition) {
                Some(t) if t.over() => {
                    t.waker = Some(cx.waker().clone());
                    drop(order);
                    channel.register_sender(cx.waker());
                    Poll::Pending
                }
                _ => Poll::Ready(()),
            }
        })
    }

    /// Updates driver `partition` under the order's lock, then sends what
    /// that released.
    fn update(
        &self,
        global: &GlobalState,
        partition: u32,
        f: impl FnOnce(&mut Track),
    ) -> Result<()> {
        let g = result_global(global)?;
        let wakers = {
            let mut order = lock(&g.order);
            if let Some(t) = order.drivers.get_mut(&partition) {
                f(t);
            }
            order.release(&self.channel)
        };
        wakers.into_iter().for_each(Waker::wake);
        Ok(())
    }
}

impl Sink for ResultSink {
    fn name(&self) -> &'static str {
        "Result"
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(ResultGlobal::default()))
    }

    fn expect_ordered(&self, global: &GlobalState, member: u32, partitions: std::ops::Range<u32>) {
        let Ok(g) = result_global(global) else {
            return;
        };
        let mut order = lock(&g.order);
        for p in partitions {
            order.drivers.insert(
                p,
                Track {
                    member,
                    position: None,
                    ended: false,
                    held: 0,
                    bytes: 0,
                    memory: None,
                    cap: usize::MAX,
                    short: false,
                    waker: None,
                },
            );
        }
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        let partition = ctx.partition();
        let mut order = lock(&result_global(global)?.order);
        // Each ordered driver may hold back its share of a quarter of the
        // query's limit (module docs).
        let drivers = order.drivers.len().max(1);
        let cap = ctx
            .query()
            .pool()
            .effective_limit()
            .map_or(usize::MAX, |l| (l / 4 / drivers).max(1));
        let ordered = match order.drivers.get_mut(&partition) {
            Some(t) => {
                t.memory = Some(ctx.consumer().reservation());
                t.cap = cap;
                true
            }
            None => false,
        };
        Ok(Box::new(ResultLocal { partition, ordered }))
    }

    fn position(&self, local: &mut LocalState, global: &GlobalState, position: u64) {
        let Ok(l) = result_local(local) else {
            return;
        };
        if l.ordered {
            // An error here means a malformed state: the next `sink` reports it.
            let _ = self.update(global, l.partition, |t| t.position = Some(position));
        }
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        global: &GlobalState,
    ) -> Result<SinkStatus> {
        let l = result_local(local)?;
        if !l.ordered {
            return Ok(self.sent(self.channel.send(input)));
        }
        let partition = l.partition;
        let g = result_global(global)?;
        let mut order = lock(&g.order);
        let min = order.min();
        let Some(t) = order.drivers.get_mut(&partition) else {
            drop(order);
            return Ok(self.sent(self.channel.send(input)));
        };
        let key = t.bound();
        if min == Some(key) {
            // Every key below this driver's is sent: its batches go now.
            let full = self.channel.send(input);
            drop(order);
            return Ok(self.sent(full));
        }
        if self.channel.receiver_gone() {
            // Drop what every driver holds, and return its memory, now.
            let wakers = order.release(&self.channel);
            drop(order);
            wakers.into_iter().for_each(Waker::wake);
            return Ok(SinkStatus::Finished);
        }
        let bytes = held_bytes(&input);
        let reserved = t.memory.as_mut().is_some_and(|m| m.try_grow(bytes).is_ok());
        t.short |= !reserved;
        t.held += 1;
        t.bytes += bytes;
        let wait = t.over();
        order
            .pending
            .entry(key)
            .or_default()
            .push_back((partition, input, bytes, reserved));
        drop(order);
        Ok(if wait {
            SinkStatus::Blocked(Box::pin(self.turn(Arc::clone(global), partition)))
        } else {
            SinkStatus::NeedMoreInput
        })
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        let l = result_local(&local)?;
        if l.ordered {
            self.update(global, l.partition, |t| {
                t.ended = true;
                if t.held == 0 {
                    t.memory = None;
                }
            })?;
        }
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        _global: &'a GlobalState,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

/// The caller's end: batches as the root pipeline produces them, then the
/// query's error, if any. Dropping it cancels the query.
pub(crate) struct ResultStream {
    pub(crate) channel: Arc<Channel>,
    pub(crate) query: Arc<QueryContext>,
}

impl Stream for ResultStream {
    type Item = Result<Batch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Result<Batch>>> {
        self.channel.poll_recv(cx)
    }
}

impl Drop for ResultStream {
    fn drop(&mut self) {
        // Cancel first, so woken drivers see it.
        self.query.cancel();
        self.channel.drop_receiver();
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::task::Waker;

    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use krishiv_memory::{MemoryConfig, MemoryPool as _, SpillOwner, TreePool};

    use super::*;
    use crate::{
        ExecConfig, ExecutorConfig, OperatorId, QueryContext, QueryId, TokioExecutor, WorkloadClass,
    };

    fn ok<T, E: std::fmt::Display>(r: std::result::Result<T, E>) -> T {
        r.unwrap_or_else(|e| unreachable!("{e}"))
    }

    fn batch(v: i64) -> Batch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        Batch::new(ok(RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![v]))],
        )))
    }

    /// A result sink over a fresh channel, a query whose pool holds
    /// `limit` bytes, and the contexts of drivers `0..drivers`.
    fn setup(limit: Option<usize>, drivers: u32) -> (ResultSink, GlobalState, Vec<TaskContext>) {
        let root = TreePool::new(&MemoryConfig {
            limit,
            quantum: 1,
            ..MemoryConfig::default()
        });
        let q = QueryContext::new(
            QueryId(1),
            WorkloadClass::Batch,
            ExecConfig::default(),
            root.child("q", limit),
        );
        let exec = ok(TokioExecutor::new(&ExecutorConfig::default()));
        let ctx = |p| {
            TaskContext::new(
                Arc::clone(&q),
                Arc::clone(&exec) as _,
                SpillOwner::new(),
                OperatorId(0),
                "Result",
            )
            .with_partition(p)
        };
        let sink = ResultSink {
            channel: Channel::new(1024),
        };
        let global = ok(sink.global_state(&ctx(0)));
        (sink, global, (0..drivers).map(ctx).collect())
    }

    /// The values the channel holds, in order.
    fn received(sink: &ResultSink) -> Vec<i64> {
        let mut cx = Context::from_waker(Waker::noop());
        let mut out = Vec::new();
        while let Poll::Ready(Some(Ok(b))) = sink.channel.poll_recv(&mut cx) {
            let col = b.data().column(0).as_any().downcast_ref::<Int64Array>();
            out.push(col.map_or(-1, |c| c.value(0)));
        }
        out
    }

    fn status(s: &SinkStatus) -> &'static str {
        match s {
            SinkStatus::NeedMoreInput => "more",
            SinkStatus::Blocked(_) => "blocked",
            SinkStatus::Finished => "finished",
            SinkStatus::NeedMemory { .. } => "memory",
        }
    }

    /// Batches leave in `(member, ordinal)` order whatever order the drivers
    /// sink them in; a later member waits for every earlier one.
    #[test]
    fn batches_leave_in_key_order() {
        let (sink, g, ctx) = setup(None, 3);
        sink.expect_ordered(&g, 0, 0..2);
        sink.expect_ordered(&g, 1, 2..3);
        let mut l: Vec<LocalState> = ctx.iter().map(|c| ok(sink.local_state(c, &g))).collect();
        sink.position(&mut l[2], &g, 0);
        assert_eq!(status(&ok(sink.sink(batch(30), &mut l[2], &g))), "more");
        sink.position(&mut l[1], &g, 1);
        assert_eq!(status(&ok(sink.sink(batch(20), &mut l[1], &g))), "more");
        sink.position(&mut l[0], &g, 0);
        assert_eq!(status(&ok(sink.sink(batch(10), &mut l[0], &g))), "more");
        assert_eq!(received(&sink), [10]);
        // Driver 0 moves past ordinal 1: driver 1's batch is next.
        sink.position(&mut l[0], &g, 2);
        assert_eq!(received(&sink), [20]);
        assert_eq!(status(&ok(sink.sink(batch(11), &mut l[0], &g))), "more");
        let mut l = l.into_iter();
        let (Some(l0), Some(l1), Some(l2)) = (l.next(), l.next(), l.next()) else {
            unreachable!("three drivers");
        };
        ok(sink.combine(l1, &g));
        assert_eq!(received(&sink), [11]);
        ok(sink.combine(l0, &g));
        assert_eq!(received(&sink), [30]);
        ok(sink.combine(l2, &g));
        assert!(received(&sink).is_empty());
    }

    /// A driver ahead of the order holds at most `ORDER_BATCHES` batches,
    /// then waits for its turn.
    #[test]
    fn a_driver_ahead_waits_once_it_holds_too_many() {
        let (sink, g, ctx) = setup(None, 2);
        sink.expect_ordered(&g, 0, 0..2);
        let mut l0 = ok(sink.local_state(&ctx[0], &g));
        let mut l1 = ok(sink.local_state(&ctx[1], &g));
        sink.position(&mut l0, &g, 0);
        sink.position(&mut l1, &g, 1);
        for i in 0..ORDER_BATCHES - 1 {
            let s = ok(sink.sink(batch(100 + i as i64), &mut l1, &g));
            assert_eq!(status(&s), "more");
        }
        let SinkStatus::Blocked(mut turn) = ok(sink.sink(batch(999), &mut l1, &g)) else {
            unreachable!("not blocked");
        };
        let mut cx = Context::from_waker(Waker::noop());
        assert!(turn.as_mut().poll(&mut cx).is_pending());
        ok(sink.combine(l0, &g));
        assert!(turn.as_mut().poll(&mut cx).is_ready());
        assert_eq!(received(&sink).len(), ORDER_BATCHES);
        ok(sink.combine(l1, &g));
    }

    /// A refused reservation makes the driver wait at once (one batch
    /// over, the one in hand), until what it holds is sent; a dropped
    /// receiver ends every wait, drops the held batches and returns their
    /// memory.
    #[test]
    fn short_memory_and_a_dropped_receiver() {
        let (sink, g, ctx) = setup(Some(1), 2);
        sink.expect_ordered(&g, 0, 0..2);
        let mut l0 = ok(sink.local_state(&ctx[0], &g));
        let mut l1 = ok(sink.local_state(&ctx[1], &g));
        sink.position(&mut l0, &g, 0);
        sink.position(&mut l1, &g, 1);
        let SinkStatus::Blocked(mut turn) = ok(sink.sink(batch(2), &mut l1, &g)) else {
            unreachable!("not blocked");
        };
        let mut cx = Context::from_waker(Waker::noop());
        assert!(turn.as_mut().poll(&mut cx).is_pending());
        sink.channel.drop_receiver();
        assert!(turn.as_mut().poll(&mut cx).is_ready());
        assert_eq!(status(&ok(sink.sink(batch(3), &mut l1, &g))), "finished");
        assert_eq!(status(&ok(sink.sink(batch(1), &mut l0, &g))), "finished");
        let order = lock(&ok(result_global(&g)).order);
        assert!(order.pending.is_empty());
        assert!(order.drivers.values().all(|t| t.held == 0 && t.bytes == 0));
    }

    /// A driver ahead holds at most its share of a quarter of the query's
    /// memory limit, then waits for its turn.
    #[test]
    fn a_driver_ahead_holds_at_most_its_share_of_the_limit() {
        let (sink, g, ctx) = setup(Some(1 << 20), 2);
        sink.expect_ordered(&g, 0, 0..2);
        let mut l0 = ok(sink.local_state(&ctx[0], &g));
        let mut l1 = ok(sink.local_state(&ctx[1], &g));
        sink.position(&mut l0, &g, 0);
        sink.position(&mut l1, &g, 1);
        // 96 KiB each: the share is 128 KiB (1 MiB / 4 / 2 drivers).
        let big = |v: i64| {
            let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
            Batch::new(ok(RecordBatch::try_new(
                schema,
                vec![Arc::new(Int64Array::from(vec![v; 12 * 1024]))],
            )))
        };
        let first = ok(sink.sink(big(1), &mut l1, &g));
        assert_eq!(status(&first), "more");
        let second = ok(sink.sink(big(2), &mut l1, &g));
        assert_eq!(status(&second), "blocked");
        ok(sink.combine(l0, &g));
        assert_eq!(received(&sink).len(), 2);
        ok(sink.combine(l1, &g));
    }

    /// A driver's share spans its morsels: moving to a later morsel neither
    /// resets what it holds nor lets it past its share, and the driver
    /// behind passing those morsels releases their batches in key order.
    #[test]
    fn a_drivers_share_spans_the_morsels_it_moves_through() {
        let (sink, g, ctx) = setup(Some(1 << 20), 2);
        sink.expect_ordered(&g, 0, 0..2);
        let mut l0 = ok(sink.local_state(&ctx[0], &g));
        let mut l1 = ok(sink.local_state(&ctx[1], &g));
        sink.position(&mut l0, &g, 0);
        sink.position(&mut l1, &g, 1);
        // 48 KiB each: the share is 128 KiB (1 MiB / 4 / 2 drivers).
        let big = |v: i64| {
            let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
            Batch::new(ok(RecordBatch::try_new(
                schema,
                vec![Arc::new(Int64Array::from(vec![v; 6 * 1024]))],
            )))
        };
        assert_eq!(status(&ok(sink.sink(big(1), &mut l1, &g))), "more");
        sink.position(&mut l1, &g, 2);
        assert_eq!(status(&ok(sink.sink(big(2), &mut l1, &g))), "more");
        sink.position(&mut l1, &g, 3);
        assert_eq!(status(&ok(sink.sink(big(3), &mut l1, &g))), "blocked");
        // Driver 0 moves past driver 1's morsels: driver 1 now holds the
        // smallest key, and everything it held goes out in key order.
        sink.position(&mut l0, &g, 4);
        assert_eq!(received(&sink), [1, 2, 3]);
        ok(sink.combine(l0, &g));
        ok(sink.combine(l1, &g));
    }

    /// A released batch stays reserved in the pool while it waits in the
    /// channel and is returned when the caller reads it (or drops the stream).
    #[test]
    fn released_batches_stay_reserved_until_the_caller_reads_them() {
        let (sink, g, ctx) = setup(Some(1 << 20), 2);
        let pool = Arc::clone(ctx[0].query().pool());
        sink.expect_ordered(&g, 0, 0..2);
        let mut l0 = ok(sink.local_state(&ctx[0], &g));
        let mut l1 = ok(sink.local_state(&ctx[1], &g));
        sink.position(&mut l0, &g, 0);
        sink.position(&mut l1, &g, 1);
        let base = pool.reserved();
        let big = |v: i64| {
            let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
            Batch::new(ok(RecordBatch::try_new(
                schema,
                vec![Arc::new(Int64Array::from(vec![v; 4 * 1024]))],
            )))
        };
        // Driver 1 is ahead: its batch is held, reserved.
        assert_eq!(status(&ok(sink.sink(big(2), &mut l1, &g))), "more");
        let held = pool.reserved();
        assert!(held > base, "{held} <= {base}");
        // Driver 0 ends: the batch is released into the channel. The caller
        // has not read it, so the memory is still reserved.
        ok(sink.combine(l0, &g));
        assert_eq!(pool.reserved(), held);
        assert_eq!(received(&sink).len(), 1);
        assert_eq!(pool.reserved(), base);
        // A stream dropped with released batches still queued gives them back.
        let mut l1 = l1;
        assert_eq!(status(&ok(sink.sink(big(3), &mut l1, &g))), "more");
        assert!(pool.reserved() >= base);
        sink.channel.drop_receiver();
        ok(sink.combine(l1, &g));
        assert_eq!(pool.reserved(), base);
    }

    /// A view column is charged its views and its long values' bytes, not
    /// the buffers it shares with the batch it was sliced from.
    #[test]
    fn held_batches_are_charged_their_rows() {
        use arrow_array::StringViewArray;
        let long = "x".repeat(100);
        let values: Vec<String> = (0..10_000).map(|i| format!("{long}{i}")).collect();
        let all = StringViewArray::from_iter_values(values.iter());
        let schema = Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::Utf8View,
            false,
        )]));
        let two = all.slice(5, 2);
        let b = Batch::new(ok(RecordBatch::try_new(schema, vec![Arc::new(two)])));
        let want = 2 * 16 + values[5].len() + values[6].len();
        assert_eq!(held_bytes(&b), want);
        assert_eq!(held_bytes(&batch(1)), 8);
    }

    /// A driver the runner did not register sends straight on.
    #[test]
    fn unordered_drivers_send_at_once() {
        let (sink, g, ctx) = setup(None, 2);
        sink.expect_ordered(&g, 0, 0..1);
        let mut l0 = ok(sink.local_state(&ctx[0], &g));
        let mut l1 = ok(sink.local_state(&ctx[1], &g));
        sink.position(&mut l1, &g, 5);
        assert_eq!(status(&ok(sink.sink(batch(1), &mut l1, &g))), "more");
        assert_eq!(received(&sink), [1]);
        ok(sink.combine(l1, &g));
        sink.position(&mut l0, &g, 0);
        ok(sink.combine(l0, &g));
    }
}
