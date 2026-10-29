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
//! memory reserved through their consumer (taken, never awaited) and at
//! most [`ORDER_BATCHES`] each, and wait once over either limit. When the
//! smallest key moves on (a driver reports a new morsel, or ends), the
//! batches it passed are sent in key order. The driver with the smallest
//! key is never held back, so the order cannot deadlock.

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
    queue: VecDeque<Batch>,
    /// Set by the query's coordinator once every task has ended.
    end: Option<Option<KrishivError>>,
    receiver_gone: bool,
    receiver: Option<Waker>,
    senders: Vec<Waker>,
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
        let (full, waker) = {
            let mut s = lock(&self.state);
            if s.receiver_gone {
                return None;
            }
            s.queue.push_back(batch);
            (s.queue.len() >= self.capacity, s.receiver.take())
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
            if s.receiver_gone || s.queue.len() < self.capacity {
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
        let (out, senders) = {
            let mut s = lock(&self.state);
            if let Some(b) = s.queue.pop_front() {
                // Wake every blocked sender, not one: a woken sender may
                // have been cancelled (its waker is stale), and waking only
                // it would strand the others. There are at most as many
                // senders as worker slots.
                (Poll::Ready(Some(Ok(b))), std::mem::take(&mut s.senders))
            } else if let Some(end) = s.end.as_mut() {
                // An error is reported once, then the stream ends.
                (Poll::Ready(end.take().map(Err)), Vec::new())
            } else {
                s.receiver = Some(cx.waker().clone());
                (Poll::Pending, Vec::new())
            }
        };
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
        let senders = {
            let mut s = lock(&self.state);
            s.receiver_gone = true;
            s.queue.clear();
            std::mem::take(&mut s.senders)
        };
        wake_all(senders);
    }
}

/// Most batches one driver holds back waiting for its turn (module docs).
pub(crate) const ORDER_BATCHES: usize = 64;

/// A held-back batch: the driver holding it, the batch and the bytes it
/// reserved.
type Held = (u32, Batch, usize);

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
    /// Batches it holds back, and the memory they hold.
    held: usize,
    memory: Option<Reservation>,
    /// Its memory ran short: it waits until nothing it holds is left.
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
        self.held >= ORDER_BATCHES || (self.short && self.held > 0)
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
    /// in key order; returns the wakers of drivers that may go on.
    fn release(&mut self, channel: &Channel) -> Vec<Waker> {
        let min = self.min();
        let mut wakers = Vec::new();
        while let Some(entry) = self.pending.first_entry() {
            if min.is_some_and(|m| *entry.key() > m) {
                break;
            }
            for (partition, batch, bytes) in entry.remove() {
                // A receiver that is gone takes nothing: the batch is dropped.
                let _ = channel.send(batch);
                if let Some(t) = self.drivers.get_mut(&partition) {
                    t.held = t.held.saturating_sub(1);
                    if let Some(m) = t.memory.as_mut() {
                        m.shrink(bytes.min(m.size()));
                    }
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
                    memory: None,
                    short: false,
                    waker: None,
                },
            );
        }
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        let partition = ctx.partition();
        let mut order = lock(&result_global(global)?.order);
        let ordered = match order.drivers.get_mut(&partition) {
            Some(t) => {
                t.memory = Some(ctx.consumer().reservation());
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
            return Ok(SinkStatus::Finished);
        }
        let bytes = input.data().get_array_memory_size();
        let reserved = t.memory.as_mut().is_some_and(|m| m.try_grow(bytes).is_ok());
        t.short |= !reserved;
        t.held += 1;
        let wait = t.over();
        let bytes = if reserved { bytes } else { 0 };
        order
            .pending
            .entry(key)
            .or_default()
            .push_back((partition, input, bytes));
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

    /// Without memory for a held batch the driver waits until its batches
    /// are sent; a dropped receiver ends every wait.
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
