// SPDX-License-Identifier: Apache-2.0
//! The relation sink over a runtime whose tasks hold scarce slots (design
//! P3-T085 §3.3a, review 2026-11-16 finding 1): a driver opens its lanes'
//! tasks together or waits holding none, a driver needing more lanes than
//! slots fails at once, and more drivers than slots finish because a
//! waiting driver keeps the combined drivers' tasks moving. Every test runs
//! under a deadline, so a hang fails it.
#![expect(
    clippy::disallowed_methods,
    reason = "test-only runtime plumbing; never runs on a compute worker"
)]

use std::future::poll_fn;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::future::BoxFuture;
use futures::task::AtomicWaker;
use krishiv_common::error::{ErrorClass, KrishivPythonWorkerLimit};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillOwner, TreePool};
use krishiv_types::batch::Batch;

use super::sink::{InputColumn, RelationLimits};
use super::{
    Frame, PreparedRelationUdf, RelationLane, RelationShape, RelationSink, RelationSlot,
    RelationSlots, RelationTicket, UdfTask,
};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus};
use crate::operators::ScalarExpr;
use crate::{
    ExecConfig, Executor, OperatorId, QueryContext, QueryId, TaskContext, TokioClock,
    TokioExecutor, WorkloadClass,
};

const DEADLINE: Duration = Duration::from_secs(10);

/// The slots of one session of a fake bounded runtime, with the worker
/// pool's rules (`krishiv-udf-pool` `CallThreads::take_slots`): at most
/// `limit` of this session's, at most `total` of all sessions' (`state`
/// is shared by the sessions), granted in arrival order to waiters holding
/// tickets. Every ticket it hands out is counted while alive, so a test sees
/// one left queued.
pub(super) struct Slots {
    pub(super) owner: usize,
    pub(super) limit: usize,
    pub(super) total: usize,
    pub(super) state: Arc<SlotState>,
}

#[derive(Default)]
pub(super) struct SlotState {
    inner: Mutex<SlotInner>,
    /// The most slots held at once, in all.
    pub(super) most: AtomicUsize,
    /// Tickets alive.
    pub(super) tickets: AtomicUsize,
}

#[derive(Default)]
struct SlotInner {
    held: std::collections::HashMap<usize, usize>,
    total: usize,
    queue: std::collections::VecDeque<FakeWaiter>,
    next: u64,
}

struct FakeWaiter {
    id: u64,
    owner: usize,
    n: usize,
    limit: usize,
    waker: Waker,
}

impl SlotState {
    /// Slots `owner` holds.
    pub(super) fn held(&self, owner: usize) -> usize {
        self.inner
            .lock()
            .unwrap()
            .held
            .get(&owner)
            .copied()
            .unwrap_or(0)
    }

    fn wakers(inner: &SlotInner) -> Vec<Waker> {
        inner.queue.iter().map(|w| w.waker.clone()).collect()
    }
}

impl Slots {
    /// Slots this session holds.
    pub(super) fn held(&self) -> usize {
        self.state.held(self.owner)
    }
}

/// One slot; frees and wakes the waiters when it drops.
struct Held(Arc<SlotState>, usize);

impl Drop for Held {
    fn drop(&mut self) {
        let wakers = {
            let mut inner = self.0.inner.lock().unwrap();
            *inner.held.entry(self.1).or_default() -= 1;
            inner.total -= 1;
            SlotState::wakers(&inner)
        };
        wakers.into_iter().for_each(Waker::wake);
    }
}

/// A waiter's place; leaves the queue when it drops.
struct FakeTicket(Arc<SlotState>, u64);

impl Drop for FakeTicket {
    fn drop(&mut self) {
        self.0.tickets.fetch_sub(1, Ordering::SeqCst);
        let wakers = {
            let mut inner = self.0.inner.lock().unwrap();
            let Some(at) = inner.queue.iter().position(|w| w.id == self.1) else {
                return;
            };
            inner.queue.remove(at);
            SlotState::wakers(&inner)
        };
        wakers.into_iter().for_each(Waker::wake);
    }
}

impl RelationSlots for Slots {
    fn limit(&self) -> usize {
        self.limit
    }

    fn poll_take(
        &self,
        n: usize,
        ticket: &mut Option<RelationTicket>,
        cx: &mut Context<'_>,
    ) -> Poll<crate::error::Result<Vec<RelationSlot>>> {
        if n > self.limit {
            return Poll::Ready(Err(KrishivPythonWorkerLimit {
                operation: "a fake operator".to_owned(),
                needed: n.to_string(),
                available: self.limit.to_string(),
                setting: "krishiv.python.worker.maxPerSession".to_owned(),
                value: (self.limit + 1).to_string(),
            }
            .into()));
        }
        let own: Option<Box<FakeTicket>> = ticket
            .take()
            .and_then(|t| t.into_inner().downcast::<FakeTicket>().ok());
        let mut inner = self.state.inner.lock().unwrap();
        let at = own
            .as_ref()
            .and_then(|t| inner.queue.iter().position(|w| w.id == t.1));
        let ahead = at.unwrap_or(inner.queue.len());
        let held = |inner: &SlotInner, o: usize| inner.held.get(&o).copied().unwrap_or(0);
        let held_back = inner
            .queue
            .iter()
            .take(ahead)
            .any(|w| w.owner == self.owner || held(&inner, w.owner) + w.n <= w.limit);
        let fits = held(&inner, self.owner) + n <= self.limit && inner.total + n <= self.total;
        if held_back || !fits {
            if let Some(w) = at.and_then(|i| inner.queue.get_mut(i)) {
                w.waker = cx.waker().clone();
                drop(inner);
                *ticket = own.map(|t| RelationTicket::new(*t));
                return Poll::Pending;
            }
            let id = inner.next;
            inner.next += 1;
            inner.queue.push_back(FakeWaiter {
                id,
                owner: self.owner,
                n,
                limit: self.limit,
                waker: cx.waker().clone(),
            });
            drop(inner);
            drop(own);
            self.state.tickets.fetch_add(1, Ordering::SeqCst);
            *ticket = Some(RelationTicket::new(FakeTicket(Arc::clone(&self.state), id)));
            return Poll::Pending;
        }
        if let Some(i) = at {
            inner.queue.remove(i);
        }
        *inner.held.entry(self.owner).or_default() += n;
        inner.total += n;
        self.state.most.fetch_max(inner.total, Ordering::SeqCst);
        let wakers = SlotState::wakers(&inner);
        drop(inner);
        drop(own);
        wakers.into_iter().for_each(Waker::wake);
        Poll::Ready(Ok((0..n)
            .map(|_| RelationSlot::new(Held(Arc::clone(&self.state), self.owner)))
            .collect()))
    }
}

/// A call site whose tasks hold their slot until their `End` is processed
/// (as a pool task holds its worker until its thread ends). `End` takes two
/// polls, so a task polled only once with a waker nobody listens to keeps
/// its slot.
struct Udf {
    slots: Arc<Slots>,
    opened: AtomicUsize,
    ends: Arc<AtomicUsize>,
    /// Rows of the one result batch a task yields at its end (0: none). A
    /// task with a result frees its slot only once that is read, as a pool
    /// task's thread waits for the operator to take its result.
    result_rows: usize,
    /// Two such results instead of one.
    twice: std::sync::atomic::AtomicBool,
}

impl PreparedRelationUdf for Udf {
    fn open(&self, slot: Option<RelationSlot>) -> crate::error::Result<Box<dyn UdfTask>> {
        let Some(slot) = slot else {
            return Err(crate::error::internal("opened without a slot".to_owned()));
        };
        self.opened.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Task {
            slot: Mutex::new(Some(slot)),
            end_polled: Mutex::new(false),
            ended: Mutex::new(false),
            reader: AtomicWaker::new(),
            ends: Arc::clone(&self.ends),
            result: Mutex::new(std::collections::VecDeque::new()),
            result_rows: self.result_rows,
            twice: self.twice.load(Ordering::SeqCst),
        }))
    }

    fn capacity(&self) -> Option<Arc<dyn RelationSlots>> {
        Some(Arc::clone(&self.slots) as Arc<dyn RelationSlots>)
    }
}

struct Task {
    slot: Mutex<Option<RelationSlot>>,
    end_polled: Mutex<bool>,
    ended: Mutex<bool>,
    reader: AtomicWaker,
    ends: Arc<AtomicUsize>,
    result: Mutex<std::collections::VecDeque<RecordBatch>>,
    result_rows: usize,
    twice: bool,
}

impl UdfTask for Task {
    fn send(&self, frame: Frame) -> BoxFuture<'_, crate::error::Result<()>> {
        let end = matches!(frame, Frame::End);
        Box::pin(poll_fn(move |cx| {
            if end {
                let mut polled = self.end_polled.lock().unwrap();
                // A task with a result ends in one poll, so `combine`'s own
                // poll (no waker) already meets the pool's refusal.
                if !*polled && self.result_rows == 0 {
                    *polled = true;
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                *self.ended.lock().unwrap() = true;
                if self.result_rows > 0 {
                    let v: Int64Array = (0..self.result_rows).map(|i| Some(i as i64)).collect();
                    let schema =
                        Arc::new(Schema::new(vec![Field::new("v", ArrowType::Int64, true)]));
                    let batch =
                        RecordBatch::try_new(schema, vec![Arc::new(v) as ArrayRef]).unwrap();
                    let mut out = self.result.lock().unwrap();
                    if self.twice {
                        out.push_back(batch.clone());
                    }
                    out.push_back(batch);
                } else {
                    drop(self.slot.lock().unwrap().take());
                }
                self.ends.fetch_add(1, Ordering::SeqCst);
                self.reader.wake();
            }
            Poll::Ready(Ok(()))
        }))
    }

    fn next(&self) -> BoxFuture<'_, crate::error::Result<Option<RecordBatch>>> {
        Box::pin(poll_fn(|cx| {
            self.reader.register(cx.waker());
            if let Some(batch) = self.result.lock().unwrap().pop_front() {
                return Poll::Ready(Ok(Some(batch)));
            }
            if *self.ended.lock().unwrap() {
                drop(self.slot.lock().unwrap().take());
                Poll::Ready(Ok(None))
            } else {
                Poll::Pending
            }
        }))
    }
}

fn input() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", ArrowType::Int32, true),
        Field::new("v", ArrowType::Int64, true),
    ]))
}

fn rows(keys: &[i32]) -> Batch {
    let k: Int32Array = keys.iter().map(|&k| Some(k)).collect();
    let v: Int64Array = (0..keys.len()).map(|i| Some(i as i64)).collect();
    Batch::new(RecordBatch::try_new(input(), vec![Arc::new(k) as ArrayRef, Arc::new(v)]).unwrap())
}

struct Rig {
    sink: Arc<RelationSink>,
    ctx: TaskContext,
    udfs: Vec<Arc<Udf>>,
    slots: Arc<Slots>,
    ends: Arc<AtomicUsize>,
}

/// A sink of `lanes` lanes (an aggregate's when more than one) over slots
/// of `limit`.
fn rig(lanes: usize, limit: usize) -> Rig {
    rig_with(lanes, limit, 64 << 20, 0)
}

/// As [`rig`], under a memory limit of `memory` bytes, each task yielding a
/// batch of `result_rows` rows at its end.
fn rig_with(lanes: usize, limit: usize, memory: usize, result_rows: usize) -> Rig {
    rig_full(lanes, limit, memory, result_rows, 0)
}

/// As [`rig_with`], frames of at most `max_rows` rows (0: no limit), so a
/// driver with fewer rows holds them until its input ends.
fn rig_full(lanes: usize, limit: usize, memory: usize, result_rows: usize, max_rows: usize) -> Rig {
    let slots = Arc::new(Slots {
        owner: 0,
        limit,
        total: usize::MAX / 2,
        state: Arc::default(),
    });
    let ends = Arc::new(AtomicUsize::new(0));
    let udfs: Vec<Arc<Udf>> = (0..lanes)
        .map(|_| {
            Arc::new(Udf {
                slots: Arc::clone(&slots),
                opened: AtomicUsize::new(0),
                ends: Arc::clone(&ends),
                result_rows,
                twice: std::sync::atomic::AtomicBool::new(false),
            })
        })
        .collect();
    let column = |index, data_type| -> Arc<dyn ScalarExpr> {
        Arc::new(InputColumn {
            index,
            data_type,
            nullable: true,
        })
    };
    let specs: Vec<RelationLane> = udfs
        .iter()
        .map(|u| RelationLane {
            name: "f".to_owned(),
            udf: Arc::clone(u) as Arc<dyn PreparedRelationUdf>,
            sides: vec![vec![column(1, ArrowType::Int64)]],
            names: vec![vec!["v".to_owned()]],
            client: vec![ArrowType::Int64],
        })
        .collect();
    let (shape, output) = if lanes == 1 {
        (
            RelationShape::Map,
            vec![Field::new("v", ArrowType::Int64, true)],
        )
    } else {
        let mut fields = vec![Field::new("k", ArrowType::Int32, true)];
        fields.extend((0..lanes).map(|i| Field::new(format!("n{i}"), ArrowType::Int64, true)));
        (
            RelationShape::Aggregate {
                keys: vec![0],
                key_values: vec![0],
            },
            fields,
        )
    };
    let sink = RelationSink::try_new(
        input(),
        shape,
        specs,
        Arc::new(Schema::new(output)),
        RelationLimits {
            max_rows,
            max_bytes: 0,
            window: 4,
        },
        None,
    )
    .unwrap();
    let root = TreePool::with_clock(
        &MemoryConfig {
            limit: Some(memory),
            quantum: 1 << 10,
            spill_wait_timeout: Duration::from_secs(5),
            abort_release_timeout: Duration::from_secs(30),
            spill_scratch_reserve: krishiv_memory::DEFAULT_SPILL_SCRATCH_RESERVE,
            resident_guard: true,
        },
        Arc::new(TokioClock),
    );
    let query = QueryContext::new(
        QueryId(86),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("q", Some(memory)),
    );
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
    let ctx = TaskContext::new(
        query,
        executor,
        SpillOwner::new(),
        OperatorId(0),
        "python_relation",
    );
    Rig {
        sink,
        ctx,
        udfs,
        slots,
        ends,
    }
}

/// Sinks `batch` into one driver's state, awaiting a `Blocked` status.
async fn feed(
    r: &Rig,
    batch: Batch,
    l: &mut LocalState,
    g: &GlobalState,
) -> crate::error::Result<()> {
    match r.sink.sink(batch, l, g)? {
        SinkStatus::Blocked(wait) => {
            wait.await;
            // A failure of the blocked future's framing is the next call's.
            match r
                .sink
                .sink(Batch::new(RecordBatch::new_empty(input())), l, g)?
            {
                SinkStatus::Blocked(wait) => wait.await,
                SinkStatus::NeedMemory { bytes } => panic!("no memory for {bytes}"),
                SinkStatus::NeedMoreInput | SinkStatus::Finished => {}
            }
            Ok(())
        }
        SinkStatus::NeedMemory { bytes } => panic!("no memory for {bytes}"),
        SinkStatus::NeedMoreInput | SinkStatus::Finished => Ok(()),
    }
}

/// An aggregate of more UDFs than the runtime has slots (9 over 8 in the
/// pool's default) fails at once, naming the setting, instead of opening the
/// tasks it can and hanging on the rest.
#[tokio::test]
async fn a_driver_needing_more_lanes_than_slots_fails_at_once() {
    // `maxPerSession` 2 and 1 both give one relation slot.
    let r = rig(2, 1);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    let err = tokio::time::timeout(DEADLINE, feed(&r, rows(&[1, 1, 2]), &mut l, &g))
        .await
        .expect("the driver hung instead of failing")
        .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivPythonWorkerLimit);
    assert_eq!(
        r.udfs
            .iter()
            .map(|u| u.opened.load(Ordering::SeqCst))
            .sum::<usize>(),
        0
    );
    assert_eq!(r.slots.held(), 0, "no slot left held");
}

/// Two drivers over one slot: the second waits for the first's task, which
/// ends only once its `End` is pumped after the first combined; the waiting
/// driver's `Blocked` future keeps it moving.
#[tokio::test]
async fn more_drivers_than_slots_finish() {
    let r = rig(1, 1);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let run = async {
        let mut a = r.sink.local_state(&r.ctx, &g)?;
        let mut b = r.sink.local_state(&r.ctx, &g)?;
        feed(&r, rows(&[1, 2]), &mut a, &g).await?;
        assert_eq!(r.slots.held(), 1);
        // `b` waits for `a`'s slot; `a` combines meanwhile.
        let SinkStatus::Blocked(wait) = r.sink.sink(rows(&[3]), &mut b, &g)? else {
            panic!("the second driver must wait for the slot");
        };
        let waiting = tokio::spawn(wait);
        tokio::task::yield_now().await;
        r.sink.combine(a, &g)?;
        waiting.await.unwrap();
        r.sink.combine(b, &g)?;
        r.sink.finalize(&g, &r.ctx).await
    };
    tokio::time::timeout(DEADLINE, run)
        .await
        .expect("the waiting driver hung")
        .unwrap();
    assert_eq!(r.udfs[0].opened.load(Ordering::SeqCst), 2);
    assert_eq!(r.ends.load(Ordering::SeqCst), 2);
    assert_eq!(r.slots.state.most.load(Ordering::SeqCst), 1);
}

/// With two slots and an aggregate of two lanes, three drivers take the
/// slots in turn, always both of a driver's at once.
#[tokio::test]
async fn a_driver_takes_all_its_lanes_slots_together() {
    let r = rig(2, 2);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let run = async {
        let mut drivers = Vec::new();
        for _ in 0..3 {
            drivers.push(r.sink.local_state(&r.ctx, &g)?);
        }
        let mut waits = Vec::new();
        for (i, l) in drivers.iter_mut().enumerate() {
            let key = i32::try_from(i).unwrap();
            match r.sink.sink(rows(&[key, key]), l, &g)? {
                SinkStatus::Blocked(wait) => waits.push(tokio::spawn(wait)),
                _ => waits.push(tokio::spawn(async {})),
            }
        }
        tokio::task::yield_now().await;
        // Driver 0 holds both slots; the others hold none.
        assert_eq!(r.slots.held(), 2);
        // Whichever driver got the slots next (any order) combines, freeing
        // them for the others.
        let mut pending: Vec<(LocalState, tokio::task::JoinHandle<()>)> =
            drivers.into_iter().zip(waits).collect();
        while !pending.is_empty() {
            let handles = pending.iter_mut().map(|(_, w)| w);
            let (done, at, _) = futures::future::select_all(handles).await;
            done.unwrap();
            let (mut l, _) = pending.swap_remove(at);
            feed(&r, Batch::new(RecordBatch::new_empty(input())), &mut l, &g).await?;
            r.sink.combine(l, &g)?;
        }
        r.sink.finalize(&g, &r.ctx).await
    };
    // An aggregate's tasks here yield no values: `finalize` names the
    // missing ones; the point is that nothing hangs.
    let err = tokio::time::timeout(DEADLINE, run)
        .await
        .expect("a driver hung waiting for slots")
        .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivPythonUdfResultMismatch,
        "{err}"
    );
    assert_eq!(r.slots.state.most.load(Ordering::SeqCst), 2);
    assert_eq!(
        r.udfs
            .iter()
            .map(|u| u.opened.load(Ordering::SeqCst))
            .sum::<usize>(),
        6
    );
    assert_eq!(r.slots.held(), 0);
}

/// A combined driver's result the memory pool refuses stops its task, which
/// then never frees its slot: the driver waiting for that slot must not
/// hang. Its `Blocked` future ends and its next `sink` asks for the memory
/// (`NeedMemory`, the input untaken), so the arbitrator spills or aborts
/// (review R2-2).
#[tokio::test]
async fn a_refused_result_of_a_combined_driver_asks_for_memory_instead_of_hanging() {
    // One slot; 1 MiB of memory; each task's result is 1.6 MB.
    let r = rig_with(1, 1, 1 << 20, 200_000);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let run = async {
        let mut a = r.sink.local_state(&r.ctx, &g)?;
        let mut b = r.sink.local_state(&r.ctx, &g)?;
        feed(&r, rows(&[1, 2]), &mut a, &g).await?;
        let SinkStatus::Blocked(wait) = r.sink.sink(rows(&[3]), &mut b, &g)? else {
            panic!("the second driver must wait for the slot");
        };
        let waiting = tokio::spawn(wait);
        tokio::task::yield_now().await;
        r.sink.combine(a, &g)?;
        waiting.await.unwrap();
        let status = r.sink.sink(rows(&[4]), &mut b, &g)?;
        assert!(
            matches!(status, SinkStatus::NeedMemory { bytes } if bytes > 0),
            "the driver must ask for the refused result's memory"
        );
        Ok::<(), krishiv_common::error::KrishivError>(())
    };
    tokio::time::timeout(DEADLINE, run)
        .await
        .expect("the waiting driver hung on a refused result")
        .unwrap();
    assert_eq!(r.slots.held(), 1, "a's task still holds its slot");
}

/// Another driver's context of the rig's query: its own consumer (as each
/// pipeline driver has) and spill owner.
fn driver_ctx(r: &Rig, op: u32) -> (TaskContext, Arc<SpillOwner>) {
    let owner = SpillOwner::new();
    let ctx = TaskContext::new(
        Arc::clone(r.ctx.query()),
        Arc::clone(r.ctx.executor()),
        Arc::clone(&owner),
        OperatorId(op),
        "python_relation",
    );
    (ctx, owner)
}

/// The rest of a two-driver run once `b` holds its slot: `b` combines and
/// the phase finalizes.
async fn finish(r: &Rig, mut b: LocalState, g: &GlobalState) -> crate::error::Result<()> {
    feed(r, Batch::new(RecordBatch::new_empty(input())), &mut b, g).await?;
    r.sink.combine(b, g)?;
    r.sink.finalize(g, &r.ctx).await
}

/// R3-1: the waiting driver's `Blocked` future reserves a combined driver's
/// refused result itself (its consumer has grant credit) and must poll that
/// lane again: the refusal happened in `combine`'s poll, which has no waker,
/// and the pump waits with none while the result is refused, so without
/// that poll the task never ends and its slot never frees.
#[tokio::test]
async fn a_combined_result_reserved_by_the_waiting_driver_moves_its_task_on() {
    // 4 MiB; each task's result is about 1.6 MB.
    let r = rig_with(1, 1, 4 << 20, 200_000);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let (ctx_a, _) = driver_ctx(&r, 1);
    let (ctx_b, owner_b) = driver_ctx(&r, 2);
    let (ctx_c, _) = driver_ctx(&r, 3);
    let run = async {
        let mut a = r.sink.local_state(&ctx_a, &g)?;
        let mut b = r.sink.local_state(&ctx_b, &g)?;
        feed(&r, rows(&[1, 2]), &mut a, &g).await?;
        let SinkStatus::Blocked(wait) = r.sink.sink(rows(&[3]), &mut b, &g)? else {
            panic!("the second driver must wait for the slot");
        };
        // b holds 2 MB of grant credit; the pool has room for a's result
        // only through it.
        let b_consumer = Arc::clone(ctx_b.consumer().consumer());
        owner_b
            .reserve_for(&b_consumer, 2_000_000, || async {})
            .await
            .unwrap();
        let mut blocker = ctx_c.consumer().consumer().reservation();
        blocker
            .try_grow((4 << 20) - 2_000_000 - (768 << 10))
            .unwrap();
        r.sink.combine(a, &g)?;
        wait.await;
        assert!(
            b_consumer.credit() < 2_000_000,
            "the result used b's credit"
        );
        drop(blocker);
        finish(&r, b, &g).await
    };
    tokio::time::timeout(DEADLINE, run)
        .await
        .expect("the combined task never moved on")
        .unwrap();
    assert_eq!(r.ends.load(Ordering::SeqCst), 2);
}

/// R3-2: a result that fits the memory limit but is refused at first: the
/// waiting driver's `NeedMemory` grant (credit on its own consumer) is what
/// its retry reserves the result with, so the credit is used once and does
/// not grow, and the phase completes.
#[tokio::test]
async fn a_need_memory_grant_for_a_combined_result_is_used_by_the_retry() {
    let r = rig_with(1, 1, 4 << 20, 200_000);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let (ctx_a, _) = driver_ctx(&r, 1);
    let (ctx_b, owner_b) = driver_ctx(&r, 2);
    let (ctx_c, _) = driver_ctx(&r, 3);
    let run = async {
        let mut a = r.sink.local_state(&ctx_a, &g)?;
        let mut b = r.sink.local_state(&ctx_b, &g)?;
        feed(&r, rows(&[1, 2]), &mut a, &g).await?;
        let mut blocker = ctx_c.consumer().consumer().reservation();
        blocker.try_grow(3 << 20).unwrap();
        let SinkStatus::Blocked(wait) = r.sink.sink(rows(&[3]), &mut b, &g)? else {
            panic!("the second driver must wait for the slot");
        };
        let waiting = tokio::spawn(wait);
        tokio::task::yield_now().await;
        r.sink.combine(a, &g)?;
        waiting.await.unwrap();
        let SinkStatus::NeedMemory { bytes } = r.sink.sink(rows(&[4]), &mut b, &g)? else {
            panic!("the driver must ask for the refused result's memory");
        };
        // The driver's loop: memory freed (a spill), then granted to b.
        drop(blocker);
        let b_consumer = Arc::clone(ctx_b.consumer().consumer());
        owner_b
            .reserve_for(&b_consumer, bytes, || async {})
            .await
            .unwrap();
        match r.sink.sink(rows(&[4]), &mut b, &g)? {
            SinkStatus::NeedMemory { .. } => panic!("the retry must use the grant"),
            SinkStatus::Blocked(wait) => wait.await,
            SinkStatus::NeedMoreInput | SinkStatus::Finished => {}
        }
        assert_eq!(b_consumer.credit(), 0, "the grant was used, not grown");
        finish(&r, b, &g).await
    };
    tokio::time::timeout(DEADLINE, run)
        .await
        .expect("the phase hung")
        .unwrap();
    assert_eq!(r.ends.load(Ordering::SeqCst), 2);
}

/// R4-1: a driver that combines with rows still held (fewer than a frame)
/// while the slots are full must not keep a place in the slot queue that
/// nobody polls: the sibling driver asking next would wait behind it until
/// `finalize`, which cannot start while that sibling is `Blocked`.
#[tokio::test]
async fn a_driver_combining_while_slots_are_full_keeps_no_place_in_the_queue() {
    let r = rig_full(1, 1, 64 << 20, 0, 4);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let run = async {
        let mut a = r.sink.local_state(&r.ctx, &g)?;
        let mut d1 = r.sink.local_state(&r.ctx, &g)?;
        let mut d3 = r.sink.local_state(&r.ctx, &g)?;
        // a takes the slot (a whole frame).
        feed(&r, rows(&[1, 2, 3, 4]), &mut a, &g).await?;
        assert_eq!(r.slots.held(), 1);
        // d1's one row is held, not framed: it needs no slot yet...
        assert!(matches!(
            r.sink.sink(rows(&[5]), &mut d1, &g)?,
            SinkStatus::NeedMoreInput
        ));
        // ...until it combines, while a still holds the slot.
        r.sink.combine(d1, &g)?;
        assert_eq!(
            r.slots.state.tickets.load(Ordering::SeqCst),
            0,
            "combine left a ticket"
        );
        r.sink.combine(a, &g)?;
        // d3 asks next and must be admitted once a's task ends.
        feed(&r, rows(&[6, 7, 8, 9]), &mut d3, &g).await?;
        r.sink.combine(d3, &g)?;
        r.sink.finalize(&g, &r.ctx).await
    };
    tokio::time::timeout(DEADLINE, run)
        .await
        .expect("a sibling driver waited behind a combined driver's ticket")
        .unwrap();
    assert_eq!(r.ends.load(Ordering::SeqCst), 3);
    assert_eq!(r.slots.state.tickets.load(Ordering::SeqCst), 0);
}

/// R4-4: a waiting driver's ticket is kept between polls, cleared on the
/// grant, and dropped with the driver's state when it stops waiting.
#[tokio::test]
async fn a_waiting_driver_s_ticket_lives_only_while_it_waits() {
    let r = rig(1, 1);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let run = async {
        let mut a = r.sink.local_state(&r.ctx, &g)?;
        let mut b = r.sink.local_state(&r.ctx, &g)?;
        let mut c = r.sink.local_state(&r.ctx, &g)?;
        feed(&r, rows(&[1]), &mut a, &g).await?;
        let SinkStatus::Blocked(wait_b) = r.sink.sink(rows(&[2]), &mut b, &g)? else {
            panic!("b must wait");
        };
        let SinkStatus::Blocked(wait_c) = r.sink.sink(rows(&[3]), &mut c, &g)? else {
            panic!("c must wait");
        };
        assert_eq!(
            r.slots.state.tickets.load(Ordering::SeqCst),
            2,
            "one ticket per waiter"
        );
        // c stops waiting (its query was cancelled): its ticket goes with it.
        drop(wait_c);
        drop(c);
        assert_eq!(r.slots.state.tickets.load(Ordering::SeqCst), 1);
        let waiting = tokio::spawn(wait_b);
        tokio::task::yield_now().await;
        r.sink.combine(a, &g)?;
        waiting.await.unwrap();
        assert_eq!(
            r.slots.state.tickets.load(Ordering::SeqCst),
            0,
            "the grant cleared it"
        );
        r.sink.combine(b, &g)?;
        r.sink.finalize(&g, &r.ctx).await
    };
    tokio::time::timeout(DEADLINE, run)
        .await
        .expect("hung")
        .unwrap();
}

/// R4-2: a combined task with two results the pool refuses one after the
/// other: the waiting driver reserves the first, the second is refused in
/// the same poll, and the driver must ask for its memory (or finish), not
/// park with the pump stalled.
#[tokio::test]
async fn a_second_refused_result_of_a_combined_task_is_not_lost() {
    // Each task yields two results of about 1.6 MB (`result_rows` twice).
    let r = rig_with(1, 1, 4 << 20, 200_000);
    r.udfs[0].twice.store(true, Ordering::SeqCst);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let (ctx_a, _) = driver_ctx(&r, 1);
    let (ctx_b, owner_b) = driver_ctx(&r, 2);
    let (ctx_c, _) = driver_ctx(&r, 3);
    let run = async {
        let mut a = r.sink.local_state(&ctx_a, &g)?;
        let mut b = r.sink.local_state(&ctx_b, &g)?;
        feed(&r, rows(&[1, 2]), &mut a, &g).await?;
        let SinkStatus::Blocked(wait) = r.sink.sink(rows(&[3]), &mut b, &g)? else {
            panic!("the second driver must wait for the slot");
        };
        // b's credit covers one result; the pool has room for neither.
        let b_consumer = Arc::clone(ctx_b.consumer().consumer());
        owner_b
            .reserve_for(&b_consumer, 2_000_000, || async {})
            .await
            .unwrap();
        let mut blocker = ctx_c.consumer().consumer().reservation();
        blocker
            .try_grow((4 << 20) - 2_000_000 - (768 << 10))
            .unwrap();
        r.sink.combine(a, &g)?;
        wait.await;
        // The second result needs memory: the next call asks for it.
        let status = r.sink.sink(rows(&[4]), &mut b, &g)?;
        assert!(
            matches!(status, SinkStatus::NeedMemory { .. }),
            "the second refused result must be asked for"
        );
        drop(blocker);
        let SinkStatus::NeedMemory { bytes } = status else {
            unreachable!()
        };
        owner_b
            .reserve_for(&b_consumer, bytes, || async {})
            .await
            .unwrap();
        match r.sink.sink(rows(&[4]), &mut b, &g)? {
            SinkStatus::Blocked(wait) => wait.await,
            SinkStatus::NeedMemory { .. } => panic!("the retry must use the grant"),
            SinkStatus::NeedMoreInput | SinkStatus::Finished => {}
        }
        // b's own two results do not fit beside a's in this store (no spill
        // scope): the phase ends with the pool's classed error, not a hang.
        match finish(&r, b, &g).await {
            Err(e) if e.error_class() == ErrorClass::UnableToAcquireMemory => Ok(()),
            other => other,
        }
    };
    tokio::time::timeout(DEADLINE, run)
        .await
        .expect("the second refused result stalled the combined task")
        .unwrap();
}

/// A small seeded generator (xorshift64*), so a failing case is named by
/// its seed and runs again the same way.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) % n.max(1)
    }

    /// A value in `lo..=hi`.
    fn pick(&mut self, lo: usize, hi: usize) -> usize {
        lo + usize::try_from(self.below(u64::try_from(hi - lo + 1).unwrap())).unwrap()
    }
}

/// One stress case's shape.
#[derive(Debug, Clone)]
struct Case {
    sessions: usize,
    drivers: usize,
    lanes: usize,
    per_session: usize,
    total: usize,
    max_rows: usize,
    memory: usize,
    result_rows: usize,
    /// Each driver's input: batches of rows.
    batches: Vec<Vec<usize>>,
    /// The session cancelled, and after how many yields.
    cancel: Option<(usize, usize)>,
}

impl Case {
    fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let drivers = rng.pick(1, 6);
        let sessions = rng.pick(1, 2);
        Case {
            sessions,
            drivers,
            lanes: rng.pick(1, 3),
            per_session: rng.pick(1, 3),
            total: rng.pick(1, 3),
            max_rows: rng.pick(1, 2),
            memory: rng.pick(64, 512) << 10,
            result_rows: rng.pick(0, 4) * 1000,
            batches: (0..drivers * sessions)
                .map(|_| (0..rng.pick(0, 3)).map(|_| rng.pick(1, 5)).collect())
                .collect(),
            cancel: (rng.below(5) == 0).then(|| (rng.pick(0, sessions - 1), rng.pick(0, 20))),
        }
    }
}

/// One session of a case: its sink (`lanes` map lanes over the session's
/// slots) and query.
struct Session {
    sink: Arc<RelationSink>,
    query: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
}

fn session(case: &Case, owner: usize, state: &Arc<SlotState>, root: &Arc<TreePool>) -> Session {
    let slots = Arc::new(Slots {
        owner,
        limit: case.per_session.min(case.total).saturating_sub(1).max(1),
        total: case.total.saturating_sub(1).max(1),
        state: Arc::clone(state),
    });
    let specs: Vec<RelationLane> = (0..case.lanes)
        .map(|_| RelationLane {
            name: "f".to_owned(),
            udf: Arc::new(Udf {
                slots: Arc::clone(&slots),
                opened: AtomicUsize::new(0),
                ends: Arc::new(AtomicUsize::new(0)),
                result_rows: case.result_rows,
                twice: std::sync::atomic::AtomicBool::new(owner % 2 == 1),
            }) as Arc<dyn PreparedRelationUdf>,
            sides: vec![vec![Arc::new(InputColumn {
                index: 1,
                data_type: ArrowType::Int64,
                nullable: true,
            }) as Arc<dyn ScalarExpr>]],
            names: vec![vec!["v".to_owned()]],
            client: vec![ArrowType::Int64],
        })
        .collect();
    let sink = RelationSink::try_new(
        input(),
        RelationShape::Map,
        specs,
        Arc::new(Schema::new(vec![Field::new("v", ArrowType::Int64, true)])),
        RelationLimits {
            max_rows: case.max_rows,
            max_bytes: 0,
            window: 2,
        },
        None,
    )
    .unwrap();
    let query = QueryContext::new(
        QueryId(87 + u64::try_from(owner).unwrap()),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("q", Some(case.memory)),
    );
    Session {
        sink,
        query,
        executor: TokioExecutor::deterministic(1).unwrap(),
    }
}

/// One driver as the pipeline driver runs it: each batch until taken
/// (`Blocked` awaited, `NeedMemory` granted the cooperative way), then
/// `combine`.
async fn drive_one(
    s: &Session,
    g: &GlobalState,
    op: u32,
    batches: Vec<usize>,
) -> crate::error::Result<()> {
    let owner = SpillOwner::new();
    let ctx = TaskContext::new(
        Arc::clone(&s.query),
        Arc::clone(&s.executor),
        Arc::clone(&owner),
        OperatorId(op),
        "python_relation",
    );
    let mut l = s.sink.local_state(&ctx, g)?;
    for (i, n) in batches.into_iter().enumerate() {
        let keys: Vec<i32> = (0..n).map(|k| i32::try_from(i * 10 + k).unwrap()).collect();
        loop {
            match s.sink.sink(rows(&keys), &mut l, g)? {
                SinkStatus::Blocked(wait) => {
                    wait.await;
                    break;
                }
                SinkStatus::NeedMemory { bytes } => {
                    let consumer = Arc::clone(ctx.consumer().consumer());
                    owner
                        .reserve_for(&consumer, bytes, || async {})
                        .await
                        .map_err(krishiv_common::error::KrishivError::from)?;
                }
                SinkStatus::NeedMoreInput | SinkStatus::Finished => break,
            }
        }
    }
    s.sink.combine(l, g)
}

/// Every driver of a session, then its `finalize`.
async fn run_session(s: &Session, case: &Case, owner: usize) -> crate::error::Result<()> {
    let ctx = TaskContext::new(
        Arc::clone(&s.query),
        Arc::clone(&s.executor),
        SpillOwner::new(),
        OperatorId(100),
        "python_relation",
    );
    let g = s.sink.global_state(&ctx)?;
    let drivers = (0..case.drivers).map(|d| {
        let batches = case.batches[owner * case.drivers + d].clone();
        drive_one(s, &g, u32::try_from(d + 1).unwrap(), batches)
    });
    futures::future::try_join_all(drivers).await?;
    s.sink.finalize(&g, &ctx).await
}

/// A seeded sweep over drivers 1-6, lanes 1-3, `maxPerSession` and
/// `maxTotal` 1-3, one or two sessions sharing the slots, frames of one or
/// two rows, small memory limits, results the pool may refuse and random
/// cancellation (a session's drivers dropped mid-way). Every case must end,
/// with its results or a classed error, within a deadline: a hang fails the
/// case by its seed (review round 4: waits nobody wakes).
#[tokio::test]
async fn seeded_relation_slot_stress_never_hangs() {
    let budget = std::time::Instant::now();
    let mut cases = 0_u64;
    while budget.elapsed() < Duration::from_secs(8) && cases < 3000 {
        let seed = cases;
        cases += 1;
        let case = Case::new(seed);
        let root = TreePool::with_clock(
            &MemoryConfig {
                limit: Some(2 << 20),
                quantum: 1 << 10,
                spill_wait_timeout: Duration::from_millis(20),
                abort_release_timeout: Duration::from_millis(50),
                spill_scratch_reserve: krishiv_memory::DEFAULT_SPILL_SCRATCH_RESERVE,
                resident_guard: true,
            },
            Arc::new(TokioClock),
        );
        let state = Arc::new(SlotState::default());
        let sessions: Vec<Session> = (0..case.sessions)
            .map(|o| session(&case, o, &state, &root))
            .collect();
        let runs = sessions.iter().enumerate().map(|(o, s)| {
            let case = &case;
            async move {
                let run = run_session(s, case, o);
                match case.cancel {
                    Some((victim, yields)) if victim == o => {
                        let cancel = async {
                            for _ in 0..yields {
                                tokio::task::yield_now().await;
                            }
                        };
                        tokio::select! {
                            r = run => r,
                            () = cancel => Ok(()),
                        }
                    }
                    _ => run.await,
                }
            }
        });
        let outcome =
            tokio::time::timeout(Duration::from_secs(3), futures::future::join_all(runs)).await;
        let Ok(results) = outcome else {
            panic!("seed {seed} hung: {case:?}");
        };
        for r in results {
            if let Err(e) = r {
                assert_ne!(
                    e.error_class(),
                    ErrorClass::InternalError,
                    "seed {seed}: {e} ({case:?})"
                );
            }
        }
        drop(sessions);
        assert_eq!(
            state.tickets.load(Ordering::SeqCst),
            0,
            "seed {seed}: tickets left"
        );
    }
    assert!(cases >= 20, "only {cases} cases ran");
}
