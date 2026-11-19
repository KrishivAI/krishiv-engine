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
/// (as a pool task holds its worker until its thread ends). A task without
/// a result takes two polls of `End`, waking itself after the first, so it
/// keeps its slot until its sender polls again.
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
    /// An aggregate lane's task: its result has one row per group it saw
    /// (`EndGroup` frames) instead of `result_rows`.
    per_group: bool,
    /// The sends of its tasks, held while the gate is closed.
    gate: Arc<Gate>,
    /// Its tasks fail at their first `EndGroup` (a UDF raising).
    fails_end_group: std::sync::atomic::AtomicBool,
    /// Its tasks fail once their results are read, instead of ending (a UDF
    /// raising after it yielded).
    fails_after_results: std::sync::atomic::AtomicBool,
    /// An aligned lane's task (a window aggregate, a scalar iterator): its
    /// result has one row per input row it saw.
    per_row: bool,
}

/// Holds every task's sends while closed, as a pool worker's thread that
/// has not taken a frame yet: the send is pending, and opening the gate
/// wakes it, so the frame is taken inside the next poll of its pump.
#[derive(Default)]
pub(super) struct Gate {
    closed: std::sync::atomic::AtomicBool,
    wakers: Mutex<Vec<Waker>>,
}

impl Gate {
    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }

    /// Opens the gate and wakes the held sends.
    fn open(&self) {
        self.closed.store(false, Ordering::SeqCst);
        let wakers = std::mem::take(&mut *self.wakers.lock().unwrap());
        wakers.into_iter().for_each(Waker::wake);
    }
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
            per_group: self.per_group,
            per_row: self.per_row,
            groups: Mutex::new(0),
            gate: Arc::clone(&self.gate),
            fails_end_group: self.fails_end_group.load(Ordering::SeqCst),
            fails_after_results: self.fails_after_results.load(Ordering::SeqCst),
            seen: Mutex::new(0),
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
    per_group: bool,
    per_row: bool,
    /// `EndGroup` frames taken.
    groups: Mutex<usize>,
    gate: Arc<Gate>,
    fails_end_group: bool,
    fails_after_results: bool,
    /// Rows of the `Batch` frames taken.
    seen: Mutex<usize>,
}

impl UdfTask for Task {
    fn send(&self, frame: Frame) -> BoxFuture<'_, crate::error::Result<()>> {
        let end = matches!(frame, Frame::End);
        let end_group = matches!(frame, Frame::EndGroup);
        let batch_rows = match &frame {
            Frame::Batch { batch, .. } => batch.num_rows(),
            _ => 0,
        };
        Box::pin(poll_fn(move |cx| {
            if self.gate.closed.load(Ordering::SeqCst) {
                self.gate.wakers.lock().unwrap().push(cx.waker().clone());
                return Poll::Pending;
            }
            if end_group && self.fails_end_group {
                return Poll::Ready(Err(crate::error::internal(
                    "the task fails on purpose".to_owned(),
                )));
            }
            if end_group {
                *self.groups.lock().unwrap() += 1;
            }
            *self.seen.lock().unwrap() += batch_rows;
            if end {
                let rows = if self.per_group {
                    *self.groups.lock().unwrap()
                } else if self.per_row {
                    *self.seen.lock().unwrap()
                } else {
                    self.result_rows
                };
                let mut polled = self.end_polled.lock().unwrap();
                // A task with a result ends in one poll, so `combine`'s own
                // poll (with the driver's waker) already meets the pool's
                // refusal.
                if !*polled && rows == 0 {
                    *polled = true;
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                *self.ended.lock().unwrap() = true;
                if rows > 0 {
                    let v: Int64Array = (0..rows).map(|i| Some(i as i64)).collect();
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
                if self.fails_after_results {
                    return Poll::Ready(Err(crate::error::internal(
                        "the task fails after its results on purpose".to_owned(),
                    )));
                }
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
    /// Every task's sends ([`Gate`]).
    gate: Arc<Gate>,
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
    rig_opts(lanes, limit, memory, result_rows, max_rows, Opts::default())
}

/// The rest of a rig's shape.
#[derive(Clone, Copy)]
struct Opts {
    /// Frames in flight per lane.
    window: usize,
    /// One lane of a grouped map (keyed by `k`) rather than a map.
    grouped: bool,
    /// The lanes' argument is a computed copy of `v` ([`Copied`]), so each
    /// frame reserves its copies.
    copied: bool,
    /// With `copied`: the argument fails the second time it is evaluated
    /// over this many rows (a frame evaluated again after the pool refused
    /// its copies).
    fails: Option<usize>,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            window: 4,
            grouped: false,
            copied: false,
            fails: None,
        }
    }
}

/// `v` copied into a buffer of its own, as a computed argument is: a frame
/// of it holds bytes beyond its rows', reserved as the frame is made.
#[derive(Debug)]
struct Copied {
    /// [`Opts::fails`].
    fails: Option<usize>,
    /// Evaluations over `fails` rows so far.
    seen: AtomicUsize,
}

impl ScalarExpr for Copied {
    fn data_type(&self) -> ArrowType {
        ArrowType::Int64
    }

    fn nullable(&self) -> bool {
        true
    }

    fn evaluate(&self, batch: &RecordBatch) -> crate::error::Result<ArrayRef> {
        if self.fails == Some(batch.num_rows()) && self.seen.fetch_add(1, Ordering::SeqCst) == 1 {
            return Err(crate::error::internal(
                "the argument fails on purpose".to_owned(),
            ));
        }
        let v = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        Ok(Arc::new(v.iter().collect::<Int64Array>()))
    }
}

/// As [`rig_full`], with `opts`.
fn rig_opts(
    lanes: usize,
    limit: usize,
    memory: usize,
    result_rows: usize,
    max_rows: usize,
    opts: Opts,
) -> Rig {
    let slots = Arc::new(Slots {
        owner: 0,
        limit,
        total: usize::MAX / 2,
        state: Arc::default(),
    });
    let ends = Arc::new(AtomicUsize::new(0));
    let gate = Arc::new(Gate::default());
    let udfs: Vec<Arc<Udf>> = (0..lanes)
        .map(|_| {
            Arc::new(Udf {
                slots: Arc::clone(&slots),
                opened: AtomicUsize::new(0),
                ends: Arc::clone(&ends),
                result_rows,
                twice: std::sync::atomic::AtomicBool::new(false),
                per_group: false,
                gate: Arc::clone(&gate),
                fails_end_group: std::sync::atomic::AtomicBool::new(false),
                fails_after_results: std::sync::atomic::AtomicBool::new(false),
                per_row: false,
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
            sides: vec![vec![if opts.copied {
                Arc::new(Copied {
                    fails: opts.fails,
                    seen: AtomicUsize::new(0),
                }) as Arc<dyn ScalarExpr>
            } else {
                column(1, ArrowType::Int64)
            }]],
            names: vec![vec!["v".to_owned()]],
            client: vec![ArrowType::Int64],
        })
        .collect();
    let (shape, output) = if opts.grouped {
        (
            RelationShape::Grouped { keys: vec![0] },
            vec![Field::new("v", ArrowType::Int64, true)],
        )
    } else if lanes == 1 {
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
            window: opts.window,
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
        gate,
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
/// lane again: the refusal happened in `combine`'s poll, after which no
/// driver of a's polls that lane, so without that poll the task never ends
/// and its slot never frees.
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

/// R7e-1: the lane the waiting driver polls again after reserving a combined
/// driver's refused result fails in that poll. The failure is the waiting
/// driver's at once: its `Blocked` future ends in the same poll (not one
/// self-woken poll later), and its next `sink` returns the error.
#[tokio::test]
async fn a_combined_lane_failing_in_the_poll_after_its_reservation_fails_the_waiting_driver() {
    let r = rig_with(1, 1, 4 << 20, 200_000);
    r.udfs[0].fails_after_results.store(true, Ordering::SeqCst);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let (ctx_a, _) = driver_ctx(&r, 1);
    let (ctx_b, owner_b) = driver_ctx(&r, 2);
    let (ctx_c, _) = driver_ctx(&r, 3);
    let run = async {
        let mut a = r.sink.local_state(&ctx_a, &g)?;
        let mut b = r.sink.local_state(&ctx_b, &g)?;
        feed(&r, rows(&[1, 2]), &mut a, &g).await?;
        let SinkStatus::Blocked(mut wait) = r.sink.sink(rows(&[3]), &mut b, &g)? else {
            panic!("the second driver must wait for the slot");
        };
        // As in R3-1: the pool has room for a's result only through b's credit.
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
        assert!(
            poll_fn(|cx| Poll::Ready(std::future::Future::poll(wait.as_mut(), cx)))
                .await
                .is_ready(),
            "the lane's failure ends the waiting driver's future in the same poll"
        );
        drop(blocker);
        Ok::<_, krishiv_common::error::KrishivError>(r.sink.sink(rows(&[4]), &mut b, &g).err())
    };
    let e = tokio::time::timeout(DEADLINE, run)
        .await
        .expect("the waiting driver hung")
        .unwrap()
        .expect("the combined lane's failure is the waiting driver's");
    assert!(
        e.to_string().contains("fails after its results on purpose"),
        "{e}"
    );
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
        let err = finish(&r, b, &g)
            .await
            .expect_err("b's results cannot fit beside a's");
        assert_eq!(
            err.error_class(),
            ErrorClass::UnableToAcquireMemory,
            "{err}"
        );
        Ok::<(), krishiv_common::error::KrishivError>(())
    };
    tokio::time::timeout(DEADLINE, run)
        .await
        .expect("the second refused result stalled the combined task")
        .unwrap();
    // a's task ended (its slot was what b's `finish` needed), and so did b's.
    assert_eq!(r.ends.load(Ordering::SeqCst), 2);
}

/// A combined driver whose `End` stayed in its backlog (its window was full
/// when it combined: the open group's held rows, then `EndGroup`) holds the
/// one slot until that `End` goes out. Its driver no longer frames it, and
/// `finalize`, which would, waits for the driver blocked on that slot: the
/// waiting driver frames the combined drivers' backlogs (`blocked`). Seed 19
/// of PR-26c-3's sweep with #820's grouped shape.
#[tokio::test(start_paused = true)]
async fn a_combined_driver_s_end_left_by_a_full_window_is_framed_by_the_driver_waiting_for_its_slot()
 {
    end_left_by_a_full_window(2).await;
}

/// As above with a window of one (review R7-1): each frame the waiting
/// driver queues is taken inside its poll of the combined lane, which then
/// waits for the next frame with no event to come (no result follows a
/// frame mid-group). One framing pass left `EndGroup` and `End` behind and
/// the waiting driver parked for good; it frames again until nothing moves.
#[tokio::test(start_paused = true)]
async fn a_combined_backlog_whose_window_empties_inside_the_poll_is_framed_to_its_end() {
    end_left_by_a_full_window(1).await;
}

/// Two drivers over one slot, a grouped map of frames of two rows in a
/// `window`: a combines with its held row, `EndGroup` and `End` (some of
/// them past its window), and b waits for a's slot.
async fn end_left_by_a_full_window(window: usize) {
    let r = rig_opts(
        1,
        1,
        64 << 20,
        0,
        2,
        Opts {
            window,
            grouped: true,
            ..Opts::default()
        },
    );
    let g = r.sink.global_state(&r.ctx).unwrap();
    let run = async {
        let mut a = r.sink.local_state(&r.ctx, &g)?;
        let mut b = r.sink.local_state(&r.ctx, &g)?;
        // a takes the slot: a frame of two rows now, one row held.
        feed(&r, rows(&[1, 1, 1]), &mut a, &g).await?;
        assert_eq!(r.slots.held(), 1);
        // The held row and `EndGroup` fill a's window of two (with one, the
        // held row does); `End` waits.
        r.sink.combine(a, &g)?;
        assert_eq!(r.ends.load(Ordering::SeqCst), 0, "a's task cannot end yet");
        // b's whole frame waits for the slot a's task holds.
        let SinkStatus::Blocked(wait) = r.sink.sink(rows(&[5, 5]), &mut b, &g)? else {
            panic!("the second driver must wait for the slot");
        };
        wait.await;
        finish(&r, b, &g).await
    };
    tokio::time::timeout(DEADLINE, run)
        .await
        .expect("the waiting driver hung on the combined driver's backlog")
        .unwrap();
    assert_eq!(r.udfs[0].opened.load(Ordering::SeqCst), 2);
    assert_eq!(r.ends.load(Ordering::SeqCst), 2);
    assert_eq!(r.slots.state.most.load(Ordering::SeqCst), 1);
}

/// The memory-pressure variant: the combined driver's backlog (the open
/// group's held rows, `EndGroup`, `End`) waits for its frame's copies, which
/// the pool refuses (and which the waiting driver's own reserved copies, a
/// one-row frame's, do not cover).
/// The driver waiting for the slot cannot frame it, and nothing it waits on
/// frees memory: its `Blocked` future ends and its next `sink` asks for the
/// copies (`NeedMemory`), so the arbitrator gives them (a spill) or fails
/// the query, never a hang. With the grant the retry frames the backlog and
/// the combined task ends.
#[tokio::test(start_paused = true)]
async fn a_combined_driver_s_backlog_refused_its_copies_asks_for_the_memory() {
    backlog_refused_its_copies(4).await;
}

/// As above with a window of two (review R7-4).
#[tokio::test(start_paused = true)]
async fn a_combined_backlog_refused_its_copies_in_a_window_of_two_asks_for_the_memory() {
    backlog_refused_its_copies(2).await;
}

/// A grouped map with a computed argument over 4 MiB and one slot: a holds
/// the slot with 5,000 rows held, b waits for it, the pool is filled.
fn refused_copies_rig(window: usize, fails: Option<usize>) -> Rig {
    rig_opts(
        1,
        1,
        4 << 20,
        0,
        10_000,
        Opts {
            window,
            grouped: true,
            copied: true,
            fails,
        },
    )
}

/// The body of the tests above, with lanes of `window` frames.
async fn backlog_refused_its_copies(window: usize) {
    let r = refused_copies_rig(window, None);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let (ctx_a, _) = driver_ctx(&r, 1);
    let (ctx_b, owner_b) = driver_ctx(&r, 2);
    let (ctx_c, _) = driver_ctx(&r, 3);
    let run = async {
        let mut a = r.sink.local_state(&ctx_a, &g)?;
        let mut b = r.sink.local_state(&ctx_b, &g)?;
        // a takes the slot: one group, a frame of 10,000 rows now, 5,000
        // held.
        let keys: Vec<i32> = std::iter::repeat_n(0, 15_000).collect();
        feed(&r, rows(&keys), &mut a, &g).await?;
        assert_eq!(r.slots.held(), 1);
        // b's first group (one row, its copies reserved) waits for the slot.
        let SinkStatus::Blocked(wait) = r.sink.sink(rows(&[1, 2]), &mut b, &g)? else {
            panic!("the second driver must wait for the slot");
        };
        let waiting = tokio::spawn(wait);
        tokio::task::yield_now().await;
        // Whatever the pool has left goes to another holder.
        let mut hog = ctx_c.consumer().consumer().reservation();
        let mut step = 1 << 20;
        while step >= 64 {
            if hog.try_grow(step).is_err() {
                step /= 2;
            }
        }
        // a's held rows are refused their copies: they, `EndGroup` and
        // `End` wait.
        r.sink.combine(a, &g)?;
        waiting.await.unwrap();
        assert_eq!(r.ends.load(Ordering::SeqCst), 0, "a's task cannot end yet");
        let empty = || Batch::new(RecordBatch::new_empty(input()));
        let status = r.sink.sink(empty(), &mut b, &g)?;
        let SinkStatus::NeedMemory { bytes } = status else {
            panic!("the driver must ask for the combined frame's copies: {status:?}");
        };
        assert!(bytes >= 40_000, "the held rows' copies: {bytes}");
        // The driver's loop: memory freed (a spill), then granted to b.
        drop(hog);
        let b_consumer = Arc::clone(ctx_b.consumer().consumer());
        owner_b
            .reserve_for(&b_consumer, bytes, || async {})
            .await
            .unwrap();
        match r.sink.sink(empty(), &mut b, &g)? {
            SinkStatus::NeedMemory { .. } => panic!("the retry must use the grant"),
            SinkStatus::Blocked(wait) => wait.await,
            SinkStatus::NeedMoreInput | SinkStatus::Finished => {}
        }
        // a's task ended: b holds the slot.
        assert_eq!(r.ends.load(Ordering::SeqCst), 1);
        finish(&r, b, &g).await
    };
    tokio::time::timeout(DEADLINE, run)
        .await
        .expect("the waiting driver hung on the combined driver's refused frame")
        .unwrap();
    assert_eq!(r.ends.load(Ordering::SeqCst), 2);
    assert_eq!(r.slots.state.most.load(Ordering::SeqCst), 1);
}

/// A combined driver's backlog frame that fails when the waiting driver
/// frames it (its argument fails the second time over its rows; the first,
/// in `combine`, was refused its copies) is the waiting driver's error
/// (review R7-2): left in the combined driver's state, its task would wait
/// for the frame holding the slot, and `finalize` could not start.
#[tokio::test(start_paused = true)]
async fn a_combined_backlog_s_framing_failure_fails_the_waiting_driver() {
    let r = refused_copies_rig(4, Some(5_000));
    let g = r.sink.global_state(&r.ctx).unwrap();
    let (ctx_a, _) = driver_ctx(&r, 1);
    let (ctx_b, _) = driver_ctx(&r, 2);
    let (ctx_c, _) = driver_ctx(&r, 3);
    let run = async {
        let mut a = r.sink.local_state(&ctx_a, &g)?;
        let mut b = r.sink.local_state(&ctx_b, &g)?;
        let keys: Vec<i32> = std::iter::repeat_n(0, 15_000).collect();
        feed(&r, rows(&keys), &mut a, &g).await?;
        let SinkStatus::Blocked(wait) = r.sink.sink(rows(&[1, 2]), &mut b, &g)? else {
            panic!("the second driver must wait for the slot");
        };
        let waiting = tokio::spawn(wait);
        tokio::task::yield_now().await;
        let mut hog = ctx_c.consumer().consumer().reservation();
        let mut step = 1 << 20;
        while step >= 64 {
            if hog.try_grow(step).is_err() {
                step /= 2;
            }
        }
        // a's held rows are evaluated (once) and refused their copies.
        r.sink.combine(a, &g)?;
        drop(hog);
        // b frames them again: the evaluation fails.
        waiting.await.unwrap();
        let empty = Batch::new(RecordBatch::new_empty(input()));
        Ok::<_, krishiv_common::error::KrishivError>(r.sink.sink(empty, &mut b, &g).map(drop))
    };
    let err = tokio::time::timeout(DEADLINE, run)
        .await
        .expect("the waiting driver hung on the combined backlog's failure")
        .unwrap()
        .expect_err("the waiting driver must fail");
    assert!(err.to_string().contains("fails on purpose"), "{err}");
}

/// R7b-1: a pass of the waiting driver's that starts with a combined window
/// full (the send pending, as a pool thread that has not taken the frame)
/// frames nothing, and the poll that follows takes the frame at once and
/// empties the window. Framing before the poll then parked the waiting
/// driver with room free and `EndGroup`, `End` unframed, and nothing to
/// come; the poll comes first in every pass.
#[tokio::test(start_paused = true)]
async fn a_combined_window_emptied_by_a_poll_after_a_full_start_is_framed_on() {
    let r = rig_opts(
        1,
        1,
        64 << 20,
        0,
        2,
        Opts {
            window: 1,
            grouped: true,
            ..Opts::default()
        },
    );
    let g = r.sink.global_state(&r.ctx).unwrap();
    let run = async {
        let mut a = r.sink.local_state(&r.ctx, &g)?;
        let mut b = r.sink.local_state(&r.ctx, &g)?;
        // a takes the slot: a frame of two rows sent, one row held.
        feed(&r, rows(&[1, 1, 1]), &mut a, &g).await?;
        // The task takes no frame now: a's held row stays in flight (its
        // window of one full), `EndGroup` and `End` wait.
        r.gate.close();
        r.sink.combine(a, &g)?;
        let SinkStatus::Blocked(wait) = r.sink.sink(rows(&[5, 5]), &mut b, &g)? else {
            panic!("the second driver must wait for the slot");
        };
        let waiting = tokio::spawn(wait);
        tokio::task::yield_now().await;
        // The task takes the frame: the waiting driver's next pass starts
        // with the window full and empties it in its poll.
        r.gate.open();
        waiting.await.unwrap();
        finish(&r, b, &g).await
    };
    tokio::time::timeout(DEADLINE, run)
        .await
        .expect("the waiting driver parked with room free in the combined window")
        .unwrap();
    assert_eq!(r.ends.load(Ordering::SeqCst), 2);
    assert_eq!(r.slots.state.most.load(Ordering::SeqCst), 1);
}

/// R7b-2: a combined aggregate driver whose one task failed (at its
/// `EndGroup`, a UDF raising) while its `End` waits behind a full window:
/// its other task holds its slot waiting for that `End`. The failure is the
/// waiting driver's error; skipping the driver's backlog would leave the
/// other slot held and the waiting driver parked for good.
#[tokio::test(start_paused = true)]
async fn a_combined_lane_s_failure_fails_the_driver_waiting_for_its_slots() {
    let r = rig_opts(
        2,
        2,
        64 << 20,
        0,
        0,
        Opts {
            window: 1,
            ..Opts::default()
        },
    );
    r.udfs[0].fails_end_group.store(true, Ordering::SeqCst);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let run = async {
        let mut a = r.sink.local_state(&r.ctx, &g)?;
        let mut b = r.sink.local_state(&r.ctx, &g)?;
        // a takes both slots: one group's rows sent to both lanes.
        feed(&r, rows(&[1, 1]), &mut a, &g).await?;
        assert_eq!(r.slots.held(), 2);
        // The tasks take no frame now: `EndGroup` fills each window of
        // one, `End` waits.
        r.gate.close();
        r.sink.combine(a, &g)?;
        let SinkStatus::Blocked(wait) = r.sink.sink(rows(&[5, 5]), &mut b, &g)? else {
            panic!("the second driver must wait for the slots");
        };
        let waiting = tokio::spawn(wait);
        tokio::task::yield_now().await;
        // Lane 0's task fails at `EndGroup`; lane 1's waits for `End`.
        r.gate.open();
        waiting.await.unwrap();
        let empty = Batch::new(RecordBatch::new_empty(input()));
        Ok::<_, krishiv_common::error::KrishivError>(r.sink.sink(empty, &mut b, &g).map(drop))
    };
    let err = tokio::time::timeout(DEADLINE, run)
        .await
        .expect("the waiting driver hung on a combined driver's failed lane")
        .unwrap()
        .expect_err("the waiting driver must fail");
    assert!(err.to_string().contains("fails on purpose"), "{err}");
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
    /// The session cancelled, and after how many of its run's yields.
    cancel: Option<(usize, usize)>,
    /// An aggregate (one row per group from each lane) rather than a map.
    aggregate: bool,
    /// An aligned shape (one value per row from each lane): a window
    /// aggregate or a scalar iterator.
    aligned: bool,
    /// The cancel goes through the query (its drivers see it) rather than
    /// by dropping the session's drivers.
    by_query: bool,
    /// Keys whose groups may span two batches ([`keys`]); round 4's cases
    /// keep its keys.
    spanning: bool,
    /// A map shaped as a UDTF's (PR-26c-4): a `TABLE` row struct computed per frame, then a carried column.
    udtf: bool,
}

/// A session's relation slots: its `maxPerSession`, within `maxTotal`, less
/// the one kept for its scalar calls (as `CallThreads::slot_limit`).
fn slot_limit(per_session: usize, total: usize) -> usize {
    per_session.min(total).saturating_sub(1).max(1)
}

impl Case {
    /// The case of `seed`. The draws of round 4's sweep come first and in
    /// its order, so `original` gives the very cases it drew (seed 207 is
    /// the case that hung under round 4's harness); the rest (shape, lanes within the limit, the way of
    /// cancelling) are drawn after them.
    fn new(seed: u64, original: bool) -> Self {
        let mut rng = Rng::new(seed);
        let drivers = rng.pick(1, 6);
        let sessions = rng.pick(1, 2);
        let mut case = Case {
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
            aggregate: false,
            aligned: false,
            by_query: false,
            spanning: false,
            udtf: false,
        };
        if original {
            return case;
        }
        case.aggregate = rng.below(2) == 0;
        // Most cases fit the slots; the rest fail at once with
        // `KRISHIV_PYTHON_WORKER_LIMIT`.
        let limit = slot_limit(case.per_session, case.total);
        case.lanes = match (case.aggregate, rng.below(5)) {
            (false, _) => 1,
            (true, 0) => rng.pick(1, 3),
            (true, _) => rng.pick(1, limit),
        };
        case.by_query = rng.below(2) == 0;
        case.spanning = true;
        // A third of the maps become aligned (PR-26c-3), their lanes within
        // the limit as an aggregate's mostly are.
        if !case.aggregate && rng.below(3) == 0 {
            case.aligned = true;
            case.lanes = match rng.below(5) {
                0 => rng.pick(1, 3),
                _ => rng.pick(1, limit),
            };
        }
        case.udtf = !case.aggregate && !case.aligned && rng.below(2) == 0;
        case
    }

    /// The rows a session's run stores when every task runs to its end
    /// (`None` for a map of several lanes, which no plan makes).
    fn rows(&self, owner: usize) -> Option<usize> {
        if !self.aggregate && !self.aligned && self.lanes > 1 {
            return None;
        }
        let rows = (0..self.drivers)
            .map(|d| {
                let batches = &self.batches[owner * self.drivers + d];
                if self.aligned {
                    // Every input row, once.
                    batches.iter().sum::<usize>()
                } else if self.aggregate {
                    // One row per group: the distinct keys of the driver.
                    let mut keys: Vec<i32> = batches
                        .iter()
                        .enumerate()
                        .flat_map(|(i, &n)| keys(i, n, self.spanning))
                        .collect();
                    keys.dedup();
                    keys.len()
                } else if batches.iter().sum::<usize>() == 0 {
                    // An empty partition opens no task.
                    0
                } else {
                    self.result_rows * if owner % 2 == 1 { 2 } else { 1 }
                }
            })
            .sum();
        Some(rows)
    }
}

/// The keys of batch `i` of a driver's input, `n` rows (at most 5): two rows
/// per key, ascending across its batches. When `spanning`, batch `i` starts
/// at key `5i / 2`, so after a full batch of an even `i` the last key's
/// group goes on in the next batch (review N6-3: the path where a group
/// spans two batches); otherwise (round 4's keys) batch `i` starts at key
/// `5i` and no group spans two.
fn keys(i: usize, n: usize, spanning: bool) -> Vec<i32> {
    let first = if spanning { i * 5 } else { i * 10 };
    (0..n)
        .map(|k| i32::try_from((first + k) / 2).unwrap())
        .collect()
}

/// One session of a case: its sink (a map, or an aggregate of `lanes`
/// lanes, over the session's slots) and query.
struct Session {
    sink: Arc<RelationSink>,
    query: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
}

fn session(case: &Case, owner: usize, state: &Arc<SlotState>, root: &Arc<TreePool>) -> Session {
    let slots = Arc::new(Slots {
        owner,
        limit: slot_limit(case.per_session, case.total),
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
                twice: std::sync::atomic::AtomicBool::new(
                    !case.aggregate && !case.aligned && owner % 2 == 1,
                ),
                per_group: case.aggregate,
                gate: Arc::default(),
                fails_end_group: std::sync::atomic::AtomicBool::new(false),
                fails_after_results: std::sync::atomic::AtomicBool::new(false),
                per_row: case.aligned,
            }) as Arc<dyn PreparedRelationUdf>,
            sides: vec![if case.udtf {
                vec![Arc::new(RowOf) as Arc<dyn ScalarExpr>, column(1)]
            } else {
                vec![column(1)]
            }],
            names: vec![if case.udtf {
                vec!["_0".to_owned(), "_1".to_owned()]
            } else {
                vec!["v".to_owned()]
            }],
            client: vec![ArrowType::Int64],
        })
        .collect();
    let (shape, output) = if case.aligned {
        let mut fields = vec![
            Field::new("k", ArrowType::Int32, true),
            Field::new("v", ArrowType::Int64, true),
        ];
        fields.extend((0..case.lanes).map(|i| Field::new(format!("n{i}"), ArrowType::Int64, true)));
        // A window aggregate's partitions for an even owner, a scalar
        // iterator's one stream for an odd one.
        let keys = owner.is_multiple_of(2).then(|| vec![0]);
        (RelationShape::Aligned { keys }, fields)
    } else if case.aggregate {
        let mut fields = vec![Field::new("k", ArrowType::Int32, true)];
        fields.extend((0..case.lanes).map(|i| Field::new(format!("n{i}"), ArrowType::Int64, true)));
        (
            RelationShape::Aggregate {
                keys: vec![0],
                key_values: vec![0],
            },
            fields,
        )
    } else {
        (
            RelationShape::Map,
            vec![Field::new("v", ArrowType::Int64, true)],
        )
    };
    let sink = RelationSink::try_new(
        input(),
        shape,
        specs,
        Arc::new(Schema::new(output)),
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

/// One driver as the pipeline driver runs it, as a task of its own (so its
/// own waker must fire for it to move): each batch until taken (`Blocked`
/// awaited, `NeedMemory` granted the cooperative way, a cancelled query
/// checked first), then `combine`.
async fn drive_one(
    s: Arc<Session>,
    g: GlobalState,
    op: u32,
    batches: Vec<usize>,
    spanning: bool,
) -> crate::error::Result<()> {
    let owner = SpillOwner::new();
    let ctx = TaskContext::new(
        Arc::clone(&s.query),
        Arc::clone(&s.executor),
        Arc::clone(&owner),
        OperatorId(op),
        "python_relation",
    );
    let mut l = s.sink.local_state(&ctx, &g)?;
    for (i, n) in batches.into_iter().enumerate() {
        loop {
            s.query.check_cancelled()?;
            match s.sink.sink(rows(&keys(i, n, spanning)), &mut l, &g)? {
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
    s.query.check_cancelled()?;
    s.sink.combine(l, &g)
}

/// Every driver of a session, each a task of its own, then its `finalize`:
/// the rows stored, or `None` when the case cancelled it.
async fn run_session(
    s: Arc<Session>,
    case: Arc<Case>,
    owner: usize,
) -> crate::error::Result<Option<usize>> {
    let ctx = TaskContext::new(
        Arc::clone(&s.query),
        Arc::clone(&s.executor),
        SpillOwner::new(),
        OperatorId(100),
        "python_relation",
    );
    let g = s.sink.global_state(&ctx)?;
    let mut drivers = tokio::task::JoinSet::new();
    for d in 0..case.drivers {
        let batches = case.batches[owner * case.drivers + d].clone();
        drivers.spawn(drive_one(
            Arc::clone(&s),
            Arc::clone(&g),
            u32::try_from(d + 1).unwrap(),
            batches,
            case.spanning,
        ));
    }
    let mut cancel = case.cancel.filter(|&(victim, _)| victim == owner);
    let mut ticks = 0;
    let mut failed = None;
    loop {
        let next = tokio::select! {
            biased;
            next = drivers.join_next() => next,
            () = tokio::task::yield_now(), if cancel.is_some() => {
                ticks += 1;
                if cancel.is_some_and(|(_, after)| ticks > after) {
                    cancel = None;
                    if !case.by_query {
                        // The session's run is dropped: its drivers with it.
                        drivers.abort_all();
                        while drivers.join_next().await.is_some() {}
                        return Ok(None);
                    }
                    s.query.cancel();
                }
                continue;
            }
        };
        match next {
            None => break,
            Some(Ok(Ok(()))) => {}
            Some(Ok(Err(e))) => {
                // The first error fails the query; the other drivers stop.
                if failed.is_none() {
                    failed = Some(e);
                    drivers.abort_all();
                }
            }
            Some(Err(e)) if e.is_cancelled() => {}
            Some(Err(e)) => panic!("a driver panicked: {e}"),
        }
    }
    if let Some(e) = failed {
        return Err(e);
    }
    s.query.check_cancelled()?;
    // A cancel not reached yet ticks on through `finalize`: the query's
    // cancel then ends `finalize`'s wait (its `cancelled()` branch), or the
    // session's run is dropped mid-`finalize` (review N6-3).
    let finalize = s.sink.finalize(&g, &ctx);
    tokio::pin!(finalize);
    loop {
        tokio::select! {
            biased;
            r = &mut finalize => {
                r?;
                break;
            }
            () = tokio::task::yield_now(), if cancel.is_some() => {
                ticks += 1;
                if cancel.is_some_and(|(_, after)| ticks > after) {
                    cancel = None;
                    if !case.by_query {
                        return Ok(None);
                    }
                    s.query.cancel();
                }
            }
        }
    }
    let stored = super::sink::store_state(&g).and_then(|store| crate::shared::memory_rows(&store));
    Ok(Some(stored.unwrap_or(usize::MAX)))
}

/// What a case did.
#[derive(Debug, Default)]
struct Outcome {
    /// Sessions that finished with every result stored.
    completed: usize,
    /// Sessions that ended with a classed error or were cancelled.
    stopped: usize,
}

/// Runs the case of `seed`: every session must end within the deadline (in
/// paused time: a run waiting on nothing but a wakeup fails at once), with
/// its results (each one once) or a classed error, and leave no slot held,
/// no ticket queued and no memory reserved.
async fn run_case(seed: u64, case: Case) -> Outcome {
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
    let case = Arc::new(case);
    let sessions: Vec<Arc<Session>> = (0..case.sessions)
        .map(|o| Arc::new(session(&case, o, &state, &root)))
        .collect();
    let mut runs = tokio::task::JoinSet::new();
    for (o, s) in sessions.iter().enumerate() {
        let (s, c) = (Arc::clone(s), Arc::clone(&case));
        runs.spawn(async move { (o, run_session(s, c, o).await) });
    }
    let all = async {
        let mut done = Vec::new();
        while let Some(r) = runs.join_next().await {
            done.push(r.unwrap_or_else(|e| panic!("seed {seed}: a session panicked: {e}")));
        }
        done
    };
    let Ok(results) = tokio::time::timeout(Duration::from_secs(3), all).await else {
        panic!("seed {seed} hung: {case:?}");
    };
    let mut outcome = Outcome::default();
    for (o, r) in results {
        match r {
            Ok(Some(rows)) => {
                if let Some(expected) = case.rows(o) {
                    assert_eq!(
                        rows, expected,
                        "seed {seed}: rows of session {o} ({case:?})"
                    );
                }
                outcome.completed += 1;
            }
            Ok(None) => outcome.stopped += 1,
            Err(e) => {
                assert_ne!(
                    e.error_class(),
                    ErrorClass::InternalError,
                    "seed {seed}: {e} ({case:?})"
                );
                outcome.stopped += 1;
            }
        }
    }
    drop(sessions);
    assert_eq!(
        state.tickets.load(Ordering::SeqCst),
        0,
        "seed {seed}: tickets left"
    );
    assert_eq!(
        (0..case.sessions).map(|o| state.held(o)).sum::<usize>(),
        0,
        "seed {seed}: slots left held"
    );
    assert_eq!(root.reserved(), 0, "seed {seed}: memory left reserved");
    outcome
}

/// A seeded sweep over drivers 1-6, maps and aggregates of 1-3 lanes,
/// `maxPerSession` and `maxTotal` 1-3, one or two sessions sharing the
/// slots, frames of one or two rows, small memory limits, results the pool
/// may refuse and cancellation (a session's drivers dropped, or its query
/// cancelled). A fixed list of seeds, each driver a task of its own, in
/// paused time, so a case runs the same way every time and a hang fails it
/// by its seed (review round 4: waits nobody wakes; round 5: R5-1).
#[tokio::test(start_paused = true)]
async fn seeded_relation_slot_stress_never_hangs() {
    let mut total = Outcome::default();
    for seed in 0..=255 {
        let o = run_case(seed, Case::new(seed, false)).await;
        total.completed += o.completed;
        total.stopped += o.stopped;
    }
    // Most sessions run to their end: the sweep is not all early failures.
    assert!(
        total.completed >= 150,
        "only {} sessions completed ({} stopped)",
        total.completed,
        total.stopped
    );
}

/// Round 4's sweep's cases as it drew them, seed 207 among them (the case
/// that hung under round 4's harness, where a combined driver's ticket held
/// its sibling back; the deterministic test of that bug is
/// `a_driver_combining_while_slots_are_full_keeps_no_place_in_the_queue`).
#[tokio::test(start_paused = true)]
async fn the_round_four_cases_never_hang() {
    for seed in 0..=255 {
        run_case(seed, Case::new(seed, true)).await;
    }
}

/// Seed 207 of round 4's sweep on its own.
#[tokio::test(start_paused = true)]
async fn round_four_seed_207_never_hangs() {
    run_case(207, Case::new(207, true)).await;
}

/// A combined driver whose `End` stayed in its backlog (its window was full
/// when it combined: the open group's last cut, `EndGroup`, `End`) holds the
/// one slot until that frame goes out; the driver waiting for the slot frames
/// it (`blocked`, the combined feeds), else `finalize`, which waits for the
/// waiting driver, never comes. Seed 19 of the sweep, reduced; it hung before
/// the fix.
#[tokio::test(start_paused = true)]
async fn a_combined_driver_s_backlog_is_framed_by_the_driver_waiting_for_its_slot() {
    let mut c = Case::new(19, false);
    c.sessions = 1;
    c.batches = vec![vec![3], vec![2]];
    c.drivers = 2;
    assert!(c.aligned && c.total == 1, "{c:?}");
    let o = run_case(19, c).await;
    assert_eq!(o.completed, 1);
}

/// Input column `index` (`v`: 1).
fn column(index: usize) -> Arc<dyn ScalarExpr> {
    Arc::new(InputColumn {
        index,
        data_type: ArrowType::Int64,
        nullable: true,
    })
}

/// A UDTF's `TABLE` row: the struct of `k` and `v`, computed per frame (`StructOf` in instantiation).
#[derive(Debug)]
struct RowOf;

impl ScalarExpr for RowOf {
    fn data_type(&self) -> ArrowType {
        ArrowType::Struct(input().fields().clone())
    }

    fn nullable(&self) -> bool {
        false
    }

    fn evaluate(&self, batch: &RecordBatch) -> crate::error::Result<ArrayRef> {
        Ok(Arc::new(arrow_array::StructArray::from(batch.clone())))
    }
}
