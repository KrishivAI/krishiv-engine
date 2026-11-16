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
    RelationSlots, UdfTask,
};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus};
use crate::operators::ScalarExpr;
use crate::{
    ExecConfig, Executor, OperatorId, QueryContext, QueryId, TaskContext, TokioClock,
    TokioExecutor, WorkloadClass,
};

const DEADLINE: Duration = Duration::from_secs(10);

/// The slots of a fake bounded runtime, as the worker pool's.
struct Slots {
    limit: usize,
    state: Arc<SlotState>,
}

#[derive(Default)]
struct SlotState {
    held: Mutex<usize>,
    most: AtomicUsize,
    waiters: Mutex<Vec<Waker>>,
}

/// One slot; frees and wakes the waiters when it drops.
struct Held(Arc<SlotState>);

impl Drop for Held {
    fn drop(&mut self) {
        *self.0.held.lock().unwrap() -= 1;
        let waiters = std::mem::take(&mut *self.0.waiters.lock().unwrap());
        for w in waiters {
            w.wake();
        }
    }
}

impl RelationSlots for Slots {
    fn limit(&self) -> usize {
        self.limit
    }

    fn poll_take(
        &self,
        n: usize,
        cx: &mut Context<'_>,
    ) -> Poll<crate::error::Result<Vec<RelationSlot>>> {
        if n > self.limit {
            return Poll::Ready(Err(KrishivPythonWorkerLimit {
                operation: "a fake operator".to_owned(),
                needed: n.to_string(),
                available: self.limit.to_string(),
                max_per_session: (self.limit + 1).to_string(),
            }
            .into()));
        }
        let mut held = self.state.held.lock().unwrap();
        if *held + n > self.limit {
            self.state.waiters.lock().unwrap().push(cx.waker().clone());
            return Poll::Pending;
        }
        *held += n;
        self.state.most.fetch_max(*held, Ordering::SeqCst);
        Poll::Ready(Ok((0..n)
            .map(|_| RelationSlot::new(Held(Arc::clone(&self.state))))
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
}

impl UdfTask for Task {
    fn send(&self, frame: Frame) -> BoxFuture<'_, crate::error::Result<()>> {
        let end = matches!(frame, Frame::End);
        Box::pin(poll_fn(move |cx| {
            if end {
                let mut polled = self.end_polled.lock().unwrap();
                if !*polled {
                    *polled = true;
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                *self.ended.lock().unwrap() = true;
                drop(self.slot.lock().unwrap().take());
                self.ends.fetch_add(1, Ordering::SeqCst);
                self.reader.wake();
            }
            Poll::Ready(Ok(()))
        }))
    }

    fn next(&self) -> BoxFuture<'_, crate::error::Result<Option<RecordBatch>>> {
        Box::pin(poll_fn(|cx| {
            self.reader.register(cx.waker());
            if *self.ended.lock().unwrap() {
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
    let slots = Arc::new(Slots {
        limit,
        state: Arc::default(),
    });
    let ends = Arc::new(AtomicUsize::new(0));
    let udfs: Vec<Arc<Udf>> = (0..lanes)
        .map(|_| {
            Arc::new(Udf {
                slots: Arc::clone(&slots),
                opened: AtomicUsize::new(0),
                ends: Arc::clone(&ends),
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
            max_rows: 0,
            max_bytes: 0,
            window: 4,
        },
        None,
    )
    .unwrap();
    let root = TreePool::with_clock(
        &MemoryConfig {
            limit: Some(64 << 20),
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
        root.child("q", Some(64 << 20)),
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
    assert_eq!(*r.slots.state.held.lock().unwrap(), 0, "no slot left held");
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
        assert_eq!(*r.slots.state.held.lock().unwrap(), 1);
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
        assert_eq!(*r.slots.state.held.lock().unwrap(), 2);
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
    // missing ones, which is fine; the point is that nothing hangs.
    let _ = tokio::time::timeout(DEADLINE, run)
        .await
        .expect("a driver hung waiting for slots");
    assert_eq!(r.slots.state.most.load(Ordering::SeqCst), 2);
    assert_eq!(
        r.udfs
            .iter()
            .map(|u| u.opened.load(Ordering::SeqCst))
            .sum::<usize>(),
        6
    );
    assert_eq!(*r.slots.state.held.lock().unwrap(), 0);
}
