// SPDX-License-Identifier: Apache-2.0
//! The relation sink driven by hand over a fake runtime whose tasks log the
//! frames they see (design P3-T085 §4): batching, group cuts, cogroup sides,
//! the bounded window, failures, cancellation and memory.
#![expect(
    clippy::disallowed_methods,
    reason = "test-only runtime plumbing; never runs on a compute worker"
)]

use std::collections::VecDeque;
use std::future::poll_fn;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::Poll;
use std::time::Duration;

use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::future::BoxFuture;
use futures::task::AtomicWaker;
use krishiv_common::error::{ErrorClass, InternalError};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillOwner, TreePool};
use krishiv_types::batch::Batch;

use super::sink::{InputColumn, RelationLimits};
use super::{Frame, PreparedRelationUdf, RelationLane, RelationShape, RelationSink, UdfTask};
use crate::operator::{Sink, SinkStatus};
use crate::operators::ScalarExpr;
use crate::{
    ExecConfig, Executor, OperatorId, QueryContext, QueryId, TaskContext, TokioClock,
    TokioExecutor, WorkloadClass,
};

/// What the fake's tasks share with the test.
#[derive(Default)]
struct Shared {
    /// Every frame every task saw: `B<side>:<rows>`, `G`, `E`.
    log: std::sync::Mutex<Vec<String>>,
    /// Tasks opened.
    opened: std::sync::atomic::AtomicUsize,
    /// While closed, `send` waits.
    closed: AtomicBool,
    /// While set, `send` waits once the log has this many frames.
    admit: std::sync::Mutex<Option<usize>>,
    gate: AtomicWaker,
    /// Fail the n-th frame (counted from 1).
    fail_at: std::sync::Mutex<Option<usize>>,
    /// Tasks dropped (cancelled or finished).
    dropped: std::sync::atomic::AtomicUsize,
    /// Result batches the operator read from the tasks.
    taken: std::sync::atomic::AtomicUsize,
    /// Polls of `next`: past [`SPUN`] the task fails, so a busy loop ends.
    polls: std::sync::atomic::AtomicUsize,
}

/// Polls of a task's `next` that only a busy loop reaches.
const SPUN: usize = 100_000;

impl Shared {
    fn open_gate(&self) {
        self.closed.store(false, Ordering::SeqCst);
        self.gate.wake();
    }

    fn log(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }
}

/// What a fake task yields.
#[derive(Clone, Copy)]
enum Yield {
    /// Each batch back, twice (output not aligned to input).
    Twice,
    /// Nothing at all.
    Nothing,
    /// One row per group: its row count (`BIGINT`).
    Count,
    /// Two rows per group (an aggregate UDF yielding too many values).
    CountTwice,
    /// Each batch back this many times.
    Burst(usize),
}

struct Fake {
    shared: Arc<Shared>,
    yields: Yield,
}

impl PreparedRelationUdf for Fake {
    fn open(&self) -> crate::error::Result<Box<dyn UdfTask>> {
        self.shared.opened.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Task {
            shared: Arc::clone(&self.shared),
            yields: self.yields,
            state: std::sync::Mutex::new(TaskState::default()),
            ready: AtomicWaker::new(),
        }))
    }
}

#[derive(Default)]
struct TaskState {
    out: VecDeque<RecordBatch>,
    rows: i64,
    ended: bool,
}

struct Task {
    shared: Arc<Shared>,
    yields: Yield,
    state: std::sync::Mutex<TaskState>,
    ready: AtomicWaker,
}

impl Drop for Task {
    fn drop(&mut self) {
        self.shared.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

fn count(n: i64) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("n", ArrowType::Int64, true)]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(vec![n])) as ArrayRef],
    )
    .unwrap()
}

impl UdfTask for Task {
    fn send(&self, frame: Frame) -> BoxFuture<'_, crate::error::Result<()>> {
        Box::pin(async move {
            poll_fn(|cx| {
                self.shared.gate.register(cx.waker());
                let full = (*self.shared.admit.lock().unwrap())
                    .is_some_and(|n| self.shared.log.lock().unwrap().len() >= n);
                if full || self.shared.closed.load(Ordering::SeqCst) {
                    Poll::Pending
                } else {
                    Poll::Ready(())
                }
            })
            .await;
            let n = {
                let mut log = self.shared.log.lock().unwrap();
                log.push(match &frame {
                    Frame::Batch { side, batch } => format!("B{side}:{}", batch.num_rows()),
                    Frame::EndGroup => "G".to_owned(),
                    Frame::End => "E".to_owned(),
                });
                log.len()
            };
            if *self.shared.fail_at.lock().unwrap() == Some(n) {
                return Err(InternalError {
                    message: "the UDF raised".to_owned(),
                }
                .into());
            }
            let mut s = self.state.lock().unwrap();
            match frame {
                Frame::Batch { batch, .. } => {
                    s.rows += i64::try_from(batch.num_rows()).unwrap();
                    let copies = match self.yields {
                        Yield::Twice => 2,
                        Yield::Burst(n) => n,
                        Yield::Nothing | Yield::Count | Yield::CountTwice => 0,
                    };
                    for _ in 0..copies {
                        s.out.push_back(batch.clone());
                    }
                }
                Frame::EndGroup => {
                    let n = std::mem::take(&mut s.rows);
                    match self.yields {
                        Yield::Count => s.out.push_back(count(n)),
                        Yield::CountTwice => {
                            s.out.push_back(count(n));
                            s.out.push_back(count(n));
                        }
                        _ => {}
                    }
                }
                Frame::End => s.ended = true,
            }
            drop(s);
            self.ready.wake();
            Ok(())
        })
    }

    fn next(&self) -> BoxFuture<'_, crate::error::Result<Option<RecordBatch>>> {
        Box::pin(poll_fn(|cx| {
            if self.shared.polls.fetch_add(1, Ordering::SeqCst) > SPUN {
                return Poll::Ready(Err(InternalError {
                    message: "the operator spun".to_owned(),
                }
                .into()));
            }
            self.ready.register(cx.waker());
            let mut s = self.state.lock().unwrap();
            match s.out.pop_front() {
                Some(b) => {
                    self.shared.taken.fetch_add(1, Ordering::SeqCst);
                    Poll::Ready(Ok(Some(b)))
                }
                None if s.ended => Poll::Ready(Ok(None)),
                None => Poll::Pending,
            }
        }))
    }
}

fn pool(limit: usize) -> Arc<TreePool> {
    TreePool::with_clock(
        &MemoryConfig {
            limit: Some(limit),
            quantum: 1 << 10,
            spill_wait_timeout: Duration::from_secs(5),
            abort_release_timeout: Duration::from_secs(30),
            spill_scratch_reserve: krishiv_memory::DEFAULT_SPILL_SCRATCH_RESERVE,
            resident_guard: true,
        },
        Arc::new(TokioClock),
    )
}

/// `k INT, v BIGINT`; for a cogroup also `side INT`.
fn input(cogroup: bool) -> SchemaRef {
    let mut fields = vec![
        Field::new("k", ArrowType::Int32, true),
        Field::new("v", ArrowType::Int64, true),
    ];
    if cogroup {
        fields.push(Field::new("side", ArrowType::Int32, false));
    }
    Arc::new(Schema::new(fields))
}

fn column(index: usize, data_type: ArrowType) -> Arc<dyn ScalarExpr> {
    Arc::new(InputColumn {
        index,
        data_type,
        nullable: true,
    })
}

struct Rig {
    sink: Arc<RelationSink>,
    root: Arc<TreePool>,
    query: Arc<QueryContext>,
    ctx: TaskContext,
    shared: Arc<Shared>,
}

fn rig(shape: RelationShape, yields: Yield, limits: RelationLimits, limit: usize) -> Rig {
    rig_in(shape, yields, limits, limit, None)
}

/// As [`rig`], every frame of the map lane being `columns` (the task yields
/// nothing).
fn rig_columns(
    limits: RelationLimits,
    limit: usize,
    columns: &[(Arc<dyn ScalarExpr>, &str)],
) -> Rig {
    let r = rig(RelationShape::Map, Yield::Nothing, limits, limit);
    let lane = RelationLane {
        name: "f".to_owned(),
        udf: Arc::new(Fake {
            shared: Arc::clone(&r.shared),
            yields: Yield::Nothing,
        }),
        sides: vec![columns.iter().map(|(e, _)| Arc::clone(e)).collect()],
        names: vec![columns.iter().map(|(_, n)| (*n).to_owned()).collect()],
        client: vec![ArrowType::Int32, ArrowType::Int64],
    };
    let sink = RelationSink::try_new(
        input(false),
        RelationShape::Map,
        vec![lane],
        r.sink.schema(),
        limits,
        None,
    )
    .unwrap();
    Rig { sink, ..r }
}

/// A computed argument (as `repeat(s, n)`): a new string of `width` bytes
/// per row, already in its client form.
#[derive(Debug)]
struct Wide {
    width: usize,
}

impl ScalarExpr for Wide {
    fn data_type(&self) -> ArrowType {
        ArrowType::Utf8
    }

    fn nullable(&self) -> bool {
        false
    }

    fn evaluate(&self, batch: &RecordBatch) -> crate::error::Result<ArrayRef> {
        let s = "x".repeat(self.width);
        let a: arrow_array::StringArray = (0..batch.num_rows()).map(|_| Some(s.as_str())).collect();
        Ok(Arc::new(a))
    }
}

/// As [`rig`], its finished rows spilling to `scope`.
fn rig_in(
    shape: RelationShape,
    yields: Yield,
    limits: RelationLimits,
    limit: usize,
    scope: Option<krishiv_memory::SpillScope>,
) -> Rig {
    let shared = Arc::new(Shared::default());
    let cogroup = matches!(shape, RelationShape::CoGrouped { .. });
    let schema = input(cogroup);
    let both = || vec![column(0, ArrowType::Int32), column(1, ArrowType::Int64)];
    let names = || vec!["k".to_owned(), "v".to_owned()];
    let sides = if cogroup { 2 } else { 1 };
    let (client, output) = match yields {
        Yield::Count | Yield::CountTwice => (
            vec![ArrowType::Int64],
            match shape {
                RelationShape::Aggregate { .. } => vec![
                    Field::new("k", ArrowType::Int32, true),
                    Field::new("n", ArrowType::Int64, true),
                ],
                _ => vec![Field::new("n", ArrowType::Int64, true)],
            },
        ),
        Yield::Twice | Yield::Nothing | Yield::Burst(_) => (
            vec![ArrowType::Int32, ArrowType::Int64],
            vec![
                Field::new("k", ArrowType::Int32, true),
                Field::new("v", ArrowType::Int64, true),
            ],
        ),
    };
    let lane = RelationLane {
        name: "f".to_owned(),
        udf: Arc::new(Fake {
            shared: Arc::clone(&shared),
            yields,
        }),
        sides: (0..sides).map(|_| both()).collect(),
        names: (0..sides).map(|_| names()).collect(),
        client,
    };
    let sink = RelationSink::try_new(
        schema,
        shape,
        vec![lane],
        Arc::new(Schema::new(output)),
        limits,
        scope,
    )
    .unwrap();
    let root = pool(limit);
    let query = QueryContext::new(
        QueryId(85),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("q", Some(limit)),
    );
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
    let ctx = TaskContext::new(
        Arc::clone(&query),
        executor,
        SpillOwner::new(),
        OperatorId(0),
        "python_relation",
    );
    Rig {
        sink,
        root,
        query,
        ctx,
        shared,
    }
}

fn limits(max_rows: usize, window: usize) -> RelationLimits {
    RelationLimits {
        max_rows,
        max_bytes: 0,
        window,
    }
}

/// Rows `(k, v)`, with a side column for a cogroup.
fn rows(keys: &[i32], side: Option<i32>) -> Batch {
    let k: Int32Array = keys.iter().map(|&k| Some(k)).collect();
    let v: Int64Array = (0..keys.len()).map(|i| Some(i as i64)).collect();
    let mut columns: Vec<ArrayRef> = vec![Arc::new(k), Arc::new(v)];
    if let Some(s) = side {
        columns.push(Arc::new(Int32Array::from(vec![s; keys.len()])));
    }
    Batch::new(RecordBatch::try_new(input(side.is_some()), columns).unwrap())
}

/// Runs `batches` through the rig's sink as one driver.
async fn drive(r: &Rig, batches: Vec<Batch>) -> crate::error::Result<()> {
    let g = r.sink.global_state(&r.ctx)?;
    let mut l = r.sink.local_state(&r.ctx, &g)?;
    for b in batches {
        // `Blocked` took the batch: wait, then the next one. `NeedMemory`
        // took nothing: the same batch again.
        match r.sink.sink(b, &mut l, &g)? {
            SinkStatus::Blocked(wait) => wait.await,
            SinkStatus::NeedMemory { bytes } => panic!("no memory for {bytes}"),
            SinkStatus::NeedMoreInput | SinkStatus::Finished => {}
        }
    }
    r.sink.combine(l, &g)?;
    r.sink.finalize(&g, &r.ctx).await
}

fn done(r: Rig) {
    drop((r.ctx, r.query));
    assert_eq!(r.root.reserved(), 0, "memory left after the phase");
}

#[tokio::test]
async fn a_map_task_gets_full_frames_then_the_rest_then_end() {
    let r = rig(RelationShape::Map, Yield::Twice, limits(4, 8), 64 << 20);
    drive(
        &r,
        vec![rows(&[1, 2, 3], None), rows(&[4, 5, 6, 7, 8, 9], None)],
    )
    .await
    .unwrap();
    assert_eq!(r.shared.log(), ["B0:4", "B0:4", "B0:1", "E"]);
    assert_eq!(
        r.shared.opened.load(Ordering::SeqCst),
        1,
        "one task per driver"
    );
    done(r);
}

#[tokio::test]
async fn an_empty_partition_opens_no_task() {
    let r = rig(RelationShape::Map, Yield::Twice, limits(4, 8), 64 << 20);
    drive(&r, Vec::new()).await.unwrap();
    assert!(r.shared.log().is_empty());
    assert_eq!(r.shared.opened.load(Ordering::SeqCst), 0);
    done(r);
}

#[tokio::test]
async fn a_grouped_task_sees_each_key_s_rows_together_and_one_end_group_per_key() {
    let r = rig(
        RelationShape::Grouped { keys: vec![0] },
        Yield::Count,
        limits(2, 16),
        64 << 20,
    );
    // Key 2 spans the two batches; key 3 has 3 rows (cut by maxRecords 2).
    drive(&r, vec![rows(&[1, 2, 2], None), rows(&[2, 3, 3, 3], None)])
        .await
        .unwrap();
    assert_eq!(
        r.shared.log(),
        ["B0:1", "G", "B0:2", "B0:1", "G", "B0:2", "B0:1", "G", "E"]
    );
    done(r);
}

#[tokio::test]
async fn a_grouped_task_without_keys_sees_one_group() {
    let r = rig(
        RelationShape::Grouped { keys: Vec::new() },
        Yield::Count,
        limits(0, 16),
        64 << 20,
    );
    drive(&r, vec![rows(&[1, 2], None), rows(&[3], None)])
        .await
        .unwrap();
    assert_eq!(r.shared.log(), ["B0:2", "B0:1", "G", "E"]);
    done(r);
}

#[tokio::test]
async fn a_cogroup_key_on_one_side_gets_a_zero_row_frame_for_the_other() {
    let r = rig(
        RelationShape::CoGrouped {
            keys: vec![0],
            side: 2,
        },
        Yield::Count,
        limits(0, 16),
        64 << 20,
    );
    // Sorted by key then side: key 1 left only, key 2 both, key 3 right only.
    drive(
        &r,
        vec![
            rows(&[1, 2], Some(0)),
            rows(&[2, 2], Some(1)),
            rows(&[3], Some(1)),
        ],
    )
    .await
    .unwrap();
    assert_eq!(
        r.shared.log(),
        [
            "B0:1", "B1:0", "G", "B0:1", "B1:2", "G", "B0:0", "B1:1", "G", "E"
        ]
    );
    done(r);
}

#[tokio::test]
async fn a_one_sided_cogroup_key_over_two_batches_gets_one_empty_frame_first() {
    // Key 1 left only and key 3 right only, each split over two batches.
    for (side, expected) in [
        (0, ["B0:1", "B0:1", "B1:0", "G", "E"]),
        (1, ["B0:0", "B1:1", "B1:1", "G", "E"]),
    ] {
        let r = rig(
            RelationShape::CoGrouped {
                keys: vec![0],
                side: 2,
            },
            Yield::Count,
            limits(0, 16),
            64 << 20,
        );
        let key = 1 + 2 * side;
        drive(&r, vec![rows(&[key], Some(side)), rows(&[key], Some(side))])
            .await
            .unwrap();
        assert_eq!(r.shared.log(), expected, "side {side}");
        done(r);
    }
}

#[tokio::test]
async fn an_aggregate_lane_gets_one_group_per_key() {
    let r = rig(
        RelationShape::Aggregate {
            keys: vec![0],
            key_values: vec![0],
        },
        Yield::Count,
        limits(0, 16),
        64 << 20,
    );
    drive(&r, vec![rows(&[5, 5, 7], None)]).await.unwrap();
    assert_eq!(r.shared.log(), ["B0:2", "G", "B0:1", "G", "E"]);
    done(r);
}

/// An aggregate shows as many key values as it compares keys: a plan
/// that says otherwise is refused, not run with misaligned keys.
#[test]
fn an_aggregate_with_key_values_unlike_its_keys_is_refused() {
    let lane = RelationLane {
        name: "f".to_owned(),
        udf: Arc::new(Fake {
            shared: Arc::new(Shared::default()),
            yields: Yield::Count,
        }),
        sides: vec![vec![
            column(0, ArrowType::Int32),
            column(1, ArrowType::Int64),
        ]],
        names: vec![vec!["k".to_owned(), "v".to_owned()]],
        client: vec![ArrowType::Int64],
    };
    let err = RelationSink::try_new(
        input(false),
        RelationShape::Aggregate {
            keys: vec![0],
            key_values: Vec::new(),
        },
        vec![lane],
        Arc::new(Schema::new(vec![Field::new("n", ArrowType::Int64, true)])),
        limits(0, 16),
        None,
    )
    .err()
    .unwrap();
    assert!(err.to_string().contains("key values"), "{err}");
}

#[tokio::test]
async fn a_full_window_blocks_the_driver_until_the_task_takes_frames() {
    let r = rig(RelationShape::Map, Yield::Nothing, limits(1, 2), 64 << 20);
    r.shared.closed.store(true, Ordering::SeqCst);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    let SinkStatus::Blocked(wait) = r.sink.sink(rows(&[1, 2, 3], None), &mut l, &g).unwrap() else {
        panic!("three frames over a window of two must block");
    };
    let waiting = tokio::spawn(wait);
    tokio::task::yield_now().await;
    assert!(
        !waiting.is_finished(),
        "the window freed while the task took nothing"
    );
    assert!(r.shared.log().is_empty());
    r.shared.open_gate();
    tokio::time::timeout(Duration::from_secs(5), waiting)
        .await
        .unwrap()
        .unwrap();
    r.sink.combine(l, &g).unwrap();
    r.sink.finalize(&g, &r.ctx).await.unwrap();
    assert_eq!(r.shared.log(), ["B0:1", "B0:1", "B0:1", "E"]);
    drop(g);
    done(r);
}

#[tokio::test]
async fn a_failed_frame_fails_the_phase_and_releases_its_memory() {
    let r = rig(
        RelationShape::Grouped { keys: vec![0] },
        Yield::Count,
        limits(0, 16),
        64 << 20,
    );
    *r.shared.fail_at.lock().unwrap() = Some(2);
    let err = drive(&r, vec![rows(&[1, 2, 3], None)]).await.unwrap_err();
    assert!(err.to_string().contains("the UDF raised"), "{err}");
    done(r);
}

#[tokio::test]
async fn a_cancel_mid_group_ends_cleanly_and_drops_the_task() {
    let r = rig(
        RelationShape::Grouped { keys: vec![0] },
        Yield::Count,
        limits(1, 64),
        64 << 20,
    );
    r.shared.closed.store(true, Ordering::SeqCst);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    assert!(matches!(
        r.sink.sink(rows(&[1, 1, 1], None), &mut l, &g).unwrap(),
        SinkStatus::NeedMoreInput
    ));
    assert!(r.root.reserved() > 0, "the queued frames hold memory");
    r.sink.combine(l, &g).unwrap();
    r.query.cancel();
    let err = tokio::time::timeout(Duration::from_secs(5), r.sink.finalize(&g, &r.ctx))
        .await
        .unwrap()
        .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::OperationCanceled);
    drop(g);
    assert_eq!(
        r.shared.dropped.load(Ordering::SeqCst),
        1,
        "the task was dropped"
    );
    done(r);
}

#[tokio::test]
async fn a_refused_reservation_sends_nothing_so_the_retry_is_exact() {
    let r = rig(RelationShape::Map, Yield::Twice, limits(0, 8), 512);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    let keys: Vec<i32> = (0..1000).collect();
    let status = r.sink.sink(rows(&keys, None), &mut l, &g).unwrap();
    assert!(
        matches!(status, SinkStatus::NeedMemory { .. }),
        "{status:?}"
    );
    assert_eq!(r.shared.opened.load(Ordering::SeqCst), 0, "nothing sent");
    drop((l, g));
    done(r);
}

/// A runtime of the scalar kinds only.
#[derive(Debug)]
struct ScalarOnly;

impl crate::python_eval::UdfRuntime for ScalarOnly {
    fn prepare(
        &self,
        spec: &crate::python_eval::UdfCallSpec,
    ) -> crate::error::Result<Arc<dyn crate::python_eval::PreparedUdf>> {
        Err(InternalError {
            message: format!("no {}", spec.name),
        }
        .into())
    }
}

#[test]
fn a_runtime_without_relation_kinds_refuses_them_as_not_available() {
    let spec = super::RelationCallSpec {
        name: "f".to_owned(),
        version: 3,
        eval_type: krishiv_plan::PythonEvalType::GroupedMapArrow,
        thread_safe: true,
        input_types: vec![vec![krishiv_types::DataType::Long]],
        input_names: vec![vec!["k".to_owned()]],
        key_count: 1,
        arity: 1,
        output_names: vec!["n".to_owned()],
        output_types: vec![krishiv_types::DataType::Long],
        output_arrow: Arc::new(Schema::new(vec![Field::new("n", ArrowType::Int64, true)])),
        conf: super::RunnerConf::default(),
    };
    let err = crate::python_eval::UdfRuntime::prepare_relation(&ScalarOnly, &spec)
        .err()
        .unwrap();
    assert_eq!(err.error_class(), ErrorClass::KrishivUdfNotAvailable);
    assert!(err.to_string().contains("f@3"), "{err}");
}

#[tokio::test]
async fn max_bytes_per_batch_cuts_frames_between_rows() {
    let limits = RelationLimits {
        max_rows: 0,
        max_bytes: 1,
        window: 64,
    };
    let r = rig(RelationShape::Map, Yield::Twice, limits, 64 << 20);
    assert!(format!("{:?}", r.sink).contains("RelationSink"));
    drive(&r, vec![rows(&[1, 2, 3], None)]).await.unwrap();
    // At least one row per frame, however small the byte cap.
    assert_eq!(r.shared.log(), ["B0:1", "B0:1", "B0:1", "E"]);
    done(r);
}

/// A computed argument of uneven width: row `i` (its `v`) is a string of
/// `widths[i]` bytes.
#[derive(Debug)]
struct Uneven {
    widths: Vec<usize>,
}

impl ScalarExpr for Uneven {
    fn data_type(&self) -> ArrowType {
        ArrowType::Utf8
    }

    fn nullable(&self) -> bool {
        false
    }

    fn evaluate(&self, batch: &RecordBatch) -> crate::error::Result<ArrayRef> {
        let v = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let a: arrow_array::StringArray = v
            .values()
            .iter()
            .map(|&i| Some("x".repeat(self.widths[usize::try_from(i).unwrap()])))
            .collect();
        Ok(Arc::new(a))
    }
}

/// The frames a map task gets for one batch of rows of `widths` bytes under
/// a `maxBytesPerBatch` of 32 KiB.
async fn frames_of_widths(widths: Vec<usize>) -> Vec<String> {
    let limits = RelationLimits {
        max_rows: 0,
        max_bytes: 32 << 10,
        window: 64,
    };
    let keys: Vec<i32> = (0..i32::try_from(widths.len()).unwrap()).collect();
    let wide: Arc<dyn ScalarExpr> = Arc::new(Uneven { widths });
    let r = rig_columns(limits, 64 << 20, &[(wide, "s")]);
    drive(&r, vec![rows(&keys, None)]).await.unwrap();
    let log = r.shared.log();
    done(r);
    log
}

#[tokio::test]
async fn max_bytes_per_batch_cuts_by_the_frame_columns_row_sizes() {
    // Two 30 KiB strings, then 100 empty ones. The input rows are a few
    // bytes each; the argument is not. As Spark's writer: the first frame
    // closes at the row that brings it to the cap (row 2), not at the
    // batch's average bytes per row.
    let mut widths = vec![30 << 10, 30 << 10];
    widths.extend([0; 100]);
    assert_eq!(frames_of_widths(widths).await, ["B0:2", "B0:100", "E"]);
    // 20 KiB rows: two per frame (the second crosses the cap).
    assert_eq!(
        frames_of_widths(vec![20 << 10; 5]).await,
        ["B0:2", "B0:2", "B0:1", "E"]
    );
}

#[tokio::test]
async fn a_row_over_max_bytes_per_batch_closes_its_frame() {
    // As Spark's writer: a row over the cap by itself ends the frame it
    // joins (rows before it under the cap share it), and alone it is a
    // frame of its own.
    assert_eq!(
        frames_of_widths(vec![10, 40 << 10, 10, 10]).await,
        ["B0:2", "B0:2", "E"]
    );
    assert_eq!(
        frames_of_widths(vec![40 << 10, 10, 10]).await,
        ["B0:1", "B0:2", "E"]
    );
}

#[tokio::test]
async fn a_result_of_another_width_fails_the_phase() {
    // The task yields `n BIGINT` where `k, v` were declared.
    let shared = Arc::new(Shared::default());
    let lane = RelationLane {
        name: "f".to_owned(),
        udf: Arc::new(Fake {
            shared: Arc::clone(&shared),
            yields: Yield::Count,
        }),
        sides: vec![vec![
            column(0, ArrowType::Int32),
            column(1, ArrowType::Int64),
        ]],
        names: vec![vec!["k".to_owned(), "v".to_owned()]],
        client: vec![ArrowType::Int32, ArrowType::Int64],
    };
    let mut r = rig(
        RelationShape::Grouped { keys: vec![0] },
        Yield::Count,
        limits(0, 8),
        64 << 20,
    );
    r.sink = RelationSink::try_new(
        input(false),
        RelationShape::Grouped { keys: vec![0] },
        vec![lane],
        Arc::new(Schema::new(vec![
            Field::new("k", ArrowType::Int32, true),
            Field::new("v", ArrowType::Int64, true),
        ])),
        limits(0, 8),
        None,
    )
    .unwrap();
    let err = drive(&r, vec![rows(&[1, 2], None)]).await.unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivPythonUdfResultMismatch,
        "{err}"
    );
    assert!(
        err.to_string().contains("columns where 2 were declared"),
        "{err}"
    );
    done(r);
}

#[tokio::test]
async fn the_window_bounds_the_frames_and_the_rest_waits_in_the_backlog() {
    let r = rig(RelationShape::Map, Yield::Nothing, limits(1, 2), 64 << 20);
    r.shared.closed.store(true, Ordering::SeqCst);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    let SinkStatus::Blocked(wait) = r
        .sink
        .sink(rows(&[1, 2, 3, 4, 5], None), &mut l, &g)
        .unwrap()
    else {
        panic!("five frames over a window of two must block");
    };
    // Two frames in flight (one being sent), three rows cut and waiting.
    assert_eq!(super::sink::queued(&mut l).unwrap(), (vec![2], 3));
    r.shared.open_gate();
    tokio::time::timeout(Duration::from_secs(5), wait)
        .await
        .unwrap();
    r.sink.combine(l, &g).unwrap();
    r.sink.finalize(&g, &r.ctx).await.unwrap();
    assert_eq!(
        r.shared.log(),
        ["B0:1", "B0:1", "B0:1", "B0:1", "B0:1", "E"]
    );
    drop(g);
    done(r);
}

#[tokio::test]
async fn a_result_the_pool_refuses_stops_the_task_s_reads_until_reserved() {
    let limit = 256 << 10;
    let r = rig(RelationShape::Map, Yield::Burst(64), limits(0, 8), limit);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    let keys: Vec<i32> = (0..1000).collect();
    let status = r.sink.sink(rows(&keys, None), &mut l, &g).unwrap();
    assert!(matches!(status, SinkStatus::NeedMoreInput), "{status:?}");
    // Each result is reserved as it is read: 64 copies of the input do
    // not fit, so the task's reads stop at the refused one.
    let taken = r.shared.taken.load(Ordering::SeqCst);
    assert!(taken > 0 && taken < 64, "{taken} results read");
    assert!(r.root.reserved() <= limit);
    // The next call asks the driver for memory before it takes its input.
    let status = r.sink.sink(rows(&[1], None), &mut l, &g).unwrap();
    assert!(
        matches!(status, SinkStatus::NeedMemory { .. }),
        "{status:?}"
    );
    assert_eq!(r.shared.taken.load(Ordering::SeqCst), taken);
    drop((l, g));
    assert_eq!(r.shared.dropped.load(Ordering::SeqCst), 1);
    done(r);
}

#[tokio::test]
async fn an_aggregate_task_yielding_another_number_of_values_than_groups_fails() {
    // Too many values fail as soon as they outnumber the groups.
    for (yields, values) in [(Yield::CountTwice, " values"), (Yield::Nothing, "0 values")] {
        let r = rig(
            RelationShape::Aggregate {
                keys: vec![0],
                key_values: vec![0],
            },
            yields,
            limits(0, 16),
            64 << 20,
        );
        let err = drive(&r, vec![rows(&[5, 5, 7], None)]).await.unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::KrishivPythonUdfResultMismatch,
            "{err}"
        );
        assert!(err.to_string().contains(values), "{err}");
        assert!(err.to_string().contains("values for 2 groups"), "{err}");
        done(r);
    }
}

#[tokio::test]
async fn a_blocked_driver_sends_each_row_once() {
    // Two frames per batch over a window of one: every batch blocks.
    let r = rig(
        RelationShape::Grouped { keys: vec![0] },
        Yield::Count,
        limits(0, 1),
        64 << 20,
    );
    drive(&r, vec![rows(&[1, 2], None), rows(&[2, 3], None)])
        .await
        .unwrap();
    assert_eq!(
        r.shared.log(),
        ["B0:1", "G", "B0:1", "B0:1", "G", "B0:1", "G", "E"]
    );
    done(r);
}

#[tokio::test]
async fn a_weighted_batch_is_refused() {
    let r = rig(RelationShape::Map, Yield::Twice, limits(0, 8), 64 << 20);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    let (data, _) = rows(&[1, 2], None).into_parts();
    let retracting = Batch::with_weights(data, krishiv_types::batch::Weights::Const(-1)).unwrap();
    let err = r.sink.sink(retracting, &mut l, &g).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(err.to_string().contains("weighted"), "{err}");
    drop((l, g));
    done(r);
}

#[tokio::test]
async fn an_aggregate_waits_for_its_slowest_lane_without_spinning() {
    let mut r = rig(
        RelationShape::Aggregate {
            keys: vec![0],
            key_values: vec![0],
        },
        Yield::Count,
        limits(0, 16),
        64 << 20,
    );
    let slow = Arc::new(Shared::default());
    slow.closed.store(true, Ordering::SeqCst);
    let lane = |shared: &Arc<Shared>, name: &str| RelationLane {
        name: name.to_owned(),
        udf: Arc::new(Fake {
            shared: Arc::clone(shared),
            yields: Yield::Count,
        }),
        sides: vec![vec![
            column(0, ArrowType::Int32),
            column(1, ArrowType::Int64),
        ]],
        names: vec![vec!["k".to_owned(), "v".to_owned()]],
        client: vec![ArrowType::Int64],
    };
    r.sink = RelationSink::try_new(
        input(false),
        RelationShape::Aggregate {
            keys: vec![0],
            key_values: vec![0],
        },
        vec![lane(&r.shared, "f"), lane(&slow, "g")],
        Arc::new(Schema::new(vec![
            Field::new("k", ArrowType::Int32, true),
            Field::new("n", ArrowType::Int64, true),
            Field::new("m", ArrowType::Int64, true),
        ])),
        limits(0, 16),
        None,
    )
    .unwrap();
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    let status = r.sink.sink(rows(&[5, 5, 7], None), &mut l, &g).unwrap();
    assert!(matches!(status, SinkStatus::NeedMoreInput), "{status:?}");
    r.sink.combine(l, &g).unwrap();
    let mut finalize = r.sink.finalize(&g, &r.ctx);
    // The fast lane's values wait for the slow lane's: pending, no busy loop.
    let first = poll_fn(|cx| Poll::Ready(finalize.as_mut().poll(cx))).await;
    assert!(first.is_pending(), "{first:?}");
    assert_eq!(r.shared.log(), ["B0:2", "G", "B0:1", "G", "E"]);
    assert!(slow.log().is_empty());
    slow.open_gate();
    tokio::time::timeout(Duration::from_secs(5), finalize)
        .await
        .unwrap()
        .unwrap();
    assert!(
        r.shared.polls.load(Ordering::SeqCst) < 100,
        "polled in a loop"
    );
    drop(g);
    done(r);
}

#[tokio::test]
async fn a_result_refused_in_finalize_spills_instead_of_waiting() {
    let dir = tempfile::tempdir().unwrap();
    let scope = krishiv_memory::SpillManager::open(&krishiv_memory::SpillConfig {
        dir: dir.path().join("spill"),
        ..krishiv_memory::SpillConfig::default()
    })
    .unwrap()
    .scope()
    .unwrap();
    let r = rig_in(
        RelationShape::Map,
        Yield::Burst(64),
        limits(0, 8),
        256 << 10,
        Some(scope),
    );
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    let keys: Vec<i32> = (0..1000).collect();
    let status = r.sink.sink(rows(&keys, None), &mut l, &g).unwrap();
    assert!(matches!(status, SinkStatus::NeedMoreInput), "{status:?}");
    assert!(
        r.shared.taken.load(Ordering::SeqCst) < 64,
        "a result refused"
    );
    r.sink.combine(l, &g).unwrap();
    // Faster than the pool's spill wait: the refused results are written
    // out, not waited for.
    tokio::time::timeout(Duration::from_secs(4), r.sink.finalize(&g, &r.ctx))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(r.shared.taken.load(Ordering::SeqCst), 64);
    drop(g);
    done(r);
}

/// A frame whose copies the pool refuses in `finalize` while a lane's
/// results hold the rest of the pool: the results are stored first, written
/// out, which frees their memory, so the frame gets its copies instead of
/// waiting for memory nothing can release (the lane queues are not
/// spillable, nor the store's parts before it finalizes).
#[tokio::test]
async fn finalize_drains_results_before_waiting_for_frame_copies() {
    let dir = tempfile::tempdir().unwrap();
    let scope = krishiv_memory::SpillManager::open(&krishiv_memory::SpillConfig {
        dir: dir.path().join("spill"),
        ..krishiv_memory::SpillConfig::default()
    })
    .unwrap()
    .scope()
    .unwrap();
    let mut r = rig(RelationShape::Map, Yield::Nothing, limits(100, 8), 1 << 20);
    // `k INT, v BIGINT, d DECIMAL(18, 2)`: lane `f` echoes `k, v` many
    // times; lane `g` takes `d`, whose frames are copied to the client's
    // `Decimal128`.
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", ArrowType::Int32, true),
        Field::new("v", ArrowType::Int64, true),
        Field::new("d", ArrowType::Decimal64(18, 2), true),
    ]));
    let quiet = Arc::new(Shared::default());
    let lane =
        |shared: &Arc<Shared>, yields, sides: Vec<Arc<dyn ScalarExpr>>, names| RelationLane {
            name: "f".to_owned(),
            udf: Arc::new(Fake {
                shared: Arc::clone(shared),
                yields,
            }),
            sides: vec![sides],
            names: vec![names],
            client: vec![ArrowType::Int32, ArrowType::Int64],
        };
    r.sink = RelationSink::try_new(
        Arc::clone(&schema),
        RelationShape::Map,
        vec![
            lane(
                &r.shared,
                Yield::Burst(2000),
                vec![column(0, ArrowType::Int32), column(1, ArrowType::Int64)],
                vec!["k".to_owned(), "v".to_owned()],
            ),
            lane(
                &quiet,
                Yield::Nothing,
                vec![column(2, ArrowType::Decimal64(18, 2))],
                vec!["d".to_owned()],
            ),
        ],
        Arc::new(Schema::new(vec![
            Field::new("k", ArrowType::Int32, true),
            Field::new("v", ArrowType::Int64, true),
        ])),
        limits(100, 8),
        Some(scope),
    )
    .unwrap();
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    // 150 rows: a frame of 100 now, 50 held for `combine`.
    let n = 150;
    let k: Int32Array = (0..n).map(Some).collect();
    let v: Int64Array = (0..i64::from(n)).map(Some).collect();
    let s = arrow_array::Decimal64Array::from_iter_values(0..i64::from(n))
        .with_precision_and_scale(18, 2)
        .unwrap();
    let batch = Batch::new(
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(k), Arc::new(v), Arc::new(s)],
        )
        .unwrap(),
    );
    let status = r.sink.sink(batch, &mut l, &g).unwrap();
    assert!(matches!(status, SinkStatus::NeedMoreInput), "{status:?}");
    assert!(
        r.shared.taken.load(Ordering::SeqCst) < 2000,
        "a result refused"
    );
    // Whatever the pool has left goes to another holder, so the held frame
    // can get its copies only from memory the results release.
    let mut hog = r.ctx.consumer().reservation();
    let mut step = 1 << 20;
    while step >= 64 {
        if hog.try_grow(step).is_err() {
            step /= 2;
        }
    }
    r.sink.combine(l, &g).unwrap();
    // The held frame is refused its copies.
    assert_eq!(quiet.log(), ["B0:100"]);
    // The other holder lets go once the held frame is sent: before that,
    // only the results' memory can be had.
    let release = async {
        while !quiet.log().iter().any(|e| e == "B0:50") {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        drop(hog);
    };
    let (finalized, ()) = tokio::time::timeout(
        Duration::from_secs(4),
        futures::future::join(r.sink.finalize(&g, &r.ctx), release),
    )
    .await
    .unwrap();
    finalized.unwrap();
    assert_eq!(quiet.log(), ["B0:100", "B0:50", "E"]);
    assert_eq!(r.shared.taken.load(Ordering::SeqCst), 4000);
    drop(g);
    done(r);
}

/// A batch of two groups: the first group's frame is taken while the
/// second waits in the backlog, which still holds its share of the batch's
/// reservation (the slices keep the batch's buffers).
#[tokio::test]
async fn a_batch_s_later_group_keeps_its_share_of_the_input_reservation() {
    let r = rig(
        RelationShape::Grouped { keys: vec![0] },
        Yield::Nothing,
        limits(0, 1),
        64 << 20,
    );
    r.shared.closed.store(true, Ordering::SeqCst);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    let keys: Vec<i32> = (0..20_000).map(|i| 1 + i / 10_000).collect();
    let batch = rows(&keys, None);
    let bytes = batch.data().get_array_memory_size();
    let SinkStatus::Blocked(wait) = r.sink.sink(batch, &mut l, &g).unwrap() else {
        panic!("a window of one must block");
    };
    // The first group's frame in flight; its end and the second group's
    // rows wait.
    assert_eq!(super::sink::queued(&mut l).unwrap(), (vec![1], 2));
    // The task takes the first frame only.
    *r.shared.admit.lock().unwrap() = Some(1);
    r.shared.open_gate();
    let mut wait = wait;
    let first = poll_fn(|cx| Poll::Ready(wait.as_mut().poll(cx))).await;
    assert!(first.is_pending(), "ready with the task stuck");
    // The first frame is taken and its memory freed; the second group's
    // rows are still reserved.
    assert_eq!(r.shared.log(), ["B0:10000"]);
    assert_eq!(super::sink::queued(&mut l).unwrap(), (vec![1], 1));
    assert!(
        r.root.reserved() >= bytes / 2,
        "{} of {bytes} bytes reserved",
        r.root.reserved()
    );
    *r.shared.admit.lock().unwrap() = None;
    r.shared.open_gate();
    tokio::time::timeout(Duration::from_secs(5), wait)
        .await
        .unwrap();
    r.sink.combine(l, &g).unwrap();
    r.sink.finalize(&g, &r.ctx).await.unwrap();
    assert_eq!(r.shared.log(), ["B0:10000", "G", "B0:10000", "G", "E"]);
    drop(g);
    done(r);
}

/// A frame column computed from the rows (`python_udaf(repeat(s, 1000))`)
/// holds memory of its own while the frame is queued: it is reserved, as
/// the copies of a client form are, not only the input rows. A column read
/// shares the input's buffers and adds nothing.
#[tokio::test]
async fn a_computed_frame_column_is_reserved_while_queued() {
    let width = 1000;
    let r = rig_columns(
        limits(0, 4),
        64 << 20,
        &[
            (column(0, ArrowType::Int32), "k"),
            (Arc::new(Wide { width }), "s"),
        ],
    );
    r.shared.closed.store(true, Ordering::SeqCst);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    let keys: Vec<i32> = (0..100).collect();
    let batch = rows(&keys, None);
    let input = batch.data().get_array_memory_size();
    let status = r.sink.sink(batch, &mut l, &g).unwrap();
    assert!(matches!(status, SinkStatus::NeedMoreInput), "{status:?}");
    // The frame waits for the closed task: its rows and its strings.
    assert_eq!(super::sink::queued(&mut l).unwrap(), (vec![1], 0));
    let reserved = r.root.reserved();
    assert!(
        reserved >= input + 100 * width,
        "{reserved} bytes reserved for {input} input bytes and {} computed",
        100 * width
    );
    r.shared.open_gate();
    r.sink.combine(l, &g).unwrap();
    r.sink.finalize(&g, &r.ctx).await.unwrap();
    assert_eq!(r.shared.log(), ["B0:100", "E"]);
    drop(g);
    done(r);
}

/// A column read adds nothing to the frame's reservation: it shares the
/// input's buffers, which the input's reservation covers.
#[tokio::test]
async fn a_frame_of_column_reads_reserves_only_the_input() {
    let r = rig_columns(
        limits(0, 4),
        64 << 20,
        &[
            (column(0, ArrowType::Int32), "k"),
            (column(1, ArrowType::Int64), "v"),
        ],
    );
    r.shared.closed.store(true, Ordering::SeqCst);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    let keys: Vec<i32> = (0..10_000).collect();
    let batch = rows(&keys, None);
    let input = batch.data().get_array_memory_size();
    let status = r.sink.sink(batch, &mut l, &g).unwrap();
    assert!(matches!(status, SinkStatus::NeedMoreInput), "{status:?}");
    let reserved = r.root.reserved();
    // The quantum rounds the input's bytes up.
    assert!(
        reserved < input + (4 << 10),
        "{reserved} bytes reserved for {input} input bytes"
    );
    r.shared.open_gate();
    r.sink.combine(l, &g).unwrap();
    r.sink.finalize(&g, &r.ctx).await.unwrap();
    drop(g);
    done(r);
}

/// More frames per batch than the window takes (`maxRecordsPerBatch=1`):
/// the driver is blocked until its backlog drained, so the cuts held never
/// outgrow one input batch, however many batches come.
#[tokio::test]
async fn the_backlog_drains_before_the_next_input_is_taken() {
    let r = rig(RelationShape::Map, Yield::Nothing, limits(1, 4), 64 << 20);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    let keys: Vec<i32> = (0..100).collect();
    for _ in 0..5 {
        // `Blocked` took the batch: wait, then the next one.
        if let SinkStatus::Blocked(wait) = r.sink.sink(rows(&keys, None), &mut l, &g).unwrap() {
            tokio::time::timeout(Duration::from_secs(5), wait)
                .await
                .unwrap();
        }
        let (_, backlog) = super::sink::queued(&mut l).unwrap();
        assert_eq!(backlog, 0, "cuts left when the next input comes");
    }
    r.sink.combine(l, &g).unwrap();
    r.sink.finalize(&g, &r.ctx).await.unwrap();
    assert_eq!(r.shared.log().len(), 501);
    drop(g);
    done(r);
}

/// A blocked driver's future frames the backlog only as the task takes
/// frames: while the task is stuck it stays pending, and the cuts keep
/// their reservation.
#[tokio::test]
async fn a_blocked_future_waits_while_the_task_takes_nothing() {
    let r = rig(RelationShape::Map, Yield::Nothing, limits(1, 2), 64 << 20);
    r.shared.closed.store(true, Ordering::SeqCst);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    let keys: Vec<i32> = (0..10).collect();
    let SinkStatus::Blocked(wait) = r.sink.sink(rows(&keys, None), &mut l, &g).unwrap() else {
        panic!("ten frames over a window of two must block");
    };
    let waiting = tokio::spawn(wait);
    tokio::task::yield_now().await;
    assert!(!waiting.is_finished(), "ready while the task took nothing");
    assert_eq!(super::sink::queued(&mut l).unwrap(), (vec![2], 8));
    r.shared.open_gate();
    tokio::time::timeout(Duration::from_secs(5), waiting)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(super::sink::queued(&mut l).unwrap().1, 0);
    r.sink.combine(l, &g).unwrap();
    r.sink.finalize(&g, &r.ctx).await.unwrap();
    assert_eq!(r.shared.log().len(), 11);
    drop(g);
    done(r);
}
