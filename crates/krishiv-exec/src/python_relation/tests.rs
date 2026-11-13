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
    gate: AtomicWaker,
    /// Fail the n-th frame (counted from 1).
    fail_at: std::sync::Mutex<Option<usize>>,
    /// Tasks dropped (cancelled or finished).
    dropped: std::sync::atomic::AtomicUsize,
}

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
                if self.shared.closed.load(Ordering::SeqCst) {
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
                    if matches!(self.yields, Yield::Twice) {
                        s.out.push_back(batch.clone());
                        s.out.push_back(batch);
                    }
                }
                Frame::EndGroup => {
                    if matches!(self.yields, Yield::Count) {
                        let n = std::mem::take(&mut s.rows);
                        s.out.push_back(count(n));
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
            self.ready.register(cx.waker());
            let mut s = self.state.lock().unwrap();
            match s.out.pop_front() {
                Some(b) => Poll::Ready(Ok(Some(b))),
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
    let shared = Arc::new(Shared::default());
    let cogroup = matches!(shape, RelationShape::CoGrouped { .. });
    let schema = input(cogroup);
    let both = || vec![column(0, ArrowType::Int32), column(1, ArrowType::Int64)];
    let names = || vec!["k".to_owned(), "v".to_owned()];
    let sides = if cogroup { 2 } else { 1 };
    let (client, output) = match yields {
        Yield::Count => (
            vec![ArrowType::Int64],
            match shape {
                RelationShape::Aggregate { .. } => vec![
                    Field::new("k", ArrowType::Int32, true),
                    Field::new("n", ArrowType::Int64, true),
                ],
                _ => vec![Field::new("n", ArrowType::Int64, true)],
            },
        ),
        Yield::Twice | Yield::Nothing => (
            vec![ArrowType::Int32, ArrowType::Int64],
            vec![
                Field::new("k", ArrowType::Int32, true),
                Field::new("v", ArrowType::Int64, true),
            ],
        ),
    };
    let lane = RelationLane {
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
        None,
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
        loop {
            match r.sink.sink(b.clone(), &mut l, &g)? {
                SinkStatus::Blocked(wait) => wait.await,
                SinkStatus::NeedMemory { bytes } => panic!("no memory for {bytes}"),
                _ => break,
            }
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
async fn an_aggregate_lane_gets_one_group_per_key() {
    let r = rig(
        RelationShape::Aggregate { keys: vec![0] },
        Yield::Count,
        limits(0, 16),
        64 << 20,
    );
    drive(&r, vec![rows(&[5, 5, 7], None)]).await.unwrap();
    assert_eq!(r.shared.log(), ["B0:2", "G", "B0:1", "G", "E"]);
    done(r);
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

#[tokio::test]
async fn a_result_of_another_width_fails_the_phase() {
    // The task yields `n BIGINT` where `k, v` were declared.
    let shared = Arc::new(Shared::default());
    let lane = RelationLane {
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
    assert!(
        err.to_string().contains("columns where 2 were declared"),
        "{err}"
    );
    done(r);
}
