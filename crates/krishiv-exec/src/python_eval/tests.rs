// SPDX-License-Identifier: Apache-2.0
//! The `PythonEval` sink driven by hand (P2-T059 unit 1): the in-flight
//! window blocks the driver, a refused reservation leaves the batch
//! unsubmitted so the retry is exact, and a failed or cancelled phase leaves
//! the query's pool empty.
#![expect(
    clippy::disallowed_methods,
    reason = "test-only runtime plumbing; never runs on a compute worker"
)]

use std::future::poll_fn;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::Poll;
use std::time::Duration;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::future::BoxFuture;
use futures::task::AtomicWaker;
use krishiv_common::error::InternalError;
use krishiv_memory::{MemoryConfig, MemoryPool, SpillOwner, TreePool};
use krishiv_types::batch::Batch;

use super::{EvalCall, PreparedUdf, PythonEvalSink};
use crate::operator::{Sink, SinkStatus};
use crate::operators::ScalarExpr;
use crate::{
    ExecConfig, Executor, OperatorId, QueryContext, QueryId, TaskContext, TokioClock,
    TokioExecutor, WorkloadClass,
};

/// Column `i` of the input.
#[derive(Debug)]
struct Col(usize);

impl ScalarExpr for Col {
    fn data_type(&self) -> ArrowType {
        ArrowType::Int64
    }

    fn nullable(&self) -> bool {
        true
    }

    fn evaluate(&self, batch: &RecordBatch) -> crate::error::Result<ArrayRef> {
        Ok(Arc::clone(batch.column(self.0)))
    }
}

/// When the fake's results may complete.
#[derive(Default)]
struct Gate {
    open: AtomicBool,
    waker: AtomicWaker,
}

impl Gate {
    fn open(&self) {
        self.open.store(true, Ordering::SeqCst);
        self.waker.wake();
    }
}

/// `x + 1` once the gate is open; fails every call when `fail` is set, and
/// at once (without waiting for the gate) a call whose first value is at
/// least `fail_at_once`.
struct Fake {
    gate: Arc<Gate>,
    calls: Arc<AtomicUsize>,
    lens: Arc<std::sync::Mutex<Vec<usize>>>,
    fail: bool,
    fail_at_once: Option<i64>,
}

impl PreparedUdf for Fake {
    fn call(&self, args: RecordBatch) -> BoxFuture<'static, crate::error::Result<ArrayRef>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.lens.lock().unwrap().push(args.num_rows());
        let gate = Arc::clone(&self.gate);
        let fail = self.fail;
        let first = args
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .filter(|x| !x.is_empty())
            .map(|x| x.value(0));
        let at_once = self
            .fail_at_once
            .is_some_and(|v| first.is_some_and(|f| f >= v));
        Box::pin(async move {
            if at_once {
                return Err(InternalError {
                    message: "the UDF raised at once".to_owned(),
                }
                .into());
            }
            poll_fn(|cx| {
                gate.waker.register(cx.waker());
                if gate.open.load(Ordering::SeqCst) {
                    Poll::Ready(())
                } else {
                    Poll::Pending
                }
            })
            .await;
            if fail {
                return Err(InternalError {
                    message: "the UDF raised".to_owned(),
                }
                .into());
            }
            let x = args
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone();
            let out: Int64Array = x.iter().map(|v| v.map(|v| v + 1)).collect();
            Ok(Arc::new(out) as ArrayRef)
        })
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

/// What a test drives: the sink, its query and pool, and the fake's state.
struct Rig {
    sink: Arc<PythonEvalSink>,
    root: Arc<TreePool>,
    query: Arc<QueryContext>,
    ctx: TaskContext,
    gate: Arc<Gate>,
    calls: Arc<AtomicUsize>,
    lens: Arc<std::sync::Mutex<Vec<usize>>>,
}

fn rig(limit: usize, max_rows: usize, window: usize, fail: bool) -> Rig {
    rig_failing_at_once(limit, max_rows, window, fail, None)
}

fn rig_failing_at_once(
    limit: usize,
    max_rows: usize,
    window: usize,
    fail: bool,
    fail_at_once: Option<i64>,
) -> Rig {
    let schema = Schema::new(vec![Field::new("x", ArrowType::Int64, true)]);
    let gate = Arc::new(Gate::default());
    let calls = Arc::new(AtomicUsize::new(0));
    let lens = Arc::new(std::sync::Mutex::new(Vec::new()));
    let call = EvalCall {
        args: vec![Arc::new(Col(0))],
        udf: Arc::new(Fake {
            gate: Arc::clone(&gate),
            calls: Arc::clone(&calls),
            lens: Arc::clone(&lens),
            fail,
            fail_at_once,
        }),
        client: ArrowType::Int64,
        canonical: ArrowType::Int64,
    };
    let sink = PythonEvalSink::new(&schema, vec![call], max_rows, window, None);
    let root = pool(limit);
    let query = QueryContext::new(
        QueryId(59),
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
        "python_eval",
    );
    Rig {
        sink,
        root,
        query,
        ctx,
        gate,
        calls,
        lens,
    }
}

fn rows(n: i64) -> Batch {
    let schema = Arc::new(Schema::new(vec![Field::new("x", ArrowType::Int64, true)]));
    let x: Int64Array = (0..n).map(Some).collect();
    Batch::new(RecordBatch::try_new(schema, vec![Arc::new(x) as ArrayRef]).unwrap())
}

#[tokio::test]
async fn a_full_window_blocks_until_the_oldest_slice_is_done() {
    let r = rig(64 << 20, 2, 2, false);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    // Six rows in slices of two: three slices, over the window of two.
    let SinkStatus::Blocked(wait) = r.sink.sink(rows(6), &mut l, &g).unwrap() else {
        panic!("a full window must block the driver");
    };
    assert_eq!(r.calls.load(Ordering::SeqCst), 3, "one call per slice");
    let waiting = tokio::spawn(wait);
    tokio::task::yield_now().await;
    assert!(
        !waiting.is_finished(),
        "the window freed before any slice finished"
    );
    r.gate.open();
    tokio::time::timeout(Duration::from_secs(5), waiting)
        .await
        .unwrap()
        .unwrap();
    r.sink.combine(l, &g).unwrap();
    r.sink.finalize(&g, &r.ctx).await.unwrap();
    drop((g, r.ctx, r.query));
    assert_eq!(r.root.reserved(), 0, "memory left after the phase");
}

#[tokio::test]
async fn a_failed_slice_behind_a_running_one_frees_a_blocked_driver_with_its_error() {
    // One driver (design P2-T059 §6.3: an ordered input runs one): the
    // oldest slice never finishes, the newer one fails at once. The blocked
    // driver must see the failure, not wait for the oldest slice.
    let r = rig_failing_at_once(64 << 20, 2, 2, false, Some(2));
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    let SinkStatus::Blocked(wait) = r.sink.sink(rows(4), &mut l, &g).unwrap() else {
        panic!("a full window must block the driver");
    };
    tokio::time::timeout(Duration::from_secs(5), wait)
        .await
        .expect("the failure must free the blocked driver");
    let err = r
        .sink
        .sink(rows(1), &mut l, &g)
        .expect_err("the next sink call must return the failure");
    assert!(err.to_string().contains("at once"), "{err}");
    assert_eq!(
        r.calls.load(Ordering::SeqCst),
        2,
        "nothing submitted after the failure"
    );
    drop((l, g, r.ctx, r.query));
    assert_eq!(r.root.reserved(), 0, "memory left after the failure");
}

#[tokio::test]
async fn a_driver_s_rows_go_in_slices_of_exactly_max_records_and_the_rest_at_its_end() {
    // Spark cuts a partition's rows into batches of exactly
    // `maxRecordsPerBatch`, whatever the input batches were, and sends the
    // rest as the last (research/spark-notes/python-udf-batching.md).
    let r = rig(64 << 20, 3, 8, false);
    r.gate.open();
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    for n in [2, 2, 2, 1] {
        assert!(matches!(
            r.sink.sink(rows(n), &mut l, &g).unwrap(),
            SinkStatus::NeedMoreInput
        ));
    }
    assert_eq!(
        *r.lens.lock().unwrap(),
        vec![3, 3],
        "full slices only while input comes"
    );
    r.sink.combine(l, &g).unwrap();
    assert_eq!(
        *r.lens.lock().unwrap(),
        vec![3, 3, 1],
        "the rest at the end of input"
    );
    r.sink.finalize(&g, &r.ctx).await.unwrap();
    drop((g, r.ctx, r.query));
    assert_eq!(r.root.reserved(), 0, "memory left after the phase");
}

#[tokio::test]
async fn a_refused_reservation_submits_nothing_so_the_retry_is_exact() {
    // A pool too small for the input's bytes.
    let small = rig(512, 0, 2, false);
    small.gate.open();
    let g = small.sink.global_state(&small.ctx).unwrap();
    let mut l = small.sink.local_state(&small.ctx, &g).unwrap();
    let status = small.sink.sink(rows(1000), &mut l, &g).unwrap();
    assert!(
        matches!(status, SinkStatus::NeedMemory { .. }),
        "{status:?}"
    );
    assert_eq!(
        small.calls.load(Ordering::SeqCst),
        0,
        "a refused batch must not be submitted"
    );
    drop((l, g, small.ctx, small.query));
    assert_eq!(small.root.reserved(), 0);
    // With room the same batch goes through as one call.
    let big = rig(64 << 20, 0, 2, false);
    big.gate.open();
    let g = big.sink.global_state(&big.ctx).unwrap();
    let mut l = big.sink.local_state(&big.ctx, &g).unwrap();
    let status = big.sink.sink(rows(1000), &mut l, &g).unwrap();
    assert!(matches!(status, SinkStatus::NeedMoreInput), "{status:?}");
    assert_eq!(big.calls.load(Ordering::SeqCst), 1);
    big.sink.combine(l, &g).unwrap();
    big.sink.finalize(&g, &big.ctx).await.unwrap();
    drop((g, big.ctx, big.query));
    assert_eq!(big.root.reserved(), 0);
}

#[tokio::test]
async fn a_failed_call_fails_the_phase_and_releases_its_memory() {
    let r = rig(64 << 20, 4, 8, true);
    r.gate.open();
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    assert!(matches!(
        r.sink.sink(rows(10), &mut l, &g).unwrap(),
        SinkStatus::NeedMoreInput
    ));
    r.sink.combine(l, &g).unwrap();
    assert!(r.sink.finalize(&g, &r.ctx).await.is_err());
    drop((g, r.ctx, r.query));
    assert_eq!(r.root.reserved(), 0, "memory left after a failed phase");
}

#[tokio::test]
async fn a_cancel_while_calls_are_in_flight_releases_everything() {
    // The gate never opens: the calls stay in flight.
    let r = rig(64 << 20, 4, 8, false);
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    assert!(matches!(
        r.sink.sink(rows(10), &mut l, &g).unwrap(),
        SinkStatus::NeedMoreInput
    ));
    assert!(r.root.reserved() > 0, "the in-flight input holds memory");
    r.sink.combine(l, &g).unwrap();
    r.query.cancel();
    let err = tokio::time::timeout(Duration::from_secs(5), r.sink.finalize(&g, &r.ctx))
        .await
        .unwrap()
        .unwrap_err();
    assert_eq!(
        err.error_class(),
        krishiv_common::error::ErrorClass::OperationCanceled
    );
    drop((g, r.ctx, r.query));
    assert_eq!(r.root.reserved(), 0, "memory left after a cancel");
}
