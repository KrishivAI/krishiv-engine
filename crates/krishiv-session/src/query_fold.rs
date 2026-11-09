// SPDX-License-Identifier: Apache-2.0
//! The bookkeeping around one running query: the process counters read when it started
//! (`QueryStart`), the one fold of its metrics into the engine's series and `system.queries`
//! when its stream ends (`QueryFold`, `MetricStream`, `PendingFold`), and the history row of a
//! query waiting for admission (`QueuedRow`).
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use futures::Stream;
use futures::stream::BoxStream;
use krishiv_common::error::KrishivError;
use krishiv_exec::{QueryContext, QueryOutcome, Status};
use krishiv_io::IoCounters;
use krishiv_memory::ArbitrationStats;
use krishiv_types::batch::Batch;

use crate::admission::Permit;
use crate::session::SessionState;

/// The counters a query-end fold attributes, and when the query started:
/// read at query start so the fold records the deltas. Bytes read and
/// written come from the query's own I/O context (design P2-E1 R3: the
/// scheduler's totals are the whole runtime's, every session's I/O).
pub(crate) struct QueryStart {
    started: Instant,
    io: krishiv_io::IoContext,
    arbitration: ArbitrationStats,
    cache_hits: u64,
    cache_misses: u64,
    bytes_read: u64,
    bytes_written: u64,
}

impl QueryStart {
    pub(crate) fn capture(state: &SessionState, io: &krishiv_io::IoContext) -> Self {
        let totals = state.io.totals();
        let own = io.counters();
        Self {
            started: Instant::now(),
            io: io.clone(),
            arbitration: state.runtime.memory_tree.stats(),
            cache_hits: IoCounters::get(&totals.cache_hits),
            cache_misses: IoCounters::get(&totals.cache_misses),
            bytes_read: IoCounters::get(&own.bytes_read),
            bytes_written: IoCounters::get(&own.bytes_written),
        }
    }

    /// Folds the query's metrics with the deltas since [`Self::capture`]:
    /// the one fold of this query. `error` is the stream's terminal error,
    /// if any; otherwise the query's own error decides the outcome (none is
    /// success). A zero wall time means the query never finished (a
    /// cancelled stream dropped early): the duration then runs to now.
    fn fold(
        self,
        state: &SessionState,
        query: &QueryContext,
        error: Option<&KrishivError>,
        metrics_on: bool,
        log_id: &str,
    ) {
        let stored = query.error();
        let (status, error_class) = Status::from_error(error.or(stored.as_ref()));
        let metrics = query.metrics();
        let wall = metrics.wall_time;
        // A zero wall time means the query never finished (a cancelled
        // stream dropped early): the duration then runs from query start
        // to the fold.
        let duration = if wall.is_zero() {
            self.started.elapsed()
        } else {
            wall
        };
        state.queries.finish(
            log_id,
            &crate::system::Finished {
                state: match status {
                    Status::Succeeded => "finished",
                    Status::Failed => "failed",
                    Status::Cancelled => "cancelled",
                    Status::Denied => "denied",
                },
                wall_ns: i64::try_from(duration.as_nanos()).unwrap_or(i64::MAX),
                peak_memory_bytes: i64::try_from(metrics.memory.peak_reserved).unwrap_or(i64::MAX),
                error_class: error_class.as_deref(),
                // The most operators one query keeps.
                operators: metrics
                    .operators
                    .iter()
                    .take(500)
                    .map(|(id, e)| crate::system::OperatorRow::of(log_id, *id, e, metrics.workers))
                    .collect(),
                totals: Some(crate::system::Totals {
                    rows_produced: metrics
                        .operators
                        .values()
                        .filter(|e| e.node.is_none() && e.role == krishiv_exec::OperatorRole::Sink)
                        .map(|e| i64::try_from(e.metrics.rows_in).unwrap_or(i64::MAX))
                        .max(),
                    // The query's own counters (its I/O context), since it started.
                    bytes_read: i64::try_from(
                        IoCounters::get(&self.io.counters().bytes_read)
                            .saturating_sub(self.bytes_read),
                    )
                    .unwrap_or(i64::MAX),
                    bytes_written: i64::try_from(
                        IoCounters::get(&self.io.counters().bytes_written)
                            .saturating_sub(self.bytes_written),
                    )
                    .unwrap_or(i64::MAX),
                    rows_read: metrics
                        .operators
                        .values()
                        .filter(|e| {
                            e.role == krishiv_exec::OperatorRole::Source && e.name.ends_with("Scan")
                        })
                        .map(|e| i64::try_from(e.metrics.rows_out).unwrap_or(i64::MAX))
                        .fold(0, i64::saturating_add),
                    rows_written: metrics
                        .operators
                        .values()
                        .filter(|e| e.name == "FileWrite")
                        .map(|e| i64::try_from(e.metrics.rows_in).unwrap_or(i64::MAX))
                        .fold(0, i64::saturating_add),
                    workload_class: query.class().as_str(),
                    cpu_ns: i64::try_from(metrics.drivers.busy.as_nanos()).unwrap_or(i64::MAX),
                    spill_bytes: metrics
                        .operators
                        .values()
                        .map(|e| i64::try_from(e.metrics.spill_write_bytes).unwrap_or(i64::MAX))
                        .fold(0, i64::saturating_add),
                    spill_files: metrics
                        .operators
                        .values()
                        .map(|e| i64::try_from(e.metrics.spill_files).unwrap_or(i64::MAX))
                        .fold(0, i64::saturating_add),
                    pipelines: i64::try_from(metrics.pipelines.len()).unwrap_or(i64::MAX),
                    workers: i64::try_from(metrics.workers).unwrap_or(i64::MAX),
                }),
            },
        );
        if !metrics_on {
            return;
        }
        let totals = state.io.totals();
        let arbitration = state.runtime.memory_tree.stats();
        state.runtime.metrics.fold(&QueryOutcome {
            class: query.class(),
            status,
            error_class: error_class.as_deref(),
            duration_secs: duration.as_secs_f64(),
            metrics: &metrics,
            arbitration_spills: arbitration
                .spill_requests
                .saturating_sub(self.arbitration.spill_requests),
            arbitration_aborts: arbitration.aborts.saturating_sub(self.arbitration.aborts),
            cache_hits: IoCounters::get(&totals.cache_hits).saturating_sub(self.cache_hits),
            cache_misses: IoCounters::get(&totals.cache_misses).saturating_sub(self.cache_misses),
            cancel_latency_secs: query
                .cancelled_at()
                .filter(|_| status == Status::Cancelled)
                .map(|at| at.elapsed().as_secs_f64()),
        });
    }
}

/// The one fold of a running query, taken when its stream ends.
pub(crate) struct QueryFold {
    pub(crate) query: Arc<QueryContext>,
    pub(crate) state: Arc<SessionState>,
    pub(crate) start: QueryStart,
    /// `krishiv.metrics.enabled` as the query started.
    pub(crate) metrics_on: bool,
    /// The query's id in `system.queries`.
    pub(crate) log_id: String,
    /// The admission slot, given back when the fold is consumed (the
    /// stream ended, failed or was dropped).
    pub(crate) _permit: crate::admission::Permit,
    /// The statement's operation id, held until the query's stream ends, so
    /// a running query keeps its id taken even after its DataFrame drops.
    pub(crate) _operation: Option<Arc<crate::history::Operation>>,
}

impl QueryFold {
    /// As [`Self::finish`] for a query that was cancelled (its stream failed or was dropped):
    /// cancelling only signals the drivers, which stop at their next cancellation point, and
    /// blocking closures can still hold memory, so the admission permit is kept until the
    /// query reports every task ended. Releasing it at once would let the next query start
    /// beside work still using the first one's slot and memory.
    pub(crate) fn finish_cancelled(mut self, error: Option<&KrishivError>) {
        let permit = std::mem::replace(&mut self._permit, Permit::unlimited());
        let query = Arc::clone(&self.query);
        let state = Arc::clone(&self.state);
        self.finish(error);
        if query.is_finished() {
            return;
        }
        // Detached: dropping the handle leaves the task running; it ends with the query.
        drop(state.runtime.executor.spawn(
            krishiv_exec::WorkloadClass::Interactive,
            Box::pin(async move {
                query.finished().await;
                drop(permit);
            }),
        ));
    }

    /// As [`Self::finish_cancelled`] for a query dropped while `submit` was still running:
    /// `setup` reports when the submit task is done and whether it started the query.
    pub(crate) fn finish_after_setup(mut self, setup: tokio::sync::oneshot::Receiver<bool>) {
        let permit = std::mem::replace(&mut self._permit, Permit::unlimited());
        let query = Arc::clone(&self.query);
        let state = Arc::clone(&self.state);
        self.finish(None);
        drop(state.runtime.executor.spawn(
            krishiv_exec::WorkloadClass::Interactive,
            Box::pin(async move {
                if setup.await.unwrap_or(false) {
                    query.finished().await;
                }
                drop(permit);
            }),
        ));
    }

    pub(crate) fn finish(self, error: Option<&KrishivError>) {
        self.start.fold(
            &self.state,
            &self.query,
            error,
            self.metrics_on,
            &self.log_id,
        );
    }
}

/// A query's history row while it waits for admission: dropped before the
/// controller answered (the caller gave up), it closes the row `cancelled`.
pub(crate) struct QueuedRow<'a> {
    pub(crate) state: Arc<SessionState>,
    pub(crate) id: &'a str,
    pub(crate) open: bool,
}

impl Drop for QueuedRow<'_> {
    fn drop(&mut self) {
        if self.open {
            self.state.queries.close_unrun(self.id, "cancelled", None);
        }
    }
}

/// The fold of a query whose stream does not exist yet: dropped while still
/// holding it (the caller cancelled), it cancels the query and folds it.
///
/// The second field is set once the run hands `submit` to a task of its own: it reports when that
/// task is done (and whether the query was started), because instantiation keeps running on the
/// blocking pool after the caller is gone.
pub(crate) struct PendingFold(
    pub(crate) Option<QueryFold>,
    pub(crate) Option<tokio::sync::oneshot::Receiver<bool>>,
);

impl Drop for PendingFold {
    fn drop(&mut self) {
        if let Some(fold) = self.0.take() {
            // Dropped before `submit` produced a stream. Instantiation may still be running
            // on the blocking pool (it is not cancelled by dropping the caller), so the
            // permit stays until the submit task is done and, if it started the query, until
            // the query's tasks have ended.
            fold.query.cancel();
            match self.1.take() {
                Some(done) => fold.finish_after_setup(done),
                None => fold.finish(None),
            }
        }
    }
}

/// The result stream with the query's fold attached: the stream's end
/// (batches, the query's error, or a drop) folds the query's metrics once.
pub(crate) struct MetricStream {
    pub(crate) inner: BoxStream<'static, Result<Batch, KrishivError>>,
    pub(crate) fold: Option<QueryFold>,
}

impl Stream for MetricStream {
    type Item = Result<Batch, KrishivError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(None) => {
                if let Some(fold) = self.fold.take() {
                    fold.finish(None);
                }
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(error))) => {
                if let Some(fold) = self.fold.take() {
                    // Stop the query's other tasks, and keep the admission slot until they ended,
                    // so the next query is not admitted beside work still holding memory.
                    fold.query.cancel();
                    fold.finish_cancelled(Some(&error));
                }
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(Some(Ok(batch))) => Poll::Ready(Some(Ok(batch))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for MetricStream {
    fn drop(&mut self) {
        if let Some(fold) = self.fold.take() {
            // The caller went away: cancel first (a recorded error wins),
            // then fold what the drivers merged so far.
            fold.query.cancel();
            fold.finish_cancelled(None);
        }
    }
}

/// A query's `system.queries` row and admission permit, taken before the query changes
/// anything. Settled by the run that takes it ([`Self::into_parts`]), by [`Self::finish_ok`] or
/// [`Self::fail`]; dropped unsettled (the caller was cancelled), it closes the row `cancelled`
/// and gives the slot back.
pub(crate) struct Admitted {
    state: Arc<SessionState>,
    log_id: String,
    permit: Permit,
    armed: bool,
}

impl std::fmt::Debug for Admitted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Admitted")
            .field("log_id", &self.log_id)
            .finish_non_exhaustive()
    }
}

impl Admitted {
    pub(crate) fn new(state: Arc<SessionState>, log_id: String, permit: Permit) -> Self {
        Self {
            state,
            log_id,
            permit,
            armed: true,
        }
    }

    /// The row id and the permit, for the run that now owns them.
    pub(crate) fn into_parts(mut self) -> (String, Permit) {
        self.armed = false;
        (
            std::mem::take(&mut self.log_id),
            std::mem::replace(&mut self.permit, Permit::unlimited()),
        )
    }

    /// Settles the row of a statement that did not run a query and succeeded (an `ignore` write
    /// that found its path).
    pub(crate) fn finish_ok(mut self) {
        self.armed = false;
        self.close("finished", None);
    }

    /// Settles the row with the error that ended the statement before its query ran.
    pub(crate) fn fail(mut self, error: &KrishivError) {
        self.armed = false;
        self.state.queries.record_failed_row(&self.log_id, error);
    }

    fn close(&self, state: &'static str, error_class: Option<&str>) {
        self.state
            .queries
            .close_unrun(&self.log_id, state, error_class);
    }
}

impl Drop for Admitted {
    /// Dropped without a settled outcome: the caller went away, so the row is `cancelled`.
    fn drop(&mut self) {
        if self.armed {
            self.close("cancelled", None);
        }
    }
}
