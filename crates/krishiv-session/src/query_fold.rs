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

use crate::session::SessionState;

/// The process counters a query-end fold attributes, and when the query
/// started: read at query start so the fold records the deltas.
pub(crate) struct QueryStart {
    started: Instant,
    arbitration: ArbitrationStats,
    cache_hits: u64,
    cache_misses: u64,
    bytes_read: u64,
    bytes_written: u64,
}

impl QueryStart {
    pub(crate) fn capture(state: &SessionState) -> Self {
        let totals = state.io.totals();
        Self {
            started: Instant::now(),
            arbitration: state.memory_tree.stats(),
            cache_hits: IoCounters::get(&totals.cache_hits),
            cache_misses: IoCounters::get(&totals.cache_misses),
            bytes_read: IoCounters::get(&totals.bytes_read),
            bytes_written: IoCounters::get(&totals.bytes_written),
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
                    // Process-wide deltas: concurrent queries see each other's I/O.
                    bytes_read: i64::try_from(
                        IoCounters::get(&state.io.totals().bytes_read)
                            .saturating_sub(self.bytes_read),
                    )
                    .unwrap_or(i64::MAX),
                    bytes_written: i64::try_from(
                        IoCounters::get(&state.io.totals().bytes_written)
                            .saturating_sub(self.bytes_written),
                    )
                    .unwrap_or(i64::MAX),
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
        let arbitration = state.memory_tree.stats();
        state.metrics.fold(&QueryOutcome {
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
}

impl QueryFold {
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
            self.state.queries.finish(
                self.id,
                &crate::system::Finished {
                    state: "cancelled",
                    wall_ns: 0,
                    peak_memory_bytes: 0,
                    error_class: None,
                    operators: Vec::new(),
                    totals: None,
                },
            );
        }
    }
}

/// The fold of a query whose stream does not exist yet: dropped while still
/// holding it (the caller cancelled), it cancels the query and folds it.
pub(crate) struct PendingFold(pub(crate) Option<QueryFold>);

impl Drop for PendingFold {
    fn drop(&mut self) {
        if let Some(fold) = self.0.take() {
            fold.query.cancel();
            fold.finish(None);
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
                    // Stop the query's other tasks before the fold gives the admission slot
                    // back, so the next query is not admitted beside work still holding memory.
                    fold.query.cancel();
                    fold.finish(Some(&error));
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
            fold.finish(None);
        }
    }
}
