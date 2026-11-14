// SPDX-License-Identifier: Apache-2.0
//! `DataFrame::observe` and the delivery of an observed plan's metrics to
//! its [`Observation`] handles when an action's result stream ends (P3-T084).
//! A child of `dataframe` so it reads the frame's private fields.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use futures::Stream;
use krishiv_common::error::KrishivError;
use krishiv_exec::QueryContext;
use krishiv_plan::PlanNode;

use super::DataFrame;
use crate::column::Column;
use crate::observation::{Observation, Shared};
use crate::session::RecordBatchStream;

impl DataFrame {
    /// This frame, observed: `metrics` (aggregate expressions, evaluated as a
    /// global aggregate over the rows that flow through; unaliased ones are
    /// named as Spark names them) are collected as the rows pass, and
    /// `observation` returns them once the first action has finished
    /// (Spark's `Dataset.observe(Observation, ...)`).
    ///
    /// # Errors
    /// `KRISHIV_OBSERVATION_REUSED` when `observation` already observes a
    /// frame. Invalid metrics (a column outside an aggregate, `DISTINCT`, a
    /// window, ...) fail the first action, as analysis errors do.
    pub fn observe(
        &self,
        observation: &Observation,
        metrics: impl IntoIterator<Item = impl Into<Column>>,
    ) -> Result<Self, KrishivError> {
        observation.shared.bind()?;
        let mut out = self.observe_named(observation.name(), metrics);
        out.keep.push(Arc::clone(&observation.shared) as _);
        Ok(out)
    }

    /// As [`Self::observe`] with a name and no handle (Spark's
    /// `Dataset.observe(String, ...)`): the metrics are computed with the
    /// query and reported to whoever reads the query's observations.
    #[doc(hidden)]
    #[must_use]
    pub fn observe_named(
        &self,
        name: &str,
        metrics: impl IntoIterator<Item = impl Into<Column>>,
    ) -> Self {
        self.derive(PlanNode::CollectMetrics {
            input: Arc::clone(&self.plan),
            name: name.to_owned(),
            metrics: metrics
                .into_iter()
                .map(|c| Arc::new(c.into().expr))
                .collect(),
        })
    }

    /// This frame, reporting the metrics of its observed nodes to `sink` when
    /// an action's result has been read to its end (Spark Connect's
    /// `ObservedMetrics`; the server has no [`Observation`] handles).
    ///
    /// For the Connect server: not part of the engine's embedding API.
    #[doc(hidden)]
    #[must_use]
    pub fn with_observed_sink(&self, sink: &ObservedSink) -> Self {
        let mut out = self.clone();
        out.keep.push(Arc::clone(&sink.slot) as _);
        // The analysis made so far did not carry the sink.
        out.analysis = Arc::default();
        out
    }

    /// The `Observation` handles this frame keeps (not its other keep-alive handles, such as the
    /// lease of a script result, which a view must not hold).
    pub(super) fn observation_handles(&self) -> Vec<Arc<dyn std::any::Any + Send + Sync>> {
        self.keep
            .iter()
            .filter(|k| Arc::clone(k).downcast::<Shared>().is_ok())
            .cloned()
            .collect()
    }

    /// The observations and sinks this frame carries, and those of the temporary views it reads
    /// (`through_views`), for the query's env.
    pub(super) fn observers_through(
        &self,
        through_views: Vec<Arc<dyn std::any::Any + Send + Sync>>,
    ) -> Observers {
        let mut out = Observers::default();
        for k in &self.keep {
            if let Ok(shared) = Arc::clone(k).downcast::<Shared>() {
                out.observations.push(shared);
            } else if let Ok(sink) = Arc::clone(k).downcast::<SinkSlot>() {
                out.sinks.push(sink);
            }
        }
        // A view's own sinks belong to the frame that made it, not to this query.
        for k in through_views {
            if let Ok(shared) = k.downcast::<Shared>()
                && !out.observations.iter().any(|o| Arc::ptr_eq(o, &shared))
            {
                out.observations.push(shared);
            }
        }
        out
    }
}

/// What a frame's runs report their observed metrics to.
#[derive(Default)]
pub(crate) struct Observers {
    observations: Vec<Arc<Shared>>,
    sinks: Vec<Arc<SinkSlot>>,
}

/// The state of an [`ObservedSink`].
pub(crate) type SinkSlot = Mutex<Option<BTreeMap<String, Result<RecordBatch, KrishivError>>>>;

/// Where an action leaves the metrics of the observed nodes of its plan, by
/// name (one row each, or the error finalizing that observation raised):
/// set when the result was read to its end without an error, replaced by
/// each later action (Spark Connect: the last action wins, a failed query
/// sends none). For the Connect server.
#[doc(hidden)]
#[derive(Clone, Default)]
pub struct ObservedSink {
    slot: Arc<SinkSlot>,
}

impl std::fmt::Debug for ObservedSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObservedSink").finish_non_exhaustive()
    }
}

impl ObservedSink {
    /// An empty sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The metrics the last successful action left, once; `None` when no
    /// action has finished without an error. An observation whose own
    /// metrics failed to finalize is an `Err` beside the others' rows.
    #[must_use]
    pub fn take(&self) -> Option<BTreeMap<String, Result<RecordBatch, KrishivError>>> {
        self.slot
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }
}

/// `stream`, completing `observers` from `query`'s observations when it ends,
/// fails or is dropped (Spark completes them whether the action succeeded or
/// not); the first action wins. The one hook every run of a query passes.
pub(crate) fn deliver(
    stream: RecordBatchStream,
    query: Arc<QueryContext>,
    observers: &Observers,
) -> RecordBatchStream {
    if observers.observations.is_empty() && observers.sinks.is_empty() {
        return stream;
    }
    Box::pin(Delivering {
        inner: stream,
        query,
        observers: Some((observers.observations.clone(), observers.sinks.clone())),
        finished: false,
    })
}

/// The observations and sinks a stream completes.
type Targets = (Vec<Arc<Shared>>, Vec<Arc<SinkSlot>>);

struct Delivering {
    inner: RecordBatchStream,
    query: Arc<QueryContext>,
    observers: Option<Targets>,
    /// The stream ended without an error.
    finished: bool,
}

impl Delivering {
    fn complete(&mut self) {
        let Some((observations, sinks)) = self.observers.take() else {
            return;
        };
        // On the carriers a caller reads (decimals, intervals), as the action's rows are.
        let observed: BTreeMap<String, Result<RecordBatch, KrishivError>> = self
            .query
            .observed_each()
            .into_iter()
            .map(|(name, row)| {
                let row = row.and_then(|row| crate::carrier::client_batch(&row));
                (name, row)
            })
            .collect();
        if self.finished {
            for sink in sinks {
                *sink.lock().unwrap_or_else(PoisonError::into_inner) = Some(observed.clone());
            }
        }
        for o in observations {
            o.complete(match observed.get(o.name()) {
                Some(Ok(row)) => Ok(Some(row)),
                Some(Err(e)) => Err(e.clone()),
                None => Ok(None),
            });
        }
    }
}

impl Stream for Delivering {
    type Item = Result<RecordBatch, KrishivError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let polled = self.inner.as_mut().poll_next(cx);
        if matches!(polled, Poll::Ready(None)) {
            self.finished = true;
        }
        if matches!(polled, Poll::Ready(None | Some(Err(_)))) {
            self.complete();
        }
        polled
    }
}

impl Drop for Delivering {
    fn drop(&mut self) {
        self.complete();
    }
}
