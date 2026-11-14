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
    #[must_use]
    pub fn with_observed_sink(&self, sink: &ObservedSink) -> Self {
        let mut out = self.clone();
        out.keep.push(Arc::clone(&sink.slot) as _);
        out
    }

    /// The sinks this frame carries.
    pub(super) fn sinks(&self) -> Vec<Arc<SinkSlot>> {
        self.keep
            .iter()
            .filter_map(|k| Arc::clone(k).downcast::<SinkSlot>().ok())
            .collect()
    }

    /// The observation handles this frame carries.
    pub(super) fn observations(&self) -> Vec<Arc<Shared>> {
        self.keep
            .iter()
            .filter_map(|k| Arc::clone(k).downcast::<Shared>().ok())
            .collect()
    }
}

/// The state of an [`ObservedSink`].
pub(crate) type SinkSlot = Mutex<Option<BTreeMap<String, RecordBatch>>>;

/// Where an action leaves the metrics of the observed nodes of its plan, by
/// name (one row each): set when the result was read to its end without an
/// error, replaced by each later action (Spark Connect: the last action
/// wins, a failed query sends none).
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
    /// action has finished without an error.
    #[must_use]
    pub fn take(&self) -> Option<BTreeMap<String, RecordBatch>> {
        self.slot
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }
}

/// `stream`, completing `observers` from `query`'s observations when it ends,
/// fails or is dropped (Spark completes them whether the action succeeded or
/// not); the first action wins.
pub(super) fn deliver(
    stream: RecordBatchStream,
    query: Arc<QueryContext>,
    observers: Vec<Arc<Shared>>,
    sinks: Vec<Arc<SinkSlot>>,
) -> RecordBatchStream {
    if observers.is_empty() && sinks.is_empty() {
        return stream;
    }
    Box::pin(Delivering {
        inner: stream,
        query,
        observers,
        sinks,
        finished: false,
    })
}

struct Delivering {
    inner: RecordBatchStream,
    query: Arc<QueryContext>,
    observers: Vec<Arc<Shared>>,
    sinks: Vec<Arc<SinkSlot>>,
    /// The stream ended without an error.
    finished: bool,
}

impl Delivering {
    fn complete(&mut self) {
        let observers = std::mem::take(&mut self.observers);
        let sinks = std::mem::take(&mut self.sinks);
        if observers.is_empty() && sinks.is_empty() {
            return;
        }
        let observed = self.query.observed();
        if self.finished
            && let Ok(all) = &observed
        {
            for sink in sinks {
                *sink.lock().unwrap_or_else(PoisonError::into_inner) = Some(all.clone());
            }
        }
        let observed = observed.unwrap_or_default();
        for o in observers {
            o.complete(observed.get(o.name()));
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
