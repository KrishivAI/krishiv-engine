// SPDX-License-Identifier: Apache-2.0
//! `DataFrame::observe` and the delivery of an observed plan's metrics to
//! its [`Observation`] handles when an action's result stream ends (P3-T084).
//! A child of `dataframe` so it reads the frame's private fields.

use std::pin::Pin;
use std::sync::Arc;
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

    /// The observation handles this frame carries.
    pub(super) fn observations(&self) -> Vec<Arc<Shared>> {
        self.keep
            .iter()
            .filter_map(|k| Arc::clone(k).downcast::<Shared>().ok())
            .collect()
    }
}

/// `stream`, completing `observers` from `query`'s observations when it ends,
/// fails or is dropped (Spark completes them whether the action succeeded or
/// not); the first action wins.
pub(super) fn deliver(
    stream: RecordBatchStream,
    query: Arc<QueryContext>,
    observers: Vec<Arc<Shared>>,
) -> RecordBatchStream {
    if observers.is_empty() {
        return stream;
    }
    Box::pin(Delivering {
        inner: stream,
        query,
        observers,
    })
}

struct Delivering {
    inner: RecordBatchStream,
    query: Arc<QueryContext>,
    observers: Vec<Arc<Shared>>,
}

impl Delivering {
    fn complete(&mut self) {
        let observers = std::mem::take(&mut self.observers);
        if observers.is_empty() {
            return;
        }
        let observed = self.query.observed().unwrap_or_default();
        for o in observers {
            o.complete(observed.get(o.name()));
        }
    }
}

impl Stream for Delivering {
    type Item = Result<RecordBatch, KrishivError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let polled = self.inner.as_mut().poll_next(cx);
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
