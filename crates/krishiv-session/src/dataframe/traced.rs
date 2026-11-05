// SPDX-License-Identifier: Apache-2.0
//! The actions of a `DataFrame` that run inside the statement's `krishiv.query`
//! span (P2-T116): `execute_stream` (and so `collect`, `count`, ...) and
//! `collect_with_metrics`. A child of `dataframe` so that it reads the frame's
//! private fields and keeps `dataframe.rs` from growing (`cargo xtask file-size`).

use std::sync::{Arc, PoisonError};

use arrow_array::RecordBatch;
use futures::StreamExt;
use krishiv_common::error::KrishivError;
use krishiv_plan::cache::CacheKey;

use super::DataFrame;
use crate::session::RecordBatchStream;

impl DataFrame {
    pub(super) async fn execute_stream_here(&self) -> Result<RecordBatchStream, KrishivError> {
        use tracing::Instrument as _;
        let span = self.span();
        async {
            // A closed session answers nothing, not even from the result cache.
            self.session.check_open()?;
            // Before the query is analyzed, so an invalidation that runs while it executes keeps
            // its rows out of the result cache.
            let generation = crate::result_cache::global().generation();
            let a = self.action_analysis().await?;
            // The reads were authorized by the analysis: a hit serves only what this principal's
            // own run of the query would return.
            // A frame that asked for a relation cache (`cache()`, `persist()`) runs the query: its
            // snapshot is filled from the run, whatever the shared store holds.
            let result = crate::result_cache::settings(&a.env)
                .filter(|_| self.cache_request.is_none())
                .and_then(|settings| self.session.result_key(&a).map(|key| (settings, key)))
                // A cached relation substituted into the plan makes the rows depend on this
                // session's state: neither served from nor stored into the shared cache.
                .filter(|_| {
                    !self.session.state.cache.would_substitute(&a.analyzed)
                        && self
                            .session
                            .state
                            .cache
                            .lookup(&CacheKey::of(&a.analyzed))
                            .is_none()
                });
            if let Some((settings, key)) = &result
                && let Some((batches, _age)) =
                    crate::result_cache::global().lookup(key, settings.ttl)
            {
                // A hit is a request answered, so it has its row in `system.queries` too; it is not a
                // physical execution, so the metrics registry does not count it.
                let id = self.session.state.queries.record_replayed(
                    a.env.text.as_deref(),
                    a.env.statement_class,
                    self.session.operation_id(),
                );
                *a.env.row.lock().unwrap_or_else(PoisonError::into_inner) = Some(id);
                return Ok(crate::carrier::client_stream(crate::result_cache::replay(
                    batches,
                )));
            }
            let (physical, fill) = self.physical_cached(&a).await?;
            let stream = self.session.run(&physical, &a.env).await?;
            let stream = fill.fill(stream, &self.session.state.cache);
            let stream = match result {
                Some((settings, key)) => {
                    crate::result_cache::fill(stream, key, generation, settings)
                }
                None => stream,
            };
            Ok(crate::carrier::client_stream(stream))
        }
        .instrument(span.clone())
        .await
        .map(|stream| crate::telemetry::in_span(stream, span))
    }

    /// The span of one action: the statement's, or the engine's alone.
    fn span(&self) -> tracing::Span {
        let session_id = &self.session.state.session_id;
        self.trace.as_ref().map_or_else(
            || crate::telemetry::plain_span(session_id),
            |t| t.span(session_id),
        )
    }

    /// `self` with the statement's trace.
    pub(crate) fn with_trace(mut self, trace: crate::telemetry::QueryTrace) -> Self {
        self.trace = Some(Arc::new(trace));
        self
    }

    /// Runs the query and returns its rows with the executor's per-operator
    /// metrics (`EXPLAIN ANALYZE`'s data): rows, batches and time of every
    /// source, operator and sink, summed over the query's drivers. The
    /// metrics are plain data owned by the caller; nothing is registered
    /// globally. Dropping the future cancels the query.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # let batch = arrow_array::RecordBatch::try_new(schema.clone(), vec![std::sync::Arc::new(arrow_array::Int32Array::from(vec![3, 1, 2]))]).unwrap();
    /// # s.register_batches("t", schema, vec![batch])?;
    /// let (rows, metrics) = futures::executor::block_on(s.table("t").collect_with_metrics())?;
    /// assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
    /// assert!(!metrics.operators.is_empty());
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// As [`Self::collect`].
    pub async fn collect_with_metrics(
        &self,
    ) -> Result<(Vec<RecordBatch>, krishiv_exec::QueryMetrics), KrishivError> {
        use tracing::Instrument as _;
        let df = self.clone();
        let span = self.span();
        self.session
            .on_executor(
                async move {
                    let a = df.action_analysis().await?;
                    let (physical, fill) = df.physical_cached(&a).await?;
                    let (stream, query) = df.session.run_with_query(&physical, &a.env).await?;
                    let mut stream = fill.fill(stream, &df.session.state.cache);
                    let mut out = Vec::new();
                    while let Some(batch) = stream.next().await {
                        out.push(crate::carrier::client_batch(&batch?)?);
                    }
                    drop(stream);
                    Ok((out, query.metrics()))
                }
                .instrument(span),
            )
            .await
    }
}
