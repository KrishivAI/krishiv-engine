// SPDX-License-Identifier: Apache-2.0
//! The actions of a `DataFrame` that run inside the statement's `krishiv.query`
//! span (P2-T116): `execute_stream` (and so `collect`, `count`, ...) and
//! `collect_with_metrics`. A child of `dataframe` so that it reads the frame's
//! private fields and keeps `dataframe.rs` from growing (`cargo xtask file-size`).

use std::sync::{Arc, PoisonError};

use arrow_array::RecordBatch;
use futures::StreamExt;
use krishiv_common::error::KrishivError;

use super::DataFrame;
use crate::session::RecordBatchStream;

impl DataFrame {
    /// The result stream; with `edge`, on the carriers a caller reads (`carrier.rs`), else on the engine's own.
    pub(super) async fn execute_stream_here(
        &self,
        edge: bool,
    ) -> Result<RecordBatchStream, KrishivError> {
        use tracing::Instrument as _;
        let span = self.span();
        async {
            // A closed session answers nothing, not even from the result cache.
            self.session.check_open()?;
            // Before the query is analyzed, so an invalidation that runs while it executes keeps
            // its rows out of the result cache.
            let generation = self.session.state.runtime.results.generation();
            // The registry first: a `cache()` call before this action counts.
            self.session.register_pending_caches().await;
            let a = self
                .action_analysis()
                .instrument(tracing::info_span!("krishiv.analyze"))
                .await?;
            // The reads were authorized by the analysis: a hit serves only what this principal's
            // own run of the query would return.
            // A plan that holds a registered relation (`cache()`, `persist()`) runs the query: the
            // snapshot is filled from the run, whatever the shared store holds.
            let result = crate::result_cache::settings(&a.env)
                .and_then(|settings| self.session.result_key(&a).map(|key| (settings, key)))
                // A cached relation in the plan makes the rows depend on this session's state:
                // neither served from nor stored into the shared cache.
                .filter(|_| !self.session.state.cache.would_substitute(&a.analyzed));
            if let Some((settings, key)) = &result
                && let Some((batches, _age)) =
                    self.session.state.runtime.results.lookup(key, settings.ttl)
            {
                // A hit is a request answered, so it has its row in `system.queries` too; it is not a
                // physical execution, so the metrics registry does not count it.
                let id = self.session.state.queries.record_replayed(
                    crate::system::TextPolicy::from_config(&a.env.config),
                    a.env.text.as_deref(),
                    a.env.statement_class,
                    self.session.operation_id(),
                );
                *a.env.row.lock().unwrap_or_else(PoisonError::into_inner) = Some(id);
                let replayed = crate::result_cache::replay(batches);
                return Ok(if edge {
                    crate::carrier::client_stream(replayed)
                } else {
                    replayed
                });
            }
            let (physical, fill) = self
                .physical_cached(&a)
                .instrument(tracing::info_span!("krishiv.plan"))
                .await?;
            // The execution span stays open while the rows are produced and read: it closes when
            // the result stream is dropped, not when `run` returns the stream.
            let exec = tracing::info_span!("krishiv.execute");
            let stream = self
                .session
                .run(&physical, &a.env)
                .instrument(exec.clone())
                .await?;
            let stream = crate::telemetry::hold_span(stream, exec);
            let stream = fill.fill(stream, &self.session.state.cache);
            let stream = match result {
                Some((settings, key)) => crate::result_cache::fill(
                    Arc::clone(&self.session.state.runtime.results),
                    stream,
                    key,
                    generation,
                    settings,
                ),
                None => stream,
            };
            Ok(if edge {
                crate::carrier::client_stream(stream)
            } else {
                stream
            })
        }
        .instrument(span.clone())
        .await
        .map(|stream| crate::telemetry::in_span(stream, span))
    }

    /// The span of one action: the statement's, or the engine's alone.
    fn span(&self) -> tracing::Span {
        let who = crate::telemetry::Who::of(&self.session.state);
        self.trace
            .as_ref()
            .map_or_else(|| crate::telemetry::plain_span(&who), |t| t.span(&who))
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
