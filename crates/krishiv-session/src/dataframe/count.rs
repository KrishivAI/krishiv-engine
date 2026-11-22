// SPDX-License-Identifier: Apache-2.0
//! `DataFrame::count`, as Spark's `Dataset.count`: an aggregate over the frame that keeps the
//! frame's statement, and the statement a derived frame inherits. A child of `dataframe` so that
//! it reads the frame's private fields and keeps `dataframe.rs` from growing
//! (`cargo xtask file-size`).

use std::sync::{Arc, PoisonError};

use futures::StreamExt;
use krishiv_common::error::KrishivError;

use super::DataFrame;
use crate::session::QueryEnv;

impl DataFrame {
    /// The statement a frame derived from this one runs as: this frame's own analysis's, else the
    /// one it inherited, so a `limit` or `filter` of a `sql` frame shows that SQL in
    /// `system.queries`. Its `system.queries` row marker is its own, not the statement's.
    pub(super) fn statement_for_derived(&self) -> Option<Arc<QueryEnv>> {
        let own = self
            .analysis
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .map(|a| Arc::clone(&a.env));
        let source = own.or_else(|| self.statement.clone())?;
        source.text.as_ref()?;
        let mut env = QueryEnv::clone(&source);
        env.row = Arc::default();
        // A derived frame's unseeded sample draws again, whatever its source drew.
        env.sample_salt = self.session.query_env().sample_salt;
        Some(Arc::new(env))
    }

    /// The number of result rows.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # let batch = arrow_array::RecordBatch::try_new(schema.clone(), vec![std::sync::Arc::new(arrow_array::Int32Array::from(vec![3, 1, 2]))]).unwrap();
    /// # s.register_batches("t", schema, vec![batch])?;
    /// assert_eq!(futures::executor::block_on(s.table("t").count())?, 3);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// As [`Self::collect`].
    pub async fn count(&self) -> Result<u64, KrishivError> {
        // As Spark's `Dataset.count` (`groupBy().count()`): an aggregate over the frame, so column pruning
        // leaves the scan reading no column of its own (a count of a JSON/CSV read of only the corrupt-record
        // column is a count, not Spark's `QUERY_ONLY_CORRUPT_RECORD_COLUMN`).
        let mut frame = self.agg([crate::functions::count_star()]);
        // Still the frame's statement: its text (a run-time error's SQL context, `system.queries`), class,
        // row and span. The count frame is analyzed in the action, as any frame is.
        frame.trace.clone_from(&self.trace);
        frame.statement = self
            .analysis
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .map(|a| Arc::clone(&a.env))
            .or_else(|| self.statement.clone());
        // Spans only for the `sql` frame itself: a derived frame's statement is its parent's text.
        frame.statement_spans = self.statement.is_none();
        let mut stream = frame.execute_stream().await?;
        let mut rows = 0_u64;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }
            let counted = batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow_array::Int64Array>()
                .map(|a| a.value(0));
            let counted = counted.ok_or_else(|| {
                KrishivError::from(krishiv_common::error::InternalError {
                    message: "count(*) did not give a bigint".to_owned(),
                })
            })?;
            rows += u64::try_from(counted).map_err(|_| {
                KrishivError::from(krishiv_common::error::InternalError {
                    message: format!("count(*) gave a negative count {counted}"),
                })
            })?;
        }
        Ok(rows)
    }
}
