// SPDX-License-Identifier: Apache-2.0
//! Boxed entry points of [`DataFrame`]: the same futures as its `async fn`s, compiled once in this crate.

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use krishiv_common::error::KrishivError;

use super::DataFrame;
use crate::session::RecordBatchStream;

impl DataFrame {
    /// [`Self::collect`] as a boxed future, compiled once in this crate
    /// ([`Session::sql_boxed`](crate::Session::sql_boxed) says why).
    ///
    /// # Errors
    /// As [`Self::collect`].
    #[inline(never)]
    pub fn collect_boxed(
        &self,
    ) -> krishiv_plan::analyzer::BoxFuture<'_, Result<Vec<RecordBatch>, KrishivError>> {
        Box::pin(self.collect())
    }

    /// [`Self::schema`] as a boxed future, compiled once in this crate.
    ///
    /// # Errors
    /// As [`Self::schema`].
    #[inline(never)]
    pub fn schema_boxed(
        &self,
    ) -> krishiv_plan::analyzer::BoxFuture<'_, Result<SchemaRef, KrishivError>> {
        Box::pin(self.schema())
    }

    /// [`Self::execute_stream`] as a boxed future, compiled once in this crate.
    ///
    /// # Errors
    /// As [`Self::execute_stream`].
    #[inline(never)]
    pub fn execute_stream_boxed(
        &self,
    ) -> krishiv_plan::analyzer::BoxFuture<'_, Result<RecordBatchStream, KrishivError>> {
        Box::pin(self.execute_stream())
    }
}
