// SPDX-License-Identifier: Apache-2.0
//! The result of a running query as an Arrow reader (pyo3-free, so that
//! `tests.rs` can compile it and check cancel-on-release).

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use futures::StreamExt as _;
use krishiv_session::RecordBatchStream;

/// The result of a running query as an Arrow reader. It never touches Python,
/// so a consumer may call it with the GIL held or from a foreign thread.
/// Dropping it drops the engine stream, which cancels the query without
/// waiting for it.
pub(crate) struct QueryReader {
    schema: SchemaRef,
    /// The batch read to learn the schema, handed out first.
    first: Option<RecordBatch>,
    stream: RecordBatchStream,
}

impl Iterator for QueryReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = match self.first.take() {
            Some(batch) => batch,
            None => match futures::executor::block_on(self.stream.next())? {
                Ok(batch) => batch,
                Err(e) => return Some(Err(ArrowError::ExternalError(Box::new(e)))),
            },
        };
        // The schema was fixed by the first batch: a batch that differs is an
        // engine bug, reported instead of exported under the wrong schema.
        Some(RecordBatch::try_new(
            SchemaRef::clone(&self.schema),
            batch.columns().to_vec(),
        ))
    }
}

impl RecordBatchReader for QueryReader {
    fn schema(&self) -> SchemaRef {
        SchemaRef::clone(&self.schema)
    }
}

impl QueryReader {
    /// A reader of `stream` under `schema`; `first` is a batch already taken
    /// from it, handed out first.
    pub(crate) fn new(
        schema: SchemaRef,
        first: Option<RecordBatch>,
        stream: RecordBatchStream,
    ) -> Self {
        Self {
            schema,
            first,
            stream,
        }
    }
}
