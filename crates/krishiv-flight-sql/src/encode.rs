// SPDX-License-Identifier: Apache-2.0
//! The `DoGet` result stream (design P3-T101 §4.2): a pull stream with no encoder task and no
//! channel. tonic polls it when hyper has send capacity; each poll that finds no encoded piece
//! left takes the next `RecordBatch` from the query and encodes it with arrow-flight's
//! `FlightDataEncoder` (`FlightData` with `data_header` and `data_body`, pieces of at most
//! [`FLIGHT_DATA_BYTES`]).
//!
//! **Bound.** A batch from the query is reserved in the memory pool (consumer
//! [`RESPONSE_CONSUMER`]) when it arrives, and released when the encoder asks for the next one,
//! which it does only once tonic has taken the last piece of the previous one. So outside the
//! engine's own (accounted) operators, Flight holds one batch and its encoded pieces plus
//! hyper's HTTP/2 send window, and a client that stops reading stops the query. A batch over
//! [`FLIGHT_BATCH_BYTES`] is cut by rows first, so one encoded copy is at most that large. A
//! grant the pool refuses fails the stream with the pool's error.
//!
//! The caller passes batches already in Spark's Arrow shapes (Connect's `execute::arrow`); the
//! IPC framing is arrow-flight's.

use std::collections::VecDeque;
use std::sync::Arc;

use arrow_array::{Array, RecordBatch};
use arrow_flight::FlightData;
use arrow_flight::encode::{FlightDataEncoderBuilder, GRPC_TARGET_MAX_FLIGHT_SIZE_BYTES};
use arrow_flight::error::FlightError;
use arrow_schema::SchemaRef;
use futures::stream::{self, BoxStream, StreamExt};
use krishiv_common::error::{InternalError, KrishivError, KrishivFlightSqlRowTooLarge};
use krishiv_memory::{ConsumerHandle, Reservation};
use tonic::Status;

use crate::status::flight_status;

/// The memory consumer the response bytes are reserved under.
pub const RESPONSE_CONSUMER: &str = "flight_sql_responses";

/// The largest batch encoded at once; a larger one is cut by rows first. Connect's
/// `ExecuteLimits::max_batch_size` (128 MiB) is too large for one Flight message.
pub const FLIGHT_BATCH_BYTES: usize = 4 << 20;

/// The largest `FlightData` piece (arrow-flight's default, 2 MiB): under grpc-java's 4 MiB
/// inbound default (the JDBC driver) and ADBC's.
pub const FLIGHT_DATA_BYTES: usize = GRPC_TARGET_MAX_FLIGHT_SIZE_BYTES;

/// How a result stream is cut (the constants above; tests use smaller ones).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeLimits {
    /// [`FLIGHT_BATCH_BYTES`].
    pub batch_bytes: usize,
    /// [`FLIGHT_DATA_BYTES`].
    pub data_bytes: usize,
}

impl Default for EncodeLimits {
    fn default() -> Self {
        Self {
            batch_bytes: FLIGHT_BATCH_BYTES,
            data_bytes: FLIGHT_DATA_BYTES,
        }
    }
}

/// The `DoGet` response of a query: the schema message, then each batch of `batches` as
/// `FlightData` pieces, accounted under `consumer` (registered as [`RESPONSE_CONSUMER`] in the
/// query's pool). An error of `batches` or of the pool ends the stream with its
/// [`flight_status`] after the pieces before it.
pub fn encode_results(
    schema: SchemaRef,
    batches: BoxStream<'static, Result<RecordBatch, KrishivError>>,
    consumer: Arc<ConsumerHandle>,
    limits: EncodeLimits,
) -> BoxStream<'static, Result<FlightData, Status>> {
    let feed = Feed {
        input: batches,
        pending: VecDeque::new(),
        held: None,
        consumer,
        batch_bytes: limits.batch_bytes.max(1),
    };
    let input = stream::unfold(feed, Feed::next).boxed();
    FlightDataEncoderBuilder::new()
        .with_schema(schema)
        .with_max_flight_data_size(limits.data_bytes)
        .build(input)
        .map(|piece| piece.map_err(|e| status_of(&e)))
        .boxed()
}

/// The encoder's input: the query's batches, cut and reserved.
struct Feed {
    input: BoxStream<'static, Result<RecordBatch, KrishivError>>,
    /// The cuts of the current batch not yet given to the encoder.
    pending: VecDeque<RecordBatch>,
    /// The current batch's grant: held until the encoder asks past its last cut.
    held: Option<Reservation>,
    consumer: Arc<ConsumerHandle>,
    batch_bytes: usize,
}

impl Feed {
    async fn next(mut self) -> Option<(Result<RecordBatch, FlightError>, Self)> {
        if let Some(cut) = self.pending.pop_front() {
            return Some((Ok(cut), self));
        }
        // The encoder asks for more only with its queue empty: the previous batch is sent.
        self.held = None;
        let batch = match self.input.next().await? {
            Ok(batch) => batch,
            Err(e) => return Some((Err(external(e)), self)),
        };
        let bytes = batch.get_array_memory_size();
        let mut reservation = self.consumer.reservation();
        if let Err(e) = reservation.grow_without_abort(bytes).await {
            return Some((Err(external(e.into())), self));
        }
        self.held = Some(reservation);
        self.pending = cut(&batch, bytes, self.batch_bytes);
        // A row is never split: one over the cut size would be a message the client refuses with
        // a transport error, so it fails the fetch with a typed error first (design §4.2, m4).
        if let Some(row) = self
            .pending
            .iter()
            .find(|c| c.num_rows() == 1 && slice_bytes(c) > self.batch_bytes)
        {
            let e: KrishivError = KrishivFlightSqlRowTooLarge {
                columns: row.num_columns().to_string(),
                limit: self.batch_bytes.to_string(),
            }
            .into();
            self.pending.clear();
            return Some((Err(external(e)), self));
        }
        let first = self.pending.pop_front()?;
        Some((Ok(first), self))
    }
}

/// `batch` in row ranges of at most about `limit` of its `bytes` (rows taken as equal in
/// size), at least one row each; one empty batch stays one (it carries the schema's columns).
fn cut(batch: &RecordBatch, bytes: usize, limit: usize) -> VecDeque<RecordBatch> {
    let rows = batch.num_rows();
    if bytes <= limit || rows <= 1 {
        return VecDeque::from([batch.clone()]);
    }
    let per_cut = (rows.saturating_mul(limit) / bytes).max(1);
    (0..rows)
        .step_by(per_cut)
        .map(|start| batch.slice(start, per_cut.min(rows - start)))
        .collect()
}

/// The bytes `batch`'s rows hold, counted over its slice only (`get_array_memory_size` counts
/// the whole buffers a slice shares).
fn slice_bytes(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|c| c.to_data().get_slice_memory_size().unwrap_or(usize::MAX))
        .fold(0, usize::saturating_add)
}

fn external(error: KrishivError) -> FlightError {
    FlightError::ExternalError(Box::new(error))
}

/// The status a failed piece ends the stream with: a carried `KrishivError` by
/// [`flight_status`]; anything arrow-flight raised itself is an internal error.
fn status_of(error: &FlightError) -> Status {
    if let FlightError::ExternalError(inner) = error
        && let Some(e) = inner.downcast_ref::<KrishivError>()
    {
        return flight_status(e);
    }
    if let FlightError::Tonic(status) = error {
        return (**status).clone();
    }
    flight_status(
        &InternalError {
            message: format!("Flight SQL result encoding: {error}"),
        }
        .into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, StringArray};
    use arrow_flight::decode::FlightRecordBatchStream;
    use arrow_schema::{DataType, Field, Schema};
    use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
    use tonic::Code;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]))
    }

    fn batch(rows: usize, width: usize) -> RecordBatch {
        let ids = Int64Array::from_iter_values((0..rows).map(|i| i as i64));
        let s = StringArray::from_iter_values((0..rows).map(|i| format!("{i:0width$}")));
        RecordBatch::try_new(schema(), vec![Arc::new(ids), Arc::new(s)]).expect("batch")
    }

    fn pool(limit: Option<usize>) -> Arc<TreePool> {
        TreePool::new(&MemoryConfig {
            limit,
            quantum: 1,
            spill_scratch_reserve: 0,
            ..MemoryConfig::default()
        })
    }

    fn consumer(pool: &Arc<TreePool>) -> Arc<ConsumerHandle> {
        let pool: Arc<dyn MemoryPool> = Arc::clone(pool) as Arc<dyn MemoryPool>;
        ConsumerHandle::register(pool, RESPONSE_CONSUMER, None)
    }

    fn source(
        batches: Vec<Result<RecordBatch, KrishivError>>,
    ) -> BoxStream<'static, Result<RecordBatch, KrishivError>> {
        stream::iter(batches).boxed()
    }

    /// Decodes a response the way a client does.
    async fn decode(
        pieces: BoxStream<'static, Result<FlightData, Status>>,
    ) -> Result<Vec<RecordBatch>, FlightError> {
        let pieces = pieces.map(|p| p.map_err(|s| FlightError::Tonic(Box::new(s))));
        FlightRecordBatchStream::new_from_flight_data(pieces)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect()
    }

    fn rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(RecordBatch::num_rows).sum()
    }

    #[tokio::test]
    async fn round_trips_rows_and_schema() {
        let p = pool(None);
        let out = encode_results(
            schema(),
            source(vec![Ok(batch(10, 4)), Ok(batch(0, 4)), Ok(batch(7, 4))]),
            consumer(&p),
            EncodeLimits::default(),
        );
        let got = decode(out).await.expect("decoded");
        assert_eq!(rows(&got), 17);
        assert!(got.iter().all(|b| b.schema() == schema()));
        assert_eq!(p.reserved(), 0, "every grant released at the end");
    }

    #[tokio::test]
    async fn empty_result_still_sends_the_schema() {
        let p = pool(None);
        let out = encode_results(
            schema(),
            source(vec![]),
            consumer(&p),
            EncodeLimits::default(),
        );
        let pieces: Vec<_> = out.collect().await;
        assert_eq!(pieces.len(), 1, "one schema message");
        let got = decode(stream::iter(pieces).boxed()).await.expect("decoded");
        assert_eq!(rows(&got), 0);
    }

    #[tokio::test]
    async fn outbound_batch_over_client_limit_is_split() {
        let p = pool(None);
        let big = batch(4_000, 200);
        let limits = EncodeLimits {
            batch_bytes: 64 << 10,
            data_bytes: 16 << 10,
        };
        let out = encode_results(schema(), source(vec![Ok(big)]), consumer(&p), limits);
        let pieces: Vec<FlightData> = out.map(|p| p.expect("piece")).collect().await;
        assert!(pieces.len() > 10, "cut into many pieces: {}", pieces.len());
        for piece in &pieces {
            // arrow-flight's target, with slack for the IPC header and padding.
            assert!(
                piece.data_body.len() <= 2 * limits.data_bytes,
                "piece of {} bytes",
                piece.data_body.len()
            );
        }
        let got = decode(stream::iter(pieces.into_iter().map(Ok)).boxed())
            .await
            .expect("decoded");
        assert_eq!(rows(&got), 4_000);
    }

    #[tokio::test]
    async fn row_over_cut_limit_typed_error() {
        let p = pool(None);
        // Small rows around one row of 100 KB: the row is over the 64 KiB cut, its neighbours are not.
        let ids = Int64Array::from_iter_values(0..3);
        let s =
            StringArray::from_iter_values(["a".to_owned(), "x".repeat(100_000), "b".to_owned()]);
        let b = RecordBatch::try_new(schema(), vec![Arc::new(ids), Arc::new(s)]).expect("batch");
        let limits = EncodeLimits {
            batch_bytes: 64 << 10,
            data_bytes: 16 << 10,
        };
        let out = encode_results(schema(), source(vec![Ok(b)]), consumer(&p), limits);
        let last = out.collect::<Vec<_>>().await.pop().expect("a piece");
        let status = last.expect_err("the stream ends with the typed error");
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
        assert_eq!(
            status
                .metadata()
                .get("x-krishiv-error-class")
                .and_then(|v| v.to_str().ok()),
            Some("KRISHIV_FLIGHT_SQL_ROW_TOO_LARGE")
        );
        assert!(
            status.message().contains("2 columns"),
            "{}",
            status.message()
        );
        assert_eq!(p.reserved(), 0, "the batch's grant released");
    }

    #[tokio::test]
    async fn slow_doget_reader_stays_bounded() {
        let p = pool(None);
        let batches: Vec<_> = (0..20).map(|_| Ok(batch(500, 50))).collect();
        let one = batch(500, 50).get_array_memory_size();
        let mut out = encode_results(
            schema(),
            source(batches),
            consumer(&p),
            EncodeLimits::default(),
        );
        let mut pieces = 0;
        while let Some(piece) = out.next().await {
            piece.expect("piece");
            pieces += 1;
            // Between two reads at most the batch being sent is held.
            assert!(p.reserved() <= one, "{} > {one}", p.reserved());
        }
        assert!(pieces > 20);
        assert!(p.peak_reserved() <= one);
        assert_eq!(p.reserved(), 0);
    }

    #[tokio::test]
    async fn a_query_error_ends_the_stream_with_its_status() {
        let p = pool(None);
        let refused: KrishivError = InternalError {
            message: "boom".into(),
        }
        .into();
        let out = encode_results(
            schema(),
            source(vec![Ok(batch(3, 2)), Err(refused)]),
            consumer(&p),
            EncodeLimits::default(),
        );
        let pieces: Vec<_> = out.collect().await;
        let last = pieces.last().expect("pieces");
        let status = last.as_ref().expect_err("ends in an error");
        assert_eq!(status.code(), Code::Internal);
        assert!(
            status
                .metadata()
                .get(crate::status::ERROR_CLASS_KEY)
                .is_some()
        );
        assert!(pieces[..pieces.len() - 1].iter().all(Result::is_ok));
        assert_eq!(p.reserved(), 0);
    }

    #[tokio::test]
    async fn a_refused_grant_fails_the_stream() {
        let p = pool(Some(1_024));
        let out = encode_results(
            schema(),
            source(vec![Ok(batch(2_000, 40))]),
            consumer(&p),
            EncodeLimits::default(),
        );
        let pieces: Vec<_> = out.collect().await;
        let status = pieces
            .iter()
            .find_map(|p| p.as_ref().err())
            .expect("a refusal");
        assert_ne!(status.code(), Code::Ok);
        assert_eq!(p.reserved(), 0);
    }

    #[tokio::test]
    async fn dropping_the_stream_releases_its_grant() {
        let p = pool(None);
        let mut out = encode_results(
            schema(),
            source(vec![Ok(batch(500, 50)), Ok(batch(500, 50))]),
            consumer(&p),
            EncodeLimits::default(),
        );
        out.next().await.expect("schema").expect("ok");
        out.next().await.expect("first batch").expect("ok");
        assert!(p.reserved() > 0);
        drop(out);
        assert_eq!(p.reserved(), 0);
    }

    #[test]
    fn cut_keeps_every_row_once() {
        let b = batch(1_001, 8);
        let bytes = b.get_array_memory_size();
        let cuts = cut(&b, bytes, bytes / 7);
        assert!(cuts.len() >= 7);
        assert_eq!(cuts.iter().map(RecordBatch::num_rows).sum::<usize>(), 1_001);
        assert_eq!(cut(&b, bytes, bytes).len(), 1);
        assert_eq!(cut(&batch(1, 8), 1_000, 1).len(), 1);
        assert_eq!(cut(&batch(0, 8), 1_000, 1).len(), 1);
    }
}
