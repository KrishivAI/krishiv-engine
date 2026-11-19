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
        // A row is never split: one over the cut size would be a message the client refuses with
        // a transport error, so it fails the fetch with a typed error first (design §4.2, m4).
        match cuts(&batch, self.batch_bytes) {
            Ok(cuts) => self.pending = cuts,
            Err(TooLarge) => {
                let e: KrishivError = KrishivFlightSqlRowTooLarge {
                    columns: batch.num_columns().to_string(),
                    limit: self.batch_bytes.to_string(),
                }
                .into();
                return Some((Err(external(e)), self));
            }
        }
        let first = self.pending.pop_front()?;
        Some((Ok(first), self))
    }
}

/// A row larger than the cut size on its own.
struct TooLarge;

/// `batch` in row ranges of at most `limit` bytes each, by each row's own size ([`row_sizes`]);
/// a batch within `limit` (or empty: it carries the schema's columns) stays one. A row over
/// `limit` on its own refuses the batch (design §4.2 m4, review m8; review of 9b66ee1fc,
/// finding 2: a row is measured by what it covers, its list and map elements included).
fn cuts(batch: &RecordBatch, limit: usize) -> Result<VecDeque<RecordBatch>, TooLarge> {
    let sizes = row_sizes(batch);
    let total = sizes.iter().fold(0usize, |a, b| a.saturating_add(*b));
    if total <= limit {
        return Ok(VecDeque::from([batch.clone()]));
    }
    let mut out = VecDeque::new();
    let (mut start, mut held) = (0, 0usize);
    for (row, size) in sizes.iter().enumerate() {
        if *size > limit {
            return Err(TooLarge);
        }
        if row > start && held.saturating_add(*size) > limit {
            out.push_back(batch.slice(start, row - start));
            (start, held) = (row, 0);
        }
        held = held.saturating_add(*size);
    }
    out.push_back(batch.slice(start, batch.num_rows() - start));
    Ok(out)
}

/// The bytes each row of `batch` covers as the encoder sends it. One pass over offsets, no slice
/// per row. Nothing is shared between rows: the encoder hydrates dictionaries
/// ([`FlightDataEncoderBuilder`]'s default `DictionaryHandling::Hydrate`), so a dictionary row
/// goes out as its value.
fn row_sizes(batch: &RecordBatch) -> Vec<usize> {
    let mut sizes = vec![0usize; batch.num_rows()];
    for column in batch.columns() {
        add_sizes(column.as_ref(), &mut sizes);
    }
    sizes
}

/// Adds to `out[i]` the bytes row `i` of `array` covers (`out.len() == array.len()`).
fn add_sizes(array: &dyn Array, out: &mut [usize]) {
    use arrow_array::cast::AsArray;
    use arrow_schema::DataType as D;
    // The validity bitmap is an eighth of a byte a row: rounded into the offsets/values below.
    match array.data_type() {
        D::Utf8 => offsets_bytes(array.as_string::<i32>().value_offsets(), 4, out),
        D::LargeUtf8 => offsets_bytes(array.as_string::<i64>().value_offsets(), 8, out),
        D::Binary => offsets_bytes(array.as_binary::<i32>().value_offsets(), 4, out),
        D::LargeBinary => offsets_bytes(array.as_binary::<i64>().value_offsets(), 8, out),
        D::List(_) => {
            let list = array.as_list::<i32>();
            nested(list.value_offsets(), list.values().as_ref(), 4, out);
        }
        D::LargeList(_) => {
            let list = array.as_list::<i64>();
            nested(list.value_offsets(), list.values().as_ref(), 8, out);
        }
        D::Map(_, _) => {
            let map = array.as_map();
            nested(map.value_offsets(), map.entries(), 4, out);
        }
        D::Struct(_) => {
            for child in array.as_struct().columns() {
                add_sizes(child.as_ref(), out);
            }
        }
        D::FixedSizeList(_, n) => {
            let list = array.as_fixed_size_list();
            let n = usize::try_from(*n).unwrap_or(0);
            if out.is_empty() || n == 0 {
                return;
            }
            // A sliced FixedSizeList slices its values too (arrow-array 60): row 0 starts at 0.
            let values = list.values().slice(0, out.len().saturating_mul(n));
            let mut inner = vec![0usize; values.len()];
            add_sizes(values.as_ref(), &mut inner);
            for (row, size) in out.iter_mut().enumerate() {
                let elements = &inner[row * n..(row + 1) * n];
                *size = elements.iter().fold(*size, |a, b| a.saturating_add(*b));
            }
        }
        D::Dictionary(_, _) => {
            // Hydrated on the wire: a row is its value's bytes (each value measured once).
            let dict = array.as_any_dictionary();
            let values = dict.values();
            let mut value_sizes = vec![0usize; values.len()];
            add_sizes(values.as_ref(), &mut value_sizes);
            let keys = dict.keys();
            for (row, (size, key)) in out.iter_mut().zip(dict.normalized_keys()).enumerate() {
                if keys.is_valid(row) {
                    let value = value_sizes.get(key).copied().unwrap_or(usize::MAX);
                    *size = size.saturating_add(value);
                }
            }
        }
        D::Null => {}
        D::Boolean => out.iter_mut().for_each(|s| *s = s.saturating_add(1)),
        other => match other.primitive_width() {
            Some(width) => out.iter_mut().for_each(|s| *s = s.saturating_add(width)),
            // Views, unions, run-ends: Spark's shapes do not send them; measured row by row.
            None => {
                for (row, size) in out.iter_mut().enumerate() {
                    let one = array.slice(row, 1).to_data().get_slice_memory_size();
                    *size = size.saturating_add(one.unwrap_or(usize::MAX));
                }
            }
        },
    }
}

/// Variable-length values: each row's offset entry and its value bytes.
fn offsets_bytes<O: arrow_array::OffsetSizeTrait>(offsets: &[O], width: usize, out: &mut [usize]) {
    for (row, size) in out.iter_mut().enumerate() {
        let len = offsets[row + 1]
            .as_usize()
            .saturating_sub(offsets[row].as_usize());
        *size = size.saturating_add(width).saturating_add(len);
    }
}

/// A list or map: each row's offset entry and the bytes of the child rows it covers.
fn nested<O: arrow_array::OffsetSizeTrait>(
    offsets: &[O],
    child: &dyn Array,
    width: usize,
    out: &mut [usize],
) {
    let (Some(first), Some(last)) = (offsets.first(), offsets.last()) else {
        return;
    };
    let (first, last) = (first.as_usize(), last.as_usize());
    let covered = child.slice(first, last.saturating_sub(first));
    let mut inner = vec![0usize; covered.len()];
    add_sizes(covered.as_ref(), &mut inner);
    // Prefix sums: a row's child bytes in one subtraction.
    let mut prefix = Vec::with_capacity(inner.len() + 1);
    prefix.push(0usize);
    for size in &inner {
        let next = prefix.last().copied().unwrap_or(0).saturating_add(*size);
        prefix.push(next);
    }
    for (row, size) in out.iter_mut().enumerate() {
        let (a, b) = (
            offsets[row].as_usize().saturating_sub(first),
            offsets[row + 1].as_usize().saturating_sub(first),
        );
        let elements = prefix[b].saturating_sub(prefix[a]);
        *size = size.saturating_add(width).saturating_add(elements);
    }
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

    /// A 100 KB row among 1,000 small ones: `cut` (rows taken as equal) puts it in a cut of
    /// hundreds of rows, which must still be refused (review m8).
    #[tokio::test]
    async fn row_over_cut_limit_inside_a_wide_cut_typed_error() {
        let p = pool(None);
        let n = 1_000;
        let ids = Int64Array::from_iter_values(0..n);
        let s = StringArray::from_iter_values((0..n).map(|i| {
            if i == 5 {
                "x".repeat(100_000)
            } else {
                "s".to_owned()
            }
        }));
        let b = RecordBatch::try_new(schema(), vec![Arc::new(ids), Arc::new(s)]).expect("batch");
        let limits = EncodeLimits {
            batch_bytes: 64 << 10,
            data_bytes: 16 << 10,
        };
        let out = encode_results(schema(), source(vec![Ok(b)]), consumer(&p), limits);
        let last = out.collect::<Vec<_>>().await.pop().expect("a piece");
        let status = last.expect_err("the stream ends with the typed error");
        assert_eq!(
            status
                .metadata()
                .get("x-krishiv-error-class")
                .and_then(|v| v.to_str().ok()),
            Some("KRISHIV_FLIGHT_SQL_ROW_TOO_LARGE")
        );
        assert_eq!(p.reserved(), 0, "the batch's grant released");
    }

    /// The bytes of a cut as `cuts` measures them.
    fn measured(c: &RecordBatch) -> usize {
        row_sizes(c).iter().sum::<usize>()
    }

    /// Ten 30 KB rows ahead of 990 small ones: each cut holds at most the limit, and every row
    /// is kept once, in order.
    #[test]
    fn cuts_follow_skewed_rows_by_their_sizes() {
        let n = 1_000;
        let ids = Int64Array::from_iter_values(0..n);
        let s = StringArray::from_iter_values((0..n).map(|i| {
            if i < 10 {
                "x".repeat(30_000)
            } else {
                "s".to_owned()
            }
        }));
        let b = RecordBatch::try_new(schema(), vec![Arc::new(ids), Arc::new(s)]).expect("batch");
        let limit = 64 << 10;
        let cuts = cuts(&b, limit).ok().expect("no row is over the limit");
        for c in &cuts {
            assert!(measured(c) <= limit, "a cut of {} rows", c.num_rows());
        }
        let first: Vec<i64> = cuts
            .iter()
            .flat_map(|c| {
                c.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .map(|a| a.values().to_vec())
                    .unwrap_or_default()
            })
            .collect();
        assert_eq!(first, (0..n).collect::<Vec<_>>());
    }

    /// `array<bigint>` rows of `width` elements (row 0 `wide` elements when given).
    fn list_batch(rows: usize, width: usize, wide: Option<usize>) -> RecordBatch {
        use arrow_array::builder::{Int64Builder, ListBuilder};
        let mut b = ListBuilder::new(Int64Builder::new());
        for row in 0..rows {
            let n = if row == 0 {
                wide.unwrap_or(width)
            } else {
                width
            };
            for i in 0..n {
                b.values().append_value(i as i64);
            }
            b.append(true);
        }
        let list = b.finish();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "l",
            list.data_type().clone(),
            true,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(list)]).expect("batch")
    }

    /// A list column whose child is several times the limit is cut by rows (each row covers its
    /// own elements, not the whole child), and every row arrives (review of 9b66ee1fc, finding 2).
    #[tokio::test]
    async fn a_list_column_over_the_limit_is_cut_by_rows() {
        let p = pool(None);
        // 20,000 rows of 20 elements: a 3.2 MB child against a 1 MiB cut.
        let b = list_batch(20_000, 20, None);
        let limits = EncodeLimits {
            batch_bytes: 1 << 20,
            data_bytes: 256 << 10,
        };
        let cuts = cuts(&b, limits.batch_bytes).ok().expect("cut");
        assert!(cuts.len() >= 3, "{} cuts", cuts.len());
        assert!(cuts.iter().all(|c| measured(c) <= limits.batch_bytes));
        let schema = b.schema();
        let out = encode_results(schema, source(vec![Ok(b)]), consumer(&p), limits);
        let got = decode(out).await.expect("every row");
        assert_eq!(rows(&got), 20_000);
        assert_eq!(p.reserved(), 0);
    }

    /// One list row over the limit among small ones is still refused.
    #[test]
    fn a_skewed_list_row_is_refused() {
        let b = list_batch(1_000, 2, Some(200_000));
        assert!(cuts(&b, 1 << 20).is_err());
        assert!(cuts(&b.slice(1, 999), 1 << 20).is_ok(), "the others fit");
    }

    /// A map column is measured by its entries as a list is.
    #[test]
    fn a_map_column_over_the_limit_is_cut_by_rows() {
        use arrow_array::builder::{Int64Builder, MapBuilder, StringBuilder};
        let mut m = MapBuilder::new(None, StringBuilder::new(), Int64Builder::new());
        for row in 0..10_000 {
            for k in 0..10 {
                m.keys().append_value(format!("k{k}"));
                m.values().append_value(row * 10 + k);
            }
            m.append(true).expect("row");
        }
        let map = m.finish();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "m",
            map.data_type().clone(),
            true,
        )]));
        let b = RecordBatch::try_new(schema, vec![Arc::new(map)]).expect("batch");
        let limit = 256 << 10;
        assert!(measured(&b) > 3 * limit);
        let cuts = cuts(&b, limit).ok().expect("cut");
        assert!(cuts.iter().all(|c| measured(c) <= limit));
        assert_eq!(
            cuts.iter().map(RecordBatch::num_rows).sum::<usize>(),
            10_000
        );
    }

    /// A vector column (`FixedSizeList<float, n>`, P3-T075) is measured per row, sliced or not.
    #[test]
    fn a_fixed_size_list_column_is_measured_per_row() {
        use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
        let mut b = FixedSizeListBuilder::new(Float32Builder::new(), 8);
        for row in 0..1_000 {
            for i in 0..8 {
                b.values().append_value((row * 8 + i) as f32);
            }
            b.append(true);
        }
        let list = b.finish();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "v",
            list.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(list)]).expect("batch");
        let sizes = row_sizes(&batch);
        assert!(sizes.iter().all(|s| *s == 32));
        let tail = row_sizes(&batch.slice(990, 10));
        assert_eq!(tail, vec![32; 10]);
        let all = cuts(&batch, 3_200).ok().expect("cut");
        assert_eq!(all.len(), 10);
    }

    /// A dictionary column goes out hydrated (the encoder's default), so a row is measured as its
    /// value: 20,000 keys over four 1 KiB strings are cut into pieces of at most the cut size on
    /// the wire, not into one ~20 MB message (review of 40489d958, finding 3).
    #[tokio::test]
    async fn a_dictionary_column_is_cut_by_its_hydrated_rows() {
        use arrow_array::{DictionaryArray, types::Int32Type};
        let values: Vec<String> = (0..4).map(|i| format!("{i:01024}")).collect();
        let keys =
            arrow_array::Int32Array::from_iter((0..20_000).map(|i| (i % 10 != 0).then_some(i % 4)));
        let dict = DictionaryArray::<Int32Type>::try_new(
            keys,
            Arc::new(StringArray::from_iter_values(values)),
        )
        .expect("dict");
        let schema = Arc::new(Schema::new(vec![Field::new(
            "d",
            dict.data_type().clone(),
            true,
        )]));
        let b = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(dict)]).expect("batch");
        let sizes = row_sizes(&b);
        assert_eq!(
            sizes[1],
            4 + 1024,
            "a valid row is its value's offset and bytes"
        );
        assert_eq!(sizes[0], 0, "a null key covers no value");
        let p = pool(None);
        let limits = EncodeLimits {
            batch_bytes: 256 << 10,
            // Larger than any cut: each cut is one message, so its size is the cut's.
            data_bytes: 64 << 20,
        };
        let out = encode_results(schema, source(vec![Ok(b)]), consumer(&p), limits);
        let pieces: Vec<FlightData> = out.map(|p| p.expect("piece")).collect().await;
        assert!(
            pieces.len() > 20,
            "cut into many messages: {}",
            pieces.len()
        );
        for piece in &pieces {
            assert!(
                piece.data_body.len() <= 2 * limits.batch_bytes,
                "message of {} bytes",
                piece.data_body.len()
            );
        }
        let got = decode(stream::iter(pieces.into_iter().map(Ok)).boxed())
            .await
            .expect("decoded");
        assert_eq!(rows(&got), 20_000);
        let nulls: usize = got.iter().map(|b| b.column(0).null_count()).sum();
        assert_eq!(nulls, 2_000);
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
    fn cuts_keep_every_row_once() {
        let b = batch(1_001, 8);
        let bytes = measured(&b);
        let all = cuts(&b, bytes / 7).ok().expect("cut");
        assert!(all.len() >= 7);
        assert_eq!(all.iter().map(RecordBatch::num_rows).sum::<usize>(), 1_001);
        assert_eq!(cuts(&b, bytes).ok().expect("one").len(), 1);
        assert!(cuts(&batch(1, 8), 1).is_err(), "one row over the limit");
        assert_eq!(cuts(&batch(0, 8), 1).ok().expect("empty").len(), 1);
    }
}
