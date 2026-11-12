// SPDX-License-Identifier: Apache-2.0
//! `ExecutePlan` for ROOT relations (design P3-T011a §3.2): the plan is
//! decoded (a compressed one through a bounded ZSTD decoder) and translated,
//! the [`Operation`] registered, and an encoder task streams the result:
//!
//! 1. one `schema` response;
//! 2. `arrow_batch` responses (one 0-row batch for an empty result), each a
//!    complete Arrow IPC stream, `row_count` and `start_offset` running from
//!    0 (`execute::arrow`);
//! 3. one `metrics` response;
//! 4. `result_complete`, only for a reattachable execution.
//!
//! Every response carries the session ids, the operation id and a fresh
//! response id (Spark's `ExecuteResponseObserver`). The encoder pushes into
//! a channel bounded in bytes (`krishiv.connect.execute.aheadBytes`, the
//! bytes reserved from the memory pool as consumer `connect_responses`);
//! tonic's response stream drains it, so the gRPC window is the
//! backpressure and the result is never held in full.
//!
//! Errors end the stream with the gRPC status of [`crate::status`]. A
//! cancelled operation (ReleaseExecute `release_all`, ReleaseSession) ends
//! with `OPERATION_CANCELED`. A client that drops the stream cancels the
//! query; in PR-11 the operation is then removed without a tombstone.
//! Commands answer `CONNECT_INVALID_PLAN` (56K00) until P3-T012.
//!
//! The core is protocol-neutral where it is cheap: the plan is decoded into
//! a `krishiv_plan::Plan`, run as a `DataFrame` and encoded as Arrow IPC;
//! only the envelope is Connect's.

pub mod arrow;
pub mod schema;

use std::io::Read;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use futures::StreamExt;
use krishiv_common::error::{
    ConnectInvalidPlanInvalidEnum, ConnectInvalidPlanInvalidOneOfFieldNotSet,
    ConnectInvalidPlanPlanSizeLargerThanMax, KrishivError, OperationCanceled,
};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::execute_plan_request::request_option::RequestOption;
use krishiv_connect::proto::v4_3::execute_plan_response::{
    ArrowBatch, Metrics, ResponseType, ResultComplete,
};
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::plan::compressed_operation::OpType as CompressedOpType;
use krishiv_connect::translate::{TranslateConf, command_not_supported, translate_relation};
use krishiv_connect::wire;
use krishiv_memory::{ConsumerHandle, Reservation};
use krishiv_session::{DataFrame, QueryOptions, RecordBatchStream};
use prost::Message;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tonic::Status;

use self::arrow::EncodeSettings;
use crate::manager::SessionEntry;
use crate::operation::{
    ExecuteLimits, Operation, Operations, new_operation_id, plan_digest, validate_tags,
};
use crate::status::status_of;

/// The default of `spark.connect.session.resultChunking.maxChunkSize`
/// (`(0.9 × 128 MiB).toInt`, Spark 4.1+).
const DEFAULT_MAX_CHUNK_SIZE: i64 = 120_795_955;

/// How long a failed operation waits for room to send its error.
const FAIL_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

/// The smallest preferred chunk size Spark honours.
const MIN_PREFERRED_CHUNK: i64 = 1024;

/// The response stream of one `ExecutePlan`.
pub type ResponseStream =
    futures::stream::BoxStream<'static, Result<pb::ExecutePlanResponse, Status>>;

/// What an `ExecutePlanRequest` asks for besides its plan.
struct Options {
    reattachable: bool,
    /// `Some(preferred)` when the client allows chunking.
    chunking: Option<Option<i64>>,
}

fn options(request: &pb::ExecutePlanRequest) -> Options {
    let mut out = Options {
        reattachable: false,
        chunking: None,
    };
    for option in &request.request_options {
        match &option.request_option {
            Some(RequestOption::ReattachOptions(r)) if r.reattachable => {
                out.reattachable = true;
            }
            Some(RequestOption::ResultChunkingOptions(c)) if c.allow_arrow_batch_chunking => {
                out.chunking = Some(c.preferred_arrow_chunk_size);
            }
            _ => {}
        }
    }
    out
}

/// Removes its operation from the session when dropped.
struct Registered {
    operations: Arc<Operations>,
    op: Arc<Operation>,
}

impl Drop for Registered {
    fn drop(&mut self) {
        self.operations.remove(&self.op);
    }
}

/// Starts `request` on the session `entry`.
///
/// # Errors
/// Before the first response: `INVALID_HANDLE.FORMAT` for an operation id
/// that is not a UUID, `KRISHIV_INVALID_OPERATION_TAG`, the refusals of
/// [`Operations::register`] and of the engine's operation table
/// (`INVALID_HANDLE.OPERATION_ABANDONED` for an id used in the last 30
/// minutes), the plan's decoding and translation errors, and analysis
/// errors.
pub async fn execute_plan(
    entry: &Arc<SessionEntry>,
    request: &pb::ExecutePlanRequest,
    limits: &ExecuteLimits,
) -> Result<ResponseStream, KrishivError> {
    let id = match &request.operation_id {
        Some(text) => krishiv_session::QueryId::parse(text)?,
        None => new_operation_id()?,
    };
    let tags = validate_tags(&request.tags)?;
    let opts = options(request);
    let plan = request.plan.clone().unwrap_or_default();
    let op = Arc::new(Operation::new(
        id.clone(),
        plan_digest(&plan.encode_to_vec()),
        tags,
        opts.reattachable,
    ));
    entry
        .operations()
        .register(&op, limits.max_operations_per_session)?;
    let registered = Registered {
        operations: Arc::clone(entry.operations()),
        op: Arc::clone(&op),
    };
    let session = entry.session();
    let conf = session.conf();
    let translate = TranslateConf {
        default_data_source: conf
            .get("spark.sql.sources.default")
            .unwrap_or_else(|| TranslateConf::default().default_data_source),
    };
    // Decompression (up to `maxPlanSize`) runs off the async workers.
    let decompressed = match &plan.op_type {
        Some(OpType::CompressedOperation(c)) => {
            let c = (**c).clone();
            let max = limits.max_plan_size;
            Some(
                tokio::task::spawn_blocking(move || decompress(&c, max))
                    .await
                    .map_err(|e| -> KrishivError {
                        krishiv_common::error::InternalError {
                            message: format!("decompressing a plan: {e}"),
                        }
                        .into()
                    })??,
            )
        }
        _ => None,
    };
    let logical = translate_plan(&plan, decompressed.as_deref(), &translate)?;
    let df = session.from_plan_with(&logical, QueryOptions::new().operation_id(id.as_str()))?;
    let schema = df.schema().await?;
    let schema_response = schema::proto_schema(&schema)?;
    let settings = encode_settings(&conf, limits, opts.chunking);

    let permits = Arc::new(Semaphore::new(
        limits.ahead_bytes.clamp(1, Semaphore::MAX_PERMITS),
    ));
    let consumer = ConsumerHandle::register(session.runtime().memory(), "connect_responses", None);
    op.set_responses_consumer(&consumer);
    let (tx, rx) = mpsc::channel::<Item>(1024);
    let sender = Sender {
        tx,
        permits,
        cap: limits.ahead_bytes.max(1),
        consumer,
        op: Arc::clone(&op),
        session_id: request.session_id.clone(),
        server_side_session_id: entry.server_side_id().to_owned(),
    };
    tokio::spawn(encode(
        sender,
        df,
        schema,
        schema_response,
        settings,
        registered,
    ));
    Ok(futures::stream::unfold(rx, |mut rx| async move {
        let item = rx.recv().await?;
        Some((item.response, rx))
    })
    .boxed())
}

/// The plan as the engine's IR: a ROOT relation translated, a compressed
/// one decompressed first; commands refused until P3-T012.
fn translate_plan(
    plan: &pb::Plan,
    decompressed: Option<&[u8]>,
    conf: &TranslateConf,
) -> Result<krishiv_plan::Plan, KrishivError> {
    match &plan.op_type {
        Some(OpType::Root(relation)) => translate_relation(relation, conf),
        Some(OpType::Command(command)) => Err(command_not_supported(command)),
        Some(OpType::CompressedOperation(c)) => {
            let bytes = decompressed.unwrap_or_default();
            match CompressedOpType::try_from(c.op_type) {
                Ok(CompressedOpType::Relation) => {
                    translate_relation(&*wire::decode_relation(bytes)?, conf)
                }
                Ok(CompressedOpType::Command) => {
                    // Decoded through the bounded decoder as the `command`
                    // field (2) of a `Plan`, which `wire` can start from.
                    let mut wrapped = Vec::with_capacity(bytes.len() + 11);
                    prost::encoding::encode_key(
                        2,
                        prost::encoding::WireType::LengthDelimited,
                        &mut wrapped,
                    );
                    prost::encoding::encode_varint(bytes.len() as u64, &mut wrapped);
                    wrapped.extend_from_slice(bytes);
                    match &wire::decode_plan(&wrapped)?.op_type {
                        Some(OpType::Command(command)) => Err(command_not_supported(command)),
                        _ => Err(command_not_supported(&pb::Command::default())),
                    }
                }
                _ => Err(ConnectInvalidPlanInvalidOneOfFieldNotSet {
                    full_name: "spark.connect.Plan.CompressedOperation".to_owned(),
                    name: "OP_TYPE_UNSPECIFIED".to_owned(),
                }
                .into()),
            }
        }
        None => Err(ConnectInvalidPlanInvalidOneOfFieldNotSet {
            full_name: "spark.connect.Plan".to_owned(),
            name: "OP_TYPE_NOT_SET".to_owned(),
        }
        .into()),
    }
}

/// Spark's `PlanCompressionUtils.decompressBytesWithZstd`: the frame's
/// declared size is checked first, then the output is read through a bound,
/// so a small input that expands past `max` fails before `max` bytes are
/// held.
fn decompress(c: &pb::plan::CompressedOperation, max: usize) -> Result<Vec<u8>, KrishivError> {
    if pb::CompressionCodec::try_from(c.compression_codec) != Ok(pb::CompressionCodec::Zstd) {
        let name = pb::CompressionCodec::try_from(c.compression_codec)
            .map_or("UNRECOGNIZED", |codec| codec.as_str_name());
        return Err(ConnectInvalidPlanInvalidEnum {
            full_name: "spark.connect.CompressionCodec".to_owned(),
            name: name.to_owned(),
            number: c.compression_codec.to_string(),
        }
        .into());
    }
    let too_large = |size: String| -> KrishivError {
        ConnectInvalidPlanPlanSizeLargerThanMax {
            plan_size: size,
            max_plan_size: max.to_string(),
        }
        .into()
    };
    if let Some(declared) = zstd::zstd_safe::get_frame_content_size(&c.data)
        .ok()
        .flatten()
        .filter(|d| *d > max as u64)
    {
        return Err(too_large(declared.to_string()));
    }
    let cannot_parse = |e: std::io::Error| -> KrishivError {
        krishiv_common::error::ConnectInvalidPlanCannotParse {
            error_msg: e.to_string(),
        }
        .into()
    };
    let decoder =
        zstd::stream::read::Decoder::with_buffer(c.data.as_slice()).map_err(cannot_parse)?;
    let mut out = Vec::new();
    // Spark's `BoundedInputStream` fires once `max` bytes were read and the
    // parser asks for more, so a plan of exactly `max` bytes is refused too.
    decoder
        .take(max as u64)
        .read_to_end(&mut out)
        .map_err(cannot_parse)?;
    if out.len() >= max {
        return Err(too_large("unknown".to_owned()));
    }
    Ok(out)
}

fn encode_settings(
    conf: &krishiv_session::RuntimeConf,
    limits: &ExecuteLimits,
    chunking: Option<Option<i64>>,
) -> EncodeSettings {
    let int = |key: &str| conf.get(key).and_then(|v| v.trim().parse::<i64>().ok());
    let max_rows = int("spark.sql.execution.arrow.maxRecordsPerBatch").unwrap_or(10_000);
    let max_chunk =
        int("spark.connect.session.resultChunking.maxChunkSize").unwrap_or(DEFAULT_MAX_CHUNK_SIZE);
    // Spark: on when the session conf is positive and the request allows it;
    // the preferred size when it lies in [1 KiB, max], else the max.
    let chunk_size = match chunking {
        Some(preferred) if max_chunk > 0 => {
            let size = preferred
                .filter(|p| (MIN_PREFERRED_CHUNK..=max_chunk).contains(p))
                .unwrap_or(max_chunk);
            usize::try_from(size).ok()
        }
        _ => None,
    };
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "Spark's (maxBatchSize * 0.7).toLong on a positive size"
    )]
    let max_bytes = (limits.max_batch_size as f64 * 0.7) as usize;
    EncodeSettings {
        max_rows: usize::try_from(max_rows).ok().filter(|r| *r > 0),
        max_bytes,
        compression: arrow::compression_codec(
            conf.get("spark.sql.execution.arrow.compression.codec")
                .as_deref(),
        ),
        chunk_size,
        time_zone: conf.get("spark.sql.session.timeZone").map(Arc::from),
    }
}

/// One response in the channel, with the bytes it holds of the ahead budget
/// and of the memory pool; both return when tonic takes it.
struct Item {
    response: Result<pb::ExecutePlanResponse, Status>,
    _permit: Option<OwnedSemaphorePermit>,
    _reservation: Option<Reservation>,
}

/// Why the encoder stopped early.
enum Stop {
    /// The client dropped the stream.
    Gone,
    /// The operation failed.
    Failed(KrishivError),
}

impl From<KrishivError> for Stop {
    fn from(e: KrishivError) -> Self {
        Self::Failed(e)
    }
}

/// The producing side of an operation's responses.
struct Sender {
    tx: mpsc::Sender<Item>,
    permits: Arc<Semaphore>,
    cap: usize,
    consumer: Arc<ConsumerHandle>,
    op: Arc<Operation>,
    session_id: String,
    server_side_session_id: String,
}

impl Sender {
    fn response(&self, response_type: Option<ResponseType>) -> pb::ExecutePlanResponse {
        pb::ExecutePlanResponse {
            session_id: self.session_id.clone(),
            server_side_session_id: self.server_side_session_id.clone(),
            operation_id: self.op.id().as_str().to_owned(),
            response_type,
            ..Default::default()
        }
    }

    /// Sends `response` once the budget has room for it (a response larger
    /// than the whole budget waits for all of it).
    async fn send(&self, mut response: pb::ExecutePlanResponse) -> Result<(), Stop> {
        let response_id = new_operation_id()?.as_str().to_owned();
        self.op.record_sent(&response_id);
        response.response_id = response_id;
        let size = response.encoded_len();
        let wanted = u32::try_from(size.clamp(1, self.cap)).unwrap_or(u32::MAX);
        let permit = Arc::clone(&self.permits)
            .acquire_many_owned(wanted)
            .await
            .map_err(|_| Stop::Gone)?;
        let mut reservation = self.consumer.reservation();
        reservation
            .grow_without_abort(size)
            .await
            .map_err(|e| Stop::Failed(e.into()))?;
        self.tx
            .send(Item {
                response: Ok(response),
                _permit: Some(permit),
                _reservation: Some(reservation),
            })
            .await
            .map_err(|_| Stop::Gone)
    }

    /// Ends the stream with `error`, waiting at most [`FAIL_WAIT`] for room
    /// in the channel: a client that reads nothing must not keep the task.
    async fn fail(&self, error: &KrishivError) {
        let send = self.tx.send(Item {
            response: Err(status_of(error)),
            _permit: None,
            _reservation: None,
        });
        let _ = tokio::time::timeout(FAIL_WAIT, send).await;
    }
}

async fn encode(
    sender: Sender,
    df: DataFrame,
    schema: SchemaRef,
    schema_response: pb::DataType,
    settings: EncodeSettings,
    registered: Registered,
) {
    let op = Arc::clone(&sender.op);
    // The client going away is seen at once, also while the encoder waits
    // for the query's first batch or for memory rather than sending.
    let outcome = tokio::select! {
        biased;
        () = op.cancelled() => Err(Stop::Failed(OperationCanceled {}.into())),
        () = sender.tx.closed() => Err(Stop::Gone),
        r = produce(&sender, &df, &schema, schema_response, &settings) => r,
    };
    // The query's stream is gone by now (dropped with `produce`).
    drop(df);
    match outcome {
        Ok(()) | Err(Stop::Gone) => {}
        Err(Stop::Failed(e)) => sender.fail(&e).await,
    }
    drop(registered);
}

async fn produce(
    sender: &Sender,
    df: &DataFrame,
    schema: &SchemaRef,
    schema_response: pb::DataType,
    settings: &EncodeSettings,
) -> Result<(), Stop> {
    let mut first = sender.response(None);
    first.schema = Some(schema_response);
    sender.send(first).await?;
    let mut stream: RecordBatchStream = df.execute_stream().await?;
    let mut offset: i64 = 0;
    let mut sent = false;
    while let Some(batch) = stream.next().await {
        let batch = arrow::spark_batch(&batch?, settings.time_zone.as_ref())?;
        for slice in arrow::slices(&batch, settings) {
            send_batch(sender, &slice, offset, settings).await?;
            offset += i64::try_from(slice.num_rows()).unwrap_or(i64::MAX);
            sent = true;
        }
    }
    drop(stream);
    if !sent {
        let empty = arrow::empty_batch(schema, settings.time_zone.as_ref())?;
        send_batch(sender, &empty, 0, settings).await?;
    }
    let mut metrics = sender.response(None);
    // The operator tree needs a session hook that returns a stream's
    // metrics (a follow-up); until then the list is empty.
    metrics.metrics = Some(Metrics::default());
    sender.send(metrics).await?;
    if sender.op.reattachable() {
        sender
            .send(sender.response(Some(ResponseType::ResultComplete(ResultComplete {}))))
            .await?;
    }
    Ok(())
}

async fn send_batch(
    sender: &Sender,
    batch: &arrow_array::RecordBatch,
    offset: i64,
    settings: &EncodeSettings,
) -> Result<(), Stop> {
    let bytes = arrow::ipc_stream(batch, settings.compression)?;
    let rows = i64::try_from(batch.num_rows()).unwrap_or(i64::MAX);
    match settings.chunk_size {
        None => {
            let response = sender.response(Some(ResponseType::ArrowBatch(ArrowBatch {
                row_count: rows,
                data: bytes,
                start_offset: Some(offset),
                chunk_index: None,
                num_chunks_in_batch: None,
            })));
            sender.send(response).await
        }
        Some(size) => {
            let chunks = arrow::chunks(&bytes, size);
            let n = i64::try_from(chunks.len()).unwrap_or(i64::MAX);
            for (i, chunk) in chunks.into_iter().enumerate() {
                let response = sender.response(Some(ResponseType::ArrowBatch(ArrowBatch {
                    row_count: rows,
                    data: chunk.to_vec(),
                    start_offset: Some(offset),
                    chunk_index: Some(i64::try_from(i).unwrap_or(i64::MAX)),
                    num_chunks_in_batch: Some(n),
                })));
                sender.send(response).await?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests;
