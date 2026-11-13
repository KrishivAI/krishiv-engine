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
//! response id (Spark's `ExecuteResponseObserver`). Before its first
//! response the operation is admitted: it reserves its worst case
//! (`aheadBytes` plus, reattachable, the retry buffer) of the server's
//! response budget (`krishiv.connect.maxResponseBytes`) once, and holds it
//! until it is closed; only a new operation waits there, never a running
//! one. The encoder then pushes into the operation's replay buffer
//! (`crate::operation::replay`), each response holding its bytes of the
//! operation's ahead budget until a sender takes it
//! (`krishiv.connect.execute.aheadBytes`) and of the memory pool (consumer
//! `connect_responses`) until it leaves the buffer. So the engine pauses
//! once `aheadBytes` are unread, attached or not (design P3-T011a §3.8), and
//! the result is never held in full. Senders (`crate::sender`) stream the
//! buffer to the client; the gRPC window is their backpressure.
//!
//! Errors end the stream with the gRPC status of [`crate::status`] after the
//! responses before them. An interrupted operation ends with
//! `OPERATION_CANCELED`. Commands answer `CONNECT_INVALID_PLAN` (56K00)
//! until P3-T012.
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
    ConnectInvalidPlanPlanSizeLargerThanMax, InvalidCursorNotReattachable, KrishivError,
    OperationCanceled,
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
use krishiv_memory::ConsumerHandle;
use krishiv_session::{DataFrame, QueryOptions, RecordBatchStream};
use prost::Message;
use tokio::sync::Semaphore;
use tonic::Status;

use self::arrow::EncodeSettings;
use crate::errors::SessionErrors;
use crate::manager::SessionEntry;
use crate::operation::replay::{Held, uuid_text};
use crate::operation::{
    Admitted, ExecuteLimits, Operation, Operations, new_operation_id, plan_digest, random_uuid,
    validate_tags,
};
use crate::sender::{self, SenderCaps};

/// The default of `spark.connect.session.resultChunking.maxChunkSize`
/// (`(0.9 × 128 MiB).toInt`, Spark 4.1+).
const DEFAULT_MAX_CHUNK_SIZE: i64 = 120_795_955;

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

/// The server's budget of response bytes held by all operations
/// (`krishiv.connect.maxResponseBytes`, design P3-T011a §3.8): each
/// operation reserves its worst case once, before its first response, and
/// returns it when it is closed (released, abandoned, its session closed,
/// or a non-reattachable operation's stream ended). Past the cap a new
/// operation waits, cancellably, until one is closed; a running one never
/// waits on it.
#[derive(Debug, Clone)]
pub struct ResponseBudget {
    permits: Arc<Semaphore>,
    cap: usize,
}

impl ResponseBudget {
    /// A budget of `bytes` (at least 1).
    #[must_use]
    pub fn new(bytes: usize) -> Self {
        let cap = bytes.clamp(1, Semaphore::MAX_PERMITS);
        Self {
            permits: Arc::new(Semaphore::new(cap)),
            cap,
        }
    }

    /// Bytes no response holds now.
    #[must_use]
    pub fn available(&self) -> usize {
        self.permits.available_permits()
    }

    /// The whole budget.
    #[must_use]
    pub fn cap(&self) -> usize {
        self.cap
    }
}

/// Removes its operation from the session when dropped, unless kept: the
/// operation ended before its stream started, cancelled (its RPC was
/// dropped) unless [`Registered::fail`] marked it failed. Spark's state for
/// either is **unverified** (its `SparkConnectGetStatusHandler` was not
/// read); this follows design P3-T011a §3.12.
struct Registered {
    operations: Arc<Operations>,
    op: Option<Arc<Operation>>,
}

impl Registered {
    /// The operation runs: it leaves the session when released, abandoned
    /// or (not reattachable) when its stream ends.
    fn keep(mut self) {
        self.op = None;
    }

    /// The setup failed (GetStatus: `FAILED`); the operation is removed.
    fn fail(self) {
        if let Some(op) = &self.op {
            op.fail();
        }
    }
}

impl Drop for Registered {
    fn drop(&mut self) {
        if let Some(op) = self.op.take() {
            // Not failed: interrupted by `close` (GetStatus: CANCELLED).
            op.mark_ended();
            op.close();
            self.operations.remove(&op);
        }
    }
}

/// Starts `request` on the session `entry`; with the id of a live
/// operation running the same plan, attaches to it from the start instead
/// (Spark's `SparkConnectExecutePlanHandler`, v4.3.0 lines 47-60).
///
/// # Errors
/// Before the first response: `INVALID_HANDLE.FORMAT` for an operation id
/// that is not a UUID, `KRISHIV_INVALID_OPERATION_TAG`, the refusals of
/// [`Operations::register`], `INVALID_CURSOR.NOT_REATTACHABLE` for the live
/// id of a non-reattachable operation, the refusals of the engine's
/// operation table (`INVALID_HANDLE.OPERATION_ABANDONED` for an id used in
/// the last 30 minutes), the plan's decoding and translation errors, and
/// analysis errors.
pub async fn execute_plan(
    entry: &Arc<SessionEntry>,
    request: &pb::ExecutePlanRequest,
    limits: &ExecuteLimits,
    budget: &ResponseBudget,
) -> Result<ResponseStream, KrishivError> {
    let id = match &request.operation_id {
        Some(text) => krishiv_session::QueryId::parse(text)?,
        None => new_operation_id()?,
    };
    let tags = validate_tags(&request.tags)?;
    let opts = options(request);
    let plan = request.plan.clone().unwrap_or_default();
    let op = Arc::new(
        Operation::new(
            id.clone(),
            plan_digest(&plan.encode_to_vec()),
            tags,
            opts.reattachable,
        )
        .with_retry_buffer(limits.retry_buffer_size),
    );
    let operations = Arc::clone(entry.operations());
    if let Admitted::Existing(live) = operations.register(&op, limits.max_operations_per_session)? {
        // Spark's `reattachExecuteHolder(holder, observer, None)`.
        if !live.reattachable() {
            return Err(InvalidCursorNotReattachable {}.into());
        }
        let caps = SenderCaps::of(&live, limits);
        return Ok(sender::attach(
            live,
            operations,
            Arc::clone(entry.errors()),
            0,
            caps,
        ));
    }
    let registered = Registered {
        operations: Arc::clone(&operations),
        op: Some(Arc::clone(&op)),
    };
    let session = entry.session();
    let conf = session.conf();
    let setup = async {
        let logical = request_plan(session, &plan, limits).await?;
        let df = session.from_plan_with(&logical, QueryOptions::new().operation_id(id.as_str()))?;
        let schema = df.schema().await?;
        let schema_response = schema::proto_schema(&schema)?;
        Ok::<_, KrishivError>((df, schema, schema_response))
    };
    // A failure is FAILED; this future dropped (its client cancelled the
    // RPC) drops `registered`, which is CANCELLED.
    let (df, schema, schema_response) = match setup.await {
        Ok(ready) => ready,
        Err(e) => {
            registered.fail();
            return Err(e);
        }
    };
    let settings = encode_settings(&conf, limits, opts.chunking);

    let sender = Sender {
        ahead: Arc::new(Semaphore::new(
            limits.ahead_bytes.clamp(1, Semaphore::MAX_PERMITS),
        )),
        ahead_cap: limits.ahead_bytes.clamp(1, Semaphore::MAX_PERMITS),
        budget: budget.clone(),
        footprint: limits.ahead_bytes.saturating_add(if opts.reattachable {
            limits.retry_buffer_size
        } else {
            0
        }),
        consumer: ConsumerHandle::register(session.runtime().memory(), "connect_responses", None),
        op: Arc::clone(&op),
        session_id: request.session_id.clone(),
        server_side_session_id: entry.server_side_id().to_owned(),
        errors: Arc::clone(entry.errors()),
    };
    // Attached before the encoder starts: the operation is never detached
    // before its first stream.
    let stream = sender::attach(
        Arc::clone(&op),
        operations,
        Arc::clone(entry.errors()),
        0,
        SenderCaps::of(&op, limits),
    );
    registered.keep();
    tokio::spawn(encode(sender, df, schema, schema_response, settings));
    Ok(stream)
}

/// The engine plan of a request's `Plan`: a compressed one decompressed
/// (off the async workers, bounded by `maxPlanSize`), then translated with
/// the session's `spark.sql.sources.default`. Shared by `ExecutePlan` and
/// `AnalyzePlan` (Spark's `RequestDecompressionInterceptor` then
/// `SparkConnectPlanner`).
///
/// # Errors
/// The decompression, decoding and translation errors.
pub(crate) async fn request_plan(
    session: &krishiv_session::Session,
    plan: &pb::Plan,
    limits: &ExecuteLimits,
) -> Result<krishiv_plan::Plan, KrishivError> {
    let translate = TranslateConf {
        default_data_source: session
            .conf()
            .get("spark.sql.sources.default")
            .unwrap_or_else(|| TranslateConf::default().default_data_source),
    };
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
    translate_plan(plan, decompressed.as_deref(), &translate)
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

/// Why the encoder stopped early.
enum Stop {
    /// The operation was closed (released, abandoned, its session closed,
    /// or a non-reattachable operation's client went away).
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
    /// The operation's ahead budget (`aheadBytes`).
    ahead: Arc<Semaphore>,
    ahead_cap: usize,
    budget: ResponseBudget,
    /// What the operation reserves of `budget`: `aheadBytes` plus its
    /// retry buffer.
    footprint: usize,
    consumer: Arc<ConsumerHandle>,
    op: Arc<Operation>,
    session_id: String,
    server_side_session_id: String,
    /// The session's errors: a failure ends the stream with an `errorId`.
    errors: Arc<SessionErrors>,
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

    /// Admits the operation: waits until the server's response budget has
    /// its footprint (clamped to the whole budget), held until the
    /// operation is closed.
    async fn admit(&self) -> Result<(), Stop> {
        let want = u32::try_from(self.footprint.clamp(1, self.budget.cap)).unwrap_or(u32::MAX);
        let permit = Arc::clone(&self.budget.permits)
            .acquire_many_owned(want)
            .await
            .map_err(|_| Stop::Gone)?;
        if self.op.replay().admit(permit) {
            Ok(())
        } else {
            Err(Stop::Gone)
        }
    }

    /// Buffers `response` once the ahead budget has room for it (a
    /// response larger than the whole budget waits for all of it) and the
    /// memory pool grants its bytes.
    async fn send(&self, mut response: pb::ExecutePlanResponse) -> Result<(), Stop> {
        let id = random_uuid()?;
        response.response_id = uuid_text(id);
        let size = response.encoded_len();
        let want = |cap: usize| u32::try_from(size.clamp(1, cap)).unwrap_or(u32::MAX);
        let ahead = Arc::clone(&self.ahead)
            .acquire_many_owned(want(self.ahead_cap))
            .await
            .map_err(|_| Stop::Gone)?;
        let mut reservation = self.consumer.reservation();
        reservation
            .grow_without_abort(size)
            .await
            .map_err(|e| Stop::Failed(e.into()))?;
        let held = Held {
            ahead: Some(ahead),
            reservation: Some(reservation),
        };
        if self.op.replay().push(response, id, size, held) {
            Ok(())
        } else {
            Err(Stop::Gone)
        }
    }
}

async fn encode(
    sender: Sender,
    df: DataFrame,
    schema: SchemaRef,
    schema_response: pb::DataType,
    settings: EncodeSettings,
) {
    let op = Arc::clone(&sender.op);
    // A close or an interrupt is seen at once, also while the encoder
    // waits for admission, the query's first batch, a budget or memory.
    let outcome = tokio::select! {
        biased;
        () = op.closed() => Err(Stop::Gone),
        () = op.cancelled() => Err(Stop::Failed(OperationCanceled {}.into())),
        r = produce(&sender, &df, &schema, schema_response, &settings) => r,
    };
    // The query's stream is gone by now (dropped with `produce`).
    drop(df);
    settle(&op, &sender.errors, outcome);
    op.mark_ended();
}

/// Ends `op`'s stream after its encoder stopped with `outcome`. The
/// operation is finished (an interrupt from now on interrupts nothing)
/// before its error is recorded; an interrupt that came first wins, and the
/// stream ends with `OPERATION_CANCELED` instead of the error (Spark's
/// `ExecuteThreadRunner.execute`, v4.3.0 lines 149-180).
fn settle(op: &Operation, errors: &SessionErrors, outcome: Result<(), Stop>) {
    match outcome {
        // `produce` finished the stream after its last response.
        Ok(()) => {}
        Err(Stop::Gone) => {
            op.finish();
            op.replay().finish(None);
        }
        Err(Stop::Failed(e)) => {
            // Failed now, or finished before (its last response failed):
            // the error stands; an interrupt that came first wins.
            let error = if op.fail() || !op.is_cancelled() {
                e
            } else {
                OperationCanceled {}.into()
            };
            op.replay().finish(Some(errors.status(&error)));
        }
    }
}

async fn produce(
    sender: &Sender,
    df: &DataFrame,
    schema: &SchemaRef,
    schema_response: pb::DataType,
    settings: &EncodeSettings,
) -> Result<(), Stop> {
    sender.admit().await?;
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
    // The end is atomic with interrupts: once finished, an interrupt
    // interrupts nothing; an interrupt that came first turns the end into
    // OPERATION_CANCELED (Spark's `ExecuteThreadRunner`, v4.3.0 lines
    // 271-293).
    if !sender.op.finish() {
        return Err(Stop::Failed(OperationCanceled {}.into()));
    }
    if sender.op.reattachable() {
        sender
            .send(sender.response(Some(ResponseType::ResultComplete(ResultComplete {}))))
            .await?;
    }
    sender.op.replay().finish(None);
    Ok(())
}

async fn send_batch(
    sender: &Sender,
    batch: &arrow_array::RecordBatch,
    offset: i64,
    settings: &EncodeSettings,
) -> Result<(), Stop> {
    // `Bytes`: chunks are views of one buffer, and a response kept for a
    // reattach shares its payload with the copy that is sent.
    let bytes = prost::bytes::Bytes::from(arrow::ipc_stream(batch, settings.compression)?);
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
                    data: bytes.slice_ref(chunk),
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
