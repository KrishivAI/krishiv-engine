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
//! `OPERATION_CANCELED`; an encoder that panics ends it with
//! `INTERNAL_ERROR` (Spark's `ExecuteThreadRunner` catches every
//! `Throwable`), so no stream waits on an orphan. A `sql_command`
//! (`spark.sql()`) answers one `sql_command_result` (`execute::sql_command`);
//! the other commands answer `CONNECT_INVALID_PLAN` (56K00) until P3-T012.
//! Every command is checked for its ADR 0046 §3b class first
//! (`crate::auth::check_command`): a caller without the privilege a command
//! needs is refused with 42501. The exception is `ml_command.clean_cache`:
//! PySpark sends it when its process exits, and with no ML cache there is
//! nothing to clean, so it is an operation with no response but
//! `result_complete` (when reattachable). Spark sends an `ml_command_result`
//! whose shape is unverified; Krishiv sends none.
//!
//! The core is protocol-neutral where it is cheap: the plan is decoded into
//! a `krishiv_plan::Plan`, run as a `DataFrame` and encoded as Arrow IPC;
//! only the envelope is Connect's.

pub mod arrow;
mod cache_command;
mod catalog_call;
#[cfg(test)]
pub(crate) mod hooks;
pub mod observed;
pub mod schema;
mod sql_command;
mod write_command;

use std::collections::BTreeMap;
use std::io::Read;
use std::sync::Arc;

use crate::auth::{Check, check_command};
use arrow_schema::SchemaRef;
use futures::StreamExt;
use krishiv_common::error::{
    ConnectInvalidPlanDataframeNotFound, ConnectInvalidPlanInvalidEnum,
    ConnectInvalidPlanInvalidOneOfFieldNotSet, ConnectInvalidPlanNotFoundCachedLocalRelation,
    ConnectInvalidPlanPlanSizeLargerThanMax, InternalError, InvalidCursorNotReattachable,
    KrishivError, OperationCanceled,
};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::execute_plan_request::request_option::RequestOption;
use krishiv_connect::proto::v4_3::execute_plan_response::{
    ArrowBatch, Metrics, ResponseType, ResultComplete,
};
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::plan::compressed_operation::OpType as CompressedOpType;
use krishiv_connect::translate::{
    CachedRemoteIds, Command as RunCommand, LocalRelations, LocalRequest, TranslateConf,
    UdfHost as TranslateHost, cached_remote_table, command_not_supported, translate_command,
    translate_relation_with,
};
use krishiv_connect::wire;
use krishiv_memory::ConsumerHandle;
use krishiv_session::{DataFrame, QueryOptions, RecordBatchStream};
use prost::Message;
use tokio::sync::{Semaphore, TryAcquireError};
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
use crate::udf::InlineUdfs;

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

    /// Waits until `bytes` (clamped to the whole budget, at least 1) are free and holds them
    /// until the returned permit is dropped: another service on the Connect listener (an Arrow
    /// Flight SQL `DoGet`, P3-T101 note §16 M1) shares `krishiv.connect.maxResponseBytes` with
    /// Connect's operations. Cancel-safe: dropping the future gives nothing back because it took
    /// nothing. `None` once the budget is closed (never, today).
    pub async fn reserve(&self, bytes: usize) -> Option<tokio::sync::OwnedSemaphorePermit> {
        let want = u32::try_from(bytes.clamp(1, self.cap)).unwrap_or(u32::MAX);
        Arc::clone(&self.permits)
            .acquire_many_owned(want)
            .await
            .ok()
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

/// The digest of the request's plan, encoded from the decoded request
/// itself (no copy); a deep plan is encoded on a big stack, as it was
/// decoded.
fn request_digest(
    request: &wire::Decoded<pb::ExecutePlanRequest>,
) -> Result<[u8; 32], KrishivError> {
    let digest = || {
        request.plan.as_ref().map_or_else(
            || plan_digest(&[]),
            |plan| plan_digest(&plan.encode_to_vec()),
        )
    };
    if !request.is_deep() {
        return Ok(digest());
    }
    krishiv_common::stack::on_big_stack(digest).ok_or_else(|| {
        krishiv_common::error::InternalError {
            message: "cannot start a thread for a deeply nested plan".to_owned(),
        }
        .into()
    })
}

/// `prepare`, unless the operation is cancelled first: then
/// `OPERATION_CANCELED` and `prepare` is dropped (a release or a closed
/// session also stops a plan that is still being analysed).
async fn until_cancelled<T>(
    op: &Operation,
    prepare: impl std::future::Future<Output = Result<T, KrishivError>>,
) -> Result<T, KrishivError> {
    tokio::select! {
        biased;
        () = op.cancelled() => Err(OperationCanceled {}.into()),
        r = prepare => r,
    }
}

/// The outcome of an operation's preparation. A preparation error settles
/// like a query error, by the state's one compare-and-swap: failed first,
/// the error (GetStatus `FAILED`); an interrupt that reported the operation
/// first, `OPERATION_CANCELED` (GetStatus `CANCELLED`). Either way the
/// operation leaves its session. (A dropped RPC drops `registered`:
/// `CANCELLED`.)
fn settle_preparation<T>(
    op: &Operation,
    registered: Registered,
    prepared: Result<T, KrishivError>,
) -> Result<(T, Registered), KrishivError> {
    match prepared {
        Ok(ready) => Ok((ready, registered)),
        Err(e) if op.fail() => {
            registered.fail();
            Err(e)
        }
        Err(_) => Err(OperationCanceled {}.into()),
    }
}

/// The trace headers of an `ExecutePlan` request (`traceparent`, `baggage`): untrusted text the
/// engine parses and bounds ([`QueryOptions::inbound_trace`]); either may be absent.
#[derive(Debug, Clone, Default)]
pub struct InboundTrace {
    /// The `traceparent` header value.
    pub traceparent: Option<String>,
    /// The `baggage` header value.
    pub baggage: Option<String>,
    /// How the request authenticated and its user token, for the query's delegated catalogs
    /// (P3-T067 §3.3).
    pub credential: Option<(
        krishiv_session::auth::Method,
        Option<krishiv_session::auth::InboundCredential>,
    )>,
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
    request: &wire::Decoded<pb::ExecutePlanRequest>,
    limits: &ExecuteLimits,
    budget: &ResponseBudget,
    inbound: &InboundTrace,
) -> Result<ResponseStream, KrishivError> {
    let id = match &request.operation_id {
        Some(text) => krishiv_session::QueryId::parse(text)?,
        None => new_operation_id()?,
    };
    let tags = validate_tags(&request.tags)?;
    // The request's tags and trace go to the engine's queries of this operation, per request.
    let mut query_options = QueryOptions::new()
        .operation_id(id.as_str())
        .tags(tags.iter())
        .inbound_trace(inbound.traceparent.as_deref(), inbound.baggage.as_deref());
    // This request's own token, not whatever a concurrent request of the session stored.
    if let Some((method, credential)) = &inbound.credential {
        query_options = query_options.inbound_credential(*method, credential.clone());
    }
    let opts = options(request);
    let op = Arc::new(
        Operation::new(
            id.clone(),
            request_digest(request)?,
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
    let empty = pb::Plan::default();
    let plan = request.plan.as_ref().unwrap_or(&empty);
    let conf = session.conf();
    let (work, registered) = if is_ml_clean_cache(plan) {
        (Work::NoResult, registered)
    } else {
        let prepared = until_cancelled(&op, async {
            #[cfg(test)]
            hooks::preparing(id.as_str()).await;
            // `spark.sql()`: a command runs now, a query is analyzed; neither returns rows.
            if let Some(command) = sql_command::of(plan, limits).await? {
                // The command's ADR 0046 §3b class is checked before it runs, as for any command.
                let wrapped = pb::Command {
                    command_type: Some(pb::command::CommandType::SqlCommand(Box::new(command))),
                };
                crate::auth::enforce(&check_command(&wrapped), session).await?;
                let Some(pb::command::CommandType::SqlCommand(command)) = wrapped.command_type
                else {
                    return Err(InternalError {
                        message: "a sql_command lost its variant".to_owned(),
                    }
                    .into());
                };
                let answer = sql_command::run(
                    session,
                    &command,
                    query_options.clone(),
                    limits.max_batch_size,
                )
                .await?;
                return Ok(Prepared::Command(answer));
            }
            // Decompression (up to `maxPlanSize`) runs off the async workers.
            match Box::pin(execution_plan(entry, plan, limits, &query_options)).await? {
                Executed::Relation(requested) => {
                    let df = requested.frame(session, query_options).await?;
                    let schema = df.schema().await?;
                    let schema_response = schema::proto_schema(&schema)?;
                    Ok(Prepared::Rows(
                        df,
                        schema,
                        schema_response,
                        observed::plan_ids(requested.plan.root()),
                    ))
                }
                Executed::Checkpointed(id) => Ok(Prepared::Checkpoint(id)),
                // A command that ran (`register_function`): no response but the end.
                Executed::Command => Ok(Prepared::Done),
            }
        })
        .await;
        match settle_preparation(&op, registered, prepared)? {
            (Prepared::Rows(df, schema, schema_response, plan_ids), registered) => {
                let query = Query {
                    df,
                    schema,
                    schema_response,
                    settings: encode_settings(&conf, limits, opts.chunking),
                    plan_ids,
                };
                (Work::Query(Box::new(query)), registered)
            }
            (Prepared::Command(answer), registered) => {
                (Work::Command(Box::new(answer)), registered)
            }
            (Prepared::Checkpoint(id), registered) => (Work::Checkpoint(id), registered),
            (Prepared::Done, registered) => (Work::NoResult, registered),
        }
    };

    let sender = new_sender(session, request, entry, &op, (limits, budget, &opts));
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
    tokio::spawn(encode(sender, work));
    Ok(stream)
}

/// A request's plan with the DataFrames its cached remote relations name (`checkpoint_command`
/// results the session holds), looked up at translation as Spark's `transformCachedRemoteRelation`
/// does. The frames are held here, so a `remove_cached_remote_relation_command` that arrives
/// while the request runs frees nothing it reads.
pub(crate) struct Requested {
    /// The translated plan; a cached remote relation is a read of the reserved table.
    pub(crate) plan: krishiv_plan::Plan,
    frames: Vec<(Vec<String>, DataFrame)>,
}

impl Requested {
    /// `plan` with the frames of `ids` bound.
    ///
    /// # Errors
    /// `CONNECT_INVALID_PLAN.DATAFRAME_NOT_FOUND` for an id the session does not hold.
    fn bind(
        entry: &SessionEntry,
        plan: krishiv_plan::Plan,
        ids: &[String],
    ) -> Result<Self, KrishivError> {
        let mut frames = Vec::with_capacity(ids.len());
        for id in ids {
            let Some(frame) = entry.cached_frame(id) else {
                return Err(ConnectInvalidPlanDataframeNotFound {
                    df_id: id.clone(),
                    session_id: entry.client_session_id().to_owned(),
                }
                .into());
            };
            frames.push((cached_remote_table(id), frame));
        }
        Ok(Self { plan, frames })
    }

    /// The plan as a `DataFrame` of `session`, the frames spliced in.
    ///
    /// # Errors
    /// `Session::from_plan_over`'s.
    pub(crate) async fn frame(
        &self,
        session: &krishiv_session::Session,
        options: QueryOptions,
    ) -> Result<DataFrame, KrishivError> {
        session
            .from_plan_over(&self.plan, options, &self.frames)
            .await
    }
}

/// What a request's plan prepared.
enum Prepared {
    /// A relation: its frame, schema, schema response and the plan ids of its observed nodes,
    /// streamed as batches.
    Rows(DataFrame, SchemaRef, pb::DataType, BTreeMap<String, i64>),
    /// A `sql_command`: the relation it answers.
    Command(sql_command::Answer),
    /// A command that ran (`register_function`): no response but the end.
    Done,
    /// A `checkpoint_command`: the id the result frame is held under.
    Checkpoint(String),
}

/// The producing side of `op`.
fn new_sender(
    session: &krishiv_session::Session,
    request: &wire::Decoded<pb::ExecutePlanRequest>,
    entry: &SessionEntry,
    op: &Arc<Operation>,
    (limits, budget, opts): (&ExecuteLimits, &ResponseBudget, &Options),
) -> Sender {
    Sender {
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
        op: Arc::clone(op),
        session_id: request.session_id.clone(),
        server_side_session_id: entry.server_side_id().to_owned(),
        errors: Arc::clone(entry.errors()),
    }
}

/// Whether `plan` is exactly `Command{ml_command{clean_cache}}`, the ML
/// cache cleanup PySpark 4.3.0 sends at exit (`_cleanup_ml_cache`, before
/// `release_session`): answered with nothing to clean (connect-mapping §4
/// row 17, decision 11). Every other command, and this one compressed, stays
/// 56K00.
///
/// The shortcut skips the command path, so it also requires the command's ADR
/// 0046 §3b class to be [`Check::OwnSession`] (`check_command`): a class change
/// sends it down the ordinary path, with its check, instead of past it.
fn is_ml_clean_cache(plan: &pb::Plan) -> bool {
    use krishiv_connect::proto::v4_3::command::CommandType;
    use krishiv_connect::proto::v4_3::ml_command::Command as MlCommand;
    matches!(
        &plan.op_type,
        Some(OpType::Command(command)) if matches!(
            &command.command_type,
            Some(CommandType::MlCommand(ml))
                if matches!(ml.command, Some(MlCommand::CleanCache(_)))
        ) && check_command(command) == Check::OwnSession
    )
}

/// What [`translate_plan`] made of a plan.
enum Translated {
    /// A ROOT relation as the engine's IR.
    Relation(krishiv_plan::Plan),
    /// A command (from a compressed operation too). The proto itself is
    /// not kept: it stays in the scrubbed request.
    Command(Box<TranslatedCommand>),
}

/// A command: its ADR 0046 §3b class, what `ExecutePlan` runs
/// (`register_function`; every other command is refused until P3-T012) and
/// the refusal `AnalyzePlan` gives.
struct TranslatedCommand {
    /// The command's class.
    check: Check,
    /// The command as translated, or why it is not run.
    run: Result<RunCommand, KrishivError>,
    /// `CONNECT_INVALID_PLAN` naming the command.
    refusal: KrishivError,
    /// The relation a `checkpoint_command` checkpoints, translated with the plan's host.
    input: Option<Result<krishiv_plan::Plan, KrishivError>>,
}

impl Translated {
    fn command(
        command: &pb::Command,
        conf: &TranslateConf,
        udfs: &mut dyn TranslateHost,
    ) -> Result<Self, KrishivError> {
        let relation = match &command.command_type {
            Some(pb::command::CommandType::CheckpointCommand(c)) => Some(c.relation.as_ref()),
            Some(pb::command::CommandType::WriteOperation(w)) => Some(w.input.as_ref()),
            Some(pb::command::CommandType::WriteOperationV2(w)) => Some(w.input.as_ref()),
            _ => None,
        };
        let empty = pb::Relation::default();
        // The error waits for the command's class check: a viewer is refused first.
        let input = relation
            .map(|relation| translate_relation_with(relation.unwrap_or(&empty), conf, udfs));
        Ok(Self::Command(Box::new(TranslatedCommand {
            check: check_command(command),
            run: translate_command(command),
            refusal: command_not_supported(command),
            input,
        })))
    }
}

/// What an `ExecutePlan` request's plan is, once its privileges are checked.
enum Executed {
    /// A ROOT relation to run; its inline Python UDFs are registered.
    Relation(Requested),
    /// A `checkpoint_command` that ran: the id its result is held under.
    Checkpointed(String),
    /// A command that ran (`register_function`, `remove_cached_remote_relation_command`).
    Command,
}

/// The session's `spark.sql.sources.default` for translation.
fn translate_conf(session: &krishiv_session::Session) -> TranslateConf {
    TranslateConf {
        default_data_source: session
            .conf()
            .get("spark.sql.sources.default")
            .unwrap_or_else(|| TranslateConf::default().default_data_source),
        parser: session.parser_options(),
        local_relations: LocalRelations::default(),
        cached_remote: CachedRemoteIds::default(),
    }
}

/// The plan of an `ExecutePlan` request, after its checks: a command's ADR
/// 0046 §3b class is enforced before it runs; a relation with a Python
/// UDF (an inline call, or a `mapInPandas`/`applyInPandas` relation's)
/// needs `Usage` on `System("udf")` before its UDFs are
/// registered with the session's workers (ADR 0046 §3b, N1: fail closed,
/// nothing is registered or started for a caller without it).
///
/// # Errors
/// The decompression, decoding and translation errors, the privilege
/// refusals (`KRISHIV_INSUFFICIENT_PERMISSIONS`), a command's errors, and
/// the refusal of a command that is not run yet.
async fn execution_plan(
    entry: &SessionEntry,
    plan: &pb::Plan,
    limits: &ExecuteLimits,
    query_options: &QueryOptions,
) -> Result<Executed, KrishivError> {
    let session = entry.session();
    let decompressed = decompressed(plan, limits).await?;
    let mut udfs = InlineUdfs::execution(entry.python());
    let (translated, cached) =
        translate_locals(entry, Input::Plan(plan), decompressed.as_deref(), &mut udfs).await?;
    match translated {
        Translated::Relation(logical) => {
            if udfs.seen() {
                crate::auth::enforce(&crate::auth::client_code(), session).await?;
                udfs.grant(session)?;
            }
            Ok(Executed::Relation(Requested::bind(
                entry, logical, &cached,
            )?))
        }
        Translated::Command(command) => {
            let TranslatedCommand {
                check, run, input, ..
            } = *command;
            crate::auth::enforce(&check, session).await?;
            match run? {
                RunCommand::RegisterPythonUdf(udf) => {
                    entry.python().register(session, udf)?;
                }
                RunCommand::Checkpoint(spec) => {
                    // No operation id or tags on the source: the result is held by the session
                    // for good, and its frame would hold them with it.
                    let source =
                        command_source(entry, input, udfs, &cached, QueryOptions::new()).await?;
                    let id = cache_command::checkpoint(entry, &source, spec).await?;
                    return Ok(Executed::Checkpointed(id));
                }
                RunCommand::RemoveCachedRemoteRelation { relation_id } => {
                    entry.remove_cached_frame(&relation_id);
                }
                RunCommand::Write(spec) => {
                    // A one-shot statement of this operation: its operation id, tags, trace and
                    // credential, never the session's latest token.
                    let source =
                        command_source(entry, input, udfs, &cached, query_options.clone()).await?;
                    Box::pin(write_command::write(session, &source, spec)).await?;
                }
                RunCommand::WriteV2(spec) => {
                    let source =
                        command_source(entry, input, udfs, &cached, query_options.clone()).await?;
                    Box::pin(write_command::write_v2(&source, spec)).await?;
                }
            }
            Ok(Executed::Command)
        }
    }
}

/// The frame of a command's relation (`checkpoint_command`, the writes), once the command's class
/// is checked: a relation with a Python UDF needs `Usage` on `System("udf")` before its UDFs are
/// registered (ADR 0046 §3b, N1), as for a ROOT relation. `options` are the frame's: the request's
/// for a write, empty for a checkpoint.
///
/// # Errors
/// The relation's translation errors, the privilege refusal, and binding errors.
async fn command_source(
    entry: &SessionEntry,
    input: Option<Result<krishiv_plan::Plan, KrishivError>>,
    udfs: InlineUdfs<'_>,
    cached: &[String],
    options: QueryOptions,
) -> Result<DataFrame, KrishivError> {
    let session = entry.session();
    let Some(input) = input else {
        return Err(krishiv_common::error::InternalError {
            message: "a command lost its relation".to_owned(),
        }
        .into());
    };
    let input = input?;
    if udfs.seen() {
        crate::auth::enforce(&crate::auth::client_code(), session).await?;
        udfs.grant(session)?;
    }
    Requested::bind(entry, input, cached)?
        .frame(session, options)
        .await
}

/// The engine plan of a request's `Plan`: a compressed one decompressed
/// (off the async workers, bounded by `maxPlanSize`), then translated with
/// the session's `spark.sql.sources.default`. Shared by `ExecutePlan` and
/// `AnalyzePlan` (Spark's `RequestDecompressionInterceptor` then
/// `SparkConnectPlanner`). A command gets its class checked, then its refusal (until
/// P3-T012 runs commands).
///
/// # Errors
/// The decompression, decoding and translation errors.
pub(crate) async fn request_plan(
    entry: &SessionEntry,
    plan: &pb::Plan,
    limits: &ExecuteLimits,
) -> Result<Requested, KrishivError> {
    let session = entry.session();
    let decompressed = decompressed(plan, limits).await?;
    // Inline Python UDFs are typed, never registered: analysis runs no client code.
    let mut udfs = InlineUdfs::analysis(entry.python());
    let (translated, cached) =
        translate_locals(entry, Input::Plan(plan), decompressed.as_deref(), &mut udfs).await?;
    match translated {
        Translated::Relation(logical) => Requested::bind(entry, logical, &cached),
        // A command's ADR 0046 §3b class is checked before it is refused, on every RPC that
        // takes a plan.
        Translated::Command(command) => {
            crate::auth::enforce(&command.check, session).await?;
            Err(command.refusal)
        }
    }
}

/// [`request_plan`] for a relation the request carries as such (`AnalyzePlan`'s `persist`,
/// `unpersist` and `get_storage_level`, which Spark reads with `transformRelation`).
///
/// # Errors
/// The translation errors, and `CONNECT_INVALID_PLAN.DATAFRAME_NOT_FOUND`.
pub(crate) async fn request_relation(
    entry: &SessionEntry,
    relation: &pb::Relation,
) -> Result<Requested, KrishivError> {
    let mut udfs = InlineUdfs::analysis(entry.python());
    let (translated, cached) =
        translate_locals(entry, Input::Relation(relation), None, &mut udfs).await?;
    match translated {
        Translated::Relation(logical) => Requested::bind(entry, logical, &cached),
        Translated::Command(_) => Err(krishiv_common::error::InternalError {
            message: "a relation translated to a command".to_owned(),
        }
        .into()),
    }
}

/// What a request translates: a `Plan` (a relation or a command) or a bare `Relation`.
#[derive(Clone, Copy)]
enum Input<'a> {
    Plan(&'a pb::Plan),
    Relation(&'a pb::Relation),
}

/// [`translate_plan`] or the translation of a bare relation.
fn translate_input(
    input: Input<'_>,
    decompressed: Option<&[u8]>,
    conf: &TranslateConf,
    udfs: &mut dyn TranslateHost,
) -> Result<Translated, KrishivError> {
    match input {
        Input::Plan(plan) => translate_plan(plan, decompressed, conf, udfs),
        Input::Relation(relation) => {
            translate_relation_with(relation, conf, udfs).map(Translated::Relation)
        }
    }
}

/// A compressed operation's bytes, decompressed off the async workers
/// (bounded by `maxPlanSize`); `None` for any other plan.
async fn decompressed(
    plan: &pb::Plan,
    limits: &ExecuteLimits,
) -> Result<Option<Vec<u8>>, KrishivError> {
    Ok(match &plan.op_type {
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
    })
}

/// The plan translated, with its local relations (`createDataFrame`) built by an async pre-pass, as
/// the translator is synchronous: a first translation records what the plan needs, the relations are
/// built, and a second translation uses them. A plan without any is translated once.
async fn translate_locals(
    entry: &SessionEntry,
    plan: Input<'_>,
    decompressed: Option<&[u8]>,
    udfs: &mut dyn TranslateHost,
) -> Result<(Translated, Vec<String>), KrishivError> {
    let translate = translate_conf(entry.session());
    let cached = translate.cached_remote.clone();
    let collecting = LocalRelations::collecting();
    let first = TranslateConf {
        local_relations: collecting.clone(),
        ..translate.clone()
    };
    let translated = translate_input(plan, decompressed, &first, udfs)?;
    // Drained, not copied: the recorded inline data is the one copy, and each request is consumed
    // (and freed) by its build below.
    let requests = collecting.take_requests();
    drop(first);
    drop(collecting);
    let has_plan = match &translated {
        Translated::Relation(_) => true,
        Translated::Command(c) => matches!(c.input, Some(Ok(_))),
    };
    if requests.is_empty() || !has_plan {
        return Ok((translated, cached.take()));
    }
    drop(translated);
    let mut built = Vec::with_capacity(requests.len());
    for request in requests {
        built.push(build_local_relation(entry, request).await?);
    }
    let second = TranslateConf {
        local_relations: LocalRelations::resolved(built),
        ..translate
    };
    let translated = translate_input(plan, decompressed, &second, udfs)?;
    Ok((translated, cached.take()))
}

/// The relation of one request of the translator's first pass: an inline one through
/// `Session::local_relation` (the IPC data decoded, a user schema applied eagerly, the batches
/// held once in the session's relation store); a cached one through the artifact store.
///
/// The artifact store is not on this branch (T011/T015b), so no hash is ever found:
/// `CONNECT_INVALID_PLAN.NOT_FOUND_CACHED_LOCAL_RELATION`, as Spark's block read of an unknown
/// `cache/<hash>` (golden-pending: Spark's `CANNOT_FIND_CACHED_LOCAL_RELATION` for a hash the
/// artifact manager never registered).
async fn build_local_relation(
    entry: &SessionEntry,
    request: LocalRequest,
) -> Result<krishiv_plan::LocalRelation, KrishivError> {
    match request {
        LocalRequest::Inline { data, schema } => {
            entry
                .session()
                .local_relation(data.as_deref(), schema)
                .await
        }
        LocalRequest::Catalog(call) => catalog_call::run(entry.session(), call).await,
        LocalRequest::Cached { hash } => Err(ConnectInvalidPlanNotFoundCachedLocalRelation {
            hash,
            session_u_u_i_d: entry.server_side_id().to_owned(),
        }
        .into()),
    }
}

/// The plan as the engine's IR: a ROOT relation translated, a compressed
/// one decompressed first; a command is returned as decoded.
fn translate_plan(
    plan: &pb::Plan,
    decompressed: Option<&[u8]>,
    conf: &TranslateConf,
    udfs: &mut dyn TranslateHost,
) -> Result<Translated, KrishivError> {
    match &plan.op_type {
        Some(OpType::Root(relation)) => {
            translate_relation_with(relation, conf, udfs).map(Translated::Relation)
        }
        Some(OpType::Command(command)) => Translated::command(command, conf, udfs),
        Some(OpType::CompressedOperation(c)) => {
            let bytes = decompressed.unwrap_or_default();
            match CompressedOpType::try_from(c.op_type) {
                Ok(CompressedOpType::Relation) => {
                    translate_relation_with(&*wire::decode_relation(bytes)?, conf, udfs)
                        .map(Translated::Relation)
                }
                Ok(CompressedOpType::Command) => match &decode_command(bytes)?.op_type {
                    Some(OpType::Command(command)) => Translated::command(command, conf, udfs),
                    _ => Translated::command(&pb::Command::default(), conf, udfs),
                },
                _ => {
                    // Spark's `PlanCompressionUtils` re-throws what parsing
                    // the operation raised as `CANNOT_PARSE` (v4.3.0).
                    let unset: KrishivError = ConnectInvalidPlanInvalidOneOfFieldNotSet {
                        full_name: "spark.connect.Plan.CompressedOperation".to_owned(),
                        name: "OP_TYPE_UNSPECIFIED".to_owned(),
                    }
                    .into();
                    Err(krishiv_common::error::ConnectInvalidPlanCannotParse {
                        error_msg: unset.to_string(),
                    }
                    .into())
                }
            }
        }
        None => Err(ConnectInvalidPlanInvalidOneOfFieldNotSet {
            full_name: "spark.connect.Plan".to_owned(),
            name: "OP_TYPE_NOT_SET".to_owned(),
        }
        .into()),
    }
}

/// A decompressed `Command`, decoded through the bounded decoder as the `command` field (2) of a
/// `Plan`, which `wire` can start from.
fn decode_command(bytes: &[u8]) -> Result<wire::Decoded<pb::Plan>, KrishivError> {
    let mut wrapped = Vec::with_capacity(bytes.len() + 11);
    prost::encoding::encode_key(2, prost::encoding::WireType::LengthDelimited, &mut wrapped);
    prost::encoding::encode_varint(bytes.len() as u64, &mut wrapped);
    wrapped.extend_from_slice(bytes);
    wire::decode_plan(&wrapped)
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
        let ahead = match Arc::clone(&self.ahead).try_acquire_many_owned(want(self.ahead_cap)) {
            Ok(permit) => permit,
            Err(TryAcquireError::Closed) => return Err(Stop::Gone),
            Err(TryAcquireError::NoPermits) => {
                let _waiting = self.op.waiting_for_client();
                Arc::clone(&self.ahead)
                    .acquire_many_owned(want(self.ahead_cap))
                    .await
                    .map_err(|_| Stop::Gone)?
            }
        };
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

/// What an operation's encoder streams.
enum Work {
    /// A query's result: schema, batches, metrics, then `result_complete`
    /// when reattachable.
    Query(Box<Query>),
    /// A `sql_command`: the `sql_command_result` relation it answers.
    Command(Box<sql_command::Answer>),
    /// A `checkpoint_command`: the id its result is held under.
    Checkpoint(String),
    /// A command with no result (`ml_command.clean_cache`, a
    /// `register_function` that ran): only `result_complete`, when
    /// reattachable.
    NoResult,
}

/// A prepared query and how its result is encoded.
struct Query {
    df: DataFrame,
    schema: SchemaRef,
    schema_response: pb::DataType,
    settings: EncodeSettings,
    /// The observed nodes' plan ids by observation name (`observed::plan_ids`).
    plan_ids: BTreeMap<String, i64>,
}

impl Work {
    async fn produce(self, sender: &Sender) -> Result<(), Stop> {
        match self {
            Self::Query(q) => {
                produce(
                    sender,
                    &q.df,
                    &q.schema,
                    q.schema_response.clone(),
                    &q.settings,
                    &q.plan_ids,
                )
                .await
            }
            Self::Command(answer) => sql_command::produce(sender, *answer).await,
            Self::Checkpoint(id) => cache_command::produce(sender, id).await,
            Self::NoResult => {
                sender.admit().await?;
                complete(sender).await
            }
        }
    }
}

/// Settles its operation when dropped unsettled: the encoder task unwound
/// (a panic in a kernel or the IPC writer). Without it the operation would
/// be an orphan (Spark's `ExecuteHolder.isOrphan`): a non-reattachable
/// stream would wait forever and a reattachable one reattach forever. Spark
/// catches every `Throwable` of the execution and sends it as the stream's
/// error (`ExecuteThreadRunner.execute`, v4.3.0 lines 122-160); Krishiv
/// sends `INTERNAL_ERROR`, or `OPERATION_CANCELED` after an interrupt.
struct Unsettled {
    op: Option<Arc<Operation>>,
    errors: Arc<SessionErrors>,
}

impl Unsettled {
    /// Settles the operation with `outcome`; its encoder has stopped.
    fn settle(mut self, outcome: Result<(), Stop>) {
        if let Some(op) = self.op.take() {
            settle(&op, &self.errors, outcome);
            op.mark_ended();
        }
    }
}

impl Drop for Unsettled {
    fn drop(&mut self) {
        if let Some(op) = self.op.take() {
            tracing::error!(
                target: "krishiv::connect",
                operation = op.id().as_str(),
                "the response encoder of a Connect operation stopped without settling it"
            );
            let error = InternalError {
                message: "the response encoder of the operation stopped unexpectedly".to_owned(),
            };
            // FAILED, or CANCELLED when an interrupt came first.
            settle(&op, &self.errors, Err(Stop::Failed(error.into())));
            op.mark_ended();
        }
    }
}

async fn encode(sender: Sender, work: Work) {
    let op = Arc::clone(&sender.op);
    let guard = Unsettled {
        op: Some(Arc::clone(&op)),
        errors: Arc::clone(&sender.errors),
    };
    // A close or an interrupt is seen at once, also while the encoder
    // waits for admission, the query's first batch, a budget or memory.
    let outcome = tokio::select! {
        biased;
        () = op.closed() => Err(Stop::Gone),
        () = op.cancelled() => Err(Stop::Failed(OperationCanceled {}.into())),
        // The query's stream goes with `produce`, which owns the work.
        r = work.produce(&sender) => r,
    };
    guard.settle(outcome);
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
            // Failed now, or finished before (its last response failed:
            // failed after all, so GetStatus agrees with the stream); an
            // interrupt that came first wins.
            let error = if op.fail() || op.fail_finished() {
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
    plan_ids: &BTreeMap<String, i64>,
) -> Result<(), Stop> {
    let sink = krishiv_session::ObservedSink::new();
    let df = &df.with_observed_sink(&sink);
    sender.admit().await?;
    let mut first = sender.response(None);
    first.schema = Some(schema_response);
    sender.send(first).await?;
    #[cfg(test)]
    hooks::encoding(sender.op.id().as_str());
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
    // The observed nodes' metrics follow the SQL metrics, all in one response as Spark sends
    // them; a query that ended in an error never gets here, so it sends none.
    // An observation whose metrics failed to finalize, or have no literal yet, fails the action
    // (Spark: the metric's error is the action's), after the others' metrics were sent.
    let mut all = Vec::new();
    let mut failed = None;
    for (name, row) in sink.take().unwrap_or_default() {
        let plan_id = plan_ids.get(&name).copied().unwrap_or(0);
        match row.and_then(|row| observed::observed_metrics(&name, plan_id, &row)) {
            Ok(metrics) => all.push(metrics),
            Err(e) => {
                failed.get_or_insert(e);
            }
        }
    }
    if !all.is_empty() {
        let mut response = sender.response(None);
        response.observed_metrics = all;
        sender.send(response).await?;
    }
    if let Some(e) = failed {
        return Err(e.into());
    }
    complete(sender).await
}

/// Ends the operation's responses after its last one: `result_complete`
/// when reattachable, then the end of its stream.
async fn complete(sender: &Sender) -> Result<(), Stop> {
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
                    // Each chunk owns its bytes: a slice of the batch would
                    // keep the whole batch alive in the retry buffer while
                    // only the chunk's length is reserved (Codex, #742).
                    data: prost::bytes::Bytes::copy_from_slice(chunk),
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
mod fault_tests;
#[cfg(test)]
mod tests;
