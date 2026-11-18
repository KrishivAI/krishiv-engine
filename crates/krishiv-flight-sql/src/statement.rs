// SPDX-License-Identifier: Apache-2.0
//! Statements (design P3-T101 §4.1, §4.2, §4.5, §16 M1/M3): `GetFlightInfo(CommandStatementQuery)`
//! takes a handle slot, analyzes a query (nothing runs) or runs a command once and stores its
//! rows; `DoGet(TicketStatementQuery)` re-analyzes a query with its pinned clock and streams it, or
//! streams a command's stored rows; `CancelFlightInfo` cancels a handle's fetches and releases it.
//! Every mark a call takes is a drop guard, released in reverse order whatever ends the call:
//! the call in the session table, the slot, the fetch, the `Operation`.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow_array::RecordBatch;
use arrow_flight::sql::{CommandStatementQuery, ProstMessageExt, TicketStatementQuery};
use arrow_flight::{FlightData, FlightEndpoint, FlightInfo, Ticket};
use arrow_schema::SchemaRef;
use futures::stream::{self, BoxStream, Stream, StreamExt};
use krishiv_common::error::{
    InvalidHandleFormat, InvalidHandleOperationNotFound, KrishivError, KrishivFlightSqlFetchIdle,
    KrishivFlightSqlResultNotStored, KrishivFlightSqlSchemaChanged,
    KrishivFlightSqlStatementTooLarge,
};
use krishiv_connect_server::execute::arrow::{empty_batch, spark_batch};
use krishiv_connect_server::operation::{new_operation_id, plan_digest};
use krishiv_connect_server::{Operation, SessionEntry, SessionOwner};
use krishiv_memory::ConsumerHandle;
use krishiv_session::auth::Authenticated;
use krishiv_session::{QueryOptions, Session, SqlOutcome, SqlQuery, StatementClass};
use prost::Message;
use tokio::sync::OwnedSemaphorePermit;
use tonic::metadata::MetadataMap;
use tonic::{Response, Status};

use crate::call::{CallSession, shutting_down};
use crate::cookies::{SessionCookie, session_cookie};
use crate::encode::{RESPONSE_CONSUMER, encode_results};
use crate::handles::HandleId;
use crate::server::{FlightSqlServer, HANDLES_CONSUMER, Kind, Payload, SMALL_RESULT_BYTES, Stored};
use crate::status::flight_status;

/// The `DoGet` response stream.
pub(crate) type DoGetStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send>>;

/// An `Operation` registered in a session for one run or fetch (design §4.1, "operation per
/// fetch"): not reattachable, so `busy()` waits for it at shutdown and a drop cancels its query.
struct OpGuard {
    entry: Arc<SessionEntry>,
    op: Arc<Operation>,
}

impl OpGuard {
    /// Registers a fresh operation for `sql` in `entry`'s session.
    fn register(entry: &Arc<SessionEntry>, sql: &str, max: usize) -> Result<Self, KrishivError> {
        let op = Arc::new(Operation::new(
            new_operation_id()?,
            plan_digest(sql.as_bytes()),
            Arc::from([]),
            false,
        ));
        entry.operations().register(&op, max)?;
        Ok(Self {
            entry: Arc::clone(entry),
            op,
        })
    }

    fn id(&self) -> &str {
        self.op.id().as_str()
    }

    /// The run or fetch ended as it should.
    fn finish(&self) {
        self.op.finish();
    }

    /// The run or fetch failed.
    fn fail(&self) {
        self.op.fail();
    }
}

impl Drop for OpGuard {
    fn drop(&mut self) {
        // Still running: the call was dropped (a client gone, a cancel), so its query is cancelled.
        if self.op.cancel() {
            let _ = self.entry.session().cancel(self.op.id());
        }
        self.op.mark_ended();
        self.entry.operations().close_and_remove(&self.op);
    }
}

/// A fetch in flight on a handle ([`crate::Handles::begin`]): ends it when dropped, so the TTL
/// runs from the stream's end.
struct FetchGuard {
    server: FlightSqlServer,
    id: HandleId,
}

impl Drop for FetchGuard {
    fn drop(&mut self) {
        self.server.inner.handles.end(&self.id, Instant::now());
    }
}

/// The slot a `GetFlightInfo` reserved (design §4.1 step 1): given back when dropped before its
/// answer (§16 M3: also after its command ran; the client never saw the ticket), else left idle.
struct SlotGuard {
    server: FlightSqlServer,
    cap: SessionOwner,
    id: HandleId,
    answered: bool,
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        let handles = &self.server.inner.handles;
        if self.answered {
            handles.end(&self.id, Instant::now());
        } else {
            let _ = handles.remove(&self.cap, &self.id);
        }
    }
}

/// The marks of a `DoGet`, released in reverse order of taking: the operation, the fetch, the
/// response budget, the call (struct fields drop in declaration order).
pub(crate) struct Marks {
    op: OpGuard,
    _fetch: FetchGuard,
    _budget: OwnedSemaphorePermit,
    _call: CallSession,
}

/// What the sweep shares with a `DoGet` stream (design §4.2, M3 of v3): since when tonic owes the
/// stream a poll (it was handed a piece and has not asked again), and the stream's marks. A reader
/// idle past `fetchIdleTimeout` loses its marks to the sweep (its query is cancelled; its
/// operation, response budget and call are released) and the stream ends `DEADLINE_EXCEEDED`.
/// The clock runs only while tonic owes a poll, never while the engine computes the next batch.
pub(crate) struct FetchCell {
    epoch: Instant,
    /// Milliseconds after `epoch` plus one since tonic owes a poll; 0 while it does not.
    owed_since: AtomicU64,
    marks: Mutex<Option<Marks>>,
    expired: AtomicBool,
}

impl FetchCell {
    fn new(marks: Marks) -> Self {
        Self {
            epoch: Instant::now(),
            owed_since: AtomicU64::new(0),
            marks: Mutex::new(Some(marks)),
            expired: AtomicBool::new(false),
        }
    }

    fn take(&self) -> Option<Marks> {
        self.marks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    fn owe_from(&self, now: Instant) {
        let ms = u64::try_from(now.saturating_duration_since(self.epoch).as_millis())
            .unwrap_or(u64::MAX - 1);
        self.owed_since.store(ms + 1, Ordering::Release);
    }

    /// Takes the marks of a stream idle over `timeout` at `now` (dropped by the caller, outside
    /// any lock); `None` when it is not idle or already ended.
    pub(crate) fn expire(&self, now: Instant, timeout: Duration) -> Option<Marks> {
        let since = self.owed_since.load(Ordering::Acquire);
        if since == 0 {
            return None;
        }
        let at = self.epoch + Duration::from_millis(since - 1);
        if now.saturating_duration_since(at) < timeout {
            return None;
        }
        let marks = self.take()?;
        self.expired.store(true, Ordering::Release);
        Some(marks)
    }
}

/// A `DoGet` stream holding its marks (in its [`FetchCell`]) until it ends or is dropped.
struct Guarded {
    inner: BoxStream<'static, Result<FlightData, Status>>,
    cell: Arc<FetchCell>,
    timeout: Duration,
    done: bool,
}

impl Stream for Guarded {
    type Item = Result<FlightData, Status>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        if this.done {
            return Poll::Ready(None);
        }
        // tonic asked: it owes nothing now.
        this.cell.owed_since.store(0, Ordering::Release);
        if this.cell.expired.load(Ordering::Acquire) {
            this.done = true;
            let e: KrishivError = KrishivFlightSqlFetchIdle {
                timeout: format!("{:?}", this.timeout),
            }
            .into();
            return Poll::Ready(Some(Err(flight_status(&e))));
        }
        let next = this.inner.as_mut().poll_next(cx);
        match &next {
            Poll::Ready(Some(Ok(_))) => this.cell.owe_from(Instant::now()),
            Poll::Ready(None) => {
                this.done = true;
                if let Some(m) = this.cell.take() {
                    m.op.finish();
                }
            }
            Poll::Ready(Some(Err(_))) => {
                this.done = true;
                if let Some(m) = this.cell.take() {
                    m.op.fail();
                }
            }
            Poll::Pending => {}
        }
        next
    }
}

fn now_micros() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_micros()).unwrap_or(i64::MAX))
}

/// The session time zone results are shaped in (as Connect's `sql_command`).
fn time_zone(session: &Session) -> Option<Arc<str>> {
    session
        .conf()
        .get("spark.sql.session.timeZone")
        .map(Arc::from)
}

/// `schema` in the Spark-to-Arrow form the stream sends (design §4.1, m11).
fn spark_schema(schema: &SchemaRef, tz: Option<&Arc<str>>) -> Result<SchemaRef, KrishivError> {
    Ok(empty_batch(schema, tz)?.schema())
}

/// The handle of a ticket's or `FlightInfo`'s `TicketStatementQuery`.
pub(crate) fn handle_of(statement_handle: &[u8]) -> Result<HandleId, KrishivError> {
    HandleId::from_bytes(statement_handle).ok_or_else(|| {
        InvalidHandleFormat {
            handle: "<ticket>".to_owned(),
        }
        .into()
    })
}

/// A handle another owner holds, an unknown one and one not answered yet get the same answer.
fn not_found() -> KrishivError {
    InvalidHandleOperationNotFound {
        handle: "<ticket>".to_owned(),
    }
    .into()
}

/// The `FlightInfo` of a statement handle (design §4.1).
fn flight_info(
    schema: &SchemaRef,
    id: &HandleId,
    descriptor: arrow_flight::FlightDescriptor,
) -> Result<FlightInfo, Status> {
    let ticket = TicketStatementQuery {
        statement_handle: id.as_bytes().to_vec().into(),
    };
    FlightInfo::new()
        .try_with_schema(schema)
        .map_err(|e| Status::internal(format!("encoding a schema: {e}")))
        .map(|info| {
            info.with_endpoint(
                FlightEndpoint::new().with_ticket(Ticket::new(ticket.as_any().encode_to_vec())),
            )
            .with_descriptor(descriptor)
            .with_total_records(-1)
            .with_total_bytes(-1)
        })
}

impl FlightSqlServer {
    /// `GetFlightInfo(CommandStatementQuery)` (design §4.1).
    pub(crate) async fn statement_info(
        &self,
        query: CommandStatementQuery,
        metadata: &MetadataMap,
        descriptor: arrow_flight::FlightDescriptor,
        caller: &Authenticated,
    ) -> Result<Response<FlightInfo>, Status> {
        let inner = &self.inner;
        let sql: Arc<str> = Arc::from(query.query.as_str());
        if sql.len() > inner.config.max_statement_bytes {
            return Err(flight_status(
                &KrishivFlightSqlStatementTooLarge {
                    bytes: sql.len().to_string(),
                    limit: inner.config.max_statement_bytes.to_string(),
                }
                .into(),
            ));
        }
        let mut call = self.call_session(metadata, caller)?;
        match self.run_statement(&mut call, sql).await {
            Ok((schema, slot)) => {
                let mut response = Response::new(flight_info(&schema, &slot.id, descriptor)?);
                call.cookie_on(&mut response);
                call.kept();
                let mut slot = slot;
                slot.answered = true;
                Ok(response)
            }
            Err(e) => {
                call.failed();
                Err(e)
            }
        }
    }

    /// Steps 1-3 of §4.1 in the call's session: the slot, the operation, the analysis or the run.
    async fn run_statement(
        &self,
        call: &mut CallSession,
        sql: Arc<str>,
    ) -> Result<(SchemaRef, SlotGuard), Status> {
        let inner = &self.inner;
        let cap = call.owner.cap_key();
        let query_start = now_micros();
        let pending = Payload {
            owner: call.owner.clone(),
            sql: Arc::clone(&sql),
            query_start_micros: query_start,
            kind: Kind::Pending,
        };
        // 1. The slot (count and text bytes, with eviction); a refusal here runs nothing.
        let inserted = inner
            .handles
            .reserve(
                &cap,
                &call.id,
                pending.clone(),
                sql.len(),
                false,
                Instant::now(),
            )
            .map_err(|e| flight_status(&e))?;
        let mut slot = SlotGuard {
            server: self.clone(),
            cap: cap.clone(),
            id: inserted.id,
            answered: false,
        };
        self.after_evictions(&inserted.evicted);
        // 2. The operation, before the statement's kind is known (a command runs in step 3).
        let entry = Arc::clone(&call.entry);
        let op = OpGuard::register(&entry, &sql, inner.config.max_operations_per_session)
            .map_err(|e| flight_status(&e))?;
        // 3. Under the statement gate until the analysis ended or the command ran (§16 m3).
        // Classified first: a query is analyzed and nothing runs; anything else is not run by
        // the classification and runs once below, under the operation.
        let gate = entry.shared().await;
        let session = entry.session();
        let classified = session
            .sql_query_with(&sql, QueryOptions::new().query_start(query_start))
            .await
            .map_err(|e| flight_status(&e))?;
        let outcome = match classified {
            SqlQuery::Query(df) => SqlOutcome::Query(df),
            SqlQuery::NotAQuery(class) => {
                if marks_changed(class, &sql) {
                    // Changed before it runs: a later call depends on what it leaves (§3.2,
                    // §16 m2), even when storing its result fails.
                    call.mark_changed();
                }
                let options = QueryOptions::new()
                    .operation_id(op.id())
                    .query_start(query_start);
                session
                    .sql_command_with(&sql, options)
                    .await
                    .map_err(|e| flight_status(&e))?
            }
            _ => return Err(Status::internal("an unknown outcome of a SQL statement")),
        };
        let (schema, kind, bytes) = match outcome {
            SqlOutcome::Query(df) => {
                let schema = df.schema().await.map_err(|e| flight_status(&e))?;
                drop(gate);
                let tz = time_zone(session);
                let schema = spark_schema(&schema, tz.as_ref()).map_err(|e| flight_status(&e))?;
                let bytes = sql.len().saturating_add(schema_bytes(&schema));
                (Arc::clone(&schema), Kind::Query(schema), bytes)
            }
            SqlOutcome::Command(df) => {
                // The command ran: any failure from here says so (§15 m3).
                let stored = self.store_command(session, df).await;
                drop(gate);
                if stored.is_err() && self.draining() {
                    return Err(Status::unavailable(
                        "server shutting down; the statement ran, its result was not stored",
                    ));
                }
                let stored = stored.map_err(|e| flight_status(&e))?;
                let bytes = sql
                    .len()
                    .saturating_add(schema_bytes(&stored.schema))
                    .saturating_add(rows_bytes(&stored.batches));
                (
                    Arc::clone(&stored.schema),
                    Kind::Command(Arc::new(stored)),
                    bytes,
                )
            }
            _ => return Err(Status::internal("an unknown outcome of a SQL statement")),
        };
        let evicted = inner
            .handles
            .store(&cap, &slot.id, Payload { kind, ..pending }, bytes)
            .map_err(|e| match bytes_of_command(bytes, &e) {
                Some(not_stored) => flight_status(&not_stored),
                None => flight_status(&e),
            });
        let evicted = match evicted {
            Ok(evicted) => evicted,
            Err(status) => {
                slot.answered = false;
                return Err(status);
            }
        };
        self.after_evictions(&evicted);
        op.finish();
        Ok((schema, slot))
    }

    /// Collects a command's result rows in Spark's Arrow shapes, in the session time zone read
    /// after it ran (§4.1, v3 review m9), bounded by `maxCommandResultBytes` and reserved in the
    /// pool as `flight_sql_handles` past 64 KiB.
    async fn store_command(
        &self,
        session: &Session,
        df: krishiv_session::DataFrame,
    ) -> Result<Stored, KrishivError> {
        let not_stored = |bytes: usize, reason: String| -> KrishivError {
            KrishivFlightSqlResultNotStored {
                bytes: bytes.to_string(),
                reason,
            }
            .into()
        };
        let tz = time_zone(session);
        let schema = df
            .schema()
            .await
            .map_err(|e| not_stored(0, e.message().to_owned()))?;
        let rows = df
            .collect()
            .await
            .map_err(|e| not_stored(0, e.message().to_owned()))?;
        let mut batches = Vec::with_capacity(rows.len());
        for b in &rows {
            batches.push(
                spark_batch(b, tz.as_ref()).map_err(|e| not_stored(0, e.message().to_owned()))?,
            );
        }
        drop(rows);
        let schema = spark_schema(&schema, tz.as_ref())
            .map_err(|e| not_stored(0, e.message().to_owned()))?;
        let bytes = rows_bytes(&batches);
        let max = self.inner.config.max_command_result_bytes;
        if bytes > max {
            return Err(not_stored(
                bytes,
                format!("over \"krishiv.flightSql.maxCommandResultBytes\" ({max})"),
            ));
        }
        let reservation = if bytes > SMALL_RESULT_BYTES {
            let consumer =
                ConsumerHandle::register(session.runtime().memory(), HANDLES_CONSUMER, None);
            let mut r = consumer.reservation();
            r.try_grow(bytes)
                .map_err(|e| not_stored(bytes, KrishivError::from(e).message().to_owned()))?;
            Some(r)
        } else {
            None
        };
        Ok(Stored {
            schema,
            batches,
            _reservation: reservation,
        })
    }

    /// `DoGet(TicketStatementQuery)` (design §4.1, §4.2).
    pub(crate) async fn statement_get(
        &self,
        ticket: TicketStatementQuery,
        metadata: &MetadataMap,
        caller: &Authenticated,
    ) -> Result<Response<DoGetStream>, Status> {
        let inner = &self.inner;
        let owner = SessionOwner::of(caller);
        let cap = owner.cap_key();
        let id = handle_of(&ticket.statement_handle).map_err(|e| flight_status(&e))?;
        let Some((sid, payload)) = inner.handles.get(&cap, &id, Instant::now()) else {
            return Err(flight_status(&not_found()));
        };
        if payload.owner != owner || matches!(payload.kind, Kind::Pending) {
            return Err(flight_status(&not_found()));
        }
        // The call is counted from before its analysis (admission waits count, §3.2).
        let cookie = session_cookie(metadata);
        let via_cookie = matches!(&cookie, SessionCookie::Present(c) if *c == sid);
        let mut call = self
            .enter(&owner, caller, &sid, via_cookie)
            .map_err(|_| flight_status(&not_found()))?;
        match &cookie {
            SessionCookie::Present(other) if *other != sid => {
                // Its own session is promoted (v3 review m10); the fetch runs in the handle's.
                let _ = inner.sessions.promote(&owner, other);
            }
            SessionCookie::Absent => call.end_as_cookieless_fetch(),
            _ => {}
        }
        let Some((_, payload)) = inner.handles.begin(&cap, &id, Instant::now()) else {
            return Err(flight_status(&not_found()));
        };
        let fetch = FetchGuard {
            server: self.clone(),
            id,
        };
        let entry = Arc::clone(&call.entry);
        let op = OpGuard::register(
            &entry,
            &payload.sql,
            inner.config.max_operations_per_session,
        )
        .map_err(|e| flight_status(&e))?;
        // The stream's share of `maxResponseBytes` (§16 M1); the wait is cancellable.
        let limits = inner.config.encode;
        let budget = inner
            .budget
            .reserve(limits.batch_bytes.saturating_add(limits.data_bytes))
            .await
            .ok_or_else(shutting_down)?;
        let session = entry.session();
        let consumer =
            ConsumerHandle::register(session.runtime().memory(), RESPONSE_CONSUMER, None);
        let (schema, batches): (
            SchemaRef,
            BoxStream<'static, Result<RecordBatch, KrishivError>>,
        ) = match &payload.kind {
            Kind::Query(announced) => {
                let gate = entry.shared().await;
                let options = QueryOptions::new()
                    .operation_id(op.id())
                    .query_start(payload.query_start_micros);
                let df = session
                    .sql_query_with(&payload.sql, options)
                    .await
                    .and_then(SqlQuery::into_query)
                    .map_err(|e| flight_status(&e))?;
                let schema = df.schema().await.map_err(|e| flight_status(&e))?;
                drop(gate);
                let tz = time_zone(session);
                let shaped = spark_schema(&schema, tz.as_ref()).map_err(|e| flight_status(&e))?;
                if shaped != *announced {
                    return Err(flight_status(&KrishivFlightSqlSchemaChanged {}.into()));
                }
                let rows = df.execute_stream().await.map_err(|e| flight_status(&e))?;
                let rows = rows
                    .map(move |b| b.and_then(|b| spark_batch(&b, tz.as_ref())))
                    .boxed();
                (shaped, rows)
            }
            Kind::Command(stored) => {
                let schema = Arc::clone(&stored.schema);
                let stored = Arc::clone(stored);
                let n = stored.batches.len();
                let rows = stream::iter(0..n)
                    .map(move |i| Ok(stored.batches[i].clone()))
                    .boxed();
                (schema, rows)
            }
            Kind::Pending => return Err(flight_status(&not_found())),
        };
        let encoded = encode_results(schema, batches, consumer, limits);
        let cell = Arc::new(FetchCell::new(Marks {
            op,
            _fetch: fetch,
            _budget: budget,
            _call: call,
        }));
        self.watch_fetch(&cell);
        let stream = Guarded {
            inner: encoded,
            cell,
            timeout: inner.config.fetch_idle_timeout,
            done: false,
        };
        Ok(Response::new(Box::pin(stream)))
    }

    /// `CancelFlightInfo` of a statement ticket (design §4.5): the handle's live fetches are
    /// cancelled and the handle is released; `true` when it was the caller's.
    pub(crate) fn cancel_statement(&self, caller: &Authenticated, statement_handle: &[u8]) -> bool {
        let inner = &self.inner;
        let owner = SessionOwner::of(caller);
        let cap = owner.cap_key();
        let Ok(id) = handle_of(statement_handle) else {
            return false;
        };
        match inner.handles.get(&cap, &id, Instant::now()) {
            Some((_, p)) if p.owner == owner => {}
            _ => return false,
        }
        let Some((sid, payload)) = inner.handles.remove(&cap, &id) else {
            return false;
        };
        let _ = payload;
        // A fetch of this handle ends its own operation when its stream is dropped by the
        // client; the operations of the handle's session that run its SQL are cancelled here.
        if let Ok(entry) =
            inner
                .manager
                .get_with(caller, &sid, krishiv_session::auth::Surface::FlightSql)
        {
            for op in entry.operations().all() {
                if op.cancel() {
                    let _ = entry.session().cancel(op.id());
                }
            }
        }
        if let Some(r) = inner
            .sessions
            .release_if_handleless(&owner, &sid, |s| self.holds(s))
        {
            self.release(&r);
        }
        true
    }
}

/// Whether a statement of `class` leaves session state a later call depends on (design §3.2,
/// §16 m2): `Session` and `Admin` statements, any script and any `EXECUTE IMMEDIATE` (their
/// class is their parts', which may hide a `SET`; checked on the text, erring towards keeping
/// the session). Plain `Read`, `Write` and `Ddl` statements leave it unchanged.
fn marks_changed(class: StatementClass, sql: &str) -> bool {
    if matches!(class, StatementClass::Session | StatementClass::Admin) {
        return true;
    }
    let upper = sql.trim_start().to_ascii_uppercase();
    upper.starts_with("BEGIN")
        || upper
            .split_ascii_whitespace()
            .collect::<Vec<_>>()
            .windows(2)
            .any(|w| w == ["EXECUTE", "IMMEDIATE"])
}

/// What a schema costs a handle (its IPC message, roughly).
fn schema_bytes(schema: &SchemaRef) -> usize {
    schema
        .fields()
        .iter()
        .map(|f| 64usize.saturating_add(f.name().len()))
        .sum()
}

fn rows_bytes(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::get_array_memory_size).sum()
}

/// A command whose rows do not fit the owner's byte cap ran: `RESULT_NOT_STORED`, never the
/// handle limit (§4.1).
fn bytes_of_command(bytes: usize, e: &KrishivError) -> Option<KrishivError> {
    (e.error_class() == krishiv_common::error::ErrorClass::KrishivFlightSqlHandleLimit).then(|| {
        KrishivFlightSqlResultNotStored {
            bytes: bytes.to_string(),
            reason: e.message().to_owned(),
        }
        .into()
    })
}
