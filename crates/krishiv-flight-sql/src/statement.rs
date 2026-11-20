// SPDX-License-Identifier: Apache-2.0
//! Statements (design P3-T101 §4.1, §4.2, §4.5, §16 M1/M3): `GetFlightInfo(CommandStatementQuery)`
//! takes a handle slot, analyzes a query (nothing runs) or runs a command once and stores its
//! rows; `DoGet(TicketStatementQuery)` re-analyzes a query with its pinned clock and streams it, or
//! streams a command's stored rows; `CancelFlightInfo` cancels a handle's fetches and releases it.
//! Every mark a call takes is a drop guard, released whatever ends the call, the `Operation`
//! before the fetch, the slot and the call in the session table (see [`Marks`]).

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow_array::RecordBatch;
use arrow_flight::sql::{CommandStatementQuery, ProstMessageExt, TicketStatementQuery};
use arrow_flight::{FlightData, FlightEndpoint, FlightInfo, Ticket};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::stream::{self, BoxStream, Stream, StreamExt};
use krishiv_common::error::{
    ErrorClass, InternalError, InvalidHandleFormat, InvalidHandleOperationNotFound, KrishivError,
    KrishivFlightSqlFetchIdle, KrishivFlightSqlResultNotStored, KrishivFlightSqlSchemaChanged,
    KrishivFlightSqlStatementTooLarge, OperationCanceled,
};
use krishiv_connect_server::execute::arrow::{empty_batch, spark_batch};
use krishiv_connect_server::operation::{new_operation_id, plan_digest};
use krishiv_connect_server::{Operation, SessionEntry, SessionOwner};
use krishiv_memory::ConsumerHandle;
use krishiv_session::auth::Authenticated;
use krishiv_session::{QueryOptions, Session, SqlOutcome, SqlQuery, StatementClass};
use krishiv_sql::lexer::{TokenKind, is_keyword};
use prost::Message;
use tokio::sync::OwnedSemaphorePermit;
use tonic::metadata::MetadataMap;
use tonic::{Response, Status};

use crate::call::{CallSession, shutting_down};
use crate::cookies::{SessionCookie, session_cookie};
use crate::encode::{RESPONSE_CONSUMER, encode_results};
use crate::handles::HandleId;
use crate::server::{
    FlightSqlServer, HANDLES_CONSUMER, Kind, LiveOps, Payload, SMALL_RESULT_BYTES, Stored,
};
use crate::status::flight_status;

/// The `DoGet` response stream.
pub(crate) type DoGetStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send>>;

/// An `Operation` registered in a session for one run or fetch (design §4.1, "operation per
/// fetch"): not reattachable, so `busy()` waits for it at shutdown and a drop cancels its query.
pub(crate) struct OpGuard {
    entry: Arc<SessionEntry>,
    op: Arc<Operation>,
    /// The handle's live set, which holds `op` until the guard drops.
    live: Arc<LiveOps>,
}

impl OpGuard {
    /// Registers a fresh operation for `sql` in `entry`'s session, live on the handle `live`.
    pub(crate) fn register(
        entry: &Arc<SessionEntry>,
        live: &Arc<LiveOps>,
        sql: &str,
        max: usize,
    ) -> Result<Self, KrishivError> {
        let op = Arc::new(Operation::new(
            new_operation_id()?,
            plan_digest(sql.as_bytes()),
            Arc::from([]),
            false,
        ));
        entry.operations().register(&op, max)?;
        let joined = live.add(&op);
        let guard = Self {
            entry: Arc::clone(entry),
            op,
            live: Arc::clone(live),
        };
        if !joined {
            // The handle was cancelled since this call found it: the guard's drop cancels and
            // removes the operation before anything ran.
            return Err(OperationCanceled {}.into());
        }
        Ok(guard)
    }

    pub(crate) fn id(&self) -> &str {
        self.op.id().as_str()
    }

    /// The run or fetch ended as it should.
    pub(crate) fn finish(&self) {
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
        // An operation refused at `register` (its handle was cancelled) never ran: the cancel
        // finds no query and its result is ignored.
        if self.op.cancel() {
            let _ = self.entry.session().cancel(self.op.id());
        }
        self.op.mark_ended();
        self.live.remove(&self.op);
        self.entry.operations().close_and_remove(&self.op);
    }
}

/// An operation in flight on a handle ([`crate::Handles::begin`]: a fetch, or a prepared
/// handle's bind or execution): ends it when dropped, so the handle does not expire while it
/// runs and its TTL runs from the operation's end.
pub(crate) struct FetchGuard {
    pub(crate) server: FlightSqlServer,
    pub(crate) id: HandleId,
}

impl Drop for FetchGuard {
    fn drop(&mut self) {
        self.server.inner.handles.end(&self.id, Instant::now());
    }
}

/// The slot a `GetFlightInfo` reserved (design §4.1 step 1): given back when dropped before its
/// answer (§16 M3: also after its command ran; the client never saw the ticket), else left idle.
pub(crate) struct SlotGuard {
    server: FlightSqlServer,
    cap: SessionOwner,
    pub(crate) id: HandleId,
    pub(crate) answered: bool,
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

/// The marks of a `DoGet`, taken in the order call, fetch, operation, response budget and
/// released (struct fields drop in declaration order) operation, fetch, budget, call: the
/// operation ends before the fetch and the call, which is what §16 M3 needs; the budget permit
/// has no order with the others (review n1).
pub(crate) struct Marks {
    op: OpGuard,
    _fetch: FetchGuard,
    _budget: OwnedSemaphorePermit,
    _call: CallSession,
}

/// What the sweep shares with a `DoGet` stream (design §4.2, M3 of v3): since when tonic owes the
/// stream a poll (it was handed a piece and has not asked again), and the stream's live part, its
/// encoder (the query stream and the reserved source batch) and its marks. A reader idle past
/// `fetchIdleTimeout` loses the live part to the sweep (its query is cancelled; its batch, pool
/// reservation, operation, response budget and call are released: review m4) and the stream
/// ends `DEADLINE_EXCEEDED`. The clock runs only while tonic owes a poll, never while the engine
/// computes the next batch. A poll and an expiry are serialized by the `live` mutex, so a reader
/// that resumed is never expired (review m3).
pub(crate) struct FetchCell {
    epoch: Instant,
    /// Milliseconds after `epoch` plus one since tonic owes a poll; 0 while it does not.
    owed_since: AtomicU64,
    /// `None` once the stream ended or the sweep took it.
    live: Mutex<Option<Live>>,
}

/// The part of a `DoGet` the sweep can take: the encoder, dropped first, then the marks.
pub(crate) struct Live {
    inner: BoxStream<'static, Result<FlightData, Status>>,
    marks: Marks,
}

impl FetchCell {
    fn new(inner: BoxStream<'static, Result<FlightData, Status>>, marks: Marks) -> Self {
        Self {
            epoch: Instant::now(),
            owed_since: AtomicU64::new(0),
            live: Mutex::new(Some(Live { inner, marks })),
        }
    }

    fn owe_from(&self, now: Instant) {
        let ms = u64::try_from(now.saturating_duration_since(self.epoch).as_millis())
            .unwrap_or(u64::MAX - 1);
        self.owed_since.store(ms + 1, Ordering::Release);
    }

    /// Takes the live part of a stream idle over `timeout` at `now` (dropped by the caller,
    /// outside any lock); `None` when it is not idle, is being polled, or already ended.
    pub(crate) fn expire(&self, now: Instant, timeout: Duration) -> Option<Live> {
        let since = self.owed_since.load(Ordering::Acquire);
        if since == 0 {
            return None;
        }
        let at = self.epoch + Duration::from_millis(since - 1);
        if now.saturating_duration_since(at) < timeout {
            return None;
        }
        // Held by a poll: the reader is not idle.
        let mut live = match self.live.try_lock() {
            Ok(g) => g,
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return None,
        };
        // A poll between the load and the lock cleared the clock.
        if self.owed_since.load(Ordering::Acquire) != since {
            return None;
        }
        live.take()
    }
}

/// A `DoGet` stream whose live part (in its [`FetchCell`]) is held until it ends or is dropped.
struct Guarded {
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
        let mut guard = this
            .cell
            .live
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        // tonic asked: it owes nothing now.
        this.cell.owed_since.store(0, Ordering::Release);
        let Some(live) = guard.as_mut() else {
            // The sweep took it: the reader was idle past the timeout.
            drop(guard);
            this.done = true;
            let e: KrishivError = KrishivFlightSqlFetchIdle {
                timeout: format!("{:?}", this.timeout),
            }
            .into();
            return Poll::Ready(Some(Err(flight_status(&e))));
        };
        let next = live.inner.as_mut().poll_next(cx);
        let ended = match &next {
            Poll::Ready(Some(Ok(_))) => {
                this.cell.owe_from(Instant::now());
                None
            }
            Poll::Ready(None) => {
                live.marks.op.finish();
                guard.take()
            }
            Poll::Ready(Some(Err(_))) => {
                live.marks.op.fail();
                guard.take()
            }
            Poll::Pending => None,
        };
        drop(guard);
        if ended.is_some() {
            this.done = true;
        }
        // Released outside the lock.
        drop(ended);
        next
    }
}

pub(crate) fn now_micros() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_micros()).unwrap_or(i64::MAX))
}

/// The session time zone results are shaped in (as Connect's `sql_command`).
pub(crate) fn time_zone(session: &Session) -> Option<Arc<str>> {
    session
        .conf()
        .get("spark.sql.session.timeZone")
        .map(Arc::from)
}

/// `schema` in the Spark-to-Arrow form the stream sends (design §4.1, m11).
pub(crate) fn spark_schema(
    schema: &SchemaRef,
    tz: Option<&Arc<str>>,
) -> Result<SchemaRef, KrishivError> {
    Ok(empty_batch(schema, tz)?.schema())
}

/// Whether `e`, or a cause of it, is `INVALID_HANDLE.SESSION_CLOSED`: the manager was shut
/// (`close_all`) while a command's result was being stored. `store_command` wraps every failure in
/// `KRISHIV_FLIGHT_SQL_RESULT_NOT_STORED`, and `error_class` is the outer class only.
fn closed_by_shutdown(e: &KrishivError) -> bool {
    std::iter::once(e)
        .chain(e.causes())
        .any(|c| c.error_class() == ErrorClass::InvalidHandleSessionClosed)
}

/// `maxStatementBytes` (design §4.1): a statement's text over it is `INVALID_ARGUMENT`
/// (`KRISHIV_FLIGHT_SQL_STATEMENT_TOO_LARGE`) before parsing.
pub(crate) fn check_statement_bytes(sql: &str, max: usize) -> Result<(), Status> {
    if sql.len() > max {
        return Err(flight_status(
            &KrishivFlightSqlStatementTooLarge {
                bytes: sql.len().to_string(),
                limit: max.to_string(),
            }
            .into(),
        ));
    }
    Ok(())
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
pub(crate) fn not_found() -> KrishivError {
    InvalidHandleOperationNotFound {
        handle: "<ticket>".to_owned(),
    }
    .into()
}

/// `schema` as a `FlightInfo` or a prepared statement announces it: each top-level decimal column
/// with the Flight SQL column metadata `ARROW:FLIGHT:SQL:PRECISION` and `ARROW:FLIGHT:SQL:SCALE`
/// (FlightSql.proto `CommandStatementQuery`). Arrow's JDBC driver reads `ResultSetMetaData`
/// precision and scale only from these, never from the Arrow type. The streamed batches carry
/// none (the keys are optional; a client's schema check compares types).
pub(crate) fn described(schema: &Schema) -> Schema {
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| {
            let (precision, scale) = match f.data_type() {
                DataType::Decimal32(p, s)
                | DataType::Decimal64(p, s)
                | DataType::Decimal128(p, s)
                | DataType::Decimal256(p, s) => (*p, *s),
                _ => return f.as_ref().clone(),
            };
            let mut metadata = f.metadata().clone();
            metadata.insert(
                "ARROW:FLIGHT:SQL:PRECISION".to_owned(),
                precision.to_string(),
            );
            metadata.insert("ARROW:FLIGHT:SQL:SCALE".to_owned(), scale.to_string());
            f.as_ref().clone().with_metadata(metadata)
        })
        .collect();
    Schema::new_with_metadata(fields, schema.metadata().clone())
}

/// The `FlightInfo` of a statement handle (design §4.1).
pub(crate) fn flight_info(
    schema: &SchemaRef,
    id: &HandleId,
    descriptor: arrow_flight::FlightDescriptor,
) -> Result<FlightInfo, Status> {
    let ticket = TicketStatementQuery {
        statement_handle: id.as_bytes().to_vec().into(),
    };
    FlightInfo::new()
        .try_with_schema(&described(schema))
        .map_err(|e| internal(format!("encoding a schema: {e}")))
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
        check_statement_bytes(&query.query, inner.config.max_statement_bytes)?;
        let sql: Arc<str> = Arc::from(query.query.as_str());
        // A session this call created is released when the call ends, by an error or a dropped
        // future, unless the answer naming it goes out (`kept`, review M4).
        let mut call = self.call_session(metadata, caller)?;
        let (schema, mut slot) = self.run_statement(&mut call, sql).await?;
        let mut response = Response::new(flight_info(&schema, &slot.id, descriptor)?);
        call.cookie_on(&mut response);
        call.kept();
        slot.answered = true;
        Ok(response)
    }

    /// Steps 1-3 of §4.1 in the call's session: the slot, the operation, the analysis or the run.
    pub(crate) async fn run_statement(
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
            live: Arc::default(),
            fetched: Arc::default(),
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
        let op = OpGuard::register(
            &entry,
            &pending.live,
            &sql,
            inner.config.max_operations_per_session,
        )
        .map_err(|e| flight_status(&e))?;
        // 3. Under the statement gate until the analysis ended or the command ran (§16 m3).
        // Classified first: a query is analyzed and nothing runs; anything else is not run by
        // the classification and runs once below, under the operation.
        let gate = entry.shared().await;
        let session = entry.session();
        let classified = session
            .sql_query_with(&sql, QueryOptions::new().query_start(query_start))
            .await
            .map_err(|e| op_failed(&op, &e))?;
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
                    .map_err(|e| op_failed(&op, &e))?
            }
            _ => return Err(internal("an unknown outcome of a SQL statement".to_owned())),
        };
        let (schema, kind, bytes) = match outcome {
            SqlOutcome::Query(df) => {
                let schema = df.schema().await.map_err(|e| op_failed(&op, &e))?;
                drop(gate);
                let tz = time_zone(session);
                let schema = spark_schema(&schema, tz.as_ref()).map_err(|e| op_failed(&op, &e))?;
                let bytes = sql.len().saturating_add(schema_bytes(&schema));
                (Arc::clone(&schema), Kind::Query(schema), bytes)
            }
            SqlOutcome::Command(df) => {
                // The command ran: any failure from here says so (§15 m3).
                let stored = self.store_command(session, df).await;
                drop(gate);
                if stored.as_ref().err().is_some_and(closed_by_shutdown) {
                    return Err(Status::unavailable(
                        "server shutting down; the statement ran, its result was not stored",
                    ));
                }
                let stored = stored.map_err(|e| op_failed(&op, &e))?;
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
            _ => return Err(internal("an unknown outcome of a SQL statement".to_owned())),
        };
        // A command ran, so any refusal to keep its handle says so (§15 m3, review m1); a query
        // did not run and gets the refusal itself (review m2).
        let ran = matches!(kind, Kind::Command(_));
        let evicted = inner
            .handles
            .store(&cap, &slot.id, Payload { kind, ..pending }, bytes)
            .map_err(|e| {
                if ran {
                    op.fail();
                    flight_status(&not_stored_after(bytes, &e))
                } else {
                    op_failed(&op, &e)
                }
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
        // The reason is the inner error's client-safe text (review M2): an operator-audience
        // error (a spill path, a store address) is redacted as it would be on its own; the full
        // error stays the cause, for the log.
        let not_stored = |bytes: Option<usize>, e: KrishivError| -> KrishivError {
            KrishivError::from(KrishivFlightSqlResultNotStored {
                size: stored_size(bytes),
                reason: client_reason(&e),
            })
            .with_cause(e)
        };
        let tz = time_zone(session);
        let schema = df.schema().await.map_err(|e| not_stored(None, e))?;
        let rows = df.collect().await.map_err(|e| not_stored(None, e))?;
        let mut batches = Vec::with_capacity(rows.len());
        for b in &rows {
            batches.push(spark_batch(b, tz.as_ref()).map_err(|e| not_stored(None, e))?);
        }
        drop(rows);
        let schema = spark_schema(&schema, tz.as_ref()).map_err(|e| not_stored(None, e))?;
        let bytes = rows_bytes(&batches);
        let max = self.inner.config.max_command_result_bytes;
        if bytes > max {
            return Err(KrishivFlightSqlResultNotStored {
                size: stored_size(Some(bytes)),
                reason: format!("over \"krishiv.flightSql.maxCommandResultBytes\" ({max})"),
            }
            .into());
        }
        let reservation = if bytes > SMALL_RESULT_BYTES {
            let consumer =
                ConsumerHandle::register(session.runtime().memory(), HANDLES_CONSUMER, None);
            let mut r = consumer.reservation();
            r.try_grow(bytes)
                .map_err(|e| not_stored(Some(bytes), KrishivError::from(e)))?;
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
        if payload.owner != owner || matches!(payload.kind, Kind::Pending | Kind::Prepared(_)) {
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
        payload
            .fetched
            .store(true, std::sync::atomic::Ordering::Release);
        let fetch = FetchGuard {
            server: self.clone(),
            id,
        };
        let entry = Arc::clone(&call.entry);
        let op = OpGuard::register(
            &entry,
            &payload.live,
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
                    .query_start(payload.query_start_micros)
                    // The ticket was issued before any drain (a drain refuses new statements): a
                    // fetch is work the server accepted, which a drain lets finish (T066 §5.1).
                    .admitted_before_drain();
                let df = session
                    .sql_query_with(&payload.sql, options)
                    .await
                    .and_then(SqlQuery::into_query)
                    .map_err(|e| op_failed(&op, &e))?;
                let schema = df.schema().await.map_err(|e| op_failed(&op, &e))?;
                drop(gate);
                let tz = time_zone(session);
                let shaped = spark_schema(&schema, tz.as_ref()).map_err(|e| op_failed(&op, &e))?;
                if shaped != *announced {
                    return Err(op_failed(&op, &KrishivFlightSqlSchemaChanged {}.into()));
                }
                let rows = df.execute_stream().await.map_err(|e| op_failed(&op, &e))?;
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
            Kind::Pending | Kind::Prepared(_) => return Err(flight_status(&not_found())),
        };
        let encoded = encode_results(schema, batches, consumer, limits);
        let cell = Arc::new(FetchCell::new(
            encoded,
            Marks {
                op,
                _fetch: fetch,
                _budget: budget,
                _call: call,
            },
        ));
        self.watch_fetch(&cell);
        let stream = Guarded {
            cell,
            timeout: inner.config.fetch_idle_timeout,
            done: false,
        };
        Ok(Response::new(Box::pin(stream)))
    }

    /// `CancelFlightInfo` / `CancelQuery` of one statement handle (design §4.5): the handle's live
    /// operations are cancelled and the handle is released; `INVALID_HANDLE.FORMAT` for a
    /// malformed handle, one not-found answer for another owner's and an unknown one.
    pub(crate) fn cancel_statement(
        &self,
        caller: &Authenticated,
        statement_handle: &[u8],
    ) -> Result<(), KrishivError> {
        let inner = &self.inner;
        let owner = SessionOwner::of(caller);
        let cap = owner.cap_key();
        let id = handle_of(statement_handle)?;
        match inner.handles.get(&cap, &id, Instant::now()) {
            // A prepared statement is closed, not cancelled (§4.4).
            Some((_, p)) if p.owner == owner && !matches!(p.kind, Kind::Prepared(_)) => {}
            _ => return Err(not_found()),
        }
        let Some((sid, payload)) = inner.handles.remove(&cap, &id) else {
            return Err(not_found());
        };
        // Only the operations live on this handle are cancelled (§4.5): another ticket's fetch or
        // a command running in the same session is left alone.
        let live = payload.live.cancel();
        if !live.is_empty()
            && let Ok(entry) =
                inner
                    .manager
                    .get_with(caller, &sid, krishiv_session::auth::Surface::FlightSql)
        {
            for op in live {
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
        Ok(())
    }
}

/// Whether a statement of `class` leaves session state a later call depends on (design §3.2,
/// §16 m2): `Session` and `Admin` statements, any script and any `EXECUTE IMMEDIATE` (their
/// class is their parts', which may hide a `SET`; checked on the text, erring towards keeping
/// the session). Plain `Read`, `Write` and `Ddl` statements leave it unchanged.
pub(crate) fn marks_changed(class: StatementClass, sql: &str) -> bool {
    if matches!(class, StatementClass::Session | StatementClass::Admin) {
        return true;
    }
    // Spark's own lexing (nested `/* */`, `--` continued by `\`+newline and ended by `\r`,
    // `EXECUTE/**/IMMEDIATE'...'`), comments skipped; no copy of the text (review n2, review of
    // 40489d958 finding 4).
    let tokens = krishiv_sql::lexer::tokenize(sql);
    let words = tokens
        .iter()
        .filter(|t| !matches!(t.kind, TokenKind::Comment | TokenKind::UnclosedComment));
    let word = |t: &krishiv_sql::lexer::Token, kw: &str| {
        t.kind == TokenKind::Word && is_keyword(t.span.text(sql), kw)
    };
    if words.clone().next().is_some_and(|t| word(t, "BEGIN")) {
        return true;
    }
    let mut previous_execute = false;
    for t in words {
        if previous_execute && word(t, "IMMEDIATE") {
            return true;
        }
        previous_execute = word(t, "EXECUTE");
    }
    false
}

/// What a schema costs a handle (its IPC message, roughly).
pub(crate) fn schema_bytes(schema: &SchemaRef) -> usize {
    schema
        .fields()
        .iter()
        .map(|f| 64usize.saturating_add(f.name().len()))
        .sum()
}

fn rows_bytes(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::get_array_memory_size).sum()
}

/// A command that ran but whose handle was not kept (the byte cap, a swept slot): `RESULT_NOT_STORED`
/// with the refusal's client-safe text, never the refusal itself (§4.1, §15 m3).
fn not_stored_after(bytes: usize, e: &KrishivError) -> KrishivError {
    KrishivError::from(KrishivFlightSqlResultNotStored {
        size: stored_size(Some(bytes)),
        reason: client_reason(e),
    })
    .with_cause(e.clone())
}

/// A command that ran but whose result could not be read afterwards (an update's count):
/// `RESULT_NOT_STORED` of unknown size with the failure's client-safe text, so the client knows
/// the statement ran and does not run it again (§15 m3).
pub(crate) fn ran_unread(e: KrishivError) -> KrishivError {
    KrishivError::from(KrishivFlightSqlResultNotStored {
        size: stored_size(None),
        reason: client_reason(&e),
    })
    .with_cause(e)
}

/// The `<size>` of `KRISHIV_FLIGHT_SQL_RESULT_NOT_STORED`: the bytes when the result was built,
/// else "size unknown" (a failure before it was, review n3).
fn stored_size(bytes: Option<usize>) -> String {
    bytes.map_or_else(|| "size unknown".to_owned(), |b| format!("{b} bytes"))
}

/// An internal failure as a Flight status, with its class, SQLSTATE and error id (review n4).
pub(crate) fn internal(message: String) -> Status {
    flight_status(&InternalError { message }.into())
}

/// A failure after `op` was registered: the operation ends failed, not cancelled (review m10),
/// and the error goes out as its Flight status.
pub(crate) fn op_failed(op: &OpGuard, e: &KrishivError) -> Status {
    op.fail();
    flight_status(e)
}

/// An inner error's text as a client may see it ([`KrishivError::for_client`]), for the reason
/// of `RESULT_NOT_STORED` (review M2).
fn client_reason(e: &KrishivError) -> String {
    e.for_client().error().message().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_common::error::{ErrorClass, KrishivSpillIoError};

    /// Note §17 A3: each top-level decimal column of an announced schema carries Flight SQL's
    /// precision and scale; other columns, and the metadata a field had, are kept as they were.
    #[test]
    fn described_decimals_carry_precision_and_scale() {
        let kept = std::collections::HashMap::from([("k".to_owned(), "v".to_owned())]);
        let schema = Schema::new_with_metadata(
            vec![
                Field::new("d", DataType::Decimal128(25, 2), true).with_metadata(kept.clone()),
                Field::new("w", DataType::Decimal256(40, 0), true),
                Field::new("s", DataType::Utf8, true),
            ],
            kept.clone(),
        );
        let out = described(&schema);
        let meta = |i: usize, key: &str| out.field(i).metadata().get(key).cloned();
        assert_eq!(meta(0, "ARROW:FLIGHT:SQL:PRECISION").as_deref(), Some("25"));
        assert_eq!(meta(0, "ARROW:FLIGHT:SQL:SCALE").as_deref(), Some("2"));
        assert_eq!(meta(0, "k").as_deref(), Some("v"));
        assert_eq!(meta(1, "ARROW:FLIGHT:SQL:PRECISION").as_deref(), Some("40"));
        assert_eq!(meta(1, "ARROW:FLIGHT:SQL:SCALE").as_deref(), Some("0"));
        assert!(out.field(2).metadata().is_empty());
        assert_eq!(out.metadata(), &kept);
        let types: Vec<_> = out.fields().iter().map(|f| f.data_type().clone()).collect();
        let before: Vec<_> = schema
            .fields()
            .iter()
            .map(|f| f.data_type().clone())
            .collect();
        assert_eq!(types, before, "types unchanged");
    }

    /// A result that was not stored because the session was closed by shutdown is told apart from
    /// one that failed for its own reason, though `store_command` wraps both (review of d54cd33f4).
    #[test]
    fn a_not_stored_result_caused_by_a_closed_session_is_a_shutdown() {
        let closed: KrishivError = krishiv_common::error::InvalidHandleSessionClosed {
            handle: "s".to_owned(),
        }
        .into();
        let wrap = |cause: KrishivError| {
            KrishivError::from(KrishivFlightSqlResultNotStored {
                size: stored_size(None),
                reason: "r".to_owned(),
            })
            .with_cause(cause)
        };
        assert!(closed_by_shutdown(&closed));
        assert!(closed_by_shutdown(&wrap(closed)));
        assert!(!closed_by_shutdown(&wrap(not_found())));
    }

    /// An operation that would join a handle after `CancelFlightInfo` took its live set is refused,
    /// so a fetch between `begin` and its registration never streams (review of 15c4ffa75, 5).
    #[test]
    fn no_operation_joins_a_cancelled_handle() {
        let op = || {
            Arc::new(Operation::new(
                new_operation_id().expect("id"),
                plan_digest(b"SELECT 1"),
                Arc::from([]),
                false,
            ))
        };
        let live = LiveOps::default();
        let running = op();
        assert!(live.add(&running));
        let taken = live.cancel();
        assert_eq!(taken.len(), 1);
        assert!(Arc::ptr_eq(&taken[0], &running));
        assert!(!live.add(&op()), "a cancelled handle takes no operation");
        assert_eq!(
            live.cancel().len(),
            1,
            "only the one that ran before the cancel"
        );
    }

    /// Scripts and `EXECUTE IMMEDIATE` mark a provisional session changed, a comment before them
    /// included, lexed as Spark does; a word that only starts with `BEGIN` does not (review n2,
    /// review of 40489d958 finding 4).
    #[test]
    fn marks_changed_sees_scripts_behind_comments() {
        let read = StatementClass::Read;
        for sql in [
            "BEGIN SELECT 1; END",
            "  begin\nSELECT 1; END",
            "-- a comment\nBEGIN SELECT 1; END",
            "/* c */ -- d\n BEGIN SELECT 1; END",
            "SELECT 1; execute   Immediate 'SET x = 1'",
            // Bracketed comments nest (SqlBaseLexer BRACKETED_COMMENT).
            "/* a /* b */ c */ BEGIN SET x = 1; END",
            // A `--` comment goes on past `\`+newline, and ends at a bare `\r`.
            "-- a \\\n still the comment\nBEGIN SET x = 1; END",
            "-- a\rBEGIN SET x = 1; END",
            // No whitespace around the words.
            "EXECUTE IMMEDIATE'SET x = 1'",
            "EXECUTE/**/IMMEDIATE 'SET x = 1'",
        ] {
            assert!(marks_changed(read, sql), "{sql:?}");
        }
        for sql in [
            "BEGINNING",
            "SELECT 'begin'",
            "-- BEGIN\nSELECT 1",
            "EXECUTE x",
            "/* a /* BEGIN */ c */ SELECT 1",
            "-- a \\\nBEGIN\nSELECT 1",
            "SELECT 'EXECUTE IMMEDIATE x'",
        ] {
            assert!(!marks_changed(read, sql), "{sql:?}");
        }
    }

    /// `RESULT_NOT_STORED` carries the inner error's client-safe text (review M2).
    #[test]
    fn result_not_stored_reason_is_redacted() {
        let inner: KrishivError = KrishivSpillIoError {
            operation: "write".into(),
            path: "/var/spill/tenant-7/x".into(),
            reason: "disk full on host-12".into(),
        }
        .into();
        let reason = client_reason(&inner);
        for secret in ["/var/spill", "tenant-7", "host-12"] {
            assert!(!reason.contains(secret), "{reason}");
        }
        let limit: KrishivError = krishiv_common::error::KrishivFlightSqlHandleLimit {
            limit_name: "/var/spill".into(),
            limit: "1".into(),
        }
        .into();
        let wrapped = not_stored_after(10, &limit);
        assert_eq!(
            wrapped.error_class(),
            ErrorClass::KrishivFlightSqlResultNotStored
        );
        let shown = flight_status(&wrapped);
        assert!(shown.message().contains("ran"), "{}", shown.message());
    }

    /// The size is said only when it is known (review n3): never "of 0 bytes".
    #[test]
    fn result_not_stored_says_the_size_only_when_known() {
        let cause: KrishivError = InternalError {
            message: "x".into(),
        }
        .into();
        let known = flight_status(&not_stored_after(10, &cause));
        assert!(
            known.message().contains("(10 bytes)"),
            "{}",
            known.message()
        );
        let unknown: KrishivError = KrishivFlightSqlResultNotStored {
            size: stored_size(None),
            reason: "r".into(),
        }
        .into();
        let unknown = flight_status(&unknown);
        assert!(
            unknown.message().contains("(size unknown)"),
            "{}",
            unknown.message()
        );
        assert!(!unknown.message().contains("0 bytes"));
    }
}
