// SPDX-License-Identifier: Apache-2.0
//! Prepared statements (design P3-T101 §4.4): `CreatePreparedStatement` classifies (a query is
//! analyzed, nothing runs) and stores a prepared handle, which is never evicted;
//! `DoPut(CommandPreparedStatementQuery)` binds one row of positional parameters (`?`, written
//! as literals as `EXECUTE IMMEDIATE ... USING` writes them); `GetFlightInfo` of it makes a
//! statement handle as §4.1; `DoPut(CommandPreparedStatementUpdate)` runs a command once per
//! bound row, the counts summed; `ClosePreparedStatement` drops the handle. Every call but the
//! create is handle-carrying: it runs in the handle's session, whatever cookie it presents
//! (§3.2, v3 review M4/m10). A bind or execution is an operation on the handle: it does not
//! expire while one runs, and its TTL runs from the end of the last (§4.1).

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{ArrayRef, Int32Array, NullArray, RecordBatch};
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::error::FlightError;
use arrow_flight::sql::{
    ActionClosePreparedStatementRequest, ActionCreatePreparedStatementRequest,
    ActionCreatePreparedStatementResult, CommandPreparedStatementQuery,
    CommandPreparedStatementUpdate, DoPutPreparedStatementResult, ProstMessageExt,
};
use arrow_flight::{FlightData, FlightDescriptor, FlightInfo, PutResult};
use arrow_schema::{DataType, Field, Schema};
use futures::stream::{self, Peekable};
use futures::{StreamExt, TryStreamExt};
use krishiv_common::error::{KrishivError, KrishivFlightSqlInvalidParameters};
use krishiv_connect_server::SessionOwner;
use krishiv_session::auth::Authenticated;
use krishiv_session::parameter_types::BoundAs;
use krishiv_session::{QueryOptions, Session, SqlQuery};
use prost::Message;
use tonic::metadata::MetadataMap;
use tonic::{Response, Status, Streaming};

use crate::call::CallSession;
use crate::cookies::{SessionCookie, session_cookie};
use crate::handles::HandleId;
use crate::server::{FlightSqlServer, Kind, Payload, Prepared};
use crate::statement::{
    FetchGuard, check_statement_bytes, flight_info, handle_of, internal, not_found, now_micros,
    spark_schema, time_zone,
};
use crate::status::flight_status;
use crate::update::{DoPutStream, update_result};

/// The `DoAction` response stream.
pub(crate) type ActionStream =
    std::pin::Pin<Box<dyn futures::Stream<Item = Result<arrow_flight::Result, Status>> + Send>>;

/// `KRISHIV_FLIGHT_SQL_INVALID_PARAMETERS` (07001, `INVALID_ARGUMENT`).
fn invalid_parameters(reason: String) -> Status {
    flight_status(&KrishivFlightSqlInvalidParameters { reason }.into())
}

/// `schema` as the IPC message `dataset_schema` / `parameter_schema` carry.
fn schema_message(schema: &Schema) -> Result<prost::bytes::Bytes, Status> {
    // `FlightInfo.schema` is the same encapsulated IPC schema message.
    FlightInfo::new()
        .try_with_schema(&crate::statement::described(schema))
        .map(|info| info.schema)
        .map_err(|e| internal(format!("encoding a schema: {e}")))
}

/// The parameter schema of markers typed `types` (design §4.4, amended 2026-11-20): each
/// nullable, in Spark's Arrow shapes as a result column of that type; a marker with no type is a
/// string. The JDBC driver binds each value into a vector of this type.
fn parameter_schema(
    types: &[Option<krishiv_types::DataType>],
    session: &Session,
) -> Result<arrow_schema::SchemaRef, KrishivError> {
    let schema = krishiv_session::parameter_types::parameter_schema(types);
    spark_schema(&schema, time_zone(session).as_ref())
}

/// One row binding every marker to `NULL`: the classification text of a parameterized
/// statement (its kind and, for a query, its reads, columns and marker types are checked at
/// create).
fn null_row(markers: usize) -> Result<RecordBatch, Status> {
    let fields: Vec<Field> = (1..=markers)
        .map(|i| Field::new(format!("parameter_{i}"), DataType::Null, true))
        .collect();
    let columns: Vec<ArrayRef> = (0..markers)
        .map(|_| Arc::new(NullArray::new(1)) as ArrayRef)
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| internal(format!("a parameter row: {e}")))
}

/// One row binding every marker to the `INT` `1`: the second classification text, for a query
/// a `NULL` cannot analyze (`LIMIT ?`, `OFFSET ?`: Spark's `INVALID_LIMIT_LIKE_EXPRESSION`).
fn one_row(markers: usize) -> Result<RecordBatch, Status> {
    let fields: Vec<Field> = (1..=markers)
        .map(|i| Field::new(format!("parameter_{i}"), DataType::Int32, false))
        .collect();
    let columns: Vec<ArrayRef> = (0..markers)
        .map(|_| Arc::new(Int32Array::from(vec![1])) as ArrayRef)
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| internal(format!("a parameter row: {e}")))
}

/// The record count of a prepared update's runs (design §4.4): their sum (a run that reports no
/// count counts `0`, `update::record_count`).
fn summed(counts: impl IntoIterator<Item = i64>) -> i64 {
    counts
        .into_iter()
        .fold(0i64, |t, c| t.saturating_add(c.max(0)))
}

/// The refusal of row `ran + 1` of a prepared update's `of` rows after `ran` had run: its own
/// code, class and text, saying how many ran (§15 m3: a retry of the whole batch would run them
/// twice).
fn partly_ran(status: &Status, ran: usize, of: usize) -> Status {
    Status::with_metadata(
        status.code(),
        format!(
            "{} ({ran} of {of} parameter rows ran before this one failed; they are not undone)",
            status.message()
        ),
        status.metadata().clone(),
    )
}

/// What classifying a statement found: whether it is a query, its columns in Spark's Arrow
/// shapes, and the type of each marker whose bound literal analysis cast (none for a command).
type Classified = (
    bool,
    Option<arrow_schema::SchemaRef>,
    Vec<Option<krishiv_types::DataType>>,
);

/// A statement's kind and, for a query, its columns in Spark's Arrow shapes and its markers'
/// types (`krishiv_session::parameter_types::marker_types`, the markers' values at `spans` of
/// `text`), analyzed (nothing runs) as `text` under the session's policy.
async fn classify(
    session: &Session,
    text: &str,
    spans: &[std::ops::Range<usize>],
    bound: BoundAs,
    query_start: i64,
) -> Result<Classified, KrishivError> {
    match session
        .sql_query_with(text, QueryOptions::new().query_start(query_start))
        .await?
    {
        SqlQuery::Query(df) => {
            let schema = df.schema().await?;
            let tz = time_zone(session);
            // A plan the caller may not be shown (definer views) leaves the markers untyped,
            // asked quietly (`DataFrame::marker_types`): preparing is not a denial.
            let types = if spans.is_empty() {
                Vec::new()
            } else {
                df.marker_types(spans, bound).await?
            };
            Ok((true, Some(spark_schema(&schema, tz.as_ref())?), types))
        }
        SqlQuery::NotAQuery(_) => Ok((false, None, Vec::new())),
        _ => Err(krishiv_common::error::InternalError {
            message: "an unknown kind of SQL statement".to_owned(),
        }
        .into()),
    }
}

/// The handle of a bind or execution: the handle is live until `_use` drops (design §4.1).
struct InUse {
    id: HandleId,
    sid: String,
    payload: Payload,
    prepared: Arc<Prepared>,
    _use: FetchGuard,
}

/// The bound rows of a `DoPut` stream, its first message (the descriptor) included: at most
/// `max` rows, more is `INVALID_ARGUMENT` (read no further). A message with no IPC header (the
/// descriptor alone) carries no rows.
async fn bound_rows(
    stream: Peekable<Streaming<FlightData>>,
    max: usize,
) -> Result<Vec<RecordBatch>, Status> {
    let data = stream
        .try_filter(|d| futures::future::ready(!d.data_header.is_empty()))
        .map_err(|s| FlightError::Tonic(Box::new(s)));
    let mut batches = FlightRecordBatchStream::new_from_flight_data(data);
    let mut out = Vec::new();
    let mut rows = 0usize;
    while let Some(batch) = batches.next().await {
        let batch = batch.map_err(|e| match e {
            FlightError::Tonic(s) => *s,
            other => invalid_parameters(format!("the parameter rows are not Arrow IPC ({other})")),
        })?;
        rows = rows.saturating_add(batch.num_rows());
        if rows > max {
            return Err(invalid_parameters(format!(
                "more than \"krishiv.flightSql.maxBatchRows\" ({max}) rows"
            )));
        }
        if batch.num_rows() > 0 {
            out.push(batch);
        }
    }
    Ok(out)
}

impl FlightSqlServer {
    /// The prepared handle `handle` names for `caller`, with its session: another owner's, an
    /// unknown one and a statement handle get one not-found answer; a malformed one
    /// `INVALID_HANDLE.FORMAT`.
    fn prepared_of(
        &self,
        caller: &Authenticated,
        handle: &[u8],
    ) -> Result<(HandleId, String, Payload, Arc<Prepared>), Status> {
        let owner = SessionOwner::of(caller);
        let id = handle_of(handle).map_err(|e| flight_status(&e))?;
        match self
            .inner
            .handles
            .get(&owner.cap_key(), &id, Instant::now())
        {
            Some((sid, payload)) if payload.owner == owner => match &payload.kind {
                Kind::Prepared(p) => {
                    let p = Arc::clone(p);
                    Ok((id, sid, payload, p))
                }
                _ => Err(flight_status(&not_found())),
            },
            _ => Err(flight_status(&not_found())),
        }
    }

    /// As [`Self::prepared_of`], and starts an operation on the handle (a bind or an execution,
    /// review of 4d7f3235f finding 1): it does not expire until the [`InUse`] drops, and its TTL
    /// runs from then.
    fn prepared_use(&self, caller: &Authenticated, handle: &[u8]) -> Result<InUse, Status> {
        let owner = SessionOwner::of(caller);
        let id = handle_of(handle).map_err(|e| flight_status(&e))?;
        let Some((sid, payload)) = self
            .inner
            .handles
            .begin(&owner.cap_key(), &id, Instant::now())
        else {
            return Err(flight_status(&not_found()));
        };
        let guard = FetchGuard {
            server: self.clone(),
            id,
        };
        match &payload.kind {
            Kind::Prepared(p) if payload.owner == owner => {
                let prepared = Arc::clone(p);
                Ok(InUse {
                    id,
                    sid,
                    payload,
                    prepared,
                    _use: guard,
                })
            }
            _ => Err(flight_status(&not_found())),
        }
    }

    /// A handle-carrying call (design §3.2): it runs in the handle's session `sid`; a cookie
    /// naming another of the caller's sessions still promotes that one (v3 review m10).
    fn handle_call(
        &self,
        caller: &Authenticated,
        metadata: &MetadataMap,
        sid: &str,
    ) -> Result<CallSession, Status> {
        let owner = SessionOwner::of(caller);
        let cookie = session_cookie(metadata);
        let via_cookie = matches!(&cookie, SessionCookie::Present(c) if c == sid);
        let call = self
            .enter(&owner, caller, sid, via_cookie)
            .map_err(|_| flight_status(&not_found()))?;
        if let SessionCookie::Present(other) = &cookie
            && other != sid
        {
            let _ = self.inner.sessions.promote(&owner, other);
        }
        Ok(call)
    }

    /// `CreatePreparedStatement` (design §4.4): classified with `sql_query_with` (nothing runs;
    /// a parameterized statement with each marker `NULL`, else, when that does not analyze, each
    /// marker `1`), stored as a prepared handle in the call's session, which it marks changed
    /// (§3.2). A query answers the columns of that analysis as its `dataset_schema`, a marker's
    /// own column as `Null`: a client (Arrow's JDBC driver) reads an empty schema as an update.
    pub(crate) async fn prepared_create(
        &self,
        request: ActionCreatePreparedStatementRequest,
        metadata: &MetadataMap,
        caller: &Authenticated,
    ) -> Result<Response<ActionStream>, Status> {
        let inner = &self.inner;
        check_statement_bytes(&request.query, inner.config.max_statement_bytes)?;
        let sql: Arc<str> = Arc::from(request.query.as_str());
        let mut call = self.call_session(metadata, caller)?;
        let entry = Arc::clone(&call.entry);
        let session = entry.session();
        let markers = krishiv_sql::parameter_markers(
            &sql,
            session.parser_options().double_quoted_identifiers,
        )
        .len();
        let query_start = now_micros();
        let gate = entry.shared().await;
        let (query, schema, mut types) = if markers == 0 {
            classify(session, &sql, &[], BoundAs::Null, query_start).await
        } else {
            let (text, spans) = session
                .bind_parameters_spanned(&sql, &null_row(markers)?, 0)
                .map_err(|e| flight_status(&e))?;
            match classify(session, &text, &spans, BoundAs::Null, query_start).await {
                Ok(c) => Ok(c),
                // Only a query is analyzed, so only a query fails here: one a `NULL` cannot
                // stand in (`LIMIT ?`) is tried with `1`; failing both, the `NULL` error stands.
                // A marker its `1` was not cast for is an `INT`, as the `1` is.
                Err(first) => {
                    let (text, spans) = session
                        .bind_parameters_spanned(&sql, &one_row(markers)?, 0)
                        .map_err(|e| flight_status(&e))?;
                    classify(session, &text, &spans, BoundAs::One, query_start)
                        .await
                        .map(|(q, s, t)| {
                            let t = t
                                .into_iter()
                                .map(|t| t.or(Some(krishiv_types::DataType::Integer)))
                                .collect();
                            (q, s, t)
                        })
                        .map_err(|_| first)
                }
            }
        }
        .map_err(|e| flight_status(&e))?;
        // A command is not analyzed: an `INSERT ... VALUES` cell's marker takes its column's type.
        if markers > 0 && !query {
            types = session.insert_parameter_types(&sql, markers).await;
        }
        let parameter_schema = if markers == 0 {
            prost::bytes::Bytes::new()
        } else {
            schema_message(&*parameter_schema(&types, session).map_err(|e| flight_status(&e))?)?
        };
        drop(gate);
        let dataset_schema = match &schema {
            Some(s) => schema_message(s)?,
            None => prost::bytes::Bytes::new(),
        };
        // The handle holds its text (and, once bound, the bound text; `prepared_bind`).
        let bytes = sql.len();
        let payload = Payload {
            owner: call.owner.clone(),
            sql: Arc::clone(&sql),
            query_start_micros: query_start,
            kind: Kind::Prepared(Arc::new(Prepared {
                query,
                markers,
                bound: None,
            })),
            live: Arc::default(),
            // A prepared handle is not a ticket: it never holds a drain open.
            fetched: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        };
        let cap = call.owner.cap_key();
        let inserted = inner
            .handles
            .insert(&cap, &call.id, payload, bytes, true, Instant::now())
            .map_err(|e| flight_status(&e))?;
        self.after_evictions(&inserted.evicted);
        call.mark_changed();
        let result = ActionCreatePreparedStatementResult {
            prepared_statement_handle: inserted.id.as_bytes().to_vec().into(),
            dataset_schema,
            parameter_schema,
        };
        let out: ActionStream = Box::pin(stream::iter([Ok(arrow_flight::Result {
            body: result.as_any().encode_to_vec().into(),
        })]));
        let mut response = Response::new(out);
        call.cookie_on(&mut response);
        call.kept();
        Ok(response)
    }

    /// `ClosePreparedStatement` (design §4.4, §3.2 rule 3): the handle goes; one already gone,
    /// another owner's or unknown answers OK (nothing to close).
    pub(crate) fn prepared_close(
        &self,
        request: &ActionClosePreparedStatementRequest,
        caller: &Authenticated,
    ) -> Result<Response<ActionStream>, Status> {
        let inner = &self.inner;
        let owner = SessionOwner::of(caller);
        if let Ok((id, _, _, _)) = self.prepared_of(caller, &request.prepared_statement_handle) {
            if let Some((sid, _)) = inner.handles.remove(&owner.cap_key(), &id)
                && let Some(r) = inner
                    .sessions
                    .release_if_handleless(&owner, &sid, |s| self.holds(s))
            {
                self.release(&r);
            }
        } else {
            // Only a malformed handle is refused.
            handle_of(&request.prepared_statement_handle).map_err(|e| flight_status(&e))?;
        }
        let out: ActionStream = Box::pin(stream::empty());
        Ok(Response::new(out))
    }

    /// `GetFlightInfo(CommandPreparedStatementQuery)` (design §4.4): the statement with its
    /// binding runs as `GetFlightInfo(CommandStatementQuery)` does, in the handle's session (a
    /// query is analyzed, a command runs once), and answers a statement handle's ticket.
    pub(crate) async fn prepared_info(
        &self,
        command: &CommandPreparedStatementQuery,
        metadata: &MetadataMap,
        descriptor: FlightDescriptor,
        caller: &Authenticated,
    ) -> Result<Response<FlightInfo>, Status> {
        let used = self.prepared_use(caller, &command.prepared_statement_handle)?;
        let mut call = self.handle_call(caller, metadata, &used.sid)?;
        let sql = used
            .prepared
            .bound
            .clone()
            .unwrap_or_else(|| Arc::clone(&used.payload.sql));
        let (schema, mut slot) = self.run_statement(&mut call, sql).await?;
        let response = Response::new(flight_info(&schema, &slot.id, descriptor)?);
        slot.answered = true;
        Ok(response)
    }

    /// `DoPut(CommandPreparedStatementQuery)` (design §4.4): binds one row (no row clears the
    /// binding), for a query or a command alike (more rows go through
    /// `DoPut(CommandPreparedStatementUpdate)`); answers the same handle.
    pub(crate) async fn prepared_bind(
        &self,
        command: CommandPreparedStatementQuery,
        stream: Peekable<Streaming<FlightData>>,
        metadata: &MetadataMap,
        caller: &Authenticated,
    ) -> Result<Response<DoPutStream>, Status> {
        let inner = &self.inner;
        let handle = command.prepared_statement_handle;
        let InUse {
            id,
            sid,
            payload,
            prepared,
            _use,
        } = self.prepared_use(caller, &handle)?;
        let call = self.handle_call(caller, metadata, &sid)?;
        let rows = bound_rows(stream, inner.config.max_batch_rows).await?;
        let count: usize = rows.iter().map(RecordBatch::num_rows).sum();
        if count > 1 {
            return Err(invalid_parameters(format!(
                "a binding is one row, not {count} (a command runs a batch through \
                 CommandPreparedStatementUpdate)"
            )));
        }
        let bound = match rows.first() {
            Some(batch) => {
                let text = self.bind_row(&call, &payload.sql, prepared.markers, batch, 0)?;
                Some(Arc::from(text.as_str()))
            }
            None => None,
        };
        let bytes = payload
            .sql
            .len()
            .saturating_add(bound.as_ref().map_or(0, |b: &Arc<str>| b.len()));
        let rebound = Payload {
            kind: Kind::Prepared(Arc::new(Prepared {
                query: prepared.query,
                markers: prepared.markers,
                bound,
            })),
            ..payload
        };
        let cap = call.owner.cap_key();
        let evicted = inner
            .handles
            .store(&cap, &id, rebound, bytes)
            .map_err(|e| {
                // Closed while it was being bound: gone, not an internal error.
                if inner.handles.get(&cap, &id, Instant::now()).is_none() {
                    flight_status(&not_found())
                } else {
                    flight_status(&e)
                }
            })?;
        self.after_evictions(&evicted);
        let result = DoPutPreparedStatementResult {
            prepared_statement_handle: Some(handle),
        };
        let out: DoPutStream = Box::pin(stream::iter([Ok(PutResult {
            app_metadata: result.encode_to_vec().into(),
        })]));
        Ok(Response::new(out))
    }

    /// `DoPut(CommandPreparedStatementUpdate)` (design §4.4): each bound row runs the command
    /// once, in order, and the counts are summed (a run that reports none counts `0`); with no row
    /// it runs once as prepared. A query is `KRISHIV_FLIGHT_SQL_QUERY_AS_UPDATE`, unrun. A row
    /// that fails after others ran says how many did (they are not undone).
    pub(crate) async fn prepared_update(
        &self,
        command: &CommandPreparedStatementUpdate,
        stream: Peekable<Streaming<FlightData>>,
        metadata: &MetadataMap,
        caller: &Authenticated,
    ) -> Result<Response<DoPutStream>, Status> {
        let InUse {
            sid,
            payload,
            prepared,
            _use,
            ..
        } = self.prepared_use(caller, &command.prepared_statement_handle)?;
        if prepared.query {
            return Err(flight_status(
                &krishiv_common::error::KrishivFlightSqlQueryAsUpdate {}.into(),
            ));
        }
        let call = self.handle_call(caller, metadata, &sid)?;
        let rows = bound_rows(stream, self.inner.config.max_batch_rows).await?;
        // Every row is bound before any runs: a bad row runs none.
        let mut texts: Vec<Arc<str>> = Vec::new();
        for batch in &rows {
            for row in 0..batch.num_rows() {
                let text = self.bind_row(&call, &payload.sql, prepared.markers, batch, row)?;
                texts.push(Arc::from(text.as_str()));
            }
        }
        if texts.is_empty() {
            texts.push(
                prepared
                    .bound
                    .clone()
                    .unwrap_or_else(|| Arc::clone(&payload.sql)),
            );
        }
        let mut counts = Vec::with_capacity(texts.len());
        for (ran, sql) in texts.iter().enumerate() {
            match self.run_update(&call, sql).await {
                Ok((count, _)) => counts.push(count),
                Err(s) if ran > 0 => return Err(partly_ran(&s, ran, texts.len())),
                Err(s) => return Err(s),
            }
        }
        Ok(Response::new(update_result(summed(counts))))
    }

    /// `sql` with row `row` of `batch` bound to its `markers` markers, one column each; the
    /// text is bounded by `maxStatementBytes`.
    fn bind_row(
        &self,
        call: &CallSession,
        sql: &str,
        markers: usize,
        batch: &RecordBatch,
        row: usize,
    ) -> Result<String, Status> {
        if batch.num_columns() != markers {
            return Err(invalid_parameters(format!(
                "the statement has {markers} parameter markers, the rows {} columns",
                batch.num_columns()
            )));
        }
        let text = call
            .entry
            .session()
            .bind_parameters(sql, batch, row)
            .map_err(|e: KrishivError| flight_status(&e))?;
        check_statement_bytes(&text, self.inner.config.max_statement_bytes)?;
        Ok(text)
    }
}

/// The `dataset_schema` a test decodes (an IPC schema message) as a schema.
#[cfg(test)]
pub(crate) fn decode_schema(bytes: &[u8]) -> Option<Arc<Schema>> {
    if bytes.is_empty() {
        return None;
    }
    let message = arrow_flight::IpcMessage(prost::bytes::Bytes::copy_from_slice(bytes));
    Schema::try_from(message).ok().map(Arc::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_schema_types_each_marker_nullable() {
        let session = Session::new().expect("session");
        let s = parameter_schema(&[Some(krishiv_types::DataType::Date), None], &session)
            .expect("schema");
        let types: Vec<&DataType> = s.fields().iter().map(|f| f.data_type()).collect();
        assert_eq!(types, [&DataType::Date32, &DataType::Utf8]);
        assert!(s.fields().iter().all(|f| f.is_nullable()));
        assert_eq!(s.field(0).name(), "parameter_1");
        let bytes = schema_message(&s).expect("message");
        assert_eq!(decode_schema(&bytes).as_deref(), Some(&*s));
        assert_eq!(decode_schema(&[]), None);
        assert_eq!(null_row(3).expect("row").num_rows(), 1);
        assert_eq!(one_row(2).expect("row").num_columns(), 2);
    }

    #[test]
    fn counts_sum() {
        assert_eq!(summed([2, 3]), 5);
        assert_eq!(summed([0, 7, 0]), 7);
        assert_eq!(summed([2, -1, 3]), 5, "a negative count is none");
        assert_eq!(summed([]), 0);
        assert_eq!(summed([i64::MAX, 1]), i64::MAX);
    }

    #[test]
    fn a_partial_batch_keeps_the_class_and_says_how_many_ran() {
        let failed = invalid_parameters("bad".to_owned());
        let s = partly_ran(&failed, 1, 3);
        assert_eq!(s.code(), failed.code());
        assert_eq!(
            s.metadata().get("x-krishiv-error-class"),
            failed.metadata().get("x-krishiv-error-class")
        );
        assert!(s.message().starts_with(failed.message()), "{}", s.message());
        assert!(
            s.message().contains("1 of 3 parameter rows ran"),
            "{}",
            s.message()
        );
    }
}
