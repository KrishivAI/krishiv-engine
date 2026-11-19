// SPDX-License-Identifier: Apache-2.0
//! Prepared statements (design P3-T101 §4.4): `CreatePreparedStatement` classifies (a query is
//! analyzed, nothing runs) and stores a prepared handle, which is never evicted;
//! `DoPut(CommandPreparedStatementQuery)` binds one row of positional parameters (`?`, written
//! as literals as `EXECUTE IMMEDIATE ... USING` writes them); `GetFlightInfo` of it makes a
//! statement handle as §4.1; `DoPut(CommandPreparedStatementUpdate)` runs a command once per
//! bound row, the counts summed; `ClosePreparedStatement` drops the handle. Every call but the
//! create is handle-carrying: it runs in the handle's session, whatever cookie it presents
//! (§3.2, v3 review M4/m10).

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{ArrayRef, NullArray, RecordBatch};
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
use krishiv_session::{QueryOptions, SqlQuery};
use prost::Message;
use tonic::metadata::MetadataMap;
use tonic::{Response, Status, Streaming};

use crate::call::CallSession;
use crate::cookies::{SessionCookie, session_cookie};
use crate::handles::HandleId;
use crate::server::{FlightSqlServer, Kind, Payload, Prepared};
use crate::statement::{
    check_statement_bytes, flight_info, handle_of, internal, not_found, now_micros, schema_bytes,
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
        .try_with_schema(schema)
        .map(|info| info.schema)
        .map_err(|e| internal(format!("encoding a schema: {e}")))
}

/// The parameter schema of `markers` markers (design §4.4): each nullable `Null`.
fn parameter_schema(markers: usize) -> Schema {
    Schema::new(
        (1..=markers)
            .map(|i| Field::new(format!("parameter_{i}"), DataType::Null, true))
            .collect::<Vec<_>>(),
    )
}

/// One row binding every marker to `NULL`: the classification text of a parameterized
/// statement (its kind and, for a query, its reads are checked at create).
fn null_row(markers: usize) -> Result<RecordBatch, Status> {
    let columns: Vec<ArrayRef> = (0..markers)
        .map(|_| Arc::new(NullArray::new(1)) as ArrayRef)
        .collect();
    RecordBatch::try_new(Arc::new(parameter_schema(markers)), columns)
        .map_err(|e| internal(format!("a parameter row: {e}")))
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
    /// a parameterized statement with each marker `NULL`), stored as a prepared handle in the
    /// call's session, which it marks changed (§3.2).
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
        let classified_text = if markers == 0 {
            sql.to_string()
        } else {
            session
                .bind_parameters(&sql, &null_row(markers)?, 0)
                .map_err(|e| flight_status(&e))?
        };
        let query_start = now_micros();
        let gate = entry.shared().await;
        let classified = session
            .sql_query_with(
                &classified_text,
                QueryOptions::new().query_start(query_start),
            )
            .await
            .map_err(|e| flight_status(&e))?;
        let (query, schema) = match classified {
            SqlQuery::Query(df) => {
                let schema = df.schema().await.map_err(|e| flight_status(&e))?;
                drop(gate);
                let tz = time_zone(session);
                let shaped = spark_schema(&schema, tz.as_ref()).map_err(|e| flight_status(&e))?;
                // A parameterized query's columns are not known before its binding (§4.4).
                (true, (markers == 0).then_some(shaped))
            }
            SqlQuery::NotAQuery(_) => (false, None),
            _ => return Err(internal("an unknown kind of SQL statement".to_owned())),
        };
        let dataset_schema = match &schema {
            Some(s) => schema_message(s)?,
            None => prost::bytes::Bytes::new(),
        };
        let parameter_schema = if markers == 0 {
            prost::bytes::Bytes::new()
        } else {
            schema_message(&parameter_schema(markers))?
        };
        let bytes = sql
            .len()
            .saturating_add(schema.as_ref().map_or(0, schema_bytes));
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
        let (_, sid, payload, prepared) =
            self.prepared_of(caller, &command.prepared_statement_handle)?;
        let mut call = self.handle_call(caller, metadata, &sid)?;
        let sql = prepared
            .bound
            .clone()
            .unwrap_or_else(|| Arc::clone(&payload.sql));
        let (schema, mut slot) = self.run_statement(&mut call, sql).await?;
        let response = Response::new(flight_info(&schema, &slot.id, descriptor)?);
        slot.answered = true;
        Ok(response)
    }

    /// `DoPut(CommandPreparedStatementQuery)` (design §4.4): binds one row (no row clears the
    /// binding); answers the same handle.
    pub(crate) async fn prepared_bind(
        &self,
        command: CommandPreparedStatementQuery,
        stream: Peekable<Streaming<FlightData>>,
        metadata: &MetadataMap,
        caller: &Authenticated,
    ) -> Result<Response<DoPutStream>, Status> {
        let inner = &self.inner;
        let handle = command.prepared_statement_handle;
        let (id, sid, payload, prepared) = self.prepared_of(caller, &handle)?;
        let call = self.handle_call(caller, metadata, &sid)?;
        let rows = bound_rows(stream, inner.config.max_batch_rows).await?;
        let count: usize = rows.iter().map(RecordBatch::num_rows).sum();
        if prepared.query && count > 1 {
            return Err(invalid_parameters(format!(
                "a query binds one row, not {count}"
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
            .map_err(|e| flight_status(&e))?;
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
    /// once, in order, and the counts are summed (`-1` when a run reports none); with no row
    /// it runs once as prepared. A query is `KRISHIV_FLIGHT_SQL_QUERY_AS_UPDATE`, unrun.
    pub(crate) async fn prepared_update(
        &self,
        command: &CommandPreparedStatementUpdate,
        stream: Peekable<Streaming<FlightData>>,
        metadata: &MetadataMap,
        caller: &Authenticated,
    ) -> Result<Response<DoPutStream>, Status> {
        let (_, sid, payload, prepared) =
            self.prepared_of(caller, &command.prepared_statement_handle)?;
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
        let mut total: Option<i64> = Some(0);
        for sql in &texts {
            let (count, _) = self.run_update(&call, sql).await?;
            total = match total {
                Some(t) if count >= 0 => Some(t.saturating_add(count)),
                _ => None,
            };
        }
        Ok(Response::new(update_result(total.unwrap_or(-1))))
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
    fn parameter_schema_is_nullable_null_per_marker() {
        let s = parameter_schema(2);
        assert_eq!(s.fields().len(), 2);
        assert!(
            s.fields()
                .iter()
                .all(|f| f.is_nullable() && f.data_type() == &DataType::Null)
        );
        let bytes = schema_message(&s).expect("message");
        assert_eq!(decode_schema(&bytes).as_deref(), Some(&s));
        assert_eq!(decode_schema(&[]), None);
        assert_eq!(null_row(3).expect("row").num_rows(), 1);
    }
}
