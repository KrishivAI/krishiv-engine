// SPDX-License-Identifier: Apache-2.0
//! Updates (design P3-T101 §4.3, §16 M2): `DoPut(CommandStatementUpdate)` runs a statement that
//! returns no rows (DML, DDL, `SET`, `USE`, ...) once, under an `Operation`, and answers
//! `DoPutUpdateResult{record_count}`. A query is refused before anything runs
//! (`KRISHIV_FLIGHT_SQL_QUERY_AS_UPDATE`). A cookie-less update that leaves no session state runs in
//! a transient session, released at the end of the call with no `set-cookie`; one that does (a
//! `SET`, a script) keeps its provisional session, marked changed, and answers with its cookie.

use std::pin::Pin;
use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch};
use arrow_flight::PutResult;
use arrow_flight::sql::{CommandStatementUpdate, DoPutUpdateResult};
use futures::{Stream, stream};
use krishiv_common::error::KrishivFlightSqlQueryAsUpdate;
use krishiv_session::{QueryOptions, SqlOutcome, SqlQuery};
use prost::Message;
use tonic::metadata::MetadataMap;
use tonic::{Response, Status};

use crate::call::{CallSession, traced};
use crate::cookies::{SessionCookie, session_cookie};
use crate::server::FlightSqlServer;
use crate::statement::{
    OpGuard, check_statement_bytes, internal, marks_changed, now_micros, op_failed, ran_unread,
};
use crate::status::flight_status;

/// The `DoPut` response stream.
pub(crate) type DoPutStream = Pin<Box<dyn Stream<Item = Result<PutResult, Status>> + Send>>;

/// The column a command's result names its affected-row count with (Delta's and the row-level
/// commands' `num_affected_rows`).
const AFFECTED_ROWS: &str = "num_affected_rows";

/// `record_count` of a command's result rows (design §4.3, amended 2026-11-20): the affected rows
/// where the statement reports them (a first column `num_affected_rows` of one `BIGINT` row),
/// else `0`, as a JDBC update count of DDL is. Not `-1` ("unknown" in Flight SQL): Arrow's JDBC
/// driver (Avatica) reads an update count of `-1` as a statement with a result set and asks for
/// it, which runs the command a second time.
pub(crate) fn record_count(batches: &[RecordBatch]) -> i64 {
    let mut rows = batches.iter().filter(|b| b.num_rows() > 0);
    let (Some(batch), None) = (rows.next(), rows.next()) else {
        return 0;
    };
    if batch.num_rows() != 1
        || batch
            .schema()
            .fields()
            .first()
            .is_none_or(|f| f.name() != AFFECTED_ROWS)
    {
        return 0;
    }
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .filter(|a| a.is_valid(0))
        .map_or(0, |a| a.value(0))
}

/// The answer of an update: one `PutResult` carrying `DoPutUpdateResult{record_count}`.
pub(crate) fn update_result(record_count: i64) -> DoPutStream {
    let result = DoPutUpdateResult { record_count };
    Box::pin(stream::iter([Ok(PutResult {
        app_metadata: result.encode_to_vec().into(),
    })]))
}

impl FlightSqlServer {
    /// `DoPut(CommandStatementUpdate)` (design §4.3, §16 M2).
    pub(crate) async fn statement_update(
        &self,
        command: CommandStatementUpdate,
        metadata: &MetadataMap,
        caller: &krishiv_session::auth::Authenticated,
    ) -> Result<Response<DoPutStream>, Status> {
        check_statement_bytes(&command.query, self.inner.config.max_statement_bytes)?;
        let cookieless = matches!(session_cookie(metadata), SessionCookie::Absent);
        let mut call = self.call_session(metadata, caller)?;
        let sql: Arc<str> = Arc::from(command.query.as_str());
        let (count, changed) = self.run_update(&call, &sql, metadata).await?;
        let mut response = Response::new(update_result(count));
        // A cookie-less update that left no session state ran in a transient session: no
        // cookie, and the guard releases it as the call ends (§16 M2).
        if !cookieless || changed {
            call.cookie_on(&mut response);
            call.kept();
        }
        Ok(response)
    }

    /// Runs `sql` (no rows expected) once in the call's session under a fresh operation: the
    /// record count, and whether it marked the session changed.
    pub(crate) async fn run_update(
        &self,
        call: &CallSession,
        sql: &Arc<str>,
        metadata: &MetadataMap,
    ) -> Result<(i64, bool), Status> {
        let entry = Arc::clone(&call.entry);
        let op = OpGuard::register(
            &entry,
            &Arc::default(),
            sql,
            self.inner.config.max_operations_per_session,
        )
        .map_err(|e| flight_status(&e))?;
        let query_start = now_micros();
        let gate = entry.shared().await;
        let session = entry.session();
        let options = || traced(QueryOptions::new().query_start(query_start), metadata);
        let class = match session
            .sql_query_with(sql, options())
            .await
            .map_err(|e| op_failed(&op, &e))?
        {
            SqlQuery::Query(_) => {
                return Err(op_failed(&op, &KrishivFlightSqlQueryAsUpdate {}.into()));
            }
            SqlQuery::NotAQuery(class) => class,
            _ => return Err(internal("an unknown kind of SQL statement".to_owned())),
        };
        let changed = marks_changed(class, sql);
        if changed {
            call.mark_changed();
        }
        let outcome = session
            .sql_command_with(sql, options().operation_id(op.id()))
            .await
            .map_err(|e| op_failed(&op, &e))?;
        let count = match outcome {
            SqlOutcome::Command(df) => df.collect().await.map(|rows| record_count(&rows)),
            SqlOutcome::Query(_) => {
                return Err(op_failed(&op, &KrishivFlightSqlQueryAsUpdate {}.into()));
            }
            _ => return Err(internal("an unknown outcome of a SQL statement".to_owned())),
        };
        drop(gate);
        // The command ran (eagerly): a failure reading its count must not invite a rerun.
        let count = count.map_err(|e| op_failed(&op, &ran_unread(e)))?;
        op.finish();
        Ok((count, changed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema};

    fn batch(name: &str, values: Vec<Option<i64>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values))]).expect("batch")
    }

    #[test]
    fn record_count_reads_num_affected_rows_only() {
        assert_eq!(record_count(&[batch(AFFECTED_ROWS, vec![Some(7)])]), 7);
        let empty = batch(AFFECTED_ROWS, vec![]);
        assert_eq!(
            record_count(&[empty.clone(), batch(AFFECTED_ROWS, vec![Some(3)]), empty]),
            3
        );
        assert_eq!(record_count(&[]), 0, "INSERT reports nothing");
        assert_eq!(record_count(&[batch("x", vec![Some(7)])]), 0);
        assert_eq!(record_count(&[batch(AFFECTED_ROWS, vec![None])]), 0);
        assert_eq!(
            record_count(&[batch(AFFECTED_ROWS, vec![Some(1), Some(2)])]),
            0
        );
        assert_eq!(
            record_count(&[
                batch(AFFECTED_ROWS, vec![Some(1)]),
                batch(AFFECTED_ROWS, vec![Some(2)])
            ]),
            0
        );
    }
}
