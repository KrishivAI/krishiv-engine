// SPDX-License-Identifier: Apache-2.0
//! [`SessionSource`]: the UI backend's data from an engine runtime (embedded mode, P2-T133).
//!
//! Rows come from `SELECT * FROM system.<table>` run by a session of the UI's own, with the principal
//! [`UI_PRINCIPAL`]. That session's queries are recorded like any other, so they are left out of every list the
//! UI serves (and, history being capped per principal, they can only push out each other, never a user's rows).
//! Only the columns a page can show are converted: text, integers, floats, booleans and timestamps (as
//! microseconds since the epoch); maps and lists become `null`.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::principal::{Principal, PrincipalKind};
use krishiv_session::{CancelReason, EngineRuntime, QueryId, Session};
use serde_json::{Number, Value};

use crate::internal::{Row, SystemTable, UiError, UiFuture, UiSource};

/// The principal of the UI's own session; its queries are hidden from the UI.
pub const UI_PRINCIPAL: &str = "krishiv-ui";

/// The UI's data source over one engine runtime.
pub struct SessionSource {
    session: Session,
    runtime: Arc<EngineRuntime>,
}

impl std::fmt::Debug for SessionSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionSource").finish_non_exhaustive()
    }
}

impl SessionSource {
    /// A source reading `runtime`'s system tables.
    ///
    /// # Errors
    /// Returns an error when the UI's session cannot be built.
    pub fn new(runtime: Arc<EngineRuntime>) -> Result<Self, UiError> {
        let session = Session::builder()
            .runtime(Arc::clone(&runtime))
            .principal(Principal::new(UI_PRINCIPAL, PrincipalKind::Service))
            .build()
            .map_err(|e| UiError(e.to_string()))?;
        Ok(Self { session, runtime })
    }
}

fn table_name(table: SystemTable) -> &'static str {
    match table {
        SystemTable::Queries => "queries",
        SystemTable::Operators => "operators",
        SystemTable::Memory => "memory",
        SystemTable::Caches => "caches",
    }
}

fn cell(column: &dyn Array, row: usize) -> Value {
    if column.is_null(row) {
        return Value::Null;
    }
    match column.data_type() {
        DataType::Utf8 => Value::String(column.as_string::<i32>().value(row).to_owned()),
        DataType::LargeUtf8 => Value::String(column.as_string::<i64>().value(row).to_owned()),
        DataType::Utf8View => Value::String(column.as_string_view().value(row).to_owned()),
        DataType::Boolean => Value::Bool(column.as_boolean().value(row)),
        DataType::Int32 => Value::from(
            column
                .as_primitive::<arrow_array::types::Int32Type>()
                .value(row),
        ),
        DataType::Int64 => Value::from(
            column
                .as_primitive::<arrow_array::types::Int64Type>()
                .value(row),
        ),
        DataType::Float64 => Number::from_f64(
            column
                .as_primitive::<arrow_array::types::Float64Type>()
                .value(row),
        )
        .map_or(Value::Null, Value::Number),
        DataType::Timestamp(TimeUnit::Microsecond, _) => Value::from(
            column
                .as_primitive::<arrow_array::types::TimestampMicrosecondType>()
                .value(row),
        ),
        _ => Value::Null,
    }
}

/// The rows of `batches` as JSON objects.
#[must_use]
pub fn batches_to_rows(batches: &[RecordBatch]) -> Vec<Row> {
    let mut out = Vec::new();
    for batch in batches {
        let schema = batch.schema();
        for row in 0..batch.num_rows() {
            let mut r = Row::new();
            for (i, field) in schema.fields().iter().enumerate() {
                r.insert(field.name().clone(), cell(batch.column(i).as_ref(), row));
            }
            out.push(r);
        }
    }
    out
}

impl UiSource for SessionSource {
    fn rows(&self, table: SystemTable) -> UiFuture<'_, Vec<Row>> {
        Box::pin(async move {
            let sql = format!("SELECT * FROM system.{}", table_name(table));
            let df = self
                .session
                .sql(&sql)
                .await
                .map_err(|e| UiError(e.to_string()))?;
            let batches = df.collect().await.map_err(|e| UiError(e.to_string()))?;
            let mut rows = batches_to_rows(&batches);
            if table == SystemTable::Queries {
                rows.retain(|r| r.get("principal").and_then(Value::as_str) != Some(UI_PRINCIPAL));
            }
            Ok(rows)
        })
    }

    fn cancel<'a>(&'a self, query_id: &'a str) -> UiFuture<'a, bool> {
        Box::pin(async move {
            let id = QueryId::parse(query_id).map_err(|e| UiError(e.to_string()))?;
            Ok(self.runtime.cancel(&id, CancelReason::Admin))
        })
    }
}
