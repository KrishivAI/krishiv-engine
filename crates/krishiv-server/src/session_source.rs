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
use krishiv_common::error::KrishivError;
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
            .map_err(ui_error)?;
        Ok(Self { session, runtime })
    }
}

/// The engine's error with its Spark class and SQLSTATE kept.
#[expect(
    clippy::needless_pass_by_value,
    reason = "used as the function of map_err"
)]
fn ui_error(e: KrishivError) -> UiError {
    UiError::with_class(e.message(), e.error_class().name(), e.sql_state())
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
            let df = self.session.sql(&sql).await.map_err(ui_error)?;
            let batches = df.collect().await.map_err(ui_error)?;
            let mut rows = batches_to_rows(&batches);
            if table == SystemTable::Queries {
                rows.retain(|r| r.get("principal").and_then(Value::as_str) != Some(UI_PRINCIPAL));
            }
            Ok(rows)
        })
    }

    fn cancel<'a>(&'a self, query_id: &'a str) -> UiFuture<'a, bool> {
        Box::pin(async move {
            // An id that is not an id names no running query.
            let Ok(id) = QueryId::parse(query_id) else {
                return Ok(false);
            };
            Ok(self.runtime.cancel(&id, CancelReason::Admin))
        })
    }
}

/// The UI served from a host process's own engine runtime (embedded mode): queries run through `runtime`
/// show up in the pages. Dropping the handle stops the server.
pub struct EmbeddedUi {
    addr: std::net::SocketAddr,
    token: String,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for EmbeddedUi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddedUi")
            .field("addr", &self.addr)
            .finish_non_exhaustive()
    }
}

impl EmbeddedUi {
    /// Binds `addr` (port 0 picks a free one) and serves the UI over `runtime` on a background thread,
    /// behind a fresh token.
    ///
    /// # Errors
    /// Returns an error when the UI session, the token, the runtime or the bind fails.
    pub fn start(runtime: Arc<EngineRuntime>, addr: std::net::SocketAddr) -> Result<Self, UiError> {
        let source = SessionSource::new(runtime)?;
        let token = crate::internal::random_token()?;
        let app = crate::internal::router_with_ui(Arc::new(source), &token);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_io()
            .enable_time()
            .build()
            .map_err(|e| UiError::new(format!("ui runtime: {e}")))?;
        let listener = std::net::TcpListener::bind(addr)
            .and_then(|l| l.set_nonblocking(true).map(|()| l))
            .map_err(|e| UiError::new(format!("bind {addr}: {e}")))?;
        let bound = listener
            .local_addr()
            .map_err(|e| UiError::new(format!("local address: {e}")))?;
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("krishiv-ui".into())
            .spawn(move || {
                let _ = rt.block_on(async {
                    let Ok(listener) = tokio::net::TcpListener::from_std(listener) else {
                        return Ok(());
                    };
                    axum::serve(listener, app)
                        .with_graceful_shutdown(async {
                            let _ = rx.await;
                        })
                        .await
                });
            })
            .map_err(|e| UiError::new(format!("ui thread: {e}")))?;
        Ok(Self {
            addr: bound,
            token,
            shutdown: Some(tx),
            thread: Some(thread),
        })
    }

    /// The address the server is bound to.
    #[must_use]
    pub fn addr(&self) -> std::net::SocketAddr {
        self.addr
    }

    /// The URL to open, with the token.
    #[must_use]
    pub fn url(&self) -> String {
        format!("http://{}/?token={}", self.addr, self.token)
    }
}

impl Drop for EmbeddedUi {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, LargeStringArray,
        StringArray, StringViewArray, TimestampMicrosecondArray,
    };
    use arrow_schema::{Field, Schema};
    use serde_json::json;

    use super::*;

    /// Every column type a system table uses, with a null in the second row of each.
    #[test]
    fn every_column_type_converts_and_nulls_stay_null() {
        let cols: Vec<(&str, ArrayRef)> = vec![
            ("utf8", Arc::new(StringArray::from(vec![Some("a"), None]))),
            (
                "large",
                Arc::new(LargeStringArray::from(vec![Some("b"), None])),
            ),
            (
                "view",
                Arc::new(StringViewArray::from(vec![Some("c"), None])),
            ),
            ("flag", Arc::new(BooleanArray::from(vec![Some(true), None]))),
            ("int", Arc::new(Int32Array::from(vec![Some(7), None]))),
            (
                "long",
                Arc::new(Int64Array::from(vec![Some(i64::MAX), None])),
            ),
            (
                "double",
                Arc::new(Float64Array::from(vec![Some(1.5), None])),
            ),
            (
                "nan",
                Arc::new(Float64Array::from(vec![Some(f64::NAN), None])),
            ),
            (
                "time",
                Arc::new(TimestampMicrosecondArray::from(vec![
                    Some(1_700_000_000_000_000),
                    None,
                ])),
            ),
        ];
        let schema = Schema::new(
            cols.iter()
                .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
                .collect::<Vec<_>>(),
        );
        let batch =
            RecordBatch::try_new(Arc::new(schema), cols.into_iter().map(|(_, a)| a).collect())
                .unwrap();
        let rows = batches_to_rows(&[batch]);
        assert_eq!(rows.len(), 2);
        assert_eq!(
            Value::Object(rows[0].clone()),
            json!({
                "utf8": "a", "large": "b", "view": "c", "flag": true, "int": 7,
                "long": i64::MAX, "double": 1.5, "nan": null, "time": 1_700_000_000_000_000_i64
            })
        );
        assert!(rows[1].values().all(Value::is_null), "{:?}", rows[1]);
    }

    /// A column of a type the pages never show (a map, a list) becomes null instead of failing the table.
    #[test]
    fn a_type_the_pages_do_not_show_is_null() {
        use arrow_array::builder::{Int32Builder, ListBuilder};
        let mut b = ListBuilder::new(Int32Builder::new());
        b.values().append_value(1);
        b.append(true);
        let list: ArrayRef = Arc::new(b.finish());
        let schema = Schema::new(vec![Field::new(
            "pipelines",
            list.data_type().clone(),
            true,
        )]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![list]).unwrap();
        assert_eq!(batches_to_rows(&[batch])[0]["pipelines"], Value::Null);
    }
}
