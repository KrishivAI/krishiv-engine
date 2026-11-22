// SPDX-License-Identifier: Apache-2.0
//! [`SessionSource`]: the UI backend's data from an engine runtime (embedded mode, P2-T133).
//!
//! Rows come from `SELECT * FROM system.<table>` run by a session of the UI's own, with the principal
//! [`UI_PRINCIPAL`]. That session's queries are recorded like any other, so they are left out of every list the
//! UI serves (and, history being capped per principal, they can only push out each other, never a user's rows).
//! Only the columns a page can show are converted: text, integers, floats, booleans and timestamps (as
//! microseconds since the epoch) and maps of those (a query's tags); lists become `null`.

use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::KrishivError;
use krishiv_common::principal::{Principal, PrincipalKind};
use krishiv_session::{CancelReason, EngineRuntime, QueryId, Session};
use serde_json::{Number, Value};

use krishiv_session::auth::{Authenticated, Role};

use crate::guard::Caller;
use crate::internal::{Row, SystemTable, UiError, UiFuture, UiSource};

/// The principal of the UI's own session; its queries are hidden from the UI.
pub const UI_PRINCIPAL: &str = "krishiv-ui";

/// Builds the session a statement of an authenticated caller runs in: the caller's principal and policy, as a
/// Spark Connect session of that caller gets (`krishiv_connect_server::role_sessions`).
pub type CallerSessions = dyn Fn(&Authenticated) -> Result<Session, KrishivError> + Send + Sync;

/// The UI's data source over one engine runtime.
pub struct SessionSource {
    session: Session,
    /// Where the editor's statements of a caller the guard did not identify run (a server with no
    /// authentication, whose start-up token is all-powerful): an ordinary session of the runtime.
    anonymous: Session,
    runtime: Arc<EngineRuntime>,
    runs: Arc<Mutex<Runs>>,
    /// How an identified caller's session is built; without it such a caller's statement is refused.
    caller_sessions: Option<Arc<CallerSessions>>,
    /// One session per caller, so `SET` and `USE` of one user never reach another.
    sessions: Mutex<std::collections::HashMap<String, (Option<Role>, Session)>>,
    /// The queries the history store held when the source was built ([`Self::with_stored_history`]): what ran
    /// before the server's last start, which the runtime's own tables no longer list.
    stored: Option<ProfilesSource>,
    /// Whether [`Self::anonymous`] is the one session the console shows ([`Self::over_session`]): a cancel from
    /// the editor is then that session's own.
    own_session: bool,
}

/// How long after a query ended its plan is still expected from the history writer.
const PLAN_WAIT_MICROS: i64 = 10_000_000;

/// Stored queries a server's history shows beside its own (the newest; the History page lists at most 1,000).
const MAX_STORED_QUERIES: usize = 1000;

/// Rows an editor result keeps.
const MAX_RESULT_ROWS: usize = 1000;
/// Editor runs remembered (the oldest finished one is forgotten first; a run still running is never).
const MAX_RUNS: usize = 32;
/// Statements one caller may have running in the editor at once.
const MAX_RUNNING_PER_CALLER: usize = 4;

/// The editor's runs, oldest first.
#[derive(Default)]
struct Runs {
    order: std::collections::VecDeque<String>,
    by_id: std::collections::HashMap<String, Arc<Mutex<Run>>>,
}

struct Run {
    tag: String,
    /// The caller the run belongs to (`None` for an unidentified one).
    owner: Option<String>,
    /// The principal the statement's query is recorded under.
    principal: Option<Principal>,
    cancel_requested: bool,
    state: &'static str,
    outcome: Option<Value>,
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
        let anonymous = Session::builder()
            .runtime(Arc::clone(&runtime))
            .build()
            .map_err(ui_error)?;
        Ok(Self {
            session,
            anonymous,
            runtime,
            runs: Arc::new(Mutex::new(Runs::default())),
            caller_sessions: None,
            sessions: Mutex::new(std::collections::HashMap::new()),
            stored: None,
            own_session: false,
        })
    }

    /// A source whose editor runs in `session` itself (the console of one embedded session,
    /// [`crate::embedded::EmbeddedSource`]): the rows are still read by a session of the UI's own on the same
    /// runtime, so reading them adds nothing to `session`'s history.
    ///
    /// # Errors
    /// Returns an error when the UI's session cannot be built.
    pub(crate) fn over_session(session: &Session) -> Result<Self, UiError> {
        let mut source = Self::new(Arc::clone(session.runtime()))?;
        source.anonymous = session.clone();
        source.own_session = true;
        Ok(source)
    }

    /// The same, running the editor's statements of an identified caller in the session `factory` builds for it
    /// (its principal and its policy: what a Spark Connect session of the caller gets).
    #[must_use]
    pub fn with_caller_sessions(mut self, factory: Arc<CallerSessions>) -> Self {
        self.caller_sessions = Some(factory);
        self
    }

    /// The same, also listing the newest of `stored`: what the history store (`krishiv.history.path`,
    /// [`krishiv_session::read_profile_history`], oldest first) held when the server started, so a server's
    /// history survives a restart (ADR 0024, the modes table). What finishes later is in the runtime's own
    /// tables; a query in both is shown as the runtime has it.
    #[must_use]
    pub fn with_stored_history(mut self, stored: &[krishiv_session::StoredProfile]) -> Self {
        // The console's own reads are stored like any query (about one a second while a page is open): they
        // are dropped before the cut, or they would take every place in it.
        let users = newest_per_query(stored.iter().filter(|p| !of_console(p)));
        let newest = &users[users.len().saturating_sub(MAX_STORED_QUERIES)..];
        self.stored = Some(ProfilesSource::of(newest.iter().copied()));
        self
    }

    /// `rows` of `table` followed by the stored rows of the queries `live` does not name. `live` is the ids of
    /// the runtime's own queries, whichever table is read: a running query has no operator rows yet, and its
    /// id must still keep an earlier run's stored operators out.
    fn with_stored(
        &self,
        table: SystemTable,
        mut rows: Vec<Row>,
        live: &std::collections::HashSet<String>,
    ) -> Vec<Row> {
        let stored = match (&self.stored, table) {
            (Some(s), SystemTable::Queries) => &s.queries,
            (Some(s), SystemTable::Operators) => &s.operators,
            _ => return rows,
        };
        rows.extend(
            stored
                .iter()
                .filter(|r| query_id(r).is_none_or(|i| !live.contains(i)))
                .cloned(),
        );
        rows
    }

    /// The ids of the runtime's own queries, the console's reads among them (a stored row of theirs is never
    /// listed). Read only by a source with a stored history.
    async fn live_ids(&self) -> Result<std::collections::HashSet<String>, UiError> {
        let sql = "SELECT query_id FROM system.queries";
        let df = self.session.sql_boxed(sql).await.map_err(ui_error)?;
        let batches = df.collect_boxed().await.map_err(ui_error)?;
        Ok(batches_to_rows(&batches)
            .iter()
            .filter_map(|r| query_id(r).map(str::to_owned))
            .collect())
    }

    /// The ids of the queries session `session_id` ran, read with one narrow statement.
    pub(crate) async fn query_ids_of(
        &self,
        session_id: &str,
    ) -> Result<std::collections::HashSet<String>, UiError> {
        let sql = "SELECT query_id, session_id FROM system.queries";
        let df = self.session.sql_boxed(sql).await.map_err(ui_error)?;
        let batches = df.collect_boxed().await.map_err(ui_error)?;
        Ok(batches_to_rows(&batches)
            .iter()
            .filter(|r| r.get("session_id").and_then(Value::as_str) == Some(session_id))
            .filter_map(|r| query_id(r).map(str::to_owned))
            .collect())
    }

    /// The session `caller` runs statements in, built on first use.
    fn session_of(&self, caller: &Caller) -> Result<(String, Session), UiError> {
        let Some(who) = authenticated(caller) else {
            return Err(UiError::new(
                "the caller's identity is not one this source knows",
            ));
        };
        let Some(factory) = &self.caller_sessions else {
            return Err(UiError::new(
                "no session policy is configured for the editor",
            ));
        };
        let key = self.owner_key(caller).unwrap_or_default();
        let mut sessions = lock(&self.sessions);
        // A session's policy is fixed for its life: a caller whose role differs from the one its session was built
        // with (a group claim that moved) gets a new session, as the Connect session manager gives. So does a
        // caller whose session was ended (the Sessions page, an idle sweep): the editor keeps working.
        if let Some((role, s)) = sessions.get(&key)
            && *role == who.role
            && !s.is_closed()
        {
            return Ok((key, s.clone()));
        }
        let session = factory(who).map_err(ui_error)?;
        sessions.insert(key.clone(), (who.role, session.clone()));
        Ok((key, session))
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Largest magnitude a JavaScript number holds exactly.
const JS_SAFE_INT: u64 = 9_007_199_254_740_992;

/// A cell of an editor result: the system-table conversion, with timestamps as text (the system tables keep
/// epoch microseconds, an editor shows a time), integers a browser cannot hold exactly as their decimal text,
/// and the display text for the types the conversion leaves null.
fn result_cell(column: &dyn Array, row: usize) -> Value {
    if column.is_null(row) {
        return Value::Null;
    }
    let text = |c: &dyn Array| {
        arrow_cast::display::array_value_to_string(c, row).map_or(Value::Null, Value::String)
    };
    match column.data_type() {
        // Nested values keep the display text an editor always showed; only the system tables render them as JSON.
        DataType::Timestamp(..)
        | DataType::Date32
        | DataType::Date64
        | DataType::List(_)
        | DataType::LargeList(_)
        | DataType::Struct(_) => text(column),
        DataType::Int64
            if column
                .as_primitive::<arrow_array::types::Int64Type>()
                .value(row)
                .unsigned_abs()
                > JS_SAFE_INT =>
        {
            text(column)
        }
        _ => match cell(column, row) {
            Value::Null => text(column),
            v => v,
        },
    }
}

/// The editor's result: `columns` come from the result's schema, so an empty result keeps them.
fn result_json(columns: &[String], batches: &[RecordBatch]) -> Value {
    let mut rows = Vec::new();
    let mut truncated = false;
    'outer: for b in batches {
        for r in 0..b.num_rows() {
            if rows.len() >= MAX_RESULT_ROWS {
                truncated = true;
                break 'outer;
            }
            rows.push(Value::Array(
                (0..b.num_columns())
                    .map(|c| result_cell(b.column(c).as_ref(), r))
                    .collect(),
            ));
        }
    }
    serde_json::json!({"columns": columns, "rows": rows, "truncated": truncated})
}

/// An identifier as a quoted SQL name.
fn quote(n: &str) -> String {
    format!("`{}`", n.replace('`', "``"))
}

/// A result the editor shows: its column names and the (capped) batches.
type RunOutput = (Vec<String>, Vec<RecordBatch>);

/// Runs one statement on a task of its own, so a panic ends the run as `failed` instead of leaving it running
/// (and counted against its caller's cap) for ever.
async fn run_to_end(
    editor: Session,
    runtime: Arc<EngineRuntime>,
    run: Arc<Mutex<Run>>,
    sql: String,
    tag: String,
) {
    supervise(&run, execute(editor, Arc::clone(&run), sql, tag)).await;
    drop(runtime);
}

/// Awaits `statement` on a task of its own and records its end in `run`; a panic is a failed run.
async fn supervise(
    run: &Mutex<Run>,
    statement: impl std::future::Future<Output = Result<Option<RunOutput>, KrishivError>>
    + Send
    + 'static,
) {
    let Ok(result) = tokio::spawn(statement).await else {
        let mut r = lock(run);
        r.state = "failed";
        r.outcome = Some(serde_json::json!({
            "error": "the statement stopped unexpectedly",
            "error_class": "INTERNAL_ERROR",
            "sql_state": "XX000",
        }));
        return;
    };
    finish(run, result);
}

async fn execute(
    editor: Session,
    run: Arc<Mutex<Run>>,
    sql: String,
    tag: String,
) -> Result<Option<RunOutput>, KrishivError> {
    // A cancel that came before the statement started: do not run it at all (a write would otherwise commit).
    if lock(&run).cancel_requested {
        return Ok(None);
    }
    let df = editor
        .sql_with_boxed(
            &sql,
            krishiv_session::QueryOptions::new().tags([tag.as_str()]),
        )
        .await?;
    // A cancel that came before the query was registered has nothing to stop yet: honour it now.
    if lock(&run).cancel_requested {
        return Ok(None);
    }
    // One row past the cap tells `result_json` the result was cut; the rest is never materialised.
    let capped = df.limit(i32::try_from(MAX_RESULT_ROWS + 1).unwrap_or(i32::MAX));
    let work = Box::pin(async {
        let columns: Vec<String> = capped
            .schema_boxed()
            .await?
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        capped.collect_boxed().await.map(|b| Some((columns, b)))
    });
    // A cancel that reached the run after the last check above may find no registered query to stop
    // (`cancel_by_tag` sees only attached queries): watch the flag while the statement runs and drop the work when
    // it is set, so an accepted cancel is never reported as a finished run.
    let watch = Box::pin(async {
        while !lock(&run).cancel_requested {
            tokio::time::sleep(CANCEL_POLL).await;
        }
    });
    match futures::future::select(work, watch).await {
        futures::future::Either::Left((result, _)) => result,
        futures::future::Either::Right(((), _)) => Ok(None),
    }
}

/// How often a running editor statement looks for a cancel that found no registered query to stop.
const CANCEL_POLL: std::time::Duration = std::time::Duration::from_millis(50);

fn finish(run: &Mutex<Run>, result: Result<Option<RunOutput>, KrishivError>) {
    let mut r = lock(run);
    match result {
        Ok(Some((columns, batches))) => {
            r.state = "finished";
            r.outcome = Some(result_json(&columns, &batches));
        }
        Ok(None) => {
            r.state = "cancelled";
            r.outcome = Some(
                serde_json::json!({"error": "cancelled", "error_class": "OPERATION_CANCELED"}),
            );
        }
        Err(e) => {
            let class = e.error_class().name();
            let cancelled = matches!(class, "OPERATION_CANCELED" | "KRISHIV_QUERY_KILLED");
            r.state = if cancelled { "cancelled" } else { "failed" };
            r.outcome = Some(serde_json::json!({
                "error": krishiv_common::redact::redact(e.message()).into_owned(),
                "error_class": class,
                "sql_state": e.sql_state(),
            }));
        }
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
        SystemTable::Sessions => "sessions",
        SystemTable::ResourceGroups => "resource_groups",
        SystemTable::PartitionAdvice => "partition_advice",
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
        // `system.queries.tags` is a map from a tag's key to its value.
        DataType::Map(_, _) => {
            let entries = column.as_map().value(row);
            let (keys, values) = (entries.column(0), entries.column(1));
            Value::Object(
                (0..entries.len())
                    .filter_map(|i| match cell(keys.as_ref(), i) {
                        Value::String(key) => Some((key, cell(values.as_ref(), i))),
                        _ => None,
                    })
                    .collect(),
            )
        }
        // `system.queries.hints` is a list of structs: the advisor's hints reach the query page through this.
        DataType::List(_) => list_cells(&column.as_list::<i32>().value(row)),
        DataType::LargeList(_) => list_cells(&column.as_list::<i64>().value(row)),
        DataType::Struct(fields) => {
            let st = column.as_struct();
            Value::Object(
                fields
                    .iter()
                    .zip(st.columns())
                    .map(|(f, c)| (f.name().clone(), cell(c.as_ref(), row)))
                    .collect(),
            )
        }
        _ => Value::Null,
    }
}

/// The elements of one list value as a JSON array.
fn list_cells(values: &dyn Array) -> Value {
    Value::Array((0..values.len()).map(|i| cell(values, i)).collect())
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

/// The catalog objects one level below `path` as `session` sees them ([`UiSource::catalog`]).
pub(crate) async fn catalog_rows(session: &Session, path: &[String]) -> Result<Vec<Row>, UiError> {
    // Every statement names its catalog, the built-in one included: a bare `SHOW SCHEMAS` or `default.t`
    // is read in the session's current catalog, which `USE <catalog>` changes.
    let sql = match path {
        [] => "SHOW CATALOGS".to_owned(),
        [c] => format!("SHOW SCHEMAS IN {}", quote(c)),
        [c, s] => format!("SHOW TABLES IN {}.{}", quote(c), quote(s)),
        [c, s, t] => format!("DESCRIBE TABLE {}.{}.{}", quote(c), quote(s), quote(t)),
        _ => return Ok(Vec::new()),
    };
    let batches = match collect(session, &sql).await {
        Ok(batches) => batches,
        // A temporary view is listed under every schema, but its name has one part: `c.s.v` does not name
        // it. The three-part name is tried first, so a table of the schema with the view's name is the one
        // described, and a name that is neither keeps the first statement's error. So does a table of that
        // name whose own `DESCRIBE` fails: the view is described only when the schema has no such table.
        Err(e) => match path {
            [c, s, t] if is_temporary_view(session, c, s, t).await => {
                collect(session, &format!("DESCRIBE TABLE {}", quote(t)))
                    .await
                    .map_err(ui_error)?
            }
            _ => return Err(ui_error(e)),
        },
    };
    let mut out = Vec::new();
    for r in batches_to_rows(&batches) {
        let first = |keys: &[&str]| {
            keys.iter()
                .find_map(|k| r.get(*k).and_then(Value::as_str))
                .map(str::to_owned)
        };
        let Some(name) = first(&["catalog", "tableName", "namespace", "col_name"]) else {
            continue;
        };
        // `DESCRIBE` ends its column list with blank and `# ...` lines about partitions and metadata.
        if path.len() == 3 && (name.is_empty() || name.starts_with('#')) {
            break;
        }
        let mut row = Row::new();
        row.insert("name".into(), Value::String(name));
        if let Some(t) = first(&["data_type"]) {
            row.insert("data_type".into(), Value::String(t));
        }
        if let Some(c) = first(&["comment"]) {
            row.insert("comment".into(), Value::String(c));
        }
        if let Some(Value::Bool(b)) = r.get("isTemporary") {
            row.insert("is_temporary".into(), Value::Bool(*b));
        }
        out.push(row);
    }
    Ok(out)
}

/// The batches of one statement of the catalog page.
async fn collect(session: &Session, sql: &str) -> Result<Vec<RecordBatch>, KrishivError> {
    session.sql_boxed(sql).await?.collect_boxed().await
}

/// Whether `SHOW TABLES IN c.s` lists `t` as a temporary view and as nothing else. `false` when the listing fails.
async fn is_temporary_view(session: &Session, c: &str, s: &str, t: &str) -> bool {
    let sql = format!("SHOW TABLES IN {}.{}", quote(c), quote(s));
    collect(session, &sql)
        .await
        .is_ok_and(|batches| only_temporary(&batches_to_rows(&batches), t))
}

/// Whether the `SHOW TABLES` rows `listing` name `t` as a temporary view and hold no other row of that name.
/// A table of the schema with the view's name is what the three-part name means, so its own error is the one
/// to show: the view's columns under the table's name would hide it.
fn only_temporary(listing: &[Row], t: &str) -> bool {
    let mut named = listing
        .iter()
        .filter(|r| r.get("tableName").and_then(Value::as_str) == Some(t))
        .map(|r| r.get("isTemporary") == Some(&Value::Bool(true)))
        .peekable();
    named.peek().is_some() && named.all(|temporary| temporary)
}

/// `system.queries` with the `client_type` of each query's session.
const QUERIES_WITH_CLIENT: &str = "SELECT q.*, s.client_type FROM system.queries q \
     LEFT JOIN system.sessions s ON q.session_id = s.session_id";

impl UiSource for SessionSource {
    fn rows(&self, table: SystemTable) -> UiFuture<'_, Vec<Row>> {
        Box::pin(async move {
            // A query row carries `client_type`: what the client of the query's session said it is
            // (`system.sessions`), null when the session is no longer listed. One statement reads both.
            let sql = if table == SystemTable::Queries {
                QUERIES_WITH_CLIENT.to_owned()
            } else {
                format!("SELECT * FROM system.{}", table_name(table))
            };
            let df = self.session.sql_boxed(&sql).await.map_err(ui_error)?;
            let batches = df.collect_boxed().await.map_err(ui_error)?;
            let mut rows = batches_to_rows(&batches);
            if table == SystemTable::Queries {
                // The runtime mints a session's id, so `system.sessions` holds one row per id and the join
                // answers each query once. Two sessions may run a query under one operation id (the client
                // chooses it, per session): both are listed.
                rows.retain(|r| r.get("principal").and_then(Value::as_str) != Some(UI_PRINCIPAL));
            }
            // The console's own session is not a client's. The session the editor runs in is listed (what
            // it runs shows on it) and marked: `close_session` refuses it, so the page offers no action.
            if table == SystemTable::Sessions {
                rows.retain(|r| r.get("principal").and_then(Value::as_str) != Some(UI_PRINCIPAL));
                let editor = self.anonymous.id();
                for row in &mut rows {
                    if row.get("session_id").and_then(Value::as_str) == Some(editor) {
                        row.insert("endable".to_owned(), Value::Bool(false));
                    }
                }
            }
            if self.stored.is_none()
                || !matches!(table, SystemTable::Queries | SystemTable::Operators)
            {
                return Ok(rows);
            }
            // The queries just read are the live ones; the operators table does not name a query that has
            // reported no operator yet, so its ids are read from the queries.
            let live = if table == SystemTable::Queries {
                rows.iter()
                    .filter_map(|r| query_id(r).map(str::to_owned))
                    .collect()
            } else {
                // The ids of the rows just read are added: a query that left the runtime's history between
                // the two reads is still this run's, and must keep an earlier run's operators out.
                let mut live = self.live_ids().await?;
                live.extend(rows.iter().filter_map(|r| query_id(r).map(str::to_owned)));
                live
            };
            Ok(self.with_stored(table, rows, &live))
        })
    }

    fn plan_pending(&self, query: &Row) -> bool {
        // The writer holds an entry for half a second before it writes a segment; a query that ended longer
        // ago than the window and is not in the store will not get there (trimmed, a failed write).
        let ended_us = query.get("ended_at").and_then(Value::as_i64);
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_micros()).unwrap_or(i64::MAX));
        self.runtime.keeps_profile_history()
            && ended_us.is_some_and(|ended| now_us.saturating_sub(ended) < PLAN_WAIT_MICROS)
    }

    fn plan<'a>(&'a self, query: &'a Row) -> UiFuture<'a, Option<Value>> {
        Box::pin(async move {
            let Some(query_id) = query_id(query) else {
                return Ok(None);
            };
            // The runtime's store, once the writer has reached the query. The store also holds earlier runs
            // (a server reads and writes one directory) and a client may reuse an id, so a profile is this
            // query's only if it ended after this query was submitted: until this run's profile is written
            // the store's newest under the id is an earlier run's, whose plan is not this run's operators'.
            let runtime = Arc::clone(&self.runtime);
            let id = query_id.to_owned();
            let own = tokio::task::spawn_blocking(move || runtime.stored_profile(&id))
                .await
                .map_err(|e| UiError::new(e.to_string()))?
                .map_err(ui_error)?;
            let submitted_ms = query
                .get("submitted_at")
                .and_then(Value::as_i64)
                .map_or(i64::MAX, |us| us.div_euclid(1000));
            if let Some(profile) = own.filter(|p| p.finished_at_ms >= submitted_ms) {
                return Ok(plan_of(&profile.profile));
            }
            // One of an earlier run that the source was given when it was built, unless this run has a
            // query of that id.
            match &self.stored {
                Some(stored) if stored.plans.contains_key(query_id) => {
                    if self.live_ids().await?.contains(query_id) {
                        return Ok(None);
                    }
                    Ok(stored.plans.get(query_id).cloned())
                }
                _ => Ok(None),
            }
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

    fn cancel_in_session<'a>(
        &'a self,
        query_id: &'a str,
        session_id: &'a str,
    ) -> UiFuture<'a, bool> {
        Box::pin(async move {
            let Ok(id) = QueryId::parse(query_id) else {
                return Ok(false);
            };
            Ok(self
                .runtime
                .cancel_in_session(session_id, &id, CancelReason::Admin))
        })
    }

    fn close_session<'a>(&'a self, session_id: &'a str) -> UiFuture<'a, Option<usize>> {
        Box::pin(async move {
            // The console reads through its own session, and the editor of an unidentified caller runs in one
            // shared session: ending either would break the console for everyone, so neither can be ended here.
            // Any other session ends as an operator's kill (`killed` / `admin_kill`), the caller's own editor
            // session included: the console does not know which session is the caller's.
            if session_id == self.session.id() || session_id == self.anonymous.id() {
                return Ok(None);
            }
            Ok(self.runtime.kill_session(session_id, CancelReason::Admin))
        })
    }

    fn catalog<'a>(&'a self, path: &'a [String]) -> UiFuture<'a, Vec<Row>> {
        Box::pin(catalog_rows(&self.session, path))
    }

    fn environment(&self) -> UiFuture<'_, Vec<(String, String)>> {
        Box::pin(async move { Ok(self.session.conf().entries()) })
    }

    fn sql_start<'a>(&'a self, sql: &'a str, caller: Option<&'a Caller>) -> UiFuture<'a, String> {
        Box::pin(async move {
            // The statement runs as the caller, under the caller's policy. Only the local administrator (a server
            // with no authentication, whoever holds its start-up token) runs in the ordinary session; a caller no
            // guard identified holds no rights and is refused.
            let caller = match caller {
                Some(c) if authenticated(c).is_some() => Some(c),
                Some(c)
                    if c.identity.as_deref().is_some_and(
                        <dyn std::any::Any + Send + Sync>::is::<crate::guard::LocalAdmin>,
                    ) =>
                {
                    None
                }
                Some(_) => {
                    return Err(UiError::new(
                        "the caller's identity is not one this source knows",
                    ));
                }
                None => None,
            };
            let (owner, session, principal) = match caller {
                Some(c) => {
                    let (key, session) = self.session_of(c)?;
                    (
                        Some(key),
                        session,
                        authenticated(c).map(|a| a.principal.clone()),
                    )
                }
                None => (None, self.anonymous.clone(), None),
            };
            let token = crate::internal::random_token()?;
            let id = token
                .get(..32)
                .ok_or_else(|| UiError::new("no run id"))?
                .to_owned();
            let tag = format!("{}{id}", crate::internal::EDITOR_TAG_PREFIX);
            let run = Arc::new(Mutex::new(Run {
                tag: tag.clone(),
                owner: owner.clone(),
                principal,
                cancel_requested: false,
                state: "running",
                outcome: None,
            }));
            {
                let mut runs = lock(&self.runs);
                let running_of_owner = runs
                    .by_id
                    .values()
                    .filter(|r| {
                        let r = lock(r);
                        r.state == "running" && r.owner == owner
                    })
                    .count();
                if running_of_owner >= MAX_RUNNING_PER_CALLER {
                    return Err(UiError::busy(
                        "too many statements are running; wait for one to finish or cancel it",
                    ));
                }
                // Make room by forgetting the oldest finished run. A run still running is never forgotten (it
                // could no longer be cancelled from the editor).
                while runs.order.len() >= MAX_RUNS {
                    let finished = runs.order.iter().position(|old| {
                        runs.by_id
                            .get(old)
                            .is_none_or(|r| lock(r).state != "running")
                    });
                    let Some(at) = finished else {
                        return Err(UiError::busy(
                            "the editor is busy with other statements; try again shortly",
                        ));
                    };
                    if let Some(old) = runs.order.remove(at) {
                        runs.by_id.remove(&old);
                    }
                }
                runs.order.push_back(id.clone());
                runs.by_id.insert(id.clone(), Arc::clone(&run));
            }
            tokio::spawn(run_to_end(
                session,
                Arc::clone(&self.runtime),
                run,
                sql.to_owned(),
                tag,
            ));
            Ok(id)
        })
    }

    fn sql_status<'a>(
        &'a self,
        run_id: &'a str,
        caller: Option<&'a Caller>,
    ) -> UiFuture<'a, Option<Value>> {
        Box::pin(async move {
            let Some(run) = self.run_of(run_id, caller) else {
                return Ok(None);
            };
            let r = lock(&run);
            let mut body = r.outcome.clone().unwrap_or_else(|| serde_json::json!({}));
            if let Value::Object(m) = &mut body {
                m.insert("state".into(), Value::String(r.state.to_owned()));
            }
            Ok(Some(body))
        })
    }

    fn sql_cancel<'a>(&'a self, run_id: &'a str, caller: Option<&'a Caller>) -> UiFuture<'a, bool> {
        Box::pin(async move {
            let Some(run) = self.run_of(run_id, caller) else {
                return Ok(false);
            };
            let (tag, principal) = {
                let mut r = lock(&run);
                if r.state != "running" {
                    return Ok(false);
                }
                r.cancel_requested = true;
                (r.tag.clone(), r.principal.clone())
            };
            // The caller's own query ends as a user's cancel (`OPERATION_CANCELED`), and so does a statement of
            // the embedded session, through the session's own interrupt; an unidentified caller's on a server
            // (the all-powerful start-up token) as an operator's kill, which the run reports as cancelled too.
            match principal {
                Some(p) => self.runtime.cancel_by_tag(&tag, &p, false),
                None if self.own_session => self.anonymous.interrupt_tag(&tag),
                None => {
                    self.runtime
                        .cancel_by_tag(&tag, &Principal::new("", PrincipalKind::User), true)
                }
            };
            Ok(true)
        })
    }
}

impl SessionSource {
    /// The run `run_id` when it belongs to `caller`: another caller's run does not exist for this one.
    fn run_of(&self, run_id: &str, caller: Option<&Caller>) -> Option<Arc<Mutex<Run>>> {
        let run = lock(&self.runs).by_id.get(run_id).cloned()?;
        let owner = caller.and_then(|c| self.owner_key(c));
        let mine = lock(&run).owner == owner;
        mine.then_some(run)
    }

    fn owner_key(&self, caller: &Caller) -> Option<String> {
        let who = authenticated(caller)?;
        // What a Connect session is bound to (`SessionOwner`): the method, the principal's kind, tenant and name, so
        // a static token and an OIDC subject of the same name share neither a session nor a run.
        Some(format!(
            "{:?}/{:?}/{:?}/{}/{:?}",
            who.method,
            who.principal.kind,
            who.principal.tenant,
            who.principal.name,
            who.principal.delegated_by
        ))
    }
}

/// The session guard's record of `caller`; `None` for a caller a guard let through without identifying a principal
/// (the start-up token's local administrator, an anonymous one): such a caller runs in the ordinary session.
fn authenticated(caller: &Caller) -> Option<&Authenticated> {
    caller.identity.as_deref()?.downcast_ref()
}

/// The UI over stored profiles, with no engine (`krishiv ui --profiles <dir>`, P3-T050): the pages that read the
/// system tables (queries, history, a query's operators) show what the history store holds; there is nothing live,
/// no memory or cache table, and nothing to cancel.
#[derive(Debug)]
pub struct ProfilesSource {
    queries: Vec<Row>,
    operators: Vec<Row>,
    plans: std::collections::HashMap<String, Value>,
}

impl ProfilesSource {
    /// Reads every profile under `dir` (`krishiv.history.path`'s store).
    ///
    /// # Errors
    /// Returns an error when the directory cannot be read.
    pub fn open(dir: &std::path::Path) -> Result<Self, UiError> {
        let stored = krishiv_session::read_profile_history(dir).map_err(ui_error)?;
        Ok(Self::from_profiles(&stored))
    }

    /// The source over profiles already read (oldest first, as [`krishiv_session::read_profile_history`] gives
    /// them). The profiles of a console's own reads ([`UI_PRINCIPAL`], which a server's store holds too) are left
    /// out, as they are from every list the UI serves, and a query id stored more than once is listed once, as
    /// its newest profile.
    #[must_use]
    pub fn from_profiles(stored: &[krishiv_session::StoredProfile]) -> Self {
        Self::of(newest_per_query(stored.iter().filter(|p| !of_console(p))).into_iter())
    }

    fn of<'a>(stored: impl Iterator<Item = &'a krishiv_session::StoredProfile>) -> Self {
        let mut queries = Vec::new();
        let mut operators = Vec::new();
        let mut plans = std::collections::HashMap::new();
        for p in stored {
            let doc: Value = serde_json::from_str(&p.profile).unwrap_or(Value::Null);
            let usage = &doc["usage"];
            let ended_us = p.finished_at_ms.saturating_mul(1000);
            let mut q = Row::new();
            let mut put = |k: &str, v: Value| {
                q.insert(k.to_owned(), v);
            };
            put("query_id", Value::String(p.query_id.clone()));
            put("state", Value::String(p.state.clone()));
            put("principal", Value::String(p.principal.clone()));
            put(
                "query_text",
                p.sql.clone().map_or(Value::Null, Value::String),
            );
            put(
                "submitted_at",
                Value::from(ended_us.saturating_sub(p.duration_ns / 1000)),
            );
            put("ended_at", Value::from(ended_us));
            put("wall_ns", Value::from(p.duration_ns));
            put("cpu_ns", usage["cpu_ns"].clone());
            put("peak_memory_bytes", usage["peak_memory_bytes"].clone());
            put("rows_produced", usage["rows_out"].clone());
            put("bytes_read", usage["scan"]["bytes"].clone());
            put("rows_read", usage["scan"]["rows"].clone());
            put("spill_bytes", usage["spill_bytes"].clone());
            put("workers", doc["query"]["workers"].clone());
            put(
                "error_class",
                p.error_class.clone().map_or(Value::Null, Value::String),
            );
            queries.push(q);
            if let Some(plan) = plan_in(&doc) {
                plans.insert(p.query_id.clone(), plan);
            }
            for o in doc["operators"].as_array().map_or(&[][..], Vec::as_slice) {
                let m = &o["metrics"];
                let mut r = Row::new();
                r.insert("query_id".into(), Value::String(p.query_id.clone()));
                r.insert("operator_id".into(), o["id"].clone());
                r.insert("node_id".into(), o["node"].clone());
                r.insert("name".into(), o["name"].clone());
                r.insert("role".into(), o["role"].clone());
                for (to, from) in [
                    ("drivers", "drivers"),
                    ("rows_in", "rows_in"),
                    ("rows_out", "rows_out"),
                    ("compute_ns", "compute_time_ns"),
                    ("max_driver_compute_ns", "max_driver_compute_time_ns"),
                    ("blocked_io_ns", "blocked_on_io_ns"),
                    ("blocked_memory_ns", "blocked_on_memory_ns"),
                    ("blocked_sink_ns", "blocked_on_sink_ns"),
                    ("peak_reserved_bytes", "peak_reserved_bytes"),
                    ("spills", "spills"),
                    ("spilled_bytes", "spilled_bytes"),
                ] {
                    r.insert(to.into(), m[from].clone());
                }
                operators.push(r);
            }
        }
        Self {
            queries,
            operators,
            plans,
        }
    }
}

/// The query a row of `system.queries` or `system.operators` belongs to.
fn query_id(row: &Row) -> Option<&str> {
    row.get("query_id").and_then(Value::as_str)
}

/// The plan tree of a profile document (`krishiv.query-profile` JSON text), `None` when it does not read or
/// has none.
fn plan_of(profile: &str) -> Option<Value> {
    plan_in(&serde_json::from_str(profile).ok()?)
}

/// The `plan` member of a profile document: its array of nodes.
fn plan_in(doc: &Value) -> Option<Value> {
    doc.get("plan").filter(|p| p.is_array()).cloned()
}

/// `stored` (oldest first) with one profile per query id, the newest, in the same order. A client may reuse an
/// operation id across runs of a server, so a store can hold several profiles of one id; listing them all would
/// repeat the id in the history and mix the operators of different runs on the query's page.
fn newest_per_query<'a>(
    stored: impl DoubleEndedIterator<Item = &'a krishiv_session::StoredProfile>,
) -> Vec<&'a krishiv_session::StoredProfile> {
    let mut seen = std::collections::HashSet::new();
    let mut newest: Vec<_> = stored
        .rev()
        .filter(|p| seen.insert(p.query_id.as_str()))
        .collect();
    newest.reverse();
    newest
}

/// Whether `profile` is of a query a console's own session ran.
fn of_console(profile: &krishiv_session::StoredProfile) -> bool {
    profile.principal == UI_PRINCIPAL
}

impl UiSource for ProfilesSource {
    fn mode(&self) -> &'static str {
        "profiles"
    }

    fn rows(&self, table: SystemTable) -> UiFuture<'_, Vec<Row>> {
        Box::pin(async move {
            Ok(match table {
                SystemTable::Queries => self.queries.clone(),
                SystemTable::Operators => self.operators.clone(),
                SystemTable::Memory
                | SystemTable::Caches
                | SystemTable::Sessions
                | SystemTable::ResourceGroups
                | SystemTable::PartitionAdvice => Vec::new(),
            })
        })
    }

    fn cancel<'a>(&'a self, _query_id: &'a str) -> UiFuture<'a, bool> {
        Box::pin(async { Ok(false) })
    }

    fn plan<'a>(&'a self, query: &'a Row) -> UiFuture<'a, Option<Value>> {
        Box::pin(async move { Ok(query_id(query).and_then(|id| self.plans.get(id).cloned())) })
    }
}

/// How long a stop waits for open connections before cutting them.
const SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_millis(500);

/// How often a server that follows a session asks whether the session has ended.
const ENDED_POLL: std::time::Duration = std::time::Duration::from_millis(100);

/// Answers `true` once what the server shows is gone.
pub(crate) type Ended = Box<dyn Fn() -> bool + Send>;

/// Resolves when the handle asks for a stop (or is gone) or when `ended` resolves.
async fn stopped(handle: tokio::sync::oneshot::Receiver<()>, ended: impl Future<Output = ()>) {
    tokio::select! {
        _ = handle => {}
        () = ended => {}
    }
}

/// The UI served from a host process's own engine runtime (embedded mode): queries run through `runtime`
/// show up in the pages. Dropping the handle stops the server and blocks until it has (at most about a second); do not drop it from
/// a thread the engine needs to make progress.
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
        Self::serve(Arc::new(SessionSource::new(runtime)?), addr, None)
    }

    /// [`EmbeddedUi::start`] over any source. With `ended`, the server also stops by itself soon after `ended`
    /// first answers `true` (the session it shows was closed).
    pub(crate) fn serve(
        source: Arc<dyn UiSource>,
        addr: std::net::SocketAddr,
        ended: Option<Ended>,
    ) -> Result<Self, UiError> {
        let token = crate::internal::random_token()?;
        let app = crate::internal::router_with_ui(source, &token);
        let listener = std::net::TcpListener::bind(addr)
            .and_then(|l| l.set_nonblocking(true).map(|()| l))
            .map_err(|e| UiError::new(format!("bind {addr}: {e}")))?;
        let bound = listener
            .local_addr()
            .map_err(|e| UiError::new(format!("local address: {e}")))?;
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
        // The runtime is built on the server's own thread, so a failure never drops one inside the caller's.
        let thread = std::thread::Builder::new()
            .name("krishiv-ui".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_io()
                    .enable_time()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("ui runtime: {e}")));
                        return;
                    }
                };
                let served = rt.block_on(async {
                    let listener = match tokio::net::TcpListener::from_std(listener) {
                        Ok(l) => l,
                        Err(e) => return Err(format!("ui listener: {e}")),
                    };
                    let _ = ready_tx.send(Ok(()));
                    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
                    let serve = std::future::IntoFuture::into_future(
                        axum::serve(listener, app).with_graceful_shutdown(async {
                            let _ = stop_rx.await;
                        }),
                    );
                    tokio::pin!(serve);
                    let ended = async {
                        match ended {
                            Some(ended) => {
                                while !ended() {
                                    tokio::time::sleep(ENDED_POLL).await;
                                }
                            }
                            None => std::future::pending().await,
                        }
                    };
                    tokio::select! {
                        r = &mut serve => r.map_err(|e| e.to_string()),
                        () = stopped(rx, ended) => {
                            let _ = stop_tx.send(());
                            // An open event stream never ends: give the others a moment, then cut it.
                            let _ = tokio::time::timeout(SHUTDOWN_GRACE, &mut serve).await;
                            Ok(())
                        }
                    }
                });
                // A serve error after start has no caller left to receive it.
                drop(served);
                rt.shutdown_background();
            })
            .map_err(|e| UiError::new(format!("ui thread: {e}")))?;
        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(UiError::new(e)),
            Err(_) => return Err(UiError::new("the UI thread stopped before it served")),
        }
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

    /// The token the `/api/internal` routes require.
    #[must_use]
    pub fn token(&self) -> &str {
        &self.token
    }

    /// Whether the server has stopped by itself: the session it follows was closed
    /// ([`EmbeddedUi::for_session`]), or serving failed.
    #[must_use]
    pub fn is_stopped(&self) -> bool {
        self.thread
            .as_ref()
            .is_none_or(std::thread::JoinHandle::is_finished)
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

    /// The page describes a temporary view by its one-part name only when the schema has no table of that
    /// name: a table whose `DESCRIBE` failed keeps its error.
    #[test]
    fn a_view_is_described_only_when_no_table_has_its_name() {
        let listing: Vec<Row> = [("t", false), ("t", true), ("v", true), ("u", false)]
            .into_iter()
            .map(|(name, temporary)| {
                let mut r = Row::new();
                r.insert("tableName".into(), json!(name));
                r.insert("isTemporary".into(), json!(temporary));
                r
            })
            .collect();
        assert!(only_temporary(&listing, "v"));
        assert!(!only_temporary(&listing, "t"), "a table has the name too");
        assert!(!only_temporary(&listing, "u"), "a table alone");
        assert!(!only_temporary(&listing, "missing"));
        assert!(!only_temporary(&[], "v"));
    }

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

    /// A list cell renders as a JSON array (so `system.queries.hints` reaches the query page) and a map cell as
    /// a JSON object (`system.queries.tags`); a type the pages never show (binary) becomes null instead of
    /// failing the table.
    #[test]
    fn a_list_and_a_map_render_and_a_type_the_pages_do_not_show_is_null() {
        use arrow_array::builder::{Int32Builder, ListBuilder, MapBuilder, StringBuilder};
        let mut b = ListBuilder::new(Int32Builder::new());
        b.values().append_value(1);
        b.append(true);
        let list: ArrayRef = Arc::new(b.finish());
        let mut m = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
        m.keys().append_value("k");
        m.values().append_value(1);
        m.append(true).unwrap();
        let map: ArrayRef = Arc::new(m.finish());
        let binary: ArrayRef = Arc::new(arrow_array::BinaryArray::from_vec(vec![b"x".as_slice()]));
        let schema = Schema::new(vec![
            Field::new("pipelines", list.data_type().clone(), true),
            Field::new("attrs", map.data_type().clone(), true),
            Field::new("raw", binary.data_type().clone(), true),
        ]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![list, map, binary]).unwrap();
        let row = &batches_to_rows(&[batch])[0];
        assert_eq!(row["pipelines"], json!([1]));
        assert_eq!(row["attrs"], json!({"k": 1}));
        assert_eq!(row["raw"], Value::Null);
    }

    fn new_run() -> Arc<Mutex<Run>> {
        Arc::new(Mutex::new(Run {
            tag: "t".into(),
            owner: None,
            principal: None,
            cancel_requested: false,
            state: "running",
            outcome: None,
        }))
    }

    /// A statement that panics ends its run as failed (it would otherwise stay running for ever).
    #[tokio::test]
    async fn a_panicking_statement_ends_failed() {
        let run = new_run();
        supervise(&run, async { panic!("boom") }).await;
        let r = lock(&run);
        assert_eq!(r.state, "failed");
        let outcome = r.outcome.clone().unwrap();
        assert_eq!(outcome["error_class"], "INTERNAL_ERROR");
        assert!(!outcome.to_string().contains("boom"));
    }

    /// An empty result keeps its column names.
    #[test]
    fn an_empty_result_keeps_its_columns() {
        let v = result_json(&["a".to_owned(), "b".to_owned()], &[]);
        assert_eq!(
            v,
            json!({"columns": ["a", "b"], "rows": [], "truncated": false})
        );
    }

    /// Editor timestamps are text; an Int64 a browser could not hold exactly is its decimal text, others stay numbers.
    #[test]
    fn editor_cells_render_times_as_text_and_keep_integers_exact() {
        let ts = TimestampMicrosecondArray::from(vec![Some(1_700_000_000_000_000)]);
        assert_eq!(result_cell(&ts, 0), json!("2023-11-14T22:13:20"));
        let big = Int64Array::from(vec![i64::MAX, 42, -9_007_199_254_740_993]);
        assert_eq!(result_cell(&big, 0), json!("9223372036854775807"));
        assert_eq!(result_cell(&big, 1), json!(42));
        assert_eq!(result_cell(&big, 2), json!("-9007199254740993"));
    }

    /// A list in an editor result keeps its display text (the system tables render it as an array instead).
    #[test]
    fn an_editor_list_cell_is_its_display_text() {
        use arrow_array::builder::{Int32Builder, ListBuilder};
        let mut b = ListBuilder::new(Int32Builder::new());
        b.values().append_value(1);
        b.values().append_value(2);
        b.append(true);
        let list = b.finish();
        assert_eq!(result_cell(&list, 0), json!("[1, 2]"));
        assert_eq!(cell(&list, 0), json!([1, 2]));
    }
}
