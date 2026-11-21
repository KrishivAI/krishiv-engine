// SPDX-License-Identifier: Apache-2.0
//! The console of one embedded session (ADR 0024): [`EmbeddedSource`] and [`EmbeddedUi::for_session`].
//!
//! A host process that calls `ui()` on its session gets the same pages `krishiv server` serves, limited to
//! that session:
//!
//! - **Queries, History, a query's profile, Compare:** the session's own queries only. Another session of the
//!   same runtime does not exist for this console.
//! - **Cancel:** the session's own running queries only, as the session's own cancel (`OPERATION_CANCELED`).
//! - **SQL editor:** runs in the session itself: it sees the session's temporary views, settings and
//!   variables, and what it creates is there for the host program.
//! - **Catalog:** what the session sees, its temporary views included. The statements that list it run in a copy
//!   of the session, under the session's principal: they are not in the console's lists, but they are rows of
//!   `system.queries` for that principal, which the host program sees if it reads the table itself.
//! - **Environment:** the session's settings.
//! - **Resources and Health:** this process (its memory pools without other sessions' queries, its caches).
//!
//! The server binds the loopback interface only, on a port the system picks unless one is asked for, behind one
//! fresh token and no roles; it stops when the handle is dropped and by itself when the session is closed.

use std::collections::HashSet;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;

use krishiv_session::{QueryId, Session};
use serde_json::Value;

use crate::guard::Caller;
use crate::internal::{Row, SystemTable, UiError, UiFuture, UiSource};
use crate::session_source::{EmbeddedUi, SessionSource, catalog_rows};

/// The UI's data source over one session of a host process.
pub struct EmbeddedSource {
    /// Reads the system tables (with a session of the UI's own) and runs the editor in [`Self::session`].
    inner: SessionSource,
    session: Session,
}

impl std::fmt::Debug for EmbeddedSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddedSource")
            .field("session", &self.session.id())
            .finish_non_exhaustive()
    }
}

impl EmbeddedSource {
    /// A source showing `session` and nothing else of its runtime.
    ///
    /// # Errors
    /// Returns an error when the UI's own session cannot be built.
    pub fn new(session: &Session) -> Result<Self, UiError> {
        Ok(Self {
            inner: SessionSource::over_session(session)?,
            session: session.clone(),
        })
    }

    /// Refuses once the session is closed: nothing of a closed session is served.
    fn open(&self) -> Result<(), UiError> {
        if self.session.is_closed() {
            return Err(UiError::new("the session is closed"));
        }
        Ok(())
    }

    /// The session's own rows of `system.queries`.
    async fn own_queries(&self) -> Result<Vec<Row>, UiError> {
        let mut rows = self.inner.rows(SystemTable::Queries).await?;
        let id = self.session.id();
        rows.retain(|r| r.get("session_id").and_then(Value::as_str) == Some(id));
        Ok(rows)
    }

    /// The ids of the session's own queries.
    async fn own_ids(&self) -> Result<HashSet<String>, UiError> {
        Ok(self
            .own_queries()
            .await?
            .iter()
            .filter_map(|r| r.get("query_id").and_then(Value::as_str))
            .map(str::to_owned)
            .collect())
    }
}

/// Whether `row` names one of the queries `ids`.
fn of_query(row: &Row, ids: &HashSet<String>) -> bool {
    row.get("query_id")
        .and_then(Value::as_str)
        .is_some_and(|q| ids.contains(q))
}

impl UiSource for EmbeddedSource {
    fn mode(&self) -> &'static str {
        "embedded"
    }

    fn session_id(&self) -> Option<String> {
        Some(self.session.id().to_owned())
    }

    fn rows(&self, table: SystemTable) -> UiFuture<'_, Vec<Row>> {
        Box::pin(async move {
            self.open()?;
            match table {
                SystemTable::Queries => self.own_queries().await,
                SystemTable::Operators => {
                    let ids = self.own_ids().await?;
                    let mut rows = self.inner.rows(table).await?;
                    rows.retain(|r| of_query(r, &ids));
                    Ok(rows)
                }
                // The process's pools, without the pools of other sessions' queries.
                SystemTable::Memory => {
                    let ids = self.own_ids().await?;
                    let mut rows = self.inner.rows(table).await?;
                    rows.retain(|r| {
                        r.get("query_id").is_none_or(Value::is_null) || of_query(r, &ids)
                    });
                    Ok(rows)
                }
                SystemTable::Caches => self.inner.rows(table).await,
                // The one session the console shows; the groups are a server's.
                SystemTable::Sessions => {
                    let id = self.session.id();
                    let mut rows = self.inner.rows(table).await?;
                    rows.retain(|r| r.get("session_id").and_then(Value::as_str) == Some(id));
                    Ok(rows)
                }
                SystemTable::ResourceGroups => Ok(Vec::new()),
            }
        })
    }

    fn plan_pending(&self, query: &Row) -> bool {
        self.inner.plan_pending(query)
    }

    fn plan<'a>(&'a self, query: &'a Row) -> UiFuture<'a, Option<Value>> {
        // `query` is a row this source listed, so one of the session's own.
        self.inner.plan(query)
    }

    fn cancel<'a>(&'a self, query_id: &'a str) -> UiFuture<'a, bool> {
        Box::pin(async move {
            self.open()?;
            // An id that is not an id names no running query; neither does a query of another session.
            let Ok(id) = QueryId::parse(query_id) else {
                return Ok(false);
            };
            Ok(self.session.cancel(&id))
        })
    }

    fn catalog<'a>(&'a self, path: &'a [String]) -> UiFuture<'a, Vec<Row>> {
        Box::pin(async move {
            self.open()?;
            // A copy of the session sees the same catalogs, temporary views and current schema, and the
            // statements that list them are recorded under the copy, not in the session's own history.
            let copy = self
                .session
                .clone_session()
                .map_err(|e| UiError::new(e.to_string()))?;
            let rows = catalog_rows(&copy, path).await;
            copy.close();
            rows
        })
    }

    fn environment(&self) -> UiFuture<'_, Vec<(String, String)>> {
        Box::pin(async move {
            self.open()?;
            Ok(self.session.conf().entries())
        })
    }

    fn sql_start<'a>(&'a self, sql: &'a str, caller: Option<&'a Caller>) -> UiFuture<'a, String> {
        Box::pin(async move {
            self.open()?;
            self.inner.sql_start(sql, caller).await
        })
    }

    fn sql_status<'a>(
        &'a self,
        run_id: &'a str,
        caller: Option<&'a Caller>,
    ) -> UiFuture<'a, Option<Value>> {
        self.inner.sql_status(run_id, caller)
    }

    fn sql_cancel<'a>(&'a self, run_id: &'a str, caller: Option<&'a Caller>) -> UiFuture<'a, bool> {
        self.inner.sql_cancel(run_id, caller)
    }
}

impl EmbeddedUi {
    /// Serves the console of `session` on the loopback interface, behind a fresh token, on a background thread.
    /// `port` 0 picks a free port. The server stops when the handle is dropped, and by itself soon after
    /// `session` is closed.
    ///
    /// # Errors
    /// Returns an error when the session is closed, or the UI session, the token, the server's runtime or the
    /// bind fails.
    pub fn for_session(session: &Session, port: u16) -> Result<Self, UiError> {
        let source = EmbeddedSource::new(session)?;
        source.open()?;
        let followed = session.clone();
        Self::serve(
            Arc::new(source),
            SocketAddr::from((Ipv4Addr::LOCALHOST, port)),
            Some(Box::new(move || followed.is_closed())),
        )
    }
}
