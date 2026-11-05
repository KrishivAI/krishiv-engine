// SPDX-License-Identifier: Apache-2.0
//! A session's place on its engine runtime (design P2-E1): child sessions,
//! closing, and per-statement options. A child of `session` so that it reads
//! the session's private state and keeps `session.rs` from growing
//! (`cargo xtask file-size`).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use krishiv_common::config::Config;
use krishiv_common::error::KrishivError;

use super::{Session, SessionParts};
use crate::dataframe::DataFrame;
use crate::runtime::EngineRuntime;

impl Session {
    /// A new session on the same runtime, run by the same principal under
    /// the same access policy, Spark profile and function registry: fresh
    /// temporary views, variables, UDFs, tags and configuration (the
    /// runtime's, without this session's changes), as Spark's
    /// `newSession()`.
    ///
    /// ```
    /// let a = krishiv_session::Session::new()?;
    /// a.conf().set("spark.sql.ansi.enabled", "false")?;
    /// let b = a.new_session();
    /// assert!(std::sync::Arc::ptr_eq(a.runtime(), b.runtime()));
    /// assert_eq!(b.conf().get("spark.sql.ansi.enabled").as_deref(), Some("true"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn new_session(&self) -> Self {
        let state = &self.state;
        let runtime = Arc::clone(&state.runtime);
        let parts = SessionParts {
            profile: state.profile,
            principal: Arc::clone(&state.principal),
            policy: Arc::clone(&state.policy),
            functions: state.base_functions.clone(),
        };
        // The configuration the session was built with is valid, so the
        // runtime's starting configuration is too (a private runtime starts
        // from the builder's).
        let config = runtime
            .session_config(Vec::new())
            .unwrap_or_else(|_| Config::new());
        match Self::on_runtime(runtime, config, parts) {
            Ok(s) => s,
            // Not reached: building a session on a runtime has no other
            // failure; the parent's handle is a working session either way.
            Err(_) => self.clone(),
        }
    }

    /// The runtime this session runs on.
    #[must_use]
    pub fn runtime(&self) -> &Arc<EngineRuntime> {
        &self.state.runtime
    }

    /// Closes the session (Spark Connect's `ReleaseSession`): its running
    /// queries are cancelled, so their streams end with
    /// `OPERATION_CANCELED`, and it takes no new statement
    /// (`INVALID_HANDLE.SESSION_CLOSED`). Dropping a session's handles does
    /// not cancel anything: an open result stream keeps its session alive.
    /// Returns the number of queries cancelled; closing again is a no-op.
    ///
    /// ```
    /// let s = krishiv_session::Session::new()?;
    /// s.close();
    /// let e = futures::executor::block_on(s.sql("SELECT 1")).unwrap_err();
    /// assert_eq!(e.error_class().name(), "INVALID_HANDLE.SESSION_CLOSED");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn close(&self) -> usize {
        if self.state.closed.swap(true, Ordering::AcqRel) {
            return 0;
        }
        self.state.runtime.cancel_session(self.state.number)
    }

    /// `INVALID_HANDLE.SESSION_CLOSED` once [`Self::close`] ran.
    pub(crate) fn check_open(&self) -> Result<(), KrishivError> {
        if self.state.closed.load(Ordering::Acquire) {
            return Err(krishiv_common::error::InvalidHandleSessionClosed {
                handle: self.state.queries.session_id().to_owned(),
            }
            .into());
        }
        Ok(())
    }

    /// As [`Self::sql`], with per-statement options: an operation id names
    /// the queries the statement and its DataFrame's actions run
    /// (`system.queries.query_id`), as Spark Connect's `operation_id` does.
    ///
    /// ```
    /// use krishiv_session::{QueryOptions, Session};
    /// let s = Session::new()?;
    /// let id = "00112233-4455-6677-8899-aabbccddeeff";
    /// let df = futures::executor::block_on(s.sql_with("SELECT 1", QueryOptions::new().operation_id(id)))?;
    /// futures::executor::block_on(df.collect())?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// As [`Self::sql`]; `INVALID_HANDLE.FORMAT` for an operation id that is
    /// not a UUID. A query started while another of this session runs under
    /// the same id fails with `INVALID_HANDLE.OPERATION_ALREADY_EXISTS`.
    pub async fn sql_with(
        &self,
        query: &str,
        options: crate::QueryOptions,
    ) -> Result<DataFrame, KrishivError> {
        let operation = options
            .operation_id
            .as_deref()
            .map(crate::QueryId::parse)
            .transpose()?;
        let handle = Session {
            state: Arc::clone(&self.state),
            locals: Arc::clone(&self.locals),
            operation,
        };
        handle.sql(query).await
    }
}
