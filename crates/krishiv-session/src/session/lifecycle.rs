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
use crate::runtime::{Attached, EngineRuntime};

impl Session {
    /// A new session on the same runtime, run by the same principal under
    /// the same access policy, Spark profile and function registry: fresh
    /// temporary views, variables, UDFs, tags and configuration (the
    /// runtime's, without this session's changes), as Spark's
    /// `newSession()`.
    ///
    /// ```
    /// let a = krishiv_session::Session::builder()
    ///     .config("spark.sql.caseSensitive", "true")
    ///     .build()?;
    /// a.conf().set("spark.sql.ansi.enabled", "false")?;
    /// let b = a.new_session();
    /// assert!(std::sync::Arc::ptr_eq(a.runtime(), b.runtime()));
    /// assert_eq!(b.conf().get("spark.sql.ansi.enabled").as_deref(), Some("true"));
    /// assert_eq!(b.conf().get("spark.sql.caseSensitive").as_deref(), Some("true"));
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
        // The runtime's starting configuration, validated when it was built
        // (for a private runtime, the builder's configuration).
        let config = Config::clone(&runtime.config);
        Self::on_runtime(runtime, config, parts)
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
        // Under the runtime's registry lock: a query that registers after
        // this is refused there (`EngineRuntime::track`), so none slips
        // between the flag and the cancellation.
        self.state.runtime.cancel_session(self.state.number)
    }

    /// The error a closed session answers with.
    pub(crate) fn closed_error(&self) -> KrishivError {
        krishiv_common::error::InvalidHandleSessionClosed {
            handle: self.state.queries.session_id().to_owned(),
        }
        .into()
    }

    /// `INVALID_HANDLE.SESSION_CLOSED` once [`Self::close`] ran.
    pub(crate) fn check_open(&self) -> Result<(), KrishivError> {
        if self.state.closed.load(Ordering::Acquire) {
            return Err(self.closed_error());
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
    /// not a UUID; `INVALID_HANDLE.OPERATION_ALREADY_EXISTS` while another
    /// statement of this session holds the id (its DataFrame or a result
    /// stream of it is alive); `INVALID_HANDLE.OPERATION_ABANDONED` within
    /// 30 minutes of the id's last use ([`crate::QueryOptions::operation_id`]).
    pub async fn sql_with(
        &self,
        query: &str,
        options: crate::QueryOptions,
    ) -> Result<DataFrame, KrishivError> {
        self.check_open()?;
        let operation = match options.operation_id.as_deref() {
            None => None,
            Some(text) => {
                let id = crate::QueryId::parse(text)?;
                let taken = self
                    .state
                    .queries
                    .operations()
                    .take(id, crate::system::now_ms())?;
                Some(Arc::new(taken))
            }
        };
        let handle = Session {
            state: Arc::clone(&self.state),
            locals: Arc::clone(&self.locals),
            operation,
            deadline: options.deadline.or(self.deadline),
        };
        handle.sql(query).await
    }

    /// The operation id of the statement this handle runs, if the caller
    /// gave one.
    pub(crate) fn operation_id(&self) -> Option<&crate::QueryId> {
        self.operation.as_deref().map(crate::history::Operation::id)
    }

    /// Attaches the query reserved for row `id` to the runtime's registry,
    /// with its time limit: the shorter of `krishiv.query.maxExecutionTime`
    /// and the statement's [`crate::QueryOptions::deadline`]. A cancellation
    /// that arrived before the query's context existed is applied now.
    ///
    /// # Errors
    /// `INVALID_HANDLE.SESSION_CLOSED` when the session closed meanwhile.
    pub(crate) fn attach(
        &self,
        query: &Arc<krishiv_exec::QueryContext>,
        id: &str,
        config: &Config,
    ) -> Result<(), KrishivError> {
        let configured = config
            .get(&krishiv_common::config::QUERY_MAX_EXECUTION_TIME)
            .and_then(krishiv_common::config::ConfigValue::as_duration);
        let limit = match (configured, self.deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let runtime = &self.state.runtime;
        match runtime.attach(query.id().0, query, limit) {
            Attached::Run => Ok(()),
            Attached::Closed => Err(self.closed_error()),
            Attached::Cancelled(reason) => {
                crate::cancel::apply(
                    query,
                    reason,
                    None,
                    &runtime.history,
                    self.state.queries.session_id(),
                    id,
                );
                Ok(())
            }
        }
    }

    /// Cancels the running query `id` of this session (Spark Connect's
    /// `Interrupt` by operation id): its stream ends with
    /// `OPERATION_CANCELED` at its next batch boundary, and
    /// `system.queries.cancel_reason` shows `user`. Whether a running query
    /// was cancelled: `false` for an unknown id, a query that ended or one
    /// of another session.
    ///
    /// ```
    /// let s = krishiv_session::Session::new()?;
    /// let id = krishiv_session::QueryId::parse("00112233-4455-6677-8899-aabbccddeeff")?;
    /// assert!(!s.cancel(&id));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn cancel(&self, id: &crate::QueryId) -> bool {
        let number = self.state.number;
        !self
            .state
            .runtime
            .cancel_where(crate::CancelReason::User, |s, q| {
                s == number && q == id.as_str()
            })
            .is_empty()
    }

    /// Cancels every running query of this session (Spark Connect's
    /// `Interrupt` of type `ALL`) and returns their ids.
    pub fn interrupt_all(&self) -> Vec<crate::QueryId> {
        let number = self.state.number;
        self.state
            .runtime
            .cancel_where(crate::CancelReason::User, |s, _| s == number)
    }
}
