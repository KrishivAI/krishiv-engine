// SPDX-License-Identifier: Apache-2.0
//! The statement an access check is asked for (ADR 0049 Update 2026-11-21, queue row 75): the
//! session sets it around a SQL statement's checks, analysis and run, so a policy provider that
//! talks to a service (`krishiv-server`'s `external`) can send the statement's class and query id
//! with its questions. A task-local: work the statement spawns onto other tasks does not see it,
//! and a check made outside a SQL statement (a DataFrame or Spark Connect plan) sees none.

use std::sync::{Arc, Mutex, PoisonError};

tokio::task_local! {
    static STATEMENT: StatementContext;
}

/// The SQL statement the current task is checking and running.
#[derive(Debug, Clone)]
pub struct StatementContext {
    class: &'static str,
    operation: Option<Arc<str>>,
    row: Arc<Mutex<Option<String>>>,
}

impl StatementContext {
    /// A statement of class `class` (`StatementClass::name`), with the caller's operation id if
    /// it gave one, and `row`, the slot the id of the `system.queries` row its query begins goes to.
    pub(crate) fn new(
        class: &'static str,
        operation: Option<&str>,
        row: Arc<Mutex<Option<String>>>,
    ) -> Self {
        Self {
            class,
            operation: operation.map(Arc::from),
            row,
        }
    }

    /// The statement's class as `system.queries.statement_class` shows it (`READ`, `WRITE`,
    /// `DDL`, `SESSION`, `ADMIN`).
    #[must_use]
    pub fn class(&self) -> &'static str {
        self.class
    }

    /// The statement's query id as `system.queries.query_id` shows it, when it is known yet: the
    /// caller's operation id (Spark Connect always sends one), else the id of the row its query
    /// already began. `None` before a statement without an operation id began its query (its
    /// first analysis comes before the row).
    #[must_use]
    pub fn query_id(&self) -> Option<String> {
        if let Some(op) = &self.operation {
            return Some(op.to_string());
        }
        self.row
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Runs `work` with `self` as the current statement; a statement inside it (`EXECUTE
    /// IMMEDIATE`, a script's statement) is current inside its own scope.
    pub(crate) async fn scope<F: Future>(self, work: F) -> F::Output {
        STATEMENT.scope(self, work).await
    }

    /// `work` under the calling task's statement, if it runs one: for a task the statement
    /// starts (`on_executor`).
    pub(crate) fn carried<F: Future>(work: F) -> impl Future<Output = F::Output> {
        match Self::current() {
            Some(context) => futures::future::Either::Left(STATEMENT.scope(context, work)),
            None => futures::future::Either::Right(work),
        }
    }

    /// The statement the current task runs, if it runs one.
    #[must_use]
    pub fn current() -> Option<Self> {
        STATEMENT.try_with(Clone::clone).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn outside_a_statement_there_is_none() {
        assert!(StatementContext::current().is_none());
    }

    #[tokio::test]
    async fn the_operation_id_comes_before_the_row() {
        let row = Arc::new(Mutex::new(None));
        let ctx = StatementContext::new("WRITE", None, Arc::clone(&row));
        let seen = ctx
            .scope(async {
                let first = StatementContext::current().and_then(|c| c.query_id());
                *row.lock().unwrap_or_else(PoisonError::into_inner) = Some("r1".to_owned());
                let current = StatementContext::current();
                (
                    first,
                    current.as_ref().map(StatementContext::class),
                    current.and_then(|c| c.query_id()),
                )
            })
            .await;
        assert_eq!(seen, (None, Some("WRITE"), Some("r1".to_owned())));
        let ctx = StatementContext::new(
            "READ",
            Some("op-1"),
            Arc::new(Mutex::new(Some("r2".to_owned()))),
        );
        assert_eq!(ctx.query_id().as_deref(), Some("op-1"));
    }

    #[tokio::test]
    async fn an_inner_statement_is_current_inside_its_scope() {
        let outer = StatementContext::new("ADMIN", None, Arc::new(Mutex::new(None)));
        let classes = outer
            .scope(async {
                let inner = StatementContext::new("DDL", None, Arc::new(Mutex::new(None)));
                let inside = inner
                    .scope(async { StatementContext::current().map(|c| c.class()) })
                    .await;
                (inside, StatementContext::current().map(|c| c.class()))
            })
            .await;
        assert_eq!(classes, (Some("DDL"), Some("ADMIN")));
    }
}
