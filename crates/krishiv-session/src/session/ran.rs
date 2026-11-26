// SPDX-License-Identifier: Apache-2.0
//! Running one SQL statement on the calling task ([`Session::sql_here_marked`]) and [`Ran`], what it did with it
//! (with `queries_only`, P3-T101 Flight SQL refuses a command at GetFlightInfo).

use super::*;

tokio::task_local! {
    /// The command this task is running, while it runs.
    static COMMAND: Arc<Command>;
    /// Set while the task runs a query that evaluates an expression of its command (a variable's value).
    static EVALUATION: ();
}

/// A statement that is not a query, for the queries it runs itself (an `INSERT`'s write): they are begun with
/// no statement of their own, and `system.queries` lists them under the command's text and class.
struct Command {
    text: Arc<str>,
    class: &'static str,
    /// Whether a query took the text: the command has its row.
    listed: std::sync::atomic::AtomicBool,
    /// The statement's slot for the id of the row a query of it began (`sql_ran` adds no refused row then).
    row: Arc<std::sync::Mutex<Option<String>>>,
    /// A statement of a script. Its queries carry the script's text (the text their error positions
    /// index), and are listed under the statement's own.
    of_script: bool,
}

/// The command a query is listed under.
pub(crate) struct Listing {
    pub(crate) text: Arc<str>,
    pub(crate) class: &'static str,
    /// Where the id of the query's row goes: the query is the statement's listed run. `None` for a query
    /// that evaluates an expression of the command: the command may still fail after it finished, so it
    /// becomes the listed run only by failing ([`evaluation_failed`]).
    pub(crate) row: Option<Arc<std::sync::Mutex<Option<String>>>>,
}

/// `work` under the command the calling task is running, if any: a task the command starts for a query of
/// its own (`on_executor`) is still the command's.
pub(super) fn in_running_command<F: Future>(work: F) -> impl Future<Output = F::Output> {
    // The statement a policy provider is told of goes with it too.
    let work = crate::access::StatementContext::carried(work);
    match COMMAND.try_with(Arc::clone) {
        Ok(command) => futures::future::Either::Left(COMMAND.scope(command, work)),
        Err(_) => futures::future::Either::Right(work),
    }
}

/// `work`, a query a command runs to evaluate one of its own expressions (the value of `SET VAR`, a
/// `DECLARE` default, the string of `EXECUTE IMMEDIATE`): the query carries a `SELECT` put together for
/// it, and is listed in `system.queries` under the command's text and class, as the command's run.
pub(crate) fn as_evaluation<F: Future>(work: F) -> impl Future<Output = F::Output> {
    EVALUATION.scope((), work)
}

/// A query run [`as_evaluation`] failed: its row (the one `slot` names) is the failed run of the command
/// the calling task is running, which gets no refused row beside it.
pub(crate) fn evaluation_failed(slot: &std::sync::Mutex<Option<String>>) {
    let id = slot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    if let Some(id) = id {
        // No command: nothing to tell.
        let _ = COMMAND.try_with(|c| {
            *c.row
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(id);
        });
    }
}

/// Tells the statement of a script the calling task is running, if any, that the refused row `id` is its
/// row: the statement gets no second refused row from [`as_listed_script_statement`].
pub(crate) fn refused_in_script_statement(id: &str) {
    let _ = COMMAND.try_with(|c| {
        if c.of_script {
            c.row
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get_or_insert_with(|| id.to_owned());
        }
    });
}

/// The command the calling task is running, for a query begun with no text of its own (`has_text` false),
/// begun by a statement of a script, or run to evaluate an expression of the command.
pub(crate) fn command_statement(has_text: bool) -> Option<Listing> {
    let evaluation = EVALUATION.try_with(|()| ()).is_ok();
    COMMAND
        .try_with(|c| {
            (!has_text || c.of_script || evaluation).then(|| {
                c.listed.store(true, Ordering::Relaxed);
                Listing {
                    text: Arc::clone(&c.text),
                    class: c.class,
                    row: (!evaluation).then(|| Arc::clone(&c.row)),
                }
            })
        })
        .ok()
        .flatten()
}

/// `work`, the run of one statement of a script: the queries it runs are listed in `system.queries` under
/// `text` (the statement's own, redacted) and `class`, not under the script's.
pub(crate) fn as_script_statement<F: Future>(
    text: &str,
    class: &'static str,
    work: F,
) -> impl Future<Output = F::Output> {
    COMMAND.scope(script_statement(text, class), work)
}

/// [`as_script_statement`] for a whole statement of a script (`work` checks access, analyzes and runs it).
/// One that fails before any query of it began is listed as refused under `text` and `class`, on
/// `session`, as the same statement typed alone is; the script's own row tells the script's failure.
pub(crate) async fn as_listed_script_statement<T, F>(
    session: &Session,
    text: &str,
    class: &'static str,
    work: F,
) -> Result<T, KrishivError>
where
    F: Future<Output = Result<T, KrishivError>>,
{
    let statement = script_statement(text, class);
    let result = COMMAND.scope(Arc::clone(&statement), work).await;
    if let Err(error) = &result
        && statement
            .row
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_none()
    {
        session.record_refused(Some(text), Some(class), error);
    }
    result
}

/// A statement of a script, as the command its queries are listed under.
fn script_statement(text: &str, class: &'static str) -> Arc<Command> {
    Arc::new(Command {
        text: Arc::from(text),
        class,
        listed: std::sync::atomic::AtomicBool::new(false),
        // The statement's own slot: the script's failure is told by the script's row.
        row: Arc::default(),
        of_script: true,
    })
}

/// What [`Session::sql_here_marked`] did with a statement.
#[expect(
    clippy::large_enum_variant,
    reason = "a return value matched at once; boxing the frame buys nothing"
)]
pub(crate) enum Ran {
    /// Its frame (a query's, analyzed; a command's result, already run) and whether it was a
    /// command.
    Frame(DataFrame, bool),
    /// Not a query, refused before it was authorized or run (`queries_only`), with its class.
    NotAQuery(krishiv_sql::StatementClass),
}

impl Ran {
    /// The frame, or `KRISHIV_FLIGHT_SQL_NOT_A_QUERY` for a refused statement.
    pub(crate) fn into_frame(self) -> Result<(DataFrame, bool), KrishivError> {
        match self {
            Self::Frame(df, command) => Ok((df, command)),
            Self::NotAQuery(class) => Err(not_a_query(class)),
        }
    }
}

/// `KRISHIV_FLIGHT_SQL_NOT_A_QUERY` (55000) for a statement of `class`.
pub(super) fn not_a_query(class: krishiv_sql::StatementClass) -> KrishivError {
    krishiv_common::error::KrishivFlightSqlNotAQuery {
        statement_class: class.name().to_owned(),
    }
    .into()
}

impl Session {
    /// Runs `query` on the calling task as `sql` would, with `row` as the statement's slot for
    /// the id of the `system.queries` row a query of it began. `parsed` gets the statement's class
    /// once it parsed: a statement refused after that is listed with its class.
    pub(crate) async fn sql_here_marked(
        &self,
        query: &str,
        row: Arc<std::sync::Mutex<Option<String>>>,
        parsed: &std::sync::OnceLock<&'static str>,
        queries_only: bool,
    ) -> Result<Ran, KrishivError> {
        // Boxed: one more nested future level overflows the compiler's query depth in the binaries.
        Box::pin(self.sql_here_inner(query, row, Vec::new(), Some(parsed), queries_only)).await
    }

    /// With `queries_only`, a statement that is not a query is not authorized or run: its class
    /// is returned ([`Ran::NotAQuery`]).
    pub(super) async fn sql_here_inner(
        &self,
        query: &str,
        row: Arc<std::sync::Mutex<Option<String>>>,
        spans: Vec<krishiv_sql::ast::Span>,
        parsed: Option<&std::sync::OnceLock<&'static str>>,
        queries_only: bool,
    ) -> Result<Ran, KrishivError> {
        let mut env = self.query_env();
        // Planning starts here: substitution, `IDENTIFIER(..)` clauses and the parse count (MODES-2).
        env.planning.parse_started(std::time::Instant::now());
        env.row = row;
        let written = query;
        let query = self.substituted(query);
        let query: &str = &query;
        // `IDENTIFIER(expr)` names are evaluated first: parsing, access checks and the run see them.
        // This runs before the statement's `StatementContext` scope below, so a check it made would
        // carry an enclosing statement's context; it makes none today (the arguments are constant).
        let resolved = self.identifier_clauses_into(query, &mut env).await?;
        let named_by_clause = matches!(resolved, std::borrow::Cow::Owned(_));
        if !spans.is_empty() && !named_by_clause && query == written {
            env.identifier_spans = spans.into();
        }
        let query: &str = &resolved;
        let redacted = krishiv_sql::redact_sql(query);
        let statement = krishiv_sql::parse_statement(query, &env.parser)
            .map_err(|e| sql_context(e.error, &redacted, e.span.start, e.span.end))?;
        // Metric views arrive in Spark 4.2: older releases fail to parse the statement, at `WITH`
        // (3.5) or at `METRICS` (4.0, 4.1), the gate goldens of P3 4.2-MV. `session.rs`'s dispatch
        // arm checks the same again, for EXECUTE IMMEDIATE and scripts; this check is only here so
        // that the refusal comes right after the parse, before any access check. Keep both.
        if let Statement::CreateMetricView(v) = &statement
            && let Some(e) = metric_view_gate_error(v, self.state.profile, query, &redacted)
        {
            return Err(e);
        }
        // Spark refuses a database on a temporary function named by an `IDENTIFIER(...)`
        // clause when it drops it; the parser sees only the evaluated name.
        if named_by_clause
            && let Statement::DropFunction(f) = &statement
            && f.temporary
        {
            krishiv_sql::check_temporary_function_name(&f.name, f.span)
                .map_err(|e| sql_context(e.error, &redacted, e.span.start, e.span.end))?;
        }
        // Before any side effect: what the statement writes, creates or
        // drops (its reads are checked by analysis, first thing).
        let class = self.statement_class(&statement, &env);
        if let Some(parsed) = parsed {
            // Set once: only `sql_ran` passes a slot, one per call; a statement run inside this one
            // (`sql_here_spanned`) passes none, so it never writes the outer statement's class.
            let _ = parsed.set(class.name());
        }
        if queries_only && !matches!(statement, Statement::Query(_)) {
            return Ok(Ran::NotAQuery(class));
        }
        // The statement's class and query id reach a policy provider's questions (ADR 0049).
        let context = crate::access::StatementContext::new(
            class.name(),
            self.operation_id().map(crate::QueryId::as_str),
            Arc::clone(&env.row),
        );
        Box::pin(context.scope(self.sql_here_checked(statement, class, env, query, &redacted)))
            .await
    }

    /// [`Self::sql_here_inner`] from the access checks of what `statement` writes on.
    async fn sql_here_checked(
        &self,
        statement: Statement,
        class: krishiv_sql::StatementClass,
        mut env: QueryEnv,
        query: &str,
        redacted: &RedactedSql,
    ) -> Result<Ran, KrishivError> {
        self.authorize_statement(&statement, class, &env).await?;
        env.statement_class = Some(class.name());
        self.refuse_changes_while_draining(class, redacted, &env)?;
        let command = !matches!(statement, Statement::Query(_));
        let script = matches!(statement, Statement::Script(_));
        if !command {
            let df = self.run_statement(statement, env, query, redacted).await?;
            df.with_statement(redacted, class.name());
            return Ok(Ran::Frame(df, false));
        }
        let running = Arc::new(Command {
            text: Arc::from(redacted.as_str()),
            class: class.name(),
            listed: std::sync::atomic::AtomicBool::new(false),
            row: Arc::clone(&env.row),
            of_script: false,
        });
        // Boxed, as `sql_here_marked`'s future is: the scope is one more nested future level.
        let run = Box::pin(self.run_statement(statement, env, query, redacted));
        let df = COMMAND.scope(Arc::clone(&running), run).await?;
        // A command that ran no query (`CREATE TABLE`, `SET`, `SHOW`) is listed when its result frame is
        // read, with the statement's text and class. One that ran a query has its row already: a read of
        // its (empty) result is not the statement again. A script's result holds the rows of its last query:
        // a read of them is listed under the script's text and class, never as a row with neither.
        if script || !running.listed.load(Ordering::Relaxed) {
            df.with_statement(redacted, class.name());
        }
        Ok(Ran::Frame(df, true))
    }
}

/// Spark's `PARSE_SYNTAX_ERROR` for a metric view on a release before 4.2, told against `query`
/// (the text `v` was parsed from). Checked right after the parse of a top-level statement (before
/// any access check) and again where the statement is dispatched, for the statements of
/// `EXECUTE IMMEDIATE` and of a script.
pub(crate) fn metric_view_gate_error(
    v: &krishiv_sql::ast::CreateMetricView,
    profile: SparkProfile,
    query: &str,
    redacted: &RedactedSql,
) -> Option<KrishivError> {
    let span = metric_view_gate(v, profile)?;
    // Errors never echo a string literal's text (`krishiv-sql/CLAUDE.md`).
    let token = if span.range() == v.body_span.range() {
        "$$"
    } else {
        query.get(span.start..span.end).unwrap_or_default()
    };
    let error = krishiv_common::error::ParseSyntaxError {
        error: format!("'{token}'"),
        hint: String::new(),
    };
    Some(sql_context(error.into(), redacted, span.start, span.end))
}

/// Where a Spark release before 4.2 stops parsing `CREATE VIEW .. WITH METRICS`, if `profile` is
/// one: at a view's `LANGUAGE` (no release before 4.2 has it), or at `WITH METRICS`, whichever
/// comes first; 3.5 has no `WITH` there, 4.0 and 4.1 take `WITH` for a schema-binding clause and
/// fail at the next token (Spark 3.5.7 and 4.1.0 runs, 2026-11-21). The statement always holds
/// one of the two (else it is no metric view), so the span never covers the YAML literal.
fn metric_view_gate(
    v: &krishiv_sql::ast::CreateMetricView,
    profile: SparkProfile,
) -> Option<krishiv_sql::ast::Span> {
    let at = v.with_metrics.map(|[with, metrics]| match profile {
        SparkProfile::V3_5 => with,
        _ => metrics,
    });
    let first = [at, v.language_keyword]
        .into_iter()
        .flatten()
        .min_by_key(|s| s.start);
    match profile {
        // Unreachable fallback, refused all the same: the literal's span, whose text is never shown.
        SparkProfile::V3_5 | SparkProfile::V4_0 | SparkProfile::V4_1 => first.or(Some(v.body_span)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A statement of a script whose work records its own refused row (an action's planning failure
    /// does) is listed once, not a second time by the script.
    #[test]
    fn a_script_statement_that_recorded_its_refusal_is_listed_once() {
        let s = Session::new().unwrap();
        let error = || -> KrishivError { crate::session::internal("planning failed".to_owned()) };
        let rows = |s: &Session| s.state.queries.snapshot(true).len();
        let before = rows(&s);
        let own = futures::executor::block_on(as_listed_script_statement(
            &s,
            "SELECT 1",
            "QUERY",
            async {
                s.record_refused(Some("SELECT 1"), Some("QUERY"), &error());
                Err::<(), _>(error())
            },
        ));
        assert!(own.is_err());
        assert_eq!(rows(&s), before + 1);
        let none = futures::executor::block_on(as_listed_script_statement(
            &s,
            "SELECT 2",
            "QUERY",
            async { Err::<(), _>(error()) },
        ));
        assert!(none.is_err());
        assert_eq!(rows(&s), before + 2);
    }
}
