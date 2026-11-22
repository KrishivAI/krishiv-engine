// SPDX-License-Identifier: Apache-2.0
//! Running one SQL statement on the calling task ([`Session::sql_here_marked`]) and [`Ran`], what it did with it
//! (with `queries_only`, P3-T101 Flight SQL refuses a command at GetFlightInfo).

use super::*;

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
    /// the id of the `system.queries` row a query of it began.
    pub(crate) async fn sql_here_marked(
        &self,
        query: &str,
        row: Arc<std::sync::Mutex<Option<String>>>,
        queries_only: bool,
    ) -> Result<Ran, KrishivError> {
        // Boxed: one more nested future level overflows the compiler's query depth in the binaries.
        Box::pin(self.sql_here_inner(query, row, Vec::new(), queries_only)).await
    }

    /// With `queries_only`, a statement that is not a query is not authorized or run: its class
    /// is returned ([`Ran::NotAQuery`]).
    pub(super) async fn sql_here_inner(
        &self,
        query: &str,
        row: Arc<std::sync::Mutex<Option<String>>>,
        spans: Vec<krishiv_sql::ast::Span>,
        queries_only: bool,
    ) -> Result<Ran, KrishivError> {
        let mut env = self.query_env();
        env.row = row;
        let written = query;
        let query = self.substituted(query);
        let query: &str = &query;
        // `IDENTIFIER(expr)` names are evaluated first: parsing, access checks and the run see them.
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
        if queries_only && !matches!(statement, Statement::Query(_)) {
            return Ok(Ran::NotAQuery(class));
        }
        self.authorize_statement(&statement, class, &env).await?;
        env.statement_class = Some(class.name());
        self.refuse_changes_while_draining(class, &redacted, &env)?;
        let command = !matches!(statement, Statement::Query(_));
        let df = self.run_statement(statement, env, query, &redacted).await?;
        // Whatever frame the statement returns (a command's result too) shows the statement's
        // text and class when it runs.
        df.with_statement(&redacted, class.name());
        Ok(Ran::Frame(df, command))
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
