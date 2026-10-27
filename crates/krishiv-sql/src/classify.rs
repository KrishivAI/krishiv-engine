// SPDX-License-Identifier: Apache-2.0
//! The syntactic statement classifier (stage 1 of statement
//! classification): what kind of effect a parsed statement has.
//!
//! [`classify`] is an exhaustive `match` over [`Statement`] with no
//! wildcard arm, so a new statement variant does not compile until it is
//! classified. A statement the classifier cannot place is
//! [`StatementClass::Admin`], the most restricted class, never `Read`;
//! every `Admin` arm carries a `// cluster-scope: <reason>` comment
//! (`cargo xtask source-guard` rejects one without it), so `Admin` never
//! becomes a silent catch-all.
//!
//! The classifier decides no policy: consumers apply their own allow-list
//! over classes (a read-only surface allows [`StatementClass::Read`]), and
//! the session checks each privilege a statement needs on its securables.

use std::fmt;

use crate::ast::{
    AlterViewAction, CacheKind, DescribeKind, ExplainMode, MergeSource, Query, SetVariableKind,
    Statement,
};

/// What kind of effect a statement has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StatementClass {
    /// Reads data or metadata and changes nothing (queries, `SHOW`,
    /// `DESCRIBE`, `EXPLAIN` of a statement it does not run).
    Read,
    /// Changes rows or files of existing objects (`INSERT`, `MERGE`,
    /// `DELETE`, `UPDATE`, `TRUNCATE`, `ANALYZE TABLE`).
    Write,
    /// Creates, alters or drops shared catalog objects (tables, views,
    /// namespaces).
    Ddl,
    /// Changes only the state of the session that runs it (`SET`,
    /// `RESET`, `USE`, session variables, temporary views and tables).
    Session,
    /// Acts beyond the session and its data (cluster-scope state, code
    /// loading), and anything the classifier cannot place.
    Admin,
}

impl StatementClass {
    /// The class name as `system.queries.statement_class` and the analyze
    /// API show it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Read => "READ",
            Self::Write => "WRITE",
            Self::Ddl => "DDL",
            Self::Session => "SESSION",
            Self::Admin => "ADMIN",
        }
    }
}

impl fmt::Display for StatementClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// How an `EXPLAIN` treats its statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExplainKind {
    /// Shows the plan without running the statement.
    Plan,
    /// `EXPLAIN ANALYZE`: runs the statement and shows its profile.
    Analyze,
}

/// The class of `statement` (see the module documentation).
///
/// ```
/// use krishiv_sql::{ParserOptions, StatementClass, classify, parse_statement};
///
/// let class = |sql: &str| classify(&parse_statement(sql, &ParserOptions::default()).unwrap());
/// assert_eq!(class("SELECT 1 /* ; DROP TABLE t */"), StatementClass::Read);
/// assert_eq!(class("INSERT INTO t SELECT 1"), StatementClass::Write);
/// assert_eq!(class("DROP TABLE t"), StatementClass::Ddl);
/// assert_eq!(class("SET spark.sql.ansi.enabled = false"), StatementClass::Session);
/// assert_eq!(class("ADD JAR '/tmp/a.jar'"), StatementClass::Admin);
/// ```
#[must_use]
#[deny(clippy::wildcard_enum_match_arm)]
pub fn classify(statement: &Statement) -> StatementClass {
    match statement {
        Statement::Query(_)
        | Statement::Show(_)
        | Statement::Describe(_)
        | Statement::ListResource(_) => StatementClass::Read,
        // `EXPLAIN ANALYZE` runs its statement: it takes that statement's
        // class; any other mode only shows a plan.
        Statement::Explain(e) => match e.mode {
            ExplainMode::Analyze => classify(&e.statement),
            ExplainMode::Simple
            | ExplainMode::Extended
            | ExplainMode::Formatted
            | ExplainMode::Codegen
            | ExplainMode::Cost
            | ExplainMode::Verbose => StatementClass::Read,
        },
        // The legacy view nodes are temporary views with a one-part name:
        // session-local, re-analyzed under the session's principal.
        Statement::Set(_)
        | Statement::Reset(_)
        | Statement::Use(_)
        | Statement::DeclareVariable(_)
        | Statement::SetVariable(_)
        | Statement::DropVariable(_)
        | Statement::CreateView(_)
        | Statement::DropView(_) => StatementClass::Session,
        Statement::CreateTable(t) => {
            if t.temporary {
                StatementClass::Session
            } else {
                StatementClass::Ddl
            }
        }
        // A global temporary view is visible to every session: shared state.
        Statement::CreatePermanentView(v) => {
            if v.temporary && !v.global {
                StatementClass::Session
            } else {
                StatementClass::Ddl
            }
        }
        Statement::CreateTableLike(_)
        | Statement::DropTable(_)
        | Statement::CreateNamespace(_)
        | Statement::DropNamespace(_)
        | Statement::DropPermanentView(_)
        | Statement::AlterTable(_)
        | Statement::AlterView(_)
        | Statement::AlterNamespace(_)
        | Statement::Repair(_) => StatementClass::Ddl,
        // `ANALYZE TABLE` writes the table's statistics.
        Statement::Insert(_)
        | Statement::Delete(_)
        | Statement::Update(_)
        | Statement::Merge(_)
        | Statement::Truncate(_)
        | Statement::Analyze(_) => StatementClass::Write,
        Statement::Cache(c) => match &c.kind {
            CacheKind::Cache { .. } | CacheKind::Uncache { .. } => StatementClass::Session,
            // cluster-scope: CLEAR CACHE drops the cached data of every session.
            CacheKind::Clear => StatementClass::Admin,
        },
        // REFRESH invalidates metadata and file listings cached for every
        // session; ADD JAR/FILE/ARCHIVE loads code and files into the
        // process.
        // cluster-scope: shared caches and process-wide code loading.
        Statement::Refresh(_) | Statement::AddResource(_) => StatementClass::Admin,
    }
}

/// How `statement` explains, if it is an `EXPLAIN`.
#[must_use]
pub fn explain_kind(statement: &Statement) -> Option<ExplainKind> {
    match statement {
        Statement::Explain(e) if e.mode == ExplainMode::Analyze => Some(ExplainKind::Analyze),
        Statement::Explain(_) => Some(ExplainKind::Plan),
        _ => None,
    }
}

/// The queries `statement` reads from, each analyzable on its own: their
/// securables are the statement's reads. A multi-table `INSERT` with a
/// shared `FROM` has no standalone query and returns `None`, as does any
/// read through a `DEFAULT` or `SET VARIABLE` expression subquery (only
/// whole queries are listed).
#[must_use]
#[deny(clippy::wildcard_enum_match_arm)]
pub fn statement_queries(statement: &Statement) -> Option<Vec<&Query>> {
    Some(match statement {
        Statement::Query(q) => vec![&**q],
        Statement::Explain(e) => return statement_queries(&e.statement),
        Statement::CreateView(v) => vec![&*v.query],
        Statement::CreateTable(t) => t.as_query.iter().map(|q| &**q).collect(),
        Statement::CreatePermanentView(v) => v.query.iter().map(|q| &**q).collect(),
        Statement::SetVariable(v) => match &v.kind {
            SetVariableKind::Row { query, .. } => vec![&**query],
            SetVariableKind::Single { assignments } => {
                if assignments.is_empty() {
                    Vec::new()
                } else {
                    return None;
                }
            }
        },
        Statement::DeclareVariable(v) => {
            if v.default.is_some() {
                return None;
            }
            Vec::new()
        }
        Statement::Insert(i) => {
            if i.source.is_some() {
                return None;
            }
            i.clauses.iter().map(|c| &*c.query).collect()
        }
        Statement::Merge(m) => match &m.source {
            MergeSource::Query { query, .. } => vec![&**query],
            MergeSource::Table { .. } => Vec::new(),
        },
        Statement::Cache(c) => match &c.kind {
            CacheKind::Cache { query, .. } => query.iter().map(|q| &**q).collect(),
            CacheKind::Uncache { .. } | CacheKind::Clear => Vec::new(),
        },
        Statement::Describe(d) => match &d.kind {
            DescribeKind::Query { query, .. } => vec![&**query],
            DescribeKind::Table { .. }
            | DescribeKind::Function { .. }
            | DescribeKind::Namespace { .. } => Vec::new(),
        },
        Statement::AlterView(a) => match &a.action {
            AlterViewAction::SetQuery { query, .. } => vec![&**query],
            AlterViewAction::SetSchemaBinding(_)
            | AlterViewAction::SetTblProperties(_)
            | AlterViewAction::UnsetTblProperties { .. }
            | AlterViewAction::RenameTo { .. } => Vec::new(),
        },
        // Reads without a query (`UPDATE`/`DELETE` predicates, `MERGE`
        // from a table) are expressions over the target: no
        // standalone query to analyze.
        Statement::Delete(d) => {
            if d.where_clause.is_some() {
                return None;
            }
            Vec::new()
        }
        Statement::Update(_) => return None,
        Statement::Set(_)
        | Statement::Reset(_)
        | Statement::DropView(_)
        | Statement::CreateTableLike(_)
        | Statement::DropTable(_)
        | Statement::CreateNamespace(_)
        | Statement::DropNamespace(_)
        | Statement::Use(_)
        | Statement::DropVariable(_)
        | Statement::DropPermanentView(_)
        | Statement::AlterTable(_)
        | Statement::AlterNamespace(_)
        | Statement::Truncate(_)
        | Statement::Show(_)
        | Statement::Refresh(_)
        | Statement::Analyze(_)
        | Statement::Repair(_)
        | Statement::AddResource(_)
        | Statement::ListResource(_) => Vec::new(),
    })
}
