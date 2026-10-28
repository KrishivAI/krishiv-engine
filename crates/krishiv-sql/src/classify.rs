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

use krishiv_plan::analyzer::Privilege;

use crate::ast::{
    AlterNamespaceAction, AlterTableAction, AlterViewAction, CacheKind, DdlName, DescribeKind,
    ExplainMode, MergeSource, Query, SetVariableKind, ShowKind, Statement,
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

/// An object a statement changes, or reads without a query, before it is
/// located: the session resolves names against its current catalog and
/// namespace and paths to their qualified form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TargetObject<'a> {
    /// The table the name names.
    Table(&'a DdlName),
    /// The table the name names inside the namespace `in` (`SHOW COLUMNS
    /// IN t IN db`).
    TableIn {
        /// The table name.
        table: &'a DdlName,
        /// The namespace.
        namespace: &'a DdlName,
    },
    /// The persistent view the name names.
    View(&'a DdlName),
    /// The global temporary view the name names (in `global_temp`).
    GlobalTempView(&'a DdlName),
    /// The namespace holding the table or view the name names.
    ParentOf(&'a DdlName),
    /// The namespace global temporary views live in.
    GlobalTempNamespace,
    /// The namespace the name names.
    Namespace(&'a DdlName),
    /// The catalog (or parent namespace) holding the namespace the name
    /// names.
    ParentOfNamespace(&'a DdlName),
    /// The current namespace.
    CurrentNamespace,
    /// The current catalog.
    CurrentCatalog,
    /// A path, as written.
    Path(&'a str),
    /// The cluster: what the `ADMIN` class acts on.
    Cluster,
}

/// One privilege a statement needs on one object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target<'a> {
    /// The object.
    pub object: TargetObject<'a>,
    /// The privilege.
    pub privilege: Privilege,
}

/// What `statement` needs beyond the reads of its queries (which analysis
/// checks): `Modify` on what it writes, `Create` on the namespace a new
/// object goes in, `Drop` on what it drops or replaces (an `ALTER` replaces
/// the object's definition: `Drop` on it and `Create` in its namespace),
/// `Select` on what it reads without a query (metadata commands, `LIKE`
/// sources, `MERGE` from a table), and `Monitor` on the cluster for the
/// `ADMIN` class. Session-class statements need nothing beyond the files
/// they name (a temporary table's `LOCATION`).
#[must_use]
#[deny(clippy::wildcard_enum_match_arm)]
#[expect(clippy::too_many_lines, reason = "one arm per statement kind")]
pub fn statement_targets(statement: &Statement) -> Vec<Target<'_>> {
    use Privilege::{Create, Drop, Modify, Monitor, Select};
    use TargetObject as O;
    const fn t(object: TargetObject<'_>, privilege: Privilege) -> Target<'_> {
        Target { object, privilege }
    }
    fn location(l: &Option<String>) -> Vec<Target<'_>> {
        l.iter()
            .map(|p| t(TargetObject::Path(p), Privilege::Modify))
            .collect()
    }
    // An ALTER replaces the definition of `object` in `parent`.
    fn alter<'a>(object: TargetObject<'a>, parent: TargetObject<'a>) -> Vec<Target<'a>> {
        vec![t(object, Privilege::Drop), t(parent, Privilege::Create)]
    }
    fn listing(namespace: &Option<DdlName>) -> Vec<Target<'_>> {
        let object = namespace
            .as_ref()
            .map_or(TargetObject::CurrentNamespace, TargetObject::Namespace);
        vec![t(object, Privilege::Select)]
    }
    match statement {
        Statement::Query(_)
        | Statement::Set(_)
        | Statement::Reset(_)
        | Statement::Use(_)
        | Statement::DeclareVariable(_)
        | Statement::SetVariable(_)
        | Statement::DropVariable(_)
        | Statement::CreateView(_)
        | Statement::DropView(_)
        | Statement::ListResource(_) => Vec::new(),
        Statement::Explain(e) => match e.mode {
            ExplainMode::Analyze => statement_targets(&e.statement),
            ExplainMode::Simple
            | ExplainMode::Extended
            | ExplainMode::Formatted
            | ExplainMode::Codegen
            | ExplainMode::Cost
            | ExplainMode::Verbose => Vec::new(),
        },
        Statement::CreateTable(c) => {
            if c.temporary {
                // Session-local, but its files are still read, and written
                // by `AS SELECT`.
                let mut out: Vec<Target<'_>> = c
                    .location
                    .iter()
                    .map(|p| t(TargetObject::Path(p), Select))
                    .collect();
                if c.as_query.is_some() {
                    out.extend(location(&c.location));
                }
                return out;
            }
            let mut out = vec![t(O::ParentOf(&c.name), Create)];
            if c.or_replace {
                out.push(t(O::Table(&c.name), Drop));
            }
            if c.as_query.is_some() {
                out.push(t(O::Table(&c.name), Modify));
            }
            out.extend(location(&c.location));
            out
        }
        Statement::CreateTableLike(c) => {
            let mut out = vec![
                t(O::ParentOf(&c.name), Create),
                t(O::Table(&c.source), Select),
            ];
            out.extend(location(&c.location));
            out
        }
        Statement::DropTable(d) => vec![t(O::Table(&d.name), Drop)],
        Statement::CreateNamespace(c) => {
            let mut out = vec![t(O::ParentOfNamespace(&c.name), Create)];
            out.extend(location(&c.location));
            out
        }
        Statement::DropNamespace(d) => vec![t(O::Namespace(&d.name), Drop)],
        Statement::CreatePermanentView(v) => {
            let (object, parent) = match (v.temporary, v.global) {
                (true, false) => return Vec::new(),
                (_, true) => (O::GlobalTempView(&v.name), O::GlobalTempNamespace),
                (false, false) => (O::View(&v.name), O::ParentOf(&v.name)),
            };
            let mut out = vec![t(parent, Create)];
            if v.or_replace {
                out.push(t(object, Drop));
            }
            out
        }
        Statement::DropPermanentView(d) => vec![t(O::View(&d.name), Drop)],
        Statement::AlterTable(a) => {
            let mut out = alter(O::Table(&a.table), O::ParentOf(&a.table));
            match &a.action {
                AlterTableAction::RenameTo { name, .. } => out.push(t(O::ParentOf(name), Create)),
                AlterTableAction::SetLocation(l) => out.push(t(O::Path(l), Modify)),
                AlterTableAction::SetPartitionLocation { location, .. } => {
                    out.push(t(O::Path(location), Modify));
                }
                AlterTableAction::AddPartition { parts, .. } => {
                    out.extend(parts.iter().flat_map(|p| location(&p.location)));
                }
                AlterTableAction::AddColumns(_)
                | AlterTableAction::DropColumns(_)
                | AlterTableAction::RenameColumn { .. }
                | AlterTableAction::ChangeColumn { .. }
                | AlterTableAction::AlterColumn { .. }
                | AlterTableAction::SetTblProperties(_)
                | AlterTableAction::UnsetTblProperties { .. }
                | AlterTableAction::SetSerdeProperties(_)
                | AlterTableAction::SetPartitionSerdeProperties { .. }
                | AlterTableAction::DropPartition { .. }
                | AlterTableAction::RenamePartition { .. }
                | AlterTableAction::RecoverPartitions => {}
            }
            out
        }
        Statement::AlterView(a) => {
            let mut out = alter(O::View(&a.name), O::ParentOf(&a.name));
            match &a.action {
                AlterViewAction::RenameTo { name, .. } => out.push(t(O::ParentOf(name), Create)),
                AlterViewAction::SetQuery { .. }
                | AlterViewAction::SetSchemaBinding(_)
                | AlterViewAction::SetTblProperties(_)
                | AlterViewAction::UnsetTblProperties { .. } => {}
            }
            out
        }
        Statement::AlterNamespace(a) => {
            let mut out = alter(O::Namespace(&a.name), O::ParentOfNamespace(&a.name));
            match &a.action {
                AlterNamespaceAction::SetLocation(l) => out.push(t(O::Path(l), Modify)),
                AlterNamespaceAction::SetProperties(_)
                | AlterNamespaceAction::DefaultCollation(_) => {}
            }
            out
        }
        Statement::Repair(r) => alter(O::Table(&r.table), O::ParentOf(&r.table)),
        Statement::Insert(i) => i
            .clauses
            .iter()
            .map(|c| t(O::Table(&c.name), Modify))
            .collect(),
        Statement::Delete(d) => vec![t(O::Table(&d.table), Modify)],
        Statement::Update(u) => vec![t(O::Table(&u.table), Modify)],
        Statement::Truncate(u) => vec![t(O::Table(&u.table), Modify)],
        Statement::Merge(m) => {
            let mut out = vec![t(O::Table(&m.target), Modify)];
            match &m.source {
                MergeSource::Table { name, .. } => out.push(t(O::Table(name), Select)),
                MergeSource::Query { .. } => {}
            }
            out
        }
        // Statistics are written from the rows read.
        Statement::Analyze(a) => vec![t(O::Table(&a.table), Select), t(O::Table(&a.table), Modify)],
        Statement::Show(s) => match &s.kind {
            ShowKind::Tables { namespace, .. }
            | ShowKind::TableExtended { namespace, .. }
            | ShowKind::Views { namespace, .. }
            | ShowKind::Functions { namespace, .. } => listing(namespace),
            ShowKind::Columns {
                table, namespace, ..
            } => vec![t(
                namespace
                    .as_ref()
                    .map_or(O::Table(table), |namespace| O::TableIn { table, namespace }),
                Select,
            )],
            ShowKind::Namespaces { .. } => vec![t(O::CurrentCatalog, Select)],
            ShowKind::CreateTable { name, .. }
            | ShowKind::TblProperties { name, .. }
            | ShowKind::Partitions { name, .. } => vec![t(O::Table(name), Select)],
            ShowKind::CurrentNamespace { .. } => Vec::new(),
        },
        Statement::Describe(d) => match &d.kind {
            DescribeKind::Table { table, .. } => vec![t(O::Table(table), Select)],
            DescribeKind::Namespace { name, .. } => vec![t(O::Namespace(name), Select)],
            // Built-in functions are public; a query's reads are analyzed.
            DescribeKind::Function { .. } | DescribeKind::Query { .. } => Vec::new(),
        },
        Statement::Cache(c) => match &c.kind {
            // Caching a table reads it whole; a cached query is analyzed.
            CacheKind::Cache { name, query, .. } => {
                if query.is_some() {
                    Vec::new()
                } else {
                    vec![t(O::Table(name), Select)]
                }
            }
            CacheKind::Uncache { .. } => Vec::new(),
            CacheKind::Clear => vec![t(O::Cluster, Monitor)],
        },
        Statement::Refresh(_) | Statement::AddResource(_) => vec![t(O::Cluster, Monitor)],
    }
}
