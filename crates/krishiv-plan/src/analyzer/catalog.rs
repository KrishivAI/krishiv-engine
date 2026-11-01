// SPDX-License-Identifier: Apache-2.0
//! What the analyzer resolves names against: catalogs (metadata only),
//! functions, the access policy and the SQL front end (spec 13 §7, ADR
//! 0036 R2).

use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_common::redact::RedactedSql;
use krishiv_common::secret::OptionValue;
use krishiv_types::DataType;

use super::{AnalysisError, SparkProfile, SqlFunction};
use crate::{Expr, FunctionId, FunctionKind, Plan, SourceId, TableVersion};

/// How a relation name was written, for the catalog that loads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NameForm {
    /// Names match as written; otherwise case-insensitively, as Spark's
    /// catalogs do.
    pub case_sensitive: bool,
    /// The name carried a namespace or catalog part. Spark resolves
    /// temporary views only for names that did not.
    pub qualified: bool,
    /// The SQL path names no schema of this catalog, only the session: the
    /// name may reach temporary views and nothing stored.
    pub temporary_only: bool,
}

/// A boxed `Send` future, as trait methods that may do I/O return.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A table's identity in its catalog.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TableIdent {
    /// The catalog name.
    pub catalog: String,
    /// The namespace parts (a database is one part).
    pub namespace: Vec<String>,
    /// The table name.
    pub name: String,
}

impl TableIdent {
    /// Every part: catalog, namespace, name.
    #[must_use]
    pub fn parts(&self) -> Vec<String> {
        let mut out = vec![self.catalog.clone()];
        out.extend(self.namespace.iter().cloned());
        out.push(self.name.clone());
        out
    }
}

impl fmt::Display for TableIdent {
    /// The quoted name, as Spark's errors print it: `` `cat`.`db`.`t` ``.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&quoted(&self.parts()))
    }
}

/// `parts` quoted as Spark's `toSQLId` does: `` `a`.`b` ``.
pub(crate) fn quoted(parts: &[String]) -> String {
    let quote = |p: &String| format!("`{}`", p.replace('`', "``"));
    parts.iter().map(quote).collect::<Vec<_>>().join(".")
}

/// A column of a table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableColumn {
    /// The name.
    pub name: String,
    /// The type.
    pub data_type: DataType,
    /// Whether it may be null.
    pub nullable: bool,
}

/// What a catalog entry is.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TableKind {
    /// A table.
    Table,
    /// A view (expanded by a later analyzer pass; rejected until then).
    View,
}

/// The metadata of a table at one version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableMeta {
    /// Its identity, as the catalog spells it.
    pub ident: TableIdent,
    /// Its columns.
    pub columns: Vec<TableColumn>,
    /// The current version, for versioned formats.
    pub version: Option<TableVersion>,
    /// Table or view.
    pub kind: TableKind,
}

/// The metadata half of a catalog (spec 13 §7); scans and writes are an
/// execution-layer concern.
pub trait Catalog: Send + Sync {
    /// The catalog's name.
    fn name(&self) -> &str;
    /// Bumped on every change made through Krishiv; part of the context
    /// fingerprint.
    fn epoch(&self) -> u64;
    /// Whether table names in this catalog take exactly one namespace part
    /// (Spark's session catalog): the analyzer raises
    /// `REQUIRES_SINGLE_PART_NAMESPACE` for any other name, without
    /// loading it.
    fn single_part_namespace(&self) -> bool {
        false
    }
    /// The table `id` (as written; the catalog applies `form`, as Spark's
    /// catalogs do) at `version`, or the current one; `None` when it does
    /// not exist.
    fn load_table<'a>(
        &'a self,
        id: &'a TableIdent,
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>>;
}

/// The session's catalogs: the current catalog and namespace, and the
/// catalogs by name.
pub trait CatalogResolver: Send + Sync {
    /// The current catalog's name.
    fn current_catalog(&self) -> &str;
    /// The current namespace.
    fn current_namespace(&self) -> &[String];
    /// The catalog named `name`.
    fn catalog(&self, name: &str) -> Option<Arc<dyn Catalog>>;
    /// Changes whenever the set of catalogs or their names change; part of
    /// the context fingerprint.
    fn epoch(&self) -> u64;
    /// The SQL path unqualified routines and relations resolve through, in
    /// order, each entry as its parts (`SET PATH`). The default is Spark's:
    /// `system.builtin`, `system.session`, the current namespace.
    fn sql_path(&self) -> Vec<Vec<String>> {
        let mut current = vec![self.current_catalog().to_owned()];
        current.extend(self.current_namespace().iter().cloned());
        vec![
            vec!["system".to_owned(), "builtin".to_owned()],
            vec!["system".to_owned(), "session".to_owned()],
            current,
        ]
    }
}

pub use krishiv_functions::signature::{
    ArgType, Arity, CallTypes, Constant, InputBuilder, InputType, Inputs, Operator, ResultType,
};

/// A function the registry knows. The canonical operators
/// ([`crate::Builtin`]) are not registry functions: the analyzer types
/// them as Spark types its operators.
#[derive(Clone)]
pub struct FunctionEntry {
    /// Its identity.
    pub id: FunctionId,
    /// Scalar, aggregate, ...
    pub kind: FunctionKind,
    /// The argument counts it takes (`WRONG_NUM_ARGS` otherwise).
    pub arity: Arity,
    /// How its arguments are implicitly cast.
    pub inputs: Inputs,
    /// The result type for the (cast) argument types.
    pub result: ResultType,
    /// Whether a call is deterministic (the optimizer never moves or
    /// duplicates a nondeterministic one).
    pub deterministic: bool,
}

impl fmt::Debug for FunctionEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FunctionEntry")
            .field("id", &self.id)
            .field("kind", &self.kind)
            .field("arity", &self.arity)
            .field("inputs", &self.inputs)
            .field("deterministic", &self.deterministic)
            .finish_non_exhaustive()
    }
}

/// A registry lookup.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum FunctionLookup {
    /// Found.
    Found(FunctionEntry),
    /// A SQL-bodied function of the session: calls are replaced by its
    /// body (see [`SqlFunction`]).
    Sql(Arc<SqlFunction>),
    /// Spark has it in this profile but Krishiv does not implement it.
    KnownUnsupported,
    /// Spark does not have it in this profile.
    NotFound,
}

/// The function registry: built-in functions resolve in-process, so the
/// lookup is synchronous. It is not asked about the canonical operators
/// ([`crate::Builtin`] names). Registries hosting session UDFs also answer
/// [`FunctionRegistry::resolve_udf`].
pub trait FunctionRegistry: Send + Sync {
    /// The function named by `name`'s parts (as written, case-insensitive;
    /// one part for a bare built-in, more for a qualified name) in
    /// `profile`.
    fn lookup(&self, name: &[String], profile: SparkProfile) -> FunctionLookup;
    /// Whether Spark analyzes the built-in `name` (case-insensitive) as
    /// deterministic: `None` when the registry does not know. Inline
    /// tables use it for functions the engine does not implement (which
    /// [`FunctionLookup::KnownUnsupported`] does not say).
    fn spark_deterministic(&self, _name: &str) -> Option<bool> {
        None
    }
    /// The session UDF named `name` (already lower-case) at `version`, for
    /// constant folding: `None` when the registry hosts no UDFs, or none
    /// at that name and version. The default has none.
    fn resolve_udf(
        &self,
        _name: &str,
        _version: u32,
    ) -> Option<Arc<dyn krishiv_udf_sdk::ScalarUdf>> {
        None
    }
    /// Changes whenever a lookup could answer differently (temporary and
    /// session functions); part of the context fingerprint.
    fn version(&self) -> u64;
}

/// A table-valued function the analyzer knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
/// What a table-function name resolves to.
pub enum TableFunctionKind {
    /// `range(start, end, step, numSlices)`.
    Range,
    /// `sql_keywords()`.
    SqlKeywords,
    /// A Spark generator TVF (`explode`, `inline`, `stack`, ...) the
    /// engine does not implement yet: lowering used to reject it with
    /// `KRISHIV_UNSUPPORTED_FEATURE` and the analyzer keeps doing so.
    UnsupportedGenerator,
}

/// What a table-function lookup found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TableFunctionLookup {
    /// A built-in table function.
    Found(TableFunctionKind),
    /// Not a table function Spark has in this profile.
    NotFound,
}

/// The table-function registry: the built-in table-valued functions,
/// looked up by lower-case name beside [`FunctionRegistry`]. No
/// per-function logic lives in lowering; the analyzer resolves through
/// this table.
pub trait TableFunctionRegistry: Send + Sync {
    /// The table function named `name` (as written; matched
    /// case-insensitively) in `profile`.
    fn lookup(&self, name: &str, profile: SparkProfile) -> TableFunctionLookup;
    /// Changes whenever a lookup could answer differently; part of the
    /// context fingerprint.
    fn version(&self) -> u64;
}

/// The built-in table functions: `range` and `sql_keywords`, plus the
/// Spark generator TVFs the engine rejects as unsupported (as lowering
/// did before they lowered).
#[derive(Debug, Clone, Copy, Default)]
pub struct BuiltinTableFunctions;

impl BuiltinTableFunctions {
    /// A new registry.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Whether `name` (lower-case) is a generator TVF Spark has but the
    /// engine does not implement yet.
    fn is_generator(lower: &str) -> bool {
        matches!(
            lower,
            "explode"
                | "explode_outer"
                | "inline"
                | "inline_outer"
                | "posexplode"
                | "posexplode_outer"
                | "json_tuple"
                | "stack"
        )
    }
}

impl TableFunctionRegistry for BuiltinTableFunctions {
    fn lookup(&self, name: &str, _profile: SparkProfile) -> TableFunctionLookup {
        let mut lower = String::with_capacity(name.len());
        lower.extend(name.chars().flat_map(char::to_lowercase));
        match lower.as_str() {
            "range" => TableFunctionLookup::Found(TableFunctionKind::Range),
            "sql_keywords" => TableFunctionLookup::Found(TableFunctionKind::SqlKeywords),
            n if Self::is_generator(n) => {
                TableFunctionLookup::Found(TableFunctionKind::UnsupportedGenerator)
            }
            _ => TableFunctionLookup::NotFound,
        }
    }

    fn version(&self) -> u64 {
        0
    }
}

/// Who runs a query. Build it with [`Principal::new`]; more attributes
/// arrive with authentication (P2-T136).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Principal {
    /// The user or service name.
    pub name: String,
    /// Group memberships.
    pub groups: Vec<String>,
    /// The tenant, in multi-tenant servers.
    pub tenant: Option<String>,
    /// User or service principal.
    pub kind: PrincipalKind,
}

/// The kind of a principal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PrincipalKind {
    /// A person.
    User,
    /// A service principal.
    Service,
}

impl Principal {
    /// A principal of `kind` named `name`, without groups or tenant.
    #[must_use]
    pub fn new(name: impl Into<String>, kind: PrincipalKind) -> Self {
        Self {
            name: name.into(),
            groups: Vec::new(),
            tenant: None,
            kind,
        }
    }

    /// The embedded default: the local OS user.
    #[must_use]
    pub fn local_user() -> Self {
        let name = ["USER", "USERNAME"]
            .iter()
            .find_map(|k| std::env::var(k).ok())
            .unwrap_or_else(|| "unknown".to_owned());
        Self {
            name,
            groups: Vec::new(),
            tenant: None,
            kind: PrincipalKind::User,
        }
    }

    /// The identity that enters the context fingerprint: kind, tenant,
    /// name and the sorted groups (a policy may decide by group).
    #[must_use]
    pub fn id(&self) -> String {
        let kind = match self.kind {
            PrincipalKind::User => "user",
            PrincipalKind::Service => "service",
        };
        let mut groups = self.groups.clone();
        groups.sort();
        groups.dedup();
        format!(
            "{kind}:{}/{}{}",
            quoted(std::slice::from_ref(
                &self.tenant.clone().unwrap_or_default()
            )),
            quoted(std::slice::from_ref(&self.name)),
            if groups.is_empty() {
                String::new()
            } else {
                format!("[{}]", quoted(&groups))
            }
        )
    }
}

/// Something a privilege is granted on (ADR 0036 R2).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Securable {
    /// A table.
    Table(TableIdent),
    /// A view.
    View(TableIdent),
    /// A catalog function.
    Function(Vec<String>),
    /// Files read or written by path (`volume`/path grant), in the
    /// qualified form the read or write uses; a URL's credentials stay
    /// typed.
    Path(OptionValue),
    /// A namespace (database or schema): the catalog name, then the
    /// namespace parts. Just the catalog name stands for the catalog
    /// itself (where namespaces are created).
    Namespace(Vec<String>),
    /// A system object: `system.*` tables (`Monitor` reads other
    /// principals' rows), or `cluster` for the cluster-scope statements
    /// of the `ADMIN` class.
    System(String),
}

impl fmt::Display for Securable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Table(t) => write!(f, "table {t}"),
            Self::View(t) => write!(f, "view {t}"),
            Self::Function(n) => write!(f, "function {}", quoted(n)),
            Self::Path(p) => write!(f, "path '{p}'"),
            Self::Namespace(n) => write!(f, "namespace {}", quoted(n)),
            Self::System(n) => write!(f, "system {}", quoted(std::slice::from_ref(n))),
        }
    }
}

/// A privilege.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Privilege {
    /// Read rows.
    Select,
    /// Insert, overwrite, update or delete rows; write files under a path.
    Modify,
    /// Create an object in a namespace or catalog, or files at a new path.
    Create,
    /// Drop (or replace) the object.
    Drop,
    /// Read other principals' rows in `system.*` tables, and run the
    /// cluster-scope statements of the `ADMIN` class.
    Monitor,
}

impl Privilege {
    /// Whether the privilege changes data or objects (`Modify`, `Create`,
    /// `Drop`): a principal whose reads of the securable are filtered is
    /// never granted one (see [`super::authorize`]).
    #[must_use]
    pub const fn changes(self) -> bool {
        matches!(self, Self::Modify | Self::Create | Self::Drop)
    }
}

impl fmt::Display for Privilege {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Select => "SELECT",
            Self::Modify => "MODIFY",
            Self::Create => "CREATE",
            Self::Drop => "DROP",
            Self::Monitor => "MONITOR",
        })
    }
}

/// Row filter and column masks for a securable, as unresolved expressions
/// over its columns.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RowColumnPolicy {
    /// Rows the principal may see.
    pub row_filter: Option<Arc<Expr>>,
    /// Column name → mask expression.
    pub column_masks: BTreeMap<String, Arc<Expr>>,
}

/// An access decision.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AccessDecision {
    /// Allowed.
    Allow,
    /// Denied; `reason` is for audit, never shown to the user.
    Deny {
        /// Why.
        reason: String,
    },
    /// Allowed through a row filter and column masks (applied from P5-T045;
    /// rejected until then).
    AllowWith(RowColumnPolicy),
}

/// The version of an access policy; part of the context fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PolicyVersion(pub u64);

/// The access-policy hook (ADR 0036 R2): called once per securable and
/// privilege a statement uses, with the session's principal, before the
/// resolver sees any of its metadata and before any side effect. Every
/// error and every denial fails the statement (see [`super::authorize`]).
pub trait AccessPolicy: Send + Sync {
    /// The policy version.
    fn version(&self) -> PolicyVersion;
    /// The decision for `principal` on `securable`.
    fn check<'a>(
        &'a self,
        principal: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>>;
}

/// The embedded default policy: everything is allowed.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllowAll;

impl AccessPolicy for AllowAll {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(0)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        _: &'a Securable,
        _: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        Box::pin(std::future::ready(Ok(AccessDecision::Allow)))
    }
}

/// A policy that denies everything (tests, and a closed default to build
/// grants on).
#[derive(Debug, Clone, Copy, Default)]
pub struct DenyAll;

impl AccessPolicy for DenyAll {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(0)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        _: &'a Securable,
        _: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        Box::pin(std::future::ready(Ok(AccessDecision::Deny {
            reason: "deny-all policy".to_owned(),
        })))
    }
}

/// The SQL front end, for `PlanNode::Sql` (and later view text): keeps this
/// crate free of the parser.
pub trait SqlFrontend: Send + Sync {
    /// Parses and lowers `query`, its spans indexing `source`.
    ///
    /// # Errors
    /// The parse or lowering error.
    fn lower(&self, query: &str, source: SourceId) -> Result<Plan, AnalysisError>;
    /// `query` with its secrets replaced.
    fn redact(&self, query: &str) -> RedactedSql;
    /// A digest of the options that change how text parses; part of the
    /// context fingerprint.
    fn fingerprint(&self) -> u64;
    /// The rows of `sql_keywords()`: `(keyword, reserved)` with the
    /// session's enforced-keywords flag applied. The list comes from the
    /// SQL front end so the analyzer never duplicates the parser's table.
    fn keywords(&self) -> Vec<(String, bool)>;
}

/// What a data source name in `` format.`path` `` is, for Spark's
/// `ResolveSQLOnFile`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SourceKind {
    /// A file format: the name reads files by path.
    File,
    /// A data source that does not read files (`jdbc`, `noop`, ...):
    /// `UNSUPPORTED_DATASOURCE_FOR_DIRECT_QUERY`.
    Other,
    /// Not a data source: the name stays a table name.
    Unknown,
}

/// Lists and reads the files of a read by path (`spark.read.parquet(path)`,
/// `` parquet.`path` ``), for their schema. The session implements it over
/// its file formats and stores; the analyzer stays free of I/O crates.
///
/// A plan that reads files depends on what the files hold when they are
/// listed, which no context input captures: [`super::AnalyzedPlan::reads_files`]
/// marks such plans, and a resolved-plan cache must list them again
/// (ADR 0037). An implementation serves one query: it lists each read once
/// and lets execution read the same files, and a new query gets a new
/// resolver (so it lists the files again).
///
/// The analyzer asks the access policy about each path in the form
/// [`Self::qualify`] gives, which is also the form the scan reads, so a
/// policy never sees one path while the query reads another.
pub trait FileSchemaResolver: Send + Sync {
    /// What the data source `name` is (case-insensitive).
    fn source_kind(&self, name: &str) -> SourceKind;
    /// `path` in the form the read uses: relative paths made absolute,
    /// `.` and `..` resolved, URLs in one spelling. The default keeps it
    /// as written.
    ///
    /// # Errors
    /// For a path the session cannot place (`KRISHIV_UNSUPPORTED_FEATURE`).
    fn qualify(&self, path: &OptionValue) -> Result<OptionValue, KrishivError> {
        Ok(path.clone())
    }
    /// The columns of reading `paths` as `format` with `options`, as the
    /// format infers them (the analyzer makes them nullable, as Spark does
    /// for every file source).
    ///
    /// # Errors
    /// `PATH_NOT_FOUND` for a path that does not exist,
    /// `UNABLE_TO_INFER_SCHEMA` when no file is left to infer from,
    /// `KRISHIV_UNSUPPORTED_FEATURE` for what the session cannot read yet,
    /// and storage failures.
    fn infer<'a>(
        &'a self,
        format: &'a str,
        paths: &'a [OptionValue],
        options: &'a BTreeMap<String, OptionValue>,
    ) -> BoxFuture<'a, Result<Vec<TableColumn>, KrishivError>>;
}

/// Tables in memory, by namespace and name, stored lower-case like Spark's
/// session catalog: lookups fold case unless `case_sensitive`.
#[derive(Debug, Clone, Default)]
pub struct MemoryCatalog {
    name: String,
    epoch: u64,
    tables: BTreeMap<(Vec<String>, String), TableMeta>,
}

impl MemoryCatalog {
    /// An empty catalog named `name`.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Adds (or replaces) a table and bumps the epoch.
    #[must_use]
    pub fn with_table(
        mut self,
        namespace: &[&str],
        name: &str,
        columns: Vec<TableColumn>,
        version: Option<TableVersion>,
    ) -> Self {
        let namespace: Vec<String> = namespace.iter().map(|s| s.to_lowercase()).collect();
        let ident = TableIdent {
            catalog: self.name.clone(),
            namespace: namespace.clone(),
            name: name.to_lowercase(),
        };
        let meta = TableMeta {
            ident: ident.clone(),
            columns,
            version,
            kind: TableKind::Table,
        };
        self.tables.insert((namespace, ident.name), meta);
        self.epoch += 1;
        self
    }
}

impl Catalog for MemoryCatalog {
    fn name(&self) -> &str {
        &self.name
    }

    fn epoch(&self) -> u64 {
        self.epoch
    }

    fn load_table<'a>(
        &'a self,
        id: &'a TableIdent,
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        let fold = |s: &String| {
            if form.case_sensitive {
                s.clone()
            } else {
                s.to_lowercase()
            }
        };
        let key = (id.namespace.iter().map(fold).collect(), fold(&id.name));
        let found = self.tables.get(&key).map(|m| TableMeta {
            version: version.or(m.version),
            ..m.clone()
        });
        Box::pin(std::future::ready(Ok(found)))
    }
}

#[cfg(test)]
mod tests {
    use super::super::BuiltinFunctions;
    use super::FunctionRegistry;

    #[test]
    fn registries_without_udfs_resolve_none() {
        // The default `resolve_udf`: registries hosting no UDFs answer `None`,
        // so constant folding leaves UDF calls alone.
        let builtins = BuiltinFunctions::new();
        assert!(builtins.resolve_udf("my_udf", 1).is_none());
        assert!(builtins.resolve_udf("upper", 2).is_none());
    }
}
