// SPDX-License-Identifier: Apache-2.0
//! Access control in the session: statement classification for policy
//! consumers (what a statement is, [`StatementClass`], and what it reads,
//! from analysis), the checks on what statements write, create and drop
//! (before any side effect), the audit record of every denial, and
//! [`PathPrefixPolicy`].
//!
//! Every privilege goes through [`krishiv_plan::analyzer::authorize`]: one
//! check per securable and privilege, failing closed with
//! `KRISHIV_INSUFFICIENT_PERMISSIONS` and never revealing the policy's
//! reason or whether the object exists.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use krishiv_common::error::{ErrorClass, KrishivError, KrishivUnsupportedFeature, PathNotFound};
use krishiv_common::secret::OptionValue;
use krishiv_exec::files::LocatedPath;
use krishiv_plan::SourceId;
use krishiv_plan::analyzer::{
    AccessDecision, AccessPolicy, BoxFuture, PolicyVersion, Principal, Privilege, Securable,
    SecurableUse, TableIdent, authorize, insufficient_permissions, namespace_parts, table_ident,
};
use krishiv_sql::ast::{DdlName, ExprKind, Literal, Statement};
use krishiv_sql::{
    ExplainKind, StatementClass, Target, TargetObject, parse_multipart_identifier,
    statement_targets,
};

use crate::dataframe::DataFrame;
use crate::session::{QueryEnv, Session, sql_context};

/// The namespace global temporary views live in (Spark's default
/// `spark.sql.globalTempDatabase`).
const GLOBAL_TEMP: &str = "global_temp";

/// What a statement is and what it reads: the input of a read-only
/// surface's allow-list (over [`Self::class`]) and of the analyze API.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StatementInfo {
    /// The statement's class (syntactic; `EXPLAIN ANALYZE` takes the class
    /// of the statement it runs).
    pub class: StatementClass,
    /// How it explains, if it is an `EXPLAIN`.
    pub explain: Option<ExplainKind>,
    /// What its queries read, with the columns they reference, in
    /// first-use order (temporary views expanded: a view is read with the
    /// rights of the session that reads it); then what it writes, creates
    /// or drops, and reads without a query, with no columns.
    pub securables: Vec<SecurableUse>,
}

impl Session {
    /// Classifies the SQL statement `sql` and lists the securables it uses
    /// (what it reads, writes, creates and drops), without running it. Reads are found by analysis, which asks the session's
    /// access policy about each of them: a denied read fails here as it
    /// would when the statement runs.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_schema::{DataType, Field, Schema};
    /// use krishiv_session::{Session, StatementClass};
    ///
    /// let s = Session::new()?;
    /// # s.register_batches("t", Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)])), vec![])?;
    /// let info = futures::executor::block_on(s.statement_info("SELECT a FROM t"))?;
    /// assert_eq!(info.class, StatementClass::Read);
    /// assert_eq!(info.securables.len(), 1);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Parse and analysis errors (`KRISHIV_INSUFFICIENT_PERMISSIONS` for a
    /// denied read); `KRISHIV_UNSUPPORTED_FEATURE` for a statement whose
    /// reads are not whole queries (a multi-table `INSERT` with a shared
    /// `FROM`, reads through `UPDATE`/`DELETE` predicates or variable
    /// expressions).
    pub async fn statement_info(&self, sql: &str) -> Result<StatementInfo, KrishivError> {
        let (session, sql) = (self.clone(), sql.to_owned());
        self.on_executor(async move { session.statement_info_here(&sql).await })
            .await
    }

    async fn statement_info_here(&self, sql: &str) -> Result<StatementInfo, KrishivError> {
        let env = self.query_env();
        let redacted = krishiv_sql::redact_sql(sql);
        let statement = krishiv_sql::parse_statement(sql, &env.parser)
            .map_err(|e| sql_context(e.error, &redacted, e.span.start, e.span.end))?;
        let class = krishiv_sql::classify(&statement);
        let explain = krishiv_sql::explain_kind(&statement);
        let queries = krishiv_sql::statement_queries(&statement).ok_or_else(|| {
            KrishivError::from(KrishivUnsupportedFeature {
                feature: "listing the reads of this statement".to_owned(),
            })
        })?;
        let mut securables: Vec<SecurableUse> = Vec::new();
        for query in queries {
            let plan = krishiv_sql::lower(query, SourceId(0))
                .map_err(|e| sql_context(e.error, &redacted, e.span.start, e.span.end))?;
            let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
            let analysis = df
                .analyzed_in_with_text(self.query_env(), Some(&redacted))
                .await?;
            for used in &analysis.analyzed.securables {
                match securables
                    .iter_mut()
                    .find(|s| s.securable == used.securable && s.privilege == used.privilege)
                {
                    Some(seen) => seen.columns.extend(used.columns.iter().cloned()),
                    None => securables.push(used.clone()),
                }
            }
        }
        // Then what it writes, creates or drops, as the session would check
        // it (the name as written).
        for target in statement_targets(&statement) {
            if let Some(securable) = self.securables_of(target.object, &env)?.into_iter().next()
                && !securables
                    .iter()
                    .any(|s| s.securable == securable && s.privilege == target.privilege)
            {
                securables.push(SecurableUse {
                    securable,
                    privilege: target.privilege,
                    columns: std::collections::BTreeSet::new(),
                });
            }
        }
        Ok(StatementInfo {
            class,
            explain,
            securables,
        })
    }
}

/// The parts of a function name as written.
fn object_parts(name: &krishiv_sql::ast::ObjectName) -> Vec<String> {
    name.parts.iter().map(|p| p.value.clone()).collect()
}

/// Logs `result`'s denial, if it is one, as a structured audit record
/// (target `krishiv::audit`): the principal, the class of what was denied,
/// and the denial (privilege and securable; never the policy's reason).
pub(crate) fn audit<T>(
    session: &Session,
    class: StatementClass,
    result: Result<T, KrishivError>,
) -> Result<T, KrishivError> {
    if let Err(e) = &result
        && e.error_class() == ErrorClass::KrishivInsufficientPermissions
    {
        log_denial(session, class, e);
    }
    result
}

/// The audit record of the denial `e` (see [`audit`]): also written when
/// an access check fails closed because it cannot name what a statement
/// touches (`KRISHIV_UNSUPPORTED_FEATURE`).
fn log_denial(session: &Session, class: StatementClass, e: &KrishivError) {
    tracing::warn!(
        target: "krishiv::audit",
        principal = %session.state.principal.name,
        statement_class = class.name(),
        error_class = e.error_class().name(),
        denial = %e,
        "access denied"
    );
}

impl Session {
    /// Checks what `statement` (of `class`) writes, creates or drops, and
    /// what it reads without a query, before it runs; an `ADMIN` statement
    /// needs `Monitor` on the cluster. Names are located as analysis locates
    /// them and asked about as written and, unless analysis is
    /// case-sensitive, folded; paths in their qualified form.
    pub(crate) async fn authorize_statement(
        &self,
        statement: &Statement,
        class: StatementClass,
        env: &QueryEnv,
    ) -> Result<(), KrishivError> {
        let mut targets = statement_targets(statement);
        if class == StatementClass::Admin
            && !targets.iter().any(|t| t.object == TargetObject::Cluster)
        {
            targets.push(Target {
                object: TargetObject::Cluster,
                privilege: Privilege::Monitor,
            });
        }
        let mut checked: Vec<(Securable, Privilege)> = Vec::new();
        for target in targets {
            let securables = match self.securables_of(target.object, env) {
                Ok(s) => s,
                Err(e) => {
                    log_denial(self, class, &e);
                    return Err(e);
                }
            };
            for securable in securables {
                if checked
                    .iter()
                    .any(|(s, p)| *s == securable && *p == target.privilege)
                {
                    continue;
                }
                let r = authorize(
                    &*self.state.policy,
                    &self.state.principal,
                    &securable,
                    target.privilege,
                )
                .await;
                audit(self, class, r)?;
                checked.push((securable, target.privilege));
            }
        }
        Ok(())
    }

    /// Asks the policy about the catalog-store lookups of a query, in plan order, before any of
    /// them is read: permanent views are inlined before binding, so the analyzer never asks about
    /// them. Each name is asked as the analyzer asks about a table (as written, then folded, whether
    /// or not it exists, so a denial cannot tell a view from a missing name), then a permanent view
    /// is asked as a view. Returns the views, for the securables the analysis reports.
    pub(crate) async fn authorize_reads(
        &self,
        reads: &[crate::catalog::ReadCandidate],
        env: &QueryEnv,
    ) -> Result<Vec<TableIdent>, KrishivError> {
        let catalogs = &*env.analysis.catalogs;
        let fold = !env.analysis.conf.case_sensitive;
        let mut views = Vec::new();
        let mut checked: Vec<Securable> = Vec::new();
        for read in reads {
            let ident =
                table_ident(catalogs, &read.written).ok_or_else(|| unplaceable("an empty name"))?;
            let mut securables = vec![Securable::Table(ident.clone())];
            if fold {
                let lower = folded(&ident);
                if lower != ident {
                    securables.push(Securable::Table(lower));
                }
            }
            if let Some((namespace, name)) = &read.view {
                let parts = [namespace.clone(), name.clone()];
                let view =
                    table_ident(catalogs, &parts).ok_or_else(|| unplaceable("an empty name"))?;
                securables.push(Securable::View(view.clone()));
                views.push(view);
            }
            for securable in securables {
                if checked.contains(&securable) {
                    continue;
                }
                let r = authorize(
                    &*self.state.policy,
                    &self.state.principal,
                    &securable,
                    Privilege::Select,
                )
                .await;
                audit(self, StatementClass::Read, r)?;
                checked.push(securable);
            }
        }
        Ok(views)
    }

    /// `SELECT` on the permanent view `namespace.name`: the statements that
    /// show a view's definition, columns or properties need it as a query of
    /// the view does, besides the table-named check made before dispatch.
    pub(crate) async fn authorize_view_select(
        &self,
        namespace: &str,
        name: &str,
        env: &QueryEnv,
    ) -> Result<(), KrishivError> {
        let parts = [namespace.to_owned(), name.to_owned()];
        let ident = table_ident(&*env.analysis.catalogs, &parts)
            .ok_or_else(|| unplaceable("an empty name"))?;
        let r = authorize(
            &*self.state.policy,
            &self.state.principal,
            &Securable::View(ident),
            Privilege::Select,
        )
        .await;
        audit(self, StatementClass::Read, r)
    }

    /// The securables `object` stands for: as written, then folded.
    fn securables_of(
        &self,
        object: TargetObject<'_>,
        env: &QueryEnv,
    ) -> Result<Vec<Securable>, KrishivError> {
        let catalogs = &*env.analysis.catalogs;
        let fold = !env.analysis.conf.case_sensitive;
        let parts = |name: &DdlName| ddl_parts(name, env);
        let table = |parts: &[String]| {
            table_ident(catalogs, parts).ok_or_else(|| unplaceable("an empty name"))
        };
        let current_catalog = || catalogs.current_catalog().to_owned();
        let tables = |ident: TableIdent, view: bool| {
            let make = |i| {
                if view {
                    Securable::View(i)
                } else {
                    Securable::Table(i)
                }
            };
            let mut out = vec![make(ident.clone())];
            if fold {
                let folded = folded(&ident);
                if folded != ident {
                    out.push(make(folded));
                }
            }
            out
        };
        let namespaces = |parts: Vec<String>| {
            let mut out = vec![Securable::Namespace(parts.clone())];
            if fold {
                let folded: Vec<String> = parts
                    .iter()
                    .enumerate()
                    .map(|(i, p)| if i == 0 { p.clone() } else { p.to_lowercase() })
                    .collect();
                if folded != parts {
                    out.push(Securable::Namespace(folded));
                }
            }
            out
        };
        Ok(match object {
            TargetObject::Table(name) => tables(table(&parts(name)?)?, false),
            TargetObject::TableIn {
                table: t,
                namespace,
            } => {
                let mut p = parts(namespace)?;
                p.extend(parts(t)?.pop());
                tables(table(&p)?, false)
            }
            TargetObject::View(name) => tables(table(&parts(name)?)?, true),
            TargetObject::GlobalTempView(name) => {
                let name = parts(name)?
                    .pop()
                    .ok_or_else(|| unplaceable("an empty name"))?;
                tables(
                    TableIdent {
                        catalog: current_catalog(),
                        namespace: vec![GLOBAL_TEMP.to_owned()],
                        name,
                    },
                    true,
                )
            }
            TargetObject::ParentOf(name) => {
                let ident = table(&parts(name)?)?;
                namespaces(
                    std::iter::once(ident.catalog)
                        .chain(ident.namespace)
                        .collect(),
                )
            }
            TargetObject::Function(name) => {
                let ident = table(&object_parts(name))?;
                let mut full = vec![ident.catalog];
                full.extend(ident.namespace);
                full.push(ident.name);
                let mut out = vec![Securable::Function(full.clone())];
                if fold {
                    let folded: Vec<String> = full
                        .iter()
                        .enumerate()
                        .map(|(i, p)| if i == 0 { p.clone() } else { p.to_lowercase() })
                        .collect();
                    if folded != full {
                        out.push(Securable::Function(folded));
                    }
                }
                out
            }
            TargetObject::ParentOfFunction(name) => {
                let ident = table(&object_parts(name))?;
                namespaces(
                    std::iter::once(ident.catalog)
                        .chain(ident.namespace)
                        .collect(),
                )
            }
            TargetObject::GlobalTempNamespace => {
                namespaces(vec![current_catalog(), GLOBAL_TEMP.to_owned()])
            }
            TargetObject::Namespace(name) => namespaces(namespace_parts(catalogs, &parts(name)?)),
            TargetObject::ParentOfNamespace(name) => {
                let mut p = namespace_parts(catalogs, &parts(name)?);
                if p.len() > 1 {
                    p.pop();
                }
                namespaces(p)
            }
            TargetObject::CurrentNamespace => namespaces(
                std::iter::once(current_catalog())
                    .chain(catalogs.current_namespace().iter().cloned())
                    .collect(),
            ),
            TargetObject::CurrentCatalog => namespaces(vec![current_catalog()]),
            TargetObject::Path(path) => vec![Securable::Path(OptionValue::from_user(
                "path",
                self.qualify_path(path)?,
            ))],
            TargetObject::Cluster => vec![Securable::System("cluster".to_owned())],
            // A target this session cannot place fails closed.
            _ => return Err(unplaceable("this statement target")),
        })
    }
}

/// The parts of `name`: as written, or the constant string of
/// `IDENTIFIER('...')` parsed as a multipart name.
pub(crate) fn ddl_parts(name: &DdlName, env: &QueryEnv) -> Result<Vec<String>, KrishivError> {
    let parts = match name {
        DdlName::Parts(n) => n.parts.iter().map(|p| p.value.clone()).collect(),
        DdlName::Dynamic(d) => match &d.expr.kind {
            ExprKind::Literal(Literal::String(text)) => {
                parse_multipart_identifier(text, &env.parser)
                    .map_err(|e| e.error)?
                    .parts
                    .into_iter()
                    .map(|p| p.value)
                    .collect()
            }
            _ => {
                return Err(unplaceable(
                    "IDENTIFIER() names that are not a string literal",
                ));
            }
        },
        _ => return Err(unplaceable("this kind of name")),
    };
    Ok(parts)
}

/// `KRISHIV_UNSUPPORTED_FEATURE` for an object the access checks cannot
/// name: the statement does not run (fail closed).
fn unplaceable(what: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: format!("access checks on {what}"),
    }
    .into()
}

/// `ident` as a case-insensitive catalog sees it (the catalog name is kept).
fn folded(ident: &TableIdent) -> TableIdent {
    TableIdent {
        catalog: ident.catalog.clone(),
        namespace: ident.namespace.iter().map(|s| s.to_lowercase()).collect(),
        name: ident.name.to_lowercase(),
    }
}

/// `at` (a local path, `file:/...`) with the symbolic links of its
/// existing part resolved: the directory a write really goes to.
pub(crate) fn physical(at: LocatedPath) -> Result<LocatedPath, KrishivError> {
    let Some(local) = at.display.strip_prefix("file:") else {
        return Ok(at);
    };
    let resolved = resolve_links(Path::new(local)).ok_or_else(|| unplaceable("this path"))?;
    let parts = resolved
        .components()
        .filter_map(|c| match c {
            Component::Normal(p) => Some(p.to_str().ok_or_else(|| unplaceable("non-UTF-8 paths"))),
            _ => None,
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LocatedPath {
        store: at.store,
        path: object_store::path::Path::from_iter(parts.iter().copied()),
        display: format!("file:/{}", parts.join("/")),
    })
}

/// Fails closed (`KRISHIV_INSUFFICIENT_PERMISSIONS`) when `at`, a path
/// [`physical`] resolved and the policy allowed, now resolves elsewhere: a
/// symbolic link was swapped into it.
pub(crate) fn still_physical(session: &Session, at: &LocatedPath) -> Result<(), KrishivError> {
    let now = physical(at.clone())?;
    if now.display == at.display {
        return Ok(());
    }
    let securable = Securable::Path(OptionValue::from_user("path", now.display));
    audit(
        session,
        StatementClass::Write,
        Err(insufficient_permissions(
            &session.state.principal,
            &securable,
            Privilege::Modify,
        )),
    )
}

/// Allows paths only under the given root directories, and delegates every
/// other decision to `inner`: the policy behind a server's data-directory
/// restriction.
///
/// A path is compared in the qualified form reads and writes use
/// (`file:/a/b`, `.` and `..` already resolved), after resolving symbolic
/// links in the part of it that exists, against the roots with their links
/// resolved. Anything else is denied: other schemes (`s3://`, ...), paths
/// with credentials, any path containing `%` (percent-encoded text would
/// not be decoded the way the comparison reads it, so even a literal `%`
/// in a file name is refused) and relative or unresolved paths. Session
/// writes go to the resolved directory, which is opened once and then used
/// by descriptor (no name is resolved through a link while the write runs),
/// and fail if a link is swapped into it before the write begins or commits.
/// Resolving links reads file metadata on the calling thread, once per
/// path and query.
///
/// ```
/// use std::sync::Arc;
/// use krishiv_session::{AllowAll, PathPrefixPolicy, Session};
///
/// let dir = tempfile::tempdir().unwrap();
/// let policy = PathPrefixPolicy::new([dir.path()], Arc::new(AllowAll))?;
/// let s = Session::builder().access_policy(Arc::new(policy)).build()?;
/// let e = futures::executor::block_on(s.sql("SELECT * FROM parquet.`/etc`")).unwrap_err();
/// assert_eq!(e.error_class().name(), "KRISHIV_INSUFFICIENT_PERMISSIONS");
/// # Ok::<(), krishiv_common::error::KrishivError>(())
/// ```
#[derive(Clone)]
pub struct PathPrefixPolicy {
    roots: Vec<PathBuf>,
    inner: Arc<dyn AccessPolicy>,
}

impl std::fmt::Debug for PathPrefixPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PathPrefixPolicy")
            .field("roots", &self.roots)
            .finish_non_exhaustive()
    }
}

impl PathPrefixPolicy {
    /// A policy allowing paths under `roots` (existing directories), with
    /// `inner` deciding everything else.
    ///
    /// # Errors
    /// `PATH_NOT_FOUND` for a root that does not exist.
    pub fn new<P: AsRef<Path>>(
        roots: impl IntoIterator<Item = P>,
        inner: Arc<dyn AccessPolicy>,
    ) -> Result<Self, KrishivError> {
        let roots = roots
            .into_iter()
            .map(|r| {
                std::fs::canonicalize(r.as_ref()).map_err(|_| {
                    KrishivError::from(PathNotFound {
                        path: format!("file:{}", r.as_ref().display()),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { roots, inner })
    }

    /// Whether the qualified path `path` lies under a root.
    fn allows(&self, path: &OptionValue) -> bool {
        let Some(text) = path.as_plain() else {
            return false;
        };
        let Some(local) = text.strip_prefix("file:") else {
            return false;
        };
        let local = Path::new(local);
        let normal = local.is_absolute()
            && local
                .components()
                .all(|c| matches!(c, Component::RootDir | Component::Normal(_)));
        if !normal || text.contains('%') {
            return false;
        }
        resolve_links(local).is_some_and(|p| self.roots.iter().any(|r| p.starts_with(r)))
    }
}

/// `path` with the symbolic links of its longest existing prefix resolved
/// (the rest does not exist yet, so it holds no links).
fn resolve_links(path: &Path) -> Option<PathBuf> {
    let mut existing = path.to_path_buf();
    let mut rest = Vec::new();
    loop {
        if let Ok(resolved) = std::fs::canonicalize(&existing) {
            return Some(rest.iter().rev().fold(resolved, |p, part| p.join(part)));
        }
        rest.push(existing.file_name()?.to_owned());
        if !existing.pop() {
            return None;
        }
    }
}

impl AccessPolicy for PathPrefixPolicy {
    fn version(&self) -> PolicyVersion {
        self.inner.version()
    }

    fn check<'a>(
        &'a self,
        principal: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        if let Securable::Path(path) = securable
            && !self.allows(path)
        {
            return Box::pin(std::future::ready(Ok(AccessDecision::Deny {
                reason: "path outside the allowed roots".to_owned(),
            })));
        }
        self.inner.check(principal, securable, privilege)
    }
}
