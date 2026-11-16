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

use krishiv_common::config::{
    Config, ConfigValue, IO_LOCAL_ALLOWED_ROOTS, IO_OBJECT_STORE_ALLOWED_PREFIXES,
};
use krishiv_common::error::{
    ErrorClass, KrishivError, KrishivInvalidConfigValue, KrishivUnsupportedFeature, PathNotFound,
};
use krishiv_common::secret::OptionValue;
use krishiv_exec::files::{LocatedPath, PathLocator};
use krishiv_io::{StoreId, parse_object_url};
use krishiv_plan::SourceId;
use krishiv_plan::analyzer::{
    AccessDecision, AccessPolicy, BoxFuture, PolicyVersion, Principal, Privilege, Securable,
    SecurableUse, TableIdent, authorize, authorize_reported, insufficient_permissions,
    namespace_parts, table_ident,
};
use krishiv_sql::ast::{DdlName, ExprKind, Literal, Statement, TableOption};
use krishiv_sql::{
    ExplainKind, ParserOptions, StatementClass, Target, TargetObject, parse_multipart_identifier,
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
    /// reads are not known from its text (a `MERGE` with a subquery in an
    /// expression, `EXECUTE IMMEDIATE` of non-literal text or with `USING`,
    /// scripts and variable expressions; `krishiv_sql::statement_reads`).
    pub async fn statement_info(&self, sql: &str) -> Result<StatementInfo, KrishivError> {
        let (session, sql) = (self.clone(), sql.to_owned());
        self.on_executor(async move { session.statement_info_here(&sql).await })
            .await
    }

    async fn statement_info_here(&self, sql: &str) -> Result<StatementInfo, KrishivError> {
        let parts = self.statement_parts(sql).await?;
        if parts.partial {
            return Err(KrishivUnsupportedFeature {
                feature: "listing the reads of this statement".to_owned(),
            }
            .into());
        }
        Ok(parts.info)
    }

    /// The statement `sql` parsed, classified and its reads analyzed (see [`Self::statement_info`]),
    /// for `statement_info` and the analyze API. When its reads are not known from its text
    /// (`krishiv_sql::statement_reads` gives `None`), nothing is analyzed and `partial` is set: the
    /// securables are then its targets only.
    pub(crate) async fn statement_parts(&self, sql: &str) -> Result<StatementParts, KrishivError> {
        let env = self.query_env();
        let sql = &*self.substituted(sql);
        let redacted = krishiv_sql::redact_sql(sql);
        let statement = krishiv_sql::parse_statement(sql, &env.parser)
            .map_err(|e| sql_context(e.error, &redacted, e.span.start, e.span.end))?;
        self.parts_of_statement(statement, &redacted).await
    }

    /// [`Self::statement_parts`] of the parsed `statement`, whose spans index `redacted`'s text
    /// (`EXPLAIN FORMAT JSON` analyzes its explained statement here).
    pub(crate) async fn parts_of_statement(
        &self,
        statement: Statement,
        redacted: &krishiv_common::redact::RedactedSql,
    ) -> Result<StatementParts, KrishivError> {
        let env = self.query_env();
        let class = krishiv_sql::classify_with(&statement, &env.parser);
        if let Statement::ExecuteImmediate(e) = &statement {
            return Box::pin(self.immediate_parts(e, class, statement.clone())).await;
        }
        let explain = krishiv_sql::explain_kind(&statement);
        let queries = krishiv_sql::statement_reads(&statement);
        let partial = queries.is_none();
        let mut securables: Vec<SecurableUse> = Vec::new();
        let mut analyzed = Vec::new();
        for query in queries.unwrap_or_default() {
            let plan = krishiv_sql::lower(&query, SourceId(0))
                .map_err(|e| sql_context(e.error, redacted, e.span.start, e.span.end))?;
            let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
            let analysis = df
                .analyzed_in_with_text(self.query_env(), Some(redacted))
                .await?;
            merge_reads(&mut securables, &analysis.analyzed.securables);
            analyzed.push(Arc::clone(&analysis.analyzed));
        }
        self.add_targets(&mut securables, &statement, &env, false)?;
        Ok(StatementParts {
            info: StatementInfo {
                class,
                explain,
                securables,
            },
            partial,
            analyzed,
            statement,
        })
    }
}

impl Session {
    /// Adds what `statement` writes, creates or drops to `securables`, as the session would check it
    /// (the name as written), and `Manage` on the cluster for `ADMIN` ([`required_targets`]). With
    /// `skip_unplaced`, a target that cannot be placed (an `IDENTIFIER(:marker)` name of text not yet
    /// bound) is left out instead of failing.
    fn add_targets(
        &self,
        securables: &mut Vec<SecurableUse>,
        statement: &Statement,
        env: &QueryEnv,
        skip_unplaced: bool,
    ) -> Result<(), KrishivError> {
        for target in required_targets(statement, &env.parser) {
            let placed = match self.securables_of(target.object, env) {
                Ok(placed) => placed,
                Err(_) if skip_unplaced => continue,
                Err(e) => return Err(e),
            };
            if let Some(securable) = placed.into_iter().next()
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
        Ok(())
    }

    /// The parts of `EXECUTE IMMEDIATE` (of `class`, see `krishiv_sql::classify_with`): with literal
    /// text and no `USING`, those of the statement the text holds (its spans index the text), under the
    /// outer class (`INTO` makes it at least `SESSION`); text with `USING` arguments is bound only when
    /// it runs, so it is `partial` with the inner statement's class and targets (the text parses with
    /// its markers); text from a variable or an expression is `ADMIN` and `partial`, permanently
    /// (design §3.2). Nothing is evaluated.
    async fn immediate_parts(
        &self,
        e: &krishiv_sql::ast::ExecuteImmediate,
        class: StatementClass,
        statement: Statement,
    ) -> Result<StatementParts, KrishivError> {
        let env = self.query_env();
        // A nested `EXECUTE IMMEDIATE` or a script is refused when it runs: nothing to list.
        let inner = krishiv_sql::immediate_statement(e, &env.parser)
            .filter(|i| !matches!(i, Statement::ExecuteImmediate(_) | Statement::Script(_)));
        let inner = match inner {
            Some(inner) if e.using.is_empty() => inner,
            known => {
                let mut securables = Vec::new();
                if let Some(inner) = &known {
                    // Its markers are unbound: a name made of one cannot be placed yet.
                    self.add_targets(&mut securables, inner, &env, true)?;
                }
                return Ok(StatementParts {
                    info: StatementInfo {
                        class,
                        explain: None,
                        securables,
                    },
                    partial: true,
                    analyzed: Vec::new(),
                    statement,
                });
            }
        };
        let inner_redacted =
            krishiv_sql::redact_sql(krishiv_sql::immediate_text(e).unwrap_or_default());
        let mut parts = self.parts_of_statement(inner, &inner_redacted).await?;
        parts.info.class = class;
        if !e.into.is_empty() {
            // The row goes to variables: no result columns or lineage.
            parts.statement = statement;
        }
        Ok(parts)
    }
}

/// A statement as [`Session::statement_parts`] found it.
pub(crate) struct StatementParts {
    /// Its class, explain kind and securables.
    pub(crate) info: StatementInfo,
    /// Whether its reads are not whole queries (none were analyzed).
    pub(crate) partial: bool,
    /// The analyses of its queries, in order.
    pub(crate) analyzed: Vec<Arc<krishiv_plan::analyzer::AnalyzedPlan>>,
    /// The parsed statement.
    pub(crate) statement: Statement,
}

/// Adds the reads `used` to `securables`: one entry per (securable, privilege), with the union of
/// the columns.
pub(crate) fn merge_reads(securables: &mut Vec<SecurableUse>, used: &[SecurableUse]) {
    for used in used {
        match securables
            .iter_mut()
            .find(|s| s.securable == used.securable && s.privilege == used.privilege)
        {
            Some(seen) => seen.columns.extend(used.columns.iter().cloned()),
            None => securables.push(used.clone()),
        }
    }
}

/// What `statement` writes, creates or drops and reads without a query (`statement_targets`), plus
/// `Manage` on the cluster for every `ADMIN` statement (ADR 0046 §4), whatever else it names: what the
/// session checks before it runs, and what `statement_info` lists. `options` read the text of an
/// `EXECUTE IMMEDIATE`.
///
/// Dynamic SQL (`EXECUTE IMMEDIATE` of text that cannot be placed, reported `ADMIN`) adds no `Manage`
/// wherever it stands, in a script or under `EXPLAIN ANALYZE` too (`krishiv_sql::needs_cluster_manage`):
/// the statement it runs is authorized under its own class when it runs (`execute_immediate.rs`).
fn required_targets<'s>(statement: &'s Statement, options: &ParserOptions) -> Vec<Target<'s>> {
    let mut targets = statement_targets(statement);
    if krishiv_sql::needs_cluster_manage(statement, options)
        && !targets
            .iter()
            .any(|t| t.object == TargetObject::Cluster && t.privilege == Privilege::Manage)
    {
        targets.push(Target {
            object: TargetObject::Cluster,
            privilege: Privilege::Manage,
        });
    }
    targets
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

/// [`audit`] of a query's read check, unless `env` is a trial analysis
/// ([`QueryEnv::quiet_denials`]), whose denials are not logged.
pub(crate) fn audit_read<T>(
    session: &Session,
    env: &QueryEnv,
    result: Result<T, KrishivError>,
) -> Result<T, KrishivError> {
    if env.quiet_denials {
        return result;
    }
    audit(session, StatementClass::Read, result)
}

/// The audit record of the denial `e` (see [`audit`]): also written when
/// an access check fails closed because it cannot name what a statement
/// touches (`KRISHIV_UNSUPPORTED_FEATURE`).
///
/// It is the one record of a refusal (review M4): a provider only counts it. `provider` is the
/// `krishiv_access_denied_total` label of who decided: with the grants provider `grants` or
/// `roles` (the role matrix decides `System(..)` and the `system` catalog), without it `roles`;
/// `policy_epoch` is the policy's version (the grants epoch mixed in); the query id is the
/// enclosing span's.
fn log_denial(session: &Session, class: StatementClass, e: &KrishivError) {
    if e.error_class() == ErrorClass::KrishivInsufficientPermissions
        && let Some((counts, surface)) = session.state.denials.get()
        && let Some((_, privilege)) = e
            .message_parameters()
            .iter()
            .find(|(name, _)| *name == "privilege")
    {
        counts.record_named(*surface, privilege);
    }
    // ADR 0046 §5: the client's own `user_id` is recorded, marked unverified, never used.
    let client_user_id = session.state.activity.client_user_id();
    let provider = if session.state.grants.is_some() {
        crate::grants::denial_provider(e)
    } else {
        crate::grants::DecidedBy::Roles
    }
    .as_str();
    // A refusal of a principal acting for this one (the definer of a `SQL SECURITY DEFINER` view
    // the session read, P3-T062 §2.4) names it: the record shows both.
    let own =
        krishiv_common::names::quote_parts(std::slice::from_ref(&session.state.principal.name));
    let acting_as = e
        .message_parameters()
        .iter()
        .find(|(name, _)| *name == "principal")
        .map(|(_, p)| p.clone())
        .filter(|p| *p != own);
    tracing::warn!(
        target: "krishiv::audit",
        principal = %session.state.principal.name,
        acting_as = acting_as.as_deref(),
        delegated_by = session.state.principal.delegated_by.as_deref(),
        definer_for = session.state.principal.definer_for.as_deref(),
        client_user_id_unverified = client_user_id.as_deref(),
        statement_class = class.name(),
        error_class = e.error_class().name(),
        denial = %e,
        decision = "deny",
        provider,
        policy_epoch = session.state.policy.version().0,
        "access denied"
    );
}

impl Session {
    /// Counts every `KRISHIV_INSUFFICIENT_PERMISSIONS` denial of this session into `counts`
    /// under `surface` from now on (a server's sessions: `krishiv_auth_denied_total`, ADR 0046
    /// §8). Set once; a second call changes nothing and returns `false`.
    pub fn count_denials(
        &self,
        counts: Arc<crate::auth::DeniedCounts>,
        surface: crate::auth::Surface,
    ) -> bool {
        self.state.denials.set((counts, surface)).is_ok()
    }

    /// Checks what `statement` (of `class`) writes, creates or drops, and
    /// what it reads without a query, before it runs; an `ADMIN` statement
    /// needs `Manage` on the cluster (ADR 0046 §4). Names are located as analysis locates
    /// them and asked about as written and, unless analysis is
    /// case-sensitive, folded; paths in their qualified form.
    pub(crate) async fn authorize_statement(
        &self,
        statement: &Statement,
        class: StatementClass,
        env: &QueryEnv,
    ) -> Result<(), KrishivError> {
        if let Err(e) = reject_system_namespace(statement) {
            log_denial(self, class, &e);
            return Err(e);
        }
        let targets = required_targets(statement, &env.parser);
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
                if let Err(e) = reject_system_write(&securable, target.privilege) {
                    log_denial(self, class, &e);
                    return Err(e);
                }
                if checked
                    .iter()
                    .any(|(s, p)| *s == securable && *p == target.privilege)
                {
                    continue;
                }
                if self
                    .first_write_of_new_table(
                        statement,
                        &securable,
                        target.privilege,
                        &checked,
                        env,
                    )
                    .await
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

    /// Whether `privilege` on `securable` is the first write of the table a `CREATE TABLE .. AS`
    /// is creating, under the grants provider (review open point 2): `Create` on its namespace,
    /// already asked and allowed in `checked`, covers it, because the creator will own the
    /// table. Only when the table does not exist (the ownership bookkeeping records the creator
    /// only if this statement really creates it, and a create of an existing table fails or
    /// writes nothing).
    async fn first_write_of_new_table(
        &self,
        statement: &Statement,
        securable: &Securable,
        privilege: Privilege,
        checked: &[(Securable, Privilege)],
        env: &QueryEnv,
    ) -> bool {
        let Statement::CreateTable(c) = statement else {
            return false;
        };
        let Securable::Table(t) = securable else {
            return false;
        };
        if self.state.grants.is_none()
            || privilege != Privilege::Modify
            || c.as_query.is_none()
            || c.temporary
            || c.location.is_some()
        {
            return false;
        }
        let parent: Vec<String> = std::iter::once(t.catalog.clone())
            .chain(t.namespace.iter().cloned())
            .collect();
        let created_in_parent = checked.iter().any(|(s, p)| {
            *p == Privilege::Create
                && matches!(s, Securable::Namespace(n) if n.len() == parent.len()
                    && n.iter().zip(&parent).all(|(a, b)| a.eq_ignore_ascii_case(b)))
        });
        created_in_parent && self.object_exists(securable, env).await == Some(false)
    }

    /// Asks the policy about the catalog-store lookups of a query, in plan order, before any of
    /// them is read: permanent views are inlined before binding, so the analyzer never asks about
    /// them. Each name is asked as the analyzer asks about a table (as written, then folded, whether
    /// or not it exists, so a denial cannot tell a view from a missing name), then a permanent view
    /// is asked as a view. Each is asked the analysis' read privilege (`Select`; `Browse` for a
    /// metadata analysis, `AnalysisMode::Metadata`). Returns the views, for the securables the
    /// analysis reports.
    pub(crate) async fn authorize_reads(
        &self,
        reads: &[crate::catalog::ReadCandidate],
        env: &QueryEnv,
    ) -> Result<Vec<TableIdent>, KrishivError> {
        let principal = Arc::clone(&self.state.principal);
        self.authorize_reads_as(reads, env, &principal).await
    }

    /// [`Self::authorize_reads`] for `principal`: the session's own, or the definer of a
    /// `SQL SECURITY DEFINER` view acting for it (P3-T062 §2.4).
    pub(crate) async fn authorize_reads_as(
        &self,
        reads: &[crate::catalog::ReadCandidate],
        env: &QueryEnv,
        principal: &Principal,
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
                // A stored view belongs to the session catalog, whatever catalog is current.
                let parts = [
                    crate::catalog::CATALOG.to_owned(),
                    namespace.clone(),
                    name.clone(),
                ];
                let view =
                    table_ident(catalogs, &parts).ok_or_else(|| unplaceable("an empty name"))?;
                securables.push(Securable::View(view.clone()));
                views.push(view);
            }
            for securable in securables {
                if checked.contains(&securable) {
                    continue;
                }
                // The analysis' read privilege (`Select`, or `Browse` for a metadata analysis). A
                // policy that decides columns is asked here only whether the name may be seen; the
                // table's rows are asked per column after analysis. A stored view is still asked
                // `Select` as a whole (column grants on views are not supported).
                let wanted = env.analysis.mode.read_privilege();
                let privilege = if wanted == Privilege::Select
                    && self.state.policy.column_level()
                    && matches!(securable, Securable::Table(_))
                {
                    Privilege::Browse
                } else {
                    wanted
                };
                let r = authorize_reported(
                    &*self.state.policy,
                    principal,
                    &securable,
                    privilege,
                    wanted,
                )
                .await;
                audit_read(self, env, r)?;
                checked.push(securable);
            }
        }
        Ok(views)
    }

    /// The analysis' read privilege (`Select`; `Browse` for a metadata analysis) on every stored
    /// function the query's analysis resolved a call to (P3-T062 §2.4: calling a function needs
    /// `Select` on it; there is no `Execute`, Q3). Built-in, temporary and session functions are
    /// not securables. A call inside a definer view's text is asked of the reader too (fails
    /// closed). Each refusal is the caller's to audit.
    pub(crate) async fn authorize_function_calls(
        &self,
        env: &QueryEnv,
    ) -> Result<(), KrishivError> {
        let privilege = env.analysis.mode.read_privilege();
        for (namespace, name) in env.sql_functions.stored_calls() {
            let securable =
                Securable::Function(vec![crate::catalog::CATALOG.to_owned(), namespace, name]);
            authorize(
                &*self.state.policy,
                &self.state.principal,
                &securable,
                privilege,
            )
            .await?;
        }
        Ok(())
    }

    /// Asks the session's principal what the `SQL SECURITY DEFINER` views of `analyzed` read
    /// (their `owner_reads`), as an invoker would be asked, before their inlined queries are shown
    /// (`EXPLAIN`, estimates, warnings; P3-T062 §2.4: an explain checks the privileges of the
    /// statement it explains). Column-level policies are asked per column read. Audited.
    ///
    /// # Errors
    /// The first refusal (42501).
    pub(crate) async fn authorize_shown_definer_reads(
        &self,
        analyzed: &krishiv_plan::analyzer::AnalyzedPlan,
    ) -> Result<(), KrishivError> {
        use krishiv_plan::analyzer::views::ViewSecurity;
        let policy = &*self.state.policy;
        for view in analyzed
            .views
            .iter()
            .filter(|v| v.security == ViewSecurity::Definer)
        {
            for used in &view.owner_reads {
                let per_column = used.privilege == Privilege::Select
                    && policy.column_level()
                    && !used.columns.is_empty();
                let asked: Vec<Securable> = match &used.securable {
                    Securable::Table(t) | Securable::View(t) if per_column => used
                        .columns
                        .iter()
                        .map(|c| Securable::Column {
                            table: t.clone(),
                            column: c.clone(),
                        })
                        .collect(),
                    other => vec![other.clone()],
                };
                for securable in asked {
                    let r =
                        authorize(policy, &self.state.principal, &securable, used.privilege).await;
                    audit(self, StatementClass::Read, r)?;
                }
            }
        }
        Ok(())
    }

    /// Whether this session's principal may see `securable` in a listing (`SHOW TABLES`,
    /// `SHOW VIEWS`, `SHOW NAMESPACES`, `spark.catalog.list*`; P3-T062 §2.4): under the `grants`
    /// provider when it may `Select` it or holds any grant on it (`GrantsPolicy::listable`; a
    /// `CREATE` on the namespace does not show every table in it), otherwise `Browse` on it.
    /// Asked quietly: leaving an entry out is not a refusal (no audit record, no denial counter).
    pub(crate) async fn may_list(&self, securable: &Securable) -> bool {
        if let Some(grants) = &self.state.grants
            && let Some(listed) = grants.listable(&self.state.principal, securable)
        {
            return listed;
        }
        krishiv_plan::analyzer::allowed(
            &*self.state.policy,
            &self.state.principal,
            securable,
            Privilege::Browse,
        )
        .await
    }

    /// The relations of the session catalog's namespace `namespace` among `tables` that this
    /// session's principal may see ([`Self::may_list`]).
    pub(crate) async fn listed_relations(
        &self,
        namespace: &str,
        tables: Vec<Arc<crate::store::CatalogTable>>,
    ) -> Vec<Arc<crate::store::CatalogTable>> {
        let mut out = Vec::with_capacity(tables.len());
        for t in tables {
            let ident = TableIdent {
                catalog: crate::catalog::CATALOG.to_owned(),
                namespace: vec![namespace.to_owned()],
                name: t.name.clone(),
            };
            let securable = if t.view.is_some() {
                Securable::View(ident)
            } else {
                Securable::Table(ident)
            };
            if self.may_list(&securable).await {
                out.push(t);
            }
        }
        out
    }

    /// Whether this session's principal may see the session catalog's namespace `namespace` in a
    /// listing: `Browse` on it, or on any relation in it (a table grant shows where the table is).
    pub(crate) async fn may_list_namespace(&self, namespace: &str) -> bool {
        let securable = Securable::Namespace(vec![
            crate::catalog::CATALOG.to_owned(),
            namespace.to_owned(),
        ]);
        if self.may_list(&securable).await {
            return true;
        }
        let tables = crate::store::CatalogStore::tables_in(&*self.state.catalog.store, namespace);
        for t in tables {
            let ident = TableIdent {
                catalog: crate::catalog::CATALOG.to_owned(),
                namespace: vec![namespace.to_owned()],
                name: t.name.clone(),
            };
            if self.may_list(&Securable::Table(ident)).await {
                return true;
            }
        }
        false
    }

    /// The error `e` of a metadata analysis (P3 PR-14d review B1), raised after the permanent
    /// `views` it inlined were allowed at `Browse`: a failure inside a view's text names the
    /// text (`(a = 424242)`, a column name), and a view's text needs `Select`. So the error is
    /// shown only once `Select` on each of them passes; otherwise the first denial is (logged
    /// by the caller, through [`audit_read`]). An ordinary analysis asked `Select` already, and
    /// an access error is shown as it is.
    pub(crate) async fn view_text_failure(
        &self,
        views: &[TableIdent],
        env: &QueryEnv,
        e: KrishivError,
    ) -> KrishivError {
        if env.analysis.mode != krishiv_plan::analyzer::AnalysisMode::Metadata
            || e.error_class() == ErrorClass::KrishivInsufficientPermissions
        {
            return e;
        }
        for view in views {
            if let Err(denied) = authorize(
                &*self.state.policy,
                &self.state.principal,
                &Securable::View(view.clone()),
                Privilege::Select,
            )
            .await
            {
                return denied;
            }
        }
        e
    }

    /// `Select` on every permanent view a metadata analysis read at `Browse` (a withheld
    /// explain, design P3 PR-14d D2a): the plan shows a view's inlined body, and a view's text
    /// is data (ADR 0046, update 2026-11-12). A SQL function's body is shown as
    /// `DESCRIBE FUNCTION EXTENDED` shows it: metadata (ADR 0046, update 2026-11-15). Asked in
    /// plan order; a denial is audited.
    pub(crate) async fn authorize_bodies(
        &self,
        analysis: &crate::dataframe::Analysis,
    ) -> Result<(), KrishivError> {
        for used in &analysis.analyzed.securables {
            if !matches!(used.securable, Securable::View(_)) {
                continue;
            }
            let r = authorize(
                &*self.state.policy,
                &self.state.principal,
                &used.securable,
                Privilege::Select,
            )
            .await;
            audit(self, StatementClass::Read, r)?;
        }
        Ok(())
    }

    /// Checks that this session's principal holds `privilege` on `securable` under the
    /// session's policy, outside any statement: what a server checks for a request before it
    /// runs one (the Connect request classes of ADR 0046 §3b). Fails closed as every check
    /// does, and a denial writes one `krishiv::audit` record with `class`.
    ///
    /// # Errors
    /// `KRISHIV_INSUFFICIENT_PERMISSIONS` (42501) when the policy does not allow it.
    pub async fn authorize_privilege(
        &self,
        securable: &Securable,
        privilege: Privilege,
        class: StatementClass,
    ) -> Result<(), KrishivError> {
        let r = authorize(
            &*self.state.policy,
            &self.state.principal,
            securable,
            privilege,
        )
        .await;
        audit(self, class, r)
    }

    /// `privilege` on the permanent view `namespace.name`, besides the table-named check made
    /// before dispatch: the statements that show a view's definition, columns or properties
    /// need `Browse` on it (ADR 0046 §4), `SHOW TABLE EXTENDED` `Select`.
    pub(crate) async fn authorize_view(
        &self,
        namespace: &str,
        name: &str,
        privilege: Privilege,
        env: &QueryEnv,
    ) -> Result<(), KrishivError> {
        let parts = [
            crate::catalog::CATALOG.to_owned(),
            namespace.to_owned(),
            name.to_owned(),
        ];
        let ident = table_ident(&*env.analysis.catalogs, &parts)
            .ok_or_else(|| unplaceable("an empty name"))?;
        let r = authorize(
            &*self.state.policy,
            &self.state.principal,
            &Securable::View(ident),
            privilege,
        )
        .await;
        audit(self, StatementClass::Read, r)
    }

    /// The columns of the system table `table` for `DESCRIBE` / `SHOW COLUMNS`: `Browse` on
    /// `system.<table>` (ADR 0046 §3), not the `Select` a read of its rows needs.
    pub(crate) async fn system_table_columns(
        &self,
        table: &str,
        env: &QueryEnv,
    ) -> Result<Vec<(String, krishiv_types::DataType)>, KrishivError> {
        let parts = ["system".to_owned(), table.to_owned()];
        let ident = table_ident(&*env.analysis.catalogs, &parts)
            .ok_or_else(|| unplaceable("an empty name"))?;
        let r = authorize(
            &*self.state.policy,
            &self.state.principal,
            &Securable::Table(ident),
            Privilege::Browse,
        )
        .await;
        audit(self, StatementClass::Read, r)?;
        crate::system::columns(table)
            .map(|cols| cols.into_iter().map(|c| (c.name, c.data_type)).collect())
            .ok_or_else(|| {
                krishiv_common::error::InternalError {
                    message: format!("system.{table}: no such system table"),
                }
                .into()
            })
    }

    /// `parts` (a name written in a statement's target), with a bare temporary relation placed in the session
    /// catalog's current namespace: a temporary relation resolves before any catalog, `system` included.
    fn temp_first(&self, parts: Vec<String>, env: &QueryEnv) -> Vec<String> {
        match parts.as_slice() {
            [name]
                if crate::system::is_system(&self.state.catalog.current_catalog())
                    && self.is_temp_name(name, env) =>
            {
                vec![
                    crate::catalog::CATALOG.to_owned(),
                    self.state.catalog.current_namespace(),
                    name.clone(),
                ]
            }
            _ => parts,
        }
    }

    /// The securables `object` stands for: as written, then folded.
    pub(crate) fn securables_of(
        &self,
        object: TargetObject<'_>,
        env: &QueryEnv,
    ) -> Result<Vec<Securable>, KrishivError> {
        let catalogs = &*env.analysis.catalogs;
        let fold = !env.analysis.conf.case_sensitive;
        let parts = |name: &DdlName| ddl_parts(name, env);
        let table = |parts: &[String]| {
            let ident = table_ident(catalogs, parts).ok_or_else(|| unplaceable("an empty name"))?;
            // `global_temp.x` belongs to the session catalog's registry, whatever catalog is current.
            Ok::<TableIdent, KrishivError>(
                if parts.len() == 2 && parts[0].eq_ignore_ascii_case(GLOBAL_TEMP) {
                    TableIdent {
                        catalog: crate::catalog::CATALOG.to_owned(),
                        ..ident
                    }
                } else {
                    ident
                },
            )
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
            TargetObject::Table(name) => tables(table(&self.temp_first(parts(name)?, env))?, false),
            TargetObject::TableIn {
                table: t,
                namespace,
            } => {
                let mut p = parts(namespace)?;
                p.extend(parts(t)?.pop());
                tables(table(&p)?, false)
            }
            TargetObject::View(name) => tables(table(&self.temp_first(parts(name)?, env))?, true),
            TargetObject::GlobalTempView(name) => {
                let name = parts(name)?
                    .pop()
                    .ok_or_else(|| unplaceable("an empty name"))?;
                tables(
                    TableIdent {
                        catalog: crate::catalog::CATALOG.to_owned(),
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
                // The `global_temp` registry belongs to the session catalog, whatever catalog is current.
                namespaces(vec![
                    crate::catalog::CATALOG.to_owned(),
                    GLOBAL_TEMP.to_owned(),
                ])
            }
            TargetObject::Namespace(name) => {
                let written = parts(name)?;
                // `SHOW TABLES IN system` lists the system catalog (the introspection handler reads the
                // argument that way), so it is checked as the catalog, as `USE system; SHOW TABLES` is.
                if crate::system::is_system(written.last().map_or("", String::as_str))
                    && (written.len() == 1
                        || (written.len() == 2
                            && written[0].eq_ignore_ascii_case(crate::catalog::CATALOG)))
                {
                    return Ok(namespaces(vec![crate::system::SYSTEM.to_owned()]));
                }
                let mut p = namespace_parts(catalogs, &written);
                // The `global_temp` registry belongs to the session catalog, whatever catalog is current.
                if p.len() == 2 && p[1].eq_ignore_ascii_case(GLOBAL_TEMP) {
                    p[0] = crate::catalog::CATALOG.to_owned();
                }
                namespaces(p)
            }
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

/// A path as an access check sees it: `at`, resolved (links followed), is what the policy is asked
/// about, the audit record names and the statement uses; `shown`, the path as the caller wrote it with
/// `.` and `..` folded from the text only (`LocalPaths::locate`), is what a denial the caller sees names.
/// Nothing read from the file system reaches `shown`, so a denial cannot tell a link's target or whether
/// anything exists.
#[derive(Debug, Clone)]
pub(crate) struct Checked {
    /// The resolved path.
    pub(crate) at: LocatedPath,
    /// The caller's text, `..` folded, qualified (`file:/...`).
    pub(crate) shown: String,
}

/// `at` (a local path, `file:/...`, as `LocalPaths::locate` gives it: `.` and `..` already folded in the
/// text, as every read and write folds them) with the symbolic links of its existing part resolved: the
/// directory a write really goes to. An object-store path is returned as it is. `at.display` is the
/// [`Checked::shown`] text.
///
/// # Errors
/// `KRISHIV_INSUFFICIENT_PERMISSIONS` for `privilege` on the shown path (audited as `class`) when the
/// path cannot be resolved: an unsearchable directory, a dangling link or a link loop on it. That is the
/// answer a policy gives a path it denies, with the same text (no existence oracle).
pub(crate) fn physical(
    session: &Session,
    at: LocatedPath,
    privilege: Privilege,
    class: StatementClass,
) -> Result<Checked, KrishivError> {
    let shown = at.display.clone();
    let Some(local) = at.display.strip_prefix("file:") else {
        return Ok(Checked { at, shown });
    };
    let Some(parts) = canonical_parts(Path::new(local)) else {
        return Err(unresolved(session, class, &shown, privilege));
    };
    Ok(Checked {
        at: LocatedPath {
            store: at.store,
            path: object_store::path::Path::from_iter(parts.iter().map(String::as_str)),
            display: format!("file:/{}", parts.join("/")),
        },
        shown,
    })
}

/// The local directory `path` (absolute) in the one form every check of a table's directories asks:
/// `.` and `..` folded in the text (`LocalPaths::locate`, as reads and writes fold them), then
/// [`physical`]. The result's `at.display` is `file:` and the canonical path.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` (audited) for a relative or non-UTF-8 path: that depends on the text
/// only, never on the disk. `KRISHIV_INSUFFICIENT_PERMISSIONS` as [`physical`] gives it.
pub(crate) fn canonical_dir(
    session: &Session,
    path: &Path,
    privilege: Privilege,
    class: StatementClass,
) -> Result<Checked, KrishivError> {
    let lexical = lexical_dir(path).inspect_err(|e| log_denial(session, class, e))?;
    physical(session, lexical, privilege, class)
}

/// The absolute local path `path` with `.` and `..` folded in its text, as `LocalPaths::locate` folds a
/// read's or write's path; no file system access.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a relative or non-UTF-8 path.
pub(crate) fn lexical_dir(path: &Path) -> Result<LocatedPath, KrishivError> {
    let text = path
        .to_str()
        .ok_or_else(|| unplaceable("non-UTF-8 paths"))?;
    if !path.is_absolute() {
        return Err(unplaceable("relative paths"));
    }
    krishiv_exec::files::LocalPaths::new(StoreId::new(crate::session::LOCAL_STORE), "/")
        .locate(text)
}

/// Asks the policy each of `privileges` on the resolved path `checked.at`, auditing each answer with that
/// path (the record stays on the server), and returns a denial naming `checked.shown`, the caller's text.
pub(crate) async fn authorize_checked(
    session: &Session,
    checked: &Checked,
    privileges: &[Privilege],
    class: StatementClass,
) -> Result<(), KrishivError> {
    let asked = Securable::Path(OptionValue::from_user("path", checked.at.display.clone()));
    for privilege in privileges {
        let r = authorize(
            &*session.state.policy,
            &session.state.principal,
            &asked,
            *privilege,
        )
        .await;
        if let Err(e) = audit(session, class, r) {
            // Re-issued with the caller's text and the same class: `authorize` names the securable only
            // in a 42501 and in the `UNSUPPORTED_FEATURE` of a filtered `Select`; any other error names
            // no path and passes as it is.
            let shown = Securable::Path(OptionValue::from_user("path", checked.shown.clone()));
            return Err(match e.error_class() {
                ErrorClass::KrishivInsufficientPermissions => {
                    insufficient_permissions(&session.state.principal, &shown, *privilege)
                }
                ErrorClass::KrishivUnsupportedFeature => KrishivUnsupportedFeature {
                    feature: format!("row filters and column masks (on {shown})"),
                }
                .into(),
                _ => e,
            });
        }
    }
    Ok(())
}

/// The denial of `privilege` on the path `display` (the caller's text) that cannot be resolved: the same
/// 42501 a policy's denial gives, with the audit record a denial writes.
///
/// `privilege` is the first one the caller would ask: a policy denying a missing path denies that one
/// first too, unless it grants it and denies a later one, which a caller sees only on paths inside its
/// rights.
pub(crate) fn unresolved(
    session: &Session,
    class: StatementClass,
    display: &str,
    privilege: Privilege,
) -> KrishivError {
    let securable = Securable::Path(OptionValue::from_user("path", display.to_owned()));
    let e = insufficient_permissions(&session.state.principal, &securable, privilege);
    log_denial(session, class, &e);
    e
}

/// The components of the absolute local path `local` with every symbolic link resolved in the part of
/// it that exists, and the part that does not exist yet appended as written (see [`resolve_links`]).
/// `None` for a relative path, one that cannot be resolved (an I/O error, a link loop, a dangling link,
/// a `..` in its missing part) and a non-UTF-8 one. Callers fold `..` in the text first.
fn canonical_parts(local: &Path) -> Option<Vec<String>> {
    if !local.is_absolute() {
        return None;
    }
    resolve_links(local)?
        .components()
        .filter_map(|c| match c {
            Component::Normal(p) => Some(p.to_str().map(str::to_owned)),
            _ => None,
        })
        .collect()
}

/// Fails closed (`KRISHIV_INSUFFICIENT_PERMISSIONS`, naming `checked.shown`) when `checked.at`, a path
/// [`physical`] resolved and the policy allowed, now resolves elsewhere (a symbolic link was swapped
/// into it) or no longer resolves.
pub(crate) fn still_physical(session: &Session, checked: &Checked) -> Result<(), KrishivError> {
    let deny = || {
        unresolved(
            session,
            StatementClass::Write,
            &checked.shown,
            Privilege::Modify,
        )
    };
    let Some(local) = checked.at.display.strip_prefix("file:") else {
        return Ok(());
    };
    match canonical_parts(Path::new(local)) {
        Some(parts) if format!("file:/{}", parts.join("/")) == checked.at.display => Ok(()),
        _ => Err(deny()),
    }
}

/// Allows paths only under the given root directories, and delegates every
/// other decision to `inner`: the policy behind a server's data-directory
/// restriction.
///
/// A path is compared in the qualified form reads and writes use
/// (`file:/a/b`, `.` and `..` already resolved), after resolving symbolic
/// links in the part of it that exists, against the roots with their links
/// resolved. Object-store paths are allowed only under the prefixes of
/// [`Self::with_object_prefixes`] (none by default). Anything else is
/// denied: other schemes, paths with credentials, any path containing `%`
/// (percent-encoded text would not be decoded the way the comparison reads
/// it, so even a literal `%` in a file name is refused) and relative or
/// unresolved paths. A glob or directory read is checked as written; the
/// runtime's local store refuses a listing that follows a link out of
/// `krishiv.io.local.allowedRoots` (set it on the runtime too). Session
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
    prefixes: Vec<ObjectPrefix>,
    inner: Arc<dyn AccessPolicy>,
}

/// One allowed object-store prefix: a store and the whole key segments under it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ObjectPrefix {
    store: StoreId,
    segments: Vec<String>,
}

impl std::fmt::Debug for PathPrefixPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PathPrefixPolicy")
            .field("roots", &self.roots)
            .field("prefixes", &self.prefixes)
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
        Ok(Self {
            roots,
            prefixes: Vec::new(),
            inner,
        })
    }

    /// Also allows object-store paths under `prefixes` (`s3://bucket/prefix`, `gs://bucket`,
    /// `abfss://container@account.dfs.core.windows.net/prefix`): a path is allowed when it
    /// names the same store (`s3` and `s3a` are one; for Azure the same container in the same
    /// account) and its key starts with the prefix's whole segments, so `s3://b/data` allows
    /// `s3://b/data/x` but not `s3://b/data2` (ADR 0046 §3a). An Azure path without an
    /// account host is never allowed, since its account would come from the client's settings.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` for a prefix that is not an object-store URL with a
    /// bucket, an Azure prefix without an `account.<host>`, or one that holds credentials,
    /// `%`, a query or fragment, a glob character, or a `.`, `..` or empty segment.
    pub fn with_object_prefixes<S: AsRef<str>>(
        mut self,
        prefixes: impl IntoIterator<Item = S>,
    ) -> Result<Self, KrishivError> {
        for prefix in prefixes {
            let prefix = prefix.as_ref();
            let parsed = parse_object_prefix(prefix)
                .filter(|_| !prefix.contains(['*', '?', '[', ']', '{', '}']))
                // An Azure prefix names its account: without one, the store would take the
                // account from the client's `fs.azure.account.name`.
                .filter(|_| {
                    parse_object_url(prefix).is_ok_and(|u| {
                        u.scheme != krishiv_io::ObjectScheme::Azure || u.account.is_some()
                    })
                })
                .ok_or_else(|| {
                    KrishivError::from(KrishivInvalidConfigValue {
                        value: format!("'{prefix}'"),
                        key: format!("\"{}\"", IO_OBJECT_STORE_ALLOWED_PREFIXES.key),
                        reason:
                            "expected an object-store prefix such as s3://bucket/prefix (Azure: \
                             abfss://container@account.dfs.core.windows.net/prefix), without \
                             credentials, `%`, a query or fragment, glob characters, or `.`, `..` \
                             or empty segments"
                                .to_owned(),
                    })
                })?;
            self.prefixes.push(parsed);
        }
        Ok(self)
    }

    /// The server-mode confinement of ADR 0046 §3a around `inner`: local paths under
    /// `krishiv.io.local.allowedRoots`, object-store paths under
    /// `krishiv.io.objectStore.allowedPrefixes` (both comma separated, default empty, so
    /// nothing by path), and nothing else, for every role.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` for a root that is not an absolute path to an existing
    /// directory, or a prefix [`Self::with_object_prefixes`] refuses.
    pub fn from_config(
        config: &Config,
        inner: Arc<dyn AccessPolicy>,
    ) -> Result<Self, KrishivError> {
        Self {
            roots: allowed_roots(config)?,
            prefixes: Vec::new(),
            inner,
        }
        .with_object_prefixes(config_list(config, &IO_OBJECT_STORE_ALLOWED_PREFIXES))
    }

    /// Whether the qualified path `path` lies under a root or an object-store prefix.
    fn allows(&self, path: &OptionValue) -> bool {
        let text = match path {
            OptionValue::Plain(text) => text.as_str(),
            // `container@account.host` reads as user info, so an Azure path arrives typed as a
            // URL with credentials; Azure parsing takes it as container and account and refuses
            // `user:secret@` forms, so only Azure URLs are compared, and the text never leaves
            // this check.
            OptionValue::Url(url) => {
                let exposed = url.expose_secret();
                let text = exposed.expose_secret();
                return parse_object_url(text)
                    .is_ok_and(|u| u.scheme == krishiv_io::ObjectScheme::Azure)
                    && self.allows_object(text);
            }
            OptionValue::Secret(_) => return false,
        };
        let Some(local) = text.strip_prefix("file:") else {
            return self.allows_object(text);
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

impl PathPrefixPolicy {
    /// Whether the object-store URL `text` lies under one of the prefixes.
    fn allows_object(&self, text: &str) -> bool {
        parse_object_prefix(text).is_some_and(|p| {
            self.prefixes
                .iter()
                .any(|a| a.store == p.store && p.segments.starts_with(&a.segments))
        })
    }
}

/// The comma-separated entries of the string setting `entry` (blank entries dropped).
fn config_list(config: &Config, entry: &krishiv_common::config::ConfigEntry) -> Vec<String> {
    match config.get(entry) {
        Some(ConfigValue::String(raw)) => raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

/// `krishiv.io.local.allowedRoots`, canonical (links resolved); empty when unset. The policy
/// checks paths against them and the runtime's local store refuses listings that follow a
/// link out of them (ADR 0046 §3a).
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` naming the key for a root that is not an absolute path to
/// an existing directory.
pub(crate) fn allowed_roots(config: &Config) -> Result<Vec<PathBuf>, KrishivError> {
    config_list(config, &IO_LOCAL_ALLOWED_ROOTS)
        .into_iter()
        .map(|root| {
            let path = Path::new(&root);
            std::fs::canonicalize(path)
                .ok()
                .filter(|c| path.is_absolute() && c.is_dir())
                .ok_or_else(|| {
                    KrishivError::from(KrishivInvalidConfigValue {
                        value: format!("'{root}'"),
                        key: format!("\"{}\"", IO_LOCAL_ALLOWED_ROOTS.key),
                        reason: "expected absolute paths of existing directories, comma \
                                 separated"
                            .to_owned(),
                    })
                })
        })
        .collect()
}

/// `text` as an object-store store and key segments, if it is a plain object-store URL: a
/// bucket, no credentials, no `%` (percent-encoded text would not be compared the way the
/// store reads it), no query or fragment, and no `.`, `..` or empty key segment (the URL
/// parser would fold them, so the text written would not be the key compared). A trailing
/// `/` is allowed.
fn parse_object_prefix(text: &str) -> Option<ObjectPrefix> {
    if text.contains(['%', '?', '#']) {
        return None;
    }
    let (_, rest) = text.split_once("://")?;
    let key = rest.split_once('/').map_or("", |(_, key)| key);
    let key = key.strip_suffix('/').unwrap_or(key);
    if !key.is_empty() && key.split('/').any(|s| matches!(s, "" | "." | "..")) {
        return None;
    }
    let url = parse_object_url(text).ok()?;
    let segments = url
        .key
        .as_ref()
        .split('/')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    Some(ObjectPrefix {
        store: url.store_id,
        segments,
    })
}

/// `path` with the symbolic links of its longest existing prefix resolved
/// (the rest does not exist yet, so it holds no links).
pub(crate) fn resolve_links(path: &Path) -> Option<PathBuf> {
    let mut existing = path.to_path_buf();
    let mut rest = Vec::new();
    loop {
        match std::fs::canonicalize(&existing) {
            Ok(resolved) => return Some(rest.iter().rev().fold(resolved, |p, part| p.join(part))),
            // Only a name that is not there at all is kept as written (below a file it cannot be there
            // either; the write then fails as a storage error). A dangling link (its own entry exists), a
            // link loop or an unreadable directory fails closed.
            Err(e)
                if absent(&e)
                    && std::fs::symlink_metadata(&existing).is_err_and(|m| absent(&m)) => {}
            Err(_) => return None,
        }
        // `None` for a missing part ending in `..`: it is not resolved against a directory that is not there.
        rest.push(existing.file_name()?.to_owned());
        if !existing.pop() {
            return None;
        }
    }
}

/// Whether `e` says the entry is not there: missing, or below a file.
fn absent(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
    )
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

    fn column_level(&self) -> bool {
        self.inner.column_level()
    }

    fn denied(&self, principal: &Principal, securable: &Securable, privilege: Privilege) {
        self.inner.denied(principal, securable, privilege);
    }
}

/// The privileges a read-only surface allows: `Select` and `Browse` (metadata, which reads nothing a
/// `Select` would not). [`ReadOnlyPolicy`] denies every other one, and the MCP server's analyze check
/// refuses a statement that needs any other (design `P3-T063-analyze-api.md` §3.4): `Monitor` (other
/// principals' `system.*` rows) and `Usage` (on `system.udf`, sending code the server runs) stay out.
pub const READ_ONLY_PRIVILEGES: [Privilege; 2] = [Privilege::Select, Privilege::Browse];

/// A policy that allows `Select` as `inner` decides and denies every other
/// privilege: the read-only surfaces (the MCP server) enforce read-only
/// through the same [`authorize`] path as every other check, even if a
/// statement classifier missed a write.
///
/// ```
/// use std::sync::Arc;
/// use krishiv_session::{AllowAll, ReadOnlyPolicy, Session};
///
/// let s = Session::builder().access_policy(Arc::new(ReadOnlyPolicy::new(Arc::new(AllowAll)))).build()?;
/// let e = futures::executor::block_on(s.sql("CREATE TABLE t (a INT)")).unwrap_err();
/// assert_eq!(e.error_class().name(), "KRISHIV_INSUFFICIENT_PERMISSIONS");
/// # Ok::<(), krishiv_common::error::KrishivError>(())
/// ```
#[derive(Clone)]
pub struct ReadOnlyPolicy {
    inner: Arc<dyn AccessPolicy>,
}

impl std::fmt::Debug for ReadOnlyPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadOnlyPolicy").finish_non_exhaustive()
    }
}

impl ReadOnlyPolicy {
    /// Read-only over `inner`, which decides every `Select` and `Browse`.
    #[must_use]
    pub fn new(inner: Arc<dyn AccessPolicy>) -> Self {
        Self { inner }
    }
}

impl AccessPolicy for ReadOnlyPolicy {
    fn version(&self) -> PolicyVersion {
        self.inner.version()
    }

    fn check<'a>(
        &'a self,
        principal: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        // `Browse` (metadata) reads nothing a `Select` would not; the inner policy decides both.
        if !READ_ONLY_PRIVILEGES.contains(&privilege) {
            return Box::pin(std::future::ready(Ok(AccessDecision::Deny {
                reason: "the session is read-only".to_owned(),
            })));
        }
        self.inner.check(principal, securable, privilege)
    }

    fn column_level(&self) -> bool {
        self.inner.column_level()
    }

    fn denied(&self, principal: &Principal, securable: &Securable, privilege: Privilege) {
        self.inner.denied(principal, securable, privilege);
    }
}

/// `KRISHIV_RESERVED_NAME` for a statement that creates, replaces, drops or
/// writes something in the `system` catalog (P2-T126): its tables belong to
/// the engine, whoever the principal is.
fn reject_system_write(securable: &Securable, privilege: Privilege) -> Result<(), KrishivError> {
    if !matches!(
        privilege,
        Privilege::Create | Privilege::Drop | Privilege::Modify
    ) {
        return Ok(());
    }
    let (name, kind) = match securable {
        Securable::Table(t) | Securable::View(t) if crate::system::is_system(&t.catalog) => {
            (t.to_string(), "tables")
        }
        Securable::Namespace(parts)
            if parts.first().is_some_and(|c| crate::system::is_system(c)) =>
        {
            (parts.join("."), "namespaces")
        }
        _ => return Ok(()),
    };
    Err(krishiv_common::error::KrishivReservedName {
        name,
        kind: kind.to_owned(),
    }
    .into())
}

/// `CREATE|DROP DATABASE|SCHEMA|NAMESPACE system`: the name itself is
/// reserved, not only what would be created inside it.
fn reject_system_namespace(statement: &Statement) -> Result<(), KrishivError> {
    let name = match statement {
        Statement::CreateNamespace(n) => &n.name,
        Statement::DropNamespace(n) => &n.name,
        _ => return Ok(()),
    };
    match name {
        // `system` or `spark_catalog.system`: the same database, however
        // it is spelled.
        DdlName::Parts(o)
            if match o.parts.as_slice() {
                [first, second] => {
                    crate::system::is_system(&first.value)
                        || (first.value.eq_ignore_ascii_case("spark_catalog")
                            && crate::system::is_system(&second.value))
                }
                parts => parts
                    .first()
                    .is_some_and(|p| crate::system::is_system(&p.value)),
            } =>
        {
            Err(krishiv_common::error::KrishivReservedName {
                name: o
                    .parts
                    .iter()
                    .map(|p| p.value.as_str())
                    .collect::<Vec<_>>()
                    .join("."),
                kind: "namespaces".to_owned(),
            }
            .into())
        }
        _ => Ok(()),
    }
}

/// Table options are stored as text and never resolved, so a `secret('scope', 'key')` value would
/// be kept as that literal text; it is refused as it is for `writeTo` (use a read or write option). The
/// options are returned when none is a secret.
pub(crate) fn refuse_secret_options<'a>(
    options: &'a [TableOption],
    original: &str,
) -> Result<&'a [TableOption], KrishivError> {
    let is_secret = |o: &TableOption| {
        o.value.as_ref().is_some_and(|v| {
            original
                .get(v.span.start..v.span.end)
                .is_some_and(calls_secret)
        })
    };
    if options.iter().any(is_secret) {
        return Err(KrishivUnsupportedFeature {
            feature: "secret table options".to_owned(),
        }
        .into());
    }
    Ok(options)
}

/// Whether expression text `t` calls `secret` anywhere: the name bare, back-quoted or double-quoted, in any case
/// (`IDENTIFIER('secret')` was already replaced by a back-quoted name, `Session::sql`), followed by `(`, inside parentheses, casts or other calls too. It reads the
/// parser's own tokens, so string literals are skipped and comments between the name and `(` are trivia, as in the
/// parser; anything else that looks like a call counts (the check fails closed: a double-quoted `"secret"` counts
/// whether or not `spark.sql.ansi.doubleQuotedIdentifiers` makes it a name).
fn calls_secret(t: &str) -> bool {
    use krishiv_sql::lexer::{Sym, TokenKind, tokenize};
    let tokens: Vec<_> = tokenize(t)
        .into_iter()
        .filter(|k| !k.kind.is_trivia())
        .collect();
    let text = |i: usize| {
        t.get(tokens[i].span.start..tokens[i].span.end)
            .unwrap_or("")
    };
    // The text between the quotes of a quoted token, with doubled quotes read as one.
    let unquoted = |i: usize, quote: char| -> Option<String> {
        let raw = text(i);
        let inner = raw.strip_prefix(quote)?.strip_suffix(quote)?;
        Some(inner.replace(&format!("{quote}{quote}"), &quote.to_string()))
    };
    let names_secret = |i: usize| -> bool {
        let Some(k) = tokens.get(i) else { return false };
        match k.kind {
            TokenKind::Word => text(i).eq_ignore_ascii_case("secret"),
            TokenKind::QuotedIdent => {
                unquoted(i, '`').is_some_and(|n| n.eq_ignore_ascii_case("secret"))
            }
            TokenKind::DoubleQuoted => {
                unquoted(i, '"').is_some_and(|n| n.eq_ignore_ascii_case("secret"))
            }
            _ => false,
        }
    };
    (0..tokens.len()).any(|i| {
        names_secret(i)
            && tokens
                .get(i + 1)
                .is_some_and(|k| k.kind == TokenKind::Sym(Sym::LParen))
    })
}

#[cfg(test)]
mod canonical_path_tests {
    use std::path::Path;

    use super::{canonical_parts, lexical_dir};

    /// `path` as the directory checks see it: `..` folded in the text, then links resolved.
    fn canonical(path: &Path) -> Option<String> {
        let lexical = lexical_dir(path).ok()?;
        let local = lexical.display.strip_prefix("file:")?.to_owned();
        canonical_parts(Path::new(&local)).map(|p| format!("file:/{}", p.join("/")))
    }

    #[test]
    fn folds_dot_dot_in_the_text_then_resolves_links_and_appends_the_rest() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let real = root.join("real");
        std::fs::create_dir(&real).unwrap();
        std::fs::create_dir(root.join("x")).unwrap();
        std::fs::create_dir(root.join("elsewhere")).unwrap();
        std::os::unix::fs::symlink(&real, root.join("link")).unwrap();
        // `l/..` is `root`, not the parent of the link's target.
        std::os::unix::fs::symlink(root.join("elsewhere"), root.join("l")).unwrap();
        let expected = format!("file:{}/new/t", real.display());
        for written in [
            real.join("new/t"),
            root.join("link/new/t"),
            root.join("x/../link/./new/t"),
            root.join("l/../link/new/t"),
            root.join("missing/../link/new/t"),
        ] {
            assert_eq!(
                canonical(&written).as_deref(),
                Some(&*expected),
                "{written:?}"
            );
        }
    }

    #[test]
    fn refuses_paths_it_cannot_make_canonical() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::os::unix::fs::symlink(dir.join("nowhere"), dir.join("dangling")).unwrap();
        std::os::unix::fs::symlink(dir.join("loop"), dir.join("loop")).unwrap();
        for path in [Path::new("relative/wh"), Path::new("file:/wh")] {
            assert!(lexical_dir(path).is_err(), "{path:?} was not refused");
        }
        for path in [
            &dir.join("dangling/t"),
            &dir.join("dangling"),
            &dir.join("loop/t"),
        ] {
            assert_eq!(canonical(path), None, "{path:?} was not refused");
        }
    }
}
