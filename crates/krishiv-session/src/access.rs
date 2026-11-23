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

use std::path::{Component, Path};
use std::sync::Arc;

use krishiv_common::error::{ErrorClass, KrishivError, KrishivUnsupportedFeature};
use krishiv_common::secret::OptionValue;
use krishiv_exec::files::{LocatedPath, PathLocator};
use krishiv_io::StoreId;
use krishiv_plan::SourceId;
use krishiv_plan::analyzer::{
    AccessDecision, AccessPolicy, BoxFuture, PolicyVersion, Principal, Privilege, Securable,
    SecurableUse, TableIdent, authorize, authorize_quietly, authorize_reported,
    insufficient_permissions, namespace_parts, table_ident,
};
use krishiv_sql::ast::{DdlName, ExprKind, Literal, Statement, TableOption};
use krishiv_sql::{
    ExplainKind, ParserOptions, StatementClass, Target, TargetObject, parse_multipart_identifier,
    statement_targets,
};

mod path_prefix;

pub use path_prefix::PathPrefixPolicy;
pub(crate) use path_prefix::{allowed_roots, resolve_links};

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
        self.parts_of_statement(statement, sql, &redacted).await
    }

    /// [`Self::statement_parts`] of the parsed `statement`, whose spans index `text` (redacted:
    /// `redacted`; `EXPLAIN FORMAT JSON` analyzes its explained statement here).
    pub(crate) async fn parts_of_statement(
        &self,
        statement: Statement,
        text: &str,
        redacted: &krishiv_common::redact::RedactedSql,
    ) -> Result<StatementParts, KrishivError> {
        let env = self.query_env();
        let class = self.statement_class(&statement, &env);
        if let Statement::ExecuteImmediate(e) = &statement {
            return Box::pin(self.immediate_parts(e, class, statement.clone(), text)).await;
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
            text: text.to_owned(),
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
        // A bare `DROP VIEW` of a stored view is checked as the qualified one when it runs.
        if let Statement::DropView(v) = statement
            && let Some(stored) = self.stored_drop_view(v, env)
            && self.stored_namespace_in(env).is_ok()
        {
            return self.add_targets(securables, &stored, env, skip_unplaced);
        }
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
        text: &str,
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
                    text: text.to_owned(),
                });
            }
        };
        let inner_text = krishiv_sql::immediate_text(e).unwrap_or_default();
        let inner_redacted = krishiv_sql::redact_sql(inner_text);
        let mut parts = self
            .parts_of_statement(inner, inner_text, &inner_redacted)
            .await?;
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
    /// The text the statement's spans index, as the run reads it (not redacted: never shown; the
    /// directories of a `CREATE TABLE` read its option values from it, `analyze/dirs.rs`).
    pub(crate) text: String,
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
    if env.quiet_denials() {
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
        self.check_targets(statement, env, Some(class)).await
    }

    /// Whether [`Self::authorize_statement`] of `statement` would pass, asked as it asks (the same
    /// questions in the same order, stopping at the first denial, the same decisions) but quietly: no
    /// `krishiv::audit` record, no session denial count, and the policy's [`AccessPolicy::denied`] is
    /// not called (`authorize_quietly`, review R2-1), so a provider counts and audits nothing either.
    /// The analyze API asks it before it reads any catalog or file-system state about a write target
    /// (T063 N1b review B2), so analysis reveals nothing about a target the caller may not write.
    pub(crate) async fn targets_allowed_quietly(
        &self,
        statement: &Statement,
        env: &QueryEnv,
    ) -> bool {
        self.check_targets(statement, env, None).await.is_ok()
    }

    /// Whether `privilege` on `object` would be allowed, asked as [`Self::targets_allowed_quietly`]
    /// asks (through `authorize_quietly`: `authorize`'s decision, with its `Browse`-to-`Select`
    /// fallback), quietly: no audit record, no denial count, no [`AccessPolicy::denied`] call. An
    /// object that cannot be placed is not allowed.
    pub(crate) async fn allowed_quietly(
        &self,
        object: TargetObject<'_>,
        privilege: Privilege,
        env: &QueryEnv,
    ) -> bool {
        let Ok(securables) = self.securables_of(object, env) else {
            return false;
        };
        for securable in &securables {
            let r = authorize_quietly(
                &*self.state.policy,
                &self.state.principal,
                securable,
                privilege,
            )
            .await;
            if r.is_err() {
                return false;
            }
        }
        true
    }

    /// The checks of [`Self::authorize_statement`]; each denial is logged under `class`, or not at all
    /// when `class` is `None` (a quiet trial, asked through `authorize_quietly`, so the policy is not
    /// told of it either).
    async fn check_targets(
        &self,
        statement: &Statement,
        env: &QueryEnv,
        class: Option<StatementClass>,
    ) -> Result<(), KrishivError> {
        let denied = |e: KrishivError| {
            if let Some(class) = class {
                log_denial(self, class, &e);
            }
            e
        };
        reject_system_namespace(statement).map_err(denied)?;
        let targets = required_targets(statement, &env.parser);
        let mut checked: Vec<(Securable, Privilege)> = Vec::new();
        for target in targets {
            let securables = self.securables_of(target.object, env).map_err(denied)?;
            for securable in securables {
                reject_system_write(&securable, target.privilege).map_err(denied)?;
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
                let r = match class {
                    Some(_) => {
                        authorize(
                            &*self.state.policy,
                            &self.state.principal,
                            &securable,
                            target.privilege,
                        )
                        .await
                    }
                    // A trial's refusal is an answer, not a refusal: the policy is not told.
                    None => {
                        authorize_quietly(
                            &*self.state.policy,
                            &self.state.principal,
                            &securable,
                            target.privilege,
                        )
                        .await
                    }
                };
                match class {
                    Some(class) => audit(self, class, r)?,
                    None => r?,
                }
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
        self.authorize_reads_as(reads, env, &principal, true).await
    }

    /// [`Self::authorize_reads`] for `principal`: the session's own, or the definer of a
    /// `SQL SECURITY DEFINER` view acting for it (P3-T062 §2.4). `audited`: a refusal is audited
    /// here (`false` when the caller audits it, as the definer's failure is).
    pub(crate) async fn authorize_reads_as(
        &self,
        reads: &[crate::catalog::ReadCandidate],
        env: &QueryEnv,
        principal: &Principal,
        audited: bool,
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
                    && env.analysis.policy.column_level()
                    && matches!(securable, Securable::Table(_))
                {
                    Privilege::Browse
                } else {
                    wanted
                };
                // The analysis' policy: the session's, quiet in a trial (`QuietPolicy`).
                let r = authorize_reported(
                    &*env.analysis.policy,
                    principal,
                    &securable,
                    privilege,
                    wanted,
                )
                .await;
                if !audited {
                    // A definer's refusal is audited once, by `Session::definer_failure`, which
                    // redacts it for the reader (one record per refusal, review M4).
                    r?;
                } else {
                    audit_read(self, env, r)?;
                }
                checked.push(securable);
            }
        }
        Ok(views)
    }

    /// The analysis' read privilege (`Select`; `Browse` for a metadata analysis) on every stored
    /// function the query's analysis resolved a call to (P3-T062 §2.4: calling a function needs
    /// `Select` on it; there is no `Execute`, Q3). Built-in, temporary and session functions are
    /// not securables. A call inside a definer view's text is asked of the reader too (fails
    /// closed). Each refusal is the caller's to audit. Asked through the analysis's policy, so a
    /// trial analysis's `QuietPolicy` keeps a refusal from the provider (T063 R2-1).
    pub(crate) async fn authorize_function_calls(
        &self,
        env: &QueryEnv,
    ) -> Result<(), KrishivError> {
        let privilege = env.analysis.mode.read_privilege();
        for (namespace, name) in env.sql_functions.principal_calls() {
            let securable =
                Securable::Function(vec![crate::catalog::CATALOG.to_owned(), namespace, name]);
            authorize(
                &*env.analysis.policy,
                &self.state.principal,
                &securable,
                privilege,
            )
            .await?;
        }
        Ok(())
    }

    /// Asks the session's principal what the `SQL SECURITY DEFINER` views of `analyzed` read
    /// (their `owner_reads`: the tables, and the views their texts inline, review M1), as an
    /// invoker would be asked, before their inlined queries are shown (`EXPLAIN`, estimates,
    /// warnings; P3-T062 §2.4: an explain checks the privileges of the statement it explains).
    /// Column-level policies are asked per column read. Audited whole; the reader is told only
    /// the refusal's class and SQLSTATE, as of a definer failure (review r2 item 6: the reader
    /// wrote none of these names).
    ///
    /// # Errors
    /// The first refusal (42501), redacted.
    pub(crate) async fn authorize_shown_definer_reads(
        &self,
        analyzed: &krishiv_plan::analyzer::AnalyzedPlan,
    ) -> Result<(), KrishivError> {
        for (securable, privilege) in self.shown_definer_reads(&analyzed.views) {
            let r = authorize(
                &*self.state.policy,
                &self.state.principal,
                &securable,
                privilege,
            )
            .await;
            audit(self, StatementClass::Read, r)
                .map_err(|e| e.redacted().without_pending_site())?;
        }
        Ok(())
    }

    /// Whether [`Self::authorize_shown_definer_reads`] would pass for the views `views`, asked
    /// quietly (no audit record): a query's profile shows its plan only then (review M2).
    pub(crate) async fn may_see_definer_reads(
        &self,
        views: &[krishiv_plan::analyzer::views::ViewExpansion],
    ) -> bool {
        for (securable, privilege) in self.shown_definer_reads(views) {
            if !krishiv_plan::analyzer::allowed(
                &*self.state.policy,
                &self.state.principal,
                &securable,
                privilege,
            )
            .await
            {
                return false;
            }
        }
        true
    }

    /// What showing the queries of the definer views among `views` asks the principal.
    fn shown_definer_reads(
        &self,
        views: &[krishiv_plan::analyzer::views::ViewExpansion],
    ) -> Vec<(Securable, Privilege)> {
        use krishiv_plan::analyzer::views::ViewSecurity;
        let column_level = self.state.policy.column_level();
        let mut out = Vec::new();
        for view in views.iter().filter(|v| v.security == ViewSecurity::Definer) {
            for used in &view.owner_reads {
                let per_column =
                    used.privilege == Privilege::Select && column_level && !used.columns.is_empty();
                match &used.securable {
                    Securable::Table(t) | Securable::View(t) if per_column => {
                        out.extend(used.columns.iter().map(|c| {
                            (
                                Securable::Column {
                                    table: t.clone(),
                                    column: c.clone(),
                                },
                                used.privilege,
                            )
                        }));
                    }
                    other => out.push((other.clone(), used.privilege)),
                }
            }
        }
        out
    }

    /// Whether this session's principal may see `securable` in a listing (`SHOW TABLES`,
    /// `SHOW VIEWS`, `SHOW NAMESPACES`, `spark.catalog.list*`; P3-T062 §2.4): under the `grants`
    /// provider when it may `Select` it or holds any grant on it (`GrantsPolicy::listable`; a
    /// `CREATE` on the namespace does not show every table in it), otherwise `Browse` on it.
    /// Asked quietly: leaving an entry out is not a refusal (no audit record, no denial counter).
    pub(crate) async fn may_list(&self, securable: &Securable) -> bool {
        if let Some(listed) = self.grants_listable(securable) {
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

    /// The grants provider's listing rule for `securable` (`GrantsPolicy::listable`): `None`
    /// without the provider or for a securable the `system` policy decides.
    fn grants_listable(&self, securable: &Securable) -> Option<bool> {
        self.state
            .grants
            .as_ref()
            .and_then(|g| g.listable(&self.state.principal, securable))
    }

    /// Whether a lookup of the one object `securable` by name answers as for a missing object
    /// (review B3; ADR 0046 §8: no existence oracle): the grants provider does not list it to this
    /// principal ([`Self::may_list`]'s rule, the same predicate). Other policies decide such a
    /// lookup by `Browse`, as they decide what a listing shows, and refuse it with 42501.
    pub(crate) fn hidden(&self, securable: &Securable) -> bool {
        self.grants_listable(securable) == Some(false)
    }

    /// Whether the grants provider hides the session catalog's namespace `namespace` from this
    /// principal, by the rule of [`Self::may_list_namespace`] (review B3).
    pub(crate) async fn namespace_hidden(&self, namespace: &str) -> bool {
        self.state.grants.is_some() && !self.may_list_namespace(namespace).await
    }

    /// The securable of the stored relation `t` of the session catalog: its view or its table.
    pub(crate) fn relation_securable(t: &crate::store::CatalogTable) -> Securable {
        let ident = TableIdent {
            catalog: crate::catalog::CATALOG.to_owned(),
            namespace: vec![t.namespace.clone()],
            name: t.name.clone(),
        };
        if t.view.is_some() {
            Securable::View(ident)
        } else {
            Securable::Table(ident)
        }
    }

    /// The relations of the session catalog among `tables` that this session's principal may
    /// see ([`Self::may_list`]).
    pub(crate) async fn listed_relations(
        &self,
        tables: Vec<Arc<crate::store::CatalogTable>>,
    ) -> Vec<Arc<crate::store::CatalogTable>> {
        let mut out = Vec::with_capacity(tables.len());
        for t in tables {
            if self.may_list(&Self::relation_securable(&t)).await {
                out.push(t);
            }
        }
        out
    }

    /// Whether this session's principal may see the session catalog's namespace `namespace` in a
    /// listing: `Browse` on it, or on any relation or function in it (a table grant shows where the
    /// table is).
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
        // Its functions count too (ADR 0048 A1 §3: "anything in it"; review r3 N4).
        let functions = crate::store::CatalogStore::functions(&*self.state.catalog.store)
            .in_namespace(namespace);
        for name in functions {
            let securable = Securable::Function(vec![
                crate::catalog::CATALOG.to_owned(),
                namespace.to_owned(),
                name,
            ]);
            if self.may_list(&securable).await {
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
                &*env.analysis.policy,
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
            // `SHOW NAMESPACES IN name`: a catalog when `name` is one, else a namespace of the
            // current catalog.
            TargetObject::NamedCatalog(name) => {
                let written = parts(name)?;
                match written.as_slice() {
                    [only] if catalogs.catalog(only).is_some() => namespaces(vec![only.clone()]),
                    _ => namespaces(namespace_parts(catalogs, &written)),
                }
            }
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

/// The policy of a trial analysis ([`crate::session::QueryEnv::set_quiet_denials`], P3 PR-14d):
/// decides exactly as `inner` does, but is never told of a refusal, so a trial's read denials,
/// which are answers and not refusals, leave no count or audit record in the provider
/// (T063 review R2-1). Never a session's policy: only a trial analysis' `AnalysisContext` holds it.
pub(crate) struct QuietPolicy {
    inner: Arc<dyn AccessPolicy>,
}

impl QuietPolicy {
    /// `inner`, quiet.
    pub(crate) fn new(inner: Arc<dyn AccessPolicy>) -> Self {
        Self { inner }
    }
}

impl AccessPolicy for QuietPolicy {
    fn version(&self) -> PolicyVersion {
        self.inner.version()
    }

    fn check<'a>(
        &'a self,
        principal: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        self.inner.check(principal, securable, privilege)
    }

    fn column_level(&self) -> bool {
        self.inner.column_level()
    }

    // `denied` keeps the default: told nothing.
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

    #[cfg(unix)]
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

    #[cfg(unix)]
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
