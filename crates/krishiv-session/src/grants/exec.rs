// SPDX-License-Identifier: Apache-2.0
//! Running the access-control statements (`GRANT`, `REVOKE`, `SHOW GRANTS`, roles) and keeping
//! ownership as objects are created and dropped.
//!
//! Who may change a grant (note §2.2): an admin, an owner of the object (or a parent), or a holder
//! of the privilege there `WITH GRANT OPTION`; a grant never exceeds the grantor's own rights
//! (`ALL PRIVILEGES` needs the authority for each). `REVOKE` takes the same authority, then
//! removes only the caller's own grants unless it owns the object or is an admin (another
//! grantor's rows are left as if absent: no existence oracle); a table-level revoke takes the
//! grantee's column grants on the table with it. Role
//! changes other than `CREATE ROLE` need an admin or the role's creator. Every refusal is
//! `KRISHIV_INSUFFICIENT_PERMISSIONS` naming only what the statement wrote, and reads the same
//! whether the object or role exists or not.

use std::collections::BTreeSet;
use std::sync::Arc;

use krishiv_common::error::{
    KrishivError, KrishivRoleAlreadyExists, KrishivRoleCycle, KrishivRoleNotFound,
    KrishivUnsupportedFeature,
};
use krishiv_plan::analyzer::access_control::{
    AccessCommand, GRANTABLE, GrantObject, GrantPrivileges, Grantee,
};
use krishiv_plan::analyzer::{
    Privilege, Securable, TableIdent, authorize, insufficient_permissions, namespace_parts,
    table_ident,
};
use krishiv_sql::ast::Statement;

use super::model::{GrantRow, GrantState, KeyKind, ObjectKey};
use super::{GrantStateHandle, GrantsPolicy};
use crate::session::QueryEnv;
use krishiv_common::redact::RedactedSql;

use crate::{DataFrame, Session};

/// An object a statement creates (it gets an owner if the statement really created it), drops
/// (its grants and owner go) or renames (they move to the new name).
pub(crate) enum OwnerChange {
    Created(Securable),
    Dropped(Securable),
    Renamed { from: Securable, to: Securable },
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

fn unsupported_without_grants(command: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: format!(
            "{command} without the grants access provider (\"krishiv.access.provider\")"
        ),
    }
    .into()
}

/// The privileges a `GRANT`/`REVOKE` list names, each with its columns (empty: the object).
fn expand(privileges: &GrantPrivileges) -> Vec<(Privilege, Vec<String>)> {
    match privileges {
        GrantPrivileges::List(l) => l.iter().map(|g| (g.privilege, g.columns.clone())).collect(),
        _ => GRANTABLE.iter().map(|p| (*p, Vec::new())).collect(),
    }
}

impl Session {
    /// Runs the parsed, authorized `statement`; with grants on, a created object gets its
    /// creator as owner and a dropped one loses its grants (P3-T062).
    pub(crate) async fn run_statement(
        &self,
        statement: Statement,
        env: QueryEnv,
        query: &str,
        redacted: &RedactedSql,
    ) -> Result<DataFrame, KrishivError> {
        let Some(change) = self.owner_change(&statement, &env) else {
            return Box::pin(self.run_statement_here(statement, env, query, redacted)).await;
        };
        let check = env.clone();
        self.with_owner_change(change, &check, async move {
            Box::pin(self.run_statement_here(statement, env, query, redacted)).await
        })
        .await
    }

    /// Runs `run`, a statement that makes `change`, and keeps the ownership records in step
    /// (review B2, B3, m4): under the grants' DDL lock, the object's existence is read before and
    /// after, so only a statement that really created an object owns it (a no-op `IF NOT EXISTS`
    /// takes nothing), a drop clears what was under the name, and a rename moves it. While a
    /// new object is being created, its own keys (and a managed location the statement makes for
    /// it) are allowed to this session (review open point 2).
    pub(crate) async fn with_owner_change<T>(
        &self,
        change: OwnerChange,
        env: &QueryEnv,
        run: impl std::future::Future<Output = Result<T, KrishivError>>,
    ) -> Result<T, KrishivError> {
        let Some(policy) = self.grants_policy().cloned() else {
            return run.await;
        };
        let _ddl = policy.grants().ddl().await;
        let before = match &change {
            OwnerChange::Created(s) | OwnerChange::Renamed { to: s, .. } => {
                self.object_exists(s, env).await
            }
            OwnerChange::Dropped(_) => None,
        };
        // A new object (or a rename's new name, with the managed location it moves to) is the
        // statement's own while it runs.
        let creating = match (&change, before) {
            (OwnerChange::Created(s) | OwnerChange::Renamed { to: s, .. }, Some(false)) => {
                ObjectKey::of(s)
            }
            _ => None,
        };
        let _creating = creating.map(|k| policy.creating(k));
        let out = run.await?;
        self.apply_owner_change(&policy, &change, before, env)
            .await?;
        Ok(out)
    }

    /// Whether `securable` exists, as the catalog resolves its name; `None` when it cannot be
    /// told (then no owner is recorded: the object is the admins' until granted).
    pub(crate) async fn object_exists(
        &self,
        securable: &Securable,
        env: &QueryEnv,
    ) -> Option<bool> {
        let cs = env.analysis.conf.case_sensitive;
        match securable {
            Securable::Table(t) | Securable::View(t) => {
                // A permanent view of the session catalog: the query catalog never loads one
                // (views are inlined before analysis), so ask the store, or its creator would
                // never own it (P3-T062 G3).
                if let ([namespace], true) = (
                    t.namespace.as_slice(),
                    t.catalog.eq_ignore_ascii_case(crate::catalog::CATALOG),
                ) {
                    use crate::store::CatalogStore as _;
                    let store = &self.state.catalog.store;
                    let found = store.table(namespace, &t.name).or_else(|| {
                        (!cs)
                            .then(|| store.table(&namespace.to_lowercase(), &t.name.to_lowercase()))
                            .flatten()
                    });
                    if found.as_ref().is_some_and(|e| e.view.is_some()) {
                        return Some(true);
                    }
                }
                let catalog = env.analysis.catalogs.catalog(&t.catalog)?;
                let form = krishiv_plan::analyzer::NameForm {
                    case_sensitive: cs,
                    qualified: true,
                    temporary_only: false,
                };
                catalog
                    .load_table(t, None, form)
                    .await
                    .ok()
                    .map(|m| m.is_some())
            }
            Securable::Namespace(n) => match n.as_slice() {
                [catalog, namespace] if catalog == crate::catalog::CATALOG => {
                    use crate::store::CatalogStore as _;
                    let store = &self.state.catalog.store;
                    Some(
                        store.has_namespace(namespace)
                            || (!cs && store.has_namespace(&namespace.to_lowercase())),
                    )
                }
                _ => None,
            },
            // A stored function of the session catalog (P3-T062 G3: its creator owns it, so it
            // can call it, which needs `Select` on it).
            Securable::Function(n) => match n.as_slice() {
                [catalog, namespace, name] if catalog == crate::catalog::CATALOG => {
                    use crate::store::CatalogStore as _;
                    let namespace = if cs {
                        namespace.clone()
                    } else {
                        namespace.to_lowercase()
                    };
                    Some(
                        self.state
                            .catalog
                            .store
                            .functions()
                            .get(&namespace, name)
                            .is_some(),
                    )
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// The grants policy of this session, if it has one.
    fn grants_policy(&self) -> Option<&Arc<GrantsPolicy>> {
        self.state.grants.as_ref()
    }

    /// The securable `object` names, located as analysis locates a read, in the catalog's
    /// spelling (review M1, A2): an existing table or view as the catalog returns it (with its
    /// columns), anything else as the catalog's create path would spell it (folded to lower case
    /// unless `spark.sql.caseSensitive`).
    async fn grant_securable(
        &self,
        object: &GrantObject,
        env: &QueryEnv,
    ) -> Result<(Securable, Vec<String>), KrishivError> {
        let securable = self.written_securable(object, env)?;
        let cs = env.analysis.conf.case_sensitive;
        let fold_tail = |parts: &[String]| -> Vec<String> {
            parts
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    if i == 0 || cs {
                        p.clone()
                    } else {
                        p.to_lowercase()
                    }
                })
                .collect()
        };
        Ok(match securable {
            Securable::Table(t) | Securable::View(t) => {
                let view = matches!(object, GrantObject::View(_));
                let found = match env.analysis.catalogs.catalog(&t.catalog) {
                    Some(c) => {
                        let form = krishiv_plan::analyzer::NameForm {
                            case_sensitive: cs,
                            qualified: true,
                            temporary_only: false,
                        };
                        c.load_table(&t, None, form).await.ok().flatten()
                    }
                    None => None,
                };
                let (ident, columns) = match found {
                    Some(meta) => (
                        meta.ident.clone(),
                        meta.columns.iter().map(|c| c.name.clone()).collect(),
                    ),
                    None => (
                        TableIdent {
                            catalog: t.catalog.clone(),
                            namespace: fold_tail(
                                &std::iter::once(String::new())
                                    .chain(t.namespace.iter().cloned())
                                    .collect::<Vec<_>>(),
                            )
                            .into_iter()
                            .skip(1)
                            .collect(),
                            name: fold_tail(&[String::new(), t.name.clone()])
                                .pop()
                                .unwrap_or_default(),
                        },
                        Vec::new(),
                    ),
                };
                (
                    if view {
                        Securable::View(ident)
                    } else {
                        Securable::Table(ident)
                    },
                    columns,
                )
            }
            Securable::Namespace(n) => (Securable::Namespace(fold_tail(&n)), Vec::new()),
            Securable::Function(n) => (Securable::Function(fold_tail(&n)), Vec::new()),
            other => (other, Vec::new()),
        })
    }

    /// The securable `object` names as written, located as analysis locates a read.
    fn written_securable(
        &self,
        object: &GrantObject,
        env: &QueryEnv,
    ) -> Result<Securable, KrishivError> {
        let catalogs = &*env.analysis.catalogs;
        let empty = || -> KrishivError {
            KrishivUnsupportedFeature {
                feature: "an empty name in GRANT".to_owned(),
            }
            .into()
        };
        let table = |parts: &[String]| table_ident(catalogs, parts).ok_or_else(empty);
        Ok(match object {
            GrantObject::Catalog(c) => Securable::Namespace(vec![c.clone()]),
            GrantObject::Namespace(n) => Securable::Namespace(namespace_parts(catalogs, n)),
            GrantObject::Table(t) => Securable::Table(table(t)?),
            GrantObject::View(v) => Securable::View(table(v)?),
            GrantObject::Function(f) => {
                let TableIdent {
                    catalog,
                    namespace,
                    name,
                } = table(f)?;
                let mut full = vec![catalog];
                full.extend(namespace);
                full.push(name);
                Securable::Function(full)
            }
            GrantObject::Path(p) => Securable::Path(p.clone()),
            GrantObject::Connection(c) => Securable::Connection(c.clone()),
            GrantObject::SecretScope(s) => Securable::Secret { scope: s.clone() },
            _ => return Err(empty()),
        })
    }

    /// A refusal of `privilege` on `securable` (as the statement wrote it), audited and counted.
    fn refuse(
        &self,
        policy: &GrantsPolicy,
        securable: &Securable,
        privilege: Privilege,
    ) -> KrishivError {
        use krishiv_plan::analyzer::AccessPolicy as _;
        policy.denied(&self.state.principal, securable, privilege);
        insufficient_permissions(&self.state.principal, securable, privilege)
    }

    /// Runs an access-control statement (already authorized by its class).
    pub(crate) async fn exec_access_control(
        &self,
        a: &krishiv_sql::ast::AccessControl,
        env: &QueryEnv,
    ) -> Result<DataFrame, KrishivError> {
        let Some(policy) = self.grants_policy().cloned() else {
            return Err(unsupported_without_grants(a.command.name()));
        };
        let class = krishiv_sql::classify(&Statement::AccessControl(Box::new(a.clone())));
        let r = self.exec_access(&policy, &a.command, env).await;
        crate::access::audit(self, class, r)
    }

    async fn exec_access(
        &self,
        policy: &GrantsPolicy,
        command: &AccessCommand,
        env: &QueryEnv,
    ) -> Result<DataFrame, KrishivError> {
        let principal = Arc::clone(&self.state.principal);
        let me = principal.name.clone();
        let admin = policy.is_admin();
        let store = policy.grants().store();
        match command {
            AccessCommand::Grant {
                privileges,
                object,
                grantees,
                with_grant_option,
            } => {
                let (securable, columns) = self.grant_securable(object, env).await?;
                let rows = self.grant_rows(policy, &securable, &columns, privileges)?;
                let at = now_ms();
                store.update(&mut |s: &mut GrantStateHandle| {
                    // A role that does not exist would go to whoever creates it (review M3).
                    missing_role(&s.0, grantees, admin, || {
                        self.refuse(policy, &securable, Privilege::Manage)
                    })?;
                    for (key, privilege) in &rows {
                        for grantee in grantees {
                            // One row per grantor (review re-check 3 n1): another grantor's row
                            // of the same grant is not this caller's to change or revoke.
                            let row = s.0.grants.iter_mut().find(|g| {
                                g.grantee == *grantee
                                    && g.privilege == *privilege
                                    && g.object == *key
                                    && g.granted_by == me
                            });
                            match row {
                                Some(g) => g.with_grant_option |= *with_grant_option,
                                None => s.0.grants.push(GrantRow {
                                    grantee: grantee.clone(),
                                    privilege: *privilege,
                                    object: key.clone(),
                                    with_grant_option: *with_grant_option,
                                    granted_by: me.clone(),
                                    granted_at: at,
                                }),
                            }
                        }
                    }
                    Ok(())
                })?;
                self.command_done().await
            }
            AccessCommand::Revoke {
                grant_option_only,
                privileges,
                object,
                grantees,
            } => {
                let (securable, columns) = self.grant_securable(object, env).await?;
                let rows = self.grant_rows(policy, &securable, &columns, privileges)?;
                let cs = ObjectKey::of(&securable).is_none_or(|k| policy.exact_for(&k));
                store.update(&mut |s: &mut GrantStateHandle| {
                    missing_role(&s.0, grantees, admin, || {
                        self.refuse(policy, &securable, Privilege::Manage)
                    })?;
                    // A table-level revoke also takes the grantee's column grants of that
                    // privilege on the table (Unity Catalog's behaviour).
                    let matches = |g: &GrantRow| {
                        grantees.contains(&g.grantee)
                            && rows.iter().any(|(k, p)| {
                                g.privilege == *p
                                    && (g.object.same(k, cs) || g.object.is_column_of(k, cs))
                            })
                    };
                    // Only the grant's grantor, an owner of its object (or a parent) or an admin
                    // revokes it. Another grantor's row is left as if it did not exist, so the
                    // answer never tells whether it does (no existence oracle); a caller without
                    // any grant authority was refused above, before anything was looked up.
                    let mine: Vec<bool> =
                        s.0.grants
                            .iter()
                            .map(|g| {
                                matches(g)
                                    && (admin
                                        || g.granted_by == me
                                        || s.0.owns(&principal, &g.object, cs))
                            })
                            .collect();
                    if *grant_option_only {
                        s.0.grants
                            .iter_mut()
                            .zip(&mine)
                            .filter(|(_, m)| **m)
                            .for_each(|(g, _)| g.with_grant_option = false);
                    } else {
                        let mut mine = mine.iter();
                        s.0.grants
                            .retain(|_| !mine.next().copied().unwrap_or(false));
                    }
                    Ok(())
                })?;
                self.command_done().await
            }
            AccessCommand::ShowGrants { object, grantee } => {
                let key = match object {
                    Some(o) => {
                        let (securable, _) = self.grant_securable(o, env).await?;
                        Some(
                            ObjectKey::of(&securable)
                                .and_then(super::physical_key)
                                .ok_or_else(|| {
                                    self.refuse(policy, &securable, Privilege::Browse)
                                })?,
                        )
                    }
                    None => None,
                };
                let state = store.snapshot();
                let mine = state.0.effective(&principal);
                let system = Securable::System(crate::system::SYSTEM.to_owned());
                // Another principal's grants need `Monitor` (note §2.2), asked through the one
                // path; without a grantee, an object's grants are everyone's only with it (a
                // probe, not a refusal: no audit record when it says no).
                let who: Option<BTreeSet<Grantee>> = match grantee {
                    Some(g) if mine.contains(g) => Some([g.clone()].into()),
                    Some(g) => {
                        authorize(&*self.state.policy, &principal, &system, Privilege::Monitor)
                            .await?;
                        Some([g.clone()].into())
                    }
                    None if object.is_some() => {
                        let probe = self
                            .state
                            .policy
                            .check(&principal, &system, Privilege::Monitor)
                            .await;
                        match probe {
                            Ok(krishiv_plan::analyzer::AccessDecision::Allow) => None,
                            _ => Some(mine),
                        }
                    }
                    None => Some(mine),
                };
                let rows = show_rows(
                    &state.0,
                    key.as_ref(),
                    who.as_ref(),
                    key.as_ref().is_none_or(|k| policy.exact_for(k)),
                );
                self.ordered_frame(
                    crate::analyze::command_schema::SHOW_GRANTS,
                    &["grant_option"],
                    rows,
                )
                .await
            }
            AccessCommand::CreateRole { name } => {
                store.update(&mut |s: &mut GrantStateHandle| {
                    if s.0.roles.contains_key(name) {
                        return Err(KrishivRoleAlreadyExists { role: quote(name) }.into());
                    }
                    s.0.roles.insert(
                        name.clone(),
                        super::model::RoleEntry {
                            owner: me.clone(),
                            members: BTreeSet::new(),
                        },
                    );
                    Ok(())
                })?;
                self.command_done().await
            }
            AccessCommand::DropRole { name }
            | AccessCommand::GrantRole { role: name, .. }
            | AccessCommand::RevokeRole { role: name, .. } => {
                // The role as a securable only for the refusal's text: what the statement wrote.
                let refused = || {
                    self.refuse(
                        policy,
                        &Securable::System(format!("role {name}")),
                        Privilege::Manage,
                    )
                };
                store.update(&mut |s: &mut GrantStateHandle| {
                    let Some(entry) = s.0.roles.get(name) else {
                        // No existence oracle for a principal that could not change it anyway.
                        return Err(if admin {
                            KrishivRoleNotFound { role: quote(name) }.into()
                        } else {
                            refused()
                        });
                    };
                    if !admin && entry.owner != me {
                        return Err(refused());
                    }
                    match command {
                        AccessCommand::DropRole { .. } => {
                            s.0.roles.remove(name);
                            let gone = Grantee::Role(name.clone());
                            for r in s.0.roles.values_mut() {
                                r.members.remove(&gone);
                            }
                            s.0.grants.retain(|g| g.grantee != gone);
                        }
                        AccessCommand::GrantRole { grantees, .. } => {
                            for g in grantees {
                                if let Grantee::Role(member) = g
                                    && !s.0.roles.contains_key(member)
                                {
                                    return Err(KrishivRoleNotFound {
                                        role: quote(member),
                                    }
                                    .into());
                                }
                                if s.0.would_cycle(name, g) {
                                    return Err(KrishivRoleCycle {
                                        role: quote(name),
                                        grantee: g.to_string(),
                                    }
                                    .into());
                                }
                                if let Some(e) = s.0.roles.get_mut(name) {
                                    e.members.insert(g.clone());
                                }
                            }
                        }
                        AccessCommand::RevokeRole { grantees, .. } => {
                            if let Some(e) = s.0.roles.get_mut(name) {
                                for g in grantees {
                                    e.members.remove(g);
                                }
                            }
                        }
                        _ => {}
                    }
                    Ok(())
                })?;
                self.command_done().await
            }
            _ => Err(unsupported_without_grants(command.name())),
        }
    }

    /// The keys and privileges a `GRANT`/`REVOKE` of `privileges` on `securable` touches, after
    /// checking the caller's authority for each (refused as a whole otherwise).
    fn grant_rows(
        &self,
        policy: &GrantsPolicy,
        securable: &Securable,
        known_columns: &[String],
        privileges: &GrantPrivileges,
    ) -> Result<Vec<(ObjectKey, Privilege)>, KrishivError> {
        let principal = &self.state.principal;
        // A system object is the role matrix's, never granted (ADR 0046 §7).
        let refuse_all = || self.refuse(policy, securable, Privilege::Manage);
        if GrantsPolicy::decided_by(securable) != super::DecidedBy::Grants {
            return Err(refuse_all());
        }
        let Some(key) = ObjectKey::of(securable).and_then(super::physical_key) else {
            return Err(refuse_all());
        };
        let cs = policy.exact_for(&key);
        let state = policy.grants().store().snapshot();
        let mut out = Vec::new();
        for (privilege, columns) in expand(privileges) {
            let keys: Vec<ObjectKey> = if columns.is_empty() {
                vec![key.clone()]
            } else {
                columns
                    .iter()
                    .map(|c| {
                        // The column in the table's spelling when the table has it.
                        let spelled = known_columns
                            .iter()
                            .find(|k| {
                                if cs {
                                    *k == c
                                } else {
                                    k.eq_ignore_ascii_case(c)
                                }
                            })
                            .unwrap_or(c);
                        let mut parts = key.parts.clone();
                        parts.push(spelled.clone());
                        ObjectKey::new(KeyKind::Column, parts)
                    })
                    .collect()
            };
            for k in keys {
                if !policy.is_admin() && !state.0.may_grant(principal, &k, privilege, cs) {
                    return Err(self.refuse(policy, securable, privilege));
                }
                out.push((k, privilege));
            }
        }
        Ok(out)
    }

    /// The empty result of a command that changes state.
    async fn command_done(&self) -> Result<DataFrame, KrishivError> {
        self.ordered_frame(&[], &[], Vec::new()).await
    }

    /// The ownership change `statement` makes when it succeeds, if grants are on.
    pub(crate) fn owner_change(
        &self,
        statement: &Statement,
        env: &QueryEnv,
    ) -> Option<OwnerChange> {
        use krishiv_sql::TargetObject as O;
        self.grants_policy()?;
        let (object, created) = match statement {
            Statement::CreateTable(c) if !c.temporary => (O::Table(&c.name), true),
            Statement::CreateTableLike(c) => (O::Table(&c.name), true),
            Statement::CreatePipelineDataset(c) => (O::Table(&c.table.name), true),
            Statement::CreatePermanentView(v) if !v.temporary && !v.global => {
                (O::View(&v.name), true)
            }
            Statement::CreateMetricView(v) => (O::View(&v.name), true),
            Statement::CreateNamespace(n) => (O::Namespace(&n.name), true),
            Statement::CreateFunction(f) if !f.temporary => (O::Function(&f.name), true),
            Statement::DropTable(d) => (O::Table(&d.name), false),
            Statement::DropPermanentView(d) => (O::View(&d.name), false),
            Statement::DropNamespace(d) => (O::Namespace(&d.name), false),
            Statement::DropFunction(f) if !f.temporary => (O::Function(&f.name), false),
            Statement::AlterTable(a) => match &a.action {
                krishiv_sql::ast::AlterTableAction::RenameTo { name, .. } => {
                    return self.renamed(O::Table(&a.table), name, false, env);
                }
                _ => return None,
            },
            Statement::AlterView(a) => match &a.action {
                krishiv_sql::ast::AlterViewAction::RenameTo { name, .. } => {
                    return self.renamed(O::View(&a.name), name, true, env);
                }
                _ => return None,
            },
            _ => return None,
        };
        let securable = self.securables_of(object, env).ok()?.into_iter().next()?;
        Some(if created {
            OwnerChange::Created(securable)
        } else {
            OwnerChange::Dropped(securable)
        })
    }

    /// Allows the managed location `location` of the table `namespace.name` of the session
    /// catalog to the running statement when it is creating that table and the location does
    /// not exist yet (review open point 2: `CREATE` on the namespace covers the directory the
    /// engine chooses; an explicit `LOCATION` stays a path that needs a grant). Confinement still
    /// applies: it wraps the grants.
    pub(crate) fn allow_managed_location(
        &self,
        namespace: &str,
        name: &str,
        location: &std::path::Path,
    ) {
        let Some(policy) = self.grants_policy() else {
            return;
        };
        let relation = ObjectKey::new(
            KeyKind::Relation,
            vec![
                crate::catalog::CATALOG.to_owned(),
                namespace.to_owned(),
                name.to_owned(),
            ],
        );
        if location.exists() || !location.is_absolute() || !policy.is_creating(&relation) {
            return;
        }
        let text = format!("file:{}", location.display());
        let path = Securable::Path(krishiv_common::secret::OptionValue::from_user("path", text));
        if let Some(key) = ObjectKey::of(&path).and_then(super::physical_key) {
            policy.also_creating(key);
        }
    }

    /// The creation of the table a DataFrame writer names (`saveAsTable`, `writeTo(..).create()`),
    /// when grants are on.
    pub(crate) fn created_table(&self, table: &str, env: &QueryEnv) -> Option<OwnerChange> {
        self.grants_policy()?;
        let parts: Vec<String> = krishiv_sql::parse_multipart_identifier(table, &env.parser)
            .ok()?
            .parts
            .into_iter()
            .map(|p| p.value)
            .collect();
        let ident = table_ident(&*env.analysis.catalogs, &parts)?;
        Some(OwnerChange::Created(Securable::Table(ident)))
    }

    /// The rename of `object` to `to` (an unqualified new name stays in the object's namespace,
    /// as Spark's session catalog renames).
    fn renamed(
        &self,
        object: krishiv_sql::TargetObject<'_>,
        to: &krishiv_sql::ast::DdlName,
        view: bool,
        env: &QueryEnv,
    ) -> Option<OwnerChange> {
        let from = self.securables_of(object, env).ok()?.into_iter().next()?;
        let (Securable::Table(old) | Securable::View(old)) = &from else {
            return None;
        };
        let parts = crate::access::ddl_parts(to, env).ok()?;
        let new = match parts.as_slice() {
            [name] => TableIdent {
                name: name.clone(),
                ..old.clone()
            },
            _ => table_ident(&*env.analysis.catalogs, &parts)?,
        };
        let to = if view {
            Securable::View(new)
        } else {
            Securable::Table(new)
        };
        Some(OwnerChange::Renamed { from, to })
    }

    /// Records `change` after its statement succeeded (under the DDL lock, see
    /// [`Self::with_owner_change`]): the creator owns an object that did not exist `before` and
    /// does now, and anything left under its name (an object dropped out of band) goes; a
    /// dropped object's owner and every grant on it or under it go, so a later object of the same
    /// name starts with none; a rename moves them, clearing what was under the new name. The
    /// change and the epoch bump are one store update.
    async fn apply_owner_change(
        &self,
        policy: &GrantsPolicy,
        change: &OwnerChange,
        before: Option<bool>,
        env: &QueryEnv,
    ) -> Result<(), KrishivError> {
        let me = self.state.principal.name.clone();
        let key = |s: &Securable| ObjectKey::of(s);
        let cs = match change {
            OwnerChange::Created(s)
            | OwnerChange::Dropped(s)
            | OwnerChange::Renamed { to: s, .. } => key(s).is_none_or(|k| policy.exact_for(&k)),
        };
        let edit: Box<dyn Fn(&mut GrantState) + Send> = match change {
            OwnerChange::Created(s) => {
                let after = self.object_exists(s, env).await;
                let (Some(false), Some(true), Some(k)) = (before, after, key(s)) else {
                    return Ok(());
                };
                Box::new(move |state| {
                    state.owners.retain(|o, _| !o.covered_by(&k, cs));
                    state.grants.retain(|g| !g.object.covered_by(&k, cs));
                    state.owners.insert(k.clone(), me.clone());
                })
            }
            OwnerChange::Dropped(s) => {
                if self.object_exists(s, env).await == Some(true) {
                    return Ok(());
                }
                let Some(k) = key(s) else {
                    return Ok(());
                };
                Box::new(move |state| {
                    state.owners.retain(|o, _| !o.covered_by(&k, cs));
                    state.grants.retain(|g| !g.object.covered_by(&k, cs));
                })
            }
            OwnerChange::Renamed { from, to } => {
                let (Some(from_key), Some(to_key)) = (key(from), key(to)) else {
                    return Ok(());
                };
                // A rename that did not happen moves nothing; one that cannot be told clears
                // the old name (its records never follow a name to another object).
                let moved = self.object_exists(to, env).await == Some(true)
                    && self.object_exists(from, env).await != Some(true);
                let still_there = self.object_exists(from, env).await == Some(true);
                if still_there {
                    return Ok(());
                }
                Box::new(move |state| {
                    state.owners.retain(|o, _| !o.covered_by(&to_key, cs));
                    state.grants.retain(|g| !g.object.covered_by(&to_key, cs));
                    let owners = std::mem::take(&mut state.owners);
                    for (o, owner) in owners {
                        match o.renamed(&from_key, &to_key, cs) {
                            Some(n) if moved => {
                                state.owners.insert(n, owner);
                            }
                            Some(_) => {}
                            None => {
                                state.owners.insert(o, owner);
                            }
                        }
                    }
                    let grants = std::mem::take(&mut state.grants);
                    for mut g in grants {
                        match g.object.renamed(&from_key, &to_key, cs) {
                            Some(n) if moved => {
                                g.object = n;
                                state.grants.push(g);
                            }
                            Some(_) => {}
                            None => state.grants.push(g),
                        }
                    }
                })
            }
        };
        policy
            .grants()
            .store()
            .update(&mut |state: &mut GrantStateHandle| {
                edit(&mut state.0);
                Ok(())
            })?;
        Ok(())
    }
}

/// `SHOW GRANTS` rows: owners as an implicit `OWNERSHIP` grant, then grant rows, of `who` (every
/// grantee when `None`) on `key` and what it covers (every key when `None`).
fn show_rows(
    state: &GrantState,
    key: Option<&ObjectKey>,
    who: Option<&BTreeSet<Grantee>>,
    cs: bool,
) -> Vec<Vec<Option<String>>> {
    let on = |k: &ObjectKey| key.is_none_or(|key| key.covered_by(k, cs) || k.covered_by(key, cs));
    let mut rows = Vec::new();
    for (k, owner) in &state.owners {
        let grantee = Grantee::User(owner.clone());
        if on(k) && who.is_none_or(|w| w.contains(&grantee)) {
            rows.push(vec![
                Some(grantee.to_string()),
                Some("OWNERSHIP".to_owned()),
                Some(k.kind.name().to_owned()),
                Some(k.display()),
                Some("true".to_owned()),
                None,
                None,
            ]);
        }
    }
    let mut grants: Vec<&GrantRow> = state
        .grants
        .iter()
        .filter(|g| on(&g.object) && who.is_none_or(|w| w.contains(&g.grantee)))
        .collect();
    grants.sort_by(|a, b| {
        (&a.object, &a.grantee, a.privilege).cmp(&(&b.object, &b.grantee, b.privilege))
    });
    for g in grants {
        rows.push(vec![
            Some(g.grantee.to_string()),
            Some(g.privilege.to_string()),
            Some(g.object.kind.name().to_owned()),
            Some(g.object.display()),
            Some(g.with_grant_option.to_string()),
            Some(g.granted_by.clone()),
            Some(g.granted_at.to_string()),
        ]);
    }
    rows
}

/// Refuses a grant to or revoke from a `role:` that does not exist (review M3): an admin learns
/// it is missing (`KRISHIV_ROLE_NOT_FOUND`), anyone else gets `refused` (no existence oracle).
fn missing_role(
    state: &GrantState,
    grantees: &[Grantee],
    admin: bool,
    refused: impl Fn() -> KrishivError,
) -> Result<(), KrishivError> {
    for g in grantees {
        if let Grantee::Role(role) = g
            && !state.roles.contains_key(role)
        {
            return Err(if admin {
                KrishivRoleNotFound { role: quote(role) }.into()
            } else {
                refused()
            });
        }
    }
    Ok(())
}

fn quote(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}
