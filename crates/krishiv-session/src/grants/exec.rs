// SPDX-License-Identifier: Apache-2.0
//! Running the access-control statements (`GRANT`, `REVOKE`, `SHOW GRANTS`, roles) and keeping
//! ownership as objects are created and dropped.
//!
//! Who may change a grant (note §2.2): an admin, an owner of the object (or a parent), or a holder
//! of the privilege there `WITH GRANT OPTION`; a grant never exceeds the grantor's own rights
//! (`ALL PRIVILEGES` needs the authority for each). `REVOKE` takes the same authority. Role
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

/// An object a statement creates (it gets an owner) or drops (its grants and owner go).
pub(crate) enum OwnerChange {
    Created(Securable),
    Dropped(Securable),
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
        let owner_change = self.owner_change(&statement, &env);
        let df = Box::pin(self.run_statement_here(statement, env, query, redacted)).await?;
        if let Some(change) = &owner_change {
            self.apply_owner_change(change)?;
        }
        Ok(df)
    }

    /// The grants policy of this session, if it has one.
    fn grants_policy(&self) -> Option<&Arc<GrantsPolicy>> {
        self.state.grants.as_ref()
    }

    /// The securable `object` names, located as analysis locates a read.
    fn grant_securable(
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
                let securable = self.grant_securable(object, env)?;
                let rows = self.grant_rows(policy, &securable, privileges)?;
                let at = now_ms();
                store.update(&mut |s: &mut GrantStateHandle| {
                    for (key, privilege) in &rows {
                        for grantee in grantees {
                            let row = s.0.grants.iter_mut().find(|g| {
                                g.grantee == *grantee
                                    && g.privilege == *privilege
                                    && g.object == *key
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
                let securable = self.grant_securable(object, env)?;
                let rows = self.grant_rows(policy, &securable, privileges)?;
                store.update(&mut |s: &mut GrantStateHandle| {
                    let matches = |g: &GrantRow| {
                        grantees.contains(&g.grantee)
                            && rows
                                .iter()
                                .any(|(k, p)| g.object == *k && g.privilege == *p)
                    };
                    if *grant_option_only {
                        s.0.grants
                            .iter_mut()
                            .filter(|g| matches(g))
                            .for_each(|g| g.with_grant_option = false);
                    } else {
                        s.0.grants.retain(|g| !matches(g));
                    }
                    Ok(())
                })?;
                self.command_done().await
            }
            AccessCommand::ShowGrants { object, grantee } => {
                let key =
                    match object {
                        Some(o) => {
                            let securable = self.grant_securable(o, env)?;
                            Some(ObjectKey::of(&securable).ok_or_else(|| {
                                self.refuse(policy, &securable, Privilege::Browse)
                            })?)
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
                let rows = show_rows(&state.0, key.as_ref(), who.as_ref());
                self.ordered_frame(
                    &[
                        "principal",
                        "privilege",
                        "object_type",
                        "object_key",
                        "grant_option",
                        "granted_by",
                        "granted_at",
                    ],
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
        privileges: &GrantPrivileges,
    ) -> Result<Vec<(ObjectKey, Privilege)>, KrishivError> {
        let principal = &self.state.principal;
        // A system object is the role matrix's, never granted (ADR 0046 §7).
        let refuse_all = || self.refuse(policy, securable, Privilege::Manage);
        if GrantsPolicy::decided_by(securable) != super::DecidedBy::Grants {
            return Err(refuse_all());
        }
        let Some(key) = ObjectKey::of(securable) else {
            return Err(refuse_all());
        };
        let state = policy.grants().store().snapshot();
        let mut out = Vec::new();
        for (privilege, columns) in expand(privileges) {
            let keys: Vec<ObjectKey> = if columns.is_empty() {
                vec![key.clone()]
            } else {
                columns
                    .iter()
                    .map(|c| {
                        let mut parts = key.parts.clone();
                        parts.push(c.to_lowercase());
                        ObjectKey::new(KeyKind::Column, parts)
                    })
                    .collect()
            };
            for k in keys {
                if !policy.is_admin() && !state.0.may_grant(principal, &k, privilege) {
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
            Statement::CreateNamespace(n) => (O::Namespace(&n.name), true),
            Statement::CreateFunction(f) if !f.temporary => (O::Function(&f.name), true),
            Statement::DropTable(d) => (O::Table(&d.name), false),
            Statement::DropPermanentView(d) => (O::View(&d.name), false),
            Statement::DropNamespace(d) => (O::Namespace(&d.name), false),
            Statement::DropFunction(f) if !f.temporary => (O::Function(&f.name), false),
            _ => return None,
        };
        let securable = self.securables_of(object, env).ok()?.into_iter().next()?;
        Some(if created {
            OwnerChange::Created(securable)
        } else {
            OwnerChange::Dropped(securable)
        })
    }

    /// Records `change` after its statement succeeded: the creator owns a new object (an object
    /// that already had an owner keeps it, so `IF NOT EXISTS` takes nothing); a dropped object's
    /// owner and every grant on it or under it go, so a later object of the same name starts
    /// with none.
    pub(crate) fn apply_owner_change(&self, change: &OwnerChange) -> Result<(), KrishivError> {
        let Some(policy) = self.grants_policy() else {
            return Ok(());
        };
        let me = self.state.principal.name.clone();
        let (OwnerChange::Created(s) | OwnerChange::Dropped(s)) = change;
        let Some(key) = ObjectKey::of(s) else {
            return Ok(());
        };
        policy
            .grants()
            .store()
            .update(&mut |state: &mut GrantStateHandle| {
                match change {
                    OwnerChange::Created(_) => {
                        state
                            .0
                            .owners
                            .entry(key.clone())
                            .or_insert_with(|| me.clone());
                    }
                    OwnerChange::Dropped(_) => {
                        state.0.owners.retain(|k, _| !k.covered_by(&key));
                        state.0.grants.retain(|g| !g.object.covered_by(&key));
                    }
                }
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
) -> Vec<Vec<Option<String>>> {
    let on = |k: &ObjectKey| key.is_none_or(|key| key.covered_by(k) || k.covered_by(key));
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

fn quote(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}
