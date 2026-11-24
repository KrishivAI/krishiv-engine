// SPDX-License-Identifier: Apache-2.0
//! The one authorization path every privilege check takes: reads in
//! analysis, writes and DDL in the session's command execution.

use krishiv_common::error::{
    KrishivError, KrishivInsufficientPermissions, KrishivUnsupportedFeature,
};
use krishiv_common::secret::{OptionValue, SecretValue};

use super::catalog::{
    AccessDecision, AccessPolicy, CatalogResolver, Principal, Privilege, Securable, TableIdent,
    quoted,
};

/// `KRISHIV_INSUFFICIENT_PERMISSIONS` (42501) for `principal` lacking
/// `privilege` on `securable`. It never carries the policy's reason, and it
/// reads the same whether the securable exists or not.
#[must_use]
pub fn insufficient_permissions(
    principal: &Principal,
    securable: &Securable,
    privilege: Privilege,
) -> KrishivError {
    KrishivInsufficientPermissions {
        principal: quoted(std::slice::from_ref(&principal.name)),
        privilege: privilege.to_string(),
        securable: securable.to_string(),
    }
    .into()
}

/// Asks `policy` whether `principal` has `privilege` on `securable`, failing
/// closed: a denial or any policy error is
/// [`insufficient_permissions`] (the policy's reason and error text are
/// not shown).
///
/// A row filter or column mask ([`AccessDecision::AllowWith`]) on a read
/// is `KRISHIV_UNSUPPORTED_FEATURE` until it can be applied. A principal
/// whose reads of the securable are filtered or masked is denied every
/// privilege that [changes](Privilege::changes) it, whatever the policy
/// answers for that privilege: a filtered reader never writes, replaces or
/// drops data it cannot see whole.
///
/// # Errors
/// As above.
pub async fn authorize(
    policy: &dyn AccessPolicy,
    principal: &Principal,
    securable: &Securable,
    privilege: Privilege,
) -> Result<(), KrishivError> {
    authorize_reported(policy, principal, securable, privilege, privilege).await
}

/// [`authorize`] asking `asked` but refusing (and telling [`AccessPolicy::denied`]) as
/// `reported`: Bind asks a column-level policy `Browse` for a read, and a refusal names the
/// `Select` the statement needs (P3-T062 review m6).
///
/// # Errors
/// As [`authorize`], naming `reported`.
pub async fn authorize_reported(
    policy: &dyn AccessPolicy,
    principal: &Principal,
    securable: &Securable,
    asked: Privilege,
    reported: Privilege,
) -> Result<(), KrishivError> {
    let result = decide(policy, principal, securable, asked).await;
    match result {
        Err(e)
            if e.error_class()
                == krishiv_common::error::ErrorClass::KrishivInsufficientPermissions =>
        {
            policy.denied(principal, securable, reported);
            Err(if asked == reported {
                e
            } else {
                insufficient_permissions(principal, securable, reported)
            })
        }
        other => other,
    }
}

/// Whether `policy` allows `principal` `privilege` on `securable`, asked as [`authorize`] asks it
/// (fail closed: a denial, a policy error or a filter on a change is `false`) but quietly: the
/// policy is not told of a refusal and nothing is traced. For what a listing shows
/// (`SHOW TABLES` lists only what the principal may `Browse`, P3-T062 §2.4): leaving an entry
/// out is not a refusal.
pub async fn allowed(
    policy: &dyn AccessPolicy,
    principal: &Principal,
    securable: &Securable,
    privilege: Privilege,
) -> bool {
    decide_quietly(policy, principal, securable, privilege)
        .await
        .is_ok()
}

/// [`authorize`] as a trial: the same questions, in the same order, and the same answer
/// (the `Browse`-to-`Select` fallback and the filtered-reader rule included), but a refusal
/// is not told to [`AccessPolicy::denied`], so the policy writes no audit record and counts
/// nothing. For checks whose refusal is an answer to the engine, not a refusal of the caller:
/// the analyze API's trial of a run's target checks (P3-T063 review R2-1).
///
/// # Errors
/// As [`authorize`].
pub async fn authorize_quietly(
    policy: &dyn AccessPolicy,
    principal: &Principal,
    securable: &Securable,
    privilege: Privilege,
) -> Result<(), KrishivError> {
    decide(policy, principal, securable, privilege).await
}

async fn decide(
    policy: &dyn AccessPolicy,
    principal: &Principal,
    securable: &Securable,
    privilege: Privilege,
) -> Result<(), KrishivError> {
    // What is asked, before any answer: the analyze API's securables are checked against these
    // events (P3-T063 m3). The securable prints as the denial text does (redacted, ADR 0038).
    tracing::trace!(
        target: "krishiv::authorize",
        principal = %principal.name,
        privilege = %privilege,
        securable = %securable,
        "authorize"
    );
    decide_quietly(policy, principal, securable, privilege).await
}

async fn decide_quietly(
    policy: &dyn AccessPolicy,
    principal: &Principal,
    securable: &Securable,
    privilege: Privilege,
) -> Result<(), KrishivError> {
    let denied = || insufficient_permissions(principal, securable, privilege);
    if privilege.changes() {
        match policy.check(principal, securable, Privilege::Select).await {
            // A write-only principal (reads denied) may still write.
            Ok(AccessDecision::Allow | AccessDecision::Deny { .. }) => {}
            Ok(AccessDecision::AllowWith(_)) | Err(_) => return Err(denied()),
        }
    }
    let decision = match policy.check(principal, securable, privilege).await {
        Ok(AccessDecision::Allow) => Ok(AccessDecision::Allow),
        // Metadata is not rows: a row filter or column mask does not hide that the object
        // exists or its schema.
        Ok(AccessDecision::AllowWith(_)) if privilege == Privilege::Browse => {
            Ok(AccessDecision::Allow)
        }
        // `Select` implies `Browse` (ADR 0046 §4): a provider that knows only `Select` keeps
        // answering metadata commands as it did before `Browse` existed.
        Ok(_) | Err(_) if privilege == Privilege::Browse => {
            match policy.check(principal, securable, Privilege::Select).await {
                Ok(AccessDecision::Allow | AccessDecision::AllowWith(_)) => {
                    Ok(AccessDecision::Allow)
                }
                _ => return Err(denied()),
            }
        }
        other => other,
    };
    match decision {
        Ok(AccessDecision::Allow) => Ok(()),
        Ok(AccessDecision::AllowWith(_)) if privilege == Privilege::Select => {
            Err(KrishivUnsupportedFeature {
                feature: format!("row filters and column masks (on {securable})"),
            }
            .into())
        }
        Ok(AccessDecision::AllowWith(_) | AccessDecision::Deny { .. }) | Err(_) => Err(denied()),
    }
}

/// The column pass of a policy that [decides columns](AccessPolicy::column_level): `Select` on
/// each column of every table and view the final plan reads (`reads`, one per use, see
/// `reads::column_reads`), and on the table itself when some appearance reads whole rows
/// (`count(*)`, a view, or a plan the walk cannot prove). Bind asked only `Browse` on them, so
/// this is the read check; a policy that does not decide columns was asked `Select` on the table
/// at Bind and is not asked again. The uses in `definer_only` (read only under a definer view's
/// text, authorized for the view's owner) are not asked.
///
/// A refused column is named only when the statement wrote a reference to it (`wrote`, given the
/// use's index and the column, asked only then; ADR 0048 §3 Denials, review r3 M1, r4 N1): a
/// column reached through `*`, `* EXCEPT`, a whole-table read or a view's text, or whose name the
/// statement wrote only for something else, is refused as a read of its table, so a refusal never
/// lists a table's columns.
///
/// # Errors
/// As [`authorize`], at `origin`, for the first column (or table) denied.
pub(super) async fn authorize_columns(
    ctx: &super::AnalysisContext,
    uses: &[super::SecurableUse],
    reads: &[super::reads::Reads],
    definer_only: &std::collections::BTreeSet<usize>,
    wrote: impl Fn(usize, &str) -> bool,
    origin: crate::Origin,
) -> Result<(), super::AnalysisError> {
    if !ctx.policy.column_level() {
        return Ok(());
    }
    // The pairs the loop below asks, in one hint first (ADR 0049 §3).
    let mut items: Vec<(Securable, Privilege)> = Vec::new();
    for (i, used) in uses.iter().enumerate() {
        let (Securable::Table(table) | Securable::View(table)) = &used.securable else {
            continue;
        };
        if definer_only.contains(&i) || used.privilege != Privilege::Select {
            continue;
        }
        let read = reads.get(i);
        if read.is_none_or(|r| r.whole) {
            items.push((used.securable.clone(), Privilege::Select));
        }
        for column in read.map(|r| &r.columns).into_iter().flatten() {
            let securable = Securable::Column {
                table: table.clone(),
                column: column.clone(),
            };
            items.push((securable, Privilege::Select));
        }
    }
    if !items.is_empty() {
        ctx.policy.prefetch(&ctx.principal, &items).await;
    }
    for (i, used) in uses.iter().enumerate() {
        if definer_only.contains(&i) {
            continue;
        }
        let (Securable::Table(table) | Securable::View(table)) = &used.securable else {
            continue;
        };
        if used.privilege != Privilege::Select {
            continue;
        }
        let at = |e| super::AnalysisError::new(e, origin);
        // A use without a derived read is asked as whole rows (fail closed).
        let read = reads.get(i);
        if read.is_none_or(|r| r.whole) {
            authorize(
                &*ctx.policy,
                &ctx.principal,
                &used.securable,
                Privilege::Select,
            )
            .await
            .map_err(at)?;
        }
        for column in read.map(|r| &r.columns).into_iter().flatten() {
            let securable = Securable::Column {
                table: table.clone(),
                column: column.clone(),
            };
            authorize(&*ctx.policy, &ctx.principal, &securable, Privilege::Select)
                .await
                .map_err(|e| {
                    if wrote(i, column) {
                        at(e)
                    } else {
                        at(insufficient_permissions(
                            &ctx.principal,
                            &used.securable,
                            Privilege::Select,
                        ))
                    }
                })?;
        }
    }
    Ok(())
}

/// The scopes of the `secret('scope', 'key')` references among `values`, each once, in order.
pub fn secret_scopes<'a>(values: impl IntoIterator<Item = &'a OptionValue>) -> Vec<String> {
    let mut scopes: Vec<String> = Vec::new();
    for v in values {
        if let OptionValue::Secret(SecretValue::Reference { scope, .. }) = v
            && !scopes.contains(scope)
        {
            scopes.push(scope.clone());
        }
    }
    scopes
}

/// Asks the policy for `USAGE` on the scope of every secret reference among `values` (E12), for
/// callers outside analysis (writes, DDL) that are about to resolve them.
///
/// # Errors
/// As [`authorize`], for the first scope the principal may not use.
pub async fn authorize_secret_refs<'a>(
    policy: &dyn AccessPolicy,
    principal: &Principal,
    values: impl IntoIterator<Item = &'a OptionValue>,
) -> Result<(), KrishivError> {
    for scope in secret_scopes(values) {
        authorize(
            policy,
            principal,
            &Securable::Secret { scope },
            Privilege::Usage,
        )
        .await?;
    }
    Ok(())
}

/// The table the name `parts` names, as analysis locates it: a first part
/// naming a catalog qualifies it; otherwise the current catalog, and the
/// current namespace for a one-part name.
#[must_use]
pub fn table_ident(catalogs: &dyn CatalogResolver, parts: &[String]) -> Option<TableIdent> {
    let (last, init) = parts.split_last()?;
    let (catalog, namespace) = if !init.is_empty() && catalogs.catalog(&init[0]).is_some() {
        (init[0].clone(), init[1..].to_vec())
    } else {
        let namespace = if init.is_empty() {
            catalogs.current_namespace().to_vec()
        } else {
            init.to_vec()
        };
        (catalogs.current_catalog().to_owned(), namespace)
    };
    Some(TableIdent {
        catalog,
        namespace,
        name: last.clone(),
    })
}

/// The namespace the name `parts` names, as [`Securable::Namespace`]
/// parts: the catalog, then the namespace (a first part naming a catalog
/// qualifies it, otherwise the current catalog).
#[must_use]
pub fn namespace_parts(catalogs: &dyn CatalogResolver, parts: &[String]) -> Vec<String> {
    match parts.first() {
        Some(first) if parts.len() > 1 && catalogs.catalog(first).is_some() => parts.to_vec(),
        _ => std::iter::once(catalogs.current_catalog().to_owned())
            .chain(parts.iter().cloned())
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use krishiv_common::error::ErrorClass;
    use krishiv_common::secret::OptionValue;

    use super::*;
    use crate::analyzer::catalog::{BoxFuture, PolicyVersion, RowColumnPolicy};
    use crate::analyzer::testing::block_on;
    use crate::analyzer::{AllowAll, DenyAll, PrincipalKind};

    /// Answers per privilege; `None` fails the check.
    struct Fixed(BTreeMap<Privilege, Option<AccessDecision>>);

    impl AccessPolicy for Fixed {
        fn version(&self) -> PolicyVersion {
            PolicyVersion(7)
        }

        fn check<'a>(
            &'a self,
            _: &'a Principal,
            _: &'a Securable,
            privilege: Privilege,
        ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
            let answer = self.0.get(&privilege).cloned().flatten().ok_or_else(|| {
                KrishivError::from(KrishivUnsupportedFeature {
                    feature: "policy store down: secret detail".to_owned(),
                })
            });
            Box::pin(std::future::ready(answer))
        }
    }

    fn deny() -> Option<AccessDecision> {
        Some(AccessDecision::Deny {
            reason: "secret reason".to_owned(),
        })
    }

    fn filtered() -> Option<AccessDecision> {
        Some(AccessDecision::AllowWith(RowColumnPolicy::default()))
    }

    fn check(policy: &dyn AccessPolicy, privilege: Privilege) -> Result<(), KrishivError> {
        let principal = Principal::new("ana", PrincipalKind::User);
        let table = Securable::Table(TableIdent {
            catalog: "spark_catalog".to_owned(),
            namespace: vec!["default".to_owned()],
            name: "t".to_owned(),
        });
        block_on(authorize(policy, &principal, &table, privilege))
    }

    fn fixed(answers: &[(Privilege, Option<AccessDecision>)]) -> Fixed {
        Fixed(answers.iter().cloned().collect())
    }

    const ALL: [Privilege; 10] = [
        Privilege::Select,
        Privilege::Modify,
        Privilege::Create,
        Privilege::Drop,
        Privilege::Monitor,
        Privilege::Usage,
        Privilege::Browse,
        Privilege::Operate,
        Privilege::Manage,
        Privilege::Impersonate,
    ];

    #[test]
    fn select_implies_browse() {
        // A provider that knows only `Select` answers metadata as before `Browse` existed.
        let select_only = fixed(&[
            (Privilege::Select, Some(AccessDecision::Allow)),
            (Privilege::Browse, deny()),
        ]);
        check(&select_only, Privilege::Browse).unwrap();
        // A provider that fails on `Browse` but allows `Select` too.
        let browse_errors = fixed(&[
            (Privilege::Select, Some(AccessDecision::Allow)),
            (Privilege::Browse, None),
        ]);
        check(&browse_errors, Privilege::Browse).unwrap();
        // `Browse` alone does not give `Select`.
        let browse_only = fixed(&[
            (Privilege::Browse, Some(AccessDecision::Allow)),
            (Privilege::Select, deny()),
        ]);
        check(&browse_only, Privilege::Browse).unwrap();
        let e = check(&browse_only, Privilege::Select).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivInsufficientPermissions);
        // Neither: denied, as `BROWSE`, without the reason.
        let neither = fixed(&[(Privilege::Select, deny()), (Privilege::Browse, deny())]);
        let e = check(&neither, Privilege::Browse).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivInsufficientPermissions);
        assert!(
            e.to_string().contains("BROWSE") && !e.to_string().contains("secret"),
            "{e}"
        );
    }

    /// Counts the refusals it is told of.
    struct Counting(Fixed, std::sync::atomic::AtomicUsize);

    impl AccessPolicy for Counting {
        fn version(&self) -> PolicyVersion {
            self.0.version()
        }

        fn check<'a>(
            &'a self,
            p: &'a Principal,
            s: &'a Securable,
            privilege: Privilege,
        ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
            self.0.check(p, s, privilege)
        }

        fn denied(&self, _: &Principal, _: &Securable, _: Privilege) {
            self.1.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[test]
    fn the_policy_is_told_of_each_refusal_once() {
        let counted = |answers: &[(Privilege, Option<AccessDecision>)], p| {
            let policy = Counting(fixed(answers), std::sync::atomic::AtomicUsize::new(0));
            let r = check(&policy, p);
            (
                r.is_ok(),
                policy.1.load(std::sync::atomic::Ordering::Relaxed),
            )
        };
        // `Browse` refused, `Select` allowed: an answer, not a refusal.
        let browse_fallback = [
            (Privilege::Browse, deny()),
            (Privilege::Select, Some(AccessDecision::Allow)),
        ];
        assert_eq!(counted(&browse_fallback, Privilege::Browse), (true, 0));
        // Both refused: one refusal, however many questions it took.
        let neither = [(Privilege::Browse, deny()), (Privilege::Select, deny())];
        assert_eq!(counted(&neither, Privilege::Browse), (false, 1));
        // A write-only principal's read probe is not a refusal.
        let write_only = [
            (Privilege::Select, deny()),
            (Privilege::Modify, Some(AccessDecision::Allow)),
        ];
        assert_eq!(counted(&write_only, Privilege::Modify), (true, 0));
        // A failing policy is a refusal too; an unapplied filter is not a permission refusal.
        assert_eq!(
            counted(&[(Privilege::Select, None)], Privilege::Select),
            (false, 1)
        );
        assert_eq!(
            counted(&[(Privilege::Select, filtered())], Privilege::Select),
            (false, 0)
        );
        // Policies decide tables by default; columns only when they say so.
        assert!(!AllowAll.column_level() && !DenyAll.column_level());
        let column = Securable::Column {
            table: TableIdent {
                catalog: "c".to_owned(),
                namespace: vec!["s".to_owned()],
                name: "t".to_owned(),
            },
            column: "a".to_owned(),
        };
        assert_eq!(column.to_string(), "column `a` of table `c`.`s`.`t`");
    }

    /// T063 review R2-1: a quiet trial decides exactly as [`authorize`] does and tells the policy of
    /// no refusal.
    #[test]
    fn a_quiet_check_decides_as_authorize_and_tells_the_policy_nothing() {
        let principal = Principal::new("ana", PrincipalKind::User);
        let table = Securable::Table(TableIdent {
            catalog: "spark_catalog".to_owned(),
            namespace: vec!["default".to_owned()],
            name: "t".to_owned(),
        });
        let cases: [&[(Privilege, Option<AccessDecision>)]; 6] = [
            &[
                (Privilege::Browse, deny()),
                (Privilege::Select, Some(AccessDecision::Allow)),
            ],
            &[(Privilege::Browse, deny()), (Privilege::Select, deny())],
            &[
                (Privilege::Select, deny()),
                (Privilege::Modify, Some(AccessDecision::Allow)),
            ],
            &[(Privilege::Select, None)],
            &[(Privilege::Select, filtered())],
            &[
                (Privilege::Select, filtered()),
                (Privilege::Drop, Some(AccessDecision::Allow)),
            ],
        ];
        for answers in cases {
            for p in ALL {
                let loud = Counting(fixed(answers), std::sync::atomic::AtomicUsize::new(0));
                let quiet = Counting(fixed(answers), std::sync::atomic::AtomicUsize::new(0));
                let a = block_on(authorize(&loud, &principal, &table, p));
                let q = block_on(authorize_quietly(&quiet, &principal, &table, p));
                assert_eq!(
                    a.as_ref().map_err(KrishivError::to_string),
                    q.as_ref().map_err(KrishivError::to_string),
                    "{p} {answers:?}"
                );
                assert_eq!(quiet.1.load(std::sync::atomic::Ordering::Relaxed), 0);
                // `authorize` itself tells the policy once of each permission refusal.
                let refused = a
                    .as_ref()
                    .is_err_and(|e| e.error_class() == ErrorClass::KrishivInsufficientPermissions);
                assert_eq!(
                    loud.1.load(std::sync::atomic::Ordering::Relaxed),
                    usize::from(refused),
                    "{p} {answers:?}"
                );
            }
        }
    }

    #[test]
    fn a_filtered_reader_may_browse() {
        // Metadata is not rows: a row filter answered for `Browse` or for its `Select`
        // fallback still shows that the object exists and its schema.
        for policy in [
            fixed(&[(Privilege::Browse, filtered()), (Privilege::Select, deny())]),
            fixed(&[(Privilege::Browse, deny()), (Privilege::Select, filtered())]),
        ] {
            check(&policy, Privilege::Browse).unwrap();
        }
    }

    #[test]
    fn allow_all_and_deny_all() {
        for p in ALL {
            check(&AllowAll, p).unwrap();
            let e = check(&DenyAll, p).unwrap_err();
            assert_eq!(e.error_class(), ErrorClass::KrishivInsufficientPermissions);
            assert_eq!(e.sql_state(), Some("42501"));
            assert!(e.to_string().contains(&p.to_string()), "{e}");
        }
    }

    #[test]
    fn a_filtered_reader_cannot_change_the_securable() {
        // Reads filtered, writes allowed: every change is denied.
        let policy = fixed(&[
            (Privilege::Select, filtered()),
            (Privilege::Modify, Some(AccessDecision::Allow)),
            (Privilege::Create, Some(AccessDecision::Allow)),
            (Privilege::Drop, Some(AccessDecision::Allow)),
            (Privilege::Monitor, Some(AccessDecision::Allow)),
        ]);
        for p in [Privilege::Modify, Privilege::Create, Privilege::Drop] {
            let e = check(&policy, p).unwrap_err();
            assert_eq!(
                e.error_class(),
                ErrorClass::KrishivInsufficientPermissions,
                "{p}"
            );
        }
        // Monitor changes nothing; the filtered read itself is not applied yet.
        check(&policy, Privilege::Monitor).unwrap();
        let e = check(&policy, Privilege::Select).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    }

    #[test]
    fn writes_need_their_own_grant_and_fail_closed() {
        // Write-only: reads denied, writes allowed.
        let write_only = fixed(&[
            (Privilege::Select, deny()),
            (Privilege::Modify, Some(AccessDecision::Allow)),
        ]);
        check(&write_only, Privilege::Modify).unwrap();
        // A filter answered for the write itself, a denial, or a failing
        // policy (for the write or for the read it asks first): denied,
        // without the reason or the policy's error text.
        for policy in [
            fixed(&[
                (Privilege::Select, Some(AccessDecision::Allow)),
                (Privilege::Modify, filtered()),
            ]),
            fixed(&[
                (Privilege::Select, Some(AccessDecision::Allow)),
                (Privilege::Modify, deny()),
            ]),
            fixed(&[
                (Privilege::Select, Some(AccessDecision::Allow)),
                (Privilege::Modify, None),
            ]),
            fixed(&[
                (Privilege::Select, None),
                (Privilege::Modify, Some(AccessDecision::Allow)),
            ]),
        ] {
            let e = check(&policy, Privilege::Modify).unwrap_err();
            assert_eq!(e.error_class(), ErrorClass::KrishivInsufficientPermissions);
            let text = e.to_string();
            assert!(!text.contains("secret"), "{text}");
        }
    }

    #[test]
    fn securables_and_privileges_print() {
        assert_eq!(
            Securable::Namespace(vec!["spark_catalog".to_owned(), "db".to_owned()]).to_string(),
            "namespace `spark_catalog`.`db`"
        );
        assert_eq!(
            Securable::System("cluster".to_owned()).to_string(),
            "system `cluster`"
        );
        assert_eq!(
            Securable::Path(OptionValue::from_user("path", "file:/a")).to_string(),
            "path 'file:/a'"
        );
        assert_eq!(
            Securable::Secret {
                scope: "prod".to_owned()
            }
            .to_string(),
            "secret `prod`"
        );
        assert_eq!(
            Securable::Connection("lake".to_owned()).to_string(),
            "connection `lake`"
        );
        let names: Vec<String> = ALL.iter().map(ToString::to_string).collect();
        assert_eq!(
            names,
            [
                "SELECT",
                "MODIFY",
                "CREATE",
                "DROP",
                "MONITOR",
                "USAGE",
                "BROWSE",
                "OPERATE",
                "MANAGE",
                "IMPERSONATE"
            ]
        );
        assert_eq!(
            ALL.map(Privilege::changes),
            [
                false, true, true, true, false, false, false, false, false, false
            ]
        );
    }
}
