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
    let denied = || insufficient_permissions(principal, securable, privilege);
    // What is asked, before any answer: the analyze API's securables are checked against these
    // events (P3-T063 m3). The securable prints as the denial text does (redacted, ADR 0038).
    tracing::trace!(
        target: "krishiv::authorize",
        principal = %principal.name,
        privilege = %privilege,
        securable = %securable,
        "authorize"
    );
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
