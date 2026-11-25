// SPDX-License-Identifier: Apache-2.0
//! Unit tests of the composition and the provider selection.

use super::*;
use krishiv_plan::analyzer::{PrincipalKind, TableIdent};

fn table(catalog: &str) -> Securable {
    Securable::Table(TableIdent {
        catalog: catalog.to_owned(),
        namespace: vec![],
        name: "queries".to_owned(),
    })
}

#[test]
fn securables_route_to_grants_or_roles() {
    assert_eq!(
        GrantsPolicy::decided_by(&table("spark_catalog")),
        DecidedBy::Grants
    );
    assert_eq!(GrantsPolicy::decided_by(&table("system")), DecidedBy::Roles);
    assert_eq!(
        GrantsPolicy::decided_by(&Securable::System("udf".to_owned())),
        DecidedBy::Roles
    );
}

#[test]
fn admin_passes_data_checks_but_never_impersonates() {
    let grants = Grants::new(Arc::new(MemoryGrantStore::new()));
    let p = Principal::new("root", PrincipalKind::User);
    let admin = GrantsPolicy::new(
        Arc::clone(&grants),
        Arc::new(krishiv_plan::analyzer::DenyAll),
        true,
    );
    let user = GrantsPolicy::new(
        Arc::clone(&grants),
        Arc::new(krishiv_plan::analyzer::AllowAll),
        false,
    );
    let t = table("spark_catalog");
    let decide = |pol: &GrantsPolicy, s: &Securable, pr| {
        futures::executor::block_on(pol.check(&p, s, pr)).unwrap()
    };
    assert_eq!(decide(&admin, &t, Privilege::Drop), AccessDecision::Allow);
    assert!(matches!(
        decide(&user, &t, Privilege::Select),
        AccessDecision::Deny { .. }
    ));
    // The role matrix (here deny-all) still decides the server's own objects for an admin.
    assert!(matches!(
        decide(
            &admin,
            &Securable::System("cluster".to_owned()),
            Privilege::Manage
        ),
        AccessDecision::Deny { .. }
    ));
    assert!(matches!(
        decide(&admin, &t, Privilege::Impersonate),
        AccessDecision::Deny { .. }
    ));
    assert!(user.column_level());
}

#[test]
fn the_version_moves_with_the_epoch() {
    let grants = Grants::new(Arc::new(MemoryGrantStore::new()));
    let policy = GrantsPolicy::new(
        Arc::clone(&grants),
        Arc::new(krishiv_plan::analyzer::AllowAll),
        false,
    );
    let before = policy.version();
    grants.store().update(&mut |_| Ok(())).unwrap();
    assert_ne!(policy.version(), before);
}

#[test]
fn the_provider_follows_the_config() {
    let config = |v: Option<&str>| {
        let mut c = Config::new();
        if let Some(v) = v {
            c.set("krishiv.access.provider", v).unwrap();
        }
        c
    };
    assert_eq!(
        access_provider(&config(None), true).unwrap(),
        AccessProvider::Grants
    );
    assert_eq!(
        access_provider(&config(None), false).unwrap(),
        AccessProvider::Roles
    );
    assert_eq!(
        access_provider(&config(Some("GRANTS")), false).unwrap(),
        AccessProvider::Grants
    );
    assert_eq!(
        access_provider(&config(Some("roles")), true).unwrap(),
        AccessProvider::Roles
    );
    assert_eq!(
        access_provider(&config(Some("external")), true).unwrap(),
        AccessProvider::External
    );
    assert_eq!(
        access_providers(&config(Some("grants + External")), true).unwrap(),
        vec![AccessProvider::Grants, AccessProvider::External]
    );
    assert_eq!(
        access_providers(&config(Some("roles+external")), true).unwrap(),
        vec![AccessProvider::Roles, AccessProvider::External]
    );
    for (value, reason) in [
        ("catalog", "G5"),
        ("grants+catalog", "G5"),
        ("grants+external", "not a single provider"),
        ("grants+grants", "twice"),
        ("grants+", "empty"),
        ("grants,external", "`+`-separated"),
        ("ranger", "grants, roles or external"),
    ] {
        let e = access_provider(&config(Some(value)), true).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INVALID_CONFIG_VALUE",
            "{value}"
        );
        assert!(e.to_string().contains(reason), "{value}: {e}");
    }
}

#[test]
fn a_denial_names_the_provider_that_decided_it() {
    let principal = Principal::new("ana", PrincipalKind::User);
    let of = |securable: &Securable| {
        denial_provider(&krishiv_plan::analyzer::insufficient_permissions(
            &principal,
            securable,
            Privilege::Select,
        ))
    };
    assert_eq!(of(&table("spark_catalog")), DecidedBy::Grants);
    assert_eq!(of(&table("SYSTEM")), DecidedBy::Roles);
    assert_eq!(of(&table("sys`tem")), DecidedBy::Grants);
    assert_eq!(
        of(&Securable::System("cluster".to_owned())),
        DecidedBy::Roles
    );
    assert_eq!(
        of(&Securable::Namespace(vec![
            "system".to_owned(),
            "x".to_owned()
        ])),
        DecidedBy::Roles
    );
    let column = |catalog: &str, column: &str| Securable::Column {
        table: TableIdent {
            catalog: catalog.to_owned(),
            namespace: vec![],
            name: "t".to_owned(),
        },
        column: column.to_owned(),
    };
    assert_eq!(of(&column("system", "a")), DecidedBy::Roles);
    // A column name that reads like the rest of the text decides nothing.
    assert_eq!(
        of(&column("spark_catalog", "x` of table `system")),
        DecidedBy::Grants
    );
    assert_eq!(
        of(&Securable::Path(
            krishiv_common::secret::OptionValue::from_user("path", "/system/x")
        )),
        DecidedBy::Grants
    );
    let other: KrishivError = krishiv_common::error::KrishivUnsupportedFeature {
        feature: "x".to_owned(),
    }
    .into();
    assert_eq!(denial_provider(&other), DecidedBy::Grants);
}
