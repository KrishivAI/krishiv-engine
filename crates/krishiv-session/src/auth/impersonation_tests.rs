// SPDX-License-Identifier: Apache-2.0
//! Trusted impersonation (design P3-T061 §4.2 and §7): each rule refuses on its own, the end
//! user's role applies, the audit record and the counters.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex, PoisonError};

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::principal::{Principal, PrincipalKind};

use super::{
    ActAs, AuthConfig, Authenticated, ImpersonationCount, Method, Role, ServerAuth, Surface,
};

fn loopback() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 40_000)
}

fn remote() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 9)), 40_000)
}

fn server(extra: &[(&str, &str)]) -> ServerAuth {
    let mut pairs = vec![
        ("krishiv.auth.principals.portal.privileges", "impersonate"),
        (
            "krishiv.auth.principals.portal.impersonates",
            "alice, team-*, group:analysts",
        ),
        ("krishiv.auth.impersonation.roles.operator", "analysts"),
        ("krishiv.auth.impersonation.roles.admin", "dbas"),
    ];
    pairs.extend_from_slice(extra);
    let config = AuthConfig::from_settings(pairs).expect("valid");
    ServerAuth::new(&config).expect("builds")
}

fn caller(name: &str, role: Role, method: Method) -> Authenticated {
    Authenticated::new(
        Principal::new(name, PrincipalKind::Service),
        Some(role),
        method,
    )
}

fn portal() -> Authenticated {
    caller("portal", Role::Admin, Method::Token).with_token_id(Some("jti-1".to_owned()))
}

fn denied(r: Result<Authenticated, KrishivError>) -> bool {
    matches!(r, Err(e) if e.error_class() == ErrorClass::KrishivAuthImpersonationDenied
        && e.sql_state() == Some("42501"))
}

#[test]
fn impersonated_principal_is_the_end_user_through_the_caller() {
    let auth = server(&[]);
    let end = auth
        .impersonate(
            &portal(),
            Surface::Connect,
            loopback(),
            &ActAs::new("alice"),
        )
        .expect("allowed");
    assert_eq!(end.principal.name, "alice");
    assert_eq!(end.principal.kind, PrincipalKind::User);
    assert_eq!(end.principal.delegated_by.as_deref(), Some("portal"));
    assert_eq!(end.method, Method::Token);
    assert_eq!(end.token_id.as_deref(), Some("jti-1"));
    assert!(
        auth.impersonate(
            &portal(),
            Surface::Connect,
            loopback(),
            &ActAs::new("team-x")
        )
        .is_ok()
    );
}

#[test]
fn end_user_role_applies_not_caller_role() {
    let auth = server(&[]);
    // An admin caller acting for a user with no listed group: viewer.
    let plain = auth
        .impersonate(
            &portal(),
            Surface::Connect,
            loopback(),
            &ActAs::new("alice"),
        )
        .expect("allowed");
    assert_eq!(plain.role, Some(Role::Viewer));
    // The highest role of the end user's groups.
    let groups = vec!["analysts".to_owned(), "dbas".to_owned()];
    let both = auth
        .impersonate(
            &portal(),
            Surface::Connect,
            loopback(),
            &ActAs::new("carol").with_groups(groups),
        )
        .expect("allowed by group");
    assert_eq!(both.role, Some(Role::Admin));
    assert_eq!(both.principal.groups, ["analysts", "dbas"]);
    // A viewer caller cannot lower anything either: the end user's role, whatever the caller's.
    let low = caller("portal", Role::Viewer, Method::Token);
    let op = auth
        .impersonate(
            &low,
            Surface::Connect,
            loopback(),
            &ActAs::new("dave").with_groups(vec!["analysts".to_owned()]),
        )
        .expect("allowed");
    assert_eq!(op.role, Some(Role::Operator));
}

#[test]
fn impersonation_without_privilege_denied() {
    let auth = server(&[]);
    let other = caller("etl", Role::Admin, Method::Token);
    assert!(denied(auth.impersonate(
        &other,
        Surface::Connect,
        loopback(),
        &ActAs::new("alice")
    )));
    // Its own name needs no right, and is the caller unchanged.
    let me = auth
        .impersonate(&other, Surface::Connect, loopback(), &ActAs::new("etl"))
        .expect("itself");
    assert_eq!(me, other);
    // Its own name with groups is asking for something: refused.
    let with_groups = ActAs::new("etl").with_groups(vec!["dbas".to_owned()]);
    assert!(denied(auth.impersonate(
        &other,
        Surface::Connect,
        loopback(),
        &with_groups
    )));
}

#[test]
fn impersonation_outside_allowlist_denied() {
    let auth = server(&[]);
    for act_as in [
        ActAs::new("bob"),
        ActAs::new("xteam-a"),
        ActAs::new("carol").with_groups(vec!["devs".to_owned()]),
    ] {
        assert!(
            denied(auth.impersonate(&portal(), Surface::Connect, loopback(), &act_as)),
            "{act_as:?}"
        );
    }
}

#[test]
fn chained_impersonation_refused() {
    let auth = server(&[]);
    let alice = auth
        .impersonate(
            &portal(),
            Surface::Connect,
            loopback(),
            &ActAs::new("alice"),
        )
        .expect("allowed");
    // Even acting as itself: a delegated principal never goes through impersonation again.
    for act_as in [ActAs::new("team-a"), ActAs::new("alice")] {
        assert!(denied(auth.impersonate(
            &alice,
            Surface::Connect,
            loopback(),
            &act_as
        )));
    }
}

#[test]
fn plain_act_as_needs_loopback_or_a_client_certificate() {
    let auth = server(&[]);
    assert!(denied(auth.impersonate(
        &portal(),
        Surface::Connect,
        remote(),
        &ActAs::new("alice")
    )));
    let mtls = caller("portal", Role::Viewer, Method::Mtls);
    assert!(
        auth.impersonate(&mtls, Surface::Connect, remote(), &ActAs::new("alice"))
            .is_ok()
    );
}

#[test]
fn malformed_requests_and_other_tenants_refused() {
    let auth = server(&[]);
    let too_many = vec!["analysts".to_owned(); 65];
    for act_as in [
        ActAs::new(""),
        ActAs::new("ali\u{200b}ce"),
        ActAs::new("a".repeat(257)),
        ActAs::new("alice").with_groups(vec![String::new()]),
        ActAs::new("alice").with_groups(too_many),
        ActAs::new("alice").with_tenant("t\n"),
    ] {
        assert!(denied(auth.impersonate(
            &portal(),
            Surface::Connect,
            loopback(),
            &act_as
        )));
    }
    let mut in_t = portal();
    in_t.principal.tenant = Some("t1".to_owned());
    assert!(denied(auth.impersonate(
        &in_t,
        Surface::Connect,
        loopback(),
        &ActAs::new("alice").with_tenant("t2")
    )));
    // The caller's tenant is the end user's.
    let same = auth
        .impersonate(&in_t, Surface::Connect, loopback(), &ActAs::new("alice"))
        .expect("allowed");
    assert_eq!(same.principal.tenant.as_deref(), Some("t1"));
    // A caller outside every tenant may name the end user's.
    let named = auth
        .impersonate(
            &portal(),
            Surface::Connect,
            loopback(),
            &ActAs::new("alice").with_tenant("t2"),
        )
        .expect("allowed");
    assert_eq!(named.principal.tenant.as_deref(), Some("t2"));
}

/// The I/O secret resolver and block-cache scopes key on the principal's name and tenant, so
/// two end users of one front end, or one end user in two tenants, never share a scope.
#[test]
fn block_cache_not_shared_across_tenants_via_impersonation() {
    let auth = server(&[]);
    let act = |user: &str, tenant: &str| {
        auth.impersonate(
            &portal(),
            Surface::Connect,
            loopback(),
            &ActAs::new(user).with_tenant(tenant),
        )
        .expect("allowed")
        .principal
    };
    let a1 = act("alice", "t1");
    assert_ne!(a1.id(), act("alice", "t2").id());
    assert_ne!(a1.id(), act("team-a", "t1").id());
    // The same end user directly: the same scope (the delegator is audit, not identity).
    let mut direct = Principal::new("alice", PrincipalKind::User);
    direct.tenant = Some("t1".to_owned());
    assert_eq!(a1.id(), direct.id());
}

#[test]
fn counters_count_each_decision_on_someone_else() {
    let auth = server(&[]);
    let _ = auth.impersonate(
        &portal(),
        Surface::Connect,
        loopback(),
        &ActAs::new("alice"),
    );
    let _ = auth.impersonate(&portal(), Surface::Connect, loopback(), &ActAs::new("bob"));
    let _ = auth.impersonate(&portal(), Surface::Connect, remote(), &ActAs::new("alice"));
    let _ = auth.impersonate(
        &portal(),
        Surface::Connect,
        loopback(),
        &ActAs::new("portal"),
    );
    let c = auth.impersonation_counts();
    assert_eq!(
        c,
        ImpersonationCount {
            allowed: 1,
            denied: 2
        }
    );
}

#[test]
fn impersonation_config_refusals() {
    for (k, v) in [
        ("krishiv.auth.impersonation.roles.root", "g"),
        ("krishiv.auth.impersonation.roleMap", "g=admin"),
        ("krishiv.auth.impersonation.roles.Admin", "g"),
    ] {
        assert!(AuthConfig::from_settings([(k, v)]).is_err(), "{k}");
    }
    let err = AuthConfig::from_settings([
        ("krishiv.auth.impersonation.roles.viewer", "g"),
        ("krishiv.auth.impersonation.roles.admin", "g"),
    ])
    .expect_err("one role per group");
    assert!(err.to_string().contains("exactly one role"), "{err}");
}

#[test]
fn audit_event_has_no_token_text() {
    #[derive(Clone, Default)]
    struct Sink(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for Sink {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let sink = Sink::default();
    let writer = sink.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_writer(move || writer.clone())
        .finish();
    let secret_group = "payroll-secret-group";
    tracing::subscriber::with_default(subscriber, || {
        let auth = server(&[]);
        let act =
            ActAs::new("carol").with_groups(vec!["analysts".to_owned(), secret_group.to_owned()]);
        let _ = auth.impersonate(&portal(), Surface::Connect, loopback(), &act);
        let _ = auth.impersonate(&portal(), Surface::Connect, remote(), &ActAs::new("bob"));
    });
    let logged = String::from_utf8(
        sink.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone(),
    )
    .unwrap_or_default();
    for want in [
        "krishiv::audit",
        "caller=portal",
        "end_user=carol",
        "decision=\"allowed\"",
        "decision=\"denied\"",
        "reason_code=\"untrusted_channel\"",
        "jti=\"jti-1\"",
        "groups_hash=",
    ] {
        assert!(logged.contains(want), "{want} in {logged}");
    }
    assert!(!logged.contains(secret_group), "{logged}");
    assert!(!logged.contains("analysts"), "{logged}");
}
