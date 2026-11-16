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
    // The groups are the caller's claim, so the caller's own role caps them (review M2): a
    // viewer front end claiming `dbas` makes a viewer, never an admin.
    let low = caller("portal", Role::Viewer, Method::Token);
    let claimed_admin = vec!["analysts".to_owned(), "dbas".to_owned()];
    for groups in [vec!["analysts".to_owned()], claimed_admin.clone()] {
        let capped = auth
            .impersonate(
                &low,
                Surface::Connect,
                loopback(),
                &ActAs::new("dave").with_groups(groups),
            )
            .expect("allowed");
        assert_eq!(capped.role, Some(Role::Viewer));
    }
    let mid = caller("portal", Role::Operator, Method::Token);
    let capped = auth
        .impersonate(
            &mid,
            Surface::Connect,
            loopback(),
            &ActAs::new("dave").with_groups(claimed_admin),
        )
        .expect("allowed");
    assert_eq!(capped.role, Some(Role::Operator));
}

/// Rights belong to service principals (review M1): a user that carries a service's name, from
/// OIDC or a user certificate, does not inherit `impersonate`.
#[test]
fn user_named_like_a_service_cannot_impersonate() {
    let auth = server(&[]);
    for method in [Method::Oidc, Method::Mtls, Method::Token] {
        let user = Authenticated::new(
            Principal::new("portal", PrincipalKind::User),
            Some(Role::Admin),
            method,
        );
        assert!(denied(auth.impersonate(
            &user,
            Surface::Connect,
            loopback(),
            &ActAs::new("alice")
        )));
    }
}

/// `krishiv.auth.principals.<name>.expires` ends the privileges for every method (review M3).
#[test]
fn expired_principal_cannot_impersonate() {
    let auth = server(&[("krishiv.auth.principals.portal.expires", "2001-01-01")]);
    let by_cert = caller("portal", Role::Admin, Method::Mtls);
    assert!(denied(auth.impersonate(
        &by_cert,
        Surface::Connect,
        loopback(),
        &ActAs::new("alice")
    )));
    let later = server(&[("krishiv.auth.principals.portal.expires", "2999-01-01")]);
    assert!(
        later
            .impersonate(&by_cert, Surface::Connect, loopback(), &ActAs::new("alice"))
            .is_ok()
    );
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
    // Naming the caller's own tenant is the same request.
    assert!(
        auth.impersonate(
            &in_t,
            Surface::Connect,
            loopback(),
            &ActAs::new("alice").with_tenant("t1")
        )
        .is_ok()
    );
    // A caller outside every tenant cannot place the end user in one (review m1).
    assert!(denied(auth.impersonate(
        &portal(),
        Surface::Connect,
        loopback(),
        &ActAs::new("alice").with_tenant("t2"),
    )));
}

/// The I/O secret resolver and block-cache scopes key on the principal's name and tenant, so
/// two end users of one front end, or one end user in two tenants, never share a scope.
#[test]
fn block_cache_not_shared_across_tenants_via_impersonation() {
    let auth = server(&[]);
    let act = |user: &str, tenant: &str| {
        let mut front = portal();
        front.principal.tenant = Some(tenant.to_owned());
        auth.impersonate(&front, Surface::Connect, loopback(), &ActAs::new(user))
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
        .with_ansi(false)
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
        // A caller-supplied name with a line break is never written (review m2).
        let forged = ActAs::new("mallory\nkrishiv::audit decision=allowed");
        let _ = auth.impersonate(&portal(), Surface::Connect, loopback(), &forged);
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
    assert!(!logged.contains("mallory"), "{logged}");
    assert!(logged.contains("malformed=true"), "{logged}");
    assert!(!logged.contains("analysts"), "{logged}");
}

#[derive(Clone, Default)]
struct LogSink(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for LogSink {
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

impl LogSink {
    fn text(&self) -> String {
        String::from_utf8(
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone(),
        )
        .unwrap_or_default()
    }
}

/// With `signedAssertion` off (and in a build without `oidc`) every assertion is refused with
/// the one error class, one audit record and one denied count; the reason is only in the log.
#[tokio::test]
async fn signed_assertion_off_is_refused_audited_and_counted() {
    let sink = LogSink::default();
    let writer = sink.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_max_level(tracing::Level::TRACE)
        .with_writer(move || writer.clone())
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);
    let auth = server(&[]);
    let r = auth
        .impersonate_signed(&portal(), Surface::Connect, remote(), "a.b.c")
        .await;
    assert!(
        matches!(&r, Err(e) if e.error_class() == ErrorClass::KrishivAuthImpersonationDenied
        && e.sql_state() == Some("42501"))
    );
    assert_eq!(
        auth.impersonation_counts(),
        ImpersonationCount {
            allowed: 0,
            denied: 1
        }
    );
    let logged = sink.text();
    assert_eq!(logged.matches("krishiv::audit").count(), 1, "{logged}");
    assert!(logged.contains("reason_code=\"assertion_off\""), "{logged}");
    assert!(logged.contains("decision=\"denied\""), "{logged}");
    assert!(!logged.contains("a.b.c"), "{logged}");
}

/// A caller that may not impersonate is refused on its own standing, before any assertion is
/// looked at (review m3): `no_privilege`, not the assertion's verdict.
#[tokio::test]
async fn signed_assertion_from_a_caller_without_the_privilege_is_refused() {
    let sink = LogSink::default();
    let writer = sink.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_max_level(tracing::Level::TRACE)
        .with_writer(move || writer.clone())
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);
    let auth = server(&[]);
    let batch = caller("batch", Role::Admin, Method::Token);
    let r = auth
        .impersonate_signed(&batch, Surface::Connect, remote(), "a.b.c")
        .await;
    assert!(matches!(&r, Err(e) if e.error_class() == ErrorClass::KrishivAuthImpersonationDenied));
    assert_eq!(auth.impersonation_counts().denied, 1);
    assert_eq!(auth.impersonation_counts().allowed, 0);
    let logged = sink.text();
    assert_eq!(logged.matches("krishiv::audit").count(), 1, "{logged}");
    assert!(logged.contains("reason_code=\"no_privilege\""), "{logged}");
}

#[test]
fn denial_reason_codes_are_stable() {
    use super::impersonation::Denial;
    for (d, code) in [
        (Denial::NoPrivilege, "no_privilege"),
        (Denial::NotAllowed, "not_allowed"),
        (Denial::Chained, "chained"),
        (Denial::UntrustedChannel, "untrusted_channel"),
        (Denial::Malformed, "malformed"),
        (Denial::OtherTenant, "other_tenant"),
        (Denial::AssertionOff, "assertion_off"),
        (Denial::BadAssertion, "bad_assertion"),
        (Denial::AssertionUnavailable, "assertion_unavailable"),
        (Denial::ActorMismatch, "actor_mismatch"),
        (Denial::Expired, "expired"),
    ] {
        assert_eq!(d.as_str(), code);
        assert_eq!(
            d.error().error_class(),
            ErrorClass::KrishivAuthImpersonationDenied
        );
    }
}

#[test]
fn signed_assertion_settings() {
    let on = [
        ("krishiv.auth.impersonation.signedAssertion", " TRUE "),
        (
            "krishiv.auth.impersonation.assertion.issuer",
            "https://portal.example.test/",
        ),
        (
            "krishiv.auth.impersonation.assertion.audience",
            "krishiv-act-as",
        ),
    ];
    // The assertion settings share OIDC's validator: a build without `oidc` refuses them.
    let on = AuthConfig::from_settings(on);
    assert_eq!(on.is_ok(), cfg!(feature = "oidc"), "{on:?}");
    if let Ok(c) = on {
        assert!(c.impersonation.signed_assertion);
    }
    let off = AuthConfig::from_settings([("krishiv.auth.impersonation.signedAssertion", "false")])
        .expect("valid");
    assert!(!off.impersonation.signed_assertion);
    // Not a boolean; on without an issuer; assertion settings while off.
    let bad = |pairs: Vec<(&str, &str)>| AuthConfig::from_settings(pairs).expect_err("refused");
    let e = bad(vec![("krishiv.auth.impersonation.signedAssertion", "yes")]);
    assert!(e.to_string().contains("expected true or false"), "{e}");
    let e = bad(vec![("krishiv.auth.impersonation.signedAssertion", "true")]);
    assert!(e.to_string().contains("assertion.issuer"), "{e}");
    let e = bad(vec![
        (
            "krishiv.auth.impersonation.assertion.issuer",
            "https://portal.example.test/",
        ),
        (
            "krishiv.auth.impersonation.assertion.audience",
            "krishiv-act-as",
        ),
    ]);
    // Without `oidc` the build refuses the settings first; with it, the missing switch.
    if cfg!(feature = "oidc") {
        assert!(e.to_string().contains("signedAssertion"), "{e}");
    }
}

/// A principal that already acts for someone never presents an assertion: refused as `chained`
/// on its own standing, counted once.
#[tokio::test]
async fn signed_assertion_from_a_delegated_principal_is_refused() {
    let auth = server(&[]);
    let alice = auth
        .impersonate(
            &portal(),
            Surface::Connect,
            loopback(),
            &ActAs::new("alice"),
        )
        .expect("allowed");
    let r = auth
        .impersonate_signed(&alice, Surface::Connect, remote(), "a.b.c")
        .await;
    assert!(matches!(&r, Err(e) if e.error_class() == ErrorClass::KrishivAuthImpersonationDenied));
    assert_eq!(
        auth.impersonation_counts(),
        ImpersonationCount {
            allowed: 1,
            denied: 1
        }
    );
}
