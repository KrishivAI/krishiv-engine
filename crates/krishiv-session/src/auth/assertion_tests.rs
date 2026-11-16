// SPDX-License-Identifier: Apache-2.0
//! Signed act-as assertions (design P3-T061 §4.1, unit I3; §7 `chained_assertion_refused`): an
//! RFC 8693 `act` JWT from the assertion issuer lifts the loopback/mTLS rule, and nothing else.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::principal::{Principal, PrincipalKind};
use serde_json::{Value, json};

use super::oidc::unix_now;
use super::oidc_tests::{
    Source, jwks, rs_header, rs256, rsa, rsa_jwk, rsa_other, token, wait_until,
};
use super::{
    ActAs, AuthConfig, Authenticated, ImpersonationCount, Method, Role, ServerAuth, Surface,
};

const ISSUER: &str = "https://portal.example.test/";
const AUDIENCE: &str = "krishiv-act-as";

fn remote() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 9)), 40_000)
}

fn base() -> Vec<(&'static str, &'static str)> {
    vec![
        ("krishiv.auth.principals.portal.privileges", "impersonate"),
        (
            "krishiv.auth.principals.portal.impersonates",
            "alice, group:analysts",
        ),
        ("krishiv.auth.principals.batch.privileges", ""),
        ("krishiv.auth.impersonation.roles.operator", "analysts"),
        ("krishiv.auth.impersonation.signedAssertion", "true"),
        ("krishiv.auth.impersonation.assertion.issuer", ISSUER),
        ("krishiv.auth.impersonation.assertion.audience", AUDIENCE),
    ]
}

fn config(extra: &[(&'static str, &'static str)]) -> Result<AuthConfig, KrishivError> {
    let mut pairs = base();
    pairs.extend_from_slice(extra);
    AuthConfig::from_settings(pairs)
}

/// A server whose assertion keys come from `source`, after its first fetch.
fn server_with(source: &Source) -> ServerAuth {
    let auth =
        ServerAuth::with_assertion_source(&config(&[]).expect("valid"), Box::new(source.clone()))
            .expect("builds");
    wait_until(|| source.fetches() >= 1);
    auth
}

fn server() -> (ServerAuth, Source) {
    let source = Source::new(vec![Ok(jwks(&[rsa_jwk(rsa(), "k1")]))]);
    let auth = server_with(&source);
    // The first fetch installed the key once the verifier answers for it.
    wait_until(|| auth.assertions().is_some_and(|a| a.jwks().key_count() == 1));
    (auth, source)
}

fn caller(name: &str) -> Authenticated {
    Authenticated::new(
        Principal::new(name, PrincipalKind::Service),
        Some(Role::Admin),
        Method::Token,
    )
    .with_token_id(Some("caller-jti".to_owned()))
}

fn claims(sub: &str, act: &Value) -> Value {
    let now = unix_now();
    json!({
        "iss": ISSUER, "aud": AUDIENCE, "sub": sub, "groups": ["analysts"],
        "iat": now, "exp": now + 120, "jti": "assert-1", "act": act,
    })
}

fn signed(claims: &Value) -> String {
    token(&rs_header("k1"), claims, rs256(rsa()))
}

fn present(
    auth: &ServerAuth,
    caller: &Authenticated,
    assertion: &str,
) -> Result<(ActAs, Authenticated), KrishivError> {
    block_on(auth.impersonate_signed(caller, Surface::Connect, remote(), assertion))
}

fn denied<T>(r: Result<T, KrishivError>) -> bool {
    matches!(r, Err(e) if e.error_class() == ErrorClass::KrishivAuthImpersonationDenied
        && e.sql_state() == Some("42501"))
}

#[test]
fn signed_assertion_acts_for_the_end_user_from_a_remote_peer() {
    let (auth, _) = server();
    let (act_as, end) = present(
        &auth,
        &caller("portal"),
        &signed(&claims("alice", &json!({"sub": "portal"}))),
    )
    .expect("allowed");
    assert_eq!(
        act_as,
        ActAs::new("alice").with_groups(vec!["analysts".to_owned()])
    );
    assert_eq!(end.principal.name, "alice");
    assert_eq!(end.principal.delegated_by.as_deref(), Some("portal"));
    // The end user's role, from the impersonation lists, not the caller's admin.
    assert_eq!(end.role, Some(Role::Operator));
    // The same request as plain act-as from that peer is refused (rule 5).
    assert!(denied(auth.impersonate(
        &caller("portal"),
        Surface::Connect,
        remote(),
        &act_as
    )));
    assert_eq!(
        auth.impersonation_counts(),
        ImpersonationCount {
            allowed: 1,
            denied: 1
        }
    );
}

#[test]
fn chained_assertion_refused() {
    let (auth, _) = server();
    let chain = json!({"sub": "portal", "act": {"sub": "gateway"}});
    assert!(denied(present(
        &auth,
        &caller("portal"),
        &signed(&claims("alice", &chain))
    )));
}

#[test]
fn assertion_for_another_caller_refused() {
    let (auth, _) = server();
    let other = signed(&claims("alice", &json!({"sub": "someone-else"})));
    assert!(denied(present(&auth, &caller("portal"), &other)));
    // `act` without `sub`, or not an object, or missing.
    for act in [json!({}), json!({"sub": 7})] {
        assert!(denied(present(
            &auth,
            &caller("portal"),
            &signed(&claims("alice", &act))
        )));
    }
    let mut no_act = claims("alice", &json!(null));
    no_act.as_object_mut().expect("object").remove("act");
    assert!(denied(present(&auth, &caller("portal"), &signed(&no_act))));
    assert!(denied(present(
        &auth,
        &caller("portal"),
        &signed(&claims("alice", &json!("portal")))
    )));
}

#[test]
fn assertion_must_verify() {
    let (auth, _) = server();
    let act = json!({"sub": "portal"});
    let good = claims("alice", &act);
    // Another key, issuer, audience; expired; longer than the 5 minute default lifetime.
    let forged = token(&rs_header("k1"), &good, rs256(rsa_other()));
    let mut wrong_iss = good.clone();
    wrong_iss["iss"] = json!("https://elsewhere.example.test/");
    let mut wrong_aud = good.clone();
    wrong_aud["aud"] = json!("krishiv");
    let now = unix_now();
    let mut expired = good.clone();
    expired["iat"] = json!(now - 600);
    expired["exp"] = json!(now - 300);
    let mut long = good.clone();
    long["exp"] = json!(now + 600);
    for t in [
        forged,
        signed(&wrong_iss),
        signed(&wrong_aud),
        signed(&expired),
        signed(&long),
        "not.a.jwt".to_owned(),
    ] {
        assert!(denied(present(&auth, &caller("portal"), &t)));
    }
    assert!(present(&auth, &caller("portal"), &signed(&good)).is_ok());
}

#[test]
fn assertion_does_not_lift_privilege_or_allowlist() {
    let (auth, _) = server();
    // A caller without `impersonate`, with an assertion naming it.
    let batch = signed(&claims("alice", &json!({"sub": "batch"})));
    assert!(denied(present(&auth, &caller("batch"), &batch)));
    // An end user outside the portal's allowlist (no listed group either).
    let mut bob = claims("bob", &json!({"sub": "portal"}));
    bob["groups"] = json!(["staff"]);
    assert!(denied(present(&auth, &caller("portal"), &signed(&bob))));
    // A caller already acting for someone.
    let acting = caller("portal");
    let mut acting = acting;
    acting.principal = acting.principal.with_delegator("gateway");
    let a = signed(&claims("alice", &json!({"sub": "portal"})));
    assert!(denied(present(&auth, &acting, &a)));
}

#[test]
fn assertion_naming_the_caller_is_no_impersonation() {
    let (auth, _) = server();
    let mut own = claims("portal", &json!({"sub": "portal"}));
    own["groups"] = json!([]);
    let (_, end) = present(&auth, &caller("portal"), &signed(&own)).expect("itself");
    assert_eq!(end.principal.name, "portal");
    assert_eq!(end.principal.delegated_by, None);
    assert_eq!(auth.impersonation_counts(), ImpersonationCount::default());
}

#[test]
fn unknown_assertion_key_waits_for_one_fetch() {
    // The first fetch has no usable key; the second brings the assertion's key.
    let source = Source::new(vec![
        Ok(jwks(&[rsa_jwk(rsa_other(), "old")])),
        Ok(jwks(&[rsa_jwk(rsa(), "k1")])),
    ]);
    let auth = server_with(&source);
    wait_until(|| auth.assertions().is_some_and(|a| a.jwks().key_count() == 1));
    let a = signed(&claims("alice", &json!({"sub": "portal"})));
    assert!(present(&auth, &caller("portal"), &a).is_ok());
    assert_eq!(source.fetches(), 2);
}

#[test]
fn assertions_off_are_refused() {
    let config = AuthConfig::from_settings([
        ("krishiv.auth.principals.portal.privileges", "impersonate"),
        ("krishiv.auth.principals.portal.impersonates", "alice"),
    ])
    .expect("valid");
    let auth = ServerAuth::new(&config).expect("builds");
    let a = signed(&claims("alice", &json!({"sub": "portal"})));
    assert!(denied(present(&auth, &caller("portal"), &a)));
    assert_eq!(auth.impersonation_counts().denied, 1);
}

#[test]
fn assertion_settings_are_checked_at_startup() {
    let refused = |pairs: Vec<(&str, &str)>| AuthConfig::from_settings(pairs).is_err();
    // On without an issuer; an issuer while off; an end-user role list under the assertion
    // prefix (roles come from krishiv.auth.impersonation.roles.*); no audience; http://.
    assert!(refused(vec![(
        "krishiv.auth.impersonation.signedAssertion",
        "true"
    )]));
    assert!(refused(vec![
        ("krishiv.auth.impersonation.assertion.issuer", ISSUER),
        ("krishiv.auth.impersonation.assertion.audience", AUDIENCE),
    ]));
    assert!(config(&[("krishiv.auth.impersonation.assertion.roles.admin", "dbas")]).is_err());
    assert!(refused(vec![
        ("krishiv.auth.impersonation.signedAssertion", "true"),
        ("krishiv.auth.impersonation.assertion.issuer", ISSUER),
    ]));
    assert!(
        config(&[(
            "krishiv.auth.impersonation.assertion.issuer",
            "http://portal.example.test/"
        )])
        .is_err()
    );
    assert!(config(&[("krishiv.auth.impersonation.signedAssertion", "yes")]).is_err());
    let c = config(&[("krishiv.auth.impersonation.assertion.maxLifetime", "10m")]).expect("valid");
    assert!(c.impersonation.signed_assertion);
    assert_eq!(
        c.impersonation.assertion.max_lifetime,
        std::time::Duration::from_mins(10)
    );
    // The OIDC settings stay their own.
    assert!(!c.oidc.is_on());
    assert_eq!(
        config(&[])
            .expect("valid")
            .impersonation
            .assertion
            .max_lifetime,
        std::time::Duration::from_mins(5)
    );
}

/// An assertion issuer and audience equal to the OIDC ones would make every assertion a login
/// token for the end user (review M4): refused at startup.
#[test]
fn assertion_config_equal_to_oidc_refused() {
    let same = config(&[
        ("krishiv.auth.oidc.issuer", ISSUER),
        ("krishiv.auth.oidc.audience", AUDIENCE),
    ]);
    assert!(same.is_err());
    let other_audience = config(&[
        ("krishiv.auth.oidc.issuer", ISSUER),
        ("krishiv.auth.oidc.audience", "krishiv"),
    ]);
    assert!(other_audience.is_ok());
}

/// The audit record of a signed impersonation names the assertion's `jti` but holds no token
/// text: neither the compact assertion nor its signature (review m6).
#[test]
fn audit_record_has_no_assertion_text() {
    use std::sync::{Arc, Mutex, PoisonError};
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
    let (auth, _) = server();
    let sink = Sink::default();
    let writer = sink.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_max_level(tracing::Level::TRACE)
        .with_writer(move || writer.clone())
        .finish();
    let good = signed(&claims("alice", &json!({"sub": "portal"})));
    let bad = signed(&claims("alice", &json!({"sub": "nobody"})));
    tracing::subscriber::with_default(subscriber, || {
        assert!(present(&auth, &caller("portal"), &good).is_ok());
        assert!(present(&auth, &caller("portal"), &bad).is_err());
    });
    let logged = String::from_utf8(
        sink.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone(),
    )
    .unwrap_or_default();
    for want in [
        "assertion_jti=\"assert-1\"",
        "jti=\"caller-jti\"",
        "reason_code=\"actor_mismatch\"",
    ] {
        assert!(logged.contains(want), "{want} in {logged}");
    }
    for t in [&good, &bad] {
        let signature = t.rsplit('.').next().unwrap_or_default();
        assert!(!logged.contains(t.as_str()), "{logged}");
        assert!(!logged.contains(signature), "{logged}");
    }
}
