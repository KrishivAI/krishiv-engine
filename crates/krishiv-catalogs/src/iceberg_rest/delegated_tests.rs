// SPDX-License-Identifier: Apache-2.0
//! Delegated catalog authentication (design P3-T067 §3.3, §8 Auth) against the fake catalog and a
//! scripted RFC 8693 token endpoint on the same server.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::secret::SecretString;
use url::Url;

use super::test_server::{Reply, Seen, Server, serve};
use super::{DelegationLimits, RestAuth, RestCaller, RestClient, RestConfig, SubjectToken};
use crate::location::RemoteLocationPolicy;

const CONFIG: &str = r#"{"defaults": {}, "overrides": {}}"#;
const EXCHANGE: &str = "urn:ietf:params:oauth:grant-type:token-exchange";

fn path(seen: &Seen) -> &str {
    seen.target.split('?').next().unwrap_or("")
}

fn form(seen: &Seen) -> Vec<(String, String)> {
    url::form_urlencoded::parse(seen.body.as_bytes())
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect()
}

fn field<'a>(form: &'a [(String, String)], name: &str) -> Option<&'a str> {
    form.iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.as_str())
}

/// A catalog and its IdP: `/token` issues `actor-token` for client credentials and
/// `x<n>-<audience>` for an exchange (`bad*` subject tokens are refused with an
/// `error_description` that repeats the token); the catalog answers any bearer except
/// `revoked-*` ones it was told to refuse.
fn idp_and_catalog(lifetime: u64) -> (Server, Arc<AtomicUsize>) {
    let exchanges = Arc::new(AtomicUsize::new(0));
    let n = Arc::clone(&exchanges);
    let server = serve(move |r| match path(r) {
        "/token" => {
            let f = form(r);
            match field(&f, "grant_type") {
                Some("client_credentials") => Reply::Json(
                    200,
                    r#"{"access_token": "actor-token", "token_type": "bearer", "expires_in": 3600}"#
                        .into(),
                ),
                Some(EXCHANGE) => {
                    let subject = field(&f, "subject_token").unwrap_or("");
                    if subject.starts_with("bad") {
                        return Reply::Json(
                            400,
                            format!(
                                r#"{{"error": "invalid_grant", "error_description": "token {subject} is bad"}}"#
                            ),
                        );
                    }
                    let i = n.fetch_add(1, Ordering::SeqCst);
                    let aud = field(&f, "audience").unwrap_or("");
                    Reply::Json(
                        200,
                        format!(
                            r#"{{"access_token": "x{i}-{aud}", "token_type": "Bearer", "expires_in": {lifetime}}}"#
                        ),
                    )
                }
                _ => Reply::Json(400, r#"{"error": "unsupported_grant_type"}"#.into()),
            }
        }
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ if r
            .header("authorization")
            .is_some_and(|a| a.contains("revoked")) =>
        {
            Reply::Json(401, "{}".into())
        }
        _ => Reply::Json(200, r#"{"namespaces": []}"#.into()),
    });
    (server, exchanges)
}

fn delegated(
    server: &Server,
    audience: &str,
    forward: bool,
    limits: DelegationLimits,
) -> RestClient {
    let mut config = RestConfig::new(&server.base).unwrap();
    config.allow_http = true;
    config.auth = RestAuth::Delegated {
        forward,
        audience: audience.to_owned(),
        token_endpoint: (!forward).then(|| Url::parse(&format!("{}/token", server.base)).unwrap()),
        scope: Some("lake.read".to_owned()),
        actor: (!forward).then(|| ("krishiv".to_owned(), SecretString::new("actor-secret"))),
        limits,
    };
    RestClient::new(audience, config, Arc::new(RemoteLocationPolicy::default())).unwrap()
}

fn exchange_client(server: &Server) -> RestClient {
    delegated(server, "lake", false, DelegationLimits::default())
}

fn subject(token: &str, claims: u8, audiences: &[&str], ttl: Duration) -> SubjectToken {
    SubjectToken::new(
        SecretString::new(token),
        false,
        audiences.iter().map(|a| (*a).to_owned()).collect(),
        [claims; 16],
        SystemTime::now() + ttl,
    )
}

const HOUR: Duration = Duration::from_secs(3600);

fn user(principal: &str, subject: Option<SubjectToken>) -> RestCaller {
    RestCaller::user(
        format!("lake/{principal}"),
        principal.to_owned(),
        None,
        "oidc",
        subject,
    )
}

fn list(client: &RestClient, caller: &RestCaller) -> Result<(), KrishivError> {
    client.list_namespaces_as(caller, &[]).map(drop)
}

/// The bearer of every catalog request (not the token endpoint's).
fn bearers(server: &Server) -> Vec<String> {
    server
        .seen()
        .iter()
        .filter(|s| path(s) != "/token" && path(s) != "/v1/config")
        .map(|s| s.header("authorization").unwrap_or("").to_owned())
        .collect()
}

fn exchange_forms(server: &Server) -> Vec<Vec<(String, String)>> {
    server
        .seen()
        .iter()
        .filter(|s| path(s) == "/token")
        .map(form)
        .filter(|f| field(f, "grant_type") == Some(EXCHANGE))
        .collect()
}

#[test]
fn delegated_exchange_sends_subject_actor_audience() {
    let (server, _) = idp_and_catalog(3600);
    let client = exchange_client(&server);
    list(
        &client,
        &user("alice", Some(subject("alice-jwt", 1, &["krishiv"], HOUR))),
    )
    .unwrap();
    let forms = exchange_forms(&server);
    assert_eq!(forms.len(), 1);
    let f = &forms[0];
    assert_eq!(field(f, "subject_token"), Some("alice-jwt"));
    assert_eq!(
        field(f, "subject_token_type"),
        Some("urn:ietf:params:oauth:token-type:jwt")
    );
    // The actor is the server's own client-credentials token for this catalog.
    assert_eq!(field(f, "actor_token"), Some("actor-token"));
    assert_eq!(
        field(f, "actor_token_type"),
        Some("urn:ietf:params:oauth:token-type:access_token")
    );
    assert_eq!(field(f, "audience"), Some("lake"));
    assert_eq!(field(f, "scope"), Some("lake.read"));
    // The catalog sees only the exchanged token, never the user's own.
    assert!(!bearers(&server).is_empty());
    for b in bearers(&server) {
        assert_eq!(b, "Bearer x0-lake");
    }
    // `/v1/config` goes with the user's exchanged token too: the server's actor token is for the
    // IdP only and never reaches the catalog (review 58b1-r2 N1).
    assert_eq!(config_bearers(&server), ["Bearer x0-lake"]);
}

fn config_bearers(server: &Server) -> Vec<String> {
    server
        .seen()
        .iter()
        .filter(|s| path(s) == "/v1/config")
        .map(|s| s.header("authorization").unwrap_or("").to_owned())
        .collect()
}

/// Without an actor client a delegated catalog has no identity of its own: `/v1/config` is
/// fetched and kept per caller identity, each with that caller's token (T067 §3.5).
#[test]
fn config_without_an_actor_is_kept_per_identity() {
    let (server, _) = idp_and_catalog(3600);
    let client = delegated(&server, "lake", true, DelegationLimits::default());
    let alice = user("alice", Some(subject("alice-tok", 1, &["lake"], HOUR)));
    let bob = user("bob", Some(subject("bob-tok", 1, &["lake"], HOUR)));
    list(&client, &alice).unwrap();
    list(&client, &bob).unwrap();
    list(&client, &alice).unwrap();
    assert_eq!(
        config_bearers(&server),
        ["Bearer alice-tok", "Bearer bob-tok"]
    );
}

#[test]
fn exchange_without_an_actor_sends_none() {
    let (server, _) = idp_and_catalog(3600);
    let mut config = RestConfig::new(&server.base).unwrap();
    config.allow_http = true;
    config.auth = RestAuth::Delegated {
        forward: false,
        audience: "lake".into(),
        token_endpoint: Some(Url::parse(&format!("{}/token", server.base)).unwrap()),
        scope: None,
        actor: None,
        limits: DelegationLimits::default(),
    };
    let client =
        RestClient::new("lake", config, Arc::new(RemoteLocationPolicy::default())).unwrap();
    list(
        &client,
        &user("alice", Some(subject("alice-jwt", 1, &[], HOUR))),
    )
    .unwrap();
    let f = &exchange_forms(&server)[0];
    assert_eq!(field(f, "actor_token"), None);
    assert_eq!(field(f, "scope"), None);
}

#[test]
fn exchanged_token_never_sent_to_other_catalog() {
    let (idp, _) = idp_and_catalog(3600);
    let (cat_b, _) = idp_and_catalog(3600);
    // Catalog `a` is the IdP's own server; catalog `b` is another host whose exchanges go to
    // the same IdP.
    let a = delegated(&idp, "a", false, DelegationLimits::default());
    let mut config = RestConfig::new(&cat_b.base).unwrap();
    config.allow_http = true;
    config.auth = RestAuth::Delegated {
        forward: false,
        audience: "b".into(),
        token_endpoint: Some(Url::parse(&format!("{}/token", idp.base)).unwrap()),
        scope: None,
        actor: None,
        limits: DelegationLimits::default(),
    };
    let b = RestClient::new("b", config, Arc::new(RemoteLocationPolicy::default())).unwrap();
    let alice = user("alice", Some(subject("alice-jwt", 1, &["krishiv"], HOUR)));
    list(&a, &alice).unwrap();
    list(&b, &alice).unwrap();
    list(&a, &alice).unwrap();
    // Two exchanges, one per catalog audience; `b` only ever saw its own.
    let audiences: Vec<_> = exchange_forms(&idp)
        .iter()
        .map(|f| field(f, "audience").unwrap_or("").to_owned())
        .collect();
    assert_eq!(audiences, ["a", "b"]);
    let seen_b = bearers(&cat_b);
    assert!(!seen_b.is_empty());
    assert!(seen_b.iter().all(|t| t.ends_with("-b")), "{seen_b:?}");
}

#[test]
fn forward_refused_without_catalog_audience() {
    let (server, _) = idp_and_catalog(3600);
    let client = delegated(&server, "lake", true, DelegationLimits::default());
    // A token issued for Krishiv only is never replayed (the confused deputy).
    let e = list(
        &client,
        &user(
            "alice",
            Some(subject("krishiv-only", 1, &["krishiv"], HOUR)),
        ),
    )
    .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivCatalogDelegationUnavailable
    );
    assert!(!e.to_string().contains("krishiv-only"), "{e}");
    // A signed assertion is never forwarded, whatever audience it claims.
    let mut assertion = subject("signed-assertion", 1, &["krishiv", "lake"], HOUR);
    assertion.assertion = true;
    let e = list(&client, &user("alice", Some(assertion))).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivCatalogDelegationUnavailable
    );
    assert!(e.to_string().contains("signed assertion"), "{e}");
    assert!(server.seen().is_empty(), "nothing was sent");
    // A token whose audience names the catalog goes as it is.
    list(
        &client,
        &user(
            "alice",
            Some(subject("multi-aud", 1, &["krishiv", "lake"], HOUR)),
        ),
    )
    .unwrap();
    assert!(bearers(&server).iter().all(|b| b == "Bearer multi-aud"));
    assert!(exchange_forms(&server).is_empty());
}

#[test]
fn delegated_refused_for_static_token_session() {
    let (server, _) = idp_and_catalog(3600);
    for client in [
        exchange_client(&server),
        delegated(&server, "lake", true, DelegationLimits::default()),
    ] {
        // No user token: never a fallback to the server's identity.
        let caller = RestCaller::user("lake/svc".into(), "svc".into(), None, "token", None);
        let e = list(&client, &caller).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivCatalogDelegationUnavailable
        );
        assert_eq!(e.error_class().sql_state(), Some("42501"));
        assert!(e.to_string().contains("authenticated with `token`"), "{e}");
        // The server identity itself is refused too.
        let e = list(&client, &RestCaller::server()).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivCatalogDelegationUnavailable
        );
    }
    // Nothing was sent at all: no actor token, no exchange, no catalog call as anyone.
    assert!(server.seen().is_empty(), "{:?}", server.seen());
}

#[test]
fn delegated_signed_assertion_is_subject_token() {
    let (server, _) = idp_and_catalog(3600);
    let client = exchange_client(&server);
    let mut assertion = subject("signed-assertion-jwt", 1, &["krishiv"], HOUR);
    assertion.assertion = true;
    list(&client, &user("alice", Some(assertion))).unwrap();
    let f = &exchange_forms(&server)[0];
    assert_eq!(field(f, "subject_token"), Some("signed-assertion-jwt"));
    assert_eq!(field(f, "actor_token"), Some("actor-token"));
}

#[test]
fn expired_subject_token_fails_refresh() {
    let (server, _) = idp_and_catalog(3600);
    let client = exchange_client(&server);
    list(
        &client,
        &user("alice", Some(subject("alice-jwt", 1, &[], HOUR))),
    )
    .unwrap();
    // The subject token has expired: the exchanged token issued before is still used.
    let expired = SubjectToken::new(
        SecretString::new("alice-jwt"),
        false,
        Vec::new(),
        [1; 16],
        SystemTime::now() - Duration::from_secs(1),
    );
    let alice = user("alice", Some(expired));
    list(&client, &alice).unwrap();
    assert_eq!(exchange_forms(&server).len(), 1);
    // A new exchange is needed (the catalog refuses the exchanged token): it fails and says so.
    let (revoking, _) = idp_and_catalog(3600);
    let mut config = RestConfig::new(&revoking.base).unwrap();
    config.allow_http = true;
    config.auth = RestAuth::Delegated {
        forward: false,
        audience: "revoked".into(),
        token_endpoint: Some(Url::parse(&format!("{}/token", revoking.base)).unwrap()),
        scope: None,
        actor: None,
        limits: DelegationLimits::default(),
    };
    let client =
        RestClient::new("lake", config, Arc::new(RemoteLocationPolicy::default())).unwrap();
    let e = list(&client, &alice).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivCatalogDelegationUnavailable
    );
    let text = e.to_string();
    assert!(
        text.contains("expired") && text.contains("reconnect"),
        "{text}"
    );
    assert!(
        exchange_forms(&revoking).is_empty(),
        "no exchange with an expired token"
    );
}

#[test]
fn a_catalog_401_drops_the_exchanged_token_and_exchanges_again() {
    let (server, exchanges) = idp_and_catalog(3600);
    // Audience `revoked`: every exchanged token is `x<n>-revoked`, which the catalog refuses.
    let client = delegated(&server, "revoked", false, DelegationLimits::default());
    let e = list(
        &client,
        &user("alice", Some(subject("alice-jwt", 1, &[], HOUR))),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivCatalogFailure);
    assert_eq!(
        exchanges.load(Ordering::SeqCst),
        2,
        "one retry, one new exchange"
    );
}

#[test]
fn exchange_limit_per_subject_token() {
    let (server, _) = idp_and_catalog(3600);
    let limits = DelegationLimits {
        max_exchanges_per_minute: 2,
        max_distinct_exchanges_per_minute: 20,
    };
    let client = delegated(&server, "revoked", false, limits);
    // A bad token is refused by the IdP once, then from memory without reaching it.
    let bad = user("alice", Some(subject("bad-jwt", 1, &[], HOUR)));
    let e = list(&client, &bad).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivCatalogDelegationUnavailable
    );
    let text = e.to_string();
    assert!(text.contains("invalid_grant"), "{text}");
    assert!(
        !text.contains("bad-jwt") && !text.contains("is bad"),
        "{text}"
    );
    let e = list(&client, &bad).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivCatalogDelegationUnavailable
    );
    let idp_calls = |s: &Server| {
        s.seen()
            .iter()
            .filter(|r| path(r) == "/token" && form(r).iter().any(|(_, v)| v == "bad-jwt"))
            .count()
    };
    assert_eq!(idp_calls(&server), 1);
    // The same principal's good token (same claims) is not locked out: it exchanges (count 2).
    // The catalog refuses the exchanged token, the retry needs a third exchange: the limit.
    let good = user("alice", Some(subject("good-jwt", 1, &[], HOUR)));
    let e = list(&client, &good).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivCatalogLimitExceeded);
    assert!(
        e.to_string()
            .contains("krishiv.catalog.delegation.maxExchangesPerMinute"),
        "{e}"
    );
    assert_eq!(exchange_forms(&server).len(), 2);
}

#[test]
fn exchange_reused_across_fresh_subject_tokens() {
    let (server, exchanges) = idp_and_catalog(3600);
    let client = exchange_client(&server);
    // A front end that mints a fresh token per call, same claims: one exchange.
    for i in 0..5 {
        let token = format!("fresh-{i}");
        list(&client, &user("alice", Some(subject(&token, 7, &[], HOUR)))).unwrap();
    }
    assert_eq!(exchanges.load(Ordering::SeqCst), 1);
}

#[test]
fn exchange_not_reused_for_narrower_claims() {
    let (server, exchanges) = idp_and_catalog(3600);
    let client = exchange_client(&server);
    list(&client, &user("alice", Some(subject("wide", 1, &[], HOUR)))).unwrap();
    // A down-scoped token of the same principal gets its own exchange.
    list(
        &client,
        &user("alice", Some(subject("narrow", 2, &[], HOUR))),
    )
    .unwrap();
    assert_eq!(exchanges.load(Ordering::SeqCst), 2);
    let b = bearers(&server);
    assert_eq!(b.last().map(String::as_str), Some("Bearer x1-lake"));
}

#[test]
fn distinct_exchange_ceiling_per_principal() {
    let (server, _) = idp_and_catalog(3600);
    let limits = DelegationLimits {
        max_exchanges_per_minute: 60,
        max_distinct_exchanges_per_minute: 2,
    };
    let client = delegated(&server, "lake", false, limits);
    list(&client, &user("alice", Some(subject("t1", 1, &[], HOUR)))).unwrap();
    list(&client, &user("alice", Some(subject("t2", 2, &[], HOUR)))).unwrap();
    let e = list(&client, &user("alice", Some(subject("t3", 3, &[], HOUR)))).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivCatalogLimitExceeded);
    assert!(
        e.to_string().contains("maxDistinctExchangesPerMinute"),
        "{e}"
    );
    // Another principal has its own ceiling.
    list(&client, &user("bob", Some(subject("t3", 3, &[], HOUR)))).unwrap();
}

#[test]
fn an_exchanged_token_is_never_shown() {
    let (server, _) = idp_and_catalog(3600);
    let client = exchange_client(&server);
    let alice = user(
        "alice",
        Some(subject("planted-subject", 1, &["krishiv"], HOUR)),
    );
    list(&client, &alice).unwrap();
    for text in [format!("{client:?}"), format!("{alice:?}")] {
        assert!(
            !text.contains("x0-lake") && !text.contains("planted-subject"),
            "{text}"
        );
    }
}

/// An IdP that refuses the server's own client: when `actor_refused`, its client-credentials
/// grant; otherwise the exchange itself, with `invalid_client`.
fn client_refusing_idp(actor_refused: bool) -> Server {
    serve(move |r| match path(r) {
        "/token" => {
            let f = form(r);
            match field(&f, "grant_type") {
                Some("client_credentials") if actor_refused => {
                    Reply::Json(401, r#"{"error": "invalid_client"}"#.into())
                }
                Some("client_credentials") => Reply::Json(
                    200,
                    r#"{"access_token": "actor-token", "token_type": "bearer", "expires_in": 0}"#
                        .into(),
                ),
                _ => Reply::Json(401, r#"{"error": "invalid_client"}"#.into()),
            }
        }
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Json(200, r#"{"namespaces": []}"#.into()),
    })
}

/// Review 58b1-r1 m3/m4: a refusal of the server's own client (the actor token, or
/// `invalid_client` on the exchange) is the catalog's failure: not the user's "reconnect", not
/// remembered against the user's token, and an actor that cannot be fetched spends no budget.
#[test]
fn actor_refusal_is_the_catalogs_failure() {
    let alice = user("alice", Some(subject("alice-jwt", 1, &[], HOUR)));
    // The exchange is refused with `invalid_client`.
    let server = client_refusing_idp(false);
    let client = delegated(&server, "lake", false, DelegationLimits::default());
    for _ in 0..2 {
        let e = list(&client, &alice).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivCatalogFailure, "{e}");
        assert!(!e.to_string().contains("reconnect"), "{e}");
    }
    // Not remembered against the user's token: both calls reached the IdP.
    assert_eq!(exchange_forms(&server).len(), 2);

    // The actor token is refused: no exchange is sent, no budget is spent (with a limit of one,
    // a charge would turn the second call into `LIMIT_EXCEEDED`), and the refusal is remembered
    // for 30 s, so user calls send the IdP one client-credentials request, not one each
    // (review 58b1-r2 N2).
    let server = client_refusing_idp(true);
    let one = DelegationLimits {
        max_exchanges_per_minute: 1,
        max_distinct_exchanges_per_minute: 20,
    };
    let client = delegated(&server, "lake", false, one);
    for i in 0..3 {
        let who = user("alice", Some(subject(&format!("t{i}"), i, &[], HOUR)));
        let e = list(&client, &who).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivCatalogFailure, "{e}");
    }
    assert!(exchange_forms(&server).is_empty());
    let grants = server
        .seen()
        .iter()
        .filter(|s| field(&form(s), "grant_type") == Some("client_credentials"))
        .count();
    assert_eq!(grants, 1, "a failing actor client is backed off");
    assert!(
        server.seen().iter().all(|s| path(s) == "/token"),
        "nothing reached the catalog"
    );
}

/// Review 58b1-r2 N1: a catalog that accepts only exchanged tokens (its audience), as Polaris or
/// Lakekeeper validating `aud` do. The server's actor token never reaches the catalog host.
#[test]
fn actor_token_never_sent_to_catalog() {
    let server = serve(move |r| match path(r) {
        "/token" => {
            let f = form(r);
            match field(&f, "grant_type") {
                Some("client_credentials") => Reply::Json(
                    200,
                    r#"{"access_token": "actor-token", "token_type": "bearer", "expires_in": 3600}"#
                        .into(),
                ),
                _ => Reply::Json(
                    200,
                    r#"{"access_token": "x-user", "token_type": "bearer", "expires_in": 3600}"#
                        .into(),
                ),
            }
        }
        _ if r.header("authorization") == Some("Bearer x-user") => {
            if path(r) == "/v1/config" {
                Reply::Json(200, CONFIG.into())
            } else {
                Reply::Json(200, r#"{"namespaces": []}"#.into())
            }
        }
        _ => Reply::Json(401, "{}".into()),
    });
    let client = delegated(&server, "lake", false, DelegationLimits::default());
    list(
        &client,
        &user("alice", Some(subject("alice-jwt", 1, &[], HOUR))),
    )
    .unwrap();
    let actor_at_catalog = server
        .seen()
        .iter()
        .filter(|s| path(s) != "/token")
        .any(|s| s.header("authorization") == Some("Bearer actor-token"));
    assert!(!actor_at_catalog, "{:?}", server.seen());
    // Only the exchange carried it.
    let f = &exchange_forms(&server)[0];
    assert_eq!(field(f, "actor_token"), Some("actor-token"));
}

fn client_credentials_requests(server: &Server) -> usize {
    server
        .seen()
        .iter()
        .filter(|s| field(&form(s), "grant_type") == Some("client_credentials"))
        .count()
}

/// Review 58b1-r3 R1: an actor fetch that fails transiently (a 5xx) is not remembered: the
/// next call asks the IdP again (only an outright refusal is backed off for 30 s).
#[test]
fn a_transient_actor_failure_is_not_backed_off() {
    let server = serve(|r| match path(r) {
        "/token" => Reply::Json(503, "{}".into()),
        _ => Reply::Json(200, CONFIG.into()),
    });
    let client = delegated(&server, "lake", false, DelegationLimits::default());
    for i in 0..3 {
        let who = user("alice", Some(subject(&format!("t{i}"), i, &[], HOUR)));
        let e = list(&client, &who).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivCatalogFailure, "{e}");
    }
    assert_eq!(client_credentials_requests(&server), 3);
    assert!(exchange_forms(&server).is_empty());
}

/// Review 58b1-r3 R2: `invalid_client` on an exchange drops the cached actor token, so the next
/// exchange fetches a new one instead of reusing a revoked token until its own expiry.
#[test]
fn a_refused_actor_token_is_dropped_and_fetched_again() {
    let grants = Arc::new(AtomicUsize::new(0));
    let n = Arc::clone(&grants);
    let server = serve(move |r| match path(r) {
        "/token" => {
            let f = form(r);
            match field(&f, "grant_type") {
                Some("client_credentials") => {
                    let i = n.fetch_add(1, Ordering::SeqCst);
                    Reply::Json(
                        200,
                        format!(
                            r#"{{"access_token": "actor-{i}", "token_type": "bearer", "expires_in": 3600}}"#
                        ),
                    )
                }
                // The IdP revoked the first actor token.
                _ if field(&f, "actor_token") == Some("actor-0") => {
                    Reply::Json(401, r#"{"error": "invalid_client"}"#.into())
                }
                _ => Reply::Json(
                    200,
                    r#"{"access_token": "x-user", "token_type": "bearer", "expires_in": 3600}"#
                        .into(),
                ),
            }
        }
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Json(200, r#"{"namespaces": []}"#.into()),
    });
    let client = delegated(&server, "lake", false, DelegationLimits::default());
    let alice = user("alice", Some(subject("alice-jwt", 1, &[], HOUR)));
    let e = list(&client, &alice).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivCatalogFailure, "{e}");
    list(&client, &alice).unwrap();
    assert_eq!(
        grants.load(Ordering::SeqCst),
        2,
        "a new actor token was fetched"
    );
    let actors: Vec<_> = exchange_forms(&server)
        .iter()
        .map(|f| field(f, "actor_token").unwrap_or("").to_owned())
        .collect();
    assert_eq!(actors, ["actor-0", "actor-1"]);
}
