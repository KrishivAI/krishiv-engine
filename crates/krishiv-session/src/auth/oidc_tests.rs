// SPDX-License-Identifier: Apache-2.0
//! OIDC tests (design §4: `oidc_basic_validation`, `oidc_algorithm_pinning`,
//! `oidc_unknown_kid_refresh_limit`, `oidc_idp_down_uses_cache`). Keys are generated per run
//! with aws-lc-rs and served from an in-memory JWK set source.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use aws_lc_rs::rand::SystemRandom;
use aws_lc_rs::signature::{
    ECDSA_P256_SHA256_FIXED_SIGNING, EcdsaKeyPair, KeyPair, RSA_PKCS1_SHA256, RsaKeyPair,
};
use futures::executor::block_on;
use serde_json::{Value, json};

use super::jose::{self, Alg};
use super::oidc::{JwksCache, KeySource, Lookup, ON_DEMAND_EVERY, unix_now};
use super::{
    AuthConfig, AuthFailure, Authenticated, Authenticator, Credentials, Method, Role, ServerAuth,
    Surface,
};

const ISSUER: &str = "https://idp.example.test/";
const AUDIENCE: &str = "krishiv";

fn rsa() -> &'static RsaKeyPair {
    static KEY: OnceLock<RsaKeyPair> = OnceLock::new();
    KEY.get_or_init(|| {
        RsaKeyPair::generate(aws_lc_rs::rsa::KeySize::Rsa2048).expect("generate an RSA key")
    })
}

fn rsa_other() -> &'static RsaKeyPair {
    static KEY: OnceLock<RsaKeyPair> = OnceLock::new();
    KEY.get_or_init(|| {
        RsaKeyPair::generate(aws_lc_rs::rsa::KeySize::Rsa2048).expect("generate an RSA key")
    })
}

fn ec() -> &'static EcdsaKeyPair {
    static KEY: OnceLock<EcdsaKeyPair> = OnceLock::new();
    KEY.get_or_init(|| {
        EcdsaKeyPair::generate(&ECDSA_P256_SHA256_FIXED_SIGNING).expect("generate a P-256 key")
    })
}

fn b64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let mut acc = 0u32;
        for (i, b) in chunk.iter().enumerate() {
            acc |= u32::from(*b) << (16 - 8 * i);
        }
        for i in 0..=chunk.len() {
            out.push(char::from(A[((acc >> (18 - 6 * i)) & 63) as usize]));
        }
    }
    out
}

/// `(n, e)` of an RSA public key's DER (`RSAPublicKey ::= SEQUENCE { n INTEGER, e INTEGER }`).
fn rsa_components(der: &[u8]) -> (Vec<u8>, Vec<u8>) {
    fn tlv(d: &[u8]) -> (&[u8], &[u8]) {
        let (len, head) = match d[1] {
            l if l < 0x80 => (usize::from(l), 2),
            0x81 => (usize::from(d[2]), 3),
            0x82 => (usize::from(d[2]) << 8 | usize::from(d[3]), 4),
            _ => panic!("unexpected length"),
        };
        (&d[head..head + len], &d[head + len..])
    }
    let (seq, _) = tlv(der);
    let (n, rest) = tlv(seq);
    let (e, _) = tlv(rest);
    let strip = |v: &[u8]| {
        v.iter()
            .skip_while(|b| **b == 0)
            .copied()
            .collect::<Vec<_>>()
    };
    (strip(n), strip(e))
}

fn rsa_jwk(key: &RsaKeyPair, kid: &str) -> Value {
    let (n, e) = rsa_components(key.public_key().as_ref());
    json!({"kty": "RSA", "kid": kid, "use": "sig", "alg": "RS256", "n": b64(&n), "e": b64(&e)})
}

fn ec_jwk(key: &EcdsaKeyPair, kid: &str) -> Value {
    let point = key.public_key().as_ref();
    json!({"kty": "EC", "kid": kid, "crv": "P-256", "x": b64(&point[1..33]), "y": b64(&point[33..65])})
}

fn jwks(keys: &[Value]) -> Vec<u8> {
    serde_json::to_vec(&json!({ "keys": keys })).expect("serialize")
}

/// A token with `header` and `claims`, signed by `sign` over the signing input.
fn token(header: &Value, claims: &Value, sign: impl Fn(&[u8]) -> Vec<u8>) -> String {
    let input = format!(
        "{}.{}",
        b64(&serde_json::to_vec(header).expect("header")),
        b64(&serde_json::to_vec(claims).expect("claims"))
    );
    let sig = sign(input.as_bytes());
    format!("{input}.{}", b64(&sig))
}

fn rs256(key: &'static RsaKeyPair) -> impl Fn(&[u8]) -> Vec<u8> {
    move |msg| {
        let mut sig = vec![0; key.public_modulus_len()];
        key.sign(&RSA_PKCS1_SHA256, &SystemRandom::new(), msg, &mut sig)
            .expect("sign");
        sig
    }
}

fn es256(msg: &[u8]) -> Vec<u8> {
    ec().sign(&SystemRandom::new(), msg)
        .expect("sign")
        .as_ref()
        .to_vec()
}

fn claims() -> Value {
    let now = unix_now();
    json!({
        "iss": ISSUER, "aud": [AUDIENCE, "other"], "sub": "alice",
        "groups": ["analysts", "staff"], "exp": now + 300, "nbf": now - 10,
    })
}

fn rs_header(kid: &str) -> Value {
    json!({"alg": "RS256", "typ": "JWT", "kid": kid})
}

/// The answers of a [`Source`], in order.
type Answers = Arc<Mutex<Vec<Result<Vec<u8>, String>>>>;

/// An in-memory JWK set source: each fetch takes the next answer (the last repeats).
#[derive(Clone)]
struct Source {
    answers: Answers,
    fetches: Arc<AtomicUsize>,
}

impl Source {
    fn new(answers: Vec<Result<Vec<u8>, String>>) -> Self {
        Self {
            answers: Arc::new(Mutex::new(answers)),
            fetches: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn fetches(&self) -> usize {
        self.fetches.load(Ordering::SeqCst)
    }
}

impl KeySource for Source {
    fn fetch(&mut self) -> Result<Vec<u8>, String> {
        self.fetches.fetch_add(1, Ordering::SeqCst);
        let mut answers = self
            .answers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if answers.len() > 1 {
            answers.remove(0)
        } else {
            answers[0].clone()
        }
    }
}

fn settings(extra: &[(&str, &str)]) -> AuthConfig {
    let mut all = vec![
        ("krishiv.auth.oidc.issuer", ISSUER),
        ("krishiv.auth.oidc.audience", AUDIENCE),
        ("krishiv.auth.oidc.algorithms", "RS256, ES256"),
        ("krishiv.auth.oidc.roles.operator", "analysts"),
        ("krishiv.auth.oidc.roles.viewer", "staff"),
    ];
    all.extend_from_slice(extra);
    AuthConfig::from_settings(all).expect("settings")
}

fn server(source: &Source, extra: &[(&str, &str)]) -> ServerAuth {
    let auth = ServerAuth::with_key_source(&settings(extra), Box::new(source.clone()))
        .expect("server auth");
    // Wait for the first fetch to land: keys installed, or the fetch done and failed.
    let keys_expected = matches!(
        source.answers.lock().map(|a| a.first().map(Result::is_ok)),
        Ok(Some(true))
    );
    let oidc = auth.oidc().expect("oidc on");
    if keys_expected {
        wait_until(|| oidc.jwks().key_count() > 0);
    } else {
        wait_until(|| source.fetches() >= 1);
    }
    auth
}

fn wait_until(mut done: impl FnMut() -> bool) {
    let start = Instant::now();
    while !done() {
        assert!(start.elapsed() < Duration::from_secs(10), "timed out");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn peer() -> SocketAddr {
    SocketAddr::from(([10, 0, 0, 7], 40_000))
}

fn auth(server: &ServerAuth, bearer: &str) -> Result<Authenticated, AuthFailure> {
    let creds = Credentials::new(peer(), Surface::Connect).with_bearer(bearer);
    block_on(server.authenticate(&creds))
}

#[test]
fn oidc_basic_validation() {
    let source = Source::new(vec![Ok(jwks(&[rsa_jwk(rsa(), "k1"), ec_jwk(ec(), "e1")]))]);
    let server = server(&source, &[]);
    let ok = auth(&server, &token(&rs_header("k1"), &claims(), rs256(rsa()))).expect("valid");
    assert_eq!(ok.method, Method::Oidc);
    assert_eq!(ok.principal.name, "alice");
    assert_eq!(ok.principal.groups, ["analysts", "staff"]);
    // The highest role any group has.
    assert_eq!(ok.role, Some(Role::Operator));

    let es = token(&json!({"alg": "ES256", "kid": "e1"}), &claims(), es256);
    assert_eq!(auth(&server, &es).expect("ES256").principal.name, "alice");

    // A single-string audience and groups claim; no group in any list means no role.
    let mut c = claims();
    c["aud"] = json!(AUDIENCE);
    c["groups"] = json!("nobody");
    let ok = auth(&server, &token(&rs_header("k1"), &c, rs256(rsa()))).expect("valid");
    assert_eq!(
        (ok.role, ok.principal.groups.as_slice()),
        (None, &["nobody".to_owned()][..])
    );
}

#[test]
fn oidc_claims_are_checked() {
    let source = Source::new(vec![Ok(jwks(&[rsa_jwk(rsa(), "k1")]))]);
    let server = server(
        &source,
        &[("krishiv.auth.failureThrottle.maxFailures", "0")],
    );
    let now = unix_now();
    let with = |name: &str, v: Value| {
        let mut c = claims();
        if v.is_null() {
            c.as_object_mut().expect("object").remove(name);
        } else {
            c[name] = v;
        }
        auth(&server, &token(&rs_header("k1"), &c, rs256(rsa())))
    };
    assert_eq!(
        with("exp", json!(now - 61)).unwrap_err(),
        AuthFailure::Expired
    );
    // Within the 60 s skew.
    assert!(with("exp", json!(now - 30)).is_ok());
    assert_eq!(with("exp", Value::Null).unwrap_err(), AuthFailure::Invalid);
    assert_eq!(
        with("exp", json!("tomorrow")).unwrap_err(),
        AuthFailure::Invalid
    );
    assert_eq!(
        with("nbf", json!(now + 120)).unwrap_err(),
        AuthFailure::Invalid
    );
    assert_eq!(
        with("iss", json!("https://idp.example.test")).unwrap_err(),
        AuthFailure::Invalid
    );
    assert_eq!(
        with("aud", json!(["other"])).unwrap_err(),
        AuthFailure::Invalid
    );
    assert_eq!(with("aud", Value::Null).unwrap_err(), AuthFailure::Invalid);
    assert_eq!(with("sub", json!("")).unwrap_err(), AuthFailure::Invalid);
    assert_eq!(
        with("sub", json!("al\u{202E}ice")).unwrap_err(),
        AuthFailure::Invalid
    );
    assert_eq!(with("sub", Value::Null).unwrap_err(), AuthFailure::Invalid);
    assert_eq!(
        with("groups", json!([1])).unwrap_err(),
        AuthFailure::Invalid
    );
    assert!(with("groups", Value::Null).is_ok_and(|a| a.role.is_none()));
    // A signature by another key with the same kid.
    let forged = token(&rs_header("k1"), &claims(), rs256(rsa_other()));
    assert_eq!(auth(&server, &forged).unwrap_err(), AuthFailure::Invalid);
    // A tampered payload.
    let good = token(&rs_header("k1"), &claims(), rs256(rsa()));
    let parts: Vec<&str> = good.split('.').collect();
    let mut c = claims();
    c["sub"] = json!("mallory");
    let tampered = format!(
        "{}.{}.{}",
        parts[0],
        b64(&serde_json::to_vec(&c).expect("json")),
        parts[2]
    );
    assert_eq!(auth(&server, &tampered).unwrap_err(), AuthFailure::Invalid);
}

#[test]
fn oidc_algorithm_pinning() {
    let source = Source::new(vec![Ok(jwks(&[rsa_jwk(rsa(), "k1"), ec_jwk(ec(), "e1")]))]);
    let server = server(
        &source,
        &[("krishiv.auth.failureThrottle.maxFailures", "0")],
    );
    // alg none, unsigned and with a junk signature.
    let none = token(&json!({"alg": "none", "kid": "k1"}), &claims(), |_| {
        b"x".to_vec()
    });
    assert_eq!(auth(&server, &none).unwrap_err(), AuthFailure::Invalid);
    let unsigned = format!("{}.", none.rsplit_once('.').expect("dot").0);
    assert_eq!(auth(&server, &unsigned).unwrap_err(), AuthFailure::Invalid);
    // HS256 with the RSA public key as the HMAC secret (the classic confusion attack).
    let public = rsa().public_key().as_ref().to_vec();
    let hs = token(&json!({"alg": "HS256", "kid": "k1"}), &claims(), |msg| {
        aws_lc_rs::hmac::sign(
            &aws_lc_rs::hmac::Key::new(aws_lc_rs::hmac::HMAC_SHA256, &public),
            msg,
        )
        .as_ref()
        .to_vec()
    });
    assert_eq!(auth(&server, &hs).unwrap_err(), AuthFailure::Invalid);
    // ES256 naming the RSA key, and RS256 naming the EC key.
    let es_on_rsa = token(&json!({"alg": "ES256", "kid": "k1"}), &claims(), es256);
    assert_eq!(auth(&server, &es_on_rsa).unwrap_err(), AuthFailure::Invalid);
    let rs_on_ec = token(&rs_header("e1"), &claims(), rs256(rsa()));
    assert_eq!(auth(&server, &rs_on_ec).unwrap_err(), AuthFailure::Invalid);
    // A verifiable algorithm outside the allowlist.
    let rs_only = Source::new(vec![Ok(jwks(&[ec_jwk(ec(), "e1")]))]);
    let strict = self::server(&rs_only, &[("krishiv.auth.oidc.algorithms", "RS256")]);
    let es = token(&json!({"alg": "ES256", "kid": "e1"}), &claims(), es256);
    assert_eq!(auth(&strict, &es).unwrap_err(), AuthFailure::Invalid);
    // The header's own key members are never used.
    let (n, e) = rsa_components(rsa_other().public_key().as_ref());
    let jwk_header = json!({"alg": "RS256", "jwk": {"kty": "RSA", "n": b64(&n), "e": b64(&e)}});
    let embedded = token(&jwk_header, &claims(), rs256(rsa_other()));
    assert_eq!(auth(&server, &embedded).unwrap_err(), AuthFailure::Invalid);
    // crit and duplicate members.
    let crit = token(
        &json!({"alg": "RS256", "kid": "k1", "crit": ["exp"]}),
        &claims(),
        rs256(rsa()),
    );
    assert_eq!(auth(&server, &crit).unwrap_err(), AuthFailure::Invalid);
    let dup_header = b64(br#"{"alg":"RS256","kid":"k1","alg":"none"}"#);
    let body = b64(&serde_json::to_vec(&claims()).expect("json"));
    let input = format!("{dup_header}.{body}");
    let sig = rs256(rsa())(input.as_bytes());
    let dup = format!("{input}.{}", b64(&sig));
    assert_eq!(auth(&server, &dup).unwrap_err(), AuthFailure::Invalid);
}

#[test]
fn oidc_unknown_kid_refresh_limit() {
    // The first fetch has k1 only; every later one also k2.
    let source = Source::new(vec![
        Ok(jwks(&[rsa_jwk(rsa(), "k1")])),
        Ok(jwks(&[rsa_jwk(rsa(), "k1"), rsa_jwk(rsa_other(), "k2")])),
    ]);
    let server = server(
        &source,
        &[("krishiv.auth.failureThrottle.maxFailures", "0")],
    );
    assert_eq!(source.fetches(), 1);
    // A rotated key: the request waits for one fetch and then passes.
    let k2 = token(&rs_header("k2"), &claims(), rs256(rsa_other()));
    assert_eq!(
        auth(&server, &k2).expect("after refresh").principal.name,
        "alice"
    );
    assert_eq!(source.fetches(), 2);
    // An unknown kid within the minute: refused at once, no fetch.
    let k3 = token(&rs_header("k3"), &claims(), rs256(rsa()));
    for _ in 0..20 {
        assert_eq!(auth(&server, &k3).unwrap_err(), AuthFailure::Invalid);
    }
    assert_eq!(source.fetches(), 2);

    // The limit itself, on a clock of its own.
    let cache = JwksCache::default();
    let t0 = Instant::now();
    assert!(matches!(
        cache.lookup(Some("a"), Alg::Rs256, t0),
        Lookup::Refresh
    ));
    assert!(matches!(
        cache.lookup(Some("a"), Alg::Rs256, t0 + Duration::from_secs(1)),
        Lookup::NoKeys
    ));
    cache
        .install(&jwks(&[rsa_jwk(rsa(), "k1")]))
        .expect("install");
    assert!(matches!(
        cache.lookup(Some("a"), Alg::Rs256, t0 + Duration::from_secs(2)),
        Lookup::Unknown
    ));
    assert!(matches!(cache.lookup(Some("k1"), Alg::Rs256, t0), Lookup::Keys(_, i) if i.len() == 1));
    // The key's type must fit the algorithm.
    assert!(matches!(
        cache.lookup(Some("k1"), Alg::Es256, t0 + Duration::from_secs(3)),
        Lookup::Unknown
    ));
    assert!(matches!(
        cache.lookup(Some("a"), Alg::Rs256, t0 + ON_DEMAND_EVERY),
        Lookup::Refresh
    ));
    // A token without kid may use any fitting key.
    assert!(matches!(
        cache.lookup(None, Alg::Rs256, t0),
        Lookup::Keys(..)
    ));
}

#[test]
fn oidc_idp_down_uses_cache() {
    let source = Source::new(vec![
        Ok(jwks(&[rsa_jwk(rsa(), "k1")])),
        Err("connection refused".to_owned()),
    ]);
    let server = server(
        &source,
        &[("krishiv.auth.failureThrottle.maxFailures", "0")],
    );
    let k1 = token(&rs_header("k1"), &claims(), rs256(rsa()));
    assert!(auth(&server, &k1).is_ok());
    // A fetch fails (the provider is down): the cached key still verifies.
    let unknown = token(&rs_header("k9"), &claims(), rs256(rsa()));
    assert_eq!(auth(&server, &unknown).unwrap_err(), AuthFailure::Invalid);
    assert_eq!(source.fetches(), 2);
    assert!(auth(&server, &k1).is_ok());
    // A document that does not parse keeps the keys too.
    let cache = JwksCache::default();
    cache
        .install(&jwks(&[rsa_jwk(rsa(), "k1")]))
        .expect("install");
    assert!(cache.install(b"<html>").is_err());
    assert!(matches!(
        cache.lookup(Some("k1"), Alg::Rs256, Instant::now()),
        Lookup::Keys(..)
    ));

    // Never any key: UNAVAILABLE, so clients retry; not counted toward the throttle.
    let down = Source::new(vec![Err("connection refused".to_owned())]);
    let server = self::server(&down, &[("krishiv.auth.failureThrottle.maxFailures", "1")]);
    for _ in 0..3 {
        let f = auth(&server, &k1).unwrap_err();
        assert_eq!(f, AuthFailure::Unavailable);
        assert!(f.is_retryable());
    }
    assert!(
        server
            .failure_counts()
            .iter()
            .any(|c| c.method == Method::Oidc && c.count == 3)
    );
}

#[test]
fn oidc_sync_path_does_not_wait() {
    let source = Source::new(vec![Err("down".to_owned())]);
    let server = server(&source, &[]);
    let k1 = token(&rs_header("k1"), &claims(), rs256(rsa()));
    let creds = Credentials::new(peer(), Surface::Admin).with_bearer(&k1);
    let before = source.fetches();
    assert_eq!(
        server.authenticate_at(&creds, Instant::now()).unwrap_err(),
        AuthFailure::Unavailable
    );
    // It still asked for a fetch in the background.
    wait_until(|| source.fetches() > before);
}

#[test]
fn static_tokens_work_beside_oidc() {
    let secret = "s".repeat(40);
    let dir = tempfile::tempdir().expect("tempdir");
    let reference = super::tests::secret_file(dir.path(), "ci", &secret, 0o600);
    let source = Source::new(vec![Ok(jwks(&[rsa_jwk(rsa(), "k1")]))]);
    let server = server(
        &source,
        &[
            ("krishiv.auth.tokens.ci.secret", reference.as_str()),
            ("krishiv.auth.tokens.ci.principal", "ci"),
            ("krishiv.auth.tokens.ci.role", "viewer"),
        ],
    );
    let ok = auth(&server, &secret).expect("static token");
    assert_eq!(
        (ok.method, ok.principal.name.as_str()),
        (Method::Token, "ci")
    );
    // A dotted token that is not a JWS goes to the static tokens and fails there.
    assert_eq!(auth(&server, "a.b.c").unwrap_err(), AuthFailure::Invalid);
    assert!(
        server
            .failure_counts()
            .iter()
            .any(|c| c.method == Method::Token && c.count == 1)
    );
}

#[test]
fn oidc_counts_as_a_bearer_method() {
    let c = settings(&[]);
    assert!(c.is_configured() && c.has_bearer_method());
    let none = AuthConfig::from_settings([]).expect("empty");
    assert!(!none.is_configured() && !none.has_bearer_method());
}

#[test]
fn real_keys_parse_from_a_jwk_set() {
    let keys = jose::parse_jwks(&jwks(&[rsa_jwk(rsa(), "k1"), ec_jwk(ec(), "e1")])).expect("parse");
    assert!(keys[0].fits(Alg::Rs256) && !keys[0].fits(Alg::Es256));
    assert!(keys[1].fits(Alg::Es256) && !keys[1].fits(Alg::Es384) && !keys[1].fits(Alg::Rs256));
}

/// A client certificate with a JWT beside it: the token must name the certificate's principal;
/// a token that cannot be checked now or has expired fails as such, not as a mismatch.
#[test]
fn certificate_with_a_jwt() {
    const ETL: &str = "spiffe://krishiv.test/etl";
    const CLIENT: &[u8] = include_bytes!("../../tests/fixtures/tls/client.der");
    let ca = format!(
        "file:{}/tests/fixtures/tls/ca.pem",
        env!("CARGO_MANIFEST_DIR")
    );
    let source = Source::new(vec![Ok(jwks(&[rsa_jwk(rsa(), "k1")]))]);
    let server = server(
        &source,
        &[
            ("krishiv.auth.clientCertificates", "optional"),
            ("krishiv.auth.clientCa", ca.as_str()),
            ("krishiv.auth.roles.operator", ETL),
            ("krishiv.auth.failureThrottle.maxFailures", "0"),
        ],
    );
    let with_sub = |sub: &str, kid: &str, exp: u64| {
        let mut c = claims();
        c["sub"] = json!(sub);
        c["exp"] = json!(exp);
        token(&rs_header(kid), &c, rs256(rsa()))
    };
    let run = |bearer: &str| {
        let mut creds = Credentials::new(peer(), Surface::Connect).with_bearer(bearer);
        creds.peer_leaf = Some(CLIENT);
        server.authenticate_at(&creds, Instant::now())
    };
    let later = unix_now() + 300;
    let ok = run(&with_sub(ETL, "k1", later)).expect("same principal");
    assert_eq!((ok.method, ok.role), (Method::Mtls, Some(Role::Operator)));
    assert_eq!(ok.principal.name, ETL);
    assert_eq!(
        run(&with_sub("alice", "k1", later)).unwrap_err(),
        AuthFailure::Invalid
    );
    assert_eq!(
        run(&with_sub(ETL, "k1", unix_now() - 3600)).unwrap_err(),
        AuthFailure::Expired
    );
    // A key not cached: retry later, not a refusal.
    let unknown = run(&with_sub(ETL, "k9", later)).unwrap_err();
    assert_eq!(unknown, AuthFailure::Unavailable);
    let counts = server.failure_counts();
    let count = |m: Method| {
        counts
            .iter()
            .filter(|c| c.method == m)
            .map(|c| c.count)
            .sum::<u64>()
    };
    assert_eq!((count(Method::Mtls), count(Method::Oidc)), (1, 2));
}

/// Certificate names and OIDC role-claim values are separate namespaces.
#[test]
fn certificate_and_group_role_lists_are_separate() {
    let source = Source::new(vec![Ok(jwks(&[rsa_jwk(rsa(), "k1")]))]);
    let server = server(&source, &[("krishiv.auth.roles.admin", "analysts")]);
    let ok = auth(&server, &token(&rs_header("k1"), &claims(), rs256(rsa()))).expect("valid");
    // `analysts` is an admin certificate name, but an operator group.
    assert_eq!(ok.role, Some(Role::Operator));
}

#[test]
fn expired_tokens_do_not_count_toward_the_throttle() {
    let source = Source::new(vec![Ok(jwks(&[rsa_jwk(rsa(), "k1")]))]);
    let server = server(
        &source,
        &[("krishiv.auth.failureThrottle.maxFailures", "2")],
    );
    let mut c = claims();
    c["exp"] = json!(unix_now() - 3600);
    let expired = token(&rs_header("k1"), &c, rs256(rsa()));
    for _ in 0..5 {
        assert_eq!(auth(&server, &expired).unwrap_err(), AuthFailure::Expired);
    }
    assert!(auth(&server, &token(&rs_header("k1"), &claims(), rs256(rsa()))).is_ok());
    // Wrong tokens do.
    let forged = token(&rs_header("k1"), &claims(), rs256(rsa_other()));
    for _ in 0..2 {
        assert_eq!(auth(&server, &forged).unwrap_err(), AuthFailure::Invalid);
    }
    assert_eq!(auth(&server, &forged).unwrap_err(), AuthFailure::Throttled);
}

#[test]
fn an_empty_key_set_keeps_the_cached_keys() {
    let cache = JwksCache::default();
    assert_eq!(cache.install(&jwks(&[])).expect("empty start"), 0);
    cache
        .install(&jwks(&[rsa_jwk(rsa(), "k1")]))
        .expect("install");
    assert!(cache.install(&jwks(&[])).is_err());
    let only_odd = jwks(&[json!({"kty": "oct", "k": "c2VjcmV0"})]);
    assert!(cache.install(&only_odd).is_err());
    assert_eq!(cache.key_count(), 1);
}

#[test]
fn a_token_without_kid_needs_exactly_one_fitting_key() {
    let now = Instant::now();
    let cache = JwksCache::default();
    cache
        .install(&jwks(&[rsa_jwk(rsa(), "k1"), ec_jwk(ec(), "e1")]))
        .expect("install");
    assert!(matches!(cache.lookup(None, Alg::Rs256, now), Lookup::Keys(_, i) if i == [0]));
    cache
        .install(&jwks(&[rsa_jwk(rsa(), "k1"), rsa_jwk(rsa_other(), "k2")]))
        .expect("install");
    assert!(matches!(
        cache.lookup(None, Alg::Rs256, now),
        Lookup::Ambiguous
    ));
    let source = Source::new(vec![Ok(jwks(&[
        rsa_jwk(rsa(), "k1"),
        rsa_jwk(rsa_other(), "k2"),
    ]))]);
    let server = server(&source, &[]);
    let no_kid = token(&json!({"alg": "RS256"}), &claims(), rs256(rsa()));
    assert_eq!(auth(&server, &no_kid).unwrap_err(), AuthFailure::Invalid);
}
