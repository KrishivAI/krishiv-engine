// SPDX-License-Identifier: Apache-2.0
//! The OIDC parsers and settings, without cryptography (so they run in every build): base64url,
//! the token shape, header refusals, JWK sets, discovery, claim checks and `krishiv.auth.oidc.*`.
//! Signatures and the key cache are tested in `oidc_tests.rs` (feature `oidc`).

use std::time::Duration;

use serde_json::{Value, json};

use super::jose::{self, Alg, ClaimRules, JoseError, KeyMaterial};
use super::oidc_config::OidcConfig;

const ISSUER: &str = "https://idp.example.test/";
const NOW: u64 = 1_800_000_000;

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

fn token_with_header(header: &str) -> String {
    format!(
        "{}.{}.c2ln",
        b64(header.as_bytes()),
        b64(b"{\"sub\":\"a\"}")
    )
}

fn rules() -> ClaimRules<'static> {
    ClaimRules {
        issuer: ISSUER,
        audience: "krishiv",
        principal_claim: "sub",
        role_claim: "groups",
        skew_secs: 60,
    }
}

fn claims(v: &Value) -> jose::Object {
    serde_json::from_value(v.clone()).expect("object")
}

fn good() -> Value {
    json!({"iss": ISSUER, "aud": "krishiv", "sub": "alice", "exp": NOW + 10, "groups": ["g1", "g2"]})
}

fn check(v: &Value) -> Result<jose::Identity, JoseError> {
    jose::check_claims(&claims(v), &rules(), NOW)
}

#[test]
fn base64url_is_strict() {
    assert_eq!(jose::b64url_decode("").expect("empty"), b"");
    assert_eq!(jose::b64url_decode("TWFu").expect("Man"), b"Man");
    assert_eq!(jose::b64url_decode("TWE").expect("Ma"), b"Ma");
    assert_eq!(jose::b64url_decode("TQ").expect("M"), b"M");
    assert_eq!(
        jose::b64url_decode("-_8").expect("url alphabet"),
        [0xfb, 0xff]
    );
    // Padding, the standard alphabet, a lone character, non-zero trailing bits.
    for bad in ["TQ==", "a+b/", "T", "TR", "TWF"] {
        assert!(jose::b64url_decode(bad).is_err(), "{bad}");
    }
    assert_eq!(b64(b"any carnal pleas"), "YW55IGNhcm5hbCBwbGVhcw");
}

#[test]
fn token_shape_and_header_refusals() {
    assert!(!jose::looks_like_jws("abc"));
    assert!(!jose::looks_like_jws("a.b.c.d"));
    assert!(!jose::looks_like_jws("eyJ.."));
    assert!(!jose::looks_like_jws(&token_with_header(
        r#"{"typ":"JWT"}"#
    )));
    assert!(!jose::looks_like_jws(&token_with_header(r#"{"alg":1}"#)));
    assert!(jose::looks_like_jws(&token_with_header(
        r#"{"alg":"RS256"}"#
    )));
    let long = format!(
        "{}.{}.c2ln",
        b64(br#"{"alg":"RS256"}"#),
        "A".repeat(jose::MAX_TOKEN_BYTES)
    );
    assert!(!jose::looks_like_jws(&long));
    assert_eq!(
        jose::parse_jws(&long).unwrap_err(),
        JoseError::Invalid("token too long")
    );

    let es384 = token_with_header(r#"{"alg":"ES384","kid":"k"}"#);
    let ok = jose::parse_jws(&es384).expect("parse");
    assert_eq!(
        (ok.header.alg, ok.header.kid.as_deref()),
        (Alg::Es384, Some("k"))
    );
    assert_eq!(ok.signature, b"sig");
    assert_eq!(
        ok.signing_input,
        es384
            .rsplit_once('.')
            .map(|(i, _)| i.as_bytes())
            .expect("dot")
    );
    assert_eq!(ok.claims.0.get("sub"), Some(&json!("a")));

    let refused = |header: &str| jose::parse_jws(&token_with_header(header)).unwrap_err();
    for (header, why) in [
        (r#"{"alg":"none"}"#, "algorithm not supported"),
        (r#"{"alg":"HS256"}"#, "algorithm not supported"),
        (r#"{"alg":"rs256"}"#, "algorithm not supported"),
        (r#"{"alg":"PS256"}"#, "algorithm not supported"),
        (r#"{"kid":"k"}"#, "no algorithm"),
        (r#"{"alg":["RS256"]}"#, "member is not a string"),
        (
            r#"{"alg":"RS256","crit":["b64"]}"#,
            "critical header extensions",
        ),
        (r#"{"alg":"RS256","alg":"none"}"#, "not a JSON object"),
        (r#"{"alg":"RS256","typ":"dpop+jwt"}"#, "not a JWT"),
        (r#"{"alg":"RS256","kid":7}"#, "member is not a string"),
        ("[]", "not a JSON object"),
    ] {
        assert_eq!(refused(header), JoseError::Invalid(why), "{header}");
    }
    for typ in ["JWT", "jwt", "at+jwt", "application/at+JWT"] {
        let h = format!(r#"{{"alg":"RS256","typ":"{typ}"}}"#);
        assert!(jose::parse_jws(&token_with_header(&h)).is_ok(), "{typ}");
    }
    // Two parts, an empty signature, a payload that is not an object, a duplicate claim.
    let h = b64(br#"{"alg":"RS256"}"#);
    for bad in [
        format!("{h}.e30"),
        format!("{h}.e30."),
        format!("{h}.{}.c2ln", b64(b"[1]")),
        format!("{h}.{}.c2ln", b64(br#"{"sub":"a","sub":"b"}"#)),
        format!("{h}.e30.c2ln.x"),
        format!("{h}.e3=.c2ln"),
    ] {
        assert!(jose::parse_jws(&bad).is_err(), "{bad}");
    }
}

#[test]
fn claims_are_checked() {
    let id = check(&good()).expect("valid");
    assert_eq!(id.principal, "alice");
    assert_eq!(id.groups, ["g1", "g2"]);
    let with = |name: &str, v: Value| {
        let mut c = good();
        if v.is_null() {
            c.as_object_mut().expect("object").remove(name);
        } else {
            c[name] = v;
        }
        check(&c)
    };
    // Expiry with the skew, both sides of the line; fractional dates.
    assert_eq!(
        with("exp", json!(NOW - 60)).unwrap_err(),
        JoseError::Expired
    );
    assert!(with("exp", json!(NOW - 59)).is_ok());
    assert!(with("exp", json!(1.8e9 - 59.5)).is_ok());
    assert_eq!(
        with("exp", Value::Null).unwrap_err(),
        JoseError::Invalid("no exp")
    );
    assert_eq!(
        with("exp", json!("soon")).unwrap_err(),
        JoseError::Invalid("date claim is not a number")
    );
    assert!(with("nbf", json!(NOW + 60)).is_ok());
    assert_eq!(
        with("nbf", json!(NOW + 61)).unwrap_err(),
        JoseError::Invalid("not valid yet")
    );
    assert_eq!(
        with("iss", json!("https://idp.example.test")).unwrap_err(),
        JoseError::Invalid("issuer")
    );
    assert_eq!(
        with("iss", Value::Null).unwrap_err(),
        JoseError::Invalid("issuer")
    );
    assert_eq!(
        with("iss", json!(1)).unwrap_err(),
        JoseError::Invalid("member is not a string")
    );
    assert!(with("aud", json!(["x", "krishiv"])).is_ok());
    for aud in [json!("x"), json!([]), json!(["x"]), json!(5), Value::Null] {
        assert_eq!(
            with("aud", aud).unwrap_err(),
            JoseError::Invalid("audience")
        );
    }
    for sub in [
        json!(""),
        json!("a\nb"),
        json!("a\u{200B}b"),
        json!("x".repeat(257)),
    ] {
        assert_eq!(
            with("sub", sub).unwrap_err(),
            JoseError::Invalid("principal claim")
        );
    }
    assert!(with("sub", json!("x".repeat(256))).is_ok());
    assert_eq!(with("groups", json!("solo")).expect("one").groups, ["solo"]);
    assert!(with("groups", Value::Null).expect("none").groups.is_empty());
    let too_many: Vec<String> = (0..=jose::MAX_GROUPS).map(|i| i.to_string()).collect();
    for groups in [json!([1]), json!({"a": 1}), json!(too_many)] {
        assert_eq!(
            with("groups", groups).unwrap_err(),
            JoseError::Invalid("role claim")
        );
    }
    assert_eq!(JoseError::Expired.to_string(), "expired");
    assert_eq!(JoseError::Invalid("issuer").to_string(), "issuer");
}

#[test]
fn jwk_sets_keep_usable_keys() {
    let n = b64(&[0, 0, 0xc5, 1, 2]);
    let rsa = |kid: &str| json!({"kty": "RSA", "kid": kid, "n": n, "e": "AQAB"});
    let ec = |crv: &str, len: usize| json!({"kty": "EC", "kid": crv, "crv": crv, "x": b64(&vec![1; len]), "y": b64(&vec![2; len])});
    let mut wrong_alg = rsa("wrong-alg");
    wrong_alg["alg"] = json!("ES256");
    let mut enc = rsa("enc");
    enc["use"] = json!("enc");
    let mut ops = rsa("ops");
    ops["key_ops"] = json!(["encrypt"]);
    let mut verify = rsa("verify");
    verify["key_ops"] = json!(["verify"]);
    verify["alg"] = json!("RS512");
    verify["use"] = json!("sig");
    let mut unknown_alg = rsa("unknown-alg");
    unknown_alg["alg"] = json!("PS256");
    let set = json!({"keys": [
        {"kty": "oct", "k": "c2VjcmV0"},
        {"kty": "OKP", "crv": "Ed25519", "x": "AA"},
        ec("P-521", 66),
        {"kty": "EC", "crv": "P-256", "x": "AQ", "y": "Ag"},
        {"kty": "RSA", "n": "not base64!", "e": "AQAB"},
        {"kty": "RSA", "n": "AAAA", "e": "AQAB"},
        {"kty": "RSA", "e": "AQAB"},
        wrong_alg, enc, ops, unknown_alg,
        rsa("k1"), verify, ec("P-256", 32), ec("P-384", 48),
    ], "other": 1});
    let keys = jose::parse_jwks(&serde_json::to_vec(&set).expect("json")).expect("parse");
    let kids: Vec<_> = keys.iter().filter_map(|k| k.kid.as_deref()).collect();
    assert_eq!(kids, ["k1", "verify", "P-256", "P-384"]);
    let KeyMaterial::Rsa { n, e } = &keys[0].material else {
        panic!("an RSA key");
    };
    assert_eq!(
        (n.as_slice(), e.as_slice()),
        (&[0xc5, 1, 2][..], &[1, 0, 1][..])
    );
    let KeyMaterial::Ec { point, curve } = &keys[3].material else {
        panic!("an EC key");
    };
    assert_eq!((point.len(), point[0], *curve), (97, 4, jose::Curve::P384));
    assert!(keys[1].fits(Alg::Rs512) && !keys[1].fits(Alg::Rs256));
    assert!(keys[2].fits(Alg::Es256) && !keys[2].fits(Alg::Es384) && !keys[2].fits(Alg::Rs256));
    assert!(keys[0].fits(Alg::Rs384) && !keys[0].fits(Alg::Es256));

    for bad in [
        &br#"{"keys": [], "keys": []}"#[..],
        b"{}",
        b"[]",
        b"{\"keys\": [1]}",
        b"{\"keys\": [{\"kty\": \"RSA\", \"kty\": \"EC\"}]}",
    ] {
        assert!(
            jose::parse_jwks(bad).is_err(),
            "{}",
            String::from_utf8_lossy(bad)
        );
    }
    let many = json!({"keys": vec![rsa("k"); jose::MAX_JWKS_KEYS + 1]});
    assert_eq!(
        jose::parse_jwks(&serde_json::to_vec(&many).expect("json")).unwrap_err(),
        JoseError::Invalid("too many keys")
    );
    let huge = vec![b' '; jose::MAX_DOCUMENT_BYTES + 1];
    assert!(jose::parse_jwks(&huge).is_err());
    assert!(jose::discovered_jwks_uri(&huge, ISSUER).is_err());
}

#[test]
fn discovery_needs_the_same_issuer() {
    let doc = json!({"issuer": ISSUER, "jwks_uri": "https://idp.example.test/keys"}).to_string();
    assert_eq!(
        jose::discovered_jwks_uri(doc.as_bytes(), ISSUER).expect("uri"),
        "https://idp.example.test/keys"
    );
    assert!(jose::discovered_jwks_uri(doc.as_bytes(), "https://evil.test/").is_err());
    let no_uri = json!({"issuer": ISSUER}).to_string();
    assert!(jose::discovered_jwks_uri(no_uri.as_bytes(), ISSUER).is_err());
}

#[test]
fn algorithm_names() {
    for alg in Alg::ALL {
        assert_eq!(Alg::parse(alg.as_str()), Some(alg));
    }
    assert_eq!(Alg::parse("ES512"), None);
    assert_eq!(Alg::Rs256.curve(), None);
    assert_eq!(
        Alg::Es256.curve().map(jose::Curve::coordinate_len),
        Some(32)
    );
}

fn parse(kv: &[(&str, &str)]) -> Result<OidcConfig, String> {
    let mut c = OidcConfig::default();
    for (k, v) in kv {
        if !c.apply(k, v).map_err(|e| e.to_string())? {
            return Err(format!("not an OIDC key: {k}"));
        }
    }
    c.validate_values().map_err(|e| e.to_string())?;
    Ok(c)
}

#[test]
fn oidc_settings() {
    let c = parse(&[
        ("krishiv.auth.oidc.issuer", ISSUER),
        ("krishiv.auth.oidc.audience", "krishiv"),
        ("krishiv.auth.oidc.algorithms", "ES256,RS512,ES256"),
        ("krishiv.auth.oidc.clockSkew", "0s"),
        ("krishiv.auth.oidc.jwksRefresh", "1h"),
        ("krishiv.auth.oidc.principalClaim", "email"),
        ("krishiv.auth.oidc.roleClaim", "roles"),
        ("krishiv.auth.oidc.jwksUri", "https://idp.example.test/keys"),
        ("krishiv.auth.oidc.allowInsecureIssuer", "false"),
    ])
    .expect("valid");
    assert!(c.is_on());
    assert_eq!(c.algorithms, [Alg::Es256, Alg::Rs512]);
    assert_eq!(c.jwks_refresh, Duration::from_hours(1));
    assert_eq!(c.clock_skew, Duration::ZERO);
    assert_eq!(
        (c.principal_claim.as_str(), c.role_claim.as_str()),
        ("email", "roles")
    );
    assert_eq!(
        c.discovery_uri().as_deref(),
        Some("https://idp.example.test/.well-known/openid-configuration")
    );
    assert!(!OidcConfig::default().is_on());
    assert_eq!(OidcConfig::default().discovery_uri(), None);
    assert!(OidcConfig::default().validate_values().is_ok());
    let issuer = ("krishiv.auth.oidc.issuer", ISSUER);
    let audience = ("krishiv.auth.oidc.audience", "krishiv");
    for (kv, what) in [
        (
            vec![("krishiv.auth.oidc.issuer", "http://idp.test"), audience],
            "http issuer",
        ),
        (
            vec![("krishiv.auth.oidc.issuer", "idp.test"), audience],
            "no scheme",
        ),
        (vec![issuer], "no audience"),
        (vec![audience], "no issuer"),
        (vec![("krishiv.auth.oidc.issuer", " ")], "empty issuer"),
        (vec![("krishiv.auth.oidc.algorithms", "HS256")], "HMAC"),
        (vec![("krishiv.auth.oidc.algorithms", "none")], "none"),
        (vec![("krishiv.auth.oidc.algorithms", " , ")], "empty list"),
        (vec![("krishiv.auth.oidc.clockSkew", "10m")], "skew"),
        (vec![("krishiv.auth.oidc.jwksRefresh", "5s")], "refresh"),
        (vec![("krishiv.auth.oidc.jwksRefresh", "2d")], "unit"),
        (
            vec![("krishiv.auth.oidc.allowInsecureIssuer", "yes")],
            "bool",
        ),
        (vec![("krishiv.auth.oidc.issuers", ISSUER)], "unknown key"),
        (
            vec![
                issuer,
                audience,
                ("krishiv.auth.oidc.jwksUri", "ftp://idp.test/keys"),
            ],
            "jwks scheme",
        ),
    ] {
        assert!(parse(&kv).is_err(), "{what}");
    }
    let insecure = parse(&[
        ("krishiv.auth.oidc.issuer", "http://127.0.0.1:9/"),
        audience,
        ("krishiv.auth.oidc.allowInsecureIssuer", "TRUE"),
    ])
    .expect("allowed for tests");
    assert!(insecure.uri_allowed("http://127.0.0.1:9/keys"));
    assert!(insecure.uri_allowed("HTTPS://idp.test/keys"));
    assert!(!insecure.uri_allowed("file:///etc/keys"));
    assert!(!c.uri_allowed("http://idp.test/keys"));
    assert!(!c.uri_allowed("https:/"));
    // No host, user information (it would be logged) or fragment.
    for bad in [
        "https://",
        "https:///keys",
        "https://user:pw@idp.test/keys",
        "https://idp.test/keys#x",
    ] {
        assert!(!c.uri_allowed(bad), "{bad}");
    }
    assert!(c.uri_allowed("https://idp.test:8443/keys?tenant=a"));
    assert!(c.uri_allowed("https://idp.test?x=a@b"));
}

#[test]
fn oidc_role_lists() {
    let c = parse(&[
        ("krishiv.auth.oidc.roles.admin", "platform-admins"),
        ("krishiv.auth.oidc.roles.viewer", "staff, interns"),
        ("krishiv.auth.oidc.roles.viewer", "staff"),
    ])
    .expect_err("an issuer is required");
    assert!(c.contains("issuer"), "{c}");
    let mut c = OidcConfig::default();
    for (k, v) in [
        ("krishiv.auth.oidc.roles.admin", "platform-admins"),
        ("krishiv.auth.oidc.roles.viewer", "staff, interns"),
        // A key given twice keeps its last value.
        ("krishiv.auth.oidc.roles.viewer", "staff"),
    ] {
        assert!(c.apply(k, v).expect("valid"));
    }
    assert_eq!(c.roles.role_of("platform-admins"), Some(super::Role::Admin));
    assert_eq!(c.roles.role_of("staff"), Some(super::Role::Viewer));
    assert_eq!(c.roles.role_of("interns"), None);
    // One role per value; unknown or miscased role names are not OIDC keys.
    assert!(
        c.apply("krishiv.auth.oidc.roles.operator", "staff")
            .is_err()
    );
    assert!(
        !c.apply("krishiv.auth.oidc.roles.Admin", "x")
            .expect("not a key")
    );
    assert!(
        !c.apply("krishiv.auth.oidc.roles.owner", "x")
            .expect("not a key")
    );
    assert!(super::AuthConfig::from_settings([("krishiv.auth.oidc.roles.owner", "x")]).is_err());
}

#[test]
fn principal_names_refuse_the_certificate_format_characters() {
    for c in (0..=0x10_ffff).filter_map(char::from_u32) {
        assert_eq!(
            jose::is_format_char(c),
            super::der::is_format(c),
            "U+{:04X}",
            u32::from(c)
        );
    }
}

/// A build without the `oidc` feature refuses every OIDC setting at startup.
#[cfg(not(feature = "oidc"))]
#[test]
fn oidc_needs_the_feature() {
    let e = super::AuthConfig::from_settings([
        ("krishiv.auth.oidc.issuer", ISSUER),
        ("krishiv.auth.oidc.audience", "krishiv"),
    ])
    .unwrap_err();
    assert!(e.to_string().contains("`oidc` feature"), "{e}");
    let e = super::AuthConfig::from_settings([("krishiv.auth.oidc.clockSkew", "5s")]).unwrap_err();
    assert!(e.to_string().contains("issuer"), "{e}");
}
