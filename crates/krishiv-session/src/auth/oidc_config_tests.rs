// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.auth.oidc.*` settings (`OidcConfig`) and the principal-name rule shared with the
//! certificate names. The token parsers are tested in `krishiv-jose` (ADR 0050).

use std::time::Duration;

use super::Alg;
use super::oidc_config::OidcConfig;

const ISSUER: &str = "https://idp.example.test/";

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
            krishiv_jose::is_format_char(c),
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

#[test]
fn max_lifetime_setting() {
    let mut c = OidcConfig::default();
    assert_eq!(c.max_lifetime, Duration::from_hours(24));
    assert!(
        c.apply("krishiv.auth.oidc.maxLifetime", "1h")
            .expect("known")
    );
    assert_eq!(c.max_lifetime, Duration::from_hours(1));
    for bad in ["25h", "30s", "0s", "x"] {
        assert!(
            c.apply("krishiv.auth.oidc.maxLifetime", bad).is_err(),
            "{bad}"
        );
    }
}
