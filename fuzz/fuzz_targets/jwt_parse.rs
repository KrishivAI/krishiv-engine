// SPDX-License-Identifier: Apache-2.0
//! The OIDC token and key-set parsers (P3-T002 PR-14d; design §3.3, threat model §4.9): a JWS
//! compact token, a JWK set and a discovery document arrive from clients and the identity
//! provider, before any signature is checked, and a signer converts a key service's DER
//! signature (ADR 0050). Each parser must return a value or an error for any bytes, never
//! panic, loop or allocate without bound. The first byte picks the parser.
#![no_main]

use krishiv_jose as jose;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    let Some((&which, rest)) = data.split_first() else {
        return;
    };
    match which % 4 {
        0 => {
            if let Ok(text) = std::str::from_utf8(rest) {
                let shaped = jose::looks_like_jws(text);
                if let Ok(jws) = jose::parse_jws(text) {
                    // A parsed token has the shape the router looks for.
                    assert!(shaped);
                    assert!(jws.signing_input.len() < text.len());
                    let rules = jose::ClaimRules {
                        issuer: "https://idp.test/",
                        audience: "krishiv",
                        principal_claim: "sub",
                        role_claim: "groups",
                        skew_secs: 60,
                        max_lifetime_secs: 24 * 3600,
                    };
                    if let Ok(id) = jose::check_claims(&jws.claims, &rules, 1_800_000_000) {
                        assert!(!id.principal.is_empty());
                        assert!(id.groups.len() <= jose::MAX_GROUPS);
                    }
                }
            }
        }
        1 => {
            if let Ok(keys) = jose::parse_jwks(rest) {
                assert!(keys.len() <= jose::MAX_JWKS_KEYS);
            }
        }
        2 => {
            let _ = jose::discovered_jwks_uri(rest, "https://idp.test/");
        }
        _ => {
            for curve in [jose::Curve::P256, jose::Curve::P384] {
                if let Ok(fixed) = jose::ecdsa_der_to_fixed(curve, rest) {
                    assert_eq!(fixed.len(), 2 * curve.coordinate_len());
                }
            }
        }
    }
});
