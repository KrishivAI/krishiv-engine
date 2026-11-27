// SPDX-License-Identifier: Apache-2.0
//! The token verifier is the `krishiv-jose` crate (ADR 0050), shared with the platform; this
//! module keeps its items at the paths the rest of `auth` uses.

#[cfg(feature = "oidc")]
pub(crate) use krishiv_jose::verify_signature;
pub(crate) use krishiv_jose::{
    Alg, ClaimRules, Identity, JoseError, Jwk, MAX_DOCUMENT_BYTES, check_claims,
    discovered_jwks_uri, is_format_char, looks_like_jws, parse_jwks, parse_jws,
};
