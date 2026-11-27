// SPDX-License-Identifier: Apache-2.0
//! The signature check on aws-lc-rs (feature `verify`).
//!
//! RSASSA-PKCS1-v1_5 on 2048 to 8192 bit keys, and ECDSA P-256/P-384 with fixed-width `r || s`
//! (RFC 7518 §3.4). There is no `none` and no HMAC path, so an HS256 token "signed" with an RSA
//! public key cannot verify (algorithm pinning by construction).

use aws_lc_rs::signature::{self, RsaPublicKeyComponents, UnparsedPublicKey};

use crate::{Alg, Jwk, KeyMaterial};

/// Whether `signature` over `input` verifies with `key` for `alg`. The key must fit `alg`
/// ([`Jwk::fits`]); a key that does not never verifies.
#[must_use]
pub fn verify_signature(alg: Alg, key: &Jwk, input: &[u8], signature: &[u8]) -> bool {
    if !key.fits(alg) {
        return false;
    }
    match (&key.material, alg) {
        (KeyMaterial::Rsa { n, e }, Alg::Rs256 | Alg::Rs384 | Alg::Rs512) => {
            let params = match alg {
                Alg::Rs256 => &signature::RSA_PKCS1_2048_8192_SHA256,
                Alg::Rs384 => &signature::RSA_PKCS1_2048_8192_SHA384,
                _ => &signature::RSA_PKCS1_2048_8192_SHA512,
            };
            RsaPublicKeyComponents { n, e }
                .verify(params, input, signature)
                .is_ok()
        }
        (KeyMaterial::Ec { point, .. }, Alg::Es256) => {
            UnparsedPublicKey::new(&signature::ECDSA_P256_SHA256_FIXED, point)
                .verify(input, signature)
                .is_ok()
        }
        (KeyMaterial::Ec { point, .. }, Alg::Es384) => {
            UnparsedPublicKey::new(&signature::ECDSA_P384_SHA384_FIXED, point)
                .verify(input, signature)
                .is_ok()
        }
        _ => false,
    }
}
