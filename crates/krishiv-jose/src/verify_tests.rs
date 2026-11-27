// SPDX-License-Identifier: Apache-2.0
//! `verify_signature` (feature `verify`) with keys generated per run: a good ECDSA signature,
//! another message, a key that does not fit the algorithm. RSA and the whole chain are tested
//! end to end by `krishiv-session`'s `oidc_tests.rs`.

use aws_lc_rs::rand::SystemRandom;
use aws_lc_rs::signature::{self, EcdsaKeyPair, KeyPair as _};

use crate::{Alg, Curve, Jwk, KeyMaterial, verify_signature};

fn es256_key() -> (EcdsaKeyPair, Jwk) {
    let rng = SystemRandom::new();
    let alg = &signature::ECDSA_P256_SHA256_FIXED_SIGNING;
    let pkcs8 = EcdsaKeyPair::generate_pkcs8(alg, &rng).expect("key");
    let pair = EcdsaKeyPair::from_pkcs8(alg, pkcs8.as_ref()).expect("key");
    let jwk = Jwk {
        kid: Some("e1".into()),
        alg: None,
        material: KeyMaterial::Ec {
            curve: Curve::P256,
            point: pair.public_key().as_ref().to_vec(),
        },
    };
    (pair, jwk)
}

#[test]
fn an_ecdsa_signature_verifies_only_for_its_message_and_algorithm() {
    let (pair, key) = es256_key();
    let sig = pair.sign(&SystemRandom::new(), b"input").expect("sign");
    assert!(verify_signature(Alg::Es256, &key, b"input", sig.as_ref()));
    assert!(!verify_signature(Alg::Es256, &key, b"other", sig.as_ref()));
    // The key does not fit these algorithms, so they never verify with it.
    for alg in [Alg::Es384, Alg::Rs256, Alg::Rs384, Alg::Rs512] {
        assert!(
            !verify_signature(alg, &key, b"input", sig.as_ref()),
            "{alg:?}"
        );
    }
}

#[test]
fn a_key_pinned_to_another_algorithm_never_verifies() {
    let (pair, mut key) = es256_key();
    let sig = pair.sign(&SystemRandom::new(), b"input").expect("sign");
    key.alg = Some(Alg::Es384);
    assert!(!verify_signature(Alg::Es256, &key, b"input", sig.as_ref()));
}
