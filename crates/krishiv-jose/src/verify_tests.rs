// SPDX-License-Identifier: Apache-2.0
//! `verify_signature` (feature `verify`) with keys generated per run: a good signature under
//! each algorithm (RS256, RS384, RS512, ES256, ES384), another message, another algorithm, a key
//! that does not fit the algorithm. The whole chain is tested end to end by `krishiv-session`'s
//! `oidc_tests.rs`.

use aws_lc_rs::rand::SystemRandom;
use aws_lc_rs::rsa::KeySize;
use aws_lc_rs::signature::{self, EcdsaKeyPair, KeyPair as _, RsaKeyPair};

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

fn ec_key(alg: &'static signature::EcdsaSigningAlgorithm, curve: Curve) -> (EcdsaKeyPair, Jwk) {
    let rng = SystemRandom::new();
    let pkcs8 = EcdsaKeyPair::generate_pkcs8(alg, &rng).expect("key");
    let pair = EcdsaKeyPair::from_pkcs8(alg, pkcs8.as_ref()).expect("key");
    let jwk = Jwk {
        kid: None,
        alg: None,
        material: KeyMaterial::Ec {
            curve,
            point: pair.public_key().as_ref().to_vec(),
        },
    };
    (pair, jwk)
}

#[test]
fn an_es384_signature_verifies_only_for_its_message_and_algorithm() {
    let (pair, key) = ec_key(&signature::ECDSA_P384_SHA384_FIXED_SIGNING, Curve::P384);
    let sig = pair.sign(&SystemRandom::new(), b"input").expect("sign");
    assert_eq!(sig.as_ref().len(), 96);
    assert!(verify_signature(Alg::Es384, &key, b"input", sig.as_ref()));
    assert!(!verify_signature(Alg::Es384, &key, b"other", sig.as_ref()));
    for alg in [Alg::Es256, Alg::Rs256, Alg::Rs384, Alg::Rs512] {
        assert!(
            !verify_signature(alg, &key, b"input", sig.as_ref()),
            "{alg:?}"
        );
    }
}

/// `(n, e)` of an RSA public key's DER (`RSAPublicKey ::= SEQUENCE { n INTEGER, e INTEGER }`),
/// without leading zero bytes, as a JWK carries them.
fn rsa_components(der: &[u8]) -> (Vec<u8>, Vec<u8>) {
    fn tlv(d: &[u8]) -> (&[u8], &[u8]) {
        let (len, head) = match d[1] {
            l if l < 0x80 => (usize::from(l), 2),
            0x81 => (usize::from(d[2]), 3),
            0x82 => (usize::from(d[2]) << 8 | usize::from(d[3]), 4),
            other => panic!("unexpected DER length byte {other:#x}"),
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

#[test]
fn an_rsa_signature_verifies_only_under_its_own_algorithm_and_message() {
    let pair = RsaKeyPair::generate(KeySize::Rsa2048).expect("generate an RSA key");
    let (n, e) = rsa_components(pair.public_key().as_ref());
    let key = Jwk {
        kid: None,
        alg: None,
        material: KeyMaterial::Rsa { n, e },
    };
    let rng = SystemRandom::new();
    let schemes = [
        (Alg::Rs256, &signature::RSA_PKCS1_SHA256),
        (Alg::Rs384, &signature::RSA_PKCS1_SHA384),
        (Alg::Rs512, &signature::RSA_PKCS1_SHA512),
    ];
    for (alg, scheme) in schemes {
        let mut sig = vec![0; pair.public_modulus_len()];
        pair.sign(scheme, &rng, b"input", &mut sig).expect("sign");
        assert!(verify_signature(alg, &key, b"input", &sig), "{alg:?}");
        assert!(
            !verify_signature(alg, &key, b"other", &sig),
            "{alg:?} other message"
        );
        for other in [Alg::Rs256, Alg::Rs384, Alg::Rs512, Alg::Es256, Alg::Es384] {
            if other != alg {
                assert!(
                    !verify_signature(other, &key, b"input", &sig),
                    "{alg:?} signature checked as {other:?}"
                );
            }
        }
    }
}
