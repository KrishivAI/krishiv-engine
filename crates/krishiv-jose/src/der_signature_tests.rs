// SPDX-License-Identifier: Apache-2.0
//! `ecdsa_der_to_fixed`: a fixed vector (RFC 6979 A.2.5, P-256 with SHA-256, message
//! "sample"), round trips through aws-lc-rs's own DER signer (feature `verify`), and the
//! refusals of malformed DER.

use crate::{Curve, JoseError, ecdsa_der_to_fixed};

fn hex(s: &str) -> Vec<u8> {
    let s: String = s.split_whitespace().collect();
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

/// RFC 6979 A.2.5: the SHA-256 signature of "sample" by the P-256 key.
const SAMPLE_R: &str = "EFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716";
const SAMPLE_S: &str = "F7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8";

/// The DER a key service returns for that signature: both numbers have the top bit set, so
/// each carries one leading zero byte.
fn sample_der() -> Vec<u8> {
    let mut der = hex("30 46 02 21 00");
    der.extend(hex(SAMPLE_R));
    der.extend(hex("02 21 00"));
    der.extend(hex(SAMPLE_S));
    der
}

#[test]
fn the_rfc_6979_vector_turns_into_r_and_s() {
    let fixed = ecdsa_der_to_fixed(Curve::P256, &sample_der()).expect("valid DER");
    let mut want = hex(SAMPLE_R);
    want.extend(hex(SAMPLE_S));
    assert_eq!(fixed, want);
}

#[cfg(feature = "verify")]
#[test]
fn the_rfc_6979_vector_verifies_with_its_key() {
    use crate::{Alg, Jwk, KeyMaterial, verify_signature};
    // RFC 6979 A.2.5: the key's public point.
    const P256_X: &str = "60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6";
    const P256_Y: &str = "7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299";
    let mut point = vec![0x04];
    point.extend(hex(P256_X));
    point.extend(hex(P256_Y));
    let key = Jwk {
        kid: None,
        alg: Some(Alg::Es256),
        material: KeyMaterial::Ec {
            curve: Curve::P256,
            point,
        },
    };
    let fixed = ecdsa_der_to_fixed(Curve::P256, &sample_der()).expect("valid DER");
    assert!(verify_signature(Alg::Es256, &key, b"sample", &fixed));
    assert!(!verify_signature(Alg::Es256, &key, b"samplf", &fixed));
    // The DER form itself is never accepted by the verifier.
    assert!(!verify_signature(
        Alg::Es256,
        &key,
        b"sample",
        &sample_der()
    ));
}

/// Short numbers are padded on the left to the coordinate length.
#[test]
fn short_numbers_are_left_padded() {
    let der = hex("30 06 02 01 01 02 01 7f");
    let fixed = ecdsa_der_to_fixed(Curve::P384, &der).expect("valid DER");
    assert_eq!(fixed.len(), 96);
    assert_eq!(fixed[47], 1);
    assert_eq!(fixed[95], 0x7f);
    assert!(fixed[..47].iter().all(|b| *b == 0));
    assert!(fixed[48..95].iter().all(|b| *b == 0));
}

#[test]
fn malformed_der_is_refused() {
    let valid = sample_der();
    let mut wrong_tag = valid.clone();
    wrong_tag[0] = 0x31;
    let mut wrong_int_tag = valid.clone();
    wrong_int_tag[2] = 0x04;
    let mut trailing = valid.clone();
    trailing.push(0);
    let mut inner_trailing = valid.clone();
    inner_trailing[1] = 0x47;
    inner_trailing.push(0);
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", vec![]),
        ("wrong sequence tag", wrong_tag),
        ("wrong integer tag", wrong_int_tag),
        ("trailing bytes after the sequence", trailing),
        ("trailing bytes inside the sequence", inner_trailing),
        (
            "sequence longer than the input",
            valid[..valid.len() - 1].to_vec(),
        ),
        (
            "long-form length beyond the input",
            hex("30 84 ff ff ff ff 02 01 01 02 01 01"),
        ),
        // DER takes the short form for lengths below 128.
        (
            "non-minimal long-form length",
            hex("30 81 06 02 01 01 02 01 01"),
        ),
        (
            "integer longer than the sequence",
            hex("30 06 02 05 01 02 01 01"),
        ),
        ("empty integer", hex("30 05 02 00 02 01 01")),
        ("negative integer", hex("30 06 02 01 81 02 01 01")),
        ("non-minimal integer", hex("30 07 02 02 00 01 02 01 01")),
        ("zero", hex("30 06 02 01 00 02 01 01")),
        ("one integer only", hex("30 03 02 01 01")),
        ("three integers", hex("30 09 02 01 01 02 01 01 02 01 01")),
    ];
    for (why, der) in cases {
        assert!(
            matches!(
                ecdsa_der_to_fixed(Curve::P256, &der),
                Err(JoseError::Invalid(_))
            ),
            "{why}"
        );
    }
}

/// A number wider than the curve's coordinate: P-384's 48 bytes do not fit P-256.
#[test]
fn a_number_too_wide_for_the_curve_is_refused() {
    let mut der = vec![0x30, 2 + 48 + 3, 0x02, 48];
    der.extend([0x11; 48]);
    der.extend(hex("02 01 01"));
    assert!(ecdsa_der_to_fixed(Curve::P256, &der).is_err());
    assert_eq!(
        ecdsa_der_to_fixed(Curve::P384, &der)
            .expect("fits P-384")
            .len(),
        96
    );
}

/// aws-lc-rs signs in DER; the conversion's output verifies as the fixed form, on both curves,
/// with fresh keys each run.
#[cfg(feature = "verify")]
#[test]
fn aws_lc_rs_der_signatures_round_trip() {
    use aws_lc_rs::rand::SystemRandom;
    use aws_lc_rs::signature::{self, EcdsaKeyPair, KeyPair as _};

    use crate::{Alg, Jwk, KeyMaterial, verify_signature};
    let rng = SystemRandom::new();
    for (alg, curve, signing) in [
        (
            Alg::Es256,
            Curve::P256,
            &signature::ECDSA_P256_SHA256_ASN1_SIGNING,
        ),
        (
            Alg::Es384,
            Curve::P384,
            &signature::ECDSA_P384_SHA384_ASN1_SIGNING,
        ),
    ] {
        let pkcs8 = EcdsaKeyPair::generate_pkcs8(signing, &rng).expect("key");
        let pair = EcdsaKeyPair::from_pkcs8(signing, pkcs8.as_ref()).expect("key");
        let key = Jwk {
            kid: None,
            alg: Some(alg),
            material: KeyMaterial::Ec {
                curve,
                point: pair.public_key().as_ref().to_vec(),
            },
        };
        for i in 0..32u8 {
            let msg = [i; 7];
            let der = pair.sign(&rng, &msg).expect("sign");
            let fixed = ecdsa_der_to_fixed(curve, der.as_ref()).expect("aws-lc-rs DER");
            assert_eq!(fixed.len(), 2 * curve.coordinate_len());
            assert!(verify_signature(alg, &key, &msg, &fixed), "{alg:?} {i}");
        }
    }
}
