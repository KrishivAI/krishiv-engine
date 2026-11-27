// SPDX-License-Identifier: Apache-2.0
//! A DER ECDSA signature (`SEQUENCE { INTEGER r, INTEGER s }`, RFC 3279 §2.2.3: what a key
//! service returns) turned into the fixed `r || s` form of a JWS (RFC 7518 §3.4).
//!
//! Strict DER only: short-form lengths (every valid signature here is under 128 bytes),
//! minimal positive integers, no trailing bytes. A signer needs this; the verifier never
//! takes DER.

use crate::{Curve, JoseError};

const SEQUENCE: u8 = 0x30;
const INTEGER: u8 = 0x02;

/// The fixed-width `r || s` of a DER-encoded ECDSA signature on `curve`: each number
/// big-endian, left-padded to the curve's coordinate length (32 bytes for P-256, 48 for P-384).
///
/// # Errors
/// [`JoseError::Invalid`] for anything but one DER `SEQUENCE` of exactly two `INTEGER`s with
/// nothing after it: another tag, a long-form or overrunning length, an empty, negative,
/// non-minimal or zero integer, or one wider than the curve's coordinate.
pub fn ecdsa_der_to_fixed(curve: Curve, der: &[u8]) -> Result<Vec<u8>, JoseError> {
    let body = element(der, SEQUENCE)?;
    if !body.rest.is_empty() {
        return Err(JoseError::Invalid("bytes after the signature"));
    }
    let r = element(body.content, INTEGER)?;
    let s = element(r.rest, INTEGER)?;
    if !s.rest.is_empty() {
        return Err(JoseError::Invalid("bytes after the signature numbers"));
    }
    let len = curve.coordinate_len();
    let mut out = vec![0; 2 * len];
    for (half, n) in out.chunks_mut(len).zip([r.content, s.content]) {
        let n = unsigned(n)?;
        if n.len() > len {
            return Err(JoseError::Invalid(
                "signature number too wide for the curve",
            ));
        }
        half[len - n.len()..].copy_from_slice(n);
    }
    Ok(out)
}

/// One DER element and what follows it.
struct Element<'a> {
    content: &'a [u8],
    rest: &'a [u8],
}

/// The element with `tag` at the start of `bytes`, with a short-form length.
fn element(bytes: &[u8], tag: u8) -> Result<Element<'_>, JoseError> {
    let [t, len, rest @ ..] = bytes else {
        return Err(JoseError::Invalid("DER element too short"));
    };
    if *t != tag {
        return Err(JoseError::Invalid("unexpected DER tag"));
    }
    // A long-form length (top bit set) is never minimal for these sizes.
    if len & 0x80 != 0 {
        return Err(JoseError::Invalid("DER long-form length"));
    }
    let len = usize::from(*len);
    if rest.len() < len {
        return Err(JoseError::Invalid("DER length beyond the input"));
    }
    let (content, rest) = rest.split_at(len);
    Ok(Element { content, rest })
}

/// The magnitude of a minimal, positive, non-zero DER integer, without its sign byte.
fn unsigned(n: &[u8]) -> Result<&[u8], JoseError> {
    match n {
        [] => Err(JoseError::Invalid("empty DER integer")),
        [b, ..] if b & 0x80 != 0 => Err(JoseError::Invalid("negative DER integer")),
        [0] => Err(JoseError::Invalid("zero signature number")),
        [0, b, ..] if b & 0x80 == 0 => Err(JoseError::Invalid("non-minimal DER integer")),
        [0, magnitude @ ..] => Ok(magnitude),
        _ => Ok(n),
    }
}
