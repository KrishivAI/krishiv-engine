// SPDX-License-Identifier: Apache-2.0
//! Minimal base64 for JSON `BINARY` values: `to_json` renders binary as
//! base64 text and `from_json` decodes it, as Spark's `JacksonGenerator`
//! and `JacksonParser` do.

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// An alphabet index as a `usize` (the caller masks to six bits).
fn idx(v: u32) -> usize {
    usize::try_from(v & 63).unwrap_or(0)
}

/// base64-encodes `bytes` (standard alphabet, `=` padding).
pub(crate) fn encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3).saturating_mul(4));
    for chunk in bytes.chunks(3) {
        let a = u32::from(chunk[0]);
        let b = chunk.get(1).copied().map_or(0, u32::from);
        let c = chunk.get(2).copied().map_or(0, u32::from);
        let n = a
            .wrapping_shl(16)
            .wrapping_add(b.wrapping_shl(8))
            .wrapping_add(c);
        out.push(ALPHABET[idx(n.wrapping_shr(18))] as char);
        out.push(ALPHABET[idx(n.wrapping_shr(12))] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[idx(n.wrapping_shr(6))] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[idx(n)] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// base64-decodes `text` (standard alphabet, padding required); `None` for
/// anything else.
pub(crate) fn decode(text: &str) -> Option<Vec<u8>> {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return Some(Vec::new());
    }
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len().div_euclid(4).saturating_mul(3));
    for chunk in bytes.chunks(4) {
        let mut n = 0_u32;
        let mut pad = 0_usize;
        for (i, &b) in chunk.iter().enumerate() {
            let v = match b {
                b'A'..=b'Z' => u32::from(b.checked_sub(b'A')?),
                b'a'..=b'z' => u32::from(b.checked_sub(b'a')?).checked_add(26)?,
                b'0'..=b'9' => u32::from(b.checked_sub(b'0')?).checked_add(52)?,
                b'+' => 62,
                b'/' => 63,
                b'=' if i >= 2 => {
                    pad = pad.saturating_add(1);
                    0
                }
                _ => return None,
            };
            n = n.wrapping_shl(6).wrapping_add(v);
        }
        if pad > 2 || (pad > 0 && chunk[3] != b'=') || (pad == 2 && chunk[2] != b'=') {
            return None;
        }
        out.push(u8::try_from(n.wrapping_shr(16)).ok()?);
        if pad < 2 {
            out.push(u8::try_from(n.wrapping_shr(8) & 0xFF).ok()?);
        }
        if pad == 0 {
            out.push(u8::try_from(n & 0xFF).ok()?);
        }
    }
    Some(out)
}
