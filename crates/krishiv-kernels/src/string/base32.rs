// SPDX-License-Identifier: Apache-2.0
//! `to_base32` and `from_base32` (Spark 4.3): RFC 4648 Base32 as Apache Commons Codec's `Base32` writes and reads it.
//!
//! Encoding pads to a multiple of eight symbols and never breaks lines. Decoding is lenient like Commons Codec's:
//! lower-case letters are accepted, a byte that is not a Base32 symbol (spaces, `!`, digits `0`, `1`, `8`, `9`, any
//! non-ASCII byte) is skipped, and the first `=` ends the input. A trailing group of 1 to 7 symbols decodes the
//! bytes its bits complete (one symbol gives `lbit >> 2`, as the library does); it is never an error.

#![expect(
    clippy::arithmetic_side_effects,
    reason = "the work area is at most 40 bits and every shift is below 64; indexes and counts are bounded by the group of 5 bytes or 8 symbols"
)]

use arrow_array::{Array, GenericBinaryArray, GenericStringArray, OffsetSizeTrait};

use super::{BinSink, StrSink, check_offset};
use crate::error::KernelError;

const ALPHABET: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

/// The 5-bit value of a Base32 symbol (either case), if it is one.
fn symbol(b: u8) -> Option<u64> {
    match b {
        b'A'..=b'Z' => Some(u64::from(b - b'A')),
        b'a'..=b'z' => Some(u64::from(b - b'a')),
        b'2'..=b'7' => Some(u64::from(b - b'2') + 26),
        _ => None,
    }
}

/// The Base32 text of `bytes`.
fn encode(bytes: &[u8], out: &mut String) {
    for group in bytes.chunks(5) {
        let n = group
            .iter()
            .enumerate()
            .fold(0_u64, |n, (i, b)| n | (u64::from(*b) << (32 - 8 * i)));
        // Five bytes make eight symbols; `g` bytes make `ceil(8 g / 5)` of them.
        let real = (group.len() * 8).div_ceil(5);
        for k in 0..8 {
            let shift = 35 - 5 * k;
            if k < real {
                let index = usize::try_from((n >> shift) & 31).unwrap_or(0);
                out.push(char::from(ALPHABET[index]));
            } else {
                out.push('=');
            }
        }
    }
}

/// The bytes `s` decodes to.
fn decode(s: &str, out: &mut Vec<u8>) {
    let (mut modulus, mut work) = (0_u32, 0_u64);
    for &b in s.as_bytes() {
        if b == b'=' {
            break;
        }
        let Some(v) = symbol(b) else { continue };
        modulus = (modulus + 1) % 8;
        work = (work << 5) + v;
        if modulus == 0 {
            out.extend((0..5).map(|i| ((work >> (32 - 8 * i)) & 0xff) as u8));
        }
    }
    // A partial group: the library's end-of-input switch on the symbols read.
    let (shift, bytes) = match modulus {
        1 | 2 => (2, 1),
        3 => (7, 1),
        4 => (4, 2),
        5 => (1, 3),
        6 => (6, 3),
        7 => (3, 4),
        _ => return,
    };
    let work = work >> shift;
    out.extend((0..bytes).map(|i| ((work >> (8 * (bytes - 1 - i))) & 0xff) as u8));
}

/// `to_base32(bin)`.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn to_base32<O: OffsetSizeTrait>(
    a: &GenericBinaryArray<O>,
) -> Result<GenericStringArray<O>, KernelError> {
    let mut sink = StrSink::<O>::new(a.len(), 0);
    let mut scratch = String::new();
    for i in 0..a.len() {
        if a.is_null(i) {
            sink.push_null();
            continue;
        }
        let bytes = a.value(i);
        let need = bytes.len().div_ceil(5).saturating_mul(8);
        check_offset::<O>(sink.bytes.saturating_add(need))?;
        scratch.clear();
        scratch
            .try_reserve(need)
            .map_err(|_| KernelError::out_of_memory(need))?;
        encode(bytes, &mut scratch);
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

/// `from_base32(str)`: never fails (see the module note).
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn from_base32<O: OffsetSizeTrait>(
    a: &GenericStringArray<O>,
) -> Result<GenericBinaryArray<O>, KernelError> {
    let mut sink = BinSink::<O>::new(a.len(), 0);
    let mut scratch = Vec::new();
    for i in 0..a.len() {
        if a.is_null(i) {
            sink.push_null();
            continue;
        }
        scratch.clear();
        decode(a.value(i), &mut scratch);
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(b: &[u8]) -> String {
        let mut s = String::new();
        encode(b, &mut s);
        s
    }

    fn dec(s: &str) -> Vec<u8> {
        let mut v = Vec::new();
        decode(s, &mut v);
        v
    }

    /// Values from a Spark 4.3.0 run (`to_base32`, `from_base32`).
    #[test]
    fn matches_spark_4_3_0() {
        for (b, want) in [
            (&b"f"[..], "MY======"),
            (&[0x66, 0x6f][..], "MZXQ===="),
            (b"foo", "MZXW6==="),
            (b"foob", "MZXW6YQ="),
            (b"fooba", "MZXW6YTB"),
            (b"foobar", "MZXW6YTBOI======"),
            (b"", ""),
            ("é".as_bytes(), "YOUQ===="),
        ] {
            assert_eq!(enc(b), want);
        }
        for (s, want) in [
            ("MZXW6===", &b"foo"[..]),
            ("mzxw6", b"foo"),
            ("MZXW6", b"foo"),
            ("MZ XW6===", b"foo"),
            ("MZXW6!!!", b"foo"),
            ("!!!", b""),
            ("M", &[0x03]),
            ("MZ", b"f"),
            ("MZX", b"f"),
            ("MZXW", &[0x66, 0x6f]),
            ("MZXW6", b"foo"),
            ("MZXW6Y", b"foo"),
            ("MZXW6YT", b"foob"),
            ("MZXW6YQ=", b"foob"),
            ("MZXW6YTBOI======", b"foobar"),
            ("MZXW6YTBOI", b"foobar"),
            ("A1", &[0x00]),
            ("ZZZZZZZZ", &[0xCE, 0x73, 0x9C, 0xE7, 0x39]),
            ("MY======MZXW6===", b"f"),
            ("=MZXW6", b""),
            ("7777777777777777", &[0xFF; 10]),
            ("", b""),
        ] {
            assert_eq!(dec(s), want, "{s:?}");
        }
    }
}
