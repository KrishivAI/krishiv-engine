// SPDX-License-Identifier: Apache-2.0
//! Murmur3 x86_32 as Spark runs it (`Murmur3_x86_32`): the tail bytes of a
//! byte string are mixed one sign-extended byte at a time, not as one
//! partial word, so this is not the reference algorithm for lengths that are
//! not multiples of four.

use super::SparkHasher;

const C1: u32 = 0xcc9e_2d51;
const C2: u32 = 0x1b87_3593;

fn mix_k1(k1: u32) -> u32 {
    k1.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2)
}

fn mix_h1(h1: u32, k1: u32) -> u32 {
    (h1 ^ k1)
        .rotate_left(13)
        .wrapping_mul(5)
        .wrapping_add(0xe654_6b64)
}

fn fmix(h1: u32, len: u32) -> u32 {
    let mut h = h1 ^ len;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

/// Spark's `hash()`: Murmur3 x86_32 with seed 42, signed.
#[derive(Debug, Clone, Copy)]
pub(super) struct Murmur3;

impl SparkHasher for Murmur3 {
    type State = i32;
    const SEED: i32 = 42;

    fn hash_int(v: i32, seed: i32) -> i32 {
        let h1 = mix_h1(seed.cast_unsigned(), mix_k1(v.cast_unsigned()));
        fmix(h1, 4).cast_signed()
    }

    fn hash_long(v: i64, seed: i32) -> i32 {
        let bits = v.cast_unsigned();
        // The low then the high 32 bits.
        let low = u32::try_from(bits & 0xffff_ffff).unwrap_or_default();
        let high = u32::try_from(bits >> 32).unwrap_or_default();
        let h1 = mix_h1(seed.cast_unsigned(), mix_k1(low));
        let h1 = mix_h1(h1, mix_k1(high));
        fmix(h1, 8).cast_signed()
    }

    fn hash_bytes(bytes: &[u8], seed: i32) -> i32 {
        let mut h1 = seed.cast_unsigned();
        let (words, tail) = bytes.as_chunks::<4>();
        for w in words {
            h1 = mix_h1(h1, mix_k1(u32::from_le_bytes(*w)));
        }
        for &b in tail {
            // `Platform.getByte` is signed and widened to int.
            h1 = mix_h1(h1, mix_k1(i32::from(b.cast_signed()).cast_unsigned()));
        }
        // Spark's length is an int; Arrow offsets keep it below 2^31 (i32) or
        // far above any real value (i64).
        fmix(h1, u32::try_from(bytes.len()).unwrap_or(u32::MAX)).cast_signed()
    }
}
