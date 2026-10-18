// SPDX-License-Identifier: Apache-2.0
//! XXH64 as Spark's `xxhash64` uses it (seed 42). `hashInt` and `hashLong`
//! equal XXH64 of the 4 or 8 little-endian bytes, written out here without
//! the buffering.

use super::SparkHasher;

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P3: u64 = 0x1656_67B1_9E37_79F9;
const P4: u64 = 0x85EB_CA77_C2B2_AE63;
const P5: u64 = 0x27D4_EB2F_1656_67C5;

fn fmix(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^ (h >> 32)
}

fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(P2))
        .rotate_left(31)
        .wrapping_mul(P1)
}

fn merge(acc: u64, v: u64) -> u64 {
    (acc ^ round(0, v)).wrapping_mul(P1).wrapping_add(P4)
}

fn le64(b: &[u8]) -> u64 {
    let mut a = [0_u8; 8];
    a.copy_from_slice(&b[..8]);
    u64::from_le_bytes(a)
}

fn le32(b: &[u8]) -> u64 {
    let mut a = [0_u8; 4];
    a.copy_from_slice(&b[..4]);
    u64::from(u32::from_le_bytes(a))
}

/// Spark's `xxhash64()`: XXH64 with seed 42, signed.
#[derive(Debug, Clone, Copy)]
pub(super) struct Xxh64;

impl SparkHasher for Xxh64 {
    type State = i64;
    const SEED: i64 = 42;

    fn hash_int(v: i32, seed: i64) -> i64 {
        let mut h = seed.cast_unsigned().wrapping_add(P5).wrapping_add(4);
        h ^= u64::from(v.cast_unsigned()).wrapping_mul(P1);
        h = h.rotate_left(23).wrapping_mul(P2).wrapping_add(P3);
        fmix(h).cast_signed()
    }

    fn hash_long(v: i64, seed: i64) -> i64 {
        let mut h = seed.cast_unsigned().wrapping_add(P5).wrapping_add(8);
        h ^= round(0, v.cast_unsigned());
        h = h.rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
        fmix(h).cast_signed()
    }

    fn hash_bytes(bytes: &[u8], seed: i64) -> i64 {
        let seed = seed.cast_unsigned();
        let mut rest = bytes;
        let mut h = if bytes.len() >= 32 {
            let mut v = [
                seed.wrapping_add(P1).wrapping_add(P2),
                seed.wrapping_add(P2),
                seed,
                seed.wrapping_sub(P1),
            ];
            while rest.len() >= 32 {
                let (block, after) = rest.split_at(32);
                let (lanes, _) = block.as_chunks::<8>();
                for (acc, lane) in v.iter_mut().zip(lanes) {
                    *acc = round(*acc, u64::from_le_bytes(*lane));
                }
                rest = after;
            }
            let mut h = v[0]
                .rotate_left(1)
                .wrapping_add(v[1].rotate_left(7))
                .wrapping_add(v[2].rotate_left(12))
                .wrapping_add(v[3].rotate_left(18));
            for acc in v {
                h = merge(h, acc);
            }
            h
        } else {
            seed.wrapping_add(P5)
        };
        h = h.wrapping_add(bytes.len() as u64);
        while rest.len() >= 8 {
            h = (h ^ round(0, le64(rest)))
                .rotate_left(27)
                .wrapping_mul(P1)
                .wrapping_add(P4);
            rest = &rest[8..];
        }
        if rest.len() >= 4 {
            h = (h ^ le32(rest).wrapping_mul(P1))
                .rotate_left(23)
                .wrapping_mul(P2)
                .wrapping_add(P3);
            rest = &rest[4..];
        }
        for &b in rest {
            h = (h ^ u64::from(b).wrapping_mul(P5))
                .rotate_left(11)
                .wrapping_mul(P1);
        }
        fmix(h).cast_signed()
    }
}
