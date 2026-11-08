// SPDX-License-Identifier: Apache-2.0
//! The Mersenne Twister of Apache Commons Math 3.6.1
//! (`org.apache.commons.math3.random.MersenneTwister`, MT19937) that Spark's
//! `RandomUUIDGenerator` draws `uuid`'s bits from, and that generator.
//!
//! Ported operation for operation from commons-math's `MersenneTwister`
//! (`setSeed(long)` through `setSeed(int[])`, `next(int)`) and
//! `BitsStreamGenerator.nextLong` (two 32-bit draws, high word first), so a
//! seed gives Spark's values exactly.

/// The state size.
const N: usize = 624;
/// The shift size.
const M: usize = 397;
/// `MAG01`.
const MAG01: [u32; 2] = [0x0, 0x9908_b0df];

/// One MT19937 generator, seeded as commons-math seeds a `long`.
pub(super) struct MersenneTwister {
    mt: Box<[u32; N]>,
    mti: usize,
}

impl MersenneTwister {
    /// `new MersenneTwister(long seed)`: `setSeed(new int[] {high, low})`.
    pub(super) fn new(seed: i64) -> Self {
        let bits = seed.cast_unsigned();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Java's (int) of the high and low halves"
        )]
        let key = [(bits >> 32) as u32, bits as u32];
        let mut g = Self {
            mt: Box::new([0; N]),
            mti: N,
        };
        g.seed_array(&key);
        g
    }

    /// `setSeed(int seed)`.
    fn seed_int(&mut self, seed: u32) {
        let mut long_mt = u64::from(seed);
        self.mt[0] = seed;
        for i in 1..N {
            // Java's `long` arithmetic masked to 32 bits.
            long_mt = (1_812_433_253_u64
                .wrapping_mul(long_mt ^ (long_mt >> 30))
                .wrapping_add(i as u64))
                & 0xffff_ffff;
            #[expect(clippy::cast_possible_truncation, reason = "masked to 32 bits")]
            {
                self.mt[i] = long_mt as u32;
            }
        }
        self.mti = N;
    }

    /// `setSeed(int[] seed)`.
    fn seed_array(&mut self, key: &[u32]) {
        self.seed_int(19_650_218);
        let (mut i, mut j) = (1_usize, 0_usize);
        for _ in 0..N.max(key.len()) {
            let l0 = u64::from(self.mt[i]);
            let l1 = u64::from(self.mt[i - 1]);
            // `l0 ^ ((l1 ^ (l1 >> 30)) * 1664525) + seed[j] + j`, in Java's
            // signed `long`: `seed[j]` is an `int` and sign-extends.
            let l = (l0 ^ (l1 ^ (l1 >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(i64::from(key[j].cast_signed()).cast_unsigned())
                .wrapping_add(j as u64);
            self.mt[i] = (l & 0xffff_ffff) as u32;
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let l0 = u64::from(self.mt[i]);
            let l1 = u64::from(self.mt[i - 1]);
            let l = (l0 ^ (l1 ^ (l1 >> 30)).wrapping_mul(1_566_083_941)).wrapping_sub(i as u64);
            self.mt[i] = (l & 0xffff_ffff) as u32;
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000;
        self.mti = N;
    }

    /// `next(32)`: the next 32 tempered bits.
    fn next_u32(&mut self) -> u32 {
        if self.mti >= N {
            let mt = &mut self.mt;
            for k in 0..N {
                let y = (mt[k] & 0x8000_0000) | (mt[(k + 1) % N] & 0x7fff_ffff);
                mt[k] = mt[(k + M) % N] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
            }
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `BitsStreamGenerator.nextLong`: the high word, then the low word.
    pub(super) fn next_long(&mut self) -> u64 {
        let high = u64::from(self.next_u32()) << 32;
        high | u64::from(self.next_u32())
    }
}

/// Spark's `RandomUUIDGenerator.getNextUUID().toString()`: two `nextLong`
/// draws with the RFC 4122 version 4 and variant bits set.
pub(super) fn next_uuid(g: &mut MersenneTwister) -> String {
    let most = (g.next_long() & 0xffff_ffff_ffff_0fff) | 0x0000_0000_0000_4000;
    let least = (g.next_long() | 0x8000_0000_0000_0000) & 0xbfff_ffff_ffff_ffff;
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        most >> 32,
        (most >> 16) & 0xffff,
        most & 0xffff,
        least >> 48,
        least & 0xffff_ffff_ffff,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spark 4.2.0 `SELECT uuid(82374)`, `uuid(237685)` (one partition, so
    /// the generator's seed is the argument), and the first draws of the
    /// reference MT19937 for commons-math's `long` seeding.
    #[test]
    fn spark_seeded_uuids() {
        assert_eq!(
            next_uuid(&mut MersenneTwister::new(82_374)),
            "8b88900e-f862-468c-8d3b-828188116155"
        );
        assert_eq!(
            next_uuid(&mut MersenneTwister::new(237_685)),
            "19974dca-21f6-47ef-b58c-73908ab52aa0"
        );
        // Partition 1 of seed 237685 (the second round of `cte-recursion`
        // case 84).
        assert_eq!(
            next_uuid(&mut MersenneTwister::new(237_686)),
            "d0032efe-ae60-461b-8582-f6a7c649f238"
        );
    }

    /// Many draws cross the state refill several times.
    #[test]
    fn refills_are_stable() {
        let mut g = MersenneTwister::new(-1);
        let first: Vec<u64> = (0..2_000).map(|_| g.next_long()).collect();
        let mut again = MersenneTwister::new(-1);
        assert!(first.iter().all(|&v| v == again.next_long()));
        assert_ne!(first[0], first[700]);
    }
}
