// SPDX-License-Identifier: Apache-2.0
//! `shuffle(array[, seed])`: the elements of each array in random order.
//!
//! Ground truth: Spark 4.2.0's `Shuffle` (`collectionOperations.scala`) and
//! `RandomIndicesGenerator` (`catalyst/util/RandomIndicesGenerator.scala`),
//! pinned with a real Spark 4.2.0 run (see the tests). Partition `p` owns one
//! generator, a commons-math 3 `MersenneTwister` seeded with `seed + p`, and
//! every non-NULL array draws `length` values from it with the "inside-out"
//! Fisher-Yates shuffle (`nextInt(i + 1)` for position `i`). A NULL array
//! draws nothing; an array of one element still draws once. The seed is an
//! `INT`/`BIGINT` literal; an absent or NULL seed draws from the query seed.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use krishiv_common::error::KrishivError;

use super::args::internal;
use super::array_edit::{MAX_ARRAY_LENGTH, list_of, rebuild, span};
use super::random::unseeded_seed;
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["shuffle"];

/// The state size of MT19937.
const N: usize = 624;
/// The shift size.
const M: usize = 397;
/// `MAG01`.
const MAG01: [u32; 2] = [0x0, 0x9908_b0df];

/// Apache Commons Math 3.6.1's `MersenneTwister` (MT19937), seeded as it
/// seeds a `long`: `setSeed(new int[] {high, low})`.
struct MersenneTwister {
    mt: Box<[u32; N]>,
    mti: usize,
}

impl MersenneTwister {
    fn new(seed: i64) -> Self {
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
            // `seed[j]` is an `int` and sign-extends in Java's `long` sum.
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

    /// `BitsStreamGenerator.nextInt(int n)` for `n > 0`.
    fn next_int(&mut self, n: u32) -> u32 {
        if n.isolate_lowest_one() == n {
            // A power of two: `(int) ((n * (long) next(31)) >> 31)`.
            let high = u64::from(n) * u64::from(self.next_u32() >> 1);
            return u32::try_from(high >> 31).unwrap_or(0);
        }
        loop {
            let bits = self.next_u32() >> 1;
            let val = bits % n;
            // Java's `bits - val + (n - 1) < 0` in `int` arithmetic.
            if bits.wrapping_sub(val).wrapping_add(n - 1).cast_signed() >= 0 {
                return val;
            }
        }
    }

    /// `RandomIndicesGenerator.getNextIndices(length)`: the inside-out
    /// Fisher-Yates permutation.
    fn next_indices(&mut self, length: usize) -> Result<Vec<u32>, KrishivError> {
        let mut indices: Vec<u32> = Vec::new();
        indices
            .try_reserve(length)
            .map_err(|_| internal(format!("a shuffle of {length} elements: out of memory")))?;
        indices.resize(length, 0);
        for i in 0..length {
            let at = u32::try_from(i).map_err(|_| internal("a shuffled array past 2^32".into()))?;
            let j = self.next_int(at + 1);
            let j = usize::try_from(j).unwrap_or(i);
            if j != i {
                indices[i] = indices[j];
            }
            indices[j] = at;
        }
        Ok(indices)
    }
}

/// The folded seed (`None` for a NULL literal).
fn seed_of(c: &ArrayRef) -> Result<Option<i64>, KrishivError> {
    if c.data_type() == &DataType::Null || c.is_null(0) {
        return Ok(None);
    }
    match c.data_type() {
        DataType::Int32 => Ok(Some(i64::from(c.as_primitive::<Int32Type>().value(0)))),
        DataType::Int64 => Ok(Some(c.as_primitive::<Int64Type>().value(0))),
        t => Err(internal(format!("a shuffle seed of type {t}"))),
    }
}

/// Binds `shuffle(array)` and `shuffle(array, seed)`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let ty = match types.as_slice() {
        [t @ DataType::List(_)]
        | [
            t @ DataType::List(_),
            DataType::Int32 | DataType::Int64 | DataType::Null,
        ] => t.clone(),
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    let DataType::List(field) = &ty else {
        return Some((listed, Err(bad_types(listed, args))));
    };
    let field = Arc::clone(field);
    let seeded = args.len() > 1;
    // Fixed per binding: an absent or NULL seed draws from the query seed.
    let queries = unseeded_seed(opts);
    let state: Mutex<HashMap<u32, MersenneTwister>> = Mutex::new(HashMap::new());
    let binding = Binding::partitioned(ty, move |x, partition| {
        let seed = if seeded {
            seed_of(&x.column(1)?)?.unwrap_or(queries)
        } else {
            queries
        };
        let c = x.column(0)?;
        let list = list_of(&c, "shuffle")?;
        let mut streams = state
            .lock()
            .map_err(|_| internal("a shuffle's generators are poisoned".to_owned()))?;
        let rng = streams
            .entry(partition)
            .or_insert_with(|| MersenneTwister::new(seed.wrapping_add(i64::from(partition))));
        let mut rows = Vec::with_capacity(x.rows());
        for row in 0..x.rows() {
            if list.is_null(row) {
                rows.push(None);
                continue;
            }
            let (start, len) = span(list, row);
            let picked = rng
                .next_indices(len)?
                .into_iter()
                .map(|k| u32::try_from(start + k as usize).ok())
                .collect::<Vec<_>>();
            if picked.iter().any(Option::is_none) {
                return Err(internal("an array past 2^32 elements".to_owned()));
            }
            rows.push(Some(picked));
        }
        rebuild(&field, list.values(), &rows, "shuffle", MAX_ARRAY_LENGTH)
    });
    Some((listed, Ok(binding)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spark 4.2.0 `shuffle(array(1..10), seed)`, one partition.
    #[test]
    fn spark_seeded_permutations() {
        let cases: [(i64, [u32; 10]); 4] = [
            (0, [4, 1, 2, 9, 7, 10, 5, 3, 6, 8]),
            (42, [6, 8, 2, 1, 4, 7, 9, 3, 5, 10]),
            (-7, [5, 2, 1, 8, 10, 4, 3, 7, 9, 6]),
            (1_234_567_890_123, [10, 7, 2, 8, 5, 3, 6, 9, 1, 4]),
        ];
        for (seed, want) in cases {
            let idx = MersenneTwister::new(seed).next_indices(10).unwrap();
            let got: Vec<u32> = idx.iter().map(|i| i + 1).collect();
            assert_eq!(got, want, "seed {seed}");
        }
    }

    /// `range(0, 6, 1, 3)` with `shuffle(array(1..6), 100)`: partition `p`
    /// seeds `100 + p` and its two rows draw consecutively.
    #[test]
    fn partitions_offset_the_seed_and_rows_share_a_stream() {
        // Spark's rows: id 0,1 -> partition 0; 2,3 -> 1; 4,5 -> 2.
        let spark: [[u32; 6]; 6] = [
            [2, 4, 1, 6, 3, 5],
            [3, 4, 5, 1, 2, 6],
            [4, 1, 2, 3, 5, 6],
            [4, 2, 3, 5, 6, 1],
            [2, 1, 4, 5, 3, 6],
            [4, 3, 2, 1, 5, 6],
        ];
        for p in 0..3_usize {
            let mut g = MersenneTwister::new(100 + p as i64);
            for r in 0..2 {
                let got: Vec<u32> = g.next_indices(6).unwrap().iter().map(|i| i + 1).collect();
                assert_eq!(got, spark[p * 2 + r], "partition {p} row {r}");
            }
        }
    }
}
