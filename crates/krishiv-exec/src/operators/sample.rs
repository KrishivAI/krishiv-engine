// SPDX-License-Identifier: Apache-2.0
//! `TABLESAMPLE`: Spark's `SampleExec` without replacement (a
//! `BernoulliCellSampler` per partition over `XORShiftRandom`).
//!
//! Ground truth: `tests/fixtures/spark-4.2.0/tablesample.json` and its
//! reference port `tests/scripts/tablesample_ref.py`. Each row of input
//! partition `i` takes one `nextDouble()` of `XORShiftRandom(seed + i)` (a
//! wrapping 64-bit sum), in input order, and is kept when `lower <= x <
//! upper`; with `upper - lower <= 0` no row is drawn for or kept.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arrow_array::BooleanArray;
use arrow_schema::SchemaRef;
use krishiv_types::batch::Batch;

use super::{PartitionState, local_mut};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{LocalState, Operator, OperatorOutput};

/// Spark's `XORShiftRandom`: the seed hashed with Scala's
/// `MurmurHash3.bytesHash`, then a 64-bit xorshift register.
#[derive(Debug, Clone)]
pub(crate) struct XorShiftRandom {
    state: u64,
}

impl XorShiftRandom {
    /// `new XORShiftRandom(seed)`.
    pub(crate) fn new(seed: i64) -> Self {
        Self {
            state: hash_seed(seed),
        }
    }

    /// `next(bits)`.
    fn next(&mut self, bits: u32) -> u64 {
        let mut x = self.state;
        x ^= x << 21;
        x ^= x >> 35;
        x ^= x << 4;
        self.state = x;
        x & ((1_u64 << bits) - 1)
    }

    /// `java.util.Random.nextDouble`: 53 bits over `2^53`.
    #[expect(clippy::cast_precision_loss, reason = "53 bits convert to f64 exactly")]
    pub(crate) fn next_double(&mut self) -> f64 {
        let high = self.next(26);
        let low = self.next(27);
        ((high << 27) + low) as f64 / 9_007_199_254_740_992.0
    }
}

/// `XORShiftRandom.hashSeed`: the seed's big-endian bytes hashed twice.
fn hash_seed(seed: i64) -> u64 {
    let bytes = seed.to_be_bytes();
    let low = bytes_hash(&bytes, 0x3c07_4a61);
    let high = bytes_hash(&bytes, low);
    (u64::from(high) << 32) | u64::from(low)
}

/// Scala 2.13's `MurmurHash3.bytesHash` of 8 bytes (no tail).
fn bytes_hash(data: &[u8; 8], seed: u32) -> u32 {
    let mut h = seed;
    for chunk in data.as_chunks::<4>().0 {
        let k = u32::from_le_bytes(*chunk);
        h = mix_last(h, k)
            .rotate_left(13)
            .wrapping_mul(5)
            .wrapping_add(0xe654_6b64);
    }
    // `finalizeHash(h, 8)`: `avalanche(h ^ 8)`.
    h ^= 8;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

fn mix_last(h: u32, k: u32) -> u32 {
    h ^ k
        .wrapping_mul(0xcc9e_2d51)
        .rotate_left(15)
        .wrapping_mul(0x1b87_3593)
}

/// Keeps each row whose draw falls in `[lower, upper)`: the draws of one
/// input partition (the batch's split, else the driver's partition, as a
/// seeded `rand` reads it) come from one stream seeded `seed + partition`,
/// shared by the drivers so a partition's rows draw in order whichever
/// driver runs them.
#[derive(Debug)]
pub struct Sample {
    schema: SchemaRef,
    lower: f64,
    upper: f64,
    seed: i64,
    streams: Mutex<HashMap<u32, XorShiftRandom>>,
}

impl Sample {
    /// A sample of batches of `schema` keeping draws in `[lower, upper)`,
    /// partition `i` drawing from `XORShiftRandom(seed + i)`.
    pub fn new(schema: SchemaRef, lower: f64, upper: f64, seed: i64) -> Arc<Self> {
        Arc::new(Self {
            schema,
            lower,
            upper,
            seed,
            streams: Mutex::new(HashMap::new()),
        })
    }

    /// The keep mask of `rows` rows of `partition`, drawn in order.
    fn mask(&self, partition: u32, rows: usize) -> Result<BooleanArray> {
        let mut streams = self
            .streams
            .lock()
            .map_err(|_| internal("a sample's streams are poisoned".to_owned()))?;
        let rng = streams
            .entry(partition)
            .or_insert_with(|| XorShiftRandom::new(self.seed.wrapping_add(i64::from(partition))));
        Ok((0..rows)
            .map(|_| {
                let x = rng.next_double();
                Some(self.lower <= x && x < self.upper)
            })
            .collect())
    }
}

impl Operator for Sample {
    fn lane_safe(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "Sample"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(PartitionState {
            partition: ctx.partition(),
        }))
    }

    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput> {
        // `BernoulliCellSampler`: an empty range draws nothing, keeps nothing.
        if input.num_rows() == 0 || self.upper - self.lower <= 0.0 {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let partition = local_mut::<PartitionState>(local, "Sample")?.partition;
        let mask = self.mask(input.split().unwrap_or(partition), input.num_rows())?;
        let out = input.filter(&mask)?;
        if out.num_rows() == 0 {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let out = if out.num_rows() < input.num_rows() {
            super::filter::compact_sparse_views(input, out)?
        } else {
            out
        };
        Ok(OperatorOutput::NeedMoreInput(Some(out)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../../../../tests/fixtures/spark-4.2.0/tablesample.json"
        ))
        .unwrap()
    }

    /// The fixture's `XORShiftRandom` vectors: the hashed seed and the
    /// first ten doubles, bit for bit.
    #[test]
    fn xorshift_matches_spark() {
        let doc = fixture();
        let rng = doc["rng"].as_object().unwrap();
        assert_eq!(rng.len(), 5);
        for (seed, v) in rng {
            let seed: i64 = seed.parse().unwrap();
            assert_eq!(
                hash_seed(seed).cast_signed(),
                v["hash_seed"].as_i64().unwrap(),
                "{seed}"
            );
            let mut r = XorShiftRandom::new(seed);
            for bits in v["first10_double_bits_hex"].as_array().unwrap() {
                let want = u64::from_str_radix(bits.as_str().unwrap(), 16).unwrap();
                assert_eq!(r.next_double().to_bits(), want, "{seed}");
            }
        }
    }

    /// The sampler over the fixture's one-partition `range(0, 1000)` cases
    /// (`view_*`): Spark's ids, partition 0 seeded with the seed.
    #[test]
    fn one_partition_cases_match_spark() {
        let doc = fixture();
        let mut checked = 0;
        for case in doc["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let sql = case["sql"].as_str().unwrap();
            if !name.starts_with("view_") || name.contains("bernoulli") {
                continue;
            }
            // `+- 'Sample 0.0, 0.1, false, 7, Bernoulli`.
            let fields: Vec<&str> = case["parsed_sample"]
                .as_str()
                .unwrap()
                .trim_start_matches("+- 'Sample ")
                .split(", ")
                .collect();
            let (lower, upper): (f64, f64) =
                (fields[0].parse().unwrap(), fields[1].parse().unwrap());
            let seed: i64 = fields[3].parse().unwrap();
            let s = Sample::new(Arc::new(arrow_schema::Schema::empty()), lower, upper, seed);
            let mask = if upper - lower <= 0.0 {
                BooleanArray::from(vec![false; 1000])
            } else {
                s.mask(0, 1000).unwrap()
            };
            let ids: Vec<i64> = (0..1000_usize)
                .filter(|&i| mask.value(i))
                .map(|i| i64::try_from(i).unwrap())
                .collect();
            let want: Vec<i64> = case["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            assert_eq!(ids, want, "{sql}");
            checked += 1;
        }
        assert!(checked >= 10, "{checked}");
    }

    /// One stream per partition, continued across batches: two masks of 500
    /// rows are the mask of 1000.
    #[test]
    fn a_partition_stream_continues_across_batches() {
        let schema = Arc::new(arrow_schema::Schema::empty());
        let whole = Sample::new(Arc::clone(&schema), 0.0, 0.5, 3)
            .mask(2, 1000)
            .unwrap();
        let split = Sample::new(schema, 0.0, 0.5, 3);
        let a = split.mask(2, 500).unwrap();
        let b = split.mask(2, 500).unwrap();
        let joined: Vec<bool> = a.iter().chain(b.iter()).map(Option::unwrap).collect();
        let whole: Vec<bool> = whole.iter().map(Option::unwrap).collect();
        assert_eq!(joined, whole);
    }
}
