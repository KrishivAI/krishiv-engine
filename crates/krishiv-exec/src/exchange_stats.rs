// SPDX-License-Identifier: Apache-2.0
//! [`ExchangeStats`]: what an exchange's map tasks wrote (design P2-T174
//! §2.3), the input of re-planning at materialized stages (T164, T166).
//!
//! Each map task reports rows and bytes per output partition and a
//! distinct-key sketch of its keys' hashes; the stage merges them. The
//! sketch's precision is fixed ([`EXCHANGE_HLL_BITS`]), so any two reports
//! merge. On one worker an exchange below an aggregate runs inside the
//! aggregate's breaker (`physical::fused_aggregate`): its map tasks are the
//! breaker's drivers, its partitions the drivers' radix partitions and its
//! rows the partial-state rows each driver's table holds (once per spill and
//! at the end of input), as a shuffle would carry them.

use crate::rows::Hll;

/// Register bits of an exchange's key sketch: 1 KiB, 3.3% standard error,
/// the same as the row tables' partition sketches (owner, 2026-10-25).
pub const EXCHANGE_HLL_BITS: u32 = 10;

/// Rows and bytes one output partition of an exchange received.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PartitionStats {
    /// Rows.
    pub rows: u64,
    /// Bytes, as the writer measured them (in memory on one worker).
    pub bytes: u64,
}

/// What the map tasks of one exchange wrote, merged.
#[derive(Clone, Debug)]
pub struct ExchangeStats {
    partitions: Vec<PartitionStats>,
    keys: Hll,
    map_tasks: u64,
}

impl ExchangeStats {
    /// No rows yet, over `partitions` output partitions.
    pub(crate) fn new(partitions: usize) -> Self {
        Self {
            partitions: vec![PartitionStats::default(); partitions.max(1)],
            keys: Hll::new(EXCHANGE_HLL_BITS),
            map_tasks: 0,
        }
    }

    /// Counts one row of `bytes` whose key hashes to `h`, in the partition
    /// of `h`'s top bits.
    #[inline]
    pub(crate) fn add(&mut self, h: u64, bytes: u64) {
        let n = self.partitions.len();
        // `n` is a power of two up to 2^16 on every caller; others fold.
        let at = if n.is_power_of_two() && n > 1 {
            usize::try_from(h >> (64 - n.trailing_zeros())).unwrap_or(0)
        } else {
            usize::try_from(h).unwrap_or(0) % n
        };
        if let Some(p) = self.partitions.get_mut(at) {
            p.rows += 1;
            p.bytes += bytes;
        }
        self.keys.add(h);
    }

    /// Counts `rows` rows of `bytes` in all into partition `p` (whose keys
    /// `keys` sketches, at [`EXCHANGE_HLL_BITS`]): a table that already
    /// sketches its radix partitions reports them without a walk.
    pub(crate) fn add_partition(&mut self, p: usize, rows: u64, bytes: u64, keys: &Hll) {
        if let Some(at) = self.partitions.get_mut(p) {
            at.rows += rows;
            at.bytes += bytes;
        }
        self.keys.merge(keys);
    }

    /// The partitions this report counts into.
    pub(crate) fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    /// Ends one map task's report.
    pub(crate) fn end_task(&mut self) {
        self.map_tasks += 1;
    }

    /// Adds `other`'s reports (same partition count; other partitions are
    /// ignored).
    pub fn merge(&mut self, other: &Self) {
        for (a, b) in self.partitions.iter_mut().zip(&other.partitions) {
            a.rows += b.rows;
            a.bytes += b.bytes;
        }
        self.keys.merge(&other.keys);
        self.map_tasks += other.map_tasks;
    }

    /// Per output partition.
    #[must_use]
    pub fn partitions(&self) -> &[PartitionStats] {
        &self.partitions
    }

    /// Rows over every partition.
    #[must_use]
    pub fn rows(&self) -> u64 {
        self.partitions.iter().map(|p| p.rows).sum()
    }

    /// Bytes over every partition.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.partitions.iter().map(|p| p.bytes).sum()
    }

    /// The estimated distinct keys (0 with no rows).
    #[must_use]
    pub fn distinct_keys(&self) -> u64 {
        if self.rows() == 0 {
            return 0;
        }
        // An estimate, never above the rows counted.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a non-negative estimate far below u64::MAX"
        )]
        let keys = self.keys.estimate().round() as u64;
        keys.clamp(1, self.rows())
    }

    /// The map tasks that reported.
    #[must_use]
    pub fn map_tasks(&self) -> u64 {
        self.map_tasks
    }

    /// `EXPLAIN ANALYZE`'s counters, `exchange_` first.
    pub(crate) fn metrics(&self) -> crate::metrics::ExtraMetrics {
        let max = self.partitions.iter().map(|p| p.rows).max().unwrap_or(0);
        let used = self.partitions.iter().filter(|p| p.rows > 0).count() as u64;
        crate::metrics::counts([
            ("exchange_rows", self.rows()),
            ("exchange_bytes", self.bytes()),
            ("exchange_keys", self.distinct_keys()),
            ("exchange_map_tasks", self.map_tasks),
            ("exchange_partitions", used),
            ("exchange_max_partition_rows", max),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64's finalizer: a well-spread hash of `i`.
    fn spread(i: u64) -> u64 {
        let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[test]
    fn rows_land_in_the_partition_of_their_top_bits_and_reports_merge() {
        let mut a = ExchangeStats::new(16);
        a.add(spread(1) >> 4, 10);
        a.add(u64::MAX, 4);
        a.add((1 << 60) | (spread(2) >> 8), 1);
        a.end_task();
        assert_eq!(a.partitions()[0].rows, 1);
        assert_eq!(a.partitions()[1].rows, 1);
        assert_eq!(a.partitions()[15], PartitionStats { rows: 1, bytes: 4 });
        let mut b = ExchangeStats::new(16);
        b.add(u64::MAX, 6);
        b.end_task();
        a.merge(&b);
        assert_eq!((a.rows(), a.bytes(), a.map_tasks()), (4, 21, 2));
        assert_eq!(a.partitions()[15], PartitionStats { rows: 2, bytes: 10 });
        assert_eq!(a.distinct_keys(), 3);
        assert_eq!(ExchangeStats::new(4).distinct_keys(), 0);
    }

    #[test]
    fn the_key_sketch_counts_keys_seen_by_several_tasks_once() {
        let mut total = ExchangeStats::new(16);
        for task in 0..4_u64 {
            let mut s = ExchangeStats::new(16);
            // Every task sees keys 0..50_000, and its own 10_000.
            for i in (0..50_000).chain(100_000 * (task + 1)..100_000 * (task + 1) + 10_000) {
                s.add(spread(i), 8);
            }
            s.end_task();
            total.merge(&s);
        }
        assert_eq!(total.rows(), 240_000);
        let keys = total.distinct_keys() as f64;
        assert!((keys / 90_000.0 - 1.0).abs() < 0.1, "{keys}");
        let max = total.partitions().iter().map(|p| p.rows).max().unwrap();
        assert!(max < 240_000 / 16 * 2, "{max}");
    }

    #[test]
    fn one_partition_takes_every_row() {
        let mut s = ExchangeStats::new(1);
        s.add(u64::MAX, 1);
        s.add(0, 1);
        assert_eq!(s.partitions().len(), 1);
        assert_eq!(s.partitions()[0].rows, 2);
        let m = s.metrics();
        assert!(m.iter().any(|(n, _)| n == "exchange_partitions"));
    }
}
