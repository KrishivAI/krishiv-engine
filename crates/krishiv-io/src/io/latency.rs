// SPDX-License-Identifier: Apache-2.0
//! Store request latency, as a histogram of power-of-two microsecond buckets.
//!
//! Every read attempt the scheduler issues to a remote store (a ranged
//! `GET`, a metadata request; local files are not counted) is timed from
//! the moment it holds its lane's permit to its last byte, so the time
//! includes how long the I/O runtime took to poll it: a starved I/O runtime
//! shows as latency here (P2-T104 §3, E01's I/O check). Pacing waits and
//! retry backoff are not in it. Every attempt counts, a retried one
//! each time, and one that timed out as about the request timeout.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Buckets: bucket `i` counts requests that took less than `2^i` µs (and at
/// least `2^(i-1)` µs); the last one counts everything slower.
pub const LATENCY_BUCKETS: usize = 32;

/// Read request latencies (a fixed histogram: recording is one relaxed
/// atomic add, no lock).
#[derive(Debug)]
pub struct LatencyHistogram {
    counts: [AtomicU64; LATENCY_BUCKETS],
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self {
            counts: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl LatencyHistogram {
    /// Counts one request that took `d`.
    pub fn record(&self, d: Duration) {
        let us = u64::try_from(d.as_micros()).unwrap_or(u64::MAX);
        // Bit length: 0 µs and 1 µs land in buckets 0 and 1, 2-3 µs in 2.
        let bucket = (u64::BITS - us.leading_zeros()) as usize;
        self.counts[bucket.min(LATENCY_BUCKETS - 1)].fetch_add(1, Ordering::Relaxed);
    }

    /// The count of every bucket, in order.
    #[must_use]
    pub fn counts(&self) -> [u64; LATENCY_BUCKETS] {
        std::array::from_fn(|i| self.counts[i].load(Ordering::Relaxed))
    }

    /// The upper bound, in microseconds, of the bucket that holds quantile
    /// `q` (0 to 1) of `counts`, a value [`counts`](Self::counts) returned
    /// (or a difference of two); `None` when no request was counted.
    #[must_use]
    pub fn quantile_us(counts: &[u64; LATENCY_BUCKETS], q: f64) -> Option<u64> {
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return None;
        }
        // The rank of the quantile, 1-based: the smallest bucket whose
        // running count reaches it.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "a rank within a u64 count, clamped to [1, total]"
        )]
        let rank = ((q.clamp(0.0, 1.0) * total as f64).ceil() as u64).clamp(1, total);
        let mut seen = 0;
        for (i, &c) in counts.iter().enumerate() {
            seen += c;
            if seen >= rank {
                return Some(
                    1u64.checked_shl(u32::try_from(i).unwrap_or(63))
                        .unwrap_or(u64::MAX),
                );
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_are_powers_of_two_microseconds() {
        let h = LatencyHistogram::default();
        h.record(Duration::from_micros(0));
        h.record(Duration::from_micros(1));
        h.record(Duration::from_micros(3));
        h.record(Duration::from_millis(5)); // 5,000 µs: below 2^13
        h.record(Duration::from_secs(1 << 40));
        let c = h.counts();
        assert_eq!((c[0], c[1], c[2], c[13]), (1, 1, 1, 1));
        assert_eq!(c[LATENCY_BUCKETS - 1], 1);
        assert_eq!(c.iter().sum::<u64>(), 5);
    }

    #[test]
    fn quantiles_name_the_bucket_bound() {
        let h = LatencyHistogram::default();
        for _ in 0..99 {
            h.record(Duration::from_micros(100)); // bucket 7, below 128 µs
        }
        h.record(Duration::from_millis(10)); // bucket 14, below 16,384 µs
        let c = h.counts();
        assert_eq!(LatencyHistogram::quantile_us(&c, 0.5), Some(128));
        assert_eq!(LatencyHistogram::quantile_us(&c, 0.99), Some(128));
        assert_eq!(LatencyHistogram::quantile_us(&c, 1.0), Some(16_384));
        assert_eq!(
            LatencyHistogram::quantile_us(&[0; LATENCY_BUCKETS], 0.5),
            None
        );
    }
}
