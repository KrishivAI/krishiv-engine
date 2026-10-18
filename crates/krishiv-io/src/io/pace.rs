// SPDX-License-Identifier: Apache-2.0
//! Prefix-aware request pacing (ADR 0014 item 11).
//!
//! Object stores throttle per key prefix (S3: about 5,500 reads and 3,500
//! writes per second per prefix, then `503 SlowDown`). A paced store keeps
//! one token bucket per prefix and per direction; a request takes a token
//! and waits (on Tokio's clock, so paused-time tests replay) when the
//! bucket is empty. The burst is one second's worth of requests. Pacing
//! happens before a request takes an in-flight permit, so a waiting request
//! never blocks other prefixes. Local and in-memory stores are not paced.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use object_store::path::Path;
use tokio::time::Instant;

/// Most prefixes tracked per direction before idle (full) buckets are
/// dropped; a dropped bucket is recreated full, which is what it was.
const MAX_TRACKED_PREFIXES: usize = 4096;

/// A store's request pacing: requests per second per key prefix, by
/// direction; `None` is unpaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StorePolicy {
    /// GETs per second per prefix.
    pub reads_per_prefix: Option<u32>,
    /// PUTs, multipart parts and deletes per second per prefix.
    pub writes_per_prefix: Option<u32>,
}

impl StorePolicy {
    /// No pacing (local files, memory).
    pub const UNPACED: Self = Self {
        reads_per_prefix: None,
        writes_per_prefix: None,
    };

    /// Amazon S3's documented per-prefix request rates.
    pub const S3: Self = Self {
        reads_per_prefix: Some(5_500),
        writes_per_prefix: Some(3_500),
    };
}

/// Which bucket a request draws from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Direction {
    Read,
    Write,
}

#[derive(Debug)]
struct Bucket {
    tokens: f64,
    at: Instant,
}

#[derive(Debug)]
struct Buckets {
    rate: f64,
    prefixes: Mutex<HashMap<String, Bucket>>,
}

impl Buckets {
    fn new(rate: u32) -> Option<Self> {
        (rate > 0).then(|| Self {
            rate: f64::from(rate),
            prefixes: Mutex::new(HashMap::new()),
        })
    }

    /// Takes a token for `prefix` and returns how long to wait for it.
    fn take(&self, prefix: &str, now: Instant) -> Duration {
        let rate = self.rate;
        let refill = |b: &mut Bucket| {
            let elapsed = now.saturating_duration_since(b.at).as_secs_f64();
            b.tokens = elapsed.mul_add(rate, b.tokens).min(rate);
            b.at = now;
        };
        let mut map = self.prefixes.lock().unwrap_or_else(PoisonError::into_inner);
        if map.len() >= MAX_TRACKED_PREFIXES && !map.contains_key(prefix) {
            map.retain(|_, b| {
                refill(b);
                b.tokens < rate
            });
        }
        let b = map.entry(prefix.to_owned()).or_insert(Bucket {
            tokens: rate,
            at: now,
        });
        refill(b);
        // A token may be borrowed: the debt is the wait, so waiters are
        // served in arrival order at exactly the rate.
        b.tokens -= 1.0;
        if b.tokens >= 0.0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(-b.tokens / rate)
        }
    }
}

/// The pacing state of one store.
#[derive(Debug)]
pub(crate) struct Pacer {
    reads: Option<Buckets>,
    writes: Option<Buckets>,
}

impl Pacer {
    pub(crate) fn new(policy: StorePolicy) -> Self {
        Self {
            reads: policy.reads_per_prefix.and_then(Buckets::new),
            writes: policy.writes_per_prefix.and_then(Buckets::new),
        }
    }

    /// Waits for a token of `path`'s prefix. Dropping the future gives up
    /// the wait (the token stays taken: pacing errs on the slow side).
    pub(crate) async fn pace(&self, direction: Direction, path: &Path) {
        let buckets = match direction {
            Direction::Read => self.reads.as_ref(),
            Direction::Write => self.writes.as_ref(),
        };
        let Some(b) = buckets else {
            return;
        };
        let wait = b.take(prefix(path), Instant::now());
        if !wait.is_zero() {
            tokio::time::sleep(wait).await;
        }
    }
}

/// The key prefix stores partition by: everything up to the last `/`.
fn prefix(path: &Path) -> &str {
    let p: &str = path.as_ref();
    p.rfind('/').map_or("", |i| &p[..i])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_are_directories() {
        assert_eq!(prefix(&Path::from("a/b/c.parquet")), "a/b");
        assert_eq!(prefix(&Path::from("c.parquet")), "");
    }

    #[test]
    fn a_full_bucket_is_dropped_when_too_many_prefixes_are_tracked() {
        let b = Buckets::new(2).unwrap_or_else(|| unreachable!());
        let now = Instant::now();
        for i in 0..MAX_TRACKED_PREFIXES {
            assert_eq!(b.take(&format!("p{i}"), now), Duration::ZERO);
        }
        // Every bucket refilled after a second: all are dropped but the new one.
        let later = now + Duration::from_secs(1);
        assert_eq!(b.take("new", later), Duration::ZERO);
        let len = b.prefixes.lock().map(|m| m.len()).unwrap_or(0);
        assert_eq!(len, 1);
        assert!(Buckets::new(0).is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn waits_follow_the_rate_per_prefix() {
        let pacer = Pacer::new(StorePolicy {
            reads_per_prefix: Some(10),
            writes_per_prefix: None,
        });
        let start = Instant::now();
        for _ in 0..30 {
            pacer.pace(Direction::Read, &Path::from("hot/f")).await;
        }
        // A burst of 10, then 20 more at 10 per second.
        let hot = start.elapsed();
        assert!(
            hot >= Duration::from_millis(1990) && hot <= Duration::from_millis(2010),
            "{hot:?}"
        );
        // Another prefix has its own bucket; writes are unpaced.
        let start = Instant::now();
        for _ in 0..10 {
            pacer.pace(Direction::Read, &Path::from("cold/f")).await;
            pacer.pace(Direction::Write, &Path::from("hot/f")).await;
        }
        assert_eq!(start.elapsed(), Duration::ZERO);
    }
}
