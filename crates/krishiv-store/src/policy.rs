// SPDX-License-Identifier: Apache-2.0
//! Request pacing budgets per cloud scheme, and the retry policy of the cloud clients.

use std::time::Duration;

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

    /// Google Cloud Storage: buckets redistribute load automatically, so
    /// there is no published per-prefix `SlowDown` budget to copy; the
    /// conservative starting point matches S3's until per-service rates
    /// are calibrated against GCS.
    pub const GCS: Self = Self {
        reads_per_prefix: Some(5_500),
        writes_per_prefix: Some(3_500),
    };

    /// Azure Blob Storage: partitions split under load (single-blob targets
    /// are lower than an account's); the conservative starting point matches
    /// S3's until per-service rates are calibrated against Azure.
    pub const AZURE: Self = Self {
        reads_per_prefix: Some(5_500),
        writes_per_prefix: Some(3_500),
    };
}

/// How the cloud clients retry a failed request: how many times, for how long, and the
/// exponential backoff with jitter between attempts. Maps onto `object_store`'s own retry
/// configuration; set it with [`StoreRegistry::with_retry_policy`](crate::StoreRegistry::with_retry_policy).
///
/// A caller that retries above this layer (the engine's request scheduler does) should lower
/// `max_retries`, so the two layers' attempts do not multiply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetryPolicy {
    /// The most times one request is retried; `0` disables retries.
    pub max_retries: usize,
    /// No retry starts after this long since the first attempt. Keep it under five minutes:
    /// retries reuse the request's credentials and payload.
    pub timeout: Duration,
    /// The wait before the first retry.
    pub initial_backoff: Duration,
    /// The longest wait between two attempts.
    pub max_backoff: Duration,
    /// The factor each wait grows by.
    pub backoff_base: f64,
}

impl RetryPolicy {
    /// `object_store`'s defaults: 10 retries within 3 minutes, backoff from 100 ms to 15 s,
    /// growing by 2.
    pub const DEFAULT: Self = Self {
        max_retries: 10,
        timeout: Duration::from_secs(3 * 60),
        initial_backoff: Duration::from_millis(100),
        max_backoff: Duration::from_secs(15),
        backoff_base: 2.0,
    };

    /// Never retries.
    pub const NONE: Self = Self {
        max_retries: 0,
        ..Self::DEFAULT
    };

    /// The policy as `object_store`'s retry configuration.
    #[cfg(feature = "cloud")]
    pub(crate) fn to_config(self) -> object_store::RetryConfig {
        object_store::RetryConfig {
            backoff: object_store::BackoffConfig {
                init_backoff: self.initial_backoff,
                max_backoff: self.max_backoff,
                base: self.backoff_base,
            },
            max_retries: self.max_retries,
            retry_timeout: self.timeout,
        }
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(all(test, feature = "cloud"))]
mod tests {
    use super::*;

    #[test]
    fn the_default_policy_is_object_stores_default() {
        let want = object_store::RetryConfig::default();
        let got = RetryPolicy::DEFAULT.to_config();
        assert_eq!(got.max_retries, want.max_retries);
        assert_eq!(got.retry_timeout, want.retry_timeout);
        assert_eq!(got.backoff.init_backoff, want.backoff.init_backoff);
        assert_eq!(got.backoff.max_backoff, want.backoff.max_backoff);
        assert!((got.backoff.base - want.backoff.base).abs() < f64::EPSILON);
    }

    #[test]
    fn none_disables_retries_and_keeps_the_rest() {
        let c = RetryPolicy::NONE.to_config();
        assert_eq!(c.max_retries, 0);
        assert_eq!(c.retry_timeout, RetryPolicy::DEFAULT.timeout);
    }
}
