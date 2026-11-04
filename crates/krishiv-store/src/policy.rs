// SPDX-License-Identifier: Apache-2.0
//! Request pacing budgets per cloud scheme.

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
