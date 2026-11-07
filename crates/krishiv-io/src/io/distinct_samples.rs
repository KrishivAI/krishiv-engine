// SPDX-License-Identifier: Apache-2.0
//! The distinct-value samples of file columns (P2-T164 unit U6): what the
//! planner counted in the first rows of a table's files, kept next to the
//! file-metadata cache so a later query over the same file versions does not
//! read them again.
//!
//! A key names the sampled files by their metadata-cache keys (store, path,
//! size and version), the column and the row budget, so a changed file or a
//! different budget is a miss, never a stale count. The cache is bounded in
//! entries: when full, it starts over (samples are cheap to take again).

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use crate::io::cache::CacheKey;

/// What one sample counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistinctSample {
    /// Non-null rows counted.
    pub rows: u64,
    /// Null rows counted.
    pub nulls: u64,
    /// Distinct non-null values among them.
    pub distinct: u64,
    /// Values seen exactly once.
    pub singletons: u64,
    /// Whether the sample read every row of the table.
    pub whole: bool,
    /// When the sample did not finish: the time it was given, in
    /// milliseconds (0 for a sample that finished). A later query with no
    /// more time does not try again.
    pub failed_within_ms: u64,
}

/// The key of a sample: the files it read, in order, the column, the row
/// budget and the table's file count (a file added after the sampled ones
/// changes whether the sample was the whole table).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SampleKey {
    files: Vec<CacheKey>,
    column: String,
    budget: u64,
    table_files: usize,
}

impl SampleKey {
    /// The key of a sample of `column` over `files` within `budget` rows, of
    /// a table of `table_files` files.
    #[must_use]
    pub fn new(files: Vec<CacheKey>, column: &str, budget: u64, table_files: usize) -> Self {
        Self {
            files,
            column: column.to_owned(),
            budget,
            table_files,
        }
    }
}

/// The samples taken in a process (one per I/O scheduler).
#[derive(Debug)]
pub struct DistinctSamples {
    map: Mutex<HashMap<SampleKey, DistinctSample>>,
    capacity: usize,
}

impl DistinctSamples {
    /// Entries kept before the cache starts over.
    pub const CAPACITY: usize = 4096;

    /// An empty cache of at most `capacity` entries.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
            capacity: capacity.max(1),
        }
    }

    /// The sample stored under `key`.
    #[must_use]
    pub fn get(&self, key: &SampleKey) -> Option<DistinctSample> {
        self.map
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(key)
            .copied()
    }

    /// Stores `sample` under `key`; a full cache starts over first.
    pub fn insert(&self, key: SampleKey, sample: DistinctSample) {
        let mut map = self.map.lock().unwrap_or_else(PoisonError::into_inner);
        if map.len() >= self.capacity && !map.contains_key(&key) {
            map.clear();
        }
        map.insert(key, sample);
    }

    /// The number of samples held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Whether no sample is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for DistinctSamples {
    fn default() -> Self {
        Self::new(Self::CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::{DistinctSample, DistinctSamples, SampleKey};
    use crate::io::cache::CacheKey;
    use crate::io::{FileRef, StoreId, VersionToken};

    fn file(path: &str, version: &str) -> CacheKey {
        let f = FileRef {
            store: StoreId::new("file"),
            path: path.into(),
            size: 10,
            version: VersionToken::ETag(version.into()),
            modified_micros: None,
        };
        CacheKey::new("parquet", &f, None)
    }

    fn sample(distinct: u64) -> DistinctSample {
        DistinctSample {
            rows: 100,
            nulls: 0,
            distinct,
            singletons: 0,
            whole: true,
            failed_within_ms: 0,
        }
    }

    #[test]
    fn a_sample_is_found_by_its_files_column_and_budget() {
        let c = DistinctSamples::default();
        let key = SampleKey::new(vec![file("a", "v1")], "x", 10, 1);
        c.insert(key.clone(), sample(3));
        assert_eq!(c.get(&key), Some(sample(3)));
        assert_eq!(
            c.get(&SampleKey::new(vec![file("a", "v2")], "x", 10, 1)),
            None,
            "another version"
        );
        assert_eq!(
            c.get(&SampleKey::new(vec![file("a", "v1")], "y", 10, 1)),
            None,
            "another column"
        );
        assert_eq!(
            c.get(&SampleKey::new(vec![file("a", "v1")], "x", 20, 1)),
            None,
            "another budget"
        );
        assert_eq!(
            c.get(&SampleKey::new(vec![file("a", "v1")], "x", 10, 2)),
            None,
            "a file added after the sampled ones"
        );
    }

    #[test]
    fn a_full_cache_starts_over() {
        let c = DistinctSamples::new(2);
        for (i, name) in ["a", "b", "c"].into_iter().enumerate() {
            c.insert(
                SampleKey::new(vec![file(name, "v")], "x", 1, 1),
                sample(i as u64),
            );
        }
        assert_eq!(c.len(), 1);
        assert!(
            c.get(&SampleKey::new(vec![file("c", "v")], "x", 1, 1))
                .is_some()
        );
        assert!(!c.is_empty());
    }
}
