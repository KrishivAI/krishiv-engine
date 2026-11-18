// SPDX-License-Identifier: Apache-2.0
//! Key facts of file tables (design P2b A5 (d), unit 2): whether a set of a
//! table's columns is unique among its rows whose columns of the set are all
//! non-NULL, as the planner counted it, kept next to the distinct-value
//! samples so a later query over the same file versions reads nothing.
//!
//! A key names every file of the table by its metadata-cache key (store,
//! path, size and version), in listing order, the sorted column names and
//! how the values were decoded ([`KeyFactKey`]): a changed, added or removed
//! file, or another decoding, is a miss, never a stale fact. A count
//! that did not finish is kept with the time it had, so it is tried again
//! only with more. The cache is bounded in entries and drops the least
//! recently used one when full.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Mutex, PoisonError};

use crate::io::cache::CacheKey;

/// What a count of a table's key columns found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyFact {
    /// No two rows with all key columns non-NULL agree on every key column.
    Unique,
    /// Two such rows agree (or two rows' hashes did: a collision only loses
    /// a rewrite).
    NotUnique,
    /// The count did not finish (out of time or memory, a read error)
    /// within the time it had, in milliseconds.
    Unknown {
        /// The time it had.
        within_ms: u64,
    },
}

/// The key of a fact: every file of the table, in order, the column names,
/// sorted, and what decides the values the count saw (`decoding`: the
/// columns' types, the format and its read options, the settings that
/// change which rows are read). The same files decoded another way (a
/// `STRING` column read as `INT` turns `1` and `01` into one value; a CSV
/// `nullValue` turns a value into NULL) are another key, never the same fact.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct KeyFactKey {
    files: Vec<CacheKey>,
    columns: Vec<String>,
    decoding: String,
}

impl fmt::Debug for KeyFactKey {
    // The decoding text can hold read options; it is compared, never shown.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeyFactKey")
            .field("files", &self.files)
            .field("columns", &self.columns)
            .finish_non_exhaustive()
    }
}

impl KeyFactKey {
    /// The key of a count of `columns` over `files` (all the table's files),
    /// decoded as `decoding` describes.
    #[must_use]
    pub fn new(files: Vec<CacheKey>, columns: &[String], decoding: String) -> Self {
        let mut columns = columns.to_vec();
        columns.sort();
        columns.dedup();
        Self {
            files,
            columns,
            decoding,
        }
    }
}

/// The key facts counted in a process (one per I/O scheduler).
#[derive(Debug)]
pub struct KeyFacts {
    inner: Mutex<Inner>,
    capacity: usize,
}

#[derive(Debug, Default)]
struct Inner {
    /// Each fact with the tick of its last use.
    map: HashMap<KeyFactKey, (KeyFact, u64)>,
    tick: u64,
}

impl KeyFacts {
    /// Entries kept before the least recently used is dropped.
    pub const CAPACITY: usize = 1024;

    /// An empty cache of at most `capacity` entries.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            capacity: capacity.max(1),
        }
    }

    /// The fact stored under `key` (a use: it becomes the most recent).
    #[must_use]
    pub fn get(&self, key: &KeyFactKey) -> Option<KeyFact> {
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        inner.tick += 1;
        let tick = inner.tick;
        let entry = inner.map.get_mut(key)?;
        entry.1 = tick;
        Some(entry.0)
    }

    /// Stores `fact` under `key`; a full cache drops its least recently used
    /// entry first.
    pub fn insert(&self, key: KeyFactKey, fact: KeyFact) {
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        inner.tick += 1;
        let tick = inner.tick;
        if inner.map.len() >= self.capacity
            && !inner.map.contains_key(&key)
            && let Some(oldest) = inner
                .map
                .iter()
                .min_by_key(|(_, (_, t))| *t)
                .map(|(k, _)| k.clone())
        {
            inner.map.remove(&oldest);
        }
        inner.map.insert(key, (fact, tick));
    }

    /// The number of facts held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .map
            .len()
    }

    /// Whether no fact is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forgets every fact.
    pub fn clear(&self) {
        self.inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .map
            .clear();
    }
}

impl Default for KeyFacts {
    fn default() -> Self {
        Self::new(Self::CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::{KeyFact, KeyFactKey, KeyFacts};
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

    fn cols(c: &[&str]) -> Vec<String> {
        c.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn a_fact_is_found_by_its_files_and_sorted_columns() {
        let c = KeyFacts::default();
        let key = KeyFactKey::new(
            vec![file("a", "v1"), file("b", "v1")],
            &cols(&["y", "x"]),
            "int".into(),
        );
        c.insert(key, KeyFact::Unique);
        assert_eq!(
            c.get(&KeyFactKey::new(
                vec![file("a", "v1"), file("b", "v1")],
                &cols(&["x", "y"]),
                "int".into()
            )),
            Some(KeyFact::Unique),
            "columns in any order"
        );
        for (files, columns, why) in [
            (
                vec![file("a", "v2"), file("b", "v1")],
                cols(&["x", "y"]),
                "a changed file",
            ),
            (vec![file("a", "v1")], cols(&["x", "y"]), "a file removed"),
            (
                vec![file("a", "v1"), file("b", "v1"), file("c", "v1")],
                cols(&["x", "y"]),
                "a file added",
            ),
            (
                vec![file("a", "v1"), file("b", "v1")],
                cols(&["x"]),
                "other columns",
            ),
        ] {
            assert_eq!(
                c.get(&KeyFactKey::new(files, &columns, "int".into())),
                None,
                "{why}"
            );
        }
        assert_eq!(
            c.get(&KeyFactKey::new(
                vec![file("a", "v1"), file("b", "v1")],
                &cols(&["x", "y"]),
                "string".into()
            )),
            None,
            "another decoding"
        );
    }

    #[test]
    fn every_answer_is_kept() {
        let c = KeyFacts::default();
        for (i, fact) in [
            KeyFact::Unique,
            KeyFact::NotUnique,
            KeyFact::Unknown { within_ms: 50 },
        ]
        .into_iter()
        .enumerate()
        {
            let key = KeyFactKey::new(
                vec![file(&i.to_string(), "v")],
                &cols(&["k"]),
                String::new(),
            );
            c.insert(key.clone(), fact);
            assert_eq!(c.get(&key), Some(fact));
        }
        assert_eq!(c.len(), 3);
        c.clear();
        assert!(c.is_empty());
    }

    #[test]
    fn a_full_cache_drops_the_least_recently_used() {
        let c = KeyFacts::new(2);
        let key = |n: &str| KeyFactKey::new(vec![file(n, "v")], &cols(&["k"]), String::new());
        c.insert(key("a"), KeyFact::Unique);
        c.insert(key("b"), KeyFact::Unique);
        // `a` used again: `b` is the oldest.
        assert!(c.get(&key("a")).is_some());
        c.insert(key("c"), KeyFact::NotUnique);
        assert_eq!(c.len(), 2);
        assert!(c.get(&key("b")).is_none());
        assert!(c.get(&key("a")).is_some());
        assert!(c.get(&key("c")).is_some());
    }
}
