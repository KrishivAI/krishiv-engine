// SPDX-License-Identifier: Apache-2.0
//! The build-match bitmap of a split nested-loop join as a value (design
//! P2-T052b, 2026-11-06): one bit per build row, numbered in the broadcast
//! exchange's order (map partition, frame, row), which every reader of the
//! committed map outputs sees alike. Partials OR: a build row is matched
//! if some task matched it.

use std::sync::Arc;

use crate::error::{Result, internal};

/// The version byte of [`BuildMarks::encode`].
const V1: u8 = 1;

/// Which build rows of a nested-loop join some probe row matched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildMarks {
    rows: u64,
    words: Arc<[u64]>,
}

impl BuildMarks {
    /// The bitmap of `rows` build rows whose bits are `words` (bit `i % 64`
    /// of word `i / 64` is row `i`).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when `words` does not hold exactly `rows` bits
    /// rounded up to a word, or sets a bit past the last row.
    pub fn new(rows: u64, words: Vec<u64>) -> Result<Self> {
        if u64::try_from(words.len()).ok() != Some(rows.div_ceil(64)) {
            return Err(internal(format!(
                "nested-loop marks of {rows} rows in {} words",
                words.len()
            )));
        }
        let tail = rows % 64;
        if tail != 0 && words.last().is_some_and(|w| w >> tail != 0) {
            return Err(internal(format!("nested-loop marks set past row {rows}")));
        }
        Ok(Self {
            rows,
            words: words.into(),
        })
    }

    /// The number of build rows.
    #[must_use]
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// The bits, one word per 64 build rows.
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Whether build row `row` was matched.
    #[must_use]
    pub fn get(&self, row: u64) -> bool {
        usize::try_from(row / 64)
            .ok()
            .and_then(|w| self.words.get(w))
            .is_some_and(|w| w & (1 << (row % 64)) != 0)
    }

    /// The rows matched in `self` or in `other`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when they number different builds (row counts
    /// differ): the tasks read different broadcast rows.
    pub fn or(&self, other: &Self) -> Result<Self> {
        if self.rows != other.rows {
            return Err(internal(format!(
                "nested-loop marks of {} and {} build rows",
                self.rows, other.rows
            )));
        }
        Ok(Self {
            rows: self.rows,
            words: self
                .words
                .iter()
                .zip(other.words.iter())
                .map(|(a, b)| a | b)
                .collect(),
        })
    }

    /// The bytes a task carries: version 1, the row count, then the words,
    /// little-endian.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(9 + self.words.len() * 8);
        out.push(V1);
        out.extend_from_slice(&self.rows.to_le_bytes());
        for w in self.words.iter() {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// The bitmap of [`encode`](Self::encode)'s bytes.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for another version, or bytes of another length
    /// than the row count implies.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let bad = || internal("malformed nested-loop marks".to_owned());
        let (&version, rest) = bytes.split_first().ok_or_else(bad)?;
        if version != V1 {
            return Err(internal(format!(
                "nested-loop marks of version {version}; this build reads {V1}"
            )));
        }
        let (rows, rest) = rest.split_first_chunk::<8>().ok_or_else(bad)?;
        let rows = u64::from_le_bytes(*rows);
        let (words, tail) = rest.as_chunks::<8>();
        if !tail.is_empty() {
            return Err(bad());
        }
        let words = words.iter().map(|c| u64::from_le_bytes(*c)).collect();
        Self::new(rows, words)
    }
}

#[cfg(test)]
mod tests {
    use super::BuildMarks;

    #[test]
    fn round_trips_and_ors() {
        let a = BuildMarks::new(70, vec![0b101, 0b10]).unwrap();
        let b = BuildMarks::new(70, vec![0b010, 0]).unwrap();
        let back = BuildMarks::decode(&a.encode()).unwrap();
        assert_eq!(back, a);
        let both = a.or(&b).unwrap();
        assert_eq!(both.words(), &[0b111, 0b10]);
        assert!(both.get(0) && both.get(1) && both.get(2) && both.get(65));
        assert!(!both.get(3) && !both.get(64) && !both.get(200));
        assert_eq!(both.rows(), 70);
    }

    #[test]
    fn refuses_mismatched_shapes() {
        assert!(BuildMarks::new(70, vec![0]).is_err());
        assert!(BuildMarks::new(3, vec![0b1000]).is_err());
        let a = BuildMarks::new(3, vec![0]).unwrap();
        let b = BuildMarks::new(4, vec![0]).unwrap();
        assert!(a.or(&b).is_err());
        let mut bytes = a.encode();
        bytes[0] = 2;
        assert!(BuildMarks::decode(&bytes).is_err());
        assert!(BuildMarks::decode(&[1, 0]).is_err());
        assert!(BuildMarks::decode(&a.encode()[..12]).is_err());
        assert_eq!(
            BuildMarks::decode(&BuildMarks::new(0, Vec::new()).unwrap().encode())
                .unwrap()
                .rows(),
            0
        );
    }
}
