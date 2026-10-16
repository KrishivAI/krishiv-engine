//! Batch == bounded-stream invariant harness (ADR 0015 / 0017).
//!
//! The same query must return the same multiset of rows whether it runs as a batch job or as a
//! streaming job replaying the same (bounded) input split into epochs. Row order is not compared.

use std::collections::BTreeMap;

/// One result row; each cell is its canonical text form, `None` for SQL NULL.
pub type Row = Vec<Option<String>>;

/// Error from a [`QueryRunner`].
#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
    /// The engine does not support this yet.
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
    /// The query failed.
    #[error("query failed: {0}")]
    Failed(String),
}

/// An engine that can run a query both ways.
pub trait QueryRunner {
    /// Run `sql` as a batch query.
    fn run_batch(&self, sql: &str) -> Result<Vec<Row>, RunnerError>;

    /// Run `sql` as a streaming query over the same input replayed as `epochs` epochs, and
    /// return the accumulated final result (the materialised output after the last epoch).
    fn run_stream_replay(&self, sql: &str, epochs: usize) -> Result<Vec<Row>, RunnerError>;
}

/// Difference between two row multisets. Empty means equal.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MultisetDiff {
    /// Rows (with surplus count) present more often in batch than in stream.
    pub only_in_batch: Vec<(Row, usize)>,
    /// Rows (with surplus count) present more often in stream than in batch.
    pub only_in_stream: Vec<(Row, usize)>,
}

impl MultisetDiff {
    /// True when the multisets are equal.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.only_in_batch.is_empty() && self.only_in_stream.is_empty()
    }
}

/// Compare two row collections as multisets (order-insensitive, duplicates counted).
#[must_use]
pub fn multiset_diff(batch: &[Row], stream: &[Row]) -> MultisetDiff {
    let mut counts: BTreeMap<&Row, (usize, usize)> = BTreeMap::new();
    for r in batch {
        counts.entry(r).or_default().0 += 1;
    }
    for r in stream {
        counts.entry(r).or_default().1 += 1;
    }
    let mut diff = MultisetDiff::default();
    for (row, (b, s)) in counts {
        if b > s {
            diff.only_in_batch.push((row.clone(), b - s));
        } else if s > b {
            diff.only_in_stream.push((row.clone(), s - b));
        }
    }
    diff
}

/// Why the invariant check failed.
#[derive(Debug, thiserror::Error)]
pub enum InvariantError {
    /// A runner failed.
    #[error(transparent)]
    Runner(#[from] RunnerError),
    /// Results differ.
    #[error("batch != stream replay ({epochs} epochs) for `{sql}`: {diff:?}")]
    Mismatch {
        /// The query.
        sql: String,
        /// Epoch count used for the replay.
        epochs: usize,
        /// The difference.
        diff: MultisetDiff,
    },
}

/// Check the batch == stream invariant for `sql` with the input replayed as `epochs` epochs.
pub fn assert_batch_equals_stream(
    runner: &dyn QueryRunner,
    sql: &str,
    epochs: usize,
) -> Result<(), InvariantError> {
    let batch = runner.run_batch(sql)?;
    let stream = runner.run_stream_replay(sql, epochs)?;
    let diff = multiset_diff(&batch, &stream);
    if diff.is_empty() {
        Ok(())
    } else {
        Err(InvariantError::Mismatch {
            sql: sql.to_owned(),
            epochs,
            diff,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(cells: &[&str]) -> Row {
        cells.iter().map(|c| Some((*c).to_owned())).collect()
    }

    /// In-memory fake: fixed batch rows, and a stream result chosen by the test.
    struct Fake {
        batch: Vec<Row>,
        stream: Vec<Row>,
    }

    impl QueryRunner for Fake {
        fn run_batch(&self, _sql: &str) -> Result<Vec<Row>, RunnerError> {
            Ok(self.batch.clone())
        }
        fn run_stream_replay(&self, _sql: &str, _epochs: usize) -> Result<Vec<Row>, RunnerError> {
            Ok(self.stream.clone())
        }
    }

    #[test]
    fn order_insensitive_equal() {
        let f = Fake {
            batch: vec![row(&["1", "a"]), row(&["2", "b"]), row(&["1", "a"])],
            stream: vec![row(&["1", "a"]), row(&["1", "a"]), row(&["2", "b"])],
        };
        assert!(assert_batch_equals_stream(&f, "q", 3).is_ok());
    }

    #[test]
    fn duplicates_are_counted() {
        let f = Fake {
            batch: vec![row(&["1"]), row(&["1"])],
            stream: vec![row(&["1"])],
        };
        match assert_batch_equals_stream(&f, "q", 2) {
            Err(InvariantError::Mismatch { diff, .. }) => {
                assert_eq!(diff.only_in_batch, vec![(row(&["1"]), 1)]);
                assert!(diff.only_in_stream.is_empty());
            }
            other => assert!(other.is_err(), "expected mismatch, got {other:?}"),
        }
    }

    #[test]
    fn null_differs_from_text_null() {
        let a = vec![vec![None]];
        let b = vec![row(&["NULL"])];
        assert!(!multiset_diff(&a, &b).is_empty());
        assert!(multiset_diff(&a, &a).is_empty());
    }

    /// Placeholder runner for the real engine; replaced by the Krishiv session in P4.
    struct Krishiv;

    impl QueryRunner for Krishiv {
        fn run_batch(&self, _sql: &str) -> Result<Vec<Row>, RunnerError> {
            Err(RunnerError::NotImplemented("batch execution (P2)"))
        }
        fn run_stream_replay(&self, _sql: &str, _epochs: usize) -> Result<Vec<Row>, RunnerError> {
            Err(RunnerError::NotImplemented("stream replay (P4)"))
        }
    }

    #[test]
    #[ignore = "P4: streaming execution not implemented"]
    fn batch_equals_stream_for_nexmark_q5() -> Result<(), InvariantError> {
        assert_batch_equals_stream(&Krishiv, "SELECT 1", 4)
    }
}
