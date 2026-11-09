// SPDX-License-Identifier: Apache-2.0
//! Plan-time distinct-value samples of file columns (P2-T164 unit U6).
//!
//! Plain Parquet files carry no distinct counts for strings, decimals or
//! timestamps, so an equality on such a column would take the default
//! selectivity. Before the optimizer runs, the session asks a [`FileTable`]
//! to sample the columns an equality or `IN` reads: the first rows of the
//! table in file order, those columns only, through the ordinary
//! [`ScanStream`] (every byte through the `IoContext`), within a row budget
//! and a time limit. Counting is exact over the sample (a map of value
//! hashes to counts, capped at [`MAX_TRACKED`] distinct values; past it the
//! sample is the prefix counted so far).
//!
//! [`estimate`] turns a sample into the table's distinct count: exact when
//! the sample read the whole table, else GEE (Charikar, Chaudhuri, Motwani,
//! Narasayya, PODS 2000: `sqrt(N/n) * f1 + (d - f1)`), clamped to
//! `[d, N]`. The first rows are not a random sample; the count only steers
//! plans, never results. Samples are cached next to the file-metadata cache
//! ([`krishiv_io::DistinctSamples`]), keyed by the files' versions.

use std::collections::HashMap;
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{Array, ArrayRef};
use arrow_row::{RowConverter, SortField};
use krishiv_io::{
    ByteRange, CacheKey, ColumnPath, DistinctSample, SampleKey, ScanRequest, ScanStream,
};

use super::file_table::FileTable;
use crate::Result;
use crate::error::internal;

/// The most distinct values a sample tracks per column; past it the column
/// stops counting (its sample is the rows counted until then).
pub(super) const MAX_TRACKED: usize = 1 << 16;

/// How much a table sample may read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleOptions {
    /// Rows read at most (`krishiv.optimizer.ndvSampling.rows`).
    pub rows: u64,
    /// Time left for this table: what is left of the query's budget
    /// (`krishiv.optimizer.ndvSampling.timeout`). Past it the table keeps
    /// its estimates; zero reads nothing.
    pub timeout: Duration,
}

/// What a table's sample did for some of its columns, as `EXPLAIN COST`
/// and `EXPLAIN VERBOSE` report it: whether the estimates of `columns`
/// come from a sample or stayed the defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleOutcome {
    /// The columns, in the order asked.
    pub columns: Vec<String>,
    /// What happened to them.
    pub result: SampleResult,
}

/// How a sample of some columns ended ([`SampleOutcome`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleResult {
    /// Read `rows` rows (every row of the table when `whole`) in `took`.
    Sampled {
        /// Rows read, nulls included.
        rows: u64,
        /// The sample is the whole table: the counts are exact.
        whole: bool,
        /// The read's time.
        took: Duration,
    },
    /// An earlier query's sample of the same files.
    Cached,
    /// The read did not finish within `after`: the defaults stay.
    TimedOut {
        /// The time it had.
        after: Duration,
    },
    /// An earlier query's sample of the same files did not finish (out of
    /// time, or a read error) within as much time as this one has: the
    /// defaults stay, nothing is read.
    TimedOutBefore,
    /// The read failed: the defaults stay.
    Failed,
    /// No sampling time was left for this table (the query's other tables
    /// spent it, or the budget is zero): the defaults stay, nothing is read.
    NoTime,
}

impl SampleOutcome {
    /// The outcome's text with the read's time (`EXPLAIN VERBOSE`); the
    /// `Display` text leaves it out, so `EXPLAIN COST` reads the same from
    /// run to run.
    #[must_use]
    pub fn with_time(&self) -> String {
        match self.result {
            SampleResult::Sampled { took, .. } => format!("{self} in {} ms", took.as_millis()),
            _ => self.to_string(),
        }
    }
}

impl fmt::Display for SampleOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.columns.join(", "))?;
        match self.result {
            SampleResult::Sampled { rows, whole, .. } => write!(
                f,
                "sampled {rows} rows{}",
                if whole { " (the whole table)" } else { "" },
            ),
            SampleResult::Cached => f.write_str("sampled by an earlier query"),
            SampleResult::TimedOut { after } => {
                write!(f, "timed out after {} ms, defaults kept", after.as_millis())
            }
            SampleResult::TimedOutBefore => f.write_str(
                "an earlier query's sample with as much time did not finish, defaults kept",
            ),
            SampleResult::Failed => f.write_str("the read failed, defaults kept"),
            SampleResult::NoTime => f.write_str("no sampling time left, defaults kept"),
        }
    }
}

/// The table's distinct non-null values estimated from `s`, for a table of
/// `table_rows` rows: exact when the sample read every row, else GEE over the
/// table's non-null rows (the sample's null fraction applied), clamped to
/// `[distinct seen, non-null rows]`. `None` when the sample holds no value.
#[must_use]
pub(super) fn estimate(s: &DistinctSample, table_rows: u64) -> Option<u64> {
    if s.rows == 0 {
        return (s.whole || s.nulls > 0).then_some(0);
    }
    if s.whole {
        return Some(s.distinct);
    }
    let counted = s.rows.saturating_add(s.nulls);
    #[expect(clippy::cast_precision_loss, reason = "an estimate")]
    let (n, f1, d, non_null) = (
        s.rows as f64,
        s.singletons as f64,
        s.distinct as f64,
        table_rows as f64 * (s.rows as f64 / counted.max(1) as f64),
    );
    let gee = (non_null / n).max(1.0).sqrt() * f1 + (d - f1);
    let clamped = gee.min(non_null).max(d);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a finite, non-negative estimate"
    )]
    Some(clamped.round() as u64)
}

/// Counts of one column's values over a sample.
#[derive(Debug, Default)]
struct Counter {
    counts: HashMap<u64, u64>,
    rows: u64,
    nulls: u64,
    /// Stopped tracking: the sample is what was counted before.
    full: bool,
}

impl Counter {
    /// Counts the rows of `column`.
    fn add(&mut self, column: &ArrayRef) -> Result<()> {
        if self.full {
            return Ok(());
        }
        let rows = RowConverter::new(vec![SortField::new(column.data_type().clone())])
            .and_then(|c| c.convert_columns(std::slice::from_ref(column)))
            .map_err(|e| internal(e.to_string()))?;
        for i in 0..column.len() {
            if column.is_null(i) {
                self.nulls += 1;
                continue;
            }
            let mut h = DefaultHasher::new();
            rows.row(i).as_ref().hash(&mut h);
            let key = h.finish();
            if !self.counts.contains_key(&key) && self.counts.len() >= MAX_TRACKED {
                self.full = true;
                return Ok(());
            }
            *self.counts.entry(key).or_insert(0) += 1;
            self.rows += 1;
        }
        Ok(())
    }

    fn sample(&self, whole: bool) -> DistinctSample {
        DistinctSample {
            rows: self.rows,
            nulls: self.nulls,
            distinct: self.counts.len() as u64,
            singletons: self.counts.values().filter(|&&c| c == 1).count() as u64,
            whole: whole && !self.full,
            failed_within_ms: 0,
        }
    }
}

impl FileTable {
    /// Samples `columns` (names of the table's file columns; partition and
    /// unknown columns are skipped) within `options`, and keeps each
    /// estimated distinct count for [`statistics`](super::table::TableProvider::statistics).
    /// Columns already sampled, or with a distinct count from the footers,
    /// are not read again. Past the time limit the table keeps its
    /// estimates. What happened is kept for
    /// [`sample_outcomes`](Self::sample_outcomes).
    ///
    /// # Errors
    /// Read errors of the sampled files.
    pub async fn sample_distinct(&self, columns: &[String], options: SampleOptions) -> Result<()> {
        let Some(stats) = self.base_statistics() else {
            return Ok(());
        };
        let wanted: Vec<&String> = columns
            .iter()
            .filter(|c| {
                self.table_schema().index_of(c).is_ok()
                    && !self.is_partition_column(c)
                    && stats.columns.get(*c).is_none_or(|s| s.distinct.is_none())
                    && !self.has_sample(c)
            })
            .collect();
        if wanted.is_empty() || options.rows == 0 {
            return Ok(());
        }
        // The files the budget reaches, by their footers' row counts.
        let mut files = Vec::new();
        let mut covered = 0_u64;
        for (file, rows) in self.files().iter().zip(self.file_rows()) {
            if covered >= options.rows {
                break;
            }
            files.push(file.clone());
            covered = covered.saturating_add(rows.unwrap_or(options.rows));
        }
        let whole = files.len() == self.files().len() && stats.rows <= options.rows;
        let format = self.format();
        let keys: Vec<CacheKey> = files
            .iter()
            .map(|f| CacheKey::new(format.name(), f, None))
            .collect();
        let cache = self.io().scheduler().distinct_samples();
        let table_files = self.files().len();
        let key = |c: &str| SampleKey::new(keys.clone(), c, options.rows, table_files);
        let timeout_ms = u64::try_from(options.timeout.as_millis()).unwrap_or(u64::MAX);
        let mut missing = Vec::new();
        let (mut cached, mut failed_before) = (Vec::new(), Vec::new());
        for c in wanted {
            match cache.get(&key(c)) {
                // A sample that did not finish is tried again only with more time.
                Some(s) if s.failed_within_ms > 0 && timeout_ms > s.failed_within_ms => {
                    missing.push(c.clone());
                }
                Some(s) => {
                    self.keep_sample(c, &s, stats.rows);
                    if s.failed_within_ms > 0 {
                        failed_before.push(c.clone());
                    } else {
                        cached.push(c.clone());
                    }
                }
                None => missing.push(c.clone()),
            }
        }
        for (columns, result) in [
            (cached, SampleResult::Cached),
            (failed_before, SampleResult::TimedOutBefore),
        ] {
            if !columns.is_empty() {
                self.note_sample(SampleOutcome { columns, result });
            }
        }
        if missing.is_empty() {
            return Ok(());
        }
        if options.timeout.is_zero() {
            // Nothing recorded in the cache: a later query with time samples.
            self.note_sample(SampleOutcome {
                columns: missing,
                result: SampleResult::NoTime,
            });
            return Ok(());
        }
        let started = Instant::now();
        let read = self.read_sample(&files, &missing, options.rows);
        let counters = match tokio::time::timeout(options.timeout, read).await {
            Ok(Ok(counters)) => counters,
            failed => {
                // Recorded, so later queries over the same files do not pay
                // for it again (unless given more time).
                let failure = DistinctSample {
                    rows: 0,
                    nulls: 0,
                    distinct: 0,
                    singletons: 0,
                    whole: false,
                    failed_within_ms: timeout_ms.max(1),
                };
                for c in &missing {
                    cache.insert(key(c), failure);
                    self.keep_sample(c, &failure, stats.rows);
                }
                return match failed {
                    Ok(Err(e)) => {
                        self.note_sample(SampleOutcome {
                            columns: missing,
                            result: SampleResult::Failed,
                        });
                        Err(e)
                    }
                    _ => {
                        tracing::debug!(columns = ?missing, "distinct-value sample timed out");
                        self.note_sample(SampleOutcome {
                            columns: missing,
                            result: SampleResult::TimedOut {
                                after: options.timeout,
                            },
                        });
                        Ok(())
                    }
                };
            }
        };
        let took = started.elapsed();
        let mut rows = 0;
        let mut all_whole = true;
        for (c, counter) in missing.iter().zip(counters) {
            let s = counter.sample(whole);
            rows = rows.max(counter.rows.saturating_add(counter.nulls));
            all_whole &= s.whole;
            cache.insert(key(c), s);
            self.keep_sample(c, &s, stats.rows);
        }
        self.note_sample(SampleOutcome {
            columns: missing,
            result: SampleResult::Sampled {
                rows,
                whole: all_whole,
                took,
            },
        });
        Ok(())
    }

    /// Reads up to `budget` rows of `columns` from `files`, in order.
    async fn read_sample(
        &self,
        files: &[krishiv_io::FileRef],
        columns: &[String],
        budget: u64,
    ) -> Result<Vec<Counter>> {
        let schema = self.table_schema();
        let projection: Vec<ColumnPath> =
            columns.iter().map(|c| ColumnPath::new(c.clone())).collect();
        let mut counters: Vec<Counter> = columns.iter().map(|_| Counter::default()).collect();
        // The query's pool, as the footer reads use; a context without one
        // (embedded tests) reads under a pool of its own.
        let pool: Arc<dyn krishiv_memory::MemoryPool> = match self.io().memory_pool() {
            Some(p) => Arc::clone(p),
            None => krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default()),
        };
        let mut left = budget;
        for file in files {
            if left == 0 {
                break;
            }
            let mut req = ScanRequest::new(Arc::clone(schema), projection.clone());
            req.limit = Some(left);
            let id = pool.register("ndv sample", None);
            let mut stream = ScanStream::open(
                self.io(),
                self.format().as_ref(),
                file.clone(),
                ByteRange::whole(file.size),
                &req,
                &self.options(),
                krishiv_memory::Reservation::new(Arc::clone(&pool), id),
                self.scan_options(),
            )
            .await?;
            while let Some(batch) = stream.next_batch().await? {
                let rows = batch.num_rows() as u64;
                for (counter, name) in counters.iter_mut().zip(columns) {
                    if let Some(column) = batch.column_by_name(name) {
                        counter.add(column)?;
                    }
                }
                left = left.saturating_sub(rows);
                if left == 0 {
                    break;
                }
            }
        }
        Ok(counters)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array, StringArray};

    use super::{Counter, MAX_TRACKED, SampleOutcome, SampleResult, estimate};
    use krishiv_io::DistinctSample;
    use std::time::Duration;

    fn sample(rows: u64, distinct: u64, singletons: u64, whole: bool) -> DistinctSample {
        DistinctSample {
            rows,
            nulls: 0,
            distinct,
            singletons,
            whole,
            failed_within_ms: 0,
        }
    }

    #[test]
    fn a_whole_table_sample_is_exact() {
        assert_eq!(estimate(&sample(25, 25, 25, true), 25), Some(25));
        assert_eq!(estimate(&sample(10, 3, 0, true), 10), Some(3));
    }

    #[test]
    fn few_repeated_values_stay_few() {
        // `o_orderstatus`: three values, none seen once, in 100K of 150M rows.
        assert_eq!(
            estimate(&sample(100_000, 3, 0, false), 150_000_000),
            Some(3)
        );
    }

    #[test]
    fn values_seen_once_scale_with_the_square_root_of_the_ratio() {
        // All distinct in 10K of 1M rows: sqrt(100) * 10K = 100K.
        assert_eq!(
            estimate(&sample(10_000, 10_000, 10_000, false), 1_000_000),
            Some(100_000)
        );
        // Ten values seen once in 10 of 40 rows: sqrt(4) * 10 = 20; never
        // fewer than seen.
        assert_eq!(estimate(&sample(10, 10, 10, false), 40), Some(20));
        assert_eq!(estimate(&sample(10, 5, 0, false), 1_000), Some(5));
    }

    #[test]
    fn nulls_are_not_values() {
        let s = DistinctSample {
            rows: 50,
            nulls: 50,
            distinct: 50,
            singletons: 50,
            whole: false,
            failed_within_ms: 0,
        };
        // Half the 10,000 rows are values: sqrt(5000 / 50) * 50 = 500.
        assert_eq!(estimate(&s, 10_000), Some(500));
        let empty = DistinctSample {
            rows: 0,
            nulls: 4,
            distinct: 0,
            singletons: 0,
            whole: false,
            failed_within_ms: 0,
        };
        assert_eq!(estimate(&empty, 100), Some(0));
        assert_eq!(estimate(&sample(0, 0, 0, false), 100), None);
    }

    #[test]
    fn a_counter_counts_values_singletons_and_nulls() {
        let mut c = Counter::default();
        let a: ArrayRef = Arc::new(StringArray::from(vec![
            Some("F"),
            Some("O"),
            None,
            Some("F"),
            Some("P"),
        ]));
        c.add(&a).unwrap();
        let s = c.sample(true);
        assert_eq!(
            (s.rows, s.nulls, s.distinct, s.singletons, s.whole),
            (4, 1, 3, 2, true)
        );
    }

    #[test]
    fn a_counter_stops_at_its_cap_and_is_no_longer_whole() {
        let mut c = Counter::default();
        let n = i32::try_from(MAX_TRACKED + 10).unwrap();
        let a: ArrayRef = Arc::new(Int32Array::from_iter_values(0..n));
        c.add(&a).unwrap();
        let s = c.sample(true);
        assert_eq!(s.distinct, MAX_TRACKED as u64);
        assert_eq!(s.rows, MAX_TRACKED as u64);
        assert!(!s.whole);
        // Further batches are not counted.
        c.add(&a).unwrap();
        assert_eq!(c.sample(true).rows, MAX_TRACKED as u64);
    }

    /// Each outcome's line, as `EXPLAIN COST` prints it after the table.
    #[test]
    fn outcomes_say_whether_the_defaults_stayed() {
        let line = |result| {
            SampleOutcome {
                columns: vec!["a".into(), "b".into()],
                result,
            }
            .to_string()
        };
        assert_eq!(
            line(SampleResult::Sampled {
                rows: 4096,
                whole: false,
                took: Duration::from_millis(12),
            }),
            "a, b: sampled 4096 rows"
        );
        let timed = SampleOutcome {
            columns: vec!["a".into()],
            result: SampleResult::Sampled {
                rows: 1,
                whole: false,
                took: Duration::from_millis(12),
            },
        };
        assert_eq!(timed.with_time(), "a: sampled 1 rows in 12 ms");
        assert_eq!(
            line(SampleResult::Sampled {
                rows: 25,
                whole: true,
                took: Duration::ZERO,
            }),
            "a, b: sampled 25 rows (the whole table)"
        );
        assert_eq!(
            line(SampleResult::Cached),
            "a, b: sampled by an earlier query"
        );
        assert_eq!(
            line(SampleResult::TimedOut {
                after: Duration::from_millis(1),
            }),
            "a, b: timed out after 1 ms, defaults kept"
        );
        assert_eq!(
            line(SampleResult::TimedOutBefore),
            "a, b: an earlier query's sample with as much time did not finish, defaults kept"
        );
        assert_eq!(
            line(SampleResult::Failed),
            "a, b: the read failed, defaults kept"
        );
        assert_eq!(
            line(SampleResult::NoTime),
            "a, b: no sampling time left, defaults kept"
        );
    }
}
