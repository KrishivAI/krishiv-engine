// SPDX-License-Identifier: Apache-2.0
//! Plan-time key counts of file tables (design P2b A5 (d), unit 2).
//!
//! `PayloadAboveAggregate` needs a dimension's key proven unique among its
//! non-NULL keys (FD2); plain Parquet declares no key. When the optimizer
//! asks for one ([`krishiv_optimizer::Statistics::want_unique`], recorded by
//! [`FileTable::want_unique`](super::table::TableProvider::want_unique)),
//! the session calls [`FileTable::discover_unique`]: the whole table (every
//! file, in order), those columns only, through the ordinary [`ScanStream`],
//! rows with a NULL key column skipped. The answer is a fact, never an
//! estimate; any doubt answers "unknown", which proves nothing:
//!
//! - one integer, date, timestamp or decimal column whose values strictly
//!   increase in file order across all files is unique, with no memory
//!   (TPC-H `customer` is written sorted by `c_custkey`); the first value
//!   that does not increase starts the hash path over;
//! - the hash path keeps a set of 64-bit hashes of the key rows (in
//!   `arrow_row` form, as distinct-value samples hash them); a repeated hash
//!   answers *not unique* at once and stops reading (a collision only loses
//!   the rewrite, never a result). The set's bytes are reserved from the
//!   query's pool as it grows; a refused reservation answers unknown;
//! - past the time limit (`krishiv.optimizer.discoverKeys.timeout`) the
//!   answer is unknown; partition and unknown columns are unknown, nothing
//!   read.
//!
//! Answers are cached in [`krishiv_io::KeyFacts`] under every file's version
//! and the sorted columns; an unknown answer is kept with the time it had,
//! so only a count with more time tries again (as samples do). A unique
//! answer joins the table's
//! [`unique_keys`](super::table::TableProvider::unique_keys).

use std::collections::HashSet;
use std::fmt;
use std::hash::{BuildHasherDefault, DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::cast::AsArray as _;
use arrow_array::types::{
    Date32Type, Date64Type, Decimal32Type, Decimal64Type, Decimal128Type, Int8Type, Int16Type,
    Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef};
use arrow_row::{RowConverter, SortField};
use arrow_schema::{DataType, TimeUnit};
use krishiv_io::{ByteRange, CacheKey, ColumnPath, KeyFact, KeyFactKey, ScanRequest, ScanStream};

use super::file_table::FileTable;
use crate::Result;
use crate::error::internal;

/// Bytes the hash set is charged per slot: the 8-byte hash and a control
/// byte, rounded up for the table's spare capacity.
const SLOT_BYTES: usize = 10;

/// How much a key count may take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyOptions {
    /// Time left for this count (what is left of
    /// `krishiv.optimizer.discoverKeys.timeout` for the query). Past it the
    /// answer is unknown; zero reads nothing.
    pub timeout: Duration,
}

/// What a key count did for some columns of a table, for `EXPLAIN VERBOSE`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyOutcome {
    /// The columns, sorted.
    pub columns: Vec<String>,
    /// What it found.
    pub result: KeyResult,
}

/// How a key count ended ([`KeyOutcome`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyResult {
    /// Counted now: `rows` rows with every key column non-NULL read in
    /// `took`.
    Discovered {
        /// Whether the columns are unique.
        unique: bool,
        /// Non-NULL key rows read (up to the first repeat when not unique).
        rows: u64,
        /// Whether the strictly increasing fast path decided it.
        sorted: bool,
        /// The read's time.
        took: Duration,
    },
    /// An earlier query's count of the same file versions.
    Cached {
        /// Whether the columns are unique.
        unique: bool,
    },
    /// Nothing proven.
    Unknown(KeyUnknown),
}

/// Why a key count proved nothing ([`KeyResult::Unknown`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyUnknown {
    /// A column is a partition column or not a column of the files: nothing
    /// is read.
    NotFileColumns,
    /// The count did not finish within `after`.
    TimedOut {
        /// The time it had.
        after: Duration,
    },
    /// An earlier query's count of the same files did not finish within as
    /// much time as this one has: nothing is read.
    TimedOutBefore,
    /// The pool refused the hash set's memory.
    NoMemory,
    /// The read failed.
    Failed,
    /// No counting time was left (the query's other counts spent it).
    NoTime,
}

impl KeyOutcome {
    /// Whether the columns were proven unique.
    #[must_use]
    pub fn is_unique(&self) -> bool {
        matches!(
            self.result,
            KeyResult::Discovered { unique: true, .. } | KeyResult::Cached { unique: true }
        )
    }

    /// The text with the read's time (`EXPLAIN VERBOSE`); `Display` leaves
    /// it out.
    #[must_use]
    pub fn with_time(&self) -> String {
        match self.result {
            KeyResult::Discovered { took, .. } => {
                format!("{self} in {} ms", took.as_millis())
            }
            _ => self.to_string(),
        }
    }
}

impl fmt::Display for KeyOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let verdict = |unique: bool| if unique { "unique" } else { "not unique" };
        write!(f, "key [{}]: ", self.columns.join(", "))?;
        match self.result {
            KeyResult::Discovered {
                unique,
                rows,
                sorted,
                ..
            } => write!(
                f,
                "discovered {} over {rows} rows{}",
                verdict(unique),
                if sorted { " (sorted)" } else { "" }
            ),
            KeyResult::Cached { unique } => {
                write!(f, "cached {} (an earlier query's count)", verdict(unique))
            }
            KeyResult::Unknown(why) => {
                f.write_str("unknown, ")?;
                match why {
                    KeyUnknown::NotFileColumns => {
                        f.write_str("not columns of the files, nothing read")
                    }
                    KeyUnknown::TimedOut { after } => {
                        write!(f, "timed out after {} ms", after.as_millis())
                    }
                    KeyUnknown::TimedOutBefore => {
                        f.write_str("an earlier query's count with as much time did not finish")
                    }
                    KeyUnknown::NoMemory => f.write_str("the pool refused the count's memory"),
                    KeyUnknown::Failed => f.write_str("the read failed"),
                    KeyUnknown::NoTime => f.write_str("no counting time left"),
                }
            }
        }
    }
}

/// A hasher for values that are hashes already.
#[derive(Default)]
struct Identity(u64);

impl Hasher for Identity {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(*b);
        }
    }

    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

/// How a count's read ended.
enum Counted {
    /// Every row read: unique.
    Unique { rows: u64, sorted: bool },
    /// A repeat after `rows` rows.
    Repeat { rows: u64 },
    /// The pool refused the set's memory.
    NoMemory,
    /// The fast path failed: start over with the hash path.
    NotSorted,
}

/// The values of an integer-like column as `i128`, in row order, `None` for
/// NULL; `None` for a type the fast path does not read.
fn ordered(column: &ArrayRef) -> Option<Vec<Option<i128>>> {
    macro_rules! read {
        ($t:ty) => {
            column
                .as_primitive_opt::<$t>()?
                .iter()
                .map(|v| v.map(i128::from))
                .collect()
        };
    }
    Some(match column.data_type() {
        DataType::Int8 => read!(Int8Type),
        DataType::Int16 => read!(Int16Type),
        DataType::Int32 => read!(Int32Type),
        DataType::Int64 => read!(Int64Type),
        DataType::UInt8 => read!(UInt8Type),
        DataType::UInt16 => read!(UInt16Type),
        DataType::UInt32 => read!(UInt32Type),
        DataType::UInt64 => read!(UInt64Type),
        DataType::Date32 => read!(Date32Type),
        DataType::Date64 => read!(Date64Type),
        DataType::Decimal32(..) => read!(Decimal32Type),
        DataType::Decimal64(..) => read!(Decimal64Type),
        DataType::Decimal128(..) => column
            .as_primitive_opt::<Decimal128Type>()?
            .iter()
            .collect(),
        DataType::Timestamp(unit, _) => match unit {
            TimeUnit::Second => read!(arrow_array::types::TimestampSecondType),
            TimeUnit::Millisecond => read!(arrow_array::types::TimestampMillisecondType),
            TimeUnit::Microsecond => read!(arrow_array::types::TimestampMicrosecondType),
            TimeUnit::Nanosecond => read!(arrow_array::types::TimestampNanosecondType),
        },
        _ => return None,
    })
}

/// The set of key-row hashes, its memory reserved as it grows.
struct Seen {
    set: HashSet<u64, BuildHasherDefault<Identity>>,
    reservation: krishiv_memory::Reservation,
}

impl Seen {
    /// Room for `more` hashes; `false` when the pool refuses it.
    fn make_room(&mut self, more: usize) -> bool {
        let want = self.set.len().saturating_add(more);
        if want <= self.set.capacity() {
            return true;
        }
        // The table's next size (7/8 load, a power of two of buckets).
        let buckets = (want.saturating_mul(8) / 7).max(4).next_power_of_two();
        let bytes = buckets.saturating_mul(SLOT_BYTES);
        let have = self.reservation.size();
        if bytes > have && self.reservation.try_grow(bytes - have).is_err() {
            return false;
        }
        self.set.reserve(more);
        let actual = self.set.capacity().saturating_mul(SLOT_BYTES);
        let have = self.reservation.size();
        actual <= have || self.reservation.try_grow(actual - have).is_ok()
    }
}

impl FileTable {
    /// Counts whether `columns` (names of the table's file columns) are
    /// unique among the table's rows whose columns of the set are all
    /// non-NULL, within `options` (module docs). A unique answer joins
    /// [`unique_keys`](super::table::TableProvider::unique_keys); the
    /// outcome is also kept for [`key_outcomes`](Self::key_outcomes).
    ///
    /// # Errors
    /// Read errors of the table's files (the outcome is then unknown).
    pub async fn discover_unique(
        &self,
        columns: &[String],
        options: KeyOptions,
    ) -> Result<KeyOutcome> {
        let mut columns = columns.to_vec();
        columns.sort();
        columns.dedup();
        let outcome = |result| KeyOutcome {
            columns: columns.clone(),
            result,
        };
        if columns.is_empty()
            || columns
                .iter()
                .any(|c| self.table_schema().index_of(c).is_err() || self.is_partition_column(c))
        {
            return Ok(self.keep_outcome(outcome(KeyResult::Unknown(KeyUnknown::NotFileColumns))));
        }
        let format = self.format();
        let files: Vec<CacheKey> = self
            .files()
            .iter()
            .map(|f| CacheKey::new(format.name(), f, None))
            .collect();
        let key = KeyFactKey::new(files, &columns);
        let cache = self.io().scheduler().key_facts();
        let timeout_ms = u64::try_from(options.timeout.as_millis()).unwrap_or(u64::MAX);
        match cache.get(&key) {
            Some(KeyFact::Unique) => {
                self.keep_key(&columns);
                return Ok(self.keep_outcome(outcome(KeyResult::Cached { unique: true })));
            }
            Some(KeyFact::NotUnique) => {
                return Ok(self.keep_outcome(outcome(KeyResult::Cached { unique: false })));
            }
            // Tried again only with more time.
            Some(KeyFact::Unknown { within_ms }) if timeout_ms <= within_ms => {
                return Ok(
                    self.keep_outcome(outcome(KeyResult::Unknown(KeyUnknown::TimedOutBefore)))
                );
            }
            _ => {}
        }
        if options.timeout.is_zero() {
            return Ok(self.keep_outcome(outcome(KeyResult::Unknown(KeyUnknown::NoTime))));
        }
        let started = Instant::now();
        let failed = KeyFact::Unknown {
            within_ms: timeout_ms.max(1),
        };
        let counted = match tokio::time::timeout(options.timeout, self.count(&columns)).await {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => {
                cache.insert(key, failed);
                self.keep_outcome(outcome(KeyResult::Unknown(KeyUnknown::Failed)));
                return Err(e);
            }
            Err(_) => {
                tracing::debug!(columns = ?columns, "key count timed out");
                cache.insert(key, failed);
                return Ok(
                    self.keep_outcome(outcome(KeyResult::Unknown(KeyUnknown::TimedOut {
                        after: options.timeout,
                    }))),
                );
            }
        };
        let took = started.elapsed();
        let result = match counted {
            Counted::Unique { rows, sorted } => {
                cache.insert(key, KeyFact::Unique);
                self.keep_key(&columns);
                KeyResult::Discovered {
                    unique: true,
                    rows,
                    sorted,
                    took,
                }
            }
            Counted::Repeat { rows } => {
                cache.insert(key, KeyFact::NotUnique);
                KeyResult::Discovered {
                    unique: false,
                    rows,
                    sorted: false,
                    took,
                }
            }
            // Not cached: memory is the query's, not the files'.
            Counted::NoMemory | Counted::NotSorted => KeyResult::Unknown(KeyUnknown::NoMemory),
        };
        Ok(self.keep_outcome(outcome(result)))
    }

    /// The count: the fast path for one integer-like column, else (or once
    /// it fails) the hash path.
    async fn count(&self, columns: &[String]) -> Result<Counted> {
        let sortable = columns.len() == 1
            && self
                .table_schema()
                .field_with_name(&columns[0])
                .is_ok_and(|f| {
                    matches!(
                        f.data_type(),
                        DataType::Int8
                            | DataType::Int16
                            | DataType::Int32
                            | DataType::Int64
                            | DataType::UInt8
                            | DataType::UInt16
                            | DataType::UInt32
                            | DataType::UInt64
                            | DataType::Date32
                            | DataType::Date64
                            | DataType::Decimal32(..)
                            | DataType::Decimal64(..)
                            | DataType::Decimal128(..)
                            | DataType::Timestamp(..)
                    )
                });
        if sortable {
            match self.read(columns, true).await? {
                Counted::NotSorted => {}
                done => return Ok(done),
            }
        }
        self.read(columns, false).await
    }

    /// Reads `columns` of every file in order: strictly increasing values of
    /// one column (`sorted`), else distinct row hashes.
    async fn read(&self, columns: &[String], sorted: bool) -> Result<Counted> {
        let schema = self.table_schema();
        let projection: Vec<ColumnPath> =
            columns.iter().map(|c| ColumnPath::new(c.clone())).collect();
        // The query's pool, as the footer reads and samples use; a context
        // without one (embedded tests) counts under a pool of its own.
        let pool: Arc<dyn krishiv_memory::MemoryPool> = match self.io().memory_pool() {
            Some(p) => Arc::clone(p),
            None => krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default()),
        };
        let set_id = pool.register("key count", None);
        let mut seen = Seen {
            set: HashSet::default(),
            reservation: krishiv_memory::Reservation::new(Arc::clone(&pool), set_id),
        };
        let fields = columns
            .iter()
            .map(|c| {
                schema
                    .field_with_name(c)
                    .map(|f| SortField::new(f.data_type().clone()))
                    .map_err(|e| internal(e.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let converter = RowConverter::new(fields).map_err(|e| internal(e.to_string()))?;
        let mut last: Option<i128> = None;
        let mut rows = 0_u64;
        let mut out = None;
        let mut scans = Vec::new();
        'files: for file in self.files() {
            let req = ScanRequest::new(Arc::clone(schema), projection.clone());
            let id = pool.register("key count scan", None);
            scans.push(id);
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
                let cols: Vec<ArrayRef> = columns
                    .iter()
                    .map(|c| {
                        batch
                            .column_by_name(c)
                            .cloned()
                            .ok_or_else(|| internal(format!("the key count read no column {c}")))
                    })
                    .collect::<Result<_>>()?;
                if sorted {
                    let Some(values) = cols.first().and_then(ordered) else {
                        out = Some(Counted::NotSorted);
                        break 'files;
                    };
                    for v in values.into_iter().flatten() {
                        if last.is_some_and(|l| v <= l) {
                            out = Some(Counted::NotSorted);
                            break 'files;
                        }
                        last = Some(v);
                        rows += 1;
                    }
                    continue;
                }
                if !seen.make_room(batch.num_rows()) {
                    out = Some(Counted::NoMemory);
                    break 'files;
                }
                let keys = converter
                    .convert_columns(&cols)
                    .map_err(|e| internal(e.to_string()))?;
                for i in 0..batch.num_rows() {
                    if cols.iter().any(|c| c.is_null(i)) {
                        continue;
                    }
                    let mut h = DefaultHasher::new();
                    keys.row(i).as_ref().hash(&mut h);
                    rows += 1;
                    if !seen.set.insert(h.finish()) {
                        out = Some(Counted::Repeat { rows });
                        break 'files;
                    }
                }
            }
        }
        drop(seen);
        pool.unregister(set_id);
        for id in scans {
            pool.unregister(id);
        }
        Ok(out.unwrap_or(Counted::Unique { rows, sorted }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use arrow_array::{ArrayRef, Date32Array, Decimal128Array, Int64Array, StringArray};

    use super::{KeyOutcome, KeyResult, KeyUnknown, ordered};

    #[test]
    fn integer_like_columns_read_in_order() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![Some(3), None, Some(-1)]));
        assert_eq!(ordered(&a), Some(vec![Some(3), None, Some(-1)]));
        let d: ArrayRef = Arc::new(Date32Array::from(vec![7]));
        assert_eq!(ordered(&d), Some(vec![Some(7)]));
        let m: ArrayRef = Arc::new(
            Decimal128Array::from(vec![12_345])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        );
        assert_eq!(ordered(&m), Some(vec![Some(12_345)]));
        let s: ArrayRef = Arc::new(StringArray::from(vec!["a"]));
        assert_eq!(ordered(&s), None);
    }

    /// The set's memory comes from the pool; a refusal stops the count.
    #[test]
    fn the_hash_set_reserves_its_memory_and_a_refusal_stops_it() {
        use krishiv_memory::{MemoryConfig, MemoryPool as _, Reservation, TreePool};
        let pool = TreePool::new(&MemoryConfig {
            limit: Some(64 * 1024),
            quantum: 1,
            spill_scratch_reserve: 0,
            ..MemoryConfig::default()
        });
        let id = pool.register("key count", None);
        let mut seen = super::Seen {
            set: std::collections::HashSet::default(),
            reservation: Reservation::new(Arc::clone(&pool) as _, id),
        };
        assert!(seen.make_room(1_000));
        assert!(seen.reservation.size() >= seen.set.capacity() * super::SLOT_BYTES);
        for h in 0..1_000 {
            seen.set.insert(h);
        }
        assert!(!seen.make_room(100_000), "past the pool's limit");
        drop(seen);
        pool.unregister(id);
    }

    #[test]
    fn outcomes_say_what_the_count_found() {
        let line = |result| {
            KeyOutcome {
                columns: vec!["a".into(), "b".into()],
                result,
            }
            .to_string()
        };
        assert_eq!(
            line(KeyResult::Discovered {
                unique: true,
                rows: 25,
                sorted: true,
                took: Duration::ZERO,
            }),
            "key [a, b]: discovered unique over 25 rows (sorted)"
        );
        assert_eq!(
            line(KeyResult::Discovered {
                unique: false,
                rows: 3,
                sorted: false,
                took: Duration::ZERO,
            }),
            "key [a, b]: discovered not unique over 3 rows"
        );
        assert_eq!(
            line(KeyResult::Cached { unique: true }),
            "key [a, b]: cached unique (an earlier query's count)"
        );
        assert_eq!(
            line(KeyResult::Unknown(KeyUnknown::TimedOut {
                after: Duration::from_millis(5)
            })),
            "key [a, b]: unknown, timed out after 5 ms"
        );
        assert_eq!(
            line(KeyResult::Unknown(KeyUnknown::NotFileColumns)),
            "key [a, b]: unknown, not columns of the files, nothing read"
        );
    }
}
