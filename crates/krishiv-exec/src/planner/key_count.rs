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
//!   the rewrite, never a result). The set's bytes, and the scans', are
//!   reserved from the session's pool ([`KeyOptions::pool`]: the planner
//!   has no query pool yet, so the arbitrator and
//!   `krishiv.session.memoryLimit` see the count there) before the set
//!   grows; a refused reservation answers unknown;
//! - past the time limit (`krishiv.optimizer.discoverKeys.timeout`) the
//!   answer is unknown; partition and unknown columns are unknown, nothing
//!   read.
//!
//! Answers are cached in [`krishiv_io::KeyFacts`] under every file's version,
//! the sorted columns and how the values were decoded (column types, format
//! and read options); a timed-out count is kept with the time it had, so
//! only a count with more time tries again (as samples do). A read error
//! and a refused reservation are not cached. A unique
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
    Int32Type, Int64Type, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef};
use arrow_row::{RowConverter, SortField};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::ErrorClass;
use krishiv_io::{ByteRange, CacheKey, ColumnPath, KeyFact, KeyFactKey, ScanRequest, ScanStream};
use krishiv_memory::{ConsumerId, MemoryPool, Reservation};

use super::file_table::FileTable;
use crate::Result;
use crate::error::internal;

/// Bytes the hash set is charged per slot: the 8-byte hash and a control
/// byte, rounded up for the table's spare capacity.
const SLOT_BYTES: usize = 10;

/// How much a key count may take.
#[derive(Debug, Clone)]
pub struct KeyOptions {
    /// Time left for this count (what is left of
    /// `krishiv.optimizer.discoverKeys.timeout` for the query). Past it the
    /// answer is unknown; zero reads nothing.
    pub timeout: Duration,
    /// The pool the count's memory (the hash set, the scans' buffers) is
    /// reserved from: the session's, so the arbitrator and
    /// `krishiv.session.memoryLimit` see it. `None` takes the I/O context's
    /// pool, else (embedded use) one of the count's own.
    pub pool: Option<Arc<dyn MemoryPool>>,
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
}

/// A consumer of the count's pool, unregistered when dropped: on the
/// normal return, on a read error, and when the count's future is dropped
/// (its time ran out, or the query was cancelled while planning). Declared
/// before the reservations it backs, so they drop (and release) first.
struct Registered {
    pool: Arc<dyn MemoryPool>,
    id: ConsumerId,
}

impl Registered {
    fn new(pool: &Arc<dyn MemoryPool>, name: &str) -> Self {
        Self {
            pool: Arc::clone(pool),
            id: pool.register(name, None),
        }
    }

    fn reservation(&self) -> Reservation {
        Reservation::new(Arc::clone(&self.pool), self.id)
    }
}

impl Drop for Registered {
    fn drop(&mut self) {
        self.pool.unregister(self.id);
    }
}

/// Whether the values of an integer-like `column` keep strictly increasing
/// after `last`, NULLs skipped, counting the values in `rows`; `None` for a
/// type the fast path does not read. Compares in place: nothing allocated.
fn increasing(column: &ArrayRef, last: &mut Option<i128>, rows: &mut u64) -> Option<bool> {
    fn go<T: arrow_array::ArrowPrimitiveType>(
        column: &ArrayRef,
        last: &mut Option<i128>,
        rows: &mut u64,
    ) -> Option<bool>
    where
        i128: From<T::Native>,
    {
        for v in column.as_primitive_opt::<T>()?.iter().flatten() {
            let v = i128::from(v);
            if last.is_some_and(|l| v <= l) {
                return Some(false);
            }
            *last = Some(v);
            *rows += 1;
        }
        Some(true)
    }
    match column.data_type() {
        DataType::Int8 => go::<Int8Type>(column, last, rows),
        DataType::Int16 => go::<Int16Type>(column, last, rows),
        DataType::Int32 => go::<Int32Type>(column, last, rows),
        DataType::Int64 => go::<Int64Type>(column, last, rows),
        DataType::UInt8 => go::<UInt8Type>(column, last, rows),
        DataType::UInt16 => go::<UInt16Type>(column, last, rows),
        DataType::UInt32 => go::<UInt32Type>(column, last, rows),
        DataType::UInt64 => go::<UInt64Type>(column, last, rows),
        DataType::Date32 => go::<Date32Type>(column, last, rows),
        DataType::Date64 => go::<Date64Type>(column, last, rows),
        DataType::Decimal32(..) => go::<Decimal32Type>(column, last, rows),
        DataType::Decimal64(..) => go::<Decimal64Type>(column, last, rows),
        DataType::Decimal128(..) => go::<Decimal128Type>(column, last, rows),
        DataType::Timestamp(unit, _) => match unit {
            TimeUnit::Second => go::<TimestampSecondType>(column, last, rows),
            TimeUnit::Millisecond => go::<TimestampMillisecondType>(column, last, rows),
            TimeUnit::Microsecond => go::<TimestampMicrosecondType>(column, last, rows),
            TimeUnit::Nanosecond => go::<TimestampNanosecondType>(column, last, rows),
        },
        _ => None,
    }
}

/// Whether the fast path reads columns of type `t`.
fn sortable(t: &DataType) -> bool {
    matches!(
        t,
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
}

/// The set of key-row hashes, its memory reserved before it grows.
struct Seen {
    set: HashSet<u64, BuildHasherDefault<Identity>>,
    reservation: Reservation,
}

/// The buckets a hash set allocates to hold `items` (a power of two, 7/8
/// full; 4 or 8 for tiny sets), never fewer than the table allocates.
fn buckets_for(items: usize) -> usize {
    if items < 4 {
        4
    } else if items < 8 {
        8
    } else {
        (items.saturating_mul(8) / 7)
            .saturating_add(1)
            .next_power_of_two()
    }
}

impl Seen {
    fn new(consumer: &Registered) -> Self {
        Self {
            set: HashSet::default(),
            reservation: consumer.reservation(),
        }
    }

    /// Room for `more` hashes, reserved before the set grows; `false` when
    /// the pool refuses it (the set is then left as it was).
    fn make_room(&mut self, more: usize) -> bool {
        let want = self.set.len().saturating_add(more);
        if want <= self.set.capacity() {
            return true;
        }
        // A growing table takes at least one more item than its capacity.
        let items = want.max(self.set.capacity().saturating_add(1));
        let bytes = buckets_for(items).saturating_mul(SLOT_BYTES);
        let have = self.reservation.size();
        if bytes > have && self.reservation.try_grow(bytes - have).is_err() {
            return false;
        }
        self.set.reserve(more);
        true
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
        let key = KeyFactKey::new(files, &columns, self.decoding(&columns));
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
        // The caller's pool (the session's, so the arbitrator and
        // `krishiv.session.memoryLimit` see the count), else the I/O
        // context's, else (embedded use) a pool of the count's own.
        let pool: Arc<dyn MemoryPool> = match (&options.pool, self.io().memory_pool()) {
            (Some(p), _) => Arc::clone(p),
            (None, Some(p)) => p,
            (None, None) => krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default()),
        };
        let started = Instant::now();
        let counted = match tokio::time::timeout(options.timeout, self.count(&columns, &pool)).await
        {
            Ok(Ok(c)) => c,
            // The scans' buffers refused by the pool: as the set's refusal.
            Ok(Err(e)) if e.error_class() == ErrorClass::UnableToAcquireMemory => Counted::NoMemory,
            Ok(Err(e)) => {
                // Not cached: a read error may be transient (a store's 503);
                // the next query counts again.
                self.keep_outcome(outcome(KeyResult::Unknown(KeyUnknown::Failed)));
                return Err(e);
            }
            Err(_) => {
                tracing::debug!(columns = ?columns, "key count timed out");
                cache.insert(
                    key,
                    KeyFact::Unknown {
                        within_ms: timeout_ms.max(1),
                    },
                );
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
            // Not cached: memory is the session's, not the files'.
            Counted::NoMemory => KeyResult::Unknown(KeyUnknown::NoMemory),
        };
        Ok(self.keep_outcome(outcome(result)))
    }

    /// What decides the values a count of `columns` sees, for the cache key:
    /// each column's Arrow field (type, nullability and the metadata that
    /// carries collation or `CHAR` semantics), the format and its parsed
    /// read options (session fallbacks included), and the scan settings
    /// that change which rows are read.
    ///
    /// The field mapping is not part of it: a `FileTable`'s scans and its
    /// count both map file columns by name, case-insensitively
    /// (`FieldMapping::default()`). If file scans ever follow
    /// `spark.sql.caseSensitive`, or a `FileTable` gets field-id or column
    /// mapping, `ScanRequest::mapping` must join this text.
    fn decoding(&self, columns: &[String]) -> String {
        let fields: Vec<String> = columns
            .iter()
            .filter_map(|c| self.table_schema().field_with_name(c).ok())
            .map(|f| {
                // The metadata in a fixed order: equal fields, equal text.
                let metadata: std::collections::BTreeMap<&String, &String> =
                    f.metadata().iter().collect();
                format!(
                    "{}:{:?}:{}:{metadata:?}",
                    f.name(),
                    f.data_type(),
                    f.is_nullable()
                )
            })
            .collect();
        let scan = self.scan_options();
        format!(
            "{}|{}|{:?}|corrupt={},missing={}",
            fields.join(";"),
            self.format().name(),
            self.options(),
            scan.ignore_corrupt_files,
            scan.ignore_missing_files,
        )
    }

    /// The count: the fast path for one integer-like column, else (or once
    /// it meets a value that does not increase) the hash path.
    async fn count(&self, columns: &[String], pool: &Arc<dyn MemoryPool>) -> Result<Counted> {
        let fast = columns.len() == 1
            && self
                .table_schema()
                .field_with_name(&columns[0])
                .is_ok_and(|f| sortable(f.data_type()));
        if fast && let Some(rows) = self.read_sorted(&columns[0], pool).await? {
            return Ok(Counted::Unique { rows, sorted: true });
        }
        self.read_hashed(columns, pool).await
    }

    /// Opens a scan of `columns` of `file` under a consumer of `pool`.
    async fn open_scan(
        &self,
        file: &krishiv_io::FileRef,
        columns: &[String],
        consumer: &Registered,
    ) -> Result<ScanStream> {
        let projection: Vec<ColumnPath> =
            columns.iter().map(|c| ColumnPath::new(c.clone())).collect();
        let req = ScanRequest::new(Arc::clone(self.table_schema()), projection);
        ScanStream::open(
            self.io(),
            self.format().as_ref(),
            file.clone(),
            ByteRange::whole(file.size),
            &req,
            &self.options(),
            consumer.reservation(),
            self.scan_options(),
        )
        .await
    }

    /// The rows of `column` when its values strictly increase over every
    /// file in order; `None` at the first that does not.
    async fn read_sorted(&self, column: &str, pool: &Arc<dyn MemoryPool>) -> Result<Option<u64>> {
        let columns = [column.to_owned()];
        let mut last = None;
        let mut rows = 0_u64;
        for file in self.files() {
            let consumer = Registered::new(pool, "key count scan");
            let mut stream = self.open_scan(file, &columns, &consumer).await?;
            while let Some(batch) = stream.next_batch().await? {
                let Some(values) = batch.column_by_name(column) else {
                    return Err(internal(format!("the key count read no column {column}")));
                };
                if increasing(values, &mut last, &mut rows) != Some(true) {
                    return Ok(None);
                }
            }
        }
        Ok(Some(rows))
    }

    /// Reads `columns` of every file in order into a set of row hashes,
    /// stopping at the first repeat.
    async fn read_hashed(&self, columns: &[String], pool: &Arc<dyn MemoryPool>) -> Result<Counted> {
        let schema = self.table_schema();
        let set_consumer = Registered::new(pool, "key count");
        let mut seen = Seen::new(&set_consumer);
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
        let mut rows = 0_u64;
        for file in self.files() {
            let consumer = Registered::new(pool, "key count scan");
            let mut stream = self.open_scan(file, columns, &consumer).await?;
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
                if !seen.make_room(batch.num_rows()) {
                    return Ok(Counted::NoMemory);
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
                        return Ok(Counted::Repeat { rows });
                    }
                }
            }
        }
        Ok(Counted::Unique {
            rows,
            sorted: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use arrow_array::{ArrayRef, Date32Array, Decimal128Array, Int64Array, StringArray};

    use super::{KeyOutcome, KeyResult, KeyUnknown, increasing};

    #[test]
    fn integer_like_columns_compare_in_place() {
        let (mut last, mut rows) = (None, 0);
        let a: ArrayRef = Arc::new(Int64Array::from(vec![Some(-3), None, Some(4)]));
        assert_eq!(increasing(&a, &mut last, &mut rows), Some(true));
        assert_eq!((last, rows), (Some(4), 2));
        // Across batches: 4 does not increase on 4.
        let b: ArrayRef = Arc::new(Int64Array::from(vec![4]));
        assert_eq!(increasing(&b, &mut last, &mut rows), Some(false));
        let d: ArrayRef = Arc::new(Date32Array::from(vec![7]));
        assert_eq!(increasing(&d, &mut None, &mut 0), Some(true));
        let m: ArrayRef = Arc::new(
            Decimal128Array::from(vec![12_345, 12_344])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        );
        assert_eq!(increasing(&m, &mut None, &mut 0), Some(false));
        let s: ArrayRef = Arc::new(StringArray::from(vec!["a"]));
        assert_eq!(increasing(&s, &mut None, &mut 0), None);
    }

    /// The set's memory is reserved before it grows; a refusal leaves the
    /// set as it was; the consumer goes when its guard does.
    #[test]
    fn the_hash_set_reserves_before_it_grows_and_a_refusal_stops_it() {
        use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
        let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig {
            limit: Some(64 * 1024),
            quantum: 1,
            spill_scratch_reserve: 0,
            ..MemoryConfig::default()
        });
        let consumers = |p: &Arc<dyn MemoryPool>| {
            p.snapshot()
                .entries
                .iter()
                .filter(|e| e.is_consumer)
                .count()
        };
        let consumer = super::Registered::new(&pool, "key count");
        assert_eq!(consumers(&pool), 1);
        let mut seen = super::Seen::new(&consumer);
        for more in [1, 10, 100, 1_000] {
            assert!(seen.make_room(more));
            assert!(
                seen.reservation.size() >= seen.set.capacity() * 9,
                "{more}: {} reserved for {} slots",
                seen.reservation.size(),
                seen.set.capacity()
            );
            for h in 0..more as u64 {
                seen.set.insert(h);
            }
        }
        let capacity = seen.set.capacity();
        assert!(!seen.make_room(100_000), "past the pool's limit");
        assert_eq!(seen.set.capacity(), capacity, "nothing allocated");
        drop(seen);
        drop(consumer);
        assert_eq!(consumers(&pool), 0);
        assert_eq!(pool.reserved(), 0);
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
