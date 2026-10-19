// SPDX-License-Identifier: Apache-2.0
//! Runtime filters (design P2-T163): what a producer — a hash join's build
//! side today, a Top-N bound later — learns about its keys while the query
//! runs, handed to the scans that feed the other side so they skip files,
//! row groups and rows the producer would drop anyway.
//!
//! # Contract
//! - A [`RuntimeFilter`] belongs to one query: the planner creates it, the
//!   producer and every consumer hold an `Arc`, and it is dropped with the
//!   plan. It targets one table column of one scan ([`RuntimeFilter::target`])
//!   and compares values in its key type ([`RuntimeFilter::key_type`]).
//! - The producer publishes immutable [`Published`] snapshots, each with
//!   its own generation, under a write lock; the atomic generation
//!   [hint](RuntimeFilter::generation_hint) only tells a consumer that
//!   something changed. A consumer remembers the generation it lowered,
//!   loads the hint at each check point (one atomic load) and takes the read
//!   lock only when it moved; the generation it adopts is the one inside
//!   the snapshot, so a generation is never paired with another snapshot.
//! - [`FilterSnapshot::Pending`] and [`FilterSnapshot::Disabled`] apply
//!   nothing: a consumer never waits for a filter, so one that arrives late
//!   (a scan below a pipeline breaker runs while the build does) is adopted
//!   at the next check point, and one that never arrives (a failed or
//!   cancelled producer publishes `Disabled`) only costs the pruning.
//! - **A filter only drops rows the producer would drop.** Every stage keeps
//!   a superset of the matching rows; the operator above the scan still does
//!   the exact work, so results never depend on runtime filters.
//!
//! # Stages
//! A [`ScanFilter`] is a filter as one scan level sees it (table or file
//! columns). [`ScanFilter::predicate`] lowers the current snapshot to a
//! [`PushdownExpr`] — min/max and value-set comparisons for statistics
//! (row groups, and page indexes once a reader exposes them), and a
//! [`PushdownExpr::Membership`] for stages that evaluate values (rows;
//! dictionaries once a reader exposes them) — re-typed to the column and
//! cached per generation. [`ScanFilter::evaluate`] filters rows.

mod bloom;
mod filter;
mod key;
mod sync;

use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, BooleanArray};
use arrow_schema::{DataType, Schema};

pub use self::bloom::{BLOCK_BYTES, RuntimeBloom, xxh64};
pub use self::filter::{KeyFilter, KeyFilterBuilder, KeyFilterOptions};
pub use self::key::{NormalizedKeys, normalize_key_bytes, supported_key_type, widens_to};
use self::sync::{AtomicU64, Mutex, Ordering, RwLock};
use crate::request::{ColumnPath, PushdownExpr};

/// A runtime filter's id, unique within its query (`EXPLAIN` prints it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FilterId(pub u64);

impl fmt::Display for FilterId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Why a filter applies nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DisabledReason {
    /// The producer failed or was cancelled before it finished.
    ProducerFailed,
    /// Turned off by configuration after planning.
    Off,
}

/// One published state of a filter. Immutable.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum FilterSnapshot {
    /// The producer is not done: apply nothing (yet).
    Pending,
    /// Apply nothing, ever.
    Disabled(DisabledReason),
    /// The producer's final knowledge (a join build): published once.
    Final(Arc<KeyFilter>),
}

/// A snapshot with the generation it was published as: they travel
/// together, so a consumer never pairs a generation with another snapshot.
#[derive(Debug, Clone)]
pub struct Published {
    /// Starts at 0 (`Pending`), one more per publication.
    pub generation: u64,
    /// The state.
    pub snapshot: FilterSnapshot,
}

/// Counters of one filter, summed over its consumers (`EXPLAIN ANALYZE`).
#[derive(Debug, Default)]
pub struct FilterStats {
    rows_tested: AtomicU64,
    rows_dropped: AtomicU64,
    units_pruned: AtomicU64,
    bytes_pruned: AtomicU64,
    splits_skipped: AtomicU64,
}

/// A copy of [`FilterStats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct FilterCounts {
    /// Rows the row stage evaluated.
    pub rows_tested: u64,
    /// Rows the row stage dropped.
    pub rows_dropped: u64,
    /// Units (row groups) pruned.
    pub units_pruned: u64,
    /// Compressed bytes of the pruned units.
    pub bytes_pruned: u64,
    /// Splits skipped without opening their file.
    pub splits_skipped: u64,
}

impl FilterStats {
    /// Rows tested and dropped at the row stage.
    pub fn add_rows(&self, tested: u64, dropped: u64) {
        self.rows_tested.fetch_add(tested, Ordering::Relaxed);
        self.rows_dropped.fetch_add(dropped, Ordering::Relaxed);
    }

    /// A unit of `bytes` pruned.
    pub fn add_unit(&self, bytes: u64) {
        self.units_pruned.fetch_add(1, Ordering::Relaxed);
        self.bytes_pruned.fetch_add(bytes, Ordering::Relaxed);
    }

    /// A split skipped before its file was opened.
    pub fn add_split(&self, bytes: u64) {
        self.splits_skipped.fetch_add(1, Ordering::Relaxed);
        self.bytes_pruned.fetch_add(bytes, Ordering::Relaxed);
    }

    /// The counters so far.
    #[must_use]
    pub fn counts(&self) -> FilterCounts {
        let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
        FilterCounts {
            rows_tested: get(&self.rows_tested),
            rows_dropped: get(&self.rows_dropped),
            units_pruned: get(&self.units_pruned),
            bytes_pruned: get(&self.bytes_pruned),
            splits_skipped: get(&self.splits_skipped),
        }
    }
}

/// A filter shared by its producer and consumers (see the module docs).
pub struct RuntimeFilter {
    id: FilterId,
    target: ColumnPath,
    key_type: DataType,
    hint: AtomicU64,
    state: RwLock<Arc<Published>>,
    stats: FilterStats,
}

impl fmt::Debug for RuntimeFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeFilter")
            .field("id", &self.id)
            .field("target", &self.target)
            .field("key_type", &self.key_type)
            .field("generation", &self.generation_hint())
            .finish_non_exhaustive()
    }
}

impl RuntimeFilter {
    /// A pending filter on the table column `target`, comparing in
    /// `key_type`.
    #[must_use]
    pub fn new(id: FilterId, target: ColumnPath, key_type: DataType) -> Self {
        Self {
            id,
            target,
            key_type,
            hint: AtomicU64::new(0),
            state: RwLock::new(Arc::new(Published {
                generation: 0,
                snapshot: FilterSnapshot::Pending,
            })),
            stats: FilterStats::default(),
        }
    }

    /// The id.
    #[must_use]
    pub fn id(&self) -> FilterId {
        self.id
    }

    /// The table column the filter applies to.
    #[must_use]
    pub fn target(&self) -> &ColumnPath {
        &self.target
    }

    /// The type keys are compared in.
    #[must_use]
    pub fn key_type(&self) -> &DataType {
        &self.key_type
    }

    /// The counters.
    #[must_use]
    pub fn stats(&self) -> &FilterStats {
        &self.stats
    }

    /// The latest published generation: a cheap "changed?" probe only.
    #[must_use]
    pub fn generation_hint(&self) -> u64 {
        self.hint.load(Ordering::Acquire)
    }

    /// The current snapshot with its generation.
    #[must_use]
    pub fn published(&self) -> Arc<Published> {
        Arc::clone(
            &self
                .state
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    /// Publishes `snapshot` as the next generation and returns it, unless
    /// the filter is already final or disabled (`None`; a final filter is
    /// published once).
    pub fn publish(&self, snapshot: FilterSnapshot) -> Option<u64> {
        let mut state = self
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !matches!(state.snapshot, FilterSnapshot::Pending) {
            return None;
        }
        let generation = state.generation + 1;
        *state = Arc::new(Published {
            generation,
            snapshot,
        });
        // Under the lock: a reader that sees the new hint finds this state.
        self.hint.store(generation, Ordering::Release);
        Some(generation)
    }

    /// Publishes [`FilterSnapshot::Disabled`] if nothing was published yet
    /// (a producer's drop path: failure or cancellation).
    pub fn abandon(&self, reason: DisabledReason) {
        self.publish(FilterSnapshot::Disabled(reason));
    }
}

/// The lowered form of one generation.
#[derive(Debug, Clone)]
struct Lowered {
    generation: u64,
    keys: Option<Arc<KeyFilter>>,
    predicate: Option<PushdownExpr>,
}

/// A runtime filter as one scan level sees it: the filter and the column it
/// applies to at that level (table, or one file's names and type), with the
/// lowering of its current snapshot cached per generation.
#[derive(Clone)]
pub struct ScanFilter {
    filter: Arc<RuntimeFilter>,
    /// The column and its type at this level; `None` when the file lacks
    /// it (every row reads as NULL).
    column: Option<(ColumnPath, DataType)>,
    cache: Arc<Mutex<Option<Arc<Lowered>>>>,
}

impl fmt::Debug for ScanFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScanFilter")
            .field("filter", &self.filter)
            .field("column", &self.column)
            .finish_non_exhaustive()
    }
}

impl ScanFilter {
    /// `filter` at the table level of `table`: `None` when its target is
    /// not a column of `table`.
    #[must_use]
    pub fn new(filter: Arc<RuntimeFilter>, table: &Schema) -> Option<Self> {
        let column_type = resolve(table, filter.target())?;
        let column = Some((filter.target().clone(), column_type));
        Some(Self::at(filter, column))
    }

    /// `filter` applied to `column` (a file-level view), with its own cache.
    #[must_use]
    pub fn at(filter: Arc<RuntimeFilter>, column: Option<(ColumnPath, DataType)>) -> Self {
        Self {
            filter,
            column,
            cache: Arc::new(Mutex::new(None)),
        }
    }

    /// The filter.
    #[must_use]
    pub fn filter(&self) -> &Arc<RuntimeFilter> {
        &self.filter
    }

    /// The column at this level and its type (`None`: missing, all NULL).
    #[must_use]
    pub fn column(&self) -> Option<&(ColumnPath, DataType)> {
        self.column.as_ref()
    }

    fn lowered(&self) -> Arc<Lowered> {
        let hint = self.filter.generation_hint();
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(l) = cache.as_ref()
            && l.generation == hint
        {
            return Arc::clone(l);
        }
        let published = self.filter.published();
        let keys = match &published.snapshot {
            FilterSnapshot::Final(k) => Some(Arc::clone(k)),
            FilterSnapshot::Pending | FilterSnapshot::Disabled(_) => None,
        };
        let predicate = keys.as_ref().and_then(|k| match &self.column {
            Some((c, t)) => k.lower(c, t, self.filter.id, published.generation),
            // A missing column is NULL in every row.
            None => Some(PushdownExpr::Constant(k.nulls_pass())),
        });
        let l = Arc::new(Lowered {
            generation: published.generation,
            keys,
            predicate,
        });
        *cache = Some(Arc::clone(&l));
        l
    }

    /// The current snapshot's knowledge, if final.
    #[must_use]
    pub fn current(&self) -> Option<Arc<KeyFilter>> {
        self.lowered().keys.clone()
    }

    /// The generation [`Self::current`] and [`Self::predicate`] are of.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.lowered().generation
    }

    /// The current snapshot as a predicate on this level's column (see the
    /// module docs); `None` while nothing applies.
    #[must_use]
    pub fn predicate(&self) -> Option<PushdownExpr> {
        self.lowered().predicate.clone()
    }

    /// Whether no row can pass: the producer has no key and NULLs do not
    /// pass (the whole input can be skipped).
    #[must_use]
    pub fn drops_everything(&self) -> bool {
        self.current()
            .is_some_and(|k| k.is_empty() && !k.nulls_pass())
    }

    /// Whether each row of `column` (this level's column) may pass; `None`
    /// when nothing applies (no final snapshot, or a type that does not
    /// widen to the key type).
    #[must_use]
    pub fn evaluate(&self, column: &dyn Array) -> Option<BooleanArray> {
        self.current()?.evaluate(column)
    }
}

/// The type of `path` in `schema` (through structs).
fn resolve(schema: &Schema, path: &ColumnPath) -> Option<DataType> {
    let (first, rest) = path.0.split_first()?;
    let mut t = schema.field_with_name(first).ok()?.data_type().clone();
    for name in rest {
        let DataType::Struct(children) = &t else {
            return None;
        };
        t = children
            .iter()
            .find(|f| f.name() == name)?
            .data_type()
            .clone();
    }
    Some(t)
}

#[cfg(test)]
mod tests;
