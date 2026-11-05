// SPDX-License-Identifier: Apache-2.0
//! The rows of `system.queries` for a whole engine runtime (design P2-E1
//! §5): every session of the runtime records into one [`QueryHistory`],
//! bounded per principal so one busy principal never flushes another's
//! history, and each session reads it through its own
//! [`QueryLog`](crate::system::QueryLog), which filters by principal.
//!
//! Also the [`QueryId`] callers name a query by and the per-statement
//! [`QueryOptions`].

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use krishiv_common::error::{
    InvalidHandleFormat, InvalidHandleOperationAlreadyExists, KrishivError,
};

use crate::system::{Finished, OperatorRow, QueryRow};

/// Finished rows one principal keeps; its oldest finished row goes first.
pub(crate) const PRINCIPAL_ROWS: usize = 1000;

/// The id of a query in `system.queries`, profiles and cancellation: a
/// UUIDv7 the runtime mints, or the operation id a caller gave
/// ([`QueryOptions::operation_id`]), as given.
///
/// ```
/// let id = krishiv_session::QueryId::parse("00112233-4455-6677-8899-aabbccddeeff")?;
/// assert_eq!(id.as_str(), "00112233-4455-6677-8899-aabbccddeeff");
/// let e = krishiv_session::QueryId::parse("not-a-uuid").unwrap_err();
/// assert_eq!(e.error_class().name(), "INVALID_HANDLE.FORMAT");
/// # Ok::<(), krishiv_common::error::KrishivError>(())
/// ```
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QueryId(Arc<str>);

impl QueryId {
    /// `text` as a query id: a UUID in its canonical form
    /// (`00112233-4455-6677-8899-aabbccddeeff`, either case), as Spark
    /// Connect checks an operation id.
    ///
    /// # Errors
    /// `INVALID_HANDLE.FORMAT` for anything else.
    pub fn parse(text: &str) -> Result<Self, KrishivError> {
        if is_uuid(text) {
            Ok(Self(Arc::from(text)))
        } else {
            Err(InvalidHandleFormat {
                handle: text.to_owned(),
            }
            .into())
        }
    }

    /// The id as text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for QueryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for QueryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QueryId({})", self.0)
    }
}

/// Whether `text` is a UUID as Java's `UUID.fromString` takes it from
/// Spark Connect clients: five groups of 8, 4, 4, 4 and 12 hex digits.
fn is_uuid(text: &str) -> bool {
    let groups: Vec<&str> = text.split('-').collect();
    groups.len() == 5
        && groups
            .iter()
            .zip([8, 4, 4, 4, 12])
            .all(|(g, n)| g.len() == n && g.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// Options of one statement ([`Session::sql_with`](crate::Session::sql_with)).
///
/// ```
/// let options = krishiv_session::QueryOptions::new()
///     .operation_id("00112233-4455-6677-8899-aabbccddeeff");
/// # let _ = options;
/// ```
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
#[must_use]
pub struct QueryOptions {
    pub(crate) operation_id: Option<String>,
}

impl QueryOptions {
    /// No options: the runtime mints the query's id.
    pub fn new() -> Self {
        Self::default()
    }

    /// Names the statement's queries `id` (Spark Connect's operation id), a
    /// UUID. While one of them runs, another statement of the same session
    /// with the same id fails (`INVALID_HANDLE.OPERATION_ALREADY_EXISTS`);
    /// once it ended, the id may be used again. Sessions never collide.
    pub fn operation_id(mut self, id: impl Into<String>) -> Self {
        self.operation_id = Some(id.into());
        self
    }
}

/// The `system.queries` rows of one runtime. The lock is held only to push,
/// update or copy.
#[derive(Debug)]
pub(crate) struct QueryHistory {
    rows: Mutex<VecDeque<QueryRow>>,
    counter: AtomicU64,
    /// Most rows of all principals together
    /// (`krishiv.system.queries.capacity`).
    capacity: usize,
}

impl Default for QueryHistory {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

/// The default of `krishiv.system.queries.capacity`.
pub(crate) const DEFAULT_CAPACITY: usize = 10_000;

impl QueryHistory {
    /// A history keeping at most `capacity` rows in all.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            rows: Mutex::new(VecDeque::new()),
            counter: AtomicU64::new(0),
            capacity: capacity.max(1),
        }
    }

    /// A UUIDv7 (RFC 9562): 48 bits of milliseconds, then version, then bits
    /// that are unique per call. The tail comes from a per-process random
    /// hasher key and a counter, not from a CSPRNG: it names queries and
    /// sessions, it guards nothing.
    pub(crate) fn mint(&self, ms: i64) -> String {
        use std::hash::{BuildHasher, Hasher, RandomState};
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let mut h = RandomState::new().build_hasher();
        h.write_u64(n);
        let rand = h.finish();
        let ms = u64::try_from(ms).unwrap_or(0) & 0xFFFF_FFFF_FFFF;
        // 12 bits after the version, then the variant and 62 more bits.
        let rand_a = (rand >> 52) & 0xFFF;
        let rand_b = (rand ^ n.rotate_left(32)) & 0x3FFF_FFFF_FFFF_FFFF;
        format!(
            "{:08x}-{:04x}-7{:03x}-{:04x}-{:012x}",
            ms >> 16,
            ms & 0xFFFF,
            rand_a,
            0x8000 | ((rand_b >> 48) & 0x3FFF),
            rand_b & 0xFFFF_FFFF_FFFF
        )
    }

    /// Adds `row`, a query that starts now, under an id the runtime minted.
    pub(crate) fn push(&self, row: QueryRow) {
        let mut rows = self.rows.lock().unwrap_or_else(PoisonError::into_inner);
        self.make_room(&mut rows, &row.principal);
        rows.push_back(row);
    }

    /// Adds `row` under an id a caller gave: refused
    /// (`INVALID_HANDLE.OPERATION_ALREADY_EXISTS`) while a query of the same
    /// session with that id runs. The check and the insertion happen under
    /// one lock, so two statements racing for one id cannot both run.
    pub(crate) fn push_given(&self, row: QueryRow) -> Result<(), KrishivError> {
        let mut rows = self.rows.lock().unwrap_or_else(PoisonError::into_inner);
        if rows
            .iter()
            .any(|r| r.ended_ms.is_none() && r.id == row.id && r.session_id == row.session_id)
        {
            return Err(InvalidHandleOperationAlreadyExists { handle: row.id }.into());
        }
        self.make_room(&mut rows, &row.principal);
        rows.push_back(row);
        Ok(())
    }

    /// Makes room for a row of `principal`: its oldest finished row once it
    /// keeps [`PRINCIPAL_ROWS`], then, at the runtime's capacity, the oldest
    /// finished row of the principal holding the most finished rows. Running
    /// rows are never evicted.
    fn make_room(&self, rows: &mut VecDeque<QueryRow>, principal: &str) {
        let own = rows.iter().filter(|r| r.principal == principal).count();
        if own >= PRINCIPAL_ROWS {
            evict_oldest_finished(rows, principal);
        }
        if rows.len() >= self.capacity {
            let mut held: HashMap<&str, usize> = HashMap::new();
            for r in rows.iter().filter(|r| r.ended_ms.is_some()) {
                *held.entry(r.principal.as_str()).or_default() += 1;
            }
            // Ties go to the principal named first, so the choice is stable.
            let heaviest = held
                .into_iter()
                .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(a.0)))
                .map(|(p, _)| p.to_owned());
            if let Some(p) = heaviest {
                evict_oldest_finished(rows, &p);
            }
        }
    }

    /// Applies `f` to the row `id` of session `session`, if it is still
    /// remembered.
    pub(crate) fn update(&self, session: &str, id: &str, f: impl FnOnce(&mut QueryRow)) {
        let mut rows = self.rows.lock().unwrap_or_else(PoisonError::into_inner);
        // The newest first: a reused id names its latest query.
        if let Some(row) = rows
            .iter_mut()
            .rev()
            .find(|r| r.id == id && r.session_id == session)
        {
            f(row);
        }
    }

    /// Marks the query `id` of session `session` finished.
    pub(crate) fn finish(&self, session: &str, id: &str, done: &Finished<'_>) {
        self.update(session, id, |row| {
            row.state = done.state;
            row.ended_ms = Some(crate::system::now_ms());
            row.wall_ns = Some(done.wall_ns);
            row.peak_memory_bytes = Some(done.peak_memory_bytes);
            row.error_class = done.error_class.map(str::to_owned);
            row.operators = Arc::from(done.operators.as_slice());
            if let Some(t) = done.totals {
                row.workload_class = Some(t.workload_class);
                row.cpu_ns = Some(t.cpu_ns);
                row.rows_produced = t.rows_produced;
                row.bytes_read = Some(t.bytes_read);
                row.bytes_written = Some(t.bytes_written);
                row.spill_bytes = Some(t.spill_bytes);
                row.spill_files = Some(t.spill_files);
                row.pipelines = Some(t.pipelines);
                row.workers = Some(t.workers);
            }
        });
    }

    /// The rows `principal` may see, oldest first: its own, or every row
    /// when `principal` is `None`.
    pub(crate) fn snapshot(&self, principal: Option<&str>) -> Vec<QueryRow> {
        self.rows
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|r| principal.is_none_or(|p| r.principal == p))
            .cloned()
            .collect()
    }

    /// The operators of the remembered queries `principal` may see, oldest
    /// query first (copied after the lock is released).
    pub(crate) fn operators(&self, principal: Option<&str>) -> Vec<OperatorRow> {
        let shared: Vec<Arc<[OperatorRow]>> = self
            .rows
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|r| principal.is_none_or(|p| r.principal == p))
            .map(|r| Arc::clone(&r.operators))
            .collect();
        shared.iter().flat_map(|o| o.iter().cloned()).collect()
    }
}

/// Removes the oldest finished row of `principal`, if it has one.
fn evict_oldest_finished(rows: &mut VecDeque<QueryRow>, principal: &str) {
    if let Some(at) = rows
        .iter()
        .position(|r| r.ended_ms.is_some() && r.principal == principal)
    {
        rows.remove(at);
    }
}
