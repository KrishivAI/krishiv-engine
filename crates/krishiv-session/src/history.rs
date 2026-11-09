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
    InvalidHandleFormat, InvalidHandleOperationAbandoned, InvalidHandleOperationAlreadyExists,
    KrishivError,
};

use crate::system::{Finished, OperatorRow, QueryRow};

/// Finished rows one principal keeps; its oldest finished row goes first.
pub(crate) const PRINCIPAL_ROWS: usize = 1000;

/// The id of a query in `system.queries`, profiles and cancellation: a
/// UUIDv7 the runtime mints, or the operation id a caller gave
/// ([`QueryOptions::operation_id`]) in its normal form.
///
/// ```
/// let id = krishiv_session::QueryId::parse("00112233-4455-6677-8899-AABBCCDDEEFF")?;
/// assert_eq!(id.as_str(), "00112233-4455-6677-8899-aabbccddeeff");
/// let short = krishiv_session::QueryId::parse("1-2-3-4-5")?;
/// assert_eq!(short.as_str(), "00000001-0002-0003-0004-000000000005");
/// let e = krishiv_session::QueryId::parse("not-a-uuid").unwrap_err();
/// assert_eq!(e.error_class().name(), "INVALID_HANDLE.FORMAT");
/// # Ok::<(), krishiv_common::error::KrishivError>(())
/// ```
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QueryId(Arc<str>);

impl QueryId {
    /// `text` as a query id, read as Spark Connect reads an operation id
    /// (`UUID.fromString(id).toString`, Spark 4.2.0
    /// `SparkConnectExecutionManager.ExecuteKey`): at most 36 characters,
    /// five groups of hex digits split by `-`, each as Java's
    /// `Long.parseLong(group, 16)` takes it (an optional `+`, at least one
    /// digit, at most `i64::MAX`), keeping the low 32, 16, 16, 16 and 48 bits.
    /// The id is the canonical lower-case form of the result:
    /// `1-2-3-4-5` is `00000001-0002-0003-0004-000000000005`.
    ///
    /// One divergence: Java also takes non-ASCII decimal digits (such as
    /// full-width ones); only ASCII hex digits are taken here.
    ///
    /// # Errors
    /// `INVALID_HANDLE.FORMAT` for anything else.
    pub fn parse(text: &str) -> Result<Self, KrishivError> {
        java_uuid(text).map(Self).ok_or_else(|| {
            InvalidHandleFormat {
                handle: text.to_owned(),
            }
            .into()
        })
    }

    /// An id the runtime minted, or one already parsed (canonical).
    pub(crate) fn minted(text: String) -> Self {
        Self(Arc::from(text))
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

/// `UUID.fromString(text).toString()` (JDK 21, the JVM of Spark 4.2.0), or
/// `None` where it throws.
fn java_uuid(text: &str) -> Option<Arc<str>> {
    if !text.is_ascii() || text.len() > 36 {
        return None;
    }
    let groups: Vec<&str> = text.split('-').collect();
    if groups.len() != 5 {
        return None;
    }
    let mut v = [0u64; 5];
    for (slot, group) in v.iter_mut().zip(&groups) {
        let digits = group.strip_prefix('+').unwrap_or(group);
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        let n = u64::from_str_radix(digits, 16).ok()?;
        if n > i64::MAX.unsigned_abs() {
            return None;
        }
        *slot = n;
    }
    Some(Arc::from(format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        v[0] & 0xFFFF_FFFF,
        v[1] & 0xFFFF,
        v[2] & 0xFFFF,
        v[3] & 0xFFFF,
        v[4] & 0xFFFF_FFFF_FFFF
    )))
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
    pub(crate) deadline: Option<std::time::Duration>,
}

impl QueryOptions {
    /// No options: the runtime mints the query's id.
    pub fn new() -> Self {
        Self::default()
    }

    /// Names the statement's queries `id` (Spark Connect's operation id), a
    /// UUID. The operation is active while the statement's DataFrame or a
    /// result stream of it is alive: another statement of the same session
    /// with the same id then fails with
    /// `INVALID_HANDLE.OPERATION_ALREADY_EXISTS`. For 30 minutes after its
    /// last use the id stays taken (`INVALID_HANDLE.OPERATION_ABANDONED`), as
    /// Spark Connect keeps a session's finished operations. Sessions never
    /// collide.
    pub fn operation_id(mut self, id: impl Into<String>) -> Self {
        self.operation_id = Some(id.into());
        self
    }

    /// Limits each query the statement runs to `limit` from the start of
    /// its execution (a SQL script is one query: the limit counts from its start) (`KRISHIV_QUERY_TIMEOUT` past it). With
    /// `krishiv.query.maxExecutionTime` also set, the shorter applies: a
    /// statement can tighten the session's limit, never loosen it.
    pub fn deadline(mut self, limit: std::time::Duration) -> Self {
        self.deadline = Some(limit);
        self
    }
}

/// How long a finished operation id stays taken after its last use: Spark
/// 4.2.0's `spark.connect.session.inactiveOperations.cacheExpiration`
/// default (30 minutes, an internal static conf; `expireAfterAccess`).
pub(crate) const INACTIVE_OPERATION_MS: i64 = 30 * 60 * 1000;

/// The operation ids of one session: active ones, and finished ones with
/// the time of their last use (Spark Connect's `SessionHolder`
/// `activeOperationIds` and `inactiveOperations`).
#[derive(Debug, Default)]
pub(crate) struct Operations {
    ids: Mutex<HashMap<Arc<str>, Option<i64>>>,
}

impl Operations {
    /// Takes `id` for a new statement at `now` (milliseconds).
    ///
    /// # Errors
    /// `INVALID_HANDLE.OPERATION_ALREADY_EXISTS` while a statement holds it;
    /// `INVALID_HANDLE.OPERATION_ABANDONED` within
    /// [`INACTIVE_OPERATION_MS`] of its last use (which counts as a use).
    pub(crate) fn take(self: &Arc<Self>, id: QueryId, now: i64) -> Result<Operation, KrishivError> {
        let mut ids = self.ids.lock().unwrap_or_else(PoisonError::into_inner);
        ids.retain(|_, last| last.is_none_or(|t| now.saturating_sub(t) < INACTIVE_OPERATION_MS));
        match ids.get_mut(&id.0) {
            Some(None) => {
                return Err(InvalidHandleOperationAlreadyExists {
                    handle: id.as_str().to_owned(),
                }
                .into());
            }
            Some(Some(last)) => {
                *last = now;
                return Err(InvalidHandleOperationAbandoned {
                    handle: id.as_str().to_owned(),
                }
                .into());
            }
            None => {}
        }
        ids.insert(Arc::clone(&id.0), None);
        Ok(Operation {
            id,
            owner: Arc::clone(self),
        })
    }

    /// Marks `id` finished at `now`.
    fn release(&self, id: &QueryId, now: i64) {
        let mut ids = self.ids.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(slot) = ids.get_mut(&id.0) {
            *slot = Some(now);
        }
    }
}

/// A statement's hold on its operation id: dropped with the last handle of
/// the statement (its DataFrame, result streams and their queries), which
/// finishes the operation.
#[derive(Debug)]
pub(crate) struct Operation {
    id: QueryId,
    owner: Arc<Operations>,
}

impl Operation {
    /// The id the statement's queries carry.
    pub(crate) fn id(&self) -> &QueryId {
        &self.id
    }
}

impl Drop for Operation {
    fn drop(&mut self) {
        self.owner.release(&self.id, crate::system::now_ms());
    }
}

/// The `system.queries` rows of one runtime. The lock is held only to push,
/// update or copy.
#[derive(Debug)]
pub(crate) struct QueryHistory {
    rows: Mutex<Rows>,
    counter: AtomicU64,
    /// Most rows of all principals together
    /// (`krishiv.system.queries.capacity`).
    capacity: usize,
}

/// The rows, oldest first, and how many each principal holds (kept beside
/// them so making room never counts the whole history).
#[derive(Debug, Default)]
struct Rows {
    rows: VecDeque<QueryRow>,
    held: HashMap<String, Held>,
    /// [`QueryRow::heap_bytes`] of every row held.
    bytes: usize,
}

/// One principal's rows: all of them, and the finished ones among them.
#[derive(Debug, Default, Clone, Copy)]
struct Held {
    total: usize,
    finished: usize,
}

impl Default for QueryHistory {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

/// Most bytes the remembered rows may hold (design P2-T126 §3: 64 MiB);
/// like the row bounds, only finished rows are evicted.
pub(crate) const MAX_BYTES: usize = 64 << 20;

/// The default of `krishiv.system.queries.capacity`.
pub(crate) const DEFAULT_CAPACITY: usize = 10_000;

impl QueryHistory {
    /// A history keeping at most `capacity` rows in all.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            rows: Mutex::new(Rows::default()),
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

    /// Adds `row`, a query that starts now. Room is made first: the
    /// principal's oldest finished row once it holds [`PRINCIPAL_ROWS`]
    /// rows, then, at the runtime's capacity, the oldest finished row of the
    /// principal holding the most finished rows. Running rows are never
    /// evicted.
    pub(crate) fn push(&self, row: QueryRow) {
        let mut rows = self.rows.lock().unwrap_or_else(PoisonError::into_inner);
        rows.trim(&row.principal, 1, self.capacity);
        rows.add(row);
    }

    /// Adds `row` under a caller's operation id: refused
    /// (`INVALID_HANDLE.OPERATION_ALREADY_EXISTS`) while an execution of the
    /// same session under that id is live (Spark Connect runs one execution
    /// per operation id, so an action run twice at once is refused). The
    /// check and the insertion happen under one lock.
    pub(crate) fn push_given(&self, row: QueryRow) -> Result<(), KrishivError> {
        let mut rows = self.rows.lock().unwrap_or_else(PoisonError::into_inner);
        if rows
            .rows
            .iter()
            .any(|r| r.ended_ms.is_none() && r.id == row.id && r.session_id == row.session_id)
        {
            return Err(InvalidHandleOperationAlreadyExists { handle: row.id }.into());
        }
        rows.trim(&row.principal, 1, self.capacity);
        rows.add(row);
        Ok(())
    }

    /// Applies `f` to the row `id` of session `session`, if it is still
    /// remembered.
    pub(crate) fn update(&self, session: &str, id: &str, f: impl FnOnce(&mut QueryRow)) {
        let mut rows = self.rows.lock().unwrap_or_else(PoisonError::into_inner);
        let Rows { rows, held, bytes } = &mut *rows;
        // The newest first: a reused id names its latest query.
        if let Some(row) = rows
            .iter_mut()
            .rev()
            .find(|r| r.id == id && r.session_id == session)
        {
            let was_running = row.ended_ms.is_none();
            let before = row.heap_bytes();
            f(row);
            *bytes = (*bytes + row.heap_bytes()).saturating_sub(before);
            if was_running && row.ended_ms.is_some() {
                held.entry(row.principal.clone()).or_default().finished += 1;
                // Rows added while everything was running may have gone past
                // the bounds: a finished row can now go.
                let principal = row.principal.clone();
                Rows::trim_parts(rows, held, bytes, &principal, 0, self.capacity);
            }
        }
    }

    /// When the query `id` of session `session` was submitted (monotonic clock), if it is still
    /// remembered.
    pub(crate) fn submitted_at(&self, session: &str, id: &str) -> Option<std::time::Instant> {
        self.rows
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .rows
            .iter()
            .rev()
            .find(|r| r.id == id && r.session_id == session)
            .map(|r| r.submitted_at)
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
                row.rows_read = Some(t.rows_read);
                row.rows_written = Some(t.rows_written);
                row.bytes_written = Some(t.bytes_written);
                row.files_read = Some(t.files_read);
                row.files_written = Some(t.files_written);
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
            .rows
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
            .rows
            .iter()
            .filter(|r| principal.is_none_or(|p| r.principal == p))
            .map(|r| Arc::clone(&r.operators))
            .collect();
        shared.iter().flat_map(|o| o.iter().cloned()).collect()
    }
}

impl Rows {
    /// Appends `row`, counting it for its principal.
    fn add(&mut self, row: QueryRow) {
        self.bytes += row.heap_bytes();
        self.held.entry(row.principal.clone()).or_default().total += 1;
        self.rows.push_back(row);
    }

    /// Makes room for `incoming` more rows of `principal` (0 after a row
    /// finished): its oldest finished rows while it holds [`PRINCIPAL_ROWS`]
    /// or more, then, while the history is at `capacity`, the oldest finished
    /// row of the principal holding the most finished rows. Running rows are
    /// never evicted, so the bounds may be passed while everything runs.
    fn trim(&mut self, principal: &str, incoming: usize, capacity: usize) {
        let Self { rows, held, bytes } = self;
        Self::trim_parts(rows, held, bytes, principal, incoming, capacity);
    }

    fn trim_parts(
        rows: &mut VecDeque<QueryRow>,
        held: &mut HashMap<String, Held>,
        bytes: &mut usize,
        principal: &str,
        incoming: usize,
        capacity: usize,
    ) {
        while held
            .get(principal)
            .is_some_and(|h| h.finished > 0 && h.total + incoming > PRINCIPAL_ROWS)
        {
            if !Self::evict_oldest_finished(rows, held, bytes, principal) {
                break;
            }
        }
        while rows.len() + incoming > capacity || *bytes > MAX_BYTES {
            // Ties go to the principal named first, so the choice is stable.
            let heaviest = held
                .iter()
                .filter(|(_, h)| h.finished > 0)
                .max_by(|a, b| a.1.finished.cmp(&b.1.finished).then_with(|| b.0.cmp(a.0)))
                .map(|(p, _)| p.clone());
            let Some(p) = heaviest else { break };
            if !Self::evict_oldest_finished(rows, held, bytes, &p) {
                break;
            }
        }
    }

    /// Removes the oldest finished row of `principal`; whether there was one.
    fn evict_oldest_finished(
        rows: &mut VecDeque<QueryRow>,
        held: &mut HashMap<String, Held>,
        bytes: &mut usize,
        principal: &str,
    ) -> bool {
        let Some(at) = rows
            .iter()
            .position(|r| r.ended_ms.is_some() && r.principal == principal)
        else {
            return false;
        };
        if let Some(gone) = rows.remove(at) {
            *bytes = bytes.saturating_sub(gone.heap_bytes());
        }
        if let Some(h) = held.get_mut(principal) {
            h.total = h.total.saturating_sub(1);
            h.finished = h.finished.saturating_sub(1);
            if h.total == 0 {
                held.remove(principal);
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ids and outcomes `java.util.UUID.fromString` gave on the JDK 21
    /// of the Spark 4.2.0 image (probe run 2026-11-05).
    #[test]
    fn operation_ids_parse_as_java_does() {
        for (text, normal) in [
            (
                "00112233-4455-6677-8899-aabbccddeeff",
                "00112233-4455-6677-8899-aabbccddeeff",
            ),
            (
                "00112233-4455-6677-8899-AABBCCDDEEFF",
                "00112233-4455-6677-8899-aabbccddeeff",
            ),
            ("1-2-3-4-5", "00000001-0002-0003-0004-000000000005"),
            ("+1-2-3-4-5", "00000001-0002-0003-0004-000000000005"),
            ("1-2-3-4-+5", "00000001-0002-0003-0004-000000000005"),
            (
                "00112233-4455-6677-8899-aabbccddeef",
                "00112233-4455-6677-8899-0aabbccddeef",
            ),
            ("123456789-1-1-1-1", "23456789-0001-0001-0001-000000000001"),
            ("1-12345-1-1-1", "00000001-2345-0001-0001-000000000001"),
            (
                "1-1-1-1-1234567890123",
                "00000001-0001-0001-0001-234567890123",
            ),
            (
                "1234567890123456-1-1-1-1",
                "90123456-0001-0001-0001-000000000001",
            ),
            (
                "7fffffffffffffff-1-1-1-1",
                "ffffffff-0001-0001-0001-000000000001",
            ),
        ] {
            assert_eq!(QueryId::parse(text).unwrap().as_str(), normal, "{text}");
        }
        for bad in [
            "",
            "-1-2-3-4-5",
            "00112233-4455-6677-8899-aabbccddeeff0",
            "g-1-2-3-4",
            "1-2-3-4",
            "1-2-3-4-5-6",
            "12345678901234567-1-1-1-1",
            "8000000000000000-1-1-1-1",
            "1-1-1-1-ffffffffffffffff",
            "1-2-3-4-++5",
            "0x1-2-3-4-5",
            " 1-2-3-4-5",
            "00112233-4455-6677-8899-aabbccddeefg",
        ] {
            let e = QueryId::parse(bad).unwrap_err();
            assert_eq!(e.error_class().name(), "INVALID_HANDLE.FORMAT", "{bad:?}");
        }
    }

    #[test]
    fn a_finished_operation_id_stays_taken_for_thirty_minutes_after_its_last_use() {
        let ops = Arc::new(Operations::default());
        let id = || QueryId::parse("1-2-3-4-5").unwrap();
        let held = ops.take(id(), 0).unwrap();
        let e = ops.take(id(), 1).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "INVALID_HANDLE.OPERATION_ALREADY_EXISTS"
        );
        let key = Arc::clone(&held.id().0);
        drop(held);
        // Finished at time 10 (the drop used the wall clock).
        ops.ids.lock().unwrap().insert(key, Some(10));
        let e = ops.take(id(), 10 + INACTIVE_OPERATION_MS - 1).unwrap_err();
        assert_eq!(e.error_class().name(), "INVALID_HANDLE.OPERATION_ABANDONED");
        // That refusal was a use: the window starts again from it.
        let again = 10 + 2 * INACTIVE_OPERATION_MS - 2;
        assert!(ops.take(id(), again).is_err());
        let later = again + INACTIVE_OPERATION_MS;
        let taken = ops.take(id(), later).unwrap();
        assert_eq!(taken.id().as_str(), "00000001-0002-0003-0004-000000000005");
        // Another id is independent.
        assert!(
            ops.take(QueryId::parse("1-2-3-4-6").unwrap(), later)
                .is_ok()
        );
    }

    #[test]
    fn eviction_counts_stay_exact() {
        let history = QueryHistory::new(3);
        let row = |id: &str, who: &str| {
            let log = crate::system::QueryLog::default();
            let mut r = log.snapshot_row_for_tests(id);
            r.principal = who.to_owned();
            r.session_id = "s".to_owned();
            r
        };
        for (i, who) in ["a", "a", "b"].iter().enumerate() {
            history.push(row(&i.to_string(), who));
        }
        // All running: nothing can go, so the history grows past capacity.
        history.push(row("3", "a"));
        assert_eq!(history.snapshot(None).len(), 4);
        for id in ["0", "1", "2"] {
            history.update("s", id, |r| r.ended_ms = Some(1));
        }
        // `0` finished first and went at once (four rows, capacity three).
        let ids: Vec<String> = history.snapshot(None).into_iter().map(|r| r.id).collect();
        assert_eq!(ids, ["1", "2", "3"]);
        // At capacity: `a` and `b` hold one finished row each; the tie goes
        // to `a`, which loses its oldest.
        history.push(row("4", "b"));
        let ids: Vec<String> = history.snapshot(None).into_iter().map(|r| r.id).collect();
        assert_eq!(ids, ["2", "3", "4"]);
        let held = history.rows.lock().unwrap().held.clone();
        assert_eq!((held["a"].total, held["a"].finished), (1, 0));
        assert_eq!((held["b"].total, held["b"].finished), (2, 1));
    }

    #[test]
    fn finished_rows_go_oldest_first_past_the_byte_bound() {
        let history = QueryHistory::new(1_000);
        let big = |id: &str| {
            let log = crate::system::QueryLog::default();
            let mut r = log.snapshot_row_for_tests(id);
            r.session_id = "s".to_owned();
            r.text = Some("x".repeat(4 << 20));
            r
        };
        // 20 rows of 4 MiB: 80 MiB, over the 64 MiB bound.
        for i in 0..20 {
            let id = i.to_string();
            history.push(big(&id));
            history.update("s", &id, |r| r.ended_ms = Some(1));
        }
        let rows = history.snapshot(None);
        assert!(rows.iter().map(QueryRow::heap_bytes).sum::<usize>() <= MAX_BYTES);
        assert_eq!(rows.last().map(|r| r.id.as_str()), Some("19"));
        assert!(rows.len() < 20);
        // A running row is never evicted for bytes.
        history.push(big("run"));
        history.push(big("run2"));
        let ids: Vec<String> = history.snapshot(None).into_iter().map(|r| r.id).collect();
        assert!(ids.contains(&"run".to_owned()) && ids.contains(&"run2".to_owned()));
    }

    /// Rows added past the capacity while everything ran are trimmed as
    /// they finish.
    #[test]
    fn finishing_rows_trims_the_history_back_to_capacity() {
        let history = QueryHistory::new(2);
        let log = crate::system::QueryLog::default();
        for id in ["0", "1", "2", "3"] {
            let mut r = log.snapshot_row_for_tests(id);
            r.session_id = "s".to_owned();
            history.push(r);
        }
        assert_eq!(history.snapshot(None).len(), 4, "running rows stay");
        for id in ["0", "1", "2", "3"] {
            history.update("s", id, |r| r.ended_ms = Some(1));
        }
        let ids: Vec<String> = history.snapshot(None).into_iter().map(|r| r.id).collect();
        assert_eq!(ids, ["2", "3"]);
    }
}
