// SPDX-License-Identifier: Apache-2.0
//! The persisted query history (P3-T051): one compact profile per finished query, kept as Parquet files
//! under `krishiv.history.path` and bounded by `krishiv.history.maxBytes`.
//!
//! A query's end hands a [`HistoryEntry`] to [`ProfileHistory::record`], which only queues it. A writer
//! thread builds the profile JSON, groups entries into one file (a *segment*) and writes it after the
//! result stream has ended, so the query's critical path pays for a channel send. A failed write, a full
//! queue and a profile that does not serialize are counted ([`ProfileHistory::failures`]) and never fail
//! the query.
//!
//! A segment is `history-<first finish, ms, 13 digits>-<writer id>-<seq, 6 digits>.parquet` (the writer id keeps
//! two engines on one directory from naming the same file), written to a temporary
//! name and renamed, so a reader never sees half a file. Retention is by whole segments, oldest first,
//! until the directory is within the byte bound; the newest segment is always kept.
//!
//! Schema (v1; a reader takes the columns it knows by name, so a later optional column loads under it):
//! `query_id`, `finished_at_ms`, `state`, `error_class`, `principal`, `sql`, `duration_ns`,
//! `profile_version`, `profile_minor`, `profile` (the versioned profile JSON of `krishiv-exec`).
//! `sql` is the statement text as the session redacted it for credentials (ADR 0038), and with
//! `krishiv.history.literals = mask` (the default) every literal in it and in the profile's plan lines is `?`.
//! The profile has no result data. A file carries `krishiv.history.schema_version` in its Parquet metadata;
//! a reader skips a file of a later major version, so a store written by a newer build never misreads.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError, sync_channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::QueryMetrics;
use krishiv_exec::planner::PhysicalPlan;
use krishiv_exec::profile::{PROFILE_MINOR, PROFILE_VERSION, ProfileIdentity, QueryProfile};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

/// Entries the queue holds before [`ProfileHistory::record`] drops one (and counts it).
const QUEUE: usize = 1024;
/// The most rows one segment holds.
const SEGMENT_ROWS: usize = 256;
/// How long the writer waits for more entries before it writes what it has.
const LINGER: Duration = Duration::from_millis(500);
/// The file name prefix and extension of a segment.
const PREFIX: &str = "history-";
const SUFFIX: &str = ".parquet";
/// The segment schema's major version and the Parquet metadata key that carries it.
const SCHEMA_VERSION: u32 = 1;
const VERSION_KEY: &str = "krishiv.history.schema_version";
/// The most bytes of statement text one row keeps.
const SQL_MAX: usize = 4096;

/// What a query leaves behind when it ends.
pub(crate) struct HistoryEntry {
    pub(crate) query_id: String,
    pub(crate) finished_at_ms: i64,
    pub(crate) state: &'static str,
    pub(crate) error_class: Option<String>,
    pub(crate) sql: Option<String>,
    pub(crate) detailed: bool,
    pub(crate) duration_ns: i64,
    pub(crate) plan: PhysicalPlan,
    pub(crate) metrics: QueryMetrics,
    pub(crate) identity: ProfileIdentity,
    pub(crate) trace: Option<serde_json::Value>,
}

/// Where a file write's entry waits for its commit: the query ends before the commit does, and a commit that
/// fails is the write's outcome, so the entry is recorded once the write has settled.
#[derive(Default)]
pub(crate) struct HistoryHold(Mutex<HoldState>);

#[derive(Default)]
enum HoldState {
    /// The query has not ended.
    #[default]
    Waiting,
    /// The query ended; its entry waits for the write's outcome.
    Held(Box<HistoryEntry>),
    /// The write is settled: a late entry is recorded as it comes.
    Settled,
}

impl HistoryHold {
    /// Holds `entry` until [`Self::settle`], or records it now when the write already settled.
    pub(crate) fn offer(&self, entry: HistoryEntry, store: &ProfileHistory) {
        let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if matches!(*state, HoldState::Settled) {
            drop(state);
            store.record(entry);
        } else {
            *state = HoldState::Held(Box::new(entry));
        }
    }

    /// Records the held entry as `cancelled`, unless the write already settled: the write's future was dropped.
    fn settle_cancelled(&self, store: &ProfileHistory) {
        let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let held = std::mem::replace(&mut *state, HoldState::Settled);
        drop(state);
        if let HoldState::Held(mut entry) = held {
            entry.state = "cancelled";
            store.record(*entry);
        }
    }

    /// Records the held entry with the write's final outcome (`error` set when it failed).
    pub(crate) fn settle(&self, error: Option<&KrishivError>, store: &ProfileHistory) {
        let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let held = std::mem::replace(&mut *state, HoldState::Settled);
        drop(state);
        if let HoldState::Held(mut entry) = held {
            if let Some(error) = error {
                let (status, class) = krishiv_exec::Status::from_error(Some(error));
                entry.state = match status {
                    krishiv_exec::Status::Succeeded => "finished",
                    krishiv_exec::Status::Failed => "failed",
                    krishiv_exec::Status::Cancelled => "cancelled",
                    krishiv_exec::Status::Denied => "denied",
                };
                entry.error_class = class;
            }
            store.record(*entry);
        }
    }
}

/// Settles a [`HistoryHold`] as `cancelled` when dropped unsettled: a write whose future is dropped
/// never reaches its own settle call, and its held entry would vanish.
pub(crate) struct HoldGuard {
    hold: Arc<HistoryHold>,
    store: Option<Arc<ProfileHistory>>,
}

impl HoldGuard {
    pub(crate) fn new(hold: Arc<HistoryHold>, store: Option<Arc<ProfileHistory>>) -> Self {
        Self { hold, store }
    }

    /// Records the held entry with the write's final outcome and disarms the guard's drop.
    pub(crate) fn settle(&self, error: Option<&KrishivError>) {
        if let Some(store) = &self.store {
            self.hold.settle(error, store);
        }
    }
}

impl Drop for HoldGuard {
    fn drop(&mut self) {
        if let Some(store) = &self.store {
            self.hold.settle_cancelled(store);
        }
    }
}

/// A profile read back from the store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StoredProfile {
    /// The query's id in `system.queries`.
    pub(crate) query_id: String,
    /// When the query ended, in milliseconds since the Unix epoch.
    pub(crate) finished_at_ms: i64,
    /// `finished`, `failed`, `cancelled` or `denied`.
    pub(crate) state: String,
    /// The Spark error class of a failed query.
    pub(crate) error_class: Option<String>,
    /// The principal that ran it.
    pub(crate) principal: String,
    /// The statement, redacted as `system.queries` shows it.
    pub(crate) sql: Option<String>,
    /// Wall time, in nanoseconds.
    pub(crate) duration_ns: i64,
    /// The profile document's major version.
    pub(crate) profile_version: i64,
    /// The profile document's minor version.
    pub(crate) profile_minor: i64,
    /// The profile document (`krishiv.query-profile` JSON), as [`QueryProfile::from_json`] reads it.
    pub(crate) profile: String,
}

/// The writer side: a bounded queue to the thread that writes segments.
pub(crate) struct ProfileHistory {
    tx: Mutex<Option<SyncSender<Msg>>>,
    failures: Arc<AtomicU64>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

enum Msg {
    Entry(Box<HistoryEntry>),
    /// Write what is queued, then answer.
    #[cfg_attr(not(test), expect(dead_code, reason = "tests flush the writer"))]
    Flush(SyncSender<()>),
}

impl std::fmt::Debug for ProfileHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProfileHistory")
            .field("failures", &self.failures())
            .finish_non_exhaustive()
    }
}

impl ProfileHistory {
    /// Opens the store at `dir` (created if missing) and starts its writer.
    ///
    /// # Errors
    /// The directory cannot be created.
    pub(crate) fn open(dir: &Path, max_bytes: u64, mask: bool) -> Result<Self, KrishivError> {
        std::fs::create_dir_all(dir)
            .map_err(|e| internal(format!("krishiv.history.path {}: {e}", dir.display())))?;
        let (tx, rx) = sync_channel(QUEUE);
        let failures = Arc::new(AtomicU64::new(0));
        let writer = Writer {
            dir: dir.to_owned(),
            max_bytes,
            failures: Arc::clone(&failures),
            seq: 0,
            id: writer_id(),
            mask,
        };
        let thread = std::thread::Builder::new()
            .name("krishiv-history".to_owned())
            .spawn(move || writer.run(&rx))
            .map_err(|e| internal(format!("cannot start the history writer: {e}")))?;
        Ok(Self {
            tx: Mutex::new(Some(tx)),
            failures,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Queues `entry`; a full queue drops it and counts a failure.
    pub(crate) fn record(&self, entry: HistoryEntry) {
        let tx = self.tx.lock().unwrap_or_else(PoisonError::into_inner);
        let sent = tx
            .as_ref()
            .is_some_and(|tx| tx.try_send(Msg::Entry(Box::new(entry))).is_ok());
        if !sent {
            self.failures.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Entries that did not reach a file: queue full, serialization or write failures.
    pub(crate) fn failures(&self) -> u64 {
        self.failures.load(Ordering::Relaxed)
    }

    /// Waits until everything queued so far is on disk.
    #[cfg(test)]
    pub(crate) fn flush(&self) {
        let (done_tx, done_rx) = sync_channel(1);
        let sent = {
            let tx = self.tx.lock().unwrap_or_else(PoisonError::into_inner);
            tx.as_ref()
                .is_some_and(|tx| tx.send(Msg::Flush(done_tx)).is_ok())
        };
        if sent {
            let _ = done_rx.recv();
        }
    }
}

impl Drop for ProfileHistory {
    fn drop(&mut self) {
        // Closing the channel makes the writer write its last segment and end.
        drop(
            self.tx
                .get_mut()
                .unwrap_or_else(PoisonError::into_inner)
                .take(),
        );
        let thread = self
            .thread
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(thread) = thread {
            let _ = thread.join();
        }
    }
}

struct Writer {
    dir: PathBuf,
    max_bytes: u64,
    failures: Arc<AtomicU64>,
    seq: u64,
    /// Names this writer's files, unique among the engines on one directory.
    id: String,
    mask: bool,
}

impl Writer {
    fn run(mut self, rx: &Receiver<Msg>) {
        let mut rows: Vec<StoredProfile> = Vec::new();
        // When the oldest buffered row entered: the flush is due LINGER after it, however often later rows arrive.
        let mut since: Option<std::time::Instant> = None;
        loop {
            if rows.is_empty() {
                since = None;
            }
            let msg = match since {
                None => rx.recv().map_err(|_| RecvTimeoutError::Disconnected),
                Some(t) => rx.recv_timeout(LINGER.saturating_sub(t.elapsed())),
            };
            if since.is_none() && matches!(msg, Ok(Msg::Entry(_))) {
                since = Some(std::time::Instant::now());
            }
            match msg {
                Ok(Msg::Entry(entry)) => {
                    match stored(*entry, self.mask) {
                        Some(row) => rows.push(row),
                        None => {
                            self.failures.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    // Take what is already queued, up to a segment.
                    while rows.len() < SEGMENT_ROWS {
                        match rx.try_recv() {
                            Ok(Msg::Entry(e)) => match stored(*e, self.mask) {
                                Some(row) => rows.push(row),
                                None => {
                                    self.failures.fetch_add(1, Ordering::Relaxed);
                                }
                            },
                            Ok(Msg::Flush(done)) => {
                                self.write(&mut rows);
                                let _ = done.send(());
                            }
                            Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                        }
                    }
                    if rows.len() >= SEGMENT_ROWS {
                        self.write(&mut rows);
                    }
                }
                Ok(Msg::Flush(done)) => {
                    self.write(&mut rows);
                    let _ = done.send(());
                }
                Err(RecvTimeoutError::Timeout) => self.write(&mut rows),
                Err(RecvTimeoutError::Disconnected) => {
                    self.write(&mut rows);
                    return;
                }
            }
        }
    }

    /// Writes `rows` as one segment and applies the byte bound; a failure counts one per row.
    fn write(&mut self, rows: &mut Vec<StoredProfile>) {
        if rows.is_empty() {
            return;
        }
        let taken = std::mem::take(rows);
        self.seq += 1;
        let first = taken.iter().map(|r| r.finished_at_ms).min().unwrap_or(0);
        let name = format!("{PREFIX}{first:013}-{}-{:06}{SUFFIX}", self.id, self.seq);
        if write_segment(&self.dir, &name, &taken).is_err() {
            self.failures
                .fetch_add(taken.len() as u64, Ordering::Relaxed);
            return;
        }
        // A failed trim leaves the store over its bound until the next write tries again.
        let _ = trim(&self.dir, self.max_bytes);
    }
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

fn stored(e: HistoryEntry, mask: bool) -> Option<StoredProfile> {
    let mut profile = QueryProfile::new(&e.plan, e.metrics, e.detailed);
    profile.trace = e.trace;
    profile.identity = Some(e.identity.clone());
    let mut sql = e.sql;
    if mask {
        // A generated output name (`SELECT 'x'` is named `x`) prints the literal's text into the plan.
        // A DataFrame query has no text to read the literals from, so every result column name is masked: it can
        // be a literal's (`lit("x")` is named `x`), and over-masking a real column name is the safe error.
        let names = match sql.as_deref() {
            Some(text) => krishiv_sql::string_literal_bodies(text),
            None => e
                .plan
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect(),
        };
        let mut names = names;
        names.sort_by_key(|n| std::cmp::Reverse(n.len()));
        // Without the statement's text no list of names is complete (a file write's result schema is its commit
        // columns, not its input's), so every alias in the plan is masked too.
        mask_plan(&mut profile.plan, &names, sql.is_none());
        sql = sql.map(|t| krishiv_sql::mask_literals(&t));
    }
    let sql = sql.map(|t| cut(t, SQL_MAX));
    Some(StoredProfile {
        query_id: e.query_id,
        finished_at_ms: e.finished_at_ms,
        state: e.state.to_owned(),
        error_class: e.error_class,
        principal: e.identity.principal,
        sql,
        duration_ns: e.duration_ns,
        profile_version: i64::try_from(PROFILE_VERSION).ok()?,
        profile_minor: i64::try_from(PROFILE_MINOR).ok()?,
        profile: profile.to_json(),
    })
}

/// Replaces the literals in every node's argument text with `?`.
/// Then every whole-word occurrence of a string literal's text (`names`), which a generated output name carries.
fn mask_plan(node: &mut krishiv_exec::profile::ProfileNode, names: &[String], aliases: bool) {
    // Names first, on the text as printed: `mask_literals` re-spaces tokens (`hunter-secret` would become
    // `hunter - secret`) and no longer finds the literal's spelling. Longest first, so a name holding another is
    // masked whole.
    let mut detail = node.detail.clone();
    for name in names.iter().filter(|n| !n.is_empty()) {
        detail = mask_word(&detail, name);
    }
    if aliases {
        detail = mask_write_names(&mask_aliases(&detail));
    }
    node.detail = krishiv_sql::mask_literals(&detail);
    for child in &mut node.children {
        mask_plan(child, names, aliases);
    }
}

/// `text` with the name of every `AS <name>#<id>` alias replaced by `?`: a generated name can carry a literal.
fn mask_aliases(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(" AS ") {
        let name_start = at + " AS ".len();
        out.push_str(&rest[..name_start]);
        rest = &rest[name_start..];
        // The name ends at the first `#` that starts a numeric id.
        let end = rest.match_indices('#').find_map(|(i, _)| {
            rest[i + 1..]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit())
                .then_some(i)
        });
        if let Some(end) = end {
            out.push('?');
            rest = &rest[end..];
        }
    }
    out.push_str(rest);
    out
}

/// `text` with the names a file write prints without an alias marker replaced by `?`: its `partitioned by [..]` list
/// and the names that lead a sort key (`name ASC NULLS FIRST`). A DataFrame's partition column can be a literal's name.
fn mask_write_names(text: &str) -> String {
    const PARTITIONED: &str = "partitioned by [";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(PARTITIONED) {
        let start = at + PARTITIONED.len();
        out.push_str(&rest[..start]);
        rest = &rest[start..];
        match rest.find(']') {
            Some(end) => {
                out.push('?');
                rest = &rest[end..];
            }
            None => {
                out.push('?');
                rest = "";
            }
        }
    }
    out.push_str(rest);
    // Sort keys: the item before each ` ASC` / ` DESC` is a name or an expression; either way it is hidden.
    let mut masked = String::with_capacity(out.len());
    for (i, item) in out.split(", ").enumerate() {
        if i > 0 {
            masked.push_str(", ");
        }
        let key_end = [" ASC", " DESC"].iter().filter_map(|k| item.find(k)).min();
        match key_end {
            Some(end) => {
                let lead = item[..end].rfind('[').map_or(0, |b| b + 1);
                masked.push_str(&item[..lead]);
                masked.push('?');
                masked.push_str(&item[end..]);
            }
            None => masked.push_str(item),
        }
    }
    masked
}

/// `text` with each occurrence of `word` that is not part of a longer word replaced by `?`.
fn mask_word(text: &str, word: &str) -> String {
    if word.is_empty() {
        return text.to_owned();
    }
    // `_` is a boundary: a generated name appends to a literal with it (`hunter2_s`, a pivot's output).
    let is_word = char::is_alphanumeric;
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(word) {
        let before = rest[..at].chars().next_back();
        let after = rest[at + word.len()..].chars().next();
        out.push_str(&rest[..at]);
        if before.is_some_and(is_word) || after.is_some_and(is_word) {
            out.push_str(word);
        } else {
            out.push('?');
        }
        rest = &rest[at + word.len()..];
    }
    out.push_str(rest);
    out
}

/// `text` cut to at most `max` bytes at a character boundary.
fn cut(mut text: String, max: usize) -> String {
    if text.len() > max {
        let mut end = max;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
    }
    text
}

/// A name for one writer: the process, the moment it opened and a process-wide count, so no two writers on a
/// directory share one, even two opened in the same clock tick.
fn writer_id() -> String {
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!(
        "{:x}{nanos:x}{:x}",
        std::process::id(),
        COUNT.fetch_add(1, Ordering::Relaxed)
    )
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("query_id", DataType::Utf8, false),
        Field::new("finished_at_ms", DataType::Int64, false),
        Field::new("state", DataType::Utf8, false),
        Field::new("error_class", DataType::Utf8, true),
        Field::new("principal", DataType::Utf8, false),
        Field::new("sql", DataType::Utf8, true),
        Field::new("duration_ns", DataType::Int64, false),
        Field::new("profile_version", DataType::Int64, false),
        Field::new("profile_minor", DataType::Int64, false),
        Field::new("profile", DataType::Utf8, false),
    ]))
}

fn write_segment(dir: &Path, name: &str, rows: &[StoredProfile]) -> Result<(), String> {
    let strings = |f: &dyn Fn(&StoredProfile) -> Option<&str>| -> ArrayRef {
        Arc::new(rows.iter().map(f).collect::<StringArray>())
    };
    let ints = |f: &dyn Fn(&StoredProfile) -> i64| -> ArrayRef {
        Arc::new(rows.iter().map(f).collect::<Int64Array>())
    };
    let batch = RecordBatch::try_new(
        schema(),
        vec![
            strings(&|r| Some(&r.query_id)),
            ints(&|r| r.finished_at_ms),
            strings(&|r| Some(&r.state)),
            strings(&|r| r.error_class.as_deref()),
            strings(&|r| Some(&r.principal)),
            strings(&|r| r.sql.as_deref()),
            ints(&|r| r.duration_ns),
            ints(&|r| r.profile_version),
            ints(&|r| r.profile_minor),
            strings(&|r| Some(&r.profile)),
        ],
    )
    .map_err(|e| e.to_string())?;
    let tmp = dir.join(format!(".{name}.tmp"));
    let file = File::create(&tmp).map_err(|e| e.to_string())?;
    // Any failure below removes the temporary file: it is outside `list_segments`, so nothing else would.
    let written = (|| {
        let props = WriterProperties::builder()
            .set_key_value_metadata(Some(vec![KeyValue::new(
                VERSION_KEY.to_owned(),
                SCHEMA_VERSION.to_string(),
            )]))
            .build();
        let mut writer =
            ArrowWriter::try_new(file, schema(), Some(props)).map_err(|e| e.to_string())?;
        writer.write(&batch).map_err(|e| e.to_string())?;
        writer.close().map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, dir.join(name)).map_err(|e| e.to_string())
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}

/// The segments of `dir`, oldest first, with their sizes.
fn list_segments(dir: &Path) -> std::io::Result<Vec<(PathBuf, u64)>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(PREFIX) && name.ends_with(SUFFIX) {
            out.push((entry.path(), entry.metadata()?.len()));
        }
    }
    // The name leads with the first finish time and ends with a sequence number.
    out.sort();
    Ok(out)
}

/// Removes the oldest segments while the directory is over `max_bytes`; the newest stays.
fn trim(dir: &Path, max_bytes: u64) -> std::io::Result<()> {
    let segments = list_segments(dir)?;
    let mut total: u64 = segments.iter().map(|(_, n)| n).sum();
    for (path, size) in segments.iter().take(segments.len().saturating_sub(1)) {
        if total <= max_bytes {
            break;
        }
        match std::fs::remove_file(path) {
            Ok(()) => {}
            // Another engine on the directory trimmed it first.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        total -= size;
    }
    Ok(())
}

/// Reads every stored profile under `dir`, oldest first. A file that does not read is skipped, as a
/// crash can leave one. Works with no engine running: this is what an offline viewer calls.
///
/// # Errors
/// The directory cannot be read.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "read by the offline profile viewer (T050)")
)]
pub(crate) fn read_history(dir: &Path) -> Result<Vec<StoredProfile>, KrishivError> {
    let segments =
        list_segments(dir).map_err(|e| internal(format!("history {}: {e}", dir.display())))?;
    let mut out = Vec::new();
    for (path, _) in segments {
        if let Ok(rows) = read_segment(&path) {
            out.extend(rows);
        }
    }
    out.sort_by_key(|r| r.finished_at_ms);
    Ok(out)
}

fn read_segment(path: &Path) -> Result<Vec<StoredProfile>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| e.to_string())?;
    // A file without the key is the first v1; one of a later major version is not this build's to read.
    let version = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .and_then(|kv| kv.iter().find(|k| k.key == VERSION_KEY))
        .and_then(|k| k.value.as_deref()?.parse::<u32>().ok())
        .unwrap_or(1);
    if version > SCHEMA_VERSION {
        return Err(format!(
            "history schema version {version} is newer than {SCHEMA_VERSION}"
        ));
    }
    let reader = builder.build().map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|e| e.to_string())?;
        let text = |name: &str| -> Option<&StringArray> {
            batch.column_by_name(name)?.as_any().downcast_ref()
        };
        let int = |name: &str| -> Option<&Int64Array> {
            batch.column_by_name(name)?.as_any().downcast_ref()
        };
        let (Some(id), Some(at), Some(state), Some(principal), Some(profile)) = (
            text("query_id"),
            int("finished_at_ms"),
            text("state"),
            text("principal"),
            text("profile"),
        ) else {
            return Err("a required column is missing".to_owned());
        };
        let (error_class, sql) = (text("error_class"), text("sql"));
        let (duration, version, minor) = (
            int("duration_ns"),
            int("profile_version"),
            int("profile_minor"),
        );
        let opt = |a: Option<&StringArray>, i: usize| {
            a.filter(|a| a.is_valid(i)).map(|a| a.value(i).to_owned())
        };
        for i in 0..batch.num_rows() {
            out.push(StoredProfile {
                query_id: id.value(i).to_owned(),
                finished_at_ms: at.value(i),
                state: state.value(i).to_owned(),
                error_class: opt(error_class, i),
                principal: principal.value(i).to_owned(),
                sql: opt(sql, i),
                duration_ns: duration.map_or(0, |a| a.value(i)),
                // A file from before the minor column: the first v1.
                profile_version: version.map_or(1, |a| a.value(i)),
                profile_minor: minor.map_or(0, |a| a.value(i)),
                profile: profile.value(i).to_owned(),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
