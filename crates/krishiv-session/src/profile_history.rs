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
//! A segment is `history-<first finish, ms, 13 digits>-<seq, 6 digits>.parquet`, written to a temporary
//! name and renamed, so a reader never sees half a file. Retention is by whole segments, oldest first,
//! until the directory is within the byte bound; the newest segment is always kept.
//!
//! Schema (v1; a reader takes the columns it knows by name, so a later optional column loads under it):
//! `query_id`, `finished_at_ms`, `state`, `error_class`, `principal`, `sql`, `duration_ns`,
//! `profile_version`, `profile_minor`, `profile` (the versioned profile JSON of `krishiv-exec`).
//! `sql` is the redacted statement text `system.queries` shows (ADR 0038); the profile has no result data.

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

/// Entries the queue holds before [`ProfileHistory::record`] drops one (and counts it).
const QUEUE: usize = 1024;
/// The most rows one segment holds.
const SEGMENT_ROWS: usize = 256;
/// How long the writer waits for more entries before it writes what it has.
const LINGER: Duration = Duration::from_millis(500);
/// The file name prefix and extension of a segment.
const PREFIX: &str = "history-";
const SUFFIX: &str = ".parquet";

/// What a query leaves behind when it ends.
pub(crate) struct HistoryEntry {
    pub(crate) query_id: String,
    pub(crate) finished_at_ms: i64,
    pub(crate) state: &'static str,
    pub(crate) error_class: Option<String>,
    pub(crate) sql: Option<String>,
    pub(crate) duration_ns: i64,
    pub(crate) plan: PhysicalPlan,
    pub(crate) metrics: QueryMetrics,
    pub(crate) identity: ProfileIdentity,
    pub(crate) trace: Option<serde_json::Value>,
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
    pub(crate) fn open(dir: &Path, max_bytes: u64) -> Result<Self, KrishivError> {
        std::fs::create_dir_all(dir)
            .map_err(|e| internal(format!("krishiv.history.path {}: {e}", dir.display())))?;
        let (tx, rx) = sync_channel(QUEUE);
        let failures = Arc::new(AtomicU64::new(0));
        let writer = Writer {
            dir: dir.to_owned(),
            max_bytes,
            failures: Arc::clone(&failures),
            seq: 0,
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
}

impl Writer {
    fn run(mut self, rx: &Receiver<Msg>) {
        let mut rows: Vec<StoredProfile> = Vec::new();
        loop {
            let msg = if rows.is_empty() {
                rx.recv().map_err(|_| RecvTimeoutError::Disconnected)
            } else {
                rx.recv_timeout(LINGER)
            };
            match msg {
                Ok(Msg::Entry(entry)) => {
                    match stored(&entry) {
                        Some(row) => rows.push(row),
                        None => {
                            self.failures.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    // Take what is already queued, up to a segment.
                    while rows.len() < SEGMENT_ROWS {
                        match rx.try_recv() {
                            Ok(Msg::Entry(e)) => match stored(&e) {
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
        let name = format!("{PREFIX}{first:013}-{:06}{SUFFIX}", self.seq);
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

fn stored(e: &HistoryEntry) -> Option<StoredProfile> {
    let mut profile = QueryProfile::new(&e.plan, e.metrics.clone(), true);
    profile.trace.clone_from(&e.trace);
    profile.identity = Some(e.identity.clone());
    Some(StoredProfile {
        query_id: e.query_id.clone(),
        finished_at_ms: e.finished_at_ms,
        state: e.state.to_owned(),
        error_class: e.error_class.clone(),
        principal: e.identity.principal.clone(),
        sql: e.sql.clone(),
        duration_ns: e.duration_ns,
        profile_version: i64::try_from(PROFILE_VERSION).ok()?,
        profile_minor: i64::try_from(PROFILE_MINOR).ok()?,
        profile: profile.to_json(),
    })
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
    let mut writer = ArrowWriter::try_new(file, schema(), None).map_err(|e| e.to_string())?;
    writer.write(&batch).map_err(|e| e.to_string())?;
    writer.close().map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, dir.join(name)).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        e.to_string()
    })
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
        std::fs::remove_file(path)?;
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
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .and_then(parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::build)
        .map_err(|e| e.to_string())?;
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
