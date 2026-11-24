// SPDX-License-Identifier: Apache-2.0
//! The query event stream (P3 E9) and the usage records it carries (E7); design `P3-E9-query-events.md`.
//!
//! Every transition of a `system.queries` row (submitted, queued, started, finished, cancelled, failed after
//! its stream) is turned into one [`QueryEvent`] by `QueryHistory`, the one choke point under `QueryLog`, so no
//! row changes without an event or the reverse. Events go to an [`EventBus`]: per registered [`EventSink`] a
//! bounded queue and one delivery thread. [`EventBus::emit`] never blocks and never fails; a full queue drops
//! events (`Progress` first, then the oldest non-terminal one) and counts them per sink and kind. With no sink
//! registered the bus is inactive and `QueryHistory` builds no event at all.
//!
//! What is not carried: error messages and credentials (the error *class* only), and statement literals:
//! the text is always masked (`krishiv_sql::mask_literals`), whatever `krishiv.system.queryText.literals`
//! says for `system.queries`, then cut to `krishiv.events.textMaxBytes`.
//!
//! Not built yet (design §6, §3.2, §4.2): `Progress` samples, `AuditEvent`s, the OTLP logs sink, `reads` /
//! `writes` / `profile` on a finished query, store request counters in the usage record.

use std::collections::VecDeque;
use std::fmt;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, PoisonError, RwLock};
use std::thread::JoinHandle;
use std::time::Duration;

use serde_json::{Value, json};

use crate::system::{QueryRow, now_ms};

/// The `schema` member of a serialized event.
pub const SCHEMA_NAME: &str = "krishiv.event.v1";

/// The version of the event schema: additive fields bump `minor`; anything else is a new `major`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SchemaVersion {
    /// Incompatible changes.
    pub major: u32,
    /// Added optional fields and kinds.
    pub minor: u32,
}

/// The version this build writes.
pub const SCHEMA_VERSION: SchemaVersion = SchemaVersion { major: 1, minor: 1 };

/// Who and what a query event is about.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct QueryRef {
    /// The query's id (the caller's operation id when it gave one).
    pub query_id: String,
    /// The session it ran in.
    pub session_id: String,
    /// The principal that ran it.
    pub principal: String,
    /// The service principal acting for `principal`, if any.
    pub delegated_by: Option<String>,
    /// `krishiv.compute.id`: what the usage is billed to.
    pub compute_id: String,
    /// The W3C trace id, if the request carried one.
    pub trace_id: Option<String>,
    /// The session's tags and the request's.
    pub tags: Vec<(String, String)>,
}

/// One event of one query.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct QueryEvent {
    /// The schema version.
    pub version: SchemaVersion,
    /// Bus-wide and increasing: one counter per runtime, so the events of one query are ordered by it but
    /// interleaved queries leave gaps in each other's, and a reused operation id never repeats a `seq`.
    pub seq: u64,
    /// When the event was emitted, milliseconds since the epoch.
    pub emitted_at_ms: i64,
    /// The query.
    pub query: QueryRef,
    /// What happened.
    pub kind: QueryEventKind,
}

/// What happened to a query.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum QueryEventKind {
    /// The statement was recorded (a refused one too: it gets a `Finished` at once).
    Submitted {
        /// `READ`, `WRITE`, `DDL`, `SESSION` or `ADMIN`; `None` when refused before it was classified.
        statement_class: Option<String>,
        /// The masked, cut statement text; `None` when the policy omits it.
        text: Option<String>,
    },
    /// Admission holds the query in its queue.
    Queued,
    /// Admission let it run.
    Started {
        /// The resource group the controller placed it in.
        resource_group: Option<String>,
        /// How long it waited in the queue.
        queued_ns: u64,
    },
    /// Usage so far (not emitted yet: design §6).
    Progress(UsageRecord),
    /// The query ended: finished, failed or denied. A query that failed after its stream (a write whose
    /// commit failed) gets a second `Finished` with a higher `seq` when its history row is still there (a row evicted meanwhile gets none); a consumer keeps the highest.
    Finished(Box<FinishedEvent>),
    /// The query was cancelled.
    Cancelled {
        /// Why: `user`, `timeout`, `admin`, `quota`, `drain` or `client_gone`; `None` when unknown.
        reason: Option<String>,
        /// What it used before it ended.
        usage: UsageRecord,
    },
}

/// The end of a query that was not cancelled.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FinishedEvent {
    /// `finished`, `failed` or `denied`.
    pub state: String,
    /// The final usage.
    pub usage: UsageRecord,
    /// The error class's name; never the message, which can hold literals.
    pub error_class: Option<String>,
    /// The statement class, when known.
    pub statement_class: Option<String>,
    /// Answered from the result cache, with no execution.
    pub result_cache_hit: bool,
}

/// Whether a usage record is a sample or the exact end figure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UsageKind {
    /// A sample while the query runs.
    Progress,
    /// The exact figures at the query's end.
    Final,
}

/// What a query used, cumulative since it started (E7): a consumer keeps the record with the highest `seq`
/// of its `record_id`, so a lost sample loses nothing.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct UsageRecord {
    /// `(session_id, query_id, seq)`: the exactly-once key (a caller's operation id is unique only per session).
    pub record_id: (String, String, u64),
    /// Sample or final.
    pub kind: UsageKind,
    /// The principal charged.
    pub principal: String,
    /// The service acting for it.
    pub delegated_by: Option<String>,
    /// The session.
    pub session_id: String,
    /// What the usage is billed to.
    pub compute_id: String,
    /// `interactive`, `batch` and so on, once the query ran.
    pub workload_class: Option<String>,
    /// The resource group.
    pub resource_group: Option<String>,
    /// The statement class.
    pub statement_class: Option<String>,
    /// The start of the window: the start of the run, the submission for a query that never ran.
    pub window_start_ms: i64,
    /// The end of the window.
    pub window_end_ms: i64,
    /// Execution time, queue excluded (as `system.queries.wall_ns`).
    pub wall_ns: Option<i64>,
    /// Time held in the admission queue.
    pub queued_ns: Option<i64>,
    /// Time spent planning before the run, outside `wall_ns` and the queue (as
    /// `system.queries.planning_ns`); `None` for a query that never ran.
    pub planning_ns: Option<i64>,
    /// Busy time of the query's drivers.
    pub cpu_ns: Option<i64>,
    /// Bytes its scans read.
    pub bytes_scanned: Option<i64>,
    /// Rows its scans produced.
    pub rows_read: Option<i64>,
    /// Files its scans opened.
    pub files_read: Option<i64>,
    /// Bytes it wrote.
    pub bytes_written: Option<i64>,
    /// Rows its write sinks took.
    pub rows_written: Option<i64>,
    /// Files it wrote.
    pub files_written: Option<i64>,
    /// Bytes spilled.
    pub spill_bytes: Option<i64>,
    /// Spill files.
    pub spill_files: Option<i64>,
    /// Peak reserved memory.
    pub peak_memory_bytes: Option<i64>,
    /// Billed as zero compute.
    pub result_cache_hit: bool,
}

/// The kinds of event, for a sink's `accept` filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EventKindTag {
    /// [`QueryEventKind::Submitted`]
    Submitted,
    /// [`QueryEventKind::Queued`]
    Queued,
    /// [`QueryEventKind::Started`]
    Started,
    /// [`QueryEventKind::Progress`]
    Progress,
    /// [`QueryEventKind::Finished`]
    Finished,
    /// [`QueryEventKind::Cancelled`]
    Cancelled,
}

impl EventKindTag {
    /// The kind's name as the JSON and the drop counters spell it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Submitted => "submitted",
            Self::Queued => "queued",
            Self::Started => "started",
            Self::Progress => "progress",
            Self::Finished => "finished",
            Self::Cancelled => "cancelled",
        }
    }

    const ALL: [Self; 6] = [
        Self::Submitted,
        Self::Queued,
        Self::Started,
        Self::Progress,
        Self::Finished,
        Self::Cancelled,
    ];

    /// An event that ends a query: dropped only when nothing else can go.
    const fn terminal(self) -> bool {
        matches!(self, Self::Finished | Self::Cancelled)
    }
}

impl QueryEventKind {
    /// The tag of this kind.
    #[must_use]
    pub const fn tag(&self) -> EventKindTag {
        match self {
            Self::Submitted { .. } => EventKindTag::Submitted,
            Self::Queued => EventKindTag::Queued,
            Self::Started { .. } => EventKindTag::Started,
            Self::Progress(_) => EventKindTag::Progress,
            Self::Finished(_) => EventKindTag::Finished,
            Self::Cancelled { .. } => EventKindTag::Cancelled,
        }
    }
}

/// Everything the bus carries (design §3.2 adds `Audit`).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EngineEvent {
    /// An event of one query.
    Query(QueryEvent),
}

impl EngineEvent {
    /// The tag of the event's kind.
    #[must_use]
    pub fn tag(&self) -> EventKindTag {
        match self {
            Self::Query(q) => q.kind.tag(),
        }
    }

    /// The event as the `krishiv.event.v1` JSON document: a stable schema, one object.
    #[must_use]
    pub fn to_json(&self) -> Value {
        match self {
            Self::Query(q) => q.to_json(),
        }
    }
}

fn opt<T: Into<Value>>(v: Option<T>) -> Value {
    v.map_or(Value::Null, Into::into)
}

impl UsageRecord {
    /// The record as JSON.
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "record_id": {"session_id": self.record_id.0, "query_id": self.record_id.1, "seq": self.record_id.2},
            "kind": match self.kind { UsageKind::Progress => "progress", UsageKind::Final => "final" },
            "principal": self.principal,
            "delegated_by": self.delegated_by,
            "session_id": self.session_id,
            "compute_id": self.compute_id,
            "workload_class": self.workload_class,
            "resource_group": self.resource_group,
            "statement_class": self.statement_class,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "wall_ns": opt(self.wall_ns),
            "queued_ns": opt(self.queued_ns),
            "planning_ns": opt(self.planning_ns),
            "cpu_ns": opt(self.cpu_ns),
            "bytes_scanned": opt(self.bytes_scanned),
            "rows_read": opt(self.rows_read),
            "files_read": opt(self.files_read),
            "bytes_written": opt(self.bytes_written),
            "rows_written": opt(self.rows_written),
            "files_written": opt(self.files_written),
            "spill_bytes": opt(self.spill_bytes),
            "spill_files": opt(self.spill_files),
            "peak_memory_bytes": opt(self.peak_memory_bytes),
            "result_cache_hit": self.result_cache_hit,
            "fidelity": match self.kind { UsageKind::Progress => "sampled", UsageKind::Final => "exact" },
        })
    }
}

impl QueryEvent {
    /// The event as the `krishiv.event.v1` JSON document.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let tags: serde_json::Map<String, Value> = self
            .query
            .tags
            .iter()
            .map(|(k, v)| (k.clone(), Value::from(v.clone())))
            .collect();
        let mut doc = json!({
            "schema": SCHEMA_NAME,
            "minor": self.version.minor,
            "seq": self.seq,
            "emitted_at_ms": self.emitted_at_ms,
            "kind": self.kind.tag().name(),
            "query": {
                "query_id": self.query.query_id,
                "session_id": self.query.session_id,
                "principal": self.query.principal,
                "delegated_by": self.query.delegated_by,
                "compute_id": self.query.compute_id,
                "trace_id": self.query.trace_id,
                "tags": tags,
            },
        });
        let body = match &self.kind {
            QueryEventKind::Submitted {
                statement_class,
                text,
            } => json!({"statement_class": statement_class, "text": text}),
            QueryEventKind::Queued => json!({}),
            QueryEventKind::Started {
                resource_group,
                queued_ns,
            } => json!({"resource_group": resource_group, "queued_ns": queued_ns}),
            QueryEventKind::Progress(u) => json!({"usage": u.to_json()}),
            QueryEventKind::Finished(f) => json!({
                "state": f.state,
                "usage": f.usage.to_json(),
                "error_class": f.error_class,
                "statement_class": f.statement_class,
                "result_cache_hit": f.result_cache_hit,
            }),
            QueryEventKind::Cancelled { reason, usage } => {
                json!({"reason": reason, "usage": usage.to_json()})
            }
        };
        if let (Some(doc), Some(body)) = (doc.as_object_mut(), body.as_object()) {
            for (k, v) in body {
                doc.insert(k.clone(), v.clone());
            }
        }
        doc
    }
}

/// A consumer of the event stream: the self-hosted platform links its own, the engine ships a JSON-lines one.
pub trait EventSink: Send + Sync + 'static {
    /// The sink's name, a label of its metrics.
    fn name(&self) -> &str;

    /// Whether the sink wants events of `kind`. It is asked on the sink's own thread after an event is dequeued, so a
    /// queue holds (and its drops count) events the sink would reject; keep it cheap, a panic here costs the batch.
    fn accept(&self, _kind: EventKindTag) -> bool {
        true
    }

    /// Handles events, in order, on the sink's own thread (never the query's).
    fn deliver(&self, batch: &[Arc<EngineEvent>]);

    /// Writes out anything buffered.
    fn flush(&self) {}

    /// Events this sink accepted but could not deliver (a failed write); counted with the drops.
    fn lost(&self) -> u64 {
        0
    }
}

/// Counted drops of one sink, by kind.
#[derive(Debug, Default)]
struct Drops([AtomicU64; 6]);

impl Drops {
    fn add(&self, kind: EventKindTag, n: u64) {
        if let Some(i) = EventKindTag::ALL.iter().position(|k| *k == kind) {
            self.0[i].fetch_add(n, Ordering::Relaxed);
        }
    }

    fn get(&self, kind: EventKindTag) -> u64 {
        EventKindTag::ALL
            .iter()
            .position(|k| *k == kind)
            .map_or(0, |i| self.0[i].load(Ordering::Relaxed))
    }
}

#[derive(Default)]
struct QueueState {
    events: VecDeque<Arc<EngineEvent>>,
    closed: bool,
}

/// One sink's bounded queue.
struct Queue {
    state: Mutex<QueueState>,
    wake: Condvar,
    capacity: usize,
    drops: Drops,
}

impl Queue {
    /// Queues `event`, or drops (and counts) what the capacity does not leave room for: `Progress`
    /// first, then the oldest non-terminal event, and the new event last of all.
    fn push(&self, event: &Arc<EngineEvent>) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.closed {
            self.drops.add(event.tag(), 1);
            return;
        }
        if state.events.len() >= self.capacity {
            let tag = event.tag();
            if tag == EventKindTag::Progress {
                self.drops.add(tag, 1);
                return;
            }
            let victim = state
                .events
                .iter()
                .position(|e| e.tag() == EventKindTag::Progress)
                .or_else(|| state.events.iter().position(|e| !e.tag().terminal()));
            match victim {
                Some(i) => {
                    if let Some(gone) = state.events.remove(i) {
                        self.drops.add(gone.tag(), 1);
                    }
                }
                None => {
                    self.drops.add(tag, 1);
                    tracing::warn!(
                        kind = tag.name(),
                        "event queue full of terminal events: the incoming event was dropped"
                    );
                    return;
                }
            }
        }
        state.events.push_back(Arc::clone(event));
        drop(state);
        self.wake.notify_one();
    }

    fn depth(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .events
            .len()
    }
}

/// A registered sink with its queue and thread.
struct SinkHandle {
    sink: Arc<dyn EventSink>,
    queue: Arc<Queue>,
    thread: Mutex<Option<JoinHandle<()>>>,
    done: Mutex<mpsc::Receiver<()>>,
}

/// The most events one delivery call gets.
const BATCH: usize = 256;

impl SinkHandle {
    fn start(sink: Arc<dyn EventSink>, capacity: usize) -> Self {
        let queue = Arc::new(Queue {
            state: Mutex::new(QueueState::default()),
            wake: Condvar::new(),
            capacity: capacity.max(1),
            drops: Drops::default(),
        });
        let (done_tx, done_rx) = mpsc::channel();
        let worker_queue = Arc::clone(&queue);
        let worker_sink = Arc::clone(&sink);
        let spawned = std::thread::Builder::new()
            .name(format!("krishiv-events-{}", sink.name().replace('\0', "")))
            .spawn(move || {
                loop {
                    let batch: Vec<Arc<EngineEvent>> = {
                        let mut state = worker_queue
                            .state
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner);
                        while state.events.is_empty() && !state.closed {
                            state = worker_queue
                                .wake
                                .wait(state)
                                .unwrap_or_else(PoisonError::into_inner);
                        }
                        if state.events.is_empty() {
                            break;
                        }
                        let n = state.events.len().min(BATCH);
                        state.events.drain(..n).collect()
                    };
                    // A panicking sink loses its batch, never the thread's queue or a query.
                    let delivered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let wanted: Vec<Arc<EngineEvent>> = batch
                            .iter()
                            .filter(|e| worker_sink.accept(e.tag()))
                            .cloned()
                            .collect();
                        if !wanted.is_empty() {
                            worker_sink.deliver(&wanted);
                        }
                    }));
                    if delivered.is_err() {
                        for e in &batch {
                            worker_queue.drops.add(e.tag(), 1);
                        }
                        tracing::warn!(sink = worker_sink.name(), "event sink panicked");
                    }
                }
                worker_sink.flush();
                // The receiver may be gone (the bus detached from a stuck sink).
                done_tx.send(()).ok();
            });
        let thread = match spawned {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::warn!(error = %e, "event sink thread did not start; its events are dropped");
                queue
                    .state
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .closed = true;
                None
            }
        };
        Self {
            sink,
            queue,
            thread: Mutex::new(thread),
            done: Mutex::new(done_rx),
        }
    }

    /// Stops accepting events, lets the thread drain what is queued and waits up to `bound` for it. What the
    /// thread has not delivered by then is counted as dropped and the thread is left to finish alone.
    fn shutdown(&self, bound: Duration) {
        {
            let mut state = self
                .queue
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            state.closed = true;
        }
        self.queue.wake.notify_all();
        let Some(thread) = self
            .thread
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        else {
            return;
        };
        let finished = self
            .done
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .recv_timeout(bound)
            .is_ok();
        if finished {
            thread.join().ok();
        } else {
            let mut state = self
                .queue
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            for e in state.events.drain(..) {
                self.queue.drops.add(e.tag(), 1);
            }
        }
    }
}

/// How long dropping a bus waits for each sink to drain.
pub const SHUTDOWN_FLUSH: Duration = Duration::from_secs(5);

/// The runtime's event bus: sinks, their queues and threads. See the module documentation.
pub struct EventBus {
    sinks: RwLock<Vec<Arc<SinkHandle>>>,
    active: AtomicBool,
    compute_id: String,
    text_max: usize,
    capacity: usize,
    next_seq: AtomicU64,
}

impl fmt::Debug for EventBus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventBus")
            .field("active", &self.is_active())
            .field("compute_id", &self.compute_id)
            .finish_non_exhaustive()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(host_name(), 4096, 10_000)
    }
}

/// The machine's name: the default `krishiv.compute.id`.
pub(crate) fn host_name() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .or_else(|| std::fs::read_to_string("/etc/hostname").ok())
        .map(|h| h.trim().to_owned())
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "localhost".to_owned())
}

impl EventBus {
    /// A bus with no sink: `compute_id` names what usage is billed to, `text_max` cuts statement text,
    /// `capacity` bounds every sink's queue.
    #[must_use]
    pub fn new(compute_id: String, text_max: usize, capacity: usize) -> Self {
        Self {
            sinks: RwLock::new(Vec::new()),
            active: AtomicBool::new(false),
            compute_id,
            text_max,
            capacity,
            next_seq: AtomicU64::new(0),
        }
    }

    /// Adds a sink, started on its own thread.
    pub fn register(&self, sink: Arc<dyn EventSink>) {
        let handle = Arc::new(SinkHandle::start(sink, self.capacity));
        self.sinks
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .push(handle);
        self.active.store(true, Ordering::Release);
    }

    /// Whether any sink is registered: when not, callers build no event.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    /// The `krishiv.compute.id` of this runtime.
    #[must_use]
    pub fn compute_id(&self) -> &str {
        &self.compute_id
    }

    /// Hands `event` to every sink that accepts its kind. Never blocks on a sink, never fails.
    pub fn emit(&self, event: EngineEvent) {
        let event = Arc::new(event);
        for h in self
            .sinks
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
        {
            // The sink's filter runs on its own thread, never here (this may hold the history lock).
            h.queue.push(&event);
        }
    }

    /// Events of `kind` the sink named `sink` dropped.
    #[must_use]
    pub fn dropped(&self, sink: &str, kind: EventKindTag) -> u64 {
        self.sinks
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|h| h.sink.name() == sink)
            .map(|h| h.queue.drops.get(kind))
            .sum()
    }

    /// Events dropped by every sink, all kinds (`krishiv_events_dropped`).
    #[must_use]
    pub fn dropped_total(&self) -> u64 {
        self.sinks
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(|h| {
                h.sink.lost()
                    + EventKindTag::ALL
                        .iter()
                        .map(|k| h.queue.drops.get(*k))
                        .sum::<u64>()
            })
            .sum()
    }

    /// Events waiting across all sinks (`krishiv_events_queue_depth`).
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        self.sinks
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(|h| h.queue.depth())
            .sum()
    }

    /// Stops every sink, each given `bound` to drain.
    pub fn shutdown(&self, bound: Duration) {
        let sinks: Vec<_> = self
            .sinks
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        for h in sinks {
            h.shutdown(bound);
        }
    }

    /// The event of `row` for the transition `kind` of query `row`, with its next `seq`.
    pub(crate) fn make(&self, row: &mut QueryRow, kind: Transition) -> EngineEvent {
        // One counter for the bus: an operation id reused by a later execution never repeats a `seq`.
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        row.event_seq = seq + 1;
        let query = QueryRef {
            query_id: row.id.clone(),
            session_id: row.session_id.clone(),
            principal: row.principal.clone(),
            delegated_by: row.delegated_by.clone(),
            compute_id: self.compute_id.clone(),
            trace_id: row.trace_id.clone(),
            tags: row.tags.clone(),
        };
        let kind = match kind {
            Transition::Submitted => QueryEventKind::Submitted {
                statement_class: row.statement_class.map(str::to_owned),
                text: row
                    .text
                    .as_deref()
                    .map(|t| cut(&krishiv_sql::mask_literals(t), self.text_max)),
            },
            Transition::Queued => QueryEventKind::Queued,
            Transition::Started => QueryEventKind::Started {
                resource_group: row.resource_group.clone(),
                queued_ns: row
                    .queued_ns
                    .and_then(|n| u64::try_from(n).ok())
                    .unwrap_or(0),
            },
            Transition::Ended if row.state == "cancelled" => QueryEventKind::Cancelled {
                reason: row.cancel_reason.map(str::to_owned),
                usage: self.usage(row, seq),
            },
            Transition::Ended => QueryEventKind::Finished(Box::new(FinishedEvent {
                state: row.state.to_owned(),
                usage: self.usage(row, seq),
                error_class: row.error_class.clone(),
                statement_class: row.statement_class.map(str::to_owned),
                result_cache_hit: row.result_cache_hit,
            })),
        };
        EngineEvent::Query(QueryEvent {
            version: SCHEMA_VERSION,
            seq,
            emitted_at_ms: now_ms(),
            query,
            kind,
        })
    }

    fn usage(&self, row: &QueryRow, seq: u64) -> UsageRecord {
        UsageRecord {
            record_id: (row.session_id.clone(), row.id.clone(), seq),
            kind: UsageKind::Final,
            principal: row.principal.clone(),
            delegated_by: row.delegated_by.clone(),
            session_id: row.session_id.clone(),
            compute_id: self.compute_id.clone(),
            workload_class: row.workload_class.map(str::to_owned),
            resource_group: row.resource_group.clone(),
            statement_class: row.statement_class.map(str::to_owned),
            window_start_ms: row.started_ms.unwrap_or(row.submitted_ms),
            window_end_ms: row.ended_ms.unwrap_or_else(now_ms),
            wall_ns: row.wall_ns,
            queued_ns: row.queued_ns,
            planning_ns: row.planning_ns,
            cpu_ns: row.cpu_ns,
            bytes_scanned: row.bytes_read,
            rows_read: row.rows_read,
            files_read: row.files_read,
            bytes_written: row.bytes_written,
            rows_written: row.rows_written,
            files_written: row.files_written,
            spill_bytes: row.spill_bytes,
            spill_files: row.spill_files,
            peak_memory_bytes: row.peak_memory_bytes,
            result_cache_hit: row.result_cache_hit,
        }
    }
}

impl Drop for EventBus {
    fn drop(&mut self) {
        self.shutdown(SHUTDOWN_FLUSH);
    }
}

/// A change of a `system.queries` row that is an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Transition {
    /// The row was added.
    Submitted,
    /// The state became `queued`.
    Queued,
    /// `started_ms` was set.
    Started,
    /// The row ended, or its outcome changed after it ended.
    Ended,
}

/// The facts of a row that decide which events an update produced.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowMark {
    state: &'static str,
    started: bool,
    ended: bool,
}

impl RowMark {
    /// The mark of `row` now.
    pub(crate) fn of(row: &QueryRow) -> Self {
        Self {
            state: row.state,
            started: row.started_ms.is_some(),
            ended: row.ended_ms.is_some(),
        }
    }

    /// The transitions from `self` to `after`, in order.
    pub(crate) fn since(self, after: Self) -> impl Iterator<Item = Transition> {
        let queued = after.state == "queued" && self.state != "queued";
        let started = after.started && !self.started;
        // The first end, or a state change of a row that had ended (`QueryLog::fail`).
        let ended = after.ended && (!self.ended || after.state != self.state);
        [
            queued.then_some(Transition::Queued),
            started.then_some(Transition::Started),
            ended.then_some(Transition::Ended),
        ]
        .into_iter()
        .flatten()
    }
}

/// Cuts `text` to at most `max` bytes at a character boundary.
fn cut(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_owned();
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_owned()
}

/// A sink that appends one JSON document per line to `<prefix>.<n>.jsonl`, moving to the next segment past
/// `max_bytes` and keeping the newest `max_files`. A segment is complete lines only; a failed write is
/// logged and counted in [`JsonLinesSink::failures`], never raised.
#[derive(Debug)]
pub struct JsonLinesSink {
    prefix: PathBuf,
    max_bytes: u64,
    max_files: usize,
    state: Mutex<JsonState>,
    failures: AtomicU64,
}

#[derive(Debug, Default)]
struct JsonState {
    file: Option<std::io::BufWriter<std::fs::File>>,
    written: u64,
    /// Bytes of the open segment known to hold only whole lines (flushed).
    committed: u64,
    /// Events of the batch in hand that an earlier segment's flush made durable.
    durable: usize,
    segment: u64,
}

impl JsonLinesSink {
    /// A sink writing under `prefix` (a directory is created for its parent).
    #[must_use]
    pub fn new(prefix: PathBuf, max_bytes: u64, max_files: usize) -> Self {
        Self {
            prefix,
            max_bytes: max_bytes.max(1),
            max_files: max_files.max(1),
            state: Mutex::new(JsonState::default()),
            failures: AtomicU64::new(0),
        }
    }

    /// Events lost to failed writes.
    #[must_use]
    pub fn failures(&self) -> u64 {
        self.failures.load(Ordering::Relaxed)
    }

    fn segment_path(&self, n: u64) -> PathBuf {
        let mut name = self.prefix.as_os_str().to_owned();
        name.push(format!(".{n:06}.jsonl"));
        PathBuf::from(name)
    }

    /// The segment numbers on disk under the prefix, ascending. A directory that cannot be listed is an
    /// error, not an empty list. Names are matched lossily so a non-UTF-8 prefix still finds its segments.
    fn existing_segments(&self) -> std::io::Result<Vec<u64>> {
        let (Some(dir), Some(stem)) = (self.prefix.parent(), self.prefix.file_name()) else {
            return Ok(Vec::new());
        };
        let dir = if dir.as_os_str().is_empty() {
            std::path::Path::new(".")
        } else {
            dir
        };
        let head = format!("{}.", stem.to_string_lossy());
        let mut found = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            if let Some(n) = name
                .strip_prefix(&head)
                .and_then(|r| r.strip_suffix(".jsonl"))
                .and_then(|r| r.parse::<u64>().ok())
            {
                found.push(n);
            }
        }
        found.sort_unstable();
        Ok(found)
    }

    fn open_next(&self, state: &mut JsonState) -> std::io::Result<()> {
        // The old file stays in the state until its flush succeeds, so a failure can still cut its tail.
        if let Some(f) = state.file.as_mut() {
            f.flush()?;
        }
        state.file = None;
        if let Some(dir) = self.prefix.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let existing = self.existing_segments()?;
        // The next number follows the highest on disk (a restart continues after the previous run, so name
        // order stays time order); it is committed to the state only once the file is open.
        let highest = existing.last().copied().unwrap_or(0).max(state.segment);
        let next = highest
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("event file segment numbers are exhausted"))?;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.segment_path(next))?;
        state.segment = next;
        state.written = file.metadata().map_or(0, |m| m.len());
        state.committed = state.written;
        state.file = Some(std::io::BufWriter::new(file));
        // Keep the newest `max_files` segments, whatever gaps or a lower limit leave behind.
        let keep = self.max_files.saturating_sub(1);
        let old: Vec<u64> = existing.into_iter().filter(|n| *n != next).collect();
        let excess = old.len().saturating_sub(keep);
        for n in &old[..excess] {
            std::fs::remove_file(self.segment_path(*n)).ok();
        }
        Ok(())
    }

    /// Writes the batch; on failure returns the error with the count of the batch's events an earlier
    /// segment's flush made durable, read under the same lock so a concurrent delivery cannot change it.
    fn write(&self, batch: &[Arc<EngineEvent>]) -> Result<(), (std::io::Error, usize)> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let result = self.write_locked(&mut state, batch);
        let durable = state.durable;
        if result.is_err() {
            // A failed write may leave a partial line: never append to that segment again, and cut it
            // back to the last flushed whole line.
            if let Some(f) = state.file.take() {
                drop(f.into_parts().0);
                let path = self.segment_path(state.segment);
                let cut = std::fs::OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .and_then(|f| f.set_len(state.committed));
                if let Err(e) = cut {
                    tracing::warn!(error = %e, "event file tail could not be repaired");
                }
            }
        }
        result.map_err(|e| (e, durable))
    }

    fn write_locked(
        &self,
        state: &mut JsonState,
        batch: &[Arc<EngineEvent>],
    ) -> std::io::Result<()> {
        state.durable = 0;
        for (i, event) in batch.iter().enumerate() {
            if state.file.is_none() || state.written >= self.max_bytes {
                if let Some(f) = state.file.as_mut() {
                    f.flush()?;
                    state.committed = state.written;
                    state.durable = i;
                }
                self.open_next(state)?;
            }
            let mut line = event.to_json().to_string();
            line.push('\n');
            let len = line.len() as u64;
            if let Some(f) = state.file.as_mut() {
                f.write_all(line.as_bytes())?;
            }
            state.written += len;
        }
        if let Some(f) = state.file.as_mut() {
            f.flush()?;
        }
        state.committed = state.written;
        state.durable = batch.len();
        Ok(())
    }
}

impl EventSink for JsonLinesSink {
    fn name(&self) -> &str {
        "jsonl"
    }

    fn deliver(&self, batch: &[Arc<EngineEvent>]) {
        if let Err((e, durable)) = self.write(batch) {
            self.failures.fetch_add(
                batch.len().saturating_sub(durable) as u64,
                Ordering::Relaxed,
            );
            tracing::warn!(error = %e, "event file write failed");
        }
    }

    fn lost(&self) -> u64 {
        self.failures()
    }

    fn flush(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(f) = state.file.as_mut()
            && let Err(e) = f.flush()
        {
            self.failures.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(error = %e, "event file flush failed");
        }
    }
}

#[cfg(test)]
mod tests;
