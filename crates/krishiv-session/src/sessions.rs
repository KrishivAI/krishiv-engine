// SPDX-License-Identifier: Apache-2.0
//! The runtime's session registry (P3-T064, design `P3-T064-sessions-tags.md` §4.1): the live
//! sessions and a bounded ring of the ones that ended, behind `system.sessions`,
//! [`crate::EngineRuntime::kill_session`] and the idle sweep's `expire_session`.
//!
//! A live entry holds a weak reference to its session (so the registry never keeps one alive)
//! and the facts that change on every call live in [`Activity`], shared with the session, so a
//! `touch` is an atomic store and takes no registry lock.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicI64, AtomicU8, AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError, Weak};

use crate::session::SessionState;
use std::sync::Arc;

use arrow_array::{ArrayRef, TimestampMicrosecondArray};

use crate::system::{Kind, Sources, labels_column, long, now_ms, text};

/// Closed sessions the registry remembers (Spark's `closedSessionsTombstonesSize` default).
pub(crate) const TOMBSTONES: usize = 1000;

/// The front end a session was opened through (`system.sessions.transport`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Transport {
    /// A library caller in the same process.
    #[default]
    Embedded,
    /// Spark Connect.
    Connect,
    /// Arrow Flight SQL.
    FlightSql,
    /// The MCP server.
    Mcp,
    /// The web UI.
    Ui,
}

impl Transport {
    fn name(self) -> &'static str {
        match self {
            Self::Embedded => "embedded",
            Self::Connect => "connect",
            Self::FlightSql => "flight_sql",
            Self::Mcp => "mcp",
            Self::Ui => "ui",
        }
    }

    fn code(self) -> u8 {
        self as u8
    }

    fn from_code(c: u8) -> Self {
        match c {
            1 => Self::Connect,
            2 => Self::FlightSql,
            3 => Self::Mcp,
            4 => Self::Ui,
            _ => Self::Embedded,
        }
    }
}

/// Why a session ended (`system.sessions.state` and `close_reason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum End {
    /// `Session::close` / `ReleaseSession`, or the last handle dropped.
    Closed(&'static str),
    /// The idle sweep.
    Expired,
    /// An administrator (or a drain).
    Killed(&'static str),
}

impl End {
    fn state(self) -> &'static str {
        match self {
            Self::Closed(_) => "closed",
            Self::Expired => "expired",
            Self::Killed(_) => "killed",
        }
    }

    fn reason(self) -> &'static str {
        match self {
            Self::Closed(r) | Self::Killed(r) => r,
            Self::Expired => "idle_timeout",
        }
    }
}

/// The per-call facts of a session, shared between it and the registry.
#[derive(Debug)]
pub(crate) struct Activity {
    created_ms: i64,
    last_active_ms: AtomicI64,
    started: AtomicU64,
    transport: AtomicU8,
    idle_timeout_ms: AtomicI64,
    client: Mutex<ClientInfo>,
}

/// What a front end says about its client (all client-asserted, shown, never trusted).
#[derive(Debug, Default, Clone)]
struct ClientInfo {
    client_type: Option<String>,
    client_user_id: Option<String>,
    server_session_id: Option<String>,
}

/// "No idle timeout set" in [`Activity::idle_timeout_ms`].
const NO_TIMEOUT: i64 = i64::MIN;

impl Default for Activity {
    fn default() -> Self {
        let now = now_ms();
        Self {
            created_ms: now,
            last_active_ms: AtomicI64::new(now),
            started: AtomicU64::new(0),
            transport: AtomicU8::new(Transport::Embedded.code()),
            idle_timeout_ms: AtomicI64::new(NO_TIMEOUT),
            client: Mutex::new(ClientInfo::default()),
        }
    }
}

impl Activity {
    /// A call arrived on the session.
    pub(crate) fn touch(&self) {
        self.last_active_ms.store(now_ms(), Ordering::Relaxed);
    }

    /// A query started.
    pub(crate) fn query_started(&self) {
        self.started.fetch_add(1, Ordering::Relaxed);
        self.touch();
    }

    pub(crate) fn set_transport(&self, t: Transport) {
        self.transport.store(t.code(), Ordering::Relaxed);
    }

    pub(crate) fn set_idle_timeout_ms(&self, ms: Option<i64>) {
        self.idle_timeout_ms
            .store(ms.unwrap_or(NO_TIMEOUT), Ordering::Relaxed);
    }

    /// Keeps the first non-empty client type and user id seen, and the latest server session id.
    pub(crate) fn set_client(
        &self,
        client_type: Option<&str>,
        user: Option<&str>,
        server: Option<&str>,
    ) {
        let mut c = self.client.lock().unwrap_or_else(PoisonError::into_inner);
        let keep = |slot: &mut Option<String>, v: Option<&str>| {
            if slot.is_none()
                && let Some(v) = v.filter(|v| !v.is_empty())
            {
                *slot = Some(v.to_owned());
            }
        };
        keep(&mut c.client_type, client_type);
        keep(&mut c.client_user_id, user);
        if let Some(s) = server.filter(|s| !s.is_empty()) {
            c.server_session_id = Some(s.to_owned());
        }
    }
}

/// One row of `system.sessions`.
#[derive(Debug, Clone)]
pub(crate) struct SessionRow {
    pub(crate) session_id: String,
    pub(crate) server_session_id: Option<String>,
    pub(crate) principal: String,
    pub(crate) client_user_id: Option<String>,
    pub(crate) client_type: Option<String>,
    pub(crate) transport: &'static str,
    pub(crate) state: &'static str,
    pub(crate) created_ms: i64,
    pub(crate) last_active_ms: i64,
    pub(crate) closed_ms: Option<i64>,
    pub(crate) close_reason: Option<&'static str>,
    pub(crate) idle_timeout_ms: Option<i64>,
    pub(crate) current_catalog: String,
    pub(crate) current_namespace: String,
    pub(crate) tags: Vec<(String, String)>,
    pub(crate) queries_running: i64,
    pub(crate) queries_started: i64,
}

impl SessionRow {
    /// The row of `s` as it is now, ended with `end` when given.
    pub(crate) fn of(s: &SessionState, end: Option<End>, running: i64) -> Self {
        let a = &s.activity;
        let c = a
            .client
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let (catalog, namespace) = s.catalog.position_pair();
        let timeout = a.idle_timeout_ms.load(Ordering::Relaxed);
        Self {
            session_id: s.session_id.clone(),
            server_session_id: c.server_session_id,
            principal: s.principal.name.clone(),
            client_user_id: c.client_user_id,
            client_type: c.client_type,
            transport: Transport::from_code(a.transport.load(Ordering::Relaxed)).name(),
            state: end.map_or("active", End::state),
            created_ms: a.created_ms,
            last_active_ms: a.last_active_ms.load(Ordering::Relaxed),
            closed_ms: end.map(|_| now_ms()),
            close_reason: end.map(End::reason),
            idle_timeout_ms: (timeout != NO_TIMEOUT).then_some(timeout),
            current_catalog: catalog,
            current_namespace: namespace,
            tags: s.queries.tags(),
            queries_running: if end.is_some() { 0 } else { running },
            queries_started: i64::try_from(a.started.load(Ordering::Relaxed)).unwrap_or(i64::MAX),
        }
    }
}

#[derive(Default)]
struct Inner {
    live: HashMap<u64, Weak<SessionState>>,
    ended: VecDeque<SessionRow>,
}

/// The sessions of one runtime.
pub(crate) struct SessionRegistry {
    inner: Mutex<Inner>,
    cap: usize,
}

impl Default for SessionRegistry {
    fn default() -> Self {
        Self::with_capacity(TOMBSTONES)
    }
}

impl SessionRegistry {
    pub(crate) fn with_capacity(cap: usize) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            cap,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn register(&self, number: u64, session: Weak<SessionState>) {
        self.lock().live.insert(number, session);
    }

    /// Moves session `number` to the ended ring with `row`; the oldest entry goes past the cap.
    /// A no-op when it already ended.
    pub(crate) fn retire(&self, number: u64, row: impl FnOnce() -> SessionRow) {
        let mut g = self.lock();
        if g.live.remove(&number).is_none() {
            return;
        }
        g.ended.push_back(row());
        while g.ended.len() > self.cap {
            g.ended.pop_front();
        }
    }

    /// The live sessions' weak handles, copied out so they upgrade (and a last strong reference
    /// drops, which retires the session) with the registry unlocked.
    fn live_handles(&self) -> Vec<(u64, Weak<SessionState>)> {
        self.lock()
            .live
            .iter()
            .map(|(n, w)| (*n, Weak::clone(w)))
            .collect()
    }

    /// The live session with id `id`.
    pub(crate) fn find(&self, id: &str) -> Option<(u64, std::sync::Arc<SessionState>)> {
        self.live_handles().into_iter().find_map(|(n, w)| {
            let s = w.upgrade()?;
            (s.session_id == id).then_some((n, s))
        })
    }

    /// Whether `id` ended and is still remembered.
    pub(crate) fn ended(&self, id: &str) -> bool {
        self.lock().ended.iter().any(|r| r.session_id == id)
    }

    /// Every session, live ones first by creation time, then the ended ones oldest first;
    /// `running` gives a live session's running queries.
    pub(crate) fn snapshot(&self, running: impl Fn(u64) -> i64) -> Vec<SessionRow> {
        let mut live: Vec<SessionRow> = self
            .live_handles()
            .into_iter()
            .filter_map(|(n, w)| {
                let s = w.upgrade()?;
                Some(SessionRow::of(&s, None, running(n)))
            })
            .collect();
        live.sort_by_key(|r| (r.created_ms, r.session_id.clone()));
        live.extend(self.lock().ended.iter().cloned());
        live
    }
}

/// The runtime's session registry and how many queries each session runs, for `system.sessions`.
pub(crate) struct SessionView {
    pub(crate) registry: Arc<crate::sessions::SessionRegistry>,
    pub(crate) running: Box<dyn Fn(u64) -> i64 + Send + Sync>,
}

impl Default for SessionView {
    fn default() -> Self {
        Self {
            registry: Arc::default(),
            running: Box::new(|_| 0),
        }
    }
}

/// The columns of `system.sessions` for the session that asked.
pub(crate) fn columns(sources: &Sources) -> Vec<ArrayRef> {
    let all = sources.monitor.sees_all();
    let me = sources.queries.principal();
    let rows: Vec<crate::sessions::SessionRow> = sources
        .sessions
        .registry
        .snapshot(&*sources.sessions.running)
        .into_iter()
        .filter(|r| all || r.principal == me)
        .collect();
    let micros = |ms: Option<i64>| ms.map(|m| m.saturating_mul(1000));
    let time = |f: &dyn Fn(&crate::sessions::SessionRow) -> Option<i64>| -> ArrayRef {
        Arc::new(
            TimestampMicrosecondArray::from_iter(rows.iter().map(|r| micros(f(r))))
                .with_timezone("UTC"),
        )
    };
    vec![
        text(rows.iter().map(|r| Some(r.session_id.as_str()))),
        text(rows.iter().map(|r| r.server_session_id.as_deref())),
        text(rows.iter().map(|r| Some(r.principal.as_str()))),
        text(rows.iter().map(|r| r.client_user_id.as_deref())),
        text(rows.iter().map(|r| r.client_type.as_deref())),
        text(rows.iter().map(|r| Some(r.transport))),
        text(rows.iter().map(|r| Some(r.state))),
        time(&|r| Some(r.created_ms)),
        time(&|r| Some(r.last_active_ms)),
        time(&|r| r.closed_ms),
        text(rows.iter().map(|r| r.close_reason)),
        long(rows.iter().map(|r| r.idle_timeout_ms)),
        text(rows.iter().map(|r| Some(r.current_catalog.as_str()))),
        text(rows.iter().map(|r| Some(r.current_namespace.as_str()))),
        labels_column(&rows.iter().map(|r| r.tags.as_slice()).collect::<Vec<_>>()),
        long(rows.iter().map(|r| Some(r.queries_running))),
        long(rows.iter().map(|r| Some(r.queries_started))),
    ]
}

/// The columns of `system.sessions`: `(name, kind, nullable)`.
pub(crate) fn layout() -> Vec<(&'static str, Kind, bool)> {
    use Kind::{Labels, Long, Text, Time};
    vec![
        ("session_id", Text, false),
        ("server_session_id", Text, true),
        ("principal", Text, false),
        ("client_user_id", Text, true),
        ("client_type", Text, true),
        ("transport", Text, false),
        ("state", Text, false),
        ("created_at", Time, false),
        ("last_active_at", Time, false),
        ("closed_at", Time, true),
        ("close_reason", Text, true),
        ("idle_timeout_ms", Long, true),
        ("current_catalog", Text, false),
        ("current_namespace", Text, false),
        ("tags", Labels, false),
        ("queries_running", Long, false),
        ("queries_started", Long, false),
    ]
}
