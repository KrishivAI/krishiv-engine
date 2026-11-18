// SPDX-License-Identifier: Apache-2.0
//! `FlightSqlServer`: the Arrow Flight SQL service's state (design P3-T101 §2): Connect's session
//! manager, auth, drain flag and response budget, Flight's own session table ([`Sessions`]) and
//! handle map ([`Handles`]), its limits from the `krishiv.flightSql.*` keys, the sweep task and
//! `close_all` for shutdown. The RPCs are in `service.rs`, `statement.rs` and `call.rs`.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use arrow_array::RecordBatch;
use arrow_flight::flight_service_server::FlightServiceServer;
use arrow_flight::sql::metadata::SqlInfoData;
use arrow_schema::SchemaRef;
use krishiv_common::config::{
    CONNECT_MAX_INBOUND_MESSAGE_SIZE, Config, ConfigEntry, ConfigValue,
    FLIGHT_SQL_FETCH_IDLE_TIMEOUT, FLIGHT_SQL_HANDLE_TTL, FLIGHT_SQL_MAX_COMMAND_RESULT_BYTES,
    FLIGHT_SQL_MAX_HANDLE_BYTES_PER_PRINCIPAL, FLIGHT_SQL_MAX_HANDLES_PER_PRINCIPAL,
    FLIGHT_SQL_MAX_HANDLES_PER_SESSION, FLIGHT_SQL_MAX_STATEMENT_BYTES, FLIGHT_SQL_PROVISIONAL_TTL,
};
use krishiv_common::error::KrishivError;
use krishiv_connect_server::execute::ResponseBudget;
use krishiv_connect_server::{ConnectAuth, ExecuteLimits, Operation, SessionManager, SessionOwner};
use krishiv_memory::Reservation;

use crate::encode::EncodeLimits;
use crate::gate::FlightGate;
use crate::handles::{HandleLimits, Handles};
use crate::metrics::Requests;
use crate::sessions::{Holds, Release, Sessions};
use crate::sql_info::{ServerInfo, sql_info_data};
use crate::statement::FetchCell;

/// The limits of a [`FlightSqlServer`], from the `krishiv.flightSql.*` keys (all Startup).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlightSqlConfig {
    /// `handleTtl`, `maxHandlesPerSession`, `maxHandlesPerPrincipal`,
    /// `maxHandleBytesPerPrincipal`.
    pub handles: HandleLimits,
    /// `provisionalTtl` (design §3.2 rule 2).
    pub provisional_ttl: Duration,
    /// `fetchIdleTimeout` (design §4.2).
    pub fetch_idle_timeout: Duration,
    /// `maxStatementBytes`: the longest SQL text, checked before parsing.
    pub max_statement_bytes: usize,
    /// `maxCommandResultBytes`: the largest result a command stores for its ticket.
    pub max_command_result_bytes: usize,
    /// `spark.connect.grpc.maxInboundMessageSize`: the Flight service's inbound message limit,
    /// as Connect's (design §4.2).
    pub max_inbound_message_size: usize,
    /// How result streams are cut (the module constants; tests use smaller ones).
    pub encode: EncodeLimits,
    /// The most operations one session holds (`krishiv.connect.maxOperationsPerSession`): a
    /// `DoGet` and a running command register one each, as Connect's `ExecutePlan` does.
    pub max_operations_per_session: usize,
}

impl Default for FlightSqlConfig {
    fn default() -> Self {
        Self {
            handles: HandleLimits::default(),
            provisional_ttl: Duration::from_secs(60),
            fetch_idle_timeout: Duration::from_secs(300),
            max_statement_bytes: 1 << 20,
            max_command_result_bytes: 16 << 20,
            max_inbound_message_size: 128 << 20,
            encode: EncodeLimits::default(),
            max_operations_per_session: ExecuteLimits::default().max_operations_per_session,
        }
    }
}

fn duration(config: &Config, entry: &ConfigEntry, default: Duration) -> Duration {
    config
        .get(entry)
        .and_then(ConfigValue::as_duration)
        .unwrap_or(default)
}

fn size(config: &Config, entry: &ConfigEntry, default: usize) -> usize {
    match config.get(entry) {
        Some(ConfigValue::Size(n)) => usize::try_from(*n).unwrap_or(usize::MAX),
        Some(ConfigValue::Int(n)) => usize::try_from(*n).unwrap_or(default),
        _ => default,
    }
}

impl FlightSqlConfig {
    /// The limits `config` sets (defaults for keys it leaves unset).
    #[must_use]
    pub fn from_config(config: &Config) -> Self {
        let d = Self::default();
        let dh = d.handles;
        Self {
            handles: HandleLimits {
                ttl: duration(config, &FLIGHT_SQL_HANDLE_TTL, dh.ttl),
                per_session: size(config, &FLIGHT_SQL_MAX_HANDLES_PER_SESSION, dh.per_session),
                per_owner: size(config, &FLIGHT_SQL_MAX_HANDLES_PER_PRINCIPAL, dh.per_owner),
                bytes_per_owner: size(
                    config,
                    &FLIGHT_SQL_MAX_HANDLE_BYTES_PER_PRINCIPAL,
                    dh.bytes_per_owner,
                ),
            },
            provisional_ttl: duration(config, &FLIGHT_SQL_PROVISIONAL_TTL, d.provisional_ttl),
            fetch_idle_timeout: duration(
                config,
                &FLIGHT_SQL_FETCH_IDLE_TIMEOUT,
                d.fetch_idle_timeout,
            ),
            max_statement_bytes: size(
                config,
                &FLIGHT_SQL_MAX_STATEMENT_BYTES,
                d.max_statement_bytes,
            ),
            max_command_result_bytes: size(
                config,
                &FLIGHT_SQL_MAX_COMMAND_RESULT_BYTES,
                d.max_command_result_bytes,
            ),
            max_inbound_message_size: size(
                config,
                &CONNECT_MAX_INBOUND_MESSAGE_SIZE,
                d.max_inbound_message_size,
            ),
            encode: d.encode,
            max_operations_per_session: ExecuteLimits::from_config(config)
                .max_operations_per_session,
        }
    }
}

/// What a handle holds (design §4.1): its full owner (the map counts under `cap_key`, so a
/// lookup checks this), the SQL text, the clock and, once its `GetFlightInfo` ran, its kind.
#[derive(Clone)]
pub(crate) struct Payload {
    pub(crate) owner: SessionOwner,
    pub(crate) sql: Arc<str>,
    pub(crate) query_start_micros: i64,
    pub(crate) kind: Kind,
    /// The operations running for this handle (its `GetFlightInfo` run and its fetches), so
    /// `CancelFlightInfo` cancels these and nothing else of the session (§4.5).
    pub(crate) live: Arc<LiveOps>,
}

/// The operations live on one handle (design §4.1, §4.5), shared by every copy of its payload.
/// Once the handle is cancelled no operation joins it, so a fetch that began just before the
/// cancel never streams (review of 15c4ffa75, finding 5).
#[derive(Default)]
pub(crate) struct LiveOps(Mutex<Live>);

#[derive(Default)]
struct Live {
    ops: Vec<Arc<Operation>>,
    cancelled: bool,
}

impl LiveOps {
    fn live(&self) -> std::sync::MutexGuard<'_, Live> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Records `op` as running for the handle; `false` (not recorded) once it is cancelled.
    #[must_use]
    pub(crate) fn add(&self, op: &Arc<Operation>) -> bool {
        let mut live = self.live();
        if live.cancelled {
            return false;
        }
        live.ops.push(Arc::clone(op));
        true
    }

    /// Forgets `op` (its run or fetch ended).
    pub(crate) fn remove(&self, op: &Arc<Operation>) {
        self.live().ops.retain(|o| !Arc::ptr_eq(o, op));
    }

    /// Marks the handle cancelled and returns the operations running now, in one step.
    pub(crate) fn cancel(&self) -> Vec<Arc<Operation>> {
        let mut live = self.live();
        live.cancelled = true;
        live.ops.clone()
    }
}

/// The kind of a handle (design §4.1).
#[derive(Clone)]
pub(crate) enum Kind {
    /// Reserved: its `GetFlightInfo` has not answered yet. A fetch of it is `NOT_FOUND` (the
    /// client never saw the ticket).
    Pending,
    /// A query: re-analyzed at each fetch; the schema its `FlightInfo` announced.
    Query(SchemaRef),
    /// A command that ran: its stored rows (Spark's Arrow shapes, the session time zone fixed
    /// when it ran), held in the pool as `flight_sql_handles` while the handle lives.
    Command(Arc<Stored>),
}

/// A command's stored result.
pub(crate) struct Stored {
    pub(crate) schema: SchemaRef,
    pub(crate) batches: Vec<RecordBatch>,
    /// The pool grant of the rows (none for a result of at most 64 KiB).
    pub(crate) _reservation: Option<Reservation>,
}

/// The memory consumer stored command results are reserved under (design §4.1, v3 review M3).
pub const HANDLES_CONSUMER: &str = "flight_sql_handles";

/// A stored result of at most this many bytes takes no pool grant (every DML and DDL result).
pub(crate) const SMALL_RESULT_BYTES: usize = 64 << 10;

/// The shared state of the service.
pub(crate) struct Inner {
    pub(crate) manager: Arc<SessionManager>,
    pub(crate) auth: Arc<ConnectAuth>,
    pub(crate) draining: Arc<AtomicBool>,
    pub(crate) budget: ResponseBudget,
    pub(crate) sessions: Sessions<SessionOwner>,
    pub(crate) handles: Handles<SessionOwner, Payload>,
    pub(crate) requests: Requests,
    pub(crate) sql_info: SqlInfoData,
    pub(crate) config: FlightSqlConfig,
    /// The live `DoGet` streams' idle cells, for the sweep (design §4.2).
    pub(crate) fetches: Mutex<Vec<Weak<FetchCell>>>,
}

/// Krishiv's Arrow Flight SQL service (module docs). Cheap to clone: the state is shared.
#[derive(Clone)]
pub struct FlightSqlServer {
    pub(crate) inner: Arc<Inner>,
}

impl std::fmt::Debug for FlightSqlServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlightSqlServer")
            .field("sessions", &self.inner.sessions)
            .field("handles", &self.inner.handles)
            .field("config", &self.inner.config)
            .finish_non_exhaustive()
    }
}

impl FlightSqlServer {
    /// A service over the Connect listener's `manager`, `auth`, drain flag
    /// (`ConnectServer::draining`) and response budget (`ConnectServer::response_budget`), with
    /// `info` for `GetSqlInfo`.
    ///
    /// # Errors
    /// What building the `GetSqlInfo` table fails with (`INTERNAL_ERROR`).
    pub fn new(
        manager: Arc<SessionManager>,
        auth: Arc<ConnectAuth>,
        draining: Arc<AtomicBool>,
        budget: ResponseBudget,
        info: &ServerInfo,
        config: FlightSqlConfig,
    ) -> Result<Self, KrishivError> {
        Ok(Self {
            inner: Arc::new(Inner {
                manager,
                auth,
                draining,
                budget,
                sessions: Sessions::new(config.provisional_ttl),
                handles: Handles::new(config.handles),
                requests: Requests::default(),
                sql_info: sql_info_data(info)?,
                config,
                fetches: Mutex::new(Vec::new()),
            }),
        })
    }

    /// The tonic service to add to the Connect listener (`ConnectServer::with_routes`): the
    /// fail-closed [`FlightGate`] in front of the arrow-rs dispatch (design §16 m5), with the
    /// inbound message limit of `spark.connect.grpc.maxInboundMessageSize`.
    #[must_use]
    pub fn service(&self) -> FlightServiceServer<FlightGate> {
        FlightServiceServer::new(FlightGate(self.clone()))
            .max_decoding_message_size(self.inner.config.max_inbound_message_size)
    }

    /// [`Self::service`] as the routes `ConnectServer::with_routes` takes.
    #[must_use]
    pub fn routes(&self) -> tonic::service::Routes {
        tonic::service::Routes::new(self.service())
    }

    /// The request counters (`krishiv_flight_sql_requests_total`) for the server's scrape.
    #[must_use]
    pub fn requests(&self) -> &Requests {
        &self.inner.requests
    }

    /// `krishiv.flight_sql.requests` as counter samples, for the server's `/metrics` scrape
    /// (design §7).
    #[must_use]
    pub fn metrics(&self) -> krishiv_session::MetricsSnapshot {
        let mut out = krishiv_session::MetricsSnapshot::default();
        for c in self.inner.requests.counts() {
            out.counters.push(krishiv_session::CounterSample {
                name: "krishiv.flight_sql.requests",
                unit: "{request}",
                labels: vec![
                    ("command".to_owned(), c.command.as_str().to_owned()),
                    ("outcome".to_owned(), c.outcome.as_str().to_owned()),
                ],
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "a counter past 2^53 loses only its low bits"
                )]
                value: c.count as f64,
            });
        }
        out
    }

    /// How many statement and prepared-statement handles are stored.
    #[must_use]
    pub fn handle_count(&self) -> usize {
        self.inner.handles.len()
    }

    /// How many Flight sessions the table holds.
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.inner.sessions.len()
    }

    /// One sweep (design §2, §3.2, §4.1) at `now`: Flight sessions the manager released leave
    /// the table, unchanged provisional sessions past `provisionalTtl` are released, expired
    /// handles and handles of gone sessions are dropped. Never stamps a session's access.
    /// Returns how many sessions were let go.
    pub fn sweep_at(&self, now: Instant) -> usize {
        let inner = &self.inner;
        // Liveness is asked outside the table's lock (the manager has its own).
        let gone: HashSet<String> = inner
            .sessions
            .snapshot()
            .into_iter()
            .filter(|(id, owner)| !inner.manager.is_live(owner, id))
            .map(|(id, _)| id)
            .collect();
        self.expire_idle_fetches(now);
        let released = inner.sessions.sweep(now, &gone);
        let n = released.len();
        for r in released {
            self.release(&r);
        }
        // `alive` runs under the handle map's lock: the session table comes first in the lock
        // order, so its snapshot is read before, never inside. A handle whose session is not in
        // the table (released, swept) goes.
        let known: HashSet<String> = inner
            .sessions
            .snapshot()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        inner.handles.sweep(now, |sid| known.contains(sid));
        n
    }

    /// Starts the sweep task on `interval` (`ConnectServerConfig::maintenance_interval`, the
    /// value Connect's sweep uses); it ends when the returned guard is dropped (`krishiv-cli`
    /// holds it for the life of `serve`).
    #[must_use]
    pub fn spawn_sweeper(&self, interval: Duration) -> SweepGuard {
        let server = self.clone();
        let task = tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval.max(Duration::from_millis(10)));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tick.tick().await;
                server.sweep_at(Instant::now());
            }
        });
        SweepGuard(task)
    }

    /// Releases every Flight session and drops every handle (design §16 m7: `krishiv-cli` calls
    /// it right after `serve` returns; the manager's `close_all` has closed the sessions).
    /// Returns how many sessions the table held.
    pub fn close_all(&self) -> usize {
        let rows = self.inner.sessions.snapshot();
        let n = rows.len();
        for (id, owner) in rows {
            if let Some(r) = self.inner.sessions.close(&owner, &id) {
                self.release(&r);
            }
        }
        n
    }

    /// Watches a `DoGet` stream's idle cell (the sweep holds it weakly).
    pub(crate) fn watch_fetch(&self, cell: &Arc<FetchCell>) {
        self.fetches().push(Arc::downgrade(cell));
    }

    fn fetches(&self) -> std::sync::MutexGuard<'_, Vec<Weak<FetchCell>>> {
        self.inner
            .fetches
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Cancels the fetches whose reader has not asked for the next piece within
    /// `fetchIdleTimeout` (design §4.2); returns how many. The marks are dropped after the lock.
    pub(crate) fn expire_idle_fetches(&self, now: Instant) -> usize {
        let timeout = self.inner.config.fetch_idle_timeout;
        let live: Vec<Arc<FetchCell>> = {
            let mut cells = self.fetches();
            cells.retain(|w| w.strong_count() > 0);
            cells.iter().filter_map(Weak::upgrade).collect()
        };
        let expired: Vec<_> = live.iter().filter_map(|c| c.expire(now, timeout)).collect();
        let n = expired.len();
        if n > 0 {
            tracing::info!(target: "krishiv::flight_sql", fetches = n, "idle fetches cancelled");
        }
        drop(expired);
        n
    }

    /// Whether the server is shutting down (`ConnectServer::draining`).
    pub(crate) fn draining(&self) -> bool {
        self.inner.draining.load(Ordering::Acquire)
    }

    /// Runs a release the session table decided, after its lock: the session's handles go, and
    /// a live session is released in the manager with no tombstone (design §3.2: Flight ids are
    /// minted and never reused).
    pub(crate) fn release(&self, r: &Release<SessionOwner>) {
        self.inner.handles.remove_session(&r.session_id);
        if r.live {
            let _ = self
                .inner
                .manager
                .release_owned(&r.owner, &r.session_id, true);
        }
    }

    /// What session `id`'s handles hold, for the session table's `holds` closures (lock order:
    /// the table, then the handles).
    pub(crate) fn holds(&self, id: &str) -> Holds {
        let (handles, prepared) = self.inner.handles.held_by_session(id);
        Holds { handles, prepared }
    }

    /// After handles were evicted to make room (design §4.1, v4 review m4): a provisional session
    /// left with none is released (rule 3). An evicted handle may be of a session of another
    /// owner with the same `cap_key` (other act-as groups), so each row's own owner is used.
    pub(crate) fn after_evictions(&self, evicted: &[String]) {
        if evicted.is_empty() {
            return;
        }
        let wanted: HashSet<&str> = evicted.iter().map(String::as_str).collect();
        let rows: Vec<(String, SessionOwner)> = self
            .inner
            .sessions
            .snapshot()
            .into_iter()
            .filter(|(id, _)| wanted.contains(id.as_str()))
            .collect();
        for (sid, owner) in rows {
            if let Some(r) = self
                .inner
                .sessions
                .release_if_handleless(&owner, &sid, |s| self.holds(s))
            {
                self.release(&r);
            }
        }
    }
}

/// Aborts the sweep task when dropped.
#[derive(Debug)]
pub struct SweepGuard(tokio::task::JoinHandle<()>);

impl Drop for SweepGuard {
    fn drop(&mut self) {
        self.0.abort();
    }
}
