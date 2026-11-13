// SPDX-License-Identifier: Apache-2.0
//! Connect operations (design P3-T011a §3.1; Spark's `ExecuteHolder` and
//! the session's share of `SparkConnectExecutionManager`): one
//! [`Operation`] per `ExecutePlan`, kept in its session's [`Operations`]
//! while it streams.
//!
//! An operation id is looked up only inside the caller's session, and a
//! session is bound to its principal (ADR 0046, threat model T-A8, T-Z5):
//! another principal's id is `INVALID_HANDLE.OPERATION_NOT_FOUND`, the answer
//! an unknown id gets. Which ids were used in the last 30 minutes is the
//! engine's operation table (`Session::from_plan_with`), shared with SQL.
//!
//! An operation that leaves the map keeps its final state for
//! [`ENDED_OPERATION_WINDOW`] after its last lookup (Spark's
//! `SessionHolder.inactiveOperations`, `expireAfterAccess` of
//! `spark.connect.session.inactiveOperations.cacheExpiration`), at most
//! [`ENDED_OPERATIONS_KEPT`] of them per session: `GetStatus` answers from it.
//!
//! The map's lock guards lookups, inserts and removals only; nothing awaits
//! under it. Lock order: the live map, then the ended one.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use krishiv_common::config::{
    CONNECT_EXECUTE_AHEAD_BYTES, CONNECT_EXECUTE_DETACHED_TIMEOUT,
    CONNECT_EXECUTE_MAINTENANCE_INTERVAL, CONNECT_EXECUTE_RETRY_BUFFER_SIZE,
    CONNECT_EXECUTE_SENDER_MAX_STREAM_DURATION, CONNECT_EXECUTE_SENDER_MAX_STREAM_SIZE,
    CONNECT_MAX_OPERATIONS_PER_SESSION, CONNECT_MAX_RESPONSE_BYTES, Config, ConfigValue,
};
use krishiv_common::error::{
    InternalError, InvalidHandleOperationAlreadyExists, KrishivError, KrishivInvalidConfigValue,
    KrishivInvalidOperationTag, KrishivOperationLimit,
};
use krishiv_session::QueryId;
use sha2::{Digest, Sha256};
use tokio::sync::Notify;

pub use self::replay::Replay;

pub mod replay;

/// The limits of `ExecutePlan` (design P3-T011a §3.1, §3.2).
#[derive(Debug, Clone)]
pub struct ExecuteLimits {
    /// Live operations of one session
    /// (`krishiv.connect.maxOperationsPerSession`, 100).
    pub max_operations_per_session: usize,
    /// Encoded responses an operation holds before its client reads them
    /// (`krishiv.connect.execute.aheadBytes`, 64 MiB).
    pub ahead_bytes: usize,
    /// The largest decompressed plan (`spark.connect.maxPlanSize`, static,
    /// 512 MiB in Spark 4.1+).
    pub max_plan_size: usize,
    /// The largest Arrow batch before the 0.7 factor
    /// (`spark.connect.grpc.arrow.maxBatchSize`, static, 128 MiB).
    pub max_batch_size: usize,
    /// Bytes of sent responses a reattachable operation keeps for a
    /// reattach (`spark.connect.execute.reattachable.observerRetryBufferSize`,
    /// 10 MiB).
    pub retry_buffer_size: usize,
    /// A reattachable stream ends (OK, no `ResultComplete`) after this long;
    /// zero never (`spark.connect.execute.reattachable.senderMaxStreamDuration`,
    /// 2 min).
    pub sender_max_stream_duration: Duration,
    /// A reattachable stream ends once it sent more than this many bytes;
    /// zero never (`spark.connect.execute.reattachable.senderMaxStreamSize`,
    /// 1 GiB).
    pub sender_max_stream_size: usize,
    /// A reattachable operation with no stream attached this long is
    /// abandoned (`spark.connect.execute.manager.detachedTimeout`, 5 min).
    pub detached_timeout: Duration,
    /// How often detached operations are looked for
    /// (`spark.connect.execute.manager.maintenanceInterval`, 30 s).
    pub maintenance_interval: Duration,
    /// Response bytes all operations of the server hold at once
    /// (`krishiv.connect.maxResponseBytes`, 1 GiB).
    pub max_response_bytes: usize,
}

impl Default for ExecuteLimits {
    fn default() -> Self {
        Self {
            max_operations_per_session: 100,
            ahead_bytes: 64 * 1024 * 1024,
            max_plan_size: 512 * 1024 * 1024,
            max_batch_size: 128 * 1024 * 1024,
            retry_buffer_size: 10 * 1024 * 1024,
            sender_max_stream_duration: Duration::from_secs(120),
            sender_max_stream_size: 1024 * 1024 * 1024,
            detached_timeout: Duration::from_secs(300),
            maintenance_interval: Duration::from_secs(30),
            max_response_bytes: 1024 * 1024 * 1024,
        }
    }
}

impl ExecuteLimits {
    /// The limits `config` gives; defaults otherwise.
    #[must_use]
    pub fn from_config(config: &Config) -> Self {
        let d = Self::default();
        let size = |e| {
            config
                .get(e)
                .and_then(ConfigValue::as_size)
                .and_then(|v| usize::try_from(v).ok())
        };
        let duration = |e| config.get(e).and_then(ConfigValue::as_duration);
        Self {
            max_operations_per_session: config
                .get(&CONNECT_MAX_OPERATIONS_PER_SESSION)
                .and_then(ConfigValue::as_int)
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(d.max_operations_per_session),
            ahead_bytes: size(&CONNECT_EXECUTE_AHEAD_BYTES).unwrap_or(d.ahead_bytes),
            retry_buffer_size: size(&CONNECT_EXECUTE_RETRY_BUFFER_SIZE)
                .unwrap_or(d.retry_buffer_size),
            sender_max_stream_duration: duration(&CONNECT_EXECUTE_SENDER_MAX_STREAM_DURATION)
                .unwrap_or(d.sender_max_stream_duration),
            sender_max_stream_size: size(&CONNECT_EXECUTE_SENDER_MAX_STREAM_SIZE)
                .unwrap_or(d.sender_max_stream_size),
            detached_timeout: duration(&CONNECT_EXECUTE_DETACHED_TIMEOUT)
                .unwrap_or(d.detached_timeout),
            maintenance_interval: duration(&CONNECT_EXECUTE_MAINTENANCE_INTERVAL)
                .unwrap_or(d.maintenance_interval),
            max_response_bytes: size(&CONNECT_MAX_RESPONSE_BYTES).unwrap_or(d.max_response_bytes),
            ..d
        }
    }

    /// Checks the limits fit together: the server's response budget holds
    /// at least one operation's worst case (`aheadBytes` plus the retry
    /// buffer), so an operation can always be admitted once the others are
    /// closed.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` for a `krishiv.connect.maxResponseBytes`
    /// below `krishiv.connect.execute.aheadBytes` +
    /// `krishiv.connect.execute.retryBufferSize`.
    pub fn validate(&self) -> Result<(), KrishivError> {
        let one = self.ahead_bytes.saturating_add(self.retry_buffer_size);
        if self.max_response_bytes < one {
            return Err(KrishivInvalidConfigValue {
                value: format!("'{}'", self.max_response_bytes),
                key: format!("\"{}\"", CONNECT_MAX_RESPONSE_BYTES.key),
                reason: format!(
                    "it must hold one operation's worst case, {} + {} = {one} bytes",
                    CONNECT_EXECUTE_AHEAD_BYTES.key, CONNECT_EXECUTE_RETRY_BUFFER_SIZE.key
                ),
            }
            .into());
        }
        Ok(())
    }
}

/// The most tags one request may carry (design P3-T064 §4: a cap Spark does
/// not have).
pub const MAX_TAGS: usize = 256;

/// The longest tag, in bytes (design P3-T064 §4).
pub const MAX_TAG_BYTES: usize = 1024;

/// Spark's `ProtoUtils.throwIfInvalidTag` (v4.3.0) plus T064's caps. Spark
/// raises a plain `IllegalArgumentException` with the same text; Krishiv
/// sends it as `KRISHIV_INVALID_OPERATION_TAG` (22023), a recorded
/// difference.
///
/// # Errors
/// `KRISHIV_INVALID_OPERATION_TAG` for a tag containing `,`, an empty tag,
/// more than [`MAX_TAGS`] tags or a tag over [`MAX_TAG_BYTES`].
pub fn validate_tags(tags: &[String]) -> Result<Arc<[Arc<str>]>, KrishivError> {
    let refuse = |reason: String| -> KrishivError { KrishivInvalidOperationTag { reason }.into() };
    if tags.len() > MAX_TAGS {
        return Err(refuse(format!(
            "A request may carry at most {MAX_TAGS} Spark Connect tags, not {}.",
            tags.len()
        )));
    }
    for tag in tags {
        // Spark's order: the separator first, then emptiness.
        if tag.contains(',') {
            return Err(refuse("Spark Connect tag cannot contain ','.".to_owned()));
        }
        if tag.is_empty() {
            return Err(refuse(
                "Spark Connect tag cannot be an empty string.".to_owned(),
            ));
        }
        if tag.len() > MAX_TAG_BYTES {
            return Err(refuse(format!(
                "Spark Connect tag cannot be longer than {MAX_TAG_BYTES} bytes."
            )));
        }
    }
    Ok(tags.iter().map(|t| Arc::from(t.as_str())).collect())
}

/// A fresh random operation id (a version 4 UUID, as Spark's
/// `UUID.randomUUID`).
///
/// # Errors
/// `INTERNAL_ERROR` when the system has no random source.
pub fn new_operation_id() -> Result<QueryId, KrishivError> {
    QueryId::parse(&replay::uuid_text(random_uuid()?))
}

/// The bits of a fresh random version 4 UUID (Spark's `UUID.randomUUID`):
/// operation and response ids.
///
/// # Errors
/// `INTERNAL_ERROR` when the system has no random source.
pub fn random_uuid() -> Result<u128, KrishivError> {
    let mut b = [0_u8; 16];
    getrandom::fill(&mut b).map_err(|e| -> KrishivError {
        InternalError {
            message: format!("no random source for an id: {e}"),
        }
        .into()
    })?;
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    Ok(u128::from_be_bytes(b))
}

/// The SHA-256 of a request's `Plan` bytes: "the same plan" for an
/// `ExecutePlan` that re-sends a live operation id (Spark compares the
/// messages; one client's re-send is byte-equal).
#[must_use]
pub fn plan_digest(plan_bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(plan_bytes).into()
}

/// [`Operation`] states.
const RUNNING: u8 = 0;
const FINISHED: u8 = 1;
const CANCELLED: u8 = 2;
const FAILED: u8 = 3;

/// How long an ended operation's final state is kept after its last lookup
/// (Spark's `spark.connect.session.inactiveOperations.cacheExpiration`, 30
/// minutes, the window the engine's operation table also keeps an id taken).
pub const ENDED_OPERATION_WINDOW: Duration = Duration::from_secs(30 * 60);

/// The most ended operations one session remembers; the least recently used
/// goes first (a Krishiv bound: Spark's cache has none).
pub const ENDED_OPERATIONS_KEPT: usize = 1000;

/// What `GetStatus` reports for an operation (design P3-T011a §3.12;
/// `GetStatusResponse.OperationStatus.OperationState` without `UNKNOWN`
/// and `UNSPECIFIED`). **Unverified against Spark**: how Spark maps an
/// operation's `ExecuteStatus` and `TerminationReason`
/// (`ExecuteHolder.scala:348-355`) to these states is in its
/// `SparkConnectGetStatusHandler`, which was not read: `TERMINATING` for an
/// interrupt whose encoder still runs, `CANCELLED` for one removed while
/// running, `FAILED` for a setup failure follow the design, not Spark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationState {
    /// Running (or waiting for admission).
    Running,
    /// Interrupted; its encoder has not stopped yet.
    Terminating,
    /// Produced its whole result.
    Succeeded,
    /// Failed.
    Failed,
    /// Interrupted, released or abandoned before it finished.
    Cancelled,
}

/// One `ExecutePlan` (Spark's `ExecuteHolder`).
#[derive(Debug)]
pub struct Operation {
    id: QueryId,
    plan_digest: [u8; 32],
    tags: Arc<[Arc<str>]>,
    reattachable: bool,
    /// Its responses and attached senders (Spark's
    /// `ExecuteResponseObserver`).
    replay: Replay,
    /// [`RUNNING`], then [`FINISHED`], [`FAILED`] or [`CANCELLED`], each
    /// set once by a compare-and-swap (Spark's `ExecuteThreadRunner`
    /// state).
    state: AtomicU8,
    /// Set once its encoder stopped (or it never started one).
    ended: AtomicBool,
    cancel: Notify,
    /// Set by [`Self::close`]; wakes the encoder.
    closed: Notify,
}

impl Operation {
    /// A new operation. A reattachable one keeps 10 MiB of sent responses
    /// for a reattach ([`Self::with_retry_buffer`]); another keeps none
    /// (Spark's `ExecuteResponseObserver`, v4.3.0 lines 106-110).
    #[must_use]
    pub fn new(
        id: QueryId,
        plan_digest: [u8; 32],
        tags: Arc<[Arc<str>]>,
        reattachable: bool,
    ) -> Self {
        let retry = if reattachable {
            ExecuteLimits::default().retry_buffer_size
        } else {
            0
        };
        Self {
            id,
            plan_digest,
            tags,
            reattachable,
            replay: Replay::new(retry, reattachable),
            state: AtomicU8::new(RUNNING),
            ended: AtomicBool::new(false),
            cancel: Notify::new(),
            closed: Notify::new(),
        }
    }

    /// This operation keeping `bytes` of sent responses for a reattach
    /// (a reattachable one only).
    #[must_use]
    pub fn with_retry_buffer(mut self, bytes: usize) -> Self {
        if self.reattachable {
            self.replay = Replay::new(bytes, true);
        }
        self
    }

    /// The operation id (the engine query id of its queries).
    #[must_use]
    pub fn id(&self) -> &QueryId {
        &self.id
    }

    /// The digest of the request's plan.
    #[must_use]
    pub fn plan_digest(&self) -> &[u8; 32] {
        &self.plan_digest
    }

    /// The request's tags.
    #[must_use]
    pub fn tags(&self) -> &[Arc<str>] {
        &self.tags
    }

    /// Whether the client asked for a reattachable execution.
    #[must_use]
    pub fn reattachable(&self) -> bool {
        self.reattachable
    }

    /// Its responses and attached senders.
    #[must_use]
    pub fn replay(&self) -> &Replay {
        &self.replay
    }

    /// The most bytes of responses the operation held at once (waiting for
    /// its client or kept for a reattach); 0 before it produced any.
    #[must_use]
    pub fn buffered_peak(&self) -> usize {
        self.replay.peak()
    }

    /// How many responses this operation produced so far.
    #[must_use]
    pub fn sent_count(&self) -> usize {
        usize::try_from(self.replay.produced()).unwrap_or(usize::MAX)
    }

    /// Asks the operation to stop: its stream ends with
    /// `OPERATION_CANCELED`. Idempotent. Whether this call interrupted it
    /// (Spark's `ExecuteThreadRunner.interrupt`, one compare-and-swap of the
    /// state): `false` when it was cancelled before or had already ended
    /// ([`Self::finish`]).
    pub fn cancel(&self) -> bool {
        let won = self
            .state
            .compare_exchange(RUNNING, CANCELLED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok();
        if won {
            // `notify_one` keeps a permit when nobody waits yet.
            self.cancel.notify_one();
        }
        won
    }

    /// Ends the operation on its own (result complete, failed, or its client
    /// gone), unless it was cancelled first; a later [`Self::cancel`]
    /// interrupts nothing. Whether it is finished: `false` when a cancel won.
    pub fn finish(&self) -> bool {
        match self
            .state
            .compare_exchange(RUNNING, FINISHED, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => true,
            Err(now) => now == FINISHED,
        }
    }

    /// Ends the operation as failed, unless it was cancelled or finished
    /// first. Whether it is failed: `false` when a cancel or the finish won.
    pub fn fail(&self) -> bool {
        match self
            .state
            .compare_exchange(RUNNING, FAILED, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => true,
            Err(now) => now == FAILED,
        }
    }

    /// Records that the operation's encoder stopped: an interrupted
    /// operation is `CANCELLED` from now on rather than `TERMINATING`.
    pub fn mark_ended(&self) {
        self.ended.store(true, Ordering::Release);
    }

    /// The operation's state for `GetStatus`.
    #[must_use]
    pub fn state(&self) -> OperationState {
        match self.state.load(Ordering::Acquire) {
            FINISHED => OperationState::Succeeded,
            FAILED => OperationState::Failed,
            CANCELLED if self.ended.load(Ordering::Acquire) => OperationState::Cancelled,
            CANCELLED => OperationState::Terminating,
            _ => OperationState::Running,
        }
    }

    /// The state it is remembered with once it left its session's map.
    /// Every path that removes an operation closes it (release, abandon,
    /// a closed stream, a released session), which interrupts one still
    /// running: running or terminating, it ends cancelled.
    fn final_state(&self) -> OperationState {
        match self.state() {
            OperationState::Running | OperationState::Terminating => OperationState::Cancelled,
            s => s,
        }
    }

    /// Whether [`Self::cancel`] interrupted the operation.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.state.load(Ordering::Acquire) == CANCELLED
    }

    /// Resolves once [`Self::cancel`] runs (one waiter: the operation's
    /// encoder).
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        self.cancel.notified().await;
    }

    /// Closes the operation (Spark's `ExecuteHolder.close`, v4.3.0 lines
    /// 283-299; ReleaseExecute `release_all`, abandonment, a released
    /// session): interrupts it if it still runs, ends every attached stream
    /// with `INVALID_CURSOR.DISCONNECTED` and drops its buffered responses
    /// (their memory and budgets return). Idempotent.
    pub fn close(&self) {
        self.cancel();
        self.replay.close();
        self.closed.notify_one();
    }

    /// Resolves once [`Self::close`] ran (one waiter: the encoder).
    pub async fn closed(&self) {
        if self.replay.is_closed() {
            return;
        }
        self.closed.notified().await;
    }
}

/// What [`Operations::register`] did.
#[derive(Debug)]
pub enum Admitted {
    /// The operation was registered.
    New,
    /// A live operation holds the id with the same plan: the request
    /// attaches to it from the start (Spark's
    /// `SparkConnectExecutePlanHandler`, v4.3.0 lines 52-55).
    Existing(Arc<Operation>),
}

/// The final states of a session's ended operations: least recently used
/// first, each with the time of its last use.
#[derive(Debug, Default)]
struct Ended {
    by_id: HashMap<QueryId, (u64, Instant, OperationState)>,
    order: BTreeMap<u64, QueryId>,
    next: u64,
}

impl Ended {
    /// Drops the entries unused for [`ENDED_OPERATION_WINDOW`] at `now`
    /// (the order is the order of use, so they are at the front).
    fn prune(&mut self, now: Instant) {
        while let Some((&seq, id)) = self.order.first_key_value() {
            let expired = self.by_id.get(id).is_none_or(|(s, at, _)| {
                *s != seq || now.saturating_duration_since(*at) >= ENDED_OPERATION_WINDOW
            });
            if !expired {
                break;
            }
            if let Some(id) = self.order.remove(&seq)
                && self.by_id.get(&id).is_some_and(|(s, _, _)| *s == seq)
            {
                self.by_id.remove(&id);
            }
        }
    }

    fn put(&mut self, id: QueryId, state: OperationState, now: Instant) {
        if let Some((old, _, _)) = self.by_id.remove(&id) {
            self.order.remove(&old);
        }
        self.next += 1;
        self.order.insert(self.next, id.clone());
        self.by_id.insert(id, (self.next, now, state));
        while self.by_id.len() > ENDED_OPERATIONS_KEPT {
            let Some((_, oldest)) = self.order.pop_first() else {
                break;
            };
            self.by_id.remove(&oldest);
        }
    }

    /// The state of `id`, refreshed (a lookup is a use).
    fn get(&mut self, id: &QueryId, now: Instant) -> Option<OperationState> {
        self.prune(now);
        let state = self.by_id.get(id)?.2;
        self.put(id.clone(), state, now);
        Some(state)
    }
}

/// The live operations of one session, and the final states of the ones
/// that ended.
#[derive(Debug, Default)]
pub struct Operations {
    live: Mutex<HashMap<QueryId, Arc<Operation>>>,
    ended: Mutex<Ended>,
}

impl Operations {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<QueryId, Arc<Operation>>> {
        self.live.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn lock_ended(&self) -> std::sync::MutexGuard<'_, Ended> {
        self.ended.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The state of operation `id` (any text; one that is not a UUID is
    /// unknown): live, else ended within the window; `None` when unknown.
    #[must_use]
    pub fn state_of(&self, id: &str) -> Option<OperationState> {
        self.state_of_at(id, Instant::now())
    }

    fn state_of_at(&self, id: &str, now: Instant) -> Option<OperationState> {
        let id = QueryId::parse(id).ok()?;
        // The live map first: an operation is recorded as ended while the
        // live lock is held, so it is always in one of the two.
        let live = self.lock().get(&id).map(Arc::clone);
        match live {
            Some(op) => Some(op.state()),
            None => self.lock_ended().get(&id, now),
        }
    }

    /// Every operation the session knows: the live ones and those ended
    /// within the window, each once.
    #[must_use]
    pub fn known(&self) -> Vec<(QueryId, OperationState)> {
        self.known_at(Instant::now())
    }

    fn known_at(&self, now: Instant) -> Vec<(QueryId, OperationState)> {
        let mut out: Vec<(QueryId, OperationState)> = self
            .lock()
            .values()
            .map(|op| (op.id().clone(), op.state()))
            .collect();
        let mut ended = self.lock_ended();
        ended.prune(now);
        let live: std::collections::HashSet<QueryId> =
            out.iter().map(|(id, _)| id.clone()).collect();
        out.extend(
            ended
                .order
                .values()
                .filter(|id| !live.contains(*id))
                .filter_map(|id| ended.by_id.get(id).map(|(_, _, s)| (id.clone(), *s))),
        );
        out
    }

    /// Records `op` as ended (called with the live lock held).
    fn record_ended(&self, op: &Operation) {
        let now = Instant::now();
        let mut ended = self.lock_ended();
        ended.prune(now);
        ended.put(op.id().clone(), op.final_state(), now);
    }

    /// Registers `op`, or finds the live operation of its id running the
    /// same plan ([`Admitted::Existing`]).
    ///
    /// # Errors
    /// `INVALID_HANDLE.OPERATION_ALREADY_EXISTS` for an id a live operation
    /// holds with another plan; `KRISHIV_OPERATION_LIMIT` past `max` live
    /// operations.
    pub fn register(&self, op: &Arc<Operation>, max: usize) -> Result<Admitted, KrishivError> {
        let mut live = self.lock();
        if let Some(existing) = live.get(op.id()) {
            return if existing.plan_digest() == op.plan_digest() {
                Ok(Admitted::Existing(Arc::clone(existing)))
            } else {
                Err(InvalidHandleOperationAlreadyExists {
                    handle: op.id().as_str().to_owned(),
                }
                .into())
            };
        }
        if live.len() >= max {
            return Err(KrishivOperationLimit {
                limit_name: "krishiv.connect.maxOperationsPerSession".to_owned(),
                limit: max.to_string(),
            }
            .into());
        }
        live.insert(op.id().clone(), Arc::clone(op));
        Ok(Admitted::New)
    }

    /// The live operation `id` (any text; an id that is not a UUID is simply
    /// unknown, as Spark does not check it here).
    #[must_use]
    pub fn get(&self, id: &str) -> Option<Arc<Operation>> {
        let id = QueryId::parse(id).ok()?;
        self.lock().get(&id).map(Arc::clone)
    }

    /// Removes `op` if it is still the registered operation of its id.
    pub fn remove(&self, op: &Arc<Operation>) -> bool {
        let mut live = self.lock();
        if live.get(op.id()).is_some_and(|o| Arc::ptr_eq(o, op)) {
            live.remove(op.id());
            self.record_ended(op);
            true
        } else {
            false
        }
    }

    /// Removes and closes every live operation (the session is released;
    /// Spark's `removeAllExecutionsForSession`). Returns them.
    pub fn close_all(&self) -> Vec<Arc<Operation>> {
        let ops: Vec<Arc<Operation>> = {
            let mut live = self.lock();
            let ops: Vec<Arc<Operation>> = live.drain().map(|(_, op)| op).collect();
            for op in &ops {
                self.record_ended(op);
            }
            ops
        };
        for op in &ops {
            op.close();
        }
        ops
    }

    /// Every live operation.
    #[must_use]
    pub fn all(&self) -> Vec<Arc<Operation>> {
        self.lock().values().map(Arc::clone).collect()
    }

    /// The number of live operations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether no operation is live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }
}

#[cfg(test)]
mod tests;
