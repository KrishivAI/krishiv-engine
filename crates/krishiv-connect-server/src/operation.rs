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
//! The map's lock guards lookups, inserts and removals only; nothing awaits
//! under it.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use krishiv_common::config::{
    CONNECT_EXECUTE_AHEAD_BYTES, CONNECT_MAX_OPERATIONS_PER_SESSION, Config, ConfigValue,
};
use krishiv_common::error::{
    InternalError, InvalidCursorDisconnected, InvalidHandleOperationAlreadyExists, KrishivError,
    KrishivInvalidOperationTag, KrishivOperationLimit,
};
use krishiv_memory::ConsumerHandle;
use krishiv_session::QueryId;
use sha2::{Digest, Sha256};
use tokio::sync::Notify;

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
}

impl Default for ExecuteLimits {
    fn default() -> Self {
        Self {
            max_operations_per_session: 100,
            ahead_bytes: 64 * 1024 * 1024,
            max_plan_size: 512 * 1024 * 1024,
            max_batch_size: 128 * 1024 * 1024,
        }
    }
}

impl ExecuteLimits {
    /// The limits `config` gives; defaults otherwise.
    #[must_use]
    pub fn from_config(config: &Config) -> Self {
        let d = Self::default();
        Self {
            max_operations_per_session: config
                .get(&CONNECT_MAX_OPERATIONS_PER_SESSION)
                .and_then(ConfigValue::as_int)
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(d.max_operations_per_session),
            ahead_bytes: config
                .get(&CONNECT_EXECUTE_AHEAD_BYTES)
                .and_then(ConfigValue::as_size)
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(d.ahead_bytes),
            ..d
        }
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
    let mut b = [0_u8; 16];
    getrandom::fill(&mut b).map_err(|e| -> KrishivError {
        InternalError {
            message: format!("no random source for an operation id: {e}"),
        }
        .into()
    })?;
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
    QueryId::parse(&format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    ))
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

/// One `ExecutePlan` (Spark's `ExecuteHolder`).
#[derive(Debug)]
pub struct Operation {
    id: QueryId,
    plan_digest: [u8; 32],
    tags: Arc<[Arc<str>]>,
    reattachable: bool,
    /// The response ids sent so far (`ReleaseExecute.release_until` accepts
    /// only these).
    sent: Mutex<HashSet<String>>,
    /// [`RUNNING`], then [`FINISHED`] or [`CANCELLED`], each set once by a
    /// compare-and-swap (Spark's `ExecuteThreadRunner` state).
    state: AtomicU8,
    /// Whether the encoder waits for the client ([`Self::is_waiting_for_client`]).
    waiting: AtomicBool,
    /// Set by [`Self::close`]: the encoder stops even after the operation
    /// finished (a release while the last responses wait for the client).
    closed: AtomicBool,
    cancel: Notify,
    /// The memory consumer its buffered responses are reserved from.
    responses: OnceLock<Arc<ConsumerHandle>>,
}

impl Operation {
    /// A new operation.
    #[must_use]
    pub fn new(
        id: QueryId,
        plan_digest: [u8; 32],
        tags: Arc<[Arc<str>]>,
        reattachable: bool,
    ) -> Self {
        Self {
            id,
            plan_digest,
            tags,
            reattachable,
            sent: Mutex::new(HashSet::new()),
            state: AtomicU8::new(RUNNING),
            waiting: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            cancel: Notify::new(),
            responses: OnceLock::new(),
        }
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

    /// Notes that the response `response_id` was produced.
    pub fn record_sent(&self, response_id: &str) {
        self.sent
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(response_id.to_owned());
    }

    /// Whether this operation produced the response `response_id`.
    #[must_use]
    pub fn has_sent(&self, response_id: &str) -> bool {
        self.sent
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .contains(response_id)
    }

    /// Sets the memory consumer of the operation's buffered responses (once).
    pub fn set_responses_consumer(&self, consumer: &Arc<ConsumerHandle>) {
        let _ = self.responses.set(Arc::clone(consumer));
    }

    /// The most bytes of responses the operation held at once, waiting for
    /// its client (`connect_responses`); 0 before it produced any.
    #[must_use]
    pub fn buffered_peak(&self) -> usize {
        self.responses.get().map_or(0, |c| c.peak())
    }

    /// How many responses this operation produced so far.
    #[must_use]
    pub fn sent_count(&self) -> usize {
        self.sent
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Whether the operation's encoder is waiting for its client to read:
    /// its responses fill the ahead budget or the channel (backpressure).
    #[must_use]
    pub fn is_waiting_for_client(&self) -> bool {
        self.waiting.load(Ordering::Acquire)
    }

    /// Marks the encoder as waiting for its client until the guard drops
    /// (also when the wait is cancelled).
    pub(crate) fn waiting_for_client(&self) -> WaitingForClient<'_> {
        self.waiting.store(true, Ordering::Release);
        WaitingForClient(self)
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

    /// Whether [`Self::cancel`] interrupted the operation.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.state.load(Ordering::Acquire) == CANCELLED
    }

    /// Closes the operation (ReleaseExecute `release_all`, a released
    /// session): interrupts it if it still runs ([`Self::cancel`]) and, in
    /// every state, stops its encoder, so a finished operation whose last
    /// responses wait for a client that reads nothing gives back its task
    /// and memory too.
    pub fn close(&self) {
        self.cancel();
        self.closed.store(true, Ordering::Release);
        self.cancel.notify_one();
    }

    /// Resolves once [`Self::cancel`] or [`Self::close`] runs (one waiter:
    /// the operation's encoder).
    pub async fn cancelled(&self) {
        if self.is_cancelled() || self.closed.load(Ordering::Acquire) {
            return;
        }
        self.cancel.notified().await;
    }
}

/// The encoder of an operation waits for its client while this lives
/// ([`Operation::waiting_for_client`]).
pub(crate) struct WaitingForClient<'a>(&'a Operation);

impl Drop for WaitingForClient<'_> {
    fn drop(&mut self) {
        self.0.waiting.store(false, Ordering::Release);
    }
}

/// The live operations of one session.
#[derive(Debug, Default)]
pub struct Operations {
    live: Mutex<HashMap<QueryId, Arc<Operation>>>,
}

impl Operations {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<QueryId, Arc<Operation>>> {
        self.live.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Registers `op`.
    ///
    /// # Errors
    /// For an id a live operation holds: `INVALID_CURSOR.DISCONNECTED` when
    /// it runs the same plan (Spark would reattach the new stream to it from
    /// the start; PR-11 has no replay buffer, a recorded difference) and
    /// `INVALID_HANDLE.OPERATION_ALREADY_EXISTS` for another plan;
    /// `KRISHIV_OPERATION_LIMIT` past `max` live operations.
    pub fn register(&self, op: &Arc<Operation>, max: usize) -> Result<(), KrishivError> {
        let mut live = self.lock();
        if let Some(existing) = live.get(op.id()) {
            return Err(if existing.plan_digest() == op.plan_digest() {
                InvalidCursorDisconnected {}.into()
            } else {
                InvalidHandleOperationAlreadyExists {
                    handle: op.id().as_str().to_owned(),
                }
                .into()
            });
        }
        if live.len() >= max {
            return Err(KrishivOperationLimit {
                limit_name: "krishiv.connect.maxOperationsPerSession".to_owned(),
                limit: max.to_string(),
            }
            .into());
        }
        live.insert(op.id().clone(), Arc::clone(op));
        Ok(())
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
            true
        } else {
            false
        }
    }

    /// Cancels and removes every live operation (the session is released).
    /// Returns how many.
    pub fn cancel_all(&self) -> usize {
        let ops: Vec<Arc<Operation>> = self.lock().drain().map(|(_, op)| op).collect();
        for op in &ops {
            op.close();
        }
        ops.len()
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
