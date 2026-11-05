// SPDX-License-Identifier: Apache-2.0
//! Why a query was cancelled, and the statement timeout (design P2-E1 §4):
//! [`CancelReason`], the error each reason ends the query's stream with,
//! and [`Deadlines`], one timer task per engine runtime that cancels
//! queries past `krishiv.query.maxExecutionTime`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::{Duration, Instant};

use krishiv_common::error::{
    KrishivError, KrishivQueryKilled, KrishivQueryTimeout, OperationCanceled,
};
use krishiv_exec::{Executor, QueryContext, TaskHandle, TokioClock, WorkloadClass};
use krishiv_memory::Clock as _;
use tokio::sync::Notify;

use crate::history::QueryHistory;

/// Why a query was cancelled (`system.queries.cancel_reason`). The first
/// reason recorded wins; later cancellations of a cancelled query are
/// no-ops.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CancelReason {
    /// The client asked ([`Session::cancel`](crate::Session::cancel),
    /// [`Session::interrupt_all`](crate::Session::interrupt_all)):
    /// `OPERATION_CANCELED`.
    User,
    /// The query ran past its time limit: `KRISHIV_QUERY_TIMEOUT`.
    Timeout,
    /// An operator stopped it: `KRISHIV_QUERY_KILLED`.
    Admin,
    /// A quota stopped it: `KRISHIV_QUERY_KILLED` until quotas have a class
    /// of their own.
    Quota,
    /// The server is draining: `KRISHIV_QUERY_KILLED`.
    Drain,
    /// The client's session closed ([`Session::close`](crate::Session::close)):
    /// `OPERATION_CANCELED`.
    ClientGone,
}

impl CancelReason {
    /// The reason as `system.queries.cancel_reason` shows it.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Timeout => "timeout",
            Self::Admin => "admin",
            Self::Quota => "quota",
            Self::Drain => "drain",
            Self::ClientGone => "client_gone",
        }
    }

    /// The error the query's stream ends with for this reason. `limit` is
    /// the time limit a timeout passed.
    pub(crate) fn error(self, query_id: &str, limit: Option<Duration>) -> KrishivError {
        let killed = |why: &str| -> KrishivError {
            KrishivQueryKilled {
                query_id: query_id.to_owned(),
                reason: why.to_owned(),
            }
            .into()
        };
        match self {
            Self::User | Self::ClientGone => OperationCanceled {}.into(),
            Self::Timeout => KrishivQueryTimeout {
                query_id: query_id.to_owned(),
                limit: limit.map_or_else(|| "its time limit".to_owned(), |d| format!("{d:?}")),
            }
            .into(),
            Self::Admin => killed("an operator cancelled it"),
            Self::Quota => killed("it exceeded a quota"),
            Self::Drain => killed("the server is shutting down"),
        }
    }
}

impl fmt::Display for CancelReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Cancels `query` (row `id` of session `session` in `history`) with
/// `reason`; whether this call cancelled it (`false` once it finished,
/// failed or was cancelled before).
pub(crate) fn apply(
    query: &QueryContext,
    reason: CancelReason,
    limit: Option<Duration>,
    history: &QueryHistory,
    session: &str,
    id: &str,
) -> bool {
    if !query.cancel_with(reason.error(id, limit)) {
        return false;
    }
    history.update(session, id, |row| {
        if row.cancel_reason.is_none() {
            row.cancel_reason = Some(reason.as_str());
        }
    });
    true
}

/// A query's deadline.
struct Deadline {
    at: Instant,
    /// Insertion order: equal instants keep their order.
    seq: u64,
    query: Weak<QueryContext>,
    limit: Duration,
    session: Arc<str>,
    id: Arc<str>,
}

impl PartialEq for Deadline {
    fn eq(&self, other: &Self) -> bool {
        (self.at, self.seq) == (other.at, other.seq)
    }
}

impl Eq for Deadline {}

impl PartialOrd for Deadline {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Deadline {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.at, self.seq).cmp(&(other.at, other.seq))
    }
}

/// The deadlines of one runtime's queries and the one task that fires them:
/// a min-heap of `(instant, query)` held weakly, so a deadline never keeps a
/// query (or the runtime) alive. A new deadline earlier than the heap's head
/// wakes the task; a query that ended is skipped when its deadline comes.
/// The task is aborted when the runtime drops.
pub(crate) struct Deadlines {
    shared: Arc<Shared>,
    task: TaskHandle,
}

struct Shared {
    heap: Mutex<BinaryHeap<Reverse<Deadline>>>,
    wake: Notify,
    seq: AtomicU64,
    history: Arc<QueryHistory>,
}

impl fmt::Debug for Deadlines {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pending = self
            .shared
            .heap
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len();
        f.debug_struct("Deadlines")
            .field("pending", &pending)
            .finish_non_exhaustive()
    }
}

impl Deadlines {
    /// Starts the timer task on `executor`.
    pub(crate) fn start(executor: &dyn Executor, history: Arc<QueryHistory>) -> Self {
        let shared = Arc::new(Shared {
            heap: Mutex::new(BinaryHeap::new()),
            wake: Notify::new(),
            seq: AtomicU64::new(0),
            history,
        });
        let task = executor.spawn(
            WorkloadClass::Background,
            Box::pin(run(Arc::downgrade(&shared))),
        );
        Self { shared, task }
    }

    /// Cancels `query` with `KRISHIV_QUERY_TIMEOUT` once `limit` has passed
    /// from now, unless it ended first.
    pub(crate) fn add(&self, query: &Arc<QueryContext>, limit: Duration, session: &str, id: &str) {
        let at = TokioClock.now() + limit;
        let deadline = Deadline {
            at,
            seq: self.shared.seq.fetch_add(1, Ordering::Relaxed),
            query: Arc::downgrade(query),
            limit,
            session: Arc::from(session),
            id: Arc::from(id),
        };
        let mut heap = self
            .shared
            .heap
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let earliest = heap.peek().is_none_or(|Reverse(head)| at < head.at);
        heap.push(Reverse(deadline));
        drop(heap);
        if earliest {
            // A stored permit if the task is not waiting yet: it re-reads
            // the heap before it sleeps again.
            self.shared.wake.notify_one();
        }
    }
}

impl Drop for Deadlines {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// The timer task: holds only a weak handle, so it never keeps the runtime's
/// deadlines alive.
async fn run(shared: Weak<Shared>) {
    loop {
        let Some(s) = shared.upgrade() else {
            return;
        };
        let now = TokioClock.now();
        let mut due = Vec::new();
        let next = {
            let mut heap = s.heap.lock().unwrap_or_else(PoisonError::into_inner);
            while heap.peek().is_some_and(|Reverse(d)| d.at <= now) {
                if let Some(Reverse(d)) = heap.pop() {
                    due.push(d);
                }
            }
            heap.peek().map(|Reverse(d)| d.at)
        };
        for d in due {
            if let Some(query) = d.query.upgrade() {
                apply(
                    &query,
                    CancelReason::Timeout,
                    Some(d.limit),
                    &s.history,
                    &d.session,
                    &d.id,
                );
            }
        }
        let wake = Arc::clone(&s);
        drop(s);
        let notified = wake.wake.notified();
        match next.map(|at| TokioClock.sleep_until(at)) {
            Some(Ok(sleep)) => {
                futures::future::select(Box::pin(notified), sleep).await;
            }
            // No deadline (or no timer to sleep on): wait for one to arrive.
            Some(Err(_)) | None => notified.await,
        }
        drop(wake);
    }
}
