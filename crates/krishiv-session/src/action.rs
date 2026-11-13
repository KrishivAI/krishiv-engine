// SPDX-License-Identifier: Apache-2.0
//! The cancellation of a session action before its query exists (P3-T021
//! follow-up, re-review ruling R1 of PR-08b part 2).
//!
//! A query reaches the runtime's registry ([`crate::runtime::EngineRuntime::reserve`])
//! only once it is admitted, after analysis and planning. Analysis can do
//! slow I/O (file listings and footers, a table format's log, an access
//! policy), and `Session::cancel` / `Session::interrupt_all` must end that
//! too. So every action ([`crate::Session::on_executor`]) registers here for
//! as long as it runs:
//!
//! - **Before a query is reserved**, a cancellation ends the action at once:
//!   its future is dropped (every await of the pipeline is drop-safe, as a
//!   caller that goes away already relies on) and it fails with the reason's
//!   error (`OPERATION_CANCELED` for `user`).
//! - **From the first reservation on**, the action hands over: a reason that
//!   came before is the reservation's pending reason (applied at `attach`), and
//!   one that comes later cancels the reserved queries through the registry,
//!   so a running query ends as it always has (its stream, its row).
//!
//! The hand-over and a cancellation take the action's one lock, so a
//! cancellation racing a reservation is never lost. The action is current
//! in a task-local while it runs; planning-time readers that hold their own
//! cancellation (a table format's log reads) can wait on
//! [`cancelled`] to end with it.

use std::collections::HashMap;
use std::future::Future;
use std::pin::pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use futures::future::{Either, select};
use krishiv_common::error::KrishivError;
use tokio::sync::watch;

use crate::cancel::CancelReason;

tokio::task_local! {
    static CURRENT: Arc<Action>;
}

/// One running action of a session.
#[derive(Debug)]
pub(crate) struct Action {
    /// The number of the session that runs it.
    session: u64,
    /// The caller's operation id, if it gave one (the id `Session::cancel`
    /// names it by).
    operation: Option<Arc<str>>,
    state: Mutex<State>,
    /// The first reason, published for the action's race and for planning-time readers.
    reason: watch::Sender<Option<CancelReason>>,
}

#[derive(Debug, Default)]
struct State {
    reason: Option<CancelReason>,
    /// The `system.queries` rows reserved under the action (the hand-over).
    rows: Vec<Arc<str>>,
}

impl Action {
    /// Whether a query has been reserved under the action.
    fn handed_over(&self) -> bool {
        !self.lock().rows.is_empty()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A row reserved under an action: its session number and `system.queries` id.
pub(crate) type HandedOver = (u64, Arc<str>);

/// The running actions of one runtime.
#[derive(Debug, Default)]
pub(crate) struct Actions {
    next: AtomicU64,
    live: Mutex<HashMap<u64, Arc<Action>>>,
}

impl Actions {
    /// Registers an action of session `session`, named `operation` when the
    /// caller gave an id; it leaves the registry when the guard drops.
    pub(crate) fn begin(self: &Arc<Self>, session: u64, operation: Option<&str>) -> ActionGuard {
        let (reason, _) = watch::channel(None);
        let action = Arc::new(Action {
            session,
            operation: operation.map(Arc::from),
            state: Mutex::new(State::default()),
            reason,
        });
        let key = self.next.fetch_add(1, Ordering::Relaxed);
        self.live
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(key, Arc::clone(&action));
        ActionGuard {
            actions: Arc::clone(self),
            key,
            action,
        }
    }

    /// Cancels, with `reason`, every action `pick` selects (by session number
    /// and operation id; an action without an id is picked by an empty id).
    /// Returns the operation ids of the actions it cancelled first, and the
    /// rows handed over by every picked action (to be cancelled through the
    /// query registry), each as (session, row).
    pub(crate) fn cancel_where(
        &self,
        reason: CancelReason,
        pick: &impl Fn(u64, &str) -> bool,
    ) -> (Vec<Arc<str>>, Vec<HandedOver>) {
        let picked: Vec<Arc<Action>> = self
            .live
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .filter(|a| pick(a.session, a.operation.as_deref().unwrap_or("")))
            .cloned()
            .collect();
        let mut ids = Vec::new();
        let mut rows = Vec::new();
        for action in picked {
            let mut state = action.lock();
            if state.reason.is_none() {
                state.reason = Some(reason);
                if let Some(id) = &action.operation {
                    ids.push(Arc::clone(id));
                }
                action.reason.send_replace(Some(reason));
            }
            rows.extend(state.rows.iter().map(|r| (action.session, Arc::clone(r))));
        }
        (ids, rows)
    }
}

/// An action's registration; it leaves the registry on drop.
#[derive(Debug)]
pub(crate) struct ActionGuard {
    actions: Arc<Actions>,
    key: u64,
    action: Arc<Action>,
}

impl ActionGuard {
    /// Runs `work` as this action: current in the task-local, and ended with
    /// the cancellation's error if one comes before a query is reserved.
    pub(crate) async fn run<T>(
        self,
        work: impl Future<Output = Result<T, KrishivError>>,
    ) -> Result<T, KrishivError> {
        let action = Arc::clone(&self.action);
        let mut reason = action.reason.subscribe();
        let mut work = pin!(CURRENT.scope(Arc::clone(&action), work));
        loop {
            // A reason set before the first poll is seen as a change too.
            if let Some(r) = *reason.borrow_and_update()
                && !action.handed_over()
            {
                return Err(r.error(action.operation.as_deref().unwrap_or(""), None));
            }
            let changed = pin!(reason.changed());
            match select(work.as_mut(), changed).await {
                Either::Left((done, _)) => return done,
                Either::Right((Ok(()), _)) => {}
                // The sender lives in `action`, which this frame holds: not reached.
                Either::Right((Err(_), _)) => return work.await,
            }
            if action.handed_over() {
                // The registry ends the query; wait for the work alone.
                return work.await;
            }
        }
    }
}

impl Drop for ActionGuard {
    fn drop(&mut self) {
        self.actions
            .live
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&self.key);
    }
}

/// Hands the current action (if it is of session `session`) over to the
/// reservation of row `row`: the reason it was already cancelled with, if any,
/// which the reservation keeps as pending.
pub(crate) fn hand_over(session: u64, row: &str) -> Option<CancelReason> {
    CURRENT
        .try_with(|action| {
            if action.session != session {
                return None;
            }
            let mut state = action.lock();
            state.rows.push(Arc::from(row));
            state.reason
        })
        .ok()
        .flatten()
}

/// Resolves when the current action is cancelled (never, outside an action):
/// for planning-time work that holds a cancellation of its own.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the Delta planning reads wire it (PR-08b part 2)")
)]
pub(crate) fn cancelled() -> impl Future<Output = CancelReason> + Send + 'static {
    let reason = CURRENT.try_with(|a| a.reason.subscribe()).ok();
    async move {
        let Some(mut reason) = reason else {
            return std::future::pending().await;
        };
        loop {
            if let Some(r) = *reason.borrow_and_update() {
                return r;
            }
            if reason.changed().await.is_err() {
                return std::future::pending().await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(_: u64, _: &str) -> bool {
        true
    }

    fn block<T>(f: impl Future<Output = T>) -> T {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(f)
    }

    #[test]
    fn a_cancel_before_the_reservation_is_handed_over_as_pending() {
        let actions = Arc::new(Actions::default());
        let guard = actions.begin(1, Some("op"));
        let action = Arc::clone(&guard.action);
        let (ids, rows) = actions.cancel_where(CancelReason::User, &all);
        assert_eq!(ids, vec![Arc::<str>::from("op")]);
        assert!(rows.is_empty());
        let pending = block(CURRENT.scope(action, async { hand_over(1, "row") }));
        assert_eq!(pending, Some(CancelReason::User));
    }

    #[test]
    fn a_cancel_after_the_reservation_names_its_rows() {
        let actions = Arc::new(Actions::default());
        let guard = actions.begin(1, None);
        let action = Arc::clone(&guard.action);
        assert_eq!(
            block(CURRENT.scope(Arc::clone(&action), async { hand_over(1, "row") })),
            None
        );
        // Another session's reservation is not this action's.
        assert_eq!(
            block(CURRENT.scope(action, async { hand_over(2, "x") })),
            None
        );
        let (ids, rows) = actions.cancel_where(CancelReason::Admin, &all);
        assert!(
            ids.is_empty(),
            "an action without an operation id has none to report"
        );
        assert_eq!(rows, vec![(1, Arc::<str>::from("row"))]);
        // The first reason wins; the rows are named again for a later cancel.
        let (ids, rows) = actions.cancel_where(CancelReason::User, &all);
        assert!(ids.is_empty());
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn a_handed_over_action_waits_for_its_work() {
        let actions = Arc::new(Actions::default());
        let guard = actions.begin(1, Some("op"));
        let a = Arc::clone(&actions);
        let out = block(guard.run(async move {
            assert_eq!(hand_over(1, "op"), None);
            let _ = a.cancel_where(CancelReason::User, &all);
            tokio::task::yield_now().await;
            Ok::<_, KrishivError>(7)
        }));
        assert_eq!(out.unwrap(), 7);
        assert!(
            actions.live.lock().unwrap().is_empty(),
            "the guard left the registry"
        );
    }

    #[test]
    fn planning_reads_see_the_cancellation() {
        let actions = Arc::new(Actions::default());
        let guard = actions.begin(1, None);
        let a = Arc::clone(&actions);
        let out = block(guard.run(async move {
            let wait = cancelled();
            let _ = a.cancel_where(CancelReason::Drain, &all);
            Ok::<_, KrishivError>(wait.await)
        }));
        // The work is polled first and sees the reason itself.
        assert_eq!(out.unwrap(), CancelReason::Drain);
        // Work that never ends is ended by the race, with the reason's error.
        let guard = actions.begin(1, None);
        let a = Arc::clone(&actions);
        let out = block(guard.run(async move {
            let _ = a.cancel_where(CancelReason::Drain, &all);
            std::future::pending::<Result<(), KrishivError>>().await
        }));
        assert_eq!(
            out.unwrap_err().error_class().name(),
            "KRISHIV_QUERY_KILLED"
        );
        // Outside an action nothing is ever cancelled.
        let mut outside = pin!(cancelled());
        let waker = futures::task::noop_waker();
        let mut cx = std::task::Context::from_waker(&waker);
        assert!(outside.as_mut().poll(&mut cx).is_pending());
    }
}
