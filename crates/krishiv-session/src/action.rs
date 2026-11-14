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
//! - **Before it hands over**, a cancellation ends the action at once: its
//!   future is dropped and it fails with the reason's error
//!   (`OPERATION_CANCELED` for `user`). Only analysis and planning are cut
//!   this way; they hold memory and rows, which a drop releases, as it does
//!   for a caller that goes away.
//! - **It hands over** at the first reservation of a query under it, or at a
//!   [`commit_point`], which a statement calls before its first side effect
//!   (an overwrite that drops a table or deletes the files it replaces). A
//!   reason that came before is the next reservation's pending reason
//!   (applied at `attach`); one that comes later cancels the reserved queries
//!   through the registry, so a running query ends as it always has (its
//!   stream, its row), and a side effect is not cut half-way by a cancel.
//! - **Nested actions** (a command that runs a statement of its own, such as
//!   `ANALYZE TABLE .. FOR COLUMNS` or `EXECUTE IMMEDIATE`) record their
//!   parent: a reservation or a commit point hands over the whole chain, so a
//!   cancel of the outer action reaches the inner query through the registry.
//!
//! Limit: once an action has handed over, a later statement of it (the next
//! statement of a SQL script, which reserves its row when it begins) is not
//! dropped during its own analysis; it ends at its next reservation (the
//! pending reason) or, for readers that wait on [`cancelled`], at once.
//!
//! The hand-over and a cancellation take the action's one lock, so a
//! cancellation racing a reservation is never lost. Lock order:
//! `EngineRuntime::live` -> `Action::state`, one action's at a time;
//! `Actions::live` is never held with either. The action is current in a
//! task-local while it runs; planning-time readers that hold their own
//! cancellation (a table format's log reads) wait on [`cancelled`] to end
//! with it.

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

/// The `query_id` an error names for an action the caller gave no id.
const NOT_STARTED: &str = "(not started)";

/// One running action of a session.
#[derive(Debug)]
pub(crate) struct Action {
    /// The number of the session that runs it.
    session: u64,
    /// The caller's operation id, if it gave one (the id `Session::cancel`
    /// names it by).
    operation: Option<Arc<str>>,
    /// The tags its queries carry, fixed at the start: the session's and the request's.
    tags: Arc<[(String, String)]>,
    /// The request tags among them (exact-match interrupts).
    request_tags: Arc<[Arc<str>]>,
    /// The action this one runs inside, if any (a nested `on_executor`).
    parent: Option<Arc<Action>>,
    /// Taken after `EngineRuntime::live` when both are held, never together
    /// with another action's.
    state: Mutex<State>,
    /// The first reason, published for the action's race and for planning-time readers.
    reason: watch::Sender<Option<CancelReason>>,
}

#[derive(Debug, Default)]
struct State {
    reason: Option<CancelReason>,
    /// Whether the action has handed over (a reservation or a commit point,
    /// its own or a nested action's): a cancel no longer drops it.
    handed_over: bool,
    /// The `system.queries` rows reserved under the action or a nested one.
    rows: Vec<HandedOver>,
}

impl Action {
    fn handed_over(&self) -> bool {
        self.lock().handed_over
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Hands the action and its ancestors over, recording `row` on each; the
    /// first reason any of them was already cancelled with. Each lock is taken
    /// alone, innermost first: an ancestor's cancel that comes before its lock
    /// is returned here, one that comes after finds the row.
    fn hand_over_chain(self: &Arc<Self>, row: Option<&HandedOver>) -> Option<CancelReason> {
        let mut first = None;
        let mut next = Some(self);
        while let Some(action) = next {
            let mut state = action.lock();
            state.handed_over = true;
            if let Some(row) = row {
                state.rows.push(row.clone());
            }
            first = first.or(state.reason);
            drop(state);
            next = action.parent.as_ref();
        }
        first
    }
}

/// A row reserved under an action: its session number and `system.queries` id.
pub(crate) type HandedOver = (u64, Arc<str>);

/// What [`Actions::cancel_where`] did.
#[derive(Debug, Default)]
pub(crate) struct ActionsCancelled {
    /// The operation ids of the actions it ended, or will end at their next
    /// reservation (no row reserved yet, an id given).
    pub(crate) ids: Vec<Arc<str>>,
    /// How many actions it ended, with an id or without.
    pub(crate) ended: usize,
    /// The rows handed over under every picked action, to cancel through the registry.
    pub(crate) rows: Vec<HandedOver>,
}

/// The running actions of one runtime.
#[derive(Debug, Default)]
pub(crate) struct Actions {
    next: AtomicU64,
    live: Mutex<HashMap<u64, Arc<Action>>>,
}

impl Actions {
    /// Registers an action of session `session`, named `operation` when the
    /// caller gave an id, inside the current action if there is one (so call
    /// it on the caller's task); it leaves the registry when the guard drops.
    #[cfg(test)]
    pub(crate) fn begin(self: &Arc<Self>, session: u64, operation: Option<&str>) -> ActionGuard {
        self.begin_tagged(session, operation, Arc::from([]), Arc::from([]))
    }

    /// [`Self::begin`] for an action whose queries carry `tags` (`request_tags` among them).
    pub(crate) fn begin_tagged(
        self: &Arc<Self>,
        session: u64,
        operation: Option<&str>,
        tags: Arc<[(String, String)]>,
        request_tags: Arc<[Arc<str>]>,
    ) -> ActionGuard {
        let (reason, _) = watch::channel(None);
        let action = Arc::new(Action {
            session,
            operation: operation.map(Arc::from),
            tags,
            request_tags,
            parent: CURRENT.try_with(Arc::clone).ok(),
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
    /// The first reason wins. An action that has not handed over is ended (its
    /// race drops it); one that has is left to the registry (its rows).
    pub(crate) fn cancel_where(
        &self,
        reason: CancelReason,
        pick: &impl Fn(u64, &str) -> bool,
    ) -> ActionsCancelled {
        self.cancel_matching(reason, &|a: &Action| {
            pick(a.session, a.operation.as_deref().unwrap_or(""))
        })
    }

    /// As [`Self::cancel_where`], picking by the action's tags: `pick` sees the session number,
    /// the effective tags and the request tags.
    pub(crate) fn cancel_tagged(
        &self,
        reason: CancelReason,
        pick: &impl Fn(u64, &[(String, String)], &[Arc<str>]) -> bool,
    ) -> ActionsCancelled {
        self.cancel_matching(reason, &|a: &Action| {
            pick(a.session, &a.tags, &a.request_tags)
        })
    }

    fn cancel_matching(
        &self,
        reason: CancelReason,
        pick: &impl Fn(&Action) -> bool,
    ) -> ActionsCancelled {
        let picked: Vec<Arc<Action>> = self
            .live
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .filter(|a| pick(a))
            .cloned()
            .collect();
        let mut out = ActionsCancelled::default();
        for action in picked {
            let mut state = action.lock();
            if state.reason.is_none() {
                state.reason = Some(reason);
                // Ended now, or at its next reservation (past a commit point, no
                // row yet). An action with rows is reported by the registry, through them.
                if state.rows.is_empty() {
                    out.ended += 1;
                    if let Some(id) = &action.operation {
                        out.ids.push(Arc::clone(id));
                    }
                }
                action.reason.send_replace(Some(reason));
            }
            out.rows.extend(state.rows.iter().cloned());
        }
        out
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
    /// A handle on the reason this action is cancelled with, which outlives it.
    pub(crate) fn reason(&self) -> ActionReason {
        ActionReason(Arc::clone(&self.action))
    }

    /// Runs `work` as this action: current in the task-local, and ended with
    /// the cancellation's error if one comes before it hands over.
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
                let id = action.operation.as_deref().unwrap_or(NOT_STARTED);
                return Err(r.error(id, None));
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

/// The reason an action was cancelled with ([`ActionGuard::reason`]).
#[derive(Debug)]
pub(crate) struct ActionReason(Arc<Action>);

impl ActionReason {
    /// The reason the action was ended with before it handed over (by its
    /// race, so no query row records it), if it was.
    pub(crate) fn ended_with(&self) -> Option<CancelReason> {
        let state = self.0.lock();
        state.reason.filter(|_| !state.handed_over)
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

/// Hands the current action (if it is of session `session`) and its ancestors
/// over to the reservation of row `row`: the reason one of them was already
/// cancelled with, if any, which the reservation keeps as pending.
pub(crate) fn hand_over(session: u64, row: &str) -> Option<CancelReason> {
    CURRENT
        .try_with(|action| {
            if action.session != session {
                return None;
            }
            action.hand_over_chain(Some(&(session, Arc::from(row))))
        })
        .ok()
        .flatten()
}

/// The reason the current action was cancelled with, if it was (a drop that
/// runs inside the action, such as a statement's row closing while it waited).
pub(crate) fn current_reason() -> Option<CancelReason> {
    CURRENT
        .try_with(|action| action.lock().reason)
        .ok()
        .flatten()
}

/// Called before a statement's first side effect that a drop would leave half
/// done (an overwrite dropping a table, or deleting the files it replaces):
/// from here a cancel no longer drops the action; it reaches the statement at
/// its next reservation (the pending reason) or through its query.
pub(crate) fn commit_point() {
    let _ = CURRENT.try_with(|action| action.hand_over_chain(None));
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
        let out = actions.cancel_where(CancelReason::User, &all);
        assert_eq!(out.ids, vec![Arc::<str>::from("op")]);
        assert_eq!(out.ended, 1);
        assert!(out.rows.is_empty());
        let pending = block(CURRENT.scope(action, async { hand_over(1, "row") }));
        assert_eq!(pending, Some(CancelReason::User));
    }

    #[test]
    fn a_cancel_after_the_reservation_names_its_rows() {
        let actions = Arc::new(Actions::default());
        let guard = actions.begin(1, Some("op"));
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
        let out = actions.cancel_where(CancelReason::Admin, &all);
        // Handed over: the registry reports the row, not the action.
        assert!(out.ids.is_empty());
        assert_eq!(out.ended, 0);
        assert_eq!(out.rows, vec![(1, Arc::<str>::from("row"))]);
        // The first reason wins; the rows are named again for a later cancel.
        let out = actions.cancel_where(CancelReason::User, &all);
        assert!(out.ids.is_empty());
        assert_eq!(out.rows.len(), 1);
    }

    #[test]
    fn an_action_without_an_id_is_ended_but_not_named() {
        let actions = Arc::new(Actions::default());
        let _guard = actions.begin(1, None);
        let out = actions.cancel_where(CancelReason::User, &all);
        assert!(out.ids.is_empty());
        assert_eq!(out.ended, 1);
    }

    #[test]
    fn a_nested_reservation_hands_over_its_parents() {
        let actions = Arc::new(Actions::default());
        let outer = actions.begin(1, Some("outer"));
        let a = Arc::clone(&actions);
        let out = block(outer.run(async move {
            let inner = a.begin(1, None);
            let reserved = inner.run(async { Ok::<_, KrishivError>(hand_over(1, "row")) });
            assert_eq!(reserved.await.unwrap(), None);
            // The outer action is not dropped: the inner row ends through the registry.
            let out = a.cancel_where(CancelReason::User, &all);
            tokio::task::yield_now().await;
            Ok::<_, KrishivError>(out)
        }));
        let out = out.unwrap();
        assert!(out.ids.is_empty());
        assert_eq!(out.rows, vec![(1, Arc::<str>::from("row"))]);
    }

    #[test]
    fn a_commit_point_keeps_the_action_and_its_reason_waits() {
        let actions = Arc::new(Actions::default());
        let guard = actions.begin(1, Some("op"));
        let a = Arc::clone(&actions);
        let out = block(guard.run(async move {
            commit_point();
            let cancelled = a.cancel_where(CancelReason::User, &all);
            // Named: it ends at its reservation, though it is not dropped now.
            assert_eq!(cancelled.ids, vec![Arc::<str>::from("op")]);
            tokio::task::yield_now().await;
            // The side effect finishes; the next reservation takes the reason.
            Ok::<_, KrishivError>(hand_over(1, "row"))
        }));
        assert_eq!(out.unwrap(), Some(CancelReason::User));
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
        // Work that never ends is ended by the race, with the reason's error,
        // which names a placeholder for an action without an id.
        let guard = actions.begin(1, None);
        let a = Arc::clone(&actions);
        let out = block(guard.run(async move {
            let _ = a.cancel_where(CancelReason::Drain, &all);
            std::future::pending::<Result<(), KrishivError>>().await
        }));
        let e = out.unwrap_err();
        assert_eq!(e.error_class().name(), "KRISHIV_QUERY_KILLED");
        assert!(e.to_string().contains(NOT_STARTED), "{e}");
        // Outside an action nothing is ever cancelled.
        let mut outside = pin!(cancelled());
        let waker = futures::task::noop_waker();
        let mut cx = std::task::Context::from_waker(&waker);
        assert!(outside.as_mut().poll(&mut cx).is_pending());
    }
}
