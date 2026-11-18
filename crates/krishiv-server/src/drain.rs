// SPDX-License-Identifier: Apache-2.0
//! The drain state machine (design P3-T066 §5.1): `SERVING` → `DRAINING` → `DRAINED`, and back to
//! `SERVING` by `resume` while the process is not exiting.
//!
//! The controller owns no queries. A driver loop reports how much work is still held (running or
//! queued queries, results a client has not fetched) to [`DrainController::step`] and does what it
//! answers: wait, cancel the rest (`CancelReason::Drain`), or stop. Time is passed in, so the rules
//! are tested without a clock. The serving flag is the one `/readyz` reads, so it flips first.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime};

/// How long cancelled queries get to stop before the state becomes `DRAINED` anyway.
pub const STOP_BOUND: Duration = Duration::from_secs(10);

/// The longest grace period a drain takes (design P3-T066 §6.5); longer ones are clamped here and
/// the admin API refuses them.
pub const MAX_GRACE: Duration = Duration::from_secs(24 * 60 * 60);

/// The shortest grace period the admin API accepts.
pub const MIN_GRACE: Duration = Duration::from_secs(1);

/// Where the server is in a drain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainState {
    /// Taking new work.
    Serving,
    /// Refusing new work, letting held work finish until the deadline.
    Draining,
    /// Nothing held (or the stop bound passed).
    Drained,
}

/// What the driver loop does next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainStep {
    /// Keep waiting.
    Wait,
    /// The deadline passed: cancel every remaining query and release held results.
    CancelRemaining,
    /// The state is now `DRAINED`; with `exit` the process returns from `run_server`.
    Drained {
        /// Whether `exit_when_drained` is on.
        exit: bool,
    },
}

/// A snapshot for `GetDrainStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrainStatus {
    /// The state.
    pub state: DrainState,
    /// When held work is cancelled, while draining.
    pub deadline: Option<Instant>,
    /// The same deadline as a wall-clock time, fixed when it was set (it does not drift between
    /// two status calls).
    pub deadline_at: Option<SystemTime>,
    /// Whether the process exits once drained.
    pub exit_when_drained: bool,
}

/// `resume` was refused: the process is going to exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exiting;

#[derive(Debug)]
struct Inner {
    state: DrainState,
    deadline: Option<Instant>,
    deadline_at: Option<SystemTime>,
    exit_when_drained: bool,
    /// When `CancelRemaining` was answered; the stop bound counts from here.
    cancelled_at: Option<Instant>,
}

/// The drain state of one server process.
#[derive(Debug)]
pub struct DrainController {
    serving: Arc<AtomicBool>,
    /// True while new work is refused; the flag the Connect service reads.
    refusing: Arc<AtomicBool>,
    inner: Mutex<Inner>,
}

impl DrainController {
    /// A controller in `SERVING`, flipping `serving` (the flag `/readyz` reads).
    #[must_use]
    pub fn new(serving: Arc<AtomicBool>) -> Self {
        Self {
            serving,
            refusing: Arc::new(AtomicBool::new(false)),
            inner: Mutex::new(Inner {
                state: DrainState::Serving,
                deadline: None,
                deadline_at: None,
                exit_when_drained: false,
                cancelled_at: None,
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Starts a drain, or adjusts a running one: a second call may shorten the deadline, never
    /// extend it, and may turn `exit_when_drained` on, never off. Returns the status either way, so
    /// a retry is safe.
    pub fn start(&self, now: Instant, grace: Duration, exit_when_drained: bool) -> DrainStatus {
        let mut s = self.lock();
        let grace = grace.min(MAX_GRACE);
        let deadline = now.checked_add(grace).unwrap_or(now);
        let deadline_at = SystemTime::now()
            .checked_add(grace)
            .unwrap_or_else(SystemTime::now);
        if s.state == DrainState::Serving {
            // Readiness flips before anything is refused.
            self.serving.store(false, Ordering::Release);
            self.refusing.store(true, Ordering::Release);
            s.state = DrainState::Draining;
            s.deadline = Some(deadline);
            s.deadline_at = Some(deadline_at);
            s.cancelled_at = None;
        } else if s.state == DrainState::Draining && s.deadline.is_none_or(|d| deadline < d) {
            s.deadline = Some(deadline);
            s.deadline_at = Some(deadline_at);
        }
        s.exit_when_drained |= exit_when_drained;
        Self::status_of(&s)
    }

    /// Back to `SERVING` from `DRAINING` or `DRAINED`.
    ///
    /// # Errors
    /// [`Exiting`] when `exit_when_drained` is on (after SIGTERM too): the orchestrator is going to
    /// stop the process anyway.
    pub fn resume(&self) -> Result<DrainStatus, Exiting> {
        let mut s = self.lock();
        if s.state != DrainState::Serving {
            if s.exit_when_drained {
                return Err(Exiting);
            }
            s.state = DrainState::Serving;
            s.deadline = None;
            s.deadline_at = None;
            s.cancelled_at = None;
            self.serving.store(true, Ordering::Release);
            self.refusing.store(false, Ordering::Release);
        }
        Ok(Self::status_of(&s))
    }

    /// The flag that is true while new work is refused, for surfaces that read a flag (the
    /// Connect service).
    #[must_use]
    pub fn refusing_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.refusing)
    }

    /// Whether new work is refused (`KRISHIV_SERVER_DRAINING`).
    #[must_use]
    pub fn refuses_new_work(&self) -> bool {
        self.lock().state != DrainState::Serving
    }

    /// The current status.
    #[must_use]
    pub fn status(&self) -> DrainStatus {
        Self::status_of(&self.lock())
    }

    fn status_of(s: &Inner) -> DrainStatus {
        DrainStatus {
            state: s.state,
            deadline: s.deadline,
            deadline_at: s.deadline_at,
            exit_when_drained: s.exit_when_drained,
        }
    }

    /// One turn of the driver loop. `held` counts the running and queued queries plus the results
    /// a client has not fetched.
    pub fn step(&self, now: Instant, held: usize) -> DrainStep {
        let mut s = self.lock();
        match s.state {
            DrainState::Serving => return DrainStep::Wait,
            // Asked again once drained (a SIGTERM after an admin `Drain`): the answer still says
            // whether to exit, so a late `exit_when_drained` is never missed.
            DrainState::Drained => {
                return DrainStep::Drained {
                    exit: s.exit_when_drained,
                };
            }
            DrainState::Draining => {}
        }
        let done = held == 0
            || s.cancelled_at
                .is_some_and(|at| now.saturating_duration_since(at) >= STOP_BOUND);
        if done {
            s.state = DrainState::Drained;
            return DrainStep::Drained {
                exit: s.exit_when_drained,
            };
        }
        if s.cancelled_at.is_none() && s.deadline.is_some_and(|d| now >= d) {
            s.cancelled_at = Some(now);
            return DrainStep::CancelRemaining;
        }
        DrainStep::Wait
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controller() -> (DrainController, Arc<AtomicBool>, Instant) {
        let flag = Arc::new(AtomicBool::new(true));
        (
            DrainController::new(Arc::clone(&flag)),
            flag,
            Instant::now(),
        )
    }

    const S: fn(u64) -> Duration = Duration::from_secs;

    #[test]
    fn start_flips_readiness_and_refuses_new_work() {
        let (c, flag, t) = controller();
        assert!(!c.refuses_new_work());
        let st = c.start(t, S(30), false);
        assert_eq!(st.state, DrainState::Draining);
        assert!(!flag.load(Ordering::SeqCst));
        assert!(c.refuses_new_work());
        assert!(c.refusing_flag().load(Ordering::SeqCst));
        c.resume().unwrap();
        assert!(!c.refusing_flag().load(Ordering::SeqCst));
    }

    #[test]
    fn a_second_drain_shortens_the_deadline_and_turns_exit_on_only() {
        let (c, _, t) = controller();
        c.start(t, S(30), false);
        assert_eq!(c.start(t, S(60), false).deadline, Some(t + S(30)));
        let st = c.start(t, S(5), true);
        assert_eq!(st.deadline, Some(t + S(5)));
        assert!(st.exit_when_drained);
        assert!(
            c.start(t, S(5), false).exit_when_drained,
            "never turned off"
        );
    }

    #[test]
    fn nothing_held_drains_at_once() {
        let (c, _, t) = controller();
        c.start(t, S(30), true);
        assert_eq!(c.step(t, 0), DrainStep::Drained { exit: true });
        assert_eq!(c.status().state, DrainState::Drained);
    }

    #[test]
    fn held_work_waits_then_is_cancelled_at_the_deadline_then_drained_after_the_stop_bound() {
        let (c, _, t) = controller();
        c.start(t, S(30), false);
        assert_eq!(c.step(t + S(29), 2), DrainStep::Wait);
        assert_eq!(c.step(t + S(30), 2), DrainStep::CancelRemaining);
        assert_eq!(c.step(t + S(35), 1), DrainStep::Wait);
        assert_eq!(c.step(t + S(40), 1), DrainStep::Drained { exit: false });
    }

    #[test]
    fn cancelled_work_that_stops_drains_before_the_bound() {
        let (c, _, t) = controller();
        c.start(t, S(1), false);
        assert_eq!(c.step(t + S(1), 3), DrainStep::CancelRemaining);
        assert_eq!(c.step(t + S(2), 0), DrainStep::Drained { exit: false });
    }

    #[test]
    fn resume_returns_to_serving_unless_the_process_is_exiting() {
        let (c, flag, t) = controller();
        c.start(t, S(30), false);
        assert!(c.resume().is_ok());
        assert!(flag.load(Ordering::SeqCst));
        assert!(!c.refuses_new_work());
        c.start(t, S(30), true);
        assert_eq!(c.resume().unwrap_err(), Exiting);
        assert!(c.refuses_new_work());
    }

    #[test]
    fn resume_while_serving_is_a_no_op() {
        let (c, _, _) = controller();
        assert_eq!(c.resume().unwrap().state, DrainState::Serving);
    }

    #[test]
    fn a_drained_server_still_answers_exit_to_a_later_drain() {
        let (c, _, t) = controller();
        c.start(t, S(30), false);
        assert_eq!(c.step(t, 0), DrainStep::Drained { exit: false });
        c.start(t, S(30), true);
        assert_eq!(c.step(t, 0), DrainStep::Drained { exit: true });
    }

    #[test]
    fn a_huge_grace_period_is_clamped_not_a_panic() {
        let (c, _, t) = controller();
        let st = c.start(t, Duration::MAX, false);
        assert_eq!(st.deadline, Some(t + MAX_GRACE));
    }

    #[test]
    fn the_wall_clock_deadline_is_fixed_when_set() {
        let (c, _, t) = controller();
        let first = c.start(t, S(60), false).deadline_at;
        let second = c.start(t, S(600), false).deadline_at;
        assert_eq!(first, second);
        assert!(first.is_some());
    }

    #[test]
    fn a_drained_server_resumes() {
        let (c, _, t) = controller();
        c.start(t, S(30), false);
        let _ = c.step(t, 0);
        assert_eq!(c.resume().unwrap().state, DrainState::Serving);
        assert_eq!(
            c.step(t, 0),
            DrainStep::Wait,
            "serving has nothing to drive"
        );
    }
}
