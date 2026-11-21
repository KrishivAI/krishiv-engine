// SPDX-License-Identifier: Apache-2.0
//! Cached readiness checks behind `/readyz` (design P3-T066 §5.2).
//!
//! The checks run in the background every `interval`, each within `timeout`;
//! `/readyz` answers from the last result, so a probe never blocks and never
//! multiplies calls to a remote service. A result older than three intervals
//! counts as not ready, so a stalled or dead checker task cannot serve "ready"
//! for ever. The serving flag (not draining, the session manager accepting:
//! check 1) is read on every probe, so a drain flips `/readyz` at once.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use tokio::task::JoinHandle;
use tokio::time::Instant;

/// How many intervals a result stays fresh.
pub const STALE_AFTER_INTERVALS: u32 = 3;

/// The future of one check: `Ok(())` when the dependency answers, else a
/// short reason (logged, never put in the probe body).
pub type CheckFuture = Pin<Box<dyn Future<Output = Result<(), String>> + Send>>;

/// One dependency the server needs before it takes traffic (the warehouse
/// object store, the authentication keys).
pub trait ReadinessCheck: Send + Sync {
    /// A stable name for logs and the detail report.
    fn name(&self) -> &'static str;
    /// Runs the check once; the caller applies the timeout.
    fn run(&self) -> CheckFuture;
}

/// Check 2 (design P3-T066 §5.2): with OIDC configured, the JWK set has been
/// fetched and is not stale. Passes at once when OIDC is not configured.
#[cfg(feature = "session")]
#[derive(Debug, Clone)]
pub struct AuthReady(pub Arc<krishiv_session::auth::ServerAuth>);

#[cfg(feature = "session")]
impl ReadinessCheck for AuthReady {
    fn name(&self) -> &'static str {
        "auth"
    }
    fn run(&self) -> CheckFuture {
        let verdict = self.0.jwks_ready().unwrap_or(Ok(()));
        Box::pin(std::future::ready(verdict))
    }
}

/// The outcome of one check in the last round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckOutcome {
    /// The check's name.
    pub name: &'static str,
    /// `None` when it passed, else the reason.
    pub failure: Option<String>,
}

/// The state of one check as the admin API reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckState {
    /// Passed in the last round.
    Ok,
    /// Failed in the last round.
    Failed,
    /// No round has run yet.
    Unknown,
}

/// One check in [`Readiness::report`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckReport {
    /// The check's name.
    pub name: String,
    /// Its state.
    pub state: CheckState,
    /// Whether its failure fails `/readyz`.
    pub required: bool,
    /// Failures in a row.
    pub consecutive_failures: u32,
}

struct Round {
    /// When each check's verdict was produced, parallel to `outcomes`: freshness is per check,
    /// so a slow optional check cannot make an old `auth` verdict look new.
    finished: Vec<Instant>,
    outcomes: Vec<CheckOutcome>,
    /// Consecutive failures per check, parallel to `outcomes`.
    streaks: Vec<u32>,
    /// Whether each check has passed in some round so far.
    ever_ok: Vec<bool>,
}

/// The cached readiness of the server.
pub struct Readiness {
    serving: Arc<AtomicBool>,
    checks: Vec<Arc<dyn ReadinessCheck>>,
    interval: Duration,
    timeout: Duration,
    /// Consecutive failures after which a required check fails readiness.
    failure_threshold: u32,
    /// Names of the checks that can fail readiness; `None` means all of them.
    required: Option<Vec<String>>,
    last: Mutex<Option<Round>>,
    /// Held for a whole round, so rounds never overlap and never publish out of order.
    round: tokio::sync::Mutex<()>,
}

impl Readiness {
    /// Readiness over `serving` (check 1) and `checks`, re-run every
    /// `interval` with `timeout` each.
    #[must_use]
    pub fn new(
        serving: Arc<AtomicBool>,
        checks: Vec<Arc<dyn ReadinessCheck>>,
        interval: Duration,
        timeout: Duration,
    ) -> Arc<Self> {
        Arc::new(Self {
            serving,
            checks,
            interval,
            timeout,
            failure_threshold: 1,
            required: None,
            last: Mutex::new(None),
            round: tokio::sync::Mutex::new(()),
        })
    }

    /// Like [`Readiness::new`], with the design's rules (P3-T066 §5.2): a check
    /// fails readiness only after `failure_threshold` consecutive failures (one
    /// slow moment is not a failure), and only checks named in `required` can
    /// fail it; the others are reported by [`Readiness::outcomes`] only.
    #[must_use]
    pub fn with_policy(
        serving: Arc<AtomicBool>,
        checks: Vec<Arc<dyn ReadinessCheck>>,
        interval: Duration,
        timeout: Duration,
        failure_threshold: u32,
        required: Vec<String>,
    ) -> Arc<Self> {
        Arc::new(Self {
            serving,
            checks,
            interval,
            timeout,
            failure_threshold: failure_threshold.max(1),
            required: Some(required),
            last: Mutex::new(None),
            round: tokio::sync::Mutex::new(()),
        })
    }

    fn is_required(&self, name: &str) -> bool {
        self.required
            .as_ref()
            .is_none_or(|names| names.iter().any(|n| n == name))
    }

    /// Whether `/readyz` answers 200: serving, and, when there are checks,
    /// a fresh round in which every one passed.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        let serving = self.serving.load(Ordering::Acquire);
        let last = self.last.lock().unwrap_or_else(PoisonError::into_inner);
        self.ready_with(serving, last.as_ref())
    }

    fn ready_with(&self, serving: bool, last: Option<&Round>) -> bool {
        if !serving {
            return false;
        }
        if self.checks.is_empty() {
            return true;
        }
        let fresh = self.interval.saturating_mul(STALE_AFTER_INTERVALS);
        last.is_some_and(|r| {
            r.outcomes
                .iter()
                .zip(&r.streaks)
                .zip(&r.ever_ok)
                .zip(&r.finished)
                .all(|(((o, streak), ever_ok), at)| {
                    // The threshold forgives a dependency that broke after it worked; one that
                    // never worked (keys never fetched) is not ready at once.
                    !self.is_required(o.name)
                        || (at.elapsed() <= fresh
                            && *streak < self.failure_threshold
                            && (*ever_ok || *streak == 0))
                })
        })
    }

    /// Every check with its state, for the admin API: the serving check first, then each
    /// registered check (`unknown` before the first round).
    #[must_use]
    pub fn report(&self) -> Vec<CheckReport> {
        self.snapshot().1
    }

    /// `is_ready` and `report` from one observation of the serving flag and one cached round,
    /// so `ready` always agrees with the checks listed beside it.
    #[must_use]
    pub fn snapshot(&self) -> (bool, Vec<CheckReport>) {
        let serving = self.serving.load(Ordering::Acquire);
        let mut out = vec![CheckReport {
            name: "serving".to_owned(),
            state: if serving {
                CheckState::Ok
            } else {
                CheckState::Failed
            },
            required: true,
            consecutive_failures: 0,
        }];
        let last = self.last.lock().unwrap_or_else(PoisonError::into_inner);
        for (i, check) in self.checks.iter().enumerate() {
            let (state, failures) = match last.as_ref() {
                None => (CheckState::Unknown, 0),
                Some(r) => (
                    if r.outcomes.get(i).is_some_and(|o| o.failure.is_none()) {
                        CheckState::Ok
                    } else {
                        CheckState::Failed
                    },
                    r.streaks.get(i).copied().unwrap_or(0),
                ),
            };
            out.push(CheckReport {
                name: check.name().to_owned(),
                state,
                required: self.is_required(check.name()),
                consecutive_failures: failures,
            });
        }
        (self.ready_with(serving, last.as_ref()), out)
    }

    /// The outcomes of the last round (empty before the first).
    #[must_use]
    pub fn outcomes(&self) -> Vec<CheckOutcome> {
        let last = self.last.lock().unwrap_or_else(PoisonError::into_inner);
        last.as_ref()
            .map(|r| r.outcomes.clone())
            .unwrap_or_default()
    }

    /// Runs every check once (each within the timeout) and stores the round.
    pub async fn run_round(&self) {
        let _round = self.round.lock().await;
        let mut outcomes = Vec::with_capacity(self.checks.len());
        let mut finished = Vec::with_capacity(self.checks.len());
        for check in &self.checks {
            let failure = match tokio::time::timeout(self.timeout, check.run()).await {
                Ok(Ok(())) => None,
                Ok(Err(reason)) => Some(reason),
                Err(_) => Some(format!("timed out after {:?}", self.timeout)),
            };
            if let Some(reason) = &failure {
                tracing::warn!(check = check.name(), %reason, "readiness check failed");
            }
            finished.push(Instant::now());
            outcomes.push(CheckOutcome {
                name: check.name(),
                failure,
            });
        }
        let mut last = self.last.lock().unwrap_or_else(PoisonError::into_inner);
        let streaks = outcomes
            .iter()
            .enumerate()
            .map(|(i, o)| match (&o.failure, last.as_ref()) {
                (None, _) => 0,
                (Some(_), Some(prev)) => prev.streaks.get(i).copied().unwrap_or(0) + 1,
                (Some(_), None) => 1,
            })
            .collect();
        let ever_ok = outcomes
            .iter()
            .enumerate()
            .map(|(i, o)| {
                o.failure.is_none()
                    || last
                        .as_ref()
                        .is_some_and(|p| p.ever_ok.get(i) == Some(&true))
            })
            .collect();
        *last = Some(Round {
            finished,
            outcomes,
            streaks,
            ever_ok,
        });
    }

    /// Starts the background checker; abort the handle to stop it. With no
    /// checks there is nothing to run.
    #[must_use]
    pub fn spawn(self: &Arc<Self>) -> Option<JoinHandle<()>> {
        if self.checks.is_empty() {
            return None;
        }
        let this = Arc::clone(self);
        Some(tokio::spawn(async move {
            loop {
                this.run_round().await;
                tokio::time::sleep(this.interval).await;
            }
        }))
    }
}

impl std::fmt::Debug for Readiness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Readiness")
            .field("checks", &self.checks.len())
            .field("interval", &self.interval)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed(&'static str, Arc<AtomicBool>);
    impl ReadinessCheck for Fixed {
        fn name(&self) -> &'static str {
            self.0
        }
        fn run(&self) -> CheckFuture {
            let ok = self.1.load(Ordering::SeqCst);
            Box::pin(async move { if ok { Ok(()) } else { Err("down".to_owned()) } })
        }
    }

    struct Hangs;
    impl ReadinessCheck for Hangs {
        fn name(&self) -> &'static str {
            "hangs"
        }
        fn run(&self) -> CheckFuture {
            Box::pin(std::future::pending())
        }
    }

    fn readiness(
        checks: Vec<Arc<dyn ReadinessCheck>>,
        interval_ms: u64,
    ) -> (Arc<Readiness>, Arc<AtomicBool>) {
        let serving = Arc::new(AtomicBool::new(true));
        let r = Readiness::new(
            Arc::clone(&serving),
            checks,
            Duration::from_millis(interval_ms),
            Duration::from_millis(50),
        );
        (r, serving)
    }

    #[tokio::test]
    async fn without_checks_the_serving_flag_decides() {
        let (r, serving) = readiness(vec![], 1000);
        assert!(r.is_ready());
        serving.store(false, Ordering::SeqCst);
        assert!(!r.is_ready());
    }

    #[tokio::test]
    async fn not_ready_before_the_first_round_then_follows_the_checks() {
        let up = Arc::new(AtomicBool::new(true));
        let (r, _) = readiness(vec![Arc::new(Fixed("store", Arc::clone(&up)))], 1000);
        assert!(!r.is_ready());
        r.run_round().await;
        assert!(r.is_ready());
        up.store(false, Ordering::SeqCst);
        r.run_round().await;
        assert!(!r.is_ready());
        assert_eq!(r.outcomes()[0].failure.as_deref(), Some("down"));
    }

    #[tokio::test]
    async fn a_draining_flag_flips_at_once_without_a_new_round() {
        let up = Arc::new(AtomicBool::new(true));
        let (r, serving) = readiness(vec![Arc::new(Fixed("store", up))], 1000);
        r.run_round().await;
        assert!(r.is_ready());
        serving.store(false, Ordering::SeqCst);
        assert!(!r.is_ready());
    }

    #[tokio::test]
    async fn a_hanging_check_times_out_as_a_failure() {
        let (r, _) = readiness(vec![Arc::new(Hangs)], 1000);
        r.run_round().await;
        assert!(!r.is_ready());
        assert!(
            r.outcomes()[0]
                .failure
                .as_deref()
                .is_some_and(|m| m.contains("timed out"))
        );
    }

    /// Runs on a paused clock: the age of a result is exact, with no real sleeping.
    #[tokio::test(start_paused = true)]
    async fn a_result_older_than_three_intervals_is_not_ready() {
        let up = Arc::new(AtomicBool::new(true));
        let (r, _) = readiness(vec![Arc::new(Fixed("store", up))], 20);
        r.run_round().await;
        assert!(r.is_ready());
        tokio::time::advance(Duration::from_millis(59)).await;
        assert!(r.is_ready(), "inside three intervals");
        tokio::time::advance(Duration::from_millis(22)).await;
        assert!(!r.is_ready(), "no new round for over three intervals");
    }

    #[tokio::test(start_paused = true)]
    async fn the_background_checker_keeps_the_result_fresh() {
        let up = Arc::new(AtomicBool::new(true));
        let (r, _) = readiness(vec![Arc::new(Fixed("store", up))], 20);
        let handle = r.spawn().expect("has checks");
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(r.is_ready());
        handle.abort();
    }

    #[tokio::test]
    async fn the_report_names_every_check_with_its_state() {
        let up = Arc::new(AtomicBool::new(false));
        let r = Readiness::with_policy(
            Arc::new(AtomicBool::new(true)),
            vec![Arc::new(Fixed("warehouse", up))],
            Duration::from_secs(10),
            Duration::from_millis(50),
            3,
            vec![],
        );
        let before = r.report();
        assert_eq!(before[1].state, CheckState::Unknown);
        r.run_round().await;
        r.run_round().await;
        let after = r.report();
        assert_eq!(after[0].name, "serving");
        assert_eq!(after[0].state, CheckState::Ok);
        assert_eq!(after[1].state, CheckState::Failed);
        assert_eq!(after[1].consecutive_failures, 2);
        assert!(!after[1].required);
    }

    #[tokio::test]
    async fn a_required_check_that_never_passed_is_not_ready_at_once() {
        let up = Arc::new(AtomicBool::new(false));
        let r = Readiness::with_policy(
            Arc::new(AtomicBool::new(true)),
            vec![Arc::new(Fixed("auth", up))],
            Duration::from_secs(10),
            Duration::from_millis(50),
            3,
            vec!["auth".to_owned()],
        );
        r.run_round().await;
        assert!(
            !r.is_ready(),
            "the first failed round is below the threshold but never worked"
        );
    }

    #[tokio::test]
    async fn a_failure_counts_only_after_the_threshold_and_recovers_at_once() {
        let up = Arc::new(AtomicBool::new(true));
        let serving = Arc::new(AtomicBool::new(true));
        let r = Readiness::with_policy(
            serving,
            vec![Arc::new(Fixed("store", Arc::clone(&up)))],
            Duration::from_secs(10),
            Duration::from_millis(50),
            3,
            vec!["store".to_owned()],
        );
        r.run_round().await;
        up.store(false, Ordering::SeqCst);
        r.run_round().await;
        r.run_round().await;
        assert!(
            r.is_ready(),
            "two failures are below the threshold of three"
        );
        r.run_round().await;
        assert!(!r.is_ready());
        up.store(true, Ordering::SeqCst);
        r.run_round().await;
        assert!(r.is_ready(), "the first success recovers");
    }

    #[tokio::test]
    async fn a_check_not_required_is_reported_but_never_fails_readiness() {
        let up = Arc::new(AtomicBool::new(false));
        let r = Readiness::with_policy(
            Arc::new(AtomicBool::new(true)),
            vec![Arc::new(Fixed("warehouse", up))],
            Duration::from_secs(10),
            Duration::from_millis(50),
            1,
            vec!["auth".to_owned()],
        );
        r.run_round().await;
        assert!(r.is_ready());
        assert_eq!(r.outcomes()[0].failure.as_deref(), Some("down"));
    }
}
