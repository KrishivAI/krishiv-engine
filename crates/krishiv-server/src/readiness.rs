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

struct Round {
    at: Instant,
    outcomes: Vec<CheckOutcome>,
}

/// The cached readiness of the server.
pub struct Readiness {
    serving: Arc<AtomicBool>,
    checks: Vec<Arc<dyn ReadinessCheck>>,
    interval: Duration,
    timeout: Duration,
    last: Mutex<Option<Round>>,
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
            last: Mutex::new(None),
        })
    }

    /// Whether `/readyz` answers 200: serving, and, when there are checks,
    /// a fresh round in which every one passed.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        if !self.serving.load(Ordering::Acquire) {
            return false;
        }
        if self.checks.is_empty() {
            return true;
        }
        let last = self.last.lock().unwrap_or_else(PoisonError::into_inner);
        let fresh = self.interval.saturating_mul(STALE_AFTER_INTERVALS);
        last.as_ref().is_some_and(|r| {
            r.at.elapsed() <= fresh && r.outcomes.iter().all(|o| o.failure.is_none())
        })
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
        let mut outcomes = Vec::with_capacity(self.checks.len());
        for check in &self.checks {
            let failure = match tokio::time::timeout(self.timeout, check.run()).await {
                Ok(Ok(())) => None,
                Ok(Err(reason)) => Some(reason),
                Err(_) => Some(format!("timed out after {:?}", self.timeout)),
            };
            if let Some(reason) = &failure {
                tracing::warn!(check = check.name(), %reason, "readiness check failed");
            }
            outcomes.push(CheckOutcome {
                name: check.name(),
                failure,
            });
        }
        let mut last = self.last.lock().unwrap_or_else(PoisonError::into_inner);
        *last = Some(Round {
            at: Instant::now(),
            outcomes,
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

    #[tokio::test]
    async fn a_result_older_than_three_intervals_is_not_ready() {
        let up = Arc::new(AtomicBool::new(true));
        let (r, _) = readiness(vec![Arc::new(Fixed("store", up))], 20);
        r.run_round().await;
        assert!(r.is_ready());
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert!(!r.is_ready(), "no new round for over three intervals");
    }

    #[tokio::test]
    async fn the_background_checker_keeps_the_result_fresh() {
        let up = Arc::new(AtomicBool::new(true));
        let (r, _) = readiness(vec![Arc::new(Fixed("store", up))], 20);
        let handle = r.spawn().expect("has checks");
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(r.is_ready());
        handle.abort();
    }
}
