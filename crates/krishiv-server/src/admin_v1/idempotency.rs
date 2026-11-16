// SPDX-License-Identifier: Apache-2.0
//! Idempotent mutating calls (design §6.2). The server remembers `(principal, request id)` with
//! the response for a time, at most [`PER_PRINCIPAL`] entries per principal (that principal's
//! oldest goes first, so one principal cannot flush another's keys). The same id with the same
//! request returns the stored response; with another request it is `KRISHIV_ADMIN_INVALID_REQUEST`
//! ("reused"); a second call with an id whose first call still runs waits for it.
//!
//! The store is per process and lost on restart: a retry after a restart, or sent to another
//! replica, is applied again.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use tokio::sync::OnceCell;

use super::{AdminError, AdminResult, ErrorKind};

/// Entries kept per principal.
pub const PER_PRINCIPAL: usize = 1_000;

struct Entry {
    id: String,
    request: Vec<u8>,
    stored: Instant,
    cell: Arc<OnceCell<AdminResult<Vec<u8>>>>,
}

/// The remembered calls of a process.
pub struct Idempotency {
    ttl: Duration,
    per_principal: usize,
    entries: Mutex<HashMap<String, VecDeque<Entry>>>,
}

impl std::fmt::Debug for Idempotency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Idempotency").finish_non_exhaustive()
    }
}

/// How a call went through the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Run {
    /// The call ran now.
    Ran,
    /// The call returned the stored response of an earlier one.
    Replayed,
}

impl Idempotency {
    /// A store keeping a call for `ttl`.
    #[must_use]
    pub fn new(ttl: Duration) -> Self {
        Self::with_limit(ttl, PER_PRINCIPAL)
    }

    /// A store with another per-principal bound (tests).
    #[must_use]
    pub fn with_limit(ttl: Duration, per_principal: usize) -> Self {
        Self {
            ttl,
            per_principal: per_principal.max(1),
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Runs `call` once for `(principal, id)` and returns its response; later calls with the same
    /// request return it without running, and calls racing the first wait for it. `request` is
    /// what makes two calls "the same request" (the RPC name and its encoded message). An empty
    /// `id` runs `call` every time.
    pub async fn run<F, Fut>(
        &self,
        principal: &str,
        id: &str,
        request: &[u8],
        call: F,
    ) -> (AdminResult<Vec<u8>>, Run)
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = AdminResult<Vec<u8>>>,
    {
        if id.is_empty() {
            return (call().await, Run::Ran);
        }
        let cell = {
            let mut map = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
            let queue = map.entry(principal.to_owned()).or_default();
            let now = Instant::now();
            queue.retain(|e| now.duration_since(e.stored) < self.ttl);
            if let Some(entry) = queue.iter().find(|e| e.id == id) {
                if entry.request != request {
                    return (Err(reused(id)), Run::Replayed);
                }
                Arc::clone(&entry.cell)
            } else {
                while queue.len() >= self.per_principal {
                    queue.pop_front();
                }
                let cell = Arc::new(OnceCell::new());
                queue.push_back(Entry {
                    id: id.to_owned(),
                    request: request.to_vec(),
                    stored: now,
                    cell: Arc::clone(&cell),
                });
                cell
            }
        };
        let mut ran = false;
        let result = cell
            .get_or_init(|| {
                ran = true;
                call()
            })
            .await
            .clone();
        (result, if ran { Run::Ran } else { Run::Replayed })
    }
}

fn reused(id: &str) -> AdminError {
    AdminError::new(
        ErrorKind::AlreadyExists,
        "KRISHIV_ADMIN_INVALID_REQUEST",
        "22023",
        format!(
            "The admin API request is not valid: the request id {id} was used for another request."
        ),
    )
    .with_parameter("reason", "request id reused with another request")
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    async fn count(n: &AtomicUsize) -> AdminResult<Vec<u8>> {
        Ok(vec![
            u8::try_from(n.fetch_add(1, Ordering::SeqCst)).unwrap_or(255),
        ])
    }

    #[tokio::test]
    async fn the_same_id_and_request_runs_once() {
        let store = Idempotency::new(Duration::from_secs(60));
        let n = AtomicUsize::new(0);
        let (a, ran) = store.run("p", "k", b"r", || count(&n)).await;
        let (b, again) = store.run("p", "k", b"r", || count(&n)).await;
        assert_eq!((ran, again), (Run::Ran, Run::Replayed));
        assert_eq!(a, b);
        assert_eq!(n.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn another_request_with_the_id_is_refused_and_no_id_runs_every_time() {
        let store = Idempotency::new(Duration::from_secs(60));
        let n = AtomicUsize::new(0);
        store
            .run("p", "k", b"one", || count(&n))
            .await
            .0
            .unwrap_or_default();
        let (result, _) = store.run("p", "k", b"two", || count(&n)).await;
        assert_eq!(result.err().map(|e| e.kind), Some(ErrorKind::AlreadyExists));
        store
            .run("p", "", b"x", || count(&n))
            .await
            .0
            .unwrap_or_default();
        store
            .run("p", "", b"x", || count(&n))
            .await
            .0
            .unwrap_or_default();
        assert_eq!(n.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn two_concurrent_calls_with_one_id_apply_once() {
        let store = Arc::new(Idempotency::new(Duration::from_secs(60)));
        let n = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for _ in 0..8 {
            let (store, n) = (Arc::clone(&store), Arc::clone(&n));
            tasks.push(tokio::spawn(async move {
                store
                    .run("p", "k", b"r", || async {
                        tokio::time::sleep(Duration::from_millis(20)).await;
                        count(&n).await
                    })
                    .await
            }));
        }
        let mut ran = 0;
        for t in tasks {
            let (result, how) = t.await.unwrap_or_else(|e| panic!("{e}"));
            assert!(result.is_ok());
            ran += usize::from(how == Run::Ran);
        }
        assert_eq!((ran, n.load(Ordering::SeqCst)), (1, 1));
    }

    #[tokio::test]
    async fn one_principals_keys_never_evict_anothers() {
        let store = Idempotency::with_limit(Duration::from_secs(60), 2);
        let n = AtomicUsize::new(0);
        store
            .run("quiet", "q", b"r", || count(&n))
            .await
            .0
            .unwrap_or_default();
        for i in 0..5 {
            store
                .run("noisy", &format!("k{i}"), b"r", || count(&n))
                .await
                .0
                .unwrap_or_default();
        }
        let (_, how) = store.run("quiet", "q", b"r", || count(&n)).await;
        assert_eq!(
            how,
            Run::Replayed,
            "the quiet principal's key is still there"
        );
        let (_, how) = store.run("noisy", "k0", b"r", || count(&n)).await;
        assert_eq!(how, Run::Ran, "the noisy principal's oldest was evicted");
    }

    #[tokio::test]
    async fn a_stored_call_expires() {
        let store = Idempotency::new(Duration::from_millis(10));
        let n = AtomicUsize::new(0);
        store
            .run("p", "k", b"r", || count(&n))
            .await
            .0
            .unwrap_or_default();
        tokio::time::sleep(Duration::from_millis(30)).await;
        let (_, how) = store.run("p", "k", b"r", || count(&n)).await;
        assert_eq!(how, Run::Ran);
    }
}
