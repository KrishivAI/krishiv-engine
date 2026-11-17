// SPDX-License-Identifier: Apache-2.0
//! Idempotent mutating calls (design §6.2). The server remembers `(principal, request id)` with
//! the response for a time, at most [`PER_PRINCIPAL`] entries per principal (that principal's
//! oldest goes first, so one principal cannot flush another's keys). The same id with the same
//! request returns the stored response; with another request it is `KRISHIV_IDEMPOTENCY_KEY_REUSED`; a second call with an id whose first call still runs waits for it.
//!
//! The store holds a 128-bit digest of each request, not the request, and at most
//! [`MAX_TOTAL`] entries in all: a sweep of expired entries runs when it is full, and a principal
//! that still finds it full loses its own oldest entry. The key is the caller's scope (its full
//! identity, not its name), so two principals with one name share nothing.
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

/// Entries kept in all, across principals.
pub const MAX_TOTAL: usize = 100_000;

/// The longest request id (bytes) the API takes, on the REST header and in the message alike.
pub const MAX_ID_LEN: usize = 128;

struct Entry {
    id: String,
    digest: [u64; 2],
    stored: Instant,
    cell: Arc<OnceCell<AdminResult<Vec<u8>>>>,
}

/// The remembered calls of a process.
pub struct Idempotency {
    ttl: Duration,
    per_principal: usize,
    max_total: usize,
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
    /// The id was used for another request: refused, nothing ran.
    Reused,
}

/// A 128-bit digest of `request`. The store is per principal, so a collision can only confuse a
/// principal's own two requests.
fn digest(request: &[u8]) -> [u64; 2] {
    use std::hash::Hasher;
    let mut out = [0_u64; 2];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        h.write_u8(u8::try_from(i).unwrap_or(0));
        h.write(request);
        *slot = h.finish();
    }
    out
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
            max_total: MAX_TOTAL,
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
            let now = Instant::now();
            let total: usize = map.values().map(|q| q.len().max(1)).sum();
            if total >= self.max_total {
                map.retain(|_, q| {
                    q.retain(|e| now.duration_since(e.stored) < self.ttl);
                    !q.is_empty()
                });
            }
            let digest = digest(request);
            let queue = map.entry(principal.to_owned()).or_default();
            queue.retain(|e| now.duration_since(e.stored) < self.ttl);
            let found = queue
                .iter()
                .find(|e| e.id == id)
                .map(|e| (e.digest, Arc::clone(&e.cell)));
            let cell = if let Some((stored, cell)) = found {
                if stored != digest {
                    return (Err(reused(id)), Run::Reused);
                }
                cell
            } else {
                while queue.len() >= self.per_principal {
                    queue.pop_front();
                }
                let cell = Arc::new(OnceCell::new());
                queue.push_back(Entry {
                    id: id.to_owned(),
                    digest,
                    stored: now,
                    cell: Arc::clone(&cell),
                });
                cell
            };
            let total: usize = map.values().map(VecDeque::len).sum();
            if total > self.max_total {
                if let Some(q) = map.get_mut(principal) {
                    if q.len() > 1 {
                        q.pop_front();
                    }
                }
            }
            cell
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
        "KRISHIV_IDEMPOTENCY_KEY_REUSED",
        "22023",
        format!(
            "The request id {id} was already used for another request. Send a new request id for a different request."
        ),
    )
    .with_parameter("requestId", id)
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
    async fn one_principals_keys_never_evict_other_principals_keys() {
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

    #[tokio::test]
    async fn a_reused_id_is_its_own_class_and_not_a_replay() {
        let store = Idempotency::new(Duration::from_secs(60));
        let n = AtomicUsize::new(0);
        store
            .run("p", "k", b"one", || count(&n))
            .await
            .0
            .unwrap_or_default();
        let (result, how) = store.run("p", "k", b"two", || count(&n)).await;
        assert_eq!(how, Run::Reused);
        assert_eq!(
            result.err().map(|e| e.error_class),
            Some("KRISHIV_IDEMPOTENCY_KEY_REUSED".to_owned())
        );
    }

    #[tokio::test]
    async fn the_store_is_bounded_across_principals_and_frees_expired_queues() {
        let mut store = Idempotency::new(Duration::from_millis(10));
        store.max_total = 4;
        let n = AtomicUsize::new(0);
        for i in 0..4 {
            store
                .run(&format!("p{i}"), "k", b"r", || count(&n))
                .await
                .0
                .unwrap_or_default();
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
        store
            .run("late", "k", b"r", || count(&n))
            .await
            .0
            .unwrap_or_default();
        let map = store.entries.lock().unwrap_or_else(PoisonError::into_inner);
        assert_eq!(map.len(), 1, "the expired principals' queues are gone");
    }
}
