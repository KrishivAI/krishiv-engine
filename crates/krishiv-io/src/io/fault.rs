// SPDX-License-Identifier: Apache-2.0
//! A fault-injecting store wrapper (tests and deterministic simulation):
//! transient failures with `Retry-After`, latency, and request accounting
//! that flags any request not issued by the [`IoScheduler`](super::IoScheduler).

use std::fmt;
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetRange, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult, Result as StoreResult,
    UploadPart,
};

/// Marker the scheduler puts in the extensions of every request it issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledRequest;

/// A store failure that is worth retrying (throttling, unavailability). Store
/// adapters wrap such failures in it (as `object_store::Error::Generic`'s
/// source); the scheduler retries them, waiting at least `retry_after`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransientError {
    /// The server's `Retry-After`, if it sent one.
    pub retry_after: Option<Duration>,
}

impl fmt::Display for TransientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "service unavailable (transient)")?;
        if let Some(d) = self.retry_after {
            write!(f, ", retry after {d:?}")?;
        }
        Ok(())
    }
}

impl std::error::Error for TransientError {}

/// Counters of a [`FaultStore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FaultStats {
    /// GET requests (including failed ones).
    pub gets: u64,
    /// Bytes returned by successful GETs.
    pub bytes: u64,
    /// GETs that asked for the end of the file (`GetRange::Suffix` or a
    /// bounded range ending at the size).
    pub tail_gets: u64,
    /// PUTs (single-part).
    pub puts: u64,
    /// Multipart uploads started, completed and aborted.
    pub uploads_started: u64,
    /// Multipart uploads completed.
    pub uploads_completed: u64,
    /// Multipart uploads aborted.
    pub uploads_aborted: u64,
    /// Requests that did not come from the scheduler.
    pub unscheduled: u64,
    /// Parts that completed after their upload was aborted (leaked storage).
    pub parts_after_abort: u64,
}

#[derive(Debug, Default)]
struct FaultState {
    fail_gets: u32,
    fail_puts: u32,
    fail_lists: u32,
    fail_deletes: u32,
    retry_after: Option<Duration>,
    latency: Duration,
    stats: FaultStats,
}

/// Wraps a store with injected faults and request accounting.
pub struct FaultStore {
    inner: Arc<dyn ObjectStore>,
    state: Arc<Mutex<FaultState>>,
}

impl fmt::Debug for FaultStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FaultStore({})", self.inner)
    }
}

impl fmt::Display for FaultStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FaultStore({})", self.inner)
    }
}

fn lock(state: &Mutex<FaultState>) -> MutexGuard<'_, FaultState> {
    // Critical sections only update counters and cannot panic.
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

fn transient(retry_after: Option<Duration>) -> object_store::Error {
    object_store::Error::Generic {
        store: "FaultStore",
        source: Box::new(TransientError { retry_after }),
    }
}

impl FaultStore {
    /// Wraps `inner`.
    #[must_use]
    pub fn new(inner: Arc<dyn ObjectStore>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            state: Arc::new(Mutex::new(FaultState::default())),
        })
    }

    /// The next `n` GETs fail transiently, with `retry_after`.
    pub fn fail_next_gets(&self, n: u32, retry_after: Option<Duration>) {
        let mut st = lock(&self.state);
        st.fail_gets = n;
        st.retry_after = retry_after;
    }

    /// The next `n` PUTs or parts fail transiently.
    pub fn fail_next_puts(&self, n: u32) {
        lock(&self.state).fail_puts = n;
    }

    /// The next `n` listings fail transiently after their first object.
    pub fn fail_next_lists(&self, n: u32) {
        lock(&self.state).fail_lists = n;
    }

    /// The next `n` deletes fail transiently.
    pub fn fail_next_deletes(&self, n: u32) {
        lock(&self.state).fail_deletes = n;
    }

    /// Every GET and upload part waits `latency` first (Tokio time).
    pub fn set_latency(&self, latency: Duration) {
        lock(&self.state).latency = latency;
    }

    /// Counters so far.
    #[must_use]
    pub fn stats(&self) -> FaultStats {
        lock(&self.state).stats
    }

    fn check_scheduled(&self, scheduled: bool) {
        if !scheduled {
            lock(&self.state).stats.unscheduled += 1;
        }
    }
}

fn is_tail(range: Option<&GetRange>, size: u64) -> bool {
    match range {
        Some(GetRange::Suffix(_)) => true,
        Some(GetRange::Bounded(r)) => r.end == size,
        _ => false,
    }
}

#[async_trait]
impl ObjectStore for FaultStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> StoreResult<PutResult> {
        self.check_scheduled(opts.extensions.get::<ScheduledRequest>().is_some());
        {
            let mut st = lock(&self.state);
            st.stats.puts += 1;
            if st.fail_puts > 0 {
                st.fail_puts -= 1;
                return Err(transient(None));
            }
        }
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> StoreResult<Box<dyn MultipartUpload>> {
        self.check_scheduled(opts.extensions.get::<ScheduledRequest>().is_some());
        let inner = self.inner.put_multipart_opts(location, opts).await?;
        lock(&self.state).stats.uploads_started += 1;
        Ok(Box::new(FaultUpload {
            inner,
            state: Arc::clone(&self.state),
            aborted: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }))
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> StoreResult<GetResult> {
        self.check_scheduled(options.extensions.get::<ScheduledRequest>().is_some());
        let latency = {
            let mut st = lock(&self.state);
            st.stats.gets += 1;
            if st.fail_gets > 0 {
                st.fail_gets -= 1;
                return Err(transient(st.retry_after));
            }
            st.latency
        };
        if !latency.is_zero() {
            tokio::time::sleep(latency).await;
        }
        let range = options.range.clone();
        let result = self.inner.get_opts(location, options).await?;
        let mut st = lock(&self.state);
        st.stats.bytes += result.range.end - result.range.start;
        if is_tail(range.as_ref(), result.meta.size) {
            st.stats.tail_gets += 1;
        }
        Ok(result)
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> StoreResult<Vec<bytes::Bytes>> {
        // Not used by the scheduler (it coalesces itself).
        self.check_scheduled(false);
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, StoreResult<Path>>,
    ) -> BoxStream<'static, StoreResult<Path>> {
        // A failure is injected before the path reaches the inner store, so a
        // delete reported as failed leaves the object in place.
        let state = Arc::clone(&self.state);
        let inner = Arc::clone(&self.inner);
        Box::pin(futures::StreamExt::flat_map(locations, move |location| {
            let fail = {
                let mut st = lock(&state);
                let fail = st.fail_deletes > 0;
                st.fail_deletes = st.fail_deletes.saturating_sub(1);
                fail
            };
            if fail {
                futures::StreamExt::boxed(futures::stream::once(async { Err(transient(None)) }))
            } else {
                inner.delete_stream(futures::StreamExt::boxed(futures::stream::once(async {
                    location
                })))
            }
        }))
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, StoreResult<ObjectMeta>> {
        let fail = {
            let mut st = lock(&self.state);
            let fail = st.fail_lists > 0;
            st.fail_lists = st.fail_lists.saturating_sub(1);
            fail
        };
        let inner = self.inner.list(prefix);
        if fail {
            // One object, then a transient failure mid-listing.
            Box::pin(futures::StreamExt::chain(
                futures::StreamExt::take(inner, 1),
                futures::stream::once(std::future::ready(Err(transient(None)))),
            ))
        } else {
            inner
        }
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> StoreResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> StoreResult<()> {
        self.check_scheduled(false);
        self.inner.copy_opts(from, to, options).await
    }
}

#[derive(Debug)]
struct FaultUpload {
    inner: Box<dyn MultipartUpload>,
    state: Arc<Mutex<FaultState>>,
    aborted: Arc<std::sync::atomic::AtomicBool>,
}

#[async_trait]
impl MultipartUpload for FaultUpload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        let latency = {
            let mut st = lock(&self.state);
            if st.fail_puts > 0 {
                st.fail_puts -= 1;
                return Box::pin(async { Err(transient(None)) });
            }
            st.latency
        };
        let part = self.inner.put_part(data);
        let (aborted, state) = (Arc::clone(&self.aborted), Arc::clone(&self.state));
        Box::pin(async move {
            if !latency.is_zero() {
                tokio::time::sleep(latency).await;
            }
            part.await?;
            if aborted.load(std::sync::atomic::Ordering::Acquire) {
                lock(&state).stats.parts_after_abort += 1;
            }
            Ok(())
        })
    }

    async fn complete(&mut self) -> StoreResult<PutResult> {
        let r = self.inner.complete().await?;
        lock(&self.state).stats.uploads_completed += 1;
        Ok(r)
    }

    async fn abort(&mut self) -> StoreResult<()> {
        self.aborted
            .store(true, std::sync::atomic::Ordering::Release);
        lock(&self.state).stats.uploads_aborted += 1;
        self.inner.abort().await
    }
}
