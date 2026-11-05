// SPDX-License-Identifier: Apache-2.0
//! [`RefreshingStore`]: one forced credential refresh and one retry when the store refuses a
//! request as stale (design note section 2).
//!
//! object_store signs a request before its retry loop and never retries a 403, and the
//! credential provider never sees a response. So a credential that expired between the
//! provider's check and the store's answer (`403 ExpiredToken`, a 401 from GCS, an Azure
//! `AuthenticationFailed`) fails the request. This wrapper recognizes that answer, marks the
//! provider's cached credential stale and retries the request once. Any other 403 fails as it
//! does today, with no refresh.

use std::error::Error as StdError;
use std::fmt;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;
use futures::stream::BoxStream;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions, Result,
};

use crate::credentials::{
    CredentialCache, CredentialRefreshError, RefreshingProvider, ServedGeneration,
};

/// Words in a store's answer that mean "the credential is no longer good", not "this
/// credential may not do that". S3 answers `ExpiredToken` (400 or 403); Azure answers
/// `AuthenticationFailed` for an expired SAS.
const STALE_MARKERS: [&str; 4] = [
    "ExpiredToken",
    "TokenRefreshRequired",
    "AuthenticationFailed",
    "Signature expired",
];

/// Whether `e` is the store refusing the credential as stale: any 401, or a 400/403 whose
/// answer names an expired token. A refresh that itself failed is not (retrying cannot help).
pub(crate) fn is_stale_credential(e: &object_store::Error) -> bool {
    if chain_has_refresh_failure(e) {
        return false;
    }
    match e {
        object_store::Error::Unauthenticated { .. } => true,
        object_store::Error::PermissionDenied { .. } | object_store::Error::Generic { .. } => {
            let mut text = e.to_string();
            let mut source = e.source();
            while let Some(s) = source {
                text.push(' ');
                text.push_str(&s.to_string());
                source = s.source();
            }
            STALE_MARKERS.iter().any(|m| text.contains(m))
        }
        _ => false,
    }
}

fn chain_has_refresh_failure(e: &object_store::Error) -> bool {
    let mut source = e.source();
    while let Some(s) = source {
        if s.downcast_ref::<CredentialRefreshError>().is_some() {
            return true;
        }
        source = s.source();
    }
    false
}

/// A store whose requests are retried once, after a forced credential refresh, when the
/// store refuses the credential as stale.
///
/// Replayable requests retry: `put`, `get` (and `head`), `copy`, `rename`, ranges, a delimited
/// listing and the start of a multipart upload. Streams (`list`, `delete_stream`, and so
/// `delete`, which object_store routes through `delete_stream`) and the parts of an upload
/// already started do not: they fail as the store reports, a refusal as stale invalidates the
/// credential (streams only), and the next request signs with a fresh credential.
pub struct RefreshingStore {
    inner: Arc<dyn ObjectStore>,
    cache: Arc<CredentialCache>,
}

impl RefreshingStore {
    /// Wraps `inner`, which signs its requests with `provider`.
    #[must_use]
    pub fn new<C>(inner: Arc<dyn ObjectStore>, provider: &RefreshingProvider<C>) -> Self {
        Self {
            inner,
            cache: Arc::clone(&provider.cache),
        }
    }

    /// Runs `call`; if the store refuses the credential as stale, invalidates the credential
    /// the request was signed with and runs `call` once more.
    async fn with_retry<T, F, Fut>(&self, call: F) -> Result<T>
    where
        F: Fn() -> Fut + Send,
        Fut: std::future::Future<Output = Result<T>> + Send,
    {
        // The provider records the generation it signs with. Without a record (the request
        // never reached the provider) the generation before the call stands in.
        let before = self.cache.generation();
        let served = ServedGeneration::new();
        match Recording::new(served.clone(), call()).await {
            Err(e) if is_stale_credential(&e) => {
                self.cache.invalidate(served.get().unwrap_or(before));
                call().await
            }
            other => other,
        }
    }

    /// Passes `stream` through, invalidating the credential when it reports a refusal as
    /// stale. The stream is not replayed: the next request signs with a fresh credential.
    fn watching<T: Send + 'static>(
        &self,
        stream: BoxStream<'static, Result<T>>,
    ) -> BoxStream<'static, Result<T>> {
        Box::pin(Watching {
            inner: stream,
            cache: Arc::clone(&self.cache),
            before: self.cache.generation(),
            served: ServedGeneration::new(),
        })
    }
}

/// A future polled with its credential generation recorded.
struct Recording<Fut> {
    served: ServedGeneration,
    inner: Pin<Box<Fut>>,
}

impl<Fut> Recording<Fut> {
    fn new(served: ServedGeneration, inner: Fut) -> Self {
        Self {
            served,
            inner: Box::pin(inner),
        }
    }
}

impl<Fut: std::future::Future> std::future::Future for Recording<Fut> {
    type Output = Fut::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let _entered = self.served.enter();
        self.inner.as_mut().poll(cx)
    }
}

/// A stream that invalidates its credential when an item is a stale-credential refusal.
struct Watching<T> {
    inner: BoxStream<'static, Result<T>>,
    cache: Arc<CredentialCache>,
    before: u64,
    served: ServedGeneration,
}

impl<T> Stream for Watching<T> {
    type Item = Result<T>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let entered = self.served.enter();
        let polled = self.inner.as_mut().poll_next(cx);
        drop(entered);
        if let Poll::Ready(Some(Err(e))) = &polled
            && is_stale_credential(e)
        {
            self.cache
                .invalidate(self.served.get().unwrap_or(self.before));
        }
        polled
    }
}

impl fmt::Display for RefreshingStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RefreshingStore({})", self.inner)
    }
}

impl fmt::Debug for RefreshingStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RefreshingStore")
            .field("scope", self.cache.scope())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl ObjectStore for RefreshingStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        self.with_retry(|| self.inner.put_opts(location, payload.clone(), opts.clone()))
            .await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.with_retry(|| self.inner.put_multipart_opts(location, opts.clone()))
            .await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        self.with_retry(|| self.inner.get_opts(location, options.clone()))
            .await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        self.with_retry(|| self.inner.get_ranges(location, ranges))
            .await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        self.watching(self.inner.delete_stream(locations))
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.watching(self.inner.list(prefix))
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        self.watching(self.inner.list_with_offset(prefix, offset))
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.with_retry(|| self.inner.list_with_delimiter(prefix))
            .await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
        self.with_retry(|| self.inner.copy_opts(from, to, options.clone()))
            .await
    }

    async fn rename_opts(&self, from: &Path, to: &Path, options: RenameOptions) -> Result<()> {
        self.with_retry(|| self.inner.rename_opts(from, to, options.clone()))
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use futures::StreamExt;
    use object_store::ObjectStoreExt;
    use object_store::memory::InMemory;

    use super::*;
    use crate::credentials::tests::{CountingSource, FakeClock, cache_for};

    #[derive(Debug)]
    struct Body(String);

    impl fmt::Display for Body {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(&self.0)
        }
    }

    impl StdError for Body {}

    /// A store whose first `refusals` `get`s fail with `error`, then answers from memory.
    #[derive(Debug)]
    struct Scripted {
        memory: InMemory,
        refusals: AtomicUsize,
        calls: AtomicUsize,
        error: Mutex<Option<fn() -> object_store::Error>>,
    }

    impl fmt::Display for Scripted {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("Scripted")
        }
    }

    #[async_trait]
    impl ObjectStore for Scripted {
        async fn put_opts(&self, l: &Path, p: PutPayload, o: PutOptions) -> Result<PutResult> {
            self.memory.put_opts(l, p, o).await
        }
        async fn put_multipart_opts(
            &self,
            l: &Path,
            o: PutMultipartOptions,
        ) -> Result<Box<dyn MultipartUpload>> {
            self.memory.put_multipart_opts(l, o).await
        }
        async fn get_opts(&self, l: &Path, o: GetOptions) -> Result<GetResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.refusals.load(Ordering::SeqCst) > 0 {
                self.refusals.fetch_sub(1, Ordering::SeqCst);
                let make = self.error.lock().unwrap().unwrap();
                return Err(make());
            }
            self.memory.get_opts(l, o).await
        }
        fn delete_stream(
            &self,
            l: BoxStream<'static, Result<Path>>,
        ) -> BoxStream<'static, Result<Path>> {
            self.memory.delete_stream(l)
        }
        fn list(&self, p: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
            if self.refusals.load(Ordering::SeqCst) > 0 {
                self.refusals.fetch_sub(1, Ordering::SeqCst);
                let make = self.error.lock().unwrap().unwrap();
                return futures::stream::iter([Err(make())]).boxed();
            }
            self.memory.list(p)
        }
        async fn list_with_delimiter(&self, p: Option<&Path>) -> Result<ListResult> {
            self.memory.list_with_delimiter(p).await
        }
        async fn copy_opts(&self, f: &Path, t: &Path, o: CopyOptions) -> Result<()> {
            self.memory.copy_opts(f, t, o).await
        }
    }

    fn denied(body: &'static str) -> fn() -> object_store::Error {
        match body {
            "expired" => || object_store::Error::PermissionDenied {
                path: "k".into(),
                source: Box::new(Body(
                    "Server returned non-2xx status code: 403 Forbidden: \
                     <Error><Code>ExpiredToken</Code></Error>"
                        .into(),
                )),
            },
            _ => || object_store::Error::PermissionDenied {
                path: "k".into(),
                source: Box::new(Body(
                    "Server returned non-2xx status code: 403 Forbidden: \
                     <Error><Code>AccessDenied</Code></Error>"
                        .into(),
                )),
            },
        }
    }

    async fn setup(
        refusals: usize,
        error: fn() -> object_store::Error,
    ) -> (RefreshingStore, Arc<Scripted>, Arc<CountingSource>) {
        let clock = FakeClock::new();
        let source = CountingSource::new(&clock, Duration::from_secs(3600));
        let cache = cache_for(&source, &clock, Duration::ZERO);
        cache.credential().await.unwrap();
        let inner = Arc::new(Scripted {
            memory: InMemory::new(),
            refusals: AtomicUsize::new(refusals),
            calls: AtomicUsize::new(0),
            error: Mutex::new(Some(error)),
        });
        inner
            .memory
            .put(&Path::from("k"), PutPayload::from_static(b"v"))
            .await
            .unwrap();
        let provider = RefreshingProvider::<()>::from_cache(cache);
        let store = RefreshingStore::new(Arc::clone(&inner) as Arc<dyn ObjectStore>, &provider);
        (store, inner, source)
    }

    #[tokio::test]
    async fn an_expired_token_forces_one_refresh_and_exactly_one_retry() {
        let (store, inner, source) = setup(1, denied("expired")).await;
        let got = store.get(&Path::from("k")).await.unwrap();
        assert_eq!(got.bytes().await.unwrap().as_ref(), b"v");
        assert_eq!(inner.calls.load(Ordering::SeqCst), 2, "one retry");
        // The retry fetched a fresh credential on its next signature: the cache was stale.
        assert!(store.cache.generation() >= 1);
        store.cache.credential().await.unwrap();
        assert_eq!(
            source.fetches.load(Ordering::SeqCst),
            2,
            "one forced refresh"
        );
    }

    #[tokio::test]
    async fn a_stale_answer_in_a_listing_invalidates_without_replaying() {
        let (store, _, source) = setup(1, denied("expired")).await;
        let first: Vec<_> = store.list(None).collect().await;
        assert!(first[0].is_err(), "the stream is not replayed");
        // The next use fetches again, though the credential had an hour left.
        store.cache.credential().await.unwrap();
        assert_eq!(source.fetches.load(Ordering::SeqCst), 2);
        let second: Vec<_> = store.list(None).collect().await;
        assert!(second.iter().all(Result::is_ok));
    }

    #[tokio::test]
    async fn a_second_expired_answer_is_not_retried_again() {
        let (store, inner, _) = setup(5, denied("expired")).await;
        assert!(store.get(&Path::from("k")).await.is_err());
        assert_eq!(inner.calls.load(Ordering::SeqCst), 2, "exactly one retry");
    }

    #[tokio::test]
    async fn any_other_403_fails_with_no_refresh_and_no_retry() {
        let (store, inner, source) = setup(1, denied("access")).await;
        let e = store.get(&Path::from("k")).await.unwrap_err();
        assert!(matches!(e, object_store::Error::PermissionDenied { .. }));
        assert_eq!(inner.calls.load(Ordering::SeqCst), 1);
        store.cache.credential().await.unwrap();
        assert_eq!(source.fetches.load(Ordering::SeqCst), 1, "no refresh");
    }

    #[tokio::test]
    async fn every_operation_passes_through_to_the_inner_store() {
        let (store, _, _) = setup(0, denied("access")).await;
        let k = Path::from("k");
        assert!(store.to_string().contains("RefreshingStore(Scripted)"));
        assert!(format!("{store:?}").contains("RefreshingStore"));
        store
            .put(&Path::from("p/a"), PutPayload::from_static(b"abc"))
            .await
            .unwrap();
        store
            .put_multipart(&Path::from("p/m"))
            .await
            .unwrap()
            .abort()
            .await
            .ok();
        let ranges = store.get_ranges(&Path::from("p/a"), &[0..1, 1..3]).await;
        assert_eq!(ranges.unwrap()[1].as_ref(), b"bc");
        let listed: Vec<_> = store.list(Some(&Path::from("p"))).collect().await;
        assert!(listed.iter().all(Result::is_ok));
        let after: Vec<_> = store
            .list_with_offset(None, &Path::from("a"))
            .collect()
            .await;
        assert!(after.iter().all(Result::is_ok));
        let dir = store.list_with_delimiter(None).await.unwrap();
        assert!(!dir.objects.is_empty() || !dir.common_prefixes.is_empty());
        store.copy(&k, &Path::from("c")).await.unwrap();
        store
            .rename(&Path::from("c"), &Path::from("r"))
            .await
            .unwrap();
        let gone: Vec<_> = store
            .delete_stream(futures::stream::iter([Ok(Path::from("r"))]).boxed())
            .collect()
            .await;
        assert!(gone.iter().all(Result::is_ok));
    }

    #[test]
    fn only_stale_credential_answers_are_recognised() {
        let missing = object_store::Error::NotFound {
            path: "k".into(),
            source: Box::new(Body("gone".into())),
        };
        assert!(!is_stale_credential(&missing));
        let unauth = object_store::Error::Unauthenticated {
            path: "k".into(),
            source: Box::new(Body("no".into())),
        };
        assert!(is_stale_credential(&unauth));
        assert!(is_stale_credential(&denied("expired")()));
        assert!(!is_stale_credential(&denied("access")()));
    }

    #[tokio::test]
    async fn a_failed_credential_refresh_is_not_retried() {
        let (store, inner, _) = setup(1, || object_store::Error::Unauthenticated {
            path: "s3://b".into(),
            source: Box::new(CredentialRefreshError::for_test("down")),
        })
        .await;
        assert!(store.get(&Path::from("k")).await.is_err());
        assert_eq!(inner.calls.load(Ordering::SeqCst), 1);
    }
}
