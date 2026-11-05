// SPDX-License-Identifier: Apache-2.0
//! Short-lived, per-location credentials (ADR 0044, design note section 2).
//!
//! A [`CredentialSource`] hands out a [`ScopedCredential`] for a [`CredentialScope`] (a
//! location such as `s3://bucket/warehouse/db/t/`, never a credential). A [`RefreshingProvider`]
//! caches the last credential, refreshes it before it expires, fetches once however many
//! requests are waiting, and never serves an expired credential. Every secret is a
//! [`SecretString`] from the moment it is built; `Debug` prints the kind and the expiry only.
//!
//! ```
//! use std::time::{Duration, SystemTime};
//! use krishiv_store::{CredentialScope, ScopedCredential};
//!
//! let scope = CredentialScope::new("s3://bucket/warehouse/db/t/")?;
//! assert_eq!(scope.to_string(), "s3://bucket/warehouse/db/t");
//! assert!(scope.contains_key("warehouse/db/t/part-0.parquet"));
//! // The prefix matches on path segments: `t1` is not under `t10`.
//! assert!(!CredentialScope::new("s3://bucket/w/t1")?.contains_key("w/t10/f"));
//!
//! let credential = ScopedCredential::s3(
//!     "AKIA...",
//!     "secret",
//!     Some("session-token".to_owned()),
//!     Some(SystemTime::now() + Duration::from_secs(3600)),
//! );
//! // Neither the key id nor the secret is printed.
//! assert!(!format!("{credential:?}").contains("AKIA"));
//! # Ok::<(), krishiv_common::error::KrishivError>(())
//! ```

use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use krishiv_common::error::{KrishivError, KrishivInvalidConfigValue};
use krishiv_common::secret::{REDACTED, SecretString};
use object_store::path::Path;

use crate::StoreId;
use crate::error::{Result, unsupported};
use crate::stores::{ObjectScheme, parse_object_url};

/// How long before its expiry a credential is refreshed (design note section 2).
pub const DEFAULT_REFRESH_SKEW: Duration = Duration::from_secs(300);

/// A location a credential is good for: a bucket (or container) and a key prefix in it.
///
/// The prefix matches on path segments, so `warehouse/t1/` does not cover `warehouse/t10/`.
#[derive(Clone, PartialEq, Eq)]
pub struct CredentialScope {
    store_id: StoreId,
    scheme: ObjectScheme,
    bucket: String,
    prefix: Path,
}

impl CredentialScope {
    /// The scope of `location` (`s3://bucket/warehouse/db/t/`, `gs://..`, `abfss://..`). The
    /// whole bucket is `s3://bucket`.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a location that does not parse, or a scheme without
    /// credentials of its own (`http(s)://`, `hf://`).
    pub fn new(location: &str) -> Result<Self> {
        let url = parse_object_url(location)?;
        if !matches!(
            url.scheme,
            ObjectScheme::S3 | ObjectScheme::Gcs | ObjectScheme::Azure
        ) {
            return Err(unsupported(format!(
                "credential sources for {} locations",
                url.scheme_text
            )));
        }
        Ok(Self {
            store_id: url.store_id,
            scheme: url.scheme,
            bucket: url.bucket,
            prefix: url.key,
        })
    }

    /// The scheme of the location.
    #[must_use]
    pub fn scheme(&self) -> ObjectScheme {
        self.scheme
    }

    /// The bucket (S3, GCS) or container (Azure).
    #[must_use]
    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    /// The key prefix within the bucket, without a leading or trailing `/` (empty for the
    /// whole bucket).
    #[must_use]
    pub fn prefix(&self) -> &str {
        self.prefix.as_ref()
    }

    /// Whether `key` (within this scope's bucket) is under the prefix, on path segments.
    #[must_use]
    pub fn contains_key(&self, key: &str) -> bool {
        Path::parse(key).is_ok_and(|key| self.covers(&key))
    }

    /// The store id of the location's bucket, without any scope suffix.
    pub(crate) fn store_id(&self) -> &StoreId {
        &self.store_id
    }

    /// Whether `key` is under the prefix, on path segments.
    pub(crate) fn covers(&self, key: &Path) -> bool {
        key.prefix_matches(&self.prefix)
    }

    /// How specific the scope is: the number of prefix segments.
    pub(crate) fn depth(&self) -> usize {
        self.prefix.parts().count()
    }
}

impl fmt::Display for CredentialScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.store_id)?;
        if !self.prefix.as_ref().is_empty() {
            write!(f, "/{}", self.prefix.as_ref())?;
        }
        Ok(())
    }
}

impl fmt::Debug for CredentialScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CredentialScope({self})")
    }
}

#[derive(Clone)]
#[cfg_attr(
    not(feature = "cloud"),
    expect(dead_code, reason = "the secrets are read only by the cloud providers")
)]
enum Kind {
    S3 {
        key_id: SecretString,
        secret: SecretString,
        token: Option<SecretString>,
    },
    Gcs {
        bearer: SecretString,
    },
    AzureSas {
        query: SecretString,
    },
    AzureBearer {
        bearer: SecretString,
    },
}

/// A credential for one scope, with the time it stops working. Built by a
/// [`CredentialSource`]; its secrets are never printed.
#[derive(Clone)]
pub struct ScopedCredential {
    kind: Kind,
    expires_at: Option<SystemTime>,
}

impl ScopedCredential {
    /// An S3 credential: access key id, secret key and optional session token (STS).
    #[must_use]
    pub fn s3(
        key_id: impl Into<String>,
        secret: impl Into<String>,
        token: Option<String>,
        expires_at: Option<SystemTime>,
    ) -> Self {
        Self {
            kind: Kind::S3 {
                key_id: SecretString::new(key_id.into()),
                secret: SecretString::new(secret.into()),
                token: token.map(SecretString::new),
            },
            expires_at,
        }
    }

    /// A GCS OAuth2 bearer token.
    #[must_use]
    pub fn gcs(bearer: impl Into<String>, expires_at: Option<SystemTime>) -> Self {
        Self {
            kind: Kind::Gcs {
                bearer: SecretString::new(bearer.into()),
            },
            expires_at,
        }
    }

    /// An Azure shared access signature as its query string (`sv=..&sig=..`, a leading `?` is
    /// ignored).
    #[must_use]
    pub fn azure_sas(query: impl Into<String>, expires_at: Option<SystemTime>) -> Self {
        Self {
            kind: Kind::AzureSas {
                query: SecretString::new(query.into()),
            },
            expires_at,
        }
    }

    /// An Azure AD bearer token.
    #[must_use]
    pub fn azure_bearer(bearer: impl Into<String>, expires_at: Option<SystemTime>) -> Self {
        Self {
            kind: Kind::AzureBearer {
                bearer: SecretString::new(bearer.into()),
            },
            expires_at,
        }
    }

    /// When the credential stops working (`None`: it does not expire).
    #[must_use]
    pub fn expires_at(&self) -> Option<SystemTime> {
        self.expires_at
    }

    fn kind_name(&self) -> &'static str {
        match self.kind {
            Kind::S3 { .. } => "s3",
            Kind::Gcs { .. } => "gcs",
            Kind::AzureSas { .. } => "azure-sas",
            Kind::AzureBearer { .. } => "azure-bearer",
        }
    }
}

impl fmt::Debug for ScopedCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The kind and the expiry only: no key id, secret, token or bearer.
        f.debug_struct("ScopedCredential")
            .field("kind", &self.kind_name())
            .field("expires_at", &self.expires_at)
            .finish_non_exhaustive()
    }
}

/// Where credentials for a scope come from: an STS call, a catalog's per-table credentials
/// endpoint, a token service. `fetch` is called again whenever the last credential nears its
/// expiry.
///
/// An error from `fetch` fails the request that needed the credential; its text is printed,
/// so it must never contain a secret.
#[async_trait]
pub trait CredentialSource: fmt::Debug + Send + Sync {
    /// A credential good for `scope`.
    ///
    /// # Errors
    /// Any [`KrishivError`]; the request that needed the credential fails with it.
    async fn fetch(&self, scope: &CredentialScope) -> Result<ScopedCredential>;
}

/// The time source of the credential cache; tests replace it.
pub(crate) trait Clock: Send + Sync + fmt::Debug {
    fn now(&self) -> SystemTime;
}

#[derive(Debug)]
struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}

/// The error an object_store request carries when no credential could be had (the refresh
/// failed, or the source returned an expired credential). Never carries a secret.
#[derive(Debug)]
pub(crate) struct CredentialRefreshError {
    scope: String,
    reason: String,
}

impl fmt::Display for CredentialRefreshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no valid credential for {}: {}", self.scope, self.reason)
    }
}

impl std::error::Error for CredentialRefreshError {}

#[cfg(test)]
impl CredentialRefreshError {
    pub(crate) fn for_test(reason: &str) -> Self {
        Self {
            scope: "s3://b".to_owned(),
            reason: reason.to_owned(),
        }
    }
}

#[derive(Default)]
struct CacheState {
    current: Option<Arc<ScopedCredential>>,
    /// Bumped by every fetch attempt that finished (success or failure).
    generation: u64,
    /// The failure of the attempt that produced `generation`, when it failed.
    failure: Option<String>,
    /// The cached credential was refused by the store: fetch again on the next use.
    stale: bool,
}

/// The shared state behind a [`RefreshingProvider`]: the source, the scope and the cached
/// credential. One per credential scope; every provider and wrapper store of the scope shares it.
pub(crate) struct CredentialCache {
    scope: CredentialScope,
    source: Arc<dyn CredentialSource>,
    skew: Duration,
    clock: Arc<dyn Clock>,
    state: Mutex<CacheState>,
    /// Held while a fetch is in flight: callers that arrive meanwhile wait on it.
    flight: tokio::sync::Mutex<()>,
}

impl fmt::Debug for CredentialCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CredentialCache")
            .field("scope", &self.scope)
            .field("skew", &self.skew)
            .finish_non_exhaustive()
    }
}

impl CredentialCache {
    pub(crate) fn new(
        scope: CredentialScope,
        source: Arc<dyn CredentialSource>,
        skew: Duration,
        clock: Arc<dyn Clock>,
    ) -> Arc<Self> {
        Arc::new(Self {
            scope,
            source,
            skew,
            clock,
            state: Mutex::new(CacheState::default()),
            flight: tokio::sync::Mutex::new(()),
        })
    }

    /// A cache on the system clock.
    pub(crate) fn system(
        scope: CredentialScope,
        source: Arc<dyn CredentialSource>,
        skew: Duration,
    ) -> Arc<Self> {
        Self::new(scope, source, skew, Arc::new(SystemClock))
    }

    pub(crate) fn scope(&self) -> &CredentialScope {
        &self.scope
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CacheState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The generation of the cached credential: what a request started under.
    pub(crate) fn generation(&self) -> u64 {
        self.lock().generation
    }

    /// The store refused a request signed under `generation` as stale: the next use fetches
    /// again, unless somebody already did (a burst of failures forces one refresh, not many).
    pub(crate) fn invalidate(&self, generation: u64) {
        let mut state = self.lock();
        if state.generation == generation {
            state.stale = true;
        }
    }

    fn fresh(&self, state: &CacheState) -> Option<Arc<ScopedCredential>> {
        if state.stale {
            return None;
        }
        let current = state.current.as_ref()?;
        let usable = match current.expires_at {
            None => true,
            Some(expiry) => self
                .clock
                .now()
                .checked_add(self.skew)
                .is_some_and(|horizon| horizon < expiry),
        };
        usable.then(|| Arc::clone(current))
    }

    /// A credential that is neither expired nor within the skew of its expiry.
    ///
    /// # Errors
    /// A [`CredentialRefreshError`] when the fetch failed or the source returned a credential
    /// that is already expired.
    pub(crate) async fn credential(&self) -> Result<Arc<ScopedCredential>, CredentialRefreshError> {
        let seen = {
            let state = self.lock();
            if let Some(ok) = self.fresh(&state) {
                return Ok(ok);
            }
            state.generation
        };
        // Single flight: one caller fetches, the others wait for its answer.
        let _flight = self.flight.lock().await;
        {
            let state = self.lock();
            if state.generation != seen {
                // Somebody fetched while we waited: use their answer, success or failure.
                if let Some(ok) = self.fresh(&state) {
                    return Ok(ok);
                }
                if let Some(reason) = &state.failure {
                    return Err(self.error(reason.clone()));
                }
            }
        }
        let result = self.source.fetch(&self.scope).await;
        let mut state = self.lock();
        state.generation += 1;
        state.stale = false;
        match result {
            Ok(credential) => {
                let expired = credential
                    .expires_at
                    .is_some_and(|expiry| expiry <= self.clock.now());
                if expired {
                    let reason = "the credential source returned an expired credential".to_owned();
                    state.failure = Some(reason.clone());
                    state.current = None;
                    return Err(self.error(reason));
                }
                state.failure = None;
                let credential = Arc::new(credential);
                state.current = Some(Arc::clone(&credential));
                Ok(credential)
            }
            Err(e) => {
                let reason = format!("the credential source failed: {e}");
                state.failure = Some(reason.clone());
                state.current = None;
                Err(self.error(reason))
            }
        }
    }

    fn error(&self, reason: String) -> CredentialRefreshError {
        CredentialRefreshError {
            scope: self.scope.to_string(),
            reason,
        }
    }
}

/// An `object_store::CredentialProvider` over a [`CredentialSource`], for the credential type
/// `C` (`AwsCredential`, `GcpCredential` or `AzureCredential`, with the `cloud` feature).
///
/// The credential is cached, refreshed [`DEFAULT_REFRESH_SKEW`] before it expires (single
/// flight: 50 concurrent requests cause one fetch) and never served once expired. A failed
/// refresh fails the request, naming the scope and never a secret.
pub struct RefreshingProvider<C> {
    pub(crate) cache: Arc<CredentialCache>,
    credential: PhantomData<fn() -> C>,
}

impl<C> RefreshingProvider<C> {
    /// A provider for `scope` fed by `source`, with the default skew.
    #[must_use]
    pub fn new(scope: CredentialScope, source: Arc<dyn CredentialSource>) -> Self {
        Self::with_skew(scope, source, DEFAULT_REFRESH_SKEW)
    }

    /// As [`Self::new`], refreshing `skew` before the expiry.
    #[must_use]
    pub fn with_skew(
        scope: CredentialScope,
        source: Arc<dyn CredentialSource>,
        skew: Duration,
    ) -> Self {
        Self::from_cache(CredentialCache::new(
            scope,
            source,
            skew,
            Arc::new(SystemClock),
        ))
    }

    pub(crate) fn from_cache(cache: Arc<CredentialCache>) -> Self {
        Self {
            cache,
            credential: PhantomData,
        }
    }
}

impl<C> Clone for RefreshingProvider<C> {
    fn clone(&self) -> Self {
        Self::from_cache(Arc::clone(&self.cache))
    }
}

impl<C> fmt::Debug for RefreshingProvider<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RefreshingProvider")
            .field("scope", self.cache.scope())
            .finish_non_exhaustive()
    }
}

/// The error a provider hands object_store: `Unauthenticated`, which the scheduler does not
/// retry; [`RefreshingStore`](crate::RefreshingStore) recognizes its source and does not retry
/// it either.
#[cfg(feature = "cloud")]
fn provider_error(e: CredentialRefreshError) -> object_store::Error {
    object_store::Error::Unauthenticated {
        path: e.scope.clone(),
        source: Box::new(e),
    }
}

/// A credential of the wrong kind for the store being built (an S3 store given a GCS bearer).
#[cfg(feature = "cloud")]
fn wrong_kind(cache: &CredentialCache, credential: &ScopedCredential) -> object_store::Error {
    provider_error(cache.error(format!(
        "the credential source returned a {} credential for a {} location",
        credential.kind_name(),
        cache.scope.scheme
    )))
}

#[cfg(feature = "cloud")]
#[async_trait]
impl object_store::CredentialProvider for RefreshingProvider<object_store::aws::AwsCredential> {
    type Credential = object_store::aws::AwsCredential;

    async fn get_credential(&self) -> object_store::Result<Arc<Self::Credential>> {
        let credential = self.cache.credential().await.map_err(provider_error)?;
        match &credential.kind {
            // The plaintext lives only inside object_store, for the life of this `Arc`.
            Kind::S3 {
                key_id,
                secret,
                token,
            } => Ok(Arc::new(object_store::aws::AwsCredential {
                key_id: key_id.expose_secret().to_owned(),
                secret_key: secret.expose_secret().to_owned(),
                token: token.as_ref().map(|t| t.expose_secret().to_owned()),
            })),
            _ => Err(wrong_kind(&self.cache, &credential)),
        }
    }
}

#[cfg(feature = "cloud")]
#[async_trait]
impl object_store::CredentialProvider for RefreshingProvider<object_store::gcp::GcpCredential> {
    type Credential = object_store::gcp::GcpCredential;

    async fn get_credential(&self) -> object_store::Result<Arc<Self::Credential>> {
        let credential = self.cache.credential().await.map_err(provider_error)?;
        match &credential.kind {
            Kind::Gcs { bearer } => Ok(Arc::new(object_store::gcp::GcpCredential {
                bearer: bearer.expose_secret().to_owned(),
            })),
            _ => Err(wrong_kind(&self.cache, &credential)),
        }
    }
}

#[cfg(feature = "cloud")]
#[async_trait]
impl object_store::CredentialProvider for RefreshingProvider<object_store::azure::AzureCredential> {
    type Credential = object_store::azure::AzureCredential;

    async fn get_credential(&self) -> object_store::Result<Arc<Self::Credential>> {
        use object_store::azure::AzureCredential;
        let credential = self.cache.credential().await.map_err(provider_error)?;
        match &credential.kind {
            Kind::AzureSas { query } => {
                let pairs = sas_query_pairs(query)
                    .map_err(|reason| provider_error(self.cache.error(reason)))?;
                Ok(Arc::new(AzureCredential::SASToken(pairs)))
            }
            Kind::AzureBearer { bearer } => Ok(Arc::new(AzureCredential::BearerToken(
                bearer.expose_secret().to_owned(),
            ))),
            _ => Err(wrong_kind(&self.cache, &credential)),
        }
    }
}

/// Splits a SAS query (`sv=..&sig=..`, a leading `?` ignored) into pairs. The complaint never
/// contains the token.
#[cfg(feature = "cloud")]
pub(crate) fn sas_query_pairs(secret: &SecretString) -> Result<Vec<(String, String)>, String> {
    let token = secret.expose_secret().trim();
    let token = token.strip_prefix('?').unwrap_or(token);
    let mut pairs = Vec::new();
    for part in token.split('&') {
        match part.split_once('=') {
            Some((name, value)) if !name.is_empty() && !value.is_empty() => {
                pairs.push((name.to_owned(), value.to_owned()));
            }
            _ => return Err("the Azure SAS token is not `name=value&...`".to_owned()),
        }
    }
    if pairs.is_empty() {
        return Err("the Azure SAS token is not `name=value&...`".to_owned());
    }
    Ok(pairs)
}

/// `KRISHIV_INVALID_CONFIG_VALUE` for a per-read credential key on a scoped location: two
/// credential sources for one read. Names the key, never the value.
pub(crate) fn two_credential_sources(key: &str, scope: &CredentialScope) -> KrishivError {
    KrishivInvalidConfigValue {
        key: format!("\"{key}\""),
        value: format!("'{REDACTED}'"),
        reason: format!(
            "{scope} gets its credentials from a credential source; a per-read credential \
             would be a second source for the same read"
        ),
    }
    .into()
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    const CANARY: &str = "Canary-5e1d2b";

    /// A clock the test moves by hand.
    #[derive(Debug)]
    pub(crate) struct FakeClock(Mutex<SystemTime>);

    impl FakeClock {
        pub(crate) fn new() -> Arc<Self> {
            Arc::new(Self(Mutex::new(
                SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000),
            )))
        }

        pub(crate) fn advance(&self, by: Duration) {
            *self.0.lock().unwrap() += by;
        }
    }

    impl Clock for FakeClock {
        fn now(&self) -> SystemTime {
            *self.0.lock().unwrap()
        }
    }

    /// A source counting its fetches; each credential lives `ttl` from the clock's now.
    #[derive(Debug)]
    pub(crate) struct CountingSource {
        pub(crate) fetches: AtomicUsize,
        pub(crate) clock: Arc<FakeClock>,
        pub(crate) ttl: Duration,
        pub(crate) fail: std::sync::atomic::AtomicBool,
        pub(crate) delay: Duration,
    }

    impl CountingSource {
        pub(crate) fn new(clock: &Arc<FakeClock>, ttl: Duration) -> Arc<Self> {
            Arc::new(Self {
                fetches: AtomicUsize::new(0),
                clock: Arc::clone(clock),
                ttl,
                fail: std::sync::atomic::AtomicBool::new(false),
                delay: Duration::ZERO,
            })
        }
    }

    #[async_trait]
    impl CredentialSource for CountingSource {
        async fn fetch(&self, _scope: &CredentialScope) -> Result<ScopedCredential> {
            let n = self.fetches.fetch_add(1, Ordering::SeqCst);
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            if self.fail.load(Ordering::SeqCst) {
                return Err(unsupported("the token service is down"));
            }
            Ok(ScopedCredential::s3(
                format!("{CANARY}-key-{n}"),
                format!("{CANARY}-secret"),
                Some(format!("{CANARY}-token")),
                Some(self.clock.now() + self.ttl),
            ))
        }
    }

    pub(crate) fn cache_for(
        source: &Arc<CountingSource>,
        clock: &Arc<FakeClock>,
        skew: Duration,
    ) -> Arc<CredentialCache> {
        CredentialCache::new(
            CredentialScope::new("s3://bucket/warehouse/t1/").unwrap(),
            Arc::clone(source) as Arc<dyn CredentialSource>,
            skew,
            Arc::clone(clock) as Arc<dyn Clock>,
        )
    }

    #[tokio::test]
    async fn a_request_after_expiry_fetches_again() {
        let clock = FakeClock::new();
        let source = CountingSource::new(&clock, Duration::from_secs(60));
        let cache = cache_for(&source, &clock, Duration::ZERO);
        cache.credential().await.unwrap();
        cache.credential().await.unwrap();
        assert_eq!(source.fetches.load(Ordering::SeqCst), 1);
        clock.advance(Duration::from_secs(61));
        cache.credential().await.unwrap();
        assert_eq!(source.fetches.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_request_inside_the_skew_window_refreshes_early() {
        let clock = FakeClock::new();
        let source = CountingSource::new(&clock, Duration::from_secs(600));
        let cache = cache_for(&source, &clock, Duration::from_secs(300));
        cache.credential().await.unwrap();
        clock.advance(Duration::from_secs(200));
        cache.credential().await.unwrap();
        assert_eq!(
            source.fetches.load(Ordering::SeqCst),
            1,
            "outside the window"
        );
        clock.advance(Duration::from_secs(150));
        cache.credential().await.unwrap();
        assert_eq!(
            source.fetches.load(Ordering::SeqCst),
            2,
            "inside the window"
        );
    }

    #[tokio::test]
    async fn fifty_concurrent_requests_cause_one_fetch() {
        let clock = FakeClock::new();
        let mut source = CountingSource::new(&clock, Duration::from_secs(3600));
        Arc::get_mut(&mut source).unwrap().delay = Duration::from_millis(50);
        let cache = cache_for(&source, &clock, Duration::ZERO);
        let tasks: Vec<_> = (0..50)
            .map(|_| {
                let cache = Arc::clone(&cache);
                tokio::spawn(async move { cache.credential().await })
            })
            .collect();
        for task in tasks {
            task.await.unwrap().unwrap();
        }
        assert_eq!(source.fetches.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_failed_refresh_fails_the_request_once_for_all_waiters() {
        let clock = FakeClock::new();
        let mut source = CountingSource::new(&clock, Duration::from_secs(3600));
        Arc::get_mut(&mut source).unwrap().delay = Duration::from_millis(50);
        source.fail.store(true, Ordering::SeqCst);
        let cache = cache_for(&source, &clock, Duration::ZERO);
        let tasks: Vec<_> = (0..20)
            .map(|_| {
                let cache = Arc::clone(&cache);
                tokio::spawn(async move { cache.credential().await })
            })
            .collect();
        for task in tasks {
            let e = task.await.unwrap().unwrap_err();
            assert!(e.to_string().contains("s3://bucket/warehouse/t1"), "{e}");
        }
        assert_eq!(
            source.fetches.load(Ordering::SeqCst),
            1,
            "the waiters share the failure"
        );
        // The next request tries again, and recovers.
        source.fail.store(false, Ordering::SeqCst);
        cache.credential().await.unwrap();
        assert_eq!(source.fetches.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn an_expired_credential_is_never_served() {
        let clock = FakeClock::new();
        // The source hands out credentials that are already over.
        let source = CountingSource::new(&clock, Duration::ZERO);
        let cache = cache_for(&source, &clock, Duration::ZERO);
        let e = cache.credential().await.unwrap_err();
        assert!(e.to_string().contains("expired"), "{e}");
    }

    #[tokio::test]
    async fn invalidating_forces_one_refresh_for_a_burst() {
        let clock = FakeClock::new();
        let source = CountingSource::new(&clock, Duration::from_secs(3600));
        let cache = cache_for(&source, &clock, Duration::ZERO);
        cache.credential().await.unwrap();
        let generation = cache.generation();
        // Three requests signed under the same credential are refused as stale.
        for _ in 0..3 {
            cache.invalidate(generation);
        }
        cache.credential().await.unwrap();
        // A late refusal of the old credential does not force another fetch.
        cache.invalidate(generation);
        cache.credential().await.unwrap();
        assert_eq!(source.fetches.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn nothing_prints_a_secret_or_a_key_id() {
        let clock = FakeClock::new();
        let source = CountingSource::new(&clock, Duration::from_secs(3600));
        let cache = cache_for(&source, &clock, Duration::ZERO);
        let credential = cache.credential().await.unwrap();
        let provider = RefreshingProvider::<()>::from_cache(Arc::clone(&cache));
        let shown = format!(
            "{credential:?} {credential:#?} {cache:?} {provider:?} {:?} {}",
            cache.scope(),
            cache.scope()
        );
        assert!(!shown.contains(CANARY), "{shown}");
        source.fail.store(true, Ordering::SeqCst);
        clock.advance(Duration::from_secs(7200));
        let e = cache.credential().await.unwrap_err();
        assert!(!format!("{e} {e:?}").contains(CANARY));
    }

    #[cfg(feature = "cloud")]
    #[tokio::test]
    async fn providers_convert_each_kind_and_refuse_the_wrong_one() {
        use object_store::CredentialProvider;
        use object_store::aws::AwsCredential;
        use object_store::azure::AzureCredential;
        use object_store::gcp::GcpCredential;

        #[derive(Debug)]
        struct One(ScopedCredential);
        #[async_trait]
        impl CredentialSource for One {
            async fn fetch(&self, _: &CredentialScope) -> Result<ScopedCredential> {
                Ok(self.0.clone())
            }
        }
        fn provider<C>(credential: ScopedCredential, scope: &str) -> RefreshingProvider<C> {
            RefreshingProvider::new(
                CredentialScope::new(scope).unwrap(),
                Arc::new(One(credential)),
            )
        }
        let aws = provider::<AwsCredential>(
            ScopedCredential::s3("kid", "sec", Some("tok".into()), None),
            "s3://b/p",
        );
        let got = aws.get_credential().await.unwrap();
        assert_eq!(
            (
                got.key_id.as_str(),
                got.secret_key.as_str(),
                got.token.as_deref()
            ),
            ("kid", "sec", Some("tok"))
        );
        let gcp = provider::<GcpCredential>(ScopedCredential::gcs("bearer", None), "gs://b/p");
        assert_eq!(gcp.get_credential().await.unwrap().bearer, "bearer");
        let sas = provider::<AzureCredential>(
            ScopedCredential::azure_sas("?sv=1&sig=abc", None),
            "az://c@acct.dfs.core.windows.net/p",
        );
        assert_eq!(
            *sas.get_credential().await.unwrap(),
            AzureCredential::SASToken(vec![
                ("sv".into(), "1".into()),
                ("sig".into(), "abc".into())
            ])
        );
        let bearer = provider::<AzureCredential>(
            ScopedCredential::azure_bearer("tok", None),
            "az://c@acct.dfs.core.windows.net/p",
        );
        assert_eq!(
            *bearer.get_credential().await.unwrap(),
            AzureCredential::BearerToken("tok".into())
        );
        // An S3 credential handed to a GCS location is refused, naming the kinds only.
        let wrong =
            provider::<GcpCredential>(ScopedCredential::s3("kid", "sec", None, None), "gs://b/p");
        let e = wrong.get_credential().await.unwrap_err().to_string();
        assert!(e.contains("s3 credential") && !e.contains("sec"), "{e}");
    }

    #[test]
    fn scopes_match_on_path_segments_and_reject_web_schemes() {
        let t1 = CredentialScope::new("s3://b/warehouse/t1/").unwrap();
        assert!(t1.contains_key("warehouse/t1/data/f.parquet"));
        assert!(t1.contains_key("warehouse/t1"));
        assert!(!t1.contains_key("warehouse/t10/f"));
        assert!(!t1.contains_key("warehouse"));
        let whole = CredentialScope::new("s3://b").unwrap();
        assert!(whole.contains_key("anything/at/all"));
        assert_eq!(whole.depth(), 0);
        assert_eq!(t1.depth(), 2);
        for web in ["https://example.com/x", "hf://datasets/org/name"] {
            assert!(CredentialScope::new(web).is_err(), "{web}");
        }
    }
}
