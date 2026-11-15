// SPDX-License-Identifier: Apache-2.0
//! [`SharedHttpClients`]: one HTTP client per set of client options, shared by the stores of
//! many registries (design note `design/P2-store-shared-http-client.md`).
//!
//! object_store's default connector builds a new `reqwest::Client` on every connect. That
//! means a TLS root store parsed again and an empty connection pool for every store of every
//! query (6-62 ms per remote query on a loaded host). A registry given a [`SharedHttpClients`]
//! makes its S3, GCS, Azure and https stores connect through it instead. Clients are cached
//! by a fingerprint of their [`ClientOptions`] config values. Those builders set no client
//! field the config values leave out (default headers, root certificates, a DNS resolver).
//! The Hugging Face store sends its token as a default header, so it never uses the cache.
//!
//! Connections carry no credentials: S3, GCS and Azure sign or authorize each request, and
//! the keys stay in the store. So one pool serves every principal, as a shared proxy would.
//! Pooled connections are tasks on the runtime that polled the request that opened them (the
//! I/O runtime for scheduled fetches, the caller's runtime for heads, lists and writes). So a
//! cache must not outlive any runtime its requests run on: the engine's `IoScheduler` owns
//! one, and its compute and I/O runtimes live as long as it does. Never make one per process.
//! (hyper drops a dead idle connection when it is checked out; only a request in flight when
//! its runtime stops fails.)

#[cfg(feature = "cloud")]
use std::collections::HashMap;
#[cfg(feature = "cloud")]
use std::sync::{Arc, Mutex, PoisonError};

#[cfg(feature = "cloud")]
use object_store::client::{ClientConfigKey, ClientOptions, HttpClient, HttpConnector};

#[cfg(feature = "cloud")]
pub(crate) mod policy;
#[cfg(feature = "cloud")]
use policy::PolicyConnector;

/// The most distinct option sets one cache holds; past it, clients are built uncached.
#[cfg(feature = "cloud")]
const MAX_CLIENTS: usize = 64;

/// HTTP clients shared by the stores of many [`StoreRegistry`]s, one per set of client
/// options (see the module docs). Cloning is cheap and shares the cache.
///
/// Without the `cloud` feature it holds nothing: no store is built.
///
/// [`StoreRegistry`]: crate::StoreRegistry
#[derive(Clone, Default)]
pub struct SharedHttpClients {
    #[cfg(feature = "cloud")]
    inner: Arc<Inner>,
}

#[cfg(feature = "cloud")]
#[derive(Default)]
struct Inner {
    clients: Mutex<HashMap<Fingerprint, HttpClient>>,
    /// Builds a client on a miss: object_store's reqwest connector, or a test's counter.
    build: Option<Arc<dyn HttpConnector>>,
    /// The egress policy's connector (server mode, design P3-T067 §5.2): every client is a
    /// policy client, its fingerprint carries the policy component.
    egress: Option<PolicyConnector>,
    /// The same policy without the provider literals, for stores with no ambient credential
    /// providers (`http(s)://`, `hf://`): a data read of a metadata address is refused
    /// (review 58a1 B2).
    egress_data: Option<PolicyConnector>,
}

impl std::fmt::Debug for SharedHttpClients {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never the fingerprints: a proxy URL may carry userinfo.
        let mut s = f.debug_struct("SharedHttpClients");
        #[cfg(feature = "cloud")]
        s.field("clients", &self.len());
        s.finish_non_exhaustive()
    }
}

impl SharedHttpClients {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty cache whose misses are built by `connector` (tests count builds with it).
    #[cfg(feature = "cloud")]
    #[must_use]
    pub fn with_connector(connector: Arc<dyn HttpConnector>) -> Self {
        Self {
            inner: Arc::new(Inner {
                clients: Mutex::new(HashMap::new()),
                build: Some(connector),
                egress: None,
                egress_data: None,
            }),
        }
    }

    /// An empty cache whose clients are all policy clients of `egress` (server mode; design
    /// P3-T067 §5.2). Only an enforcing scope makes them: an `allow` policy builds plain clients.
    #[cfg(feature = "cloud")]
    #[must_use]
    pub fn with_egress(egress: PolicyConnector) -> Self {
        Self {
            inner: Arc::new(Inner {
                clients: Mutex::new(HashMap::new()),
                build: None,
                egress_data: egress
                    .scope()
                    .enforced()
                    .then(|| egress.clone().without_provider_literals()),
                egress: egress.scope().enforced().then_some(egress),
            }),
        }
    }

    /// Whether the clients of this cache apply an enforcing egress policy. A store that builds
    /// its own client otherwise (`hf://`) connects through the cache when this holds.
    #[must_use]
    pub fn enforces_egress(&self) -> bool {
        #[cfg(feature = "cloud")]
        {
            self.inner.egress.is_some()
        }
        #[cfg(not(feature = "cloud"))]
        {
            false
        }
    }

    /// How many clients the cache holds.
    #[cfg(feature = "cloud")]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner
            .clients
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Whether the cache holds no client.
    #[cfg(feature = "cloud")]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The connector for stores without ambient credential providers: the data-only policy
    /// clients when the cache enforces the egress policy, else the cache itself.
    #[cfg(feature = "cloud")]
    #[must_use]
    pub fn data_only(&self) -> DataOnlyClients {
        DataOnlyClients(self.clone())
    }

    /// Checks a configured store endpoint against the egress rule (no provider exemption): `Ok`
    /// without an enforcing policy.
    ///
    /// # Errors
    /// The rule refuses the endpoint's address literal.
    #[cfg(feature = "cloud")]
    pub fn check_endpoint(&self, endpoint: &str) -> Result<(), crate::egress::EgressRefusal> {
        match &self.inner.egress {
            Some(egress) => egress.scope().check_endpoint(endpoint),
            None => Ok(()),
        }
    }

    #[cfg(feature = "cloud")]
    fn build(
        &self,
        options: &ClientOptions,
        egress: Option<&PolicyConnector>,
    ) -> object_store::Result<HttpClient> {
        match (egress, &self.inner.build) {
            (Some(egress), _) => egress.connect(options),
            (None, Some(connector)) => connector.connect(options),
            (None, None) => object_store::client::ReqwestConnector::default().connect(options),
        }
    }
}

/// Every client config key, in a fixed order. `ClientConfigKey` is `#[non_exhaustive]`, so a
/// key added by an object_store upgrade is not caught here: re-list it when bumping the pin
/// (Cargo.toml). Until our builders set such a key, it comes only from the process
/// environment and is equal for every store.
#[cfg(feature = "cloud")]
const KEYS: [ClientConfigKey; 20] = [
    ClientConfigKey::AllowHttp,
    ClientConfigKey::AllowInvalidCertificates,
    ClientConfigKey::NoSystemCertificates,
    ClientConfigKey::ConnectTimeout,
    ClientConfigKey::ReadTimeout,
    ClientConfigKey::DefaultContentType,
    ClientConfigKey::Http1Only,
    ClientConfigKey::Http2KeepAliveInterval,
    ClientConfigKey::Http2KeepAliveTimeout,
    ClientConfigKey::Http2KeepAliveWhileIdle,
    ClientConfigKey::Http2MaxFrameSize,
    ClientConfigKey::Http2Only,
    ClientConfigKey::PoolIdleTimeout,
    ClientConfigKey::PoolMaxIdlePerHost,
    ClientConfigKey::ProxyUrl,
    ClientConfigKey::ProxyCaCertificate,
    ClientConfigKey::ProxyExcludes,
    ClientConfigKey::RandomizeAddresses,
    ClientConfigKey::Timeout,
    ClientConfigKey::UserAgent,
];

/// A cache key: the egress policy component (§5.2: generation, exemption set, redirects, proxy),
/// then each listed key's config value, or none.
#[cfg(feature = "cloud")]
type Fingerprint = (Option<String>, Vec<(ClientConfigKey, Option<String>)>);

/// The cache key of `options`, or `None` when they hold a field no config value shows (default
/// headers, a DNS resolver): such a client is never shared. Root certificates have no getter;
/// no builder that connects through the cache adds any.
#[cfg(feature = "cloud")]
fn fingerprint(options: &ClientOptions, egress: Option<&PolicyConnector>) -> Option<Fingerprint> {
    if options.get_default_headers().is_some() || options.dns_resolver().is_some() {
        return None;
    }
    Some((
        egress.map(PolicyConnector::fingerprint),
        KEYS.iter()
            .map(|key| (*key, options.get_config_value(key)))
            .collect(),
    ))
}

#[cfg(feature = "cloud")]
impl HttpConnector for SharedHttpClients {
    fn connect(&self, options: &ClientOptions) -> object_store::Result<HttpClient> {
        self.connect_via(options, self.inner.egress.as_ref())
    }
}

/// [`SharedHttpClients`] for stores with no ambient credential providers (`http(s)://`,
/// `hf://`): under an enforcing policy their clients never admit the provider literals, so a
/// data read of a cloud metadata address is refused before it is dialled (review 58a1 B2).
#[cfg(feature = "cloud")]
#[derive(Debug, Clone)]
pub struct DataOnlyClients(SharedHttpClients);

#[cfg(feature = "cloud")]
impl HttpConnector for DataOnlyClients {
    fn connect(&self, options: &ClientOptions) -> object_store::Result<HttpClient> {
        self.0
            .connect_via(options, self.0.inner.egress_data.as_ref())
    }
}

#[cfg(feature = "cloud")]
impl SharedHttpClients {
    fn connect_via(
        &self,
        options: &ClientOptions,
        egress: Option<&PolicyConnector>,
    ) -> object_store::Result<HttpClient> {
        let Some(key) = fingerprint(options, egress) else {
            return self.build(options, egress);
        };
        if let Some(client) = self
            .inner
            .clients
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&key)
        {
            return Ok(client.clone());
        }
        // Built outside the lock: a build parses the root store (milliseconds). Two racing
        // misses build twice; the first insert wins and both callers get its client, so one
        // pool serves the options.
        let client = self.build(options, egress)?;
        let mut clients = self
            .inner
            .clients
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(first) = clients.get(&key) {
            return Ok(first.clone());
        }
        if clients.len() < MAX_CLIENTS {
            clients.insert(key, client.clone());
        }
        Ok(client)
    }
}

#[cfg(all(test, feature = "cloud"))]
mod tests;

#[cfg(test)]
mod debug_tests {
    use super::SharedHttpClients;

    #[test]
    fn debug_names_the_cache_and_no_options() {
        let text = format!("{:?}", SharedHttpClients::new());
        assert!(text.starts_with("SharedHttpClients"), "{text}");
        #[cfg(feature = "cloud")]
        assert!(text.contains("clients: 0"), "{text}");
    }
}
