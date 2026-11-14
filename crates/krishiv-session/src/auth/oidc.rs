// SPDX-License-Identifier: Apache-2.0
//! OIDC access tokens (feature `oidc`; design §3.3): the signature check on aws-lc-rs, the JWK
//! set cache and the thread that fetches it.
//!
//! - **Signatures.** RSASSA-PKCS1-v1_5 on 2048 to 8192 bit keys, and ECDSA P-256/P-384 with
//!   fixed-width `r || s` (RFC 7518 §3.4). There is no `none` and no HMAC path, so an HS256
//!   token "signed" with an RSA public key cannot verify (algorithm pinning by construction).
//! - **Keys.** Only from the configured or discovered JWK set, never from the token header.
//!   One thread (`krishiv-jwks`) fetches it at start and every `jwksRefresh`, with the blocking
//!   reqwest client over rustls, never on a tokio worker. A token whose `kid` is not cached asks
//!   for one extra fetch at most every [`ON_DEMAND_EVERY`] and waits for it; other tokens with
//!   an unknown `kid` meanwhile fail at once (threat model T-A5's JWKS flood). A failed fetch,
//!   or a set with no usable key, keeps the cached keys; with none cached, the answer is
//!   [`AuthFailure::Unavailable`] so clients retry. One fetch (discovery and key set together)
//!   ends within [`FETCH_DEADLINE`] plus one read timeout, which bounds how long a waiting
//!   request waits.
//! - **One process.** The cache, the fetch limit and the failure throttle are per process: N
//!   servers fetch N times.

use std::io::Read as _;
use std::sync::mpsc;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use aws_lc_rs::signature::{self, RsaPublicKeyComponents, UnparsedPublicKey};
use krishiv_common::error::KrishivError;
use tokio::sync::oneshot;

use super::AuthFailure;
use super::jose::{self, Alg, ClaimRules, Identity, JoseError, Jwk, KeyMaterial};
use super::oidc_config::OidcConfig;
use super::secret_ref::invalid;

/// The least time between two fetches asked for by unknown key ids (design §3.3 ⚗).
pub(crate) const ON_DEMAND_EVERY: Duration = Duration::from_mins(1);
/// The time limit of one fetch (discovery and key set together).
pub(crate) const FETCH_DEADLINE: Duration = Duration::from_secs(10);
/// The time limit of one blocking read of a response body.
const READ_TIMEOUT: Duration = Duration::from_secs(5);

/// Where the JWK set comes from. The HTTP source in production; tests supply their own.
pub(crate) trait KeySource: Send + 'static {
    /// The JWK set document's bytes. The error text is logged (it never holds a token).
    fn fetch(&mut self) -> Result<Vec<u8>, String>;
}

/// Whether `signature` over `input` verifies with `key` for `alg`. The key must fit `alg`
/// ([`Jwk::fits`]); a key that does not never verifies.
#[must_use]
pub(crate) fn verify_signature(alg: Alg, key: &Jwk, input: &[u8], signature: &[u8]) -> bool {
    if !key.fits(alg) {
        return false;
    }
    match (&key.material, alg) {
        (KeyMaterial::Rsa { n, e }, Alg::Rs256 | Alg::Rs384 | Alg::Rs512) => {
            let params = match alg {
                Alg::Rs256 => &signature::RSA_PKCS1_2048_8192_SHA256,
                Alg::Rs384 => &signature::RSA_PKCS1_2048_8192_SHA384,
                _ => &signature::RSA_PKCS1_2048_8192_SHA512,
            };
            RsaPublicKeyComponents { n, e }
                .verify(params, input, signature)
                .is_ok()
        }
        (KeyMaterial::Ec { point, .. }, Alg::Es256) => {
            UnparsedPublicKey::new(&signature::ECDSA_P256_SHA256_FIXED, point)
                .verify(input, signature)
                .is_ok()
        }
        (KeyMaterial::Ec { point, .. }, Alg::Es384) => {
            UnparsedPublicKey::new(&signature::ECDSA_P384_SHA384_FIXED, point)
                .verify(input, signature)
                .is_ok()
        }
        _ => false,
    }
}

/// The cached keys and when they may be fetched again on demand.
#[derive(Debug, Default)]
struct CacheState {
    keys: Arc<[Jwk]>,
    /// The last fetch asked for by a request.
    last_on_demand: Option<Instant>,
}

/// The answer of [`JwksCache::lookup`].
#[derive(Debug)]
pub(crate) enum Lookup {
    /// The keys that may verify the token: the cached set and the indexes of the fitting keys.
    Keys(Arc<[Jwk]>, Vec<usize>),
    /// A token without `kid` that several cached keys fit: refused rather than tried against
    /// each.
    Ambiguous,
    /// No cached key fits, and this request may wait for one fetch.
    Refresh,
    /// No cached key fits, and a fetch was asked for too recently.
    Unknown,
    /// No key at all is cached, and a fetch was asked for too recently.
    NoKeys,
}

/// The JWK set cache, shared by requests and the fetch thread.
#[derive(Debug, Default)]
pub(crate) struct JwksCache {
    state: Mutex<CacheState>,
}

impl JwksCache {
    fn state(&self) -> std::sync::MutexGuard<'_, CacheState> {
        // The state is replaced whole under the lock, so a poisoned one is still consistent.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The keys that may verify a token with `kid` and `alg`, at `now`. A token without a `kid`
    /// may use the one fitting key, if exactly one fits. Asking for a fetch is rate-limited
    /// here: the request that gets [`Lookup::Refresh`] has taken the slot.
    pub(crate) fn lookup(&self, kid: Option<&str>, alg: Alg, now: Instant) -> Lookup {
        let keys = Arc::clone(&self.state().keys);
        let fitting: Vec<usize> = keys
            .iter()
            .enumerate()
            .filter(|(_, k)| k.fits(alg) && (kid.is_none() || k.kid.as_deref() == kid))
            .map(|(i, _)| i)
            .collect();
        if kid.is_none() && fitting.len() > 1 {
            return Lookup::Ambiguous;
        }
        if !fitting.is_empty() {
            return Lookup::Keys(keys, fitting);
        }
        let mut state = self.state();
        let may = state
            .last_on_demand
            .is_none_or(|t| now.saturating_duration_since(t) >= ON_DEMAND_EVERY);
        if may {
            state.last_on_demand = Some(now);
            Lookup::Refresh
        } else if state.keys.is_empty() {
            Lookup::NoKeys
        } else {
            Lookup::Unknown
        }
    }

    /// Replaces the keys with the set in `document`. A document that does not parse, or holds
    /// no usable key while keys are cached, keeps the old keys (an empty set from a provider in
    /// trouble must not lock every user out).
    ///
    /// # Errors
    /// The parse error, or `no usable key`.
    pub(crate) fn install(&self, document: &[u8]) -> Result<usize, JoseError> {
        let keys = jose::parse_jwks(document)?;
        let n = keys.len();
        let mut state = self.state();
        if n == 0 && !state.keys.is_empty() {
            return Err(JoseError::Invalid("no usable key"));
        }
        state.keys = keys.into();
        Ok(n)
    }

    /// How many keys are cached (tests).
    #[cfg(test)]
    pub(crate) fn key_count(&self) -> usize {
        self.state().keys.len()
    }
}

/// A request to the fetch thread; `Some` wants to know when the fetch is done.
type FetchRequest = Option<oneshot::Sender<()>>;

/// The OIDC verifier of one server.
#[derive(Debug)]
pub(crate) struct Oidc {
    issuer: String,
    audience: String,
    algorithms: Vec<Alg>,
    principal_claim: String,
    role_claim: String,
    skew_secs: u64,
    jwks: Arc<JwksCache>,
    /// Requests to the fetch thread; dropping it stops the thread.
    fetch: mpsc::Sender<FetchRequest>,
}

/// The outcome of one verification.
#[derive(Debug)]
pub(crate) enum Verified {
    /// A valid token.
    Ok(Identity),
    /// Refused.
    Failed(AuthFailure),
    /// No cached key fits; wait for [`Oidc::refresh`], then verify again without a refresh.
    Refresh,
}

impl Oidc {
    /// Starts the verifier and its fetch thread over HTTP.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` when OIDC is not fully configured, or the thread cannot
    /// start.
    pub(crate) fn start(config: &OidcConfig) -> Result<Self, KrishivError> {
        let source = HttpKeySource::new(config);
        Self::with_source(config, Box::new(source))
    }

    /// Starts the verifier with `source` (tests use an in-memory one).
    ///
    /// # Errors
    /// As [`Oidc::start`].
    pub(crate) fn with_source(
        config: &OidcConfig,
        mut source: Box<dyn KeySource>,
    ) -> Result<Self, KrishivError> {
        let (Some(issuer), Some(audience)) = (&config.issuer, &config.audience) else {
            return Err(invalid(
                "krishiv.auth.oidc.issuer",
                "",
                "OIDC needs an issuer and an audience",
            ));
        };
        let jwks = Arc::new(JwksCache::default());
        let (tx, rx) = mpsc::channel::<FetchRequest>();
        let cache = Arc::clone(&jwks);
        let every = config.jwks_refresh;
        std::thread::Builder::new()
            .name("krishiv-jwks".to_owned())
            .spawn(move || fetch_loop(source.as_mut(), &cache, &rx, every))
            .map_err(|e| {
                invalid(
                    "krishiv.auth.oidc.issuer",
                    issuer,
                    &format!("cannot start the JWK set thread: {e}"),
                )
            })?;
        Ok(Self {
            issuer: issuer.clone(),
            audience: audience.clone(),
            algorithms: config.algorithms.clone(),
            principal_claim: config.principal_claim.clone(),
            role_claim: config.role_claim.clone(),
            skew_secs: config.clock_skew.as_secs(),
            jwks,
            fetch: tx,
        })
    }

    /// Verifies `token` at `now` (monotonic, for the fetch limit) and `unix_now` (seconds since
    /// the epoch, for `exp`/`nbf`). With `may_refresh`, a token whose key is not cached gets
    /// [`Verified::Refresh`] when a fetch is allowed; without, that fetch is still started and
    /// the token is refused as [`AuthFailure::Unavailable`].
    pub(crate) fn verify(
        &self,
        token: &str,
        now: Instant,
        unix_now: u64,
        may_refresh: bool,
    ) -> Verified {
        let jws = match jose::parse_jws(token) {
            Ok(jws) => jws,
            Err(e) => return refused(e),
        };
        let alg = jws.header.alg;
        if !self.algorithms.contains(&alg) {
            return refused(JoseError::Invalid("algorithm not allowed"));
        }
        let (keys, fitting) = match self.jwks.lookup(jws.header.kid.as_deref(), alg, now) {
            Lookup::Keys(keys, fitting) => (keys, fitting),
            Lookup::Refresh if may_refresh => return Verified::Refresh,
            Lookup::Refresh => {
                self.request(None);
                return Verified::Failed(AuthFailure::Unavailable);
            }
            Lookup::NoKeys => return Verified::Failed(AuthFailure::Unavailable),
            Lookup::Unknown => return refused(JoseError::Invalid("unknown key")),
            Lookup::Ambiguous => return refused(JoseError::Invalid("no kid and several keys")),
        };
        let signed = fitting
            .iter()
            .filter_map(|i| keys.get(*i))
            .any(|k| verify_signature(alg, k, jws.signing_input, &jws.signature));
        if !signed {
            return refused(JoseError::Invalid("signature"));
        }
        let rules = ClaimRules {
            issuer: &self.issuer,
            audience: &self.audience,
            principal_claim: &self.principal_claim,
            role_claim: &self.role_claim,
            skew_secs: self.skew_secs,
        };
        match jose::check_claims(&jws.claims, &rules, unix_now) {
            Ok(identity) => Verified::Ok(identity),
            Err(e) => refused(e),
        }
    }

    /// Asks the fetch thread for a fetch now; the receiver completes when it is done (or
    /// errors when the thread has stopped).
    pub(crate) fn refresh(&self) -> oneshot::Receiver<()> {
        let (tx, rx) = oneshot::channel();
        self.request(Some(tx));
        rx
    }

    fn request(&self, done: FetchRequest) {
        if self.fetch.send(done).is_err() {
            // The receiver of a waiting request then errors at once, so nobody hangs.
            tracing::error!(target: "krishiv::auth", "the JWK set thread has stopped");
        }
    }

    /// The key cache (tests).
    #[cfg(test)]
    pub(crate) fn jwks(&self) -> &JwksCache {
        &self.jwks
    }
}

/// A refusal, with its fixed reason logged at debug level (never token content).
fn refused(e: JoseError) -> Verified {
    tracing::debug!(target: "krishiv::auth", reason = %e, "OIDC token refused");
    Verified::Failed(match e {
        JoseError::Expired => AuthFailure::Expired,
        JoseError::Invalid(_) => AuthFailure::Invalid,
    })
}

/// Seconds since the Unix epoch now (0 before 1970, which no token is valid at).
#[must_use]
pub(crate) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The fetch thread: once at start, then every `every` and on request, until the verifier is
/// dropped. Requests that arrive during a fetch are answered by the next one.
fn fetch_loop(
    source: &mut dyn KeySource,
    cache: &JwksCache,
    rx: &mpsc::Receiver<FetchRequest>,
    every: Duration,
) {
    let mut waiters: Vec<oneshot::Sender<()>> = Vec::new();
    loop {
        match source.fetch() {
            Ok(bytes) => match cache.install(&bytes) {
                Ok(n) => tracing::debug!(target: "krishiv::auth", keys = n, "JWK set fetched"),
                Err(e) => tracing::error!(
                    target: "krishiv::auth",
                    error = %e,
                    "the JWK set does not parse; keeping the cached keys"
                ),
            },
            Err(e) => tracing::error!(
                target: "krishiv::auth",
                error = %e,
                "cannot fetch the JWK set; keeping the cached keys"
            ),
        }
        for w in waiters.drain(..) {
            // A waiter that gave up has dropped its receiver; nothing to tell it.
            w.send(()).ok();
        }
        match rx.recv_timeout(every) {
            Ok(request) => {
                waiters.extend(request);
                // Requests queued meanwhile share this fetch.
                while let Ok(request) = rx.try_recv() {
                    waiters.extend(request);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}

/// The JWK set over HTTPS: `jwksUri`, or the one the issuer's discovery document names.
pub(crate) struct HttpKeySource {
    issuer: String,
    configured: Option<String>,
    discovery: Option<String>,
    discovered: Option<String>,
    config: OidcConfig,
    /// Built on the fetch thread at the first fetch (a blocking client must not be built or
    /// dropped inside a tokio runtime).
    client: Option<reqwest::blocking::Client>,
}

impl HttpKeySource {
    pub(crate) fn new(config: &OidcConfig) -> Self {
        Self {
            issuer: config.issuer.clone().unwrap_or_default(),
            configured: config.jwks_uri.clone(),
            discovery: config.discovery_uri(),
            discovered: None,
            config: config.clone(),
            client: None,
        }
    }

    /// GETs `uri` before `deadline`: no redirects, at most [`jose::MAX_DOCUMENT_BYTES`].
    fn get(&mut self, uri: &str, deadline: Instant) -> Result<Vec<u8>, String> {
        if !self.config.uri_allowed(uri) {
            return Err(format!("refusing the location {uri}"));
        }
        let client = match &self.client {
            Some(c) => c,
            None => {
                let c = reqwest::blocking::Client::builder()
                    // Applied to connecting, to the headers and to each read of the body.
                    .timeout(READ_TIMEOUT)
                    .connect_timeout(READ_TIMEOUT)
                    .redirect(reqwest::redirect::Policy::none())
                    .build()
                    .map_err(|e| format!("cannot build the HTTP client: {e}"))?;
                self.client.insert(c)
            }
        };
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(format!("GET {uri}: out of time"));
        }
        let mut resp = client
            .get(uri)
            .header("accept", "application/json")
            // The whole request, body included, ends by the deadline.
            .timeout(left)
            .send()
            .map_err(|e| format!("GET {uri}: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("GET {uri}: HTTP {}", resp.status()));
        }
        let too_large = || format!("GET {uri}: the document is larger than 1 MiB");
        if resp
            .content_length()
            .is_some_and(|n| usize::try_from(n).map_or(true, |n| n > jose::MAX_DOCUMENT_BYTES))
        {
            return Err(too_large());
        }
        let mut body = Vec::new();
        let mut buf = [0u8; 16 * 1024];
        loop {
            if Instant::now() >= deadline {
                return Err(format!("GET {uri}: out of time"));
            }
            let n = resp.read(&mut buf).map_err(|e| {
                if Instant::now() >= deadline {
                    format!("GET {uri}: out of time")
                } else {
                    format!("GET {uri}: {e}")
                }
            })?;
            if n == 0 {
                return Ok(body);
            }
            if body.len() + n > jose::MAX_DOCUMENT_BYTES {
                return Err(too_large());
            }
            body.extend_from_slice(buf.get(..n).unwrap_or_default());
        }
    }
}

impl KeySource for HttpKeySource {
    fn fetch(&mut self) -> Result<Vec<u8>, String> {
        let deadline = Instant::now() + FETCH_DEADLINE;
        if let Some(uri) = self.configured.clone() {
            return self.get(&uri, deadline);
        }
        let uri = match self.discovered.clone() {
            Some(uri) => uri,
            None => {
                let discovery = self.discovery.clone().unwrap_or_default();
                let doc = self.get(&discovery, deadline)?;
                let uri = jose::discovered_jwks_uri(&doc, &self.issuer)
                    .map_err(|e| format!("{discovery}: {e}"))?;
                tracing::info!(target: "krishiv::auth", location = %uri, "JWK set location discovered");
                self.discovered.insert(uri).clone()
            }
        };
        let result = self.get(&uri, deadline);
        if result.is_err() {
            // The provider may have moved its keys: discover again next time.
            self.discovered = None;
        }
        result
    }
}
