// SPDX-License-Identifier: Apache-2.0
//! The policy HTTP client (design P3-T067 §5.2, review-1 pass 2 N2): Krishiv's copy of object_store
//! 0.14.2's `ClientOptions::client()` (`pub(crate)` there, `client/mod.rs` 867) with the egress
//! rule built in.
//!
//! The copy carries every setting the original does, read back through
//! `ClientOptions::get_config_value` and parsed the way object_store parses it: every
//! [`ClientConfigKey`] (an option this copy does not know fails the build, never dropped
//! silently), `https_only` from `allow_http`, default headers, and no response decompression.
//! Root certificates have no getter; no Krishiv store sets any, so the copy adds none. Without any
//! system root (an empty trust store), a client that may speak plain HTTP is built with an empty root
//! set (HTTPS fails verification, closed) and an HTTPS-only one fails with [`NO_TLS_ROOTS`]. On top of
//! the original:
//! - the **egress resolver** ([`EgressResolver`]) replaces object_store's address-shuffling
//!   resolver: it resolves, drops every address the rule refuses, fails when none is left and
//!   shuffles the rest itself, so the connection goes to an address the rule checked (no
//!   rebinding window between check and connect);
//! - a **redirect policy** that checks every hop (literal addresses included) and stops at the
//!   first refusal, or follows no redirect at all (catalog and token clients);
//! - `no_proxy()`: environment proxies resolve on the proxy and would skip the check. Only the
//!   admin's `krishiv.io.egress.proxy` is used, and then the policy is the proxy's job;
//! - every request whose host is an address literal is checked before it is sent
//!   ([`PolicyService`]), with the provider literals admitted (§5.1 N1).
//!
//! Upgrading object_store: re-read `ClientOptions::client()` and `KEYS` (the version pin is checked
//! by a test in `tests.rs`).

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use object_store::client::{
    ClientConfigKey, ClientOptions, HttpClient, HttpConnector, HttpError, HttpErrorKind,
    HttpRequest, HttpResponse, HttpService,
};
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::redirect;

use crate::egress::EgressScope;

/// object_store's own default user agent (`client/mod.rs` 88), kept so servers see the same.
const DEFAULT_USER_AGENT: &str = "object_store/0.14.2";

/// Whether a policy client follows redirects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Redirects {
    /// Follow up to ten hops, each checked by the egress rule (object-store clients).
    Checked,
    /// Follow none: a 3xx is the answer (catalog and token clients, LH §5.6).
    None,
}

fn generic(reason: impl Into<String>) -> object_store::Error {
    object_store::Error::Generic {
        store: "HTTP client",
        source: reason.into().into(),
    }
}

fn parse_bool(key: ClientConfigKey, v: &str) -> object_store::Result<bool> {
    match v.to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" | "y" => Ok(true),
        "0" | "false" | "off" | "no" | "n" => Ok(false),
        _ => Err(generic(format!("{} is not a boolean", key.as_ref()))),
    }
}

fn parse_duration(key: ClientConfigKey, v: &str) -> object_store::Result<Duration> {
    humantime::parse_duration(v).map_err(|_| generic(format!("{} is not a duration", key.as_ref())))
}

fn parse_num<T: std::str::FromStr>(key: ClientConfigKey, v: &str) -> object_store::Result<T> {
    v.parse()
        .map_err(|_| generic(format!("{} is not a number", key.as_ref())))
}

/// One client option applied to `builder` (`client()`'s body, key by key). Every key object_store
/// 0.14.2 has is an arm; an unknown key with a value fails the build (`_` arm), so a key added
/// by an upgrade is refused, not dropped.
fn apply(
    builder: reqwest::ClientBuilder,
    key: ClientConfigKey,
    value: &str,
) -> object_store::Result<reqwest::ClientBuilder> {
    use ClientConfigKey as K;
    Ok(match key {
        // Policy clients exist wherever `krishiv.io.egress.privateAddresses=deny`: the server-mode
        // default, and an embedded runtime that sets it too. Certificate verification stays on in
        // both: a store or catalog endpoint with a certificate the system store does not trust is
        // refused, never reached without verification (object_store would disable verification
        // here).
        K::AllowInvalidCertificates if parse_bool(key, value)? => {
            return Err(generic(format!(
                "client option {} is refused by the egress policy client: certificate \
                 verification stays on while krishiv.io.egress.privateAddresses=deny",
                key.as_ref()
            )));
        }
        // No root certificates are added (see the module docs): the system store or none.
        K::NoSystemCertificates if parse_bool(key, value)? => {
            builder.tls_certs_only(std::iter::empty())
        }
        K::ConnectTimeout => builder.connect_timeout(parse_duration(key, value)?),
        K::ReadTimeout => builder.read_timeout(parse_duration(key, value)?),
        K::Http1Only if parse_bool(key, value)? => builder.http1_only(),
        K::Http2KeepAliveInterval => builder.http2_keep_alive_interval(parse_duration(key, value)?),
        K::Http2KeepAliveTimeout => builder.http2_keep_alive_timeout(parse_duration(key, value)?),
        K::Http2KeepAliveWhileIdle if parse_bool(key, value)? => {
            builder.http2_keep_alive_while_idle(true)
        }
        K::Http2MaxFrameSize => builder.http2_max_frame_size(Some(parse_num(key, value)?)),
        K::Http2Only if parse_bool(key, value)? => builder.http2_prior_knowledge(),
        K::PoolIdleTimeout => builder.pool_idle_timeout(parse_duration(key, value)?),
        K::PoolMaxIdlePerHost => builder.pool_max_idle_per_host(parse_num(key, value)?),
        K::Timeout => builder.timeout(parse_duration(key, value)?),
        K::UserAgent => builder.user_agent(value),
        // The egress resolver always shuffles (`EgressResolver`); the value is parsed only so a
        // bad one fails as in object_store.
        K::RandomizeAddresses => {
            parse_bool(key, value)?;
            builder
        }
        // Booleans whose value was parsed above and is `false`; `AllowHttp` (`https_only`, set
        // last in `build_policy_client`); the proxy keys (applied with the admin's egress proxy
        // there); `DefaultContentType` (a store setting object_store reads per request).
        K::AllowHttp
        | K::AllowInvalidCertificates
        | K::NoSystemCertificates
        | K::Http1Only
        | K::Http2KeepAliveWhileIdle
        | K::Http2Only
        | K::ProxyUrl
        | K::ProxyCaCertificate
        | K::ProxyExcludes
        | K::DefaultContentType => builder,
        _ => {
            return Err(generic(format!(
                "client option {} is not supported by the egress policy client",
                key.as_ref()
            )));
        }
    })
}

/// Builds the policy `reqwest` client for `options` (see the module docs).
///
/// # Errors
/// A value object_store would refuse, an option this copy does not know, a custom DNS resolver
/// in `options`, or a proxy in `options` when the admin set no egress proxy.
pub(crate) fn build_policy_client(
    options: &ClientOptions,
    scope: &EgressScope,
    redirects: Redirects,
    egress_proxy: Option<&str>,
    answers: Option<&Answers>,
) -> object_store::Result<reqwest::Client> {
    if options.dns_resolver().is_some() {
        return Err(generic(
            "a custom DNS resolver cannot be combined with the egress policy",
        ));
    }
    let allow_http = match options.get_config_value(&ClientConfigKey::AllowHttp) {
        Some(v) => parse_bool(ClientConfigKey::AllowHttp, &v)?,
        None => false,
    };
    let configured = || configure(options, scope, redirects, egress_proxy, answers.cloned());
    match configured()?.build() {
        Ok(client) => Ok(client),
        // No system root (a container without `/etc/ssl/certs`): a client that may speak plain HTTP is
        // built with an empty root set, so `http://` works and every HTTPS server fails verification
        // (closed, never open); a client that may only speak HTTPS cannot work and says why.
        Err(e) if lacks_tls_roots(&e) => {
            if allow_http {
                configured()?
                    .tls_certs_only(std::iter::empty())
                    .build()
                    .map_err(|_| generic("the egress policy client could not be built"))
            } else {
                Err(generic(NO_TLS_ROOTS))
            }
        }
        Err(_) => Err(generic("the egress policy client could not be built")),
    }
}

/// The reason a client that may only speak HTTPS cannot be built without a system root certificate.
pub const NO_TLS_ROOTS: &str = "no TLS root certificates were found: the system trust store is empty \
    (install the CA certificates, or point SSL_CERT_FILE at a bundle)";

/// Whether `error` (from [`PolicyConnector::catalog_client`] or a store build) is the missing system
/// root certificates of [`NO_TLS_ROOTS`].
#[must_use]
pub fn is_missing_tls_roots(error: &object_store::Error) -> bool {
    matches!(error, object_store::Error::Generic { source, .. } if source.to_string() == NO_TLS_ROOTS)
}

/// Whether a `reqwest` build error is the platform verifier finding no root certificate
/// (rustls-platform-verifier 0.7 on Linux: "No CA certificates were loaded from the system"; pinned by
/// the test `no_system_roots`).
fn lacks_tls_roots(error: &reqwest::Error) -> bool {
    let mut cause: Option<&dyn std::error::Error> = Some(error);
    while let Some(e) = cause {
        if e.to_string().contains("No CA certificates were loaded") {
            return true;
        }
        cause = e.source();
    }
    false
}

/// The policy client's builder for `options`, before `build` (see [`build_policy_client`]).
fn configure(
    options: &ClientOptions,
    scope: &EgressScope,
    redirects: Redirects,
    egress_proxy: Option<&str>,
    answers: Option<Answers>,
) -> object_store::Result<reqwest::ClientBuilder> {
    let mut builder = reqwest::ClientBuilder::new().user_agent(DEFAULT_USER_AGENT);
    if let Some(headers) = options.get_default_headers() {
        builder = builder.default_headers(headers.clone());
    }
    for key in super::KEYS {
        if let Some(value) = options.get_config_value(&key) {
            builder = apply(builder, key, &value)?;
        }
    }
    builder = match (
        egress_proxy,
        options.get_config_value(&ClientConfigKey::ProxyUrl),
    ) {
        (Some(proxy), _) => {
            let mut proxy = reqwest::Proxy::all(proxy).map_err(|_| generic("bad egress proxy"))?;
            if let Some(excludes) = options.get_config_value(&ClientConfigKey::ProxyExcludes) {
                proxy = proxy.no_proxy(reqwest::NoProxy::from_string(&excludes));
            }
            if let Some(pem) = options.get_config_value(&ClientConfigKey::ProxyCaCertificate) {
                let cert = reqwest::Certificate::from_pem(pem.as_bytes())
                    .map_err(|_| generic("bad proxy CA certificate"))?;
                builder = builder.tls_certs_merge(std::iter::once(cert));
            }
            builder.proxy(proxy)
        }
        (None, Some(_)) => {
            return Err(generic(
                "a store proxy needs krishiv.io.egress.proxy under the egress policy",
            ));
        }
        (None, None) => builder.no_proxy(),
    };
    let allow_http = match options.get_config_value(&ClientConfigKey::AllowHttp) {
        Some(v) => parse_bool(ClientConfigKey::AllowHttp, &v)?,
        None => false,
    };
    let redirect = match redirects {
        Redirects::None => redirect::Policy::none(),
        Redirects::Checked => {
            let scope = scope.clone();
            redirect::Policy::custom(move |attempt| {
                if attempt.previous().len() >= 10 {
                    return attempt.error("too many redirects");
                }
                match scope.check_redirect(attempt.url().host_str().unwrap_or_default()) {
                    Ok(()) => attempt.follow(),
                    Err(refusal) => attempt.error(refusal),
                }
            })
        }
    };
    Ok(builder
        .no_gzip()
        .no_brotli()
        .no_zstd()
        .no_deflate()
        .redirect(redirect)
        .dns_resolver(Arc::new(EgressResolver::new(scope.clone(), answers)))
        .https_only(!allow_http))
}

/// The egress resolver: resolves, keeps only the addresses the rule allows (failing when none is
/// left) and rotates them, so connections spread over the allowed addresses.
pub(crate) struct EgressResolver {
    scope: EgressScope,
    turn: AtomicUsize,
    /// Fixed answers instead of DNS (tests).
    answers: Option<Answers>,
}

/// Fixed DNS answers by host name (tests).
pub(crate) type Answers = Arc<dyn Fn(&str) -> Vec<IpAddr> + Send + Sync>;

impl std::fmt::Debug for EgressResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EgressResolver")
            .field("scope", &self.scope)
            .finish_non_exhaustive()
    }
}

impl EgressResolver {
    pub(crate) fn new(scope: EgressScope, answers: Option<Answers>) -> Self {
        Self {
            scope,
            turn: AtomicUsize::new(0),
            answers,
        }
    }
}

/// The addresses of `resolved` for `host` that `scope` allows, rotated by `turn`.
pub(crate) fn checked(
    scope: &EgressScope,
    host: &str,
    resolved: &[IpAddr],
    turn: usize,
) -> Result<Vec<SocketAddr>, crate::egress::EgressRefusal> {
    let mut kept = scope.filter(host, resolved)?;
    let n = kept.len();
    kept.rotate_left(turn % n);
    Ok(kept.into_iter().map(|ip| SocketAddr::new(ip, 0)).collect())
}

impl Resolve for EgressResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_owned();
        let scope = self.scope.clone();
        let answers = self.answers.clone();
        let turn = self.turn.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move {
            let resolved = match answers {
                Some(answers) => answers(&host),
                None => lookup(&host).await?,
            };
            let addrs = checked(&scope, &host, &resolved, turn)?;
            Ok(Box::new(addrs.into_iter()) as Addrs)
        })
    }
}

async fn lookup(host: &str) -> Result<Vec<IpAddr>, Box<dyn std::error::Error + Send + Sync>> {
    Ok(tokio::net::lookup_host((host, 0))
        .await?
        .map(|a| a.ip())
        .collect())
}

/// A policy client as object_store's [`HttpService`]: refuses a request whose host is an address
/// literal the rule does not allow, before anything is sent.
#[derive(Debug)]
pub(crate) struct PolicyService {
    client: reqwest::Client,
    scope: EgressScope,
    /// Whether the provider literals pass (clients of stores with ambient credential providers).
    provider_literals: bool,
}

#[async_trait::async_trait]
impl HttpService for PolicyService {
    async fn call(&self, req: HttpRequest) -> Result<HttpResponse, HttpError> {
        let host = req.uri().host().unwrap_or_default();
        let checked = if self.provider_literals {
            self.scope.permits_request_literal(host)
        } else {
            self.scope.check_redirect(host)
        };
        // Names pass here (the egress resolver checks what they resolve to); literals in any
        // spelling, and hosts no URL parser reads as either, are checked now.
        if let Err(refusal) = checked {
            return Err(HttpError::new(HttpErrorKind::Connect, refusal));
        }
        self.client.call(req).await
    }
}

/// An object_store [`HttpConnector`] that builds policy clients for one egress scope (uncached;
/// [`super::SharedHttpClients`] caches them).
#[derive(Clone)]
pub struct PolicyConnector {
    scope: EgressScope,
    redirects: Redirects,
    proxy: Option<String>,
    answers: Option<Answers>,
    provider_literals: bool,
}

impl std::fmt::Debug for PolicyConnector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never the proxy URL: it may carry userinfo.
        f.debug_struct("PolicyConnector")
            .field("scope", &self.scope)
            .field("redirects", &self.redirects)
            .finish_non_exhaustive()
    }
}

impl PolicyConnector {
    /// A connector for clients of `scope` (`proxy`: the admin's `krishiv.io.egress.proxy`).
    #[must_use]
    pub fn new(scope: EgressScope, redirects: Redirects, proxy: Option<String>) -> Self {
        Self {
            scope,
            redirects,
            proxy,
            answers: None,
            provider_literals: true,
        }
    }

    /// The same connector whose clients never admit the provider literals: for stores with no
    /// ambient credential providers, whose every request is a data request (review 58a1 B2).
    #[must_use]
    pub fn without_provider_literals(mut self) -> Self {
        self.provider_literals = false;
        self
    }

    /// The same connector resolving through fixed answers instead of DNS (tests).
    #[cfg(test)]
    pub(crate) fn with_answers(mut self, answers: Answers) -> Self {
        self.answers = Some(answers);
        self
    }

    /// The scope its clients apply.
    #[must_use]
    pub fn scope(&self) -> &EgressScope {
        &self.scope
    }

    /// The fingerprint component of its clients (§5.2).
    pub(crate) fn fingerprint(&self) -> String {
        format!(
            "{}:{:?}:{}:{}",
            self.scope.fingerprint(),
            self.redirects,
            self.proxy.is_some(),
            self.provider_literals
        )
    }

    /// A bare `reqwest` client of this connector, for non-object_store callers (catalog and
    /// token clients).
    ///
    /// # Errors
    /// As [`HttpConnector::connect`].
    pub fn reqwest_client(&self, options: &ClientOptions) -> object_store::Result<reqwest::Client> {
        build_policy_client(
            options,
            &self.scope,
            self.redirects,
            self.proxy.as_deref(),
            self.answers.as_ref(),
        )
    }
}

impl PolicyConnector {
    /// The client of a catalog or token endpoint (design P3-T067 §3.4, §5.2): this connector's
    /// scope and proxy, **no redirect followed** whatever the connector's own mode, `timeout` to
    /// connect and for the whole request, `https` only unless `allow_http`, and `user_agent`.
    /// `reqwest` never resolves an address-literal host, so the egress resolver does not see it:
    /// the caller checks every request's host with [`Self::check_request_host`] before sending.
    ///
    /// # Errors
    /// As [`HttpConnector::connect`].
    pub fn catalog_client(
        &self,
        timeout: Duration,
        allow_http: bool,
        user_agent: &str,
    ) -> object_store::Result<reqwest::Client> {
        let options = ClientOptions::new()
            .with_timeout(timeout)
            .with_connect_timeout(timeout)
            .with_allow_http(allow_http)
            .with_config(ClientConfigKey::UserAgent, user_agent);
        build_policy_client(
            &options,
            &self.scope,
            Redirects::None,
            self.proxy.as_deref(),
            self.answers.as_ref(),
        )
    }

    /// Checks the host of one request a non-object_store caller is about to send (what
    /// the connector's `PolicyService` does for object_store's requests): an address literal in any spelling
    /// must pass the rule, the provider literals only on a connector that admits them; a host
    /// no URL parser reads is refused; names pass (the egress resolver checks what they resolve
    /// to).
    ///
    /// # Errors
    /// [`crate::egress::EgressRefusal::PrivateAddress`] for a refused host.
    pub fn check_request_host(&self, host: &str) -> Result<(), crate::egress::EgressRefusal> {
        if self.provider_literals {
            self.scope.permits_request_literal(host)
        } else {
            self.scope.check_redirect(host)
        }
    }
}

impl HttpConnector for PolicyConnector {
    fn connect(&self, options: &ClientOptions) -> object_store::Result<HttpClient> {
        Ok(HttpClient::new(PolicyService {
            client: self.reqwest_client(options)?,
            scope: self.scope.clone(),
            provider_literals: self.provider_literals,
        }))
    }
}

#[cfg(test)]
mod tests;
