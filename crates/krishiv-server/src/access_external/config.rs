// SPDX-License-Identifier: Apache-2.0
//! `krishiv.access.external.*` (ADR 0049 §1): the endpoint, the deadline, the token and what the
//! service decides, read once at startup.

use std::net::IpAddr;
use std::path::PathBuf;
use std::time::Duration;

use krishiv_common::config::{
    ACCESS_EXTERNAL_ENDPOINT, ACCESS_EXTERNAL_SECURABLES, ACCESS_EXTERNAL_TIMEOUT,
    ACCESS_EXTERNAL_TOKEN, Config, ConfigEntry, ConfigValue,
};
use krishiv_common::error::{KrishivError, KrishivInvalidConfigValue};
use krishiv_session::auth::SecretRef;
use tonic::transport::Channel;

/// Where the policy service listens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Endpoint {
    /// `https://host[:port]`, verified against the platform's roots.
    Https(String),
    /// `http://` to a loopback address.
    LoopbackHttp(String),
    /// A Unix socket.
    Unix(PathBuf),
}

impl Endpoint {
    /// Parses `raw` (the value of `krishiv.access.external.endpoint`).
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` for anything but `https://host`, `http://` to a loopback
    /// host, or `unix:/absolute/path`; a URI with user info, a query or a fragment is refused
    /// too (a credential belongs in `krishiv.access.external.token`).
    pub fn parse(raw: &str) -> Result<Self, KrishivError> {
        let raw = raw.trim();
        // An endpoint may hold a credential in any position (user info, a query, a path): never
        // echoed in an error, which names the key and the reason only (review m5).
        let raw_shown = "<redacted>";
        if let Some(path) = raw.strip_prefix("unix:") {
            let path = PathBuf::from(path);
            if !path.is_absolute() {
                return Err(invalid(
                    &ACCESS_EXTERNAL_ENDPOINT,
                    raw_shown,
                    "a Unix socket path must be absolute",
                ));
            }
            return Ok(Self::Unix(path));
        }
        let (scheme, rest) = raw.split_once("://").ok_or_else(|| {
            invalid(
                &ACCESS_EXTERNAL_ENDPOINT,
                raw_shown,
                "expected https://host[:port], http:// to a loopback address, or unix:/path",
            )
        })?;
        let authority = rest.split('/').next().unwrap_or_default();
        if authority.is_empty() {
            return Err(invalid(
                &ACCESS_EXTERNAL_ENDPOINT,
                raw_shown,
                "the URI names no host",
            ));
        }
        if authority.contains('@') || rest.contains(['?', '#']) {
            return Err(invalid(
                &ACCESS_EXTERNAL_ENDPOINT,
                "<redacted>",
                "the URI may not carry user info, a query or a fragment",
            ));
        }
        match scheme.to_ascii_lowercase().as_str() {
            "https" => Ok(Self::Https(format!("https://{rest}"))),
            "http" if is_loopback(authority) => Ok(Self::LoopbackHttp(format!("http://{rest}"))),
            "http" => Err(invalid(
                &ACCESS_EXTERNAL_ENDPOINT,
                raw_shown,
                "http:// is allowed only to a loopback address; use https://",
            )),
            _ => Err(invalid(
                &ACCESS_EXTERNAL_ENDPOINT,
                raw_shown,
                "expected https://, http:// to a loopback address, or unix:",
            )),
        }
    }

    /// A lazily connecting channel to the endpoint.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` when the URI or TLS cannot be set up.
    pub fn channel(&self) -> Result<Channel, KrishivError> {
        let bad = |e: tonic::transport::Error| {
            invalid(&ACCESS_EXTERNAL_ENDPOINT, &self.shown(), &e.to_string())
        };
        match self {
            Self::Https(uri) => {
                let tls = tonic::transport::ClientTlsConfig::new().with_native_roots();
                Ok(tonic::transport::Endpoint::from_shared(uri.clone())
                    .map_err(bad)?
                    .tls_config(tls)
                    .map_err(bad)?
                    .connect_lazy())
            }
            Self::LoopbackHttp(uri) => Ok(tonic::transport::Endpoint::from_shared(uri.clone())
                .map_err(bad)?
                .connect_lazy()),
            Self::Unix(path) => {
                let path = path.clone();
                // The URI is not used to connect; the connector dials the socket.
                Ok(
                    tonic::transport::Endpoint::from_static("http://[::1]:50051")
                        .connect_with_connector_lazy(tower::service_fn(
                            move |_: tonic::transport::Uri| {
                                let path = path.clone();
                                async move {
                                    let stream = tokio::net::UnixStream::connect(path).await?;
                                    Ok::<_, std::io::Error>(hyper_util::rt::TokioIo::new(stream))
                                }
                            },
                        )),
                )
            }
        }
    }

    fn shown(&self) -> String {
        match self {
            Self::Https(u) | Self::LoopbackHttp(u) => u.clone(),
            Self::Unix(p) => format!("unix:{}", p.display()),
        }
    }
}

/// Whether `authority` (`host[:port]`) is a loopback IP literal.
fn is_loopback(authority: &str) -> bool {
    let host = if let Some(inner) = authority.strip_prefix('[') {
        inner.split(']').next().unwrap_or_default()
    } else {
        authority.rsplit_once(':').map_or(authority, |(h, _)| h)
    };
    // An IP literal only: a name such as `localhost` is resolved when the channel connects, and a
    // resolver may answer an address off this host, where the plaintext call and its token would
    // go.
    host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// The `external` provider's settings.
#[derive(Debug, Clone)]
pub struct ExternalConfig {
    /// The service.
    pub endpoint: Endpoint,
    /// The deadline of one call.
    pub timeout: Duration,
    /// The bearer token's reference, if any.
    pub token: Option<SecretRef>,
    /// Whether the service decides `System(..)` securables too.
    pub system: bool,
}

impl ExternalConfig {
    /// Reads `krishiv.access.external.*` from `config`.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` for a missing or bad endpoint, a zero timeout, a literal
    /// token, or a securable class other than `data` and `system`.
    pub fn from_config(config: &Config) -> Result<Self, KrishivError> {
        let endpoint = match config.get(&ACCESS_EXTERNAL_ENDPOINT) {
            Some(ConfigValue::String(s)) if !s.trim().is_empty() => Endpoint::parse(s)?,
            _ => {
                return Err(invalid(
                    &ACCESS_EXTERNAL_ENDPOINT,
                    "",
                    "the external access provider needs the policy service's endpoint",
                ));
            }
        };
        let timeout = match config.get(&ACCESS_EXTERNAL_TIMEOUT) {
            Some(ConfigValue::Duration(d)) => *d,
            _ => Duration::from_millis(200),
        };
        if timeout.is_zero() {
            return Err(invalid(
                &ACCESS_EXTERNAL_TIMEOUT,
                "0",
                "the deadline must be positive",
            ));
        }
        let token = match config.get(&ACCESS_EXTERNAL_TOKEN) {
            Some(ConfigValue::String(s)) if !s.trim().is_empty() => Some(SecretRef::parse(
                ACCESS_EXTERNAL_TOKEN.key,
                s.trim(),
                false,
            )?),
            _ => None,
        };
        let classes = config
            .get(&ACCESS_EXTERNAL_SECURABLES)
            .and_then(ConfigValue::as_str)
            .unwrap_or("data")
            .to_owned();
        let system = false;
        for class in classes.split(',').map(str::trim) {
            match class.to_ascii_lowercase().as_str() {
                "data" => {}
                // The admin API and the UI guards still decide the server's objects by the role
                // matrix alone, so the service would decide them on the SQL path only (review M3).
                "system" => {
                    return Err(invalid(
                        &ACCESS_EXTERNAL_SECURABLES,
                        &classes,
                        "`system` is not available yet: the admin API and UI guards do not ask the \
                         policy service; use `data`",
                    ));
                }
                _ => {
                    return Err(invalid(
                        &ACCESS_EXTERNAL_SECURABLES,
                        &classes,
                        "expected `data` (the one class available)",
                    ));
                }
            }
        }
        Ok(Self {
            endpoint,
            timeout,
            token,
            system,
        })
    }

    /// The `authorization` header value, the token resolved now.
    ///
    /// # Errors
    /// The secret reference's errors, or `KRISHIV_INVALID_CONFIG_VALUE` when the token is not
    /// valid header text.
    pub fn bearer(
        &self,
    ) -> Result<Option<tonic::metadata::MetadataValue<tonic::metadata::Ascii>>, KrishivError> {
        let Some(token) = &self.token else {
            return Ok(None);
        };
        let secret = token.resolve(ACCESS_EXTERNAL_TOKEN.key)?;
        let value = format!("Bearer {}", secret.expose_secret().trim());
        let mut header = tonic::metadata::MetadataValue::try_from(value).map_err(|_| {
            invalid(
                &ACCESS_EXTERNAL_TOKEN,
                "<redacted>",
                "the token is not valid header text",
            )
        })?;
        // Kept out of h2's header compression tables and debug output.
        header.set_sensitive(true);
        Ok(Some(header))
    }
}

fn invalid(entry: &ConfigEntry, value: &str, reason: &str) -> KrishivError {
    KrishivInvalidConfigValue {
        value: format!("'{value}'"),
        key: format!("\"{}\"", entry.key),
        reason: reason.to_owned(),
    }
    .into()
}
