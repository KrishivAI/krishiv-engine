// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.server.tls.*` settings (design §3.5) and the startup check that a listener's
//! address, authentication and TLS make sense together (threat model §6 item 9). Read without
//! the `tls` feature, so a build without TLS still refuses a TLS configuration it cannot serve.

use std::net::SocketAddr;
use std::path::PathBuf;

use krishiv_common::error::KrishivError;

use super::config::AuthConfig;
use super::secret_ref::invalid;

pub(crate) const CERT_KEY: &str = "krishiv.server.tls.certFile";
pub(crate) const KEY_KEY: &str = "krishiv.server.tls.keyFile";
pub(crate) const MIN_VERSION_KEY: &str = "krishiv.server.tls.minVersion";

/// The `krishiv.server.tls.*` settings.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TlsConfig {
    /// The PEM certificate chain, leaf first.
    pub cert_file: Option<PathBuf>,
    /// The PEM private key.
    pub key_file: Option<PathBuf>,
    /// Whether TLS 1.2 is refused (`minVersion = 1.3`).
    pub tls13_only: bool,
}

impl TlsConfig {
    /// Reads the `krishiv.server.tls.*` settings of `settings` (other keys are ignored).
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` for an unknown `krishiv.server.tls.*` key, a bad
    /// `minVersion`, or a certificate without a key (or a key without a certificate).
    pub fn from_settings<'a>(
        settings: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self, KrishivError> {
        let mut out = Self::default();
        for (key, raw) in settings {
            match key {
                CERT_KEY => out.cert_file = Some(path(key, raw)?),
                KEY_KEY => out.key_file = Some(path(key, raw)?),
                MIN_VERSION_KEY => {
                    out.tls13_only = match raw.trim() {
                        "1.2" => false,
                        "1.3" => true,
                        _ => return Err(invalid(key, raw, "expected 1.2 or 1.3")),
                    };
                }
                _ if key.starts_with("krishiv.server.tls.") => {
                    return Err(invalid(key, key, "not a known TLS setting"));
                }
                _ => {}
            }
        }
        match (&out.cert_file, &out.key_file) {
            (Some(_), None) => Err(invalid(
                KEY_KEY,
                "",
                "required with krishiv.server.tls.certFile",
            )),
            (None, Some(_)) => Err(invalid(
                CERT_KEY,
                "",
                "required with krishiv.server.tls.keyFile",
            )),
            _ => Ok(out),
        }
    }

    /// Whether the listeners serve TLS.
    #[must_use]
    pub fn is_on(&self) -> bool {
        self.cert_file.is_some()
    }
}

/// A plain path or `file:/path`.
fn path(key: &str, raw: &str) -> Result<PathBuf, KrishivError> {
    let p = raw.strip_prefix("file:").unwrap_or(raw).trim();
    if p.is_empty() {
        return Err(invalid(key, raw, "a file path is required"));
    }
    Ok(PathBuf::from(p))
}

/// Refuses a listener on `addr` whose authentication and TLS do not fit together (design §3.5):
/// - a non-loopback address with no authentication method;
/// - a non-loopback address with a bearer method and no TLS (a stock client would send the token
///   over TLS and could not connect; a custom one would send it in clear);
/// - client certificates without TLS.
///
/// Loopback with a token and no TLS is allowed: that is the stock client's `localhost` path.
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` naming the setting to change.
pub fn check_listener(
    addr: SocketAddr,
    auth: &AuthConfig,
    tls: &TlsConfig,
) -> Result<(), KrishivError> {
    if auth.mtls.is_on() && !tls.is_on() {
        return Err(invalid(
            "krishiv.auth.clientCertificates",
            auth.mtls.client_certificates.as_str(),
            "client certificates need krishiv.server.tls.certFile and keyFile",
        ));
    }
    if addr.ip().is_loopback() {
        return Ok(());
    }
    let shown = addr.to_string();
    if !auth.is_configured() {
        return Err(invalid(
            "krishiv.server.host",
            &shown,
            "a non-loopback address needs an authentication method (krishiv.auth.*)",
        ));
    }
    if !auth.tokens.is_empty() && !tls.is_on() {
        return Err(invalid(
            "krishiv.server.host",
            &shown,
            "bearer tokens on a non-loopback address need TLS (krishiv.server.tls.certFile)",
        ));
    }
    Ok(())
}
