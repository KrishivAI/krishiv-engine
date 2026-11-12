// SPDX-License-Identifier: Apache-2.0
//! Client certificates (design §3.4; ADR 0046 §6): whether the listeners ask for one, and the
//! one rule that names the principal of a verified leaf certificate.
//!
//! rustls verifies the chain, the client CA and the validity dates at the handshake
//! ([`super::tls`]); this reads the principal's name out of the leaf. No revocation (CRL, OCSP)
//! in P3.

use std::collections::BTreeSet;
use std::path::PathBuf;

use krishiv_common::error::KrishivError;
use krishiv_common::principal::PrincipalKind;

use super::der::CertNames;
use super::secret_ref::invalid;

/// `krishiv.auth.clientCertificates`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClientCertificates {
    /// No client certificate is asked for (the default).
    #[default]
    Off,
    /// A client may present one; one without a certificate authenticates by bearer token.
    Optional,
    /// The handshake fails without a certificate the client CA verifies.
    Required,
}

impl ClientCertificates {
    /// The setting's value.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Optional => "optional",
            Self::Required => "required",
        }
    }

    fn parse(key: &str, raw: &str) -> Result<Self, KrishivError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "optional" => Ok(Self::Optional),
            "required" => Ok(Self::Required),
            _ => Err(invalid(key, raw, "expected off, optional or required")),
        }
    }
}

/// `krishiv.auth.mtls.principalFrom`: which field of the leaf names the principal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrincipalFrom {
    /// The single URI subject alternative name, e.g. a SPIFFE id (the default).
    #[default]
    SanUri,
    /// The single DNS subject alternative name.
    SanDns,
    /// The single subject common name.
    Cn,
}

impl PrincipalFrom {
    fn parse(key: &str, raw: &str) -> Result<Self, KrishivError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "san-uri" => Ok(Self::SanUri),
            "san-dns" => Ok(Self::SanDns),
            "cn" => Ok(Self::Cn),
            _ => Err(invalid(key, raw, "expected san-uri, san-dns or cn")),
        }
    }

    /// The setting's value.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SanUri => "san-uri",
            Self::SanDns => "san-dns",
            Self::Cn => "cn",
        }
    }
}

/// The `krishiv.auth.clientCertificates`, `krishiv.auth.clientCa` and `krishiv.auth.mtls.*`
/// settings.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MtlsConfig {
    /// Whether a client certificate is asked for.
    pub client_certificates: ClientCertificates,
    /// The PEM file of the CA certificates that sign client certificates (`file:/path`).
    pub client_ca: Option<PathBuf>,
    /// The field that names the principal.
    pub principal_from: PrincipalFrom,
    /// Certificate principals that are users; every other one is a service.
    pub user_principals: BTreeSet<String>,
}

impl MtlsConfig {
    /// Applies the setting `rest` (the key after `krishiv.auth.`), if it is one of these.
    /// Returns whether it was.
    pub(crate) fn apply(&mut self, key: &str, rest: &str, raw: &str) -> Result<bool, KrishivError> {
        match rest {
            "clientCertificates" => self.client_certificates = ClientCertificates::parse(key, raw)?,
            "clientCa" => {
                let path = raw
                    .strip_prefix("file:")
                    .filter(|p| !p.is_empty())
                    .ok_or_else(|| invalid(key, raw, "expected file:/path of a PEM file"))?;
                self.client_ca = Some(PathBuf::from(path));
            }
            "mtls.principalFrom" => self.principal_from = PrincipalFrom::parse(key, raw)?,
            "mtls.userPrincipals" => {
                self.user_principals = raw
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect();
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Whether client certificates are asked for.
    #[must_use]
    pub fn is_on(&self) -> bool {
        self.client_certificates != ClientCertificates::Off
    }

    /// Startup checks that need no other setting: a client CA exactly when certificates are on.
    pub(crate) fn validate(&self) -> Result<(), KrishivError> {
        match (self.is_on(), &self.client_ca) {
            (true, None) => Err(invalid(
                "krishiv.auth.clientCa",
                "",
                "required when krishiv.auth.clientCertificates is optional or required",
            )),
            (false, Some(_)) => Err(invalid(
                "krishiv.auth.clientCa",
                "",
                "set, but krishiv.auth.clientCertificates is off",
            )),
            _ => Ok(()),
        }
    }

    /// The principal named by the verified leaf certificate `leaf` (DER), and its kind.
    ///
    /// `None` when the certificate cannot be read or has zero or several values of the
    /// configured kind (no fallback to another kind).
    #[must_use]
    pub fn principal_of(&self, leaf: &[u8]) -> Option<(String, PrincipalKind)> {
        let names = CertNames::from_der(leaf).ok()?;
        let values = match self.principal_from {
            PrincipalFrom::SanUri => names.uris,
            PrincipalFrom::SanDns => names.dns,
            PrincipalFrom::Cn => names.common_names,
        };
        let [name] = <[String; 1]>::try_from(values).ok()?;
        if name.is_empty() {
            return None;
        }
        let kind = if self.user_principals.contains(&name) {
            PrincipalKind::User
        } else {
            PrincipalKind::Service
        };
        Some((name, kind))
    }
}
