// SPDX-License-Identifier: Apache-2.0
//! Server TLS (design §3.5, threat model T-T1): the rustls configuration both listeners serve,
//! on the `aws-lc-rs` provider (no OpenSSL, ADR 0028).
//!
//! - `krishiv.server.tls.certFile` / `keyFile`: PEM files; the key file must not be readable by
//!   group or others. `krishiv.server.tls.minVersion`: `1.2` (default) or `1.3`.
//! - Client certificates ([`MtlsConfig`]): rustls checks the chain against the client CA and
//!   the validity dates at the handshake; with `optional` a client may present none.
//! - [`ServerTls::reload`] re-reads the certificate and key (the binary calls it on `SIGHUP`):
//!   new handshakes get the new material, connections already up keep theirs, and material that
//!   fails to load leaves the old one serving.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use krishiv_common::error::KrishivError;
use rustls::crypto::CryptoProvider;
use rustls::pki_types::pem::PemObject as _;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::server::{ClientHello, ResolvesServerCert, WebPkiClientVerifier};
use rustls::sign::CertifiedKey;
use rustls::{RootCertStore, ServerConfig};

use super::mtls::{ClientCertificates, MtlsConfig};
use super::secret_ref::{invalid, read_private_file, read_public_file};
use super::tls_config::{CERT_KEY, KEY_KEY, MIN_VERSION_KEY, TlsConfig};

/// The longest certificate chain or CA file read.
const MAX_CERT_FILE_BYTES: usize = 1024 * 1024;

/// The longest private key file read (a PEM RSA-8192 key is about 6.3 KiB).
const MAX_KEY_FILE_BYTES: usize = 64 * 1024;

/// The certificate the server presents, swapped whole on reload.
#[derive(Debug)]
struct Current {
    key: RwLock<Arc<CertifiedKey>>,
}

impl ResolvesServerCert for Current {
    fn resolve(&self, _hello: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        // A poisoned lock still holds a whole `Arc`: serve it.
        let guard = self
            .key
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Some(Arc::clone(&guard))
    }
}

/// The server's TLS: the rustls configuration and the material behind it.
#[derive(Debug, Clone)]
pub struct ServerTls {
    config: Arc<ServerConfig>,
    current: Arc<Current>,
    cert_file: PathBuf,
    key_file: PathBuf,
    provider: Arc<CryptoProvider>,
}

impl ServerTls {
    /// Loads the certificate, key and (when client certificates are on) the client CA.
    /// `None` when TLS is off.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` when a file cannot be read or parsed, the key file is
    /// readable by group or others, the key does not match the certificate, or the client CA
    /// holds no usable certificate.
    pub fn new(tls: &TlsConfig, mtls: &MtlsConfig) -> Result<Option<Self>, KrishivError> {
        let (Some(cert_file), Some(key_file)) = (&tls.cert_file, &tls.key_file) else {
            return Ok(None);
        };
        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        let current = Arc::new(Current {
            key: RwLock::new(Arc::new(load_key(cert_file, key_file, &provider)?)),
        });
        let versions: &[&rustls::SupportedProtocolVersion] = if tls.tls13_only {
            &[&rustls::version::TLS13]
        } else {
            &[&rustls::version::TLS13, &rustls::version::TLS12]
        };
        let builder = ServerConfig::builder_with_provider(Arc::clone(&provider))
            .with_protocol_versions(versions)
            .map_err(|e| invalid(MIN_VERSION_KEY, "", &e.to_string()))?;
        let builder = match (mtls.client_certificates, &mtls.client_ca) {
            (ClientCertificates::Off, _) => builder.with_no_client_auth(),
            (_, None) => {
                // `MtlsConfig::validate` refuses this; never serve without the verifier asked for.
                return Err(invalid(
                    "krishiv.auth.clientCa",
                    "",
                    "required when krishiv.auth.clientCertificates is optional or required",
                ));
            }
            (mode, Some(ca)) => {
                let roots = Arc::new(client_roots(ca)?);
                let verifier =
                    WebPkiClientVerifier::builder_with_provider(roots, Arc::clone(&provider));
                let verifier = if mode == ClientCertificates::Optional {
                    verifier.allow_unauthenticated()
                } else {
                    verifier
                };
                let verifier = verifier
                    .build()
                    .map_err(|e| invalid("krishiv.auth.clientCa", &shown(ca), &e.to_string()))?;
                builder.with_client_cert_verifier(verifier)
            }
        };
        let mut config =
            builder.with_cert_resolver(Arc::clone(&current) as Arc<dyn ResolvesServerCert>);
        // gRPC needs HTTP/2; the admin listener also speaks HTTP/1.1.
        config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
        Ok(Some(Self {
            config: Arc::new(config),
            current,
            cert_file: cert_file.clone(),
            key_file: key_file.clone(),
            provider,
        }))
    }

    /// The configuration both listeners serve.
    #[must_use]
    pub fn server_config(&self) -> Arc<ServerConfig> {
        Arc::clone(&self.config)
    }

    /// Re-reads the certificate and key files. New handshakes get the new material; on an
    /// error the old material keeps serving. Blocking file I/O: an async caller (the `SIGHUP`
    /// handler) runs it on a blocking thread.
    ///
    /// # Errors
    /// As [`ServerTls::new`] for the certificate and key.
    pub fn reload(&self) -> Result<(), KrishivError> {
        let key = load_key(&self.cert_file, &self.key_file, &self.provider)?;
        let mut guard = self
            .current
            .key
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *guard = Arc::new(key);
        Ok(())
    }
}

fn shown(path: &Path) -> String {
    format!("file:{}", path.display())
}

/// The certificate chain and key, checked to belong together.
fn load_key(
    cert_file: &Path,
    key_file: &Path,
    provider: &CryptoProvider,
) -> Result<CertifiedKey, KrishivError> {
    let pem = read_public_file(CERT_KEY, cert_file, MAX_CERT_FILE_BYTES)?;
    let chain = CertificateDer::pem_slice_iter(&pem)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid(CERT_KEY, &shown(cert_file), "not a PEM certificate chain"))?;
    if chain.is_empty() {
        return Err(invalid(CERT_KEY, &shown(cert_file), "holds no certificate"));
    }
    let key_pem = read_private_file(KEY_KEY, key_file, MAX_KEY_FILE_BYTES)?;
    let key = PrivateKeyDer::from_pem_slice(key_pem.expose_secret().as_bytes())
        .map_err(|_| invalid(KEY_KEY, &shown(key_file), "not a PEM private key"))?;
    CertifiedKey::from_der(chain, key, provider).map_err(|e| {
        invalid(
            KEY_KEY,
            &shown(key_file),
            &format!("does not match the certificate or is not supported: {e}"),
        )
    })
}

/// The client CA certificates.
fn client_roots(ca: &Path) -> Result<RootCertStore, KrishivError> {
    let key = "krishiv.auth.clientCa";
    let pem = read_public_file(key, ca, MAX_CERT_FILE_BYTES)?;
    let mut roots = RootCertStore::empty();
    for cert in CertificateDer::pem_slice_iter(&pem) {
        let cert = cert.map_err(|_| invalid(key, &shown(ca), "not a PEM certificate file"))?;
        roots.add(cert).map_err(|e| {
            invalid(
                key,
                &shown(ca),
                &format!("not a usable CA certificate: {e}"),
            )
        })?;
    }
    if roots.is_empty() {
        return Err(invalid(key, &shown(ca), "holds no certificate"));
    }
    Ok(roots)
}
