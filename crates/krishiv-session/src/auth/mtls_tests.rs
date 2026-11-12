// SPDX-License-Identifier: Apache-2.0
//! Client certificates, the DER walk, listener coherence and (feature `tls`) server TLS
//! (P3-T002 PR-14c; design §3.4, §3.5; ADR 0046 §6). Fixtures: `tests/fixtures/tls/README.md`.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Instant;

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::principal::PrincipalKind;

use super::der::{CertNames, DerError};
use super::{
    AuthConfig, AuthFailure, Credentials, Method, PrincipalFrom, Role, ServerAuth, Surface,
    TlsConfig, check_listener,
};

const CLIENT: &[u8] = include_bytes!("../../tests/fixtures/tls/client.der");
const CLIENT2: &[u8] = include_bytes!("../../tests/fixtures/tls/client2.der");
const NOSAN: &[u8] = include_bytes!("../../tests/fixtures/tls/nosan.der");
const SERVER: &[u8] = include_bytes!("../../tests/fixtures/tls/server.der");

const ETL: &str = "spiffe://krishiv.test/etl";
const TOKEN: &str = "cccccccccccccccccccccccccccccccccccccccc";

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/tls/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn peer() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 9)), 50_000)
}

fn config(pairs: &[(&str, &str)]) -> Result<AuthConfig, KrishivError> {
    AuthConfig::from_settings(pairs.iter().copied())
}

fn invalid_config(e: &KrishivError) -> bool {
    e.error_class() == ErrorClass::KrishivInvalidConfigValue
}

/// Client certificates on (`mode`), the principal from `from`, `etl` an operator.
fn mtls(mode: &str, from: &str) -> AuthConfig {
    let ca = format!("file:{}", fixture("ca.pem"));
    config(&[
        ("krishiv.auth.clientCertificates", mode),
        ("krishiv.auth.clientCa", &ca),
        ("krishiv.auth.mtls.principalFrom", from),
        ("krishiv.auth.roles.operator", &format!("{ETL}, etl-job")),
        ("krishiv.auth.roles.viewer", "reporting"),
    ])
    .unwrap()
}

#[test]
fn der_reads_subject_and_alt_names() {
    let names = CertNames::from_der(CLIENT).unwrap();
    assert_eq!(names.common_names, ["etl-job"]);
    assert_eq!(names.uris, [ETL]);
    assert_eq!(names.dns, ["etl.krishiv.test"]);

    let two = CertNames::from_der(CLIENT2).unwrap();
    assert_eq!(two.common_names, ["a", "b"]);
    assert_eq!(two.uris.len(), 2);
    assert_eq!(two.dns, ["a.test", "b.test"]);

    let none = CertNames::from_der(NOSAN).unwrap();
    assert_eq!(none.common_names, ["reporting"]);
    assert!(none.uris.is_empty() && none.dns.is_empty());

    // An IP SAN is skipped, not refused.
    let server = CertNames::from_der(SERVER).unwrap();
    assert_eq!(server.dns, ["localhost"]);
}

#[test]
fn der_refuses_damaged_input_without_panicking() {
    for len in 0..CLIENT.len() {
        assert!(CertNames::from_der(&CLIENT[..len]).is_err(), "prefix {len}");
    }
    let mut longer = CLIENT.to_vec();
    longer.push(0);
    assert_eq!(CertNames::from_der(&longer), Err(DerError::Trailing));
    // Indefinite length (BER) at the outer sequence.
    let mut indefinite = CLIENT.to_vec();
    indefinite[1] = 0x80;
    assert_eq!(CertNames::from_der(&indefinite), Err(DerError::BadLength));
    // Every single-byte change either parses or fails; none panics.
    for i in 0..CLIENT.len() {
        let mut flipped = CLIENT.to_vec();
        flipped[i] ^= 0xff;
        let _ = CertNames::from_der(&flipped);
    }
}

#[test]
fn der_refuses_a_non_minimal_length() {
    // SEQUENCE, long form for a length under 128.
    assert_eq!(
        CertNames::from_der(&[0x30, 0x81, 0x01, 0x00]),
        Err(DerError::BadLength)
    );
    assert_eq!(
        CertNames::from_der(&[0x30, 0x82, 0x00, 0x81]),
        Err(DerError::BadLength)
    );
}

/// One DER element.
fn tlv(tag: u8, contents: &[u8]) -> Vec<u8> {
    let mut out = vec![tag];
    let n = contents.len();
    if n < 0x80 {
        out.push(u8::try_from(n).unwrap());
    } else {
        let bytes: Vec<u8> = n
            .to_be_bytes()
            .into_iter()
            .skip_while(|b| *b == 0)
            .collect();
        out.push(0x80 | u8::try_from(bytes.len()).unwrap());
        out.extend(bytes);
    }
    out.extend_from_slice(contents);
    out
}

/// A `commonName` attribute with value tag `tag`.
fn cn_attr(tag: u8, value: &[u8]) -> Vec<u8> {
    tlv(
        0x30,
        &[tlv(0x06, &[0x55, 0x04, 0x03]), tlv(tag, value)].concat(),
    )
}

/// A certificate shell (unsigned) with `subject` (the contents of the `Name`) and, if given, a
/// subjectAltName extension holding `alt` (the contents of `GeneralNames`).
fn shell(subject: &[u8], alt: Option<&[u8]>) -> Vec<u8> {
    let mut tbs = [
        tlv(0xa0, &tlv(0x02, &[2])),
        tlv(0x02, &[1]),
        tlv(0x30, &[]),
        tlv(0x30, &[]),
        tlv(0x30, &[]),
        tlv(0x30, subject),
        tlv(0x30, &[]),
    ]
    .concat();
    if let Some(alt) = alt {
        let ext = tlv(
            0x30,
            &[tlv(0x06, &[0x55, 0x1d, 0x11]), tlv(0x04, &tlv(0x30, alt))].concat(),
        );
        tbs.extend(tlv(0xa3, &tlv(0x30, &ext)));
    }
    tlv(
        0x30,
        &[tlv(0x30, &tbs), tlv(0x30, &[]), tlv(0x03, &[0])].concat(),
    )
}

#[test]
fn der_reads_a_multi_valued_rdn() {
    // One SET holding two common names: both are read, so the `cn` rule refuses.
    let rdn = tlv(0x31, &[cn_attr(0x0c, b"a"), cn_attr(0x13, b"b")].concat());
    let cert = shell(&rdn, None);
    assert_eq!(CertNames::from_der(&cert).unwrap().common_names, ["a", "b"]);
    let c = mtls("required", "cn");
    assert!(c.mtls.principal_of(&cert).is_none());
}

#[test]
fn der_refuses_constructed_and_invisible_names() {
    let subject = tlv(0x31, &cn_attr(0x0c, b"svc"));
    // A constructed [6] next to a primitive URI would hide a second name.
    let alt = [
        tlv(0x86, b"spiffe://x/a"),
        tlv(0xa6, &tlv(0x16, b"spiffe://x/b")),
    ]
    .concat();
    assert_eq!(
        CertNames::from_der(&shell(&subject, Some(&alt))),
        Err(DerError::Unexpected)
    );
    let alt = tlv(0xa2, &tlv(0x16, b"b.test"));
    assert_eq!(
        CertNames::from_der(&shell(&subject, Some(&alt))),
        Err(DerError::Unexpected)
    );
    // A right-to-left override in a common name (UTF8String).
    let bidi = tlv(0x31, &cn_attr(0x0c, "svc\u{202e}cba".as_bytes()));
    let names = CertNames::from_der(&shell(&bidi, None)).unwrap();
    assert_eq!(names.undecodable_common_names, 1);
    // Other kinds (an IP address, a directory name) are skipped.
    let alt = [
        tlv(0x87, &[127, 0, 0, 1]),
        tlv(0xa4, &tlv(0x30, &[])),
        tlv(0x86, b"u"),
    ]
    .concat();
    assert_eq!(
        CertNames::from_der(&shell(&subject, Some(&alt)))
            .unwrap()
            .uris,
        ["u"]
    );
}

#[test]
fn an_undecodable_common_name_matters_only_to_the_cn_rule() {
    // A BMPString common name next to one URI SAN.
    let subject = tlv(0x31, &cn_attr(0x1e, &[0, b'x']));
    let cert = shell(&subject, Some(&tlv(0x86, b"spiffe://x/svc")));
    let names = CertNames::from_der(&cert).unwrap();
    assert_eq!(names.undecodable_common_names, 1);
    assert_eq!(
        mtls("required", "san-uri")
            .mtls
            .principal_of(&cert)
            .unwrap()
            .0,
        "spiffe://x/svc"
    );
    assert!(mtls("required", "cn").mtls.principal_of(&cert).is_none());
    // With a decodable common name as well, `cn` still refuses: the other could be the second.
    let subject = [
        tlv(0x31, &cn_attr(0x0c, b"svc")),
        tlv(0x31, &cn_attr(0x1e, &[0, b'x'])),
    ]
    .concat();
    assert!(
        mtls("required", "cn")
            .mtls
            .principal_of(&shell(&subject, None))
            .is_none()
    );
}

#[test]
fn mtls_principal_rule() {
    // ADR 0046 §6: exactly one value of the configured kind, no fallback to another kind.
    let cases = [
        ("san-uri", CLIENT, Some(ETL)),
        ("san-dns", CLIENT, Some("etl.krishiv.test")),
        ("cn", CLIENT, Some("etl-job")),
        ("san-uri", CLIENT2, None),
        ("san-dns", CLIENT2, None),
        ("cn", CLIENT2, None),
        ("san-uri", NOSAN, None),
        ("san-dns", NOSAN, None),
        ("cn", NOSAN, Some("reporting")),
        ("san-uri", &[0x30, 0x00], None),
    ];
    for (from, cert, want) in cases {
        let c = mtls("required", from);
        assert_eq!(
            c.mtls.principal_from,
            match from {
                "san-uri" => PrincipalFrom::SanUri,
                "san-dns" => PrincipalFrom::SanDns,
                _ => PrincipalFrom::Cn,
            }
        );
        let got = c.mtls.principal_of(cert).map(|(n, _)| n);
        assert_eq!(got.as_deref(), want, "{from}");
    }
}

#[test]
fn certificate_principals_are_services_unless_listed() {
    let mut c = mtls("required", "san-uri");
    assert_eq!(
        c.mtls.principal_of(CLIENT).unwrap().1,
        PrincipalKind::Service
    );
    c.mtls.user_principals.insert(ETL.to_owned());
    assert_eq!(c.mtls.principal_of(CLIENT).unwrap().1, PrincipalKind::User);
}

#[test]
fn a_certificate_authenticates_with_its_listed_role() {
    let auth = ServerAuth::new(&mtls("required", "san-uri")).unwrap();
    assert!(auth.is_enabled());
    let mut c = Credentials::new(peer(), Surface::Connect);
    c.peer_leaf = Some(CLIENT);
    let a = auth.authenticate_at(&c, Instant::now()).unwrap();
    assert_eq!(
        (a.principal.name.as_str(), a.role, a.method),
        (ETL, Some(Role::Operator), Method::Mtls)
    );
    // A principal in no list has no role (everything is denied later).
    let auth = ServerAuth::new(&mtls("required", "san-dns")).unwrap();
    let a = auth.authenticate_at(&c, Instant::now()).unwrap();
    assert_eq!(a.role, None);
}

#[test]
fn a_certificate_without_one_principal_is_refused_and_counted() {
    let auth = ServerAuth::new(&mtls("required", "san-uri")).unwrap();
    let mut c = Credentials::new(peer(), Surface::Admin);
    c.peer_leaf = Some(CLIENT2);
    assert_eq!(
        auth.authenticate_at(&c, Instant::now()).unwrap_err(),
        AuthFailure::Invalid
    );
    let mtls_admin = auth
        .failure_counts()
        .into_iter()
        .find(|f| f.surface == Surface::Admin && f.method == Method::Mtls)
        .unwrap();
    assert_eq!(mtls_admin.count, 1);
}

#[test]
fn required_certificates_never_fall_back_to_a_token() {
    let dir = tempfile::tempdir().unwrap();
    let secret = secret_file(dir.path());
    let ca = format!("file:{}", fixture("ca.pem"));
    let c = config(&[
        ("krishiv.auth.clientCertificates", "required"),
        ("krishiv.auth.clientCa", &ca),
        ("krishiv.auth.tokens.t.secret", &secret),
        ("krishiv.auth.tokens.t.principal", "tok"),
        ("krishiv.auth.tokens.t.role", "admin"),
    ])
    .unwrap();
    let auth = ServerAuth::new(&c).unwrap();
    let mut cred = Credentials::new(peer(), Surface::Connect);
    cred.bearer = Some(TOKEN);
    assert_eq!(
        auth.authenticate_at(&cred, Instant::now()).unwrap_err(),
        AuthFailure::Missing
    );
}

#[test]
fn optional_certificates_let_a_token_through() {
    let dir = tempfile::tempdir().unwrap();
    let secret = secret_file(dir.path());
    let ca = format!("file:{}", fixture("ca.pem"));
    let c = config(&[
        ("krishiv.auth.clientCertificates", "optional"),
        ("krishiv.auth.clientCa", &ca),
        ("krishiv.auth.tokens.t.secret", &secret),
        ("krishiv.auth.tokens.t.principal", ETL),
        ("krishiv.auth.tokens.t.role", "admin"),
        (
            "krishiv.auth.tokens.u.secret",
            &secret_file_named(dir.path(), "u", 'u'),
        ),
        ("krishiv.auth.tokens.u.principal", "someone-else"),
        ("krishiv.auth.tokens.u.role", "viewer"),
    ])
    .unwrap();
    let auth = ServerAuth::new(&c).unwrap();
    let mut cred = Credentials::new(peer(), Surface::Connect);
    cred.bearer = Some(TOKEN);
    let a = auth.authenticate_at(&cred, Instant::now()).unwrap();
    assert_eq!(a.method, Method::Token);

    // A certificate and a token for the same principal: the certificate decides (role from the
    // lists, here none).
    cred.peer_leaf = Some(CLIENT);
    let a = auth.authenticate_at(&cred, Instant::now()).unwrap();
    assert_eq!((a.principal.name.as_str(), a.method), (ETL, Method::Mtls));

    // A token naming another principal next to a certificate: refused (a confused gateway).
    let other = "u".repeat(40);
    cred.bearer = Some(&other);
    assert_eq!(
        auth.authenticate_at(&cred, Instant::now()).unwrap_err(),
        AuthFailure::Invalid
    );
    // So is an unusable `Authorization` header.
    cred.bearer = None;
    cred.malformed = true;
    assert_eq!(
        auth.authenticate_at(&cred, Instant::now()).unwrap_err(),
        AuthFailure::Invalid
    );
}

fn secret_file(dir: &std::path::Path) -> String {
    secret_file_named(dir, "t", 'c')
}

fn secret_file_named(dir: &std::path::Path, name: &str, fill: char) -> String {
    let path = dir.join(name);
    std::fs::write(&path, fill.to_string().repeat(40)).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    format!("file:{}", path.display())
}

#[test]
fn mtls_settings_are_checked_at_startup() {
    let ca = format!("file:{}", fixture("ca.pem"));
    let e = config(&[("krishiv.auth.clientCertificates", "required")]).unwrap_err();
    assert!(
        invalid_config(&e) && e.to_string().contains("clientCa"),
        "{e}"
    );
    let e = config(&[("krishiv.auth.clientCa", &ca)]).unwrap_err();
    assert!(e.to_string().contains("is off"), "{e}");
    let e = config(&[("krishiv.auth.clientCertificates", "sometimes")]).unwrap_err();
    assert!(invalid_config(&e), "{e}");
    let e = config(&[("krishiv.auth.mtls.principalFrom", "email")]).unwrap_err();
    assert!(invalid_config(&e), "{e}");
    let e = config(&[("krishiv.auth.clientCa", "/no/file/prefix")]).unwrap_err();
    assert!(invalid_config(&e), "{e}");
}

#[test]
fn role_lists_give_one_role_per_principal() {
    let e = config(&[
        ("krishiv.auth.roles.viewer", "a, b"),
        ("krishiv.auth.roles.admin", "b"),
    ])
    .unwrap_err();
    assert!(e.to_string().contains("exactly one role"), "{e}");
    let e = config(&[("krishiv.auth.roles.superuser", "a")]).unwrap_err();
    assert!(invalid_config(&e), "{e}");
    let e = config(&[("krishiv.auth.roles.Admin", "a")]).unwrap_err();
    assert!(invalid_config(&e), "{e}");
    // A key given twice keeps its last value.
    let c = config(&[
        ("krishiv.auth.roles.viewer", "a"),
        ("krishiv.auth.roles.viewer", "b"),
        ("krishiv.auth.roles.admin", "a"),
    ])
    .unwrap();
    assert_eq!(c.roles.role_of("a"), Some(Role::Admin));
    assert_eq!(c.roles.role_of("b"), Some(Role::Viewer));
    assert_eq!(c.roles.role_of("c"), None);
}

#[test]
fn startup_incoherent_configs() {
    let dir = tempfile::tempdir().unwrap();
    let secret = secret_file(dir.path());
    let tokens = config(&[
        ("krishiv.auth.tokens.t.secret", &secret),
        ("krishiv.auth.tokens.t.principal", "p"),
        ("krishiv.auth.tokens.t.role", "admin"),
    ])
    .unwrap();
    let none = config(&[]).unwrap();
    // Built directly: a build without the `tls` feature refuses these settings.
    let tls = TlsConfig {
        cert_file: Some(fixture("server.pem").into()),
        key_file: Some(fixture("server.key").into()),
        tls13_only: false,
    };
    let plain = TlsConfig::default();
    let public: SocketAddr = "0.0.0.0:15002".parse().unwrap();
    let local: SocketAddr = "127.0.0.1:15002".parse().unwrap();
    let local6: SocketAddr = "[::1]:15002".parse().unwrap();

    // A non-loopback address with no authentication.
    let e = check_listener(public, &none, &tls).unwrap_err();
    assert!(
        e.to_string().contains("needs an authentication method"),
        "{e}"
    );
    // A non-loopback address with a token and no TLS.
    let e = check_listener(public, &tokens, &plain).unwrap_err();
    assert!(e.to_string().contains("need TLS"), "{e}");
    // Client certificates without TLS, even on loopback.
    let e = check_listener(local, &mtls("optional", "san-uri"), &plain).unwrap_err();
    assert!(e.to_string().contains("client certificates need"), "{e}");
    // Allowed: loopback with a token and no TLS (the stock client's localhost path), loopback
    // with nothing, and a public address with tokens over TLS.
    check_listener(local, &tokens, &plain).unwrap();
    check_listener(local6, &none, &plain).unwrap();
    check_listener(public, &tokens, &tls).unwrap();
    check_listener(public, &mtls("required", "san-uri"), &tls).unwrap();
}

#[test]
fn tls_settings_are_checked() {
    #[cfg(feature = "tls")]
    {
        let e = TlsConfig::from_settings([("krishiv.server.tls.certFile", "/x.pem")]).unwrap_err();
        assert!(e.to_string().contains("keyFile"), "{e}");
        let e = TlsConfig::from_settings([("krishiv.server.tls.keyFile", "/x.key")]).unwrap_err();
        assert!(e.to_string().contains("certFile"), "{e}");
    }
    let e = TlsConfig::from_settings([("krishiv.server.tls.minVersion", "1.1")]).unwrap_err();
    assert!(invalid_config(&e), "{e}");
    let e = TlsConfig::from_settings([("krishiv.server.tls.ciphers", "x")]).unwrap_err();
    assert!(invalid_config(&e), "{e}");
    let only = TlsConfig::from_settings([("krishiv.server.tls.minVersion", "1.3")]).unwrap();
    assert!(!only.is_on() && only.tls13_only);
}

#[cfg(not(feature = "tls"))]
#[test]
fn tls_settings_need_the_tls_feature() {
    // S1: a build that cannot serve TLS refuses a configuration asking for it.
    for key in ["krishiv.server.tls.certFile", "krishiv.server.tls.keyFile"] {
        let e = TlsConfig::from_settings([(key, "/a.pem")]).unwrap_err();
        assert!(e.to_string().contains("no TLS"), "{e}");
    }
}

#[cfg(feature = "tls")]
#[test]
fn tls_settings_with_files() {
    let c = TlsConfig::from_settings([
        ("krishiv.server.tls.certFile", "file:/a.pem"),
        ("krishiv.server.tls.keyFile", "/a.key"),
        ("krishiv.server.tls.minVersion", "1.3"),
        ("krishiv.other", "ignored"),
    ])
    .unwrap();
    assert!(c.is_on() && c.tls13_only);
    assert_eq!(c.cert_file.as_deref(), Some(std::path::Path::new("/a.pem")));
}

/// Server TLS through rustls, with in-memory handshakes (no sockets).
#[cfg(feature = "tls")]
mod tls {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use rustls::pki_types::pem::PemObject as _;
    use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
    use rustls::{ClientConfig, ClientConnection, RootCertStore, ServerConnection};

    use super::super::{MtlsConfig, ServerTls, TlsConfig};
    use super::{CLIENT, fixture, mtls};

    /// The server's certificate and key copied to `dir` (key mode 0600), as settings.
    fn server_files(dir: &Path, which: &str) -> (PathBuf, PathBuf) {
        let cert = dir.join("server.pem");
        let key = dir.join("server.key");
        std::fs::copy(fixture(&format!("{which}.pem")), &cert).unwrap();
        std::fs::copy(fixture(&format!("{which}.key")), &key).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        (cert, key)
    }

    fn tls_config(cert: &Path, key: &Path) -> TlsConfig {
        TlsConfig {
            cert_file: Some(cert.to_owned()),
            key_file: Some(key.to_owned()),
            tls13_only: false,
        }
    }

    fn provider() -> Arc<rustls::crypto::CryptoProvider> {
        Arc::new(rustls::crypto::aws_lc_rs::default_provider())
    }

    /// A client trusting the test CA, presenting `identity` (a fixture name) if given.
    fn client(identity: Option<&str>) -> Arc<ClientConfig> {
        let mut roots = RootCertStore::empty();
        roots
            .add(CertificateDer::from_pem_file(fixture("ca.pem")).unwrap())
            .unwrap();
        let b = ClientConfig::builder_with_provider(provider())
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_root_certificates(roots);
        Arc::new(match identity {
            None => b.with_no_client_auth(),
            Some(name) => {
                let chain =
                    vec![CertificateDer::from_pem_file(fixture(&format!("{name}.pem"))).unwrap()];
                let key = PrivateKeyDer::from_pem_file(fixture(&format!("{name}.key"))).unwrap();
                b.with_client_auth_cert(chain, key).unwrap()
            }
        })
    }

    /// One handshake in memory. On success, the server's view of the client's leaf and the
    /// leaf the client saw from the server.
    fn handshake(
        server: &ServerTls,
        client: Arc<ClientConfig>,
    ) -> Result<(Option<Vec<u8>>, Vec<u8>), String> {
        let mut s = ServerConnection::new(server.server_config()).map_err(|e| e.to_string())?;
        let name = ServerName::try_from("localhost").map_err(|e| e.to_string())?;
        let mut c = ClientConnection::new(client, name).map_err(|e| e.to_string())?;
        for _ in 0..20 {
            if !c.is_handshaking() && !s.is_handshaking() {
                break;
            }
            let mut buf = Vec::new();
            while c.wants_write() {
                c.write_tls(&mut buf).map_err(|e| e.to_string())?;
            }
            if !buf.is_empty() {
                s.read_tls(&mut buf.as_slice()).map_err(|e| e.to_string())?;
                s.process_new_packets().map_err(|e| e.to_string())?;
            }
            let mut buf = Vec::new();
            while s.wants_write() {
                s.write_tls(&mut buf).map_err(|e| e.to_string())?;
            }
            if !buf.is_empty() {
                c.read_tls(&mut buf.as_slice()).map_err(|e| e.to_string())?;
                c.process_new_packets().map_err(|e| e.to_string())?;
            }
        }
        if c.is_handshaking() || s.is_handshaking() {
            return Err("handshake did not finish".to_owned());
        }
        let peer = s
            .peer_certificates()
            .and_then(|chain| chain.first())
            .map(|leaf| leaf.as_ref().to_vec());
        let served = c
            .peer_certificates()
            .and_then(|chain| chain.first())
            .map(|leaf| leaf.as_ref().to_vec())
            .unwrap_or_default();
        Ok((peer, served))
    }

    #[test]
    fn tls_reload() {
        let dir = tempfile::tempdir().unwrap();
        let (cert, key) = server_files(dir.path(), "server");
        let server = ServerTls::new(&tls_config(&cert, &key), &MtlsConfig::default())
            .unwrap()
            .unwrap();
        let first = handshake(&server, client(None)).unwrap();
        assert_eq!(first.1, super::SERVER);

        // Swap the files and reload: new handshakes get the new certificate.
        server_files(dir.path(), "server2");
        server.reload().unwrap();
        let second = handshake(&server, client(None)).unwrap();
        assert_eq!(
            second.1,
            include_bytes!("../../tests/fixtures/tls/server2.der")
        );

        // Material that fails to load leaves the current one serving.
        std::fs::write(&cert, "not a certificate").unwrap();
        let e = server.reload().unwrap_err();
        assert!(e.to_string().contains("certFile"), "{e}");
        let third = handshake(&server, client(None)).unwrap();
        assert_eq!(third.1, second.1);
    }

    #[test]
    fn required_client_certificates_are_verified_at_the_handshake() {
        let dir = tempfile::tempdir().unwrap();
        let (cert, key) = server_files(dir.path(), "server");
        let c = mtls("required", "san-uri");
        let server = ServerTls::new(&tls_config(&cert, &key), &c.mtls)
            .unwrap()
            .unwrap();
        // The verified leaf reaches the principal rule.
        let (leaf, _) = handshake(&server, client(Some("client"))).unwrap();
        let leaf = leaf.unwrap();
        assert_eq!(leaf, CLIENT);
        assert_eq!(c.mtls.principal_of(&leaf).unwrap().0, super::ETL);
        // No certificate, or an expired one: the handshake fails.
        assert!(handshake(&server, client(None)).is_err());
        let e = handshake(&server, client(Some("expired"))).unwrap_err();
        assert!(e.to_lowercase().contains("expired"), "{e}");
        // A certificate of the same CA without the clientAuth usage (the server's own).
        assert!(handshake(&server, client(Some("server"))).is_err());
    }

    #[test]
    fn optional_client_certificates_allow_none() {
        let dir = tempfile::tempdir().unwrap();
        let (cert, key) = server_files(dir.path(), "server");
        let c = mtls("optional", "san-uri");
        let server = ServerTls::new(&tls_config(&cert, &key), &c.mtls)
            .unwrap()
            .unwrap();
        let (leaf, _) = handshake(&server, client(None)).unwrap();
        assert!(leaf.is_none());
        let (leaf, _) = handshake(&server, client(Some("client"))).unwrap();
        assert_eq!(leaf.as_deref(), Some(CLIENT));
    }

    #[test]
    fn tls_material_is_checked_at_startup() {
        let dir = tempfile::tempdir().unwrap();
        let (cert, key) = server_files(dir.path(), "server");
        // A key readable by others.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o644)).unwrap();
            let e = ServerTls::new(&tls_config(&cert, &key), &MtlsConfig::default()).unwrap_err();
            assert!(e.to_string().contains("readable by group or others"), "{e}");
            std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        // A key that does not belong to the certificate.
        std::fs::copy(fixture("server2.key"), &key).unwrap();
        let e = ServerTls::new(&tls_config(&cert, &key), &MtlsConfig::default()).unwrap_err();
        assert!(e.to_string().contains("does not match"), "{e}");
        // Client certificates asked for with no CA: never served without the verifier.
        server_files(dir.path(), "server");
        let no_ca = MtlsConfig {
            client_certificates: super::super::ClientCertificates::Required,
            ..MtlsConfig::default()
        };
        let e = ServerTls::new(&tls_config(&cert, &key), &no_ca).unwrap_err();
        assert!(e.to_string().contains("clientCa"), "{e}");
        std::fs::copy(fixture("server2.key"), &key).unwrap();
        // TLS off.
        assert!(
            ServerTls::new(&TlsConfig::default(), &MtlsConfig::default())
                .unwrap()
                .is_none()
        );
        // TLS 1.3 only refuses a 1.2-only client.
        server_files(dir.path(), "server");
        let mut only13 = tls_config(&cert, &key);
        only13.tls13_only = true;
        let server = ServerTls::new(&only13, &MtlsConfig::default())
            .unwrap()
            .unwrap();
        let mut roots = RootCertStore::empty();
        roots
            .add(CertificateDer::from_pem_file(fixture("ca.pem")).unwrap())
            .unwrap();
        let old = ClientConfig::builder_with_provider(provider())
            .with_protocol_versions(&[&rustls::version::TLS12])
            .unwrap()
            .with_root_certificates(roots)
            .with_no_client_auth();
        assert!(handshake(&server, Arc::new(old)).is_err());
    }
}
