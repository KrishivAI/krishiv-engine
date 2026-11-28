// SPDX-License-Identifier: Apache-2.0
//! The policy HTTP client (design P3-T067 §5.2, §8 egress tests).

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use object_store::ObjectStoreExt;
use object_store::client::{ClientConfigKey, ClientOptions, HttpConnector, ReqwestConnector};
use object_store::path::Path;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use super::{PolicyConnector, Redirects, checked};
use crate::SharedHttpClients;
use crate::egress::{Cidr, EgressPolicy, EgressScope, PrivateAddresses, ProviderEndpoints};
use crate::http_clients::KEYS;
use crate::http_clients::tests::changed;

fn scope_with(exempt: &[&str], env: &[(&str, &str)]) -> EgressScope {
    let env: HashMap<String, String> = env
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    EgressPolicy::new(
        PrivateAddresses::Deny,
        Vec::<Cidr>::new(),
        ProviderEndpoints::from_env(|k| env.get(k).cloned()),
    )
    .scope(exempt.iter().copied())
}

fn connector(exempt: &[&str]) -> PolicyConnector {
    PolicyConnector::new(scope_with(exempt, &[]), Redirects::Checked, None)
}

/// A one-shot-per-connection HTTP server on loopback answering `response` to every request;
/// returns its port and the request heads it saw.
async fn server(response: String) -> (u16, Arc<std::sync::Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let mut buf = vec![0u8; 8192];
            let n = socket.read(&mut buf).await.unwrap_or(0);
            log.lock()
                .unwrap()
                .push(String::from_utf8_lossy(&buf[..n]).into_owned());
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.shutdown().await;
        }
    });
    (port, seen)
}

fn ok_response() -> String {
    "HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok".to_owned()
}

fn error_chain(e: &(dyn std::error::Error + 'static)) -> String {
    let mut text = e.to_string();
    let mut source = e.source();
    while let Some(s) = source {
        text.push_str(" / ");
        text.push_str(&s.to_string());
        source = s.source();
    }
    text
}

#[test]
fn object_store_pin_reviewed() {
    // The copy mirrors object_store 0.14.2's `ClientOptions::client()`. Bumping the pin fails
    // here until the copy (and `KEYS`) are re-read against the new version.
    let manifest = include_str!("../../../../../Cargo.toml");
    assert!(
        manifest.contains("object_store = { version = \"=0.14.2\""),
        "object_store moved: re-read ClientOptions::client() and update the policy client"
    );
}

#[test]
fn policy_client_honours_every_client_config_key() {
    let plain = connector(&[]);
    let proxied = PolicyConnector::new(
        scope_with(&[], &[]),
        Redirects::Checked,
        Some("http://127.0.0.1:3128".to_owned()),
    );
    for key in KEYS {
        let options = changed(key);
        // object_store's own verdict on the same options: the copy accepts and refuses alike.
        let original = ReqwestConnector::default().connect(&options).is_ok();
        let result = plain.reqwest_client(&options);
        if key == ClientConfigKey::AllowInvalidCertificates {
            // The one deliberate difference: verification is never turned off under the egress
            // rule, whichever mode chose `deny` (an embedded runtime may too; review #803 n1).
            let e = result
                .as_ref()
                .err()
                .map(ToString::to_string)
                .unwrap_or_default();
            assert!(
                e.contains("krishiv.io.egress.privateAddresses=deny"),
                "{key:?}: {e}"
            );
            assert!(result.is_err(), "{key:?}");
            assert!(proxied.reqwest_client(&options).is_err(), "{key:?}");
        } else if key == ClientConfigKey::ProxyUrl {
            // A store proxy without the admin's egress proxy is refused, never silently used
            // or dropped; with it, the store's proxy settings apply.
            assert!(result.is_err(), "{key:?}");
            assert_eq!(
                proxied.reqwest_client(&options).is_ok(),
                original,
                "{key:?}"
            );
        } else {
            assert_eq!(result.is_ok(), original, "{key:?}: {:?}", result.err());
            assert_eq!(
                proxied.reqwest_client(&options).is_ok(),
                original,
                "{key:?}"
            );
        }
    }
    // Asking for verification (the default) is accepted.
    let verified =
        ClientOptions::new().with_config(ClientConfigKey::AllowInvalidCertificates, "false");
    assert!(plain.reqwest_client(&verified).is_ok());
    // The proxy CA is parsed as object_store parses it, once a proxy is in use.
    let with_ca =
        changed(ClientConfigKey::ProxyUrl).with_config(ClientConfigKey::ProxyCaCertificate, "PEM");
    assert_eq!(
        proxied.reqwest_client(&with_ca).is_ok(),
        ReqwestConnector::default().connect(&with_ca).is_ok()
    );
    // A bad value fails as object_store fails it.
    for (key, bad) in [
        (ClientConfigKey::Timeout, "soon"),
        (ClientConfigKey::Http2Only, "maybe"),
        (ClientConfigKey::PoolMaxIdlePerHost, "-1"),
    ] {
        let options = ClientOptions::new().with_config(key, bad);
        assert!(plain.reqwest_client(&options).is_err(), "{key:?}={bad}");
    }
    // A custom resolver would bypass the egress resolver.
    #[derive(Debug)]
    struct Fixed;
    impl object_store::client::DnsResolver for Fixed {
        fn resolve(&self, _: &str) -> object_store::client::DnsFuture {
            Box::pin(async { Ok(Vec::new()) })
        }
    }
    let options = ClientOptions::new().with_dns_resolver(Arc::new(Fixed));
    assert!(plain.reqwest_client(&options).is_err());
}

#[tokio::test]
async fn policy_client_sends_the_user_agent_and_keeps_https_only() {
    let (port, seen) = server(ok_response()).await;
    let conn = connector(&["127.0.0.1"]);
    let url = format!("http://127.0.0.1:{port}/x");
    // `allow_http` unset: https only, nothing is sent.
    let strict = conn.reqwest_client(&ClientOptions::new()).unwrap();
    assert!(strict.get(&url).send().await.is_err());
    assert!(seen.lock().unwrap().is_empty());
    let options = ClientOptions::new()
        .with_allow_http(true)
        .with_config(ClientConfigKey::UserAgent, "krishiv-ua-test");
    let client = conn.reqwest_client(&options).unwrap();
    assert_eq!(client.get(&url).send().await.unwrap().status(), 200);
    let head = seen.lock().unwrap()[0].to_ascii_lowercase();
    assert!(head.contains("user-agent: krishiv-ua-test"), "{head}");
    assert!(!head.contains("gzip"), "{head}");
}

#[tokio::test]
async fn object_store_redirect_to_metadata_address_refused() {
    for target in [
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "http://[fd00:ec2::254]/latest/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://10.1.2.3/",
    ] {
        let (port, seen) = server(format!(
            "HTTP/1.1 302 Found\r\nlocation: {target}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
        ))
        .await;
        let store = object_store::http::HttpBuilder::new()
            .with_url(format!("http://127.0.0.1:{port}"))
            .with_client_options(ClientOptions::new().with_allow_http(true))
            .with_retry(object_store::RetryConfig {
                max_retries: 0,
                ..object_store::RetryConfig::default()
            })
            .with_http_connector(connector(&["127.0.0.1"]))
            .build()
            .unwrap();
        let err = store.get(&Path::from("data.parquet")).await.unwrap_err();
        let text = error_chain(&err);
        assert!(text.contains("private address"), "{target}: {text}");
        assert_eq!(seen.lock().unwrap().len(), 1, "{target}");
    }
}

#[tokio::test]
async fn catalog_clients_follow_no_redirect() {
    let (port, _) = server(
        "HTTP/1.1 302 Found\r\nlocation: http://127.0.0.1:1/\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
            .to_owned(),
    )
    .await;
    let conn = PolicyConnector::new(scope_with(&["127.0.0.1"], &[]), Redirects::None, None);
    let client = conn
        .reqwest_client(&ClientOptions::new().with_allow_http(true))
        .unwrap();
    let response = client
        .get(format!("http://127.0.0.1:{port}/v1/config"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 302);
}

#[tokio::test]
async fn catalog_client_follows_no_redirect_and_checks_hosts() {
    let (port, seen) = server(
        "HTTP/1.1 302 Found\r\nlocation: http://127.0.0.1:1/\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
            .to_owned(),
    )
    .await;
    // A connector that follows checked redirects for stores still gives catalog clients none.
    let conn = connector(&["127.0.0.1"]).without_provider_literals();
    let client = conn
        .catalog_client(
            std::time::Duration::from_secs(5),
            true,
            "krishiv-catalog-test",
        )
        .unwrap();
    let url = format!("http://127.0.0.1:{port}/v1/config");
    let response = client.get(&url).send().await.unwrap();
    assert_eq!(response.status(), 302);
    let head = seen.lock().unwrap()[0].to_ascii_lowercase();
    assert!(head.contains("user-agent: krishiv-catalog-test"), "{head}");
    // https only unless allowed.
    let strict = conn
        .catalog_client(std::time::Duration::from_secs(5), false, "ua")
        .unwrap();
    assert!(strict.get(&url).send().await.is_err());
    // The request host check: names pass, the declared literal passes, any other private or
    // provider literal in any spelling is refused, never named.
    conn.check_request_host("catalog.example.com").unwrap();
    conn.check_request_host("127.0.0.1").unwrap();
    for host in ["10.0.0.9", "2852039166", "169.254.169.254", "0x7f.0.0.2"] {
        let e = conn.check_request_host(host).unwrap_err();
        assert_eq!(e.to_string(), "private address", "{host}");
    }
}

#[tokio::test]
async fn literal_request_checked_before_it_is_sent() {
    let conn = connector(&[]);
    let client = conn
        .connect(&ClientOptions::new().with_allow_http(true))
        .unwrap();
    let request = http::Request::builder()
        .uri("http://10.0.0.9:9/x")
        .body(object_store::client::HttpRequestBody::empty())
        .unwrap();
    let err = client.execute(request).await.unwrap_err();
    assert!(error_chain(&err).contains("private address"), "{err}");
}

#[tokio::test]
async fn egress_resolver_pins_checked_address() {
    let (port, seen) = server(ok_response()).await;
    // A name that first answers loopback (exempt by name), then a link-local address: the
    // second request is refused at resolution and never connects.
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let answers: super::Answers = Arc::new(move |host: &str| {
        assert_eq!(host, "rebind.test");
        if counter.fetch_add(1, Ordering::SeqCst) == 0 {
            vec!["127.0.0.1".parse::<IpAddr>().unwrap()]
        } else {
            vec!["169.254.169.254".parse::<IpAddr>().unwrap()]
        }
    });
    let conn = connector(&["rebind.test"]).with_answers(answers);
    let options = ClientOptions::new()
        .with_allow_http(true)
        .with_config(ClientConfigKey::PoolMaxIdlePerHost, "0");
    let client = conn.reqwest_client(&options).unwrap();
    let url = format!("http://rebind.test:{port}/");
    assert_eq!(client.get(&url).send().await.unwrap().status(), 200);
    let err = client.get(&url).send().await.unwrap_err();
    assert!(error_chain(&err).contains("no allowed address"), "{err}");
    assert_eq!(seen.lock().unwrap().len(), 1);
    // Mixed answers: only the allowed ones are dialled.
    let scope = scope_with(&[], &[]);
    let kept = checked(
        &scope,
        "mixed.test",
        &["10.0.0.1".parse().unwrap(), "8.8.8.8".parse().unwrap()],
        0,
    )
    .unwrap();
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].ip().to_string(), "8.8.8.8");
}

#[tokio::test]
async fn ambient_instance_credentials_still_load_under_policy() {
    // Azure App Service leg: `IDENTITY_ENDPOINT` names a loopback host; the provider pin admits it
    // with no by-name exemption on the client.
    let (port, seen) = server(ok_response()).await;
    let endpoint = format!("http://localhost:{port}/msi/token");
    let scope = scope_with(&[], &[("IDENTITY_ENDPOINT", endpoint.as_str())]);
    let conn = PolicyConnector::new(scope.clone(), Redirects::Checked, None);
    let client = conn
        .reqwest_client(&ClientOptions::new().with_allow_http(true))
        .unwrap();
    let response = client
        .get(format!("http://127.0.0.1:{port}/").replace("127.0.0.1", "localhost"))
        .send()
        .await;
    // `localhost` may resolve to ::1 first where the server is not listening; the pin admits
    // both, and reqwest falls back to 127.0.0.1.
    assert_eq!(response.unwrap().status(), 200);
    assert_eq!(seen.lock().unwrap().len(), 1);
    // GCP leg: the metadata host name passes resolution only to its own address.
    let metadata: IpAddr = "169.254.169.254".parse().unwrap();
    assert!(checked(&scope, "metadata.google.internal", &[metadata], 0).is_ok());
    assert!(
        checked(
            &scope,
            "metadata.google.internal",
            &["10.0.0.1".parse().unwrap()],
            0
        )
        .is_err()
    );
    // AWS leg: the instance-metadata literal passes for a request the store built itself.
    assert!(scope.permits_request_literal("169.254.169.254").is_ok());
    // Without the variable the same loopback request is refused.
    let bare = PolicyConnector::new(scope_with(&[], &[]), Redirects::Checked, None)
        .reqwest_client(&ClientOptions::new().with_allow_http(true))
        .unwrap();
    assert!(
        bare.get(format!("http://localhost:{port}/"))
            .send()
            .await
            .is_err()
    );
}

#[test]
fn policy_clients_shared() {
    let policy = EgressPolicy::new(
        PrivateAddresses::Deny,
        Vec::new(),
        ProviderEndpoints::default(),
    );
    let clients = SharedHttpClients::with_egress(PolicyConnector::new(
        policy.scope(["minio.internal"]),
        Redirects::Checked,
        None,
    ));
    assert!(clients.enforces_egress());
    let options = ClientOptions::new().with_allow_http(true);
    for _ in 0..3 {
        clients.connect(&options).unwrap();
    }
    assert_eq!(clients.len(), 1);
    clients
        .connect(&options.with_config(ClientConfigKey::Timeout, "9s"))
        .unwrap();
    assert_eq!(clients.len(), 2);
    // The fingerprint carries the scope: another exemption set is another client.
    let a = PolicyConnector::new(policy.scope(["a"]), Redirects::Checked, None);
    let b = PolicyConnector::new(policy.scope(["b"]), Redirects::Checked, None);
    assert_ne!(a.fingerprint(), b.fingerprint());
    // An allow policy builds plain clients.
    let allow = EgressPolicy::new(
        PrivateAddresses::Allow,
        Vec::new(),
        ProviderEndpoints::default(),
    );
    let plain = SharedHttpClients::with_egress(PolicyConnector::new(
        allow.scope::<_, &str>([]),
        Redirects::Checked,
        None,
    ));
    assert!(!plain.enforces_egress());
}

/// Run as a child process with proxy variables set (see the parent test below).
#[tokio::test]
async fn env_proxy_ignored_child() {
    let Ok(port) = std::env::var("KRISHIV_EGRESS_PROXY_CHILD_PORT") else {
        return;
    };
    let client = connector(&["127.0.0.1"])
        .reqwest_client(&ClientOptions::new().with_allow_http(true))
        .unwrap();
    let response = client
        .get(format!("http://127.0.0.1:{port}/"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn env_proxy_ignored_by_policy_clients() {
    let (port, seen) = server(ok_response()).await;
    let exe = std::env::current_exe().unwrap();
    let dead_proxy = "http://127.0.0.1:9";
    let output = tokio::process::Command::new(exe)
        .args([
            "--exact",
            "http_clients::policy::tests::env_proxy_ignored_child",
        ])
        .env("KRISHIV_EGRESS_PROXY_CHILD_PORT", port.to_string())
        .env("HTTP_PROXY", dead_proxy)
        .env("http_proxy", dead_proxy)
        .env("ALL_PROXY", dead_proxy)
        .env("all_proxy", dead_proxy)
        .env_remove("NO_PROXY")
        .env_remove("no_proxy")
        .output()
        .await
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(stdout.contains("1 passed"), "{stdout}");
    assert_eq!(seen.lock().unwrap().len(), 1, "the request went direct");
}

/// Review B1: a request whose host is a non-canonical IPv4 literal is checked as that address
/// (reqwest dials such a host without the resolver).
#[tokio::test]
async fn noncanonical_literal_request_refused() {
    let (port, seen) = server(ok_response()).await;
    let client = connector(&[])
        .connect(&ClientOptions::new().with_allow_http(true))
        .unwrap();
    for host in [
        "2130706433",
        "127.1",
        "0x7f.0.0.1",
        "0177.0.0.1",
        "127.0.0.1.",
    ] {
        let request = http::Request::builder()
            .uri(format!("http://{host}:{port}/x"))
            .body(object_store::client::HttpRequestBody::empty())
            .unwrap();
        let err = client.execute(request).await.unwrap_err();
        assert!(
            error_chain(&err).contains("private address"),
            "{host}: {err}"
        );
    }
    assert!(
        seen.lock().unwrap().is_empty(),
        "nothing reached the server"
    );
}

/// Review B2: the provider literals are admitted only on the clients of stores that have
/// ambient credential providers; a data read of a metadata URL is refused before it is dialled,
/// while the S3 client still reaches the instance-metadata literal.
#[tokio::test]
async fn data_read_of_provider_literal_refused() {
    use crate::{StoreRegistry, parse_object_url};
    let policy = EgressPolicy::new(
        PrivateAddresses::Deny,
        Vec::new(),
        ProviderEndpoints::from_env(|_| None),
    );
    let clients = SharedHttpClients::with_egress(PolicyConnector::new(
        policy.scope::<_, &str>([]),
        Redirects::Checked,
        None,
    ));
    let registry =
        StoreRegistry::new(Arc::new(std::collections::BTreeMap::<String, String>::new()))
            .with_http_clients(clients.clone());
    for location in [
        "https://169.254.169.254/latest/meta-data/iam/security-credentials/role",
        "https://169.254.170.2/v2/credentials/x",
        "https://[fd00:ec2::254]/latest/meta-data/",
    ] {
        let url = parse_object_url(location).unwrap();
        let store = registry.store(&url).unwrap();
        // The policy's own refusal, not a connect or timeout error: the request was stopped
        // before any dial (no wall-clock bound, which load makes flaky).
        let err = store.get(&url.key).await.unwrap_err();
        assert!(
            error_chain(&err).contains("private address"),
            "{location}: {err}"
        );
    }
    // A store with ambient credential providers still builds on the same cache.
    assert!(
        registry
            .store(&parse_object_url("s3://bucket/key").unwrap())
            .is_ok()
    );
    assert!(
        registry
            .store(&parse_object_url("gs://bucket/key").unwrap())
            .is_ok()
    );
    // The credential path keeps the provider literals (N1).
    let client = clients
        .connect(&ClientOptions::new().with_allow_http(true))
        .unwrap();
    let request = http::Request::builder()
        .uri("http://10.0.0.9:9/x")
        .body(object_store::client::HttpRequestBody::empty())
        .unwrap();
    assert!(error_chain(&client.execute(request).await.unwrap_err()).contains("private address"));
    let request = http::Request::builder()
        .uri("http://169.254.169.254:9/latest/api/token")
        .body(object_store::client::HttpRequestBody::empty())
        .unwrap();
    let err = client.execute(request).await.unwrap_err();
    assert!(!error_chain(&err).contains("private address"), "{err}");
}

/// Review B2: an S3, GCS or Azure endpoint (startup or per read) at an address the rule refuses
/// fails when the store is built, so the provider-literal admission of those clients never
/// carries a data request.
#[test]
fn store_endpoint_at_provider_literal_refused_at_build() {
    use crate::{StoreRegistry, parse_object_url};
    let policy = EgressPolicy::new(
        PrivateAddresses::Deny,
        Vec::new(),
        ProviderEndpoints::from_env(|_| None),
    );
    let clients = SharedHttpClients::with_egress(PolicyConnector::new(
        policy.scope::<_, &str>([]),
        Redirects::Checked,
        None,
    ));
    for (scheme_url, key, endpoint) in [
        ("s3://b/k", "fs.s3a.endpoint", "http://169.254.169.254"),
        ("s3://b/k", "fs.s3a.endpoint", "http://2852039166:80"),
        (
            "gs://b/k",
            "fs.gs.storage.root.url",
            "http://169.254.169.254",
        ),
        (
            "abfss://c@a.dfs.core.windows.net/k",
            "fs.azure.endpoint",
            "http://169.254.169.254",
        ),
    ] {
        let settings: std::collections::BTreeMap<String, String> =
            [(key.to_owned(), endpoint.to_owned())].into();
        let registry = StoreRegistry::new(Arc::new(settings)).with_http_clients(clients.clone());
        let err = registry
            .store(&parse_object_url(scheme_url).unwrap())
            .err()
            .unwrap_or_else(|| panic!("{key}={endpoint} built"));
        let text = err.to_string();
        assert!(text.contains("egress"), "{key}: {text}");
        assert!(
            !text.contains("169.254") && !text.contains("2852039166"),
            "{text}"
        );
    }
}

/// Runs this test binary's test `name` again with an empty system trust store (rustls-native-certs reads
/// `SSL_CERT_FILE` / `SSL_CERT_DIR`), so the process environment of this one is never changed.
fn rerun_without_system_roots(name: &str) {
    let dir = std::env::temp_dir().join(format!("krishiv-no-roots-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture"])
        .env("KRISHIV_TEST_NO_ROOTS", "1")
        .env("SSL_CERT_FILE", dir.join("absent.pem"))
        .env("SSL_CERT_DIR", &dir)
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "{text}");
    assert!(text.contains("1 passed"), "the child ran nothing: {text}");
}

#[tokio::test]
async fn no_system_roots() {
    if std::env::var_os("KRISHIV_TEST_NO_ROOTS").is_none() {
        rerun_without_system_roots("http_clients::policy::tests::no_system_roots");
        return;
    }
    let conn = connector(&["127.0.0.1"]);
    // HTTPS only: refused, by name.
    let err = conn
        .catalog_client(std::time::Duration::from_secs(5), false, "krishiv/test")
        .unwrap_err();
    assert!(super::is_missing_tls_roots(&err), "{err}");
    assert!(
        err.to_string().contains("no TLS root certificates"),
        "{err}"
    );
    // Plain HTTP allowed: the client builds and an http:// request goes out.
    let (port, seen) =
        server("HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}".to_owned())
            .await;
    let client = conn
        .catalog_client(std::time::Duration::from_secs(5), true, "krishiv/test")
        .unwrap();
    let response = client
        .get(format!("http://127.0.0.1:{port}/v1/config"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(seen.lock().unwrap().len(), 1);
}
