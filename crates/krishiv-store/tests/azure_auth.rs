// SPDX-License-Identifier: Apache-2.0
//! Azure credentials from Hadoop ABFS keys against a loopback Azure stand-in: the request each
//! configuration sends (its `Authorization` header or SAS query), the OAuth token requests
//! object_store makes for `ClientCredsTokenProvider`, `WorkloadIdentityTokenProvider` and
//! `MsiTokenProvider`, the per-host scoping of every key, and the precedence of explicit
//! configuration over the process's `AZURE_*` environment (checked in a child process, since a
//! test cannot change its own environment safely).
#![cfg(feature = "cloud")]
#![expect(clippy::unwrap_used, reason = "test code")]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use krishiv_common::error::ErrorClass;
use krishiv_store::StoreRegistry;
use object_store::ObjectStoreExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// One request the stand-in saw.
#[derive(Debug, Clone)]
struct Seen {
    /// `GET /path?query HTTP/1.1`.
    line: String,
    /// The `Authorization` header value, if any.
    auth: Option<String>,
    /// The request body (a token request's form).
    body: String,
}

type Log = Arc<Mutex<Vec<Seen>>>;

/// A loopback stand-in for Azure Blob Storage, Entra ID's token endpoint and IMDS: token paths
/// answer a token named after the flow, every other request is a 404 (so a `head` fails fast,
/// with no retry, after one signed request).
async fn stand_in() -> (String, Log) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let log: Log = Arc::default();
    let seen = Arc::clone(&log);
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let seen = Arc::clone(&seen);
            tokio::spawn(async move {
                let mut request = Vec::new();
                let mut buf = [0u8; 8192];
                let head_end = loop {
                    if let Some(i) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        break i + 4;
                    }
                    let n = socket.read(&mut buf).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    request.extend_from_slice(&buf[..n]);
                };
                let head = String::from_utf8_lossy(&request[..head_end]).to_string();
                let header = |name: &str| {
                    head.lines().find_map(|l| {
                        let (k, v) = l.split_once(':')?;
                        k.eq_ignore_ascii_case(name).then(|| v.trim().to_owned())
                    })
                };
                let length: usize = header("content-length")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                while request.len() < head_end + length {
                    let n = socket.read(&mut buf).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    request.extend_from_slice(&buf[..n]);
                }
                let line = head.lines().next().unwrap_or_default().to_owned();
                let body = String::from_utf8_lossy(&request[head_end..]).to_string();
                seen.lock().unwrap().push(Seen {
                    line: line.clone(),
                    auth: header("authorization"),
                    body,
                });
                let in_an_hour = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    + 3600;
                let (status, body) = if line.contains("/oauth2/v2.0/token") {
                    let token = if line.contains("/wi-tenant/") {
                        "workload-token"
                    } else {
                        "client-token"
                    };
                    (
                        "200 OK",
                        format!(
                            r#"{{"token_type":"Bearer","expires_in":3600,"access_token":"{token}"}}"#
                        ),
                    )
                } else if line.contains("/metadata/identity/oauth2/token") {
                    (
                        "200 OK",
                        format!(
                            r#"{{"token_type":"Bearer","expires_on":"{in_an_hour}","access_token":"msi-token"}}"#
                        ),
                    )
                } else {
                    ("404 The specified blob does not exist.", String::new())
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    });
    (endpoint, log)
}

fn registry(pairs: &[(&str, &str)]) -> StoreRegistry {
    let settings: BTreeMap<String, String> = pairs
        .iter()
        .map(|&(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    StoreRegistry::new(Arc::new(settings))
}

/// `head`s `url` through `registry` (a 404 from the stand-in) and returns the storage request.
async fn storage_request(registry: &StoreRegistry, url: &str, log: &Log) -> Seen {
    let opened = registry.open(url).unwrap();
    let error = opened.store.head(&opened.key).await.unwrap_err();
    assert!(
        matches!(error, object_store::Error::NotFound { .. }),
        "{error}"
    );
    let seen = log.lock().unwrap();
    seen.iter()
        .rev()
        .find(|s| s.line.starts_with("HEAD "))
        .cloned()
        .unwrap()
}

/// The `fs.azure.endpoint` of the stand-in for the Azurite-style account `acct`.
fn account_endpoint(endpoint: &str) -> String {
    format!("{endpoint}/acct")
}

const URL: &str = "abfss://cont@acct.dfs.core.windows.net/dir/a.csv";
/// A valid base64 account key (not a real one).
const ACCOUNT_KEY: &str = "a2V5LWZvci10ZXN0cy1vbmx5LW5vdC1hLXJlYWwta2V5LTAxMjM0NTY3ODk=";

#[tokio::test]
async fn shared_key_sas_and_static_bearer_sign_the_request() {
    let (endpoint, log) = stand_in().await;
    let base = account_endpoint(&endpoint);

    // SharedKey (the Hadoop default auth type), account key scoped by host.
    let shared = registry(&[
        ("fs.azure.endpoint", &base),
        (
            "fs.azure.account.key.acct.dfs.core.windows.net",
            ACCOUNT_KEY,
        ),
    ]);
    let seen = storage_request(&shared, URL, &log).await;
    assert!(
        seen.auth.as_deref().unwrap().starts_with("SharedKey acct:"),
        "{seen:?}"
    );
    assert!(
        seen.line.starts_with("HEAD /acct/cont/dir/a.csv "),
        "{seen:?}"
    );

    // SAS: Hadoop's fixed token key, scoped by host.
    let sas = registry(&[
        ("fs.azure.endpoint", &base),
        (
            "fs.azure.account.auth.type.acct.dfs.core.windows.net",
            "SAS",
        ),
        (
            "fs.azure.sas.fixed.token.acct.dfs.core.windows.net",
            "sv=2024-01-01&sp=r&sig=c2lnbmF0dXJl",
        ),
    ]);
    let seen = storage_request(&sas, URL, &log).await;
    assert_eq!(seen.auth, None, "{seen:?}");
    assert!(seen.line.contains("sig=c2lnbmF0dXJl"), "{seen:?}");

    // A static bearer token (Krishiv's key) under OAuth.
    let bearer = registry(&[
        ("fs.azure.endpoint", &base),
        ("fs.azure.account.auth.type", "OAuth"),
        ("fs.azure.account.oauth2.access.token", "static-token-1"),
    ]);
    let seen = storage_request(&bearer, URL, &log).await;
    assert_eq!(seen.auth.as_deref(), Some("Bearer static-token-1"));
}

#[tokio::test]
async fn client_credentials_fetch_a_token_from_the_endpoint_tenant() {
    let (endpoint, log) = stand_in().await;
    let registry = registry(&[
        ("fs.azure.endpoint", &account_endpoint(&endpoint)),
        ("fs.azure.account.auth.type", "OAuth"),
        (
            "fs.azure.account.oauth.provider.type",
            "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
        ),
        ("fs.azure.account.oauth2.client.id", "app-id"),
        ("fs.azure.account.oauth2.client.secret", "app-secret"),
        (
            "fs.azure.account.oauth2.client.endpoint",
            &format!("{endpoint}/my-tenant/oauth2/token"),
        ),
    ]);
    let seen = storage_request(&registry, URL, &log).await;
    assert_eq!(seen.auth.as_deref(), Some("Bearer client-token"));
    let token = log
        .lock()
        .unwrap()
        .iter()
        .find(|s| s.line.contains("/oauth2/v2.0/token"))
        .cloned()
        .unwrap();
    assert!(
        token.line.starts_with("POST /my-tenant/oauth2/v2.0/token "),
        "{token:?}"
    );
    assert!(token.body.contains("client_id=app-id"), "{token:?}");
    assert!(token.body.contains("client_secret=app-secret"), "{token:?}");
    assert!(
        token.body.contains("grant_type=client_credentials"),
        "{token:?}"
    );
}

#[tokio::test]
async fn workload_identity_and_managed_identity_fetch_their_tokens() {
    let (endpoint, log) = stand_in().await;
    let dir = std::env::temp_dir().join(format!("krishiv-azure-wi-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let token_file = dir.join("token");
    std::fs::write(&token_file, "federated-jwt").unwrap();
    let workload = registry(&[
        ("fs.azure.endpoint", &account_endpoint(&endpoint)),
        ("fs.azure.account.auth.type", "OAuth"),
        (
            "fs.azure.account.oauth.provider.type",
            "org.apache.hadoop.fs.azurebfs.oauth2.WorkloadIdentityTokenProvider",
        ),
        ("fs.azure.account.oauth2.client.id", "wi-app"),
        ("fs.azure.account.oauth2.msi.tenant", "wi-tenant"),
        (
            "fs.azure.account.oauth2.msi.authority",
            &format!("{endpoint}/"),
        ),
        (
            "fs.azure.account.oauth2.token.file",
            token_file.to_str().unwrap(),
        ),
    ]);
    let seen = storage_request(&workload, URL, &log).await;
    assert_eq!(seen.auth.as_deref(), Some("Bearer workload-token"));
    let token = log
        .lock()
        .unwrap()
        .iter()
        .find(|s| s.line.contains("/wi-tenant/oauth2/v2.0/token"))
        .cloned()
        .unwrap();
    assert!(token.body.contains("client_id=wi-app"), "{token:?}");
    assert!(token.body.contains("federated-jwt"), "{token:?}");
    std::fs::remove_dir_all(&dir).unwrap();

    let managed = registry(&[
        ("fs.azure.endpoint", &account_endpoint(&endpoint)),
        ("fs.azure.account.auth.type", "OAuth"),
        (
            "fs.azure.account.oauth.provider.type",
            "org.apache.hadoop.fs.azurebfs.oauth2.MsiTokenProvider",
        ),
        ("fs.azure.account.oauth2.client.id", "mi-client"),
        (
            "fs.azure.account.oauth2.msi.endpoint",
            &format!("{endpoint}/metadata/identity/oauth2/token"),
        ),
    ]);
    let seen = storage_request(&managed, URL, &log).await;
    assert_eq!(seen.auth.as_deref(), Some("Bearer msi-token"));
    let token = log
        .lock()
        .unwrap()
        .iter()
        .find(|s| s.line.contains("/metadata/identity/oauth2/token"))
        .cloned()
        .unwrap();
    assert!(token.line.contains("client_id=mi-client"), "{token:?}");
}

#[tokio::test]
async fn host_scoped_keys_beat_bare_keys_as_abfs_resolves_them() {
    let (endpoint, log) = stand_in().await;
    // `fs.azure.account.auth.type.<host>` beats the bare key (AbfsConfiguration.getEnum), and a
    // bare key applies to every account.
    let registry = registry(&[
        ("fs.azure.endpoint", &account_endpoint(&endpoint)),
        ("fs.azure.account.auth.type", "SharedKey"),
        ("fs.azure.account.key", ACCOUNT_KEY),
        (
            "fs.azure.account.auth.type.acct.dfs.core.windows.net",
            "OAuth",
        ),
        (
            "fs.azure.account.oauth2.access.token.acct.dfs.core.windows.net",
            "scoped-token",
        ),
        ("fs.azure.account.oauth2.access.token", "bare-token"),
    ]);
    let seen = storage_request(&registry, URL, &log).await;
    assert_eq!(seen.auth.as_deref(), Some("Bearer scoped-token"));
    // Another account takes the bare settings.
    let seen = storage_request(
        &registry,
        "abfss://cont@other.dfs.core.windows.net/dir/a.csv",
        &log,
    )
    .await;
    assert!(
        seen.auth
            .as_deref()
            .unwrap()
            .starts_with("SharedKey other:"),
        "{seen:?}"
    );
}

#[tokio::test]
async fn a_new_static_token_reaches_the_next_request() {
    let (endpoint, log) = stand_in().await;
    let base = account_endpoint(&endpoint);
    let settings = |token: &str| {
        registry(&[
            ("fs.azure.endpoint", &base),
            ("fs.azure.account.auth.type", "OAuth"),
            ("fs.azure.account.oauth2.access.token", token),
        ])
    };
    // Each query builds its registry from the session's settings snapshot: setting the key again
    // gives the next query a store under another id, signing with the new token.
    let (old, new) = (settings("token-a"), settings("token-b"));
    let old_id = old.open(URL).unwrap().id;
    let new_id = new.open(URL).unwrap().id;
    assert_ne!(old_id, new_id);
    assert!(!old_id.to_string().contains("token-a"));
    let seen = storage_request(&old, URL, &log).await;
    assert_eq!(seen.auth.as_deref(), Some("Bearer token-a"));
    let seen = storage_request(&new, URL, &log).await;
    assert_eq!(seen.auth.as_deref(), Some("Bearer token-b"));
}

#[tokio::test]
async fn onelake_paths_reach_the_fabric_endpoint() {
    let (endpoint, log) = stand_in().await;
    // The endpoint override stands in for `https://onelake.blob.fabric.microsoft.com`; the
    // request path is the workspace then the item path, as on OneLake.
    let registry = registry(&[
        ("fs.azure.endpoint", &format!("{endpoint}/onelake")),
        ("fs.azure.account.auth.type", "OAuth"),
        ("fs.azure.account.oauth2.access.token", "entra-token"),
    ]);
    let seen = storage_request(
        &registry,
        "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Files/a.csv",
        &log,
    )
    .await;
    assert!(
        seen.line
            .starts_with("HEAD /onelake/ws/lh.Lakehouse/Files/a.csv "),
        "{seen:?}"
    );
    assert_eq!(seen.auth.as_deref(), Some("Bearer entra-token"));
}

#[test]
fn unsupported_auth_types_and_providers_are_refused_naming_the_key() {
    for (pairs, key, class) in [
        (
            vec![("fs.azure.account.auth.type", "Custom")],
            "fs.azure.account.auth.type",
            ErrorClass::KrishivUnsupportedFeature,
        ),
        (
            vec![("fs.azure.account.auth.type", "Kerberos")],
            "fs.azure.account.auth.type",
            ErrorClass::KrishivInvalidConfigValue,
        ),
        (
            vec![
                ("fs.azure.account.auth.type", "OAuth"),
                (
                    "fs.azure.account.oauth.provider.type",
                    "org.apache.hadoop.fs.azurebfs.oauth2.RefreshTokenBasedTokenProvider",
                ),
            ],
            "RefreshTokenBasedTokenProvider",
            ErrorClass::KrishivUnsupportedFeature,
        ),
        (
            vec![
                ("fs.azure.account.auth.type", "OAuth"),
                ("fs.azure.account.oauth.provider.type", "com.example.Mine"),
            ],
            "com.example.Mine",
            ErrorClass::KrishivUnsupportedFeature,
        ),
        (
            vec![("fs.azure.account.auth.type", "OAuth")],
            "fs.azure.account.oauth.provider.type",
            ErrorClass::KrishivInvalidConfigValue,
        ),
        (
            vec![
                ("fs.azure.account.auth.type", "OAuth"),
                (
                    "fs.azure.account.oauth.provider.type",
                    "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
                ),
                ("fs.azure.account.oauth2.client.id", "id"),
                (
                    "fs.azure.account.oauth2.client.endpoint",
                    "https://login.microsoftonline.com/t/oauth2/token",
                ),
            ],
            "fs.azure.account.oauth2.client.secret",
            ErrorClass::KrishivInvalidConfigValue,
        ),
        (
            vec![
                ("fs.azure.account.auth.type", "OAuth"),
                (
                    "fs.azure.account.oauth.provider.type",
                    "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
                ),
                ("fs.azure.account.oauth2.client.id", "id"),
                ("fs.azure.account.oauth2.client.secret", "Canary-7f3a9c"),
                (
                    "fs.azure.account.oauth2.client.endpoint",
                    "http://login.example.com/t/oauth2/token",
                ),
            ],
            "fs.azure.account.oauth2.client.endpoint",
            ErrorClass::KrishivInvalidConfigValue,
        ),
        (
            vec![("fs.azure.account.auth.type", "SAS")],
            "fs.azure.sas.fixed.token",
            ErrorClass::KrishivInvalidConfigValue,
        ),
        (
            vec![("fs.azure.account.auth.type", "SharedKey")],
            "fs.azure.account.key",
            ErrorClass::KrishivInvalidConfigValue,
        ),
        (
            vec![
                ("fs.azure.account.auth.type", "OAuth"),
                (
                    "fs.azure.account.oauth.provider.type",
                    "org.apache.hadoop.fs.azurebfs.oauth2.MsiTokenProvider",
                ),
                ("fs.azure.account.oauth2.access.token", "Canary-7f3a9c"),
            ],
            "fs.azure.account.oauth2.access.token",
            ErrorClass::KrishivInvalidConfigValue,
        ),
    ] {
        let error = registry(&pairs).open(URL).map(|_| ()).unwrap_err();
        assert_eq!(error.error_class(), class, "{pairs:?}: {error}");
        let text = error.to_string();
        assert!(text.contains(key), "{pairs:?}: {text}");
        assert!(!text.contains("Canary-7f3a9c"), "{pairs:?}: {text}");
    }
}

/// The child half of [`explicit_credentials_win_over_the_ambient_environment`]: runs only when
/// the parent started this binary with the stand-in's endpoint and an `AZURE_*` environment.
#[tokio::test]
async fn child_explicit_credentials_win() {
    let Ok(endpoint) = std::env::var("KRISHIV_TEST_AZURE_CHILD_ENDPOINT") else {
        return;
    };
    // The parent's stand-in logs; the child only reads the answer: a 404 after a signed request.
    let shared = registry(&[
        ("fs.azure.endpoint", &account_endpoint(&endpoint)),
        (
            "fs.azure.account.key.acct.dfs.core.windows.net",
            ACCOUNT_KEY,
        ),
    ]);
    let opened = shared.open(URL).unwrap();
    let error = opened.store.head(&opened.key).await.unwrap_err();
    assert!(
        matches!(error, object_store::Error::NotFound { .. }),
        "{error}"
    );
    let sas = registry(&[
        ("fs.azure.endpoint", &account_endpoint(&endpoint)),
        ("fs.azure.account.auth.type", "SAS"),
        ("fs.azure.sas.fixed.token", "sv=2024-01-01&sig=Y2hpbGQ"),
    ]);
    let opened = sas.open(URL).unwrap();
    let error = opened.store.head(&opened.key).await.unwrap_err();
    assert!(
        matches!(error, object_store::Error::NotFound { .. }),
        "{error}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explicit_credentials_win_over_the_ambient_environment() {
    if std::env::var("KRISHIV_TEST_AZURE_CHILD_ENDPOINT").is_ok() {
        return;
    }
    let (endpoint, log) = stand_in().await;
    let token_file = std::env::temp_dir().join(format!("krishiv-azure-env-{}", std::process::id()));
    std::fs::write(&token_file, "ambient-jwt").unwrap();
    // What a CI runner with Azure OIDC (and the benchmark) exports: every one of these would
    // choose the credential in `MicrosoftAzureBuilder::from_env()`'s auto resolution.
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child_explicit_credentials_win", "--nocapture"])
        .env("KRISHIV_TEST_AZURE_CHILD_ENDPOINT", &endpoint)
        .env("AZURE_STORAGE_TOKEN", "ambient-token")
        .env("AZURE_CLIENT_ID", "ambient-client")
        .env("AZURE_TENANT_ID", "ambient-tenant")
        .env("AZURE_FEDERATED_TOKEN_FILE", &token_file)
        .env("AZURE_AUTHORITY_HOST", &endpoint)
        .env("AZURE_STORAGE_ACCOUNT_KEY", ACCOUNT_KEY)
        .status()
        .unwrap();
    std::fs::remove_file(&token_file).unwrap();
    assert!(status.success());
    let seen = log.lock().unwrap().clone();
    assert!(
        seen.iter().all(|s| !s.line.contains("oauth2")),
        "no ambient token request: {seen:?}"
    );
    let heads: Vec<&Seen> = seen
        .iter()
        .filter(|s| s.line.starts_with("HEAD "))
        .collect();
    assert_eq!(heads.len(), 2, "{seen:?}");
    assert!(
        heads[0]
            .auth
            .as_deref()
            .unwrap()
            .starts_with("SharedKey acct:"),
        "{seen:?}"
    );
    assert_eq!(heads[1].auth, None, "{seen:?}");
    assert!(heads[1].line.contains("sig=Y2hpbGQ"), "{seen:?}");
}

#[test]
fn oauth_keys_are_secret_or_server_scoped() {
    use krishiv_common::secret::OptionValue;
    for key in [
        "fs.azure.account.oauth2.client.secret",
        "fs.azure.account.oauth2.access.token",
        "fs.azure.account.oauth2.access.token.acct.dfs.core.windows.net",
        "fs.azure.sas.fixed.token",
    ] {
        assert!(
            OptionValue::from_user(key, "v").as_plain().is_none(),
            "{key} is a secret"
        );
        assert!(!krishiv_store::is_server_scoped_option(key), "{key}");
    }
    // Keys that choose a host, a file or the server's own identity need `Manage` in server mode.
    for key in [
        "fs.azure.account.auth.type",
        "fs.azure.account.auth.type.acct.dfs.core.windows.net",
        "fs.azure.account.oauth.provider.type",
        "fs.azure.account.oauth2.client.id",
        "fs.azure.account.oauth2.client.endpoint",
        "fs.azure.account.oauth2.msi.endpoint",
        "fs.azure.account.oauth2.msi.tenant",
        "fs.azure.account.oauth2.msi.authority",
        "fs.azure.account.oauth2.token.file",
    ] {
        assert!(krishiv_store::is_server_scoped_option(key), "{key}");
    }
}
