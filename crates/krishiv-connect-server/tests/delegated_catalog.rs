// SPDX-License-Identifier: Apache-2.0
//! A delegated catalog end to end through Connect (design P3-T067 §3.3, §8): an OIDC session's
//! bearer token reaches the declared Iceberg REST catalog only as the token the IdP exchanged it
//! for, each user with their own; a static-token session gets
//! `KRISHIV_CATALOG_DELEGATION_UNAVAILABLE` and the catalog sees nothing.
#![cfg(unix)]
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener as StdListener;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use krishiv_catalogs::declared::IcebergRestFactory;
use krishiv_common::config::Config;
use krishiv_common::principal::{Principal, PrincipalKind};
use krishiv_common::secret::SecretString;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect_server::{
    ConnectAuth, ConnectServer, ConnectServerConfig, SessionLimits, SessionManager, role_sessions,
};
use krishiv_session::auth::{
    AuthFailure, Authenticated, Authenticator, CredentialKind, Credentials, ExchangeClaims,
    InboundCredential, Method, Role,
};
use krishiv_session::catalog_decl::DeclMode;
use krishiv_session::catalog_registry::CatalogSetup;
use krishiv_session::{BoxFuture, EngineRuntime, Session};
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::transport::Channel;

/// One session id per caller (a session id belongs to one owner).
fn session_of(token: &str) -> &'static str {
    match token {
        "alice-jwt" => "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10",
        "bob-jwt" => "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d11",
        _ => "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d12",
    }
}

/// One request the fake saw: path, `authorization`, body.
type Seen = Arc<Mutex<Vec<(String, String, String)>>>;

/// A fake IdP and Iceberg REST catalog on one loopback port: `/token` exchanges `<subject>` for
/// `x-<subject>`; `db.t` exists only for `x-alice-jwt`, answered with `table` (a `loadTable`
/// body) when given, else with a table on `s3://b/t` this test never reads.
fn fake_catalog(table: Option<String>) -> (String, Seen) {
    fake_catalog_with(move |path, auth| {
        match path {
        "/v1/namespaces/db/tables/t" if auth == "Bearer x-alice-jwt" => Some((
            200,
            table.clone().unwrap_or_else(|| {
                r#"{"metadata-location": "s3://b/t/metadata/v1.metadata.json",
                        "metadata": {"format-version": 2, "table-uuid": "u", "location": "s3://b/t"},
                        "config": {}}"#
                    .to_owned()
            }),
        )),
        _ => None,
    }
    })
}

/// [`fake_catalog`]'s IdP and `/v1/config`, with `route(path, authorization)` answering every other request
/// (`None`: 404 `NoSuchTableException`).
fn fake_catalog_with(
    route: impl Fn(&str, &str) -> Option<(u16, String)> + Send + 'static,
) -> (String, Seen) {
    let listener = StdListener::bind("127.0.0.1:0").expect("bind");
    let base = format!("http://{}", listener.local_addr().expect("addr"));
    let seen: Seen = Arc::default();
    let log = Arc::clone(&seen);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let mut reader = BufReader::new(stream.try_clone().expect("clone"));
            let mut line = String::new();
            if reader.read_line(&mut line).is_err() {
                continue;
            }
            let target = line.split_whitespace().nth(1).unwrap_or("").to_owned();
            let path = target.split('?').next().unwrap_or("").to_owned();
            let (mut auth, mut len) = (String::new(), 0_usize);
            loop {
                let mut h = String::new();
                if reader.read_line(&mut h).is_err() || h.trim_end().is_empty() {
                    break;
                }
                let (k, v) = h.trim_end().split_once(':').unwrap_or(("", ""));
                match k.to_ascii_lowercase().as_str() {
                    "authorization" => auth = v.trim().to_owned(),
                    "content-length" => len = v.trim().parse().unwrap_or(0),
                    _ => {}
                }
            }
            let mut body = vec![0; len];
            let _ = reader.read_exact(&mut body);
            let body = String::from_utf8_lossy(&body).into_owned();
            log.lock()
                .expect("log")
                .push((path.clone(), auth.clone(), body.clone()));
            let (status, json) = match path.as_str() {
                "/token" => {
                    let subject = body
                        .split('&')
                        .find_map(|p| p.strip_prefix("subject_token="))
                        .unwrap_or("");
                    (
                        200,
                        format!(r#"{{"access_token": "x-{subject}", "token_type": "bearer"}}"#),
                    )
                }
                "/v1/config" => (200, r#"{"defaults": {}, "overrides": {}}"#.to_owned()),
                other => route(other, &auth).unwrap_or_else(|| {
                    (
                        404,
                        r#"{"error": {"message": "no", "type": "NoSuchTableException", "code": 404}}"#
                            .to_owned(),
                    )
                }),
            };
            let head = format!(
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                json.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(json.as_bytes());
        }
    });
    (base, seen)
}

/// Bearer tokens ending in `-jwt` are OIDC tokens of the user before the dash (the validated
/// result the T061 validator would give, credential kept); any other is a static token of the
/// service `svc`, which keeps none.
struct TestAuth;

impl Authenticator for TestAuth {
    fn authenticate<'a>(
        &'a self,
        credentials: &'a Credentials<'a>,
    ) -> BoxFuture<'a, Result<Authenticated, AuthFailure>> {
        Box::pin(async move {
            let token = credentials.bearer.ok_or(AuthFailure::Missing)?;
            Ok(match token.strip_suffix("-jwt") {
                Some(user) => Authenticated::new(
                    Principal::new(user, PrincipalKind::User),
                    Some(Role::Operator),
                    Method::Oidc,
                )
                .with_credential(Some(InboundCredential::new(
                    CredentialKind::Oidc,
                    SecretString::new(token),
                    ExchangeClaims::with_audiences(vec!["krishiv".to_owned()]),
                    SystemTime::now() + Duration::from_secs(3600),
                ))),
                None => Authenticated::new(
                    Principal::new("svc", PrincipalKind::Service),
                    Some(Role::Operator),
                    Method::Token,
                ),
            })
        })
    }
}

/// The session manager `krishiv server` builds, over a runtime declaring the delegated catalog
/// `lake` at `base` in `mode`.
fn manager(
    base: &str,
    mode: DeclMode,
    config: &[(&str, &str)],
) -> (
    Arc<SessionManager>,
    Arc<krishiv_session::auth::DeniedCounts>,
) {
    let token_endpoint = format!("{base}/token");
    // The row reads of the embedded case use the session's own storage, not vended credentials
    // (unit 3); the server-mode cases keep the declaration's default.
    let storage = (mode == DeclMode::Embedded)
        .then_some(("krishiv.catalog.lake.storage", "allowed-prefixes"));
    let setup = [
        ("krishiv.catalog.lake.type", "iceberg-rest"),
        ("krishiv.catalog.lake.uri", base),
        ("krishiv.catalog.lake.allowInsecure", "true"),
        ("krishiv.catalog.lake.auth", "delegated"),
        ("krishiv.catalog.lake.audience", "lake"),
        ("krishiv.catalog.lake.actorToken", "none"),
        (
            "krishiv.catalog.lake.tokenEndpoint",
            token_endpoint.as_str(),
        ),
    ]
    .into_iter()
    .chain(storage)
    .fold(
        CatalogSetup::new(mode).factory(Arc::new(IcebergRestFactory)),
        |s, (k, v)| s.setting(k, v),
    );
    let runtime = config
        .iter()
        .fold(EngineRuntime::builder().catalogs(setup), |b, (k, v)| {
            b.config(*k, *v)
        })
        .build()
        .expect("runtime");
    let base_session = Session::builder()
        .runtime(runtime)
        .build()
        .expect("session");
    let denials = Arc::new(krishiv_session::auth::DeniedCounts::new());
    let factory =
        role_sessions(base_session, &Config::new(), Arc::clone(&denials)).expect("factory");
    (
        Arc::new(SessionManager::new(factory, SessionLimits::default())),
        denials,
    )
}

async fn start(base: &str, mode: DeclMode) -> (Channel, oneshot::Sender<()>) {
    start_with(base, mode, &[]).await
}

/// [`start`] over a runtime with `config` too.
async fn start_with(
    base: &str,
    mode: DeclMode,
    config: &[(&str, &str)],
) -> (Channel, oneshot::Sender<()>) {
    let (manager, denials) = manager(base, mode, config);
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = ConnectServer::new(manager, ConnectServerConfig::default())
        .with_auth(ConnectAuth::new(Arc::new(TestAuth)).with_denials(denials));
    let (stop, stopped) = oneshot::channel::<()>();
    tokio::spawn(server.serve(listener, async {
        let _ = stopped.await;
    }));
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");
    (channel, stop)
}

fn sql(query: &str) -> pb::Plan {
    pb::Plan {
        op_type: Some(OpType::Root(Box::new(pb::Relation {
            common: None,
            rel_type: Some(RelType::Sql(pb::Sql {
                query: query.to_owned(),
                ..Default::default()
            })),
        }))),
    }
}

async fn execute(
    channel: &Channel,
    query: &str,
    token: &str,
) -> Result<Vec<pb::ExecutePlanResponse>, tonic::Status> {
    let request = pb::ExecutePlanRequest {
        session_id: session_of(token).to_owned(),
        user_context: Some(pb::UserContext {
            user_id: "anyone".to_owned(),
            ..Default::default()
        }),
        plan: Some(sql(query)),
        ..Default::default()
    };
    let mut r = tonic::Request::new(request);
    r.metadata_mut().insert(
        "authorization",
        format!("Bearer {token}").parse().expect("ascii"),
    );
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    let mut s = grpc
        .server_streaming(
            r,
            "/spark.connect.SparkConnectService/ExecutePlan"
                .parse()
                .expect("path"),
            tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
        )
        .await?
        .into_inner();
    let mut out = Vec::new();
    while let Some(m) = s.message().await? {
        out.push(m);
    }
    Ok(out)
}

/// The `(id, name)` rows of `responses`, sorted.
fn rows(responses: &[pb::ExecutePlanResponse]) -> Vec<(i64, String)> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use krishiv_connect::proto::v4_3::execute_plan_response::ResponseType;
    let mut out = Vec::new();
    for r in responses {
        let Some(ResponseType::ArrowBatch(b)) = &r.response_type else {
            continue;
        };
        let reader =
            arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(b.data.clone()), None)
                .expect("ipc");
        for batch in reader {
            let batch = batch.expect("batch");
            let ids = batch.column(0).as_primitive::<Int64Type>();
            let names = batch.column(1).as_string::<i32>();
            for i in 0..batch.num_rows() {
                out.push((ids.value(i), names.value(i).to_owned()));
            }
        }
    }
    out.sort();
    out
}

#[derive(Clone, PartialEq, prost::Message)]
struct RpcStatus {
    #[prost(int32, tag = "1")]
    code: i32,
    #[prost(string, tag = "2")]
    message: String,
    #[prost(message, repeated, tag = "3")]
    details: Vec<prost_types::Any>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct ErrorInfo {
    #[prost(string, tag = "1")]
    reason: String,
    #[prost(string, tag = "2")]
    domain: String,
    #[prost(btree_map = "string, string", tag = "3")]
    metadata: BTreeMap<String, String>,
}

fn error_class(status: &tonic::Status) -> String {
    let rpc = RpcStatus::decode(status.details()).expect("google.rpc.Status");
    ErrorInfo::decode(rpc.details[0].value.as_slice())
        .expect("ErrorInfo")
        .metadata
        .get("errorClass")
        .cloned()
        .unwrap_or_default()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oidc_session_reaches_delegated_catalog_with_the_exchanged_token() {
    let (base, seen) = fake_catalog(None);
    let (channel, _stop) = start(&base, DeclMode::Server).await;
    // Alice is granted `db.t` in the catalog: it is found (and not readable until the table
    // provider over declared catalogs exists; the analyzer reports a catalog's refusal as
    // `KRISHIV_CATALOG_FAILURE`, hiding its text).
    let s = execute(&channel, "SELECT * FROM lake.db.t", "alice-jwt")
        .await
        .expect_err("found, not readable yet");
    assert_eq!(error_class(&s), "KRISHIV_CATALOG_FAILURE", "{s:?}");
    // Bob is not: the same name does not exist for him.
    let s = execute(&channel, "SELECT * FROM lake.db.t", "bob-jwt")
        .await
        .expect_err("absent for bob");
    assert_eq!(error_class(&s), "TABLE_OR_VIEW_NOT_FOUND", "{s:?}");
    // A static-token session has no user token: refused, the catalog asked nothing.
    let before = seen.lock().expect("seen").len();
    let s = execute(&channel, "SELECT * FROM lake.db.t", "static-service-token")
        .await
        .expect_err("no user token");
    assert_eq!(
        error_class(&s),
        "KRISHIV_CATALOG_DELEGATION_UNAVAILABLE",
        "{s:?}"
    );
    assert!(!s.message().contains("static-service-token"), "{s:?}");
    assert_eq!(seen.lock().expect("seen").len(), before);

    let seen = seen.lock().expect("seen").clone();
    let loads: Vec<&str> = seen
        .iter()
        .filter(|(p, _, _)| p == "/v1/namespaces/db/tables/t")
        .map(|(_, a, _)| a.as_str())
        .collect();
    assert_eq!(loads, ["Bearer x-alice-jwt", "Bearer x-bob-jwt"]);
    // The user's own token went only to the token endpoint, as the subject token.
    for (path, auth, body) in &seen {
        assert!(
            !auth.contains("alice-jwt") || auth == "Bearer x-alice-jwt",
            "{path}"
        );
        if path == "/token" {
            assert!(body.contains("subject_token="), "{body}");
        }
    }
}

/// Review 58b1-r3 R3: AnalyzePlan reads a delegated catalog with its own request's token, not
/// with whatever another request of the session stored last.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn analyze_plan_uses_its_own_requests_token() {
    use krishiv_connect::proto::v4_3::analyze_plan_request::{self as areq, Analyze};
    let (base, seen) = fake_catalog(None);
    let (manager, _) = manager(&base, DeclMode::Server, &[]);
    let credential = |token: &str| {
        InboundCredential::new(
            CredentialKind::Oidc,
            SecretString::new(token),
            ExchangeClaims::with_audiences(vec!["krishiv".to_owned()]),
            SystemTime::now() + Duration::from_secs(3600),
        )
    };
    let alice = Authenticated::new(
        Principal::new("alice", PrincipalKind::User),
        Some(Role::Operator),
        Method::Oidc,
    )
    .with_credential(Some(credential("alice-jwt")));
    let entry = manager
        .get_or_create(&alice, session_of("alice-jwt"), None, "alice")
        .expect("session");
    // Another request of the session stored a different token meanwhile.
    entry
        .session()
        .set_inbound_credential(Method::Oidc, Some(credential("stale-jwt")));
    let request = pb::AnalyzePlanRequest {
        session_id: session_of("alice-jwt").to_owned(),
        analyze: Some(Analyze::Schema(Box::new(areq::Schema {
            plan: Some(sql("SELECT * FROM lake.db.t")),
        }))),
        ..Default::default()
    };
    let options = krishiv_session::QueryOptions::new()
        .inbound_credential(Method::Oidc, Some(credential("alice-jwt")));
    let r = krishiv_connect_server::analyze::analyze_plan(
        &entry,
        &request,
        &krishiv_connect_server::ExecuteLimits::default(),
        &options,
    )
    .await;
    // Found for alice (and refused until the table provider exists).
    assert!(r.is_err());
    let loads: Vec<String> = seen
        .lock()
        .expect("seen")
        .iter()
        .filter(|(p, _, _)| p == "/v1/namespaces/db/tables/t")
        .map(|(_, a, _)| a.clone())
        .collect();
    assert_eq!(loads, ["Bearer x-alice-jwt"]);
}

/// Rows of a declared table through Connect: Alice reads `lake.db.t` (a local Iceberg table the
/// connector's `testing::write_table` wrote) with the token the IdP exchanged hers for; Bob, for
/// whom the catalog has no such table, gets `TABLE_OR_VIEW_NOT_FOUND`. Embedded mode, as a
/// server-mode declaration refuses `file:` table locations (`RemoteLocationPolicy`; server-mode
/// confinement is tested in krishiv-catalogs `declared/read_tests.rs`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oidc_session_reads_rows_of_a_delegated_catalog_table() {
    let dir = tempfile::tempdir().expect("tempdir");
    // `write_table` runs a runtime of its own, so not on this test's.
    let path = dir.path().to_owned();
    let table = tokio::task::spawn_blocking(move || {
        krishiv_connectors_iceberg::testing::write_table(&path)
    })
    .await
    .expect("join")
    .expect("table");
    let body = format!(
        r#"{{"metadata-location": {}, "metadata": {}, "config": {{}}}}"#,
        serde_json::Value::String(table.metadata_location.clone()),
        table.metadata
    );
    let (base, seen) = fake_catalog(Some(body));
    let (channel, _stop) = start(&base, DeclMode::Embedded).await;
    let got = execute(&channel, "SELECT id, name FROM lake.db.t", "alice-jwt")
        .await
        .expect("alice reads");
    assert_eq!(
        rows(&got),
        [(1, "a"), (2, "b"), (3, "c")].map(|(i, n)| (i, n.to_owned()))
    );
    let s = execute(&channel, "SELECT id, name FROM lake.db.t", "bob-jwt")
        .await
        .expect_err("absent for bob");
    assert_eq!(error_class(&s), "TABLE_OR_VIEW_NOT_FOUND", "{s:?}");
    let loads: Vec<String> = seen
        .lock()
        .expect("seen")
        .iter()
        .filter(|(p, _, _)| p == "/v1/namespaces/db/tables/t")
        .map(|(_, a, _)| a.clone())
        .collect();
    assert_eq!(loads, ["Bearer x-alice-jwt", "Bearer x-bob-jwt"]);
}

/// Review 58b2 F5 (§3.3): `spark.readStream.table("lake.db.t")` is refused before the catalog is
/// asked: no streaming read of a declared table exists, so none can reach a delegated catalog
/// without the user's token rules. When streaming reads land, this test fails first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_stream_of_a_delegated_catalog_table_refused() {
    use krishiv_connect::proto::v4_3::read::{NamedTable, ReadType};
    let (base, seen) = fake_catalog(None);
    let (channel, _stop) = start(&base, DeclMode::Server).await;
    let plan = pb::Plan {
        op_type: Some(OpType::Root(Box::new(pb::Relation {
            common: None,
            rel_type: Some(RelType::Read(pb::Read {
                read_type: Some(ReadType::NamedTable(NamedTable {
                    unparsed_identifier: "lake.db.t".to_owned(),
                    ..Default::default()
                })),
                is_streaming: true,
            })),
        }))),
    };
    let request = pb::ExecutePlanRequest {
        session_id: session_of("alice-jwt").to_owned(),
        user_context: Some(pb::UserContext {
            user_id: "anyone".to_owned(),
            ..Default::default()
        }),
        plan: Some(plan),
        ..Default::default()
    };
    let mut r = tonic::Request::new(request);
    r.metadata_mut()
        .insert("authorization", "Bearer alice-jwt".parse().expect("ascii"));
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    let result = async {
        let mut s = grpc
            .server_streaming(
                r,
                "/spark.connect.SparkConnectService/ExecutePlan"
                    .parse()
                    .expect("path"),
                tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(
                ),
            )
            .await?
            .into_inner();
        while s.message().await?.is_some() {}
        Ok::<(), tonic::Status>(())
    }
    .await;
    let s = result.expect_err("a streaming read is refused");
    assert_eq!(error_class(&s), "KRISHIV_UNSUPPORTED_FEATURE", "{s:?}");
    assert!(s.message().contains("is_streaming"), "{s:?}");
    assert!(
        !seen
            .lock()
            .expect("seen")
            .iter()
            .any(|(p, _, _)| p.starts_with("/v1/namespaces")),
        "the catalog was asked for a table"
    );
}

/// A `loadTable` body of `table` vending `key_id` for `s3://bucket/wh` for an hour.
fn vended_body(table: &krishiv_connectors_iceberg::testing::WrittenTable, key_id: &str) -> String {
    let expires = (SystemTime::now() + Duration::from_secs(3600))
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("epoch")
        .as_millis();
    format!(
        r#"{{"metadata-location": {}, "metadata": {}, "config": {{}},
            "storage-credentials": [{{"prefix": "s3://bucket/wh", "config": {{
                "s3.access-key-id": "{key_id}", "s3.secret-access-key": "secret-of-{key_id}",
                "s3.session-token-expires-at-ms": "{expires}"}}}}]}}"#,
        serde_json::Value::String(table.metadata_location.clone()),
        table.metadata
    )
}

/// The table on a loopback S3 stand-in (written with a key every later request is denied) and a delegated catalog
/// vending each user `AKIA<USER>` for it.
async fn vending_lake() -> (krishiv_catalogs::testing::S3Server, String, Seen) {
    let s3 = krishiv_catalogs::testing::serve_s3();
    let writer = s3.clone();
    let table = tokio::task::spawn_blocking(move || {
        krishiv_catalogs::testing::write_s3_table(&writer, "s3://bucket/wh", "AKIAWRITER")
    })
    .await
    .expect("join");
    s3.deny("AKIAWRITER");
    let (base, seen) = fake_catalog_with(move |path, auth| {
        let user = auth.strip_prefix("Bearer x-")?.strip_suffix("-jwt")?;
        (path == "/v1/namespaces/db/tables/t").then(|| {
            (
                200,
                vended_body(&table, &format!("AKIA{}", user.to_ascii_uppercase())),
            )
        })
    });
    (s3, base, seen)
}

/// Design P3-T067 §4.4 `vended_blocks_not_shared_across_delegated_users`: over single-node Connect, Alice and Bob
/// read one vended table with the key the catalog vended each of them; Bob's read fetches the data files with his own
/// key, never from blocks Alice's read cached.
///
/// Not the mutation guard for the §4.4 scope encoding: a vended read's stores are scoped (`#scope:<nonce>` ids, one
/// nonce per query), so no two queries share vended blocks whatever `cache_scope` returns. The encoding itself is
/// pinned by `krishiv-catalogs` `declared::vended::tests::block_cache_scope_encoding_no_collision`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn vended_blocks_not_shared_across_delegated_users() {
    let (s3, base, _) = vending_lake().await;
    let config = [
        ("fs.s3a.endpoint", s3.endpoint.as_str()),
        ("fs.s3a.region", "us-east-1"),
    ];
    let (channel, _stop) = start_with(&base, DeclMode::Server, &config).await;
    let want = [(1, "a"), (2, "b"), (3, "c")].map(|(i, n)| (i, n.to_owned()));
    let mut data_reads = Vec::new();
    for (user, key) in [
        ("alice", "AKIAALICE"),
        ("bob", "AKIABOB"),
        ("alice", "AKIAALICE"),
    ] {
        s3.clear_seen();
        let got = execute(
            &channel,
            "SELECT id, name FROM lake.db.t",
            &format!("{user}-jwt"),
        )
        .await
        .unwrap_or_else(|e| panic!("{user} reads: {e:?}"));
        assert_eq!(rows(&got), want, "{user}");
        let seen = s3.seen();
        assert!(
            seen.iter().all(|r| r.key_id.as_deref() == Some(key)),
            "{user}: {seen:?}"
        );
        data_reads.push(
            seen.iter()
                .filter(|r| r.method == "GET" && r.object.ends_with(".parquet"))
                .count(),
        );
    }
    assert!(data_reads[0] > 0, "{data_reads:?}");
    assert!(
        data_reads[1] > 0,
        "Bob's read was served from Alice's blocks: {data_reads:?}"
    );
}

/// Design P3-T067 §6 `catalog_mode_matrix` (the legs krishiv2 runs): one vended table read by an embedded session
/// and over single-node Connect gives the same rows, each signed only with a vended key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn catalog_mode_matrix() {
    let (s3, base, _) = vending_lake().await;
    let config = [
        ("fs.s3a.endpoint", s3.endpoint.as_str()),
        ("fs.s3a.region", "us-east-1"),
    ];
    // Single-node Connect, server-mode declaration, delegated: Alice's vended key.
    let (channel, _stop) = start_with(&base, DeclMode::Server, &config).await;
    s3.clear_seen();
    let connect = rows(
        &execute(&channel, "SELECT id, name FROM lake.db.t", "alice-jwt")
            .await
            .expect("connect reads"),
    );
    assert_eq!(s3.key_ids(), [Some("AKIAALICE".to_owned())].into());
    // Embedded session: the same declaration in embedded mode, the caller's token set on the session.
    let token_endpoint = format!("{base}/token");
    let embedded_setup = [
        ("krishiv.catalog.lake.type", "iceberg-rest"),
        ("krishiv.catalog.lake.uri", base.as_str()),
        ("krishiv.catalog.lake.allowInsecure", "true"),
        ("krishiv.catalog.lake.auth", "delegated"),
        ("krishiv.catalog.lake.audience", "lake"),
        ("krishiv.catalog.lake.actorToken", "none"),
        (
            "krishiv.catalog.lake.tokenEndpoint",
            token_endpoint.as_str(),
        ),
        ("krishiv.catalog.lake.storage", "vended"),
    ]
    .iter()
    .fold(
        CatalogSetup::new(DeclMode::Embedded).factory(Arc::new(IcebergRestFactory)),
        |s, (k, v)| s.setting(*k, *v),
    );
    let runtime = config
        .iter()
        .fold(
            EngineRuntime::builder().catalogs(embedded_setup),
            |b, (k, v)| b.config(*k, *v),
        )
        .build()
        .expect("runtime");
    let session = Session::builder()
        .runtime(runtime)
        .principal(Principal::new("alice", PrincipalKind::User))
        .build()
        .expect("session");
    session.set_inbound_credential(
        Method::Oidc,
        Some(InboundCredential::new(
            CredentialKind::Oidc,
            SecretString::new("alice-jwt"),
            ExchangeClaims::with_audiences(vec!["krishiv".to_owned()]),
            SystemTime::now() + Duration::from_secs(3600),
        )),
    );
    s3.clear_seen();
    let embedded = async {
        session
            .sql("SELECT id, name FROM lake.db.t")
            .await?
            .collect()
            .await
    }
    .await
    .expect("embedded reads");
    assert_eq!(s3.key_ids(), [Some("AKIAALICE".to_owned())].into());
    let mut embedded_rows: Vec<(i64, String)> = Vec::new();
    for b in &embedded {
        use arrow_array::cast::AsArray;
        let ids = b.column(0).as_primitive::<arrow_array::types::Int64Type>();
        let names = b.column(1);
        for i in 0..b.num_rows() {
            let name = names
                .as_string_view_opt()
                .map(|v| v.value(i).to_owned())
                .or_else(|| names.as_string_opt::<i32>().map(|v| v.value(i).to_owned()))
                .expect("a string column");
            embedded_rows.push((ids.value(i), name));
        }
    }
    embedded_rows.sort();
    assert_eq!(embedded_rows, connect);
}
