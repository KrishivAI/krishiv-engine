// SPDX-License-Identifier: Apache-2.0
//! `AddArtifacts`, `ArtifactStatus` and `FetchErrorDetails` end to end over
//! a raw tonic client (design P3-T011a §3.9, §3.10, §5: `artifact_names`,
//! `artifact_quota`, `artifact_crc`, `fetch_error_details_once`).
#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration-test helpers are outside #[test] fns"
)]
#![expect(
    clippy::disallowed_methods,
    reason = "the tests read back status details and what is on disk"
)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use krishiv_common::principal::Principal;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::add_artifacts_request::{
    ArtifactChunk, Batch, BeginChunkedArtifact, Payload, SingleChunkArtifact,
};
use krishiv_connect::proto::v4_3::config_request::operation::OpType;
use krishiv_connect::proto::v4_3::config_request::{self as creq, Operation};
use krishiv_connect::proto::v4_3::fetch_error_details_response::query_context::ContextType;
use krishiv_connect::proto::v4_3::plan::OpType as PlanOp;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect_server::artifacts::{ARTIFACT_OVERHEAD, crc32};
use krishiv_connect_server::{
    ArtifactLimits, ConnectServer, ConnectServerConfig, SessionEntry, SessionLimits, SessionManager,
};
use krishiv_session::Session;
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::transport::Channel;

const SESSION: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10";
const OTHER_SESSION: &str = "1d6a3c1e-77b2-4c55-8d0f-0a6c2e9b4f21";

struct Running {
    manager: Arc<SessionManager>,
    channel: Channel,
    parent: tempfile::TempDir,
    _stop: oneshot::Sender<()>,
}

impl Running {
    fn entry(&self) -> Arc<SessionEntry> {
        self.manager
            .get_or_create(&Principal::local_user(), SESSION, None, "u")
            .expect("session")
    }

    /// The server's artifact root (the one directory under `parent`).
    fn root(&self) -> Option<PathBuf> {
        std::fs::read_dir(self.parent.path())
            .expect("parent")
            .next()
            .map(|e| e.expect("entry").path())
    }

    fn session_dir(&self) -> PathBuf {
        self.entry().artifacts().dir_path().expect("session dir")
    }
}

async fn start_with(max_session: u64, max_artifact: u64) -> Running {
    let manager = Arc::new(SessionManager::from_base(
        Session::new().expect("session"),
        SessionLimits::default(),
    ));
    let parent = tempfile::tempdir().expect("tempdir");
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let config = ConnectServerConfig {
        artifacts: ArtifactLimits {
            max_session_bytes: max_session,
            max_artifact_bytes: max_artifact,
            parent: Some(parent.path().to_owned()),
        },
        ..ConnectServerConfig::default()
    };
    let server = ConnectServer::new(Arc::clone(&manager), config);
    let (stop, stopped) = oneshot::channel::<()>();
    tokio::spawn(server.serve(listener, async {
        let _ = stopped.await;
    }));
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");
    Running {
        manager,
        channel,
        parent,
        _stop: stop,
    }
}

async fn start() -> Running {
    start_with(1 << 30, 256 << 20).await
}

fn path(rpc: &str) -> tonic::codegen::http::uri::PathAndQuery {
    format!("/spark.connect.SparkConnectService/{rpc}")
        .parse()
        .expect("path")
}

async fn unary<Req, Resp>(channel: &Channel, rpc: &str, request: Req) -> Result<Resp, tonic::Status>
where
    Req: prost::Message + Send + Sync + 'static,
    Resp: prost::Message + Default + Send + Sync + 'static,
{
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    grpc.unary(
        tonic::Request::new(request),
        path(rpc),
        tonic_prost::ProstCodec::<Req, Resp>::default(),
    )
    .await
    .map(tonic::Response::into_inner)
}

async fn add_in(
    channel: &Channel,
    session: &str,
    payloads: Vec<Payload>,
) -> Result<pb::AddArtifactsResponse, tonic::Status> {
    let requests: Vec<pb::AddArtifactsRequest> = payloads
        .into_iter()
        .map(|p| pb::AddArtifactsRequest {
            session_id: session.to_owned(),
            payload: Some(p),
            ..Default::default()
        })
        .collect();
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    grpc.client_streaming(
        tonic::Request::new(futures::stream::iter(requests)),
        path("AddArtifacts"),
        tonic_prost::ProstCodec::<pb::AddArtifactsRequest, pb::AddArtifactsResponse>::default(),
    )
    .await
    .map(tonic::Response::into_inner)
}

async fn add(
    run: &Running,
    payloads: Vec<Payload>,
) -> Result<pb::AddArtifactsResponse, tonic::Status> {
    add_in(&run.channel, SESSION, payloads).await
}

fn chunk(data: &[u8]) -> ArtifactChunk {
    ArtifactChunk {
        data: data.to_vec(),
        crc: i64::from(crc32(data)),
    }
}

fn single(name: &str, data: &[u8]) -> SingleChunkArtifact {
    SingleChunkArtifact {
        name: name.to_owned(),
        data: Some(chunk(data)),
    }
}

fn batch(artifacts: Vec<SingleChunkArtifact>) -> Payload {
    Payload::Batch(Batch { artifacts })
}

/// A chunked artifact as the Python client sends it (`artifact.py`
/// `_add_chunked_artifact`).
fn chunked(name: &str, data: &[u8], size: usize) -> Vec<Payload> {
    let parts: Vec<&[u8]> = data.chunks(size).collect();
    let mut out = vec![Payload::BeginChunk(BeginChunkedArtifact {
        name: name.to_owned(),
        total_bytes: i64::try_from(data.len()).expect("len"),
        num_chunks: i64::try_from(parts.len()).expect("len"),
        initial_chunk: Some(chunk(parts[0])),
    })];
    out.extend(parts[1..].iter().map(|p| Payload::Chunk(chunk(p))));
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

fn info(status: &tonic::Status) -> ErrorInfo {
    let rpc = RpcStatus::decode(status.details()).expect("google.rpc.Status");
    ErrorInfo::decode(rpc.details[0].value.as_slice()).expect("ErrorInfo")
}

fn error_class(status: &tonic::Status) -> String {
    info(status)
        .metadata
        .get("errorClass")
        .cloned()
        .unwrap_or_default()
}

fn files_under(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_owned()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries {
            let p = e.expect("entry").path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(
                    p.strip_prefix(dir)
                        .expect("under")
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
    }
    out.sort();
    out
}

async fn eventually(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

// ------------------------------------------------------------ AddArtifacts

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn add_artifacts_batch_and_chunked() {
    let run = start().await;
    let big: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
    let mut payloads = vec![batch(vec![
        single("pyfiles/util.py", b"def f(): return 1\n"),
        single("files/sub/data.txt", b"hello"),
        single("archives/env.zip#env", b"PK\x03\x04"),
    ])];
    payloads.extend(chunked("files/big.bin", &big, 32 * 1024));
    let r = add(&run, payloads).await.expect("add");
    assert_eq!(r.session_id, SESSION);
    assert_eq!(r.server_side_session_id, run.entry().server_side_id());
    let names: Vec<(&str, bool)> = r
        .artifacts
        .iter()
        .map(|a| (a.name.as_str(), a.is_crc_successful))
        .collect();
    assert_eq!(
        names,
        [
            ("pyfiles/util.py", true),
            ("files/sub/data.txt", true),
            ("archives/env.zip#env", true),
            ("files/big.bin", true),
        ]
    );
    let dir = run.session_dir();
    assert_eq!(
        files_under(&dir),
        [
            "archives/env.zip",
            "files/big.bin",
            "files/sub/data.txt",
            "pyfiles/util.py"
        ]
    );
    assert_eq!(std::fs::read(dir.join("files/big.bin")).expect("read"), big);
    assert_eq!(
        std::fs::read(dir.join("pyfiles/util.py")).expect("read"),
        b"def f(): return 1\n"
    );
    // Archives are stored as sent, never unpacked.
    assert_eq!(
        std::fs::read(dir.join("archives/env.zip")).expect("read"),
        b"PK\x03\x04"
    );
    let used = run.entry().artifacts().used_bytes();
    assert_eq!(used, 4 * ARTIFACT_OVERHEAD + 18 + 5 + 4 + 100_000);
    // Private directories.
    use std::os::unix::fs::PermissionsExt;
    for d in [run.root().expect("root"), dir.clone(), dir.join("files")] {
        let mode = std::fs::metadata(&d).expect("meta").permissions().mode() & 0o777;
        assert_eq!(mode, 0o700, "{}", d.display());
    }
    // Only a cached blob exists for ArtifactStatus.
    let s: pb::ArtifactStatusesResponse = unary(
        &run.channel,
        "ArtifactStatus",
        pb::ArtifactStatusesRequest {
            session_id: SESSION.to_owned(),
            names: vec!["pyfiles/util.py".to_owned(), "files/big.bin".to_owned()],
            ..Default::default()
        },
    )
    .await
    .expect("status");
    assert!(s.statuses.values().all(|v| !v.exists), "{:?}", s.statuses);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn artifact_crc() {
    let run = start().await;
    let mut bad = chunk(b"payload");
    bad.crc += 1;
    let data: Vec<u8> = vec![7; 70_000];
    let mut payloads = vec![batch(vec![
        SingleChunkArtifact {
            name: "files/bad.txt".to_owned(),
            data: Some(bad),
        },
        single("files/good.txt", b"ok"),
        // No data at all: an empty chunk, whose CRC is 0.
        SingleChunkArtifact {
            name: "files/empty.txt".to_owned(),
            data: None,
        },
    ])];
    // One bad chunk in the middle spoils the chunked artifact.
    let mut parts = chunked("files/chunked.bin", &data, 32 * 1024);
    if let Payload::Chunk(c) = &mut parts[1] {
        c.crc ^= 1;
    }
    payloads.extend(parts);
    let r = add(&run, payloads).await.expect("add");
    let crc: Vec<(&str, bool)> = r
        .artifacts
        .iter()
        .map(|a| (a.name.as_str(), a.is_crc_successful))
        .collect();
    assert_eq!(
        crc,
        [
            ("files/bad.txt", false),
            ("files/good.txt", true),
            ("files/empty.txt", true),
            ("files/chunked.bin", false),
        ]
    );
    let dir = run.session_dir();
    assert_eq!(files_under(&dir), ["files/empty.txt", "files/good.txt"]);
    // The discarded artifacts gave their bytes back.
    assert_eq!(
        run.entry().artifacts().used_bytes(),
        2 * ARTIFACT_OVERHEAD + 2
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn artifact_names() {
    let run = start().await;
    for name in [
        "../x",
        "/etc/passwd",
        "pyfiles/../../x",
        "pyfiles/a\0b",
        "files//a",
        "files/a\u{202e}b",
        "unknown/a",
        "pyfiles\\..\\x",
    ] {
        let status = add(&run, vec![batch(vec![single(name, b"x")])])
            .await
            .unwrap_err();
        assert_eq!(status.code(), tonic::Code::Internal, "{name:?}");
        assert_eq!(error_class(&status), "INVALID_ARTIFACT_PATH", "{name:?}");
        assert_eq!(info(&status).metadata["sqlState"], "22023");
        // An error on the session carries an errorId.
        assert!(info(&status).metadata.contains_key("errorId"));
    }
    for name in ["jars/a.jar", "classes/A.class", "forward_to_fs/tmp/a"] {
        let status = add(&run, vec![batch(vec![single(name, b"x")])])
            .await
            .unwrap_err();
        assert_eq!(
            error_class(&status),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{name}"
        );
    }
    // Nothing was created: no root, no session directory.
    assert!(run.root().is_none());
    assert!(run.entry().artifacts().dir_path().is_none());
    assert_eq!(run.entry().artifacts().used_bytes(), 0);
    // A bad name later in a stream discards what came before it.
    let status = add(
        &run,
        vec![
            batch(vec![single("files/ok.txt", b"ok")]),
            batch(vec![single("files/../x", b"x")]),
        ],
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&status), "INVALID_ARTIFACT_PATH");
    let dir = run.session_dir();
    eventually("the staged file to go", || files_under(&dir).is_empty()).await;
    assert_eq!(run.entry().artifacts().used_bytes(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn artifact_already_exists() {
    let run = start().await;
    add(&run, vec![batch(vec![single("files/a.txt", b"one")])])
        .await
        .expect("first");
    let used = run.entry().artifacts().used_bytes();
    // The same bytes again: a no-op.
    let r = add(&run, vec![batch(vec![single("files/a.txt", b"one")])])
        .await
        .expect("same");
    assert!(r.artifacts[0].is_crc_successful);
    assert_eq!(run.entry().artifacts().used_bytes(), used);
    // Other bytes: refused, the others of the stream still added.
    let status = add(
        &run,
        vec![batch(vec![
            single("files/a.txt", b"two"),
            single("files/b.txt", b"b"),
        ])],
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&status), "ARTIFACT_ALREADY_EXISTS");
    assert_eq!(info(&status).metadata["sqlState"], "42713");
    assert_eq!(
        info(&status).metadata["messageParameters"],
        r#"{"normalizedRemoteRelativePath":"files/a.txt"}"#
    );
    let dir = run.session_dir();
    assert_eq!(
        std::fs::read(dir.join("files/a.txt")).expect("read"),
        b"one"
    );
    assert_eq!(std::fs::read(dir.join("files/b.txt")).expect("read"), b"b");
    assert!(files_under(&dir.join(".staging")).is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn artifact_quota() {
    // 64 KiB per session, 16 KiB per artifact.
    let run = start_with(64 << 10, 16 << 10).await;
    let exhausted = |s: &tonic::Status| {
        assert_eq!(s.code(), tonic::Code::ResourceExhausted, "{s:?}");
        assert_eq!(error_class(s), "KRISHIV_ARTIFACT_LIMIT");
        info(s).metadata["messageParameters"].clone()
    };
    // One artifact over its cap, in one batch.
    let s = add(
        &run,
        vec![batch(vec![single("files/a", &vec![1; 17 << 10])])],
    )
    .await
    .unwrap_err();
    assert!(exhausted(&s).contains("krishiv.connect.artifacts.maxArtifactBytes"));
    // A chunked one that announces it.
    let s = add(&run, chunked("files/b", &vec![1; 20 << 10], 4096))
        .await
        .unwrap_err();
    assert!(exhausted(&s).contains("maxArtifactBytes"));
    // A chunked one that lies about its size and its chunk count: refused
    // at the chunk that passes the cap.
    let mut lying = chunked("files/c", &vec![1; 24 << 10], 4096);
    if let Payload::BeginChunk(b) = &mut lying[0] {
        b.total_bytes = 4096;
        b.num_chunks = 1000;
    }
    let s = add(&run, lying).await.unwrap_err();
    assert!(exhausted(&s).contains("maxArtifactBytes"));
    let dir = run.session_dir();
    eventually("the staged files to go", || files_under(&dir).is_empty()).await;
    assert_eq!(run.entry().artifacts().used_bytes(), 0);
    // Many small artifacts: each costs its overhead, so the session's cap
    // holds (64 KiB / 4 KiB overhead: fewer than 16 empty files).
    let many: Vec<SingleChunkArtifact> = (0..20)
        .map(|i| single(&format!("files/e{i}"), b""))
        .collect();
    let s = add(&run, vec![batch(many)]).await.unwrap_err();
    assert!(exhausted(&s).contains("krishiv.connect.artifacts.maxBytes"));
    eventually("the staged files to go", || files_under(&dir).is_empty()).await;
    assert_eq!(run.entry().artifacts().used_bytes(), 0);
    // Installed artifacts hold their bytes across streams.
    for i in 0..3 {
        add(
            &run,
            vec![batch(vec![single(
                &format!("files/k{i}"),
                &vec![2; 13 << 10],
            )])],
        )
        .await
        .expect("fits");
    }
    let s = add(
        &run,
        vec![batch(vec![single("files/k3", &vec![2; 13 << 10])])],
    )
    .await
    .unwrap_err();
    assert!(exhausted(&s).contains("maxBytes"));
    assert_eq!(files_under(&dir).len(), 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn broken_chunk_sequences_discard_the_stream() {
    let run = start().await;
    let data = vec![3u8; 10_000];
    let cases: Vec<(&str, Vec<Payload>)> = vec![
        ("missing chunk", {
            let mut p = chunked("files/a", &data, 4096);
            p.pop();
            p
        }),
        ("extra chunk", {
            let mut p = chunked("files/a", &data, 4096);
            p.push(Payload::Chunk(chunk(b"more")));
            p
        }),
        (
            "chunk without a beginning",
            vec![Payload::Chunk(chunk(b"x"))],
        ),
        ("batch while chunking", {
            let mut p = chunked("files/a", &data, 4096);
            p.insert(1, batch(vec![single("files/b", b"b")]));
            p
        }),
        ("wrong total", {
            let mut p = chunked("files/a", &data, 4096);
            if let Payload::BeginChunk(b) = &mut p[0] {
                b.total_bytes = 9_999;
            }
            p
        }),
        ("no chunks", {
            let mut p = chunked("files/a", &data, 4096);
            if let Payload::BeginChunk(b) = &mut p[0] {
                b.num_chunks = 0;
            }
            p
        }),
    ];
    for (what, payloads) in cases {
        let s = add(&run, payloads).await.unwrap_err();
        assert_eq!(
            error_class(&s),
            "KRISHIV_INVALID_ARTIFACT_TRANSFER",
            "{what}"
        );
    }
    let s = add(
        &run,
        vec![
            batch(vec![single("files/ok", b"1")]),
            Payload::Chunk(chunk(b"")),
        ],
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&s), "KRISHIV_INVALID_ARTIFACT_TRANSFER");
    let dir = run.session_dir();
    eventually("the staged files to go", || files_under(&dir).is_empty()).await;
    assert_eq!(run.entry().artifacts().used_bytes(), 0);
    // A one-chunk artifact is complete with its beginning.
    let r = add(&run, chunked("files/one", b"single", 4096))
        .await
        .expect("one chunk");
    assert!(r.artifacts[0].is_crc_successful);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_client_that_goes_away_leaves_nothing_behind() {
    let run = start().await;
    // Make the session directory exist first.
    add(&run, vec![batch(vec![single("files/keep", b"k")])])
        .await
        .expect("add");
    let used = run.entry().artifacts().used_bytes();
    let (tx, rx) = tokio::sync::mpsc::channel::<pb::AddArtifactsRequest>(4);
    let requests =
        futures::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|m| (m, rx)) });
    let channel = run.channel.clone();
    let call = tokio::spawn(async move {
        let mut grpc = tonic::client::Grpc::new(channel);
        grpc.ready().await.expect("ready");
        grpc.client_streaming(
            tonic::Request::new(requests),
            path("AddArtifacts"),
            tonic_prost::ProstCodec::<pb::AddArtifactsRequest, pb::AddArtifactsResponse>::default(),
        )
        .await
    });
    let begin = chunked("files/partial", &vec![5; 50_000], 4096).remove(0);
    tx.send(pb::AddArtifactsRequest {
        session_id: SESSION.to_owned(),
        payload: Some(begin),
        ..Default::default()
    })
    .await
    .expect("send");
    let staging = run.session_dir().join(".staging");
    eventually("a staged file", || !files_under(&staging).is_empty()).await;
    assert!(run.entry().artifacts().used_bytes() > used);
    // The client goes away mid-artifact.
    call.abort();
    drop(tx);
    eventually("the staged file to go", || files_under(&staging).is_empty()).await;
    eventually("the quota to come back", || {
        run.entry().artifacts().used_bytes() == used
    })
    .await;
    assert_eq!(files_under(&run.session_dir()), ["files/keep"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cached_artifacts_live_in_memory_and_answer_artifact_status() {
    let run = start().await;
    let blob = vec![9u8; 50_000];
    let hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let small = "aa11";
    let mut payloads = chunked(&format!("cache/{hash}"), &blob, 32 * 1024);
    payloads.push(batch(vec![single(&format!("cache/{small}"), b"tiny")]));
    let r = add(&run, payloads).await.expect("add");
    assert!(r.artifacts.iter().all(|a| a.is_crc_successful));
    let entry = run.entry();
    assert_eq!(entry.artifacts().cached(hash).expect("cached"), blob);
    assert_eq!(
        entry.artifacts().cached(small).expect("cached"),
        &b"tiny"[..]
    );
    // In memory, not on disk.
    assert!(entry.artifacts().dir_path().is_none());
    let status = |session: &str, names: Vec<String>| {
        let channel = run.channel.clone();
        let session = session.to_owned();
        async move {
            let r: pb::ArtifactStatusesResponse = unary(
                &channel,
                "ArtifactStatus",
                pb::ArtifactStatusesRequest {
                    session_id: session,
                    names,
                    ..Default::default()
                },
            )
            .await
            .expect("status");
            r
        }
    };
    let r = status(
        SESSION,
        vec![
            format!("cache/{hash}"),
            "cache/ffff".to_owned(),
            format!("files/{hash}"),
            hash.to_owned(),
        ],
    )
    .await;
    assert_eq!(r.session_id, SESSION);
    assert_eq!(r.server_side_session_id, entry.server_side_id());
    let exists: Vec<(&str, bool)> = r
        .statuses
        .iter()
        .map(|(k, v)| (k.as_str(), v.exists))
        .collect();
    assert_eq!(exists.len(), 4);
    for (k, v) in exists {
        assert_eq!(v, k == format!("cache/{hash}"), "{k}");
    }
    // Another session does not see it (T-A8).
    let r = status(OTHER_SESSION, vec![format!("cache/{hash}")]).await;
    assert!(r.statuses.values().all(|v| !v.exists));
    // The same hash again is a no-op.
    let used = entry.artifacts().used_bytes();
    add(
        &run,
        vec![batch(vec![single(&format!("cache/{small}"), b"tiny")])],
    )
    .await
    .expect("again");
    assert_eq!(entry.artifacts().used_bytes(), used);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn released_sessions_remove_their_artifacts() {
    let run = start().await;
    add(
        &run,
        vec![batch(vec![
            single("files/a", b"a"),
            single("cache/abc", b"blob"),
        ])],
    )
    .await
    .expect("add");
    let entry = run.entry();
    let dir = run.session_dir();
    assert!(dir.join("files/a").exists());
    let _: pb::ReleaseSessionResponse = unary(
        &run.channel,
        "ReleaseSession",
        pb::ReleaseSessionRequest {
            session_id: SESSION.to_owned(),
            ..Default::default()
        },
    )
    .await
    .expect("release");
    eventually("the session directory to go", || !dir.exists()).await;
    assert!(entry.artifacts().cached("abc").is_none());
    assert_eq!(entry.artifacts().used_bytes(), 0);
    // The root stays for other sessions.
    assert!(run.root().is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn planted_links_in_the_session_directory_are_not_followed() {
    let run = start().await;
    add(&run, vec![batch(vec![single("pyfiles/first.py", b"1")])])
        .await
        .expect("add");
    let dir = run.session_dir();
    let outside = tempfile::tempdir().expect("outside");
    std::fs::write(outside.path().join("victim"), b"keep").expect("write");
    std::os::unix::fs::symlink(outside.path(), dir.join("files")).expect("link");
    std::os::unix::fs::symlink(outside.path().join("victim"), dir.join("pyfiles/v.py"))
        .expect("link");
    let s = add(&run, vec![batch(vec![single("files/victim", b"evil")])])
        .await
        .unwrap_err();
    assert_eq!(error_class(&s), "INVALID_ARTIFACT_PATH");
    let s = add(&run, vec![batch(vec![single("files/new", b"evil")])])
        .await
        .unwrap_err();
    assert_eq!(error_class(&s), "INVALID_ARTIFACT_PATH");
    let s = add(&run, vec![batch(vec![single("pyfiles/v.py", b"keep")])])
        .await
        .unwrap_err();
    assert_eq!(error_class(&s), "ARTIFACT_ALREADY_EXISTS");
    assert_eq!(
        std::fs::read(outside.path().join("victim")).expect("read"),
        b"keep"
    );
    assert!(!outside.path().join("new").exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn add_artifacts_on_an_unknown_session_format_has_no_error_id() {
    let run = start().await;
    let s = add_in(
        &run.channel,
        "not-a-uuid",
        vec![batch(vec![single("files/a", b"a")])],
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&s), "INVALID_HANDLE.FORMAT");
    assert!(!info(&s).metadata.contains_key("errorId"));
}

// ------------------------------------------------------- FetchErrorDetails

fn sql(query: &str) -> pb::Plan {
    pb::Plan {
        op_type: Some(PlanOp::Root(Box::new(pb::Relation {
            common: None,
            rel_type: Some(RelType::Sql(pb::Sql {
                query: query.to_owned(),
                ..Default::default()
            })),
        }))),
    }
}

/// The status a query ends with.
async fn failing(run: &Running, session: &str, query: &str) -> tonic::Status {
    let mut grpc = tonic::client::Grpc::new(run.channel.clone());
    grpc.ready().await.expect("ready");
    let request = pb::ExecutePlanRequest {
        session_id: session.to_owned(),
        plan: Some(sql(query)),
        ..Default::default()
    };
    let mut stream = match grpc
        .server_streaming(
            tonic::Request::new(request),
            path("ExecutePlan"),
            tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
        )
        .await
    {
        Ok(s) => s.into_inner(),
        Err(status) => return status,
    };
    loop {
        match stream.message().await {
            Ok(Some(_)) => {}
            Ok(None) => panic!("{query} did not fail"),
            Err(status) => return status,
        }
    }
}

async fn fetch(run: &Running, session: &str, id: &str) -> pb::FetchErrorDetailsResponse {
    unary(
        &run.channel,
        "FetchErrorDetails",
        pb::FetchErrorDetailsRequest {
            session_id: session.to_owned(),
            error_id: id.to_owned(),
            ..Default::default()
        },
    )
    .await
    .expect("fetch")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fetch_error_details_once() {
    let run = start().await;
    // An analysis error with a SQL context.
    let status = failing(&run, SESSION, "SELECT nope FROM range(3)").await;
    let meta = info(&status).metadata;
    let id = meta.get("errorId").expect("errorId").clone();
    assert_eq!(id.len(), 36, "a UUID: {id}");
    let r = fetch(&run, SESSION, &id).await;
    assert_eq!(r.root_error_idx, Some(0));
    assert!(r.session_id.is_empty() && r.server_side_session_id.is_empty());
    let root = &r.errors[0];
    assert!(root.stack_trace.is_empty());
    assert_eq!(root.message, status.message());
    let t = root.spark_throwable.as_ref().expect("throwable");
    assert_eq!(t.error_class.as_deref(), Some(meta["errorClass"].as_str()));
    assert_eq!(t.sql_state.as_deref(), Some(meta["sqlState"].as_str()));
    assert_eq!(
        root.error_type_hierarchy.first().map(String::as_str),
        Some(info(&status).reason.as_str())
    );
    // The contexts are the ones the error carries (the session attaches SQL
    // spans to `sql()` text; a Connect `Sql` relation carries none yet, so
    // the unit tests of `errors::details` cover the mapping).
    assert!(
        t.query_contexts
            .iter()
            .all(|c| c.context_type == ContextType::Sql as i32 || c.fragment.is_empty())
    );
    // Fetched once.
    assert_eq!(
        fetch(&run, SESSION, &id).await,
        pb::FetchErrorDetailsResponse::default()
    );
    // Unknown and other sessions' ids are empty answers.
    assert_eq!(
        fetch(&run, SESSION, "no-such-id").await,
        pb::FetchErrorDetailsResponse::default()
    );
    let status = failing(&run, SESSION, "SELECT 1/0").await;
    let id = info(&status).metadata["errorId"].clone();
    assert_eq!(
        fetch(&run, OTHER_SESSION, &id).await,
        pb::FetchErrorDetailsResponse::default()
    );
    // A run-time error is kept too, and still there for its own session.
    let r = fetch(&run, SESSION, &id).await;
    assert_eq!(
        r.errors[0]
            .spark_throwable
            .as_ref()
            .and_then(|t| t.error_class.as_deref()),
        Some("DIVIDE_BY_ZERO")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn enrich_error_off_sends_no_error_id() {
    let run = start().await;
    let _: pb::ConfigResponse = unary(
        &run.channel,
        "Config",
        pb::ConfigRequest {
            session_id: SESSION.to_owned(),
            operation: Some(Operation {
                op_type: Some(OpType::Set(creq::Set {
                    pairs: vec![pb::KeyValue {
                        key: "spark.sql.connect.enrichError.enabled".to_owned(),
                        value: Some("false".to_owned()),
                    }],
                    silent: None,
                })),
            }),
            ..Default::default()
        },
    )
    .await
    .expect("set");
    let status = failing(&run, SESSION, "SELECT 1/0").await;
    assert!(!info(&status).metadata.contains_key("errorId"));
    assert!(run.entry().errors().is_empty());
}

/// T-C8: a planted secret or an internal path never comes back from
/// `FetchErrorDetails` (the grep `config_redaction` in `tests/rpcs.rs` runs).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fetch_error_details_redacts() {
    const PLANTED: &str = "hunter2-planted-value";
    let run = start().await;
    let missing = tempfile::tempdir().expect("dir");
    let internal = missing.path().join("internal-planted-dir");
    let queries = [
        format!("SELECT * FROM parquet.`s3://user:{PLANTED}@bucket/x` WHERE"),
        format!("SELECT * FROM parquet.`s3://bucket/x?X-Amz-Signature={PLANTED}`"),
        format!("SELECT * FROM parquet.`{}`", internal.display()),
        format!("CREATE TABLE t USING parquet OPTIONS (password '{PLANTED}') AS SELECT 1/0 AS x"),
    ];
    for q in &queries {
        let status = failing(&run, SESSION, q).await;
        let Some(id) = info(&status).metadata.get("errorId").cloned() else {
            panic!("{q}: no errorId in {status:?}");
        };
        let r = fetch(&run, SESSION, &id).await;
        let text = format!("{r:?} {:?}", r.encode_to_vec());
        let bytes = String::from_utf8_lossy(&r.encode_to_vec()).into_owned();
        for haystack in [&text, &bytes, &status.message().to_owned()] {
            assert!(!haystack.contains(PLANTED), "{q}: {haystack}");
            assert!(!haystack.contains("pw-planted"), "{q}: {haystack}");
        }
        assert!(!bytes.contains("stack"), "{q}: {bytes}");
        // Every element carries no stack trace.
        assert!(r.errors.iter().all(|e| e.stack_trace.is_empty()));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn errors_on_every_rpc_carry_an_error_id() {
    let run = start().await;
    let status = unary::<_, pb::ConfigResponse>(
        &run.channel,
        "Config",
        pb::ConfigRequest {
            session_id: SESSION.to_owned(),
            operation: Some(Operation {
                op_type: Some(OpType::Get(creq::Get {
                    keys: vec!["no.such.key".to_owned()],
                })),
            }),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert_eq!(error_class(&status), "SQL_CONF_NOT_FOUND");
    let id = info(&status).metadata["errorId"].clone();
    let r = fetch(&run, SESSION, &id).await;
    assert_eq!(
        r.errors[0]
            .spark_throwable
            .as_ref()
            .and_then(|t| t.error_class.as_deref()),
        Some("SQL_CONF_NOT_FOUND")
    );
    // 20 kept per session: the 21st newest pushes out the oldest.
    let mut ids = Vec::new();
    for _ in 0..21 {
        ids.push(info(&failing(&run, SESSION, "SELECT 1/0").await).metadata["errorId"].clone());
    }
    assert_eq!(run.entry().errors().len(), 20);
    assert_eq!(
        fetch(&run, SESSION, &ids[0]).await,
        pb::FetchErrorDetailsResponse::default()
    );
    assert!(!fetch(&run, SESSION, &ids[20]).await.errors.is_empty());
}
