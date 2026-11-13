// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_common::error::ErrorClass;
use krishiv_common::principal::Principal;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::add_artifacts_request::{
    ArtifactChunk, Batch, BeginChunkedArtifact, Payload, SingleChunkArtifact,
};
use krishiv_session::Session;

use super::{ArtifactLimits, ArtifactStore, Transfer, crc32};
use crate::config::ConfigRole;
use crate::manager::{SessionEntry, SessionLimits, SessionManager};

const S1: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10";
const S2: &str = "1d6a3c1e-77b2-4c55-8d0f-0a6c2e9b4f21";

#[test]
fn crc32_matches_zlib() {
    // zlib.crc32 values.
    assert_eq!(crc32(b""), 0);
    assert_eq!(crc32(b"a"), 0xe8b7_be43);
    assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    assert_eq!(
        crc32(b"The quick brown fox jumps over the lazy dog"),
        0x414f_a339
    );
}

#[test]
fn limits_from_config() {
    let d = ArtifactLimits::from_config(&Config::new());
    assert_eq!(d, ArtifactLimits::default());
    let mut c = Config::new();
    c.set("krishiv.connect.artifacts.maxBytes", "3m").unwrap();
    c.set("krishiv.connect.artifacts.maxArtifactBytes", "1m")
        .unwrap();
    let l = ArtifactLimits::from_config(&c);
    assert_eq!(
        (l.max_session_bytes, l.max_artifact_bytes),
        (3 << 20, 1 << 20)
    );
    assert_eq!(l.parent, None);
}

fn sessions() -> SessionManager {
    SessionManager::from_base(Session::new().unwrap(), SessionLimits::default())
}

fn entry(m: &SessionManager, id: &str) -> Arc<SessionEntry> {
    m.get_or_create(&Principal::local_user(), id, None, "u")
        .unwrap()
}

fn store(parent: &std::path::Path) -> Arc<ArtifactStore> {
    Arc::new(ArtifactStore::new(ArtifactLimits {
        parent: Some(parent.to_owned()),
        ..ArtifactLimits::default()
    }))
}

fn transfer(store: &Arc<ArtifactStore>, e: &SessionEntry) -> Transfer {
    Transfer::new(
        Arc::clone(store),
        e,
        Principal::local_user(),
        ConfigRole::Admin,
    )
}

fn request(payload: Option<Payload>) -> pb::AddArtifactsRequest {
    pb::AddArtifactsRequest {
        session_id: S1.to_owned(),
        payload,
        ..Default::default()
    }
}

fn single(name: &str, data: &[u8]) -> Payload {
    Payload::Batch(Batch {
        artifacts: vec![SingleChunkArtifact {
            name: name.to_owned(),
            data: Some(ArtifactChunk {
                data: data.to_vec(),
                crc: i64::from(crc32(data)),
            }),
        }],
    })
}

#[test]
fn one_root_serves_every_session() {
    let tmp = tempfile::tempdir().unwrap();
    let store = store(tmp.path());
    assert!(store.root_path().is_none());
    let m = sessions();
    let (a, b) = (entry(&m, S1), entry(&m, S2));
    for e in [&a, &b] {
        let mut t = transfer(&store, e);
        t.on_message(&request(Some(single("files/x", b"x"))))
            .unwrap();
        t.finish().unwrap();
    }
    let root = store.root_path().unwrap();
    let (da, db) = (
        a.artifacts().dir_path().unwrap(),
        b.artifacts().dir_path().unwrap(),
    );
    assert_ne!(da, db);
    assert!(da.starts_with(&root) && db.starts_with(&root));
    let debug = format!("{:?}", a.artifacts());
    assert!(
        debug.contains("used") && debug.contains("closed: false"),
        "{debug}"
    );
}

#[test]
fn a_root_that_cannot_be_made_is_an_internal_error_without_its_path() {
    let tmp = tempfile::tempdir().unwrap();
    let parent = tmp.path().join("no-such-parent-planted");
    let store = store(&parent);
    let m = sessions();
    let e = entry(&m, S1);
    let err = transfer(&store, &e)
        .on_message(&request(Some(single("files/x", b"x"))))
        .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError);
    assert!(!err.to_string().contains("planted"), "{err}");
    assert_eq!(e.artifacts().used_bytes(), 0);
}

#[test]
fn a_session_closed_mid_transfer_takes_nothing_more() {
    let tmp = tempfile::tempdir().unwrap();
    let store = store(tmp.path());
    let m = sessions();
    let e = entry(&m, S1);
    // Closed between two requests.
    let mut t = transfer(&store, &e);
    t.on_message(&request(Some(single("files/a", b"a"))))
        .unwrap();
    e.artifacts().close();
    let err = t
        .on_message(&request(Some(single("files/b", b"b"))))
        .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidHandleSessionClosed);
    // Closed before the stream ends: nothing is installed.
    let e = entry(&m, S2);
    let mut t = transfer(&store, &e);
    t.on_message(&request(Some(single("files/a", b"a"))))
        .unwrap();
    t.on_message(&request(Some(single("cache/ab12", b"c"))))
        .unwrap();
    e.artifacts().close();
    let err = t.finish().unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidHandleSessionClosed);
    assert!(e.artifacts().cached("ab12").is_none());
    assert_eq!(e.artifacts().used_bytes(), 0);
}

#[test]
fn protocol_refusals() {
    let tmp = tempfile::tempdir().unwrap();
    let store = store(tmp.path());
    let m = sessions();
    let e = entry(&m, S1);
    let err = transfer(&store, &e).on_message(&request(None)).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
    // A cached blob may be chunked too; the chunk count still holds.
    let mut t = transfer(&store, &e);
    t.on_message(&request(Some(Payload::BeginChunk(BeginChunkedArtifact {
        name: "cache/ff00".to_owned(),
        total_bytes: 2,
        num_chunks: 2,
        initial_chunk: None,
    }))))
    .unwrap();
    let err = t.finish().unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivInvalidArtifactTransfer
    );
    assert_eq!(e.artifacts().used_bytes(), 0);
}

/// Review L6: a clone dropped mid-copy (its unpublished entry closed) gets
/// nothing more: no quota, no file, no directory; and closing it after a
/// partial copy returns what it had.
#[test]
fn a_copy_into_a_closed_session_leaves_nothing_behind() {
    let tmp = tempfile::tempdir().unwrap();
    let store = store(tmp.path());
    let m = sessions();
    let source = entry(&m, S1);
    let mut t = transfer(&store, &source);
    t.on_message(&request(Some(single("files/a/x", b"xyz"))))
        .unwrap();
    t.on_message(&request(Some(single("cache/abc", b"blob"))))
        .unwrap();
    t.finish().unwrap();
    let held = store.used_bytes();
    let snapshot = source.artifacts().snapshot().unwrap();
    assert!(!snapshot.is_empty());
    let pool = source.session().runtime().memory();

    // Closed before the copy (the handler dropped while it waited).
    let target = Arc::new(super::SessionArtifacts::default());
    target.close();
    let e = target.copy_from(&snapshot, &store, &pool).unwrap_err();
    assert_eq!(e.error_class().name(), "INVALID_HANDLE.SESSION_CLOSED");
    assert_eq!(target.used_bytes(), 0);
    assert!(target.dir_path().is_none());
    assert!(target.cached("abc").is_none());
    assert_eq!(store.used_bytes(), held);

    // Copied, then closed (dropped after the copy, before publication).
    let target = Arc::new(super::SessionArtifacts::default());
    target.copy_from(&snapshot, &store, &pool).unwrap();
    let dir = target.dir_path().unwrap();
    assert!(dir.join("files/a/x").exists());
    assert_eq!(store.used_bytes(), 2 * held);
    target.close();
    assert_eq!(target.used_bytes(), 0);
    assert_eq!(store.used_bytes(), held);
    // The directory goes once its last holder drops it.
    drop(target);
    assert!(!dir.exists());
    // The source session keeps its own artifacts.
    assert!(source.artifacts().snapshot().is_some());
}
