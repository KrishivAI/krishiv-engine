// SPDX-License-Identifier: Apache-2.0
//! Tests of the REST `CommitTable` (`rest_commit`, PR-36c part 2): the status mapping table one
//! row per test, the request body's shape, and the commit loop end to end over a wire: a
//! transport that decodes each `CommitTableRequest`, lands it on `commit_tests`' catalog (its
//! requirements checked, its updates applied through iceberg-rust's builder, the metadata file
//! written) and answers as an Iceberg REST server does, with scripted faults (a lost 200, a 401,
//! a 503 with `Retry-After`, a duplicated request, a request that never left).

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use iceberg::spec::TableMetadata;
use iceberg::{TableIdent, TableRequirement, TableUpdate};
use krishiv_common::error::{ErrorClass, KrishivError};
use serde_json::Value;

use super::append::WRITE_ID_PROP;
use super::append_tests::{MAX, iceberg_rust_files, ids, rows};
use super::commit::{CommitContext, CommitOutcome, commit_append};
use super::commit_tests::{
    Case, INIT, MINE, OUR_KEY, OURS, Refusal, TestCatalog, case, exists, refs_only,
};
use super::rest_commit::{RestCommitTable, outcome, request_body};
use crate::rest_commit::{RestCommitTransport, TransportFailure, TransportReply};

fn reply(status: u16, body: &str) -> Result<TransportReply, TransportFailure> {
    Ok(TransportReply::new(status, body.as_bytes().to_vec()))
}

fn model(status: u16, kind: &str, message: &str) -> Result<TransportReply, TransportFailure> {
    let body = serde_json::json!({"error": {"message": message, "type": kind, "code": status}});
    Ok(TransportReply::new(status, body.to_string().into_bytes()))
}

fn map(answer: Result<TransportReply, TransportFailure>) -> CommitOutcome {
    outcome("lake.db.t", &[], answer)
}

fn failed_class(o: &CommitOutcome) -> ErrorClass {
    match o {
        CommitOutcome::Failed(e) => e.error_class(),
        other => panic!("not a failure: {other:?}"),
    }
}

/// Row 200: committed with the `LoadTableResult`'s metadata.
#[test]
fn a_200_is_committed_with_the_answers_metadata() {
    let c = case();
    let body = serde_json::json!({
        "metadata-location": "file:///x/v9.metadata.json",
        "metadata": c.base,
        "config": {},
    });
    match map(reply(200, &body.to_string())) {
        CommitOutcome::Committed(m) => {
            assert_eq!(m.current_snapshot_id(), c.base.current_snapshot_id());
        }
        other => panic!("{other:?}"),
    }
    // A 200 whose metadata does not parse may still have been applied: unknown, never failed.
    assert!(matches!(
        map(reply(200, r#"{"metadata": {"format-version": 9}}"#)),
        CommitOutcome::Unknown(_)
    ));
    assert!(matches!(
        map(reply(200, "not json")),
        CommitOutcome::Unknown(_)
    ));
}

/// Row 409 `CommitFailedException`: a plain conflict.
#[test]
fn a_409_is_a_conflict() {
    let o = map(model(
        409,
        "CommitFailedException",
        "Requirement failed: branch main has changed: expected id 1 != 2",
    ));
    assert!(
        matches!(o, CommitOutcome::Conflict { stale: false }),
        "{o:?}"
    );
    // No body at all is still a conflict.
    assert!(matches!(
        map(reply(409, "")),
        CommitOutcome::Conflict { stale: false }
    ));
}

/// Row stale: iceberg-rust's three refusals (400) and Java 1.12's 409 "Validation failed".
#[test]
fn stale_answers_are_stale_conflicts() {
    for (status, text) in [
        (
            400,
            "Cannot add snapshot with sequence number 3 older than last sequence number 4",
        ),
        (
            400,
            "Cannot add a snapshot, first-row-id is behind table next-row-id: 1 < 5",
        ),
        (
            400,
            "Invalid snapshot timestamp 1: before last snapshot timestamp 900000",
        ),
        (
            409,
            "Validation failed, please retry: Cannot add snapshot with sequence number 3 older than last sequence number 4",
        ),
    ] {
        let o = map(model(status, "ValidationException", text));
        assert!(
            matches!(o, CommitOutcome::Conflict { stale: true }),
            "{status} {text}: {o:?}"
        );
    }
}

/// Row duplicate snapshot id: "probably committed", never a failure, Java's and iceberg-rust's
/// wording alike; the stale texts do not match it.
#[test]
fn a_duplicate_snapshot_id_is_unknown() {
    for (status, text) in [
        (400, "Snapshot already exists for id: 7000"),
        (400, "Snapshot already exists for: '7000'"),
        (409, "Snapshot already exists for id: 7000"),
    ] {
        let o = map(model(status, "BadRequestException", text));
        assert!(matches!(o, CommitOutcome::Unknown(_)), "{status}: {o:?}");
    }
    // Another 400 is a known refusal.
    let o = map(model(400, "BadRequestException", "Malformed request"));
    assert_eq!(failed_class(&o), ErrorClass::KrishivIcebergCommitFailed);
}

/// Rows 500/502/503/504: unknown; a `Retry-After` is named in the reason.
#[test]
fn server_errors_are_unknown_and_keep_retry_after() {
    for status in [500, 502, 503, 504] {
        let o = map(model(status, "ServiceFailureException", "boom"));
        assert!(matches!(o, CommitOutcome::Unknown(_)), "{status}: {o:?}");
    }
    let answer = Ok(TransportReply::new(503, Vec::new()).with_retry_after(Duration::from_secs(7)));
    match map(answer) {
        CommitOutcome::Unknown(reason) => {
            assert!(
                reason.contains("503") && reason.contains("Retry-After 7"),
                "{reason}"
            );
        }
        other => panic!("{other:?}"),
    }
    // 501 (Java: `UnsupportedOperationException`) is a refusal.
    assert_eq!(
        failed_class(&map(reply(501, ""))),
        ErrorClass::KrishivIcebergCommitFailed
    );
}

/// No answer: unknown when the request may have left, a known failure when it never did.
#[test]
fn transport_failures_split_on_whether_the_request_left() {
    let o = map(Err(TransportFailure::new(true, "the connection was reset")));
    assert!(matches!(o, CommitOutcome::Unknown(_)), "{o:?}");
    let o = map(Err(TransportFailure::new(true, "the request timed out")));
    assert!(matches!(o, CommitOutcome::Unknown(_)), "{o:?}");
    let o = map(Err(TransportFailure::new(false, "egress policy refused")));
    assert_eq!(failed_class(&o), ErrorClass::KrishivIcebergCommitFailed);
}

/// Rows 401, 403, 404, 422 and other 4xx: known refusals naming the status and the type, never
/// the server's message.
#[test]
fn client_errors_are_failures_without_the_servers_message() {
    for (status, kind) in [
        (401, "NotAuthorizedException"),
        (403, "ForbiddenException"),
        (404, "NoSuchTableException"),
        (422, "UnprocessableEntityException"),
        (418, "Teapot"),
    ] {
        let o = map(model(status, kind, "secret=hunter2"));
        let CommitOutcome::Failed(e) = o else {
            panic!("{status}: {o:?}");
        };
        assert_eq!(e.error_class(), ErrorClass::KrishivIcebergCommitFailed);
        let text = e.to_string();
        assert!(
            text.contains(&status.to_string()) && text.contains(kind),
            "{text}"
        );
        assert!(!text.contains("hunter2"), "{text}");
    }
}

/// The body is the spec's `CommitTableRequest`: identifier, requirements, updates (iceberg's
/// serde forms).
#[test]
fn the_request_body_is_a_commit_table_request() {
    let ident = TableIdent::from_strs(["db", "sub", "t"]).unwrap();
    let requirements = vec![TableRequirement::RefSnapshotIdMatch {
        r#ref: "main".to_owned(),
        snapshot_id: Some(5),
    }];
    let updates = vec![TableUpdate::SetProperties {
        updates: [("k".to_owned(), "v".to_owned())].into(),
    }];
    let body: Value =
        serde_json::from_slice(&request_body(&ident, &requirements, &updates).unwrap()).unwrap();
    assert_eq!(
        body["identifier"],
        serde_json::json!({"namespace": ["db", "sub"], "name": "t"})
    );
    assert_eq!(body["requirements"][0]["type"], "assert-ref-snapshot-id");
    assert_eq!(body["requirements"][0]["snapshot-id"], 5);
    assert_eq!(body["updates"][0]["action"], "set-properties");
}

/// What the wire does with one request.
#[derive(Debug, Clone, Copy)]
enum Wire {
    /// Landed (or refused) and answered as a REST server does.
    Answer,
    /// Landed; the answer is lost (no answer after sending).
    LostAfterApply,
    /// Answered with this status without landing.
    Status(u16),
    /// Delivered twice: the first copy landed, the client sees the second's answer.
    Duplicated,
    /// Refused before it left (the egress policy).
    NeverSent,
}

/// A REST server over `commit_tests`' catalog (module docs).
struct Server {
    cat: &'static TestCatalog,
    script: Mutex<VecDeque<Wire>>,
    /// Requests that reached the server.
    posts: Mutex<usize>,
    /// The `all_snapshots` flag of every load.
    loads: Mutex<Vec<bool>>,
}

impl Server {
    fn new(cat: &'static TestCatalog, script: &[Wire]) -> Arc<Self> {
        Arc::new(Self {
            cat,
            script: Mutex::new(script.iter().copied().collect()),
            posts: Mutex::new(0),
            loads: Mutex::new(Vec::new()),
        })
    }

    fn land(&self, request: &(Vec<TableRequirement>, Vec<TableUpdate>)) -> (u16, String) {
        let landed = self.cat.state.lock().unwrap().land(request);
        let (status, kind, message) = match landed {
            Ok(m) => {
                let (_, location) = self.cat.metadata();
                let body = serde_json::json!({
                    "metadata-location": location,
                    "metadata": m,
                    "config": {},
                });
                return (200, body.to_string());
            }
            Err(Refusal::Conflict) => (
                409,
                "CommitFailedException",
                "Requirement failed: branch main has changed".to_owned(),
            ),
            Err(Refusal::Stale) => (
                400,
                "BadRequestException",
                "Cannot add snapshot with sequence number 2 older than last sequence number 2"
                    .to_owned(),
            ),
            Err(Refusal::Duplicate) => (
                400,
                "BadRequestException",
                format!("Snapshot already exists for: '{OURS}'"),
            ),
            Err(Refusal::Other(s)) => (400, "BadRequestException", s),
        };
        let body = serde_json::json!({"error": {"message": message, "type": kind, "code": status}});
        (status, body.to_string())
    }
}

#[async_trait]
impl RestCommitTransport for Server {
    fn name(&self) -> String {
        "lake.db.t".to_owned()
    }

    async fn post_commit(&self, body: Vec<u8>) -> Result<TransportReply, TransportFailure> {
        let wire = self
            .script
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(Wire::Answer);
        if matches!(wire, Wire::NeverSent) {
            return Err(TransportFailure::new(
                false,
                "egress policy refused the host",
            ));
        }
        *self.posts.lock().unwrap() += 1;
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            value["identifier"],
            serde_json::json!({"namespace": ["db"], "name": "t"})
        );
        let request: (Vec<TableRequirement>, Vec<TableUpdate>) = (
            serde_json::from_value(value["requirements"].clone()).unwrap(),
            serde_json::from_value(value["updates"].clone()).unwrap(),
        );
        let (status, text) = match wire {
            Wire::Answer => self.land(&request),
            Wire::LostAfterApply => {
                let _ = self.land(&request);
                return Err(TransportFailure::new(true, "the connection was reset"));
            }
            Wire::Status(s) => (s, String::new()),
            Wire::Duplicated => {
                let _ = self.land(&request);
                self.land(&request)
            }
            Wire::NeverSent => unreachable!(),
        };
        let mut reply = TransportReply::new(status, text.into_bytes());
        if status == 503 {
            reply = reply.with_retry_after(Duration::from_secs(2));
        }
        Ok(reply)
    }

    async fn load(&self, all_snapshots: bool) -> Result<Vec<u8>, KrishivError> {
        self.loads.lock().unwrap().push(all_snapshots);
        let (m, location) = self.cat.metadata();
        let m = if all_snapshots { m } else { refs_only(&m) };
        Ok(
            serde_json::json!({"metadata-location": location, "metadata": m})
                .to_string()
                .into_bytes(),
        )
    }
}

/// `c`'s catalog moved out and served over the wire with `script`.
fn wire(c: &mut Case, script: &[Wire]) -> Arc<Server> {
    let placeholder = TestCatalog::new(c.base.clone(), String::new());
    let cat: &'static TestCatalog = Box::leak(Box::new(std::mem::replace(&mut c.cat, placeholder)));
    Server::new(cat, script)
}

fn run(c: &mut Case, server: &Arc<Server>) -> Result<TableMetadata, KrishivError> {
    let table = RestCommitTable::new(
        Arc::clone(server) as Arc<dyn RestCommitTransport>,
        TableIdent::from_strs(["db", "t"]).map_err(|e| {
            KrishivError::from(krishiv_common::error::InternalError {
                message: e.to_string(),
            })
        })?,
    );
    let io = c.env.reg.file_io();
    let cx = CommitContext {
        io: &io,
        clock: c.clock.as_ref(),
        max_metadata_bytes: MAX,
    };
    c.env
        .rt
        .block_on(commit_append(&table, cx, &c.base, &mut c.new, ids(OUR_KEY)))
}

/// `Case::check` over the server's catalog.
fn check(c: &Case, server: &Server, expected: &[&str]) {
    let (m, loc) = server.cat.metadata();
    let files = iceberg_rust_files(&c.env, &m, &loc);
    let unique: HashSet<&String> = files.iter().collect();
    assert_eq!(unique.len(), files.len(), "a file listed twice: {files:?}");
    let ours = m
        .snapshots()
        .filter(|s| {
            s.summary().additional_properties.get(WRITE_ID_PROP) == Some(&OUR_KEY.to_owned())
        })
        .count();
    assert!(ours <= 1, "the write committed {ours} times");
    assert_eq!(rows(&c.env, &loc), expected);
}

/// The plain path over the wire: one POST, committed, the answer's metadata returned.
#[test]
fn an_append_commits_over_the_wire() {
    let mut c = case();
    let server = wire(&mut c, &[]);
    let m = run(&mut c, &server).unwrap();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(*server.posts.lock().unwrap(), 1);
    assert!(server.loads.lock().unwrap().is_empty());
    check(&c, &server, &[INIT, MINE]);
}

/// A foreign commit landed first: the 409 is refreshed with `snapshots=all`, the own-key search
/// finds nothing, the rebase commits on the new head.
#[test]
fn a_foreign_commit_409_is_rebased() {
    let mut c = case();
    let foreign = c.foreign("f.parquet", 900, "main");
    let server = wire(&mut c, &[]);
    let request = c.env.rt.block_on(foreign(server.cat.metadata().0));
    server
        .cat
        .state
        .lock()
        .unwrap()
        .land(&request)
        .ok()
        .unwrap();
    let m = run(&mut c, &server).unwrap();
    assert_eq!(
        m.snapshot_by_id(OURS).unwrap().parent_snapshot_id(),
        Some(900)
    );
    assert_eq!(*server.posts.lock().unwrap(), 2);
    assert_eq!(*server.loads.lock().unwrap(), [true]);
    check(&c, &server, &[INIT, MINE, "900|1|f.parquet"]);
}

/// A commit on another branch makes the request stale: iceberg-rust's 400 text is a stale
/// conflict, so the write is rebased, never failed.
#[test]
fn a_stale_400_is_rebased() {
    let mut c = case();
    let foreign = c.foreign("dev.parquet", 901, "dev");
    let server = wire(&mut c, &[]);
    let request = c.env.rt.block_on(foreign(server.cat.metadata().0));
    server
        .cat
        .state
        .lock()
        .unwrap()
        .land(&request)
        .ok()
        .unwrap();
    let m = run(&mut c, &server).unwrap();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(*server.posts.lock().unwrap(), 2);
    check(&c, &server, &[INIT, MINE]);
}

/// A lost 200 (the connection reset after the apply): the probe loads with `snapshots=all` and
/// finds the snapshot; nothing is resent.
#[test]
fn a_lost_200_is_found_by_the_snapshots_all_probe() {
    let mut c = case();
    let server = wire(&mut c, &[Wire::LostAfterApply]);
    let m = run(&mut c, &server).unwrap();
    assert!(m.snapshot_by_id(OURS).is_some());
    assert_eq!(*server.posts.lock().unwrap(), 1);
    assert_eq!(*server.loads.lock().unwrap(), [true]);
    assert!(exists(&c.list(1)) && exists(&c.data()));
    check(&c, &server, &[INIT, MINE]);
}

/// A request delivered twice: the second copy's duplicate-id 400 is "probably committed", the
/// probe finds it; committed once.
#[test]
fn a_duplicate_id_answer_is_found_by_the_probe() {
    let mut c = case();
    let server = wire(&mut c, &[Wire::Duplicated]);
    let m = run(&mut c, &server).unwrap();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(*server.posts.lock().unwrap(), 1);
    check(&c, &server, &[INIT, MINE]);
}

/// A 401 to an attempt's first request (the transport already refreshed its token once) is a
/// known failure: everything the write produced is deleted.
#[test]
fn a_401_to_the_first_request_fails_and_deletes_the_write() {
    let mut c = case();
    let server = wire(&mut c, &[Wire::Status(401)]);
    let e = run(&mut c, &server).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivIcebergCommitFailed);
    assert!(e.to_string().contains("401"), "{e}");
    assert!(!exists(&c.list(1)) && !exists(&c.manifest()) && !exists(&c.data()));
    check(&c, &server, &[INIT]);
}

/// A 503 (with `Retry-After`) then a 401 to the resend: the 401 says nothing about the first
/// request, so the write stays unknown, and the next resend commits.
#[test]
fn a_401_to_a_resend_after_a_503_stays_unknown() {
    let mut c = case();
    let server = wire(&mut c, &[Wire::Status(503), Wire::Status(401)]);
    let m = run(&mut c, &server).unwrap();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(*server.posts.lock().unwrap(), 3);
    check(&c, &server, &[INIT, MINE]);
}

/// A request the egress policy refused never left: a known failure, nothing reached the
/// server, the write's files are deleted.
#[test]
fn a_request_that_never_left_fails_before_anything_is_sent() {
    let mut c = case();
    let server = wire(&mut c, &[Wire::NeverSent]);
    let e = run(&mut c, &server).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivIcebergCommitFailed);
    assert_eq!(*server.posts.lock().unwrap(), 0);
    assert!(!exists(&c.data()));
    check(&c, &server, &[INIT]);
}
