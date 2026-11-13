// SPDX-License-Identifier: Apache-2.0
//! `Session::cancel` and `Session::interrupt_all` reach a statement while it is
//! still being analyzed, before its query exists (P3-T021 follow-up, re-review
//! ruling R1 of PR-08b part 2). The slow step is an access-policy check, which
//! analysis awaits for every table it reads, as it awaits a table format's log.

// `cfg(test)` lets clippy's allow-unwrap-in-tests apply to this file, as in the other test files.
#![cfg(test)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::{BoxFuture, PolicyVersion};
use krishiv_session::{
    AccessDecision, AccessPolicy, AdmissionController, Permit, Principal, PrincipalKind, Privilege,
    QueryId, QueryOptions, QueryRequest, Securable, Session,
};

const OP: &str = "00112233-4455-6677-8899-aabbccddeeff";

/// Allows everything; while `hang` is set, a check says so on `entered` and never answers.
struct Hanging {
    hang: AtomicBool,
    entered: Mutex<Sender<()>>,
}

impl AccessPolicy for Hanging {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(0)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        _: &'a Securable,
        _: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        if self.hang.load(Ordering::SeqCst) {
            let _ = self.entered.lock().unwrap().send(());
            return Box::pin(std::future::pending());
        }
        Box::pin(std::future::ready(Ok(AccessDecision::Allow)))
    }
}

/// A session with the table `t` whose policy hangs once `hang` is set, and the
/// receiver told when a check hangs.
fn session() -> (Session, Arc<Hanging>, Receiver<()>) {
    let (tx, rx) = channel();
    let policy = Arc::new(Hanging {
        hang: AtomicBool::new(false),
        entered: Mutex::new(tx),
    });
    let s = Session::builder()
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
        .build()
        .unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    (s, policy, rx)
}

/// Starts `SELECT a FROM t` named `OP` on another thread once the policy hangs, and
/// waits until analysis is inside the policy; the receiver gets the statement's error
/// class (or `ok`).
fn hung_statement(s: &Session, policy: &Hanging, entered: &Receiver<()>) -> Receiver<String> {
    policy.hang.store(true, Ordering::SeqCst);
    let (tx, rx) = channel();
    let s = s.clone();
    std::thread::spawn(move || {
        let out = block_on(s.sql_with("SELECT a FROM t", QueryOptions::new().operation_id(OP)));
        let _ = tx.send(match out {
            Ok(_) => "ok".to_owned(),
            Err(e) => e.error_class().name().to_owned(),
        });
    });
    entered
        .recv_timeout(Duration::from_secs(30))
        .expect("analysis reached the policy");
    rx
}

/// `state|cancel_reason` of the `system.queries` rows named `id`.
fn outcome(s: &Session, id: &str) -> String {
    let df = block_on(s.sql(&format!(
        "SELECT concat(state, '|', coalesce(cancel_reason, '-')) FROM system.queries \
         WHERE query_id = '{id}'"
    )))
    .unwrap();
    let batches = block_on(df.collect()).unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let any = b.column(0).as_any();
        for i in 0..b.num_rows() {
            let v = any.downcast_ref::<arrow_array::StringArray>().map_or_else(
                || {
                    any.downcast_ref::<arrow_array::StringViewArray>()
                        .map(|a| a.value(i).to_owned())
                        .unwrap_or_default()
                },
                |a| a.value(i).to_owned(),
            );
            out.push(v);
        }
    }
    out.join(",")
}

#[test]
fn interrupt_all_ends_a_statement_during_analysis() {
    let (s, policy, entered) = session();
    let ended = hung_statement(&s, &policy, &entered);
    let ids = s.interrupt_all();
    assert_eq!(ids, vec![QueryId::parse(OP).unwrap()]);
    let class = ended
        .recv_timeout(Duration::from_secs(30))
        .expect("the statement ended");
    assert_eq!(class, "OPERATION_CANCELED");
    policy.hang.store(false, Ordering::SeqCst);
    // Its row says why, as a cancelled running query's does.
    assert_eq!(outcome(&s, OP), "cancelled|user");
}

/// Admits everything; the first request after `hang` is set says so and never answers.
#[derive(Debug)]
struct Queue {
    hang: AtomicBool,
    entered: Mutex<Sender<()>>,
}

impl AdmissionController for Queue {
    fn admit<'a>(&'a self, _: &'a QueryRequest) -> BoxFuture<'a, Result<Permit, KrishivError>> {
        if self.hang.swap(false, Ordering::SeqCst) {
            let _ = self.entered.lock().unwrap().send(());
            return Box::pin(std::future::pending());
        }
        Box::pin(std::future::ready(Ok(Permit::unlimited())))
    }

    fn uses_estimate(&self) -> bool {
        false
    }
}

#[test]
fn a_statement_cancelled_while_queued_names_the_reason() {
    let (tx, entered) = channel();
    let queue = Arc::new(Queue {
        hang: AtomicBool::new(false),
        entered: Mutex::new(tx),
    });
    let s = Session::builder()
        .admission_controller(Arc::clone(&queue) as Arc<dyn AdmissionController>)
        .build()
        .unwrap();
    let df = block_on(s.sql_with("SELECT 1 AS x", QueryOptions::new().operation_id(OP))).unwrap();
    queue.hang.store(true, Ordering::SeqCst);
    let (done, ended) = channel();
    std::thread::spawn(move || {
        let out = block_on(df.collect());
        let _ =
            done.send(out.map_or_else(|e| e.error_class().name().to_owned(), |_| "ok".to_owned()));
    });
    entered
        .recv_timeout(Duration::from_secs(30))
        .expect("the query queued");
    assert!(s.cancel(&QueryId::parse(OP).unwrap()));
    assert_eq!(
        ended.recv_timeout(Duration::from_secs(30)).unwrap(),
        "OPERATION_CANCELED"
    );
    assert_eq!(outcome(&s, OP), "cancelled|user");
}

#[test]
fn cancel_by_operation_id_ends_a_statement_during_analysis() {
    let (s, policy, entered) = session();
    let ended = hung_statement(&s, &policy, &entered);
    let other = QueryId::parse("ffeeddcc-bbaa-9988-7766-554433221100").unwrap();
    assert!(!s.cancel(&other), "another id names nothing");
    assert!(s.cancel(&QueryId::parse(OP).unwrap()));
    let class = ended
        .recv_timeout(Duration::from_secs(30))
        .expect("the statement ended");
    assert_eq!(class, "OPERATION_CANCELED");
    // Its registration is gone with it.
    assert!(!s.cancel(&QueryId::parse(OP).unwrap()));
}

#[test]
fn an_idle_frame_is_not_a_running_statement() {
    let (s, _policy, _entered) = session();
    let df = block_on(s.sql_with("SELECT a FROM t", QueryOptions::new().operation_id(OP))).unwrap();
    assert!(s.interrupt_all().is_empty());
    assert!(!s.cancel(&QueryId::parse(OP).unwrap()));
    // The frame still runs.
    assert_eq!(
        block_on(df.collect())
            .unwrap()
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        3
    );
}

#[test]
fn another_session_is_not_interrupted() {
    let (s, policy, entered) = session();
    let other = s.new_session();
    let ended = hung_statement(&s, &policy, &entered);
    assert!(other.interrupt_all().is_empty());
    assert!(!other.cancel(&QueryId::parse(OP).unwrap()));
    assert!(
        ended.recv_timeout(Duration::from_millis(200)).is_err(),
        "still analyzing"
    );
    assert_eq!(s.interrupt_all().len(), 1);
    assert_eq!(
        ended.recv_timeout(Duration::from_secs(30)).unwrap(),
        "OPERATION_CANCELED"
    );
}

#[test]
fn an_operator_kill_during_analysis_names_the_statement() {
    let (s, policy, entered) = session();
    let ended = hung_statement(&s, &policy, &entered);
    assert!(s.runtime().cancel(
        &QueryId::parse(OP).unwrap(),
        krishiv_session::CancelReason::Admin
    ));
    let class = ended
        .recv_timeout(Duration::from_secs(30))
        .expect("the statement ended");
    assert_eq!(class, "KRISHIV_QUERY_KILLED");
}
