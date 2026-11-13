// SPDX-License-Identifier: Apache-2.0
use std::sync::Arc;
use std::time::{Duration, Instant};

use krishiv_common::config::Config;
use krishiv_session::QueryId;

use super::*;

const A: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10";
const B: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d11";

fn op(id: &str, plan: &[u8]) -> Arc<Operation> {
    Arc::new(Operation::new(
        QueryId::parse(id).unwrap(),
        plan_digest(plan),
        Arc::from(Vec::new()),
        true,
    ))
}

#[test]
fn tags_are_checked_as_spark_checks_them() {
    let ok = validate_tags(&["a".to_owned(), "b c".to_owned()]).unwrap();
    assert_eq!(ok.len(), 2);
    for (tags, text) in [
        (
            vec!["a,b".to_owned()],
            "Spark Connect tag cannot contain ','.",
        ),
        (
            vec![String::new()],
            "Spark Connect tag cannot be an empty string.",
        ),
        (
            vec!["x".repeat(MAX_TAG_BYTES + 1)],
            "Spark Connect tag cannot be longer than 1024 bytes.",
        ),
        (
            vec!["t".to_owned(); MAX_TAGS + 1],
            "A request may carry at most 256 Spark Connect tags, not 257.",
        ),
    ] {
        let e = validate_tags(&tags).unwrap_err();
        assert_eq!(e.error_class().name(), "KRISHIV_INVALID_OPERATION_TAG");
        assert_eq!(e.sql_state(), Some("22023"));
        assert_eq!(e.message(), text);
    }
}

#[test]
fn minted_ids_are_version_4_uuids() {
    let a = new_operation_id().unwrap();
    let b = new_operation_id().unwrap();
    assert_ne!(a, b);
    let s = a.as_str();
    assert_eq!(s.len(), 36);
    assert_eq!(&s[14..15], "4");
    assert!(matches!(&s[19..20], "8" | "9" | "a" | "b"), "{s}");
}

#[test]
fn register_refuses_a_live_id_and_the_cap() {
    let ops = Operations::default();
    let first = op(A, b"plan");
    ops.register(&first, 2).unwrap();
    // The same plan under a live id attaches to it (Spark reattaches).
    match ops.register(&op(A, b"plan"), 2).unwrap() {
        Admitted::Existing(live) => assert!(Arc::ptr_eq(&live, &first)),
        Admitted::New => panic!("registered twice"),
    }
    let e = ops.register(&op(A, b"other"), 2).unwrap_err();
    assert_eq!(
        e.error_class().name(),
        "INVALID_HANDLE.OPERATION_ALREADY_EXISTS"
    );
    ops.register(&op(B, b"plan"), 2).unwrap();
    let e = ops
        .register(&op("8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d12", b"p"), 2)
        .unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_OPERATION_LIMIT");
    assert_eq!(e.sql_state(), Some("53400"));
    assert_eq!(ops.len(), 2);
    assert!(!ops.is_empty());
}

#[test]
fn lookup_remove_and_close_all() {
    let ops = Operations::default();
    let first = op(A, b"plan");
    ops.register(&first, 10).unwrap();
    // Java's UUID.fromString reading: upper case finds the same operation.
    assert!(ops.get(&A.to_uppercase()).is_some());
    assert!(ops.get("not a uuid").is_none());
    assert!(ops.get(B).is_none());
    // Only the registered instance is removed.
    assert!(!ops.remove(&op(A, b"plan")));
    assert!(ops.remove(&first));
    assert!(!ops.remove(&first));
    let second = op(B, b"plan");
    ops.register(&second, 10).unwrap();
    assert_eq!(ops.close_all().len(), 1);
    assert!(second.is_cancelled());
    assert!(second.replay().is_closed());
    assert!(ops.is_empty());
}

#[test]
fn cancellation_and_close() {
    let o = op(A, b"p");
    assert!(o.reattachable());
    assert!(o.tags().is_empty());
    assert_eq!(o.sent_count(), 0);
    assert_eq!(o.buffered_peak(), 0);
    assert!(!o.is_cancelled());
    assert!(o.cancel(), "the first cancel interrupts");
    assert!(!o.cancel(), "a second interrupts nothing");
    assert!(o.is_cancelled());
    // Resolves at once after a cancel, whether or not anyone waited.
    futures::executor::block_on(o.cancelled());
    o.close();
    o.close();
    futures::executor::block_on(o.closed());
    // A closed finished operation was not interrupted.
    let f = op(B, b"q");
    assert!(f.finish());
    f.close();
    assert!(!f.is_cancelled());
    futures::executor::block_on(f.closed());
}

#[test]
fn the_retry_buffer_is_for_reattachable_operations() {
    let plain = Operation::new(
        QueryId::parse(A).unwrap(),
        plan_digest(b"p"),
        Arc::from(Vec::new()),
        false,
    )
    .with_retry_buffer(1 << 20);
    assert!(!plain.reattachable());
    let id = random_uuid().unwrap();
    assert_eq!(id >> 76 & 0xf, 4, "version 4");
    assert_eq!(
        replay::uuid_bits(&replay::uuid_text(id)),
        Some(id),
        "canonical text"
    );
}

#[test]
fn a_finished_operation_is_not_interrupted() {
    // Completion wins: the interrupt finds it finished.
    let o = op(A, b"p");
    assert!(o.finish(), "a running operation finishes");
    assert!(o.finish(), "finishing twice is still finished");
    assert!(!o.cancel());
    assert!(!o.is_cancelled());
    // Interrupt wins: the completion finds it cancelled.
    let c = op(B, b"q");
    assert!(c.cancel());
    assert!(!c.finish(), "a cancelled operation does not finish");
    assert!(c.is_cancelled());
    assert!(!c.cancel());
    let ops = Operations::default();
    ops.register(&o, 2).unwrap();
    ops.register(&op(B, b"q"), 2).unwrap();
    let mut ids: Vec<String> = ops
        .all()
        .iter()
        .map(|o| o.id().as_str().to_owned())
        .collect();
    ids.sort();
    assert_eq!(ids, vec![A.to_owned(), B.to_owned()]);
}

/// Racing threads: exactly one of `cancel` and `finish` wins, and their
/// answers agree with the final state.
#[test]
fn cancel_and_finish_race_to_one_winner() {
    for _ in 0..2_000 {
        let o = op(A, b"p");
        let (c, f) = std::thread::scope(|s| {
            let c = s.spawn(|| o.cancel());
            let f = s.spawn(|| o.finish());
            (c.join().unwrap(), f.join().unwrap())
        });
        assert!(c != f, "cancelled {c}, finished {f}");
        assert_eq!(o.is_cancelled(), c);
    }
}

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "#[tokio::test] blocks on its own runtime"
)]
async fn cancel_wakes_a_waiter() {
    let o = op(A, b"p");
    let waiter = {
        let o = Arc::clone(&o);
        tokio::spawn(async move { o.cancelled().await })
    };
    tokio::task::yield_now().await;
    o.cancel();
    waiter.await.unwrap();
}

#[test]
fn limits_from_config() {
    let mut c = Config::new();
    assert_eq!(
        ExecuteLimits::from_config(&c).max_operations_per_session,
        100
    );
    c.set("krishiv.connect.maxOperationsPerSession", "7")
        .unwrap();
    c.set("krishiv.connect.execute.aheadBytes", "2m").unwrap();
    let l = ExecuteLimits::from_config(&c);
    assert_eq!(l.max_operations_per_session, 7);
    assert_eq!(l.ahead_bytes, 2 << 20);
    assert_eq!(l.max_plan_size, 512 << 20);
    assert_eq!(l.max_batch_size, 128 << 20);
}

#[test]
fn states_follow_the_operation() {
    let running = op(A, b"p");
    assert_eq!(running.state(), OperationState::Running);
    let done = op(A, b"p");
    assert!(done.finish());
    assert_eq!(done.state(), OperationState::Succeeded);
    let failed = op(A, b"p");
    assert!(failed.fail(), "a running operation fails");
    assert!(failed.fail(), "failing twice is still failed");
    assert!(!failed.finish(), "a failed operation does not finish");
    assert!(!failed.cancel(), "a failed operation is not interrupted");
    assert_eq!(failed.state(), OperationState::Failed);
    // An interrupt is TERMINATING until its encoder stopped, then CANCELLED.
    let cancelled = op(A, b"p");
    assert!(cancelled.cancel());
    assert!(!cancelled.fail(), "an interrupt that came first wins");
    assert_eq!(cancelled.state(), OperationState::Terminating);
    cancelled.mark_ended();
    assert_eq!(cancelled.state(), OperationState::Cancelled);
}

#[test]
fn ended_operations_keep_their_final_state_for_the_window() {
    let ops = Operations::default();
    let done = op(A, b"p");
    let gone = op(B, b"q");
    ops.register(&done, 10).unwrap();
    ops.register(&gone, 10).unwrap();
    assert!(done.finish());
    assert_eq!(ops.state_of(A), Some(OperationState::Succeeded));
    let now = Instant::now();
    assert!(ops.remove(&done));
    // Removed while it runs: closed, so interrupted.
    gone.close();
    assert!(ops.remove(&gone));
    assert_eq!(ops.state_of(A), Some(OperationState::Succeeded));
    // An interrupted operation removed before its encoder stopped ended
    // cancelled.
    assert_eq!(ops.state_of(B), Some(OperationState::Cancelled));
    assert_eq!(ops.state_of("not a uuid"), None);
    let mut known = ops.known();
    known.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        known,
        vec![
            (QueryId::parse(A).unwrap(), OperationState::Succeeded),
            (QueryId::parse(B).unwrap(), OperationState::Cancelled),
        ]
    );
    // A lookup by id is an access (Spark's `expireAfterAccess`): A stays,
    // B, untouched for the whole window, goes.
    let later = now + ENDED_OPERATION_WINDOW - Duration::from_secs(1);
    assert!(ops.state_of_at(A, later).is_some());
    let past = now + ENDED_OPERATION_WINDOW + Duration::from_secs(1);
    assert_eq!(ops.state_of_at(B, past), None);
    assert_eq!(
        ops.known_at(past),
        vec![(QueryId::parse(A).unwrap(), OperationState::Succeeded)]
    );
    let gone_too = later + ENDED_OPERATION_WINDOW + Duration::from_secs(1);
    assert!(ops.known_at(gone_too).is_empty());
}

#[test]
fn ended_operations_are_capped_oldest_first() {
    let ops = Operations::default();
    let ids: Vec<String> = (0..=ENDED_OPERATIONS_KEPT)
        .map(|i| format!("00000000-0000-4000-8000-{i:012x}"))
        .collect();
    for id in &ids {
        let o = op(id, b"p");
        ops.register(&o, usize::MAX).unwrap();
        assert!(o.finish());
        assert!(ops.remove(&o));
    }
    assert_eq!(ops.known().len(), ENDED_OPERATIONS_KEPT);
    assert_eq!(ops.state_of(&ids[0]), None, "the oldest went first");
    assert_eq!(
        ops.state_of(&ids[ENDED_OPERATIONS_KEPT]),
        Some(OperationState::Succeeded)
    );
    // A live operation is listed with its live state; close_all records it.
    let live = op(A, b"p");
    ops.register(&live, usize::MAX).unwrap();
    assert!(
        ops.known()
            .contains(&(QueryId::parse(A).unwrap(), OperationState::Running))
    );
    ops.close_all();
    assert_eq!(ops.state_of(A), Some(OperationState::Cancelled));
}
