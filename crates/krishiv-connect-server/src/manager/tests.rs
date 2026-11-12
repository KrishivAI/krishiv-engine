// SPDX-License-Identifier: Apache-2.0
use std::sync::Arc;
use std::time::Duration;

use krishiv_common::principal::{Principal, PrincipalKind};
use krishiv_session::Session;

use super::{SessionLimits, SessionManager};

const ID: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10";
const ID2: &str = "1d6a3c1e-77b2-4c55-8d0f-0a6c2e9b4f21";

fn alice() -> Principal {
    Principal::new("alice", PrincipalKind::User)
}

fn bob() -> Principal {
    Principal::new("bob", PrincipalKind::User)
}

fn manager(limits: SessionLimits) -> SessionManager {
    let base = Session::new().expect("session");
    SessionManager::from_base(base, limits)
}

fn class(e: &krishiv_common::error::KrishivError) -> &'static str {
    e.error_class().name()
}

#[test]
fn get_or_create_then_lookup_returns_the_same_session() {
    let m = manager(SessionLimits::default());
    let a = m.get_or_create(&alice(), ID, None, "u").expect("create");
    let b = m.get(&alice(), ID, None).expect("get");
    assert!(Arc::ptr_eq(&a, &b));
    assert_ne!(
        a.server_side_id(),
        ID,
        "server-side id differs from the client's"
    );
    assert_eq!(a.client_user_id(), "u");
    assert_eq!(m.counts().live, 1);
}

#[test]
fn a_malformed_id_is_format_on_create_and_not_found_on_lookup() {
    let m = manager(SessionLimits::default());
    let e = m
        .get_or_create(&alice(), "not-a-uuid", None, "u")
        .unwrap_err();
    assert_eq!(class(&e), "INVALID_HANDLE.FORMAT");
    let e = m.get(&alice(), "not-a-uuid", None).unwrap_err();
    assert_eq!(class(&e), "INVALID_HANDLE.SESSION_NOT_FOUND");
    assert_eq!(m.counts().live, 0);
}

#[test]
fn any_uuid_form_java_accepts_creates() {
    let m = manager(SessionLimits::default());
    // UUID.fromString accepts short groups; the key is the id as sent.
    m.get_or_create(&alice(), "1-2-3-4-5", None, "u")
        .expect("create");
    m.get(&alice(), "1-2-3-4-5", None).expect("same text");
}

#[test]
fn a_released_id_is_session_closed_until_its_tombstone_is_evicted() {
    let m = manager(SessionLimits {
        tombstones: 1,
        ..SessionLimits::default()
    });
    let entry = m.get_or_create(&alice(), ID, None, "u").expect("create");
    let released = m.release(&alice(), ID, false).expect("released");
    assert_eq!(released, entry.server_side_id());
    assert!(entry.session().is_closed());
    for e in [
        m.get(&alice(), ID, None).unwrap_err(),
        m.get_or_create(&alice(), ID, None, "u").unwrap_err(),
    ] {
        assert_eq!(class(&e), "INVALID_HANDLE.SESSION_CLOSED");
    }
    // A second release evicts the first tombstone (capacity 1).
    m.get_or_create(&alice(), ID2, None, "u").expect("create");
    m.release(&alice(), ID2, false);
    assert_eq!(m.counts().tombstones, 1);
    m.get_or_create(&alice(), ID, None, "u")
        .expect("evicted tombstone: the id is new again");
}

#[test]
fn allow_reconnect_writes_no_tombstone() {
    let m = manager(SessionLimits::default());
    let first = m.get_or_create(&alice(), ID, None, "u").expect("create");
    m.release(&alice(), ID, true);
    assert_eq!(m.counts().tombstones, 0);
    let again = m.get_or_create(&alice(), ID, None, "u").expect("recreate");
    assert_ne!(first.server_side_id(), again.server_side_id());
}

#[test]
fn releasing_an_unknown_id_is_a_no_op() {
    let m = manager(SessionLimits::default());
    assert_eq!(m.release(&alice(), ID, false), None);
    assert_eq!(m.counts().tombstones, 0);
}

#[test]
fn a_changed_server_side_id_is_session_changed() {
    let m = manager(SessionLimits::default());
    let entry = m.get_or_create(&alice(), ID, None, "u").expect("create");
    let seen = entry.server_side_id().to_owned();
    m.get(&alice(), ID, Some(&seen)).expect("same id");
    m.get_or_create(&alice(), ID, Some(&seen), "u")
        .expect("same id");
    let e = m.get(&alice(), ID, Some("other")).unwrap_err();
    assert_eq!(class(&e), "INVALID_HANDLE.SESSION_CHANGED");
    // After a restart (a new manager), a client's old server-side id
    // creates a fresh session and is then refused as changed.
    let restarted = manager(SessionLimits::default());
    let e = restarted
        .get_or_create(&alice(), ID, Some(&seen), "u")
        .unwrap_err();
    assert_eq!(class(&e), "INVALID_HANDLE.SESSION_CHANGED");
}

#[test]
fn another_principals_session_id_is_not_found_and_not_shared() {
    let m = manager(SessionLimits::default());
    m.get_or_create(&alice(), ID, None, "u").expect("create");
    for e in [
        m.get(&bob(), ID, None).unwrap_err(),
        m.get_or_create(&bob(), ID, None, "alice").unwrap_err(),
    ] {
        assert_eq!(class(&e), "INVALID_HANDLE.SESSION_NOT_FOUND");
    }
    // Bob's release of Alice's id does nothing.
    assert_eq!(m.release(&bob(), ID, false), None);
    m.get(&alice(), ID, None).expect("still alice's");
    assert_eq!(m.counts().live, 1);
}

#[test]
fn session_caps_refuse_before_building() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let built = Arc::new(AtomicUsize::new(0));
    let base = Session::new().expect("session");
    let counter = Arc::clone(&built);
    let m = SessionManager::new(
        Box::new(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(base.new_session())
        }),
        SessionLimits {
            max_sessions: 2,
            max_sessions_per_principal: 1,
            ..SessionLimits::default()
        },
    );
    m.get_or_create(&alice(), ID, None, "u").expect("first");
    assert_eq!(built.load(Ordering::SeqCst), 1);
    let e = m.get_or_create(&alice(), ID2, None, "u").unwrap_err();
    assert_eq!(built.load(Ordering::SeqCst), 1, "refused before building");
    assert_eq!(class(&e), "KRISHIV_SESSION_LIMIT");
    assert!(
        e.to_string()
            .contains("krishiv.connect.maxSessionsPerPrincipal")
    );
    m.get_or_create(&bob(), ID2, None, "u")
        .expect("bob's first");
    let carol = Principal::new("carol", PrincipalKind::User);
    let e = m
        .get_or_create(&carol, "2b1c6e0a-0f3d-4f7e-9a51-7c6d5e4f3a21", None, "u")
        .unwrap_err();
    assert!(e.to_string().contains("krishiv.connect.maxSessions"));
    assert_eq!(built.load(Ordering::SeqCst), 2, "refused before building");
    // Releasing frees a slot.
    m.release(&alice(), ID, false);
    m.get_or_create(&alice(), "3c2d7f1b-1a4e-4a8f-8b62-8d7e6f5a4b32", None, "u")
        .expect("a slot is free again");
}

#[test]
fn sweep_releases_idle_sessions_and_tombstones_them() {
    let m = manager(SessionLimits {
        idle_timeout: Some(Duration::from_millis(1)),
        ..SessionLimits::default()
    });
    let entry = m.get_or_create(&alice(), ID, None, "u").expect("create");
    // Idle past the 1 ms timeout, without sleeping a thread.
    let since = std::time::Instant::now();
    while since.elapsed() < Duration::from_millis(5) {
        std::hint::spin_loop();
    }
    assert_eq!(m.sweep(), 1);
    assert!(entry.session().is_closed());
    let e = m.get(&alice(), ID, None).unwrap_err();
    assert_eq!(class(&e), "INVALID_HANDLE.SESSION_CLOSED");
}

#[test]
fn sweep_with_no_timeout_keeps_everything() {
    let m = manager(SessionLimits {
        idle_timeout: None,
        ..SessionLimits::default()
    });
    m.get_or_create(&alice(), ID, None, "u").expect("create");
    assert_eq!(m.sweep(), 0);
    assert_eq!(m.counts().live, 1);
}

#[test]
fn close_all_closes_every_session() {
    let m = manager(SessionLimits::default());
    let a = m.get_or_create(&alice(), ID, None, "u").expect("a");
    let b = m.get_or_create(&bob(), ID2, None, "u").expect("b");
    assert_eq!(m.close_all(), 2);
    assert!(a.session().is_closed() && b.session().is_closed());
    assert_eq!(m.counts().live, 0);
}

#[test]
fn a_failing_factory_creates_nothing() {
    let m = SessionManager::new(
        Box::new(|_| {
            Err(krishiv_common::error::InternalError {
                message: "no".to_owned(),
            }
            .into())
        }),
        SessionLimits::default(),
    );
    m.get_or_create(&alice(), ID, None, "u").unwrap_err();
    assert_eq!(m.counts().live, 0);
}

#[test]
fn limits_come_from_the_config() {
    let mut c = krishiv_common::config::Config::new();
    assert_eq!(SessionLimits::from_config(&c).max_sessions, 1000);
    c.set("krishiv.connect.maxSessions", "3").unwrap();
    c.set("krishiv.connect.session.idleTimeout", "0s").unwrap();
    c.set(
        "spark.connect.session.manager.closedSessionsTombstonesSize",
        "5",
    )
    .unwrap();
    let l = SessionLimits::from_config(&c);
    assert_eq!((l.max_sessions, l.idle_timeout, l.tombstones), (3, None, 5));
}

#[test]
fn no_session_is_created_after_close_all() {
    let m = manager(SessionLimits::default());
    m.close_all();
    let e = m.get_or_create(&alice(), ID, None, "u").unwrap_err();
    assert_eq!(class(&e), "INVALID_HANDLE.SESSION_CLOSED");
    assert_eq!(m.counts().live, 0);
}

#[test]
fn racing_creates_of_one_key_share_one_session_and_close_the_losers() {
    let m = Arc::new(manager(SessionLimits::default()));
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let (m, barrier) = (Arc::clone(&m), Arc::clone(&barrier));
            std::thread::spawn(move || {
                barrier.wait();
                m.get_or_create(&alice(), ID, None, "u").expect("create")
            })
        })
        .collect();
    let entries: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("join"))
        .collect();
    for e in &entries {
        assert!(Arc::ptr_eq(e, &entries[0]), "every caller got the winner");
        assert!(!e.session().is_closed());
    }
    assert_eq!(m.counts().live, 1);
}

#[test]
fn get_if_present_finds_only_the_owners_live_session() {
    let m = manager(SessionLimits::default());
    assert!(m.get_if_present(&alice(), ID).is_none());
    let a = m.get_or_create(&alice(), ID, None, "u").expect("create");
    let found = m.get_if_present(&alice(), ID).expect("present");
    assert!(Arc::ptr_eq(&a, &found));
    assert!(m.get_if_present(&bob(), ID).is_none());
    m.release(&alice(), ID, false);
    assert!(m.get_if_present(&alice(), ID).is_none());
}

#[test]
fn zero_tombstones_keeps_no_released_id() {
    let limits = SessionLimits {
        tombstones: 0,
        ..SessionLimits::default()
    };
    let m = manager(limits);
    assert_eq!(m.limits().tombstones, 0);
    m.get_or_create(&alice(), ID, None, "u").expect("create");
    m.release(&alice(), ID, false);
    assert_eq!(m.counts().tombstones, 0);
    // With no tombstone the id can be created again.
    m.get_or_create(&alice(), ID, None, "u").expect("recreate");
    let debug = format!("{m:?}");
    assert!(debug.starts_with("SessionManager"), "{debug}");
    assert!(debug.contains("live: 1"), "{debug}");
}

#[test]
fn release_and_close_all_cancel_the_operations() {
    use crate::operation::{Operation, plan_digest};
    let op = |id: &str| {
        Arc::new(Operation::new(
            krishiv_session::QueryId::parse(id).expect("id"),
            plan_digest(b"p"),
            Arc::from(Vec::new()),
            false,
        ))
    };
    let m = manager(SessionLimits::default());
    let entry = m.get_or_create(&alice(), ID, None, "u").expect("create");
    let first = op(ID);
    entry.operations().register(&first, 10).expect("register");
    m.release(&alice(), ID, false).expect("released");
    assert!(first.is_cancelled());
    assert!(entry.operations().is_empty());

    let entry = m.get_or_create(&alice(), ID2, None, "u").expect("create");
    let second = op(ID2);
    entry.operations().register(&second, 10).expect("register");
    assert_eq!(m.close_all(), 1);
    assert!(second.is_cancelled());
}
