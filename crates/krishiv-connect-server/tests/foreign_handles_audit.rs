// SPDX-License-Identifier: Apache-2.0
//! ADR 0046 §3b: a session or operation id another principal holds answers as
//! a missing one and still writes one `krishiv::audit` denial record; an
//! unknown id writes none. One test per binary: the subscriber is
//! process-wide.
#![cfg(test)]

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use krishiv_common::principal::{Principal, PrincipalKind};
use krishiv_connect_server::manager::{SessionLimits, SessionManager};
use krishiv_connect_server::operation::{Operation, plan_digest};
use krishiv_session::auth::{Authenticated, Method, Role};
use krishiv_session::{QueryId, Session};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct Capture;

struct Fields(String);

impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let _ = write!(self.0, "{}={value:?} ", field.name());
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        let _ = write!(self.0, "{}={value} ", field.name());
    }
}

impl Subscriber for Capture {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        if event.metadata().target() == "krishiv::audit" {
            let mut fields = Fields(String::new());
            event.record(&mut fields);
            RECORDS.lock().unwrap().push(fields.0);
        }
    }

    fn enter(&self, _: &Id) {}

    fn exit(&self, _: &Id) {}
}

const ID: &str = "8f0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d10";
const OP: &str = "1d6a3c1e-77b2-4c55-8d0f-0a6c2e9b4f21";
const UNKNOWN: &str = "5b1f7a2e-0c3d-4e8f-9a6b-2d4c6e8f0a1b";

fn user(name: &str) -> Authenticated {
    Authenticated::new(
        Principal::new(name, PrincipalKind::User),
        Some(Role::Operator),
        Method::Token,
    )
}

#[test]
fn foreign_ids_are_answered_as_unknown_and_recorded() {
    tracing::subscriber::set_global_default(Capture).unwrap();
    let take = || std::mem::take(&mut *RECORDS.lock().unwrap());
    let m = SessionManager::from_base(Session::new().unwrap(), SessionLimits::default());
    let (alice, bob) = (user("alice"), user("bob"));
    let entry = m.get_or_create(&alice, ID, None, "u").unwrap();
    let op = Arc::new(Operation::new(
        QueryId::parse(OP).unwrap(),
        plan_digest(b"p"),
        Arc::from(Vec::new()),
        false,
    ));
    entry.operations().register(&op, 10).unwrap();
    m.get_or_create(&bob, UNKNOWN, None, "u").unwrap();
    assert!(take().is_empty(), "own sessions write nothing");

    // Unknown ids: no record.
    let unknown = "0a0b0c0d-0e0f-4a1b-8c2d-3e4f5a6b7c8d";
    m.get(&bob, unknown, None).unwrap_err();
    assert_eq!(m.release(&bob, unknown, false), None);
    m.audit_foreign_operation(&bob, unknown);
    assert!(take().is_empty(), "unknown ids write nothing");

    // Alice's ids, asked for by bob: one record each, naming bob and the kind of id.
    for e in [
        m.get(&bob, ID, None).unwrap_err(),
        m.get_or_create(&bob, ID, None, "alice").unwrap_err(),
    ] {
        assert_eq!(e.error_class().name(), "INVALID_HANDLE.SESSION_NOT_FOUND");
    }
    assert_eq!(m.release(&bob, ID, false), None);
    m.audit_foreign_operation(&bob, OP);
    let records = take();
    assert_eq!(records.len(), 4, "{records:?}");
    for (r, handle) in records
        .iter()
        .zip(["session", "session", "session", "operation"])
    {
        assert!(r.contains("principal=bob "), "{r}");
        assert!(r.contains("surface=connect "), "{r}");
        assert!(r.contains("role=operator "), "{r}");
        assert!(r.contains(&format!("handle={handle} ")), "{r}");
        assert!(!r.contains("alice"), "the owner is never named: {r}");
    }

    // Alice's own operation, from her own session: nothing.
    m.audit_foreign_operation(&alice, OP);
    assert!(take().is_empty());
    m.get(&alice, ID, None).unwrap();
}
