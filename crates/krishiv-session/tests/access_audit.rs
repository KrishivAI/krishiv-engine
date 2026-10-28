// SPDX-License-Identifier: Apache-2.0
//! Every denial leaves one structured audit record (`krishiv::audit`):
//! principal, the class of what was denied and the denial, never the
//! policy's reason. One test per binary: the subscriber is process-wide.
#![cfg(test)]

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::{DenyAll, Principal, PrincipalKind, Session};
use tracing::field::{Field as TraceField, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Keeps the `krishiv::audit` events as `field=value` lines.
struct Capture;

struct Fields(String);

impl Visit for Fields {
    fn record_debug(&mut self, field: &TraceField, value: &dyn std::fmt::Debug) {
        let _ = write!(self.0, "{}={value:?} ", field.name());
    }

    fn record_str(&mut self, field: &TraceField, value: &str) {
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

#[test]
fn each_denial_leaves_one_audit_record() {
    tracing::subscriber::set_global_default(Capture).unwrap();
    let s = Session::builder()
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(Arc::new(DenyAll))
        .build()
        .unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1]))],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("o");

    let take = || std::mem::take(&mut *RECORDS.lock().unwrap());
    block_on(s.sql("SELECT * FROM t")).unwrap_err();
    let read = take();
    block_on(s.sql("DROP TABLE t")).unwrap_err();
    let drop = take();
    block_on(s.table("t").collect()).unwrap_err();
    let frame = take();
    let df = block_on(s.sql("SELECT 1 AS a")).unwrap();
    block_on(df.write().parquet(out.to_str().unwrap())).unwrap_err();
    let write = take();
    block_on(s.sql("DROP TABLE IDENTIFIER('a' || 'b')")).unwrap_err();
    let unplaceable = take();
    block_on(s.sql("SET a = b")).unwrap();
    assert!(
        take().is_empty(),
        "allowed statements leave no audit record"
    );

    // A check that fails closed because it cannot name the object is
    // recorded too.
    assert_eq!(unplaceable.len(), 1, "{unplaceable:?}");
    assert!(
        unplaceable[0].contains("error_class=KRISHIV_UNSUPPORTED_FEATURE"),
        "{unplaceable:?}"
    );
    assert!(
        unplaceable[0].contains("statement_class=DDL"),
        "{unplaceable:?}"
    );

    for (records, class, privilege) in [
        (read, "READ", "SELECT"),
        (drop, "DDL", "DROP"),
        (frame, "READ", "SELECT"),
        (write, "WRITE", "MODIFY"),
    ] {
        assert_eq!(records.len(), 1, "{records:?}");
        let r = &records[0];
        assert!(r.contains("principal=ana"), "{r}");
        assert!(r.contains(&format!("statement_class={class}")), "{r}");
        assert!(
            r.contains("error_class=KRISHIV_INSUFFICIENT_PERMISSIONS"),
            "{r}"
        );
        assert!(r.contains(privilege), "{r}");
        assert!(
            !r.contains("deny-all"),
            "the policy's reason stays out: {r}"
        );
    }
}
