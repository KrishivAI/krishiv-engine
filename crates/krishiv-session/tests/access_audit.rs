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
    let _ = tracing::subscriber::set_global_default(Capture);
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

    // A name computed by `IDENTIFIER(...)` is recorded as the name it gives.
    assert_eq!(unplaceable.len(), 1, "{unplaceable:?}");
    assert!(
        unplaceable[0].contains("error_class=KRISHIV_INSUFFICIENT_PERMISSIONS"),
        "{unplaceable:?}"
    );
    assert!(
        unplaceable[0].contains("statement_class=DDL") && unplaceable[0].contains("`ab`"),
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
        assert!(
            !r.contains("client_user_id_unverified"),
            "no client id reported, none recorded: {r}"
        );
    }

    // ADR 0046 §5: a client's own `user_id` is recorded beside the principal, marked
    // unverified (review N6).
    s.set_client_info(
        krishiv_session::Transport::Connect,
        None,
        Some("dana"),
        None,
    );
    block_on(s.sql("SELECT * FROM t")).unwrap_err();
    let records = take();
    assert_eq!(records.len(), 1, "{records:?}");
    assert!(records[0].contains("principal=ana "), "{}", records[0]);
    assert!(
        records[0].contains("client_user_id_unverified=dana "),
        "{}",
        records[0]
    );
}

/// `Browse` only: a viewer.
struct BrowseOnly;

impl krishiv_session::AccessPolicy for BrowseOnly {
    fn version(&self) -> krishiv_plan::analyzer::PolicyVersion {
        krishiv_plan::analyzer::PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a krishiv_session::Principal,
        _: &'a krishiv_session::Securable,
        privilege: krishiv_session::Privilege,
    ) -> krishiv_plan::analyzer::BoxFuture<
        'a,
        Result<krishiv_session::AccessDecision, krishiv_common::error::KrishivError>,
    > {
        Box::pin(std::future::ready(Ok(
            if privilege == krishiv_session::Privilege::Browse {
                krishiv_session::AccessDecision::Allow
            } else {
                krishiv_session::AccessDecision::Deny {
                    reason: "viewer".to_owned(),
                }
            },
        )))
    }
}

/// P3 PR-14d: the `Select` trial of a viewer's explain is an answer, not a refusal: no record.
/// A refusal the viewer is shown (`cost`) is one.
#[test]
fn a_withheld_explain_leaves_no_audit_record() {
    let _ = tracing::subscriber::set_global_default(Capture);
    let s = Session::builder()
        .principal(Principal::new("vee", PrincipalKind::User))
        .access_policy(Arc::new(BrowseOnly))
        .build()
        .unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    s.register_batches("t", schema, vec![]).unwrap();
    let vee = || -> Vec<String> {
        RECORDS
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.contains("principal=vee"))
            .cloned()
            .collect()
    };
    let text = block_on(
        s.table("t")
            .explain_for_metadata(krishiv_session::ExplainMode::Simple),
    )
    .unwrap();
    assert!(text.contains("Statistics withheld"), "{text}");
    assert!(vee().is_empty(), "{:?}", vee());
    block_on(
        s.table("t")
            .explain_for_metadata(krishiv_session::ExplainMode::Cost),
    )
    .unwrap_err();
    assert_eq!(vee().len(), 1, "{:?}", vee());
}
