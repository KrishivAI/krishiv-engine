// SPDX-License-Identifier: Apache-2.0
//! P3-T062 review M4: under the grants provider each refusal leaves exactly one `krishiv::audit`
//! record, carrying the provider and the policy epoch. One test per binary: the subscriber is
//! process-wide.
#![cfg(test)]

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use futures::executor::block_on;
use krishiv_plan::analyzer::PolicyVersion;
use krishiv_session::auth::{Role, RolePolicy};
use krishiv_session::grants::{Grants, GrantsPolicy, MemoryGrantStore};
use krishiv_session::{AccessPolicy, AllowAll, EngineRuntime, Principal, PrincipalKind, Session};
use tracing::field::{Field as TraceField, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct Capture;

struct Fields(String);

impl Visit for Fields {
    fn record_debug(&mut self, field: &TraceField, value: &dyn std::fmt::Debug) {
        let _ = write!(self.0, "{}={value:?} ", field.name());
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
fn each_refusal_leaves_exactly_one_record() {
    tracing::subscriber::set_global_default(Capture).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let rt = EngineRuntime::builder()
        .config(
            "spark.sql.warehouse.dir",
            dir.path().join("wh").to_str().unwrap(),
        )
        .build()
        .unwrap();
    let grants = Grants::new(Arc::new(MemoryGrantStore::new()));
    let viewer = || -> Arc<dyn AccessPolicy> {
        Arc::new(RolePolicy::new(
            Some(Role::Viewer),
            Role::Admin,
            PolicyVersion(1),
        ))
    };
    let session_over = |name: &str, admin: bool, system: Arc<dyn AccessPolicy>| {
        let policy = Arc::new(GrantsPolicy::new(Arc::clone(&grants), system, admin));
        Session::builder()
            .runtime(Arc::clone(&rt))
            .principal(Principal::new(name, PrincipalKind::User))
            .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
            .grants(policy)
            .build()
            .unwrap()
    };
    let session = |name: &str, admin: bool| session_over(name, admin, Arc::new(AllowAll));
    let (root, ana) = (session("root", true), session("ana", false));
    block_on(async {
        root.sql("CREATE TABLE t (a INT, b INT) USING parquet")
            .await?
            .collect()
            .await
    })
    .unwrap();
    block_on(async {
        root.sql("GRANT SELECT (a) ON t TO user:ana")
            .await?
            .collect()
            .await
    })
    .unwrap();
    let take = || std::mem::take(&mut *RECORDS.lock().unwrap());
    take();
    for sql in [
        "SELECT * FROM t",
        "SELECT b FROM t",
        "DROP TABLE t",
        "GRANT SELECT ON t TO user:bob",
    ] {
        let e = block_on(async { ana.sql(sql).await?.collect().await }).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INSUFFICIENT_PERMISSIONS",
            "{sql}"
        );
        let records = take();
        assert_eq!(records.len(), 1, "{sql}: {records:?}");
        assert!(
            records[0].contains("provider=\"grants\"") && records[0].contains("policy_epoch="),
            "{sql}: {records:?}"
        );
    }
    let e = block_on(async { ana.table("t").collect().await }).unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    assert_eq!(take().len(), 1);

    // A revoke of another grantor's grant is a no-op without a record; one without any grant
    // authority is refused with one (T062 REVOKE fix).
    for sql in [
        "CREATE TABLE t2 (a INT) USING parquet",
        "GRANT SELECT ON t2 TO user:ana WITH GRANT OPTION",
        "GRANT SELECT ON t2 TO user:carl",
    ] {
        block_on(async { root.sql(sql).await?.collect().await }).unwrap();
    }
    take();
    let refused = |s: &Session, sql: &str, provider: &str| {
        let e = block_on(async { s.sql(sql).await?.collect().await }).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INSUFFICIENT_PERMISSIONS",
            "{sql}"
        );
        let records = take();
        assert_eq!(records.len(), 1, "{sql}: {records:?}");
        assert!(
            records[0].contains(&format!("provider=\"{provider}\"")),
            "{sql}: {records:?}"
        );
    };
    block_on(async {
        ana.sql("REVOKE SELECT ON t2 FROM user:carl")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert!(take().is_empty());
    refused(
        &session("bob", false),
        "REVOKE SELECT ON t2 FROM user:carl",
        "grants",
    );
    // The role matrix's refusals carry the counter's label, `roles`, with or without grants.
    let viewer_with_grants = session_over("vic", false, viewer());
    refused(&viewer_with_grants, "SELECT * FROM system.memory", "roles");
    let roles_only = Session::builder()
        .runtime(Arc::clone(&rt))
        .principal(Principal::new("val", PrincipalKind::User))
        .access_policy(viewer())
        .build()
        .unwrap();
    refused(
        &roles_only,
        "CREATE TABLE t3 (a INT) USING parquet",
        "roles",
    );
}
