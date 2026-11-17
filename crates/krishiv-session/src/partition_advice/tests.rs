// SPDX-License-Identifier: Apache-2.0
//! The history-fed partition advisor: a planted workload finds its partition key, the advice reaches `EXPLAIN` and
//! `system.partition_advice`, applying it reads at least half the bytes less, and nothing about it runs DDL.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_exec::profile::QueryProfile;

use super::from_history;
use crate::profile_history::read_history;
use crate::{EngineRuntime, Session};

const EVENTS: &str =
    "SELECT id, CAST(id % 365 AS INT) AS d, CAST(id % 3 AS INT) AS c FROM range(0, 20000)";

fn setup(history: &std::path::Path, warehouse: &std::path::Path) -> (Arc<EngineRuntime>, Session) {
    let rt = EngineRuntime::builder()
        .config("krishiv.history.path", history.to_str().unwrap())
        .build()
        .unwrap();
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    session
        .conf()
        .set("spark.sql.warehouse.dir", warehouse.to_str().unwrap())
        .unwrap();
    (rt, session)
}

fn run(session: &Session, sql: &str) -> Vec<arrow_array::RecordBatch> {
    block_on(async { session.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn text(session: &Session, sql: &str) -> String {
    run(session, sql)
        .iter()
        .flat_map(|b| {
            (0..b.num_rows())
                .map(|i| cell(b.column(0).as_ref(), i))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn cell(c: &dyn Array, i: usize) -> String {
    match c.data_type() {
        arrow_schema::DataType::Utf8View => c.as_string_view().value(i).to_owned(),
        arrow_schema::DataType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
        other => panic!("unexpected type {other}"),
    }
}

/// 100 queries filtering `table` on `d` and 20 on `c`.
fn workload(session: &Session, table: &str) {
    for i in 0..100 {
        run(
            session,
            &format!("SELECT sum(id) FROM {table} WHERE d = {}", i * 3),
        );
    }
    for i in 0..20 {
        run(
            session,
            &format!("SELECT sum(id) FROM {table} WHERE c = {}", i % 3),
        );
    }
}

fn flush(rt: &EngineRuntime) {
    rt.profile_history.as_ref().unwrap().flush();
}

/// The bytes the scans read, summed over the stored queries whose text contains `needle`.
fn scanned_bytes(dir: &std::path::Path, needle: &str) -> u64 {
    read_history(dir)
        .unwrap()
        .iter()
        .filter(|r| r.sql.as_deref().is_some_and(|s| s.contains(needle)))
        .map(|r| {
            krishiv_exec::profile::QueryUsage::of(
                &QueryProfile::from_json(&r.profile).unwrap().metrics,
            )
            .scan_bytes
        })
        .sum()
}

#[test]
fn advisor_finds_the_planted_key_from_the_history() {
    let (history, warehouse) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let (rt, session) = setup(history.path(), warehouse.path());
    run(
        &session,
        &format!("CREATE TABLE ev USING parquet AS {EVENTS}"),
    );
    workload(&session, "ev");
    flush(&rt);
    let advice = from_history(history.path());
    assert_eq!(advice.len(), 1, "{advice:?}");
    let t = &advice[0];
    assert!(t.table.ends_with(".ev"), "{}", t.table);
    assert_eq!(t.queries, 120);
    assert_eq!(t.advice[0].column, "d");
    assert_eq!(t.advice[0].queries, 100);
    assert_eq!(t.advice[0].distinct, 365);
    assert_eq!(t.advice[1].column, "c");
}

#[test]
fn explain_and_the_system_table_show_the_advice_and_no_ddl_runs() {
    let (history, warehouse) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let (rt, session) = setup(history.path(), warehouse.path());
    run(
        &session,
        &format!("CREATE TABLE ev USING parquet AS {EVENTS}"),
    );
    workload(&session, "ev");
    flush(&rt);
    let plan = text(&session, "EXPLAIN SELECT sum(id) FROM ev WHERE d = 7");
    assert!(
        plan.contains("partition advice: d (100 queries, 365 partitions"),
        "{plan}"
    );
    // A query on another table has no advice section.
    let other = text(&session, "EXPLAIN SELECT 1");
    assert!(!other.contains("partition advice"), "{other}");
    let rows = run(
        &session,
        "SELECT table_name, rank, column_name, column_queries FROM system.partition_advice ORDER BY rank",
    );
    let b = &rows[0];
    assert!(b.num_rows() >= 2);
    assert_eq!(cell(b.column(2).as_ref(), 0), "d");
    // The table is as it was: advice runs no DDL, so no partition directory appeared.
    assert!(!has_partition_dirs(warehouse.path()));
}

fn has_partition_dirs(dir: &std::path::Path) -> bool {
    std::fs::read_dir(dir).unwrap().flatten().any(|e| {
        e.file_name().to_string_lossy().contains('=')
            || (e.path().is_dir() && has_partition_dirs(&e.path()))
    })
}

#[test]
fn applying_the_advice_reads_at_least_half_the_bytes_less() {
    let (history, warehouse) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let (rt, session) = setup(history.path(), warehouse.path());
    run(
        &session,
        &format!("CREATE TABLE ev USING parquet AS {EVENTS}"),
    );
    workload(&session, "ev");
    flush(&rt);
    let advised = from_history(history.path())[0].advice[0].column.clone();
    // The test script applies the advice: the same data partitioned by the advised column.
    run(
        &session,
        &format!("CREATE TABLE evp USING parquet PARTITIONED BY ({advised}) AS {EVENTS}"),
    );
    workload(&session, "evp");
    flush(&rt);
    let before = scanned_bytes(history.path(), "FROM ev WHERE");
    let after = scanned_bytes(history.path(), "FROM evp WHERE");
    assert!(before > 0, "the unpartitioned table read no bytes");
    assert!(after * 2 <= before, "after {after} before {before}");
}

/// Allows everything except `MONITOR` on `system`, which only `admin` holds.
struct MonitorForAdmin;

impl crate::AccessPolicy for MonitorForAdmin {
    fn version(&self) -> crate::PolicyVersion {
        crate::PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        principal: &'a crate::Principal,
        securable: &'a crate::Securable,
        privilege: crate::Privilege,
    ) -> crate::BoxFuture<'a, Result<crate::AccessDecision, krishiv_common::error::KrishivError>>
    {
        let decision = match (securable, privilege) {
            (crate::Securable::System(_), crate::Privilege::Monitor)
                if principal.name != "admin" =>
            {
                crate::AccessDecision::Deny {
                    reason: "monitor".to_owned(),
                }
            }
            _ => crate::AccessDecision::Allow,
        };
        Box::pin(std::future::ready(Ok(decision)))
    }
}

fn as_user(rt: &Arc<EngineRuntime>, who: &str, warehouse: &std::path::Path) -> Session {
    let s = Session::builder()
        .runtime(Arc::clone(rt))
        .principal(crate::Principal::new(who, crate::PrincipalKind::User))
        .access_policy(Arc::new(MonitorForAdmin))
        .build()
        .unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", warehouse.to_str().unwrap())
        .unwrap();
    s
}

#[test]
fn a_caller_without_monitor_sees_no_advice_on_either_surface() {
    let (history, warehouse) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let rt = EngineRuntime::builder()
        .config("krishiv.history.path", history.path().to_str().unwrap())
        .build()
        .unwrap();
    let admin = as_user(&rt, "admin", warehouse.path());
    run(
        &admin,
        &format!("CREATE TABLE ev USING parquet AS {EVENTS}"),
    );
    workload(&admin, "ev");
    flush(&rt);
    let query = "EXPLAIN SELECT sum(id) FROM ev WHERE d = 7";
    let table = "SELECT CAST(count(*) AS STRING) FROM system.partition_advice";
    // The holder of MONITOR sees the history of every principal.
    assert!(text(&admin, query).contains("partition advice: d"));
    assert_ne!(text(&admin, table), "0");
    // Another principal, with SELECT on the table, sees neither.
    let bob = as_user(&rt, "bob", warehouse.path());
    let plan = text(&bob, query);
    assert!(!plan.contains("partition advice"), "{plan}");
    assert!(!plan.contains("Partition Advice"), "{plan}");
    assert_eq!(text(&bob, table), "0");
}

#[test]
fn a_unique_key_is_not_advised_through_the_history() {
    let (history, warehouse) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let (rt, session) = setup(history.path(), warehouse.path());
    run(
        &session,
        &format!("CREATE TABLE ev USING parquet AS {EVENTS}"),
    );
    // Only the unique `id` is filtered by.
    for i in 0..50 {
        run(&session, &format!("SELECT sum(c) FROM ev WHERE id = {i}"));
    }
    flush(&rt);
    let advice = from_history(history.path());
    assert!(
        advice
            .iter()
            .all(|t| t.advice.iter().all(|a| a.column != "id")),
        "{advice:?}"
    );
}

#[test]
fn a_table_is_one_table_whatever_the_case_of_its_name() {
    let (history, warehouse) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let (rt, session) = setup(history.path(), warehouse.path());
    run(
        &session,
        &format!("CREATE TABLE ev USING parquet AS {EVENTS}"),
    );
    for i in 0..50 {
        run(&session, &format!("SELECT sum(id) FROM EV WHERE d = {i}"));
        run(&session, &format!("SELECT sum(id) FROM ev WHERE d = {i}"));
    }
    flush(&rt);
    let advice = from_history(history.path());
    assert_eq!(advice.len(), 1, "{advice:?}");
    assert_eq!(advice[0].queries, 100);
    let plan = text(&session, "EXPLAIN SELECT sum(id) FROM EV WHERE d = 7");
    assert!(plan.contains("partition advice: d"), "{plan}");
}
