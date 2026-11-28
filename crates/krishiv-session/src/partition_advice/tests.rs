// SPDX-License-Identifier: Apache-2.0
//! The history-fed partition advisor: a planted workload finds its partition key, the advice reaches `EXPLAIN` and
//! `system.partition_advice`, applying it reads at least half the bytes less, and nothing about it runs DDL.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_exec::profile::QueryProfile;

use krishiv_exec::profile::{ProfileScan, ScanColumn};

use super::{TableAdvice, advice_of, from_history, section_of};
use crate::profile_history::read_history;
use crate::{EngineRuntime, Session};

const EVENTS: &str =
    "SELECT id, CAST(id % 365 AS INT) AS d, CAST(id % 3 AS INT) AS c FROM range(0, 20000)";

fn setup(history: &std::path::Path, warehouse: &std::path::Path) -> (Arc<EngineRuntime>, Session) {
    let rt = EngineRuntime::builder()
        .config("krishiv.history.path", history.to_str().unwrap())
        .build()
        .unwrap();
    let session = Session::builder()
        .runtime(Arc::clone(&rt))
        .config("spark.sql.warehouse.dir", warehouse.to_str().unwrap())
        .build()
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

/// 100 queries filtering `table` on `d` and 20 on `c`; their results, in order.
fn workload(session: &Session, table: &str) -> Vec<String> {
    let d = (0..100).map(|i| {
        format!(
            "SELECT CAST(sum(id) AS STRING) FROM {table} WHERE d = {}",
            i * 3
        )
    });
    let c = (0..20).map(|i| {
        format!(
            "SELECT CAST(sum(id) AS STRING) FROM {table} WHERE c = {}",
            i % 3
        )
    });
    d.chain(c).map(|sql| text(session, &sql)).collect()
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
    assert_eq!(t.table, "spark_catalog.default.ev");
    assert_eq!(t.queries, 120);
    assert_eq!(t.advice.len(), 2, "{advice:?}");
    assert_eq!(t.advice[0].column, "d");
    assert_eq!(t.advice[0].queries, 100);
    assert_eq!(t.advice[0].distinct, 365);
    assert_eq!(t.advice[1].column, "c");
    assert_eq!(t.advice[1].queries, 20);
    assert_eq!(t.advice[1].distinct, 3);
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
    let got: Vec<(String, i32, String, i64)> = rows
        .iter()
        .flat_map(|b| {
            (0..b.num_rows()).map(move |i| {
                (
                    cell(b.column(0).as_ref(), i),
                    b.column(1)
                        .as_primitive::<arrow_array::types::Int32Type>()
                        .value(i),
                    cell(b.column(2).as_ref(), i),
                    b.column(3)
                        .as_primitive::<arrow_array::types::Int64Type>()
                        .value(i),
                )
            })
        })
        .collect();
    let table = "spark_catalog.default.ev".to_owned();
    assert_eq!(
        got,
        [
            (table.clone(), 1, "d".to_owned(), 100),
            (table, 2, "c".to_owned(), 20),
        ]
    );
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
    let unpartitioned = workload(&session, "ev");
    flush(&rt);
    let advised = from_history(history.path())[0].advice[0].column.clone();
    // The test script applies the advice: the same data partitioned by the advised column.
    run(
        &session,
        &format!("CREATE TABLE evp USING parquet PARTITIONED BY ({advised}) AS {EVENTS}"),
    );
    // Pruning keeps every row a query reads: each result is the unpartitioned table's.
    assert_eq!(workload(&session, "evp"), unpartitioned);
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
    Session::builder()
        .runtime(Arc::clone(rt))
        .principal(crate::Principal::new(who, crate::PrincipalKind::User))
        .access_policy(Arc::new(MonitorForAdmin))
        .config("spark.sql.warehouse.dir", warehouse.to_str().unwrap())
        .build()
        .unwrap()
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
    assert_eq!(text(&admin, table), "2");
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
    // The scans did reach the history (so the empty advice is a judgement, not a missing record).
    let seen: usize = read_history(history.path())
        .unwrap()
        .iter()
        .filter_map(|r| QueryProfile::from_json(&r.profile).ok())
        .flat_map(|p| p.scans)
        .filter(|s| {
            s.table.to_lowercase().ends_with("ev")
                && s.equality.iter().any(|c| c.name.eq_ignore_ascii_case("id"))
        })
        .count();
    assert_eq!(seen, 50);
    let advice = from_history(history.path());
    let suggestions: Vec<_> = advice.iter().flat_map(|t| t.advice.iter()).collect();
    assert!(suggestions.is_empty(), "{advice:?}");
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

fn scan(table: &str, rows: Option<u64>, columns: &[(&str, Option<u64>)]) -> Vec<ProfileScan> {
    vec![ProfileScan {
        table: table.to_owned(),
        rows,
        equality: columns
            .iter()
            .map(|(name, distinct)| ScanColumn {
                name: (*name).to_owned(),
                distinct: *distinct,
            })
            .collect(),
    }]
}

/// A distinct count is weighed against the row count of the query it was planned with: 60 distinct of 100 rows is
/// a key, and a later 10,000 rows with no statistics for `d` does not make it a partition column.
#[test]
fn a_distinct_count_is_not_weighed_against_a_later_row_count() {
    let advice = advice_of([
        scan("c.s.t", Some(100), &[("d", Some(60))]),
        scan("c.s.t", Some(10_000), &[("d", None)]),
    ]);
    assert_eq!(advice, Vec::<TableAdvice>::new());
    // Statistics of one snapshot still count every query that filtered on the column.
    let advice = advice_of([
        scan("c.s.t", Some(10_000), &[("d", None)]),
        scan("c.s.t", Some(10_000), &[("d", Some(10))]),
    ]);
    assert_eq!(advice.len(), 1, "{advice:?}");
    assert_eq!(advice[0].table, "c.s.t");
    assert_eq!(advice[0].queries, 2);
    assert_eq!(
        (
            advice[0].advice[0].column.as_str(),
            advice[0].advice[0].queries,
            advice[0].advice[0].distinct
        ),
        ("d", 2, 10)
    );
}

/// The history's table names and `EXPLAIN`'s scanned tables match case-insensitively, Unicode included (`Ä`/`ä`),
/// as ASCII (`EV`/`ev`) always did.
#[test]
fn advice_matches_a_table_name_in_any_case_unicode_included() {
    let advice = advice_of((0..4).map(|i| {
        scan(
            if i % 2 == 0 {
                "c.s.`\u{c4}`"
            } else {
                "c.s.`\u{e4}`"
            },
            Some(1000),
            &[("d", Some(10))],
        )
    }));
    assert_eq!(advice.len(), 1, "{advice:?}");
    assert_eq!(advice[0].table, "c.s.`\u{e4}`");
    assert_eq!(advice[0].queries, 4);
    let line = "c.s.`\u{e4}`: partition advice: d (4 queries, 10 partitions, ~90% of bytes pruned)";
    let want = Some(format!("== Partition Advice ==\n{line}\n"));
    for scanned in ["c.s.`\u{c4}`", "c.s.`\u{e4}`", "C.S.`\u{c4}`"] {
        assert_eq!(
            section_of(&advice, &[scanned.to_owned()]),
            want,
            "{scanned}"
        );
    }
    let ev = advice_of((0..2).map(|_| scan("c.s.EV", Some(1000), &[("d", Some(10))])));
    assert_eq!(ev[0].table, "c.s.ev");
    assert!(section_of(&ev, &["c.s.ev".to_owned()]).is_some());
    assert!(section_of(&ev, &["C.S.Ev".to_owned()]).is_some());
    assert_eq!(section_of(&ev, &["c.s.other".to_owned()]), None);
}

/// A runtime with a history, and sessions under one `grants` provider (`system` allowed to all, so
/// every principal holds `MONITOR`): root (admin) owns `ev`, ana may read it and owns the definer
/// view `dv` over it, bob may read `dv` only.
struct DefinerWorld {
    rt: Arc<EngineRuntime>,
    grants: Arc<crate::grants::Grants>,
    warehouse: tempfile::TempDir,
}

impl DefinerWorld {
    fn new(history: &std::path::Path) -> Self {
        let rt = EngineRuntime::builder()
            .config("krishiv.history.path", history.to_str().unwrap())
            .build()
            .unwrap();
        let w = Self {
            rt,
            grants: crate::grants::Grants::new(Arc::new(crate::grants::MemoryGrantStore::new())),
            warehouse: tempfile::tempdir().unwrap(),
        };
        let (root, ana) = (w.session("root", true), w.session("ana", false));
        run(&root, &format!("CREATE TABLE ev USING parquet AS {EVENTS}"));
        run(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
        run(&root, "GRANT SELECT ON TABLE ev TO user:ana");
        run(
            &ana,
            "CREATE VIEW dv SQL SECURITY DEFINER AS SELECT id, d, c FROM ev",
        );
        run(&ana, "GRANT SELECT ON VIEW dv TO user:bob");
        w
    }

    fn session(&self, who: &str, admin: bool) -> Session {
        let policy = Arc::new(crate::grants::GrantsPolicy::new(
            Arc::clone(&self.grants),
            Arc::new(crate::AllowAll),
            admin,
        ));
        Session::builder()
            .runtime(Arc::clone(&self.rt))
            .principal(crate::Principal::new(who, crate::PrincipalKind::User))
            .access_policy(Arc::clone(&policy) as Arc<dyn crate::AccessPolicy>)
            .grants(policy)
            .config(
                "spark.sql.warehouse.dir",
                self.warehouse.path().to_str().unwrap(),
            )
            .build()
            .unwrap()
    }
}

/// #868 merge review N1: a query through a definer view whose reads its reader may not see keeps
/// no persisted profile, so its scans never reach the history, the advisor or
/// `system.partition_advice`.
#[test]
fn a_definer_views_withheld_reads_leave_nothing_in_the_history() {
    let history = tempfile::tempdir().unwrap();
    let w = DefinerWorld::new(history.path());
    let (root, bob) = (w.session("root", true), w.session("bob", false));
    workload(&bob, "dv");
    flush(&w.rt);
    let stored = |needle: &str| {
        read_history(history.path())
            .unwrap()
            .iter()
            .filter(|r| r.sql.as_deref().is_some_and(|s| s.contains(needle)))
            .count()
    };
    assert_eq!(stored("FROM dv"), 0, "bob's queries were stored");
    assert!(from_history(history.path()).is_empty());
    let table = "SELECT CAST(count(*) AS STRING) FROM system.partition_advice";
    assert_eq!(text(&root, table), "0");
    // The definer, who may see the reads, has her queries stored.
    let ana = w.session("ana", false);
    text(&ana, "SELECT CAST(sum(id) AS STRING) FROM dv WHERE d = 3");
    flush(&w.rt);
    assert_eq!(stored("FROM dv"), 1);
}

/// #868 merge review N2: a `MONITOR` holder's `EXPLAIN` of a definer view whose reads it may not
/// see shows no `== Partition Advice ==` section, though advice for the view's table exists.
#[test]
fn explain_shows_no_advice_when_the_definer_reads_are_withheld() {
    let history = tempfile::tempdir().unwrap();
    let w = DefinerWorld::new(history.path());
    let root = w.session("root", true);
    workload(&root, "ev");
    flush(&w.rt);
    let query = "EXPLAIN SELECT sum(id) FROM dv WHERE d = 7";
    // Whoever may see the view's reads sees the advice for `ev` through it.
    assert!(text(&root, query).contains("partition advice: d"));
    // Bob may read `dv` but not see its definer's reads: EXPLAIN shows him a plan, and no advice
    // for the table behind the view.
    let bob = w.session("bob", false);
    let batches = block_on(async { bob.sql(query).await?.collect().await })
        .unwrap_or_else(|e| panic!("bob's EXPLAIN is a plan, not a refusal: {e}"));
    let plan = batches
        .iter()
        .flat_map(|b| (0..b.num_rows()).map(|i| cell(b.column(0).as_ref(), i)))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!plan.is_empty());
    assert!(!plan.contains("Partition Advice"), "{plan}");
    assert!(!plan.contains("partition advice"), "{plan}");
}
