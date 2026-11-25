// SPDX-License-Identifier: Apache-2.0
//! `krishiv.workload.resourceGroups.file` installs the group controller for every session of a
//! runtime (P3-T003): the limits hold across sessions, and a bad file stops the start.
#![cfg(test)]

use krishiv_common::error::ErrorClass;
use krishiv_session::{EngineRuntime, Session};

fn write_file(name: &str, json: &str) -> String {
    let dir = std::env::temp_dir().join(format!("krishiv-rg-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
    let path = dir.join("resource-groups.json");
    std::fs::write(&path, json).unwrap_or_else(|e| panic!("{e}"));
    path.to_string_lossy().into_owned()
}

const ONE_SLOT: &str = r#"{
  "rootGroups": [{"name": "root", "hardConcurrencyLimit": 1, "maxQueued": 0}],
  "defaultGroup": "root"
}"#;

#[test]
fn a_missing_group_file_stops_the_runtime() {
    let err = EngineRuntime::builder()
        .config(
            "krishiv.workload.resourceGroups.file",
            "/no/such/resource-groups.json",
        )
        .build()
        .err()
        .unwrap_or_else(|| panic!("a missing file must fail"));
    assert_eq!(
        err.error_class().name(),
        ErrorClass::KrishivResourceGroupConfigInvalid.name()
    );
}

#[test]
fn a_malformed_group_file_stops_the_runtime_naming_the_key() {
    let path = write_file("bad", r#"{"rootGroups": [], "cpuQuota": 1}"#);
    let err = EngineRuntime::builder()
        .config("krishiv.workload.resourceGroups.file", path)
        .build()
        .err()
        .unwrap_or_else(|| panic!("a bad file must fail"));
    assert_eq!(
        err.error_class().name(),
        ErrorClass::KrishivResourceGroupConfigInvalid.name()
    );
}

#[tokio::test]
async fn the_limit_holds_across_the_sessions_of_a_runtime() {
    let path = write_file("one", ONE_SLOT);
    let rt = EngineRuntime::builder()
        .config("krishiv.workload.resourceGroups.file", path)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let a = Session::builder()
        .runtime(std::sync::Arc::clone(&rt))
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let b = Session::builder()
        .runtime(rt)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    // Session `a` holds the only slot while its stream is open; `b` finds the queue full.
    let held = a
        .sql("select 1")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .execute_stream()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let err = b
        .sql("select 2")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .err()
        .unwrap_or_else(|| panic!("the second session must be refused"));
    assert_eq!(
        err.error_class().name(),
        ErrorClass::KrishivResourceGroupQueueFull.name()
    );
    drop(held);
    // The slot is given back when the dropped stream's query folds; allow it a moment under load.
    let mut last = None;
    for _ in 0..100 {
        match b
            .sql("select 2")
            .await
            .unwrap_or_else(|e| panic!("{e}"))
            .collect()
            .await
        {
            Ok(_) => return,
            Err(e) => last = Some(e),
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("the slot never came back: {last:?}");
}

#[tokio::test]
async fn system_queries_shows_the_group_and_the_queued_time() {
    use arrow_array::cast::AsArray;
    let path = write_file("rows", ONE_SLOT);
    let rt = EngineRuntime::builder()
        .config("krishiv.workload.resourceGroups.file", path)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let s = Session::builder()
        .runtime(rt)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    s.sql("select 1")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let batches = s
        .sql(
            "select resource_group, queued_ns from system.queries where resource_group is not null",
        )
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let rows: usize = batches.iter().map(arrow_array::RecordBatch::num_rows).sum();
    assert_eq!(rows, 1, "one admitted query has a group");
    let b = &batches[0];
    assert_eq!(b.column(0).as_string::<i32>().value(0), "root");
    assert!(
        b.column(1)
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(0)
            >= 0
    );
}

#[tokio::test]
async fn the_group_sets_the_workload_class() {
    use arrow_array::cast::AsArray;
    let path = write_file(
        "class",
        r#"{
  "rootGroups": [{"name": "root", "hardConcurrencyLimit": 4, "maxQueued": 4, "priority": 7,
                  "workloadClass": "background"}],
  "defaultGroup": "root"
}"#,
    );
    let rt = EngineRuntime::builder()
        .config("krishiv.workload.resourceGroups.file", path)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let s = Session::builder()
        .runtime(rt)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    s.sql("select 1")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let batches = s
        .sql("select workload_class from system.queries where resource_group is not null")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        batches[0].column(0).as_string::<i32>().value(0),
        "background"
    );
}

#[test]
fn the_shipped_example_tree_loads() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/examples/resource-groups.json"
    );
    krishiv_session::resource_groups::ResourceGroups::from_file(path, 8 << 30)
        .unwrap_or_else(|e| panic!("{e}"));
}

#[tokio::test]
async fn a_queued_query_shows_its_wait_in_queued_ns() {
    use arrow_array::cast::AsArray;
    let path = write_file(
        "queued",
        r#"{"rootGroups": [{"name": "root", "hardConcurrencyLimit": 1, "maxQueued": 2}],
            "defaultGroup": "root"}"#,
    );
    let rt = EngineRuntime::builder()
        .config("krishiv.workload.resourceGroups.file", path)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let a = Session::builder()
        .runtime(std::sync::Arc::clone(&rt))
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let b = Session::builder()
        .runtime(rt)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let held = a
        .sql("select 1")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .execute_stream()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let b2 = b.clone();
    let waiter = tokio::spawn(async move {
        b2.sql("select 2")
            .await
            .unwrap_or_else(|e| panic!("{e}"))
            .collect()
            .await
    });
    // Wait until the second query shows as queued, then hold it for 200 ms,
    // so a slow plan cannot shorten the wait the test measures.
    let queued_seen = std::time::Instant::now();
    loop {
        let rows = a
            .sql("select count(*) from system.queries where state = 'queued'")
            .await
            .unwrap_or_else(|e| panic!("{e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        let n = rows[0]
            .column(0)
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(0);
        if n > 0 {
            break;
        }
        assert!(
            queued_seen.elapsed() < std::time::Duration::from_secs(30),
            "the second query never showed as queued"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    drop(held);
    waiter
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .unwrap_or_else(|e| panic!("{e}"));
    let batches = b
        .sql("select max(queued_ns) from system.queries where resource_group is not null")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let waited = batches[0]
        .column(0)
        .as_primitive::<arrow_array::types::Int64Type>()
        .value(0);
    assert!(
        waited >= 100_000_000,
        "queued for about 200 ms, saw {waited} ns"
    );
}

#[tokio::test]
async fn a_spent_scan_quota_refuses_the_next_query() {
    let data = std::env::temp_dir().join(format!("krishiv-rg-{}-scan.csv", std::process::id()));
    let rows: String = (0..2000).map(|i| format!("{i},row-{i}\n")).collect();
    std::fs::write(&data, rows).unwrap_or_else(|e| panic!("{e}"));
    let path = write_file(
        "quota",
        r#"{"rootGroups": [{"name": "root", "hardConcurrencyLimit": 4, "maxQueued": 4,
                           "scanQuota": 1, "scanQuotaPeriod": "1h"}],
            "defaultGroup": "root"}"#,
    );
    let rt = EngineRuntime::builder()
        .config("krishiv.workload.resourceGroups.file", path)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let s = Session::builder()
        .runtime(rt)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let sql = format!("select count(*) from csv.`{}`", data.display());
    s.sql(&sql)
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("the first scan is admitted: {e}"));
    // The slot's drop charges the scanned bytes; allow the fold a moment.
    let mut refused = None;
    for _ in 0..100 {
        match s
            .sql("select 1")
            .await
            .unwrap_or_else(|e| panic!("{e}"))
            .collect()
            .await
        {
            Err(e) => {
                refused = Some(e);
                break;
            }
            Ok(_) => tokio::time::sleep(std::time::Duration::from_millis(50)).await,
        }
    }
    let err = refused.unwrap_or_else(|| panic!("the spent quota never refused a query"));
    assert_eq!(
        err.error_class().name(),
        ErrorClass::KrishivResourceGroupQuotaExhausted.name()
    );
}

#[tokio::test]
async fn system_resource_groups_lists_the_tree_with_its_counters() {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    let path = write_file(
        "table",
        r#"{"rootGroups": [{"name": "root", "hardConcurrencyLimit": 4, "maxQueued": 8, "priority": 60,
              "subGroups": [{"name": "etl", "hardConcurrencyLimit": 2, "maxQueued": 4,
                             "workloadClass": "batch", "scanQuota": 1000000, "scanQuotaPeriod": "1h"}]}],
            "defaultGroup": "root.etl"}"#,
    );
    let rt = EngineRuntime::builder()
        .config("krishiv.workload.resourceGroups.file", path)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let s = Session::builder()
        .runtime(rt)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let held = s
        .sql("select 1")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .execute_stream()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let batches = s
        .sql(
            "select group_id, parent_id, state, workload_class, hard_concurrency_limit, running, \
             scan_quota_bytes from system.resource_groups order by group_id",
        )
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let b = &batches[0];
    assert_eq!(b.num_rows(), 2);
    let ids = b.column(0).as_string::<i32>();
    assert_eq!((ids.value(0), ids.value(1)), ("root", "root.etl"));
    assert!(b.column(1).is_null(0), "the root has no parent");
    assert_eq!(b.column(1).as_string::<i32>().value(1), "root");
    assert_eq!(b.column(2).as_string::<i32>().value(0), "active");
    assert_eq!(b.column(3).as_string::<i32>().value(1), "batch");
    let int = |i: usize, r: usize| {
        b.column(i)
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(r)
    };
    assert_eq!((int(4, 0), int(4, 1)), (4, 2));
    assert_eq!(
        (int(5, 0), int(5, 1)),
        (1, 1),
        "the held query counts up to the root"
    );
    assert!(b.column(6).is_null(0));
    assert_eq!(
        b.column(6)
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(1),
        1_000_000
    );
    drop(held);
}

const ROUTED: &str = r#"{
  "rootGroups": [{"name": "root", "hardConcurrencyLimit": 8, "maxQueued": 8,
    "subGroups": [
      {"name": "agent", "hardConcurrencyLimit": 4, "maxQueued": 4},
      {"name": "writes", "hardConcurrencyLimit": 4, "maxQueued": 4},
      {"name": "rest", "hardConcurrencyLimit": 4, "maxQueued": 4}]}],
  "selectors": [
    {"source": "krishiv-agent.*", "group": "root.agent"},
    {"queryType": "DDL", "group": "root.writes"}],
  "defaultGroup": "root.rest"
}"#;

/// The group of the one `system.queries` row whose text starts with `prefix`.
async fn group_of(s: &Session, prefix: &str) -> Option<String> {
    use arrow_array::cast::AsArray;
    let batches = s
        .sql(&format!(
            "select resource_group from system.queries where query_text like '{prefix}%' \
             and resource_group is not null"
        ))
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    batches
        .iter()
        .find(|b| b.num_rows() > 0)
        .map(|b| b.column(0).as_string::<i32>().value(0).to_owned())
}

#[tokio::test]
async fn the_source_and_the_statement_class_route_a_query() {
    let path = write_file("routed", ROUTED);
    let rt = EngineRuntime::builder()
        .config("krishiv.workload.resourceGroups.file", path)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let agent = Session::builder()
        .runtime(std::sync::Arc::clone(&rt))
        .source("krishiv-agent-7")
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let plain = Session::builder()
        .runtime(rt)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    for (s, text) in [(&agent, "select 11 as a"), (&plain, "select 12 as b")] {
        s.sql(text)
            .await
            .unwrap_or_else(|e| panic!("{e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("{e}"));
    }
    plain
        .sql("create table routed_t (a int) using parquet")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        group_of(&agent, "select 11").await.as_deref(),
        Some("root.agent")
    );
    assert_eq!(
        group_of(&plain, "select 12").await.as_deref(),
        Some("root.rest")
    );
    assert_eq!(
        group_of(&plain, "create table routed_t").await.as_deref(),
        Some("root.writes")
    );
}

#[tokio::test]
async fn a_front_end_client_type_routes_and_does_not_replace_the_builder_source() {
    let path = write_file("routed-front-end", ROUTED);
    let rt = EngineRuntime::builder()
        .config("krishiv.workload.resourceGroups.file", path)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    // No builder source: the Connect / Flight SQL path sets the client type afterwards.
    let connect = Session::builder()
        .runtime(std::sync::Arc::clone(&rt))
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    connect.set_client_info(
        krishiv_session::Transport::Connect,
        Some("krishiv-agent-1"),
        None,
        None,
    );
    // A builder source stays when a front end reports another type later.
    let built = Session::builder()
        .runtime(rt)
        .source("krishiv-agent-2")
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    built.set_client_info(
        krishiv_session::Transport::Connect,
        Some("other"),
        None,
        None,
    );
    for (s, text) in [(&connect, "select 21 as a"), (&built, "select 22 as b")] {
        s.sql(text)
            .await
            .unwrap_or_else(|e| panic!("{e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("{e}"));
    }
    assert_eq!(
        group_of(&connect, "select 21").await.as_deref(),
        Some("root.agent")
    );
    assert_eq!(
        group_of(&built, "select 22").await.as_deref(),
        Some("root.agent")
    );
}

const BY_TYPE: &str = r#"{
  "rootGroups": [{"name": "root", "hardConcurrencyLimit": 8, "maxQueued": 8,
    "subGroups": [
      {"name": "reads", "hardConcurrencyLimit": 4, "maxQueued": 4},
      {"name": "writes", "hardConcurrencyLimit": 4, "maxQueued": 4},
      {"name": "rest", "hardConcurrencyLimit": 4, "maxQueued": 4}]}],
  "selectors": [
    {"queryType": "READ", "group": "root.reads"},
    {"queryType": "WRITE", "group": "root.writes"}],
  "defaultGroup": "root.rest"
}"#;

#[tokio::test]
async fn a_dataframe_read_and_write_match_the_query_type_selectors() {
    use arrow_array::cast::AsArray;
    let path = write_file("by-type", BY_TYPE);
    let rt = EngineRuntime::builder()
        .config("krishiv.workload.resourceGroups.file", path)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let s = Session::builder()
        .runtime(rt)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
    let dir = tmp.path().to_path_buf();
    let (src, out) = (dir.join("src"), dir.join("out"));
    s.sql("select 1 as a")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .write()
        .parquet(src.to_str().unwrap_or_default())
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    // No SQL text started these two: the entry point names the class.
    let df = s.read().parquet([src.to_str().unwrap_or_default()]);
    df.collect().await.unwrap_or_else(|e| panic!("{e}"));
    df.write()
        .parquet(out.to_str().unwrap_or_default())
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    // A write of a frame a SQL `SELECT` started is a write too.
    s.sql("select 1 as a")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .write()
        .parquet(dir.join("out2").to_str().unwrap_or_default())
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let batches = s
        .sql("select resource_group from system.queries where query_text is null")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let mut groups: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            let c = b.column(0).as_string::<i32>();
            (0..b.num_rows())
                .map(|i| c.value(i).to_owned())
                .collect::<Vec<_>>()
        })
        .collect();
    groups.sort();
    assert_eq!(groups, ["root.reads", "root.writes"]);
    // Both writes of `select 1 as a` ran in the writes group, whatever the frame's SQL class.
    let batches = s
        .sql("select resource_group from system.queries where query_text is not null and resource_group = 'root.writes'")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let rows: usize = batches.iter().map(arrow_array::RecordBatch::num_rows).sum();
    assert_eq!(rows, 2, "the select-born writes: src and out2");
}

#[tokio::test]
async fn system_resource_groups_rows_name_the_document_they_came_from() {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    let path = write_file(
        "etag_columns",
        r#"{"rootGroups": [{"name": "root", "hardConcurrencyLimit": 4, "maxQueued": 8,
              "subGroups": [{"name": "etl", "hardConcurrencyLimit": 2, "maxQueued": 4}]}],
            "defaultGroup": "root.etl"}"#,
    );
    let rt = EngineRuntime::builder()
        .config("krishiv.workload.resourceGroups.file", path)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let s = Session::builder()
        .runtime(rt)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let batches = s
        .sql("select config_etag, loaded_at from system.resource_groups order by group_id")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let b = &batches[0];
    assert_eq!(b.num_rows(), 2);
    let etags = b.column(0).as_string::<i32>();
    assert_eq!(etags.value(0), etags.value(1), "one document, one etag");
    assert_eq!(etags.value(0).len(), 64, "a SHA-256 in hex");
    assert!(etags.value(0).bytes().all(|c| c.is_ascii_hexdigit()));
    assert!(!b.column(1).is_null(0) && !b.column(1).is_null(1));
}

#[tokio::test]
async fn a_viewer_sees_only_the_groups_holding_its_own_queries() {
    use arrow_array::cast::AsArray;
    use krishiv_common::error::KrishivError;
    use krishiv_plan::analyzer::{Privilege, Securable};
    use krishiv_session::{
        AccessDecision, AccessPolicy, BoxFuture, PolicyVersion, Principal, PrincipalKind,
    };
    /// Allows everything except `Monitor`: a caller who reads the table but sees no other tenant's
    /// work. (The viewer role's table list is the ADR 0046 matrix; it is not widened here.)
    struct NoMonitor;
    impl AccessPolicy for NoMonitor {
        fn version(&self) -> PolicyVersion {
            PolicyVersion(1)
        }
        fn check<'a>(
            &'a self,
            _: &'a Principal,
            _: &'a Securable,
            privilege: Privilege,
        ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
            let decision = if privilege == Privilege::Monitor {
                AccessDecision::Deny {
                    reason: "test".to_owned(),
                }
            } else {
                AccessDecision::Allow
            };
            Box::pin(std::future::ready(Ok(decision)))
        }
    }
    let path = write_file(
        "own_groups",
        r#"{"rootGroups": [{"name": "root", "hardConcurrencyLimit": 8, "maxQueued": 8,
              "subGroups": [{"name": "etl", "hardConcurrencyLimit": 4, "maxQueued": 4,
                             "subGroups": [{"name": "nightly", "hardConcurrencyLimit": 2, "maxQueued": 4}]},
                            {"name": "other", "hardConcurrencyLimit": 2, "maxQueued": 4}]}],
            "defaultGroup": "root.etl.nightly"}"#,
    );
    let rt = EngineRuntime::builder()
        .config("krishiv.workload.resourceGroups.file", path)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let admin = Session::builder()
        .runtime(std::sync::Arc::clone(&rt))
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    // Another tenant's query holds the same leaf for the whole test.
    let _held = admin
        .sql("select 1")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .execute_stream()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let viewer = Session::builder()
        .runtime(std::sync::Arc::clone(&rt))
        .principal(Principal::new("viewer", PrincipalKind::User))
        .access_policy(std::sync::Arc::new(NoMonitor))
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    // The viewer's own running query: the table lists what the caller holds when the scan is
    // planned (a query takes its group when it starts), so it asks from a second statement.
    let _own = viewer
        .sql("select 1")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .execute_stream()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let batches = viewer
        .sql("select group_id, running from system.resource_groups order by group_id")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let b = &batches[0];
    let ids = b.column(0).as_string::<i32>();
    let got: Vec<&str> = (0..b.num_rows()).map(|i| ids.value(i)).collect();
    assert_eq!(
        got,
        ["root", "root.etl", "root.etl.nightly"],
        "its group and the ancestors, not root.other"
    );
    let running = b.column(1);
    assert!(
        running.is_null(0) && running.is_null(1),
        "totals over other tenants are null"
    );
    assert_eq!(
        running
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(2),
        2,
        "its own group keeps the counter: the other tenant's query and its own held one"
    );
}
