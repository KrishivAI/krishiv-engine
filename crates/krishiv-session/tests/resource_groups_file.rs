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
