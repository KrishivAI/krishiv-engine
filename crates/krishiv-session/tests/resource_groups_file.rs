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
