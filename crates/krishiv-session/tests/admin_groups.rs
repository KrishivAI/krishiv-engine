// SPDX-License-Identifier: Apache-2.0
//! What the admin API reads of the resource groups (`EngineRuntime::resource_group_infos` and the
//! document calls, design P3-T066 §5.4).
#![cfg(test)]

use krishiv_plan::analyzer::{Principal, PrincipalKind};
use krishiv_session::EngineRuntime;

const DOC: &str = r#"{
  "rootGroups": [{"name": "root", "hardConcurrencyLimit": 2, "maxQueued": 3,
    "subGroups": [{"name": "etl", "hardConcurrencyLimit": 1, "maxQueued": 1}]}],
  "defaultGroup": "root.etl"
}"#;

fn runtime(dir: &tempfile::TempDir) -> std::sync::Arc<EngineRuntime> {
    let path = dir.path().join("groups.json");
    std::fs::write(&path, DOC).unwrap();
    EngineRuntime::builder()
        .config(
            "krishiv.workload.resourceGroups.file",
            path.to_str().unwrap(),
        )
        .build()
        .unwrap()
}

#[test]
fn monitor_sees_every_group_with_counters_and_a_caller_with_no_query_sees_none() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(&dir);
    let alice = Principal::new("alice", PrincipalKind::User);
    let all = rt.resource_group_infos(&alice, true);
    let ids: Vec<&str> = all.iter().map(|g| g.group_id.as_str()).collect();
    assert_eq!(ids, ["root", "root.etl"]);
    assert!(
        all.iter()
            .all(|g| g.running == Some(0) && g.queued == Some(0))
    );
    assert_eq!(all[1].parent_id.as_deref(), Some("root"));
    assert_eq!(all[1].hard_concurrency_limit, 1);
    assert!(rt.resource_group_infos(&alice, false).is_empty());
}

#[test]
fn the_document_validate_replace_and_reload_follow_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(&dir);
    let doc = rt.resource_group_document();
    assert_eq!(doc.etag.len(), 64);
    assert_eq!(doc.file_etag.as_deref(), Some(doc.etag.as_str()));
    assert!(rt.validate_resource_group_document(&doc.document).is_ok());
    assert!(rt.validate_resource_group_document("{").is_err());
    let next = doc
        .document
        .replace("\"hardConcurrencyLimit\":2", "\"hardConcurrencyLimit\":5");
    assert_ne!(next, doc.document);
    let err = rt
        .replace_resource_group_document(&next, "stale")
        .unwrap_err();
    assert_eq!(
        err.error_class().name(),
        "KRISHIV_ADMIN_PRECONDITION_FAILED"
    );
    rt.replace_resource_group_document(&next, &doc.etag)
        .unwrap();
    assert_ne!(rt.resource_group_document().etag, doc.etag);
    std::fs::write(dir.path().join("groups.json"), DOC).unwrap();
    rt.reload_resource_group_document().unwrap();
    assert_eq!(rt.resource_group_document().etag, doc.etag);
}

#[test]
fn without_a_group_file_the_document_is_empty_and_changes_are_read_only() {
    let rt = EngineRuntime::builder().build().unwrap();
    let alice = Principal::new("alice", PrincipalKind::User);
    assert!(rt.resource_group_infos(&alice, true).is_empty());
    assert!(rt.resource_group_document().document.is_empty());
    for err in [
        rt.replace_resource_group_document("{}", "").unwrap_err(),
        rt.reload_resource_group_document().unwrap_err(),
    ] {
        assert_eq!(
            err.error_class().name(),
            "KRISHIV_RESOURCE_GROUP_CONFIG_READ_ONLY"
        );
    }
}

/// A session of `name` on `rt`.
fn session_of(rt: &std::sync::Arc<EngineRuntime>, name: &str) -> krishiv_session::Session {
    krishiv_session::Session::builder()
        .runtime(std::sync::Arc::clone(rt))
        .principal(Principal::new(name, PrincipalKind::User))
        .build()
        .unwrap()
}

#[tokio::test]
async fn a_caller_sees_the_group_of_its_running_and_of_its_queued_query() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(&dir);
    let alice = session_of(&rt, "alice");
    let bob = session_of(&rt, "bob");
    let nobody = Principal::new("carol", PrincipalKind::User);
    // alice takes the one slot of root.etl; bob's query waits behind it.
    let held = alice
        .sql("select 1")
        .await
        .unwrap()
        .execute_stream()
        .await
        .unwrap();
    let waiting = {
        let bob = bob.clone();
        tokio::spawn(async move { bob.sql("select 2").await.unwrap().collect().await })
    };
    // Wait until bob's query holds its place in the queue (a fixed sleep flakes on a loaded machine).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while !rt.query_infos().iter().any(|q| q.state == "queued") {
        assert!(
            std::time::Instant::now() < deadline,
            "bob's query never queued"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    let alice_p = Principal::new("alice", PrincipalKind::User);
    let own = rt.resource_group_infos(&alice_p, false);
    let ids: Vec<&str> = own.iter().map(|g| g.group_id.as_str()).collect();
    assert_eq!(ids, ["root", "root.etl"], "the group and its ancestor");
    assert_eq!(
        own[0].running, None,
        "an ancestor shows no subtree counters"
    );
    assert_eq!(own[1].running, Some(1));
    assert_eq!(own[1].queued, Some(1), "bob's query is queued in it");

    // bob holds no slot, only a place in the queue: the group is still his to see.
    let bob_p = Principal::new("bob", PrincipalKind::User);
    let queued = rt.resource_group_infos(&bob_p, false);
    assert_eq!(
        queued
            .iter()
            .map(|g| g.group_id.as_str())
            .collect::<Vec<_>>(),
        ["root", "root.etl"],
        "a queued query shows its group"
    );
    assert!(
        rt.resource_group_infos(&nobody, false).is_empty(),
        "another principal's groups are not listed"
    );
    drop(held);
    waiting.await.unwrap().unwrap();
}
