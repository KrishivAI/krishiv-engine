// SPDX-License-Identifier: Apache-2.0
//! The server console's Sessions and Workload pages over a real runtime: `SessionSource` lists the sessions
//! of every client and the resource groups, and ends a session as an operator's kill.
#![cfg(feature = "session")]
#![expect(
    clippy::unwrap_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use krishiv_common::principal::{Principal, PrincipalKind};
use krishiv_server::guard::Caller;
use krishiv_server::internal::{Row, SystemTable, UiSource};
use krishiv_server::session_source::{SessionSource, UI_PRINCIPAL};
use krishiv_session::auth::{Authenticated, Method, Role};
use krishiv_session::{EngineRuntime, Session};
use serde_json::Value;

/// A cross join no machine finishes soon.
const ENDLESS: &str =
    "SELECT count(*) FROM range(0, 1000000000) a CROSS JOIN range(0, 1000000000) b";

fn session_as(rt: &Arc<EngineRuntime>, name: &str) -> Session {
    Session::builder()
        .runtime(Arc::clone(rt))
        .principal(Principal::new(name, PrincipalKind::User))
        .build()
        .unwrap()
}

async fn sessions(source: &SessionSource) -> Vec<Row> {
    source.rows(SystemTable::Sessions).await.unwrap()
}

fn row_of<'a>(rows: &'a [Row], id: &str) -> &'a Row {
    let row = rows.iter().find(|r| r["session_id"] == id);
    assert!(row.is_some(), "no session {id} in {rows:?}");
    row.unwrap()
}

/// The editor run `id` of `caller` once it has left `running`.
async fn ended(source: &SessionSource, id: &str, caller: Option<&Caller>) -> Value {
    let start = Instant::now();
    loop {
        let v = source.sql_status(id, caller).await.unwrap().unwrap();
        if v["state"] != "running" {
            return v;
        }
        assert!(start.elapsed() < Duration::from_secs(30), "{v}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn the_sessions_of_every_client_are_listed_and_the_consoles_own_is_not() {
    let rt = EngineRuntime::builder().build().unwrap();
    let source = SessionSource::new(Arc::clone(&rt)).unwrap();
    let ann = session_as(&rt, "ann");
    let bob = session_as(&rt, "bob");
    ann.sql("SELECT 1").await.unwrap().collect().await.unwrap();

    let rows = sessions(&source).await;
    let of_ann = row_of(&rows, ann.id());
    assert_eq!(of_ann["principal"], "ann");
    assert_eq!(of_ann["state"], "active");
    assert_eq!(of_ann["queries_started"], 1, "{of_ann:?}");
    assert_eq!(row_of(&rows, bob.id())["queries_started"], 0);
    assert!(
        rows.iter().all(|r| r["principal"] != UI_PRINCIPAL),
        "the session the console reads through is not a client's: {rows:?}"
    );
    // Reading the list is not a client's query either.
    assert_eq!(
        row_of(&sessions(&source).await, ann.id())["queries_started"],
        1
    );
}

#[tokio::test]
async fn ending_a_session_kills_its_queries_and_refuses_its_next_statement() {
    let rt = EngineRuntime::builder().build().unwrap();
    let source = SessionSource::new(Arc::clone(&rt)).unwrap();
    let ann = session_as(&rt, "ann");
    let bob = session_as(&rt, "bob");
    let running = {
        let ann = ann.clone();
        tokio::spawn(async move { ann.sql(ENDLESS).await?.collect().await })
    };
    let start = Instant::now();
    while row_of(&sessions(&source).await, ann.id())["queries_running"] != 1 {
        assert!(start.elapsed() < Duration::from_secs(20), "never ran");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    assert_eq!(source.close_session(ann.id()).await.unwrap(), Some(1));
    let failed = tokio::time::timeout(Duration::from_secs(20), running)
        .await
        .expect("the query ends once its session does")
        .unwrap()
        .unwrap_err();
    assert_eq!(
        failed.error_class().name(),
        "KRISHIV_QUERY_KILLED",
        "{failed}"
    );
    let refused = ann.sql("SELECT 1").await.unwrap_err();
    assert_eq!(
        refused.error_class().name(),
        "INVALID_HANDLE.SESSION_CLOSED",
        "{refused}"
    );

    let rows = sessions(&source).await;
    let of_ann = row_of(&rows, ann.id());
    assert_eq!(of_ann["state"], "killed", "{of_ann:?}");
    assert_eq!(of_ann["close_reason"], "admin_kill", "{of_ann:?}");
    // Another client's session is untouched, and an ended or unknown session is not ended again.
    assert_eq!(row_of(&rows, bob.id())["state"], "active");
    bob.sql("SELECT 1").await.unwrap().collect().await.unwrap();
    assert_eq!(source.close_session(ann.id()).await.unwrap(), None);
    assert_eq!(source.close_session("no-such-session").await.unwrap(), None);
}

#[tokio::test]
async fn the_sessions_the_console_itself_runs_in_cannot_be_ended() {
    let rt = EngineRuntime::builder().build().unwrap();
    let source = SessionSource::new(Arc::clone(&rt)).unwrap();
    // The one client session listed before any client connects is the editor's shared one.
    let rows = sessions(&source).await;
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0]["endable"], false, "{rows:?}");
    let shared = rows[0]["session_id"].as_str().unwrap().to_owned();
    assert_eq!(source.close_session(&shared).await.unwrap(), None);
    let id = source.sql_start("SELECT 7", None).await.unwrap();
    assert_eq!(ended(&source, &id, None).await["state"], "finished");

    // The session the console reads through, found from another session of the runtime.
    let other = session_as(&rt, "ann");
    let sql = format!("SELECT session_id FROM system.sessions WHERE principal = '{UI_PRINCIPAL}'");
    let batches = other.sql(&sql).await.unwrap().collect().await.unwrap();
    let own = krishiv_server::session_source::batches_to_rows(&batches);
    assert_eq!(own.len(), 1, "{own:?}");
    let own = own[0]["session_id"].as_str().unwrap();
    assert_eq!(source.close_session(own).await.unwrap(), None);
    assert!(
        !sessions(&source).await.is_empty(),
        "the console still reads"
    );
}

#[tokio::test]
async fn an_editor_caller_whose_session_was_ended_gets_a_new_one() {
    let rt = EngineRuntime::builder().build().unwrap();
    let base = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let factory = move |who: &Authenticated| {
        Session::builder()
            .runtime(Arc::clone(base.runtime()))
            .principal(who.principal.clone())
            .build()
    };
    let source = SessionSource::new(Arc::clone(&rt))
        .unwrap()
        .with_caller_sessions(Arc::new(factory));
    let who = Authenticated::new(
        Principal::new("ann", PrincipalKind::User),
        Some(Role::Operator),
        Method::Token,
    );
    let ann = Caller {
        principal: "ann".to_owned(),
        role: Some("operator".to_owned()),
        method: "token".to_owned(),
        identity: Some(Arc::new(who)),
    };
    let id = source.sql_start("SELECT 1", Some(&ann)).await.unwrap();
    assert_eq!(ended(&source, &id, Some(&ann)).await["state"], "finished");
    let rows = sessions(&source).await;
    let first: Vec<&Row> = rows.iter().filter(|r| r["principal"] == "ann").collect();
    assert_eq!(first.len(), 1, "{rows:?}");
    let first = first[0]["session_id"].as_str().unwrap().to_owned();

    assert_eq!(source.close_session(&first).await.unwrap(), Some(0));
    let id = source.sql_start("SELECT 2", Some(&ann)).await.unwrap();
    let again = ended(&source, &id, Some(&ann)).await;
    assert_eq!(again["state"], "finished", "{again}");
    let rows = sessions(&source).await;
    let open: Vec<&Row> = rows
        .iter()
        .filter(|r| r["principal"] == "ann" && r["state"] == "active")
        .collect();
    assert_eq!(open.len(), 1, "{rows:?}");
    assert_ne!(open[0]["session_id"], first.as_str());
}

#[tokio::test]
async fn the_resource_groups_are_listed_with_their_limits_and_none_without_a_group_file() {
    let rt = EngineRuntime::builder().build().unwrap();
    let source = SessionSource::new(rt).unwrap();
    assert!(
        source
            .rows(SystemTable::ResourceGroups)
            .await
            .unwrap()
            .is_empty()
    );

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("resource-groups.json");
    std::fs::write(
        &file,
        r#"{
  "rootGroups": [{"name": "root", "hardConcurrencyLimit": 3, "maxQueued": 5,
                  "subGroups": [{"name": "etl", "hardConcurrencyLimit": 1, "maxQueued": 2}]}],
  "defaultGroup": "root.etl"
}"#,
    )
    .unwrap();
    let rt = EngineRuntime::builder()
        .config(
            "krishiv.workload.resourceGroups.file",
            file.to_str().unwrap(),
        )
        .build()
        .unwrap();
    let source = SessionSource::new(rt).unwrap();
    let groups = source.rows(SystemTable::ResourceGroups).await.unwrap();
    let ids: Vec<&str> = groups
        .iter()
        .filter_map(|g| g["group_id"].as_str())
        .collect();
    assert_eq!(ids, ["root", "root.etl"], "{groups:?}");
    assert_eq!(groups[0]["hard_concurrency_limit"], 3);
    assert_eq!(groups[1]["parent_id"], "root");
    assert_eq!(groups[1]["max_queued"], 2);
    assert_eq!(groups[1]["state"], "active");
}
