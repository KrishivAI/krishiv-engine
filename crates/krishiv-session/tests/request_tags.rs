// SPDX-License-Identifier: Apache-2.0
//! Request tags (P3-T064 part 2): they reach `system.queries`, the admission request and the
//! cancel-by-tag entry points.
#![cfg(test)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow_array::cast::AsArray;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::{
    AdmissionController, BoxFuture, EngineRuntime, Permit, Principal, PrincipalKind, QueryOptions,
    QueryRequest, Session,
};

/// Never admits: every query waits queued (and so can be cancelled while it waits), and the tags
/// of each request are kept.
#[derive(Debug, Default)]
struct Never {
    seen: Mutex<Vec<Vec<(String, String)>>>,
}

impl AdmissionController for Never {
    fn admit<'a>(
        &'a self,
        request: &'a QueryRequest,
    ) -> BoxFuture<'a, Result<Permit, KrishivError>> {
        self.seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(request.tags.clone());
        Box::pin(std::future::pending())
    }
}

fn id(n: u8) -> String {
    format!("00000000-0000-4000-8000-0000000000{n:02}")
}

fn session(rt: &Arc<EngineRuntime>, gate: &Arc<Never>) -> Session {
    Session::builder()
        .runtime(Arc::clone(rt))
        .admission_controller(Arc::clone(gate) as Arc<dyn AdmissionController>)
        .build()
        .unwrap_or_else(|e| panic!("{e}"))
}

/// Starts `select 1` of `s` under `options` and returns the task that waits for its result.
fn start(s: &Session, options: QueryOptions) -> tokio::task::JoinHandle<Result<(), KrishivError>> {
    let s = s.clone();
    tokio::spawn(async move {
        s.sql_with("select 1", options)
            .await?
            .collect()
            .await
            .map(|_| ())
    })
}

async fn settle() {
    tokio::time::sleep(Duration::from_millis(300)).await;
}

fn class(r: Result<Result<(), KrishivError>, tokio::task::JoinError>) -> String {
    r.unwrap_or_else(|e| panic!("{e}"))
        .err()
        .map(|e| e.error_class().name().to_owned())
        .unwrap_or_default()
}

#[tokio::test]
async fn request_tags_reach_the_admission_request_with_the_session_tags() {
    let rt = EngineRuntime::builder()
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let gate = Arc::new(Never::default());
    let s = session(&rt, &gate);
    s.set_tag("team", "x").unwrap_or_else(|e| panic!("{e}"));
    let task = start(
        &s,
        QueryOptions::new()
            .operation_id(id(1))
            .tags(["etl", "job=42"]),
    );
    settle().await;
    let seen = gate
        .seen
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert!(!seen.is_empty(), "the query reached admission");
    let tags = &seen[0];
    assert!(tags.contains(&("team".to_owned(), "x".to_owned())));
    assert!(tags.contains(&("etl".to_owned(), String::new())));
    assert!(tags.contains(&("job=42".to_owned(), String::new())));
    s.interrupt_all();
    let _ = task.await;
}

#[tokio::test]
async fn interrupt_tag_is_session_scoped_and_exact() {
    let rt = EngineRuntime::builder()
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let gate = Arc::new(Never::default());
    let (a, b) = (session(&rt, &gate), session(&rt, &gate));
    let a_t = start(&a, QueryOptions::new().operation_id(id(1)).tags(["t"]));
    let a_ab = start(&a, QueryOptions::new().operation_id(id(2)).tags(["ab"]));
    let a_kv = start(&a, QueryOptions::new().operation_id(id(3)).tags(["a=1"]));
    let b_t = start(&b, QueryOptions::new().operation_id(id(4)).tags(["t"]));
    settle().await;

    assert!(
        a.interrupt_tag("nothing").is_empty(),
        "an unknown tag matches nothing"
    );
    let ids = a.interrupt_tag("t");
    assert_eq!(
        ids.iter()
            .map(|i| i.as_str().to_owned())
            .collect::<Vec<_>>(),
        [id(1)]
    );
    assert_eq!(class(a_t.await), ErrorClass::OperationCanceled.name());

    // `a` does not match `ab` or `a=1`, and session B's query with the same tag is untouched.
    assert!(a.interrupt_tag("a").is_empty());
    settle().await;
    assert!(!a_ab.is_finished() && !a_kv.is_finished() && !b_t.is_finished());

    a.interrupt_all();
    b.interrupt_all();
    for t in [a_ab, a_kv, b_t] {
        let _ = t.await;
    }
}

#[tokio::test]
async fn session_tags_never_match_an_interrupt_by_tag() {
    let rt = EngineRuntime::builder()
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let gate = Arc::new(Never::default());
    let s = session(&rt, &gate);
    s.set_tag("team", "x").unwrap_or_else(|e| panic!("{e}"));
    let task = start(&s, QueryOptions::new().operation_id(id(1)));
    settle().await;
    assert!(s.interrupt_tag("team").is_empty());
    assert!(!task.is_finished());
    s.interrupt_all();
    let _ = task.await;
}

fn session_of(rt: &Arc<EngineRuntime>, gate: &Arc<Never>, who: &str) -> Session {
    Session::builder()
        .runtime(Arc::clone(rt))
        .admission_controller(Arc::clone(gate) as Arc<dyn AdmissionController>)
        .principal(Principal::new(who, PrincipalKind::User))
        .build()
        .unwrap_or_else(|e| panic!("{e}"))
}

#[tokio::test]
async fn cancel_by_tag_scopes_by_principal_and_picks_the_reason() {
    let rt = EngineRuntime::builder()
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let gate = Arc::new(Never::default());
    let (alice, bob) = (
        session_of(&rt, &gate, "alice"),
        session_of(&rt, &gate, "bob"),
    );
    alice.set_tag("job", "42").unwrap_or_else(|e| panic!("{e}"));
    let a_job = start(&alice, QueryOptions::new().operation_id(id(1)));
    let b_etl = start(&bob, QueryOptions::new().operation_id(id(2)).tags(["etl"]));
    let a_etl = start(
        &alice,
        QueryOptions::new().operation_id(id(3)).tags(["etl"]),
    );
    let b_other = start(
        &bob,
        QueryOptions::new().operation_id(id(4)).tags(["other"]),
    );
    settle().await;

    // Bob without `Operate` reaches only his own query with the tag, and cannot tell that
    // alice's carry it too.
    let ids: Vec<String> = rt
        .cancel_by_tag("etl", "bob", false)
        .into_iter()
        .map(|i| i.as_str().to_owned())
        .collect();
    assert_eq!(ids, [id(2)]);
    assert_eq!(class(b_etl.await), ErrorClass::OperationCanceled.name());
    assert!(!a_etl.is_finished());

    // An admin (here bob with `Operate`) reaches the other principals' queries too, as a kill:
    // `k=v` matches alice's session tag, a bare `k` her request tag.
    let mut ids: Vec<String> = rt
        .cancel_by_tag("job=42", "bob", true)
        .into_iter()
        .chain(rt.cancel_by_tag("etl", "bob", true))
        .map(|i| i.as_str().to_owned())
        .collect();
    ids.sort();
    assert_eq!(ids, [id(1), id(3)]);
    assert_eq!(class(a_job.await), ErrorClass::KrishivQueryKilled.name());
    assert_eq!(class(a_etl.await), ErrorClass::KrishivQueryKilled.name());
    assert!(!b_other.is_finished());
    bob.interrupt_all();
    let _ = b_other.await;
}

/// A query already executing outlives the action that started it: cancel by tag finds it
/// through the live set.
#[tokio::test]
async fn tag_cancels_reach_a_running_query() {
    use futures::StreamExt;
    let rt = EngineRuntime::builder()
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let s = Session::builder()
        .runtime(Arc::clone(&rt))
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let mut stream = s
        .sql_with(
            "select count(*) from range(0, 1000000000)",
            QueryOptions::new().operation_id(id(1)).tags(["t"]),
        )
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .execute_stream()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let ids = s.interrupt_tag("t");
    assert_eq!(
        ids.iter()
            .map(|i| i.as_str().to_owned())
            .collect::<Vec<_>>(),
        [id(1)],
        "the running query was cancelled"
    );
    let end = stream.next().await;
    assert!(
        matches!(&end, Some(Err(e)) if e.error_class().name() == ErrorClass::OperationCanceled.name()),
        "the stream ends with OPERATION_CANCELED, saw {end:?}"
    );
}

#[tokio::test]
async fn a_key_appears_once_in_the_tags_of_a_row() {
    let s = Session::new().unwrap_or_else(|e| panic!("{e}"));
    s.set_tag("etl", "x").unwrap_or_else(|e| panic!("{e}"));
    s.sql_with("select 1", QueryOptions::new().tags(["etl", "a", "a"]))
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let batches = s
        .sql("select size(map_keys(tags)) from system.queries where tags['a'] is not null")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let n = batches[0]
        .column(0)
        .as_primitive::<arrow_array::types::Int32Type>()
        .value(0);
    assert_eq!(n, 2, "etl (the session's) and a, each once");
}

#[tokio::test]
async fn request_tags_show_in_system_queries() {
    let s = Session::new().unwrap_or_else(|e| panic!("{e}"));
    s.sql_with("select 1", QueryOptions::new().tags(["etl"]))
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    s.sql("select 2")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let batches = s
        .sql("select count(*) from system.queries where tags['etl'] is not null")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let n = batches[0]
        .column(0)
        .as_primitive::<arrow_array::types::Int64Type>()
        .value(0);
    assert_eq!(n, 1, "only the tagged query carries the request tag");
}

#[tokio::test]
async fn a_request_tag_selects_the_resource_group() {
    use krishiv_session::resource_groups::ResourceGroups;
    let groups = ResourceGroups::from_json(
        r#"{
  "rootGroups": [{"name": "root", "hardConcurrencyLimit": 4, "maxQueued": 4,
    "subGroups": [{"name": "etl", "hardConcurrencyLimit": 4, "maxQueued": 4},
                  {"name": "job", "hardConcurrencyLimit": 4, "maxQueued": 4},
                  {"name": "rest", "hardConcurrencyLimit": 4, "maxQueued": 4}]}],
  "selectors": [{"clientTags": ["etl"], "group": "root.etl"},
                {"clientTags": ["job=42"], "group": "root.job"}],
  "defaultGroup": "root.rest"
}"#,
        1 << 30,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let groups = Arc::new(groups);
    let s = Session::builder()
        .admission_controller(Arc::clone(&groups) as Arc<dyn AdmissionController>)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    // Each query is held open (its stream is not read), so its group counts it as running.
    let mut held = Vec::new();
    for (options, group) in [
        (QueryOptions::new().tags(["etl"]), "root.etl"),
        (QueryOptions::new().tags(["job=42"]), "root.job"),
        (QueryOptions::new(), "root.rest"),
    ] {
        held.push(
            s.sql_with("select 1", options)
                .await
                .unwrap_or_else(|e| panic!("{e}"))
                .execute_stream()
                .await
                .unwrap_or_else(|e| panic!("{e}")),
        );
        assert_eq!(groups.counts(group), Some((1, 0)), "{group} runs one query");
    }
}
