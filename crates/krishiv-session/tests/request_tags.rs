// SPDX-License-Identifier: Apache-2.0
//! Request tags (P3-T064 part 2): they reach `system.queries`, the admission request and the
//! cancel-by-tag entry points.
#![cfg(test)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow_array::Array;
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
    arrived: tokio::sync::Notify,
}

impl Never {
    /// Returns once `n` queries have reached this gate. A query that is here has its row and
    /// its tags, so a cancel that follows finds it; a test waits on this, never on the clock.
    ///
    /// The bound only turns a query that never arrives (it failed before admission) into a
    /// failure with a message, where the wait would otherwise hang the run.
    async fn queued(&self, n: usize) {
        let arrivals = async {
            loop {
                // Registered before the count is read, so an arrival between the two is not lost.
                let arrived = self.arrived.notified();
                if self.seen() >= n {
                    return;
                }
                arrived.await;
            }
        };
        let arrived = tokio::time::timeout(Duration::from_secs(120), arrivals).await;
        assert!(
            arrived.is_ok(),
            "{} of {n} queries reached admission",
            self.seen()
        );
    }

    fn seen(&self) -> usize {
        self.seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
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
        self.arrived.notify_waiters();
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
    gate.queued(1).await;
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
    gate.queued(4).await;

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
    // Only the first cancel to reach a query names it, so each of the three being named by
    // its own tag here shows that no cancel above reached it.
    for (session, tag, n) in [(&a, "ab", 2), (&a, "a=1", 3), (&b, "t", 4)] {
        let ids = session.interrupt_tag(tag);
        assert_eq!(
            ids.iter()
                .map(|i| i.as_str().to_owned())
                .collect::<Vec<_>>(),
            [id(n)],
            "{tag} was not cancelled before"
        );
    }
    for t in [a_ab, a_kv, b_t] {
        assert_eq!(class(t.await), ErrorClass::OperationCanceled.name());
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
    gate.queued(1).await;
    assert!(s.interrupt_tag("team").is_empty());
    // Still queued: the cancel that ends it is the first to name it.
    let ids = s.interrupt_all();
    assert_eq!(
        ids.iter()
            .map(|i| i.as_str().to_owned())
            .collect::<Vec<_>>(),
        [id(1)]
    );
    assert_eq!(class(task.await), ErrorClass::OperationCanceled.name());
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
    gate.queued(4).await;

    // Bob without `Operate` reaches only his own query with the tag, and cannot tell that
    // alice's carry it too.
    let ids: Vec<String> = rt
        .cancel_by_tag("etl", &Principal::new("bob", PrincipalKind::User), false)
        .into_iter()
        .map(|i| i.as_str().to_owned())
        .collect();
    assert_eq!(ids, [id(2)]);
    assert_eq!(class(b_etl.await), ErrorClass::OperationCanceled.name());
    assert!(!a_etl.is_finished());

    // An admin (here bob with `Operate`) reaches the other principals' queries too, as a kill:
    // `k=v` matches alice's session tag, a bare `k` her request tag.
    let mut ids: Vec<String> = rt
        .cancel_by_tag("job=42", &Principal::new("bob", PrincipalKind::User), true)
        .into_iter()
        .chain(rt.cancel_by_tag("etl", &Principal::new("bob", PrincipalKind::User), true))
        .map(|i| i.as_str().to_owned())
        .collect();
    ids.sort();
    assert_eq!(ids, [id(1), id(3)]);
    assert_eq!(class(a_job.await), ErrorClass::KrishivQueryKilled.name());
    assert_eq!(class(a_etl.await), ErrorClass::KrishivQueryKilled.name());
    // Bob's other query was reached by none of these: the cancel that ends it names it.
    let ids: Vec<String> = bob
        .interrupt_all()
        .into_iter()
        .map(|i| i.as_str().to_owned())
        .collect();
    assert_eq!(ids, [id(4)]);
    assert_eq!(class(b_other.await), ErrorClass::OperationCanceled.name());
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

#[tokio::test]
async fn refused_and_replayed_statements_keep_their_request_tags() {
    let s = Session::new().unwrap_or_else(|e| panic!("{e}"));
    // A statement that fails analysis never reaches admission.
    let refused = s
        .sql_with(
            "select nope from nowhere",
            QueryOptions::new().tags(["bad"]),
        )
        .await;
    assert!(refused.is_err());
    let batches = s
        .sql("select count(*) from system.queries where tags['bad'] is not null")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let n = batches[0]
        .column(0)
        .as_primitive::<arrow_array::types::Int64Type>()
        .value(0);
    assert_eq!(n, 1, "the refused statement's row carries its request tag");
}

#[tokio::test]
async fn a_set_tag_during_admission_does_not_change_the_tags_a_query_is_cancelled_by() {
    let rt = EngineRuntime::builder()
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let gate = Arc::new(Never::default());
    let s = session(&rt, &gate);
    s.set_tag("team", "a").unwrap_or_else(|e| panic!("{e}"));
    let task = start(&s, QueryOptions::new().operation_id(id(1)));
    gate.queued(1).await;
    // The query waits in admission with team=a; a later tag change does not move it.
    s.set_tag("team", "b").unwrap_or_else(|e| panic!("{e}"));
    assert!(
        rt.cancel_by_tag(
            "team=b",
            &Principal::new("nobody", PrincipalKind::User),
            true
        )
        .is_empty()
    );
    let ids = rt.cancel_by_tag(
        "team=a",
        &Principal::new("nobody", PrincipalKind::User),
        true,
    );
    assert_eq!(ids.len(), 1, "cancelled by the tag it started with");
    let _ = task.await;
}

#[tokio::test]
async fn a_queued_query_without_an_operation_id_is_named_when_cancelled_by_tag() {
    let rt = EngineRuntime::builder()
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let gate = Arc::new(Never::default());
    let s = session(&rt, &gate);
    let task = start(&s, QueryOptions::new().tags(["t"]));
    gate.queued(1).await;
    let ids = s.interrupt_tag("t");
    assert_eq!(
        ids.len(),
        1,
        "the minted row id of the queued query is returned"
    );
    assert_eq!(class(task.await), ErrorClass::OperationCanceled.name());
}

#[tokio::test]
async fn a_statements_own_trace_never_borrows_or_changes_the_sessions() {
    let s = Session::new().unwrap_or_else(|e| panic!("{e}"));
    let a = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let b = "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-00f067aa0ba902b7-01";
    s.set_trace_context(a).unwrap_or_else(|e| panic!("{e}"));
    for options in [
        QueryOptions::new().inbound_trace(Some(b), Some("tenant=acme")),
        QueryOptions::new().inbound_trace(None, Some("tenant=acme")),
        QueryOptions::new().inbound_trace(Some("junk"), None),
        QueryOptions::new(),
    ] {
        s.sql_with("select 1", options)
            .await
            .unwrap_or_else(|e| panic!("{e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("{e}"));
    }
    // The session's own trace is untouched by the statements' traces.
    assert_eq!(
        s.trace_id().as_deref(),
        Some("4bf92f3577b34da6a3ce929d0e0e4736")
    );
    let batches = s
        .sql("select trace_id from system.queries where state = 'finished' order by submitted_at")
        .await
        .unwrap_or_else(|e| panic!("{e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let col = batches[0].column(0).as_string::<i32>();
    let seen: Vec<Option<String>> = (0..batches[0].num_rows())
        .map(|i| (!col.is_null(i)).then(|| col.value(i).to_owned()))
        .collect();
    assert_eq!(
        seen,
        [
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned()),
            None,
            None,
            Some("4bf92f3577b34da6a3ce929d0e0e4736".to_owned()),
        ],
        "its own trace, none without a valid traceparent, the session's only for a plain statement"
    );
}

#[tokio::test]
async fn cancel_by_tag_decides_ownership_on_the_whole_principal_not_the_name() {
    let rt = EngineRuntime::builder()
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let gate = Arc::new(Never::default());
    let mut service = Principal::new("bob", PrincipalKind::Service);
    service.tenant = Some("other".to_owned());
    let s = Session::builder()
        .runtime(Arc::clone(&rt))
        .admission_controller(Arc::clone(&gate) as Arc<dyn AdmissionController>)
        .principal(service)
        .build()
        .unwrap_or_else(|e| panic!("{e}"));
    let task = start(&s, QueryOptions::new().operation_id(id(1)).tags(["etl"]));
    gate.queued(1).await;
    // A user named "bob" without `Operate` does not own a service principal's query of that name.
    let bob = Principal::new("bob", PrincipalKind::User);
    assert!(rt.cancel_by_tag("etl", &bob, false).is_empty());
    assert!(!task.is_finished());
    s.interrupt_all();
    let _ = task.await;
}
