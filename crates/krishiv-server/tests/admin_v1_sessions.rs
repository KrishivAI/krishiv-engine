// SPDX-License-Identifier: Apache-2.0
//! The session and query RPCs of `krishiv.admin.v1` over a real engine runtime (design P3-T066
//! section 4): a role-aware guard, queries held in the admission queue, and the REST transport.
#![cfg(all(feature = "admin-api", feature = "session"))]
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_server::admin_v1::{AdminConfig, SessionAdmin, router};
use krishiv_server::guard::{Access, Caller, Guard, GuardRequest, Refusal};
use krishiv_session::auth::{Authenticated, Method, Role};
use krishiv_session::{
    AdmissionController, EngineRuntime, Permit, Principal, PrincipalKind, QueryOptions,
    QueryRequest, Session,
};
use serde_json::Value;
use tower::ServiceExt;

/// Never admits: every query waits queued, so it is listed and can be cancelled.
#[derive(Debug, Default)]
struct Never {
    seen: Mutex<usize>,
    arrived: tokio::sync::Notify,
}

impl Never {
    async fn queued(&self, n: usize) {
        let arrivals = async {
            loop {
                let arrived = self.arrived.notified();
                if *self.seen.lock().unwrap() >= n {
                    return;
                }
                arrived.await;
            }
        };
        tokio::time::timeout(Duration::from_secs(120), arrivals)
            .await
            .unwrap_or_else(|_| panic!("{n} queries did not reach admission"));
    }
}

impl AdmissionController for Never {
    fn admit<'a>(&'a self, _: &'a QueryRequest) -> BoxFuture<'a, Result<Permit, KrishivError>> {
        *self.seen.lock().unwrap() += 1;
        self.arrived.notify_waiters();
        Box::pin(std::future::pending())
    }
}

/// `Authorization: Bearer <name>:<role>` is the caller `name` with that role.
struct Roles;

impl Guard for Roles {
    fn check<'a>(&'a self, r: GuardRequest<'a>) -> BoxFuture<'a, Result<(), Refusal>> {
        Box::pin(async move { Self::who(&r).map(|_| ()) })
    }

    fn authenticate<'a>(&'a self, r: GuardRequest<'a>) -> BoxFuture<'a, Result<Caller, Refusal>> {
        Box::pin(async move {
            let (name, role) = Self::who(&r)?;
            let mut caller = Caller::anonymous();
            name.clone_into(&mut caller.principal);
            caller.role = Some(role.as_str().to_owned());
            caller.method = "token".to_owned();
            caller.identity = Some(Arc::new(Authenticated::new(
                Principal::new(name, PrincipalKind::User),
                Some(role),
                Method::Token,
            )));
            Ok(caller)
        })
    }
}

impl Roles {
    fn who(r: &GuardRequest<'_>) -> Result<(String, Role), Refusal> {
        let _ = Access::Authenticated;
        let token = r.token.ok_or_else(|| panic!("no token"))?;
        let (name, role) = token.split_once(':').unwrap();
        let role = match role {
            "viewer" => Role::Viewer,
            "operator" => Role::Operator,
            _ => Role::Admin,
        };
        Ok((name.to_owned(), role))
    }
}

struct Fixture {
    app: Router,
    gate: Arc<Never>,
    rt: Arc<EngineRuntime>,
}

fn fixture() -> Fixture {
    let rt = EngineRuntime::builder().build().unwrap();
    let gate = Arc::new(Never::default());
    let app = router(
        Arc::new(SessionAdmin::new(Arc::clone(&rt))),
        Arc::new(Roles),
        AdminConfig {
            loopback: true,
            allowed_hosts: Vec::new(),
            idempotency_ttl: Duration::from_secs(60),
        },
    );
    Fixture { app, gate, rt }
}

impl Fixture {
    fn session(&self, who: &str) -> Session {
        Session::builder()
            .runtime(Arc::clone(&self.rt))
            .admission_controller(Arc::clone(&self.gate) as Arc<dyn AdmissionController>)
            .principal(Principal::new(who, PrincipalKind::User))
            .build()
            .unwrap()
    }

    /// Starts a query of `s` that waits in the admission queue.
    fn start(
        &self,
        s: &Session,
        id: u8,
        tags: &[&str],
    ) -> tokio::task::JoinHandle<Result<(), KrishivError>> {
        let s = s.clone();
        let tags: Vec<String> = tags.iter().map(|t| (*t).to_owned()).collect();
        tokio::spawn(async move {
            s.sql_with(
                "select 1",
                QueryOptions::new()
                    .operation_id(qid(id))
                    .tags(tags.iter().map(String::as_str)),
            )
            .await?
            .collect()
            .await
            .map(|_| ())
        })
    }

    /// `system.queries.cancel_reason` of query `id`, read by a session of its own.
    async fn cancel_reason(&self, id: u8) -> String {
        use arrow_array::cast::AsArray;
        let s = Session::builder()
            .runtime(Arc::clone(&self.rt))
            .build()
            .unwrap();
        let batches = s
            .sql(&format!(
                "select cancel_reason from system.queries where query_id = '{}'",
                qid(id)
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        batches[0].column(0).as_string::<i32>().value(0).to_owned()
    }

    async fn call(
        &self,
        who: &str,
        method: &str,
        path: &str,
        body: Option<&str>,
    ) -> (StatusCode, Value) {
        let req = Request::builder()
            .method(method)
            .uri(format!("/api/admin/v1{path}"))
            .header(header::HOST, "localhost:7700")
            .header(header::AUTHORIZATION, format!("Bearer {who}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.unwrap_or_default().to_owned()))
            .unwrap();
        let response = self.app.clone().oneshot(req).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, json)
    }
}

fn qid(n: u8) -> String {
    format!("00000000-0000-4000-8000-0000000000{n:02}")
}

fn ids(v: &Value, field: &str, key: &str) -> Vec<String> {
    let mut out: Vec<String> = v[field]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|r| r[key].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

#[tokio::test]
async fn a_viewer_lists_only_its_own_queries_and_an_operator_lists_all() {
    let f = fixture();
    let (a, b) = (f.session("alice"), f.session("bob"));
    let _ta = f.start(&a, 1, &[]);
    let _tb = f.start(&b, 2, &[]);
    f.gate.queued(2).await;

    let (status, body) = f.call("alice:viewer", "GET", "/queries", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(ids(&body, "queries", "queryId"), [qid(1)]);

    let (_, body) = f.call("carol:operator", "GET", "/queries", None).await;
    assert_eq!(ids(&body, "queries", "queryId"), [qid(1), qid(2)]);

    // The `states` query parameter filters; a finished-only filter finds nothing running.
    let (status, body) = f
        .call("carol:operator", "GET", "/queries?states=finished", None)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(ids(&body, "queries", "queryId").is_empty());

    let (_, body) = f.call("alice:viewer", "GET", "/sessions", None).await;
    assert_eq!(ids(&body, "sessions", "principal"), ["alice"]);
    let (_, body) = f.call("carol:admin", "GET", "/sessions", None).await;
    assert_eq!(ids(&body, "sessions", "principal"), ["alice", "bob"]);
}

#[tokio::test]
async fn a_viewer_cannot_cancel_another_principals_query_and_cannot_tell_it_exists() {
    let f = fixture();
    let (a, b) = (f.session("alice"), f.session("bob"));
    let _ta = f.start(&a, 1, &[]);
    let tb = f.start(&b, 2, &[]);
    f.gate.queued(2).await;

    let (status, other) = f
        .call(
            "alice:viewer",
            "POST",
            &format!("/queries/{}:cancel", qid(2)),
            Some("{}"),
        )
        .await;
    let (status_missing, missing) = f
        .call(
            "alice:viewer",
            "POST",
            &format!("/queries/{}:cancel", qid(99)),
            Some("{}"),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(status, status_missing);
    assert_eq!(other["error"]["errorClass"], missing["error"]["errorClass"]);
    assert_eq!(
        other["error"]["errorClass"],
        "INVALID_HANDLE.OPERATION_NOT_FOUND"
    );
    assert!(!tb.is_finished(), "bob's query is still queued");

    // An operator ends it, and the reason says someone else acted.
    let (status, body) = f
        .call(
            "carol:operator",
            "POST",
            &format!("/queries/{}:cancel", qid(2)),
            Some("{}"),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(tb.await.unwrap().is_err());
    assert_eq!(f.cancel_reason(2).await, "admin");
}

#[tokio::test]
async fn an_own_cancel_records_user() {
    let f = fixture();
    let a = f.session("alice");
    let ta = f.start(&a, 1, &[]);
    f.gate.queued(1).await;
    let (status, body) = f
        .call(
            "alice:viewer",
            "POST",
            &format!("/queries/{}:cancel", qid(1)),
            Some("{}"),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(ta.await.unwrap().is_err());
    assert_eq!(f.cancel_reason(1).await, "user");
}

#[tokio::test]
async fn cancel_by_tag_answers_the_same_whether_or_not_others_queries_match() {
    let f = fixture();
    let (a, b) = (f.session("alice"), f.session("bob"));
    let _ta = f.start(&a, 1, &["etl"]);
    let _tb = f.start(&b, 2, &["etl"]);
    f.gate.queued(2).await;

    let (status, body) = f
        .call(
            "alice:viewer",
            "POST",
            "/queries:cancelByTag",
            Some(r#"{"selector":"etl"}"#),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["queryIds"],
        serde_json::json!([qid(1)]),
        "only its own"
    );

    // Bob's tag matches only Bob's query; Alice's call answers as for a tag nothing carries.
    let (_, none) = f
        .call(
            "alice:viewer",
            "POST",
            "/queries:cancelByTag",
            Some(r#"{"selector":"etl"}"#),
        )
        .await;
    let (_, unknown) = f
        .call(
            "alice:viewer",
            "POST",
            "/queries:cancelByTag",
            Some(r#"{"selector":"nothing-has-this"}"#),
        )
        .await;
    assert_eq!(none, unknown);

    let (status, _) = f
        .call(
            "alice:viewer",
            "POST",
            "/queries:cancelByTag",
            Some(r#"{"selector":""}"#),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn kill_session_follows_the_same_ownership_rules() {
    let f = fixture();
    let (a, b) = (f.session("alice"), f.session("bob"));
    let _ta = f.start(&a, 1, &[]);
    let tb = f.start(&b, 2, &[]);
    f.gate.queued(2).await;
    let bob_session =
        f.rt.session_infos()
            .into_iter()
            .find(|s| s.principal == "bob")
            .unwrap()
            .session_id;

    let (status, body) = f
        .call(
            "alice:viewer",
            "POST",
            &format!("/sessions/{bob_session}:kill"),
            Some("{}"),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(
        body["error"]["errorClass"],
        "INVALID_HANDLE.SESSION_NOT_FOUND"
    );
    assert!(!tb.is_finished());

    let (status, body) = f
        .call(
            "carol:operator",
            "POST",
            &format!("/sessions/{bob_session}:kill"),
            Some("{}"),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["queriesCancelled"], "1", "int64 is a JSON string");
    assert!(tb.await.unwrap().is_err());

    // A wrong suffix is not a route.
    let (status, _) = f
        .call(
            "carol:admin",
            "POST",
            &format!("/sessions/{bob_session}:explode"),
            Some("{}"),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn a_repeated_request_id_replays_and_a_reused_one_is_refused() {
    let f = fixture();
    let a = f.session("alice");
    let ta = f.start(&a, 1, &[]);
    let _tb = f.start(&a, 2, &[]);
    f.gate.queued(2).await;

    let path = format!("/queries/{}:cancel", qid(1));
    let body = r#"{"requestId":"req-1"}"#;
    let (first, _) = f.call("alice:viewer", "POST", &path, Some(body)).await;
    assert_eq!(first, StatusCode::OK);
    assert!(ta.await.unwrap().is_err());
    // The query is gone now, yet the same request id replays the first answer.
    let (second, _) = f.call("alice:viewer", "POST", &path, Some(body)).await;
    assert_eq!(second, StatusCode::OK);
    // The same id for a different request is not a replay.
    let other = format!("/queries/{}:cancel", qid(2));
    let (third, body) = f.call("alice:viewer", "POST", &other, Some(body)).await;
    assert_ne!(third, StatusCode::OK, "{body}");
}
