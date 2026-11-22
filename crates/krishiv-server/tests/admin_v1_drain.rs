// SPDX-License-Identifier: Apache-2.0
//! `GetReadiness`, `GetDrainStatus`, `Drain` and `ResumeServing` of `krishiv.admin.v1` (design
//! P3-T066 section 4 and 5.1): who may call them, and the drain state machine behind them.
#![cfg(all(feature = "admin-api", feature = "session"))]
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use futures::future::BoxFuture;
use krishiv_server::admin_v1::{AdminConfig, DrainOps, SessionAdmin, router};
use krishiv_server::drain::DrainController;
use krishiv_server::guard::{Access, Caller, Guard, GuardRequest, Refusal};
use krishiv_server::readiness::Readiness;
use krishiv_session::EngineRuntime;
use krishiv_session::auth::{Authenticated, Method, Role};
use krishiv_session::{Principal, PrincipalKind};
use serde_json::Value;
use tower::ServiceExt;

/// Collects the `krishiv::admin_action` records as `field=value` strings.
#[derive(Clone, Default)]
struct Records(Arc<std::sync::Mutex<Vec<String>>>);

struct Fields(String);

impl tracing::field::Visit for Fields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write as _;
        let _ = write!(self.0, "{}={value:?} ", field.name());
    }
}

impl tracing::Subscriber for Records {
    fn enabled(&self, m: &tracing::Metadata<'_>) -> bool {
        m.target() == "krishiv::admin_action"
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, e: &tracing::Event<'_>) {
        let mut f = Fields(String::new());
        e.record(&mut f);
        self.0.lock().unwrap().push(f.0);
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// One global subscriber, so a callsite first hit by a parallel test is never cached as off.
fn records() -> &'static Records {
    static RECORDS: std::sync::OnceLock<Records> = std::sync::OnceLock::new();
    RECORDS.get_or_init(|| {
        let r = Records::default();
        let _ = tracing::subscriber::set_global_default(r.clone());
        r
    })
}

/// Whether a `Drain` record with an `ok` result and `target_part` in its target was written.
fn drain_recorded(target_part: &str) -> bool {
    records().0.lock().unwrap().iter().any(|r| {
        r.contains("rpc=\"Drain\"") && r.contains("result=\"ok\"") && r.contains(target_part)
    })
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
    serving: Arc<AtomicBool>,
    rt: Arc<EngineRuntime>,
}

fn fixture(with_ops: bool) -> Fixture {
    let rt = EngineRuntime::builder().build().unwrap();
    let serving = Arc::new(AtomicBool::new(true));
    let drain = Arc::new(DrainController::new(Arc::clone(&serving)));
    let readiness = Readiness::new(
        Arc::clone(&serving),
        vec![],
        Duration::from_secs(10),
        Duration::from_secs(2),
    );
    let (start_rt, resume_rt) = (Arc::clone(&rt), Arc::clone(&rt));
    let (start_drain, resume_drain) = (Arc::clone(&drain), Arc::clone(&drain));
    let ops = DrainOps {
        drain,
        readiness,
        start: Box::new(move |grace, exit| {
            start_rt.set_draining(true);
            start_drain.start(std::time::Instant::now(), grace, exit);
        }),
        resume: Box::new(move || {
            resume_drain.resume()?;
            resume_rt.set_draining(false);
            Ok(())
        }),
        load: Box::new(|| (0, 0)),
        default_grace: Duration::from_secs(30),
    };
    let mut admin = SessionAdmin::new(Arc::clone(&rt));
    if with_ops {
        admin = admin.with_ops(Arc::new(ops));
    }
    let app = router(
        Arc::new(admin),
        Arc::new(Roles),
        AdminConfig {
            loopback: true,
            allowed_hosts: Vec::new(),
            idempotency_ttl: Duration::from_secs(60),
        },
    );
    Fixture { app, serving, rt }
}

impl Fixture {
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
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }
}

#[tokio::test]
async fn readiness_and_drain_status_need_monitor() {
    let f = fixture(true);
    for path in ["/readiness", "/drain"] {
        let (status, _) = f.call("v:viewer", "GET", path, None).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{path}");
    }
    let (status, body) = f.call("o:operator", "GET", "/readiness", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["ready"], true);
    assert_eq!(body["checks"][0]["name"], "serving");
    assert_eq!(body["checks"][0]["state"], "ok");
    let (status, body) = f.call("o:operator", "GET", "/drain", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"]["state"], "serving");
}

#[tokio::test]
async fn drain_and_resume_need_manage_and_follow_the_state_machine() {
    let _ = records();
    let f = fixture(true);
    let (status, _) = f.call("o:operator", "POST", "/drain", Some("{}")).await;
    assert_eq!(status, StatusCode::FORBIDDEN, "an operator may not drain");
    assert!(!f.rt.is_draining());

    let (status, body) = f
        .call("a:admin", "POST", "/drain", Some(r#"{"gracePeriod":"5s"}"#))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"]["state"], "draining");
    assert!(body["status"]["deadline"].is_string(), "{body}");
    assert!(
        drain_recorded("grace 5s, exit_when_drained false"),
        "the Drain action record carries its grace and exit flag"
    );
    assert!(f.rt.is_draining());
    assert!(
        !f.serving.load(std::sync::atomic::Ordering::SeqCst),
        "readiness flips first"
    );
    let (_, body) = f.call("o:operator", "GET", "/readiness", None).await;
    assert_eq!(body["ready"], false);
    assert_eq!(body["checks"][0]["state"], "failed");

    let (status, body) = f.call("a:admin", "DELETE", "/drain", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"]["state"], "serving");
    assert!(!f.rt.is_draining());
    assert!(f.serving.load(std::sync::atomic::Ordering::SeqCst));
}

#[tokio::test]
async fn resume_after_exit_when_drained_is_a_failed_precondition() {
    let _ = records();
    let f = fixture(true);
    let (status, _) = f
        .call(
            "a:admin",
            "POST",
            "/drain",
            Some(r#"{"exitWhenDrained":true}"#),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(drain_recorded("exit_when_drained true"));
    let (status, body) = f.call("a:admin", "DELETE", "/drain", None).await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED, "{body}");
    assert_eq!(
        body["error"]["errorClass"],
        "KRISHIV_ADMIN_PRECONDITION_FAILED"
    );
    assert_eq!(body["error"]["messageParameters"]["reason"], "exiting");
    assert!(f.rt.is_draining(), "still draining: the process exits");
}

#[tokio::test]
async fn a_second_drain_never_extends_the_deadline() {
    let f = fixture(true);
    let (_, first) = f
        .call(
            "a:admin",
            "POST",
            "/drain",
            Some(r#"{"gracePeriod":"60s"}"#),
        )
        .await;
    let (_, second) = f
        .call(
            "a:admin",
            "POST",
            "/drain",
            Some(r#"{"gracePeriod":"600s"}"#),
        )
        .await;
    assert_eq!(
        first["status"]["deadline"], second["status"]["deadline"],
        "a longer grace period changes nothing"
    );
}

#[tokio::test]
async fn a_server_without_drain_support_says_so() {
    let f = fixture(false);
    let (status, body) = f.call("a:admin", "GET", "/drain", None).await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED, "{body}");
    assert_eq!(body["error"]["errorClass"], "KRISHIV_UNSUPPORTED_FEATURE");
}

#[tokio::test]
async fn a_grace_period_outside_one_second_to_a_day_is_refused() {
    let f = fixture(true);
    for grace in ["0s", "86401s", "315576000000s"] {
        let body = format!(r#"{{"gracePeriod":"{grace}"}}"#);
        let (status, response) = f.call("a:admin", "POST", "/drain", Some(&body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{grace}: {response}");
    }
    assert!(!f.rt.is_draining(), "a refused drain starts nothing");
}
