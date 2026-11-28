// SPDX-License-Identifier: Apache-2.0
//! The admin.v1 surface end to end over the router, without a socket.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use futures::future::{BoxFuture, FutureExt};
use prost::Message;
use tower::ServiceExt;

use super::*;
use crate::guard::{GuardRequest, Refusal, TokenGuard};

struct Fake;

impl AdminApi for Fake {
    fn get_version(&self, _: &Call) -> BoxFuture<'_, AdminResult<GetVersionResponse>> {
        async {
            Ok(GetVersionResponse {
                version: "9.9.9".to_owned(),
                admin_v1_revision: ADMIN_V1_REVISION,
                preview: PREVIEW,
            })
        }
        .boxed()
    }

    fn get_capabilities(&self, _: &Call) -> BoxFuture<'_, AdminResult<GetCapabilitiesResponse>> {
        async {
            Ok(GetCapabilitiesResponse {
                capabilities: vec![Capability {
                    kind: "surface".to_owned(),
                    name: "admin.v1".to_owned(),
                    value: ADMIN_V1_REVISION.to_string(),
                    detail: String::new(),
                }],
            })
        }
        .boxed()
    }

    fn list_sessions(
        &self,
        _: &Call,
        _: ListSessionsRequest,
    ) -> BoxFuture<'_, AdminResult<ListSessionsResponse>> {
        async { Ok(ListSessionsResponse::default()) }.boxed()
    }

    fn kill_session(
        &self,
        _: &Call,
        _: KillSessionRequest,
    ) -> BoxFuture<'_, AdminResult<KillSessionResponse>> {
        async { Ok(KillSessionResponse::default()) }.boxed()
    }

    fn list_queries(
        &self,
        _: &Call,
        _: ListQueriesRequest,
    ) -> BoxFuture<'_, AdminResult<ListQueriesResponse>> {
        async { Ok(ListQueriesResponse::default()) }.boxed()
    }

    fn cancel_query(
        &self,
        _: &Call,
        _: CancelQueryRequest,
    ) -> BoxFuture<'_, AdminResult<CancelQueryResponse>> {
        async { Ok(CancelQueryResponse::default()) }.boxed()
    }

    fn cancel_queries_by_tag(
        &self,
        _: &Call,
        _: CancelQueriesByTagRequest,
    ) -> BoxFuture<'_, AdminResult<CancelQueriesByTagResponse>> {
        async { Ok(CancelQueriesByTagResponse::default()) }.boxed()
    }
}

struct Open;

impl Guard for Open {
    fn check<'a>(&'a self, _: GuardRequest<'a>) -> BoxFuture<'a, Result<(), Refusal>> {
        async { Ok(()) }.boxed()
    }
}

fn app(guard: Arc<dyn Guard>, loopback: bool) -> Router {
    router(
        Arc::new(Fake),
        guard,
        AdminConfig {
            loopback,
            allowed_hosts: vec!["admin.internal".to_owned()],
            idempotency_ttl: std::time::Duration::from_secs(60),
        },
    )
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let response = app
        .clone()
        .oneshot(req)
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let (parts, body) = response.into_parts();
    let bytes = axum::body::to_bytes(body, 1 << 20)
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    (parts.status, parts.headers, bytes.to_vec())
}

fn get(path: &str) -> Request<Body> {
    Request::builder()
        .uri(path)
        .header(header::HOST, "localhost:7700")
        .body(Body::empty())
        .unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn the_descriptor_and_the_route_table_agree() {
    let service = descriptor_pool()
        .get_service_by_name("krishiv.admin.v1.AdminService")
        .unwrap_or_else(|| panic!("the descriptor has no AdminService"));
    let mut methods: Vec<String> = service.methods().map(|m| m.name().to_owned()).collect();
    let mut rpcs: Vec<String> = routes().iter().map(|r| r.rpc.clone()).collect();
    methods.sort();
    rpcs.sort();
    assert_eq!(
        methods, rpcs,
        "every RPC has a route and every route an RPC"
    );
    assert!(routes().iter().all(|r| r.path.starts_with('/')));
}

#[tokio::test]
async fn rest_returns_the_proto_json_of_the_messages() {
    let app = app(Arc::new(Open), true);
    let (status, headers, body) = send(&app, get("/api/admin/v1/version")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers[header::CONTENT_TYPE], "application/json");
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
    assert_eq!(json["version"], "9.9.9");
    assert_eq!(
        json["adminV1Revision"],
        ADMIN_V1_REVISION.to_string(),
        "64-bit integers are strings"
    );
    assert_eq!(json["preview"], PREVIEW);
    let (status, _, body) = send(&app, get("/api/admin/v1/capabilities")).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
    assert_eq!(json["capabilities"][0]["name"], "admin.v1");
}

#[tokio::test]
async fn a_rest_request_with_an_unknown_field_is_refused() {
    let app = app(Arc::new(Open), true);
    let req = Request::builder()
        .uri("/api/admin/v1/version")
        .header(header::HOST, "localhost")
        .body(Body::from(r#"{"typo": 1}"#))
        .unwrap_or_else(|e| panic!("{e}"));
    let (status, _, body) = send(&app, req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
    assert_eq!(json["error"]["errorClass"], "KRISHIV_ADMIN_INVALID_REQUEST");
    assert_eq!(json["error"]["sqlState"], "22023");
    assert!(
        json["error"]["requestId"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
    );
}

#[tokio::test]
async fn a_request_without_the_token_is_unauthenticated_and_a_cookie_is_no_credential() {
    let app = app(
        Arc::new(TokenGuard::new("secret-secret-secret-secret-0000")),
        true,
    );
    let (status, headers, body) = send(&app, get("/api/admin/v1/version")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(headers[header::WWW_AUTHENTICATE], "Bearer");
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
    assert_eq!(json["error"]["errorClass"], "KRISHIV_ADMIN_UNAUTHENTICATED");
    let cookie = Request::builder()
        .uri("/api/admin/v1/version")
        .header(header::HOST, "localhost")
        .header(
            header::COOKIE,
            "krishiv_ui_token=secret-secret-secret-secret-0000",
        )
        .body(Body::empty())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(send(&app, cookie).await.0, StatusCode::UNAUTHORIZED);
    let bearer = Request::builder()
        .uri("/api/admin/v1/version")
        .header(header::HOST, "localhost")
        .header(
            header::AUTHORIZATION,
            "Bearer secret-secret-secret-secret-0000",
        )
        .body(Body::empty())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(send(&app, bearer).await.0, StatusCode::OK);
}

#[tokio::test]
async fn a_form_post_is_415_and_a_foreign_host_is_421() {
    let app = app(Arc::new(Open), true);
    let form = Request::builder()
        .method("POST")
        .uri("/api/admin/v1/version")
        .header(header::HOST, "localhost")
        .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
        .body(Body::empty())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(send(&app, form).await.0, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    let none = Request::builder()
        .method("POST")
        .uri("/api/admin/v1/version")
        .header(header::HOST, "localhost")
        .body(Body::empty())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(send(&app, none).await.0, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    for host in [
        "evil.example",
        "localhost.evil.example:7700",
        "127.0.0.1.evil.example",
    ] {
        let req = Request::builder()
            .uri("/api/admin/v1/version")
            .header(header::HOST, host)
            .body(Body::empty())
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            send(&app, req).await.0,
            StatusCode::MISDIRECTED_REQUEST,
            "{host}"
        );
    }
    for host in [
        "localhost",
        "127.0.0.1:7700",
        "[::1]:7700",
        "admin.internal",
    ] {
        let req = Request::builder()
            .uri("/api/admin/v1/version")
            .header(header::HOST, host)
            .body(Body::empty())
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(send(&app, req).await.0, StatusCode::OK, "{host}");
    }
    // A listener that is not loopback does not filter hosts.
    let open = super::tests::app(Arc::new(Open), false);
    let req = Request::builder()
        .uri("/api/admin/v1/version")
        .header(header::HOST, "anything.example")
        .body(Body::empty())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(send(&open, req).await.0, StatusCode::OK);
}

#[tokio::test]
async fn admin_v1_sends_no_cors_headers_and_answers_no_preflight() {
    let app = app(Arc::new(Open), true);
    let (_, headers, _) = send(&app, get("/api/admin/v1/version")).await;
    assert!(
        headers
            .keys()
            .all(|k| !k.as_str().starts_with("access-control-")),
        "{headers:?}"
    );
    let preflight = Request::builder()
        .method("OPTIONS")
        .uri("/api/admin/v1/version")
        .header(header::HOST, "localhost")
        .header(header::ORIGIN, "https://evil.example")
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, "GET")
        .body(Body::empty())
        .unwrap_or_else(|e| panic!("{e}"));
    let (status, headers, _) = send(&app, preflight).await;
    assert!(!status.is_success());
    assert!(!headers.contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN));
}

fn grpc(path: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(header::HOST, "localhost")
        .header(header::CONTENT_TYPE, "application/grpc")
        .body(Body::from(vec![0_u8; 5]))
        .unwrap_or_else(|e| panic!("{e}"))
}

#[tokio::test]
async fn grpc_and_rest_answer_alike() {
    let app = app(Arc::new(Open), true);
    let (status, _, body) = send(&app, grpc("/krishiv.admin.v1.AdminService/GetVersion")).await;
    assert_eq!(status, StatusCode::OK);
    // One length-prefixed message: flag byte, 4 length bytes, the message.
    assert_eq!(body[0], 0);
    let len = u32::from_be_bytes([body[1], body[2], body[3], body[4]]) as usize;
    let message = GetVersionResponse::decode(&body[5..5 + len]).unwrap_or_default();
    assert_eq!(message.version, "9.9.9");
    assert_eq!(message.admin_v1_revision, ADMIN_V1_REVISION);
    let (_, _, rest) = send(&app, get("/api/admin/v1/version")).await;
    let json: serde_json::Value = serde_json::from_slice(&rest).unwrap_or_default();
    assert_eq!(json["version"], message.version);
}

#[tokio::test]
async fn an_unauthenticated_grpc_call_gets_a_grpc_status_not_an_http_error() {
    let app = app(
        Arc::new(TokenGuard::new("secret-secret-secret-secret-0000")),
        true,
    );
    let (status, headers, _) = send(&app, grpc("/krishiv.admin.v1.AdminService/GetVersion")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["grpc-status"], "16", "UNAUTHENTICATED");
}

#[tokio::test]
async fn a_grpc_call_with_the_wrong_content_type_is_415() {
    let app = app(Arc::new(Open), true);
    let req = Request::builder()
        .method("POST")
        .uri("/krishiv.admin.v1.AdminService/GetVersion")
        .header(header::HOST, "localhost")
        .header(header::CONTENT_TYPE, "text/plain")
        .body(Body::empty())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(send(&app, req).await.0, StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

#[test]
fn error_kinds_map_to_the_statuses_of_the_design_table() {
    use axum::http::StatusCode as S;
    let cases = [
        (
            ErrorKind::InvalidArgument,
            S::BAD_REQUEST,
            tonic::Code::InvalidArgument,
        ),
        (
            ErrorKind::Unauthenticated,
            S::UNAUTHORIZED,
            tonic::Code::Unauthenticated,
        ),
        (
            ErrorKind::PermissionDenied,
            S::FORBIDDEN,
            tonic::Code::PermissionDenied,
        ),
        (ErrorKind::NotFound, S::NOT_FOUND, tonic::Code::NotFound),
        (
            ErrorKind::FailedPrecondition,
            S::PRECONDITION_FAILED,
            tonic::Code::FailedPrecondition,
        ),
        (
            ErrorKind::AlreadyExists,
            S::CONFLICT,
            tonic::Code::AlreadyExists,
        ),
        (
            ErrorKind::ResourceExhausted,
            S::PAYLOAD_TOO_LARGE,
            tonic::Code::ResourceExhausted,
        ),
        (
            ErrorKind::Unavailable,
            S::SERVICE_UNAVAILABLE,
            tonic::Code::Unavailable,
        ),
        (
            ErrorKind::Internal,
            S::INTERNAL_SERVER_ERROR,
            tonic::Code::Internal,
        ),
    ];
    for (kind, http, code) in cases {
        let error = AdminError::new(kind, "X.Y", "00000", "m").with_parameter("a", "b");
        assert_eq!(super::rest::http_status(kind), http);
        let status = super::grpc::status_of(&error);
        assert_eq!(status.code(), code);
        assert_eq!(
            status
                .metadata()
                .get("krishiv-error-reason")
                .map(|v| v.to_str().ok()),
            Some(Some("X"))
        );
        assert_eq!(error.reason(), "X");
    }
}

#[tokio::test]
async fn the_admin_router_merges_with_a_router_that_has_a_fallback_and_unknown_paths_stay_404() {
    use axum::routing::get;
    // The UI router carries a catch-all fallback; merging must not panic, and an unknown path is
    // that router's answer, not a gRPC status.
    let ui = Router::new()
        .route("/ui", get(|| async { "ui" }))
        .fallback(|| async { StatusCode::NOT_FOUND });
    let merged = ui.merge(app(Arc::new(Open), true));
    let (status, headers, _) = send(&merged, get_req("/nowhere")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(!headers.contains_key("grpc-status"));
    assert_eq!(
        send(&merged, get_req("/api/admin/v1/version")).await.0,
        StatusCode::OK
    );
    // Alone, a path under the service with no route-table row is a gRPC status (UNIMPLEMENTED).
    let alone = app(Arc::new(Open), true);
    let (status, _, _) = send(&alone, grpc("/krishiv.admin.v1.AdminService/Nope")).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "a gRPC path without a row is a gRPC status"
    );
}

fn get_req(path: &str) -> Request<Body> {
    get(path)
}

#[tokio::test]
async fn a_malformed_host_is_refused() {
    let app = app(Arc::new(Open), true);
    for host in [
        "[::1]garbage",
        "[::1",
        "localhost:",
        "localhost:80x",
        "a@localhost",
    ] {
        let req = Request::builder()
            .uri("/api/admin/v1/version")
            .header(header::HOST, host)
            .body(Body::empty())
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            send(&app, req).await.0,
            StatusCode::MISDIRECTED_REQUEST,
            "{host}"
        );
    }
}

#[tokio::test]
async fn the_bearer_scheme_is_case_insensitive() {
    let app = app(
        Arc::new(TokenGuard::new("secret-secret-secret-secret-0000")),
        true,
    );
    for scheme in ["Bearer", "bearer", "BEARER"] {
        let req = Request::builder()
            .uri("/api/admin/v1/version")
            .header(header::HOST, "localhost")
            .header(
                header::AUTHORIZATION,
                format!("{scheme} secret-secret-secret-secret-0000"),
            )
            .body(Body::empty())
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(send(&app, req).await.0, StatusCode::OK, "{scheme}");
    }
}

#[tokio::test]
async fn the_version_alias_needs_the_credential_the_guard_asks_for() {
    use axum::routing::get;
    let app = guard_version_alias(
        Router::new()
            .route("/api/v1/version", get(|| async { "v" }))
            .route("/other", get(|| async { "o" })),
        Arc::new(TokenGuard::new("secret-secret-secret-secret-0000")),
    );
    assert_eq!(
        send(&app, get_req("/api/v1/version")).await.0,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        send(&app, get_req("/other")).await.0,
        StatusCode::OK,
        "other paths pass through"
    );
    let with = Request::builder()
        .uri("/api/v1/version")
        .header(
            header::AUTHORIZATION,
            "Bearer secret-secret-secret-secret-0000",
        )
        .body(Body::empty())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(send(&app, with).await.0, StatusCode::OK);
}

struct Unavailable;

impl Guard for Unavailable {
    fn check<'a>(&'a self, _: GuardRequest<'a>) -> BoxFuture<'a, Result<(), Refusal>> {
        async { Err(Refusal::Unavailable) }.boxed()
    }
}

#[tokio::test]
async fn an_unavailable_guard_is_503_with_its_registry_class() {
    let app = app(Arc::new(Unavailable), true);
    let (status, _, body) = send(&app, get("/api/admin/v1/version")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
    assert_eq!(json["error"]["errorClass"], "KRISHIV_ADMIN_UNAVAILABLE");
    assert_eq!(json["error"]["sqlState"], "08004");
}

#[tokio::test]
async fn a_body_over_the_cap_is_413_with_its_registry_class() {
    let app = app(Arc::new(Open), true);
    let req = Request::builder()
        .uri("/api/admin/v1/version")
        .header(header::HOST, "localhost")
        .body(Body::from(vec![b' '; (1 << 20) + 1]))
        .unwrap_or_else(|e| panic!("{e}"));
    let (status, _, body) = send(&app, req).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
    assert_eq!(
        json["error"]["errorClass"],
        "KRISHIV_ADMIN_REQUEST_TOO_LARGE"
    );
    assert_eq!(json["error"]["messageParameters"]["limit"], "1048576");
}

#[test]
fn every_admin_error_class_is_in_the_registry() {
    // The permanent codes clients branch on are registry classes, not strings of this crate.
    let json = include_str!("../../../krishiv-common/krishiv-error-classes.json");
    for class in [
        "KRISHIV_ADMIN_INVALID_REQUEST",
        "KRISHIV_ADMIN_REQUEST_TOO_LARGE",
        "KRISHIV_ADMIN_UNAUTHENTICATED",
        "KRISHIV_ADMIN_UNAVAILABLE",
    ] {
        assert!(json.contains(&format!("\"{class}\"")), "{class}");
    }
}
