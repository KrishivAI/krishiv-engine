// SPDX-License-Identifier: Apache-2.0
//! The REST side of `krishiv.admin.v1`: axum routes built from the route table, bodies mapped
//! to and from the proto3 JSON of the descriptor, and the error envelope (design §6.2, §6.4).

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{MethodRouter, delete, get, patch, post, put};
use prost::Message;
use prost_reflect::{DynamicMessage, MethodDescriptor, SerializeOptions};

use super::{
    AdminApi, AdminError, AdminResult, ErrorKind, GetCapabilitiesRequest, GetVersionRequest,
    Privilege, REST_PREFIX, descriptor_pool, routes,
};
use crate::guard::{Access, Guard, GuardRequest, GuardSurface, Refusal, peer_leaf_of, peer_of};

/// A request body over this is refused (design §6.2; a group document is the largest request).
pub(super) const MAX_BODY: usize = 1 << 20;

const SERVICE: &str = "krishiv.admin.v1.AdminService";

struct Rest {
    api: Arc<dyn AdminApi>,
    guard: Arc<dyn Guard>,
}

pub(super) fn router(api: Arc<dyn AdminApi>, guard: Arc<dyn Guard>) -> Router {
    let state = Arc::new(Rest { api, guard });
    let mut router = Router::new();
    for route in routes() {
        let rpc = route.rpc.clone();
        let privilege = route.privilege;
        let state = Arc::clone(&state);
        let handler = move |req: Request| {
            let state = Arc::clone(&state);
            let rpc = rpc.clone();
            async move { state.call(&rpc, privilege, req).await }
        };
        let method: Option<MethodRouter> = match route.method.as_str() {
            "GET" => Some(get(handler)),
            "POST" => Some(post(handler)),
            "PUT" => Some(put(handler)),
            "PATCH" => Some(patch(handler)),
            "DELETE" => Some(delete(handler)),
            _ => None,
        };
        if let Some(method) = method {
            router = router.route(&format!("{REST_PREFIX}{}", route.path), method);
        }
    }
    router
}

impl Rest {
    async fn call(&self, rpc: &str, privilege: Privilege, req: Request) -> Response {
        let request_id = request_id_of(&req);
        match self.run(rpc, privilege, req).await {
            Ok(body) => json_response(StatusCode::OK, body),
            Err(e) => error_response(&e, &request_id),
        }
    }

    async fn run(&self, rpc: &str, privilege: Privilege, req: Request) -> AdminResult<String> {
        let (token, malformed) = bearer(req.headers());
        let access = match privilege {
            Privilege::Authenticated => Access::Authenticated,
        };
        let guard_request = GuardRequest::new(GuardSurface::Admin, access)
            .with_token(token.as_deref())
            .with_malformed(malformed)
            .with_peer(peer_of(req.extensions()))
            .with_peer_leaf(peer_leaf_of(req.extensions()));
        self.guard
            .check(guard_request)
            .await
            .map_err(refusal_error)?;
        let bytes = axum::body::to_bytes(req.into_body(), MAX_BODY)
            .await
            .map_err(|_| {
                AdminError::new(
                    ErrorKind::ResourceExhausted,
                    "INVALID_PARAMETER_VALUE",
                    "22023",
                    "the request body is over 1 MiB",
                )
            })?;
        let method = method_of(rpc)?;
        let request = if bytes.is_empty() {
            Vec::new()
        } else {
            // Unknown fields are refused (REST only; design §6.2): a typo guard for hand-written JSON.
            let mut json = serde_json::Deserializer::from_slice(&bytes);
            let message = DynamicMessage::deserialize(method.input(), &mut json).map_err(|e| {
                AdminError::invalid_argument(format!("the request body is not valid: {e}"))
            })?;
            json.end().map_err(|e| {
                AdminError::invalid_argument(format!("the request body is not valid: {e}"))
            })?;
            message.encode_to_vec()
        };
        let response = invoke(self.api.as_ref(), rpc, &request).await?;
        let output = DynamicMessage::decode(method.output(), response.as_slice())
            .map_err(|e| AdminError::internal(format!("a response did not decode: {e}")))?;
        let options = SerializeOptions::new().skip_default_fields(false);
        let mut out = Vec::new();
        output
            .serialize_with_options(&mut serde_json::Serializer::new(&mut out), &options)
            .map_err(|e| AdminError::internal(format!("a response did not serialize: {e}")))?;
        String::from_utf8(out).map_err(|e| AdminError::internal(e.to_string()))
    }
}

fn method_of(rpc: &str) -> AdminResult<MethodDescriptor> {
    descriptor_pool()
        .get_service_by_name(SERVICE)
        .and_then(|s| s.methods().find(|m| m.name() == rpc))
        .ok_or_else(|| AdminError::internal(format!("the descriptor has no method {rpc}")))
}

/// Runs one RPC on encoded messages: the one place REST and the typed API meet.
async fn invoke(api: &dyn AdminApi, rpc: &str, request: &[u8]) -> AdminResult<Vec<u8>> {
    let bad = |e: prost::DecodeError| AdminError::invalid_argument(e.to_string());
    match rpc {
        "GetVersion" => {
            GetVersionRequest::decode(request).map_err(bad)?;
            Ok(api.get_version().await?.encode_to_vec())
        }
        "GetCapabilities" => {
            GetCapabilitiesRequest::decode(request).map_err(bad)?;
            Ok(api.get_capabilities().await?.encode_to_vec())
        }
        other => Err(AdminError::internal(format!("no handler for {other}"))),
    }
}

/// The bearer token of a request and whether an `Authorization` header was there but unusable.
/// A cookie authenticates nothing here (design §3).
pub(super) fn bearer(headers: &axum::http::HeaderMap) -> (Option<String>, bool) {
    let value = headers.get(header::AUTHORIZATION);
    let token = value
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_owned);
    let malformed = token.is_none() && value.is_some();
    (token, malformed)
}

/// What a guard refusal is on the admin API.
pub(super) fn refusal_error(refusal: Refusal) -> AdminError {
    match refusal {
        Refusal::Unauthenticated => AdminError::new(
            ErrorKind::Unauthenticated,
            "UNAUTHENTICATED.NO_CREDENTIALS",
            "28000",
            "a valid credential is required",
        ),
        Refusal::Denied => AdminError::new(
            ErrorKind::PermissionDenied,
            "KRISHIV_INSUFFICIENT_PERMISSIONS",
            "42501",
            "the principal's role does not allow this",
        ),
        Refusal::Unavailable => AdminError::new(
            ErrorKind::Unavailable,
            "KRISHIV_ADMIN_UNAVAILABLE",
            "08004",
            "authentication is not possible now; retry later",
        ),
        _ => AdminError::internal("the server cannot authenticate this request"),
    }
}

/// The HTTP status of an error kind (design §6.4).
pub(super) fn http_status(kind: ErrorKind) -> StatusCode {
    match kind {
        ErrorKind::InvalidArgument => StatusCode::BAD_REQUEST,
        ErrorKind::Unauthenticated => StatusCode::UNAUTHORIZED,
        ErrorKind::PermissionDenied => StatusCode::FORBIDDEN,
        ErrorKind::NotFound => StatusCode::NOT_FOUND,
        ErrorKind::FailedPrecondition => StatusCode::PRECONDITION_FAILED,
        ErrorKind::AlreadyExists => StatusCode::CONFLICT,
        ErrorKind::ResourceExhausted => StatusCode::PAYLOAD_TOO_LARGE,
        ErrorKind::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        ErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// The request id a response and its error carry: the caller's `x-request-id` when sane, else a
/// fresh one.
pub(super) fn request_id_of(req: &Request) -> String {
    let given = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty() && v.len() <= 128 && v.bytes().all(|b| b.is_ascii_graphic()));
    if let Some(id) = given {
        return id.to_owned();
    }
    let mut raw = [0_u8; 8];
    if getrandom::fill(&mut raw).is_err() {
        return "unknown".to_owned();
    }
    raw.iter().map(|b| format!("{b:02x}")).collect()
}

fn json_response(status: StatusCode, body: String) -> Response {
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

/// The REST error envelope of design §6.4.
pub(super) fn error_response(error: &AdminError, request_id: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "errorClass": error.error_class,
            "sqlState": error.sql_state,
            "messageParameters": error.parameters,
            "message": error.message,
            "requestId": request_id,
        }
    });
    let mut response = json_response(http_status(error.kind), body.to_string());
    if error.kind == ErrorKind::Unauthenticated {
        response
            .headers_mut()
            .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    }
    response
}

impl IntoResponse for AdminError {
    fn into_response(self) -> Response {
        error_response(&self, "")
    }
}
