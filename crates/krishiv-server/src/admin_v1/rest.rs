// SPDX-License-Identifier: Apache-2.0
//! The REST side of `krishiv.admin.v1`: axum routes built from the route table, bodies mapped
//! to and from the proto3 JSON of the descriptor, and the error envelope (design §6.2, §6.4).

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::{RawPathParams, Request};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{MethodRouter, delete, get, patch, post, put};
use prost::Message;
use prost_reflect::{DynamicMessage, MethodDescriptor, SerializeOptions};

use super::core::Core;
use super::{
    AdminError, AdminResult, Call, ErrorKind, Privilege, REST_PREFIX, descriptor_pool, routes,
};
use crate::guard::{Access, Guard, GuardRequest, GuardSurface, Refusal, peer_leaf_of, peer_of};

/// A request body over this is refused (design §6.2; a group document is the largest request).
pub(super) const MAX_BODY: usize = 1 << 20;

const SERVICE: &str = "krishiv.admin.v1.AdminService";

struct Rest {
    core: Arc<Core>,
    guard: Arc<dyn Guard>,
}

pub(super) fn router(core: Arc<Core>, guard: Arc<dyn Guard>) -> Router {
    let state = Arc::new(Rest { core, guard });
    let mut router = Router::new();
    for route in routes() {
        let rpc = route.rpc.clone();
        let privilege = route.privilege;
        let (axum_path, suffix) = axum_path(&route.path);
        let state = Arc::clone(&state);
        let handler = move |params: RawPathParams, req: Request| {
            let state = Arc::clone(&state);
            let rpc = rpc.clone();
            let suffix = suffix.clone();
            let params: Vec<(String, String)> = params
                .iter()
                .map(|(k, v)| (k.to_owned(), v.to_owned()))
                .collect();
            async move { state.call(&rpc, privilege, &suffix, params, req).await }
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
            router = router.route(&format!("{REST_PREFIX}{axum_path}"), method);
        }
    }
    router
}

/// The axum path of a documented path and the `:verb` suffix its last parameter carries: a custom
/// method such as `/sessions/{session_id}:kill` is routed as `/sessions/{session_id}` (the
/// parameter captures the whole segment) and the suffix is checked and stripped by the handler.
fn axum_path(path: &str) -> (String, String) {
    let mut suffix = String::new();
    let rebuilt: Vec<String> = path
        .split('/')
        .map(
            |segment| match segment.strip_prefix('{').and_then(|r| r.split_once('}')) {
                Some((name, rest)) if rest.starts_with(':') => {
                    rest.clone_into(&mut suffix);
                    format!("{{{name}}}")
                }
                _ => segment.to_owned(),
            },
        )
        .collect();
    (rebuilt.join("/"), suffix)
}

/// Percent-decodes a query component (`+` is a space).
fn decode_component(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' => {
                let hex = text.get(i + 1..i + 3)?;
                out.push(u8::from_str_radix(hex, 16).ok()?);
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

/// Sets request field `name` (its proto or its JSON name) to `value` in the JSON object `into`;
/// a repeated field collects every value given.
fn put_field(
    into: &mut serde_json::Map<String, serde_json::Value>,
    message: &prost_reflect::MessageDescriptor,
    name: &str,
    value: String,
) -> AdminResult<()> {
    let field = message
        .get_field_by_name(name)
        .or_else(|| message.get_field_by_json_name(name))
        .ok_or_else(|| AdminError::invalid_argument(format!("unknown parameter {name}")))?;
    let key = field.json_name().to_owned();
    if field.is_list() {
        let entry = into
            .entry(key)
            .or_insert_with(|| serde_json::Value::Array(Vec::new()));
        if let Some(list) = entry.as_array_mut() {
            list.push(serde_json::Value::String(value));
        }
    } else {
        into.insert(key, serde_json::Value::String(value));
    }
    Ok(())
}

impl Rest {
    async fn call(
        &self,
        rpc: &str,
        privilege: Privilege,
        suffix: &str,
        params: Vec<(String, String)>,
        req: Request,
    ) -> Response {
        let request_id = request_id_of(&req);
        match self
            .run(rpc, privilege, suffix, params, req, &request_id)
            .await
        {
            Ok(body) => json_response(StatusCode::OK, body),
            Err(e) => error_response(&e, &request_id),
        }
    }

    async fn run(
        &self,
        rpc: &str,
        privilege: Privilege,
        suffix: &str,
        params: Vec<(String, String)>,
        req: Request,
        request_id: &str,
    ) -> AdminResult<String> {
        let (token, malformed) = bearer(req.headers());
        let access = match privilege {
            Privilege::Authenticated => Access::Authenticated,
        };
        let guard_request = GuardRequest::new(GuardSurface::Admin, access)
            .with_token(token.as_deref())
            .with_malformed(malformed)
            .with_peer(peer_of(req.extensions()))
            .with_peer_leaf(peer_leaf_of(req.extensions()));
        let caller = self
            .guard
            .authenticate(guard_request)
            .await
            .map_err(refusal_error)?;
        let header_id = req
            .headers()
            .get("idempotency-key")
            .and_then(|v| v.to_str().ok())
            .filter(|v| !v.is_empty() && v.len() <= 128)
            .map(str::to_owned);
        let query = req.uri().query().map(str::to_owned);
        let bytes = axum::body::to_bytes(req.into_body(), MAX_BODY)
            .await
            .map_err(|_| {
                AdminError::new(
                    ErrorKind::ResourceExhausted,
                    "KRISHIV_ADMIN_REQUEST_TOO_LARGE",
                    "54000",
                    format!("The admin API request body is over {MAX_BODY} bytes."),
                )
                .with_parameter("limit", MAX_BODY.to_string())
            })?;
        let method = method_of(rpc)?;
        let mut object = if bytes.iter().all(u8::is_ascii_whitespace) {
            serde_json::Map::new()
        } else {
            match serde_json::from_slice::<serde_json::Value>(&bytes) {
                Ok(serde_json::Value::Object(o)) => o,
                Ok(_) => {
                    return Err(AdminError::invalid_argument(
                        "the request body must be a JSON object",
                    ));
                }
                Err(e) => {
                    return Err(AdminError::invalid_argument(format!(
                        "the request body is not valid JSON: {e}"
                    )));
                }
            }
        };
        // Query parameters and path parameters fill request fields (design §6.2).
        for pair in query
            .iter()
            .flat_map(|q| q.split('&'))
            .filter(|p| !p.is_empty())
        {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            let (Some(k), Some(v)) = (decode_component(k), decode_component(v)) else {
                return Err(AdminError::invalid_argument(
                    "a query parameter is not valid",
                ));
            };
            put_field(&mut object, &method.input(), &k, v)?;
        }
        for (name, value) in params {
            let value = match value.strip_suffix(suffix) {
                Some(v) if !suffix.is_empty() => v.to_owned(),
                _ if suffix.is_empty() => value,
                _ => {
                    return Err(AdminError::new(
                        ErrorKind::NotFound,
                        "KRISHIV_ADMIN_INVALID_REQUEST",
                        "22023",
                        "The admin API request is not valid: no such route.",
                    ));
                }
            };
            let Some(value) = decode_component(&value.replace('+', "%2B")) else {
                return Err(AdminError::invalid_argument(
                    "a path parameter is not valid",
                ));
            };
            put_field(&mut object, &method.input(), &name, value)?;
        }
        // Unknown fields are refused (REST only; design §6.2): a typo guard for hand-written JSON.
        let message =
            DynamicMessage::deserialize(method.input(), serde_json::Value::Object(object))
                .map_err(|e| {
                    AdminError::invalid_argument(format!("the request is not valid: {e}"))
                })?;
        let call = Call {
            caller,
            transport: "rest",
            request_id: request_id.to_owned(),
        };
        let response = self
            .core
            .call(rpc, call, &message.encode_to_vec(), header_id.as_deref())
            .await?;
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

/// The bearer token of a request and whether an `Authorization` header was there but unusable.
/// A cookie authenticates nothing here (design §3).
pub(super) fn bearer(headers: &axum::http::HeaderMap) -> (Option<String>, bool) {
    let value = headers.get(header::AUTHORIZATION);
    let token = value
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            // The scheme is case-insensitive (RFC 9110 section 11.1).
            let (scheme, rest) = v.split_once(' ')?;
            scheme.eq_ignore_ascii_case("bearer").then_some(rest)
        })
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
            "KRISHIV_ADMIN_UNAUTHENTICATED",
            "28000",
            "A valid credential is required for the admin API.",
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
            "The admin API cannot answer now: authentication is not possible, retry later.",
        )
        .with_parameter("reason", "authentication is not possible"),
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
    fresh_request_id()
}

/// A fresh request id: 8 random bytes in hex.
pub(super) fn fresh_request_id() -> String {
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

/// The middleware of [`super::guard_version_alias`].
pub(super) async fn guard_version_alias(
    axum::extract::State(guard): axum::extract::State<Arc<dyn Guard>>,
    req: Request,
    next: axum::middleware::Next,
) -> Response {
    if req.uri().path() != "/api/v1/version" {
        return next.run(req).await;
    }
    let (token, malformed) = bearer(req.headers());
    let request = GuardRequest::new(GuardSurface::Admin, Access::Authenticated)
        .with_token(token.as_deref())
        .with_malformed(malformed)
        .with_peer(peer_of(req.extensions()))
        .with_peer_leaf(peer_leaf_of(req.extensions()));
    match guard.check(request).await {
        Ok(()) => next.run(req).await,
        Err(refusal) => error_response(&refusal_error(refusal), &request_id_of(&req)),
    }
}
