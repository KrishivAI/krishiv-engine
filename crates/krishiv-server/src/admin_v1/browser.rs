// SPDX-License-Identifier: Apache-2.0
//! Requests from a browser (design §3; TM T-W1, T-W5). With authentication off there is no token,
//! so the shape of a request is the only defence against a web page in the operator's browser:
//! a non-GET request must be `application/json` (REST) or `application/grpc*` (gRPC), which no
//! CORS-simple cross-origin request can be (415), and on a loopback listener a `Host` outside
//! `localhost`, `127.0.0.1`, `[::1]` and the allowed hosts is refused (421, DNS rebinding).
//! `/api/admin/v1` sends no CORS headers and answers no preflight.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use super::rest::{error_response, request_id_of};
use super::{AdminConfig, AdminError, GRPC_PREFIX};

pub(super) async fn enforce(
    State(config): State<Arc<AdminConfig>>,
    req: Request,
    next: Next,
) -> Response {
    let grpc = req.uri().path().starts_with(GRPC_PREFIX);
    if config.loopback && !host_allowed(&req, &config.allowed_hosts) {
        return refuse(
            &req,
            grpc,
            StatusCode::MISDIRECTED_REQUEST,
            "the Host header is not one this listener serves",
        );
    }
    if !matches!(*req.method(), Method::GET | Method::HEAD) && !content_type_ok(&req, grpc) {
        return refuse(
            &req,
            grpc,
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            if grpc {
                "the Content-Type must be application/grpc"
            } else {
                "the Content-Type must be application/json"
            },
        );
    }
    next.run(req).await
}

fn content_type_ok(req: &Request, grpc: bool) -> bool {
    let Some(value) = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
    else {
        return false;
    };
    let essence = value
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if grpc {
        essence == "application/grpc" || essence.starts_with("application/grpc+")
    } else {
        essence == "application/json"
    }
}

/// The host part of `Host` (no port) is a loopback name or one the operator allowed.
fn host_allowed(req: &Request, allowed: &[String]) -> bool {
    let Some(host) = req
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .or_else(|| req.uri().host())
    else {
        return false;
    };
    // `[v6]` or `[v6]:port` for an IPv6 literal, else `name` or `name:port`; anything else is refused.
    let name = if let Some(rest) = host.strip_prefix('[') {
        let Some((inner, tail)) = rest.split_once(']') else {
            return false;
        };
        if !(tail.is_empty() || tail.strip_prefix(':').is_some_and(is_port)) {
            return false;
        }
        format!("[{inner}]")
    } else {
        match host.rsplit_once(':') {
            Some((h, port)) if is_port(port) => h.to_owned(),
            Some(_) => return false,
            None => host.to_owned(),
        }
    };
    let name = name.to_ascii_lowercase();
    matches!(name.as_str(), "localhost" | "127.0.0.1" | "[::1]")
        || allowed.iter().any(|a| a.eq_ignore_ascii_case(&name))
}

/// A port: digits only.
fn is_port(port: &str) -> bool {
    !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit())
}

fn refuse(req: &Request, grpc: bool, status: StatusCode, message: &str) -> Response {
    if grpc {
        return status.into_response();
    }
    let error = AdminError::invalid_argument(message);
    let mut response = error_response(&error, &request_id_of(req));
    *response.status_mut() = status;
    response
}
