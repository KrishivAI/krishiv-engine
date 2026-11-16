// SPDX-License-Identifier: Apache-2.0
//! The gRPC side of `krishiv.admin.v1`: the generated tonic service over [`AdminApi`], behind the
//! admin guard, with admin.v1's own status mapping (design §6.4: real status codes and one
//! `google.rpc.ErrorInfo`, not Spark Connect's encoding).

use std::sync::Arc;

use axum::Router;
use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use tonic::{Code, Status};

use super::rest::{MAX_BODY, bearer, refusal_error};
use super::{
    AdminApi, AdminError, AdminService, AdminServiceServer, ErrorKind, GetCapabilitiesRequest,
    GetCapabilitiesResponse, GetVersionRequest, GetVersionResponse,
};
use super::{GRPC_PREFIX, Privilege, routes};
use crate::guard::{Access, Guard, GuardRequest, GuardSurface, peer_leaf_of, peer_of};

struct Service {
    api: Arc<dyn AdminApi>,
}

#[tonic::async_trait]
impl AdminService for Service {
    async fn get_version(
        &self,
        _: tonic::Request<GetVersionRequest>,
    ) -> Result<tonic::Response<GetVersionResponse>, Status> {
        Ok(tonic::Response::new(
            self.api.get_version().await.map_err(|e| status_of(&e))?,
        ))
    }

    async fn get_capabilities(
        &self,
        _: tonic::Request<GetCapabilitiesRequest>,
    ) -> Result<tonic::Response<GetCapabilitiesResponse>, Status> {
        Ok(tonic::Response::new(
            self.api
                .get_capabilities()
                .await
                .map_err(|e| status_of(&e))?,
        ))
    }
}

pub(super) fn router(api: Arc<dyn AdminApi>, guard: Arc<dyn Guard>) -> Router {
    let server = AdminServiceServer::new(Service { api })
        .max_decoding_message_size(MAX_BODY)
        .max_encoding_message_size(MAX_BODY * 16);
    // `route_service`, not tonic's `Routes`: that router carries a catch-all `unimplemented`
    // fallback, which would collide with the UI router's fallback when merged, and would answer
    // every unknown admin path as gRPC instead of 404.
    Router::new()
        .route_service(&format!("{GRPC_PREFIX}{{*rest}}"), server)
        .layer(axum::middleware::from_fn_with_state(guard, authorize))
}

/// The gRPC code of an error kind (design §6.4).
fn code_of(kind: ErrorKind) -> Code {
    match kind {
        ErrorKind::InvalidArgument => Code::InvalidArgument,
        ErrorKind::Unauthenticated => Code::Unauthenticated,
        ErrorKind::PermissionDenied => Code::PermissionDenied,
        ErrorKind::NotFound => Code::NotFound,
        ErrorKind::FailedPrecondition => Code::FailedPrecondition,
        ErrorKind::AlreadyExists => Code::AlreadyExists,
        ErrorKind::ResourceExhausted => Code::ResourceExhausted,
        ErrorKind::Unavailable => Code::Unavailable,
        ErrorKind::Internal => Code::Internal,
    }
}

/// The status of an admin error: its code, the message, and the class as ASCII metadata
/// (`krishiv-error-*`, `krishiv-param-*`). These keys stand in for the `google.rpc.ErrorInfo`
/// detail design section 6.4 specifies, until PR-11a vendors that type; they are preview-only
/// and go away with it.
pub(super) fn status_of(error: &AdminError) -> Status {
    let mut status = Status::new(code_of(error.kind), error.message.clone());
    let meta = status.metadata_mut();
    let mut put = |key: &'static str, value: &str| {
        if let Ok(v) = value.parse() {
            meta.insert(key, v);
        }
    };
    put("krishiv-error-domain", "krishiv.admin.v1");
    put("krishiv-error-reason", error.reason());
    put("krishiv-error-class", &error.error_class);
    put("krishiv-sql-state", &error.sql_state);
    for (name, value) in &error.parameters {
        if let (Ok(key), Ok(v)) = (
            format!("krishiv-param-{}", name.to_ascii_lowercase())
                .parse::<tonic::metadata::MetadataKey<tonic::metadata::Ascii>>(),
            value.parse(),
        ) {
            meta.insert(key, v);
        }
    }
    status
}

/// The guard in front of every gRPC call. A refusal is a trailers-only gRPC response (HTTP 200
/// and `grpc-status`), which is what a gRPC client can read.
async fn authorize(State(guard): State<Arc<dyn Guard>>, req: Request, next: Next) -> Response {
    // What the RPC needs comes from the route table, as on REST (one matrix for both transports);
    // a path with no row is refused, never served.
    let privilege = req
        .uri()
        .path()
        .strip_prefix(GRPC_PREFIX)
        .and_then(|rpc| routes().iter().find(|r| r.rpc == rpc))
        .map(|r| r.privilege);
    let Some(privilege) = privilege else {
        return Status::new(Code::Unimplemented, "unknown method")
            .into_http::<axum::body::Body>()
            .into_response();
    };
    let access = match privilege {
        Privilege::Authenticated => Access::Authenticated,
    };
    let (token, malformed) = bearer(req.headers());
    let request = GuardRequest::new(GuardSurface::Admin, access)
        .with_token(token.as_deref())
        .with_malformed(malformed)
        .with_peer(peer_of(req.extensions()))
        .with_peer_leaf(peer_leaf_of(req.extensions()));
    match guard.check(request).await {
        Ok(()) => next.run(req).await,
        Err(refusal) => status_of(&refusal_error(refusal))
            .into_http::<axum::body::Body>()
            .into_response(),
    }
}
