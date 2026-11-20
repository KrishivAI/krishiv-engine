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

use super::core::Core;
use super::rest::{MAX_BODY, bearer, fresh_request_id, refusal_error};
use super::{
    AdminError, AdminService, AdminServiceServer, Call, CancelQueriesByTagRequest,
    CancelQueriesByTagResponse, CancelQueryRequest, CancelQueryResponse, DrainRequest,
    DrainResponse, ErrorKind, GetCapabilitiesRequest, GetCapabilitiesResponse,
    GetDrainStatusRequest, GetDrainStatusResponse, GetReadinessRequest, GetReadinessResponse,
    GetVersionRequest, GetVersionResponse, KillSessionRequest, KillSessionResponse,
    ListQueriesRequest, ListQueriesResponse, ListSessionsRequest, ListSessionsResponse,
    ResumeServingRequest, ResumeServingResponse,
};
use super::{GRPC_PREFIX, Privilege, routes};
use crate::guard::{Access, Caller, Guard, GuardRequest, GuardSurface, peer_leaf_of, peer_of};

struct Service {
    core: Arc<Core>,
}

impl Service {
    /// Runs `rpc` for the caller the guard resolved (it put it in the request's extensions).
    async fn run<Req, Resp>(
        &self,
        rpc: &str,
        request: tonic::Request<Req>,
    ) -> Result<tonic::Response<Resp>, Status>
    where
        Req: prost::Message,
        Resp: prost::Message + Default,
    {
        let Some(caller) = request.extensions().get::<Caller>().cloned() else {
            return Err(status_of(&AdminError::internal(
                "the guard resolved no caller",
            )));
        };
        let call = Call {
            caller,
            transport: "grpc",
            request_id: fresh_request_id(),
        };
        let bytes = self
            .core
            .call(rpc, call, &request.get_ref().encode_to_vec(), None)
            .await
            .map_err(|e| status_of(&e))?;
        let response = Resp::decode(bytes.as_slice()).map_err(|e| {
            status_of(&AdminError::internal(format!(
                "a response did not decode: {e}"
            )))
        })?;
        Ok(tonic::Response::new(response))
    }
}

#[tonic::async_trait]
impl AdminService for Service {
    async fn get_version(
        &self,
        request: tonic::Request<GetVersionRequest>,
    ) -> Result<tonic::Response<GetVersionResponse>, Status> {
        self.run("GetVersion", request).await
    }

    async fn get_capabilities(
        &self,
        request: tonic::Request<GetCapabilitiesRequest>,
    ) -> Result<tonic::Response<GetCapabilitiesResponse>, Status> {
        self.run("GetCapabilities", request).await
    }

    async fn list_sessions(
        &self,
        request: tonic::Request<ListSessionsRequest>,
    ) -> Result<tonic::Response<ListSessionsResponse>, Status> {
        self.run("ListSessions", request).await
    }

    async fn kill_session(
        &self,
        request: tonic::Request<KillSessionRequest>,
    ) -> Result<tonic::Response<KillSessionResponse>, Status> {
        self.run("KillSession", request).await
    }

    async fn list_queries(
        &self,
        request: tonic::Request<ListQueriesRequest>,
    ) -> Result<tonic::Response<ListQueriesResponse>, Status> {
        self.run("ListQueries", request).await
    }

    async fn cancel_query(
        &self,
        request: tonic::Request<CancelQueryRequest>,
    ) -> Result<tonic::Response<CancelQueryResponse>, Status> {
        self.run("CancelQuery", request).await
    }

    async fn cancel_queries_by_tag(
        &self,
        request: tonic::Request<CancelQueriesByTagRequest>,
    ) -> Result<tonic::Response<CancelQueriesByTagResponse>, Status> {
        self.run("CancelQueriesByTag", request).await
    }

    async fn get_readiness(
        &self,
        request: tonic::Request<GetReadinessRequest>,
    ) -> Result<tonic::Response<GetReadinessResponse>, Status> {
        self.run("GetReadiness", request).await
    }

    async fn get_drain_status(
        &self,
        request: tonic::Request<GetDrainStatusRequest>,
    ) -> Result<tonic::Response<GetDrainStatusResponse>, Status> {
        self.run("GetDrainStatus", request).await
    }

    async fn drain(
        &self,
        request: tonic::Request<DrainRequest>,
    ) -> Result<tonic::Response<DrainResponse>, Status> {
        self.run("Drain", request).await
    }

    async fn resume_serving(
        &self,
        request: tonic::Request<ResumeServingRequest>,
    ) -> Result<tonic::Response<ResumeServingResponse>, Status> {
        self.run("ResumeServing", request).await
    }
}

pub(super) fn router(core: Arc<Core>, guard: Arc<dyn Guard>) -> Router {
    let server = AdminServiceServer::new(Service { core })
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
    match guard.authenticate(request).await {
        Ok(caller) => {
            let mut req = req;
            req.extensions_mut().insert(caller);
            next.run(req).await
        }
        Err(refusal) => status_of(&refusal_error(refusal))
            .into_http::<axum::body::Body>()
            .into_response(),
    }
}
