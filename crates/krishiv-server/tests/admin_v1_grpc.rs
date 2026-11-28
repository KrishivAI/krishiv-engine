// SPDX-License-Identifier: Apache-2.0
//! `krishiv.admin.v1` over a real socket with HTTP/2: a gRPC client calls the service, and the REST
//! side answers the same on the same port (design P3-T066 section 3: one port, two transports).
#![cfg(feature = "admin-api")]

use std::sync::Arc;

use futures::future::{BoxFuture, FutureExt};
use krishiv_server::admin_v1::{
    ADMIN_V1_REVISION, AdminApi, AdminConfig, AdminResult, Call, CancelQueriesByTagRequest,
    CancelQueriesByTagResponse, CancelQueryRequest, CancelQueryResponse, GetCapabilitiesResponse,
    GetVersionRequest, GetVersionResponse, KillSessionRequest, KillSessionResponse,
    ListQueriesRequest, ListQueriesResponse, ListSessionsRequest, ListSessionsResponse, PREVIEW,
    router,
};
use krishiv_server::guard::{Guard, GuardRequest, Refusal};
use tonic::transport::Channel;

struct Api;

impl AdminApi for Api {
    fn get_version(&self, _: &Call) -> BoxFuture<'_, AdminResult<GetVersionResponse>> {
        async {
            Ok(GetVersionResponse {
                version: "1.2.3".to_owned(),
                admin_v1_revision: ADMIN_V1_REVISION,
                preview: PREVIEW,
            })
        }
        .boxed()
    }

    fn get_capabilities(&self, _: &Call) -> BoxFuture<'_, AdminResult<GetCapabilitiesResponse>> {
        async { Ok(GetCapabilitiesResponse::default()) }.boxed()
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

#[tokio::test]
async fn a_grpc_client_and_a_rest_client_share_one_port() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let addr = listener.local_addr().unwrap_or_else(|e| panic!("{e}"));
    let app = router(
        Arc::new(Api),
        Arc::new(Open),
        AdminConfig::for_listener(addr.ip(), Vec::new()),
    );
    let server = tokio::spawn(async move { axum::serve(listener, app).await });

    let channel = Channel::from_shared(format!("http://{addr}"))
        .unwrap_or_else(|e| panic!("{e}"))
        .connect()
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let mut grpc = tonic::client::Grpc::new(channel);
    grpc.ready().await.unwrap_or_else(|e| panic!("{e}"));
    let response: tonic::Response<GetVersionResponse> = grpc
        .unary(
            tonic::Request::new(GetVersionRequest {}),
            "/krishiv.admin.v1.AdminService/GetVersion"
                .parse()
                .unwrap_or_else(|e| panic!("{e}")),
            tonic_prost::ProstCodec::default(),
        )
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(response.get_ref().version, "1.2.3");
    assert_eq!(response.get_ref().admin_v1_revision, ADMIN_V1_REVISION);

    // The same port answers REST (HTTP/1.1) with the same message.
    let mut tcp = tokio::net::TcpStream::connect(addr)
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    tcp.write_all(
        format!("GET /api/admin/v1/version HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n")
            .as_bytes(),
    )
    .await
    .unwrap_or_else(|e| panic!("{e}"));
    let mut text = String::new();
    tcp.read_to_string(&mut text)
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(text.starts_with("HTTP/1.1 200"), "{text}");
    assert!(text.contains("\"version\":\"1.2.3\""), "{text}");
    server.abort();
}
