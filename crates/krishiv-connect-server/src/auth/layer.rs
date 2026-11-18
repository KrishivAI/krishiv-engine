// SPDX-License-Identifier: Apache-2.0
//! [`AuthLayer`]: the tower layer in front of the Connect service that authenticates each
//! request from its headers and peer address before tonic reads or decodes the body (design
//! P3-T002 §3.1). A refusal is answered here as a gRPC status; an accepted request carries its
//! `Authenticated` caller in the request extensions, where [`super::ConnectAuth::caller`] reads
//! it. The layer wraps the whole router, so a service added with
//! `ConnectServer::with_routes` (Flight SQL, P3-T101 §2 H1) passes it too: the request's
//! [`Surface`] comes from its path ([`surface_of_path`], H2) and rides in the extensions beside
//! the caller ([`super::request_surface`]).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use tonic::codegen::http::{Request, Response};
use tower::Service;

use krishiv_session::auth::Surface;

use super::ConnectAuth;
use crate::server::PeerInfo;

/// The path prefix of the Arrow Flight service (`arrow.flight.protocol.FlightService`), which
/// Flight SQL runs on.
const FLIGHT_SERVICE_PREFIX: &str = "/arrow.flight.protocol.FlightService/";

/// The surface a request for `path` is for: `FlightSql` for the Arrow Flight service, `Connect`
/// for anything else (Spark Connect, and paths no service answers, which tonic refuses as
/// `UNIMPLEMENTED` after authentication).
#[must_use]
pub fn surface_of_path(path: &str) -> Surface {
    if path.starts_with(FLIGHT_SERVICE_PREFIX) {
        Surface::FlightSql
    } else {
        Surface::Connect
    }
}

/// Authenticates every request before the wrapped service sees it.
#[derive(Debug, Clone)]
pub struct AuthLayer {
    auth: Arc<ConnectAuth>,
}

impl AuthLayer {
    /// The layer of `auth`.
    #[must_use]
    pub fn new(auth: Arc<ConnectAuth>) -> Self {
        Self { auth }
    }
}

impl<S> tower::Layer<S> for AuthLayer {
    type Service = AuthService<S>;

    fn layer(&self, inner: S) -> AuthService<S> {
        AuthService {
            inner,
            auth: Arc::clone(&self.auth),
        }
    }
}

/// The service [`AuthLayer`] wraps around the Connect router.
#[derive(Debug, Clone)]
pub struct AuthService<S> {
    inner: S,
    auth: Arc<ConnectAuth>,
}

type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for AuthService<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    S::Error: Send + 'static,
    ReqBody: Send + 'static,
    ResBody: Default + Send + 'static,
{
    type Response = Response<ResBody>;
    type Error = S::Error;
    type Future = BoxFuture<Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: Request<ReqBody>) -> Self::Future {
        // The service that was polled ready serves this request; a fresh clone stays behind
        // for the next one (tower's usual clone-and-swap).
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        let auth = Arc::clone(&self.auth);
        Box::pin(async move {
            let info = request.extensions().get::<PeerInfo>().cloned();
            let peer = info.as_ref().and_then(|i| i.remote);
            let leaf = info.as_ref().and_then(|i| i.leaf.as_deref());
            let surface = surface_of_path(request.uri().path());
            match auth
                .authenticate(request.headers(), peer, leaf, surface)
                .await
            {
                Ok(caller) => {
                    request.extensions_mut().insert(caller);
                    request.extensions_mut().insert(surface);
                    inner.call(request).await
                }
                Err(status) => Ok(status.into_http()),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_surface_follows_the_service_path() {
        assert_eq!(
            surface_of_path("/arrow.flight.protocol.FlightService/GetFlightInfo"),
            Surface::FlightSql
        );
        assert_eq!(
            surface_of_path("/spark.connect.SparkConnectService/ExecutePlan"),
            Surface::Connect
        );
        // A look-alike service name is not the Flight service.
        assert_eq!(
            surface_of_path("/arrow.flight.protocol.FlightServiceX/DoGet"),
            Surface::Connect
        );
        assert_eq!(surface_of_path("/"), Surface::Connect);
    }
}
