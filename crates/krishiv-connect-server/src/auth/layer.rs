// SPDX-License-Identifier: Apache-2.0
//! [`AuthLayer`]: the tower layer in front of the Connect service that authenticates each
//! request from its headers and peer address before tonic reads or decodes the body (design
//! P3-T002 §3.1). A refusal is answered here as a gRPC status; an accepted request carries its
//! `Authenticated` caller in the request extensions, where [`super::ConnectAuth::caller`] reads
//! it.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use tonic::codegen::http::{Request, Response};
use tower::Service;

use super::ConnectAuth;
use crate::server::PeerInfo;

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
            match auth.authenticate(request.headers(), peer, leaf).await {
                Ok(caller) => {
                    request.extensions_mut().insert(caller);
                    inner.call(request).await
                }
                Err(status) => Ok(status.into_http()),
            }
        })
    }
}
