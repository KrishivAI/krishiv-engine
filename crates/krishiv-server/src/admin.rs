// SPDX-License-Identifier: Apache-2.0
//! The admin endpoints of `krishiv server` (design P3-T001 §3.5) and the
//! HTTP limits of the admin listener (threat model T-D7).
//!
//! - `/healthz`: 200 while the process runs; open, no detail (T-A10).
//! - `/readyz` (the simple form, decision 3): 200 `{"status":"ready"}` once
//!   the Connect port is bound and the session manager accepts, 503
//!   `{"status":"not_ready"}` before that and while draining. The full form
//!   is T066's.
//! - Limits: a request timeout (the event stream is exempt), an explicit body
//!   limit and a cap on open connections.

use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use axum::Router;
use axum::extract::{DefaultBodyLimit, Request};
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// The admin listener's limits.
#[derive(Debug, Clone, Copy)]
pub struct HttpLimits {
    /// A request that takes longer is answered 408 (the UI's event stream
    /// is exempt).
    pub request_timeout: Duration,
    /// The largest request body.
    pub body_limit: usize,
    /// Open connections; past it the accept loop waits for one to close.
    pub max_connections: usize,
}

impl Default for HttpLimits {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(30),
            body_limit: 2 * 1024 * 1024,
            max_connections: 256,
        }
    }
}

/// `/healthz` and `/readyz`, answering from `ready`.
pub fn health_routes(ready: Arc<AtomicBool>) -> Router {
    Router::new().route("/healthz", get(healthz)).route(
        "/readyz",
        get(move || {
            let ready = ready.load(Ordering::Acquire);
            async move { readyz(ready) }
        }),
    )
}

async fn healthz() -> Response {
    json(StatusCode::OK, r#"{"status":"ok"}"#)
}

fn readyz(ready: bool) -> Response {
    if ready {
        json(StatusCode::OK, r#"{"status":"ready"}"#)
    } else {
        json(StatusCode::SERVICE_UNAVAILABLE, r#"{"status":"not_ready"}"#)
    }
}

fn json(status: StatusCode, body: &'static str) -> Response {
    (status, [(header::CONTENT_TYPE, "application/json")], body).into_response()
}

/// `app` with the request timeout and body limit of `limits`.
pub fn with_limits(app: Router, limits: HttpLimits) -> Router {
    let timeout = limits.request_timeout;
    app.layer(middleware::from_fn(move |req: Request, next: Next| {
        request_timeout(timeout, req, next)
    }))
    .layer(DefaultBodyLimit::max(limits.body_limit))
}

async fn request_timeout(timeout: Duration, req: Request, next: Next) -> Response {
    // Server-sent events stay open on purpose.
    if req.uri().path().ends_with("/events") {
        return next.run(req).await;
    }
    match tokio::time::timeout(timeout, next.run(req)).await {
        Ok(response) => response,
        Err(_) => StatusCode::REQUEST_TIMEOUT.into_response(),
    }
}

/// Serves `app` on `listener`, at most `max_connections` open at once,
/// until `shutdown` resolves; then stops accepting and lets requests in
/// flight finish (the event streams end with the process).
///
/// # Errors
/// An I/O error of the server.
pub async fn serve_capped(
    listener: TcpListener,
    app: Router,
    max_connections: usize,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> io::Result<()> {
    let listener = CappedListener {
        inner: listener,
        permits: Arc::new(Semaphore::new(
            max_connections.clamp(1, Semaphore::MAX_PERMITS),
        )),
    };
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
}

/// A listener whose connections each hold a permit until they close.
struct CappedListener {
    inner: TcpListener,
    permits: Arc<Semaphore>,
}

impl axum::serve::Listener for CappedListener {
    type Io = CappedStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (CappedStream, SocketAddr) {
        let mut backoff = Duration::from_millis(10);
        loop {
            let Ok(permit) = Arc::clone(&self.permits).acquire_owned().await else {
                // The semaphore is never closed; wait as a closed listener would.
                std::future::pending::<()>().await;
                continue;
            };
            match self.inner.accept().await {
                Ok((stream, addr)) => {
                    return (
                        CappedStream {
                            inner: stream,
                            _permit: permit,
                        },
                        addr,
                    );
                }
                // Out of file descriptors every retry fails: log and back off.
                Err(e) => {
                    tracing::warn!(target: "krishiv::server", error = %e, "admin accept failed");
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(1));
                }
            }
        }
    }

    fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }
}

/// An admin connection holding one of the listener's permits.
#[derive(Debug)]
pub struct CappedStream {
    inner: TcpStream,
    _permit: OwnedSemaphorePermit,
}

impl AsyncRead for CappedStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for CappedStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}
