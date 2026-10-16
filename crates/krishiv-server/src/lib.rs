//! Admin and UI API server (layer L6).
//!
//! Serves the embedded SPA at `/` and the versioned REST surface under
//! `/api/v1`. Binds loopback by default (ADR 0024); auth arrives later.

#![forbid(unsafe_code)]

use std::io;
use std::net::SocketAddr;

use axum::Router;
use axum::http::{StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use tokio::net::TcpListener;

/// Crate (and workspace) version reported by `/api/v1/version`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

async fn version() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/json")],
        format!("{{\"version\":\"{VERSION}\"}}"),
    )
}

/// Serves an embedded asset; extension-less paths fall back to the SPA shell.
async fn assets(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let found = krishiv_ui_assets::get(path).or_else(|| {
        let last = path.rsplit('/').next().unwrap_or(path);
        (!last.contains('.'))
            .then(|| krishiv_ui_assets::get(""))
            .flatten()
    });
    match found {
        Some(a) => (
            [(header::CONTENT_TYPE, a.content_type)],
            a.data.into_owned(),
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// The admin/UI router.
pub fn router() -> Router {
    Router::new()
        .route("/api/v1/version", get(version))
        .fallback(assets)
}

/// Serves on an already-bound listener until the future is dropped.
///
/// # Errors
/// Returns an I/O error if the server fails.
pub async fn serve(listener: TcpListener) -> io::Result<()> {
    axum::serve(listener, router()).await
}

/// Binds `addr` and serves on a fresh multi-thread runtime, blocking forever.
/// `on_bound` is called with the actual address (useful for port 0).
///
/// # Errors
/// Returns an I/O error if the runtime, bind or server fails.
pub fn run_blocking(addr: SocketAddr, on_bound: impl FnOnce(SocketAddr)) -> io::Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()?;
    rt.block_on(async {
        let listener = TcpListener::bind(addr).await?;
        on_bound(listener.local_addr()?);
        serve(listener).await
    })
}
