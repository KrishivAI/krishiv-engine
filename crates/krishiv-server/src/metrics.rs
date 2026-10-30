// SPDX-License-Identifier: Apache-2.0
//! `GET /metrics` over the engine metrics snapshot (feature
//! `metrics-prometheus`).
//!
//! The route renders the registry snapshot the caller provides as
//! Prometheus text exposition 0.0.4. The exporter only reads snapshots:
//! it holds a provider closure, never the registry, and the render is
//! pure, so scraping never changes a counter. The route exists only when
//! the feature is on and the caller passes `enabled` (the runtime
//! `krishiv.metrics.enabled`); otherwise `/metrics` is an explicit 404,
//! never the SPA shell.

use std::sync::Arc;

use axum::Router;
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::get;
use krishiv_exec::{MetricsSnapshot, prometheus};

/// Provides the snapshot a scrape renders: called once per `GET
/// /metrics`, on the request task. Must be cheap: [`MetricsSnapshot`] is
/// plain data the registry hands over.
pub type SnapshotProvider = Arc<dyn Fn() -> MetricsSnapshot + Send + Sync>;

/// Renders `snapshot` as Prometheus text exposition 0.0.4.
#[must_use]
pub fn render(snapshot: &MetricsSnapshot) -> String {
    prometheus::render(snapshot)
}

async fn metrics(snapshot: SnapshotProvider) -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        render(&snapshot()),
    )
}

/// `/metrics` when metrics are disabled at runtime: a JSON 404, the same
/// shape as the unknown-`/api/*` one.
async fn metrics_disabled() -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        [(header::CONTENT_TYPE, "application/json")],
        "{\"error\":\"not_found\"}",
    )
}

/// The admin/UI router with `GET /metrics`: the exposition of
/// `snapshot()` when `enabled`, an explicit 404 otherwise (the fallback
/// would serve the SPA shell for the extension-less path).
pub fn router_with_metrics(enabled: bool, snapshot: SnapshotProvider) -> Router {
    let router = crate::router();
    if enabled {
        router.route(
            "/metrics",
            get(move || {
                let snapshot = Arc::clone(&snapshot);
                async move { metrics(snapshot).await }
            }),
        )
    } else {
        router.route("/metrics", get(metrics_disabled))
    }
}
