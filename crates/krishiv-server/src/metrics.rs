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
use axum::extract::Request;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use krishiv_exec::{MetricsSnapshot, prometheus};

#[cfg(test)]
use crate::guard::same_secret;
use crate::guard::{Access, Guard, GuardRequest, GuardSurface, Refusal, TokenGuard, peer_of};

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

/// `/metrics` refused: a JSON 401 naming the scheme and nothing else (no hint whether a token
/// exists, or how it differs from the one sent).
fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [
            (header::CONTENT_TYPE, "application/json"),
            (header::WWW_AUTHENTICATE, "Bearer"),
        ],
        "{\"error\":\"unauthorized\"}",
    )
        .into_response()
}

/// The token of the first `Authorization: Bearer <token>` header (scheme ignoring ASCII case, one
/// space, the rest verbatim), and whether an `Authorization` header was there without one.
fn bearer(headers: &HeaderMap) -> (Option<&str>, bool) {
    let Some(value) = headers.get(header::AUTHORIZATION) else {
        return (None, false);
    };
    let token = value.to_str().ok().and_then(|v| {
        let (scheme, presented) = v.split_once(' ')?;
        scheme.eq_ignore_ascii_case("Bearer").then_some(presented)
    });
    (token, token.is_none())
}

/// Whether `headers` carry `Authorization: Bearer <token>` for exactly `token`. An empty `token`
/// matches nothing: a misconfigured guard refuses everyone.
#[cfg(test)]
fn bearer_matches(headers: &HeaderMap, token: &str) -> bool {
    !token.is_empty()
        && bearer(headers)
            .0
            .is_some_and(|presented| same_secret(presented.as_bytes(), token.as_bytes()))
}

/// A refusal as `/metrics` answers it: JSON, no detail.
fn refused(refusal: Refusal) -> Response {
    let (status, body) = match refusal {
        Refusal::Unauthenticated => return unauthorized(),
        Refusal::Denied => (StatusCode::FORBIDDEN, "{\"error\":\"forbidden\"}"),
        Refusal::Unavailable => (
            StatusCode::SERVICE_UNAVAILABLE,
            "{\"error\":\"unavailable\"}",
        ),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "{\"error\":\"misconfigured\"}",
        ),
    };
    (status, [(header::CONTENT_TYPE, "application/json")], body).into_response()
}

/// The admin/UI router with `GET /metrics`: the exposition of
/// `snapshot()` when `enabled`, an explicit 404 otherwise (the fallback
/// would serve the SPA shell for the extension-less path). No
/// authentication: bind loopback, or use [`router_with_metrics_auth`].
pub fn router_with_metrics(enabled: bool, snapshot: SnapshotProvider) -> Router {
    router_with_metrics_auth(enabled, snapshot, None)
}

/// [`router_with_metrics`] where, with `token` set, a scrape must send
/// `Authorization: Bearer <token>` (compared in constant time); any other request is a 401 and
/// the snapshot is not taken. A disabled `/metrics` is a 404 whatever the request carries, so
/// the path does not tell an unauthenticated caller whether metrics exist when they are off.
pub fn router_with_metrics_auth(
    enabled: bool,
    snapshot: SnapshotProvider,
    token: Option<&str>,
) -> Router {
    crate::router().merge(metrics_route(enabled, snapshot, token))
}

/// The `/metrics` route alone (no fallback), for merging into another
/// router (`krishiv server`'s admin router); auth as
/// [`router_with_metrics_auth`].
pub fn metrics_route(enabled: bool, snapshot: SnapshotProvider, token: Option<&str>) -> Router {
    let guard = token.map(|t| Arc::new(TokenGuard::new(t)) as Arc<dyn Guard>);
    metrics_route_guarded(enabled, snapshot, guard)
}

/// The `/metrics` route behind `guard` ([`Access::Metrics`]: every role scrapes, ADR 0046 §3c),
/// or open with `None`. A refusal takes no snapshot; a disabled `/metrics` is a 404 whatever the
/// request carries.
pub fn metrics_route_guarded(
    enabled: bool,
    snapshot: SnapshotProvider,
    guard: Option<Arc<dyn Guard>>,
) -> Router {
    let router = Router::new();
    if !enabled {
        return router.route("/metrics", get(metrics_disabled));
    }
    router.route(
        "/metrics",
        get(move |request: Request| {
            let (snapshot, guard) = (Arc::clone(&snapshot), guard.clone());
            async move {
                if let Some(guard) = guard {
                    let (token, malformed) = bearer(request.headers());
                    let check = GuardRequest::new(GuardSurface::Metrics, Access::Metrics)
                        .with_token(token)
                        .with_malformed(malformed)
                        .with_peer(peer_of(request.extensions()));
                    if let Err(refusal) = guard.check(check).await {
                        return refused(refusal);
                    }
                }
                metrics(snapshot).await.into_response()
            }
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_bearer_header_for_the_exact_token_matches() {
        let with = |value: &str| {
            let mut h = HeaderMap::new();
            h.insert(
                header::AUTHORIZATION,
                header::HeaderValue::from_bytes(value.as_bytes())
                    .unwrap_or_else(|e| panic!("test header {value:?}: {e}")),
            );
            h
        };
        assert!(bearer_matches(&with("Bearer s3cret"), "s3cret"));
        assert!(bearer_matches(&with("bearer s3cret"), "s3cret"));
        assert!(!bearer_matches(&with("Bearer s3cre"), "s3cret"));
        assert!(!bearer_matches(&with("Bearer s3cret2"), "s3cret"));
        assert!(!bearer_matches(&with("Basic s3cret"), "s3cret"));
        assert!(!bearer_matches(&with("s3cret"), "s3cret"));
        // A tab or a second space after the scheme is not the single space the scheme allows.
        assert!(!bearer_matches(&with("Bearer\ts3cret"), "s3cret"));
        assert!(!bearer_matches(&with("Bearer  s3cret"), "s3cret"));
        // Non-ASCII credentials never match.
        assert!(!bearer_matches(&with("Bearer s3cr\u{e9}t"), "s3cret"));
        // With two Authorization headers only the first is read, so a later one cannot stand in.
        let mut two = with("Bearer wrong");
        two.append(
            header::AUTHORIZATION,
            header::HeaderValue::from_static("Bearer s3cret"),
        );
        assert!(!bearer_matches(&two, "s3cret"));
        assert!(!bearer_matches(&HeaderMap::new(), "s3cret"));
        // An empty token never matches, not even an empty credential.
        assert!(!bearer_matches(&with("Bearer "), ""));
        assert!(!bearer_matches(&HeaderMap::new(), ""));
    }
}
