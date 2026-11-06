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
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
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

/// Whether `a` and `b` are equal, in time that depends only on their lengths.
fn same_secret(a: &[u8], b: &[u8]) -> bool {
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        diff |= usize::from(a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0));
    }
    // `black_box` keeps the compiler from turning the fold into an early exit. The length is not
    // secret from a timing observer; the token's bytes are not revealed by the time taken.
    std::hint::black_box(diff) == 0
}

/// Whether `headers` carry `Authorization: Bearer <token>` for exactly `token`. An empty `token`
/// matches nothing: a misconfigured guard refuses everyone.
fn bearer_matches(headers: &HeaderMap, token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            let (scheme, presented) = v.split_once(' ')?;
            scheme.eq_ignore_ascii_case("Bearer").then_some(presented)
        })
        .is_some_and(|presented| same_secret(presented.as_bytes(), token.as_bytes()))
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
    let router = crate::router();
    if !enabled {
        return router.route("/metrics", get(metrics_disabled));
    }
    let token: Option<Arc<str>> = token.map(Arc::from);
    router.route(
        "/metrics",
        get(move |headers: HeaderMap| {
            let (snapshot, token) = (Arc::clone(&snapshot), token.clone());
            async move {
                match token {
                    Some(t) if !bearer_matches(&headers, &t) => unauthorized(),
                    _ => metrics(snapshot).await.into_response(),
                }
            }
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secrets_compare_by_content_and_length() {
        assert!(same_secret(b"abc", b"abc"));
        assert!(!same_secret(b"abc", b"abd"));
        assert!(!same_secret(b"abc", b"abcd"));
        assert!(!same_secret(b"abcd", b"abc"));
        assert!(same_secret(b"", b""));
    }

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
