// SPDX-License-Identifier: Apache-2.0
//! P3-T071 over a real socket: only the embeddable view is framable and only by the listed origins, a signed
//! link opens one query's detail without a token, and nothing else.
#![cfg(feature = "session")]
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;
use std::time::Duration;

use krishiv_server::embed::EmbedPolicy;
use krishiv_server::guard::TokenGuard;
use krishiv_server::internal::router_with_ui_policy;
use krishiv_server::pages::UiPolicy;
use krishiv_server::session_source::SessionSource;
use krishiv_session::EngineRuntime;
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const TOKEN: &str = "embed-token";

async fn start(origins: &[&str]) -> std::net::SocketAddr {
    start_on(EngineRuntime::builder().build().unwrap(), origins).await
}

async fn start_on(rt: Arc<EngineRuntime>, origins: &[&str]) -> std::net::SocketAddr {
    let source = SessionSource::new(rt).unwrap();
    let embed = EmbedPolicy::new(
        origins.iter().map(|o| (*o).to_owned()).collect(),
        false,
        Duration::from_secs(600),
    );
    let app = router_with_ui_policy(
        Arc::new(source),
        Arc::new(TokenGuard::new(TOKEN)),
        UiPolicy::loopback().with_embed(embed),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await });
    addr
}

async fn get(addr: std::net::SocketAddr, path: &str, auth: bool) -> (u16, String, String) {
    let mut s = TcpStream::connect(addr).await.unwrap();
    let a = if auth {
        format!("Authorization: Bearer {TOKEN}\r\n")
    } else {
        String::new()
    };
    let req = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n{a}Connection: close\r\n\r\n");
    s.write_all(req.as_bytes()).await.unwrap();
    let mut out = Vec::new();
    tokio::time::timeout(Duration::from_secs(20), s.read_to_end(&mut out))
        .await
        .expect("read timed out")
        .unwrap();
    let text = String::from_utf8(out).unwrap();
    let (head, body) = text.split_once("\r\n\r\n").unwrap();
    (
        head.split_whitespace().nth(1).unwrap().parse().unwrap(),
        head.to_lowercase(),
        body.to_owned(),
    )
}

fn header<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.lines()
        .find_map(|l| l.strip_prefix(&format!("{name}: ")))
}

#[tokio::test]
async fn by_default_no_page_can_be_framed() {
    let addr = start(&[]).await;
    for path in ["/", "/?embed=1", "/queries/q1/profile?embed=1"] {
        let (_, head, _) = get(addr, path, false).await;
        assert!(
            header(&head, "content-security-policy")
                .unwrap()
                .contains("frame-ancestors 'none'"),
            "{path}"
        );
        assert_eq!(header(&head, "x-frame-options"), Some("deny"), "{path}");
    }
}

#[tokio::test]
async fn only_the_embed_view_is_framable_and_only_by_the_listed_origins() {
    let addr = start(&["https://console.example.com", "https://b.example.org:8443"]).await;
    let (_, head, _) = get(addr, "/queries/q1/profile?embed=1&x=1", false).await;
    let csp = header(&head, "content-security-policy").unwrap();
    assert!(
        csp.contains("frame-ancestors https://console.example.com https://b.example.org:8443;"),
        "{csp}"
    );
    assert!(!csp.contains("frame-ancestors 'none'"), "{csp}");
    assert_eq!(header(&head, "x-frame-options"), None);
    // The same page without the embed flag, other pages and the API stay unframable.
    for path in [
        "/queries/q1/profile",
        "/",
        "/?embed=0",
        "/sql?embed=1",
        "/queries/q1/profile?emb%65d=0&embed=1",
        "/queries/q1?embed=1",
        "/queries/q1/profile/x?embed=1",
        "/queries//profile?embed=1",
        "/sql?embed=0&embed=1",
        "/queries/q1/profile?embed=0&embed=1",
        "/queries/q1/profile?embed=1&embed=1",
        "/?notembed=1",
        "/api/internal/embed/config?embed=1",
    ] {
        let (_, head, _) = get(addr, path, false).await;
        assert!(
            header(&head, "content-security-policy")
                .unwrap()
                .contains("frame-ancestors 'none'"),
            "{path}"
        );
        assert_eq!(header(&head, "x-frame-options"), Some("deny"), "{path}");
    }
}

#[tokio::test]
async fn the_embed_config_lists_the_origins_without_a_token() {
    let addr = start(&["https://console.example.com"]).await;
    let (status, _, body) = get(addr, "/api/internal/embed/config", false).await;
    assert_eq!(status, 200);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["allowed_origins"][0], "https://console.example.com");
    assert_eq!(v["hide_attribution"], false);
}

#[tokio::test]
async fn a_link_needs_the_token_to_issue_and_nothing_to_open() {
    let addr = start(&["https://console.example.com"]).await;
    let (status, ..) = get(addr, "/api/internal/queries/q1/embed-link", false).await;
    assert_eq!(status, 401);
    let (status, _, body) = get(addr, "/api/internal/queries/q1/embed-link?ttl=60", true).await;
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let url = v["url"].as_str().unwrap();
    assert!(url.starts_with("/queries/q1/profile?embed=1&exp="), "{url}");
    let query = url.split_once('?').unwrap().1;
    // The link opens the data of its query (here a query the engine never ran: 404, not 401)...
    let (status, ..) = get(
        addr,
        &format!("/api/internal/embed/queries/q1?{query}"),
        false,
    )
    .await;
    assert_eq!(status, 404);
    // ...and no other query, no tampered expiry, no missing signature.
    let (status, ..) = get(
        addr,
        &format!("/api/internal/embed/queries/q2?{query}"),
        false,
    )
    .await;
    assert_eq!(status, 401);
    let tampered = query.replacen("exp=", "exp=9", 1);
    let (status, ..) = get(
        addr,
        &format!("/api/internal/embed/queries/q1?{tampered}"),
        false,
    )
    .await;
    assert_eq!(status, 401);
    let (status, ..) = get(addr, "/api/internal/embed/queries/q1", false).await;
    assert_eq!(status, 401);
    // The link is no credential for any other route.
    for path in [
        "/api/internal/queries/q1",
        "/api/internal/queries",
        "/api/internal/environment",
    ] {
        let (status, ..) = get(addr, &format!("{path}?{query}"), false).await;
        assert_eq!(status, 401, "{path}");
    }
}

#[tokio::test]
async fn the_life_is_capped_and_the_theme_is_checked_when_the_link_is_issued() {
    let addr = start(&[]).await;
    let (_, _, body) = get(addr, "/api/internal/queries/q1/embed-link?ttl=999999", true).await;
    let v: Value = serde_json::from_str(&body).unwrap();
    let exp = v["expires_at"].as_u64().unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    assert!(exp <= now + 600 && exp + 5 >= now + 600, "{exp} vs {now}");
    let (status, ..) = get(addr, "/api/internal/queries/q1/embed-link?ttl=0", true).await;
    assert_eq!(status, 400);
    let good =
        "%7B%22mode%22%3A%22dark%22%2C%22tokens%22%3A%7B%22accent%22%3A%22%23ff8800%22%7D%7D";
    let (status, _, body) = get(
        addr,
        &format!("/api/internal/queries/q1/embed-link?theme={good}"),
        true,
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("&theme="), "{body}");
    let bad = "%7B%22tokens%22%3A%7B%22accent%22%3A%22url(https%3A%2F%2Fx)%22%7D%7D";
    let (status, ..) = get(
        addr,
        &format!("/api/internal/queries/q1/embed-link?theme={bad}"),
        true,
    )
    .await;
    assert_eq!(status, 400);
}

/// The paths of `ui/routes.json`, `{id}` replaced by `q1`.
fn manifest_paths() -> Vec<(String, String, bool)> {
    let dir = env!("CARGO_MANIFEST_DIR");
    let text = std::fs::read_to_string(format!("{dir}/../../ui/routes.json")).unwrap();
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["version"], 1);
    v["routes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let p = r["path"].as_str().unwrap().to_owned();
            (
                p.clone(),
                p.replace("{id}", "q1"),
                r["embeddable"].as_bool().unwrap(),
            )
        })
        .collect()
}

#[tokio::test]
async fn every_route_of_the_manifest_serves_the_page_and_its_assets() {
    let addr = start(&[]).await;
    let (_, _, root) = get(addr, "/", false).await;
    let js = root
        .split("./assets/")
        .nth(1)
        .and_then(|r| r.split('"').next())
        .expect("the shell links a script");
    for (_, path, _) in manifest_paths() {
        let (status, head, body) = get(addr, &path, false).await;
        assert_eq!(status, 200, "{path}");
        assert!(
            head.contains("text/html") && body.contains("<div id=\"app\">"),
            "{path}"
        );
        // The shell's relative asset links resolve from a deep link too.
        let prefix = path.rsplit_once('/').map_or("", |(p, _)| p).to_owned();
        let (status, head, _) = get(addr, &format!("{prefix}/assets/{js}"), false).await;
        assert_eq!(status, 200, "{prefix}/assets/{js}");
        assert!(head.contains("javascript"), "{head}");
    }
}

#[test]
fn the_manifest_and_the_pages_router_agree() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let routes_ts = std::fs::read_to_string(format!("{dir}/../../ui/src/routes.ts")).unwrap();
    let paths = manifest_paths();
    for (pattern, _, _) in &paths {
        assert!(
            routes_ts.contains(&format!("\"{pattern}\"")),
            "{pattern} is not in routes.ts"
        );
    }
    // routes.ts lists no route the manifest does not.
    let listed = routes_ts.matches("  \"/").count();
    assert_eq!(
        listed,
        paths.len(),
        "routes.ts and routes.json list different routes"
    );
    // Only the profile view is embeddable.
    let embeddable: Vec<_> = paths.iter().filter(|p| p.2).map(|p| p.0.as_str()).collect();
    assert_eq!(embeddable, ["/queries/{id}/profile"]);
}

/// A link for a query that ran returns that query's detail (its operators), and the same link is refused once its
/// life has passed.
#[tokio::test]
async fn a_signed_link_opens_a_finished_query_detail_until_it_expires() {
    let rt = EngineRuntime::builder().build().unwrap();
    let session = krishiv_session::Session::builder()
        .runtime(Arc::clone(&rt))
        .build()
        .unwrap();
    session
        .sql("SELECT id % 3 AS k, count(*) AS n FROM range(0, 300) GROUP BY id % 3")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let addr = start_on(rt, &["https://console.example.com"]).await;
    let (status, _, body) = get(addr, "/api/internal/history", true).await;
    assert_eq!(status, 200, "{body}");
    let queries: Value = serde_json::from_str(&body).unwrap();
    let id = queries["queries"][0]["query_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let issue = |ttl: u32| format!("/api/internal/queries/{id}/embed-link?ttl={ttl}");
    let link_query = |body: &str| {
        let v: Value = serde_json::from_str(body).unwrap();
        v["url"]
            .as_str()
            .unwrap()
            .split_once('?')
            .unwrap()
            .1
            .to_owned()
    };
    let (_, _, body) = get(addr, &issue(60), true).await;
    let query = link_query(&body);
    let (status, _, body) = get(
        addr,
        &format!("/api/internal/embed/queries/{id}?{query}"),
        false,
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let detail: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(detail["query"]["query_id"], Value::String(id.clone()));
    assert!(
        !detail["operators"].as_array().unwrap().is_empty(),
        "{detail}"
    );
    // A one-second link works now and is refused after its second.
    let (_, _, body) = get(addr, &issue(1), true).await;
    let short = link_query(&body);
    let path = format!("/api/internal/embed/queries/{id}?{short}");
    assert_eq!(get(addr, &path, false).await.0, 200);
    tokio::time::sleep(Duration::from_millis(2500)).await;
    assert_eq!(get(addr, &path, false).await.0, 401);
}
