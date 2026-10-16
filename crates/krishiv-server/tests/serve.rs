//! Integration test: serves the embedded UI and API fallbacks over a real socket.
#![allow(clippy::unwrap_used, clippy::expect_used)] // integration-test helpers are outside #[test] fns

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn get(addr: std::net::SocketAddr, path: &str) -> String {
    let mut s = TcpStream::connect(addr).await.unwrap();
    let req = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    s.write_all(req.as_bytes()).await.unwrap();
    let mut out = Vec::new();
    tokio::time::timeout(std::time::Duration::from_secs(10), s.read_to_end(&mut out))
        .await
        .expect("read timed out")
        .unwrap();
    String::from_utf8(out).unwrap()
}

#[tokio::test]
async fn serves_index_and_version() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(krishiv_server::serve(listener));

    let root = get(addr, "/").await;
    assert!(root.starts_with("HTTP/1.1 200"), "{root}");
    assert!(root.to_lowercase().contains("content-type: text/html"));
    assert!(root.contains("<div id=\"app\">"));

    let v = get(addr, "/api/v1/version").await;
    assert!(v.starts_with("HTTP/1.1 200"), "{v}");
    assert!(v.to_lowercase().contains("content-type: application/json"));
    assert!(v.contains(&format!("{{\"version\":\"{}\"}}", krishiv_server::VERSION)));

    let missing = get(addr, "/assets/nope.js").await;
    assert!(missing.starts_with("HTTP/1.1 404"), "{missing}");
}
