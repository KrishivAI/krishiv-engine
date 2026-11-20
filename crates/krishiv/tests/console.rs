// SPDX-License-Identifier: Apache-2.0
//! `krishiv::ui`: the console of an embedded session (ADR 0024).
#![cfg(feature = "ui")]
#![expect(clippy::unwrap_used, reason = "tests fail loudly on unexpected errors")]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use krishiv::Session;

/// One HTTP/1.1 GET, the whole response as text.
fn get(addr: SocketAddr, path: &str, token: Option<&str>) -> String {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let auth = token.map_or(String::new(), |t| format!("Authorization: Bearer {t}\r\n"));
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {addr}\r\n{auth}Connection: close\r\n\r\n"
    )
    .unwrap();
    let mut out = String::new();
    stream.read_to_string(&mut out).unwrap();
    out
}

fn wait_stopped(console: &krishiv::Console) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !console.is_stopped() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(console.is_stopped());
}

#[test]
fn ui_serves_the_console_of_the_session_on_loopback_behind_a_token() {
    let session = Session::new().unwrap();
    futures::executor::block_on(async {
        let df = session.sql("SELECT 1 AS seen_in_the_console").await?;
        df.collect().await
    })
    .unwrap();
    let console = krishiv::ui(&session).unwrap();
    let addr = console.addr();
    assert!(addr.ip().is_loopback());
    assert_ne!(addr.port(), 0);
    assert_eq!(
        console.url(),
        format!("http://{addr}/?token={}", console.token())
    );

    let page = get(addr, "/", None);
    assert!(page.starts_with("HTTP/1.1 200"), "{page}");
    assert!(page.contains("<script"));

    assert!(get(addr, "/api/internal/health", None).starts_with("HTTP/1.1 401"));
    let health = get(addr, "/api/internal/health", Some(console.token()));
    assert!(health.starts_with("HTTP/1.1 200"), "{health}");
    assert!(health.contains(r#""mode":"embedded""#), "{health}");
    assert!(health.contains(session.id()), "{health}");

    let history = get(addr, "/api/internal/history", Some(console.token()));
    assert!(history.contains("seen_in_the_console"), "{history}");
}

#[test]
fn the_console_stops_with_the_session_and_on_drop() {
    let session = Session::new().unwrap();
    let console = krishiv::ui(&session).unwrap();
    let addr = console.addr();
    session.close();
    wait_stopped(&console);
    assert!(TcpStream::connect(addr).is_err());
    // A closed session has no console.
    assert!(krishiv::ui(&session).is_err());

    let session = Session::new().unwrap();
    let console = krishiv::ui(&session).unwrap();
    let addr = console.addr();
    drop(console);
    assert!(TcpListener::bind(addr).is_ok());
}

#[test]
fn ui_on_port_takes_the_port_asked_for_and_fails_on_a_taken_one() {
    let session = Session::new().unwrap();
    let free = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let console = krishiv::ui_on_port(&session, free).unwrap();
    assert_eq!(console.addr().port(), free);
    let err = krishiv::ui_on_port(&session, free).unwrap_err();
    assert!(err.message.contains("bind 127.0.0.1"), "{}", err.message);
}
