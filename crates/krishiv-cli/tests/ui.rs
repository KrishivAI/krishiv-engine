// SPDX-License-Identifier: Apache-2.0
//! `krishiv ui --profiles <dir>` (P3-T050): the pages over a stored history, with no server and no engine.
#![cfg(feature = "ui")]
#![expect(
    clippy::unwrap_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::sync::Arc;

use krishiv_session::{EngineRuntime, Session};

fn get(addr: &str, token: &str, path: &str) -> String {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(
        s,
        "GET {path} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {token}\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    assert!(out.starts_with("HTTP/1.1 200"), "{path}: {out}");
    out.split_once("\r\n\r\n").unwrap().1.to_owned()
}

#[test]
fn stored_profiles_open_with_no_server_and_show_their_operators() {
    let dir = tempfile::tempdir().unwrap();
    {
        let rt = EngineRuntime::builder()
            .config("krishiv.history.path", dir.path().to_str().unwrap())
            .build()
            .unwrap();
        let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
        let df = futures::executor::block_on(
            session.sql("SELECT id % 7 AS k, count(*) AS n FROM range(0, 5000) GROUP BY id % 7"),
        )
        .unwrap();
        futures::executor::block_on(df.collect()).unwrap();
        // Dropping the runtime flushes the history writer.
    }
    let mut child = Command::new(env!("CARGO_BIN_EXE_krishiv"))
        .args(["ui", "--profiles"])
        .arg(dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let url = line
        .trim()
        .strip_prefix("krishiv ui: open http://")
        .unwrap_or_else(|| panic!("{line}"));
    let (addr, token) = url.split_once("/?token=").unwrap();
    let history: serde_json::Value =
        serde_json::from_str(&get(addr, token, "/api/internal/history")).unwrap();
    let queries = history["queries"].as_array().unwrap();
    assert_eq!(queries.len(), 1, "{history}");
    assert_eq!(queries[0]["state"], "finished");
    let id = queries[0]["query_id"].as_str().unwrap();
    let detail: serde_json::Value =
        serde_json::from_str(&get(addr, token, &format!("/api/internal/queries/{id}"))).unwrap();
    let ops = detail["operators"].as_array().unwrap();
    assert!(!ops.is_empty(), "{detail}");
    assert!(ops.iter().any(|o| o["compute_ns"].is_u64()), "{detail}");
    child.kill().unwrap();
    child.wait().unwrap();
}

#[test]
fn a_missing_directory_is_a_clean_failure() {
    let out = Command::new(env!("CARGO_BIN_EXE_krishiv"))
        .args(["ui", "--profiles", "/nonexistent/krishiv-profiles"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
}
