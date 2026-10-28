// SPDX-License-Identifier: Apache-2.0
//! Stdio conformance: a scripted JSON-RPC client drives the real server
//! over stdio pipes through `initialize` → `notifications/initialized` →
//! `tools/list` → `tools/call`, asserting every message on the wire.

use std::process::Stdio;
use std::time::Duration;

use krishiv_mcp::{HANDSHAKE_VERSION, PINNED_PROTOCOL_VERSION};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};

/// The shim binary serving stdio for these tests (its stdout carries only
/// the JSON-RPC stream, so every non-empty line must parse as JSON).
const SHIM_EXE: &str = env!("CARGO_BIN_EXE_mcp_stdio");
const READ_TIMEOUT: Duration = Duration::from_secs(10);

fn spawn_helper() -> Result<Child, Box<dyn std::error::Error>> {
    let child = Command::new(SHIM_EXE)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()?;
    Ok(child)
}

async fn send_json<W>(writer: &mut W, message: &Value) -> Result<(), Box<dyn std::error::Error>>
where
    W: AsyncWrite + Unpin,
{
    writer
        .write_all(serde_json::to_string(message)?.as_bytes())
        .await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}

async fn read_message<R>(reader: &mut BufReader<R>) -> Result<Value, Box<dyn std::error::Error>>
where
    R: tokio::io::AsyncRead + Unpin,
{
    // The child is a dedicated shim binary, so stdout carries only the
    // JSON-RPC stream: a non-JSON line is server pollution (a banner or
    // log line) that breaks MCP stdio clients, and fails the test.
    loop {
        let mut line = String::new();
        let read: usize = tokio::time::timeout(READ_TIMEOUT, reader.read_line(&mut line)).await??;
        if read == 0 {
            return Err("server closed stdout".into());
        }
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(line.trim()) {
            Ok(value) => return Ok(value),
            Err(e) => {
                return Err(
                    format!("server wrote a non-JSON line to stdout: {line:?} ({e})").into(),
                );
            }
        }
    }
}

async fn read_result<R>(
    reader: &mut BufReader<R>,
    id: u64,
) -> Result<Value, Box<dyn std::error::Error>>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let deadline = tokio::time::Instant::now() + READ_TIMEOUT;
    while tokio::time::Instant::now() < deadline {
        let message = read_message(reader).await?;
        if message.get("id").and_then(Value::as_u64) == Some(id) {
            if message.get("error").is_some() {
                return Err(format!("server answered id {id} with an error: {message}").into());
            }
            return Ok(message);
        }
    }
    Err(format!("no response for id {id}").into())
}

fn initialize(version: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": version,
            "capabilities": {},
            "clientInfo": { "name": "conformance-client", "version": "0.0.0" }
        }
    })
}

async fn shutdown(mut child: Child) {
    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
    if child.id().is_some() {
        let _ = child.kill().await;
    }
}

#[tokio::test]
async fn stdio_round_trip_on_the_handshake_revision() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = spawn_helper()?;
    let mut writer = child.stdin.take().expect("helper stdin");
    let mut reader = BufReader::new(child.stdout.take().expect("helper stdout"));

    send_json(&mut writer, &initialize(HANDSHAKE_VERSION.as_str())).await?;
    let initialized = read_result(&mut reader, 1).await?;
    assert_eq!(
        initialized.pointer("/result/protocolVersion"),
        Some(&json!(HANDSHAKE_VERSION.as_str())),
        "handshake revision must be echoed"
    );
    assert_eq!(
        initialized.pointer("/result/serverInfo/name"),
        Some(&json!("krishiv-mcp"))
    );
    assert!(
        initialized.pointer("/result/capabilities/tools").is_some(),
        "tools capability must be advertised: {initialized}"
    );

    send_json(
        &mut writer,
        &json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
    )
    .await?;

    send_json(
        &mut writer,
        &json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }),
    )
    .await?;
    let listed = read_result(&mut reader, 2).await?;
    let tools = listed
        .pointer("/result/tools")
        .and_then(Value::as_array)
        .expect("tools/list must return a tools array");
    assert!(
        tools
            .iter()
            .any(|tool| tool.get("name") == Some(&json!("server_status"))),
        "server_status must be listed: {listed}"
    );
    assert!(
        tools
            .iter()
            .any(|tool| tool.pointer("/annotations/readOnlyHint") == Some(&json!(true))),
        "server_status must carry the read-only hint: {listed}"
    );

    send_json(
        &mut writer,
        &json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": { "name": "server_status", "arguments": {} }
        }),
    )
    .await?;
    let called = read_result(&mut reader, 3).await?;
    let text = called
        .pointer("/result/content")
        .and_then(Value::as_array)
        .and_then(|content| content.first())
        .and_then(|block| block.get("text"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        text.contains(PINNED_PROTOCOL_VERSION.as_str()),
        "tool output must name the pin, got: {text}"
    );
    assert_eq!(
        called.pointer("/result/structuredContent/protocolVersion"),
        Some(&json!(PINNED_PROTOCOL_VERSION.as_str()))
    );

    drop(writer);
    shutdown(child).await;
    Ok(())
}

#[tokio::test]
async fn stdio_downgrades_an_older_client() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = spawn_helper()?;
    let mut writer = child.stdin.take().expect("helper stdin");
    let mut reader = BufReader::new(child.stdout.take().expect("helper stdout"));

    // 2024-11-05 is outside the supported set: the server answers with the
    // newest revision the handshake can name instead of echoing it.
    send_json(&mut writer, &initialize("2024-11-05")).await?;
    let initialized = read_result(&mut reader, 1).await?;
    assert_eq!(
        initialized.pointer("/result/protocolVersion"),
        Some(&json!(HANDSHAKE_VERSION.as_str())),
        "older client must be moved to the handshake revision"
    );

    send_json(
        &mut writer,
        &json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
    )
    .await?;
    send_json(
        &mut writer,
        &json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }),
    )
    .await?;
    let listed = read_result(&mut reader, 2).await?;
    assert!(
        listed.pointer("/result/tools").is_some(),
        "downgraded session must serve tools/list: {listed}"
    );

    drop(writer);
    shutdown(child).await;
    Ok(())
}

/// Offline conformance note: the official MCP inspector CLI is not
/// vendored and needs network (`npx -y @modelcontextprotocol/inspector`),
/// so it cannot run in CI. To run it manually against this server, point
/// the inspector at the `mcp_stdio` shim binary (`cargo run -p krishiv-mcp
/// --bin mcp_stdio`) and walk `initialize` → `tools/list` →
/// `tools/call server_status`, checking the pin
/// (`PINNED_PROTOCOL_VERSION`) in the tool output. Set `MCP_INSPECTOR_CMD`
/// to have this test execute that command instead of just printing these
/// instructions.
#[test]
#[ignore = "needs the external inspector CLI plus network; see doc comment"]
fn inspector_conformance() {
    match std::env::var("MCP_INSPECTOR_CMD") {
        Err(_) => {
            eprintln!(
                "inspector_conformance: set MCP_INSPECTOR_CMD to the full inspector \
                 invocation (e.g. a shell command driving the inspector CLI against \
                 a stdio shim running KrishivMcpServer::serve_stdio) to execute it."
            );
        }
        Ok(command) => {
            let status = std::process::Command::new("sh")
                .arg("-c")
                .arg(&command)
                .status()
                .expect("run MCP_INSPECTOR_CMD");
            assert!(
                status.success(),
                "inspector command failed with {status}: {command}"
            );
        }
    }
}
