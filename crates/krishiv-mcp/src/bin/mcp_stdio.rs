// SPDX-License-Identifier: Apache-2.0
//! Stdio shim for the MCP conformance tests: serves [`KrishivMcpServer`]
//! on stdio. This is a plain binary, not a test harness, so its stdout
//! carries only the JSON-RPC stream: the test client rejects any non-JSON
//! line as server pollution.

use krishiv_mcp::{KrishivMcpServer, McpConfig};

/// Serve stdio until the client closes it.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `--data <dir>` serves the read-only query tools over that directory.
    let mut args = std::env::args().skip(1);
    let server = match (args.next().as_deref(), args.next()) {
        (Some("--data"), Some(dir)) => KrishivMcpServer::with_tools(McpConfig::new(dir)),
        _ => KrishivMcpServer::default(),
    };
    server.serve_stdio().await?.waiting().await?;
    Ok(())
}
