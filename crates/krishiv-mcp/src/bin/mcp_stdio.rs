// SPDX-License-Identifier: Apache-2.0
//! Stdio shim for the MCP conformance tests: serves [`KrishivMcpServer`]
//! on stdio. This is a plain binary, not a test harness, so its stdout
//! carries only the JSON-RPC stream: the test client rejects any non-JSON
//! line as server pollution.

use krishiv_mcp::KrishivMcpServer;

/// Serve stdio until the client closes it.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    KrishivMcpServer.serve_stdio().await?.waiting().await?;
    Ok(())
}
