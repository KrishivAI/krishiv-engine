// SPDX-License-Identifier: Apache-2.0
//! MCP adapter (layer L6; P2 stdio, read-only).
//!
//! [`server::KrishivMcpServer`] is pinned to `2026-07-28`, the newest MCP
//! revision the pinned [`rmcp`] release fully supports, and answers the
//! `initialize` handshake with `2025-11-25`, the newest revision that
//! still has one. See [`protocol`] for the pin and [`server`] for the tools.

#![forbid(unsafe_code)]

pub mod protocol;
pub mod server;

pub use protocol::{HANDSHAKE_VERSION, PINNED_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS};
pub use server::{
    KrishivMcpServer, TOOL_SERVER_STATUS, dispatch_tool, server_status_payload, status_tool,
};
