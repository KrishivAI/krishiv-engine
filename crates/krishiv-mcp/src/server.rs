// SPDX-License-Identifier: Apache-2.0
//! Read-only MCP server pinned to [`PINNED_PROTOCOL_VERSION`].
//!
//! Lifecycle handling is delegated to [`rmcp`]'s default `initialize`
//! implementation, which echoes a requested revision only when this server
//! lists it in [`ServerHandler::supported_protocol_versions`]; anything
//! else over `initialize` falls back to [`HANDSHAKE_VERSION`](crate::protocol::HANDSHAKE_VERSION),
//! and per-request `_meta` versions outside the supported list are rejected
//! with `UNSUPPORTED_PROTOCOL_VERSION`. That is the spec's negotiation rule,
//! not a local reimplementation: this crate only narrows the supported set.
//!
//! Tools are pure functions of the request: the server holds no session
//! state, so the stateless `2026-07-28` revision and the session-based
//! `2025-11-25` revision behave identically. Identity, caps and audit must
//! come from the request or the transport, never from stored state.

use std::borrow::Cow;
use std::sync::Arc;

use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation,
    ListToolsResult, PaginatedRequestParams, ProtocolVersion, ServerCapabilities, ServerConfig,
    Tool, ToolAnnotations,
};
use rmcp::service::{RequestContext, RoleServer, RunningService, ServerInitializeError};
use rmcp::{ErrorData as McpError, ServerHandler, ServiceExt};

use crate::protocol::{PINNED_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS};
use crate::tools::{McpConfig, Tools, definitions};

/// Name this server reports in `initialize` results.
pub const SERVER_NAME: &str = "krishiv-mcp";

/// Version this server reports in `initialize` results.
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The single read-only tool served in this change.
///
/// Later phases add the catalog and query tools from ADR 0025; each of
/// those must stay a pure function of its arguments to keep the stateless
/// guarantee.
pub const TOOL_SERVER_STATUS: &str = "server_status";

/// Read-only MCP server. Holds no per-client state: safe to share across
/// connections and to serve revisions with and without sessions.
#[derive(Debug, Clone, Default)]
pub struct KrishivMcpServer {
    tools: Option<Arc<Tools>>,
}

impl KrishivMcpServer {
    /// A server that also serves the read-only query tools over `config`.
    #[must_use]
    pub fn with_tools(config: McpConfig) -> Self {
        Self {
            tools: Some(Arc::new(Tools::new(config))),
        }
    }

    /// Serve this server on stdio (`initialize` then framed JSON-RPC).
    ///
    /// # Errors
    ///
    /// Returns [`ServerInitializeError`] when the transport closes before
    /// the handshake completes or the handshake itself fails.
    #[expect(
        clippy::result_large_err,
        reason = "rmcp owns the error type and callers match on its variants; keep it unboxed"
    )]
    pub async fn serve_stdio(
        self,
    ) -> Result<RunningService<RoleServer, Self>, ServerInitializeError> {
        self.serve(rmcp::transport::stdio()).await
    }
}

/// The `server_status` tool definition: no input, read-only hint set.
#[must_use]
pub fn status_tool() -> Tool {
    let mut schema = serde_json::Map::new();
    schema.insert(
        "type".to_owned(),
        serde_json::Value::String("object".to_owned()),
    );
    schema.insert(
        "properties".to_owned(),
        serde_json::Value::Object(serde_json::Map::new()),
    );
    Tool::new(
        TOOL_SERVER_STATUS,
        "Report this server's name, version and pinned MCP revision. Read-only.",
        Arc::new(schema),
    )
    .annotate(ToolAnnotations::new().read_only(true))
}

/// The `server_status` payload. Pure: identical for every caller.
#[must_use]
pub fn server_status_payload() -> serde_json::Value {
    serde_json::json!({
        "name": SERVER_NAME,
        "version": SERVER_VERSION,
        "protocolVersion": PINNED_PROTOCOL_VERSION.as_str(),
        "supportedVersions": SUPPORTED_PROTOCOL_VERSIONS
            .iter()
            .map(ProtocolVersion::as_str)
            .collect::<Vec<_>>(),
        "readOnly": true,
    })
}

/// Run one tool by name. Pure: no session lookup, no stored state.
///
/// # Errors
///
/// Returns `INVALID_PARAMS` (`tool not found`, as in [`rmcp`]'s router)
/// for any name other than [`TOOL_SERVER_STATUS`].
pub fn dispatch_tool(name: &str) -> Result<CallToolResponse, McpError> {
    if name != TOOL_SERVER_STATUS {
        return Err(McpError::invalid_params("tool not found", None));
    }
    let payload = server_status_payload();
    let mut result = CallToolResult::success(vec![ContentBlock::text(format!(
        "{SERVER_NAME} {SERVER_VERSION} protocol {}",
        PINNED_PROTOCOL_VERSION.as_str()
    ))]);
    result.structured_content = Some(payload);
    Ok(result.into())
}

impl ServerHandler for KrishivMcpServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(PINNED_PROTOCOL_VERSION.clone())
            .with_server_info(Implementation::new(SERVER_NAME, SERVER_VERSION))
            .with_instructions(
                "Read-only Krishiv adapter. Tools hold no session state: identity, \
                 caps and audit come from the request or the transport.",
            )
    }

    fn supported_protocol_versions(&self) -> Cow<'static, [ProtocolVersion]> {
        Cow::Borrowed(SUPPORTED_PROTOCOL_VERSIONS)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let mut tools = vec![status_tool()];
        if self.tools.is_some() {
            tools.extend(definitions().into_iter().map(|(name, doc, schema)| {
                Tool::new(name, doc, Arc::new(schema))
                    .annotate(ToolAnnotations::new().read_only(true))
            }));
        }
        Ok(ListToolsResult::with_all_items(tools))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let name = request.name.as_ref();
        let Some(tools) = self.tools.as_ref().filter(|_| name != TOOL_SERVER_STATUS) else {
            return dispatch_tool(name);
        };
        let args = request.arguments.unwrap_or_default();
        let answer = tools.call(name, &args).await;
        let content = vec![ContentBlock::text(answer.text)];
        Ok(if answer.is_error {
            CallToolResult::error(content)
        } else {
            CallToolResult::success(content)
        }
        .into())
    }
}

#[cfg(test)]
mod tests {
    use rmcp::model::{ClientCapabilities, ErrorCode, InitializeRequestParams};

    use crate::protocol::HANDSHAKE_VERSION;

    use super::*;

    fn init_params(version: ProtocolVersion) -> InitializeRequestParams {
        InitializeRequestParams::new(
            ClientCapabilities::default(),
            Implementation::new("test-client", "0.0.0"),
        )
        .with_protocol_version(version)
    }

    fn negotiated(requested: ProtocolVersion) -> ProtocolVersion {
        KrishivMcpServer::default()
            .negotiate_initialize(&init_params(requested))
            .expect("negotiation over a handshake revision must succeed")
            .protocol_version
    }

    #[test]
    fn supported_handshake_version_is_echoed() {
        assert_eq!(
            negotiated(ProtocolVersion::V_2025_11_25),
            ProtocolVersion::V_2025_11_25
        );
    }

    #[test]
    fn stateless_pin_falls_back_to_handshake_over_initialize() {
        // 2026-07-28 dropped the handshake, so agreeing to it here would
        // leave the peers on a revision with no handshake at all.
        assert_eq!(negotiated(ProtocolVersion::V_2026_07_28), HANDSHAKE_VERSION);
    }

    #[test]
    fn older_revisions_are_moved_to_the_handshake_version() {
        for older in [
            ProtocolVersion::V_2024_11_05,
            ProtocolVersion::V_2025_03_26,
            ProtocolVersion::V_2025_06_18,
        ] {
            let agreed = negotiated(older.clone());
            assert_eq!(agreed, HANDSHAKE_VERSION, "client asked for {older}");
            assert_ne!(agreed, older, "older revisions must not be echoed");
        }
    }

    #[test]
    fn unknown_revision_falls_back_to_the_handshake_version() {
        let unknown: ProtocolVersion =
            serde_json::from_str(r#""1999-01-01""#).expect("version literal must deserialize");
        assert_eq!(negotiated(unknown), HANDSHAKE_VERSION);
    }

    #[test]
    fn info_advertises_pin_tools_and_identity() {
        let info = KrishivMcpServer::default().get_info();
        assert_eq!(info.protocol_version, PINNED_PROTOCOL_VERSION);
        assert_eq!(info.server_info.name, SERVER_NAME);
        assert!(
            info.capabilities.tools.is_some(),
            "tools/list must be advertised"
        );
    }

    #[test]
    fn status_tool_is_read_only() {
        let tool = status_tool();
        assert_eq!(tool.name.as_ref(), TOOL_SERVER_STATUS);
        assert_eq!(
            tool.annotations
                .as_ref()
                .and_then(|annotations| annotations.read_only_hint),
            Some(true)
        );
    }

    #[test]
    fn status_call_reports_the_pin() {
        let response = dispatch_tool(TOOL_SERVER_STATUS).expect("known tool must run");
        let rmcp::model::CallToolResponse::Complete(result) = response else {
            panic!("stateless tools must complete inline, without follow-ups");
        };
        // `success` marks non-errors explicitly as `Some(false)`.
        assert_eq!(result.is_error, Some(false));
        let text = result
            .content
            .iter()
            .filter_map(rmcp::model::ContentBlock::as_text)
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains(PINNED_PROTOCOL_VERSION.as_str()),
            "tool output must name the pin, got: {text}"
        );
        assert_eq!(result.structured_content, Some(server_status_payload()));
    }

    #[test]
    fn unknown_tool_is_invalid_params() {
        let error = dispatch_tool("drop_table").expect_err("unknown tool must fail");
        assert_eq!(error.code, ErrorCode::INVALID_PARAMS);
    }

    #[test]
    fn tools_hold_no_session_state() {
        // Two instances (as two connections would see) agree exactly.
        let (first, second) = (KrishivMcpServer::default(), KrishivMcpServer::default());
        assert_eq!(
            first.get_info().protocol_version,
            second.get_info().protocol_version
        );
        assert_eq!(status_tool(), status_tool());
        assert_eq!(
            server_status_payload(),
            serde_json::json!({
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
                "protocolVersion": "2026-07-28",
                "supportedVersions": ["2025-11-25", "2026-07-28"],
                "readOnly": true,
            })
        );
    }
}
