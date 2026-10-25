// SPDX-License-Identifier: Apache-2.0
//! MCP protocol revision pin.
//!
//! The workspace pins `rmcp ^3.5.1` (resolved to 3.7.0; no requirement
//! change was needed). Both releases know the same five revisions
//! (`ProtocolVersion::KNOWN_VERSIONS`): `2024-11-05`, `2025-03-26`,
//! `2025-06-18`, `2025-11-25` and `2026-07-28`.
//!
//! * `2026-07-28` is `LATEST` and is fully supported, but it is stateless:
//!   it is also `NO_INITIALIZE`, so it has no `initialize` handshake and
//!   negotiates per request through `_meta` instead.
//! * `2025-11-25` is `LATEST_WITH_INITIALIZE`: the newest revision that
//!   still runs the `initialize` handshake, and therefore the newest
//!   revision a server can name in an `initialize` result.
//!
//! The pin below advertises the stateless revision while answering the
//! handshake with the newest revision the handshake can name. No `rmcp`
//! upgrade is needed: the pinned release already implements both revisions.

use rmcp::model::ProtocolVersion;

/// Newest MCP revision [`rmcp`] fully supports, and this server's pin.
///
/// This is the revision [`crate::server::KrishivMcpServer`] advertises in
/// `get_info`/`discover`. It has no `initialize` handshake
/// (`!has_initialize`), so `initialize` answers never name it; see
/// [`HANDSHAKE_VERSION`].
pub const PINNED_PROTOCOL_VERSION: ProtocolVersion = ProtocolVersion::V_2026_07_28;

/// Newest revision with an `initialize` handshake.
///
/// A server replying to `initialize` must name this revision rather than
/// [`PINNED_PROTOCOL_VERSION`], which dropped the handshake. Kept as a
/// named constant so the fallback has one definition site next to the pin.
pub const HANDSHAKE_VERSION: ProtocolVersion = ProtocolVersion::V_2025_11_25;

/// Every revision this server advertises and negotiates, oldest first.
///
/// The list bounds three behaviours in [`rmcp`]: what `discover`
/// advertises, what `initialize` negotiation may agree to, and what
/// per-request `_meta` versions are accepted. Older revisions are not
/// echoed: a client naming one is answered with [`HANDSHAKE_VERSION`]
/// over `initialize` (a downgrade to a supported revision, per the
/// lifecycle section of the spec), and a per-request version outside this
/// list is rejected with `UNSUPPORTED_PROTOCOL_VERSION`.
pub const SUPPORTED_PROTOCOL_VERSIONS: &[ProtocolVersion] =
    &[ProtocolVersion::V_2025_11_25, ProtocolVersion::V_2026_07_28];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pin_is_the_newest_revision_rmcp_knows() {
        assert_eq!(PINNED_PROTOCOL_VERSION, ProtocolVersion::LATEST);
        assert_eq!(
            PINNED_PROTOCOL_VERSION,
            ProtocolVersion::NO_INITIALIZE,
            "the pin must track the stateless revision, not a handshake one"
        );
    }

    #[test]
    fn handshake_version_is_the_newest_with_initialize() {
        assert_eq!(HANDSHAKE_VERSION, ProtocolVersion::LATEST_WITH_INITIALIZE);
        assert!(HANDSHAKE_VERSION.has_initialize());
        assert!(!PINNED_PROTOCOL_VERSION.has_initialize());
    }

    #[test]
    fn supported_list_is_exactly_the_pin_and_the_handshake() {
        assert_eq!(
            SUPPORTED_PROTOCOL_VERSIONS,
            &[ProtocolVersion::V_2025_11_25, ProtocolVersion::V_2026_07_28,]
        );
        assert!(SUPPORTED_PROTOCOL_VERSIONS.contains(&PINNED_PROTOCOL_VERSION));
        assert!(SUPPORTED_PROTOCOL_VERSIONS.contains(&HANDSHAKE_VERSION));
    }
}
