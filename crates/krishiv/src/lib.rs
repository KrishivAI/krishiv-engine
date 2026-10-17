// SPDX-License-Identifier: Apache-2.0
//! Krishiv facade crate (layer L7; empty skeleton).

#![forbid(unsafe_code)]

/// Admin and UI server (feature `ui`).
#[cfg(feature = "ui")]
pub use krishiv_server as server;
