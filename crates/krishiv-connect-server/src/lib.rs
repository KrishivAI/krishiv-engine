// SPDX-License-Identifier: Apache-2.0
//! The Spark Connect gRPC server (layer L6, design P3-T001): the tonic
//! service over the generated stub, the authentication of every RPC, the session manager keyed
//! by the authenticated caller and session id, the limits applied before admission, and graceful
//! shutdown. Decoding and translation are `krishiv-connect`'s; the admin
//! HTTP server is `krishiv-server`'s; the process around both is
//! `krishiv server` (`krishiv-cli`).

#![forbid(unsafe_code)]
// Requests decode only through `krishiv_connect::wire` (the codec here).
#![deny(clippy::disallowed_methods)]

pub mod analyze;
pub mod auth;
pub mod codec;
pub mod config;
pub mod execute;
pub mod host;
pub mod interrupt;
pub mod manager;
pub mod operation;
pub mod sender;
pub mod server;
pub mod service;
pub mod status;

pub use auth::{
    AuthLayer, Check, ConnectAuth, Rpc, SessionOwner, check_command, check_rpc, role_sessions,
};
pub use manager::{ManagerCounts, SessionEntry, SessionLimits, SessionManager};
pub use operation::{ExecuteLimits, Operation, Operations};
pub use server::{ConnectServer, ConnectServerConfig};
pub use service::ConnectService;
pub use status::status_of;
