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
pub mod artifacts;
pub mod auth;
mod cache;
pub mod codec;
pub mod config;
pub mod errors;
pub mod execute;
pub mod get_status;
pub mod host;
pub mod interrupt;
pub mod local;
pub mod manager;
pub mod operation;
pub mod sender;
pub mod server;
pub mod service;
pub mod status;
pub mod udf;

pub use artifacts::{ArtifactLimits, ArtifactStore, SessionArtifacts};
pub use auth::{
    ACT_AS_GROUPS, ACT_AS_TENANT, ACT_AS_TOKEN, ACT_AS_USER, AuthLayer, Check, ConnectAuth,
    ExtraProvider, Rpc, SessionOwner, check_command, check_rpc, provider_sessions, role_sessions,
};
pub use errors::SessionErrors;
pub use local::LocalServer;
pub use manager::{ManagerCounts, SessionEntry, SessionFactory, SessionLimits, SessionManager};
pub use operation::{ExecuteLimits, Operation, OperationState, Operations};
pub use server::{ConnectServer, ConnectServerConfig};
pub use service::ConnectService;
pub use status::status_of;
pub use udf::{PythonArchive, RunnerConf, SessionUdfRuntime, UdfHost};
