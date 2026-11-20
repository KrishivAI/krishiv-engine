// SPDX-License-Identifier: Apache-2.0
//! Arrow Flight SQL server (layer L6; design `../krishiv-internal/design/P3-T101-flight-sql.md`):
//! a second gRPC service on the Spark Connect listener, on its session manager (P3-T101,
//! single endpoint).

#![forbid(unsafe_code)]

mod call;
pub mod cookies;
pub mod encode;
mod gate;
pub mod handles;
mod metadata;
pub mod metrics;
mod prepared;
pub mod server;
mod service;
pub mod session_options;
pub mod sessions;
pub mod sql_info;
mod statement;
pub mod status;
mod update;
mod xdbc_info;

pub use cookies::{SessionCookie, session_cookie, set_session_cookie};
pub use encode::{EncodeLimits, encode_results};
pub use gate::FlightGate;
pub use handles::{HandleId, HandleLimits, Handles, Inserted};
pub use metrics::{Command, Outcome, RequestCount, Requests};
pub use server::{FlightSqlConfig, FlightSqlServer, HANDLES_CONSUMER, SweepGuard};
pub use sessions::{Exit, Holds, Release, SessionState, Sessions, mint_session_id};
pub use sql_info::{ServerInfo, sql_info_data};
pub use status::{code_of, flight_status};
