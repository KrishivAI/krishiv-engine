// SPDX-License-Identifier: Apache-2.0
//! Arrow Flight SQL server (layer L6; design `../krishiv-internal/design/P3-T101-flight-sql.md`):
//! a second gRPC service on the Spark Connect listener, on its session manager (P3-T101,
//! single endpoint).

#![forbid(unsafe_code)]

pub mod cookies;
pub mod encode;
pub mod handles;
pub mod metrics;
pub mod sessions;
pub mod sql_info;
pub mod status;

pub use cookies::{SessionCookie, session_cookie, set_session_cookie};
pub use encode::{EncodeLimits, encode_results};
pub use handles::{HandleId, HandleLimits, Handles, Inserted};
pub use metrics::{Command, Outcome, RequestCount, Requests};
pub use sessions::{Exit, Holds, Release, SessionState, Sessions, mint_session_id};
pub use sql_info::{ServerInfo, sql_info_data};
pub use status::{code_of, flight_status};
