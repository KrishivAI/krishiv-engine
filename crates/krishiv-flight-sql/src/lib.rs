// SPDX-License-Identifier: Apache-2.0
//! Arrow Flight SQL server (layer L6; design `../krishiv-internal/design/P3-T101-flight-sql.md`):
//! a second gRPC service on the Spark Connect listener, on its session manager (P3-T101,
//! single endpoint).

#![forbid(unsafe_code)]

pub mod cookies;
pub mod handles;
pub mod sql_info;
pub mod status;

pub use cookies::{SessionCookie, session_cookie, set_session_cookie};
pub use handles::{HandleId, HandleLimits, Handles};
pub use sql_info::{ServerInfo, sql_info_data};
pub use status::{code_of, flight_status};
