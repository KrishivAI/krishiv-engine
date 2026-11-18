// SPDX-License-Identifier: Apache-2.0
//! Arrow Flight SQL server (layer L6; design `../krishiv-internal/design/P3-T101-flight-sql.md`):
//! a second gRPC service on the Spark Connect listener, on its session manager (P3-T101,
//! single endpoint).

#![forbid(unsafe_code)]

pub mod handles;
pub mod status;

pub use handles::{HandleId, HandleLimits, Handles};
pub use status::{code_of, flight_status};
