// SPDX-License-Identifier: Apache-2.0
//! Krishiv's own `delta_kernel` `Engine` over Arrow 60 (design `P3-T021-delta.md` §3).
//!
//! The kernel is built without its Arrow features (they pull `chrono-tz`, banned by ADR 0028), so the
//! data and schema bridges the kernel's default engine provides are ours.

pub mod data;
mod get_data;
pub mod schema;
