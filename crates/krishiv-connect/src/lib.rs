// SPDX-License-Identifier: Apache-2.0
//! Spark Connect (layer L6): the protocol types, the translation of a
//! Connect plan into Krishiv's logical-plan IR, and the Arrow shapes of
//! result columns that Spark sends differently from how they are held
//! ([`arrow`]).

#![forbid(unsafe_code)]
// Untrusted bytes decode only through `wire`'s bounded decoder: the banned
// prost paths live in the workspace `clippy.toml`, denied here (the
// workspace level stays silent for crates with no compute worker).
// Deliberate use carries `#[expect(clippy::disallowed_methods, reason =
// "...")]`.
#![deny(clippy::disallowed_methods)]

pub mod arrow;
pub mod proto;
mod stack;
pub mod translate;
pub mod wire;
