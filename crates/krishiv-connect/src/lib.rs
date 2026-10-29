// SPDX-License-Identifier: Apache-2.0
//! Spark Connect (layer L6): the protocol types, the translation of a
//! Connect plan into Krishiv's logical-plan IR, and the Arrow shapes of
//! result columns that Spark sends differently from how they are held
//! ([`arrow`]).

#![forbid(unsafe_code)]

pub mod arrow;
pub mod proto;
mod stack;
pub mod translate;
pub mod wire;
