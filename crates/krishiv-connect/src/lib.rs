// SPDX-License-Identifier: Apache-2.0
//! Spark Connect (layer L6): the protocol types and the translation of a
//! Connect plan into Krishiv's logical-plan IR.

#![forbid(unsafe_code)]

pub mod proto;
mod stack;
pub mod wire;
