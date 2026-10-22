// SPDX-License-Identifier: Apache-2.0
//! `krishiv.plan.v1`: the stable serialized form of a physical plan
//! (ADR 0003, P2-T173).
//!
//! A [`PhysicalPlan`](krishiv_exec::physical::PhysicalPlan) is pure data
//! (design P2-T171, invariant I1); this crate is the proof and the wire
//! format. [`encode`] turns a plan into protobuf bytes, [`decode`] turns
//! bytes back into the plan, and `decode(encode(p)) == p` for every plan
//! the planner produces. A coordinator ships these bytes to workers, which
//! [`instantiate`](krishiv_exec::physical::instantiate) them with their own
//! resources.
//!
//! # Format
//! - The schema is `proto/krishiv/plan/v1/plan.proto`; the Rust types are
//!   generated from it by `tools/plan-protogen` (CI checks they are
//!   current). Fields are only added with new numbers; removed ones are
//!   `reserved`.
//! - Compatibility is N to N-1 in both directions (ADR 0026): unknown
//!   fields are skipped; a writer that relies on something an older reader
//!   would ignore raises `min_reader_version`, and a reader older than that
//!   rejects the plan ([`CodecError::TooNew`]). The golden plans under
//!   `tests/golden/` pin the bytes of every released format version.
//! - Nodes, expressions and explain nodes are flat tables in post-order
//!   with child indices, so nothing recurses (deep plans are safe) and an
//!   expression shared through `Arc` is stored once.
//! - Spark types are an enum; Arrow schemas and types are Arrow's own IPC
//!   `Schema` flatbuffer, verified on decode.
//!
//! # What a plan may carry
//! Values only (ADR 0038): resolved expressions, schemas, ids, options,
//! sources as table names or file paths. A credential is a
//! `secret(scope, key)` reference. An inline secret or a URL with embedded
//! credentials is not encodable ([`CodecError::Unsupported`]); the encoded
//! bytes never hold secret plaintext. Registry functions travel as
//! (name, registry id): the worker must run the registry version the
//! coordinator planned with (checked at worker registration, P3).
//!
//! # Decoding untrusted bytes
//! [`decode_with`] bounds the size, the table sizes and the plan and expression depth
//! ([`Limits`]), requires every reference to point to an earlier entry,
//! every node to be used once and counts to agree.

#![forbid(unsafe_code)]

mod arrow;
mod decode;
mod encode;
mod error;
/// The generated protobuf types of `krishiv.plan.v1`.
pub mod proto;

pub use decode::{decode, decode_with, from_message};
pub use encode::{encode, to_message};
pub use error::{CodecError, Result};

/// The format version this crate writes.
pub const FORMAT_VERSION: u32 = 2;

/// The `min_reader_version` this crate writes: readers older than this
/// cannot understand the plan.
pub const MIN_READER_VERSION: u32 = 2;

/// What [`decode_with`] accepts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Limits {
    /// The most bytes of a plan.
    pub max_bytes: usize,
    /// The most plan nodes (and explain nodes).
    pub max_nodes: usize,
    /// The most expressions.
    pub max_exprs: usize,
    /// The deepest plan, explain tree and expression. Plans and expressions
    /// are dropped, compared and compiled by recursion, so this bounds the
    /// stack they need.
    pub max_depth: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_bytes: 256 << 20,
            max_nodes: 1 << 20,
            max_exprs: 1 << 22,
            max_depth: 1024,
        }
    }
}
