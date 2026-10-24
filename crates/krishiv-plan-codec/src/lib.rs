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
//! - Compatibility is by named features (ADR 0040, [`features`]): the
//!   header lists the [`Feature`]s a plan uses, and a reader decodes every
//!   plan of its epoch whose features it knows. A reader checks the epoch
//!   and the features before it reads a node
//!   ([`CodecError::UnsupportedEpoch`], [`CodecError::UnsupportedFeatures`]),
//!   and a plan that uses a construct its header does not declare (or
//!   declares one it does not use) is corrupt, so [`features_of`] can trust
//!   the header without decoding the plan. Unknown fields are skipped: an
//!   addition that an older reader may ignore needs no feature.
//! - Plans written before the header (format versions 1 to 5, no epoch)
//!   still decode: their features follow from their content, and their
//!   `min_reader_version` gate still applies ([`CodecError::TooNew`]). New
//!   plans keep writing `format_version` 5 and a `min_reader_version` that
//!   makes such older readers refuse what they cannot read.
//! - The golden plans under `tests/golden/` pin the bytes: `base/` for plans
//!   without features, one directory per feature, and the frozen `v<N>`
//!   directories of the sequential versions.
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
//!
//! # Plans in other records
//! The header is derived from the plan alone, never from the cluster, so a
//! fragment's bytes are the same wherever it runs. A record that stores
//! plans to run later stores their [`PlanFeatures`] with them: checkpoint
//! manifests (P4-T011) and job records (P5-T002) do not exist yet and take
//! it when they are built, so a restore or a handoff is refused before it
//! reaches a reader that lacks a feature. A cache of plans keyed by query
//! text must include the reader capabilities the plan was built for in its
//! key; a cache keyed by the encoded bytes needs nothing more.

#![forbid(unsafe_code)]

mod arrow;
mod decode;
mod encode;
mod error;
pub mod features;
/// The generated protobuf types of `krishiv.plan.v1`.
pub mod proto;

pub use decode::{PlanFeatures, decode, decode_with, features_of, from_message};
pub use encode::{encode, to_message};
pub use error::{CodecError, Result};
pub use features::{
    Feature, FeatureKind, MIN_READER_VERSION, POST_MIGRATION_READER_VERSION, ReaderFeatures,
    plan_features,
};

/// The `format_version` this crate writes: frozen at the last sequential
/// version. Readers that predate features compare `min_reader_version`
/// with their own format version.
pub const FORMAT_VERSION: u32 = 5;

/// The wire epoch this crate reads and writes. It changes only for a
/// wire-incompatible restructure, which is also the only time features fold
/// into the baseline or retire.
pub const EPOCH: u32 = 1;

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
