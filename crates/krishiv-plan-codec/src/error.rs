// SPDX-License-Identifier: Apache-2.0
//! The codec's error type.

/// Why a plan could not be encoded or decoded. Messages name structure
/// (a node, a field, an option key), never a value: a plan may hold
/// literals and options that are sensitive (ADR 0038).
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodecError {
    /// The plan holds something `krishiv.plan.v1` cannot carry: an
    /// unresolved expression, an inline secret, a URL with credentials.
    #[error("cannot encode the plan: {0}")]
    Unsupported(String),
    /// The bytes are not a valid `krishiv.plan.v1` plan: malformed
    /// protobuf, a reference out of range, a count that does not match.
    #[error("corrupt plan: {0}")]
    Corrupt(String),
    /// A plan written before the feature header that needs a newer reader
    /// version than the frozen format version.
    #[error("the plan needs reader version {required}, this reader implements {supported}")]
    TooNew {
        /// The writer's `min_reader_version`.
        required: u32,
        /// The version this reader implements.
        supported: u32,
    },
    /// The plan is of a wire epoch this reader does not implement.
    #[error("the plan is of epoch {epoch}, this reader implements epoch {supported}")]
    UnsupportedEpoch {
        /// The plan's epoch.
        epoch: u32,
        /// The epoch this reader implements.
        supported: u32,
    },
    /// The plan uses features this reader does not implement (ids unknown to
    /// it, or left out of its [`ReaderFeatures`](crate::ReaderFeatures)).
    #[error("the plan uses features [{}] that this reader does not implement", missing.join(", "))]
    UnsupportedFeatures {
        /// The ids of the missing features, sorted.
        missing: Vec<String>,
    },
    /// The plan is larger or deeper than the decoder's [`Limits`](crate::Limits).
    #[error("the plan exceeds a decode limit: {0}")]
    Limit(String),
}

/// A [`CodecError::Corrupt`].
pub(crate) fn corrupt(what: impl Into<String>) -> CodecError {
    CodecError::Corrupt(what.into())
}

/// Result of the codec.
pub type Result<T> = std::result::Result<T, CodecError>;
