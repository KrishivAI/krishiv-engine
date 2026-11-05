// SPDX-License-Identifier: Apache-2.0
//! Errors of the store layer. Every error is a [`KrishivError`] with a Spark-style class and
//! SQLSTATE: `KRISHIV_UNSUPPORTED_FEATURE` for a URL or a build this layer does not support,
//! `KRISHIV_INVALID_CONFIG_VALUE` for a bad setting (the key and the redaction marker, never the
//! value) and `KRISHIV_OBJECT_STORE_REQUEST_FAILED` when a store builder rejects its
//! configuration.

#[cfg(feature = "cloud")]
use krishiv_common::error::KrishivObjectStoreRequestFailed;
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};

/// Result alias for this crate.
pub type Result<T, E = KrishivError> = std::result::Result<T, E>;

/// `KRISHIV_OBJECT_STORE_REQUEST_FAILED`: a storage failure (here: a builder that rejected its
/// configuration), never treated as corruption.
#[cfg(feature = "cloud")]
pub(crate) fn store_failed(path: &str, attempts: u32, reason: &str) -> KrishivError {
    KrishivObjectStoreRequestFailed {
        path: path.to_owned(),
        attempts: attempts.to_string(),
        reason: reason.to_owned(),
    }
    .into()
}

/// `KRISHIV_UNSUPPORTED_FEATURE`.
pub(crate) fn unsupported(feature: impl Into<String>) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.into(),
    }
    .into()
}
