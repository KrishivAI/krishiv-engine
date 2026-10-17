// SPDX-License-Identifier: Apache-2.0
//! Spill file errors and their Spark-style error classes.

use std::io;
use std::path::{Path, PathBuf};

use krishiv_common::error::{
    ErrorClass, InternalError, KrishivError, KrishivInvalidConfigValue, KrishivSpillCapExceeded,
    KrishivSpillFileCorrupt, KrishivSpillIoError, OperationCanceled,
};

use crate::MemoryError;

/// A spill operation that failed. Never a panic: every failure of the
/// directory manager, a writer or a reader is one of these, and each maps to
/// a Spark-style error class through [`KrishivError`]. After any error a
/// writer or reader is finished: it accepts no more work and its files are
/// removed when it is dropped, with every byte it charged returned to the
/// spill cap and the memory pool.
#[derive(Debug, thiserror::Error)]
pub enum SpillError {
    /// The write would take the spill files over `krishiv.spill.maxBytes`
    /// (`KRISHIV_SPILL_CAP_EXCEEDED`, SQLSTATE 53100).
    #[error("{}", KrishivError::from(self))]
    CapExceeded {
        /// Bytes the write needed.
        requested: u64,
        /// Bytes spill files occupied when it was refused.
        used: u64,
        /// The configured cap.
        cap: u64,
    },
    /// The file system refused an operation, for example because the disk
    /// is full or the directory is gone (`KRISHIV_SPILL_IO_ERROR`,
    /// SQLSTATE 58030). Arrow encoding failures are reported the same way.
    #[error("{}", KrishivError::from(self))]
    Io {
        /// What was attempted, as a verb phrase ("create", "write to", ...).
        operation: &'static str,
        /// The file or directory involved.
        path: PathBuf,
        /// The cause.
        #[source]
        source: io::Error,
    },
    /// A spill file does not decode: truncated, bit-flipped, from an unknown
    /// format version, or inconsistent with what was written
    /// (`KRISHIV_SPILL_FILE_CORRUPT`, SQLSTATE XX001).
    #[error("{}", KrishivError::from(self))]
    Corrupt {
        /// The file.
        path: PathBuf,
        /// What is wrong with it.
        reason: String,
    },
    /// The query was cancelled while spilling (`OPERATION_CANCELED`,
    /// SQLSTATE HY008). Its spill files are already gone.
    #[error("{}", KrishivError::from(self))]
    Cancelled,
    /// A `krishiv.spill.*` value the spill directory cannot use
    /// (`KRISHIV_INVALID_CONFIG_VALUE`, SQLSTATE 22023).
    #[error("{}", KrishivError::from(self))]
    InvalidConfig {
        /// The config key.
        key: &'static str,
        /// The value as given.
        value: String,
        /// Why it is unusable.
        reason: String,
    },
    /// A caller bug, such as writing a batch whose schema differs from the
    /// writer's (`INTERNAL_ERROR`, SQLSTATE XX000).
    #[error("{}", KrishivError::from(self))]
    Internal(String),
    /// The memory pool refused the scratch memory of a write or read.
    #[error(transparent)]
    Memory(#[from] MemoryError),
}

impl SpillError {
    /// The error class this error is reported with.
    pub fn error_class(&self) -> ErrorClass {
        match self {
            SpillError::CapExceeded { .. } => ErrorClass::KrishivSpillCapExceeded,
            SpillError::Io { .. } => ErrorClass::KrishivSpillIoError,
            SpillError::Corrupt { .. } => ErrorClass::KrishivSpillFileCorrupt,
            SpillError::Cancelled => ErrorClass::OperationCanceled,
            SpillError::InvalidConfig { .. } => ErrorClass::KrishivInvalidConfigValue,
            SpillError::Internal(_) => ErrorClass::InternalError,
            SpillError::Memory(e) => e.error_class(),
        }
    }

    pub(crate) fn io(operation: &'static str, path: &Path, source: io::Error) -> Self {
        SpillError::Io {
            operation,
            path: path.to_owned(),
            source,
        }
    }

    pub(crate) fn corrupt(path: &Path, reason: impl Into<String>) -> Self {
        SpillError::Corrupt {
            path: path.to_owned(),
            reason: reason.into(),
        }
    }
}

impl From<&SpillError> for KrishivError {
    fn from(e: &SpillError) -> Self {
        match e {
            SpillError::CapExceeded {
                requested,
                used,
                cap,
            } => KrishivSpillCapExceeded {
                requested_bytes: requested.to_string(),
                cap_bytes: cap.to_string(),
                used_bytes: used.to_string(),
            }
            .into(),
            SpillError::Io {
                operation,
                path,
                source,
            } => KrishivSpillIoError {
                operation: (*operation).to_owned(),
                path: path.display().to_string(),
                reason: source.to_string(),
            }
            .into(),
            SpillError::Corrupt { path, reason } => KrishivSpillFileCorrupt {
                path: path.display().to_string(),
                reason: reason.clone(),
            }
            .into(),
            SpillError::InvalidConfig { key, value, reason } => KrishivInvalidConfigValue {
                value: value.clone(),
                key: (*key).to_owned(),
                reason: reason.clone(),
            }
            .into(),
            SpillError::Cancelled => OperationCanceled {}.into(),
            SpillError::Internal(message) => InternalError {
                message: message.clone(),
            }
            .into(),
            SpillError::Memory(e) => KrishivError::from(e),
        }
    }
}

impl From<SpillError> for KrishivError {
    fn from(e: SpillError) -> Self {
        Self::from(&e)
    }
}
