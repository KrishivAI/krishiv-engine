// SPDX-License-Identifier: Apache-2.0
//! Errors of the I/O layer. Every error is a [`KrishivError`]:
//!
//! | Situation | Error class |
//! |---|---|
//! | footer / metadata cannot be decoded | `FAILED_READ_FILE.CANNOT_READ_FILE_FOOTER` |
//! | data cannot be decoded | `FAILED_READ_FILE.NO_HINT` |
//! | a `STRING` value is not valid UTF-8 | `INVALID_UTF8_STRING` |
//! | the store failed (after retries, or with a non-retryable error) | `KRISHIV_OBJECT_STORE_REQUEST_FAILED` |
//! | the file was deleted after listing | `FAILED_READ_FILE.FILE_NOT_EXIST` |
//! | a stored type cannot be read as the requested type | `FAILED_READ_FILE.PARQUET_COLUMN_DATA_TYPE_MISMATCH` |
//! | case-insensitive resolution matched several fields | `FAILED_READ_FILE.NO_HINT` caused by `_LEGACY_ERROR_TEMP_2093` |
//! | the file changed between planning and reading | `KRISHIV_FILE_CHANGED_DURING_READ` |
//! | an upload or an encoder failed | `TASK_WRITE_FAILED` |
//! | unknown / invalid format option | `KRISHIV_UNKNOWN_FORMAT_OPTION` / `KRISHIV_INVALID_FORMAT_OPTION_VALUE` |
//! | the scan was cancelled | `OPERATION_CANCELED` |
//!
//! Spark wraps reader failures as `FAILED_READ_FILE.NO_HINT` with the detail in
//! the Java cause. Where Spark's cause is a classified condition (the
//! duplicate-field error), the error carries it as its
//! [`cause`](KrishivError::cause); the detail of an unclassified cause is
//! logged with the redacted path.

use std::fmt;

use krishiv_common::error::{
    FailedReadFileCannotReadFileFooter, FailedReadFileFileNotExist, FailedReadFileNoHint,
    FailedReadFileParquetColumnDataTypeMismatch, InternalError, InvalidUtf8String, KrishivError,
    KrishivFileChangedDuringRead, KrishivObjectStoreRequestFailed, KrishivUnsupportedFeature,
    OperationCanceled, TaskWriteFailed,
};

/// Result alias for this crate.
pub type Result<T, E = KrishivError> = std::result::Result<T, E>;

/// Result of a [`FileFormat`](crate::FileFormat) or decoder call.
pub type FormatResult<T> = Result<T, FormatError>;

/// An error a format reports without knowing the file's printable path: the
/// framework adds the redacted path ([`Self::into_error`]), so a format can
/// neither forget the path nor leak credentials in it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormatError {
    /// The footer or file metadata cannot be decoded (truncated, bad magic,
    /// lengths out of range, wrong decryption key).
    CorruptFooter(String),
    /// Data cannot be decoded (truncated column chunk, bad checksum).
    Corrupt(String),
    /// A column's stored type cannot be read as the requested type.
    TypeMismatch {
        /// The column, e.g. `[a, b]`.
        column: String,
        /// The requested Spark type, e.g. `bigint`.
        expected: String,
        /// The stored type, e.g. `BINARY`.
        actual: String,
    },
    /// A `STRING` value holds invalid UTF-8: Krishiv strings are always
    /// valid UTF-8, so the scan fails (the reason says where).
    InvalidUtf8(String),
    /// A file cannot be written (the encoder failed); the reason is logged.
    WriteFailed(String),
    /// A record or file Spark fails with `FAILED_READ_FILE.NO_HINT` caused
    /// by this classified error (a malformed CSV record in `FAILFAST` mode).
    Failed(KrishivError),
    /// Any other error, already classified (I/O through the
    /// [`IoContext`](crate::IoContext), memory, options, unsupported features).
    Other(KrishivError),
}

impl FormatError {
    /// The classified error for the file printed as `path` (already
    /// redacted, see [`FileRef::display_path`](crate::FileRef::display_path)).
    #[must_use]
    pub fn into_error(self, path: &str) -> KrishivError {
        match self {
            Self::CorruptFooter(reason) => {
                tracing::warn!(path, reason, "cannot read file footer");
                FailedReadFileCannotReadFileFooter {
                    path: path.to_owned(),
                }
                .into()
            }
            Self::Corrupt(reason) => {
                tracing::warn!(path, reason, "cannot decode file");
                FailedReadFileNoHint {
                    path: path.to_owned(),
                }
                .into()
            }
            Self::InvalidUtf8(reason) => InvalidUtf8String {
                str: format!("a STRING value in {path} ({reason})"),
            }
            .into(),
            Self::WriteFailed(reason) => write_failed(path, &reason),
            Self::Failed(cause) => KrishivError::from(FailedReadFileNoHint {
                path: path.to_owned(),
            })
            .with_cause(cause),
            Self::TypeMismatch {
                column,
                expected,
                actual,
            } => FailedReadFileParquetColumnDataTypeMismatch {
                path: path.to_owned(),
                column,
                expected_type: expected,
                actual_type: actual,
            }
            .into(),
            Self::Other(e) => e,
        }
    }
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CorruptFooter(r) => write!(f, "cannot read footer: {r}"),
            Self::Corrupt(r) => write!(f, "corrupt data: {r}"),
            Self::InvalidUtf8(r) => write!(f, "invalid UTF-8 in a string: {r}"),
            Self::WriteFailed(r) => write!(f, "cannot write: {r}"),
            Self::Failed(e) => write!(f, "cannot read: {e}"),
            Self::TypeMismatch {
                column,
                expected,
                actual,
            } => write!(f, "column {column}: expected {expected}, found {actual}"),
            Self::Other(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for FormatError {}

impl From<KrishivError> for FormatError {
    fn from(e: KrishivError) -> Self {
        Self::Other(e)
    }
}

/// `FAILED_READ_FILE.NO_HINT` with the reason logged.
pub(crate) fn read_failed(path: &str, reason: &str) -> KrishivError {
    tracing::warn!(path, reason, "file read failed");
    FailedReadFileNoHint {
        path: path.to_owned(),
    }
    .into()
}

/// `KRISHIV_OBJECT_STORE_REQUEST_FAILED`: a storage failure, never treated
/// as corruption (`ignoreCorruptFiles` does not skip it).
pub(crate) fn store_failed(path: &str, attempts: u32, reason: &str) -> KrishivError {
    KrishivObjectStoreRequestFailed {
        path: path.to_owned(),
        attempts: attempts.to_string(),
        reason: reason.to_owned(),
    }
    .into()
}

/// `FAILED_READ_FILE.FILE_NOT_EXIST`.
pub(crate) fn not_found(path: &str) -> KrishivError {
    FailedReadFileFileNotExist {
        path: path.to_owned(),
    }
    .into()
}

/// `KRISHIV_FILE_CHANGED_DURING_READ`.
pub(crate) fn changed(path: &str, version: &str) -> KrishivError {
    KrishivFileChangedDuringRead {
        path: path.to_owned(),
        version: version.to_owned(),
    }
    .into()
}

/// `TASK_WRITE_FAILED` with the reason logged.
pub(crate) fn write_failed(path: &str, reason: &str) -> KrishivError {
    tracing::warn!(path, reason, "file write failed");
    TaskWriteFailed {
        path: path.to_owned(),
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

/// `INTERNAL_ERROR`, for a broken invariant (not a user error).
pub(crate) fn internal(message: impl Into<String>) -> KrishivError {
    InternalError {
        message: message.into(),
    }
    .into()
}

/// `OPERATION_CANCELED`.
pub(crate) fn cancelled() -> KrishivError {
    OperationCanceled {}.into()
}
