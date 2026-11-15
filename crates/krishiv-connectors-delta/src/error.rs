// SPDX-License-Identifier: Apache-2.0
//! Kernel errors as Krishiv errors (design note P3-T021 section 3.7).
//!
//! A Krishiv error raised inside a handler (a parse limit, a refused path, a store failure)
//! travels through the kernel as [`KernelError::GenericError`] and comes back out unchanged.
//! Otherwise: a missing file is Spark's `FAILED_READ_FILE.FILE_NOT_EXIST`, a feature the kernel
//! does not support is `UNSUPPORTED_FEATURE.TABLE_OPERATION` with the kernel's text, an invalid
//! protocol is `DELTA_INVALID_PROTOCOL_VERSION`, a cancelled call is `OPERATION_CANCELED`, and
//! anything else is `INTERNAL_ERROR` with the kernel's message cut to [`MAX_MESSAGE_CHARS`]
//! characters.

use delta_kernel::KernelError;
use krishiv_common::error::{
    DeltaInvalidProtocolVersion, FailedReadFileFileNotExist, InternalError, KrishivError,
    OperationCanceled, UnsupportedFeatureTableOperation,
};

/// Characters of a kernel message kept in an `INTERNAL_ERROR`.
pub const MAX_MESSAGE_CHARS: usize = 2048;

/// The reader protocol versions Krishiv reads (`<supportedReaders>`).
pub const SUPPORTED_READERS: &str = "[1, 2, 3]";

/// The writer protocol versions Krishiv knows (`<supportedWriters>`).
pub const SUPPORTED_WRITERS: &str = "[1, 2, 3, 4, 5, 6, 7]";

/// `<deltaVersion>` in Delta's feature and protocol classes: `Krishiv <version>`.
#[must_use]
pub fn delta_version() -> String {
    format!("Krishiv {}", env!("CARGO_PKG_VERSION"))
}

/// `error` as a kernel error, so a handler can return it through the kernel and
/// [`from_kernel`] gets it back unchanged.
#[must_use]
pub fn to_kernel(error: KrishivError) -> KernelError {
    KernelError::generic_err(error)
}

/// The Krishiv error for `error`, raised while reading or writing the table `table`.
#[must_use]
pub fn from_kernel(error: &KernelError, table: &str) -> KrishivError {
    match error.without_backtrace() {
        KernelError::GenericError { source } => match source.downcast_ref::<KrishivError>() {
            Some(ours) => ours.clone(),
            None => internal(&error.to_string()),
        },
        KernelError::FileNotFound(path) => FailedReadFileFileNotExist { path: path.clone() }.into(),
        KernelError::Unsupported(what) => UnsupportedFeatureTableOperation {
            table_name: table.to_owned(),
            operation: what.clone(),
        }
        .into(),
        KernelError::InvalidProtocol(_) => DeltaInvalidProtocolVersion {
            table_name_or_path: table.to_owned(),
            reader_required: "unknown".to_owned(),
            writer_required: "unknown".to_owned(),
            delta_version: delta_version(),
            supported_readers: SUPPORTED_READERS.to_owned(),
            supported_writers: SUPPORTED_WRITERS.to_owned(),
        }
        .into(),
        KernelError::Cancelled => OperationCanceled {}.into(),
        other => internal(&other.to_string()),
    }
}

fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.chars().take(MAX_MESSAGE_CHARS).collect(),
    }
    .into()
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::{ErrorClass, KrishivCatalogLimitExceeded};

    use super::*;

    fn class(e: &KernelError) -> ErrorClass {
        from_kernel(e, "t").error_class()
    }

    #[test]
    fn kernel_errors_map_to_their_classes() {
        assert_eq!(
            class(&KernelError::file_not_found("a.parquet")),
            ErrorClass::FailedReadFileFileNotExist
        );
        let e = from_kernel(&KernelError::unsupported("copy_atomic"), "db.t");
        assert_eq!(
            e.error_class(),
            ErrorClass::UnsupportedFeatureTableOperation
        );
        assert!(e.to_string().contains("copy_atomic"), "{e}");
        assert_eq!(
            class(&KernelError::invalid_protocol("reader 4")),
            ErrorClass::DeltaInvalidProtocolVersion
        );
        assert_eq!(
            class(&KernelError::Cancelled),
            ErrorClass::OperationCanceled
        );
        assert_eq!(
            class(&KernelError::generic("boom")),
            ErrorClass::InternalError
        );
        // A backtrace wrapper maps by what it wraps.
        let wrapped = KernelError::file_not_found("a").with_backtrace();
        assert_eq!(class(&wrapped), ErrorClass::FailedReadFileFileNotExist);
    }

    #[test]
    fn a_krishiv_error_comes_back_through_the_kernel_unchanged() {
        let ours: KrishivError = KrishivCatalogLimitExceeded {
            source: "s".to_owned(),
            limit: "l".to_owned(),
            value: "v".to_owned(),
            config: "c".to_owned(),
        }
        .into();
        assert_eq!(from_kernel(&to_kernel(ours.clone()), "t"), ours);
    }

    #[test]
    fn an_internal_message_is_cut() {
        let e = from_kernel(&KernelError::generic("x".repeat(5000)), "t");
        assert_eq!(e.error_class(), ErrorClass::InternalError);
        assert!(
            e.message().chars().count() <= MAX_MESSAGE_CHARS + 64,
            "{}",
            e.message().len()
        );
    }
}
