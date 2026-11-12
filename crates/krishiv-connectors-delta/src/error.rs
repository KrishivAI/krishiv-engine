// SPDX-License-Identifier: Apache-2.0
//! Kernel errors as Krishiv errors (design `P3-T021-delta.md` §3.7).
//!
//! A missing data file is Spark's `FAILED_READ_FILE.FILE_NOT_EXIST`, a feature the kernel does
//! not support is `UNSUPPORTED_FEATURE.TABLE_OPERATION` with the kernel's text, a cancelled
//! kernel call is `OPERATION_CANCELED`, and every other failure is `INTERNAL_ERROR` with the
//! kernel's message cut to [`MAX_MESSAGE_CHARS`] characters. Delta's own classes (`DELTA_*`,
//! vendored in `krishiv-common`) are raised where the connector has the parameters, not here.

use delta_kernel::{Error as KernelWrapper, KernelError};
use krishiv_common::error::{
    FailedReadFileFileNotExist, InternalError, KrishivError, OperationCanceled,
    UnsupportedFeatureTableOperation,
};

/// Characters of a kernel message kept in an `INTERNAL_ERROR`.
pub const MAX_MESSAGE_CHARS: usize = 2048;

/// The Krishiv error for `error`, raised while reading or writing the table `table`.
#[must_use]
pub fn from_kernel(error: &KernelWrapper, table: &str) -> KrishivError {
    let KernelWrapper::Kernel(kernel) = error else {
        return internal(&error.to_string());
    };
    match kernel.without_backtrace() {
        KernelError::FileNotFound(path) => FailedReadFileFileNotExist { path: path.clone() }.into(),
        KernelError::Unsupported(what) => UnsupportedFeatureTableOperation {
            table_name: table.to_owned(),
            operation: what.clone(),
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
    use krishiv_common::error::ErrorClass;

    use super::*;

    fn class(e: KernelError) -> ErrorClass {
        from_kernel(&KernelWrapper::Kernel(e), "t").error_class()
    }

    #[test]
    fn kernel_errors_map_to_their_classes() {
        assert_eq!(
            class(KernelError::file_not_found("a.parquet")),
            ErrorClass::FailedReadFileFileNotExist
        );
        assert_eq!(
            class(KernelError::unsupported("deletion vectors")),
            ErrorClass::UnsupportedFeatureTableOperation
        );
        assert_eq!(class(KernelError::Cancelled), ErrorClass::OperationCanceled);
        assert_eq!(
            class(KernelError::generic("boom")),
            ErrorClass::InternalError
        );
    }

    #[test]
    fn a_backtraced_error_maps_by_what_it_wraps() {
        let wrapped = KernelError::file_not_found("a").with_backtrace();
        assert_eq!(class(wrapped), ErrorClass::FailedReadFileFileNotExist);
    }

    #[test]
    fn an_internal_message_is_cut() {
        let e = from_kernel(
            &KernelWrapper::Kernel(KernelError::generic("x".repeat(5000))),
            "t",
        );
        assert!(e.message().chars().count() < MAX_MESSAGE_CHARS + 200);
    }
}
