// SPDX-License-Identifier: Apache-2.0
//! Kernel errors: a registry [`KrishivError`] plus the failing row.
//!
//! | Situation | Error class |
//! |---|---|
//! | ANSI integral overflow (`int`, `bigint`) | `ARITHMETIC_OVERFLOW` |
//! | ANSI decimal result out of range | `NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION` |
//! | output buffer cannot be reserved | `UNABLE_TO_ACQUIRE_MEMORY` |
//! | decimal plan Krishiv does not support yet | `KRISHIV_UNSUPPORTED_FEATURE` |
//! | caller broke a kernel precondition (an engine bug) | `INTERNAL_ERROR` |
//!
//! Messages follow Spark 4.2 (the vendored registry's version).

use krishiv_common::error::{
    ArithmeticOverflow, ErrorClass, InternalError, KrishivError, KrishivUnsupportedFeature,
    NumericValueOutOfRangeWithSuggestion, UnableToAcquireMemory,
};

use crate::ops::ArithOp;

/// Spark's `toSQLConf(spark.sql.ansi.enabled)`.
const ANSI_CONF: &str = "\"spark.sql.ansi.enabled\"";

/// A kernel failure: the registry error and, for data errors, the row (index
/// into the kernel's logical input) that raised it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{error}")]
pub struct KernelError {
    row: Option<usize>,
    error: KrishivError,
}

impl KernelError {
    /// The first failing row, for data errors; `None` for engine errors.
    #[must_use]
    pub fn row(&self) -> Option<usize> {
        self.row
    }

    /// The error class.
    #[must_use]
    pub fn error_class(&self) -> ErrorClass {
        self.error.error_class()
    }

    /// The registry error.
    #[must_use]
    pub fn error(&self) -> &KrishivError {
        &self.error
    }

    /// `INTERNAL_ERROR`: a caller broke a kernel precondition.
    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self {
            row: None,
            error: InternalError {
                message: message.into(),
            }
            .into(),
        }
    }

    /// `KRISHIV_UNSUPPORTED_FEATURE`.
    pub(crate) fn unsupported(feature: impl Into<String>) -> Self {
        Self {
            row: None,
            error: KrishivUnsupportedFeature {
                feature: feature.into(),
            }
            .into(),
        }
    }

    /// `UNABLE_TO_ACQUIRE_MEMORY` for an output buffer of `bytes`.
    pub(crate) fn out_of_memory(bytes: usize) -> Self {
        Self {
            row: None,
            error: UnableToAcquireMemory {
                requested_bytes: bytes.to_string(),
                received_bytes: "0".to_owned(),
            }
            .into(),
        }
    }

    /// `ARITHMETIC_OVERFLOW` from Spark's `MathUtils.*Exact` for `int` and
    /// `bigint`: Spark 4.2 canonicalizes the JDK's `"long overflow"` /
    /// `"integer overflow"` to `"overflow"` and suggests the `try_` function.
    pub(crate) fn arithmetic_overflow(row: usize, op: ArithOp) -> Self {
        Self {
            row: Some(row),
            error: ArithmeticOverflow {
                message: "overflow".to_owned(),
                alternative: format!(
                    " Use '{}' to tolerate overflow and return NULL instead.",
                    op.try_function()
                ),
                config: ANSI_CONF.to_owned(),
            }
            .into(),
        }
    }

    /// `NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION`; `value` is Spark's
    /// `toPlainString` of the unrounded result.
    pub(crate) fn decimal_out_of_range(
        row: usize,
        value: String,
        precision: u8,
        scale: i8,
    ) -> Self {
        Self {
            row: Some(row),
            error: NumericValueOutOfRangeWithSuggestion {
                value,
                precision: precision.to_string(),
                scale: scale.to_string(),
                config: ANSI_CONF.to_owned(),
            }
            .into(),
        }
    }
}

impl From<KernelError> for KrishivError {
    fn from(e: KernelError) -> Self {
        e.error
    }
}

/// Reserves a zeroed output of `len` values, reporting allocation failure as
/// an error instead of aborting.
pub(crate) fn zeroed<T: Copy + Default>(len: usize) -> Result<Vec<T>, KernelError> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| KernelError::out_of_memory(len.saturating_mul(size_of::<T>())))?;
    v.resize(len, T::default());
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_message_matches_spark_4_2() {
        let e = KernelError::arithmetic_overflow(7, ArithOp::Add);
        assert_eq!(e.row(), Some(7));
        assert_eq!(e.error_class(), ErrorClass::ArithmeticOverflow);
        assert_eq!(
            e.to_string(),
            "[ARITHMETIC_OVERFLOW] overflow. Use 'try_add' to tolerate overflow and return NULL \
             instead. If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this \
             error. SQLSTATE: 22003"
        );
    }

    #[test]
    fn engine_errors_have_no_row() {
        let e = KernelError::internal("x");
        assert_eq!(e.row(), None);
        assert_eq!(e.error_class(), ErrorClass::InternalError);
        assert_eq!(KrishivError::from(e.clone()), *e.error());
        let e = KernelError::unsupported("y");
        assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
        let e = KernelError::out_of_memory(8);
        assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory);
        assert!(zeroed::<u8>(usize::MAX).is_err());
    }
}
