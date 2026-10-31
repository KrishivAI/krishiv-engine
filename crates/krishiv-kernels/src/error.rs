// SPDX-License-Identifier: Apache-2.0
//! Kernel errors: a registry [`KrishivError`] plus the failing row.
//!
//! | Situation | Error class |
//! |---|---|
//! | ANSI integral overflow (`int`, `bigint`; unary `-`, `abs`; `div`), `date` ± year-month interval | `ARITHMETIC_OVERFLOW` |
//! | ANSI `tinyint`/`smallint` `+ - *` overflow | `BINARY_ARITHMETIC_OVERFLOW` |
//! | ANSI division by zero (`/`, `div`) | `DIVIDE_BY_ZERO` |
//! | ANSI remainder by zero (`%`) | `REMAINDER_BY_ZERO` (Spark 4.0: `DIVIDE_BY_ZERO`) |
//! | ANSI decimal result out of range | `NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION` |
//! | ANSI cast of a malformed string | `CAST_INVALID_INPUT` |
//! | ANSI cast to a narrower integral type | `CAST_OVERFLOW` |
//! | ANSI string to decimal with over 38 integral digits | `NUMERIC_OUT_OF_SUPPORTED_RANGE` |
//! | a date or timestamp outside the microsecond range | `DATETIME_OVERFLOW` |
//! | `make_date` with a field out of range (ANSI) | `DATETIME_FIELD_OUT_OF_BOUNDS.WITH_SUGGESTION` |
//! | `next_day` with an unknown weekday name (ANSI) | `ILLEGAL_DAY_OF_WEEK` |
//! | a `LIKE` pattern ending in its escape, or escaping a character that cannot be escaped | `INVALID_FORMAT.ESC_AT_THE_END`, `INVALID_FORMAT.ESC_IN_THE_MIDDLE` |
//! | ANSI `BINARY` -> `STRING` of bytes that are not UTF-8 | `INVALID_UTF8_STRING` |
//! | `validate_utf8` of bytes that are not UTF-8 | `INVALID_UTF8_STRING` |
//! | `split_part(.., 0)` | `INVALID_INDEX_OF_ZERO` |
//! | ANSI `elt` with an index outside the arguments | `INVALID_ARRAY_INDEX` |
//! | `to_binary` of text that is not in its format | `CONVERSION_INVALID_INPUT` |
//! | `unbase64` of text Java's MIME decoder rejects | `KRISHIV_INVALID_BASE64_INPUT` |
//! | `encode`/`decode` with a charset outside Spark's default set, or text it cannot hold | `INVALID_PARAMETER_VALUE.CHARSET`, `MALFORMED_CHARACTER_CODING` |
//! | `url_decode` of a malformed escape | `CANNOT_DECODE_URL` |
//! | decimal `%` whose integral quotient exceeds 39 significant digits (every mode) | `KRISHIV_DECIMAL_REMAINDER_IMPOSSIBLE` |
//! | output buffer cannot be reserved | `UNABLE_TO_ACQUIRE_MEMORY` |
//! | decimal plan Krishiv does not support yet | `KRISHIV_UNSUPPORTED_FEATURE` |
//! | caller broke a kernel precondition (an engine bug) | `INTERNAL_ERROR` |
//!
//! Messages follow Spark 4.2 (the vendored registry's version); kernels
//! choose the version-specific classes and texts of
//! [`crate::version::SparkVersion`].

use krishiv_common::error::{
    ArithmeticOverflow, BinaryArithmeticOverflow, CannotParseTimestamp, CastInvalidInput,
    CastOverflow, DatetimeFieldOutOfBoundsWithSuggestion,
    DatetimeFieldOutOfBoundsWithoutSuggestion, DatetimeOverflow, DivideByZero, ErrorClass,
    IllegalDayOfWeek, InternalError, InvalidFormatEscAtTheEnd, InvalidFormatEscInTheMiddle,
    InvalidFractionOfSecond, InvalidUtf8String, KrishivDecimalRemainderImpossible, KrishivError,
    KrishivUnsupportedFeature, NumericOutOfSupportedRange, NumericValueOutOfRangeWithSuggestion,
    NumericValueOutOfRangeWithoutSuggestion, RemainderByZero, UnableToAcquireMemory,
};

use crate::policy::Fault;
use crate::version::SparkVersion;

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

    /// Wraps `error` as a kernel failure at `row` (`None` for a failure of
    /// the whole call, such as an invalid pattern compiled once for every
    /// row, rather than of one row's data).
    #[must_use]
    pub fn with_row(row: Option<usize>, error: KrishivError) -> Self {
        Self { row, error }
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

    /// `MALFORMED_RECORD_IN_PARSING` for a `FAILFAST` malformed CSV record at
    /// input row `row` (0-based).
    #[must_use]
    pub fn malformed_record(row: usize, bad_record: &str) -> Self {
        use krishiv_common::error::MalformedRecordInParsingWithoutSuggestion;
        Self {
            row: Some(row),
            error: MalformedRecordInParsingWithoutSuggestion {
                bad_record: bad_record.to_owned(),
                fail_fast_mode: "FAILFAST".to_owned(),
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

    /// `ARITHMETIC_OVERFLOW` from Spark's `MathUtils.*Exact`; `message` is
    /// [`SparkVersion::overflow_message`] (or Spark's own text, e.g. for
    /// `div`). `try_function` is the suggested `try_*` function; Spark's
    /// hint-less overloads pass `None`.
    pub(crate) fn arithmetic_overflow(
        row: usize,
        message: &str,
        try_function: Option<&str>,
    ) -> Self {
        Self {
            row: Some(row),
            error: ArithmeticOverflow {
                message: message.to_owned(),
                alternative: try_function.map_or_else(String::new, |f| {
                    format!(" Use '{f}' to tolerate overflow and return NULL instead.")
                }),
                config: ANSI_CONF.to_owned(),
            }
            .into(),
        }
    }

    /// `BINARY_ARITHMETIC_OVERFLOW` of `tinyint`/`smallint` `+ - *`: Spark
    /// renders both operands with an `S` suffix (`127S + 1S`).
    pub(crate) fn binary_arithmetic_overflow(
        row: usize,
        left: i64,
        symbol: &str,
        right: i64,
        try_function: &str,
    ) -> Self {
        Self {
            row: Some(row),
            error: BinaryArithmeticOverflow {
                value1: format!("{left}S"),
                symbol: symbol.to_owned(),
                value2: format!("{right}S"),
                function_name: format!("`{try_function}`"),
            }
            .into(),
        }
    }

    /// `DIVIDE_BY_ZERO`.
    pub(crate) fn divide_by_zero(row: usize) -> Self {
        Self {
            row: Some(row),
            error: DivideByZero {
                config: ANSI_CONF.to_owned(),
            }
            .into(),
        }
    }

    /// A remainder by zero: `REMAINDER_BY_ZERO`, or `DIVIDE_BY_ZERO` in
    /// Spark 4.0.
    pub(crate) fn remainder_by_zero(row: usize, version: SparkVersion) -> Self {
        if !version.has_remainder_by_zero() {
            return Self::divide_by_zero(row);
        }
        Self {
            row: Some(row),
            error: RemainderByZero {
                config: ANSI_CONF.to_owned(),
            }
            .into(),
        }
    }

    /// `CAST_INVALID_INPUT`; `value` is Spark's SQL rendering of the input
    /// and the types are SQL type names (`STRING`, `DECIMAL(10,2)`).
    pub(crate) fn cast_invalid_input(row: usize, value: String, from: &str, to: &str) -> Self {
        Self {
            row: Some(row),
            error: CastInvalidInput {
                expression: value,
                source_type: format!("\"{from}\""),
                target_type: format!("\"{to}\""),
            }
            .into(),
        }
    }

    /// `CAST_OVERFLOW`; `value` is Spark's SQL rendering of the input.
    pub(crate) fn cast_overflow(row: usize, value: String, from: &str, to: &str) -> Self {
        Self {
            row: Some(row),
            error: CastOverflow {
                value,
                source_type: format!("\"{from}\""),
                target_type: format!("\"{to}\""),
            }
            .into(),
        }
    }

    /// `INVALID_UTF8_STRING`: Spark renders every byte as `\xHH`.
    pub(crate) fn invalid_utf8(row: usize, bytes: &[u8]) -> Self {
        use std::fmt::Write as _;
        let mut s = String::with_capacity(bytes.len().saturating_mul(4));
        for b in bytes {
            let _ = write!(s, "\\x{b:02X}");
        }
        Self {
            row: Some(row),
            error: InvalidUtf8String { str: s }.into(),
        }
    }

    /// `NUMERIC_OUT_OF_SUPPORTED_RANGE`: a string with more than 38
    /// integral digits cast to a decimal; `value` is the string as given.
    pub(crate) fn numeric_out_of_supported_range(row: usize, value: String) -> Self {
        Self {
            row: Some(row),
            error: NumericOutOfSupportedRange { value }.into(),
        }
    }

    /// `DATETIME_OVERFLOW` for a date or timestamp outside the microsecond
    /// range. (Spark raises an unclassified `ArithmeticException("long
    /// overflow")` here, in both ANSI modes; Krishiv raises the registry
    /// class that describes it.)
    pub(crate) fn datetime_overflow(row: usize, operation: &str) -> Self {
        Self {
            row: Some(row),
            error: DatetimeOverflow {
                operation: operation.to_owned(),
            }
            .into(),
        }
    }

    /// `ILLEGAL_DAY_OF_WEEK`: the weekday name of `next_day` is not one.
    pub(crate) fn illegal_day_of_week(row: usize, name: &str) -> Self {
        Self {
            row: Some(row),
            error: IllegalDayOfWeek {
                string: name.to_owned(),
            }
            .into(),
        }
    }

    /// `CANNOT_PARSE_TIMESTAMP`: the string of a `to_*`/`unix_timestamp`
    /// call does not match its pattern; `func` is the `try_*` function
    /// that tolerates it (`` `try_to_timestamp` ``), `message` Java's
    /// failure text.
    pub(crate) fn cannot_parse_timestamp(row: usize, message: String, func: &str) -> Self {
        Self {
            row: Some(row),
            error: CannotParseTimestamp {
                message,
                func: format!("`{func}`"),
            }
            .into(),
        }
    }

    /// `DATETIME_FIELD_OUT_OF_BOUNDS.WITH_SUGGESTION`: `message` is the text
    /// of the `DateTimeException` Spark's `java.time` call raised.
    pub(crate) fn datetime_field_out_of_bounds(row: usize, message: String) -> Self {
        Self {
            row: Some(row),
            error: DatetimeFieldOutOfBoundsWithSuggestion {
                ansi_config: ANSI_CONF.to_owned(),
                range_message: message,
            }
            .into(),
        }
    }

    /// `DATETIME_FIELD_OUT_OF_BOUNDS.WITHOUT_SUGGESTION`: a `TIME` value or
    /// field outside its range (no ANSI switch to suggest).
    pub(crate) fn time_field_out_of_bounds(row: usize, message: String) -> Self {
        Self {
            row: Some(row),
            error: DatetimeFieldOutOfBoundsWithoutSuggestion {
                range_message: message,
            }
            .into(),
        }
    }

    /// `INVALID_FRACTION_OF_SECOND`: a `make_timestamp` seconds value of 60
    /// with a fraction.
    pub(crate) fn invalid_fraction_of_second(row: usize, sec_and_micros: String) -> Self {
        Self {
            row: Some(row),
            error: InvalidFractionOfSecond { sec_and_micros }.into(),
        }
    }

    /// `INVALID_FORMAT.ESC_AT_THE_END`: a `LIKE` pattern ending in its escape.
    pub(crate) fn like_escape_at_end(row: usize, pattern: &str) -> Self {
        Self {
            row: Some(row),
            error: InvalidFormatEscAtTheEnd {
                format: crate::cast::sql_string(pattern),
            }
            .into(),
        }
    }

    /// `INVALID_FORMAT.ESC_IN_THE_MIDDLE`: the escape before `c`, which it
    /// cannot escape.
    pub(crate) fn like_escape_before(row: usize, pattern: &str, c: char) -> Self {
        Self {
            row: Some(row),
            error: InvalidFormatEscInTheMiddle {
                format: crate::cast::sql_string(pattern),
                char: crate::cast::sql_string(c.encode_utf8(&mut [0; 4])),
            }
            .into(),
        }
    }

    /// The error for a raised `fault` at `row`; `overflow` builds the
    /// kernel-specific overflow error.
    pub(crate) fn for_fault(
        row: usize,
        fault: Fault,
        overflow: impl FnOnce(usize) -> Self,
    ) -> Self {
        match fault {
            Fault::Overflow => overflow(row),
            Fault::DivideByZero => Self::divide_by_zero(row),
            Fault::Impossible => {
                Self::internal("a kernel without unrepresentable rows reported one")
            }
        }
    }

    /// `KRISHIV_DECIMAL_REMAINDER_IMPOSSIBLE` (Spark: an unclassified
    /// `ArithmeticException("Division impossible")`, raised in every mode).
    pub(crate) fn decimal_remainder_impossible(
        row: usize,
        dividend: String,
        divisor: String,
    ) -> Self {
        Self {
            row: Some(row),
            error: KrishivDecimalRemainderImpossible { dividend, divisor }.into(),
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

    /// `NUMERIC_VALUE_OUT_OF_RANGE.WITHOUT_SUGGESTION` of `round` / `ceil` /
    /// `floor` on a decimal: `rounded` is the plain text of the result,
    /// `original` Java's `BigDecimal.toString` of the rounded quotient.
    pub(crate) fn decimal_round_out_of_range(
        row: usize,
        rounded: String,
        original: String,
        precision: u8,
        scale: i8,
    ) -> Self {
        Self {
            row: Some(row),
            error: NumericValueOutOfRangeWithoutSuggestion {
                rounded_value: rounded,
                original_value: original,
                precision: precision.to_string(),
                scale: scale.to_string(),
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

/// An empty vector with room for `len` values, reporting allocation
/// failure as an error instead of aborting.
pub(crate) fn with_capacity<T>(len: usize) -> Result<Vec<T>, KernelError> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| KernelError::out_of_memory(len.saturating_mul(size_of::<T>())))?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_message_matches_spark_4_2() {
        let e = KernelError::arithmetic_overflow(7, "overflow", Some("try_add"));
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
    fn faults_map_to_spark_classes() {
        let hintless = KernelError::for_fault(1, Fault::Overflow, |r| {
            KernelError::arithmetic_overflow(r, "overflow", None)
        });
        assert_eq!(hintless.row(), Some(1));
        assert!(
            hintless
                .to_string()
                .starts_with("[ARITHMETIC_OVERFLOW] overflow. If necessary")
        );
        let div = KernelError::for_fault(2, Fault::DivideByZero, |_| unreachable!());
        assert_eq!(
            (div.row(), div.error_class()),
            (Some(2), ErrorClass::DivideByZero)
        );
        assert!(div.to_string().contains("try_divide"));
    }

    /// Texts recorded from Spark 4.2 (4.0 for the remainder case).
    #[test]
    fn messages_match_spark() {
        let cases = [
            (
                KernelError::binary_arithmetic_overflow(0, 127, "+", 1, "try_add"),
                "[BINARY_ARITHMETIC_OVERFLOW] 127S + 1S caused overflow. Use `try_add` to ignore \
                 overflow problem and return NULL. SQLSTATE: 22003",
            ),
            (
                KernelError::remainder_by_zero(0, SparkVersion::V4_2),
                "[REMAINDER_BY_ZERO] Remainder by zero. Use `try_mod` to tolerate divisor being 0 \
                 and return NULL instead. If necessary set \"spark.sql.ansi.enabled\" to \"false\" \
                 to bypass this error. SQLSTATE: 22012",
            ),
            (
                KernelError::cast_invalid_input(0, "'12.0'".to_owned(), "STRING", "INT"),
                "[CAST_INVALID_INPUT] The value '12.0' of the type \"STRING\" cannot be cast to \
                 \"INT\" because it is malformed. Correct the value as per the syntax, or change \
                 its target type. Use `try_cast` to tolerate malformed input and return NULL \
                 instead. SQLSTATE: 22018",
            ),
            (
                KernelError::cast_overflow(0, "1.0E10D".to_owned(), "DOUBLE", "INT"),
                "[CAST_OVERFLOW] The value 1.0E10D of the type \"DOUBLE\" cannot be cast to \
                 \"INT\" due to an overflow. Use `try_cast` to tolerate overflow and return NULL \
                 instead. SQLSTATE: 22003",
            ),
            (
                KernelError::numeric_out_of_supported_range(0, " 1e400 ".to_owned()),
                "[NUMERIC_OUT_OF_SUPPORTED_RANGE] The value  1e400  cannot be interpreted as a \
                 numeric since it has more than 38 digits. SQLSTATE: 22003",
            ),
        ];
        for (e, text) in cases {
            assert_eq!(e.to_string(), text);
        }
        let v40 = KernelError::remainder_by_zero(3, SparkVersion::V4_0);
        assert_eq!(
            (v40.row(), v40.error_class()),
            (Some(3), ErrorClass::DivideByZero)
        );
        let dt = KernelError::datetime_overflow(1, "cast to timestamp");
        assert_eq!(dt.error_class(), ErrorClass::DatetimeOverflow);
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
        assert!(with_capacity::<u8>(usize::MAX).is_err());
        assert_eq!(with_capacity::<u64>(3).unwrap().capacity(), 3);
    }
}
