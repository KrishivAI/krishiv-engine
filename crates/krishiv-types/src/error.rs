// SPDX-License-Identifier: Apache-2.0
//! Error constructors for type parsing and mapping.
//!
//! Every error is a [`KrishivError`] with Spark 4.2's error class for the
//! same situation:
//!
//! | Situation | Error class |
//! |---|---|
//! | precision above 38 | `DECIMAL_PRECISION_EXCEEDS_MAX_PRECISION` |
//! | scale above precision | `_LEGACY_ERROR_TEMP_1228` |
//! | negative scale (from Arrow) | `NEGATIVE_SCALE_DISALLOWED` |
//! | malformed type string | `PARSE_SYNTAX_ERROR` |
//! | empty type string | `PARSE_EMPTY_STATEMENT` |
//! | unterminated `/*` comment | `UNCLOSED_BRACKETED_COMMENT` |
//! | non-ASCII unquoted name | `INVALID_IDENTIFIER` |
//! | unknown type name | `UNSUPPORTED_DATATYPE` |
//! | a Spark type Krishiv does not support yet | `KRISHIV_UNSUPPORTED_FEATURE` |
//! | a [`crate::batch::Batch`] invariant broken by the caller (an engine bug) | `INTERNAL_ERROR` |

use krishiv_common::error::{
    DecimalPrecisionExceedsMaxPrecision, InternalError, InvalidIdentifier, KrishivError,
    KrishivUnsupportedFeature, LegacyErrorTemp1228, NegativeScaleDisallowed, ParseEmptyStatement,
    ParseSyntaxError, UnclosedBracketedComment, UnsupportedDatatype,
};

use crate::DecimalType;

/// Result alias for this crate.
pub type Result<T, E = KrishivError> = std::result::Result<T, E>;

/// `DECIMAL_PRECISION_EXCEEDS_MAX_PRECISION`; `precision` is the value as written.
pub(crate) fn precision_exceeds_max(precision: &str) -> KrishivError {
    DecimalPrecisionExceedsMaxPrecision {
        precision: precision.to_owned(),
        max_precision: DecimalType::MAX_PRECISION.to_string(),
    }
    .into()
}

/// `_LEGACY_ERROR_TEMP_1228`.
pub(crate) fn scale_exceeds_precision(precision: &str, scale: &str) -> KrishivError {
    LegacyErrorTemp1228 {
        scale: scale.to_owned(),
        precision: precision.to_owned(),
    }
    .into()
}

/// `NEGATIVE_SCALE_DISALLOWED`.
pub(crate) fn negative_scale(scale: i64) -> KrishivError {
    NegativeScaleDisallowed {
        scale: scale.to_string(),
        sql_conf: "\"spark.sql.legacy.allowNegativeScaleOfDecimal\"".to_owned(),
    }
    .into()
}

/// `UNCLOSED_BRACKETED_COMMENT`.
pub(crate) fn unclosed_comment() -> KrishivError {
    UnclosedBracketedComment {}.into()
}

/// `INVALID_IDENTIFIER`, for an unquoted name that is not ASCII.
pub(crate) fn invalid_identifier(ident: &str) -> KrishivError {
    InvalidIdentifier {
        ident: ident.to_owned(),
    }
    .into()
}

/// `PARSE_EMPTY_STATEMENT`.
pub(crate) fn empty_statement() -> KrishivError {
    ParseEmptyStatement {}.into()
}

/// `PARSE_SYNTAX_ERROR` at a token (`'tok'`) or at `end of input`.
pub(crate) fn syntax_error(at: &str) -> KrishivError {
    ParseSyntaxError {
        error: at.to_owned(),
        hint: String::new(),
    }
    .into()
}

/// `PARSE_SYNTAX_ERROR` at a token Spark drops as `extra input`.
pub(crate) fn extra_input(at: &str) -> KrishivError {
    ParseSyntaxError {
        error: at.to_owned(),
        hint: format!(": extra input {at}"),
    }
    .into()
}

/// `UNSUPPORTED_DATATYPE`; `type_name` is upper-cased by the caller, and
/// double-quoted here, as Spark renders it.
pub(crate) fn unsupported_data_type(type_name: &str) -> KrishivError {
    UnsupportedDatatype {
        type_name: format!("\"{type_name}\""),
    }
    .into()
}

/// `KRISHIV_UNSUPPORTED_FEATURE` for a feature Krishiv does not support yet.
pub(crate) fn unsupported(feature: String) -> KrishivError {
    KrishivUnsupportedFeature { feature }.into()
}

/// `INTERNAL_ERROR`, for a broken engine invariant (not a user error).
pub(crate) fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}
