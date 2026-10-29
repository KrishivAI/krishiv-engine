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
//! | time precision above 6 | `UNSUPPORTED_TIME_PRECISION` |
//! | an SRID Spark does not support | `ST_INVALID_SRID_VALUE` |
//! | malformed WKB ([`crate::geo`]) | `WKB_PARSE_ERROR` |
//! | a Spark type Krishiv does not support yet | `KRISHIV_UNSUPPORTED_FEATURE` |
//! | bare `ARRAY`, `MAP` or `STRUCT` | `INCOMPLETE_TYPE_DEFINITION.*` |
//! | a type nested deeper than [`crate::DataType::MAX_NESTING_DEPTH`] | `FAILED_TO_PARSE_TOO_COMPLEX` |
//! | JSON that is not a Spark type | `INVALID_JSON_DATA_TYPE` |
//! | a [`crate::batch::Batch`] invariant broken by the caller (an engine bug) | `INTERNAL_ERROR` |

use krishiv_common::error::{
    DecimalPrecisionExceedsMaxPrecision, FailedToParseTooComplex, IncompleteTypeDefinitionArray,
    IncompleteTypeDefinitionMap, IncompleteTypeDefinitionStruct, InternalError, InvalidIdentifier,
    InvalidJsonDataType, KrishivError, KrishivUnsupportedFeature, LegacyErrorTemp1228,
    NegativeScaleDisallowed, ParseEmptyStatement, ParseSyntaxError, StInvalidSridValue,
    UnclosedBracketedComment, UnsupportedDatatype, UnsupportedTimePrecision, WkbParseError,
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

/// `PARSE_SYNTAX_ERROR` at `at` where Spark reports a `missing` token.
pub(crate) fn missing(at: &str, missing: &str) -> KrishivError {
    ParseSyntaxError {
        error: at.to_owned(),
        hint: format!(": missing {missing}"),
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

/// `UNSUPPORTED_TIME_PRECISION`; `precision` is the value as written.
pub(crate) fn time_precision(precision: &str) -> KrishivError {
    UnsupportedTimePrecision {
        precision: precision.to_owned(),
    }
    .into()
}

/// `ST_INVALID_SRID_VALUE`; `srid` is the value as written.
pub(crate) fn invalid_srid(srid: &str) -> KrishivError {
    StInvalidSridValue {
        srid: srid.to_owned(),
    }
    .into()
}

/// `WKB_PARSE_ERROR` with Spark's message for the problem at byte `pos`.
pub(crate) fn wkb(message: impl Into<String>, pos: usize) -> KrishivError {
    WkbParseError {
        parse_error: message.into(),
        pos: pos.to_string(),
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

/// `INCOMPLETE_TYPE_DEFINITION.*` for a bare `array`, `map` or `struct`
/// (`name` lower-case); `None` for another name.
pub(crate) fn incomplete_type(name: &str) -> Option<KrishivError> {
    Some(match name {
        "array" => IncompleteTypeDefinitionArray {
            element_type: "<INT>".to_owned(),
        }
        .into(),
        "map" => IncompleteTypeDefinitionMap {}.into(),
        "struct" => IncompleteTypeDefinitionStruct {}.into(),
        _ => return None,
    })
}

/// `FAILED_TO_PARSE_TOO_COMPLEX`, for a type nested deeper than
/// [`crate::DataType::MAX_NESTING_DEPTH`].
pub(crate) fn too_deep() -> KrishivError {
    FailedToParseTooComplex {}.into()
}

/// `INVALID_JSON_DATA_TYPE`; `json` is the offending JSON, as Spark quotes it.
pub(crate) fn invalid_json_type(json: &str) -> KrishivError {
    InvalidJsonDataType {
        invalid_type: json.to_owned(),
    }
    .into()
}
