// SPDX-License-Identifier: Apache-2.0
//! `flight_status`: a `KrishivError` as the gRPC status an Arrow Flight SQL client gets (design
//! P3-T101 §6). Connect's `status_of` is Spark-specific (always `INTERNAL` plus Java class
//! names) and stays Connect's; only the client-safe form of the error is shared.

use krishiv_common::error::KrishivError;
use tonic::metadata::MetadataValue;
use tonic::{Code, Status};

/// Trailing metadata: the error class of a refused call.
pub const ERROR_CLASS_KEY: &str = "x-krishiv-error-class";
/// Trailing metadata: the SQLSTATE of a refused call.
pub const SQL_STATE_KEY: &str = "x-krishiv-sqlstate";
/// Trailing metadata: the id the server's log knows the full error by.
pub const ERROR_ID_KEY: &str = "x-krishiv-error-id";

/// `error` as a Flight SQL status: the code of [`code_of`], the message Spark prints
/// (`[ERROR_CLASS] message SQLSTATE: xxxxx`) of the client-safe form
/// ([`KrishivError::for_client`]: no cause chain, operator-only parameters redacted), and the
/// class, SQLSTATE and error id as trailing metadata. No stack trace, path or address.
#[must_use]
pub fn flight_status(error: &KrishivError) -> Status {
    let client = error.for_client();
    let safe = client.error();
    let class = safe.error_class().name();
    let state = safe.sql_state();
    let mut status = Status::new(code_of(class, state), safe.to_string());
    let metadata = status.metadata_mut();
    if let Ok(v) = MetadataValue::try_from(class) {
        metadata.insert(ERROR_CLASS_KEY, v);
    }
    if let Some(v) = state.and_then(|s| MetadataValue::try_from(s).ok()) {
        metadata.insert(SQL_STATE_KEY, v);
    }
    if let Ok(v) = MetadataValue::try_from(client.error_id()) {
        metadata.insert(ERROR_ID_KEY, v);
    }
    status
}

/// The gRPC code of an error class and SQLSTATE (design §6): specific classes first, then the
/// SQLSTATE class (its first two characters), else `INTERNAL`.
#[must_use]
pub fn code_of(class: &str, sql_state: Option<&str>) -> Code {
    match class {
        "KRISHIV_INSUFFICIENT_PERMISSIONS" => return Code::PermissionDenied,
        "OPERATION_CANCELED" | "KRISHIV_QUERY_KILLED" => return Code::Cancelled,
        "KRISHIV_QUERY_TIMEOUT" | "KRISHIV_FLIGHT_SQL_FETCH_IDLE" => return Code::DeadlineExceeded,
        "INVALID_HANDLE.FORMAT" => return Code::InvalidArgument,
        "INVALID_HANDLE.OPERATION_ALREADY_EXISTS" => return Code::AlreadyExists,
        "KRISHIV_SESSION_LIMIT" | "KRISHIV_OPERATION_LIMIT" | "KRISHIV_FLIGHT_SQL_HANDLE_LIMIT" => {
            return Code::ResourceExhausted;
        }
        "KRISHIV_FLIGHT_SQL_SCHEMA_CHANGED" => return Code::FailedPrecondition,
        "TABLE_OR_VIEW_NOT_FOUND" | "SCHEMA_NOT_FOUND" | "UNRESOLVED_ROUTINE" => {
            return Code::NotFound;
        }
        _ => {}
    }
    if class == "INVALID_HANDLE" || class.starts_with("INVALID_HANDLE.") {
        return Code::NotFound;
    }
    match sql_state.and_then(|s| s.get(..2)) {
        Some("42" | "22" | "23" | "54" | "07") => Code::InvalidArgument,
        Some("0A") => Code::Unimplemented,
        Some("40") => Code::Aborted,
        Some("53") => Code::ResourceExhausted,
        Some("25" | "55") => Code::FailedPrecondition,
        _ => Code::Internal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_common::error::{KrishivError, KrishivSessionLimit, TableOrViewNotFound};

    #[test]
    fn each_row_of_the_design_table() {
        for (class, state, code) in [
            (
                "KRISHIV_INSUFFICIENT_PERMISSIONS",
                Some("42501"),
                Code::PermissionDenied,
            ),
            ("TABLE_OR_VIEW_NOT_FOUND", Some("42P01"), Code::NotFound),
            ("SCHEMA_NOT_FOUND", Some("42704"), Code::NotFound),
            ("UNRESOLVED_ROUTINE", Some("42883"), Code::NotFound),
            (
                "INVALID_HANDLE.SESSION_CLOSED",
                Some("HY000"),
                Code::NotFound,
            ),
            ("PARSE_SYNTAX_ERROR", Some("42601"), Code::InvalidArgument),
            ("DIVIDE_BY_ZERO", Some("22012"), Code::InvalidArgument),
            ("X_23", Some("23505"), Code::InvalidArgument),
            ("X_54", Some("54000"), Code::InvalidArgument),
            ("X_07", Some("07001"), Code::InvalidArgument),
            ("UNSUPPORTED_FEATURE.X", Some("0A000"), Code::Unimplemented),
            ("X_53", Some("53200"), Code::ResourceExhausted),
            (
                "KRISHIV_SESSION_LIMIT",
                Some("53400"),
                Code::ResourceExhausted,
            ),
            (
                "KRISHIV_OPERATION_LIMIT",
                Some("53400"),
                Code::ResourceExhausted,
            ),
            (
                "KRISHIV_FLIGHT_SQL_HANDLE_LIMIT",
                Some("53400"),
                Code::ResourceExhausted,
            ),
            ("OPERATION_CANCELED", Some("HY008"), Code::Cancelled),
            ("KRISHIV_QUERY_KILLED", Some("HY008"), Code::Cancelled),
            (
                "KRISHIV_QUERY_TIMEOUT",
                Some("HY008"),
                Code::DeadlineExceeded,
            ),
            (
                "KRISHIV_FLIGHT_SQL_FETCH_IDLE",
                Some("HY008"),
                Code::DeadlineExceeded,
            ),
            (
                "INVALID_HANDLE.FORMAT",
                Some("HY000"),
                Code::InvalidArgument,
            ),
            (
                "INVALID_HANDLE.OPERATION_ALREADY_EXISTS",
                Some("HY000"),
                Code::AlreadyExists,
            ),
            ("X_40", Some("40001"), Code::Aborted),
            ("X_25", Some("25000"), Code::FailedPrecondition),
            (
                "KRISHIV_FLIGHT_SQL_SCHEMA_CHANGED",
                Some("55000"),
                Code::FailedPrecondition,
            ),
            ("INTERNAL_ERROR", Some("XX000"), Code::Internal),
            ("X_HY", Some("HY000"), Code::Internal),
            ("X_NONE", None, Code::Internal),
        ] {
            assert_eq!(code_of(class, state), code, "{class} {state:?}");
        }
    }

    #[test]
    fn the_status_carries_sparks_message_and_the_class_in_metadata() {
        let e: KrishivError = TableOrViewNotFound {
            relation_name: "`t`".to_owned(),
            search_path: "[`spark_catalog`.`default`]".to_owned(),
        }
        .into();
        let s = flight_status(&e);
        assert_eq!(s.code(), Code::NotFound);
        assert!(
            s.message().starts_with("[TABLE_OR_VIEW_NOT_FOUND]"),
            "{}",
            s.message()
        );
        assert!(s.message().contains("SQLSTATE: 42P01"), "{}", s.message());
        let md = s.metadata();
        assert_eq!(md.get(ERROR_CLASS_KEY).unwrap(), "TABLE_OR_VIEW_NOT_FOUND");
        assert_eq!(md.get(SQL_STATE_KEY).unwrap(), "42P01");
        assert!(md.get(ERROR_ID_KEY).is_some());
    }

    #[test]
    fn a_limit_is_resource_exhausted() {
        let e: KrishivError = KrishivSessionLimit {
            limit_name: "krishiv.connect.maxSessions".to_owned(),
            limit: "1".to_owned(),
        }
        .into();
        assert_eq!(flight_status(&e).code(), Code::ResourceExhausted);
    }
}
