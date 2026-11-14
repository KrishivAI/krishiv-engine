// SPDX-License-Identifier: Apache-2.0
//! A [`KrishivError`] as the gRPC status Spark Connect sends
//! (`ErrorUtils.buildStatusFromThrowable`, Spark 4.2.0): code `INTERNAL`,
//! the error's message, and a `google.rpc.ErrorInfo` detail in the domain
//! `org.apache.spark` naming the exception class and its superclasses, the
//! SQLSTATE, the error class and its message parameters. PySpark raises the
//! matching Python exception from that detail. Krishiv's own refusals use
//! other codes: `RESOURCE_EXHAUSTED` for the session and operation limits,
//! `PERMISSION_DENIED` for `KRISHIV_INSUFFICIENT_PERMISSIONS` (ADR 0046 §8).
//!
//! Only the client-safe form of an error leaves ([`KrishivError::for_client`]).
//! The exception classes are the ones Spark raises where the Connect server
//! itself throws (`SparkSQLException`, `InvalidPlanInput`), and otherwise
//! chosen from the error class and SQLSTATE (`exception_classes`) so that
//! PySpark raises the Python exception Spark's would. `errorId` and
//! `stackTrace` (Spark's enriched errors and `FetchErrorDetails`) are
//! PR-15b's (design P3-T011a §3.3, §3.10).

use std::collections::BTreeMap;

use krishiv_common::error::{ErrorClass, KrishivError};
use prost::Message;
use tonic::{Code, Status};

/// `google.rpc.Status`.
#[derive(Clone, PartialEq, prost::Message)]
struct RpcStatus {
    #[prost(int32, tag = "1")]
    code: i32,
    #[prost(string, tag = "2")]
    message: String,
    #[prost(message, repeated, tag = "3")]
    details: Vec<prost_types::Any>,
}

/// `google.rpc.ErrorInfo`.
#[derive(Clone, PartialEq, prost::Message)]
struct ErrorInfo {
    #[prost(string, tag = "1")]
    reason: String,
    #[prost(string, tag = "2")]
    domain: String,
    #[prost(btree_map = "string, string", tag = "3")]
    metadata: BTreeMap<String, String>,
}

/// `java.lang.Exception` and above, as `ErrorUtils.allClasses` ends every
/// list (it appends `java.lang.Object`).
macro_rules! classes {
    ($($c:literal),* $(,)?) => {
        &[$($c,)* "java.lang.Exception", "java.lang.Throwable", "java.lang.Object"]
    };
}

/// `SparkSQLException` (`INVALID_HANDLE.*`, `INVALID_CURSOR.*` and the plan
/// decompression errors: `SparkConnectExecutionManager`, the Reattach and
/// ReleaseExecute handlers and `PlanCompressionUtils` at v4.3.0).
const SPARK_SQL_EXCEPTION: &[&str] = classes!(
    "org.apache.spark.SparkSQLException",
    "java.sql.SQLException"
);
/// `SparkNoSuchElementException` (`SQL_CONF_NOT_FOUND`,
/// `QueryExecutionErrors.sqlConfigNotFoundError`, v4.3.0 line 1985).
const SPARK_NO_SUCH_ELEMENT_EXCEPTION: &[&str] = classes!(
    "org.apache.spark.SparkNoSuchElementException",
    "java.util.NoSuchElementException",
    "java.lang.RuntimeException",
);
/// `SparkException`.
const SPARK_EXCEPTION: &[&str] = classes!("org.apache.spark.SparkException");
/// `InvalidPlanInput` (`CONNECT_INVALID_PLAN.*` from the planner;
/// `sql/connect/common/.../InvalidPlanInput.scala`, `extends Exception`).
const INVALID_PLAN_INPUT: &[&str] =
    classes!("org.apache.spark.sql.connect.common.InvalidPlanInput");
/// `AnalysisException` (`extends Exception`).
const ANALYSIS_EXCEPTION: &[&str] = classes!("org.apache.spark.sql.AnalysisException");
/// `ParseException` (`extends AnalysisException`).
const PARSE_EXCEPTION: &[&str] = classes!(
    "org.apache.spark.sql.catalyst.parser.ParseException",
    "org.apache.spark.sql.AnalysisException",
);
/// `SparkRuntimeException` (`extends RuntimeException`).
const SPARK_RUNTIME_EXCEPTION: &[&str] = classes!(
    "org.apache.spark.SparkRuntimeException",
    "java.lang.RuntimeException",
);
/// `SparkNumberFormatException`.
const SPARK_NUMBER_FORMAT_EXCEPTION: &[&str] = classes!(
    "org.apache.spark.SparkNumberFormatException",
    "java.lang.NumberFormatException",
    "java.lang.IllegalArgumentException",
    "java.lang.RuntimeException",
);
/// `SparkArithmeticException`.
const SPARK_ARITHMETIC_EXCEPTION: &[&str] = classes!(
    "org.apache.spark.SparkArithmeticException",
    "java.lang.ArithmeticException",
    "java.lang.RuntimeException",
);
/// `SparkDateTimeException`.
const SPARK_DATE_TIME_EXCEPTION: &[&str] = classes!(
    "org.apache.spark.SparkDateTimeException",
    "java.time.DateTimeException",
    "java.lang.RuntimeException",
);
/// `SparkIllegalArgumentException`.
const SPARK_ILLEGAL_ARGUMENT_EXCEPTION: &[&str] = classes!(
    "org.apache.spark.SparkIllegalArgumentException",
    "java.lang.IllegalArgumentException",
    "java.lang.RuntimeException",
);
/// `SparkUnsupportedOperationException`.
const SPARK_UNSUPPORTED_OPERATION_EXCEPTION: &[&str] = classes!(
    "org.apache.spark.SparkUnsupportedOperationException",
    "java.lang.UnsupportedOperationException",
    "java.lang.RuntimeException",
);
/// `SparkArrayIndexOutOfBoundsException`.
const SPARK_ARRAY_INDEX_EXCEPTION: &[&str] = classes!(
    "org.apache.spark.SparkArrayIndexOutOfBoundsException",
    "java.lang.ArrayIndexOutOfBoundsException",
    "java.lang.IndexOutOfBoundsException",
    "java.lang.RuntimeException",
);
/// `SparkUpgradeException` (`extends RuntimeException`).
const SPARK_UPGRADE_EXCEPTION: &[&str] = classes!(
    "org.apache.spark.SparkUpgradeException",
    "java.lang.RuntimeException",
);

/// The Java class hierarchy Spark would raise for `class`, as
/// `ErrorUtils.allClasses` lists it. PySpark 4.3.0
/// (`pyspark/errors/exceptions/connect.py`, `EXCEPTION_CLASS_MAPPING`) picks
/// the Python exception from the first entry it knows, so the list decides
/// whether the client raises `AnalysisException`, `ParseException`,
/// `NumberFormatException`, ... Krishiv does not carry the thrown class, so
/// it is chosen from the class name and SQLSTATE with the rules of the
/// Python facade (`python/krishiv/errors.py`, `_class_for`).
pub(crate) fn exception_classes(class: ErrorClass) -> &'static [&'static str] {
    let name = class.name();
    let state = class.sql_state();
    let by_prefix: &[(&str, &[&str])] = &[
        ("INVALID_HANDLE.", SPARK_SQL_EXCEPTION),
        ("INVALID_CURSOR.", SPARK_SQL_EXCEPTION),
        (
            "CONNECT_INVALID_PLAN.PLAN_SIZE_LARGER_THAN_MAX",
            SPARK_SQL_EXCEPTION,
        ),
        ("CONNECT_INVALID_PLAN.CANNOT_PARSE", SPARK_SQL_EXCEPTION),
        ("CONNECT_INVALID_PLAN.", INVALID_PLAN_INPUT),
        // The Interrupt handler throws `SparkSQLException` (v4.3.0 lines 45-61).
        (
            "INVALID_PARAMETER_VALUE.INTERRUPT_TYPE_",
            SPARK_SQL_EXCEPTION,
        ),
        ("UNSUPPORTED_FEATURE.INTERRUPT_TYPE", SPARK_SQL_EXCEPTION),
        ("SQL_CONF_NOT_FOUND", SPARK_NO_SUCH_ELEMENT_EXCEPTION),
        // `QueryCompilationErrors.cannotModifyValueOf*ConfigError`.
        ("CANNOT_MODIFY_", ANALYSIS_EXCEPTION),
        ("KRISHIV_SESSION_LIMIT", SPARK_EXCEPTION),
        ("KRISHIV_OPERATION_LIMIT", SPARK_EXCEPTION),
        ("INTERNAL_ERROR", SPARK_EXCEPTION),
        // Spark's `ExecuteThreadRunner` throws a `SparkSQLException`
        // (v4.3.0, lines 105 and 175): PySpark raises `UnknownException`.
        ("OPERATION_CANCELED", SPARK_SQL_EXCEPTION),
        ("CAST_INVALID_INPUT", SPARK_NUMBER_FORMAT_EXCEPTION),
        (
            "INCONSISTENT_BEHAVIOR_CROSS_VERSION",
            SPARK_UPGRADE_EXCEPTION,
        ),
        ("INVALID_ARRAY_INDEX", SPARK_ARRAY_INDEX_EXCEPTION),
        ("INVALID_PARAMETER_VALUE", SPARK_ILLEGAL_ARGUMENT_EXCEPTION),
        ("PARSE_", PARSE_EXCEPTION),
        ("UNSUPPORTED_DATATYPE", PARSE_EXCEPTION),
        ("INVALID_SQL_SYNTAX", PARSE_EXCEPTION),
    ];
    if let Some((_, classes)) = by_prefix.iter().find(|(p, _)| name.starts_with(p)) {
        return classes;
    }
    match state {
        Some("42601") => PARSE_EXCEPTION,
        Some("22012" | "22003") => SPARK_ARITHMETIC_EXCEPTION,
        Some("22007" | "22008") => SPARK_DATE_TIME_EXCEPTION,
        Some("22023") => SPARK_ILLEGAL_ARGUMENT_EXCEPTION,
        Some("0A000") => SPARK_UNSUPPORTED_OPERATION_EXCEPTION,
        Some(s) if s.starts_with("42") => ANALYSIS_EXCEPTION,
        _ => SPARK_RUNTIME_EXCEPTION,
    }
}

/// The gRPC status for `error`, in Spark Connect's shape.
#[must_use]
pub fn status_of(error: &KrishivError) -> Status {
    let error = error.for_client();
    let error = error.error();
    let class = error.error_class();
    let classes = exception_classes(class);
    let mut metadata = BTreeMap::new();
    metadata.insert("classes".to_owned(), json_array(classes));
    if let Some(state) = error.sql_state() {
        metadata.insert("sqlState".to_owned(), state.to_owned());
    }
    // Spark leaves both out when the parameters exceed
    // `spark.connect.grpc.maxMetadataSize` (1024 by default).
    let parameters = json_object(error.message_parameters());
    if utf16_len(&parameters) <= MAX_METADATA_SIZE {
        metadata.insert("errorClass".to_owned(), class.name().to_owned());
        metadata.insert("messageParameters".to_owned(), parameters);
    }
    let info = ErrorInfo {
        reason: classes.first().copied().unwrap_or_default().to_owned(),
        domain: "org.apache.spark".to_owned(),
        metadata,
    };
    // Spark sends INTERNAL for every SparkThrowable; a session limit is a
    // refusal before admission (threat model T-D4), RESOURCE_EXHAUSTED; an
    // authorization refusal is PERMISSION_DENIED (ADR 0046 §8).
    let code = match class {
        ErrorClass::KrishivSessionLimit | ErrorClass::KrishivOperationLimit => {
            Code::ResourceExhausted
        }
        ErrorClass::KrishivInsufficientPermissions => Code::PermissionDenied,
        _ => Code::Internal,
    };
    let message = abbreviate(&error.to_string(), MAX_MESSAGE);
    let details = RpcStatus {
        code: code as i32,
        message: message.clone(),
        details: vec![prost_types::Any {
            type_url: "type.googleapis.com/google.rpc.ErrorInfo".to_owned(),
            value: info.encode_to_vec(),
        }],
    };
    Status::with_details(code, message, details.encode_to_vec().into())
}

/// Spark's `spark.connect.grpc.maxMetadataSize` default, in UTF-16 units
/// (Java string length).
const MAX_METADATA_SIZE: usize = 1024;

/// The longest status message Spark sends (`extractErrorMessage`).
const MAX_MESSAGE: usize = 2048;

fn utf16_len(s: &str) -> usize {
    s.chars().map(char::len_utf16).sum()
}

/// Spark's `abbreviate(s, len)`: `s` if it fits in `len` UTF-16 units,
/// else its first `len - 3` units and `...`.
fn abbreviate(s: &str, len: usize) -> String {
    if utf16_len(s) <= len {
        return s.to_owned();
    }
    let mut out = String::new();
    let mut used = 0;
    for c in s.chars() {
        let w = c.len_utf16();
        if used + w > len - 3 {
            break;
        }
        used += w;
        out.push(c);
    }
    out.push_str("...");
    out
}

/// A JSON array of strings (Jackson's compact form).
fn json_array(items: &[&str]) -> String {
    serde_json::Value::from(items.to_vec()).to_string()
}

/// A JSON object of string values.
fn json_object(pairs: &[(&'static str, String)]) -> String {
    let map: serde_json::Map<String, serde_json::Value> = pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), serde_json::Value::from(v.as_str())))
        .collect();
    serde_json::Value::Object(map).to_string()
}

#[cfg(test)]
mod tests;
