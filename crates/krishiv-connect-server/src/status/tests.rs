// SPDX-License-Identifier: Apache-2.0
use krishiv_common::error::{
    InvalidHandleSessionClosed, KrishivError, KrishivQueryKilled, KrishivQueryTimeout,
    KrishivSessionLimit,
};
use prost::Message;
use tonic::Code;

use super::{ErrorInfo, RpcStatus, status_of};

#[expect(
    clippy::disallowed_methods,
    reason = "the test reads back a message it built itself"
)]
fn details(status: &tonic::Status) -> (RpcStatus, ErrorInfo) {
    let rpc = RpcStatus::decode(status.details()).expect("google.rpc.Status");
    let info = ErrorInfo::decode(rpc.details[0].value.as_slice()).expect("ErrorInfo");
    (rpc, info)
}

#[test]
fn an_invalid_handle_is_sent_as_sparks_sql_exception() {
    let e: KrishivError = InvalidHandleSessionClosed {
        handle: "abc".to_owned(),
    }
    .into();
    let status = status_of(&e);
    assert_eq!(status.code(), Code::Internal);
    assert_eq!(status.message(), e.to_string());
    assert!(
        status
            .message()
            .starts_with("[INVALID_HANDLE.SESSION_CLOSED] ")
    );
    let (rpc, info) = details(&status);
    assert_eq!(rpc.code, Code::Internal as i32);
    assert_eq!(rpc.message, status.message());
    assert_eq!(
        rpc.details[0].type_url,
        "type.googleapis.com/google.rpc.ErrorInfo"
    );
    assert_eq!(info.reason, "org.apache.spark.SparkSQLException");
    assert_eq!(info.domain, "org.apache.spark");
    assert_eq!(
        info.metadata["classes"],
        r#"["org.apache.spark.SparkSQLException","java.sql.SQLException","java.lang.Exception","java.lang.Throwable","java.lang.Object"]"#
    );
    assert_eq!(info.metadata["errorClass"], "INVALID_HANDLE.SESSION_CLOSED");
    // The vendored 4.2.0 registry's SQLSTATE (4.3.0 moves it to 08003).
    assert_eq!(info.metadata["sqlState"], "HY000");
    assert_eq!(e.sql_state(), Some("HY000"));
    assert_eq!(info.metadata["messageParameters"], r#"{"handle":"abc"}"#);
}

#[test]
fn a_session_limit_is_resource_exhausted() {
    let e: KrishivError = KrishivSessionLimit {
        limit_name: "krishiv.connect.maxSessions".to_owned(),
        limit: "2".to_owned(),
    }
    .into();
    let status = status_of(&e);
    assert_eq!(status.code(), Code::ResourceExhausted);
    let (rpc, info) = details(&status);
    assert_eq!(rpc.code, Code::ResourceExhausted as i32);
    assert_eq!(info.reason, "org.apache.spark.SparkException");
    assert_eq!(info.metadata["errorClass"], "KRISHIV_SESSION_LIMIT");
    assert_eq!(info.metadata["sqlState"], "53400");
}

#[test]
fn parameters_are_json_escaped() {
    let e: KrishivError = InvalidHandleSessionClosed {
        handle: "a\"b\\c\n".to_owned(),
    }
    .into();
    let (_, info) = details(&status_of(&e));
    let parsed: serde_json::Value =
        serde_json::from_str(&info.metadata["messageParameters"]).expect("json");
    assert_eq!(parsed["handle"], "a\"b\\c\n");
}

#[test]
fn a_long_handle_is_bounded_as_spark_bounds_it() {
    let e: KrishivError = InvalidHandleSessionClosed {
        handle: "h".repeat(5000),
    }
    .into();
    let status = status_of(&e);
    assert_eq!(status.message().chars().count(), 2048);
    assert!(status.message().ends_with("..."));
    let (rpc, info) = details(&status);
    assert_eq!(rpc.message, status.message());
    // Parameters past 1024 bytes: errorClass and messageParameters left out.
    assert!(!info.metadata.contains_key("errorClass"));
    assert!(!info.metadata.contains_key("messageParameters"));
    assert!(info.metadata.contains_key("sqlState"));
}

#[test]
fn exception_classes_pick_the_python_exception_spark_would_raise() {
    use krishiv_common::error::ErrorClass;

    use super::exception_classes;
    let first = |c: ErrorClass| exception_classes(c)[0];
    assert_eq!(
        first(ErrorClass::InvalidCursorDisconnected),
        "org.apache.spark.SparkSQLException"
    );
    assert_eq!(
        first(ErrorClass::ConnectInvalidPlanPlanSizeLargerThanMax),
        "org.apache.spark.SparkSQLException"
    );
    assert_eq!(
        first(ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet),
        "org.apache.spark.sql.connect.common.InvalidPlanInput"
    );
    assert_eq!(
        first(ErrorClass::KrishivOperationLimit),
        "org.apache.spark.SparkException"
    );
    // A null config value is Spark's `require`: PySpark raises `IllegalArgumentException`.
    assert_eq!(
        first(ErrorClass::InvalidConfValueRequirement),
        "java.lang.IllegalArgumentException"
    );
    assert_eq!(
        first(ErrorClass::OperationCanceled),
        "org.apache.spark.SparkSQLException"
    );
    // Spark raises both artifact errors as `SparkRuntimeException`
    // (PySpark's `SparkRuntimeException`, not `AnalysisException`).
    for c in [
        ErrorClass::ArtifactAlreadyExists,
        ErrorClass::InvalidArtifactPath,
    ] {
        assert_eq!(first(c), "org.apache.spark.SparkRuntimeException");
    }
    assert_eq!(
        first(ErrorClass::KrishivArtifactLimit),
        "org.apache.spark.SparkException"
    );
    assert_eq!(
        exception_classes(ErrorClass::CastInvalidInput)[1],
        "java.lang.NumberFormatException"
    );
    assert_eq!(
        exception_classes(ErrorClass::DivideByZero)[1],
        "java.lang.ArithmeticException"
    );
    assert_eq!(
        exception_classes(ErrorClass::KrishivInvalidOperationTag)[1],
        "java.lang.IllegalArgumentException"
    );
    assert_eq!(
        exception_classes(ErrorClass::KrishivUnsupportedFeature)[1],
        "java.lang.UnsupportedOperationException"
    );
    assert_eq!(
        first(ErrorClass::UnresolvedColumnWithSuggestion),
        "org.apache.spark.sql.AnalysisException"
    );
    assert_eq!(
        first(ErrorClass::ParseSyntaxError),
        "org.apache.spark.sql.catalyst.parser.ParseException"
    );
    // SQL-18: unnamed errors by their number range (no SQLSTATE).
    assert_eq!(
        first(ErrorClass::LegacyErrorTemp1035),
        "org.apache.spark.sql.AnalysisException"
    );
    assert_eq!(
        first(ErrorClass::LegacyErrorTemp0035),
        "org.apache.spark.sql.catalyst.parser.ParseException"
    );
    // Every list ends as ErrorUtils.allClasses ends it.
    for c in [ErrorClass::DivideByZero, ErrorClass::InvalidHandleFormat] {
        let list = exception_classes(c);
        assert_eq!(
            &list[list.len() - 3..],
            &[
                "java.lang.Exception",
                "java.lang.Throwable",
                "java.lang.Object"
            ]
        );
    }
}

#[test]
fn an_operation_limit_is_resource_exhausted() {
    let e: KrishivError = krishiv_common::error::KrishivOperationLimit {
        limit_name: "krishiv.connect.maxOperationsPerSession".to_owned(),
        limit: "1".to_owned(),
    }
    .into();
    assert_eq!(status_of(&e).code(), Code::ResourceExhausted);
}

/// ADR 0046 §8: an authorization refusal is `PERMISSION_DENIED`, with the
/// same `ErrorInfo` as every other error.
#[test]
fn insufficient_permissions_is_permission_denied() {
    let e: KrishivError = krishiv_common::error::KrishivInsufficientPermissions {
        principal: "ann".to_owned(),
        privilege: "SET".to_owned(),
        securable: "configuration `krishiv.resultCache.ttl`".to_owned(),
    }
    .into();
    let status = status_of(&e);
    assert_eq!(status.code(), Code::PermissionDenied);
    let (rpc, info) = details(&status);
    assert_eq!(rpc.code, Code::PermissionDenied as i32);
    assert_eq!(
        info.metadata["errorClass"],
        "KRISHIV_INSUFFICIENT_PERMISSIONS"
    );
    assert_eq!(info.metadata["sqlState"], "42501");
}

/// P3-T061 §4.2 rule 1: a refused act-as request is `PERMISSION_DENIED` too.
#[test]
fn impersonation_denied_is_permission_denied() {
    let e: KrishivError = krishiv_common::error::KrishivAuthImpersonationDenied {}.into();
    let status = status_of(&e);
    assert_eq!(status.code(), Code::PermissionDenied);
    let (_, info) = details(&status);
    assert_eq!(
        info.metadata["errorClass"],
        "KRISHIV_AUTH_IMPERSONATION_DENIED"
    );
    assert_eq!(info.metadata["sqlState"], "42501");
}

#[test]
fn a_python_udf_error_is_sparks_python_exception_and_a_crash_a_spark_exception() {
    use krishiv_common::error::{KrishivPythonWorkerExited, PythonException};
    let e: KrishivError = PythonException {
        msg: "An exception was thrown from the Python worker".to_owned(),
        traceback: "Traceback (most recent call last):\nValueError: boom".to_owned(),
    }
    .into();
    let status = status_of(&e);
    assert_eq!(status.code(), Code::Internal);
    assert!(
        status.message().starts_with("[PYTHON_EXCEPTION] "),
        "{}",
        status.message()
    );
    assert!(status.message().contains("ValueError: boom"));
    let (_, info) = details(&status);
    assert_eq!(info.reason, "org.apache.spark.api.python.PythonException");
    assert_eq!(
        info.metadata["classes"],
        r#"["org.apache.spark.api.python.PythonException","java.lang.RuntimeException","java.lang.Exception","java.lang.Throwable","java.lang.Object"]"#
    );
    assert_eq!(info.metadata["sqlState"], "38000");
    assert_eq!(info.metadata["errorClass"], "PYTHON_EXCEPTION");
    let crashed: KrishivError = KrishivPythonWorkerExited {
        reason: "killed by signal 9".to_owned(),
    }
    .into();
    let (_, info) = details(&status_of(&crashed));
    assert_eq!(info.reason, "org.apache.spark.SparkException");
    assert_eq!(info.metadata["errorClass"], "KRISHIV_PYTHON_WORKER_EXITED");
}

#[test]
fn a_timeout_and_a_kill_are_sparks_sql_exception_with_their_own_class() {
    let timeout: KrishivError = KrishivQueryTimeout {
        query_id: "q1".to_owned(),
        limit: "300ms".to_owned(),
    }
    .into();
    let killed: KrishivError = KrishivQueryKilled {
        query_id: "q2".to_owned(),
        reason: "an operator cancelled it".to_owned(),
    }
    .into();
    for (e, class) in [
        (timeout, "KRISHIV_QUERY_TIMEOUT"),
        (killed, "KRISHIV_QUERY_KILLED"),
    ] {
        let (_, info) = details(&status_of(&e));
        assert_eq!(info.reason, "org.apache.spark.SparkSQLException");
        assert_eq!(info.metadata["errorClass"], class);
        assert_eq!(info.metadata["sqlState"], "HY008");
    }
}
