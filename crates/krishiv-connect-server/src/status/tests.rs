// SPDX-License-Identifier: Apache-2.0
use krishiv_common::error::{InvalidHandleSessionClosed, KrishivError, KrishivSessionLimit};
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
    assert_eq!(
        first(ErrorClass::OperationCanceled),
        "org.apache.spark.SparkSQLException"
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
