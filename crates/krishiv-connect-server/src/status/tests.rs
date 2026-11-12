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
    assert_eq!(Some(info.metadata["sqlState"].as_str()), e.sql_state());
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
