// SPDX-License-Identifier: Apache-2.0
//! A [`KrishivError`] as the gRPC status Spark Connect sends
//! (`ErrorUtils.buildStatusFromThrowable`, Spark 4.2.0): code `INTERNAL`,
//! the error's message, and a `google.rpc.ErrorInfo` detail in the domain
//! `org.apache.spark` naming the exception class and its superclasses, the
//! SQLSTATE, the error class and its message parameters. PySpark raises the
//! matching Python exception from that detail.
//!
//! Only the client-safe form of an error leaves ([`KrishivError::for_client`]).
//! The exception class is the one Spark throws for the error classes the
//! session RPCs raise (`INVALID_HANDLE.*`: `SparkSQLException`); every other
//! class is sent as a `SparkException` until T011a maps the full set.
//! `errorId` and `stackTrace` (Spark's enriched errors and
//! `FetchErrorDetails`) are T011a's.

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

/// The Java classes of `SparkSQLException`, itself first, as
/// `ErrorUtils.allClasses` lists them (it appends `java.lang.Object`).
const SPARK_SQL_EXCEPTION: &[&str] = &[
    "org.apache.spark.SparkSQLException",
    "java.sql.SQLException",
    "java.lang.Exception",
    "java.lang.Throwable",
    "java.lang.Object",
];

/// The Java classes of `SparkException`.
const SPARK_EXCEPTION: &[&str] = &[
    "org.apache.spark.SparkException",
    "java.lang.Exception",
    "java.lang.Throwable",
    "java.lang.Object",
];

/// The gRPC status for `error`, in Spark Connect's shape.
#[must_use]
pub fn status_of(error: &KrishivError) -> Status {
    let error = error.for_client();
    let error = error.error();
    let class = error.error_class();
    let classes = if class.name().starts_with("INVALID_HANDLE.") {
        SPARK_SQL_EXCEPTION
    } else {
        SPARK_EXCEPTION
    };
    let mut metadata = BTreeMap::new();
    metadata.insert("classes".to_owned(), json_array(classes));
    if let Some(state) = error.sql_state() {
        metadata.insert("sqlState".to_owned(), state.to_owned());
    }
    metadata.insert("errorClass".to_owned(), class.name().to_owned());
    metadata.insert(
        "messageParameters".to_owned(),
        json_object(error.message_parameters()),
    );
    let info = ErrorInfo {
        reason: classes.first().copied().unwrap_or_default().to_owned(),
        domain: "org.apache.spark".to_owned(),
        metadata,
    };
    // Spark sends INTERNAL for every SparkThrowable; a session limit is a
    // refusal before admission (threat model T-D4), RESOURCE_EXHAUSTED.
    let code = if class == ErrorClass::KrishivSessionLimit {
        Code::ResourceExhausted
    } else {
        Code::Internal
    };
    let message = error.to_string();
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
