// SPDX-License-Identifier: Apache-2.0
//! Engine errors as JavaScript errors: the message is a JSON object
//! (`errorClass`, `sqlState`, `messageParameters`, `message`) that the package's
//! `embedded.ts` turns into a `KrishivError` (design/js-package-layout.md).

use krishiv_common::error::{InternalError, KrishivError};

/// The JSON text a JavaScript `Error` carries for `error`.
pub(crate) fn to_json(error: &KrishivError) -> String {
    let parameters: serde_json::Map<String, serde_json::Value> = error
        .message_parameters()
        .iter()
        .map(|(key, value)| ((*key).to_owned(), serde_json::Value::String(value.clone())))
        .collect();
    serde_json::json!({
        "errorClass": error.error_class().name(),
        "sqlState": error.sql_state(),
        "messageParameters": parameters,
        "message": error.to_string(),
    })
    .to_string()
}

/// The napi error for `error`.
pub(crate) fn to_napi(error: &KrishivError) -> napi::Error {
    napi::Error::new(napi::Status::GenericFailure, to_json(error))
}

/// An `INTERNAL_ERROR` for a failure of the binding itself (Arrow IPC encoding).
pub(crate) fn internal(context: &str, cause: impl std::fmt::Display) -> KrishivError {
    InternalError {
        message: format!("krishiv-node: {context}: {cause}"),
    }
    .into()
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::OperationCanceled;

    use super::*;

    #[test]
    fn the_json_names_the_class_state_and_parameters() {
        let error = internal("encode", "boom");
        let value: serde_json::Value = serde_json::from_str(&to_json(&error)).unwrap();
        assert_eq!(value["errorClass"], "INTERNAL_ERROR");
        assert_eq!(value["sqlState"], error.sql_state().unwrap());
        assert_eq!(
            value["messageParameters"]["message"],
            "krishiv-node: encode: boom"
        );
        assert_eq!(value["message"], error.to_string());
    }

    #[test]
    fn a_cancel_is_operation_canceled() {
        let error = KrishivError::from(OperationCanceled {});
        let value: serde_json::Value = serde_json::from_str(&to_json(&error)).unwrap();
        assert_eq!(value["errorClass"], "OPERATION_CANCELED");
        assert_eq!(value["sqlState"], "HY008");
    }
}
