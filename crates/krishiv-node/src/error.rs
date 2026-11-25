// SPDX-License-Identifier: Apache-2.0
//! Engine errors as JavaScript errors: the message is a JSON object
//! (`errorClass`, `sqlState`, `messageParameters`, `message`) that the package's
//! `embedded.ts` turns into a `KrishivError` (design/js-package-layout.md).

use std::any::Any;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
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
/// `INTERNAL_ERROR` for a panic caught at the Node-API boundary, with its message.
pub(crate) fn panicked(payload: &(dyn Any + Send)) -> KrishivError {
    let message = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with no message".to_owned());
    internal("panic", message)
}

/// Runs `f`; a panic becomes `INTERNAL_ERROR` instead of unwinding into Node-API's
/// `extern "C"` callback, which would abort the whole Node process.
pub(crate) fn guarded<T>(f: impl FnOnce() -> Result<T, KrishivError>) -> Result<T, KrishivError> {
    std::panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| Err(panicked(&*p)))
}

/// [`guarded`] for a future: a panic while it is polled becomes `INTERNAL_ERROR`, so
/// the promise rejects rather than never settling.
pub(crate) async fn guarded_async<T>(
    f: impl Future<Output = Result<T, KrishivError>>,
) -> Result<T, KrishivError> {
    AssertUnwindSafe(f)
        .catch_unwind()
        .await
        .unwrap_or_else(|p| Err(panicked(&*p)))
}

/// [`guarded`], then the error as the JSON a JavaScript `Error` carries.
pub(crate) fn napi_guarded<T>(f: impl FnOnce() -> Result<T, KrishivError>) -> napi::Result<T> {
    guarded(f).map_err(|e| to_napi(&e))
}

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
    fn a_panic_is_internal_error() {
        let e = guarded::<()>(|| panic!("boom in a kernel")).unwrap_err();
        assert_eq!(e.error_class().name(), "INTERNAL_ERROR");
        assert_eq!(
            e.message_parameters()
                .iter()
                .find(|(k, _)| *k == "message")
                .map(|(_, v)| v.as_str()),
            Some("krishiv-node: panic: boom in a kernel")
        );
        let formatted = guarded::<()>(|| panic!("{} rows", 3)).unwrap_err();
        assert!(formatted.to_string().contains("3 rows"));
        assert_eq!(guarded(|| Ok(7)).unwrap(), 7);
    }

    #[test]
    fn a_panic_while_polled_is_internal_error() {
        let e = futures::executor::block_on(guarded_async::<()>(async {
            futures::future::ready(()).await;
            panic!("boom in a step")
        }))
        .unwrap_err();
        assert_eq!(e.error_class().name(), "INTERNAL_ERROR");
        assert!(e.to_string().contains("boom in a step"));
        let json: serde_json::Value = serde_json::from_str(&to_json(&e)).unwrap();
        assert_eq!(json["errorClass"], "INTERNAL_ERROR");
    }

    #[test]
    fn a_cancel_is_operation_canceled() {
        let error = KrishivError::from(OperationCanceled {});
        let value: serde_json::Value = serde_json::from_str(&to_json(&error)).unwrap();
        assert_eq!(value["errorClass"], "OPERATION_CANCELED");
        assert_eq!(value["sqlState"], "HY008");
    }
}
