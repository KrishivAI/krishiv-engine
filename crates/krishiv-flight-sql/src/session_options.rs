// SPDX-License-Identifier: Apache-2.0
//! The session-option actions (design P3-T101 §3.2): `SetSessionOptions`, `GetSessionOptions`,
//! `CloseSession`. arrow-flight 60 has neither their messages nor a trait method for them, so the
//! messages are declared here from Arrow's `format/Flight.proto` (apache-arrow-19.0.0, the
//! "Session option messages" section), and the gate dispatches the actions itself. Their bodies
//! are the plain messages, not `Any` (arrow-java v19.0.0 `FlightClient.setSessionOptions`:
//! `request.serialize()`; the names are `FlightConstants.SET_SESSION_OPTIONS` and so on).
//!
//! - `SetSessionOptions`: options `catalog` and `schema` are `USE` (the session's current catalog
//!   and namespace); every other option is a `SET`: a `Startup` key is refused for everyone
//!   (`check_static`), any other key asks the session's policy as SQL `SET` does
//!   (`Session::authorize_set_key`), as Connect's `Config.set` does. A valueless option unsets
//!   the key. Each refused option is named in the result's `errors` (the call itself succeeds,
//!   as the protocol says); the call marks the session changed (§3.2).
//! - `GetSessionOptions`: the explicitly set keys, without the ones `Config` redacts, and the
//!   current `catalog` and `schema`.
//! - `CloseSession`: the cookie's session goes, with its handles; without a cookie, or with one
//!   this caller does not hold, it answers `CLOSED` and does nothing.

use std::collections::HashMap;

use futures::stream;
use krishiv_common::error::KrishivError;
use krishiv_connect_server::SessionOwner;
use krishiv_connect_server::config::{check_static, is_redacted};
use krishiv_session::Session;
use krishiv_session::auth::Authenticated;
use prost::Message;
use tonic::metadata::MetadataMap;
use tonic::{Response, Status};

use crate::cookies::{SessionCookie, session_cookie};
use crate::prepared::ActionStream;
use crate::server::FlightSqlServer;

/// The Flight action `SetSessionOptions`.
pub const SET_SESSION_OPTIONS: &str = "SetSessionOptions";
/// The Flight action `GetSessionOptions`.
pub const GET_SESSION_OPTIONS: &str = "GetSessionOptions";
/// The Flight action `CloseSession`.
pub const CLOSE_SESSION: &str = "CloseSession";

/// The option that sets the current catalog.
const CATALOG: &str = "catalog";
/// The option that sets the current namespace.
const SCHEMA: &str = "schema";

/// `SessionOptionValue`.
#[derive(Clone, PartialEq, Message)]
pub struct SessionOptionValue {
    /// The value; none asks to unset the option.
    #[prost(oneof = "OptionValue", tags = "1, 2, 3, 4, 5")]
    pub option_value: Option<OptionValue>,
}

/// `SessionOptionValue.option_value`.
#[derive(Clone, PartialEq, prost::Oneof)]
pub enum OptionValue {
    /// `string_value`.
    #[prost(string, tag = "1")]
    StringValue(String),
    /// `bool_value`.
    #[prost(bool, tag = "2")]
    BoolValue(bool),
    /// `int64_value` (`sfixed64`).
    #[prost(sfixed64, tag = "3")]
    Int64Value(i64),
    /// `double_value`.
    #[prost(double, tag = "4")]
    DoubleValue(f64),
    /// `string_list_value`.
    #[prost(message, tag = "5")]
    StringListValue(StringListValue),
}

/// `SessionOptionValue.StringListValue`.
#[derive(Clone, PartialEq, Message)]
pub struct StringListValue {
    /// `values`.
    #[prost(string, repeated, tag = "1")]
    pub values: Vec<String>,
}

/// `SetSessionOptionsRequest`.
#[derive(Clone, PartialEq, Message)]
pub struct SetSessionOptionsRequest {
    /// `session_options`.
    #[prost(map = "string, message", tag = "1")]
    pub session_options: HashMap<String, SessionOptionValue>,
}

/// `SetSessionOptionsResult.ErrorValue`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, prost::Enumeration)]
#[repr(i32)]
pub enum ErrorValue {
    /// `UNSPECIFIED`.
    Unspecified = 0,
    /// `INVALID_NAME`.
    InvalidName = 1,
    /// `INVALID_VALUE`.
    InvalidValue = 2,
    /// `ERROR`.
    Error = 3,
}

/// `SetSessionOptionsResult.Error`.
#[derive(Clone, PartialEq, Message)]
pub struct SetSessionOptionError {
    /// `value`.
    #[prost(enumeration = "ErrorValue", tag = "1")]
    pub value: i32,
}

/// `SetSessionOptionsResult`.
#[derive(Clone, PartialEq, Message)]
pub struct SetSessionOptionsResult {
    /// The options not set, by name.
    #[prost(map = "string, message", tag = "1")]
    pub errors: HashMap<String, SetSessionOptionError>,
}

/// `GetSessionOptionsResult`.
#[derive(Clone, PartialEq, Message)]
pub struct GetSessionOptionsResult {
    /// `session_options`.
    #[prost(map = "string, message", tag = "1")]
    pub session_options: HashMap<String, SessionOptionValue>,
}

/// `CloseSessionResult.Status`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, prost::Enumeration)]
#[repr(i32)]
pub enum CloseStatus {
    /// `UNSPECIFIED`.
    Unspecified = 0,
    /// `CLOSED`.
    Closed = 1,
    /// `CLOSING`.
    Closing = 2,
    /// `NOT_CLOSEABLE`.
    NotCloseable = 3,
}

/// `CloseSessionResult`.
#[derive(Clone, PartialEq, Message)]
pub struct CloseSessionResult {
    /// `status`.
    #[prost(enumeration = "CloseStatus", tag = "1")]
    pub status: i32,
}

/// One result message as the action's only answer.
fn one(body: Vec<u8>) -> ActionStream {
    Box::pin(stream::iter([Ok(arrow_flight::Result {
        body: body.into(),
    })]))
}

/// A string option's value.
fn string(value: String) -> SessionOptionValue {
    SessionOptionValue {
        option_value: Some(OptionValue::StringValue(value)),
    }
}

/// The text a `SET` gets for `value`; `None` for a value no `SET` takes (a string list).
fn set_text(value: &OptionValue) -> Option<String> {
    match value {
        OptionValue::StringValue(s) => Some(s.clone()),
        OptionValue::BoolValue(b) => Some(b.to_string()),
        OptionValue::Int64Value(n) => Some(n.to_string()),
        OptionValue::DoubleValue(d) => Some(d.to_string()),
        OptionValue::StringListValue(_) => None,
    }
}

/// The error a refused `SET` of a key answers: a value the key's check refused is
/// `INVALID_VALUE`, anything else (a `Startup` key, a denial) `ERROR`.
fn set_error(e: &KrishivError) -> ErrorValue {
    if e.error_class().name().starts_with("INVALID_CONF_VALUE") {
        ErrorValue::InvalidValue
    } else {
        ErrorValue::Error
    }
}

/// Sets option `name` to `value` on `session`.
async fn set_option(
    session: &Session,
    name: &str,
    value: Option<&OptionValue>,
) -> Result<(), (ErrorValue, Option<KrishivError>)> {
    if name == CATALOG || name == SCHEMA {
        let Some(OptionValue::StringValue(target)) = value else {
            return Err((ErrorValue::InvalidValue, None));
        };
        let catalog = session.catalog();
        let set = if name == CATALOG {
            catalog.set_current_catalog(target).await
        } else {
            catalog.set_current_database(target).await
        };
        return set.map_err(|e| (ErrorValue::InvalidValue, Some(e)));
    }
    let refused = |e: KrishivError| (set_error(&e), Some(e));
    check_static(name).map_err(refused)?;
    session.authorize_set_key(name).await.map_err(refused)?;
    let conf = session.conf();
    match value {
        None => conf.unset(name).map_err(refused),
        Some(v) => {
            let Some(text) = set_text(v) else {
                return Err((ErrorValue::InvalidValue, None));
            };
            conf.set(name, &text).map_err(refused)
        }
    }
}

impl FlightSqlServer {
    /// `SetSessionOptions` (design §3.2): in the call's session, under its statement gate,
    /// options in name order; the session is marked changed and its cookie set.
    pub(crate) async fn set_session_options(
        &self,
        request: SetSessionOptionsRequest,
        metadata: &MetadataMap,
        caller: &Authenticated,
    ) -> Result<Response<ActionStream>, Status> {
        let mut call = self.call_session(metadata, caller)?;
        let entry = std::sync::Arc::clone(&call.entry);
        let mut options: Vec<_> = request.session_options.into_iter().collect();
        options.sort_by(|a, b| a.0.cmp(&b.0));
        let mut errors = HashMap::new();
        {
            let _gate = entry.shared().await;
            let session = entry.session();
            for (name, value) in &options {
                if let Err((error, cause)) =
                    set_option(session, name, value.option_value.as_ref()).await
                {
                    tracing::debug!(
                        target: "krishiv::flight_sql",
                        option = %name,
                        class = cause.as_ref().map(|e| e.error_class().name()),
                        "session option not set"
                    );
                    errors.insert(
                        name.clone(),
                        SetSessionOptionError {
                            value: error as i32,
                        },
                    );
                }
            }
        }
        call.mark_changed();
        let mut response = Response::new(one(SetSessionOptionsResult { errors }.encode_to_vec()));
        call.cookie_on(&mut response);
        call.kept();
        Ok(response)
    }

    /// `GetSessionOptions` (design §3.2): the session's set keys (redacted as `Config`
    /// redacts) and its current catalog and schema. Without a cookie it runs in a transient
    /// session, released as the call ends.
    pub(crate) async fn get_session_options(
        &self,
        metadata: &MetadataMap,
        caller: &Authenticated,
    ) -> Result<Response<ActionStream>, Status> {
        let call = self.call_session(metadata, caller)?;
        let entry = std::sync::Arc::clone(&call.entry);
        let _gate = entry.shared().await;
        let session = entry.session();
        let mut options: HashMap<String, SessionOptionValue> = session
            .conf()
            .entries()
            .into_iter()
            .filter(|(k, v)| !is_redacted(k, Some(v)))
            .map(|(k, v)| (k, string(v)))
            .collect();
        let catalog = session.catalog();
        options.insert(CATALOG.to_owned(), string(catalog.current_catalog()));
        options.insert(SCHEMA.to_owned(), string(catalog.current_database()));
        let body = GetSessionOptionsResult {
            session_options: options,
        }
        .encode_to_vec();
        Ok(Response::new(one(body)))
    }

    /// `CloseSession` (design §3.2): the cookie's session, if this caller holds it, is released
    /// with its handles; the answer is `CLOSED` either way.
    pub(crate) fn close_session(
        &self,
        metadata: &MetadataMap,
        caller: &Authenticated,
    ) -> Response<ActionStream> {
        if let SessionCookie::Present(id) = session_cookie(metadata)
            && let Some(r) = self.inner.sessions.close(&SessionOwner::of(caller), &id)
        {
            self.release(&r);
        }
        let body = CloseSessionResult {
            status: CloseStatus::Closed as i32,
        }
        .encode_to_vec();
        Response::new(one(body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_as_set_text() {
        assert_eq!(
            set_text(&OptionValue::BoolValue(true)).as_deref(),
            Some("true")
        );
        assert_eq!(
            set_text(&OptionValue::Int64Value(-3)).as_deref(),
            Some("-3")
        );
        assert_eq!(
            set_text(&OptionValue::DoubleValue(1.5)).as_deref(),
            Some("1.5")
        );
        assert_eq!(
            set_text(&OptionValue::StringValue("UTC".to_owned())).as_deref(),
            Some("UTC")
        );
        let list = OptionValue::StringListValue(StringListValue {
            values: vec!["a".to_owned()],
        });
        assert_eq!(set_text(&list), None);
    }

    /// The wire form matches `Flight.proto`: `int64_value` is `sfixed64` (tag 3, wire type 1).
    #[test]
    fn int64_is_sfixed64_on_the_wire() {
        let v = SessionOptionValue {
            option_value: Some(OptionValue::Int64Value(1)),
        };
        assert_eq!(v.encode_to_vec(), [0x19, 1, 0, 0, 0, 0, 0, 0, 0]);
        let closed = CloseSessionResult {
            status: CloseStatus::Closed as i32,
        };
        assert_eq!(closed.encode_to_vec(), [0x08, 1]);
    }
}
