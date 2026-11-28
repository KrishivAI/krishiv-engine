// SPDX-License-Identifier: Apache-2.0
//! The JavaScript surface: the `#[napi]` methods of [`NativeSession`] and [`NativeQuery`],
//! each a thin shell over the Rust side in `session.rs` / `query.rs` that adds the panic
//! guard and turns an engine error into a JavaScript `Error` (`error::to_json`).
//!
//! Not part of the Rust test binary (`lib.rs` builds it outside `cfg(test)`): a
//! `napi::Error` or `Buffer` drops through Node-API, which only a Node process links. The
//! lib build, clippy and `napi build` compile it; the JavaScript tests (`js/test/`) and the
//! package smoke run it.
// The `#[napi]` attribute moves method attributes, so the exception is the module's.
#![expect(
    clippy::needless_pass_by_value,
    reason = "napi hands JavaScript strings to `set_conf` as owned `String`s"
)]

use std::collections::HashMap;

use krishiv_common::error::KrishivError;
use napi::bindgen_prelude::Buffer;
use napi_derive::napi;

use crate::error::{guarded, guarded_async, to_json};
use crate::{NativeQuery, NativeSession};

/// The napi error for `error`.
fn to_napi(error: &KrishivError) -> napi::Error {
    napi::Error::new(napi::Status::GenericFailure, to_json(error))
}

/// [`guarded`], then the error as the JSON a JavaScript `Error` carries.
fn napi_guarded<T>(f: impl FnOnce() -> Result<T, KrishivError>) -> napi::Result<T> {
    guarded(f).map_err(|e| to_napi(&e))
}

#[napi]
impl NativeSession {
    /// Opens an embedded session with the configuration `conf` (`krishiv.*` and Spark keys).
    #[napi(factory)]
    pub fn open(conf: HashMap<String, String>) -> napi::Result<Self> {
        napi_guarded(|| Self::build(conf))
    }

    /// Starts the SQL statement `sql`; its result is read with `NativeQuery.next()`.
    #[napi]
    pub fn execute(&self, sql: String) -> napi::Result<NativeQuery> {
        napi_guarded(|| Ok(NativeQuery::new(self.session()?, sql)))
    }

    /// Sets the session configuration `key` to `value`, as SQL `SET` does.
    #[napi]
    pub fn set_conf(&self, key: String, value: String) -> napi::Result<()> {
        napi_guarded(|| self.session()?.conf().set(&key, &value))
    }

    /// The session configuration value of `key`, or `undefined` when it is not set and has no
    /// default, as `SET key` reads it.
    #[napi]
    pub fn get_conf(&self, key: String) -> napi::Result<Option<String>> {
        napi_guarded(|| self.conf_value(&key))
    }

    /// Closes the session: its running queries end and its state is freed. Idempotent.
    #[napi]
    pub fn close(&self) -> napi::Result<()> {
        napi_guarded(|| {
            self.shut();
            Ok(())
        })
    }
}

#[napi]
impl NativeQuery {
    /// The next Arrow IPC chunk of the result (schema first, then one per batch, then the
    /// end-of-stream marker), or `null` once the stream has ended.
    #[napi]
    pub async fn next(&self) -> napi::Result<Option<Buffer>> {
        guarded_async(self.step())
            .await
            .map(|chunk| chunk.map(Buffer::from))
            .map_err(|e| to_napi(&e))
    }

    /// Cancels the query, before or while it streams: the next `next()` rejects with
    /// `OPERATION_CANCELED` unless the stream had already ended.
    #[napi]
    pub fn cancel(&self) -> napi::Result<()> {
        napi_guarded(|| {
            self.stop();
            Ok(())
        })
    }
}
