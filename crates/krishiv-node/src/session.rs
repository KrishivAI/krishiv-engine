// SPDX-License-Identifier: Apache-2.0
//! [`NativeSession`]: an embedded engine session (its own engine runtime, as an
//! embedded Python session has).
// The `#[napi]` attribute moves method attributes, so the exception is the module's.
#![expect(
    clippy::needless_pass_by_value,
    reason = "napi hands JavaScript strings to `set_conf` as owned `String`s"
)]

use krishiv_common::error::{InvalidHandleSessionClosed, KrishivError};
use krishiv_session::Session;
use napi_derive::napi;
use std::collections::HashMap;

use crate::error::napi_guarded;
use crate::query::NativeQuery;

/// The session behind a JavaScript `NativeSession`. Kept after `close()`, so a later call
/// is refused with the engine's own `INVALID_HANDLE.SESSION_CLOSED`, as in remote mode.
#[napi]
pub struct NativeSession {
    inner: Session,
}

impl std::fmt::Debug for NativeSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeSession").finish_non_exhaustive()
    }
}

impl NativeSession {
    /// A session with the configuration `conf`.
    pub(crate) fn build(conf: HashMap<String, String>) -> Result<Self, KrishivError> {
        let mut builder = Session::builder().source("krishiv-node");
        let mut conf: Vec<_> = conf.into_iter().collect();
        // One order whatever the object's: a later key never depends on hash order.
        conf.sort();
        for (key, value) in conf {
            builder = builder.config(key, value);
        }
        Ok(Self {
            inner: builder.build()?,
        })
    }

    /// The open session, or `INVALID_HANDLE.SESSION_CLOSED`.
    pub(crate) fn session(&self) -> Result<Session, KrishivError> {
        if self.inner.is_closed() {
            return Err(InvalidHandleSessionClosed {
                handle: self.inner.id().to_owned(),
            }
            .into());
        }
        Ok(self.inner.clone())
    }

    pub(crate) fn conf_value(&self, key: &str) -> Result<Option<String>, KrishivError> {
        Ok(self.session()?.conf().get(key))
    }

    pub(crate) fn shut(&self) {
        self.inner.close();
    }
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
