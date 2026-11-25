// SPDX-License-Identifier: Apache-2.0
//! [`NativeSession`]: an embedded engine session (its own engine runtime, as an
//! embedded Python session has).
// The `#[napi]` attribute moves method attributes, so the exception is the module's.
#![expect(
    clippy::needless_pass_by_value,
    reason = "napi hands JavaScript strings to `set_conf` as owned `String`s"
)]

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use krishiv_common::error::{InternalError, KrishivError};
use krishiv_session::Session;
use napi_derive::napi;

use crate::error::to_napi;
use crate::query::NativeQuery;

/// The session behind a JavaScript `NativeSession`; `None` once closed.
#[napi]
pub struct NativeSession {
    inner: Mutex<Option<Session>>,
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
            inner: Mutex::new(Some(builder.build()?)),
        })
    }

    pub(crate) fn session(&self) -> Result<Session, KrishivError> {
        self.inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
            .ok_or_else(|| {
                InternalError {
                    message: "krishiv-node: the session is closed".to_owned(),
                }
                .into()
            })
    }

    pub(crate) fn shut(&self) {
        let taken = self
            .inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(session) = taken {
            session.close();
        }
    }
}

#[napi]
impl NativeSession {
    /// Opens an embedded session with the configuration `conf` (`krishiv.*` and Spark keys).
    #[napi(factory)]
    pub fn open(conf: HashMap<String, String>) -> napi::Result<Self> {
        Self::build(conf).map_err(|e| to_napi(&e))
    }

    /// Starts the SQL statement `sql`; its result is read with `NativeQuery.next()`.
    #[napi]
    pub fn execute(&self, sql: String) -> napi::Result<NativeQuery> {
        let session = self.session().map_err(|e| to_napi(&e))?;
        Ok(NativeQuery::new(session, sql))
    }

    /// Sets the session configuration `key` to `value`, as SQL `SET` does.
    #[napi]
    pub fn set_conf(&self, key: String, value: String) -> napi::Result<()> {
        let session = self.session().map_err(|e| to_napi(&e))?;
        session.conf().set(&key, &value).map_err(|e| to_napi(&e))
    }

    /// Closes the session: its running queries end and its state is freed. Idempotent.
    #[napi]
    pub fn close(&self) {
        self.shut();
    }
}
