// SPDX-License-Identifier: Apache-2.0
//! [`NativeSession`]: an embedded engine session (its own engine runtime, as an
//! embedded Python session has).

use krishiv_common::error::{InvalidHandleSessionClosed, KrishivError};
use krishiv_session::Session;
use napi_derive::napi;
use std::collections::HashMap;

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
