// SPDX-License-Identifier: Apache-2.0
//! `Debug` for [`Session`] and the `Drop` of `SessionState`, which retires the session's row.

use std::fmt;

use super::{Session, SessionState};

impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("profile", &self.state.profile)
            .field("principal", &self.state.principal.name)
            .finish_non_exhaustive()
    }
}

impl Drop for SessionState {
    fn drop(&mut self) {
        self.runtime.sessions.retire(self.number, || {
            crate::sessions::SessionRow::of(
                self,
                Some(crate::sessions::End::Closed("client_gone")),
                0,
            )
        });
        self.runtime.forget_session(self.number);
    }
}
