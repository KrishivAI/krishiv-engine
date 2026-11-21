// SPDX-License-Identifier: Apache-2.0
//! The session of one Flight SQL call (design P3-T101 §3.2): the cookie's session, a new
//! provisional one for a call without a cookie, or a reset for a refused cookie; and the drop
//! guard that ends the call in Flight's session table whatever ends the call (§16 M3).

use std::sync::Arc;
use std::time::Instant;

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_connect_server::{SessionEntry, SessionOwner};
use krishiv_session::Transport;
use krishiv_session::auth::{Authenticated, Surface};
use tonic::metadata::MetadataMap;
use tonic::{Response, Status};

use crate::cookies::{SessionCookie, session_cookie, set_session_cookie};
use crate::server::FlightSqlServer;
use crate::sessions::{Exit, SessionState, mint_session_id};
use crate::status::flight_status;

/// The answer to a call made while the server shuts down (as Connect's).
pub(crate) fn shutting_down() -> Status {
    Status::unavailable("server shutting down")
}

/// The answer to new work while the server drains (design P3-T066 §5.1): `KRISHIV_SERVER_DRAINING`,
/// which a driver does not retry the way it retries `UNAVAILABLE`.
pub(crate) fn draining() -> Status {
    flight_status(&krishiv_common::error::KrishivServerDraining {}.into())
}

/// A call in flight on a Flight session: counted in the session table from creation until
/// dropped (design §3.2: the sweep never releases a session with a call in flight).
pub(crate) struct CallSession {
    server: FlightSqlServer,
    pub(crate) owner: SessionOwner,
    pub(crate) id: String,
    pub(crate) entry: Arc<SessionEntry>,
    /// The session id the response's `set-cookie` carries (a session this call created).
    pub(crate) set_cookie: Option<String>,
    exit: Exit,
    /// Release the session this call created when the call ends, unless the answer carrying its
    /// cookie or ticket went out ([`Self::kept`]): the client never learned its id, whatever its
    /// state (provisional, changed, or a reset cookie session), and whatever ended the call (an
    /// error, a dropped future; §15 m3, §16 M3, review M4).
    release_unless_kept: bool,
}

impl CallSession {
    /// The end of this call is the end of a cookie-less fetch (design §3.2 rule 1).
    pub(crate) fn end_as_cookieless_fetch(&mut self) {
        self.exit = Exit::CookielessFetch;
    }

    /// The answer naming the session (its cookie or a ticket) is going out: no release at the
    /// end of the call.
    pub(crate) fn kept(&mut self) {
        self.release_unless_kept = false;
    }

    /// Marks the session changed (design §3.2: a later call depends on its state).
    pub(crate) fn mark_changed(&self) {
        self.server
            .inner
            .sessions
            .mark_changed(&self.owner, &self.id);
    }

    /// Adds this call's `set-cookie` to `response`.
    pub(crate) fn cookie_on<T>(&self, response: &mut Response<T>) {
        // Minted ids are always valid header text; a refusal would leave a cookie-less client.
        if let Some(id) = &self.set_cookie
            && !set_session_cookie(response.metadata_mut(), id)
        {
            tracing::warn!(target: "krishiv::flight_sql", "a minted session id was not a valid cookie");
        }
    }
}

impl Drop for CallSession {
    fn drop(&mut self) {
        let server = &self.server;
        let now = Instant::now();
        if let Some(r) = server
            .inner
            .sessions
            .exit(&self.id, self.exit, now, |s| server.holds(s))
        {
            server.release(&r);
            return;
        }
        if self.release_unless_kept
            && let Some(r) = server.inner.sessions.close(&self.owner, &self.id)
        {
            server.release(&r);
        }
    }
}

/// The client's `user-agent`, shown as `system.sessions.client_type` (client-asserted).
fn user_agent(metadata: &MetadataMap) -> Option<String> {
    metadata
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.chars().take(256).collect())
}

impl FlightSqlServer {
    /// The session a session-creating call runs in (design §3.2): the cookie's, else a new one
    /// (provisional without a cookie; a cookie session when the cookie was refused, with a
    /// `set-cookie` either way).
    ///
    /// # Errors
    /// `UNAVAILABLE "server shutting down"` when the manager is shut; the manager's refusals
    /// (`KRISHIV_SESSION_LIMIT`) as Flight statuses.
    pub(crate) fn call_session(
        &self,
        metadata: &MetadataMap,
        caller: &Authenticated,
    ) -> Result<CallSession, Status> {
        let owner = SessionOwner::of(caller);
        let state = match session_cookie(metadata) {
            SessionCookie::Absent => SessionState::Provisional { changed: false },
            SessionCookie::Present(id) => match self.enter(&owner, caller, &id, true) {
                Ok(call) => return Ok(call),
                Err(reason) => {
                    tracing::info!(target: "krishiv::flight_sql", reason, "session reset");
                    SessionState::Cookie
                }
            },
            SessionCookie::Malformed => {
                tracing::info!(target: "krishiv::flight_sql", reason = "malformed cookie", "session reset");
                SessionState::Cookie
            }
        };
        self.create(owner, caller, state, metadata)
    }

    /// Starts a call on the existing Flight session `id` of `owner` (`via_cookie`: the call
    /// presented it; a handle's session otherwise). `Err(reason)` when the table does not hold
    /// it for `owner` or the manager refuses it (the row is left as it was).
    pub(crate) fn enter(
        &self,
        owner: &SessionOwner,
        caller: &Authenticated,
        id: &str,
        via_cookie: bool,
    ) -> Result<CallSession, &'static str> {
        let inner = &self.inner;
        if !inner.sessions.enter(owner, id, via_cookie) {
            return Err("not a session of this caller");
        }
        // From here the call is counted: the guard ends it whatever happens.
        match inner.manager.get_with(caller, id, Surface::FlightSql) {
            Ok(entry) => Ok(CallSession {
                server: self.clone(),
                owner: owner.clone(),
                id: id.to_owned(),
                entry,
                set_cookie: None,
                exit: Exit::Call,
                release_unless_kept: false,
            }),
            Err(e) => {
                if let Some(r) = inner
                    .sessions
                    .exit(id, Exit::Call, Instant::now(), |s| self.holds(s))
                {
                    self.release(&r);
                }
                Err(
                    if e.error_class() == ErrorClass::InvalidHandleSessionClosed {
                        "session closed"
                    } else {
                        "session not found"
                    },
                )
            }
        }
    }

    /// Creates a Flight session of `state` for `caller` once (design §3.2: never a retry loop).
    fn create(
        &self,
        owner: SessionOwner,
        caller: &Authenticated,
        state: SessionState,
        metadata: &MetadataMap,
    ) -> Result<CallSession, Status> {
        let inner = &self.inner;
        let id = mint_session_id().map_err(|e| flight_status(&e))?;
        let entry = inner
            .manager
            .get_or_create_with(
                caller,
                &id,
                None,
                &caller.principal.name,
                Transport::FlightSql,
            )
            .map_err(|e| create_status(&e))?;
        entry.session().set_client_info(
            Transport::FlightSql,
            user_agent(metadata).as_deref(),
            Some(entry.client_user_id()),
            Some(entry.server_side_id()),
        );
        inner.sessions.insert(&owner, &id, state, Instant::now());
        Ok(CallSession {
            server: self.clone(),
            owner,
            set_cookie: Some(id.clone()),
            id,
            entry,
            exit: Exit::Call,
            release_unless_kept: true,
        })
    }
}

/// A failed create: `SESSION_CLOSED` means the manager is shut (`close_all`), answered as
/// shutdown, never as `SESSION_CLOSED` (design §3.2, v2 review m6).
fn create_status(e: &KrishivError) -> Status {
    if e.error_class() == ErrorClass::InvalidHandleSessionClosed {
        shutting_down()
    } else {
        flight_status(e)
    }
}
