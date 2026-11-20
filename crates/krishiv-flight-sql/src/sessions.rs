// SPDX-License-Identifier: Apache-2.0
//! `Sessions`: Flight SQL's own session table (design P3-T101 §3.2). Flight SQL commands carry
//! no session id, so Flight mints one per session (a UUID, never a client's choice), sends it
//! as the `arrow_flight_session_id` cookie and records it here with its owner, its state
//! ([`SessionState::Provisional`] until a call presents the cookie, then
//! [`SessionState::Cookie`]), its calls in flight and the end of its last call. An id not in this
//! table is not a Flight session: the caller resets it without asking the session manager.
//!
//! Lock discipline (v4 review m1): one mutex; every rule reads and changes a row under it and
//! nothing else runs under it except the `holds` closures, which read the handle map (lock order:
//! this table, then the handles, never the reverse). Releases are **decided** here and
//! **returned** as [`Release`]s; the caller runs `SessionManager::release_owned` and removes the
//! session's handles after the lock is dropped. Liveness checks (`SessionManager::is_live`) run
//! outside the lock too: [`Sessions::snapshot`] lists the rows, the caller asks the manager,
//! and [`Sessions::sweep`] gets the ids found gone.
//!
//! Owner match (v4 review m2): promotion, the in-flight count, closing and every release act on a
//! row only for the caller whose owner equals the row's; for anyone else the id answers as unknown
//! and the row is unchanged.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use krishiv_common::error::{InternalError, KrishivError};

/// The state of a Flight session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Created by a call without a cookie; its response set the cookie, no call has presented it
    /// yet. `changed` once a call left session state a later call depends on
    /// (`SetSessionOptions`, `CreatePreparedStatement`, a `StatementClass::Session` command).
    Provisional {
        /// Whether a later call depends on this session's state.
        changed: bool,
    },
    /// A call presented the cookie: the session lives until `CloseSession`, the manager's idle
    /// sweep or shutdown, as a Connect session.
    Cookie,
}

/// What a session's handles hold, read under the table's lock by the caller's `holds` closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Holds {
    /// Handles of the session (statement and prepared).
    pub handles: usize,
    /// Prepared-statement handles of the session.
    pub prepared: usize,
}

/// A session this table let go: the caller removes its handles and, when `live`, releases it in
/// the manager (`release_owned(owner, session_id, true)`, design §3.2 tombstones) after the
/// table's lock is dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Release<O> {
    /// The minted session id.
    pub session_id: String,
    /// The session's owner.
    pub owner: O,
    /// `false` when the manager had already released it (found gone by the sweep): only its
    /// handles are left to remove.
    pub live: bool,
}

/// How a call ends ([`Sessions::exit`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// Any call but a cookie-less fetch.
    Call,
    /// The end of a `DoGet` stream that presented no cookie (design §3.2 rule 1: a cookie-less
    /// client gets one fetch per ticket).
    CookielessFetch,
}

struct Row<O> {
    owner: O,
    state: SessionState,
    in_flight: usize,
    /// When the last call ended (or the row was created).
    last_end: Instant,
    /// Rule 1 met while other calls were in flight: release when the last one ends.
    release_after_fetch: bool,
    /// Rule 3 met while other calls were in flight: release when the last one ends.
    release_when_handleless: bool,
}

/// Flight SQL's session table (module docs), keyed by minted session id, each row with its owner
/// `O` (the full `SessionOwner`, act-as groups included).
pub struct Sessions<O> {
    provisional_ttl: Duration,
    rows: Mutex<HashMap<String, Row<O>>>,
}

impl<O> std::fmt::Debug for Sessions<O> {
    /// The TTL and the count only: never an id or an owner.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self
            .rows
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len();
        f.debug_struct("Sessions")
            .field("provisional_ttl", &self.provisional_ttl)
            .field("len", &len)
            .finish()
    }
}

/// A fresh session id: a random (version 4) UUID in its 36-character text form, which
/// [`crate::session_cookie`] accepts.
///
/// # Errors
/// `INTERNAL_ERROR` when the system has no random source.
pub fn mint_session_id() -> Result<String, KrishivError> {
    let mut b = [0_u8; 16];
    getrandom::fill(&mut b).map_err(|e| -> KrishivError {
        InternalError {
            message: format!("no random source for a Flight SQL session id: {e}"),
        }
        .into()
    })?;
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let mut s = String::with_capacity(36);
    for (i, byte) in b.iter().enumerate() {
        if matches!(i, 4 | 6 | 8 | 10) {
            s.push('-');
        }
        // Writing to a String cannot fail.
        let _ = write!(s, "{byte:02x}");
    }
    Ok(s)
}

impl<O: Eq + Clone> Sessions<O> {
    /// An empty table; unchanged provisional sessions are released `provisional_ttl`
    /// (`krishiv.flightSql.provisionalTtl`) after their last call ended.
    #[must_use]
    pub fn new(provisional_ttl: Duration) -> Self {
        Self {
            provisional_ttl,
            rows: Mutex::new(HashMap::new()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<String, Row<O>>> {
        // Each rule changes one row in place; a panic between two of its statements leaves at
        // worst a count that `exit` saturates. Keep serving.
        self.rows.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Records the session `id` the caller just created in the manager for `owner`, in `state`,
    /// with the creating call in flight (end it with [`Self::exit`]). A provisional session for a
    /// call without a cookie; a cookie session for a call whose cookie was refused (design §3.2
    /// B5: it presented a cookie). An id already present is left as it is (ids are minted).
    pub fn insert(&self, owner: &O, id: &str, state: SessionState, now: Instant) {
        self.lock().entry(id.to_owned()).or_insert(Row {
            owner: owner.clone(),
            state,
            in_flight: 1,
            last_end: now,
            release_after_fetch: false,
            release_when_handleless: false,
        });
    }

    /// Starts a call of `owner` on session `id`, before it uses the session (a `DoGet` from before
    /// its analysis); `via_cookie` when the call presented `id` as its cookie, which promotes a
    /// provisional session. `false` for an id not in the table or another owner's (the row is
    /// unchanged): the caller treats the cookie as refused, or a handle's session as gone.
    pub fn enter(&self, owner: &O, id: &str, via_cookie: bool) -> bool {
        let mut rows = self.lock();
        let Some(row) = rows.get_mut(id).filter(|r| r.owner == *owner) else {
            return false;
        };
        row.in_flight += 1;
        if via_cookie {
            Self::promote_row(row);
        }
        true
    }

    /// Promotes session `id` because a call presented its cookie while it ran on another session
    /// (a handle-carrying call, design §3.2, v3 review m10). `false` for an unknown id or another
    /// owner's.
    pub fn promote(&self, owner: &O, id: &str) -> bool {
        let mut rows = self.lock();
        match rows.get_mut(id).filter(|r| r.owner == *owner) {
            Some(row) => {
                Self::promote_row(row);
                true
            }
            None => false,
        }
    }

    fn promote_row(row: &mut Row<O>) {
        row.state = SessionState::Cookie;
        row.release_after_fetch = false;
        row.release_when_handleless = false;
    }

    /// Marks provisional session `id` changed (design §3.2): it is then kept past
    /// `provisionalTtl`, as a cookie session is. No effect on a cookie session, an unknown id or
    /// another owner's.
    pub fn mark_changed(&self, owner: &O, id: &str) {
        let mut rows = self.lock();
        if let Some(row) = rows.get_mut(id).filter(|r| r.owner == *owner)
            && let SessionState::Provisional { changed } = &mut row.state
        {
            *changed = true;
        }
    }

    /// Ends a call [`Self::insert`] or [`Self::enter`] started on `id` at `now`. Returns the
    /// session to release when this was its last call in flight and a rule applies: the end of a
    /// cookie-less fetch of a provisional session holding no prepared handle (rule 1), or a rule
    /// met earlier while other calls were in flight. `holds` reads the session's handles under
    /// this table's lock and must not call back into it. A row removed meanwhile is ignored.
    pub fn exit(
        &self,
        id: &str,
        how: Exit,
        now: Instant,
        holds: impl Fn(&str) -> Holds,
    ) -> Option<Release<O>> {
        let mut rows = self.lock();
        let row = rows.get_mut(id)?;
        row.in_flight = row.in_flight.saturating_sub(1);
        row.last_end = now;
        if !matches!(row.state, SessionState::Provisional { .. }) {
            return None;
        }
        if how == Exit::CookielessFetch {
            row.release_after_fetch = true;
        }
        if row.in_flight > 0 || !(row.release_after_fetch || row.release_when_handleless) {
            return None;
        }
        let held = holds(id);
        let release = (row.release_after_fetch && held.prepared == 0)
            || (row.release_when_handleless && held.handles == 0);
        if !release {
            return None;
        }
        Self::take(&mut rows, id, true)
    }

    /// Releases provisional session `id` once it holds no handle (rule 3: `CancelFlightInfo` or
    /// `ClosePreparedStatement` of its last handle; v4 review m4: an eviction of it). With calls
    /// in flight the release waits for the last one's [`Self::exit`]. `None` for a cookie
    /// session, one that still holds a handle, an unknown id or another owner's.
    pub fn release_if_handleless(
        &self,
        owner: &O,
        id: &str,
        holds: impl Fn(&str) -> Holds,
    ) -> Option<Release<O>> {
        let mut rows = self.lock();
        let row = rows.get_mut(id).filter(|r| r.owner == *owner)?;
        if !matches!(row.state, SessionState::Provisional { .. }) || holds(id).handles > 0 {
            return None;
        }
        if row.in_flight > 0 {
            row.release_when_handleless = true;
            return None;
        }
        Self::take(&mut rows, id, true)
    }

    /// Removes session `id` for `CloseSession` (its handles go with it); `None` for an unknown id
    /// or another owner's, which `CloseSession` answers as closed and leaves alone.
    pub fn close(&self, owner: &O, id: &str) -> Option<Release<O>> {
        let mut rows = self.lock();
        if rows.get(id).is_some_and(|r| r.owner == *owner) {
            Self::take(&mut rows, id, true)
        } else {
            None
        }
    }

    /// Every row's id and owner, for the caller to ask the manager which are gone
    /// (`SessionManager::is_live`, which stamps nothing) outside this table's lock.
    #[must_use]
    pub fn snapshot(&self) -> Vec<(String, O)> {
        self.lock()
            .iter()
            .map(|(id, r)| (id.clone(), r.owner.clone()))
            .collect()
    }

    /// The sweep (design §2, §3.2): under one lock, drops the rows with no call in flight that the
    /// manager released (`gone`, from [`Self::snapshot`]; returned with `live: false`) and
    /// releases the unchanged provisional sessions whose last call ended over `provisionalTtl`
    /// before `now`. A row with a call in flight is never released, whatever `gone` says.
    pub fn sweep(&self, now: Instant, gone: &HashSet<String>) -> Vec<Release<O>> {
        let ttl = self.provisional_ttl;
        let mut rows = self.lock();
        let due: Vec<(String, bool)> = rows
            .iter()
            .filter(|(_, r)| r.in_flight == 0)
            .filter_map(|(id, r)| {
                if gone.contains(id) {
                    Some((id.clone(), false))
                } else if r.state == (SessionState::Provisional { changed: false })
                    && now.saturating_duration_since(r.last_end) > ttl
                {
                    Some((id.clone(), true))
                } else {
                    None
                }
            })
            .collect();
        due.into_iter()
            .filter_map(|(id, live)| Self::take(&mut rows, &id, live))
            .collect()
    }

    fn take(rows: &mut HashMap<String, Row<O>>, id: &str, live: bool) -> Option<Release<O>> {
        rows.remove_entry(id).map(|(session_id, row)| Release {
            session_id,
            owner: row.owner,
            live,
        })
    }

    /// The state of session `id` for `owner`; `None` for an unknown id or another owner's.
    #[must_use]
    pub fn state(&self, owner: &O, id: &str) -> Option<SessionState> {
        self.lock()
            .get(id)
            .filter(|r| r.owner == *owner)
            .map(|r| r.state)
    }

    /// How many sessions the table holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether it holds none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cookies::{SessionCookie, session_cookie};

    const TTL: Duration = Duration::from_secs(60);

    fn none(_: &str) -> Holds {
        Holds::default()
    }

    fn at(t0: Instant, s: u64) -> Instant {
        t0 + Duration::from_secs(s)
    }

    fn gone(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|s| (*s).to_owned()).collect()
    }

    fn released(id: &str, owner: &'static str, live: bool) -> Release<&'static str> {
        Release {
            session_id: id.to_owned(),
            owner,
            live,
        }
    }

    #[test]
    fn minted_ids_are_v4_uuids_the_cookie_parser_accepts() {
        let a = mint_session_id().unwrap();
        let b = mint_session_id().unwrap();
        assert_ne!(a, b);
        assert_eq!(a.len(), 36);
        assert_eq!(a.as_bytes()[14], b'4');
        assert!(matches!(a.as_bytes()[19], b'8' | b'9' | b'a' | b'b'));
        let mut md = tonic::metadata::MetadataMap::new();
        md.insert(
            "cookie",
            format!("arrow_flight_session_id={a}").parse().unwrap(),
        );
        assert_eq!(session_cookie(&md), SessionCookie::Present(a));
    }

    #[test]
    fn a_cookie_presented_promotes_and_only_for_its_owner() {
        let s: Sessions<&str> = Sessions::new(TTL);
        let t0 = Instant::now();
        s.insert(
            &"alice",
            "p",
            SessionState::Provisional { changed: false },
            t0,
        );
        assert_eq!(s.exit("p", Exit::Call, t0, none), None);
        // Another owner presenting the id: unknown, row unchanged (v4 review m2).
        assert!(!s.enter(&"bob", "p", true));
        assert!(!s.promote(&"bob", "p"));
        assert_eq!(s.close(&"bob", "p"), None);
        assert_eq!(
            s.state(&"alice", "p"),
            Some(SessionState::Provisional { changed: false })
        );
        assert_eq!(s.state(&"bob", "p"), None);
        assert!(!s.enter(&"alice", "unknown", true));
        // Its owner presenting it: promoted at the call's start (cookie_session_keeps_use_and_set).
        assert!(s.enter(&"alice", "p", true));
        assert_eq!(s.state(&"alice", "p"), Some(SessionState::Cookie));
        assert_eq!(s.exit("p", Exit::Call, at(t0, 1), none), None);
        // A cookie session outlives the provisional TTL.
        assert!(s.sweep(at(t0, 1000), &gone(&[])).is_empty());
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn an_unchanged_provisional_session_goes_after_the_ttl_a_changed_one_stays() {
        // unchanged_provisional_session_released_after_ttl,
        // idle_jdbc_connection_keeps_catalog_past_provisional_ttl.
        let s: Sessions<&str> = Sessions::new(TTL);
        let t0 = Instant::now();
        s.insert(
            &"a",
            "plain",
            SessionState::Provisional { changed: false },
            t0,
        );
        s.insert(
            &"a",
            "jdbc",
            SessionState::Provisional { changed: false },
            t0,
        );
        s.mark_changed(&"b", "jdbc");
        assert_eq!(
            s.state(&"a", "jdbc"),
            Some(SessionState::Provisional { changed: false }),
            "another owner cannot mark it"
        );
        s.mark_changed(&"a", "jdbc");
        s.exit("plain", Exit::Call, at(t0, 5), none);
        s.exit("jdbc", Exit::Call, at(t0, 5), none);
        assert!(
            s.sweep(at(t0, 65), &gone(&[])).is_empty(),
            "60 s since the end"
        );
        assert_eq!(
            s.sweep(at(t0, 66), &gone(&[])),
            vec![released("plain", "a", true)]
        );
        assert!(s.sweep(at(t0, 10_000), &gone(&[])).is_empty());
        assert_eq!(
            s.state(&"a", "jdbc"),
            Some(SessionState::Provisional { changed: true })
        );
    }

    #[test]
    fn the_sweep_never_releases_a_session_with_a_call_in_flight() {
        // sweep_never_releases_session_with_call_in_flight, admission_wait_outlives_provisional_ttl.
        let s: Sessions<&str> = Sessions::new(TTL);
        let t0 = Instant::now();
        s.insert(&"a", "p", SessionState::Provisional { changed: false }, t0);
        s.exit("p", Exit::Call, t0, none);
        // A DoGet enters (cookie-less: no promotion) and waits for admission past the TTL.
        assert!(s.enter(&"a", "p", false));
        assert!(s.sweep(at(t0, 500), &gone(&["p"])).is_empty());
        assert_eq!(s.len(), 1);
        // The TTL runs from the end of the last call, not the start.
        s.exit("p", Exit::Call, at(t0, 500), none);
        assert!(s.sweep(at(t0, 550), &gone(&[])).is_empty());
        assert_eq!(
            s.sweep(at(t0, 561), &gone(&[])),
            vec![released("p", "a", true)]
        );
    }

    #[test]
    fn rows_the_manager_released_are_dropped_without_a_second_release() {
        let s: Sessions<&str> = Sessions::new(TTL);
        let t0 = Instant::now();
        s.insert(&"a", "c", SessionState::Cookie, t0);
        s.exit("c", Exit::Call, t0, none);
        s.insert(&"b", "d", SessionState::Cookie, t0);
        s.exit("d", Exit::Call, t0, none);
        let mut ids: Vec<String> = s.snapshot().into_iter().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, ["c", "d"]);
        assert_eq!(s.sweep(t0, &gone(&["c"])), vec![released("c", "a", false)]);
        assert_eq!(s.len(), 1);
        // An exit after the row went is ignored.
        assert_eq!(s.exit("c", Exit::Call, t0, none), None);
    }

    #[test]
    fn a_cookieless_fetch_releases_its_provisional_session_at_the_end() {
        // cookieless_session_released_after_doget.
        let s: Sessions<&str> = Sessions::new(TTL);
        let t0 = Instant::now();
        s.insert(&"a", "p", SessionState::Provisional { changed: true }, t0);
        s.exit("p", Exit::Call, t0, none);
        assert!(s.enter(&"a", "p", false));
        assert_eq!(
            s.exit("p", Exit::CookielessFetch, at(t0, 1), none),
            Some(released("p", "a", true))
        );
        assert!(s.is_empty());
        // A cookie session's cookie-less fetch (a second connection with the ticket) keeps it.
        s.insert(&"a", "c", SessionState::Cookie, t0);
        assert_eq!(s.exit("c", Exit::CookielessFetch, t0, none), None);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn a_cookieless_fetch_waits_for_the_other_calls_and_spares_prepared_sessions() {
        let s: Sessions<&str> = Sessions::new(TTL);
        let t0 = Instant::now();
        s.insert(&"a", "p", SessionState::Provisional { changed: false }, t0);
        // Two fetches of its handles in flight (the creating call still counts too).
        assert!(s.enter(&"a", "p", false));
        assert!(s.enter(&"a", "p", false));
        assert_eq!(s.exit("p", Exit::Call, t0, none), None);
        assert_eq!(s.exit("p", Exit::CookielessFetch, t0, none), None);
        assert_eq!(
            s.exit("p", Exit::Call, at(t0, 2), none),
            Some(released("p", "a", true)),
            "released when the last call ends"
        );
        // Holding a prepared handle: kept, so the statement runs again (design §3.2).
        s.insert(&"a", "q", SessionState::Provisional { changed: true }, t0);
        let prepared = |_: &str| Holds {
            handles: 1,
            prepared: 1,
        };
        assert_eq!(s.exit("q", Exit::CookielessFetch, t0, prepared), None);
        assert_eq!(s.len(), 1);
        // A later cookie promotes it and cancels the pending release.
        assert!(s.enter(&"a", "q", true));
        assert_eq!(s.exit("q", Exit::CookielessFetch, t0, none), None);
        assert_eq!(s.state(&"a", "q"), Some(SessionState::Cookie));
    }

    #[test]
    fn a_provisional_session_goes_with_its_last_handle() {
        // Rule 3 and v4 review m4 (cancel, close, eviction of the last handle).
        let s: Sessions<&str> = Sessions::new(TTL);
        let t0 = Instant::now();
        let one = |_: &str| Holds {
            handles: 1,
            prepared: 0,
        };
        s.insert(&"a", "p", SessionState::Provisional { changed: true }, t0);
        s.exit("p", Exit::Call, t0, none);
        assert_eq!(s.release_if_handleless(&"a", "p", one), None);
        assert_eq!(s.release_if_handleless(&"b", "p", none), None);
        assert_eq!(
            s.release_if_handleless(&"a", "p", none),
            Some(released("p", "a", true))
        );
        // With a call in flight the release waits for its end.
        s.insert(&"a", "q", SessionState::Provisional { changed: false }, t0);
        assert_eq!(s.release_if_handleless(&"a", "q", none), None);
        assert_eq!(s.len(), 1);
        assert_eq!(
            s.exit("q", Exit::Call, t0, none),
            Some(released("q", "a", true))
        );
        // A cookie session stays.
        s.insert(&"a", "c", SessionState::Cookie, t0);
        s.exit("c", Exit::Call, t0, none);
        assert_eq!(s.release_if_handleless(&"a", "c", none), None);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn close_session_takes_the_owners_row_only() {
        let s: Sessions<&str> = Sessions::new(TTL);
        let t0 = Instant::now();
        s.insert(&"a", "c", SessionState::Cookie, t0);
        assert_eq!(s.close(&"b", "c"), None);
        assert_eq!(s.close(&"a", "c"), Some(released("c", "a", true)));
        assert_eq!(s.close(&"a", "c"), None);
        assert!(s.is_empty());
    }

    #[test]
    fn debug_shows_no_ids() {
        let s: Sessions<&str> = Sessions::new(TTL);
        s.insert(
            &"secret-owner",
            "secret-id",
            SessionState::Cookie,
            Instant::now(),
        );
        let d = format!("{s:?}");
        assert!(!d.contains("secret"), "{d}");
        assert!(d.contains("len: 1"), "{d}");
    }
}
