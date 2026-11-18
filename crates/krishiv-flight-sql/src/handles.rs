// SPDX-License-Identifier: Apache-2.0
//! `Handles`: the server-wide map of Flight SQL statement and prepared-statement handles (design
//! P3-T101 §4.1). A handle is 128 random bits that mean nothing outside this map; it is bound to
//! its owner (another owner gets the answer an unknown handle gets), to one session, and holds
//! no rows. Limits per session and per owner (count and bytes), a TTL that runs from the end of
//! the handle's last operation and never while one is live (B3 of the v2 review), and removal with
//! the session. At a cap the oldest idle handle of the scope is evicted (design v4 §4.1: Flight
//! SQL has no RPC that closes an ad-hoc ticket); only a scope whose every handle has a fetch in
//! flight refuses.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use krishiv_common::error::{InternalError, KrishivError, KrishivFlightSqlHandleLimit};

/// The bytes of a handle as it travels in a ticket or a prepared-statement handle.
pub const HANDLE_BYTES: usize = 16;

/// One handle id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HandleId([u8; HANDLE_BYTES]);

impl HandleId {
    /// A fresh random id.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when the system has no random source.
    pub fn random() -> Result<Self, KrishivError> {
        let mut b = [0_u8; HANDLE_BYTES];
        getrandom::fill(&mut b).map_err(|e| -> KrishivError {
            InternalError {
                message: format!("no random source for a Flight SQL handle: {e}"),
            }
            .into()
        })?;
        Ok(Self(b))
    }

    /// The id in `bytes` (a ticket's or a handle field's contents); `None` when they cannot be
    /// one, which the caller answers as for an unknown handle.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        <[u8; HANDLE_BYTES]>::try_from(bytes).ok().map(Self)
    }

    /// The bytes to send.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; HANDLE_BYTES] {
        &self.0
    }
}

/// The limits of a [`Handles`] map (`krishiv.flightSql.{handleTtl,maxHandlesPerSession,
/// maxHandlesPerPrincipal,maxHandleBytesPerPrincipal}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandleLimits {
    /// How long a handle lives after its last operation ended (or after its creation when never
    /// fetched); a handle with a live operation does not expire.
    pub ttl: Duration,
    /// The most handles one session holds.
    pub per_session: usize,
    /// The most handles one owner holds across its sessions.
    pub per_owner: usize,
    /// The most bytes (SQL text, bound literals, encoded schema) one owner's handles hold.
    pub bytes_per_owner: usize,
}

impl Default for HandleLimits {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(600),
            per_session: 64,
            per_owner: 1024,
            bytes_per_owner: 64 << 20,
        }
    }
}

struct Entry<K, P> {
    owner: K,
    session_id: String,
    payload: P,
    bytes: usize,
    /// A prepared-statement handle: evicted only after every statement handle of the scope.
    prepared: bool,
    /// Operations started with [`Handles::begin`] and not yet ended.
    live: usize,
    /// When the handle last became idle (created, or its last operation ended).
    idle_since: Instant,
}

impl<K, P> Entry<K, P> {
    fn is_expired(&self, ttl: Duration, now: Instant) -> bool {
        self.live == 0 && now.saturating_duration_since(self.idle_since) > ttl
    }
}

/// What one owner holds: handles and their bytes.
#[derive(Default, Clone, Copy)]
struct Held {
    count: usize,
    bytes: usize,
}

struct Inner<K, P> {
    entries: HashMap<HandleId, Entry<K, P>>,
    per_session: HashMap<String, usize>,
    per_owner: HashMap<K, Held>,
}

/// The handles of a server, keyed by [`HandleId`], each with its owner `K` (the session key's
/// owner, counted under `SessionOwner::cap_key`) and its payload `P` (what a fetch needs: SQL text,
/// schema, query start).
pub struct Handles<K, P> {
    limits: HandleLimits,
    inner: Mutex<Inner<K, P>>,
}

impl<K, P> std::fmt::Debug for Handles<K, P> {
    /// The limits and the count only: never an id or a payload.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self
            .inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entries
            .len();
        f.debug_struct("Handles")
            .field("limits", &self.limits)
            .field("len", &len)
            .finish()
    }
}

impl<K: Eq + Hash + Clone, P: Clone> Handles<K, P> {
    /// An empty map with `limits`.
    #[must_use]
    pub fn new(limits: HandleLimits) -> Self {
        Self {
            limits,
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                per_session: HashMap::new(),
                per_owner: HashMap::new(),
            }),
        }
    }

    /// The limits.
    #[must_use]
    pub fn limits(&self) -> HandleLimits {
        self.limits
    }

    fn lock(&self) -> MutexGuard<'_, Inner<K, P>> {
        // The map holds no invariant a panic between two statements could break beyond a count
        // that `remove` saturates; keep serving.
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Stores `payload` (holding `bytes` of SQL text, literals, schema and stored rows; a
    /// prepared statement's when `prepared`) under a fresh id for `owner` in session
    /// `session_id`, idle from `now`. At a cap, expired handles go first, then the scope's oldest
    /// idle ones (statement handles before prepared ones).
    ///
    /// # Errors
    /// `KRISHIV_FLIGHT_SQL_HANDLE_LIMIT` when the session's or the owner's count or the owner's
    /// bytes stay over the cap with every handle of the scope fetching (or the handle alone is
    /// over the byte cap), `INTERNAL_ERROR` without a random source.
    pub fn insert(
        &self,
        owner: &K,
        session_id: &str,
        payload: P,
        bytes: usize,
        prepared: bool,
        now: Instant,
    ) -> Result<HandleId, KrishivError> {
        let id = HandleId::random()?;
        let mut inner = self.lock();
        let full = |inner: &Inner<K, P>| {
            let held = inner.per_owner.get(owner).copied().unwrap_or_default();
            if inner.per_session.get(session_id).copied().unwrap_or(0) >= self.limits.per_session {
                Some((
                    "krishiv.flightSql.maxHandlesPerSession",
                    self.limits.per_session,
                    true,
                ))
            } else if held.count >= self.limits.per_owner {
                Some((
                    "krishiv.flightSql.maxHandlesPerPrincipal",
                    self.limits.per_owner,
                    false,
                ))
            } else if held.bytes.saturating_add(bytes) > self.limits.bytes_per_owner {
                Some((
                    "krishiv.flightSql.maxHandleBytesPerPrincipal",
                    self.limits.bytes_per_owner,
                    false,
                ))
            } else {
                None
            }
        };
        let refuse = |name: &str, limit: usize| -> KrishivError {
            KrishivFlightSqlHandleLimit {
                limit_name: name.to_owned(),
                limit: limit.to_string(),
            }
            .into()
        };
        if bytes > self.limits.bytes_per_owner {
            // No eviction makes room for it: refuse before dropping anything.
            return Err(refuse(
                "krishiv.flightSql.maxHandleBytesPerPrincipal",
                self.limits.bytes_per_owner,
            ));
        }
        if full(&inner).is_some() {
            self.expire_locked(&mut inner, now);
        }
        while let Some((name, limit, in_session)) = full(&inner) {
            let victim = inner
                .entries
                .iter()
                .filter(|(_, e)| {
                    e.live == 0
                        && if in_session {
                            e.session_id == session_id
                        } else {
                            e.owner == *owner
                        }
                })
                .min_by_key(|(_, e)| (e.prepared, e.idle_since))
                .map(|(id, _)| *id);
            match victim {
                Some(v) => {
                    Self::remove_locked(&mut inner, &v);
                }
                None => return Err(refuse(name, limit)),
            }
        }
        if inner.entries.contains_key(&id) {
            // 128 random bits do not collide; refuse rather than replace if they ever do.
            return Err(InternalError {
                message: "a Flight SQL handle id collided".to_owned(),
            }
            .into());
        }
        *inner.per_session.entry(session_id.to_owned()).or_insert(0) += 1;
        let held = inner.per_owner.entry(owner.clone()).or_default();
        held.count += 1;
        held.bytes = held.bytes.saturating_add(bytes);
        inner.entries.insert(
            id,
            Entry {
                owner: owner.clone(),
                session_id: session_id.to_owned(),
                payload,
                bytes,
                prepared,
                live: 0,
                idle_since: now,
            },
        );
        Ok(id)
    }

    /// The session id and payload of `id` for `owner`, without starting an operation (cancel,
    /// close). `None` for an unknown id, another owner's, or one past its TTL (which is removed).
    pub fn get(&self, owner: &K, id: &HandleId, now: Instant) -> Option<(String, P)> {
        self.lookup(owner, id, now, false)
    }

    /// As [`Self::get`], and starts an operation on the handle (a `DoGet`): until the matching
    /// [`Self::end`] the handle does not expire.
    pub fn begin(&self, owner: &K, id: &HandleId, now: Instant) -> Option<(String, P)> {
        self.lookup(owner, id, now, true)
    }

    /// Ends an operation [`Self::begin`] started on `id`; the TTL runs from `now` once none is
    /// live. A handle removed meanwhile (cancel, session gone) is ignored.
    pub fn end(&self, id: &HandleId, now: Instant) {
        let mut inner = self.lock();
        if let Some(e) = inner.entries.get_mut(id) {
            e.live = e.live.saturating_sub(1);
            if e.live == 0 {
                e.idle_since = now;
            }
        }
    }

    fn lookup(&self, owner: &K, id: &HandleId, now: Instant, start: bool) -> Option<(String, P)> {
        let ttl = self.limits.ttl;
        let mut inner = self.lock();
        let expired = match inner.entries.get_mut(id) {
            None => return None,
            Some(e) if e.owner != *owner => return None,
            Some(e) if e.is_expired(ttl, now) => true,
            Some(e) => {
                if start {
                    e.live += 1;
                }
                return Some((e.session_id.clone(), e.payload.clone()));
            }
        };
        if expired {
            Self::remove_locked(&mut inner, id);
        }
        None
    }

    /// Removes `id` if `owner` holds it; returns its session id and payload.
    pub fn remove(&self, owner: &K, id: &HandleId) -> Option<(String, P)> {
        let mut inner = self.lock();
        match inner.entries.get(id) {
            Some(e) if e.owner == *owner => Self::remove_locked(&mut inner, id),
            _ => None,
        }
    }

    /// Removes every handle of session `session_id` (its release, sweep or close); returns how
    /// many.
    pub fn remove_session(&self, session_id: &str) -> usize {
        let mut inner = self.lock();
        let ids: Vec<HandleId> = inner
            .entries
            .iter()
            .filter(|(_, e)| e.session_id == session_id)
            .map(|(id, _)| *id)
            .collect();
        for id in &ids {
            Self::remove_locked(&mut inner, id);
        }
        ids.len()
    }

    /// Removes the handles past their TTL at `now` and those whose session `alive` says is gone;
    /// returns how many. `alive` runs under this map's lock: it must not call back into it, and
    /// must not stamp the session's access (`SessionManager::is_live`, not `get_if_present`).
    pub fn sweep(&self, now: Instant, alive: impl Fn(&str) -> bool) -> usize {
        let ttl = self.limits.ttl;
        let mut inner = self.lock();
        let ids: Vec<HandleId> = inner
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired(ttl, now) || !alive(&e.session_id))
            .map(|(id, _)| *id)
            .collect();
        for id in &ids {
            Self::remove_locked(&mut inner, id);
        }
        ids.len()
    }

    /// How many handles are stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    /// Whether none is stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn expire_locked(&self, inner: &mut Inner<K, P>, now: Instant) {
        let ttl = self.limits.ttl;
        let ids: Vec<HandleId> = inner
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired(ttl, now))
            .map(|(id, _)| *id)
            .collect();
        for id in &ids {
            Self::remove_locked(inner, id);
        }
    }

    fn remove_locked(inner: &mut Inner<K, P>, id: &HandleId) -> Option<(String, P)> {
        let e = inner.entries.remove(id)?;
        if let Some(n) = inner.per_session.get_mut(&e.session_id) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                inner.per_session.remove(&e.session_id);
            }
        }
        if let Some(held) = inner.per_owner.get_mut(&e.owner) {
            held.count = held.count.saturating_sub(1);
            held.bytes = held.bytes.saturating_sub(e.bytes);
            if held.count == 0 {
                inner.per_owner.remove(&e.owner);
            }
        }
        Some((e.session_id, e.payload))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_common::error::ErrorClass;

    fn limits(per_session: usize, per_owner: usize) -> HandleLimits {
        HandleLimits {
            ttl: Duration::from_secs(10),
            per_session,
            per_owner,
            bytes_per_owner: 100,
        }
    }

    #[test]
    fn a_handle_is_its_owners_only() {
        let h: Handles<&str, u32> = Handles::new(HandleLimits::default());
        let now = Instant::now();
        let id = h.insert(&"alice", "s1", 7, 0, false, now).unwrap();
        assert_eq!(h.get(&"alice", &id, now), Some(("s1".to_owned(), 7)));
        assert_eq!(h.get(&"bob", &id, now), None);
        assert_eq!(h.remove(&"bob", &id), None);
        assert_eq!(h.len(), 1);
        assert_eq!(h.remove(&"alice", &id), Some(("s1".to_owned(), 7)));
        assert!(h.is_empty());
    }

    #[test]
    fn ids_round_trip_through_bytes_and_other_lengths_are_none() {
        let id = HandleId::random().unwrap();
        assert_eq!(HandleId::from_bytes(id.as_bytes()), Some(id));
        assert_eq!(HandleId::from_bytes(&[0; 15]), None);
        assert_eq!(HandleId::from_bytes(&[0; 17]), None);
        assert_ne!(HandleId::random().unwrap(), id);
    }

    #[test]
    fn a_handle_expires_after_its_ttl_since_its_last_operation_ended() {
        let h: Handles<&str, u32> = Handles::new(limits(4, 4));
        let t0 = Instant::now();
        let id = h.insert(&"a", "s", 1, 0, false, t0).unwrap();
        let at = |s| t0 + Duration::from_secs(s);
        // A lookup without an operation does not extend the TTL.
        assert!(h.get(&"a", &id, at(8)).is_some());
        // An operation from 9 s to 15 s: lives until 25 s.
        assert!(h.begin(&"a", &id, at(9)).is_some());
        h.end(&id, at(15));
        assert!(h.get(&"a", &id, at(24)).is_some());
        assert!(h.get(&"a", &id, at(26)).is_none());
        assert!(h.is_empty());
    }

    #[test]
    fn a_handle_with_a_live_operation_does_not_expire() {
        let h: Handles<&str, u32> = Handles::new(limits(4, 4));
        let t0 = Instant::now();
        let id = h.insert(&"a", "s", 1, 0, false, t0).unwrap();
        let at = |s| t0 + Duration::from_secs(s);
        // Two concurrent fetches; a long one outlives the TTL many times over.
        assert!(h.begin(&"a", &id, at(1)).is_some());
        assert!(h.begin(&"a", &id, at(2)).is_some());
        assert_eq!(h.sweep(at(100), |_| true), 0);
        h.end(&id, at(101));
        assert_eq!(h.sweep(at(200), |_| true), 0, "one fetch is still live");
        assert!(h.get(&"a", &id, at(200)).is_some());
        h.end(&id, at(300));
        assert_eq!(h.sweep(at(305), |_| true), 0);
        assert_eq!(h.sweep(at(311), |_| true), 1);
        // An end after the removal is ignored.
        h.end(&id, at(312));
        assert!(h.is_empty());
    }

    #[test]
    fn the_owner_byte_limit_counts_handle_bytes_and_frees_them() {
        let h: Handles<&str, u32> = Handles::new(limits(100, 100));
        let now = Instant::now();
        let a = h.insert(&"a", "s", 1, 60, false, now).unwrap();
        // Fetching, so not evictable: 60 + 41 is over 100.
        h.begin(&"a", &a, now).unwrap();
        let e = h.insert(&"a", "t", 2, 41, false, now).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivFlightSqlHandleLimit);
        assert!(e.to_string().contains("maxHandleBytesPerPrincipal"), "{e}");
        h.insert(&"a", "t", 2, 40, false, now).unwrap();
        h.insert(&"b", "u", 3, 100, false, now).unwrap();
        // One handle alone over the cap is refused and evicts nothing.
        assert!(h.insert(&"b", "u", 4, 101, false, now).is_err());
        assert_eq!(h.len(), 3);
        h.remove(&"a", &a).unwrap();
        h.insert(&"a", "t", 4, 60, false, now).unwrap();
    }

    #[test]
    fn a_full_scope_refuses_only_when_every_handle_is_fetching() {
        let h: Handles<&str, u32> = Handles::new(limits(2, 100));
        let now = Instant::now();
        let x = h.insert(&"a", "s", 1, 0, false, now).unwrap();
        let y = h.insert(&"a", "s", 2, 0, false, now).unwrap();
        h.begin(&"a", &x, now).unwrap();
        h.begin(&"a", &y, now).unwrap();
        let e = h.insert(&"a", "s", 3, 0, false, now).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivFlightSqlHandleLimit);
        assert!(e.to_string().contains("maxHandlesPerSession"), "{e}");
        // Another session of the same owner is not full.
        h.insert(&"a", "t", 4, 0, false, now).unwrap();
        // Once one fetch ends, its handle makes room.
        h.end(&y, now);
        h.insert(&"a", "s", 5, 0, false, now).unwrap();
        assert!(
            h.get(&"a", &y, now).is_none(),
            "the idle handle was evicted"
        );
        assert!(h.get(&"a", &x, now).is_some(), "the fetching one stays");
    }

    #[test]
    fn a_session_runs_more_statements_than_its_cap() {
        let h: Handles<&str, u32> = Handles::new(limits(4, 100));
        let t0 = Instant::now();
        let mut ids = Vec::new();
        for i in 0..100_u32 {
            let at = t0 + Duration::from_millis(u64::from(i));
            let id = h.insert(&"a", "s", i, 0, false, at).unwrap();
            h.begin(&"a", &id, at).unwrap();
            h.end(&id, at);
            ids.push(id);
        }
        assert_eq!(h.len(), 4);
        let now = t0 + Duration::from_millis(100);
        // The four newest stay.
        for id in &ids[96..] {
            assert!(h.get(&"a", id, now).is_some());
        }
        assert!(h.get(&"a", &ids[95], now).is_none());
    }

    #[test]
    fn eviction_takes_the_oldest_idle_statement_handle_before_prepared_ones() {
        let h: Handles<&str, u32> = Handles::new(limits(3, 100));
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        let prepared = h.insert(&"a", "s", 1, 0, true, at(0)).unwrap();
        let older = h.insert(&"a", "s", 2, 0, false, at(1)).unwrap();
        let newer = h.insert(&"a", "s", 3, 0, false, at(2)).unwrap();
        // `older` was fetched last, so `newer` has been idle longest.
        h.begin(&"a", &older, at(3)).unwrap();
        h.end(&older, at(4));
        h.insert(&"a", "s", 4, 0, false, at(5)).unwrap();
        assert!(h.get(&"a", &newer, at(5)).is_none());
        assert!(h.get(&"a", &older, at(5)).is_some());
        assert!(h.get(&"a", &prepared, at(5)).is_some());
    }

    #[test]
    fn the_owner_limit_counts_across_sessions_and_evicts_in_any() {
        let h: Handles<&str, u32> = Handles::new(limits(100, 2));
        let t0 = Instant::now();
        let s = h.insert(&"a", "s", 1, 0, false, t0).unwrap();
        let t = h
            .insert(&"a", "t", 2, 0, false, t0 + Duration::from_millis(1))
            .unwrap();
        h.begin(&"a", &t, t0).unwrap();
        h.insert(&"a", "u", 3, 0, false, t0 + Duration::from_millis(2))
            .unwrap();
        assert!(
            h.get(&"a", &s, t0).is_none(),
            "the idle one of session s went"
        );
        // `u` is idle: it makes room for `v`; with `t` and `v` fetching the owner is full.
        let v = h.insert(&"a", "v", 4, 0, false, t0).unwrap();
        h.begin(&"a", &v, t0).unwrap();
        let e = h.insert(&"a", "w", 5, 0, false, t0).unwrap_err();
        assert!(e.to_string().contains("maxHandlesPerPrincipal"), "{e}");
        // Another owner has its own count.
        h.insert(&"b", "u", 3, 0, false, t0).unwrap();
    }

    #[test]
    fn a_full_map_drops_expired_handles_before_evicting() {
        let h: Handles<&str, u32> = Handles::new(limits(1, 1));
        let t0 = Instant::now();
        let x = h.insert(&"a", "s", 1, 0, false, t0).unwrap();
        h.begin(&"a", &x, t0).unwrap();
        assert!(h.insert(&"a", "s", 2, 0, false, t0).is_err());
        h.end(&x, t0);
        h.insert(&"a", "s", 2, 0, false, t0 + Duration::from_secs(11))
            .unwrap();
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn removing_a_session_takes_its_handles_and_frees_the_counts() {
        let h: Handles<&str, u32> = Handles::new(limits(2, 2));
        let now = Instant::now();
        h.insert(&"a", "s", 1, 0, false, now).unwrap();
        h.insert(&"a", "s", 2, 0, false, now).unwrap();
        assert_eq!(h.remove_session("s"), 2);
        assert!(h.is_empty());
        h.insert(&"a", "s", 3, 0, false, now).unwrap();
        h.insert(&"a", "s", 4, 0, false, now).unwrap();
    }

    #[test]
    fn the_sweep_drops_expired_handles_and_those_of_gone_sessions() {
        let h: Handles<&str, u32> = Handles::new(limits(8, 8));
        let t0 = Instant::now();
        let old = h.insert(&"a", "live", 1, 0, false, t0).unwrap();
        let fresh = h
            .insert(&"a", "live", 2, 0, false, t0 + Duration::from_secs(5))
            .unwrap();
        let orphan = h
            .insert(&"a", "gone", 3, 0, false, t0 + Duration::from_secs(5))
            .unwrap();
        let removed = h.sweep(t0 + Duration::from_secs(12), |s| s == "live");
        assert_eq!(removed, 2);
        let at = t0 + Duration::from_secs(12);
        assert!(h.get(&"a", &old, at).is_none());
        assert!(h.get(&"a", &orphan, at).is_none());
        assert!(h.get(&"a", &fresh, at).is_some());
    }
}
