// SPDX-License-Identifier: Apache-2.0
//! `Handles`: the server-wide map of Flight SQL statement and prepared-statement handles (design
//! P3-T101 §4.1). A handle is 128 random bits that mean nothing outside this map; it is bound to
//! its owner (another owner gets the answer an unknown handle gets), to one session, and holds
//! no rows. Limits per session and per owner, a TTL since last use, and removal with the session.

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
/// maxHandlesPerPrincipal}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandleLimits {
    /// How long a handle lives after its last use.
    pub ttl: Duration,
    /// The most handles one session holds.
    pub per_session: usize,
    /// The most handles one owner holds across its sessions.
    pub per_owner: usize,
}

impl Default for HandleLimits {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(600),
            per_session: 64,
            per_owner: 1024,
        }
    }
}

struct Entry<K, P> {
    owner: K,
    session_id: String,
    payload: P,
    last_used: Instant,
}

struct Inner<K, P> {
    entries: HashMap<HandleId, Entry<K, P>>,
    per_session: HashMap<String, usize>,
    per_owner: HashMap<K, usize>,
}

/// The handles of a server, keyed by [`HandleId`], each with its owner `K` (the session key's
/// owner) and its payload `P` (what a fetch needs: SQL text, schema, query start).
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

    /// Stores `payload` under a fresh id for `owner` in session `session_id`, at `now`.
    ///
    /// # Errors
    /// `KRISHIV_FLIGHT_SQL_HANDLE_LIMIT` past the session's or the owner's limit (expired
    /// handles of either are dropped first), `INTERNAL_ERROR` without a random source.
    pub fn insert(
        &self,
        owner: &K,
        session_id: &str,
        payload: P,
        now: Instant,
    ) -> Result<HandleId, KrishivError> {
        let id = HandleId::random()?;
        let mut inner = self.lock();
        let full = |inner: &Inner<K, P>| {
            if inner.per_session.get(session_id).copied().unwrap_or(0) >= self.limits.per_session {
                Some((
                    "krishiv.flightSql.maxHandlesPerSession",
                    self.limits.per_session,
                ))
            } else if inner.per_owner.get(owner).copied().unwrap_or(0) >= self.limits.per_owner {
                Some((
                    "krishiv.flightSql.maxHandlesPerPrincipal",
                    self.limits.per_owner,
                ))
            } else {
                None
            }
        };
        if full(&inner).is_some() {
            self.expire_locked(&mut inner, now);
        }
        if let Some((name, limit)) = full(&inner) {
            return Err(KrishivFlightSqlHandleLimit {
                limit_name: name.to_owned(),
                limit: limit.to_string(),
            }
            .into());
        }
        if inner.entries.contains_key(&id) {
            // 128 random bits do not collide; refuse rather than replace if they ever do.
            return Err(InternalError {
                message: "a Flight SQL handle id collided".to_owned(),
            }
            .into());
        }
        *inner.per_session.entry(session_id.to_owned()).or_insert(0) += 1;
        *inner.per_owner.entry(owner.clone()).or_insert(0) += 1;
        inner.entries.insert(
            id,
            Entry {
                owner: owner.clone(),
                session_id: session_id.to_owned(),
                payload,
                last_used: now,
            },
        );
        Ok(id)
    }

    /// The session id and payload of `id` for `owner`, marking it used at `now`. `None` for an
    /// unknown id, another owner's, or one past its TTL (which is removed).
    pub fn get(&self, owner: &K, id: &HandleId, now: Instant) -> Option<(String, P)> {
        let mut inner = self.lock();
        let expired = match inner.entries.get_mut(id) {
            None => return None,
            Some(e) if e.owner != *owner => return None,
            Some(e) if self.is_expired(e.last_used, now) => true,
            Some(e) => {
                e.last_used = now;
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
    /// returns how many. `alive` runs under this map's lock: it must not call back into it.
    pub fn sweep(&self, now: Instant, alive: impl Fn(&str) -> bool) -> usize {
        let mut inner = self.lock();
        let ids: Vec<HandleId> = inner
            .entries
            .iter()
            .filter(|(_, e)| self.is_expired(e.last_used, now) || !alive(&e.session_id))
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

    fn is_expired(&self, last_used: Instant, now: Instant) -> bool {
        now.saturating_duration_since(last_used) > self.limits.ttl
    }

    fn expire_locked(&self, inner: &mut Inner<K, P>, now: Instant) {
        let ids: Vec<HandleId> = inner
            .entries
            .iter()
            .filter(|(_, e)| self.is_expired(e.last_used, now))
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
        if let Some(n) = inner.per_owner.get_mut(&e.owner) {
            *n = n.saturating_sub(1);
            if *n == 0 {
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
        }
    }

    #[test]
    fn a_handle_is_its_owners_only() {
        let h: Handles<&str, u32> = Handles::new(HandleLimits::default());
        let now = Instant::now();
        let id = h.insert(&"alice", "s1", 7, now).unwrap();
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
    fn a_handle_expires_after_its_ttl_since_last_use() {
        let h: Handles<&str, u32> = Handles::new(limits(4, 4));
        let t0 = Instant::now();
        let id = h.insert(&"a", "s", 1, t0).unwrap();
        // Used at 8 s: lives until 18 s.
        assert!(h.get(&"a", &id, t0 + Duration::from_secs(8)).is_some());
        assert!(h.get(&"a", &id, t0 + Duration::from_secs(17)).is_some());
        assert!(h.get(&"a", &id, t0 + Duration::from_secs(28)).is_none());
        assert!(h.is_empty());
    }

    #[test]
    fn the_session_limit_refuses_with_the_handle_limit_class() {
        let h: Handles<&str, u32> = Handles::new(limits(2, 100));
        let now = Instant::now();
        h.insert(&"a", "s", 1, now).unwrap();
        h.insert(&"a", "s", 2, now).unwrap();
        let e = h.insert(&"a", "s", 3, now).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivFlightSqlHandleLimit);
        assert!(e.to_string().contains("maxHandlesPerSession"), "{e}");
        // Another session of the same owner is not full.
        h.insert(&"a", "t", 4, now).unwrap();
    }

    #[test]
    fn the_owner_limit_counts_across_sessions() {
        let h: Handles<&str, u32> = Handles::new(limits(100, 2));
        let now = Instant::now();
        h.insert(&"a", "s", 1, now).unwrap();
        h.insert(&"a", "t", 2, now).unwrap();
        let e = h.insert(&"a", "u", 3, now).unwrap_err();
        assert!(e.to_string().contains("maxHandlesPerPrincipal"), "{e}");
        h.insert(&"b", "u", 3, now).unwrap();
    }

    #[test]
    fn a_full_map_drops_expired_handles_before_refusing() {
        let h: Handles<&str, u32> = Handles::new(limits(1, 1));
        let t0 = Instant::now();
        h.insert(&"a", "s", 1, t0).unwrap();
        assert!(h.insert(&"a", "s", 2, t0).is_err());
        h.insert(&"a", "s", 2, t0 + Duration::from_secs(11))
            .unwrap();
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn removing_a_session_takes_its_handles_and_frees_the_counts() {
        let h: Handles<&str, u32> = Handles::new(limits(2, 2));
        let now = Instant::now();
        h.insert(&"a", "s", 1, now).unwrap();
        h.insert(&"a", "s", 2, now).unwrap();
        assert_eq!(h.remove_session("s"), 2);
        assert!(h.is_empty());
        h.insert(&"a", "s", 3, now).unwrap();
        h.insert(&"a", "s", 4, now).unwrap();
    }

    #[test]
    fn the_sweep_drops_expired_handles_and_those_of_gone_sessions() {
        let h: Handles<&str, u32> = Handles::new(limits(8, 8));
        let t0 = Instant::now();
        let old = h.insert(&"a", "live", 1, t0).unwrap();
        let fresh = h
            .insert(&"a", "live", 2, t0 + Duration::from_secs(5))
            .unwrap();
        let orphan = h
            .insert(&"a", "gone", 3, t0 + Duration::from_secs(5))
            .unwrap();
        let removed = h.sweep(t0 + Duration::from_secs(12), |s| s == "live");
        assert_eq!(removed, 2);
        let at = t0 + Duration::from_secs(12);
        assert!(h.get(&"a", &old, at).is_none());
        assert!(h.get(&"a", &orphan, at).is_none());
        assert!(h.get(&"a", &fresh, at).is_some());
    }
}
