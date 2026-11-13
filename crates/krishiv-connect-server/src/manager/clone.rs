// SPDX-License-Identifier: Apache-2.0
//! `CloneSession` (design P3-T011a §3.11, threat model T-Z6; Spark's
//! `SparkConnectSessionManager.cloneSession` and
//! `validateCloneTargetSession`, `SM.scala` v4.3.0 lines 145-205, read: the
//! order source lookup, observed-id check, format, closed, exists; no session
//! cap). That Spark's `cloneSession` copies artifacts
//! (`SparkSession.scala:295`, `BaseSessionStateBuilder.scala:468`) is from
//! the design's reading, not re-read here: **unverified** in this PR.
//!
//! The source is looked up only (by the handler, its server-side id
//! checked); the clone belongs to the caller's principal, which is the
//! source's (only its owner can look it up). The target id must be a UUID
//! (`INVALID_CLONE_SESSION_REQUEST.TARGET_SESSION_ID_FORMAT`), not a released
//! one (`..._ALREADY_CLOSED`) and not a live one (`..._ALREADY_EXISTS`); an id
//! another principal holds is `INVALID_HANDLE.SESSION_NOT_FOUND`, as on
//! every create. The clone counts against `krishiv.connect.maxSessions` and
//! `maxSessionsPerPrincipal` (Spark checks no cap here).
//!
//! **The copy.** The engine's `Session::clone_session` is not atomic
//! against statements running on the source, so the source's statement gate
//! is held exclusively while the session is copied and its artifacts are
//! listed (both synchronous, no `await` under the guard). The artifacts
//! listed are then copied without the gate, on the blocking pool, charged to
//! the clone's quota and to the server-wide `maxTotalBytes`; a copy that does
//! not fit fails, and the clone, never published, is closed (its directory
//! removed, its quota returned). The clone enters the map only once it is
//! complete.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use krishiv_common::error::{
    InternalError, InvalidCloneSessionRequestTargetSessionIdAlreadyClosed,
    InvalidCloneSessionRequestTargetSessionIdAlreadyExists,
    InvalidCloneSessionRequestTargetSessionIdFormat, InvalidHandleSessionClosed, KrishivError,
};
use krishiv_common::principal::Principal;
use krishiv_session::QueryId;

use super::{Inner, Key, SessionEntry, SessionLimits, SessionManager};
use crate::artifacts::ArtifactStore;

/// A clone not yet in the map: closed if dropped (a failure, or the handler
/// dropped by its client), so nothing of it is left behind.
struct Unpublished(Option<Arc<SessionEntry>>);

impl Unpublished {
    fn publish(mut self) -> Option<Arc<SessionEntry>> {
        self.0.take()
    }
}

impl Drop for Unpublished {
    fn drop(&mut self) {
        if let Some(entry) = self.0.take() {
            entry.close();
        }
    }
}

impl SessionManager {
    /// Clones `source` (a live session of `principal`, looked up by the
    /// caller) as the session `target` of `principal`.
    ///
    /// # Errors
    /// `INVALID_CLONE_SESSION_REQUEST.TARGET_SESSION_ID_FORMAT`,
    /// `..._ALREADY_CLOSED` or `..._ALREADY_EXISTS` for the target id;
    /// `INVALID_HANDLE.SESSION_NOT_FOUND` for an id another principal holds;
    /// `KRISHIV_SESSION_LIMIT` past a session cap;
    /// `INVALID_HANDLE.SESSION_CLOSED` when the source is released meanwhile
    /// or the server shuts down; `KRISHIV_ARTIFACT_LIMIT` when its artifacts
    /// do not fit the clone's or the server's quota.
    pub async fn clone_session(
        &self,
        store: &Arc<ArtifactStore>,
        principal: &Principal,
        source: &Arc<SessionEntry>,
        target: &str,
    ) -> Result<Arc<SessionEntry>, KrishivError> {
        if QueryId::parse(target).is_err() {
            return Err(InvalidCloneSessionRequestTargetSessionIdFormat {
                target_session_id: target.to_owned(),
            }
            .into());
        }
        let key = Self::key(principal, target);
        // Checked before anything is built, and again when it is inserted.
        Self::check_clone_locked(&mut self.lock(), &key, &self.limits)?;
        let source_closed = || -> KrishivError {
            InvalidHandleSessionClosed {
                handle: source.session().id().to_owned(),
            }
            .into()
        };
        let (session, snapshot) = {
            let _gate = source.exclusive().await;
            let session = source.session().clone_session()?;
            let snapshot = source.artifacts().snapshot();
            (session, snapshot)
        };
        let entry = self.new_entry(session, source.client_user_id());
        let pending = Unpublished(Some(Arc::clone(&entry)));
        let snapshot = snapshot.ok_or_else(source_closed)?;
        if !snapshot.is_empty() {
            let artifacts = Arc::clone(entry.artifacts());
            let store = Arc::clone(store);
            let pool = entry.session().runtime().memory();
            tokio::task::spawn_blocking(move || artifacts.copy_from(&snapshot, &store, &pool))
                .await
                .map_err(|e| -> KrishivError {
                    InternalError {
                        message: format!("CloneSession: {e}"),
                    }
                    .into()
                })??;
        }
        {
            let mut inner = self.lock();
            Self::check_clone_locked(&mut inner, &key, &self.limits)?;
            // The idle clock starts now, not before the copy (review L3): a
            // copy longer than the idle timeout must not get the clone
            // swept as soon as it is published.
            entry.last_access.store(self.now_ms(), Ordering::Relaxed);
            inner.live.insert(key.clone(), Arc::clone(&entry));
            inner
                .owners
                .insert(key.session_id.clone(), key.principal.clone());
            *inner
                .per_principal
                .entry(key.principal.clone())
                .or_insert(0) += 1;
        }
        let _published = pending.publish();
        Ok(entry)
    }

    /// Spark's `validateCloneTargetSession` (the format is checked by the
    /// caller), then the checks of every create.
    fn check_clone_locked(
        inner: &mut Inner,
        key: &Key,
        limits: &SessionLimits,
    ) -> Result<(), KrishivError> {
        if !inner.shut {
            if inner.closed.contains(key) {
                return Err(InvalidCloneSessionRequestTargetSessionIdAlreadyClosed {
                    target_session_id: key.session_id.clone(),
                }
                .into());
            }
            if inner.live.contains_key(key) {
                return Err(InvalidCloneSessionRequestTargetSessionIdAlreadyExists {
                    target_session_id: key.session_id.clone(),
                }
                .into());
            }
        }
        // Shut down, another principal's id, the caps.
        Self::check_create_locked(inner, key, limits)
    }
}
