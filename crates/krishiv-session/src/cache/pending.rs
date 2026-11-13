// SPDX-License-Identifier: Apache-2.0
//! The pending-registration list (design P3-T083 3.1): `persist` stays
//! synchronous by putting the frame's plan here; every entry point that
//! reads the registry registers the list first, in call order.

use std::sync::{Arc, PoisonError};

use krishiv_common::error::KrishivError;
use krishiv_plan::LogicalPlan;

use super::{RegKey, RelationCache};
use crate::{DataFrame, Session, StorageLevel};

/// A `persist` call not registered yet: the frame's unanalyzed plan and
/// what it needs to analyze again. It holds no session, so a list never
/// registered (no reader came) does not keep the session alive.
pub(crate) struct PendingRegistration {
    pub(crate) plan: Arc<LogicalPlan>,
    pub(crate) keep: Vec<Arc<dyn std::any::Any + Send + Sync>>,
    pub(crate) level: StorageLevel,
}

impl std::fmt::Debug for PendingRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingRegistration")
            .field("plan", &self.plan.to_string())
            .field("level", &self.level)
            .finish_non_exhaustive()
    }
}

impl RelationCache {
    /// Puts a `persist` call on the pending list.
    pub(crate) fn push_pending(&self, pending: PendingRegistration) {
        self.pending
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(pending);
    }

    fn take_pending(&self) -> Vec<PendingRegistration> {
        std::mem::take(&mut *self.pending.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

impl Session {
    /// Registers every pending `persist` call, in call order. A plan that
    /// fails analysis is dropped with a `WARN`; the frame's own first action
    /// raises the error, as before.
    pub(crate) async fn register_pending(&self) {
        let cache = Arc::clone(&self.state.cache);
        let _registering = cache.registering.lock().await;
        loop {
            let batch = cache.take_pending();
            if batch.is_empty() {
                return;
            }
            for pending in batch {
                let mut df = DataFrame::new(self.clone(), pending.plan);
                df.keep = pending.keep;
                if let Err(error) = Box::pin(self.register_frame(&df, pending.level)).await {
                    tracing::warn!(
                        target: "krishiv::session",
                        "Not caching a plan that does not analyze: {error}"
                    );
                }
            }
        }
    }

    /// Registers `df`'s analyzed plan at `level` (Spark's
    /// `CacheManager.cacheQuery`): a plan registered already keeps its first
    /// level and logs Spark's warning. Returns the key.
    ///
    /// # Errors
    /// The plan's analysis errors.
    pub(crate) async fn register_frame(
        &self,
        df: &DataFrame,
        level: StorageLevel,
    ) -> Result<RegKey, KrishivError> {
        let a = df.analysis().await?;
        let key = RegKey::of(&a.analyzed);
        let mut tables = super::read_tables(a.analyzed.plan.root());
        tables.extend(super::unresolved_tables(&df.plan));
        tables.sort();
        tables.dedup();
        if !self
            .state
            .cache
            .register(key.clone(), level, tables, a.analyzed.reads_files)
        {
            tracing::warn!(target: "krishiv::session", "Asked to cache already cached data.");
        }
        Ok(key)
    }
}
