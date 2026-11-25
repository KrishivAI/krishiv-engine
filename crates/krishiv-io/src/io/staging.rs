// SPDX-License-Identifier: Apache-2.0
//! The upload-staging files of the local stores (PY-054).
//!
//! A local store stages an upload as `<name>#<digits>` until it completes, and hides the file from
//! its listings. A write killed mid-file leaves the partial file behind, where no listing finds
//! it and a reader of the directory (pyarrow) still takes it. The store that staged a file is the
//! one that deletes it: [`StagingSweep`] is implemented by the local stores, registered next to
//! them ([`IoScheduler::register_staging_sweeper`]), and called through [`IoContext::sweep_staging`].
//! A store without a sweeper has nothing to sweep.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use object_store::path::Path;

use super::{IoContext, IoScheduler, StoreId};

/// What decides that a staging file goes: its name (`<base>#<digits>`, the whole name) and its
/// modification time.
pub type StagingDoomed<'a> = dyn Fn(&str, DateTime<Utc>) -> bool + Send + Sync + 'a;

/// A local store that deletes the staging files it hides from listings. Blocking.
pub trait StagingSweep: Send + Sync {
    /// Deletes, below the directory `dir` (links are never followed), each staging file for which
    /// `doomed` says so. `Ok(())` only when every directory was read and every doomed file removed
    /// (a file or directory that vanished meanwhile is no failure).
    ///
    /// # Errors
    /// The first failure to read a directory or to remove a file; the rest of the walk still ran.
    fn sweep_staging(&self, dir: &Path, doomed: &StagingDoomed<'_>) -> std::io::Result<()>;
}

impl IoScheduler {
    /// Registers the sweeper of the store `id` (removed with the store, by
    /// [`Self::unregister_store`]).
    pub fn register_staging_sweeper(&self, id: StoreId, sweeper: Arc<dyn StagingSweep>) {
        self.sweepers
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id, sweeper);
    }
}

impl IoContext {
    /// Deletes the staging files below `dir` in the store `store` that `doomed` selects. Returns
    /// whether the sweep finished clean: `true` also when the store has no sweeper (nothing staged
    /// there is hidden).
    pub async fn sweep_staging(
        &self,
        store: &StoreId,
        dir: &Path,
        doomed: impl Fn(&str, DateTime<Utc>) -> bool + Send + Sync + 'static,
    ) -> bool {
        let sweeper = self
            .sched
            .sweepers
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(store)
            .cloned();
        let Some(sweeper) = sweeper else { return true };
        let dir = dir.clone();
        tokio::task::spawn_blocking(move || sweeper.sweep_staging(&dir, &doomed))
            .await
            .is_ok_and(|r| r.is_ok())
    }
}
