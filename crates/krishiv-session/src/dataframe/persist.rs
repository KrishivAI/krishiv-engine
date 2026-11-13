// SPDX-License-Identifier: Apache-2.0
//! `DataFrame.cache`, `persist`, `unpersist`, `is_cached` and
//! `storage_level` (design P3-T083 3.1), and the fill of registered
//! subtrees an action uses. A child of `dataframe` so that it reads the
//! frame's private fields and keeps `dataframe.rs` small.

use std::sync::Arc;

use futures::StreamExt;
use krishiv_common::error::KrishivError;
use krishiv_plan::LogicalPlan;
use krishiv_plan::analyzer::AnalyzedPlan;

use super::{Analysis, CacheFill, DataFrame, StorageLevel};
use crate::cache::{FillClaim, PendingRegistration, RegKey};

impl DataFrame {
    /// Caches this frame's rows (PySpark's `cache`): `persist` at
    /// `spark.sql.defaultCacheStorageLevel` (`MEMORY_AND_DISK` by default).
    /// Registration is eager: [`Self::is_cached`] is true from the call on.
    /// Materialization is lazy and all-or-nothing: the first action whose
    /// plan contains this one stores the snapshot, and later actions of any
    /// frame containing the same plan read it instead of recomputing it;
    /// when the snapshot would exceed `krishiv.cache.maxBytes` the query
    /// runs uncached.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let df = s.table("t").cache();
    /// assert!(futures::executor::block_on(df.is_cached())?);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn cache(&self) -> Self {
        let level = StorageLevel::default_of(&self.session.state.conf.snapshot());
        self.persist(level)
    }

    /// Caches this frame's rows at `level` (PySpark's `persist`; see
    /// [`Self::cache`]). Every level is held in memory for now; the level is
    /// reported as registered ([`Self::storage_level`]). A plan cached
    /// already keeps its first level (logging Spark's "Asked to cache
    /// already cached data."); [`StorageLevel::NONE`] does nothing; the
    /// frame of a command is not cached (a `WARN`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// use krishiv_session::StorageLevel;
    /// let df = s.table("t").persist(StorageLevel::DISK_ONLY);
    /// assert_eq!(futures::executor::block_on(df.storage_level())?, StorageLevel::DISK_ONLY);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn persist(&self, level: StorageLevel) -> Self {
        if level == StorageLevel::NONE {
            return self.clone();
        }
        if self.command {
            tracing::warn!(
                target: "krishiv::session",
                "Asked to cache a plan that is inapplicable for caching: {}",
                self.plan
            );
            return self.clone();
        }
        if let Some(error) = &self.invalid {
            // The frame's own first action raises it.
            tracing::warn!(
                target: "krishiv::session",
                "Not caching a plan that does not analyze: {error}"
            );
            return self.clone();
        }
        self.session.state.cache.push_pending(PendingRegistration {
            plan: Arc::clone(&self.plan),
            keep: self.keep.clone(),
            level,
        });
        self.clone()
    }

    /// Removes this frame's plan from the cache, registration and data
    /// (PySpark's `unpersist`), returning the pool memory to baseline.
    /// Other registrations are unaffected (Spark's `cascade = false`): one
    /// whose plan contains this one keeps its data, or reads the source
    /// when it fills. Unpersisting an uncached frame does nothing.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let df = futures::executor::block_on(s.table("t").cache().unpersist())?;
    /// assert!(!futures::executor::block_on(df.is_cached())?);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Spark's analysis errors.
    pub async fn unpersist(&self) -> Result<Self, KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move {
                df.session.register_pending().await;
                let a = df.analysis().await?;
                df.session.state.cache.unregister(&RegKey::of(&a.analyzed));
                Ok(df)
            })
            .await
    }

    /// Whether this frame's plan is cached: registered by a `cache`,
    /// `persist` or `CACHE TABLE` of the same plan (by any frame), under
    /// this principal and policy version, whether or not an action has
    /// materialized it yet.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// assert!(!futures::executor::block_on(s.table("t").is_cached())?);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Spark's analysis errors.
    pub async fn is_cached(&self) -> Result<bool, KrishivError> {
        Ok(self.registered_level().await?.is_some())
    }

    /// The storage level this frame's plan is cached at (PySpark's
    /// `storageLevel`): the level it was registered with, or
    /// [`StorageLevel::NONE`] when it is not cached.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// use krishiv_session::StorageLevel;
    /// let df = s.table("t");
    /// assert_eq!(futures::executor::block_on(df.storage_level())?, StorageLevel::NONE);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Spark's analysis errors.
    pub async fn storage_level(&self) -> Result<StorageLevel, KrishivError> {
        Ok(self.registered_level().await?.unwrap_or(StorageLevel::NONE))
    }

    async fn registered_level(&self) -> Result<Option<StorageLevel>, KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move {
                df.session.register_pending().await;
                let a = df.analysis().await?;
                Ok(df.session.state.cache.level(&RegKey::of(&a.analyzed)))
            })
            .await
    }

    /// Fills the registered proper subtrees of `a`'s plan whose data is
    /// missing, each as its own query under `a`'s configuration snapshot
    /// and principal. A fill that fails or is refused leaves its
    /// registration pending: the action then computes the subtree itself,
    /// and raises any error it hits there.
    pub(super) async fn fill_subtrees(&self, a: &Arc<Analysis>) {
        let claims = self.session.state.cache.claim_subtrees(&a.analyzed);
        for (node, claim) in claims {
            if let Err(error) = Box::pin(self.fill_subtree(a, node, claim)).await {
                tracing::debug!(
                    target: "krishiv::session",
                    "a cached subplan was not filled: {error}"
                );
            }
        }
    }

    /// Runs `node`, a subtree of `a`'s plan, and stores its rows for the
    /// registration `claim` holds.
    async fn fill_subtree(
        &self,
        a: &Arc<Analysis>,
        node: Arc<LogicalPlan>,
        claim: FillClaim,
    ) -> Result<(), KrishivError> {
        let Some(output) = node.resolved_output() else {
            return Ok(());
        };
        let columns = output.len();
        let analyzed = AnalyzedPlan {
            plan: krishiv_plan::Plan::try_new(node)?,
            output: output.clone(),
            output_declared: vec![None; columns],
            output_comments: vec![None; columns],
            warnings: Vec::new(),
            ..(*a.analyzed).clone()
        };
        // Its own row in `system.queries`, not the statement's.
        let mut env = crate::session::QueryEnv::clone(&a.env);
        env.row = Arc::default();
        let sub = Arc::new(Analysis {
            env: Arc::new(env),
            analyzed: Arc::new(analyzed),
        });
        let (physical, _) = Box::pin(self.physical_cached(&sub, true)).await?;
        let fill = CacheFill::of(claim, &sub, &output);
        let stream = self.session.run(&physical, &sub.env).await?;
        let mut stream = fill.fill(stream, &self.session.state.cache);
        while let Some(batch) = stream.next().await {
            batch?;
        }
        Ok(())
    }
}
