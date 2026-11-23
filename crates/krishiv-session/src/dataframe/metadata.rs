// SPDX-License-Identifier: Apache-2.0
//! Metadata analysis (ADR 0046 §3b, P3 PR-14c): the analysis of a frame whose plan is never run, for
//! Connect `AnalyzePlan` (schema, tree string, `is_local`, semantic hash, same semantics). Its catalog
//! table and view reads ask `Browse` instead of `Select` (`AnalysisMode::Metadata`), so a viewer sees
//! the stored schema of what it may not read. Reads by path keep `Select`: a path has no stored schema,
//! and inference reads file contents (CSV `header=true` names are the file's first line).
//!
//! No reuse across privileges: a metadata analysis is never kept in the frame (`DataFrame::analysis`,
//! which every action reads, holds only `Select`-checked analyses), and its context fingerprint carries
//! the mode, so the session's prefix cache never serves its resolution to a run (nor the reverse).
//! Bind asks the policy on every analysis either way. A lazy checkpoint the frame holds is taken as
//! for any analysis: that runs its query through the ordinary, `Select`-checked path.

use std::sync::Arc;

use arrow_schema::SchemaRef;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::{AnalysisMode, AnalyzedPlan};

use super::{Analysis, DataFrame, ExplainMode};

impl DataFrame {
    /// The result's schema, as [`Self::schema`], for a caller that only shows it: every catalog table
    /// and view the plan reads needs `Browse` (metadata), not `Select` (ADR 0046 §3b); a read by path
    /// still needs `Select` (its schema is inferred from file contents). The analysis is
    /// not kept, so a later action analyzes again and asks `Select`.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![
    /// #     arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false),
    /// # ]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let schema = futures::executor::block_on(s.table("t").schema_for_metadata())?;
    /// assert_eq!(schema.field(0).name(), "a");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Spark's analysis errors; `KRISHIV_INSUFFICIENT_PERMISSIONS` without `Browse` on a read.
    pub async fn schema_for_metadata(&self) -> Result<SchemaRef, KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move {
                let analyzed = df.metadata_analysis_here().await?;
                Ok(super::client_schema_of(&analyzed))
            })
            .await
    }

    /// The query's plans as text, as [`Self::explain`], for a caller that only shows them (Connect
    /// `AnalyzePlan.explain`; ADR 0046 §3b). A principal whose reads pass the ordinary `Select`
    /// check gets [`Self::explain`]'s text. One that holds only `Browse` on the catalog tables gets
    /// the plan chosen with no table statistics, no distinct-value samples and no result-cache note,
    /// ending in a `== Note ==` section that says so. Reads by path and permanent views still
    /// need `Select` (their contents or text are shown), and so does [`ExplainMode::Cost`]; a
    /// SQL function's body is shown, as Spark's `DESCRIBE FUNCTION EXTENDED` shows it.
    ///
    /// ```
    /// use krishiv_session::ExplainMode;
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![
    /// #     arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false),
    /// # ]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let df = s.table("t").limit(1);
    /// let text = futures::executor::block_on(df.explain_for_metadata(ExplainMode::Simple))?;
    /// assert_eq!(text, futures::executor::block_on(df.explain(ExplainMode::Simple))?);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Analysis and planning errors; `KRISHIV_INSUFFICIENT_PERMISSIONS` without `Browse` on a
    /// table, or without `Select` on what the withheld plan still needs it for.
    pub async fn explain_for_metadata(&self, mode: ExplainMode) -> Result<String, KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move { df.explain_for_metadata_text(mode, false).await })
            .await
    }

    /// The analysis of this DataFrame's plan, as [`Self::analyzed`], for a caller that never runs it
    /// (see [`Self::schema_for_metadata`]): reads need `Browse`, and the result is not kept. A plan
    /// holding a definer view's inlined query is shown only as [`Self::analyzed`] shows it (review
    /// r3 X2); a caller that only validates the plan asks [`Self::schema_for_metadata`].
    ///
    /// # Errors
    /// As [`Self::schema_for_metadata`]; `KRISHIV_INSUFFICIENT_PERMISSIONS`, redacted, as
    /// [`Self::analyzed`].
    #[cfg(feature = "unstable-ir")]
    pub async fn analyzed_for_metadata(&self) -> Result<Arc<AnalyzedPlan>, KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move {
                let analyzed = df.metadata_analysis_here().await?;
                df.session.authorize_shown_definer_reads(&analyzed).await?;
                Ok(analyzed)
            })
            .await
    }

    /// A fresh metadata analysis, never stored in the frame.
    async fn metadata_analysis_here(&self) -> Result<Arc<AnalyzedPlan>, KrishivError> {
        Ok(Arc::clone(&self.metadata_analysis().await?.analyzed))
    }

    /// A fresh metadata analysis with the environment it ran in (a withheld explain plans in
    /// it), never stored in the frame.
    pub(crate) async fn metadata_analysis(&self) -> Result<Arc<Analysis>, KrishivError> {
        if let Some(e) = &self.invalid {
            return Err(e.clone());
        }
        let mut env = self.session.query_env();
        env.analysis.mode = AnalysisMode::Metadata;
        let plan = self.plan_with_checkpoints().await?;
        // A metadata analysis runs nothing: no observation to complete.
        let (analyzed, _keeps) = self.session.analyze_in(&plan, &env, None).await?;
        env.catalog.check_reference_options(analyzed.plan.root())?;
        Ok(Arc::new(Analysis {
            env: Arc::new(env),
            analyzed: Arc::new(analyzed),
        }))
    }
}
