// SPDX-License-Identifier: Apache-2.0
//! Session methods that plan an analyzed query: listing and opening its files, optimizing, the physical plan.

use super::*;

impl Session {
    /// The physical plan of `analyzed`, once the files it scans are
    /// listed and their footers read.
    pub(crate) async fn physical(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
    ) -> Result<(Plan, PhysicalPlan), KrishivError> {
        let (optimized, physical, _) = self.physical_with(analyzed, env, false, false).await?;
        Ok((optimized, physical))
    }

    /// As [`Self::physical`] for a plan that is only inspected (EXPLAIN,
    /// estimates): a recursive CTE is not run and reads as an empty table.
    pub(crate) async fn physical_inspected(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
    ) -> Result<(Plan, PhysicalPlan), KrishivError> {
        // A plan that is shown shows the queries of its definer views (P3-T062 §2.4).
        self.authorize_shown_definer_reads(analyzed).await?;
        let (optimized, physical, _) = self.physical_with(analyzed, env, false, false).await?;
        Ok((optimized, physical))
    }

    /// As [`Self::physical_inspected`] (the plan is only shown, never run);
    /// with `trace`, also every optimizer rule application that changed the
    /// plan (`EXPLAIN VERBOSE`).
    pub(crate) async fn physical_traced(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
        trace: bool,
    ) -> Result<(Plan, PhysicalPlan, Vec<krishiv_optimizer::RuleStep>), KrishivError> {
        self.authorize_shown_definer_reads(analyzed).await?;
        self.physical_with(analyzed, env, trace, false).await
    }

    /// As [`Self::physical_inspected`] with every table's statistics and planning hints withheld
    /// and no distinct-value samples taken (P3 PR-14d): the plan a principal without `Select`
    /// may see, whose shape does not depend on the data.
    pub(crate) async fn physical_withheld(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
    ) -> Result<(Plan, PhysicalPlan), KrishivError> {
        self.authorize_shown_definer_reads(analyzed).await?;
        let (optimized, physical, _) = self.physical_with(analyzed, env, false, true).await?;
        Ok((optimized, physical))
    }

    async fn physical_with(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
        trace: bool,
        withheld: bool,
    ) -> Result<(Plan, PhysicalPlan, Vec<krishiv_optimizer::RuleStep>), KrishivError> {
        let start = query_start_micros();
        // A recursive CTE runs as an operator of the physical plan; the
        // session only tells each loop how many rows its readers need.
        let limited;
        let analyzed = if crate::recursion::has_loop(analyzed.plan.root()) {
            limited = crate::recursion::push_limits(analyzed)?;
            &limited
        } else {
            analyzed
        };
        let tables = if withheld {
            env.catalog
                .open_files_unsampled(analyzed.plan.root())
                .await?;
            crate::explain::withheld::resolver(
                Arc::clone(&env.catalog) as Arc<dyn krishiv_exec::planner::TableProviderResolver>
            )
        } else {
            env.catalog.open_files(analyzed.plan.root()).await?;
            Arc::clone(&env.catalog) as Arc<dyn krishiv_exec::planner::TableProviderResolver>
        };
        let stats = ProviderStatistics(Arc::clone(&tables));
        let (optimized, steps) = if trace {
            krishiv_optimizer::optimize_traced_with_udfs(
                &analyzed.plan,
                &stats,
                &self.state.udfs,
                &env.config,
            )?
        } else {
            (
                krishiv_optimizer::optimize_with_udfs(
                    &analyzed.plan,
                    &stats,
                    &self.state.udfs,
                    &env.config,
                )?,
                Vec::new(),
            )
        };
        let cx = PlanningContext::new(tables, Arc::clone(&env.config))
            .with_query_start(start)
            .with_sample_salt(env.sample_salt)
            .with_user(self.state.principal.name.as_str())
            .with_udfs(Arc::clone(&self.state.udfs));
        let physical = self.state.planner.plan(&optimized, &cx)?;
        // The admission request's estimate, only for a controller that reads it.
        if !withheld && self.state.admission.uses_estimate() {
            env.catalog.set_estimate(crate::explain::estimate_of(
                &optimized,
                &physical,
                &env.catalog,
            ));
        }
        // Spark logs what analysis and planning ignore (hints) as
        // warnings; `DataFrame::warnings` returns them too.
        for w in analyzed.warnings.iter().chain(physical.warnings()) {
            tracing::warn!(target: "krishiv::session::warnings", "{w}");
        }
        Ok((optimized, physical, steps))
    }
}
