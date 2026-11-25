// SPDX-License-Identifier: Apache-2.0
//! Session methods that plan an analyzed query: listing and opening its files, optimizing, the physical plan.

use super::*;

/// Now, as microseconds since the epoch.
fn query_start_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_micros()).ok())
        .unwrap_or_default()
}

impl Session {
    /// The physical plan of `analyzed`, once the files it scans are
    /// listed and their footers read.
    pub(crate) async fn physical(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
    ) -> Result<(Plan, PhysicalPlan), KrishivError> {
        let (optimized, physical, _) = self
            .physical_with(analyzed, env, false, false, false, true)
            .await?;
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
        let (optimized, physical, _) = self
            .physical_with(analyzed, env, false, false, true, true)
            .await?;
        Ok((optimized, physical))
    }

    /// As [`Self::physical_inspected`] for a plan that only feeds a number or a list
    /// (`DataFrame::warnings`, `DataFrame::estimate`): no scan that decides once runs user code.
    pub(crate) async fn physical_counted(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
    ) -> Result<(Plan, PhysicalPlan), KrishivError> {
        self.authorize_shown_definer_reads(analyzed).await?;
        let (optimized, physical, _) = self
            .physical_with(analyzed, env, false, false, true, false)
            .await?;
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
        self.physical_with(analyzed, env, trace, false, true, true)
            .await
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
        let (optimized, physical, _) = self
            .physical_with(analyzed, env, false, true, true, false)
            .await?;
        Ok((optimized, physical))
    }

    /// As [`Self::physical`] for a plan made only to find the first scan (the row-level probe): no
    /// scan that decides once runs user code.
    pub(crate) async fn physical_probe(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
    ) -> Result<(Plan, PhysicalPlan), KrishivError> {
        let (optimized, physical, _) = self
            .physical_with(analyzed, env, false, false, false, false)
            .await?;
        Ok((optimized, physical))
    }

    async fn physical_with(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
        trace: bool,
        withheld: bool,
        inspected: bool,
        decisions: bool,
    ) -> Result<(Plan, PhysicalPlan, Vec<krishiv_optimizer::RuleStep>), KrishivError> {
        let start = self.query_start.unwrap_or_else(query_start_micros);
        // A metric-view read no aggregate rewrote, or a `measure(..)` left over, is analyzable but
        // never plannable (design note `P3-metric-views.md` §4).
        krishiv_plan::analyzer::refuse_unplannable(analyzed.plan.root())?;
        // A recursive CTE runs as an operator of the physical plan; the
        // session only tells each loop how many rows its readers need.
        let limited;
        let analyzed = if crate::recursion::has_loop(analyzed.plan.root()) {
            limited = crate::recursion::push_limits(analyzed)?;
            &limited
        } else {
            analyzed
        };
        // `input_file_name()` and the block functions read the file of the rows, after the cache substituted
        // its snapshots (whose rows have no file).
        let bound;
        let analyzed = match krishiv_plan::analyzer::bind_input_files(&analyzed.plan, &|s| {
            env.catalog.scan_file_metadata(s)
        })? {
            Some(plan) => {
                bound = AnalyzedPlan {
                    plan,
                    ..analyzed.clone()
                };
                &bound
            }
            None => analyzed,
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
        let optimize = || {
            if trace {
                krishiv_optimizer::optimize_traced_with_udfs(
                    &analyzed.plan,
                    &stats,
                    &self.state.udfs,
                    &env.config,
                )
            } else {
                krishiv_optimizer::optimize_with_udfs(
                    &analyzed.plan,
                    &stats,
                    &self.state.udfs,
                    &env.config,
                )
                .map(|p| (p, Vec::new()))
            }
        };
        // Key counts (design P2b A5 (d)): the optimizer asks for the keys it
        // could use; with `krishiv.optimizer.discoverKeys` on they are
        // counted and, when one is proven, the plan is optimized once more
        // (never again). A withheld plan never counts: it must not depend on
        // the data, and its providers record no request.
        if !withheld {
            env.catalog.clear_key_requests();
        }
        let (mut optimized, mut steps) = optimize()?;
        // Boxed: the count's future would deepen every caller's (layout depth).
        if !withheld && Box::pin(env.catalog.discover_keys()).await {
            (optimized, steps) = optimize()?;
        }
        let cx = PlanningContext::new(tables, Arc::clone(&env.config))
            .with_query_start(start)
            .with_sample_salt(env.sample_salt)
            .with_user(self.state.principal.name.as_str())
            .with_udfs(Arc::clone(&self.state.udfs))
            .with_inspection(inspected)
            .with_spark_version(self.state.profile.kernel_version())
            .with_decisions(decisions);
        // Planning recurses on a `krishiv-deep` thread this worker joins, and a data source's
        // `decide` may run user code there: tell the runtime the worker is blocked so its core
        // goes to another task (a one-thread executor would otherwise stall every other query).
        let plan = || self.state.planner.plan(&optimized, &cx);
        // `krishiv-deep` threads lose the action's task-local: hand its cancellation handle on.
        let plan = || match crate::action::cancel_handle() {
            Some(handle) => krishiv_common::stack::with_inherited(handle, plan),
            None => plan(),
        };
        let physical = if tokio::runtime::Handle::try_current()
            .is_ok_and(|h| h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread)
        {
            tokio::task::block_in_place(plan)?
        } else {
            plan()?
        };
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
