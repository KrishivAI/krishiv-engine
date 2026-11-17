// SPDX-License-Identifier: Apache-2.0
//! Stored view reads: `SQL SECURITY DEFINER` views (P3-T062 §2.4), a view's query read with its
//! definer's rights, and the schema-binding checks of the stored views a query inlined.

use super::*;

impl Session {
    /// The `SQL SECURITY DEFINER` view `d` read with its definer's rights (P3-T062 §2.4): the
    /// lookups of its text, then its query analyzed, both as the definer acting for this session's
    /// principal (`Principal::definer`; a derived principal holds only its own grants, never
    /// the reader's role). Inside it, the queries of other definer views (`texts`, trusted by node
    /// identity) are theirs.
    /// Returns what its query reads (its tables and the views its text inlines), for the view
    /// side table.
    ///
    /// # Errors
    /// The definer's denial (42501, naming the definer), or its query's analysis error.
    pub(super) async fn authorize_definer(
        &self,
        d: &crate::catalog::DefinerRead,
        texts: &krishiv_plan::analyzer::views::DefinerTexts,
        env: &QueryEnv,
    ) -> Result<Vec<krishiv_plan::analyzer::SecurableUse>, KrishivError> {
        let Some(query) = &d.query else {
            return Err(krishiv_common::error::InternalError {
                message: format!("definer view `{}` was not inlined", d.name.join(".")),
            }
            .into());
        };
        let definer = Arc::new(Principal::definer(
            d.definer.clone(),
            self.state.principal.name.clone(),
        ));
        let views = self.authorize_reads_as(&d.reads, env, &definer).await?;
        let mut ctx = env.analysis.clone();
        ctx.principal = definer;
        ctx.definer_texts = texts.without(query);
        let plan = Plan::try_new(Arc::clone(query))?;
        let analyzed = DefaultAnalyzer
            .analyze(&plan, &ctx)
            .await
            .map_err(|e| e.error)?;
        // The views its text inlines show in a plan of it too (review M1): a reader shown the
        // plan is asked `Select` on each, as on a view it read itself.
        let mut reads = analyzed.securables;
        for view in views {
            let securable = krishiv_plan::analyzer::Securable::View(view);
            if !reads.iter().any(|u| u.securable == securable) {
                reads.push(krishiv_plan::analyzer::SecurableUse {
                    securable,
                    privilege: Privilege::Select,
                    columns: std::collections::BTreeSet::new(),
                });
            }
        }
        Ok(reads)
    }

    /// What the reader is told of the failure `e` of a definer view's read (review B2; ADR 0048
    /// §3: a denial names only what the statement wrote; design §6: the canary never shows in
    /// error text): its class and SQLSTATE, every message parameter redacted, no position in a
    /// text. The record of a refusal keeps the whole error (the definer, the securable): it is
    /// audited here, before the redaction.
    pub(crate) fn definer_failure(&self, env: &QueryEnv, e: KrishivError) -> KrishivError {
        let Err(e) = crate::access::audit_read::<()>(self, env, Err(e)) else {
            return krishiv_common::error::InternalError {
                message: "a definer failure without an error".to_owned(),
            }
            .into();
        };
        e.redacted().without_pending_site()
    }

    /// The schema binding of each stored view `reads` inlined (`view_schema`): its query analyzed alone, inner
    /// views first. Returns the evolved entries to store once the whole query analyzed, and the renames of reads.
    ///
    /// # Errors
    /// What [`crate::view_schema::check_read`] raises, with whether the failing view is a definer view or lies
    /// inside one's text ([`crate::view_schema::StoredRead::in_definer`], review r2 R1).
    pub(super) async fn check_stored_views(
        &self,
        reads: &[crate::view_schema::StoredRead],
        env: &QueryEnv,
        analysis: &AnalysisContext,
    ) -> Result<crate::view_schema::StoredViewChecks, (KrishivError, bool)> {
        let compensation = crate::view_schema::compensation(&env.config);
        let mut synced = Vec::new();
        let mut renames = Vec::new();
        for read in reads {
            // A view whose query does not analyze alone is left to the outer query.
            let (query, texts) = crate::view_schema::apply_renames_keeping(
                &read.query,
                &renames,
                &analysis.definer_texts,
            );
            let Ok(plan) = Plan::try_new(query) else {
                continue;
            };
            let renamed;
            let analysis = if renames.is_empty() {
                analysis
            } else {
                let mut ctx = analysis.clone();
                ctx.definer_texts = texts;
                renamed = ctx;
                &renamed
            };
            let Ok(analyzed) = DefaultAnalyzer.analyze(&plan, analysis).await else {
                continue;
            };
            let case_sensitive = env.analysis.conf.case_sensitive;
            let check =
                crate::view_schema::check_read(read, &analyzed, case_sensitive, compensation)
                    .map_err(|e| (e, read.in_definer))?;
            if let Some(changed) = check.changed {
                synced.push((Arc::clone(&read.entry), changed));
            }
            renames.extend(check.rename);
        }
        Ok((synced, renames))
    }
}
