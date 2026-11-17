// SPDX-License-Identifier: Apache-2.0
//! `SQL SECURITY DEFINER` views (P3-T062 §2.4): a view's query read with its definer's rights.

use super::*;

impl Session {
    /// The `SQL SECURITY DEFINER` view `d` read with its definer's rights (P3-T062 §2.4): the
    /// lookups of its text, then its query analyzed, both as the definer acting for this session's
    /// principal (`Principal::definer`; a derived principal holds only its own grants, never
    /// the reader's role). Inside it, the queries of other definer views (`texts`, trusted by node
    /// identity) are theirs.
    /// Returns what its query reads, for the view side table.
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
        self.authorize_reads_as(&d.reads, env, &definer).await?;
        let mut ctx = env.analysis.clone();
        ctx.principal = definer;
        ctx.definer_texts = texts.without(query);
        let plan = Plan::try_new(Arc::clone(query))?;
        let analyzed = DefaultAnalyzer
            .analyze(&plan, &ctx)
            .await
            .map_err(|e| e.error)?;
        Ok(analyzed.securables)
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
}
