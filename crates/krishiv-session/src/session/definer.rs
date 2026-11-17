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
        // Defence in depth (review r4 Y1c): every table name of the view's text was qualified to its
        // namespace when it was inlined; a bare read left over would bind wherever an analysis is current,
        // so it is refused, never trusted.
        if let Some(name) =
            krishiv_plan::analyzer::unscoped_read(query, env.analysis.conf.case_sensitive)
        {
            let table = krishiv_plan::analyzer::TableIdent {
                catalog: crate::catalog::CATALOG.to_owned(),
                namespace: vec![d.namespace.clone()],
                name: name.join("."),
            };
            return Err(krishiv_plan::analyzer::insufficient_permissions(
                &definer,
                &krishiv_plan::analyzer::Securable::Table(table),
                Privilege::Select,
            ));
        }
        let views = self
            .authorize_reads_as(&d.reads, env, &definer, false)
            .await?;
        let mut ctx = env.analysis.clone();
        ctx.principal = Arc::clone(&definer);
        ctx.definer_texts = texts.without(query);
        // Bound where the view was created, never where the reader is (review r4 Y1b): its own `WITH`s
        // were resolved under its captured policy before it was inlined, so none is left to the reader's.
        ctx.catalogs = Arc::new(ViewCatalogs::new(
            Arc::clone(&env.analysis.catalogs),
            d.namespace.clone(),
            d.path.clone(),
        ));
        ctx.conf.cte_precedence = krishiv_plan::analyzer::CtePrecedencePolicy::Corrected;
        // All of it is a stored view's text: no variable of the reader's session reads in it (review
        // r5 Z2), and its calls are those the inlining qualified (Z1).
        ctx.variables = Arc::new(krishiv_plan::analyzer::VariableSnapshot::default());
        ctx.stored_view_texts =
            krishiv_plan::analyzer::views::DefinerTexts::new(vec![Arc::clone(query)]);
        // The stored functions its text calls are the definer's to call (review r5 M7): recorded
        // apart from the query's, asked of the definer.
        let calls = Arc::new(env.sql_functions.recording_fork());
        ctx.functions = Arc::clone(&calls) as Arc<dyn krishiv_plan::analyzer::FunctionRegistry>;
        let plan = Plan::try_new(Arc::clone(query))?;
        let analyzed = DefaultAnalyzer
            .analyze(&plan, &ctx)
            .await
            .map_err(|e| e.error)?;
        let privilege = env.analysis.mode.read_privilege();
        // Its own calls and what their bodies look up, not those of the definer views inside it,
        // which are their definers' (review r6 m1); the bodies read as the definer (r6 M8).
        let scope = calls.definer_scope(query, &ctx.definer_texts, self.state.profile);
        self.authorize_reads_as(&scope.body.reads, env, &definer, false)
            .await?;
        self.authorize_reads_as(&scope.body.definer_reads, env, &definer, false)
            .await?;
        if let Some((e, _)) = scope.body.deferred {
            return Err(e);
        }
        for call in &scope.calls {
            krishiv_plan::analyzer::authorize(
                &*self.state.policy,
                &definer,
                &function_securable(call),
                privilege,
            )
            .await?;
        }
        // Each body reads with the intersection of the definer and the function's owner (review
        // r7 M9): the owner wrote its text, so every read and call in it is asked of the owner too.
        for owned in &scope.owned {
            let Some(owner) = self.function_owner(&owned.function) else {
                continue;
            };
            if owner == d.definer {
                continue;
            }
            let owner = Arc::new(Principal::definer(owner, self.state.principal.name.clone()));
            self.authorize_reads_as(&owned.reads.reads, env, &owner, false)
                .await?;
            self.authorize_reads_as(&owned.reads.definer_reads, env, &owner, false)
                .await?;
            for call in &owned.calls {
                krishiv_plan::analyzer::authorize(
                    &*self.state.policy,
                    &owner,
                    &function_securable(call),
                    privilege,
                )
                .await?;
            }
        }
        env.sql_functions.note_definer_calls(scope.calls);
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

    /// The owner the grants provider records for the stored function `(namespace, name)`; `None`
    /// without the grants provider, or for a function it records no owner of (then the definer
    /// alone is asked about its body).
    fn function_owner(&self, function: &(String, String)) -> Option<String> {
        self.state
            .grants
            .as_ref()?
            .owner_of(&function_securable(function))
    }

    /// The views the bodies of the functions a query called read, once the policy allows them; a
    /// failure inside a view's text after that. The bodies only a definer view's text calls were
    /// its definer's to read (review r6 M8). The lookups of a definer view's text a body reads are
    /// the principal's too (fails closed), but a refusal of one, or a failure inside its text, is
    /// told as a definer failure: the reader wrote none of its names.
    ///
    /// Boxed: `analyze_in` runs on the deepest stacks (a script's nested
    /// statements), where the size of its future counts.
    pub(super) fn authorize_body_reads<'a>(
        &'a self,
        env: &'a QueryEnv,
    ) -> futures::future::BoxFuture<'a, Result<Vec<krishiv_plan::analyzer::TableIdent>, KrishivError>>
    {
        Box::pin(async move {
            let body = env.sql_functions.body_reads();
            let mut views = if body.reads.is_empty() {
                Vec::new()
            } else {
                Box::pin(self.authorize_reads(&body.reads, env)).await?
            };
            if !body.definer_reads.is_empty() {
                let principal = Arc::clone(&self.state.principal);
                match Box::pin(self.authorize_reads_as(&body.definer_reads, env, &principal, false))
                    .await
                {
                    Ok(more) => views.extend(more),
                    Err(e) => return Err(self.definer_failure(env, e)),
                }
            }
            match body.deferred {
                Some((e, true)) => Err(self.definer_failure(env, e)),
                Some((e, false)) => Err(e),
                None => Ok(views),
            }
        })
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
            let (query, mut kept) = crate::view_schema::apply_renames_keeping(
                &read.query,
                &renames,
                &[&analysis.definer_texts, &analysis.stored_view_texts],
            );
            let Ok(plan) = Plan::try_new(query) else {
                continue;
            };
            let renamed;
            let analysis = if renames.is_empty() {
                analysis
            } else {
                let mut ctx = analysis.clone();
                ctx.stored_view_texts = kept.pop().unwrap_or_default();
                ctx.definer_texts = kept.pop().unwrap_or_default();
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

/// The stored function `(namespace, name)` of the session catalog as a securable.
fn function_securable((namespace, name): &(String, String)) -> krishiv_plan::analyzer::Securable {
    krishiv_plan::analyzer::Securable::Function(vec![
        crate::catalog::CATALOG.to_owned(),
        namespace.clone(),
        name.clone(),
    ])
}

/// The query's catalogs as a stored view's text sees them (Spark's `viewCatalogAndNamespace` and
/// the frozen SQL path): the view's catalog and namespace are current, and unqualified routines
/// and relations resolve through its frozen path. The catalogs themselves are the query's.
pub(crate) struct ViewCatalogs {
    inner: Arc<dyn krishiv_plan::analyzer::CatalogResolver>,
    namespace: Vec<String>,
    path: Vec<Vec<String>>,
}

impl ViewCatalogs {
    /// `inner` seen from a view created in `namespace` of the session catalog under `path` (empty
    /// when the view was stored without one: the built-ins, then its namespace).
    pub(crate) fn new(
        inner: Arc<dyn krishiv_plan::analyzer::CatalogResolver>,
        namespace: String,
        path: Vec<Vec<String>>,
    ) -> Self {
        Self {
            inner,
            namespace: vec![namespace],
            path,
        }
    }
}

impl krishiv_plan::analyzer::CatalogResolver for ViewCatalogs {
    fn current_catalog(&self) -> &str {
        crate::catalog::CATALOG
    }

    fn current_namespace(&self) -> &[String] {
        &self.namespace
    }

    fn catalog(&self, name: &str) -> Option<Arc<dyn krishiv_plan::analyzer::Catalog>> {
        self.inner.catalog(name)
    }

    fn epoch(&self) -> u64 {
        self.inner.epoch()
    }

    fn sql_path(&self) -> Vec<Vec<String>> {
        if !self.path.is_empty() {
            return self.path.clone();
        }
        let mut current = vec![crate::catalog::CATALOG.to_owned()];
        current.extend(self.namespace.iter().cloned());
        vec![vec!["system".to_owned(), "builtin".to_owned()], current]
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use krishiv_plan::analyzer::{CatalogResolver, MemoryCatalog};

    use super::ViewCatalogs;

    /// The reader's catalogs, current in `ns2`.
    struct Reader;

    impl CatalogResolver for Reader {
        fn current_catalog(&self) -> &str {
            "spark_catalog"
        }
        fn current_namespace(&self) -> &[String] {
            const NS: &[String] = &[];
            NS
        }
        fn catalog(&self, name: &str) -> Option<Arc<dyn krishiv_plan::analyzer::Catalog>> {
            (name == "spark_catalog").then(|| {
                Arc::new(MemoryCatalog::new("spark_catalog"))
                    as Arc<dyn krishiv_plan::analyzer::Catalog>
            })
        }
        fn epoch(&self) -> u64 {
            7
        }
        fn sql_path(&self) -> Vec<Vec<String>> {
            vec![vec!["spark_catalog".to_owned(), "ns2".to_owned()]]
        }
    }

    /// Review r4 Y1b: the definer's text is analyzed in the view's own catalog, namespace and
    /// frozen path (Spark's `viewCatalogAndNamespace`), never the reader's current ones; the
    /// catalogs themselves are the reader's query snapshot.
    #[test]
    fn a_definer_text_binds_in_the_views_namespace() {
        let parts = |p: &[&str]| p.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        let frozen = vec![
            parts(&["system", "builtin"]),
            parts(&["spark_catalog", "vns"]),
        ];
        let view = ViewCatalogs::new(Arc::new(Reader), "vns".to_owned(), frozen.clone());
        assert_eq!(view.current_catalog(), "spark_catalog");
        assert_eq!(view.current_namespace(), ["vns"]);
        assert_eq!(view.sql_path(), frozen);
        assert_eq!(view.epoch(), 7);
        assert!(view.catalog("spark_catalog").is_some());
        // A view stored without a path resolves through the built-ins and its namespace only.
        let bare = ViewCatalogs::new(Arc::new(Reader), "vns".to_owned(), Vec::new());
        assert_eq!(
            bare.sql_path(),
            vec![
                parts(&["system", "builtin"]),
                parts(&["spark_catalog", "vns"])
            ]
        );
    }
}
