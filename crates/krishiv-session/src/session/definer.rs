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
        // Every stored function its text reaches, at any depth, is one the definer owns and last
        // defined, or the version the view bound (review r9, ADR 0048 A1); else it fails closed.
        for call in &scope.calls {
            if !self.definer_may_call(d, &calls, call) {
                return Err(krishiv_plan::analyzer::insufficient_permissions(
                    &definer,
                    &function_securable(call),
                    env.analysis.mode.read_privilege(),
                ));
            }
        }
        // So is every stored view it reaches, nested definer views included: the version the view
        // pinned, or one its definer owns and last defined (review r10 M10, ADR 0048 A1).
        let reached: Vec<&crate::catalog::ReachedView> =
            d.views.iter().chain(&scope.body.views).collect();
        for v in &reached {
            if d.bound.views.get(&v.view) != Some(&v.revision)
                && !self.definer_wrote(&d.definer, &view_securable(&v.view), &v.author)
            {
                return Err(krishiv_plan::analyzer::insufficient_permissions(
                    &definer,
                    &view_securable(&v.view),
                    env.analysis.mode.read_privilege(),
                ));
            }
        }
        self.authorize_reads_as(&scope.body.reads, env, &definer, false)
            .await?;
        self.authorize_reads_as(&scope.body.definer_reads, env, &definer, false)
            .await?;
        if let Some((e, _)) = scope.body.deferred {
            return Err(e);
        }
        // Through the analysis's policy, never the session's: a trial analysis installed a
        // `QuietPolicy`, so a refusal there tells the provider nothing (T063 R2-1).
        for call in &scope.calls {
            krishiv_plan::analyzer::authorize(
                &*env.analysis.policy,
                &definer,
                &function_securable(call),
                privilege,
            )
            .await?;
        }
        // Each body reads with the intersection of the definer, the function's owner and the
        // author of its version (review r7 M9, r10 m1): they wrote its text, so every read and call
        // in it is asked of them too.
        for owned in &scope.owned {
            let author = calls
                .stored_identity(&owned.function)
                .map(|(_, author)| author.to_owned())
                .unwrap_or_default();
            let securable = function_securable(&owned.function);
            for writer in self.writers(&d.definer, &securable, &author) {
                let writer = Arc::new(Principal::definer(
                    writer,
                    self.state.principal.name.clone(),
                ));
                self.authorize_reads_as(&owned.reads.reads, env, &writer, false)
                    .await?;
                self.authorize_reads_as(&owned.reads.definer_reads, env, &writer, false)
                    .await?;
                Self::authorize_calls_as(&owned.calls, env, &writer, privilege).await?;
                // And the columns it reads, through the column pass the definer's analysis
                // made (review r11 B1).
                let krishiv_plan::analyzer::FunctionLookup::Sql(def) =
                    krishiv_plan::analyzer::FunctionRegistry::lookup(
                        &*calls,
                        &[
                            crate::catalog::CATALOG.to_owned(),
                            owned.function.0.clone(),
                            owned.function.1.clone(),
                        ],
                        self.state.profile,
                    )
                else {
                    // Fail closed: a body whose text cannot be found is not checked, so refuse.
                    return Err(self.definer_failure(
                        env,
                        krishiv_common::error::InternalError {
                            message: "a definer's function body is not a stored SQL function"
                                .to_owned(),
                        }
                        .into(),
                    ));
                };
                let qualifier = [crate::catalog::CATALOG.to_owned(), owned.function.0.clone()];
                let call = crate::sql_functions::null_call(&def, &qualifier);
                Self::analyze_as_writer(&call, &ctx, &calls, &writer).await?;
            }
        }
        // So does each stored view it reaches (review r10 M10): its lookups, and the calls and
        // bodies' lookups of its text, are asked of its owner and author too.
        for v in &reached {
            let writers = self.writers(&d.definer, &view_securable(&v.view), &v.author);
            if writers.is_empty() {
                continue;
            }
            let under = v
                .query
                .as_ref()
                .map(|q| calls.definer_scope(q, &ctx.definer_texts, self.state.profile));
            for writer in writers {
                let writer = Arc::new(Principal::definer(
                    writer,
                    self.state.principal.name.clone(),
                ));
                self.authorize_reads_as(&v.reads, env, &writer, false)
                    .await?;
                if let Some(under) = &under {
                    self.authorize_reads_as(&under.body.reads, env, &writer, false)
                        .await?;
                    self.authorize_reads_as(&under.body.definer_reads, env, &writer, false)
                        .await?;
                    Self::authorize_calls_as(&under.calls, env, &writer, privilege).await?;
                }
                // And the columns its text reads, through the column pass the definer's
                // analysis made (review r11 B1).
                let Some(q) = &v.query else {
                    // Fail closed: a view whose text is missing is not checked, so refuse.
                    return Err(self.definer_failure(
                        env,
                        krishiv_common::error::InternalError {
                            message: "a definer's reached view has no text".to_owned(),
                        }
                        .into(),
                    ));
                };
                Self::analyze_as_writer(q, &ctx, &calls, &writer).await?;
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

    /// Whether the definer view `d`'s text may call the stored function `call` (review r9, ADR
    /// 0048 A1): its version is the one the view bound when created or last altered, or its
    /// definer last defined it and owns it (the grants provider records the definer as its owner,
    /// or no grants provider is set). A replacement by anyone else, or a function resolved at a
    /// read (in an invoker view's text or a body) that is neither, is refused.
    fn definer_may_call(
        &self,
        d: &crate::catalog::DefinerRead,
        functions: &crate::sql_functions::QueryFunctions,
        call: &(String, String),
    ) -> bool {
        let Some((revision, author)) = functions.stored_identity(call) else {
            return false;
        };
        d.bound.versions.get(call) == Some(&revision)
            || self.definer_wrote(&d.definer, &function_securable(call), author)
    }

    /// Whether `definer` last defined `securable` (a stored function or view whose current version
    /// `author` wrote) and owns it: the grants provider records `definer` as its owner, or no
    /// grants provider is set (review r9, r10 M10).
    fn definer_wrote(
        &self,
        definer: &str,
        securable: &krishiv_plan::analyzer::Securable,
        author: &str,
    ) -> bool {
        author == definer
            && self
                .state
                .grants
                .as_ref()
                .is_none_or(|g| g.owner_of(securable).as_deref() == Some(definer))
    }

    /// Besides `definer`, who wrote `securable` (a stored function or view whose current version
    /// `author` wrote), each asked about what its text reads (review r7 M9, r10 m1 and M10): the
    /// owner the grants provider records, and the author, when not the definer. Empty when the
    /// definer wrote it alone.
    fn writers(
        &self,
        definer: &str,
        securable: &krishiv_plan::analyzer::Securable,
        author: &str,
    ) -> Vec<String> {
        let mut writers = Vec::new();
        if let Some(owner) = self
            .state
            .grants
            .as_ref()
            .and_then(|g| g.owner_of(securable))
            && owner != definer
        {
            writers.push(owner);
        }
        if !author.is_empty() && author != definer && !writers.iter().any(|w| w == author) {
            writers.push(author.to_owned());
        }
        writers
    }

    /// Analyzes `text`, a writer's text inside a definer view's read (a reached stored view's
    /// inlined query, or a call of a reached function body), as `writer` in the definer's
    /// analysis context `ctx` (review r11 B1): Bind asks the writer about its tables, and under a
    /// column-level policy the analyzer's column pass asks it `Select` on exactly the columns the
    /// text reads (every column a `*` expands to; whole rows ask the table), the check the
    /// definer gets, never only whether the table may be seen. The texts of the definer views
    /// inside it stay their definers' (`ctx.definer_texts`). Its calls are recorded apart and
    /// dropped: the caller asks the writer about them.
    ///
    /// # Errors
    /// The writer's refusal (42501), or the text's analysis error: a definer failure.
    async fn analyze_as_writer(
        text: &Arc<LogicalPlan>,
        ctx: &AnalysisContext,
        calls: &crate::sql_functions::QueryFunctions,
        writer: &Arc<Principal>,
    ) -> Result<(), KrishivError> {
        let mut ctx = ctx.clone();
        ctx.principal = Arc::clone(writer);
        ctx.functions =
            Arc::new(calls.recording_fork()) as Arc<dyn krishiv_plan::analyzer::FunctionRegistry>;
        ctx.stored_view_texts =
            krishiv_plan::analyzer::views::DefinerTexts::new(vec![Arc::clone(text)]);
        let plan = Plan::try_new(Arc::clone(text))?;
        DefaultAnalyzer
            .analyze(&plan, &ctx)
            .await
            .map(|_| ())
            .map_err(|e| e.error)
    }

    /// Asks the analysis's policy (`env.analysis.policy`: the session's, or the `QuietPolicy` of a
    /// trial analysis, T063 R2-1) whether `principal` may call each stored function of `calls`.
    async fn authorize_calls_as(
        calls: &[(String, String)],
        env: &QueryEnv,
        principal: &Principal,
        privilege: Privilege,
    ) -> Result<(), KrishivError> {
        for call in calls {
            krishiv_plan::analyzer::authorize(
                &*env.analysis.policy,
                principal,
                &function_securable(call),
                privilege,
            )
            .await?;
        }
        Ok(())
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
            let mut sets = vec![&analysis.definer_texts, &analysis.stored_view_texts];
            sets.extend(analysis.view_collations.iter().map(|(_, t)| t));
            let (query, mut kept) =
                crate::view_schema::apply_renames_keeping(&read.query, &renames, &sets);
            let Ok(plan) = Plan::try_new(query) else {
                continue;
            };
            let renamed;
            // The view's own text analyses under its own default collation.
            let own = read.collation != analysis.conf.default_collation;
            let analysis = if renames.is_empty() && !own {
                analysis
            } else {
                let mut ctx = analysis.clone();
                if !renames.is_empty() {
                    let mut rest = kept.split_off(2).into_iter();
                    for (_, texts) in &mut ctx.view_collations {
                        *texts = rest.next().unwrap_or_default();
                    }
                    ctx.stored_view_texts = kept.pop().unwrap_or_default();
                    ctx.definer_texts = kept.pop().unwrap_or_default();
                }
                ctx.conf.default_collation = read.collation;
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

/// The stored view `(namespace, name)` of the session catalog as a securable.
fn view_securable((namespace, name): &(String, String)) -> krishiv_plan::analyzer::Securable {
    krishiv_plan::analyzer::Securable::View(krishiv_plan::analyzer::TableIdent {
        catalog: crate::catalog::CATALOG.to_owned(),
        namespace: vec![namespace.clone()],
        name: name.clone(),
    })
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

/// The analysis context that trusts the inlined texts by identity: the definer views' queries
/// (read as their definers), every permanent view's and every temporary view's text; `None` when
/// the query inlined none, so the session's own context is used as it is.
pub(super) fn trusting_context(
    base: &krishiv_plan::analyzer::AnalysisContext,
    definers: &krishiv_plan::analyzer::views::DefinerTexts,
    stored: &krishiv_plan::analyzer::views::DefinerTexts,
    temp: &krishiv_plan::analyzer::views::DefinerTexts,
    collations: &[(
        krishiv_types::Collation,
        krishiv_plan::analyzer::views::DefinerTexts,
    )],
) -> Option<krishiv_plan::analyzer::AnalysisContext> {
    if definers.is_empty() && stored.is_empty() && temp.is_empty() {
        return None;
    }
    let mut ctx = base.clone();
    ctx.view_collations = collations.to_vec();
    ctx.definer_texts = definers.clone();
    ctx.stored_view_texts = stored.clone();
    ctx.temp_view_texts = temp.clone();
    Some(ctx)
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
