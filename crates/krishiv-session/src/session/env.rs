// SPDX-License-Identifier: Apache-2.0
//! The session's per-query environment, parser options and statement-context helpers.

use super::*;

/// The parser's configs (`ParserOptions::from_conf` reads these).
pub(super) const PARSER_KEYS: [&str; 19] = [
    "spark.sql.ansi.enabled",
    "spark.sql.ansi.enforceReservedKeywords",
    "spark.sql.ansi.doubleQuotedIdentifiers",
    "spark.sql.legacy.setopsPrecedence.enabled",
    "spark.sql.parser.escapedStringLiterals",
    "spark.sql.legacy.bangEqualsNot",
    "spark.sql.legacy.identifierClause",
    "spark.sql.legacy.consecutiveStringLiterals.enabled",
    "spark.sql.sources.bucketing.maxBuckets",
    "spark.sql.parser.singleCharacterPipeOperator.enabled",
    "spark.sql.legacy.viewSchemaBindingMode",
    "spark.sql.caseSensitive",
    "spark.sql.parser.quotedRegexColumnNames",
    "spark.sql.legacy.parser.havingWithoutGroupByAsWhere",
    "spark.sql.collation.schemaLevel.enabled",
    "spark.sql.timestampNanosTypes.enabled",
    "spark.sql.timestampType",
    "spark.sql.join.asofJoin.enabled",
    "spark.sql.timeTravel.atSyntax.enabled",
];

/// `KRISHIV_UNSUPPORTED_FEATURE` for a statement `Session::sql` cannot run yet.
pub(crate) fn unsupported_statement(what: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: format!("{what} in sessions"),
    }
    .into()
}

/// An empty redaction, for analyses with no SQL text (DataFrame errors
/// carry a DataFrame context, which needs no text).
pub(super) fn empty_redacted() -> RedactedSql {
    RedactedSql::opaque("")
}

/// An analysis error with its query context attached: the SQL span of
/// `text`, or the DataFrame operation that raised it. With no text, a
/// SQL span has no fragment to attach, so only DataFrame contexts cross.
pub(super) fn attach_analysis(e: AnalysisError, text: Option<&RedactedSql>) -> KrishivError {
    let empty;
    let redacted = match text {
        Some(t) => t,
        None => {
            empty = empty_redacted();
            &empty
        }
    };
    match e.query_context(redacted) {
        Some(PlanQueryContext::Sql {
            start,
            stop,
            fragment,
        }) if text.is_some() => e
            .error
            .with_query_context(ErrorQueryContext::sql(fragment, start, stop)),
        Some(PlanQueryContext::DataFrame { .. }) => {
            e.error.with_query_context(ErrorQueryContext::dataframe())
        }
        // A DataFrame action with no origin (e.g. an unbound `col("x")`):
        // the error still arose from a DataFrame operation.
        None if text.is_none() => e.error.with_query_context(ErrorQueryContext::dataframe()),
        _ => e.error,
    }
}

/// The parser options `config` sets, with the grammar of `profile` (Spark 4.3's type grammar from
/// 4.3 on).
pub(super) fn parser_options(config: &Config, profile: SparkProfile) -> ParserOptions {
    let values = conf_values(config, &PARSER_KEYS);
    ParserOptions {
        spark_4_3: profile >= SparkProfile::V4_3,
        ..ParserOptions::from_conf(|k| values.get(k).map(String::as_str))
    }
}

/// The values of `keys` that are set or have a default.
pub(super) fn conf_values<'k>(config: &Config, keys: &[&'k str]) -> BTreeMap<&'k str, String> {
    keys.iter()
        .filter_map(|k| conf::get(config, k).map(|v| (*k, v)))
        .collect()
}

/// The analysis inputs of one query and the versions they were taken at.
#[derive(Clone)]
pub(crate) struct QueryEnv {
    pub(crate) config: Arc<Config>,
    /// Set by a file write, whose profile is recorded after its commit: see [`crate::profile_history::HistoryHold`].
    pub(crate) history_hold: Option<Arc<crate::profile_history::HistoryHold>>,
    /// The query's I/O context: every read, scan and write of the query goes
    /// through it, so its counters are the query's own.
    pub(crate) io: krishiv_io::IoContext,
    pub(crate) parser: ParserOptions,
    pub(crate) analysis: AnalysisContext,
    pub(crate) catalog: Arc<QueryCatalog>,
    pub(crate) conf_version: u64,
    pub(crate) catalog_epoch: u64,
    /// The variable version the query's snapshot was taken at: a cached
    /// resolved plan is reused only while it is unchanged.
    pub(crate) variables_version: u64,
    /// The cache's invalidation generation before the query was analyzed.
    pub(crate) cache_generation: u64,
    /// `krishiv.cache.maxBytes` of the query's configuration snapshot.
    pub(crate) cache_cap: usize,
    /// The registry in `analysis`, as itself: the function map the query
    /// reads and the texts of function bodies.
    pub(crate) sql_functions: Arc<QueryFunctions>,
    /// [`FunctionRegistry::version`] of `analysis.functions`.
    pub(crate) functions_version: u64,
    /// The access policy's version when the query started: an analysis made under another one
    /// is not reused, since its reads were authorized under the old rules.
    pub(crate) policy_version: u64,
    /// The statement as the session may show it (redacted), for
    /// `system.queries`; `None` for a query no SQL text started.
    pub(crate) text: Option<Arc<str>>,
    /// The same text with the map from original offsets, to render the span of a run-time
    /// error ([`crate::run_site::resolve_site`]).
    pub(crate) redacted: Option<Arc<RedactedSql>>,
    /// The statement's class (`StatementClass::name`), for `system.queries`.
    pub(crate) statement_class: Option<&'static str>,
    /// The id of the `system.queries` row the statement's latest query began, if any: `sql`
    /// records a refusal only for a statement that never got a row, and a write corrects its row
    /// when a step after its run fails. Statement-local, so another statement of the same
    /// session cannot set or clear it.
    pub(crate) row: Arc<std::sync::Mutex<Option<String>>>,
    /// The spans of the names substituted for `IDENTIFIER(..)` clauses (`lower_identified`).
    pub(crate) identifier_spans: Arc<[krishiv_sql::ast::Span]>,
    /// The `IDENTIFIER(..)` text each of `identifier_spans` replaced (empty for `CACHE TABLE`).
    pub(crate) identifier_written: Arc<[String]>,
    /// The salt of the statement's unseeded `TABLESAMPLE` seeds
    /// ([`PlanningContext::sample_salt`](krishiv_exec::planner::PlanningContext::sample_salt)):
    /// drawn once per DataFrame and kept when it is analyzed again, so its
    /// actions sample the same rows.
    pub(crate) sample_salt: u64,
    /// The observations and sinks of the frame the query runs (`Dataset.observe`): every run of
    /// the query, whatever the action, completes them (`run_on_query_logged`).
    pub(crate) observers: Arc<crate::dataframe::Observers>,
    /// Set by `analyze_in` when the plan inlines a `SQL SECURITY DEFINER` view whose query the
    /// principal may not see as an invoker (review M2, the rule `EXPLAIN` follows): the query's
    /// operators (names, text metrics, row counts of the definer's plan) are kept out of
    /// `system.operators` and its profile out of the persisted history.
    pub(crate) withhold_profile: Arc<std::sync::atomic::AtomicBool>,
    /// A trial analysis (P3 PR-14d): its read denials are answers, not refusals, so they write no
    /// `krishiv::audit` record ([`crate::access::audit_read`]) and the policy is not told of them.
    /// Built [`QuietDenials::default`] (off); only [`Self::set_quiet_denials`] turns it on, together
    /// with the quiet `analysis.policy`, and only a caller that falls back to an analysis of its own,
    /// which is audited, calls it (T063 review R2-1 Q1).
    pub(crate) quiet: QuietDenials,
    /// The session's table functions, as `analysis.table_functions` holds them: asking the analysis's
    /// policy ([`Self::set_quiet_denials`]), so a trial's `Usage` check before a Python UDTF's `analyze`
    /// tells the policy nothing either.
    pub(crate) udtfs: Arc<crate::python_udfs::SessionTableFunctions>,
    /// Why this query's spill directory could not be set up, if it could not (PY-078): the query
    /// then reports its memory error as that failure. Shared by the clones of one statement's env.
    pub(crate) spill_failure: Arc<std::sync::Mutex<Option<KrishivError>>>,
}

/// Whether a [`QueryEnv`] is a trial analysis; its flag is private to this module, so it is set only
/// through [`QueryEnv::set_quiet_denials`], with the policy to match.
#[derive(Clone, Copy, Default)]
pub(crate) struct QuietDenials(bool);

impl QueryEnv {
    /// Whether this is a trial analysis ([`Self::set_quiet_denials`]).
    pub(crate) fn quiet_denials(&self) -> bool {
        self.quiet.0
    }

    /// Whether this analysis is an ordinary one that asks `policy` itself (the session's): what a kept
    /// analysis must be, so the checks made with it later are audited and told to the policy.
    pub(crate) fn asks_policy(
        &self,
        policy: &Arc<dyn krishiv_plan::analyzer::AccessPolicy>,
    ) -> bool {
        !self.quiet.0 && std::ptr::addr_eq(Arc::as_ptr(&self.analysis.policy), Arc::as_ptr(policy))
    }

    /// Makes this analysis a trial (`quiet`, P3 PR-14d) or an ordinary one: [`Self::quiet_denials`],
    /// and the policy every read check of the analysis asks (`analysis.policy`, Bind's and the
    /// session's alike): `policy` itself, or for a trial `policy` behind
    /// [`crate::access::QuietPolicy`], which decides as `policy` does but is never told of a refusal
    /// (T063 review R2-1), so the provider counts and audits nothing for a trial either.
    pub(crate) fn set_quiet_denials(
        &mut self,
        quiet: bool,
        policy: &Arc<dyn krishiv_plan::analyzer::AccessPolicy>,
    ) {
        self.quiet = QuietDenials(quiet);
        self.analysis.policy = if quiet {
            Arc::new(crate::access::QuietPolicy::new(Arc::clone(policy)))
        } else {
            Arc::clone(policy)
        };
        let udtfs = Arc::new(self.udtfs.with_policy(&self.analysis.policy));
        self.analysis.table_functions = Arc::clone(&udtfs) as _;
        self.udtfs = udtfs;
    }
}

impl QueryEnv {
    /// Whether every catalog in `epochs` (an analysis's [`ContextFingerprint::catalog_epochs`],
    /// the catalogs it consulted) is still visible to this query and at the epoch it was read at.
    /// `catalog_epoch` covers only the session's own catalog; a declared catalog's table changes
    /// with that catalog's epoch, and a catalog no longer visible is absent here.
    ///
    /// [`ContextFingerprint::catalog_epochs`]: krishiv_plan::analyzer::ContextFingerprint::catalog_epochs
    pub(crate) fn catalogs_at(&self, epochs: &BTreeMap<String, u64>) -> bool {
        epochs.iter().all(|(name, &epoch)| {
            self.analysis
                .catalogs
                .catalog(name)
                .is_some_and(|c| c.epoch() == epoch)
        })
    }
}

/// A batch query's result rows: insert-only, so a weight other than 1 is a
/// broken invariant, never dropped silently.
pub(super) fn into_record_batch(batch: Batch) -> Result<RecordBatch, KrishivError> {
    match batch.into_parts() {
        (data, Weights::Const(1)) => Ok(data),
        _ => Err(internal("a batch query produced weighted rows".to_owned())),
    }
}

pub(crate) fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// The `path` option values of `paths`, typed (a URL's credentials stay
/// secret).
pub(crate) fn typed_paths<S: AsRef<str>>(paths: impl IntoIterator<Item = S>) -> Vec<OptionValue> {
    paths
        .into_iter()
        .map(|p| OptionValue::from_user("path", p.as_ref()))
        .collect()
}

/// A read by path of `format`.
pub(crate) fn read_files(
    format: &str,
    paths: Vec<OptionValue>,
    options: BTreeMap<String, OptionValue>,
) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Read(ReadSource::files(
        format.to_owned(),
        paths,
        options,
    )))
}

impl Session {
    /// What a query's plans are instantiated with in this process: the
    /// query's tables, its UDFs, and its spill scope under the session's
    /// spill directory (created at instantiation). Writes build their sink
    /// from the plan with the query's I/O context, formats and configuration;
    /// batch splits resolve through the session's relation store.
    pub(super) fn worker(&self, env: &QueryEnv) -> WorkerContext {
        let worker = WorkerContext::new(
            Arc::clone(&env.catalog) as Arc<dyn krishiv_exec::planner::TableProviderResolver>
        )
        .with_udfs(Arc::clone(&self.state.udfs))
        .with_relations(Arc::clone(&self.state.relations))
        .with_spill_scopes(Arc::new(QuerySpill {
            state: Arc::clone(&self.state),
            failure: Arc::clone(&env.spill_failure),
        }))
        .with_io(env.io.clone())
        .with_formats(Arc::clone(&self.state.runtime.formats))
        .with_config(Arc::clone(&env.config));
        self.with_python_runtime(worker)
    }
}

#[cfg(test)]
mod quiet_tests {
    use std::sync::{Arc, Mutex};

    use futures::executor::block_on;
    use krishiv_common::error::{ErrorClass, KrishivError};
    use krishiv_common::secret::OptionValue;
    use krishiv_plan::analyzer::{
        AccessDecision, AccessPolicy, BoxFuture, PolicyVersion, Principal, Privilege, Securable,
    };
    use krishiv_plan::{LogicalPlan, PlanNode, ReadSource};

    /// Denies `USAGE` on every secret scope; records each refusal it is told of.
    #[derive(Default)]
    struct NoSecrets(Mutex<Vec<String>>);

    impl AccessPolicy for NoSecrets {
        fn version(&self) -> PolicyVersion {
            PolicyVersion(1)
        }

        fn check<'a>(
            &'a self,
            _: &'a Principal,
            securable: &'a Securable,
            privilege: Privilege,
        ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
            let d =
                if matches!(securable, Securable::Secret { .. }) && privilege == Privilege::Usage {
                    AccessDecision::Deny {
                        reason: "private".to_owned(),
                    }
                } else {
                    AccessDecision::Allow
                };
            Box::pin(std::future::ready(Ok(d)))
        }

        fn denied(&self, _: &Principal, securable: &Securable, privilege: Privilege) {
            self.0
                .lock()
                .unwrap()
                .push(format!("{privilege} {securable}"));
        }
    }

    /// Review Q3: a trial analysis refused by a secret's `USAGE` (asked by Bind, through the
    /// analysis' policy) tells the policy nothing; the ordinary analysis of the same frame does.
    #[test]
    fn a_trials_secret_usage_refusal_is_not_told() {
        let policy = Arc::new(NoSecrets::default());
        let s = crate::Session::builder()
            .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
            .build()
            .unwrap();
        let read = LogicalPlan::new(PlanNode::Read(ReadSource::files(
            "parquet",
            vec![OptionValue::from_user("path", "/data/x")],
            [("token".to_owned(), OptionValue::reference("prod", "key"))]
                .into_iter()
                .collect(),
        )));
        let df = crate::DataFrame::new(s, Arc::new(read));
        let denied = |r: Result<_, KrishivError>| matches!(r, Err(e) if e.error_class() == ErrorClass::KrishivInsufficientPermissions);
        assert!(denied(block_on(df.trial_analysis())));
        assert!(policy.0.lock().unwrap().is_empty());
        assert!(denied(block_on(df.analysis())));
        assert_eq!(policy.0.lock().unwrap().len(), 1);
    }

    /// T063 review R2-1 Q2: a trial analysis that passes is kept as the frame's ordinary one, which
    /// asks the session's own policy again (not the trial's `QuietPolicy`), so a later check made
    /// with it is told to the policy.
    #[test]
    fn a_kept_trial_analysis_asks_the_sessions_policy() {
        let s = crate::Session::new().unwrap();
        let policy = &s.state.policy;
        // A fresh env is ordinary and asks the session's policy; made a trial, it is quiet and asks a
        // `QuietPolicy` instead.
        let mut env = s.query_env();
        assert!(env.asks_policy(policy) && !env.quiet_denials());
        env.set_quiet_denials(true, policy);
        assert!(env.quiet_denials() && !env.asks_policy(policy));
        // A frame never analyzed, analyzed as a trial, keeps an ordinary analysis.
        let df = s.range(0, 3, 1, None);
        let kept = block_on(df.trial_analysis()).unwrap();
        assert!(kept.env.asks_policy(policy));
        assert!(block_on(df.analysis()).unwrap().env.asks_policy(policy));
    }
}
