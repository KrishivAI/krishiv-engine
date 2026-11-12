// SPDX-License-Identifier: Apache-2.0
//! The session's per-query environment, parser options and statement-context helpers.

use super::*;

/// The parser's configs (`ParserOptions::from_conf` reads these).
pub(super) const PARSER_KEYS: [&str; 15] = [
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

/// The parser options `config` sets.
pub(super) fn parser_options(config: &Config) -> ParserOptions {
    let values = conf_values(config, &PARSER_KEYS);
    ParserOptions::from_conf(|k| values.get(k).map(String::as_str))
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
    LogicalPlan::new(PlanNode::Read(ReadSource::Files {
        format: format.to_owned(),
        paths,
        options,
    }))
}
