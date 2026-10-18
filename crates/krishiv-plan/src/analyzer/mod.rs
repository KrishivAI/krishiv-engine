// SPDX-License-Identifier: Apache-2.0
//! The analyzer: an unresolved [`Plan`] in, a resolved, deterministic,
//! cache-ready [`AnalyzedPlan`] out (ADR 0003 #6, ADR 0036 R2, ADR 0037 §4).
//!
//! Two stages, no fixed point:
//!
//! 1. **Bind** (async): one walk over the plan expands `PlanNode::Sql`
//!    through the [`SqlFrontend`], checks that every function exists
//!    (Spark checks functions before anything else), loads each distinct
//!    table once from its [`Catalog`] and pins its version for the whole
//!    query, and asks the [`AccessPolicy`] once per securable. Denials and
//!    policy errors fail the query (`KRISHIV_INSUFFICIENT_PERMISSIONS`)
//!    before the resolver sees any metadata; row filters and column masks
//!    are rejected until their barrier semantics land (P5-T045). Reads
//!    by path (`spark.read.parquet(path)`, `` parquet.`path` `` when no
//!    catalog has the name) are listed once through the
//!    [`FileSchemaResolver`], after their paths are authorized; Spark's
//!    `ResolveSQLOnFile` rules decide which SQL names read files.
//! 2. **Resolve** (sync, pure): one bottom-up pass over the relations,
//!    with explicit stacks; each operator is resolved once after its
//!    children and yields its output scope. Names resolve in Spark 4.2's
//!    order: the operator's input (qualifier = the longest matching suffix
//!    of `catalog.namespace.table` or the alias), then aliases defined
//!    earlier in the same select list (lateral column aliases), then
//!    select-list aliases where Spark allows them (`GROUP BY`), then
//!    enclosing queries; a DataFrame `df["a"]` resolves among the
//!    relations carrying its `plan_id`. Rewrites run inside the pass in a
//!    fixed order: `WithColumns` to `Project`, `USING` expansion,
//!    `GROUP BY` / `ORDER BY` ordinals to the items they number, aggregates
//!    and grouping columns that `HAVING` / `ORDER BY` use but the
//!    aggregate lacks, a `SELECT` with aggregates to an aggregate, lateral
//!    aliases to projections below the items reading them (above the
//!    aggregate, for an aggregate's select list), and columns a filter or
//!    sort names that a projection below dropped added back to it (Spark's
//!    missing references) and dropped again above.
//!
//! Expressions are typed as Spark types them under ANSI as they are bound
//! (`AnsiTypeCoercion`, `DecimalPrecision`): the canonical operators by
//! Spark's operator rules, registry functions by their declared
//! [`Inputs`]. Implicit casts are plain [`crate::ExprNode::Cast`] nodes in
//! the pinned mode; names and error texts show expressions as written.
//! `+` / `-` on dates and timestamps, and operators' string coercions with
//! ANSI off, fail with `KRISHIV_UNSUPPORTED_FEATURE`.
//!
//! Resolution reads only the [`AnalysisContext`], whose configuration is a
//! [`ResolutionConf`]: an input it does not hold cannot change the result,
//! so `(input key, context fingerprint, pins)` is a sound cache key.
//! Resolution does not depend on how the input shares subtrees: relations
//! equal field by field (`plan_id`s included) are resolved as one, so equal
//! input keys give equal resolved plans. Results are memoised per
//! (relation, scope environment), and a relation without outer references
//! per (relation, CTE scope): a plan reached both as a child and inside a
//! subquery is resolved once. Where a relation meets itself (a self-join,
//! `df.where(df.exists())`) the second instance is a
//! [`crate::PlanNode::Shift`] of the first, never a copy.
//!
//! Bind asks the policy about a table under the name as located before
//! reading any metadata, so a denied principal gets
//! `KRISHIV_INSUFFICIENT_PERMISSIONS` whether the table exists or not (no
//! existence oracle), then again under the catalog's own identifier.
//! Catalog failures surface as `KRISHIV_CATALOG_FAILURE`, without the
//! provider's text. Catalog functions are not policy-checked yet (no
//! catalog functions exist before P3).

mod bind;
mod catalog;
mod coerce;
mod exprs;
mod ops;
mod resolve;
mod subquery;
#[cfg(any(test, feature = "testing"))]
pub mod testing;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use krishiv_common::error::KrishivError;
use krishiv_common::redact::RedactedSql;
use twox_hash::XxHash3_128;

pub use catalog::{
    AccessDecision, AccessPolicy, AllowAll, ArgType, BoxFuture, Catalog, CatalogResolver,
    FileSchemaResolver, FunctionEntry, FunctionLookup, FunctionRegistry, InputType, Inputs,
    MemoryCatalog, PolicyVersion, Principal, PrincipalKind, Privilege, ResultType, RowColumnPolicy,
    Securable, SourceKind, SqlFrontend, TableColumn, TableIdent, TableKind, TableMeta,
};

use crate::{BoundColumn, Origin, Plan, StructuralHash, TableVersion};

/// The Spark version whose behaviour a session follows (ADR 0003, 0008).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[non_exhaustive]
pub enum SparkProfile {
    /// Spark 3.5.
    V3_5,
    /// Spark 4.0.
    V4_0,
    /// Spark 4.1.
    V4_1,
    /// Spark 4.2 (the default).
    #[default]
    V4_2,
}

/// The configuration resolution may read, and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per Spark config, read once"
)]
pub struct ResolutionConf {
    /// `spark.sql.caseSensitive`.
    pub case_sensitive: bool,
    /// `spark.sql.ansi.enabled`: pins `CastMode::Default`.
    pub ansi: bool,
    /// `spark.sql.groupByAliases`.
    pub group_by_aliases: bool,
    /// `spark.sql.groupByOrdinal`.
    pub group_by_ordinal: bool,
    /// `spark.sql.orderByOrdinal`.
    pub order_by_ordinal: bool,
    /// `spark.sql.lateralColumnAlias.enableImplicitResolution`.
    pub lateral_column_alias: bool,
    /// `spark.sql.decimalOperations.allowPrecisionLoss`: decimal
    /// arithmetic result types.
    pub decimal_precision_loss: bool,
    /// `spark.sql.runSQLOnFiles`: `` format.`path` `` reads files.
    pub run_sql_on_files: bool,
}

impl Default for ResolutionConf {
    /// Spark 4.2's defaults.
    fn default() -> Self {
        Self {
            case_sensitive: false,
            ansi: true,
            group_by_aliases: true,
            group_by_ordinal: true,
            order_by_ordinal: true,
            lateral_column_alias: true,
            decimal_precision_loss: true,
            run_sql_on_files: true,
        }
    }
}

impl ResolutionConf {
    /// The session configs [`Self::from_conf`] reads: the only ones.
    pub const KEYS: [&'static str; 8] = [
        "spark.sql.caseSensitive",
        "spark.sql.ansi.enabled",
        "spark.sql.groupByAliases",
        "spark.sql.groupByOrdinal",
        "spark.sql.orderByOrdinal",
        "spark.sql.lateralColumnAlias.enableImplicitResolution",
        "spark.sql.decimalOperations.allowPrecisionLoss",
        "spark.sql.runSQLOnFiles",
    ];

    /// The flags from session configs; `conf` returns a config's value if
    /// set. Unset configs take Spark 4.2's defaults.
    pub fn from_conf<'a>(conf: impl Fn(&str) -> Option<&'a str>) -> Self {
        let d = Self::default();
        let defaults = [
            d.case_sensitive,
            d.ansi,
            d.group_by_aliases,
            d.group_by_ordinal,
            d.order_by_ordinal,
            d.lateral_column_alias,
            d.decimal_precision_loss,
            d.run_sql_on_files,
        ];
        let mut v = [false; 8];
        for ((slot, key), default) in v.iter_mut().zip(Self::KEYS).zip(defaults) {
            *slot = conf(key).map_or(default, |s| s.trim().eq_ignore_ascii_case("true"));
        }
        Self {
            case_sensitive: v[0],
            ansi: v[1],
            group_by_aliases: v[2],
            group_by_ordinal: v[3],
            order_by_ordinal: v[4],
            lateral_column_alias: v[5],
            decimal_precision_loss: v[6],
            run_sql_on_files: v[7],
        }
    }
}

/// Everything resolution may read.
#[derive(Clone)]
pub struct AnalysisContext {
    /// The Spark version semantics follow.
    pub profile: SparkProfile,
    /// The resolution flags.
    pub conf: ResolutionConf,
    /// Who runs the query.
    pub principal: Arc<Principal>,
    /// The access policy.
    pub policy: Arc<dyn AccessPolicy>,
    /// The catalogs.
    pub catalogs: Arc<dyn CatalogResolver>,
    /// The function registry.
    pub functions: Arc<dyn FunctionRegistry>,
    /// The SQL front end, for SQL text inside plans.
    pub sql: Option<Arc<dyn SqlFrontend>>,
    /// Files read by path; without it such reads are
    /// `KRISHIV_UNSUPPORTED_FEATURE`.
    pub files: Option<Arc<dyn FileSchemaResolver>>,
}

/// The context inputs of a resolution: with the input key and the pins,
/// the key of a resolved-plan cache. Every input resolution reads through
/// the [`AnalysisContext`] is here (a test varies each one).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContextFingerprint {
    /// The Spark profile.
    pub profile: SparkProfile,
    /// The resolution flags.
    pub conf: ResolutionConf,
    /// The current catalog and namespace.
    pub current: Vec<String>,
    /// [`CatalogResolver::epoch`].
    pub catalogs: u64,
    /// The epoch of every catalog consulted.
    pub catalog_epochs: BTreeMap<String, u64>,
    /// [`FunctionRegistry::version`].
    pub functions: u64,
    /// [`SqlFrontend::fingerprint`], when there is a front end.
    pub sql: Option<u64>,
    /// [`Principal::id`].
    pub principal: String,
    /// The access-policy version.
    pub policy: PolicyVersion,
    /// Whether files can be read by path (a [`FileSchemaResolver`]).
    pub files: bool,
}

impl ContextFingerprint {
    /// A 128-bit digest of the fingerprint: XXH3-128 over an explicit,
    /// length-prefixed layout of every field.
    #[must_use]
    pub fn digest(&self) -> u128 {
        let mut b = Vec::new();
        let num = |b: &mut Vec<u8>, v: u64| b.extend_from_slice(&v.to_le_bytes());
        let text = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        let profile = match self.profile {
            SparkProfile::V3_5 => 35,
            SparkProfile::V4_0 => 40,
            SparkProfile::V4_1 => 41,
            SparkProfile::V4_2 => 42,
        };
        num(&mut b, profile);
        let c = self.conf;
        let flags = [
            c.case_sensitive,
            c.ansi,
            c.group_by_aliases,
            c.group_by_ordinal,
            c.order_by_ordinal,
            c.lateral_column_alias,
            c.decimal_precision_loss,
            c.run_sql_on_files,
            self.files,
        ];
        num(
            &mut b,
            flags.iter().fold(0, |acc, f| acc << 1 | u64::from(*f)),
        );
        num(&mut b, self.current.len() as u64);
        self.current.iter().for_each(|p| text(&mut b, p));
        num(&mut b, self.catalogs);
        num(&mut b, self.catalog_epochs.len() as u64);
        for (name, epoch) in &self.catalog_epochs {
            text(&mut b, name);
            num(&mut b, *epoch);
        }
        num(&mut b, self.functions);
        match self.sql {
            None => num(&mut b, 0),
            Some(v) => {
                num(&mut b, 1);
                num(&mut b, v);
            }
        }
        text(&mut b, &self.principal);
        num(&mut b, self.policy.0);
        XxHash3_128::oneshot(&b)
    }
}

/// A securable a query reads, and the columns it references.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecurableUse {
    /// The securable.
    pub securable: Securable,
    /// The privilege checked.
    pub privilege: Privilege,
    /// The columns referenced (catalog spelling).
    pub columns: BTreeSet<String>,
}

/// Where analysis time went (ADR 0037 §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AnalysisTimings {
    /// Bind: metadata loads and policy checks.
    pub metadata: Duration,
    /// Resolve.
    pub resolve: Duration,
}

/// A resolved plan and what a cache, the policy layer and EXPLAIN need.
/// Equality ignores [`Self::timings`].
#[derive(Debug, Clone)]
pub struct AnalyzedPlan {
    /// The resolved plan.
    pub plan: Plan,
    /// Its output columns.
    pub output: Vec<BoundColumn>,
    /// The version pinned per table.
    pub pins: BTreeMap<TableIdent, TableVersion>,
    /// The securables read, in first-use order.
    pub securables: Vec<SecurableUse>,
    /// The structural hash of the unresolved input.
    pub input_key: StructuralHash,
    /// The context inputs.
    pub context: ContextFingerprint,
    /// The query text, redacted, when the plan was SQL text.
    pub query_text: Option<RedactedSql>,
    /// Whether the plan reads files by path: it then depends on the
    /// files as listed, which [`Self::context`] does not capture.
    pub reads_files: bool,
    /// Stage timings.
    pub timings: AnalysisTimings,
}

impl PartialEq for AnalyzedPlan {
    fn eq(&self, other: &Self) -> bool {
        self.plan == other.plan
            && self.output == other.output
            && self.pins == other.pins
            && self.securables == other.securables
            && self.input_key == other.input_key
            && self.context == other.context
            && self.query_text == other.query_text
            && self.reads_files == other.reads_files
    }
}

impl Eq for AnalyzedPlan {}

/// An analysis error: a Spark error class and where in the plan it arose.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{error}")]
pub struct AnalysisError {
    /// The error.
    pub error: KrishivError,
    /// The node it refers to (a SQL span, a DataFrame `plan_id`).
    pub origin: Origin,
}

/// Spark's query context of an error.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueryContext {
    /// A span of SQL text.
    Sql {
        /// First byte.
        start: usize,
        /// Last byte (inclusive, as Spark reports it).
        stop: usize,
        /// The redacted text of the span.
        fragment: String,
    },
    /// A DataFrame operation, by the `plan_id` of the relation it built
    /// (the front end maps it to its call site).
    DataFrame {
        /// The relation's `plan_id`.
        plan_id: crate::PlanId,
    },
}

impl AnalysisError {
    pub(crate) fn new(error: impl Into<KrishivError>, origin: Origin) -> Self {
        Self {
            error: error.into(),
            origin,
        }
    }

    /// The query context: a span of `text` (the redacted query the
    /// error's spans index), else the DataFrame that raised it; `None` when
    /// the error has neither.
    #[must_use]
    pub fn query_context(&self, text: &RedactedSql) -> Option<QueryContext> {
        if let Some((_, span)) = self.origin.sql {
            let (start, end) = (span.start as usize, span.end as usize);
            return Some(QueryContext::Sql {
                start,
                stop: end.saturating_sub(1),
                fragment: text.fragment(start..end).to_owned(),
            });
        }
        self.origin
            .plan_id
            .map(|plan_id| QueryContext::DataFrame { plan_id })
    }
}

impl From<AnalysisError> for KrishivError {
    fn from(e: AnalysisError) -> Self {
        e.error
    }
}

impl fmt::Debug for AnalysisContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnalysisContext")
            .field("profile", &self.profile)
            .field("conf", &self.conf)
            .field("principal", &self.principal)
            .finish_non_exhaustive()
    }
}

/// The fingerprint of `ctx`, with the epochs of the catalogs consulted.
pub(crate) fn fingerprint(
    ctx: &AnalysisContext,
    catalog_epochs: BTreeMap<String, u64>,
) -> ContextFingerprint {
    let mut current = vec![ctx.catalogs.current_catalog().to_owned()];
    current.extend(ctx.catalogs.current_namespace().iter().cloned());
    ContextFingerprint {
        profile: ctx.profile,
        conf: ctx.conf,
        current,
        catalogs: ctx.catalogs.epoch(),
        catalog_epochs,
        functions: ctx.functions.version(),
        sql: ctx.sql.as_ref().map(|f| f.fingerprint()),
        principal: ctx.principal.id(),
        policy: ctx.policy.version(),
        files: ctx.files.is_some(),
    }
}

/// Resolves plans.
pub trait Analyzer: Send + Sync {
    /// Resolves `plan` in `ctx`.
    fn analyze<'a>(
        &'a self,
        plan: &'a Plan,
        ctx: &'a AnalysisContext,
    ) -> BoxFuture<'a, Result<AnalyzedPlan, AnalysisError>>;
}

/// The analyzer of this module.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultAnalyzer;

impl Analyzer for DefaultAnalyzer {
    fn analyze<'a>(
        &'a self,
        plan: &'a Plan,
        ctx: &'a AnalysisContext,
    ) -> BoxFuture<'a, Result<AnalyzedPlan, AnalysisError>> {
        Box::pin(async move {
            let started = std::time::Instant::now();
            let bindings = bind::bind(plan, ctx).await?;
            let metadata = started.elapsed();
            let resolving = std::time::Instant::now();
            let (root, output, securables) = resolve::resolve(plan, ctx, &bindings)?;
            let resolved =
                Plan::try_new(root).map_err(|e| AnalysisError::new(e, plan.root().origin))?;
            let bind::Bindings {
                pins,
                epochs,
                query_text,
                reads_files,
                ..
            } = bindings;
            Ok(AnalyzedPlan {
                plan: resolved,
                output,
                pins,
                securables,
                input_key: plan.structural_hash(),
                context: fingerprint(ctx, epochs),
                query_text,
                reads_files,
                timings: AnalysisTimings {
                    metadata,
                    resolve: resolving.elapsed(),
                },
            })
        })
    }
}
