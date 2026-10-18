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
//!    are rejected until their barrier semantics land (P5-T045).
//! 2. **Resolve** (sync, pure): one bottom-up pass over the relations,
//!    with explicit stacks; each operator is resolved once after its
//!    children and yields its output scope. Names resolve in Spark 4.2's
//!    order: the operator's input (qualifier = the longest matching suffix
//!    of `catalog.namespace.table` or the alias), then select-list aliases
//!    where Spark allows them (`GROUP BY`), then enclosing queries; a
//!    DataFrame `df["a"]` resolves among the relations carrying its
//!    `plan_id`. Rewrites run inside the pass in a fixed order:
//!    `WithColumns` to `Project`, `USING` expansion, aggregates and grouping
//!    columns that `HAVING` / `ORDER BY` use but the aggregate lacks, a
//!    `SELECT` with aggregates to an aggregate.
//!
//! Lateral column aliases, `GROUP BY` / `ORDER BY` ordinals and `ORDER BY`
//! columns outside the select list come in a follow-up unit of P2-T071a;
//! until then they fail with `KRISHIV_UNSUPPORTED_FEATURE`, never silently.
//!
//! Resolution reads only the [`AnalysisContext`], whose configuration is a
//! [`ResolutionConf`]: an input it does not hold cannot change the result,
//! so `(input key, context fingerprint, pins)` is a sound cache key.
//! Results are memoised per (node, scope environment), so a subtree shared
//! through `Arc` is resolved once; where it meets itself (a self-join) the
//! second instance is a [`crate::PlanNode::Shift`] of the first.

mod bind;
mod catalog;
mod exprs;
mod ops;
mod resolve;
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
    FunctionEntry, FunctionLookup, FunctionRegistry, MemoryCatalog, PolicyVersion, Principal,
    PrincipalKind, Privilege, RowColumnPolicy, Securable, SqlFrontend, TableColumn, TableIdent,
    TableKind, TableMeta,
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
        }
    }
}

impl ResolutionConf {
    /// The session configs [`Self::from_conf`] reads: the only ones.
    pub const KEYS: [&'static str; 6] = [
        "spark.sql.caseSensitive",
        "spark.sql.ansi.enabled",
        "spark.sql.groupByAliases",
        "spark.sql.groupByOrdinal",
        "spark.sql.orderByOrdinal",
        "spark.sql.lateralColumnAlias.enableImplicitResolution",
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
        ];
        let mut v = [false; 6];
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
}

/// The context inputs of a resolution: with the input key and the pins,
/// the key of a resolved-plan cache.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContextFingerprint {
    /// The Spark profile.
    pub profile: SparkProfile,
    /// The resolution flags.
    pub conf: ResolutionConf,
    /// The epoch of every catalog consulted.
    pub catalog_epochs: BTreeMap<String, u64>,
    /// [`Principal::id`].
    pub principal: String,
    /// The access-policy version.
    pub policy: PolicyVersion,
}

impl ContextFingerprint {
    /// A 128-bit digest of the fingerprint (XXH3-128 over a fixed layout).
    #[must_use]
    pub fn digest(&self) -> u128 {
        let mut b = format!("{:?}|{:?}|", self.profile, self.conf).into_bytes();
        for (name, epoch) in &self.catalog_epochs {
            b.extend_from_slice(&(name.len() as u64).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&epoch.to_le_bytes());
        }
        b.extend_from_slice(&(self.principal.len() as u64).to_le_bytes());
        b.extend_from_slice(self.principal.as_bytes());
        b.extend_from_slice(&self.policy.0.to_le_bytes());
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Stage timings.
    pub timings: AnalysisTimings,
}

/// An analysis error: a Spark error class and where in the plan it arose.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{error}")]
pub struct AnalysisError {
    /// The error.
    pub error: KrishivError,
    /// The node it refers to (a SQL span, a DataFrame `plan_id`).
    pub origin: Origin,
}

/// Spark's SQL query context of an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryContext {
    /// First byte.
    pub start: usize,
    /// Last byte (inclusive, as Spark reports it).
    pub stop: usize,
    /// The redacted text of the span.
    pub fragment: String,
}

impl AnalysisError {
    pub(crate) fn new(error: impl Into<KrishivError>, origin: Origin) -> Self {
        Self {
            error: error.into(),
            origin,
        }
    }

    /// The query context in `text`, the redacted query the error's spans
    /// index; `None` when the error has no SQL span.
    #[must_use]
    pub fn query_context(&self, text: &RedactedSql) -> Option<QueryContext> {
        let (_, span) = self.origin.sql?;
        let (start, end) = (span.start as usize, span.end as usize);
        Some(QueryContext {
            start,
            stop: end.saturating_sub(1),
            fragment: text.fragment(start..end).to_owned(),
        })
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
                ..
            } = bindings;
            Ok(AnalyzedPlan {
                plan: resolved,
                output,
                pins,
                securables,
                input_key: plan.structural_hash(),
                context: ContextFingerprint {
                    profile: ctx.profile,
                    conf: ctx.conf,
                    catalog_epochs: epochs,
                    principal: ctx.principal.id(),
                    policy: ctx.policy.version(),
                },
                query_text,
                timings: AnalysisTimings {
                    metadata,
                    resolve: resolving.elapsed(),
                },
            })
        })
    }
}
