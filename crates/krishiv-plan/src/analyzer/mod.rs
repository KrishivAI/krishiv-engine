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
//! A call of a SQL-bodied function (`FunctionLookup::Sql`, the session's
//! `CREATE TEMPORARY FUNCTION`) is replaced while expressions are bound by
//! the function's body, resolved in its own scope with the arguments
//! standing for its parameters (`sqlfn`); Bind also binds what the bodies
//! read. The resolved plan holds no call of such a function.
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

mod access;
mod analytics;
mod bind;
mod builtins;
mod catalog;
mod coerce;
mod exprs;
mod fold;
mod fold_ops;
mod generate;
mod hints;
mod hof;
pub(crate) mod legacy;
mod ops;
mod pivot;
mod resolve;
mod setops;
mod sqlfn;
mod subquery;
mod suggest;
pub mod table_functions;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
mod unexecutable;
mod unpivot;
mod variables;
mod windows;

use krishiv_kernels::datetime::{TimeZoneRules, ZoneRef, resolve_zone};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use krishiv_common::error::KrishivError;
use krishiv_common::redact::RedactedSql;
use twox_hash::XxHash3_128;

pub use access::{authorize, insufficient_permissions, namespace_parts, table_ident};
pub use builtins::BuiltinFunctions;
pub use catalog::{
    AccessDecision, AccessPolicy, AllowAll, ArgType, Arity, BoxFuture, BuiltinTableFunctions,
    CallTypes, Catalog, CatalogResolver, Constant, DenyAll, FileSchemaResolver, FunctionEntry,
    FunctionLookup, FunctionRegistry, InputBuilder, InputType, Inputs, MemoryCatalog, NameForm,
    Operator, PolicyVersion, Principal, PrincipalKind, Privilege, ResultType, RowColumnPolicy,
    Securable, SourceKind, SqlFrontend, TableColumn, TableFunctionKind, TableFunctionLookup,
    TableFunctionRegistry, TableIdent, TableKind, TableMeta,
};
pub use sqlfn::{SqlFunction, SqlParam};
pub use variables::{LocalFrame, SessionVariable, VariableSnapshot, variable_name};

use crate::{BoundColumn, Origin, Plan, StructuralHash, TableVersion};

pub use krishiv_functions::signature::SparkProfile;

/// The session time zone, resolved once per analysis: a fixed offset, or a
/// region of the bundled time-zone database. `Copy` so the conf stays
/// `Copy`; regions are interned by lower-cased id (the database matches ids
/// case-insensitively), and only ids it resolves are kept, so the interner
/// holds at most one entry per zone id the database knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionZone {
    /// A fixed offset in seconds east of UTC.
    Fixed(i32),
    /// A region id, interned.
    Region(&'static RegionZone),
}

/// An interned region zone; equal when the ids are.
#[derive(Debug)]
pub struct RegionZone {
    id: String,
    zone: ZoneRef,
}

impl PartialEq for RegionZone {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RegionZone {}

impl std::hash::Hash for RegionZone {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl SessionZone {
    /// The zone `text` names; `None` when it is not a zone id.
    #[must_use]
    pub fn resolve(text: &str) -> Option<Self> {
        use std::collections::HashMap;
        use std::sync::{Mutex, OnceLock};
        static REGIONS: OnceLock<Mutex<HashMap<String, &'static RegionZone>>> = OnceLock::new();
        match resolve_zone(text).ok()? {
            ZoneRef::Fixed(fixed) => Some(Self::Fixed(fixed.0)),
            zone @ ZoneRef::Named(_) => {
                let mut regions = REGIONS
                    .get_or_init(Mutex::default)
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let interned = regions.entry(text.to_ascii_lowercase()).or_insert_with(|| {
                    Box::leak(Box::new(RegionZone {
                        id: text.to_ascii_lowercase(),
                        zone,
                    }))
                });
                Some(Self::Region(interned))
            }
        }
    }
}

impl SessionZone {
    /// The offset in seconds when the zone is a fixed one (the constant
    /// folder has no tz database, so it reads no region zone).
    #[must_use]
    pub fn fixed(self) -> Option<i32> {
        match self {
            Self::Fixed(off) => Some(off),
            Self::Region(_) => None,
        }
    }

    /// `us` (microseconds since the epoch) shifted to wall-clock time in
    /// `zone`; UTC when there is none.
    #[must_use]
    pub fn local_micros(zone: Option<Self>, us: i64) -> i64 {
        let off = zone.map_or(0, |z| z.offset_at_utc(us.div_euclid(1_000_000)));
        us.saturating_add(i64::from(off).saturating_mul(1_000_000))
    }
}

impl TimeZoneRules for SessionZone {
    fn offset_at_utc(&self, utc_seconds: i64) -> i32 {
        match self {
            Self::Fixed(off) => *off,
            Self::Region(r) => r.zone.offset_at_utc(utc_seconds),
        }
    }

    fn offset_for_local(&self, local_seconds: i64) -> i32 {
        match self {
            Self::Fixed(off) => *off,
            Self::Region(r) => r.zone.offset_for_local(local_seconds),
        }
    }
}

/// How nested `WITH` definitions with the same name resolve
/// (`spark.sql.legacy.ctePrecedencePolicy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CtePrecedencePolicy {
    /// Inner definitions shadow outer ones (Spark 4.2's default).
    Corrected,
    /// Outer definitions take precedence (Spark's legacy inlining).
    Legacy,
    /// A nested shadowing definition is `AMBIGUOUS_ALIAS_IN_NESTED_CTE`.
    Exception,
}

impl Default for CtePrecedencePolicy {
    /// Spark 4.2's default.
    fn default() -> Self {
        Self::Corrected
    }
}

impl CtePrecedencePolicy {
    /// The policy `value` names (case-insensitive, trimmed). The valid
    /// values are the config registry's
    /// `spark.sql.legacy.ctePrecedencePolicy` enum (`EXCEPTION`, `LEGACY`,
    /// `CORRECTED`): that entry is the single validation source, so an
    /// unknown value is `None` rather than a silent default. Callers holding
    /// an already-validated config map `None` to the default.
    #[must_use]
    pub fn from_value(value: &str) -> Option<Self> {
        if value.trim().eq_ignore_ascii_case("LEGACY") {
            Some(Self::Legacy)
        } else if value.trim().eq_ignore_ascii_case("EXCEPTION") {
            Some(Self::Exception)
        } else if value.trim().eq_ignore_ascii_case("CORRECTED") {
            Some(Self::Corrected)
        } else {
            None
        }
    }
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
    /// `spark.sql.mapKeyDedupPolicy = LAST_WIN`: duplicate map keys keep
    /// the value inserted last (folded map calls depend on it).
    pub map_key_dedup_last_win: bool,
    /// `spark.sql.legacy.javaCharsets`: `encode` and `decode` take Java's
    /// names.
    pub legacy_java_charsets: bool,
    /// `spark.sql.legacy.codingErrorAction`: `encode` and `decode` replace
    /// what they cannot map.
    pub legacy_coding_error_action: bool,
    /// `spark.sql.runSQLOnFiles`: `` format.`path` `` reads files.
    pub run_sql_on_files: bool,
    /// `spark.sql.legacy.percentileDiscCalculation`: `percentile_disc`
    /// returns the continuous lower bound instead of the discrete value.
    pub legacy_percentile_disc: bool,
    /// `spark.sql.timestampType = TIMESTAMP_NTZ`: plain `TIMESTAMP '...'`
    /// without a zone is a `TIMESTAMP_NTZ`.
    pub timestamp_ntz: bool,
    /// `spark.sql.session.timeZone` resolved against the bundled time-zone
    /// database (a fixed offset or a region id); `None` for a malformed or
    /// unknown zone. Unset is `UTC`, the default the cast path uses, so the
    /// default session resolves timestamp literals.
    pub session_zone: Option<SessionZone>,
    /// `spark.sql.legacy.negativeIndexInArrayInsert`: a negative
    /// `array_insert` position counts one place further from the end.
    pub legacy_negative_index_in_array_insert: bool,
    /// `spark.sql.legacy.ctePrecedencePolicy`.
    pub cte_precedence: CtePrecedencePolicy,
    /// `spark.sql.legacy.allowParameterlessCount`: `count()` counts no
    /// rows (0) instead of failing with `WRONG_NUM_ARGS`.
    pub allow_parameterless_count: bool,
    /// `spark.sql.legacy.allowStarWithSingleTableIdentifierInCount`:
    /// `count(tbl.*)` expands to the table's columns instead of failing
    /// with `INVALID_USAGE_OF_STAR_WITH_TABLE_IDENTIFIER_IN_COUNT`.
    pub allow_star_with_single_table_identifier_in_count: bool,
    /// `spark.sql.legacy.allowUdfParameterToShadowParameterlessFunction`:
    /// a SQL function's parameter named like a parameterless function
    /// (`current_user`) is the parameter, not the function.
    pub udf_parameter_shadows_function: bool,
    /// `spark.sql.legacy.timeParserPolicy = LEGACY`: Spark's
    /// `SimpleDateFormat` parser, which Krishiv does not implement (the
    /// datetime functions with a pattern are refused).
    pub legacy_time_parser: bool,
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
            map_key_dedup_last_win: false,
            legacy_java_charsets: false,
            legacy_coding_error_action: false,
            run_sql_on_files: true,
            legacy_percentile_disc: false,
            legacy_negative_index_in_array_insert: false,
            timestamp_ntz: false,
            session_zone: Some(SessionZone::Fixed(0)),
            cte_precedence: CtePrecedencePolicy::default(),
            allow_parameterless_count: false,
            allow_star_with_single_table_identifier_in_count: false,
            udf_parameter_shadows_function: false,
            legacy_time_parser: false,
        }
    }
}

impl ResolutionConf {
    /// The session configs [`Self::from_conf`] reads: the only ones.
    pub const KEYS: [&'static str; 20] = [
        "spark.sql.caseSensitive",
        "spark.sql.ansi.enabled",
        "spark.sql.groupByAliases",
        "spark.sql.groupByOrdinal",
        "spark.sql.orderByOrdinal",
        "spark.sql.lateralColumnAlias.enableImplicitResolution",
        "spark.sql.decimalOperations.allowPrecisionLoss",
        "spark.sql.runSQLOnFiles",
        "spark.sql.legacy.percentileDiscCalculation",
        "spark.sql.mapKeyDedupPolicy",
        "spark.sql.timestampType",
        "spark.sql.session.timeZone",
        "spark.sql.legacy.ctePrecedencePolicy",
        "spark.sql.legacy.allowParameterlessCount",
        "spark.sql.legacy.allowStarWithSingleTableIdentifierInCount",
        "spark.sql.legacy.allowUdfParameterToShadowParameterlessFunction",
        "spark.sql.legacy.timeParserPolicy",
        "spark.sql.legacy.javaCharsets",
        "spark.sql.legacy.codingErrorAction",
        "spark.sql.legacy.negativeIndexInArrayInsert",
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
            d.legacy_percentile_disc,
        ];
        let mut v = [false; 9];
        for ((slot, key), default) in v.iter_mut().zip(Self::KEYS[..9].iter()).zip(defaults) {
            *slot = conf(key).map_or(default, |s| s.trim().eq_ignore_ascii_case("true"));
        }
        let on = |key: &str| conf(key).is_some_and(|s| s.trim().eq_ignore_ascii_case("true"));
        Self {
            case_sensitive: v[0],
            ansi: v[1],
            group_by_aliases: v[2],
            group_by_ordinal: v[3],
            order_by_ordinal: v[4],
            lateral_column_alias: v[5],
            decimal_precision_loss: v[6],
            run_sql_on_files: v[7],
            legacy_percentile_disc: v[8],
            map_key_dedup_last_win: conf(Self::KEYS[9])
                .is_some_and(|s| s.trim().eq_ignore_ascii_case("LAST_WIN")),
            timestamp_ntz: conf(Self::KEYS[10])
                .is_some_and(|s| s.trim().eq_ignore_ascii_case("TIMESTAMP_NTZ")),
            session_zone: match conf(Self::KEYS[11]) {
                None => Some(SessionZone::Fixed(0)),
                Some(s) => SessionZone::resolve(s),
            },
            cte_precedence: conf(Self::KEYS[12])
                .and_then(CtePrecedencePolicy::from_value)
                .unwrap_or_default(),
            allow_parameterless_count: on(Self::KEYS[13]),
            allow_star_with_single_table_identifier_in_count: on(Self::KEYS[14]),
            udf_parameter_shadows_function: conf(Self::KEYS[15])
                .map_or(d.udf_parameter_shadows_function, |s| {
                    s.trim().eq_ignore_ascii_case("true")
                }),
            legacy_time_parser: conf(Self::KEYS[16])
                .is_some_and(|s| s.trim().eq_ignore_ascii_case("LEGACY")),
            legacy_java_charsets: on(Self::KEYS[17]),
            legacy_coding_error_action: on(Self::KEYS[18]),
            legacy_negative_index_in_array_insert: on(Self::KEYS[19]),
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
    /// The table-function registry.
    pub table_functions: Arc<dyn TableFunctionRegistry>,
    /// The SQL front end, for SQL text inside plans.
    pub sql: Option<Arc<dyn SqlFrontend>>,
    /// Files read by path, for this query only (see
    /// [`FileSchemaResolver`]); without it such reads are
    /// `KRISHIV_UNSUPPORTED_FEATURE`.
    pub files: Option<Arc<dyn FileSchemaResolver>>,
    /// The session variables the query resolves, taken once per query
    /// (see [`VariableSnapshot`]).
    pub variables: Arc<VariableSnapshot>,
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
    /// [`TableFunctionRegistry::version`].
    pub table_functions: u64,
    /// [`SqlFrontend::fingerprint`], when there is a front end.
    pub sql: Option<u64>,
    /// [`Principal::id`].
    pub principal: String,
    /// The access-policy version.
    pub policy: PolicyVersion,
    /// Whether files can be read by path (a [`FileSchemaResolver`]).
    pub files: bool,
    /// [`VariableSnapshot::version`]: a resolved plan is never reused
    /// after a variable changed.
    pub variables: u64,
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
        num(&mut b, self.profile.code().into());
        let c = self.conf;
        let flags = [
            c.case_sensitive,
            c.ansi,
            c.group_by_aliases,
            c.group_by_ordinal,
            c.order_by_ordinal,
            c.lateral_column_alias,
            c.decimal_precision_loss,
            c.map_key_dedup_last_win,
            c.legacy_java_charsets,
            c.legacy_coding_error_action,
            c.run_sql_on_files,
            c.timestamp_ntz,
            c.legacy_time_parser,
            self.files,
            c.allow_parameterless_count,
            c.allow_star_with_single_table_identifier_in_count,
            c.udf_parameter_shadows_function,
            c.legacy_negative_index_in_array_insert,
        ];
        num(
            &mut b,
            flags.iter().fold(0, |acc, f| acc << 1 | u64::from(*f)),
        );
        // The CTE precedence policy.
        num(
            &mut b,
            match c.cte_precedence {
                CtePrecedencePolicy::Corrected => 0,
                CtePrecedencePolicy::Legacy => 1,
                CtePrecedencePolicy::Exception => 2,
            },
        );
        // The session zone: its id (or offset) is one input.
        match c.session_zone {
            Some(SessionZone::Fixed(v)) => {
                num(&mut b, 1);
                num(&mut b, u64::from(u32::from_ne_bytes(v.to_ne_bytes())));
            }
            Some(SessionZone::Region(r)) => {
                num(&mut b, 2);
                text(&mut b, &r.id);
            }
            None => num(&mut b, 0),
        }
        num(&mut b, self.current.len() as u64);
        self.current.iter().for_each(|p| text(&mut b, p));
        num(&mut b, self.catalogs);
        num(&mut b, self.catalog_epochs.len() as u64);
        for (name, epoch) in &self.catalog_epochs {
            text(&mut b, name);
            num(&mut b, *epoch);
        }
        num(&mut b, self.functions);
        num(&mut b, self.table_functions);
        match self.sql {
            None => num(&mut b, 0),
            Some(v) => {
                num(&mut b, 1);
                num(&mut b, v);
            }
        }
        text(&mut b, &self.principal);
        num(&mut b, self.policy.0);
        num(&mut b, self.variables);
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
    /// What Spark logs as warnings during analysis: hints it ignores.
    pub warnings: Vec<String>,
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
            && self.warnings == other.warnings
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

/// `current_path()`: the SQL path as the comma-separated entries.
pub(crate) fn current_path(catalogs: &dyn CatalogResolver) -> String {
    catalogs
        .sql_path()
        .iter()
        .map(|e| krishiv_common::sql_path::render(e))
        .collect::<Vec<_>>()
        .join(",")
}

/// The fingerprint of `ctx`, with the epochs of the catalogs consulted.
pub(crate) fn fingerprint(
    ctx: &AnalysisContext,
    catalog_epochs: BTreeMap<String, u64>,
) -> ContextFingerprint {
    let mut current = vec![ctx.catalogs.current_catalog().to_owned()];
    current.extend(ctx.catalogs.current_namespace().iter().cloned());
    current.push(current_path(&*ctx.catalogs));
    ContextFingerprint {
        profile: ctx.profile,
        conf: ctx.conf,
        current,
        catalogs: ctx.catalogs.epoch(),
        catalog_epochs,
        functions: ctx.functions.version(),
        table_functions: ctx.table_functions.version(),
        sql: ctx.sql.as_ref().map(|f| f.fingerprint()),
        principal: ctx.principal.id(),
        policy: ctx.policy.version(),
        files: ctx.files.is_some(),
        variables: ctx.variables.version,
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
            let policy = legacy::CtePolicy {
                precedence: ctx.conf.cte_precedence,
                case_sensitive: ctx.conf.case_sensitive,
            };
            if policy.precedence == CtePrecedencePolicy::Exception
                && let Some(conflict) = legacy::nested_conflict(plan.root(), policy)?
            {
                return Err(conflict);
            }
            let input = plan;
            let owned;
            let plan = if policy.precedence == CtePrecedencePolicy::Legacy {
                let inlined = legacy::eliminate(plan.root(), policy)
                    .map_err(|e| AnalysisError::new(e, plan.root().origin))?;
                owned = Plan::try_new(inlined)
                    .map_err(|e| AnalysisError::new(e, plan.root().origin))?;
                &owned
            } else {
                plan
            };
            let bindings = bind::bind(plan, ctx).await?;
            let metadata = started.elapsed();
            let resolving = std::time::Instant::now();
            let (root, output, securables, warnings) = resolve::resolve(plan, ctx, &bindings)?;
            unexecutable::check(&root)?;
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
                input_key: input.structural_hash(),
                context: fingerprint(ctx, epochs),
                query_text,
                reads_files,
                timings: AnalysisTimings {
                    metadata,
                    resolve: resolving.elapsed(),
                },
                warnings,
            })
        })
    }
}
