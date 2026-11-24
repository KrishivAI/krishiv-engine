// SPDX-License-Identifier: Apache-2.0
//! The physical planner (P2-T074): an optimized, resolved logical plan
//! ([`krishiv_plan::Plan`]) in, an executable [`PhysicalPlan`] out.
//!
//! # The seam
//! - A session analyzes and optimizes a query, then calls
//!   [`PhysicalPlanner::plan`] with a [`PlanningContext`]: the query's
//!   configuration snapshot and the tables it reads
//!   ([`TableProviderResolver`]), whose metadata planning reads.
//! - The result is a [`PhysicalPlan`]: data (see [`crate::physical`]). It
//!   goes to a [`Scheduler`](crate::Scheduler) with the
//!   [`WorkerContext`](crate::physical::WorkerContext) it is instantiated
//!   with (tables, spill scope); its [`Display`](std::fmt::Display) is the
//!   physical section of `EXPLAIN`.
//! - [`DefaultPlanner`] is the embedded implementation.
//! - Tables are opened through [`TableProvider`]s (spec 13 §7): the
//!   scan factory with its pushdowns ([`ScanPushdown`](crate::ScanPushdown))
//!   and the statistics the optimizer orders joins by
//!   ([`ProviderStatistics`]).
//!
//! # Single node
//! There is no exchange: each pipeline runs `min(worker slots, morsels)`
//! drivers that pull morsels from their source, so the scans' partitions
//! (Spark's file splits, sized by `spark.sql.files.maxPartitionBytes`) are
//! the unit of parallelism, and breakers (join builds, aggregates, sorts)
//! merge their drivers' states at `finalize`.

mod aggregate;
mod build;
mod build_project;
mod decide;
pub(crate) mod eval;
mod explain;
mod file_table;
mod join;
mod key_count;
mod late_fetch;
mod ndv_sample;
pub(crate) mod nodes;
mod observe;
mod orient;
mod probe_select;
mod properties;
pub(crate) mod pushdown;
mod python_relation;
mod python_udf;
mod python_udtf;
mod row_level;
mod runtime_filter;
mod scan_facts;
mod source_filter;
pub(crate) mod subquery;
mod table;
mod window;
mod write;

use std::fmt;
use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_plan::Plan;
use krishiv_udf_sdk::UdfLibrary;

pub use build::DefaultPlanner;
pub(crate) use build::concat_schemas;
pub use explain::{ExplainNode, FilterGap, FilterGapReason, InputFacts, NodeFacts};
pub use file_table::FileTable;
pub use key_count::{KeyOptions, KeyOutcome, KeyResult, KeyUnknown};
pub use ndv_sample::{SampleOptions, SampleOutcome, SampleResult};
pub use row_level::{
    CardinalityRule, RowIdScan, RowLevelCapabilities, RowLevelCounts, RowLevelOperation,
    RowLevelProblem, RowLevelTable, RowLevelWrite,
};
pub use table::{
    MemoryTable, MemoryTables, ProviderStatistics, SpilledTable, TableProvider,
    TableProviderResolver,
};
pub use write::{
    IDEMPOTENCY_KEY_OPTION, TableCapabilities, WriteMode, WriteRequest, idempotency_key_option,
};

use crate::error::Result;
pub use crate::physical::PhysicalPlan;

/// Turns an optimized, resolved logical plan into an executable plan.
pub trait PhysicalPlanner: Send + Sync + fmt::Debug {
    /// The physical plan of `plan`.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a node or expression without a
    /// physical form yet; `INTERNAL_ERROR` for a plan that is not resolved
    /// or not consistent; the table providers' errors.
    fn plan(&self, plan: &Plan, cx: &PlanningContext) -> Result<PhysicalPlan>;
}

/// What planning one query needs besides its plan. Only metadata: the
/// worker resources a plan runs with (spill scope, scan sources) come at
/// instantiation ([`WorkerContext`](crate::physical::WorkerContext)).
#[derive(Clone)]
#[non_exhaustive]
pub struct PlanningContext {
    /// The tables the plan's scans read: their schemas, statistics and
    /// filter pushdown.
    pub tables: Arc<dyn TableProviderResolver>,
    /// The query's configuration snapshot (`spark.sql.ansi.enabled`, the
    /// session time zone, ...).
    pub config: Arc<Config>,
    /// The session's user (`current_user()`).
    pub user: Arc<str>,
    /// The session's UDFs, for validating UDF calls at planning. The
    /// implementation a plan runs is resolved from the worker at
    /// instantiation, never stored in the plan.
    pub udfs: Option<Arc<UdfLibrary>>,
    /// The query's start instant in microseconds since the epoch, when the
    /// caller fixes it (a query planned in several pieces reads one
    /// `current_timestamp()`); `None`: planning takes it.
    pub query_start_micros: Option<i64>,
    /// The salt unseeded `TABLESAMPLE` seeds are drawn from: the same salt
    /// draws the same seeds, so a caller keeping it for one DataFrame keeps
    /// its sampled rows across actions (Spark's Dataset keeps its executed
    /// plan). `None`: each plan draws fresh seeds.
    pub sample_salt: Option<u64>,
    /// The plan is only shown (`EXPLAIN`, estimates), never run: checks Spark makes when the plan
    /// executes, such as `QUERY_ONLY_CORRUPT_RECORD_COLUMN` (raised by the file format's reader), are
    /// skipped.
    pub inspect: bool,
    /// Whether a scan that decides once (`ScanPushdown::decides_once`) is asked to decide. On for plans that
    /// run and for inspected plans (`EXPLAIN` asks the source once, as Spark's does); off only for
    /// `physical_counted` and `physical_probe`, which run no user code.
    pub decisions: bool,
    /// The Spark release line the session follows (`None`: the default, 4.3).
    pub spark_version: Option<krishiv_kernels::SparkVersion>,
}

impl PlanningContext {
    /// A context over `tables` and `config`.
    pub fn new(tables: Arc<dyn TableProviderResolver>, config: Arc<Config>) -> Self {
        Self {
            tables,
            config,
            user: Arc::from(krishiv_plan::analyzer::Principal::local_user().name),
            udfs: None,
            query_start_micros: None,
            sample_salt: None,
            inspect: false,
            decisions: true,
            spark_version: None,
        }
    }

    /// The same context planning for the Spark release line `version`.
    #[must_use]
    pub fn with_spark_version(mut self, version: krishiv_kernels::SparkVersion) -> Self {
        self.spark_version = Some(version);
        self
    }

    /// The same context for a plan that is only inspected when `inspect`.
    #[must_use]
    pub fn with_inspection(mut self, inspect: bool) -> Self {
        self.inspect = inspect;
        self
    }

    /// The same context asking deciding scans to decide only when `decisions`.
    #[must_use]
    pub fn with_decisions(mut self, decisions: bool) -> Self {
        self.decisions = decisions;
        self
    }

    /// The same context drawing unseeded sample seeds from `salt`.
    #[must_use]
    pub fn with_sample_salt(mut self, salt: u64) -> Self {
        self.sample_salt = Some(salt);
        self
    }

    /// The same context for the session user `user`.
    #[must_use]
    pub fn with_user(mut self, user: impl Into<Arc<str>>) -> Self {
        self.user = user.into();
        self
    }

    /// The same context planning with the query start `micros` (microseconds
    /// since the epoch).
    #[must_use]
    pub fn with_query_start(mut self, micros: i64) -> Self {
        self.query_start_micros = Some(micros);
        self
    }

    /// The same context validating UDF calls against `udfs`.
    #[must_use]
    pub fn with_udfs(mut self, udfs: Arc<UdfLibrary>) -> Self {
        self.udfs = Some(udfs);
        self
    }
}

impl fmt::Debug for PlanningContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlanningContext")
            .field("tables", &self.tables)
            .finish_non_exhaustive()
    }
}
