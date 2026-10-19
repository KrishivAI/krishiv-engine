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
pub(crate) mod eval;
mod explain;
mod file_table;
mod join;
pub(crate) mod nodes;
pub(crate) mod pushdown;
mod runtime_filter;
pub(crate) mod subquery;
mod table;

use std::fmt;
use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_plan::Plan;

pub use build::DefaultPlanner;
pub(crate) use build::concat_schemas;
pub use explain::ExplainNode;
pub use file_table::FileTable;
pub use table::{
    MemoryTable, MemoryTables, ProviderStatistics, TableProvider, TableProviderResolver,
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
}

impl PlanningContext {
    /// A context over `tables` and `config`.
    pub fn new(tables: Arc<dyn TableProviderResolver>, config: Arc<Config>) -> Self {
        Self {
            tables,
            config,
            user: Arc::from(krishiv_plan::analyzer::Principal::local_user().name),
        }
    }

    /// The same context for the session user `user`.
    #[must_use]
    pub fn with_user(mut self, user: impl Into<Arc<str>>) -> Self {
        self.user = user.into();
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
