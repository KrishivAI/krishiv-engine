// SPDX-License-Identifier: Apache-2.0
//! The physical planner (P2-T074): an optimized, resolved logical plan
//! ([`krishiv_plan::Plan`]) in, an executable [`PhysicalPlan`] out.
//!
//! # The seam
//! - A session analyzes and optimizes a query, then calls
//!   [`PhysicalPlanner::plan`] with a [`PlanningContext`]: the query's
//!   configuration snapshot, its spill scope and the tables it reads
//!   ([`TableProviderResolver`]).
//! - The result's [`PhysicalPlan::root`] goes to a
//!   [`Scheduler`](crate::Scheduler) (`QueryRunner::submit`); its
//!   [`Display`](std::fmt::Display) is the physical section of `EXPLAIN`.
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

mod build;
mod explain;
mod nodes;
mod table;

use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use krishiv_common::config::Config;
use krishiv_memory::SpillScope;
use krishiv_plan::Plan;

pub use build::DefaultPlanner;
pub use explain::ExplainNode;
pub use table::{
    MemoryTable, MemoryTables, ProviderStatistics, TableProvider, TableProviderResolver,
};

use crate::error::Result;
use crate::pipeline::ExecNode;

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

/// What planning one query needs besides its plan.
#[derive(Clone)]
#[non_exhaustive]
pub struct PlanningContext {
    /// The tables the plan's scans read.
    pub tables: Arc<dyn TableProviderResolver>,
    /// The query's configuration snapshot (`spark.sql.ansi.enabled`, the
    /// session time zone, ...).
    pub config: Arc<Config>,
    /// The query's spill scope, shared by its spilling operators; `None`
    /// makes them unspillable.
    pub spill: Option<SpillScope>,
}

impl PlanningContext {
    /// A context over `tables` and `config`, without spilling.
    pub fn new(tables: Arc<dyn TableProviderResolver>, config: Arc<Config>) -> Self {
        Self {
            tables,
            config,
            spill: None,
        }
    }

    /// The same context spilling into `scope` (one per query).
    #[must_use]
    pub fn with_spill(mut self, scope: SpillScope) -> Self {
        self.spill = Some(scope);
        self
    }
}

impl fmt::Debug for PlanningContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlanningContext")
            .field("tables", &self.tables)
            .field("spill", &self.spill.is_some())
            .finish_non_exhaustive()
    }
}

/// An executable plan: the root node to submit, the schema of its result
/// batches and a printable tree.
#[derive(Clone)]
pub struct PhysicalPlan {
    root: Arc<dyn ExecNode>,
    schema: SchemaRef,
    explain: ExplainNode,
    warnings: Vec<String>,
}

impl PhysicalPlan {
    /// A plan of `root`, whose result batches have `schema`, printed as
    /// `explain`.
    pub fn new(root: Arc<dyn ExecNode>, schema: SchemaRef, explain: ExplainNode) -> Self {
        Self {
            root,
            schema,
            explain,
            warnings: Vec::new(),
        }
    }

    /// The root node, for [`Scheduler::submit`](crate::Scheduler::submit).
    /// A plan runs once: its nodes hold per-query state.
    pub fn root(&self) -> &Arc<dyn ExecNode> {
        &self.root
    }

    /// The schema of the result batches: the logical plan's output names.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// The printable tree.
    pub fn explain(&self) -> &ExplainNode {
        &self.explain
    }

    /// Warnings planning produced (an ignored hint, as Spark logs it).
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// The same plan with `warnings` added.
    #[must_use]
    pub fn with_warnings(mut self, warnings: Vec<String>) -> Self {
        self.warnings.extend(warnings);
        self
    }
}

impl fmt::Debug for PhysicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PhysicalPlan")
            .field("schema", &self.schema)
            .field("explain", &self.explain.to_string())
            .field("warnings", &self.warnings)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for PhysicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.explain.fmt(f)
    }
}
