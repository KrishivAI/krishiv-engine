// SPDX-License-Identifier: Apache-2.0
//! The read of one `rewrite_data_files` group, run by at most as many drivers as the group is
//! expected to write files.
//!
//! On one worker an exchange is lowered into the operators around it, so the group's write runs
//! one sink driver per scan driver, and each driver's sink writes files of its own: two small
//! files read by two drivers would be "compacted" into two files. Iceberg 1.12.0 reads a group
//! in about `expected_output_files` splits (`SizeBasedFileRewritePlanner.inputSplitSize`), one
//! Spark task each, so a group expected to write one file is read and written by one task.
//! [`CappedScan`] bounds the drivers the same way; within that bound the morsels are claimed by
//! whichever driver is free, so the output files are not balanced by bytes as Spark's splits are.

use std::sync::Arc;

use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_exec::planner::{RowLevelOperation, RowLevelTable, TableProvider};
use krishiv_exec::{
    FilterPushdown, GlobalState, LocalState, PushedFilter, ScanPushdown, ScanPushdowns, Source,
    TaskContext,
};
use krishiv_io::FilterId;
use krishiv_types::batch::Batch;

/// The group's files of `table` as a provider whose scans run at most `tasks` drivers.
#[derive(Debug)]
pub(super) struct GroupTarget {
    pub(super) table: Arc<dyn RowLevelTable>,
    pub(super) files: Vec<Vec<u8>>,
    pub(super) tasks: usize,
}

impl TableProvider for GroupTarget {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        Ok(Arc::new(CappedScan {
            inner: self.table.scan_files(&self.files)?,
            tasks: self.tasks.max(1),
        }))
    }

    fn row_level(&self, op: RowLevelOperation) -> Option<Arc<dyn RowLevelTable>> {
        (op == RowLevelOperation::Delete).then(|| Arc::clone(&self.table))
    }
}

/// `inner`, whose sources run at most `tasks` drivers.
struct CappedScan {
    inner: Arc<dyn ScanPushdown>,
    tasks: usize,
}

impl ScanPushdown for CappedScan {
    fn table_schema(&self) -> SchemaRef {
        self.inner.table_schema()
    }

    fn filter_pushdown(&self, filter: &PushedFilter) -> FilterPushdown {
        self.inner.filter_pushdown(filter)
    }

    fn only_reads_corrupt_record(&self, referenced: &[&str]) -> bool {
        self.inner.only_reads_corrupt_record(referenced)
    }

    fn supports_selection(&self) -> bool {
        self.inner.supports_selection()
    }

    fn build(&self, pushdowns: ScanPushdowns) -> krishiv_exec::Result<Arc<dyn Source>> {
        Ok(Arc::new(Capped {
            inner: self.inner.build(pushdowns)?,
            tasks: self.tasks,
        }))
    }

    fn explain(&self, pushdowns: &ScanPushdowns) -> Option<String> {
        self.inner.explain(pushdowns)
    }

    fn output_schema(&self, pushdowns: &ScanPushdowns) -> krishiv_exec::Result<SchemaRef> {
        self.inner.output_schema(pushdowns)
    }
}

/// A source run by at most `tasks` drivers; otherwise `inner` as it is.
struct Capped {
    inner: Arc<dyn Source>,
    tasks: usize,
}

impl Source for Capped {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    fn global_state(&self, ctx: &TaskContext) -> krishiv_exec::Result<GlobalState> {
        self.inner.global_state(ctx)
    }

    fn max_parallelism(&self, global: &GlobalState) -> usize {
        self.inner.max_parallelism(global).min(self.tasks)
    }

    fn deterministic_split(&self, global: &GlobalState) -> Option<usize> {
        self.inner
            .deterministic_split(global)
            .map(|n| n.min(self.tasks))
    }

    fn partitions(&self, global: &GlobalState) -> usize {
        self.inner.partitions(global)
    }

    fn single_partition(&self, global: &GlobalState) -> bool {
        self.inner.single_partition(global)
    }

    fn local_state(
        &self,
        ctx: &TaskContext,
        global: &GlobalState,
    ) -> krishiv_exec::Result<LocalState> {
        self.inner.local_state(ctx, global)
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, krishiv_exec::Result<Option<Batch>>> {
        self.inner.get_data(local, global)
    }

    fn morsels_claimed(&self, local: &LocalState) -> u64 {
        self.inner.morsels_claimed(local)
    }

    fn reports_positions(&self) -> bool {
        self.inner.reports_positions()
    }

    fn position(&self, local: &LocalState) -> Option<u64> {
        self.inner.position(local)
    }

    fn runtime_filters(&self) -> Vec<FilterId> {
        self.inner.runtime_filters()
    }

    fn extra_metrics(&self) -> krishiv_exec::ExtraMetrics {
        self.inner.extra_metrics()
    }
}
