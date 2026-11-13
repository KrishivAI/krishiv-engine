// SPDX-License-Identifier: Apache-2.0
//! The scan of a relation held in one spill file (design P3-T083 §3.2: a
//! `DISK_ONLY` cache entry, or a `MEMORY_AND_DISK` one spilled under
//! memory pressure).
//!
//! One driver streams the file in the order it was written, a batch at a
//! time: each read runs on the blocking pool and holds its frame in the
//! reading driver's memory consumer (the [`SpillReader`]'s reservation), so
//! the reading query pays for what it reads, never the whole relation. The
//! pushed filters, the runtime filters' row stage and the projection apply
//! to each batch as [`MemoryScan`](super::MemoryScan) applies them; row
//! selections and `_metadata` columns are refused, as there.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_io::{ColumnPath, MetadataCols, ScanFilter};
use krishiv_memory::{ConsumerHandle, SpillFile, SpillReader};
use krishiv_types::batch::Batch;

use super::expr::Predicate;
use super::scan::{FilterPushdown, PushedFilter, ScanPushdown, ScanPushdowns};
use super::{RuntimeFilters, ScanRows, global_ref, local_mut};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::executor::{Executor, run_blocking};
use crate::operator::{GlobalState, LocalState, Source};

/// Scans the batches of one spill file (see the [module docs](self)).
#[derive(Debug)]
pub struct SpilledScan {
    table: SchemaRef,
    file: Arc<SpillFile>,
    projection: Option<Vec<ColumnPath>>,
    filters: Vec<Arc<dyn Predicate>>,
    limit: Option<u64>,
    output: SchemaRef,
    runtime: RuntimeFilters,
}

struct SpilledGlobal {
    /// Runtime filters on the table's columns.
    runtime: Vec<ScanFilter>,
    /// Rows returned (for a pushed limit).
    returned: AtomicU64,
}

struct SpilledLocal {
    /// The open reader; `None` before the first read and while a read runs.
    reader: Option<SpillReader>,
    consumer: Arc<ConsumerHandle>,
    executor: Arc<dyn Executor>,
    rows: ScanRows,
    partition: u32,
    started: bool,
    done: bool,
}

impl SpilledScan {
    /// A scan of every column of `file`, whose batches have schema `table`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if the file's fields differ from `table`'s.
    pub fn try_new(table: SchemaRef, file: Arc<SpillFile>) -> Result<Self> {
        if file.schema().fields() != table.fields() {
            return Err(internal(format!(
                "spill file has fields {:?}, expected {:?}",
                file.schema().fields(),
                table.fields()
            )));
        }
        Ok(Self {
            output: Arc::clone(&table),
            table,
            file,
            projection: None,
            filters: Vec::new(),
            limit: None,
            runtime: RuntimeFilters::default(),
        })
    }

    fn output_for(&self, pushdowns: &ScanPushdowns) -> Result<SchemaRef> {
        if pushdowns.selection.is_some() {
            return Err(internal(
                "a spilled relation's scan cannot honour a row selection".to_owned(),
            ));
        }
        if pushdowns.metadata != MetadataCols::default() {
            return Err(internal(
                "a spilled relation's scan has no _metadata columns".to_owned(),
            ));
        }
        match &pushdowns.projection {
            Some(paths) => krishiv_io::project_schema(&self.table, paths),
            None => Ok(Arc::clone(&self.table)),
        }
    }

    /// Filters, runtime filters and projection over one batch read back.
    fn shape(&self, batch: Batch, local: &mut SpilledLocal) -> Result<Batch> {
        let mut piece = batch;
        for filter in &self.filters {
            if piece.num_rows() == 0 {
                break;
            }
            let mask = filter.evaluate_in(piece.data(), local.partition)?;
            piece = piece.filter(&mask)?;
        }
        if piece.num_rows() > 0
            && let Some(mask) = local.rows.mask(piece.data())?
        {
            piece = piece.filter(&mask)?;
        }
        let piece = match &self.projection {
            Some(paths) => {
                let (data, weights) = piece.into_parts();
                let data = krishiv_io::project_batch(&data, paths)?;
                Batch::with_weights(data, weights)?
            }
            None => piece,
        };
        Ok(piece.with_split(Some(local.partition)))
    }
}

impl ScanPushdown for SpilledScan {
    fn table_schema(&self) -> SchemaRef {
        Arc::clone(&self.table)
    }

    fn filter_pushdown(&self, _filter: &PushedFilter) -> FilterPushdown {
        FilterPushdown::Exact
    }

    fn build(&self, pushdowns: ScanPushdowns) -> Result<Arc<dyn Source>> {
        let output = self.output_for(&pushdowns)?;
        let filters: Vec<Arc<dyn Predicate>> = pushdowns
            .filters
            .iter()
            .map(|f| Arc::clone(&f.predicate))
            .collect();
        Ok(Arc::new(Self {
            table: Arc::clone(&self.table),
            file: Arc::clone(&self.file),
            projection: pushdowns.projection,
            limit: pushdowns.limit.filter(|_| filters.is_empty()),
            filters,
            output,
            runtime: pushdowns.dynamic,
        }))
    }

    fn output_schema(&self, pushdowns: &ScanPushdowns) -> Result<SchemaRef> {
        self.output_for(pushdowns)
    }
}

impl Source for SpilledScan {
    fn name(&self) -> &'static str {
        "SpilledScan"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.output)
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(SpilledGlobal {
            runtime: self.runtime.scan_filters(&self.table),
            returned: AtomicU64::new(0),
        }))
    }

    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        // One file read in order: one driver.
        1
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        let g = global_ref::<SpilledGlobal>(global, "SpilledScan")?;
        let mut rows = self
            .runtime
            .scan_stage(&g.runtime, &self.table, ctx.partition());
        rows.start_partition();
        Ok(Box::new(SpilledLocal {
            reader: None,
            consumer: Arc::clone(ctx.consumer().consumer()),
            executor: Arc::clone(ctx.executor()),
            rows,
            partition: ctx.partition(),
            started: false,
            done: false,
        }))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(async move {
            let g = global_ref::<SpilledGlobal>(global, "SpilledScan")?;
            let local = local_mut::<SpilledLocal>(local, "SpilledScan")?;
            if local.done
                || self
                    .limit
                    .is_some_and(|l| g.returned.load(Ordering::Relaxed) >= l)
            {
                return Ok(None);
            }
            local.started = true;
            let reader = local.reader.take();
            let file = Arc::clone(&self.file);
            let reservation = local.consumer.reservation();
            let (batch, reader) = run_blocking(&*local.executor, move || {
                let mut reader = match reader {
                    Some(r) => r,
                    None => file.reader(reservation).map_err(KrishivError::from)?,
                };
                let batch = reader.next().transpose().map_err(KrishivError::from)?;
                Ok((batch, reader))
            })
            .await?;
            let Some(batch) = batch else {
                local.done = true;
                return Ok(None);
            };
            local.reader = Some(reader);
            let piece = self.shape(batch, local)?;
            g.returned
                .fetch_add(piece.num_rows() as u64, Ordering::Relaxed);
            Ok(Some(piece))
        })
    }

    fn morsels_claimed(&self, local: &LocalState) -> u64 {
        local
            .downcast_ref::<SpilledLocal>()
            .map_or(0, |l| u64::from(l.started))
    }

    fn reports_positions(&self) -> bool {
        true
    }

    fn position(&self, local: &LocalState) -> Option<u64> {
        local
            .downcast_ref::<SpilledLocal>()
            .and_then(|l| l.started.then_some(0))
    }

    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
        self.runtime.extra_metrics()
    }
}
