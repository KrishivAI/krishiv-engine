// SPDX-License-Identifier: Apache-2.0
//! The in-memory scan and the pushdown interface that file scans will
//! implement later.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_types::batch::Batch;

use super::expr::Predicate;
use super::{global_ref, local_mut};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::morsel::MorselCursor;
use crate::operator::{GlobalState, LocalState, Source};

/// How far a scan honoured a pushed filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterPushdown {
    /// The scan ignores the filter: the planner keeps a `Filter` above it.
    Unsupported,
    /// The scan may drop rows that fail the filter but may still return
    /// some (for example by skipping row groups on statistics): the planner
    /// keeps the `Filter` above it.
    Inexact,
    /// Every row the scan returns satisfies the filter: the planner drops
    /// the `Filter`.
    Exact,
}

/// What the planner can push into a scan, before the plan is built. The
/// file-format scans (`FileFormat`, P2-T080a) implement the same trait.
pub trait ScanPushdown {
    /// Reads only the columns at `indices` of the scan's **current** output,
    /// in that order (so successive calls compose). The output schema
    /// changes accordingly.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if an index is out of range.
    fn push_projection(&mut self, indices: &[usize]) -> Result<()>;

    /// Offers `predicate`. It is evaluated against batches of the scan's
    /// **full, unprojected** schema, so pushing it does not depend on the
    /// projection. The scan reports how far it honours it.
    fn push_filter(&mut self, predicate: Arc<dyn Predicate>) -> FilterPushdown;
}

/// Scans batches held in memory. A morsel is a run of consecutive batches
/// of about `krishiv.exec.morsel.rows` rows; the source slices (zero-copy)
/// batches larger than `krishiv.exec.batchSize` so a downstream `LIMIT`
/// never reads more than one such batch past its limit. Pushed filters are
/// applied before the projection and honoured exactly.
///
/// Batches with no rows are skipped. Reports the morsels claimed per driver
/// ([`Source::morsels_claimed`]).
#[derive(Debug)]
pub struct MemoryScan {
    batches: Vec<Batch>,
    projection: Option<Vec<usize>>,
    filters: Vec<Arc<dyn Predicate>>,
    output: SchemaRef,
    rows_read: AtomicU64,
}

pub(super) struct ScanGlobal {
    pub(super) cursor: MorselCursor,
    pub(super) morsels: Vec<Range<usize>>,
    pub(super) batch_rows: usize,
}

pub(super) struct ScanLocal {
    /// Batches of the claimed morsel not fully read yet.
    pub(super) batches: Range<usize>,
    /// Rows already read of `batches.start`.
    pub(super) offset: usize,
    pub(super) claimed: u64,
}

impl MemoryScan {
    /// A scan of `batches`, all of schema `table`. Push projection and
    /// filters on it, then wrap it in an `Arc` as the plan's source.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if a batch's fields differ from `table`'s.
    pub fn try_new(table: SchemaRef, batches: Vec<Batch>) -> Result<Self> {
        if let Some(bad) = batches
            .iter()
            .find(|b| b.schema().fields() != table.fields())
        {
            return Err(internal(format!(
                "scan batch has fields {:?}, expected {:?}",
                bad.schema().fields(),
                table.fields()
            )));
        }
        Ok(Self {
            output: table,
            batches,
            projection: None,
            filters: Vec::new(),
            rows_read: AtomicU64::new(0),
        })
    }

    /// Rows read from memory so far, by every driver, before filtering.
    pub fn rows_read(&self) -> u64 {
        self.rows_read.load(Ordering::Relaxed)
    }

    /// Reads the next piece of the claimed (non-empty) morsel, filtered and
    /// projected.
    pub(super) fn next_piece(&self, local: &mut ScanLocal, batch_rows: usize) -> Result<Batch> {
        let batch = self
            .batches
            .get(local.batches.start)
            .ok_or_else(|| internal("morsel beyond the scan's batches".to_owned()))?;
        let rows = batch.num_rows();
        let take = batch_rows.min(rows - local.offset);
        let piece = if local.offset == 0 && take == rows {
            batch.clone()
        } else {
            batch.slice(local.offset, take)?
        };
        local.offset += take;
        if local.offset >= rows {
            local.batches.start += 1;
            local.offset = 0;
        }
        self.rows_read.fetch_add(take as u64, Ordering::Relaxed);
        let mut piece = piece;
        for filter in &self.filters {
            if piece.num_rows() == 0 {
                break;
            }
            let mask = filter.evaluate(piece.data())?;
            piece = piece.filter(&mask)?;
        }
        match &self.projection {
            Some(indices) => piece.project(indices),
            None => Ok(piece),
        }
    }

    pub(super) fn next_batch(
        &self,
        local: &mut LocalState,
        global: &GlobalState,
    ) -> Result<Option<Batch>> {
        let g = global_ref::<ScanGlobal>(global, "MemoryScan")?;
        let local = local_mut::<ScanLocal>(local, "MemoryScan")?;
        loop {
            if local.batches.is_empty() {
                let Some(m) = g.cursor.claim() else {
                    return Ok(None);
                };
                local.batches =
                    g.morsels.get(m).cloned().ok_or_else(|| {
                        internal(format!("morsel {m} out of {}", g.morsels.len()))
                    })?;
                local.offset = 0;
                local.claimed += 1;
            }
            let piece = self.next_piece(local, g.batch_rows)?;
            if piece.num_rows() > 0 {
                return Ok(Some(piece));
            }
        }
    }
}

impl ScanPushdown for MemoryScan {
    fn push_projection(&mut self, indices: &[usize]) -> Result<()> {
        // Validate against the current output (also shapes the new schema).
        let fields = self.output.fields();
        let picked = indices
            .iter()
            .map(|&i| {
                fields
                    .get(i)
                    .cloned()
                    .ok_or_else(|| internal(format!("projection index {i} out of range")))
            })
            .collect::<Result<Vec<_>>>()?;
        let composed = match &self.projection {
            Some(current) => indices
                .iter()
                .map(|&i| {
                    current
                        .get(i)
                        .copied()
                        .ok_or_else(|| internal(format!("projection index {i} out of range")))
                })
                .collect::<Result<Vec<_>>>()?,
            None => indices.to_vec(),
        };
        self.output = Arc::new(arrow_schema::Schema::new_with_metadata(
            picked,
            self.output.metadata().clone(),
        ));
        self.projection = Some(composed);
        Ok(())
    }

    fn push_filter(&mut self, predicate: Arc<dyn Predicate>) -> FilterPushdown {
        self.filters.push(predicate);
        FilterPushdown::Exact
    }
}

impl Source for MemoryScan {
    fn name(&self) -> &'static str {
        "MemoryScan"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.output)
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        let config = ctx.query().config();
        let morsel_rows = config.morsel_rows.max(1);
        let mut morsels = Vec::new();
        let (mut start, mut rows) = (0, 0usize);
        for (i, b) in self.batches.iter().enumerate() {
            rows += b.num_rows();
            if rows >= morsel_rows {
                morsels.push(start..i + 1);
                (start, rows) = (i + 1, 0);
            }
        }
        if start < self.batches.len() {
            morsels.push(start..self.batches.len());
        }
        Ok(Arc::new(ScanGlobal {
            cursor: MorselCursor::new(morsels.len(), config.seed),
            morsels,
            batch_rows: config.batch_size.max(1),
        }))
    }

    fn max_parallelism(&self, global: &GlobalState) -> usize {
        global_ref::<ScanGlobal>(global, "MemoryScan").map_or(1, |g| g.morsels.len().max(1))
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(ScanLocal {
            batches: 0..0,
            offset: 0,
            claimed: 0,
        }))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(std::future::ready(self.next_batch(local, global)))
    }

    fn morsels_claimed(&self, local: &LocalState) -> u64 {
        local.downcast_ref::<ScanLocal>().map_or(0, |l| l.claimed)
    }
}
