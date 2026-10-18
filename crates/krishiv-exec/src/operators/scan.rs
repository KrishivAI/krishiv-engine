// SPDX-License-Identifier: Apache-2.0
//! The in-memory scan and the scan pushdown interface, which the file scan
//! ([`FileScan`](super::FileScan)) implements too.
//!
//! **Pushdown is decided once.** The planner asks a scan how far it honours
//! each filter ([`ScanPushdown::filter_pushdown`]), then builds the source
//! with every pushdown at once ([`ScanPushdown::build`]): an immutable
//! [`ScanPushdowns`], never ordered calls on a mutable builder. For a file
//! scan it becomes the format framework's immutable
//! [`ScanRequest`](krishiv_io::ScanRequest).

use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_io::{
    ColumnPath, DynamicFilter, FieldMapping, FileRef, MetadataCols, PushdownExpr, RowSelection,
};
use krishiv_types::batch::Batch;

use super::expr::Predicate;
use super::{global_ref, local_mut};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::morsel::MorselCursor;
use crate::operator::{GlobalState, LocalState, Source};

/// How far a scan honours a pushed filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterPushdown {
    /// The scan ignores the filter: the planner keeps a `Filter` above it.
    Unsupported,
    /// The scan may drop rows that fail the filter but may still return
    /// some (for example by skipping row groups on statistics): the planner
    /// keeps the `Filter` above it. File scans never report more: their
    /// pruning keeps a superset of the matching rows.
    Inexact,
    /// Every row the scan returns satisfies the filter: the planner drops
    /// the `Filter`.
    Exact,
}

/// One filter offered to a scan: the row evaluator, over the scan's full
/// table schema, and, when the planner could translate it, the typed tree
/// that file formats prune with (statistics, dictionaries, page indexes).
#[derive(Clone, Debug)]
pub struct PushedFilter {
    /// Evaluates the filter on rows of the table schema.
    pub predicate: Arc<dyn Predicate>,
    /// The same filter as a typed predicate over the table's columns, if it
    /// can be expressed as one.
    pub expr: Option<PushdownExpr>,
}

impl PushedFilter {
    /// A filter with only a row evaluator (no pruning).
    #[must_use]
    pub fn opaque(predicate: Arc<dyn Predicate>) -> Self {
        Self {
            predicate,
            expr: None,
        }
    }

    /// A filter with its typed form.
    #[must_use]
    pub fn typed(predicate: Arc<dyn Predicate>, expr: PushdownExpr) -> Self {
        Self {
            predicate,
            expr: Some(expr),
        }
    }
}

/// File-relative row selections (deletion vectors, positional deletes) by
/// file, for scans whose format honours them.
pub trait FileSelections: Send + Sync + fmt::Debug {
    /// The rows of `file` to read, or `None` for all of them.
    fn selection(&self, file: &FileRef) -> Option<RowSelection>;
}

/// Everything the planner pushes into one scan, decided at once. Build it
/// with struct-update syntax over [`ScanPushdowns::default`], so new
/// pushdowns keep the default for planners that do not know them.
#[derive(Clone, Debug, Default)]
pub struct ScanPushdowns {
    /// Columns to read, in output order; nested struct fields allowed
    /// (nested-schema pruning). `None` reads every column of the table.
    pub projection: Option<Vec<ColumnPath>>,
    /// Filters (a conjunction), each over the table schema.
    pub filters: Vec<PushedFilter>,
    /// At most this many rows are needed (a `LIMIT` with no operator that
    /// drops rows in between). A hint: the scan may stop early but returns
    /// whole batches. Ignored while filters are pushed (the `Filter` above
    /// the scan may drop rows).
    pub limit: Option<u64>,
    /// Runtime filters (join build sides, Top-N bounds) that tighten while
    /// the scan runs; file formats prune with them at unit boundaries.
    pub dynamic: Vec<Arc<dyn DynamicFilter>>,
    /// Rows to read per file; only for scans that
    /// [support](ScanPushdown::supports_selection) it.
    pub selection: Option<Arc<dyn FileSelections>>,
    /// How table columns find file columns.
    pub mapping: FieldMapping,
    /// `_metadata` columns to append (file scans).
    pub metadata: MetadataCols,
}

/// What a planner can push into a scan. Implemented by scan factories: the
/// in-memory [`MemoryScan`] and the file scan
/// ([`FileScan`](super::FileScan)), which builds a
/// [`FileScanSource`](super::FileScanSource).
pub trait ScanPushdown: Send + Sync {
    /// The schema projections and filters refer to.
    fn table_schema(&self) -> SchemaRef;
    /// How far a scan built with `filter` honours it. The planner keeps the
    /// `Filter` above the scan unless this is [`FilterPushdown::Exact`].
    fn filter_pushdown(&self, filter: &PushedFilter) -> FilterPushdown;
    /// Whether [`ScanPushdowns::selection`] is honoured.
    fn supports_selection(&self) -> bool {
        false
    }
    /// The source reading `pushdowns`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a projection path that does not exist or a
    /// selection the scan cannot honour.
    fn build(&self, pushdowns: ScanPushdowns) -> Result<Arc<dyn Source>>;
}

/// Scans batches held in memory. A morsel is a run of consecutive batches
/// of about `krishiv.exec.morsel.rows` rows; the source slices (zero-copy)
/// batches larger than `krishiv.exec.batchSize` so a downstream `LIMIT`
/// never reads more than one such batch past its limit. Pushed filters are
/// applied before the projection and honoured exactly; nested projections
/// prune struct fields; a pushed limit stops the scan once that many rows
/// were returned. Dynamic filters and field mapping do not apply to
/// in-memory data; row selections and `_metadata` columns are refused.
///
/// One piece is returned per call, possibly with no rows after a pushed
/// filter, so the driver reaches a batch boundary after every piece read.
/// Reports the morsels claimed per driver
/// ([`Source::morsels_claimed`]).
#[derive(Debug)]
pub struct MemoryScan {
    table: SchemaRef,
    batches: Arc<[Batch]>,
    projection: Option<Vec<ColumnPath>>,
    filters: Vec<Arc<dyn Predicate>>,
    limit: Option<u64>,
    output: SchemaRef,
    rows_read: AtomicU64,
}

pub(super) struct ScanGlobal {
    pub(super) cursor: MorselCursor,
    pub(super) morsels: Vec<Range<usize>>,
    pub(super) batch_rows: usize,
    /// Rows returned by every driver (for a pushed limit).
    pub(super) returned: AtomicU64,
}

pub(super) struct ScanLocal {
    /// Batches of the claimed morsel not fully read yet.
    pub(super) batches: Range<usize>,
    /// Rows already read of `batches.start`.
    pub(super) offset: usize,
    pub(super) claimed: u64,
}

impl MemoryScan {
    /// A scan of every column of `batches`, all of schema `table`. Build a
    /// scan with pushdowns from it with [`Self::with_pushdowns`] (or
    /// [`ScanPushdown::build`]).
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
            output: Arc::clone(&table),
            table,
            batches: batches.into(),
            projection: None,
            filters: Vec::new(),
            limit: None,
            rows_read: AtomicU64::new(0),
        })
    }

    /// A scan of the same batches reading `pushdowns` (which replace this
    /// scan's own). The batches are shared, not copied.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if a projected path does not exist, or a selection
    /// or `_metadata` columns are pushed (in-memory batches have no files).
    pub fn with_pushdowns(&self, pushdowns: ScanPushdowns) -> Result<Self> {
        if pushdowns.selection.is_some() {
            return Err(internal(
                "an in-memory scan cannot honour a row selection".to_owned(),
            ));
        }
        if pushdowns.metadata != MetadataCols::default() {
            return Err(internal(
                "an in-memory scan has no _metadata columns".to_owned(),
            ));
        }
        let output = match &pushdowns.projection {
            Some(paths) => krishiv_io::project_schema(&self.table, paths)?,
            None => Arc::clone(&self.table),
        };
        let filters: Vec<Arc<dyn Predicate>> = pushdowns
            .filters
            .iter()
            .map(|f| Arc::clone(&f.predicate))
            .collect();
        Ok(Self {
            table: Arc::clone(&self.table),
            batches: Arc::clone(&self.batches),
            projection: pushdowns.projection,
            limit: pushdowns.limit.filter(|_| filters.is_empty()),
            filters,
            output,
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
            Some(paths) => {
                let (data, weights) = piece.into_parts();
                let data = krishiv_io::project_batch(&data, paths)?;
                Ok(Batch::with_weights(data, weights)?)
            }
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
        if self
            .limit
            .is_some_and(|l| g.returned.load(Ordering::Relaxed) >= l)
        {
            return Ok(None);
        }
        if local.batches.is_empty() {
            let Some(m) = g.cursor.claim() else {
                return Ok(None);
            };
            local.batches = g
                .morsels
                .get(m)
                .cloned()
                .ok_or_else(|| internal(format!("morsel {m} out of {}", g.morsels.len())))?;
            local.offset = 0;
            local.claimed += 1;
        }
        // One piece per call, even an empty input batch or one a pushed
        // filter emptied: the driver must reach a batch boundary (cancel,
        // abort and spill checks) after every piece read.
        let piece = self.next_piece(local, g.batch_rows)?;
        g.returned
            .fetch_add(piece.num_rows() as u64, Ordering::Relaxed);
        Ok(Some(piece))
    }
}

impl ScanPushdown for MemoryScan {
    fn table_schema(&self) -> SchemaRef {
        Arc::clone(&self.table)
    }

    fn filter_pushdown(&self, _filter: &PushedFilter) -> FilterPushdown {
        FilterPushdown::Exact
    }

    fn build(&self, pushdowns: ScanPushdowns) -> Result<Arc<dyn Source>> {
        Ok(Arc::new(self.with_pushdowns(pushdowns)?))
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
            returned: AtomicU64::new(0),
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
