// SPDX-License-Identifier: Apache-2.0
//! The Delta scan: [`DeltaScan`] (what the planner pushes into) and the [`DeltaScanSource`] it
//! builds (design note P3-T021 section 3.4).
//!
//! - **Planning.** Pushed filters are translated into one kernel predicate
//!   ([`super::predicate`]), up to the first filter with a conjunct that may raise (the rows
//!   that conjunct must see are never pruned). Every translatable filter is
//!   [`FilterPushdown::Inexact`], every other one [`FilterPushdown::Unsupported`]: the filter
//!   above the scan always stays. With a predicate, the source lists the snapshot's files again
//!   through the kernel when it starts (on the log pool, once per run), so the kernel prunes by
//!   partition values and statistics; without one it reads the table's listing. The predicate
//!   is recorded on the source ([`DeltaScanSource::pushed_predicate`]) for the read set.
//! - **Data.** One unit per file, claimed by drivers in the kernel's order, read whole through
//!   krishiv-io's [`ScanStream`] with the table's field mapping (Delta column mapping by id or
//!   physical name) and its schema adaptation (type widening). Partition columns are the
//!   kernel-parsed values of the file's `partitionValues`, repeated, never read from the file
//!   or its path. A deletion vector is read through the kernel (`DvInfo::get_row_indexes`, on
//!   the log pool), charged to the driver's memory reservation while the file is read, and
//!   applied as the request's row selection. Kernel transforms are never evaluated on data.

use std::any::Any;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions, UInt32Array};
use arrow_schema::{FieldRef, Schema, SchemaRef};
use delta_kernel::expressions::{Predicate, PredicateRef};
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::{
    FilterPushdown, GlobalState, LocalState, PushedFilter, ScanPushdown, ScanPushdowns, Source,
    TaskContext,
};
use krishiv_io::{
    ByteRange, ColumnPath, FileRef, FormatOptions, MetadataCols, MetadataNeeds, RowLevel,
    RowSelection, ScanRequest, ScanStream, SchemaMapper, VersionToken,
};
use krishiv_memory::Reservation;
use krishiv_types::batch::Batch;

use super::{Inner, ScanUnit, list_units, partition_array, predicate};
use crate::engine::columnar::{format, format_options};
use crate::error::from_kernel;

/// The scan of a [`DeltaTable`](super::DeltaTable), before pushdown.
#[derive(Debug, Clone)]
pub struct DeltaScan {
    inner: Arc<Inner>,
}

impl DeltaScan {
    pub(super) fn new(inner: Arc<Inner>) -> Self {
        Self { inner }
    }

    /// The kernel predicate of `filters` (see the module docs), if any translates.
    #[must_use]
    pub fn kernel_predicate(&self, filters: &[PushedFilter]) -> Option<Predicate> {
        let schema = self.inner.snapshot.schema();
        let parts: Vec<Predicate> = filters
            .iter()
            .take_while(|f| matches!(f.rows, RowLevel::Full))
            .filter_map(|f| predicate::to_kernel(f.expr.as_ref()?, &schema))
            .collect();
        match parts.len() {
            0 => None,
            1 => parts.into_iter().next(),
            _ => Some(Predicate::and_from(parts)),
        }
    }

    /// The source of `p`, typed.
    ///
    /// # Errors
    /// As [`ScanPushdown::build`].
    pub fn source(&self, p: &ScanPushdowns) -> Result<DeltaScanSource, KrishivError> {
        if p.selection.is_some() || p.row_id {
            return Err(unsupported("row selections and row ids over a Delta scan"));
        }
        if p.metadata != MetadataCols::default() {
            return Err(unsupported("_metadata columns of a Delta table"));
        }
        let inner = &self.inner;
        let projection: Vec<ColumnPath> = p.projection.clone().unwrap_or_else(|| {
            inner
                .schema
                .fields()
                .iter()
                .map(|f| ColumnPath::new(f.name()))
                .collect()
        });
        let partition_of = |c: &ColumnPath| {
            inner
                .partitions
                .iter()
                .position(|pc| c.0.len() == 1 && pc.field.name() == c.root())
        };
        let data_table: SchemaRef = Arc::new(Schema::new(
            inner
                .schema
                .fields()
                .iter()
                .filter(|f| {
                    !inner
                        .partitions
                        .iter()
                        .any(|pc| pc.field.name() == f.name())
                })
                .cloned()
                .collect::<Vec<_>>(),
        ));
        let data_projection: Vec<ColumnPath> = projection
            .iter()
            .filter(|c| partition_of(c).is_none())
            .cloned()
            .collect();
        let mut request = ScanRequest::new(Arc::clone(&data_table), data_projection);
        request.mapping = inner.mapping.clone();
        let decoded = Arc::clone(SchemaMapper::new(&request, data_table, "")?.output_schema());
        // One output column per distinct top-level column of the projection, in order of
        // first appearance (nested paths of one column merge into it), as the file scan does.
        let mut fields: Vec<FieldRef> = Vec::new();
        let mut layout = Vec::new();
        let mut seen: Vec<&str> = Vec::new();
        let mut data = 0;
        for c in &projection {
            if seen.contains(&c.root()) {
                continue;
            }
            seen.push(c.root());
            if let Some(k) = partition_of(c) {
                fields.push(Arc::new(
                    inner.partitions[k]
                        .field
                        .as_ref()
                        .clone()
                        .with_nullable(true),
                ));
                layout.push(Slot::Partition(k));
            } else {
                let f = decoded.fields().get(data).ok_or_else(|| {
                    internal(format!("projection of {} has no decoded column", c.root()))
                })?;
                fields.push(Arc::clone(f));
                layout.push(Slot::Data(data));
                data += 1;
            }
        }
        Ok(DeltaScanSource {
            inner: Arc::clone(inner),
            request,
            layout,
            output: Arc::new(Schema::new(fields)),
            pushed: self.kernel_predicate(&p.filters).map(Arc::new),
        })
    }
}

fn unsupported(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}

impl ScanPushdown for DeltaScan {
    fn table_schema(&self) -> SchemaRef {
        Arc::clone(&self.inner.schema)
    }

    fn filter_pushdown(&self, filter: &PushedFilter) -> FilterPushdown {
        let schema = self.inner.snapshot.schema();
        match filter
            .expr
            .as_ref()
            .and_then(|e| predicate::to_kernel(e, &schema))
        {
            Some(_) => FilterPushdown::Inexact,
            None => FilterPushdown::Unsupported,
        }
    }

    fn build(&self, pushdowns: ScanPushdowns) -> krishiv_exec::Result<Arc<dyn Source>> {
        Ok(Arc::new(self.source(&pushdowns)?))
    }

    fn output_schema(&self, pushdowns: &ScanPushdowns) -> krishiv_exec::Result<SchemaRef> {
        Ok(Arc::clone(&self.source(pushdowns)?.output))
    }
}

/// Where a column of the output comes from.
#[derive(Clone, Copy, Debug)]
enum Slot {
    /// The decoded batch's column.
    Data(usize),
    /// The partition column (index into the table's partition columns).
    Partition(usize),
}

/// The source of a Delta scan (see the module docs).
#[derive(Debug)]
pub struct DeltaScanSource {
    inner: Arc<Inner>,
    /// The table-level request every file reads (the data columns only).
    request: ScanRequest,
    layout: Vec<Slot>,
    output: SchemaRef,
    pushed: Option<PredicateRef>,
}

impl DeltaScanSource {
    /// The kernel predicate the scan pruned files with (the read set of a write, section 5.1).
    #[must_use]
    pub fn pushed_predicate(&self) -> Option<&PredicateRef> {
        self.pushed.as_ref()
    }

    /// The files the scan reads: the table's listing, or with a pushed predicate the kernel's
    /// pruned listing (on the log pool).
    async fn plan(&self) -> Result<Arc<[ScanUnit]>, KrishivError> {
        let Some(predicate) = self.pushed.clone() else {
            return Ok(Arc::clone(&self.inner.units));
        };
        let snapshot = Arc::clone(&self.inner.snapshot);
        let name = self.inner.name.clone();
        let units = self
            .inner
            .engine
            .delta_blocking(move |e| {
                list_units(e, &snapshot, Some(predicate)).map_err(|err| from_kernel(&err, &name))
            })
            .await??;
        Ok(units.into())
    }

    async fn open(&self, local: &mut Local, unit: &ScanUnit) -> Result<(), KrishivError> {
        let engine = &self.inner.engine;
        let url = engine
            .table_root()
            .join(&unit.path)
            .map_err(|e| unsupported(&format!("the Delta data file path {}: {e}", unit.path)))?;
        let path = engine.root().confine(&url)?;
        let file = FileRef {
            store: engine.store().clone(),
            path,
            size: unit.size,
            // Delta data files are never rewritten in place.
            version: VersionToken::Immutable,
            modified_micros: unit.modification_time.checked_mul(1000),
        };
        let fmt = format()?;
        let options: FormatOptions = format_options()?;
        // A refusal fails the query, so the growth is unguarded (krishiv-memory's rule).
        let mut dv_memory = local.memory.split(0).unguarded();
        let mut request;
        let req = if unit.dv.has_vector() {
            let dv = unit.dv.clone();
            let root = engine.table_root().clone();
            let name = self.inner.name.clone();
            let deleted = engine
                .delta_blocking(move |e| {
                    dv.get_row_indexes(e, &root)
                        .map_err(|err| from_kernel(&err, &name))
                })
                .await??
                .unwrap_or_default();
            // The vector and the selection made of it, while the file is read.
            let bytes = deleted
                .len()
                .saturating_add(1)
                .saturating_mul(std::mem::size_of::<std::ops::Range<u64>>() + 8);
            dv_memory.try_grow(bytes)?;
            let meta = engine
                .io()
                .read_metadata(fmt.as_ref(), &file, MetadataNeeds::default(), &options)
                .await?;
            let rows = meta
                .statistics()
                .rows
                .exact()
                .copied()
                .or(unit.records)
                .ok_or_else(|| {
                    unsupported("a Delta deletion vector on a file of unknown row count")
                })?;
            request = self.request.clone();
            request.selection = Some(RowSelection::excluding(rows, deleted));
            &request
        } else {
            &self.request
        };
        let partitions = self
            .layout
            .iter()
            .filter_map(|s| match s {
                Slot::Partition(k) => Some(*k),
                Slot::Data(_) => None,
            })
            .map(|k| {
                let pc = &self.inner.partitions[k];
                Ok((
                    k,
                    partition_array(pc, unit.partition_values.get(&pc.physical))?,
                ))
            })
            .collect::<Result<Vec<_>, KrishivError>>()?;
        let stream = ScanStream::open(
            engine.io(),
            fmt.as_ref(),
            file,
            ByteRange::whole(unit.size),
            req,
            &options,
            local.memory.split(0),
            self.inner.scan_options,
        )
        .await?
        .with_gate(Arc::clone(&local.gate));
        local.current = Some(Current {
            stream,
            partitions,
            _dv_memory: dv_memory,
        });
        Ok(())
    }

    /// The decoded `batch` with the partition columns between its data columns.
    fn assemble(
        &self,
        batch: &RecordBatch,
        partitions: &[(usize, ArrayRef)],
    ) -> Result<RecordBatch, KrishivError> {
        let rows = batch.num_rows();
        let zeros = UInt32Array::from(vec![0_u32; rows]);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.layout.len());
        for slot in &self.layout {
            columns.push(match *slot {
                Slot::Data(i) => Arc::clone(batch.column(i)),
                Slot::Partition(k) => {
                    let value = partitions
                        .iter()
                        .find(|(p, _)| *p == k)
                        .map(|(_, v)| v)
                        .ok_or_else(|| internal(format!("partition column {k} without a value")))?;
                    arrow_select::take::take(value, &zeros, None)
                        .map_err(|e| internal(format!("a partition column of {rows} rows: {e}")))?
                }
            });
        }
        RecordBatch::try_new_with_options(
            Arc::clone(&self.output),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .map_err(|e| internal(format!("a Delta scan batch: {e}")))
    }

    async fn next(
        &self,
        local: &mut Local,
        global: &Global,
    ) -> Result<Option<Batch>, KrishivError> {
        loop {
            if let Some(cur) = local.current.as_mut() {
                match cur.stream.next_batch().await? {
                    Some(b) => {
                        let b = self.assemble(&b, &cur.partitions)?;
                        return Ok(Some(Batch::new(b)));
                    }
                    None => {
                        local.current = None;
                        continue;
                    }
                }
            }
            let units = global.units.get_or_try_init(|| self.plan()).await?;
            let i = global.next.fetch_add(1, Ordering::Relaxed);
            let Some(unit) = units.get(i) else {
                return Ok(None);
            };
            local.claimed += 1;
            self.open(local, unit).await?;
        }
    }
}

struct Global {
    units: tokio::sync::OnceCell<Arc<[ScanUnit]>>,
    next: AtomicUsize,
}

struct Current {
    stream: ScanStream,
    partitions: Vec<(usize, ArrayRef)>,
    /// The deletion vector's bytes, held while the file is read.
    _dv_memory: Reservation,
}

struct Local {
    /// An empty reservation of the driver's consumer; every read's reservation splits off it.
    memory: Reservation,
    gate: Arc<krishiv_io::ScanGate>,
    current: Option<Current>,
    claimed: u64,
}

fn state_error(what: &str) -> KrishivError {
    internal(format!(
        "a Delta scan given the {what} state of another source"
    ))
}

fn internal(message: String) -> KrishivError {
    krishiv_common::error::InternalError { message }.into()
}

impl Source for DeltaScanSource {
    fn name(&self) -> &'static str {
        "DeltaScan"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.output)
    }

    fn global_state(&self, _ctx: &TaskContext) -> krishiv_exec::Result<GlobalState> {
        Ok(Arc::new(Global {
            units: tokio::sync::OnceCell::new(),
            next: AtomicUsize::new(0),
        }))
    }

    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        self.inner.units.len().max(1)
    }

    fn local_state(
        &self,
        ctx: &TaskContext,
        _global: &GlobalState,
    ) -> krishiv_exec::Result<LocalState> {
        Ok(Box::new(Local {
            memory: ctx.consumer().reservation(),
            gate: Arc::clone(ctx.query().scan_gate()),
            current: None,
            claimed: 0,
        }))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> Pin<Box<dyn Future<Output = krishiv_exec::Result<Option<Batch>>> + Send + 'a>> {
        Box::pin(async move {
            let global = (global.as_ref() as &dyn Any)
                .downcast_ref::<Global>()
                .ok_or_else(|| state_error("global"))?;
            let local = local
                .downcast_mut::<Local>()
                .ok_or_else(|| state_error("local"))?;
            self.next(local, global).await
        })
    }

    fn morsels_claimed(&self, local: &LocalState) -> u64 {
        local.downcast_ref::<Local>().map_or(0, |l| l.claimed)
    }
}
