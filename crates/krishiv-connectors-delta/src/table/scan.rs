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
//! - **Cancellation.** Every kernel call of a run (the re-listing, the deletion-vector reads)
//!   goes through the table's engine bound to the running query's cancellation
//!   ([`DeltaEngine::for_query`]), never the planning engine's token.

use std::any::Any;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::builder::make_view;
use arrow_array::{ArrayRef, BinaryViewArray, RecordBatch, RecordBatchOptions, UInt32Array};
use arrow_buffer::{Buffer, ScalarBuffer};
use arrow_schema::{DataType as ArrowType, Field, FieldRef, Schema, SchemaRef};
use delta_kernel::expressions::{Predicate, PredicateRef};
use krishiv_common::error::{FailedReadFileNoHint, KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::{
    FilterPushdown, GlobalState, LocalState, PushedFilter, ROW_FILE_COLUMN, ROW_POSITION_COLUMN,
    RuntimeFilters, ScanPushdown, ScanPushdowns, Source, SourceRows, TaskContext,
};
use krishiv_io::{
    ByteRange, ColumnPath, FileRef, FormatOptions, MetadataCols, MetadataNeeds, RowLevel,
    RowSelection, ScanRequest, ScanStream, SchemaMapper, VersionToken,
};
use krishiv_memory::Reservation;
use krishiv_types::batch::Batch;

use super::{Inner, ScanUnit, list_units, partition_array, predicate};
use crate::engine::DeltaEngine;
use crate::engine::columnar::{format, format_options};
use crate::error::from_kernel;

/// Bytes of one `RowSelection` range.
const RANGE_BYTES: usize = std::mem::size_of::<std::ops::Range<u64>>();

/// Peak bytes per deleted row while a deletion vector becomes a row selection (see
/// [`DeltaScanSource::open`]).
const DV_PEAK_BYTES_PER_ROW: usize = 2 * std::mem::size_of::<u64>() + RANGE_BYTES;

/// The scan of a [`DeltaTable`](super::DeltaTable), before pushdown.
#[derive(Debug, Clone)]
pub struct DeltaScan {
    inner: Arc<Inner>,
    /// The scan of a row-level statement: it reads the row id
    /// ([`ScanPushdowns::row_id`]) and says it [`supports_take`](ScanPushdown::supports_take),
    /// so [`krishiv_exec::planner::RowIdScan`] can wrap it (it never takes rows by position).
    row_ids: bool,
}

impl DeltaScan {
    pub(super) fn new(inner: Arc<Inner>) -> Self {
        Self {
            inner,
            row_ids: false,
        }
    }

    /// The scan of a row-level statement over `inner` (see the `row_ids` field).
    pub(crate) fn with_row_ids(inner: Arc<Inner>) -> Self {
        Self {
            inner,
            row_ids: true,
        }
    }

    /// The kernel predicate of `filters` (see the module docs), if any translates.
    #[must_use]
    pub fn kernel_predicate(&self, filters: &[PushedFilter]) -> Option<Predicate> {
        let schema = self.inner.snapshot.schema();
        let parts: Vec<Predicate> = filters
            .iter()
            // Stricter than krishiv-io's whole-unit rule (which prunes units by every filter even
            // after one that may raise): skipping a file by a filter written after a conjunct
            // that may raise could hide that conjunct's error, so pruning stops there.
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
        // TODO(08b-2): Delta row-tracking columns, once `MetadataCols` can name them, are refused
        // with `crate::features::row_tracking_refused` (UNSUPPORTED_FEATURE.TABLE_OPERATION).
        if p.selection.is_some() || (p.row_id && !self.row_ids) {
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
        // The row index within the file, counted over the file's rows (a deletion vector's
        // selection keeps the deleted rows' positions out, not their numbers).
        request.metadata_cols.row_index = p.row_id;
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
                // The decoder reads every nested field as nullable (Spark's rule for file
                // sources); a Delta table keeps its nested `NOT NULL` fields, and the plan is
                // typed by the table, so the column gets the table's nested nullability.
                let f = match inner.schema.field_with_name(f.name()) {
                    Ok(t) => Arc::new(
                        f.as_ref()
                            .clone()
                            .with_data_type(nested_like(f.data_type(), t.data_type())),
                    ),
                    Err(_) => Arc::clone(f),
                };
                fields.push(f);
                layout.push(Slot::Data(data));
                data += 1;
            }
        }
        if p.row_id {
            // The file scan's row id types (`FileScan::with_row_id`), appended last: the file
            // as the log names it, then the decoder's row index (the decoded batch's last column).
            fields.push(Arc::new(Field::new(
                ROW_FILE_COLUMN,
                ArrowType::BinaryView,
                false,
            )));
            layout.push(Slot::RowFile);
            fields.push(Arc::new(Field::new(
                ROW_POSITION_COLUMN,
                ArrowType::Int64,
                false,
            )));
            layout.push(Slot::Data(data));
        }
        Ok(DeltaScanSource {
            inner: Arc::clone(inner),
            request,
            layout,
            output: Arc::new(Schema::new(fields)),
            pushed: self.kernel_predicate(&p.filters).map(Arc::new),
            runtime: p.dynamic.clone(),
        })
    }
}

/// `decoded` with the nullability of the nested fields `table` declares: a struct field, list
/// element or map value of the same name (struct fields matched by name, as a pruned struct
/// keeps only some). Anything else keeps the decoded type.
fn nested_like(decoded: &ArrowType, table: &ArrowType) -> ArrowType {
    let field = |d: &FieldRef, t: &Field| {
        Arc::new(
            d.as_ref()
                .clone()
                .with_nullable(t.is_nullable())
                .with_data_type(nested_like(d.data_type(), t.data_type())),
        )
    };
    match (decoded, table) {
        (ArrowType::Struct(d), ArrowType::Struct(t)) => ArrowType::Struct(
            d.iter()
                .map(|f| match t.iter().find(|g| g.name() == f.name()) {
                    Some(g) => field(f, g),
                    None => Arc::clone(f),
                })
                .collect(),
        ),
        (ArrowType::List(d), ArrowType::List(t)) => ArrowType::List(field(d, t)),
        (ArrowType::LargeList(d), ArrowType::LargeList(t)) => ArrowType::LargeList(field(d, t)),
        (ArrowType::Map(d, sorted), ArrowType::Map(t, _)) => ArrowType::Map(field(d, t), *sorted),
        _ => decoded.clone(),
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

    fn supports_take(&self) -> bool {
        self.row_ids
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
    /// The row id's file: the path the log names, as bytes.
    RowFile,
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
    /// The join's runtime filters, applied last to each batch (or by the probe above when the
    /// planner handed the row stage to it).
    runtime: RuntimeFilters,
}

impl DeltaScanSource {
    /// The kernel predicate the scan pruned files with (the read set of a write, section 5.1).
    #[must_use]
    pub fn pushed_predicate(&self) -> Option<&PredicateRef> {
        self.pushed.as_ref()
    }

    /// The files the scan reads: the table's listing, or with a pushed predicate the kernel's
    /// pruned listing (on the log pool).
    async fn plan(&self, engine: &DeltaEngine) -> Result<Arc<[ScanUnit]>, KrishivError> {
        let Some(predicate) = self.pushed.clone() else {
            return Ok(Arc::clone(&self.inner.units));
        };
        let snapshot = Arc::clone(&self.inner.snapshot);
        let name = self.inner.name.clone();
        let units = engine
            .delta_blocking(move |e| {
                list_units(e, &snapshot, Some(predicate)).map_err(|err| from_kernel(&err, &name))
            })
            .await??;
        Ok(units.into())
    }

    async fn open(
        &self,
        engine: &DeltaEngine,
        local: &mut Local,
        unit: &ScanUnit,
    ) -> Result<(), KrishivError> {
        let url = engine.table_root().join(&unit.path).map_err(|e| {
            KrishivError::from(FailedReadFileNoHint {
                path: unit.path.clone(),
            })
            .with_cause(internal(format!(
                "the Delta log names an unresolvable path: {e}"
            )))
        })?;
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
            'dv: {
                // Reserved from the descriptor's cardinality before the vector is decoded, at the
                // peak: the deleted rows (8 B each), `RowSelection::excluding`'s filtered copy of them
                // (8 B) and its ranges (16 B per range, at most one more than the rows).
                let name = self.inner.name.clone();
                let cardinality = unit
                    .dv
                    .cardinality()
                    .map_err(|err| from_kernel(&err, &name))?
                    .unwrap_or(0);
                let slots = usize::try_from(cardinality)
                    .unwrap_or(usize::MAX)
                    .saturating_add(1);
                let peak = slots.saturating_mul(DV_PEAK_BYTES_PER_ROW);
                dv_memory.try_grow(peak)?;
                let dv = unit.dv.clone();
                let root = engine.table_root().clone();
                let deleted = engine
                    .delta_blocking(move |e| {
                        dv.get_row_indexes(e, &root)
                            .map_err(|err| from_kernel(&err, &name))
                    })
                    .await??
                    .unwrap_or_default();
                let meta = match engine
                    .io()
                    .read_metadata(fmt.as_ref(), &file, MetadataNeeds::default(), &options)
                    .await
                {
                    Ok(meta) => meta,
                    // A missing or corrupt file the scan options skip: no selection, and the file's
                    // own open below applies the same rule, as for a file without a DV.
                    Err(e) if self.inner.scan_options.skips(&e) => {
                        dv_memory.shrink(dv_memory.size());
                        break 'dv &self.request;
                    }
                    Err(e) => return Err(e),
                };
                let rows = meta
                    .statistics()
                    .rows
                    .exact()
                    .copied()
                    .or(unit.records)
                    .ok_or_else(|| {
                        unsupported("a Delta deletion vector on a file of unknown row count")
                    })?;
                let selection = RowSelection::excluding(rows, deleted);
                // What stays held while the file is read: the selection's range vector, whose
                // capacity `RowSelection::excluding` sizes for one more range than deleted rows,
                // however few ranges a contiguous delete leaves.
                let kept = slots.saturating_mul(RANGE_BYTES);
                dv_memory.shrink(dv_memory.size().saturating_sub(kept));
                request = self.request.clone();
                request.selection = Some(selection);
                &request
            }
        } else {
            &self.request
        };
        let partitions = self
            .layout
            .iter()
            .filter_map(|s| match s {
                Slot::Partition(k) => Some(*k),
                Slot::Data(_) | Slot::RowFile => None,
            })
            .map(|k| {
                let pc = &self.inner.partitions[k];
                Ok((
                    k,
                    partition_array(
                        pc,
                        unit.partition_values.get(&pc.physical),
                        &self.inner.name,
                    )?,
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
            row_file: Buffer::from(unit.path.as_bytes().to_vec()),
            _dv_memory: dv_memory,
        });
        Ok(())
    }

    /// The decoded `batch` with the partition columns between its data columns.
    fn assemble(
        &self,
        batch: &RecordBatch,
        partitions: &[(usize, ArrayRef)],
        row_file: &Buffer,
    ) -> Result<RecordBatch, KrishivError> {
        let rows = batch.num_rows();
        let zeros = UInt32Array::from(vec![0_u32; rows]);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.layout.len());
        for slot in &self.layout {
            columns.push(match *slot {
                Slot::Data(i) => {
                    let column = batch.column(i);
                    let to = self.output.field(columns.len()).data_type();
                    crate::engine::relabel::relabel(column, to)
                        .map_err(|e| internal(format!("a Delta column as the table's type: {e}")))?
                }
                Slot::Partition(k) => {
                    let value = partitions
                        .iter()
                        .find(|(p, _)| *p == k)
                        .map(|(_, v)| v)
                        .ok_or_else(|| internal(format!("partition column {k} without a value")))?;
                    arrow_select::take::take(value, &zeros, None)
                        .map_err(|e| internal(format!("a partition column of {rows} rows: {e}")))?
                }
                Slot::RowFile => {
                    // Every row views the one buffer.
                    let view = make_view(row_file.as_slice(), 0, 0);
                    Arc::new(
                        BinaryViewArray::try_new(
                            ScalarBuffer::from(vec![view; rows]),
                            vec![row_file.clone()],
                            None,
                        )
                        .map_err(|e| internal(format!("a row file column: {e}")))?,
                    )
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
                        let b = self.assemble(&b, &cur.partitions, &cur.row_file)?;
                        // After every other filter, as Spark evaluates a runtime filter last.
                        let b = local.rows.apply(b)?;
                        return Ok(Some(Batch::new(b)));
                    }
                    None => {
                        local.current = None;
                        continue;
                    }
                }
            }
            let units = global
                .units
                .get_or_try_init(|| self.plan(&global.engine))
                .await?;
            let i = global.next.fetch_add(1, Ordering::Relaxed);
            let Some(unit) = units.get(i) else {
                return Ok(None);
            };
            local.claimed += 1;
            // Runtime filter decisions are per file.
            local.rows.start_partition();
            self.open(&global.engine, local, unit).await?;
        }
    }
}

struct Global {
    /// The table's engine bound to the running query's cancellation.
    engine: DeltaEngine,
    units: tokio::sync::OnceCell<Arc<[ScanUnit]>>,
    next: AtomicUsize,
}

struct Current {
    stream: ScanStream,
    partitions: Vec<(usize, ArrayRef)>,
    /// The file's path, the row id's file.
    row_file: Buffer,
    /// The deletion vector's bytes, held while the file is read.
    _dv_memory: Reservation,
}

struct Local {
    /// An empty reservation of the driver's consumer; every read's reservation splits off it.
    memory: Reservation,
    gate: Arc<krishiv_io::ScanGate>,
    current: Option<Current>,
    claimed: u64,
    rows: SourceRows,
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

    fn global_state(&self, ctx: &TaskContext) -> krishiv_exec::Result<GlobalState> {
        Ok(Arc::new(Global {
            engine: self.inner.engine.for_query(ctx.query()),
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
            rows: self.runtime.source_rows(&self.output, ctx.partition()),
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

#[cfg(test)]
mod nested_like_tests {
    use std::sync::Arc;

    use arrow_schema::{DataType as ArrowType, Field, Fields};

    use super::nested_like;

    #[test]
    fn a_struct_field_the_table_lacks_keeps_its_decoded_nullability() {
        let decoded = ArrowType::Struct(Fields::from(vec![
            Field::new("a", ArrowType::Int32, true),
            Field::new("extra", ArrowType::Utf8, true),
        ]));
        let table = ArrowType::Struct(Fields::from(vec![Field::new("a", ArrowType::Int32, false)]));
        let ArrowType::Struct(out) = nested_like(&decoded, &table) else {
            panic!("a struct stays a struct");
        };
        assert!(
            !out[0].is_nullable(),
            "a matched field takes the table's nullability"
        );
        assert!(
            out[1].is_nullable(),
            "an unmatched field keeps the decoded nullability"
        );
    }

    #[test]
    fn a_large_list_element_takes_the_table_nullability() {
        let decoded = ArrowType::LargeList(Arc::new(Field::new("element", ArrowType::Int64, true)));
        let table = ArrowType::LargeList(Arc::new(Field::new("element", ArrowType::Int64, false)));
        let ArrowType::LargeList(out) = nested_like(&decoded, &table) else {
            panic!("a large list stays a large list");
        };
        assert!(!out.is_nullable());
    }
}
