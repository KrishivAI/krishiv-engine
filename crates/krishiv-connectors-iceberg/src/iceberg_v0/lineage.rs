// SPDX-License-Identifier: Apache-2.0
//! v3 row lineage on read and on rewrite (P3-T027, PR-36b unit 2a; design note
//! §8 item 3, §11 item 13): the metadata columns `_row_id` (field id
//! 2147483540) and `_last_updated_sequence_number` (2147483539), as the
//! Iceberg table spec's "Row Lineage" section and its v3 changes list define
//! them (`spec.md`, apache-iceberg 1.12.0, the reserved field ids table and
//! "Row lineage assignment").
//!
//! **Read** ([`LineageSource`], through the provider's lineage scan): for a
//! data file with a non-null `first_row_id` a stored non-null value is read
//! unchanged; a null one, or the column missing from the file, inherits
//! `first_row_id + _pos` (`_row_id`) and the file's data sequence number
//! (`_last_updated_sequence_number`). A data file with a null `first_row_id`
//! (written before the table's upgrade to v3) reads null for both, whatever
//! the file stores ("When a data file has a null `first_row_id`, readers must
//! produce null for `_row_id` and `_last_updated_sequence_number`"). A file
//! whose data sequence number the manifest leaves null reads a null
//! `_last_updated_sequence_number` where none is stored (the spec gives no
//! value to inherit). Deletes apply first: the values are derived on the rows
//! the delete-aware source returns, with `_pos` the row's position in its
//! file (`_krishiv_row_index`), so a survivor keeps the id it had.
//!
//! **Write** ([`super::write::DataFileWriter::with_lineage`]): the columns a
//! rewrite carries are written under their reserved ids, optional longs, so a
//! moved row keeps its values ("When writing an existing row into a new data
//! file, writers must write `_row_id` and `_last_updated_sequence_number` if
//! they are non-null"); a column the write does not carry is not written, and
//! its rows inherit on read. The write decides per row what to carry (an
//! updated row a null sequence number, an inserted row nulls for both): that
//! is the statement's (36b unit 2).
//!
//! Only format v3 and later have row lineage ("In v3 and later, an Iceberg
//! table must track row lineage fields"); on v1 and v2 tables the columns are
//! refused (`KRISHIV_UNSUPPORTED_FEATURE`), never invented. Whether Spark's
//! Iceberg catalog hides them or fails a query naming them on v1/v2 is not
//! verified (its `SparkTable` source was not supplied).

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, RecordBatchOptions};
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
use futures::future::BoxFuture;
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::{
    GlobalState, LocalState, ROW_FILE_COLUMN, ROW_POSITION_COLUMN, Source, TaskContext,
};
use krishiv_io::{ColumnPath, FilterId, ScalarValue};
use krishiv_types::batch::Batch;

use crate::plan::DataFileEntry;

/// `_row_id`'s reserved field id.
pub(crate) const ROW_ID_FIELD_ID: i32 = 2_147_483_540;
/// `_last_updated_sequence_number`'s reserved field id.
pub(crate) const LAST_UPDATED_FIELD_ID: i32 = 2_147_483_539;
/// The metadata column of a row's id.
pub(crate) const ROW_ID: &str = "_row_id";
/// The metadata column of the sequence number that last updated a row.
pub(crate) const LAST_UPDATED: &str = "_last_updated_sequence_number";

/// The value stored in the file under [`ROW_ID_FIELD_ID`] (null when the file
/// lacks the column), an input of the inner scan.
pub(super) const STORED_ROW_ID: &str = "_krishiv_iceberg_stored_row_id";
/// The value stored under [`LAST_UPDATED_FIELD_ID`].
pub(super) const STORED_LAST_UPDATED: &str = "_krishiv_iceberg_stored_last_updated";
/// The file's `first_row_id` (per-file constant).
pub(super) const FIRST_ROW_ID: &str = "_krishiv_iceberg_first_row_id";
/// The file's data sequence number (per-file constant).
pub(super) const DATA_SEQUENCE: &str = "_krishiv_iceberg_data_sequence";

/// The inner scan's lineage inputs, in the order it reads them.
pub(super) const INPUTS: [&str; 4] = [
    STORED_ROW_ID,
    STORED_LAST_UPDATED,
    FIRST_ROW_ID,
    DATA_SEQUENCE,
];

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// An optional long carrying Iceberg field id `id`.
fn long_with_id(name: &str, id: i32) -> Field {
    Field::new(name, DataType::Int64, true).with_metadata(HashMap::from([(
        krishiv_io::FIELD_ID_KEY.to_owned(),
        id.to_string(),
    )]))
}

/// The two metadata columns as a lineage scan exposes and a rewrite writes
/// them: optional longs under their reserved ids.
pub(crate) fn lineage_fields() -> [FieldRef; 2] {
    [
        Arc::new(long_with_id(ROW_ID, ROW_ID_FIELD_ID)),
        Arc::new(long_with_id(LAST_UPDATED, LAST_UPDATED_FIELD_ID)),
    ]
}

/// The per-file constant columns of a lineage scan (`first_row_id`, data
/// sequence number), nullable longs.
pub(super) fn constant_fields() -> [FieldRef; 2] {
    [
        Arc::new(Field::new(FIRST_ROW_ID, DataType::Int64, true)),
        Arc::new(Field::new(DATA_SEQUENCE, DataType::Int64, true)),
    ]
}

/// `read` (the inner file scan's table schema) with the stored values (by
/// their reserved ids, so a file without them reads null) and the per-file
/// constants appended.
pub(super) fn with_inputs(read: &SchemaRef) -> SchemaRef {
    let mut fields: Vec<FieldRef> = read.fields().iter().cloned().collect();
    fields.push(Arc::new(long_with_id(STORED_ROW_ID, ROW_ID_FIELD_ID)));
    fields.push(Arc::new(long_with_id(
        STORED_LAST_UPDATED,
        LAST_UPDATED_FIELD_ID,
    )));
    fields.extend(constant_fields());
    Arc::new(Schema::new_with_metadata(fields, read.metadata().clone()))
}

/// The per-file constant values of `e`, in [`constant_fields`] order.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the seam 36b unit 2's copy-on-write rewrite calls"
    )
)]
pub(super) fn constants_of(e: &DataFileEntry) -> [ScalarValue; 2] {
    let long = |v: Option<i64>| v.map_or(ScalarValue::Null, ScalarValue::Int64);
    [long(e.file.first_row_id), long(e.data_sequence_number)]
}

/// The lineage columns are refused on a table of `format_version` below 3
/// (see the module docs) and when the table has a column of their name.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE`.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the seam 36b unit 2's copy-on-write rewrite calls"
    )
)]
pub(crate) fn check_table(format_version: u8, columns: &Schema) -> Result<(), KrishivError> {
    if format_version < 3 {
        return Err(KrishivUnsupportedFeature {
            feature: format!(
                "the Iceberg row lineage columns {ROW_ID} and {LAST_UPDATED} on a format \
                 version {format_version} table (row lineage starts at format version 3)"
            ),
        }
        .into());
    }
    if let Some(f) = columns
        .fields()
        .iter()
        .find(|f| f.name() == ROW_ID || f.name() == LAST_UPDATED)
    {
        return Err(KrishivUnsupportedFeature {
            feature: format!(
                "the Iceberg row lineage columns on a table with a column named {}",
                f.name()
            ),
        }
        .into());
    }
    Ok(())
}

/// `_row_id` per the spec: null when the file has no `first_row_id`, else
/// the stored value, else `first_row_id + pos`.
///
/// # Errors
/// `INTERNAL_ERROR` when the sum overflows a long.
pub(crate) fn derive_row_ids(
    stored: &Int64Array,
    first_row_id: &Int64Array,
    pos: &Int64Array,
) -> Result<Int64Array, KrishivError> {
    let mut out = Int64Array::builder(stored.len());
    for i in 0..stored.len() {
        if first_row_id.is_null(i) {
            out.append_null();
        } else if stored.is_valid(i) {
            out.append_value(stored.value(i));
        } else {
            let id = first_row_id
                .value(i)
                .checked_add(pos.value(i))
                .ok_or_else(|| internal("a row id past the long range".to_owned()))?;
            out.append_value(id);
        }
    }
    Ok(out.finish())
}

/// `_last_updated_sequence_number` per the spec: null when the file has no
/// `first_row_id`, else the stored value, else the file's data sequence
/// number (null when the manifest leaves it null).
#[must_use]
pub(crate) fn derive_last_updated(
    stored: &Int64Array,
    first_row_id: &Int64Array,
    data_sequence: &Int64Array,
) -> Int64Array {
    (0..stored.len())
        .map(|i| {
            if first_row_id.is_null(i) {
                None
            } else if stored.is_valid(i) {
                Some(stored.value(i))
            } else {
                data_sequence.is_valid(i).then(|| data_sequence.value(i))
            }
        })
        .collect()
}

/// Where a column of a lineage scan's output comes from.
#[derive(Debug, Clone, Copy)]
enum Slot {
    /// The inner batch's column.
    Inner(usize),
    RowId,
    LastUpdated,
}

/// How a lineage scan's output is made from its inner scan's batches.
#[derive(Debug, Clone)]
pub(super) struct LineageProjection {
    slots: Vec<Slot>,
    /// The inner columns: stored row id, stored sequence number, first row
    /// id, data sequence number, row position.
    inputs: [usize; 5],
    schema: SchemaRef,
}

/// What a lineage scan's caller asked for, split into the inner scan's
/// projection and the output's layout.
#[derive(Debug, Clone)]
pub(super) struct LineageSplit {
    /// The caller's top-level columns in output order (first appearance).
    roots: Vec<String>,
    /// The inner projection.
    pub(super) inner: Vec<ColumnPath>,
    /// Whether either lineage column is asked for.
    pub(super) wanted: bool,
    /// Whether the caller asked for the row id columns itself.
    caller_row_id: bool,
}

impl LineageSplit {
    /// The split of `projection` (`None`: every column of `table` and both
    /// lineage columns). With `caller_row_id` the output ends with the row
    /// id columns.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a nested path into a lineage column.
    pub(super) fn new(
        table: &Schema,
        projection: Option<&[ColumnPath]>,
        caller_row_id: bool,
    ) -> Result<Self, KrishivError> {
        let all: Vec<ColumnPath>;
        let projection = match projection {
            Some(p) => p,
            None => {
                all = table
                    .fields()
                    .iter()
                    .map(|f| ColumnPath::new(f.name()))
                    .chain([ColumnPath::new(ROW_ID), ColumnPath::new(LAST_UPDATED)])
                    .collect();
                &all
            }
        };
        let is_lineage = |c: &ColumnPath| c.root() == ROW_ID || c.root() == LAST_UPDATED;
        if let Some(c) = projection.iter().find(|c| is_lineage(c) && c.0.len() > 1) {
            return Err(internal(format!(
                "a path into a long column: {}",
                c.0.join(".")
            )));
        }
        let mut roots: Vec<String> = Vec::new();
        for c in projection {
            if !roots.iter().any(|r| r == c.root()) {
                roots.push(c.root().to_owned());
            }
        }
        let wanted = projection.iter().any(is_lineage);
        let mut inner: Vec<ColumnPath> = projection
            .iter()
            .filter(|c| !is_lineage(c))
            .cloned()
            .collect();
        if wanted {
            inner.extend(INPUTS.iter().map(|n| ColumnPath::new(*n)));
        }
        Ok(Self {
            roots,
            inner,
            wanted,
            caller_row_id,
        })
    }

    /// The projection of batches of `inner` (the inner source's schema).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when a column is not in `inner`.
    pub(super) fn projection(&self, inner: &Schema) -> Result<LineageProjection, KrishivError> {
        let find = |name: &str| {
            inner
                .index_of(name)
                .map_err(|e| internal(format!("a lineage scan's inner column {name}: {e}")))
        };
        let [row_id_field, last_updated_field] = lineage_fields();
        let mut slots = Vec::new();
        let mut fields: Vec<FieldRef> = Vec::new();
        for r in &self.roots {
            if r == ROW_ID {
                slots.push(Slot::RowId);
                fields.push(Arc::clone(&row_id_field));
            } else if r == LAST_UPDATED {
                slots.push(Slot::LastUpdated);
                fields.push(Arc::clone(&last_updated_field));
            } else {
                let i = find(r)?;
                slots.push(Slot::Inner(i));
                fields.push(Arc::new(inner.field(i).clone()));
            }
        }
        if self.caller_row_id {
            for name in [ROW_FILE_COLUMN, ROW_POSITION_COLUMN] {
                let i = find(name)?;
                slots.push(Slot::Inner(i));
                fields.push(Arc::new(inner.field(i).clone()));
            }
        }
        let inputs = [
            find(STORED_ROW_ID)?,
            find(STORED_LAST_UPDATED)?,
            find(FIRST_ROW_ID)?,
            find(DATA_SEQUENCE)?,
            find(ROW_POSITION_COLUMN)?,
        ];
        Ok(LineageProjection {
            slots,
            inputs,
            schema: Arc::new(Schema::new_with_metadata(fields, inner.metadata().clone())),
        })
    }
}

impl LineageProjection {
    /// The output schema.
    pub(super) fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// The output batch of an inner batch.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for an input of another type and a row id past the
    /// long range.
    pub(super) fn apply(&self, batch: &RecordBatch) -> Result<RecordBatch, KrishivError> {
        let long = |i: usize| -> Result<&Int64Array, KrishivError> {
            batch
                .column(i)
                .as_primitive_opt::<Int64Type>()
                .ok_or_else(|| {
                    internal(format!(
                        "a lineage input of type {}",
                        batch.column(i).data_type()
                    ))
                })
        };
        let [stored_id, stored_seq, first, seq, pos] = self.inputs;
        let wants = |s: fn(&Slot) -> bool| self.slots.iter().any(s);
        let row_ids: Option<ArrayRef> = if wants(|s| matches!(s, Slot::RowId)) {
            Some(Arc::new(derive_row_ids(
                long(stored_id)?,
                long(first)?,
                long(pos)?,
            )?))
        } else {
            None
        };
        let last_updated: Option<ArrayRef> = if wants(|s| matches!(s, Slot::LastUpdated)) {
            Some(Arc::new(derive_last_updated(
                long(stored_seq)?,
                long(first)?,
                long(seq)?,
            )))
        } else {
            None
        };
        let missing = || internal("a lineage column not derived".to_owned());
        let columns = self
            .slots
            .iter()
            .map(|s| match s {
                Slot::Inner(i) => Ok(Arc::clone(batch.column(*i))),
                Slot::RowId => row_ids.clone().ok_or_else(missing),
                Slot::LastUpdated => last_updated.clone().ok_or_else(missing),
            })
            .collect::<Result<Vec<_>, KrishivError>>()?;
        RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )
        .map_err(|e| internal(format!("a lineage batch: {e}")))
    }
}

/// A delete-aware scan's source with the lineage columns derived on its
/// batches ([`LineageProjection`]).
pub(super) struct LineageSource {
    inner: Arc<dyn Source>,
    projection: LineageProjection,
}

impl fmt::Debug for LineageSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LineageSource")
            .field("schema", &self.projection.schema)
            .finish_non_exhaustive()
    }
}

impl LineageSource {
    /// The source over `inner`, its batches projected by `projection`.
    pub(super) fn new(inner: Arc<dyn Source>, projection: LineageProjection) -> Self {
        Self { inner, projection }
    }
}

impl Source for LineageSource {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn schema(&self) -> SchemaRef {
        self.projection.schema()
    }

    fn global_state(&self, ctx: &TaskContext) -> krishiv_exec::Result<GlobalState> {
        self.inner.global_state(ctx)
    }

    fn max_parallelism(&self, global: &GlobalState) -> usize {
        self.inner.max_parallelism(global)
    }

    fn deterministic_split(&self, global: &GlobalState) -> Option<usize> {
        self.inner.deterministic_split(global)
    }

    fn partitions(&self, global: &GlobalState) -> usize {
        self.inner.partitions(global)
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
        Box::pin(async move {
            let Some(batch) = self.inner.get_data(local, global).await? else {
                return Ok(None);
            };
            let out = self.projection.apply(batch.data())?;
            Ok(Some(Batch::new(out).with_split(batch.split())))
        })
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
