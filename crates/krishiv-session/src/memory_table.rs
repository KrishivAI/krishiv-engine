// SPDX-License-Identifier: Apache-2.0
//! `batches` as an in-memory table: the part of [`Session::register_batches`] that a script's result
//! shares.

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_exec::MemoryScan;
use krishiv_exec::ScanPushdown;
use krishiv_exec::planner::TableProvider;
use krishiv_memory::{MemoryPool, Reservation};
use krishiv_plan::analyzer::TableColumn;
use krishiv_types::DataType;
use krishiv_types::batch::Batch;

use crate::catalog::TempTable;
use crate::session::{Session, internal};

impl Session {
    /// `batches` (of `schema`) as an in-memory table and its columns: the batches are held once, as
    /// content-addressed IPC in the session's store, memory-accounted; the table holds the lease.
    pub(crate) fn memory_table(
        &self,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> Result<(Vec<TableColumn>, krishiv_exec::planner::MemoryTable), KrishivError> {
        let columns = schema
            .fields()
            .iter()
            .map(|f| {
                Ok(TableColumn::new(
                    f.name().clone(),
                    DataType::from_arrow_field(f)?,
                    f.is_nullable(),
                ))
            })
            .collect::<Result<Vec<_>, KrishivError>>()?;
        if let Some(b) = batches.iter().find(|b| b.schema() != schema) {
            return Err(internal(format!(
                "a batch of schema {} for a table of schema {schema}",
                b.schema()
            )));
        }
        let schema = crate::carrier::canonical_schema(schema);
        let batches = batches
            .into_iter()
            .map(|b| crate::carrier::recarried(&b, &schema).map(Batch::new))
            .collect::<Result<Vec<Batch>, KrishivError>>()?;
        let rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
        let lease = self.state.relations.insert(&schema, &batches)?;
        Ok((
            columns,
            krishiv_exec::planner::MemoryTable::from_lease(schema, rows, lease),
        ))
    }
}

/// What a `SourceTable` calls each time a plan reads it: the table's rows as batches of its schema.
pub type BatchSource = Arc<dyn Fn() -> Result<Vec<RecordBatch>, KrishivError> + Send + Sync>;

/// A table whose rows come from a callback, asked afresh for every physical plan of a query (not
/// once per registration): the Python Data Source API's reader lists its partitions and reads
/// them when the plan is made, as Spark's `PythonScanBuilder` does (design P3-T040 §3.3).
struct SourceTable {
    label: String,
    schema: SchemaRef,
    source: BatchSource,
    /// The session's pool; every scan counts its bytes in a child group of its own, so an abort of
    /// one scan's group never reaches a later plan.
    memory: Arc<dyn MemoryPool>,
    /// The bytes of the latest scan's batches, counted in the session's pool until the next scan
    /// replaces them or the table is dropped (a plan's scan cannot hold them itself).
    held: std::sync::Mutex<Option<(Arc<dyn MemoryPool>, Reservation)>>,
}

/// The most bytes of batches one plan of a [`SourceTable`] may hold: the interim embedded path
/// materialises a source when a query is planned, so it is capped (⚗, design P3-T040 §3.3).
const MAX_SOURCE_BYTES: usize = 1 << 30;

impl std::fmt::Debug for SourceTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceTable")
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl TableProvider for SourceTable {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        let batches = (self.source)()?;
        let bytes: usize = batches.iter().map(RecordBatch::get_array_memory_size).sum();
        if bytes > MAX_SOURCE_BYTES {
            return Err(krishiv_common::error::KrishivPythonDataSourceLimit {
                source: self.label.clone(),
                bytes: bytes.to_string(),
                limit: MAX_SOURCE_BYTES.to_string(),
            }
            .into());
        }
        // The previous scan's bytes go back first: a re-plan needs one copy of the source, not two.
        let mut held = self
            .held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *held = None;
        let pool = self.memory.child("python_source", None);
        let consumer = pool.register("python_source", None);
        let mut reservation = Reservation::new(Arc::clone(&pool), consumer);
        reservation.try_grow(bytes).map_err(KrishivError::from)?;
        *held = Some((pool, reservation));
        drop(held);
        let batches = batches
            .into_iter()
            .map(|b| {
                if b.schema() != self.schema {
                    return Err(internal(format!(
                        "a batch of schema {} for a table of schema {}",
                        b.schema(),
                        self.schema
                    )));
                }
                crate::carrier::recarried(&b, &self.schema).map(Batch::new)
            })
            .collect::<Result<Vec<Batch>, KrishivError>>()?;
        Ok(Arc::new(MemoryScan::try_new(
            Arc::clone(&self.schema),
            batches,
        )?))
    }
}

impl Session {
    /// Registers the temporary table `name` of `schema` whose rows `source` returns each time a
    /// query over it is planned. The schema is fixed here (the analysis); the rows are not (a
    /// source that lists files sees new ones on the next action).
    ///
    /// `label` names the source in errors. Each plan's batches are counted in the session's memory
    /// pool (until the next plan or the table's drop) and capped at 1 GiB.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a column type Spark has no equivalent of;
    /// `KRISHIV_RESERVED_TEMP_VIEW_NAME` for a reserved name.
    pub fn register_batch_source(
        &self,
        name: &str,
        label: &str,
        schema: SchemaRef,
        source: BatchSource,
    ) -> Result<(), KrishivError> {
        if crate::catalog::is_reserved_view_name(name) {
            return Err(crate::catalog::reserved_temp_view_name(name));
        }
        let columns = schema
            .fields()
            .iter()
            .map(|f| {
                Ok(TableColumn::new(
                    f.name().clone(),
                    DataType::from_arrow_field(f)?,
                    f.is_nullable(),
                ))
            })
            .collect::<Result<Vec<_>, KrishivError>>()?;
        let schema = crate::carrier::canonical_schema(schema);
        self.state.catalog.put(
            name,
            TempTable::Provider {
                columns,
                table: Arc::new(SourceTable {
                    label: label.to_owned(),
                    schema,
                    source,
                    memory: Arc::clone(&self.state.memory),
                    held: std::sync::Mutex::new(None),
                }),
            },
        );
        self.state.cache.forget_table(name);
        Ok(())
    }
}
