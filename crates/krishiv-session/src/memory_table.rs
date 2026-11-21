// SPDX-License-Identifier: Apache-2.0
//! `batches` as an in-memory table: the part of [`Session::register_batches`] that a script's result
//! shares.

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_exec::ScanPushdown;
use krishiv_exec::planner::TableProvider;
use krishiv_exec::{FilterPushdown, MemoryScan, PushedFilter, ScanPushdowns, Source};
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

/// What a `SourceTable` calls each time a scan reads it: the table's rows as batches of its schema.
pub type BatchSource = Arc<dyn Fn() -> Result<Vec<RecordBatch>, KrishivError> + Send + Sync>;

/// A table whose rows come from a callback, asked afresh for every scan of a query (not
/// once per registration): the Python Data Source API's reader lists its partitions and reads
/// them when the scan runs (planning and `EXPLAIN` read nothing), as Spark's `PythonScanBuilder` does (design P3-T040 §3.3).
#[derive(Clone)]
struct SourceTable {
    label: String,
    schema: SchemaRef,
    source: BatchSource,
    /// The most bytes one scan's batches may hold ([`MAX_SOURCE_BYTES`]; a field so a test can lower it).
    max_bytes: usize,
    /// The session's pool; every scan counts its bytes in a child group of its own, so an abort of
    /// one scan's group never reaches a later scan.
    memory: Arc<dyn MemoryPool>,
    /// The schema `source`'s batches have (before the carrier's canonical form), which they are
    /// checked against.
    declared: SchemaRef,
}

/// A scan's bytes in the session's pool, released when the scan and every scan derived from it
/// are gone.
struct ScanHold {
    _pool: Arc<dyn MemoryPool>,
    // Never locked: the Mutex only makes the guard `Sync`; dropping it releases the reservation.
    _reservation: std::sync::Mutex<Reservation>,
}

/// The most bytes of batches one scan of a [`SourceTable`] may hold: the interim embedded path
/// materialises a source when the scan runs, so it is capped (⚗, design P3-T040 §3.3).
const MAX_SOURCE_BYTES: usize = 1 << 30;

impl std::fmt::Debug for SourceTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceTable")
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl SourceTable {
    /// Reads the source and holds its bytes: one read of the callback.
    fn materialize(&self) -> krishiv_exec::Result<MemoryScan> {
        let batches = (self.source)()?;
        let bytes: usize = batches.iter().map(RecordBatch::get_array_memory_size).sum();
        if bytes > self.max_bytes {
            return Err(krishiv_common::error::KrishivPythonDataSourceLimit {
                source: self.label.clone(),
                bytes: bytes.to_string(),
                limit: self.max_bytes.to_string(),
            }
            .into());
        }
        let batches = batches
            .into_iter()
            .map(|b| {
                if b.schema() != self.declared {
                    return Err(internal(format!(
                        "a batch of schema {} for a table of schema {}",
                        b.schema(),
                        self.declared
                    )));
                }
                crate::carrier::recarried(&b, &self.schema).map(Batch::new)
            })
            .collect::<Result<Vec<Batch>, KrishivError>>()?;
        let pool = self.memory.child("python_source", None);
        let consumer = pool.register("python_source", None);
        let mut reservation = Reservation::new(Arc::clone(&pool), consumer);
        reservation.try_grow(bytes).map_err(KrishivError::from)?;
        let hold = ScanHold {
            _pool: pool,
            _reservation: std::sync::Mutex::new(reservation),
        };
        Ok(MemoryScan::try_new(Arc::clone(&self.schema), batches)?.with_guard(Arc::new(hold)))
    }
}

impl TableProvider for SourceTable {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        Ok(Arc::new(LazySourceScan {
            table: self.clone(),
            read: std::sync::Mutex::new(None),
        }))
    }
}

/// A scan of a [`SourceTable`] that reads the source when its data is first needed. The planner
/// asks a provider for a scan only to learn the schema, and execution asks again, so reading in
/// `scan` would read the source twice and could consume a one-shot reader before the query runs.
struct LazySourceScan {
    table: SourceTable,
    read: std::sync::Mutex<Option<Arc<MemoryScan>>>,
}

impl LazySourceScan {
    fn read(&self) -> krishiv_exec::Result<Arc<MemoryScan>> {
        let mut slot = self
            .read
            .lock()
            .map_err(|_| internal("a poisoned Python source read".to_owned()))?;
        if let Some(scan) = slot.as_ref() {
            return Ok(Arc::clone(scan));
        }
        let scan = Arc::new(self.table.materialize()?);
        *slot = Some(Arc::clone(&scan));
        Ok(scan)
    }
}

impl ScanPushdown for LazySourceScan {
    fn table_schema(&self) -> SchemaRef {
        Arc::clone(&self.table.schema)
    }

    fn filter_pushdown(&self, _filter: &PushedFilter) -> FilterPushdown {
        FilterPushdown::Exact
    }

    fn build(&self, pushdowns: ScanPushdowns) -> krishiv_exec::Result<Arc<dyn Source>> {
        self.read()?.build(pushdowns)
    }

    fn output_schema(&self, pushdowns: &ScanPushdowns) -> krishiv_exec::Result<SchemaRef> {
        if pushdowns.selection.is_some()
            || pushdowns.metadata != krishiv_io::MetadataCols::default()
        {
            // The in-memory scan refuses these; ask it (reads the source).
            return self.read()?.output_schema(pushdowns);
        }
        match &pushdowns.projection {
            Some(paths) => krishiv_io::project_schema(&self.table.schema, paths),
            None => Ok(Arc::clone(&self.table.schema)),
        }
    }

    fn with_memory_batches(
        &self,
        batches: Vec<Batch>,
    ) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        Ok(Arc::new(MemoryScan::try_new(
            Arc::clone(&self.table.schema),
            batches,
        )?))
    }
}

impl Session {
    /// Registers the temporary table `name` of `schema` whose rows `source` returns each time a
    /// scan of it runs. The schema is fixed here (the analysis); the rows are not (a
    /// source that lists files sees new ones on the next action).
    ///
    /// `label` names the source in errors. Each scan's batches are counted in the session's memory
    /// pool (until that scan's sources are dropped) and capped at 1 GiB.
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
        let declared = Arc::clone(&schema);
        let schema = crate::carrier::canonical_schema(schema);
        self.state.catalog.put(
            name,
            TempTable::Provider {
                columns,
                table: Arc::new(SourceTable {
                    label: label.to_owned(),
                    schema,
                    source,
                    max_bytes: MAX_SOURCE_BYTES,
                    memory: Arc::clone(&self.state.memory),
                    declared,
                }),
            },
        );
        self.state.cache.forget_table(name);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::{DataType as ArrowType, Field, Schema};

    #[test]
    fn planning_reads_the_schema_only_and_a_scan_reads_the_source_once() {
        let session = Session::builder().build().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("k", ArrowType::Int32, true)]));
        let reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (s, r) = (Arc::clone(&schema), Arc::clone(&reads));
        let table = SourceTable {
            label: "demo".to_owned(),
            schema: Arc::clone(&schema),
            source: Arc::new(move || {
                r.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(vec![
                    RecordBatch::try_new(
                        Arc::clone(&s),
                        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
                    )
                    .unwrap(),
                ])
            }),
            max_bytes: MAX_SOURCE_BYTES,
            memory: Arc::clone(&session.state.memory),
            declared: Arc::clone(&schema),
        };
        let planning = table.scan().unwrap();
        planning.table_schema();
        planning.output_schema(&ScanPushdowns::default()).unwrap();
        assert_eq!(reads.load(std::sync::atomic::Ordering::SeqCst), 0);
        let execution = table.scan().unwrap();
        execution.build(ScanPushdowns::default()).unwrap();
        execution.build(ScanPushdowns::default()).unwrap();
        assert_eq!(reads.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn a_plan_over_the_byte_cap_is_refused_and_the_table_debugs_by_schema() {
        let session = Session::builder().build().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("k", ArrowType::Int32, true)]));
        let s = Arc::clone(&schema);
        let table = SourceTable {
            label: "demo".to_owned(),
            schema: Arc::clone(&schema),
            source: Arc::new(move || {
                Ok(vec![
                    RecordBatch::try_new(
                        Arc::clone(&s),
                        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
                    )
                    .unwrap(),
                ])
            }),
            max_bytes: 1,
            memory: Arc::clone(&session.state.memory),
            declared: Arc::clone(&schema),
        };
        let error = table
            .scan()
            .unwrap()
            .build(ScanPushdowns::default())
            .err()
            .expect("over the cap")
            .to_string();
        assert!(error.contains("demo"), "{error}");
        let shown = format!("{table:?}");
        assert!(
            shown.contains("SourceTable") && shown.contains("schema"),
            "{shown}"
        );
    }
}
