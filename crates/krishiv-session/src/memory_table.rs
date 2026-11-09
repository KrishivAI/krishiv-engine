// SPDX-License-Identifier: Apache-2.0
//! `batches` as an in-memory table: the part of [`Session::register_batches`] that a script's result
//! shares.

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::TableColumn;
use krishiv_types::DataType;
use krishiv_types::batch::Batch;

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
