// SPDX-License-Identifier: Apache-2.0
//! Copy-on-write `DELETE` and `UPDATE` on a table whose provider offers a row-level table
//! (design `P3-PR-DML-row-level.md` §3; `TableProvider::row_level`). Two queries over one pinned
//! snapshot of the target:
//!
//! 1. **Find**: `SELECT _krishiv_row_file, count(*) FROM <target> WHERE coalesce(c, false)
//!    GROUP BY 1` over [`RowLevelTable::scan`]: the touched files and the hit count.
//! 2. **Rewrite**: over [`RowLevelTable::scan_files`] of the touched files, the rows to keep
//!    (`DELETE`: `WHERE NOT coalesce(c, false)`) or every row with the new values where the
//!    condition holds (`UPDATE`: `CASE WHEN coalesce(c, false) THEN v ELSE col END`), written
//!    through a [`PhysNode::TableWrite`](krishiv_exec::physical::PhysNode::TableWrite) into the
//!    table's [`rewriter`](RowLevelTable::rewriter), which commits once.
//!
//! Both queries read the target as a hidden table whose provider is the one row-level table
//! (`SessionCatalog::put_hidden_provider`), through a frame over its resolved scan (no SQL text
//! names a hidden table), so they and the rewrite's commit see the snapshot the statement read.
//! The condition and the new values are the statement's SQL text, parsed as expressions over
//! the target under the statement's alias. The rewriter is called with no touched files too: whether an empty change
//! commits is the table's choice. The result is the table's [`output`](RowLevelTable::output)
//! for the hit count, else no rows.
//!
//! The statement runs in-process: a hidden provider table gets no splits
//! (`QueryCatalog`'s split planning), so a distributed stage would read it whole in every task.
//!
//! Not here yet: `MERGE` (PR-37c), merge-on-read (PR-37d), `SET c = DEFAULT`, the conflict
//! check's read filter (the rewrite is checked as a whole-table read, which is wider), the
//! CHAR/VARCHAR write checks of an `UPDATE`, and a condition that names the table by a
//! qualified name longer than its own (`db.t.c`: the target is read under its last name part or
//! the statement's alias).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use futures::StreamExt;
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::physical::{PhysNode, TableWriteRequest};
use krishiv_exec::planner::{
    RowLevelCounts, RowLevelOperation, RowLevelTable, RowLevelWrite, TableProvider,
    TableProviderResolver as _,
};
use krishiv_exec::{ROW_FILE_COLUMN, ROW_POSITION_COLUMN, ScanPushdown};
use krishiv_plan::analyzer::TableColumn;
use krishiv_types::DataType;

use crate::dataframe::DataFrame;
use crate::session::{Session, internal};

/// A column of the statement's `SET` list: the column and the value's SQL text (`None`:
/// `DEFAULT`).
pub(crate) struct Assignment {
    pub(crate) column: String,
    pub(crate) value: Option<String>,
}

/// The target as a provider: the one row-level table, read whole or restricted to `files`.
#[derive(Debug)]
struct Target {
    table: Arc<dyn RowLevelTable>,
    op: RowLevelOperation,
    files: Option<Vec<Vec<u8>>>,
}

impl TableProvider for Target {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        match &self.files {
            None => self.table.scan(),
            Some(files) => self.table.scan_files(files),
        }
    }

    fn row_level(&self, op: RowLevelOperation) -> Option<Arc<dyn RowLevelTable>> {
        (op == self.op).then(|| Arc::clone(&self.table))
    }
}

/// `name` as SQL text: a backquoted identifier.
fn ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// The first scan of `node`, depth first.
fn first_scan(node: &PhysNode) -> Option<krishiv_plan::Scan> {
    match node {
        PhysNode::Scan(s) => Some(s.scan.clone()),
        _ => node.children().into_iter().find_map(first_scan),
    }
}

impl Session {
    /// Runs `op` on the table `quoted` if its provider offers a row-level table for it; `None`
    /// leaves the statement to the caller's refusal. `alias` is the statement's alias, else the
    /// table's own last name part; `condition` and the assignments are SQL text of the statement.
    pub(crate) async fn run_row_level(
        &self,
        op: RowLevelOperation,
        quoted: &str,
        alias: &str,
        condition: Option<&str>,
        assignments: &[Assignment],
    ) -> Result<Option<DataFrame>, KrishivError> {
        let plain = self.with_locals(Vec::new());
        let probe = plain
            .analyze_variable_query(&format!("SELECT * FROM {quoted}"))
            .await?;
        let (_, physical) = plain.physical(&probe.analyzed, &probe.env).await?;
        let Some(scan) = first_scan(physical.root()) else {
            return Ok(None);
        };
        let provider = probe.env.catalog.provider(&scan)?;
        let Some(table) = provider.row_level(op) else {
            return Ok(None);
        };
        let columns: Vec<TableColumn> = probe
            .analyzed
            .output
            .iter()
            .zip(&probe.analyzed.output_declared)
            .map(|(c, declared)| TableColumn {
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                nullable: c.nullable,
                char_varchar: declared.clone(),
            })
            .collect();
        for a in assignments {
            if a.value.is_none() {
                return Err(KrishivUnsupportedFeature {
                    feature: format!("{} with SET {} = DEFAULT", op.name(), a.column),
                }
                .into());
            }
            if !columns
                .iter()
                .any(|c| c.name.eq_ignore_ascii_case(&a.column))
            {
                return Err(KrishivUnsupportedFeature {
                    feature: format!("{} of the nested field {}", op.name(), a.column),
                }
                .into());
            }
        }
        let mut with_row_id = columns.clone();
        with_row_id.push(TableColumn {
            name: ROW_FILE_COLUMN.to_owned(),
            data_type: DataType::Binary,
            nullable: false,
            char_varchar: None,
        });
        with_row_id.push(TableColumn {
            name: ROW_POSITION_COLUMN.to_owned(),
            data_type: DataType::Long,
            nullable: false,
            char_varchar: None,
        });
        let hit =
            condition.map_or_else(|| "true".to_owned(), |c| format!("coalesce(({c}), false)"));

        // 1. The touched files and the hit count, over the whole snapshot.
        let (whole, whole_entry) = self.state.catalog.put_hidden_provider(
            with_row_id.clone(),
            Arc::new(Target {
                table: Arc::clone(&table),
                op,
                files: None,
            }),
        );
        let found = self
            .hidden_frame(&whole, whole_entry, &with_row_id)
            .alias(alias)
            .filter(crate::functions::expr(&hit)?)
            .group_by([crate::functions::col(ROW_FILE_COLUMN)])
            .agg([crate::functions::expr("count(*)")?]);
        let batches = found.collect().await?;
        let (touched, affected) = touched_files(&batches)?;

        // 2. The rewrite of the touched files, into the table's rewriter.
        let (part, part_entry) = self.state.catalog.put_hidden_provider(
            with_row_id.clone(),
            Arc::new(Target {
                table: Arc::clone(&table),
                op,
                files: Some(touched.clone()),
            }),
        );
        let select: Vec<String> = columns
            .iter()
            .map(|c| {
                let name = ident(&c.name);
                match assignments
                    .iter()
                    .rfind(|a| a.column.eq_ignore_ascii_case(&c.name))
                {
                    Some(a) => format!(
                        "CAST(CASE WHEN {hit} THEN ({value}) ELSE {alias}.{name} END AS {ty}) AS {name}",
                        value = a.value.as_deref().unwrap_or("NULL"),
                        alias = ident(alias),
                        ty = c.data_type.sql(),
                    ),
                    None => format!("{}.{name}", ident(alias)),
                }
            })
            .collect();
        let mut rewrite = self
            .hidden_frame(&part, part_entry, &with_row_id)
            .alias(alias);
        if op == RowLevelOperation::Delete {
            rewrite = rewrite.filter(crate::functions::expr(&format!("NOT {hit}"))?);
        }
        let rewrite = rewrite.select_expr(&select)?;
        let analysis = rewrite.action_analysis().await?;
        let (_, physical) = self.physical(&analysis.analyzed, &analysis.env).await?;
        let target = first_scan(physical.root())
            .ok_or_else(|| internal("a row-level rewrite that reads no table".to_owned()))?;
        let mut request =
            RowLevelWrite::new(op, quoted, Arc::clone(physical.schema()), touched, None);
        request.write_id = Some(crate::session::job_uuid());
        let plan = physical.into_table_write(target, TableWriteRequest::RowLevel(request));
        let mut stream = self.run(&plan, &analysis.env).await?;
        while let Some(batch) = stream.next().await {
            batch?;
        }
        drop(stream);
        self.refresh_files();

        // 3. The statement's result.
        match table.output(&RowLevelCounts::affected(affected)) {
            Some(batch) => self
                .script_result(batch.schema(), vec![batch])
                .await
                .map(Some),
            None => self.empty_frame().await.map(Some),
        }
    }
}

impl Session {
    /// A frame reading the hidden table `name` (`columns`), which owns `entry`: a resolved
    /// scan, as a script result's frame reads its table, since no SQL text names it.
    fn hidden_frame(
        &self,
        name: &str,
        entry: Arc<crate::catalog::TempTable>,
        columns: &[TableColumn],
    ) -> DataFrame {
        let output = columns
            .iter()
            .zip(0_u64..)
            .map(|(c, i)| krishiv_plan::BoundColumn {
                id: krishiv_plan::AttrId(i),
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                nullable: c.nullable,
            })
            .collect();
        let plan =
            krishiv_plan::LogicalPlan::new(krishiv_plan::PlanNode::Scan(krishiv_plan::Scan {
                source: krishiv_plan::ReadSource::Table {
                    name: vec![name.to_owned()],
                    options: std::collections::BTreeMap::new(),
                },
                output,
                version: None,
            }));
        let mut df = DataFrame::new(self.clone(), Arc::new(plan));
        df.keep.push(entry);
        df
    }
}

/// The touched files (each once) and the total hit count of the find query's rows.
fn touched_files(batches: &[RecordBatch]) -> Result<(Vec<Vec<u8>>, u64), KrishivError> {
    let mut files = Vec::new();
    let mut hits = 0_u64;
    for b in batches {
        let column = b.column(0);
        let file: Vec<Option<Vec<u8>>> = match column.data_type() {
            ArrowType::Binary => column
                .as_binary::<i32>()
                .iter()
                .map(|v| v.map(<[u8]>::to_vec))
                .collect(),
            ArrowType::BinaryView => column
                .as_binary_view()
                .iter()
                .map(|v| v.map(<[u8]>::to_vec))
                .collect(),
            other => {
                return Err(internal(format!("a touched file column of type {other}")));
            }
        };
        let count = b
            .column(1)
            .as_primitive_opt::<arrow_array::types::Int64Type>()
            .ok_or_else(|| internal("a row-level hit count that is not a long".to_owned()))?;
        for i in 0..b.num_rows() {
            if let Some(Some(f)) = file.get(i) {
                files.push(f.clone());
            }
            hits = hits.saturating_add(u64::try_from(count.value(i)).unwrap_or(0));
        }
    }
    Ok((files, hits))
}

#[cfg(test)]
mod tests;
