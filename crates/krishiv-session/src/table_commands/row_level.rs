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
use krishiv_common::error::{
    DuplicateAssignments, IncompatibleDataForTableCannotSafelyCast, KrishivError,
    KrishivUnsupportedFeature,
};
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
        // A table that cannot be planned keeps the statement's old refusal (a v1 table's error
        // class does not change): only a table offering a row-level table runs it.
        let Ok(probe) = plain
            .analyze_variable_query(&format!("SELECT * FROM {quoted}"))
            .await
        else {
            return Ok(None);
        };
        let Ok((_, physical)) = plain.physical(&probe.analyzed, &probe.env).await else {
            return Ok(None);
        };
        let Some(scan) = first_scan(physical.root()) else {
            return Ok(None);
        };
        let Ok(provider) = probe.env.catalog.provider(&scan) else {
            return Ok(None);
        };
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
        let mut seen: Vec<String> = Vec::new();
        let mut duplicates: Vec<String> = Vec::new();
        for a in assignments {
            let key = a.column.to_lowercase();
            if seen.contains(&key) {
                if !duplicates.contains(&key) {
                    duplicates.push(key);
                }
            } else {
                seen.push(key);
            }
        }
        if !duplicates.is_empty() {
            let names: Vec<String> = assignments
                .iter()
                .filter(|a| duplicates.contains(&a.column.to_lowercase()))
                .map(|a| ident(&a.column))
                .fold(Vec::new(), |mut acc, n| {
                    if !acc.iter().any(|m: &String| m.eq_ignore_ascii_case(&n)) {
                        acc.push(n);
                    }
                    acc
                });
            return Err(DuplicateAssignments {
                name_list: names.join(", "),
            }
            .into());
        }
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
        self.check_assignments(
            quoted,
            alias,
            &hit,
            &columns,
            &with_row_id,
            &table,
            op,
            assignments,
        )
        .await?;

        if table.capabilities().merge_on_read {
            return self
                .run_merge_on_read(
                    &table,
                    op,
                    quoted,
                    alias,
                    &hit,
                    &columns,
                    &with_row_id,
                    assignments,
                )
                .await;
        }

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
        let column_of =
            |name: &str| crate::functions::expr(&format!("{}.{}", ident(alias), ident(name)));
        let mut select = Vec::with_capacity(columns.len());
        for c in &columns {
            let value = match assignments
                .iter()
                .find(|a| a.column.eq_ignore_ascii_case(&c.name))
            {
                // Lazy, as `CASE`: the new value is computed for hit rows only, and stored as
                // Spark's ANSI store assignment stores it whatever `spark.sql.ansi.enabled` says.
                Some(a) => crate::functions::call_function(
                    "if",
                    [
                        crate::functions::expr(&hit)?,
                        crate::functions::expr(a.value.as_deref().unwrap_or("NULL"))?
                            .cast_ansi(c.data_type.clone()),
                        column_of(&c.name)?,
                    ],
                ),
                None => column_of(&c.name)?,
            };
            select.push(value.alias(&c.name));
        }
        let mut rewrite = self
            .hidden_frame(&part, part_entry, &with_row_id)
            .alias(alias);
        if op == RowLevelOperation::Delete {
            rewrite = rewrite.filter(crate::functions::expr(&format!("NOT {hit}"))?);
        }
        let rewrite = rewrite.select(select);
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
    /// A merge-on-read `DELETE` or `UPDATE`: one query over the snapshot, the rows the condition
    /// hits as `(_krishiv_row_file, _krishiv_row_index)` (`UPDATE`: after the new row, whose
    /// columns are the table's), into the table's rewriter, which records the deletes and writes
    /// the new rows in one commit. No file is read twice and none is rewritten. The result is the
    /// table's [`output`](RowLevelTable::output); the hit count is not known before the commit
    /// (Spark's Iceberg tables return no rows), so it is `0` for a table that wants one.
    #[expect(clippy::too_many_arguments, reason = "the statement's parts")]
    async fn run_merge_on_read(
        &self,
        table: &Arc<dyn RowLevelTable>,
        op: RowLevelOperation,
        quoted: &str,
        alias: &str,
        hit: &str,
        columns: &[TableColumn],
        with_row_id: &[TableColumn],
        assignments: &[Assignment],
    ) -> Result<Option<DataFrame>, KrishivError> {
        let (whole, whole_entry) = self.state.catalog.put_hidden_provider(
            with_row_id.to_vec(),
            Arc::new(Target {
                table: Arc::clone(table),
                op,
                files: None,
            }),
        );
        let column_of =
            |name: &str| crate::functions::expr(&format!("{}.{}", ident(alias), ident(name)));
        let mut select = Vec::new();
        if op == RowLevelOperation::Update {
            for c in columns {
                let value = match assignments
                    .iter()
                    .find(|a| a.column.eq_ignore_ascii_case(&c.name))
                {
                    Some(a) => crate::functions::expr(a.value.as_deref().unwrap_or("NULL"))?
                        .cast_ansi(c.data_type.clone()),
                    None => column_of(&c.name)?,
                };
                select.push(value.alias(&c.name));
            }
        }
        select.push(column_of(ROW_FILE_COLUMN)?.alias(ROW_FILE_COLUMN));
        select.push(column_of(ROW_POSITION_COLUMN)?.alias(ROW_POSITION_COLUMN));
        let frame = self
            .hidden_frame(&whole, whole_entry, with_row_id)
            .alias(alias)
            .filter(crate::functions::expr(hit)?)
            .select(select);
        let analysis = frame.action_analysis().await?;
        let (_, physical) = self.physical(&analysis.analyzed, &analysis.env).await?;
        let target = first_scan(physical.root())
            .ok_or_else(|| internal("a row-level change that reads no table".to_owned()))?;
        let mut request =
            RowLevelWrite::new(op, quoted, Arc::clone(physical.schema()), Vec::new(), None);
        request.write_id = Some(crate::session::job_uuid());
        let plan = physical.into_table_write(target, TableWriteRequest::RowLevel(request));
        let mut stream = self.run(&plan, &analysis.env).await?;
        while let Some(batch) = stream.next().await {
            batch?;
        }
        drop(stream);
        self.refresh_files();
        match table.output(&RowLevelCounts::affected(0)) {
            Some(batch) => self
                .script_result(batch.schema(), vec![batch])
                .await
                .map(Some),
            None => self.empty_frame().await.map(Some),
        }
    }
}

impl Session {
    /// The checks of the statement's condition and new values before anything runs: the
    /// condition is deterministic (the find and the rewrite each evaluate it), and each new
    /// value's type may be stored in its column (Spark's ANSI store assignment, as an `INSERT`
    /// checks it: `INCOMPATIBLE_DATA_FOR_TABLE.CANNOT_SAFELY_CAST`).
    #[expect(
        clippy::too_many_arguments,
        reason = "the statement's parts, each read once"
    )]
    async fn check_assignments(
        &self,
        quoted: &str,
        alias: &str,
        hit: &str,
        columns: &[TableColumn],
        with_row_id: &[TableColumn],
        table: &Arc<dyn RowLevelTable>,
        op: RowLevelOperation,
        assignments: &[Assignment],
    ) -> Result<(), KrishivError> {
        let (name, entry) = self.state.catalog.put_hidden_provider(
            with_row_id.to_vec(),
            Arc::new(Target {
                table: Arc::clone(table),
                op,
                files: Some(Vec::new()),
            }),
        );
        let mut texts: Vec<String> = assignments
            .iter()
            .map(|a| a.value.clone().unwrap_or_else(|| "NULL".to_owned()))
            .collect();
        texts.push(hit.to_owned());
        let probe = self
            .hidden_frame(&name, entry, with_row_id)
            .alias(alias)
            .select_expr(&texts)?;
        let analysis = probe.action_analysis().await?;
        let deterministic = match &analysis.analyzed.plan.root().node {
            krishiv_plan::PlanNode::Project { exprs, .. } => exprs
                .last()
                .is_none_or(|e| krishiv_plan::is_deterministic(e)),
            _ => true,
        };
        if !deterministic {
            return Err(KrishivUnsupportedFeature {
                feature: format!("{} with a non-deterministic condition", op.name()),
            }
            .into());
        }
        for (a, out) in assignments.iter().zip(&analysis.analyzed.output) {
            let Some(column) = columns
                .iter()
                .find(|c| c.name.eq_ignore_ascii_case(&a.column))
            else {
                continue;
            };
            if !super::can_store(&out.data_type, &column.data_type) {
                return Err(IncompatibleDataForTableCannotSafelyCast {
                    table_name: quoted.to_owned(),
                    col_name: ident(&column.name),
                    src_type: format!("\"{}\"", out.data_type.catalog_string().to_uppercase()),
                    target_type: format!(
                        "\"{}\"",
                        column.data_type.catalog_string().to_uppercase()
                    ),
                }
                .into());
            }
        }
        Ok(())
    }

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
