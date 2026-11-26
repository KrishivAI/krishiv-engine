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
//! The table raises its own errors through [`RowLevelTable::error`] (a Delta table refuses a
//! condition with a subquery, `DELTA_UNSUPPORTED_SUBQUERY`, and wraps an error raised while the
//! rewrite writes its rows in `TASK_WRITE_FAILED`); the planner holds no table format's code.
//! The conflict check's read set is the table's (the Delta connector reads the condition's
//! partition conjuncts).
//!
//! `MERGE` is `merge.rs` (PR-37c), over the same hidden tables. Not here yet: merge-on-read (PR-37d), `SET c = DEFAULT`, the
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
    RowLevelCounts, RowLevelOperation, RowLevelProblem, RowLevelTable, RowLevelWrite,
    TableProvider, TableProviderResolver as _,
};
use krishiv_exec::{ROW_FILE_COLUMN, ROW_POSITION_COLUMN, ScanPushdown};
use krishiv_plan::analyzer::TableColumn;
use krishiv_types::DataType;

use crate::dataframe::DataFrame;
use crate::session::{Session, internal};

/// The rewrite's option holding the rows the condition holds for (the find query's count): a
/// table's metrics and result rows read it (Delta's `numDeletedRows` / `numUpdatedRows`).
pub(crate) const AFFECTED_OPTION: &str = "__krishiv_row_level_affected";

/// The rewrite's option holding the statement's condition as written (absent: none), for the
/// table's commit record (Delta's `operationParameters.predicate`).
pub(crate) const CONDITION_OPTION: &str = "__krishiv_row_level_condition";

/// The rewrite's option holding the find query's milliseconds (Delta's `scanTimeMs`).
pub(crate) const SCAN_MS_OPTION: &str = "__krishiv_row_level_scan_ms";

/// The rewrite's option holding the name the statement reads the target under (its alias,
/// else the table's last name part): the qualifier the condition's column names may carry.
pub(crate) const ALIAS_OPTION: &str = "__krishiv_row_level_alias";

/// A column of the statement's `SET` list: the column and the value's SQL text (`None`:
/// `DEFAULT`).
#[derive(Debug, Clone)]
pub(crate) struct Assignment {
    /// The column's name parts joined by `.` (one part for a top-level column, whose name may
    /// itself hold a `.` when it was quoted).
    pub(crate) column: String,
    /// The key names a nested field (more than one name part).
    pub(crate) nested: bool,
    pub(crate) value: Option<String>,
}

/// The target as a provider: the one row-level table, read whole or restricted to `files`.
#[derive(Debug)]
pub(super) struct Target {
    pub(super) table: Arc<dyn RowLevelTable>,
    pub(super) op: RowLevelOperation,
    pub(super) files: Option<Vec<Vec<u8>>>,
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
pub(super) fn ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// The first scan of `node`, depth first.
pub(super) fn first_scan(node: &PhysNode) -> Option<krishiv_plan::Scan> {
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
        let Ok((_, physical)) = plain.physical_probe(&probe.analyzed, &probe.env).await else {
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
        // A table whose commit records the key takes it; any other refuses it here, before any scan or
        // path split (E34).
        let caps = table.capabilities();
        // The validated key travels with the statement: the rewrite's own config snapshot is never read
        // for it, so a `conf().set` between the two cannot commit under another key.
        let mut key_options = std::collections::BTreeMap::new();
        if caps.idempotent_commit {
            // An empty key is refused here, before the find scan.
            apply_idempotency_key(&probe.env.config, &mut key_options)?;
        } else {
            refuse_idempotency_key(&probe.env.config)?;
        }
        // A retry of a statement that already committed under this key changes nothing and
        // evaluates nothing: its SET values (an ANSI overflow, say) never run again.
        if let Some(key) = krishiv_exec::planner::idempotency_key_option(&key_options)
            && table.replayed(&key).await?
        {
            return match table.output(&RowLevelCounts::affected(0)) {
                Some(batch) => self
                    .script_result(batch.schema(), vec![batch])
                    .await
                    .map(Some),
                None => self.empty_frame().await.map(Some),
            };
        }
        // The texts are read again under the statement's parser options (as MERGE does), so a
        // double-quoted name under `spark.sql.ansi.doubleQuotedIdentifiers` stays a column.
        let parser = &probe.env.parser;
        // A Delta table's root as delta-spark's errors name it (`file:/..`), for a table by path
        // and for a catalog table over a Delta location alike (its scan keeps the table's name).
        #[cfg(feature = "delta")]
        let delta_root = probe.env.catalog.delta_root(&scan).transpose()?;
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
        // A table may refuse a subquery in the condition (delta-spark does, after resolution:
        // `PreprocessTableDelete.scala:35-45`, `PreprocessTableUpdate.scala:44`).
        if let Some(c) = condition
            && has_subquery(&crate::functions::expr_with(c, parser)?.expr)
            && let Some(e) = table.error(
                op,
                &RowLevelProblem::Subquery {
                    what: "condition",
                    condition: c,
                },
            )
        {
            return Err(e);
        }
        let hit =
            condition.map_or_else(|| "true".to_owned(), |c| format!("coalesce(({c}), false)"));
        let value_types = self
            .check_assignments(
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
                    parser,
                    &key_options,
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
            .filter(crate::functions::expr_with(&hit, parser)?)
            .group_by([crate::functions::col(ROW_FILE_COLUMN)])
            .agg([crate::functions::expr_with("count(*)", parser)?]);
        let find_started = std::time::Instant::now();
        let batches = found.collect().await?;
        let scan_ms = find_started.elapsed().as_millis();
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
                        crate::functions::expr_with(&hit, parser)?,
                        crate::functions::expr_with(a.value.as_deref().unwrap_or("NULL"), parser)?
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
            rewrite = rewrite.filter(crate::functions::expr_with(&format!("NOT {hit}"), parser)?);
        }
        let rewrite = rewrite.select(select);
        let analysis = rewrite.action_analysis().await?;
        let (_, physical) = self.physical(&analysis.analyzed, &analysis.env).await?;
        let target = first_scan(physical.root())
            .ok_or_else(|| internal("a row-level rewrite that reads no table".to_owned()))?;
        // The rewrite's own read is the hidden provider: a scan of the target in it is a subquery's
        // (or a view's) read of the table. A Delta target is compared by its resolved root, so
        // `/t`, `/t/`, `file:/t` and the catalog table over that location are one table.
        #[cfg(feature = "delta")]
        let rereads_target = match &delta_root {
            Some(root) => crate::catalog::reads_delta_root(
                analysis.analyzed.plan.root(),
                root,
                &analysis.env.catalog,
            ),
            None => crate::catalog::reads_source(analysis.analyzed.plan.root(), &scan.source),
        };
        #[cfg(not(feature = "delta"))]
        let rereads_target =
            crate::catalog::reads_source(analysis.analyzed.plan.root(), &scan.source);
        let read_filter = read_filter(condition, rereads_target, parser)?;
        #[cfg(test)]
        tests::READ_FILTERS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((quoted.to_string(), read_filter.is_some()));
        let mut request = RowLevelWrite::new(
            op,
            quoted,
            Arc::clone(physical.schema()),
            touched,
            read_filter,
        );
        request.write_id = Some(crate::session::job_uuid());
        request
            .options
            .insert(AFFECTED_OPTION.to_owned(), affected.to_string());
        request
            .options
            .insert(SCAN_MS_OPTION.to_owned(), scan_ms.to_string());
        request
            .options
            .insert(ALIAS_OPTION.to_owned(), alias.to_owned());
        if let Some(c) = condition {
            request
                .options
                .insert(CONDITION_OPTION.to_owned(), c.to_owned());
        }
        request
            .options
            .extend(key_options.iter().map(|(k, v)| (k.clone(), v.clone())));
        let plan = physical.into_table_write(target, TableWriteRequest::RowLevel(request));
        let written = async {
            let mut stream = self.run(&plan, &analysis.env).await?;
            while let Some(batch) = stream.next().await {
                batch?;
            }
            Ok::<(), KrishivError>(())
        }
        .await;
        if let Err(e) = written {
            return Err(write_failed(&table, op, e, &columns, &value_types));
        }
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
        parser: &krishiv_sql::ParserOptions,
        key_options: &std::collections::BTreeMap<String, String>,
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
                    Some(a) => {
                        crate::functions::expr_with(a.value.as_deref().unwrap_or("NULL"), parser)?
                            .cast_ansi(c.data_type.clone())
                    }
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
            .filter(crate::functions::expr_with(hit, parser)?)
            .select(select);
        let analysis = frame.action_analysis().await?;
        let (_, physical) = self.physical(&analysis.analyzed, &analysis.env).await?;
        let target = first_scan(physical.root())
            .ok_or_else(|| internal("a row-level change that reads no table".to_owned()))?;
        let mut request =
            RowLevelWrite::new(op, quoted, Arc::clone(physical.schema()), Vec::new(), None);
        request.write_id = Some(crate::session::job_uuid());
        request
            .options
            .extend(key_options.iter().map(|(k, v)| (k.clone(), v.clone())));
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
    ) -> Result<Vec<(String, DataType)>, KrishivError> {
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
        let mut value_types = Vec::with_capacity(assignments.len());
        for (a, out) in assignments.iter().zip(&analysis.analyzed.output) {
            let Some(column) = columns
                .iter()
                .find(|c| c.name.eq_ignore_ascii_case(&a.column))
            else {
                continue;
            };
            value_types.push((column.name.clone(), out.data_type.clone()));
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
        Ok(value_types)
    }

    /// A frame reading the hidden table `name` (`columns`), which owns `entry`: a resolved
    /// scan, as a script result's frame reads its table, since no SQL text names it.
    pub(super) fn hidden_frame(
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

/// `fut`, after which (on completion, failure or drop: a cancelled statement too) the hidden
/// tables nothing holds any more go from `catalog`: a row-level statement's target scans and
/// its source copy (a spill file) do not wait for the catalog's next change. The future is
/// dropped before the sweep, so the entries it held are free by then.
pub(super) struct Swept<F> {
    fut: Option<std::pin::Pin<Box<F>>>,
    catalog: Arc<crate::catalog::SessionCatalog>,
}

impl<F: Future> Swept<F> {
    pub(super) fn new(catalog: Arc<crate::catalog::SessionCatalog>, fut: F) -> Self {
        Self {
            fut: Some(Box::pin(fut)),
            catalog,
        }
    }
}

impl<F: Future> Future for Swept<F> {
    type Output = F::Output;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.get_mut();
        let Some(fut) = this.fut.as_mut() else {
            return std::task::Poll::Pending;
        };
        let out = fut.as_mut().poll(cx);
        if out.is_ready() {
            this.fut = None;
            this.catalog.sweep_hidden();
        }
        out
    }
}

impl<F> Drop for Swept<F> {
    fn drop(&mut self) {
        if self.fut.take().is_some() {
            self.catalog.sweep_hidden();
        }
    }
}

/// What the statement read, for the table's conflict check: the condition (the parsed,
/// unresolved expression: column names as written), or `None` for every row. A statement whose
/// subquery (in an assignment, or through a view) reads the target table again reads every row:
/// delta-spark records that scan's own predicates in the transaction's read set, and the whole
/// table is a superset of them, so no conflict is missed (an over-approximation: delta-spark
/// conflicts only on the subquery's predicates). A subquery over another table records nothing on
/// the target, so the condition stays (#909 review, finding 2).
fn read_filter(
    condition: Option<&str>,
    rereads_target: bool,
    parser: &krishiv_sql::ParserOptions,
) -> Result<Option<Arc<krishiv_plan::Expr>>, KrishivError> {
    if rereads_target {
        return Ok(None);
    }
    condition
        .map(|c| Ok(Arc::new(crate::functions::expr_with(c, parser)?.expr)))
        .transpose()
}

/// Whether `e` holds a subquery anywhere.
fn has_subquery(e: &krishiv_plan::Expr) -> bool {
    let mut stack: Vec<&krishiv_plan::Expr> = vec![e];
    while let Some(x) = stack.pop() {
        if matches!(x.node, krishiv_plan::ExprNode::Subquery { .. }) {
            return true;
        }
        x.for_each_child(&mut |c| stack.push(c));
    }
    false
}

/// An error raised while a row-level rewrite writes its rows, as the table raises it
/// ([`RowLevelProblem::WriteFailed`]; delta-spark wraps it in `TASK_WRITE_FAILED`), else as it
/// came. `value_types` are the assigned columns and their new values' types.
pub(super) fn write_failed(
    table: &Arc<dyn RowLevelTable>,
    op: RowLevelOperation,
    e: KrishivError,
    columns: &[TableColumn],
    value_types: &[(String, DataType)],
) -> KrishivError {
    let overflow = store_assignment_overflow(&e, columns, value_types);
    table
        .error(
            op,
            &RowLevelProblem::WriteFailed {
                error: &e,
                overflow,
            },
        )
        .unwrap_or(e)
}

/// The assigned column an arithmetic error `e` (SQLSTATE 22003 or 22012: Spark's
/// `ArithmeticException`s, such as `CAST_OVERFLOW`, `ARITHMETIC_OVERFLOW` and
/// `DIVIDE_BY_ZERO`) may come from: one whose store-assignment cast can overflow (a column
/// whose type holds an integral or decimal type, at any depth, and that the value's type does
/// not `Cast.canUpCast` to; delta-spark's `CheckOverflowInTableWrite`,
/// `UpdateExpressionsSupport.scala:719-757`), with the value's and the column's types. Krishiv
/// cannot tell which value raised: the first such column in table order.
fn store_assignment_overflow<'a>(
    e: &KrishivError,
    columns: &'a [TableColumn],
    value_types: &'a [(String, DataType)],
) -> Option<(&'a str, &'a DataType, &'a DataType)> {
    if !matches!(e.sql_state(), Some("22003" | "22012")) {
        return None;
    }
    // delta-spark's `canCauseCastOverflow` (`:729-732`).
    let can_overflow = |from: &DataType, to: &DataType| {
        contains_integral_or_decimal(to) && !krishiv_plan::analyzer::can_up_cast(from, to)
    };
    columns.iter().find_map(|c| {
        value_types
            .iter()
            .find(|(name, value)| {
                name.eq_ignore_ascii_case(&c.name) && can_overflow(value, &c.data_type)
            })
            .map(|(_, value)| (c.name.as_str(), value, &c.data_type))
    })
}

/// delta-spark's `containsIntegralOrDecimalType` (`UpdateExpressionsSupport.scala:719-727`).
fn contains_integral_or_decimal(t: &DataType) -> bool {
    match t {
        DataType::Byte
        | DataType::Short
        | DataType::Integer
        | DataType::Long
        | DataType::Decimal(_) => true,
        DataType::Array(a) => contains_integral_or_decimal(a.element_type()),
        DataType::Map(m) => {
            contains_integral_or_decimal(m.key_type())
                || contains_integral_or_decimal(m.value_type())
        }
        DataType::Struct(s) => s
            .fields()
            .iter()
            .any(|f| contains_integral_or_decimal(f.data_type())),
        _ => false,
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

#[cfg(all(test, feature = "delta"))]
#[test]
fn the_delta_connector_reads_the_same_option_keys() {
    use krishiv_connectors_delta::write::dml as d;
    assert_eq!(AFFECTED_OPTION, d::ROW_LEVEL_AFFECTED_OPTION);
    assert_eq!(CONDITION_OPTION, d::ROW_LEVEL_CONDITION_OPTION);
    assert_eq!(SCAN_MS_OPTION, d::ROW_LEVEL_SCAN_MS_OPTION);
    assert_eq!(ALIAS_OPTION, d::ROW_LEVEL_ALIAS_OPTION);
}

pub(crate) mod rewrite_files;
mod rewrite_scan;

#[cfg(test)]
pub(crate) mod tests;

/// A table whose commit records no key (`idempotent_commit` false) does not take
/// `krishiv.write.idempotencyKey` (E34): a set key is refused instead of deduplicating on some table
/// formats only.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` when the session setting names a key.
pub(super) fn refuse_idempotency_key(
    config: &krishiv_common::config::Config,
) -> Result<(), KrishivError> {
    match config
        .get(&krishiv_common::config::WRITE_IDEMPOTENCY_KEY)
        .and_then(krishiv_common::config::ConfigValue::as_str)
    {
        Some(_) => Err(krishiv_common::error::KrishivUnsupportedFeature {
            feature: "krishiv.write.idempotencyKey on UPDATE, DELETE or MERGE; unset it for this statement"
                .to_owned(),
        }
        .into()),
        None => Ok(()),
    }
}

/// Puts `krishiv.write.idempotencyKey` into a write's `options` when it names none (the per-write
/// option wins over the session setting). E34.
///
/// # Errors
/// `INVALID_CONF_VALUE` for an empty key.
pub(super) fn apply_idempotency_key(
    config: &krishiv_common::config::Config,
    options: &mut std::collections::BTreeMap<String, String>,
) -> Result<(), KrishivError> {
    use krishiv_exec::planner::{IDEMPOTENCY_KEY_OPTION, idempotency_key_option};
    let key = match idempotency_key_option(options) {
        Some(k) => k,
        None => {
            match config
                .get(&krishiv_common::config::WRITE_IDEMPOTENCY_KEY)
                .and_then(krishiv_common::config::ConfigValue::as_str)
            {
                Some(k) => k.to_owned(),
                None => return Ok(()),
            }
        }
    };
    if key.is_empty() {
        return Err(krishiv_common::error::KrishivInvalidConfigValue {
            key: IDEMPOTENCY_KEY_OPTION.to_owned(),
            value: key,
            reason: "an idempotency key must not be empty".to_owned(),
        }
        .into());
    }
    options.retain(|k, _| !k.eq_ignore_ascii_case(IDEMPOTENCY_KEY_OPTION));
    options.insert(IDEMPOTENCY_KEY_OPTION.to_owned(), key);
    Ok(())
}
